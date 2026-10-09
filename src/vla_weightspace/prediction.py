"""Train-only PCA and ridge with nested grouped validation and negative controls."""
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from .io import read_json, write_json, write_npz


def folds(records):
    for task in sorted({record["task"] for record in records}):
        yield "task", task, [i for i, r in enumerate(records) if r["task"] != task], [i for i, r in enumerate(records) if r["task"] == task]
    for request in sorted({record["request"] for record in records}):
        yield "request", request, [i for i, r in enumerate(records) if r["request"] != request], [i for i, r in enumerate(records) if r["request"] == request]


def pca_ridge(x_train, y_train, x_test, alpha, components):
    """Snapshot PCA: eigendecompose n-by-n Gram, never a weight-sized covariance."""
    mean_x, std_x = x_train.mean(axis=0), x_train.std(axis=0)
    std_x = np.maximum(std_x, 1e-8)
    x = (x_train - mean_x) / std_x
    xt = (x_test - mean_x) / std_x
    y_mean = y_train.mean(axis=0)
    centered = y_train - y_mean
    eigenvalues, vectors = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(eigenvalues)[::-1]
    threshold = max(float(eigenvalues.max()), 1.0) * 1e-12
    order = order[eigenvalues[order] > threshold][:min(components, len(y_train) - 1)]
    if not len(order):
        return np.broadcast_to(y_mean, (len(x_test), len(y_mean))).copy()
    scores = vectors[:, order] * np.sqrt(eigenvalues[order])
    basis = (vectors[:, order].T @ centered) / np.sqrt(eigenvalues[order])[:, None]
    coefficients = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ scores)
    return y_mean + (xt @ coefficients) @ basis


def inner_groups(records, outer_mode):
    if outer_mode == "task":
        return [record["task"] for record in records]
    # Keep all deletion requests for a checkpoint together during inner validation.
    return [f"{record['task']}:{record['seed']}" for record in records]


def select_alpha(x, y, records, mode, grid, components):
    labels = np.array(inner_groups(records, mode))
    scores = {}
    for alpha in grid:
        error, count = 0.0, 0
        for label in sorted(set(labels)):
            training, validation = np.flatnonzero(labels != label), np.flatnonzero(labels == label)
            if len(training) < 2:
                raise ValueError("Too few training groups for leakage-free ridge selection")
            prediction = pca_ridge(x[training], y[training], x[validation], alpha, components)
            error += float(np.sum((prediction - y[validation]) ** 2))
            count += len(validation)
        scores[float(alpha)] = error / count
    return min(scores, key=lambda alpha: (scores[alpha], alpha)), scores


def shuffle_within_models(x, records, seed=991):
    rng = np.random.default_rng(seed)
    result = x.copy()
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[(record["task"], record["seed"])].append(i)
    for indices in groups.values():
        # Derangement avoids leaving a request's own conditioning as a "shuffle".
        order = rng.permutation(indices)
        source = np.roll(order, 1)
        result[order] = x[source]
    return result


def layer_metrics(predicted, observed):
    p, y = predicted.astype(np.float64).ravel(), observed.astype(np.float64).ravel()
    error = float(np.sum((p - y) ** 2))
    target = float(y @ y)
    prediction = float(p @ p)
    cosine = float(p @ y / np.sqrt(target * prediction)) if target > 0 and prediction > 0 else None
    return {"sse": error, "target_squared_norm": target, "predicted_squared_norm": prediction,
            "dot": float(p @ y), "cosine": cosine, "gain_over_zero": 1 - error / target if target > 1e-20 else None,
            "coordinates": len(y)}


def predict(manifest):
    root = Path(manifest["config"]["output"])
    index = read_json(root / "pairs" / "index.json")
    if index["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Extracted collection does not match the manifest")
    records = index["pairs"]
    expected = len(manifest["tasks"]) * len(manifest["config"]["seeds"]) * len(manifest["config"]["request_seeds"])
    if len(records) != expected:
        raise ValueError("Incomplete pair collection")
    layers = records[0]["layers"]
    if any(record["layers"] != layers for record in records):
        raise ValueError("Incompatible adapter coordinates across models")
    out = root / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows, audits = [], []
    grid = manifest["config"]["prediction"]["ridge_grid"]
    components = manifest["config"]["prediction"]["components"]
    # Load a single layer at a time: full adapter collections can exceed host RAM.
    for layer_index, name in enumerate(layers):
        ys, xs, gs = [], [], []
        for record in records:
            with np.load(root / "pairs" / f"{record['id']}.npz", allow_pickle=False) as data:
                ys.append(data[f"delta_{layer_index}"].astype(np.float64))
                xs.append(data[f"features_{layer_index}"].astype(np.float64))
                gs.append(data[f"gradient_{layer_index}"].astype(np.float64))
        shape = ys[0].shape
        y, x, g = np.stack([value.ravel() for value in ys]), np.stack(xs), np.stack([value.ravel() for value in gs])
        for mode, heldout, train_idx, test_idx in folds(records):
            train_records = [records[i] for i in train_idx]
            alpha, scores = select_alpha(x[train_idx], y[train_idx], train_records, mode, grid, components)
            learned = pca_ridge(x[train_idx], y[train_idx], x[test_idx], alpha, components)
            shuffled_x = shuffle_within_models(x[train_idx], train_records)
            shuffle_alpha, _ = select_alpha(shuffled_x, y[train_idx], train_records, mode, grid, components)
            shuffled = pca_ridge(shuffled_x, y[train_idx], x[test_idx], shuffle_alpha, components)
            mean = np.broadcast_to(y[train_idx].mean(axis=0), learned.shape).copy()
            denominator = float(np.sum(g[train_idx] ** 2))
            scale = float(np.sum(g[train_idx] * y[train_idx]) / denominator) if denominator > 1e-20 else 0.0
            scaled = scale * g[test_idx]
            audits.append({"mode": mode, "heldout": heldout, "layer": name, "train_ids": [records[i]["id"] for i in train_idx],
                           "test_ids": [records[i]["id"] for i in test_idx], "alpha": alpha,
                           "validation_sse": {str(k): v for k, v in scores.items()}, "gradient_scale": scale,
                           "components_requested": components, "components_maximum": min(components, len(train_idx) - 1)})
            for method, predicted in (("zero", np.zeros_like(mean)), ("mean", mean), ("scaled_gradient", scaled), ("ridge", learned), ("shuffled_ridge", shuffled)):
                for local, pair_index in enumerate(test_idx):
                    pair = records[pair_index]
                    prediction_id = f"{mode}__{heldout}__{method}__{pair['id']}"
                    directory = out / prediction_id
                    directory.mkdir(parents=True, exist_ok=True)
                    write_npz(directory / f"layer-{layer_index}.npz", {"change": predicted[local].reshape(shape).astype(np.float32)})
                    rows.append({"prediction_id": prediction_id, "mode": mode, "heldout": heldout, "method": method,
                                 "pair": pair["id"], "task": pair["task"], "seed": pair["seed"], "request": pair["request"], "layer": name,
                                 **layer_metrics(predicted[local], y[pair_index])})
    result = {"fingerprint": manifest["fingerprint"], "synthetic": manifest["provenance"]["synthetic"],
              "rows": rows, "audits": audits, "prediction_seconds": time.perf_counter() - start}
    write_json(out / "metrics.json", result)
    return result
