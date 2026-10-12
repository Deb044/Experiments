"""Train-only scaling, validation-only threshold calibration, held-out trajectory metrics."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import Windows
from .io import read_json, write_json


def calibrate_threshold(probabilities, labels, maximum_fpr):
    probabilities, labels = np.asarray(probabilities), np.asarray(labels, dtype=bool)
    negative = probabilities[~labels]
    if not len(negative) or not np.any(labels):
        raise ValueError("threshold calibration requires both classes")
    # Smallest threshold obeying the constraint, including ties; nextafter makes all-negative
    # rejection possible rather than returning 1 and accidentally accepting a probability of 1.
    candidates = np.unique(np.concatenate(([0.0], np.nextafter(probabilities, np.inf))))
    feasible = [x for x in candidates if np.mean(negative >= x) <= maximum_fpr]
    return float(min(feasible))


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities)
    predicted = probabilities >= threshold
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "recall": float(np.mean(predicted[labels])),
        "false_positive_rate": float(np.mean(predicted[~labels])),
        "precision": float(np.mean(labels[predicted])) if predicted.any() else None,
        "positive_windows": int(labels.sum()),
        "negative_windows": int((~labels).sum()),
    }


class Detector:
    def __init__(self, payload):
        self.payload = payload
        self.threshold = payload["threshold"]

    def probability(self, visual, text):
        x = np.asarray(text if self.payload["feature_type"] == "text" else np.concatenate([visual, text]))
        scaled = (x - np.asarray(self.payload["mean"])) / np.asarray(self.payload["scale"])
        if self.payload["kind"] == "logistic":
            logit = float(np.dot(scaled, self.payload["coefficient"]) + self.payload["intercept"])
        else:
            positive, negative = np.asarray(self.payload["positive"]), np.asarray(self.payload["negative"])
            logit = float(np.square(scaled - negative).mean() - np.square(scaled - positive).mean())
        return float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))


def cache_features(cfg, backend, manifest, deadline):
    path = cfg.root / "features.npz"
    metadata = cfg.root / "features.json"
    if path.exists() and metadata.exists():
        if read_json(metadata)["manifest_digest"] != manifest["digest"]:
            raise ValueError("stale feature cache")
        return
    windows = Windows(cfg)
    backend.adapters.gate(0)
    visual, text = [], []
    try:
        for index, record in enumerate(manifest["records"]):
            deadline.check()
            v, t = backend.features(windows.get(record))
            visual.append(v)
            text.append(t)
            if index % 100 == 0:
                print(f"Frozen features: {index}/{len(manifest['records'])}", flush=True)
        np.savez_compressed(path, visual=np.asarray(visual), text=np.asarray(text))
        write_json(
            metadata,
            {
                "manifest_digest": manifest["digest"],
                "keys": [r["key"] for r in manifest["records"]],
                "source": "frozen pooled vision patches + prompt input embeddings; no future/action labels",
            },
        )
    finally:
        windows.close()


def fit_detectors(cfg, manifest):
    arrays = np.load(cfg.root / "features.npz", allow_pickle=False)
    records = manifest["records"]
    if read_json(cfg.root / "features.json")["keys"] != [r["key"] for r in records]:
        raise ValueError("feature/record alignment mismatch")
    rows = []
    for stage in range(1, 4):
        eligible = np.array([r["role"] == "retain" or 0 < r["stage"] <= stage for r in records])
        labels = np.array([0 < r["stage"] <= stage for r in records], dtype=int)
        splits = {
            name: eligible & np.array([r["split"] == name for r in records])
            for name in ["train", "validation", "test"]
        }
        for feature_type, kind in [
            ("multimodal", "logistic"),
            ("text", "logistic"),
            ("multimodal", "prototype"),
        ]:
            features = (
                arrays["text"]
                if feature_type == "text"
                else np.concatenate([arrays["visual"], arrays["text"]], axis=1)
            )
            scaler = StandardScaler().fit(features[splits["train"]])
            transformed = scaler.transform(features)
            payload = {
                "stage": stage,
                "feature_type": feature_type,
                "kind": kind,
                "manifest_digest": manifest["digest"],
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "threshold": 0.5,
            }
            if kind == "logistic":
                model = LogisticRegression(
                    C=1.0, max_iter=500, class_weight="balanced", random_state=cfg.seed
                ).fit(transformed[splits["train"]], labels[splits["train"]])
                payload.update(coefficient=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))
            else:
                payload.update(
                    positive=transformed[splits["train"] & (labels == 1)].mean(0).tolist(),
                    negative=transformed[splits["train"] & (labels == 0)].mean(0).tolist(),
                )
            detector = Detector(payload)
            probabilities = np.array(
                [detector.probability(v, t) for v, t in zip(arrays["visual"], arrays["text"])]
            )
            valid = splits["validation"]
            payload["threshold"] = calibrate_threshold(
                probabilities[valid], labels[valid], cfg.validation_false_positive_rate
            )
            test = splits["test"]
            row = {
                "stage": stage,
                "feature_type": feature_type,
                "kind": kind,
                "threshold": payload["threshold"],
                **binary_metrics(labels[test], probabilities[test], payload["threshold"]),
            }
            # Counts and per-trajectory means expose frame correlation; no frame-level confidence intervals.
            grouped = {}
            trajectory_labels = {}
            for i in np.flatnonzero(test):
                key = f"{records[i]['task_id']}/{records[i]['trajectory']}"
                grouped.setdefault(key, []).append(float(probabilities[i]))
                trajectory_labels[key] = int(labels[i])
            row["trajectory_mean_probabilities"] = {k: float(np.mean(v)) for k, v in grouped.items()}
            row.update(
                {
                    f"trajectory_{key}": value
                    for key, value in binary_metrics(
                        [trajectory_labels[k] for k in grouped],
                        [float(np.mean(v)) for v in grouped.values()],
                        payload["threshold"],
                    ).items()
                }
            )
            rows.append(row)
            write_json(cfg.root / "detectors" / f"stage-{stage}-{feature_type}-{kind}.json", payload)
    write_json(cfg.root / "detector-metrics.json", rows)


def load_detector(cfg, stage=3):
    payload = read_json(cfg.root / "detectors" / f"stage-{stage}-multimodal-logistic.json")
    if payload["manifest_digest"] != read_json(cfg.root / "manifest.json")["digest"]:
        raise ValueError("detector does not belong to this manifest")
    return Detector(payload)
