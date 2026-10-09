"""Tables and plots, with synthetic provenance and pilot limitations visible."""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from .io import read_json, write_json


def csv_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["method"])].append(row)
    result = []
    for (mode, method), group in sorted(groups.items()):
        sse = sum(row["sse"] for row in group)
        target = sum(row["target_squared_norm"] for row in group)
        pred = sum(row["predicted_squared_norm"] for row in group)
        dot = sum(row["dot"] for row in group)
        result.append({"mode": mode, "method": method, "sse": sse,
                       "relative_l2_error": np.sqrt(sse / target) if target > 1e-20 else None,
                       "cosine": dot / np.sqrt(target * pred) if target > 0 and pred > 0 else None,
                       "gain_over_zero": 1 - sse / target if target > 1e-20 else None,
                       "pairs": len({row["pair"] for row in group})})
    return result


def variability(manifest):
    root = Path(manifest["config"]["output"])
    records = read_json(root / "pairs" / "index.json")["pairs"]
    lookup = {(r["task"], r["seed"], r["request"]): r for r in records}
    result = []
    for task in manifest["tasks"]:
        seed_distances, request_distances = [], []
        layers = records[0]["layers"]
        for seed in manifest["config"]["seeds"]:
            requests = list(manifest["tasks"][task]["requests"])
            for i, left in enumerate(requests):
                for right in requests[i + 1:]:
                    a, b = lookup[(task, seed, left)], lookup[(task, seed, right)]
                    with np.load(root / "pairs" / f"{a['id']}.npz") as fa, np.load(root / "pairs" / f"{b['id']}.npz") as fb:
                        request_distances.append(sum(float(np.sum((fa[f"delta_{k}"].astype(np.float64) - fb[f"delta_{k}"]) ** 2)) for k in range(len(layers))))
        seeds = manifest["config"]["seeds"]
        for i, left in enumerate(seeds):
            for right in seeds[i + 1:]:
                for request in manifest["tasks"][task]["requests"]:
                    a, b = lookup[(task, left, request)], lookup[(task, right, request)]
                    with np.load(root / "pairs" / f"{a['id']}.npz") as fa, np.load(root / "pairs" / f"{b['id']}.npz") as fb:
                        seed_distances.append(sum(float(np.sum((fa[f"delta_{k}"].astype(np.float64) - fb[f"delta_{k}"]) ** 2)) for k in range(len(layers))))
        result.append({"task": task, "mean_between_request_squared_distance": float(np.mean(request_distances)),
                       "mean_between_seed_squared_distance": float(np.mean(seed_distances)) if seed_distances else None,
                       "request_comparisons": len(request_distances), "seed_comparisons": len(seed_distances)})
    return result


def report(manifest):
    root = Path(manifest["config"]["output"])
    metrics = read_json(root / "predictions" / "metrics.json")
    if metrics["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Stale prediction report input")
    rows = aggregate(metrics["rows"])
    functional_path = root / "functional" / "metrics.json"
    functional = read_json(functional_path) if functional_path.exists() else None
    if functional and functional["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Stale functional report input")
    if functional:
        for row in rows:
            selected = [value for value in functional["rows"] if value["mode"] == row["mode"] and value["method"] == row["method"]]
            row["mean_functional_discrepancy"] = float(np.mean([value["discrepancy"] for value in selected]))
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    csv_table(out / "prediction_summary.csv", rows)
    csv_table(out / "layer_metrics.csv", metrics["rows"])
    variation = variability(manifest)
    csv_table(out / "variability.csv", variation)
    if functional:
        csv_table(out / "functional_metrics.csv", functional["rows"])
    write_json(out / "summary.json", {"synthetic": manifest["provenance"]["synthetic"], "fingerprint": manifest["fingerprint"],
                                      "prediction": rows, "variability": variation, "functional_evaluated": functional is not None})
    _plots(metrics["rows"], out, manifest["provenance"]["synthetic"])
    label = "SYNTHETIC PIPELINE CHECK — NOT VLA RESEARCH RESULTS" if manifest["provenance"]["synthetic"] else "VLA weight-space predictability pilot"
    text = [f"# {label}", "", "Do deletion requests produce predictable changes in VLA weight space?", "",
            "This report concerns fixed-budget retained-only retraining of adapters, not certified unlearning.", "",
            "| Evaluation | Predictor | Relative L2 error | Cosine | Gain over zero |", "|---|---|---:|---:|---:|"]
    for row in rows:
        fmt = lambda value: "undefined" if value is None else f"{value:.4f}"
        text.append(f"| {row['mode']} | {row['method']} | {fmt(row['relative_l2_error'])} | {fmt(row['cosine'])} | {fmt(row['gain_over_zero'])} |")
    text += ["", "Assess ridge against zero, mean, scaled-gradient and shuffled conditioning **per held-out task and model seed**, not only in the pooled table.",
             "Layers and deletion requests from one model are not independent scientific replicates. No confidence claim is made from only two seeds and three tasks.",
             "Near-zero deletion changes yield undefined gains; a small prediction error alone is not evidence of deletion-specific structure.",
             "Functional checks: " + ("completed on matched held-out windows; see functional_metrics.csv." if functional else "NOT RUN. Run evaluate before interpreting functional quality."),
             "", "## Artifacts", "", "- prediction_summary.csv: pooled prediction metrics", "- layer_metrics.csv: task/seed/request/layer results",
             "- variability.csv: request variation versus seed variation", "- magnitude.png, layer_predictability.png, heldout_tasks.png: diagnostic plots", ""]
    benchmark = root / "benchmark.json"
    if benchmark.exists():
        hardware = read_json(benchmark)
        text += ["## Timing", "", f"Measured mean update: {hardware['timing']['mean_seconds']:.4f} s. Projected adapter collection: {hardware['training_hours_estimate']:.2f} h.",
                 "This estimate excludes extraction, evaluation, loading/checkpoints, and queue delays. Synthetic CPU timing does not estimate RTX PRO 6000 performance.", ""]
    (out / "results.md").write_text("\n".join(text), encoding="utf-8")
    return out


def _plots(rows, out, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suffix = " [SYNTHETIC]" if synthetic else ""
    selected = [row for row in rows if row["mode"] == "task" and row["method"] == "ridge"]
    fig, ax = plt.subplots(figsize=(6, 4))
    actual = np.sqrt([row["target_squared_norm"] for row in selected])
    predicted = np.sqrt([row["predicted_squared_norm"] for row in selected])
    ax.scatter(actual, predicted, s=12, alpha=0.6)
    top = max(max(actual), max(predicted), 1e-8)
    ax.plot([0, top], [0, top], "k--")
    ax.set(xlabel="Observed change norm", ylabel="Predicted change norm", title="Held-out task change magnitude" + suffix)
    fig.tight_layout()
    fig.savefig(out / "magnitude.png", dpi=160)
    plt.close(fig)
    by_layer = defaultdict(list)
    for row in selected:
        by_layer[row["layer"]].append(row)
    names = sorted(by_layer)
    gains = [1 - sum(row["sse"] for row in by_layer[name]) / max(sum(row["target_squared_norm"] for row in by_layer[name]), 1e-20) for name in names]
    fig, ax = plt.subplots(figsize=(max(6, min(14, len(names) * 0.15)), 4))
    ax.bar(range(len(names)), gains)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Adapter layer (order in layer_metrics.csv)", ylabel="Gain over zero", title="Layer-wise predictability" + suffix)
    fig.tight_layout()
    fig.savefig(out / "layer_predictability.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = ["mean", "scaled_gradient", "ridge", "shuffled_ridge"]
    tasks = sorted({row["task"] for row in selected})
    for method in methods:
        values = []
        for task in tasks:
            group = [row for row in rows if row["mode"] == "task" and row["task"] == task and row["method"] == method]
            values.append(1 - sum(row["sse"] for row in group) / max(sum(row["target_squared_norm"] for row in group), 1e-20))
        ax.plot(tasks, values, "o-", label=method)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="Gain over zero", title="Transfer to held-out tasks" + suffix)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "heldout_tasks.png", dpi=160)
    plt.close(fig)

