"""Matched-noise functional checks of predicted effective updates."""
from pathlib import Path
import time

import numpy as np
import torch

from .backends import create_backend
from .extraction import completed_factors
from .features import deterministic_seed
from .io import read_json, read_npz, write_json, write_npz
from .lora import apply_updates, effective_updates, load_factors
from .runtime import precision_context


def flow_probes(backend, episodes, seed):
    cfg = backend.cfg
    count, draws = cfg["probes"]["windows"], cfg["probes"]["draws"]
    microbatch = cfg["training"]["microbatch"]
    all_outputs, losses, valid_masks = [], [], []
    backend.policy.eval()
    with torch.no_grad():
        for draw in range(draws):
            # Same windows for every noise draw and every compared policy.
            rng = np.random.default_rng(seed)
            generator = torch.Generator(device=backend.device).manual_seed(seed + draw)
            outputs = []
            masks = []
            weighted_loss = 0.0
            for start in range(0, count, microbatch):
                n = min(microbatch, count - start)
                batch = backend.sample(episodes, n, rng)
                noise, time_value = backend.random_flow(batch, generator)
                with precision_context(cfg, backend.device):
                    output, loss = backend.flow_and_loss(batch, noise, time_value)
                outputs.append(output.cpu().float().numpy())
                pad = batch.get("action_is_pad")
                masks.append(np.ones(output.shape[:2], dtype=bool) if pad is None else (~pad).cpu().numpy())
                weighted_loss += float(loss) * n / count
            all_outputs.append(np.concatenate(outputs))
            valid_masks.append(np.concatenate(masks))
            losses.append(weighted_loss)
    return np.stack(all_outputs), np.stack(valid_masks), float(np.mean(losses))


def functional_mse(prediction, reference, valid):
    mask = np.broadcast_to(valid[..., None], prediction.shape)
    if not mask.any():
        raise ValueError("No valid action entries in functional probes")
    return float(np.mean((prediction.astype(np.float64)[mask] - reference.astype(np.float64)[mask]) ** 2))


def evaluate(manifest):
    cfg = manifest["config"]
    root = Path(cfg["output"])
    prediction_metrics = read_json(root / "predictions" / "metrics.json")
    if prediction_metrics["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Predictions were generated under a different manifest")
    records = read_json(root / "pairs" / "index.json")["pairs"]
    by_pair = {}
    for row in prediction_metrics["rows"]:
        by_pair.setdefault(row["pair"], {})[row["prediction_id"]] = row
    rows = []
    start_time = time.perf_counter()
    reference_root = root / "functional" / "references"
    reference_root.mkdir(parents=True, exist_ok=True)
    backend, group = None, None
    for pair in records:
        key = (pair["task"], pair["seed"])
        if key != group:
            if backend is not None:
                backend.close()
                del backend
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            backend = create_backend(manifest, *key)
            group = key
        task, seed, request = pair["task"], pair["seed"], pair["request"]
        full_values, _ = completed_factors(manifest, f"{task}/seed-{seed}/full")
        clean_values, _ = completed_factors(manifest, f"{task}/seed-{seed}/clean-{request}")
        full = effective_updates(full_values)
        probe_seed = deterministic_seed(cfg["probes"]["seed"], task, request, "heldout-flow")
        episodes = manifest["tasks"][task]["heldout"]
        cache = reference_root / f"{pair['id']}.npz"
        cache_meta = cache.with_suffix(".json")
        compatible_cache = False
        if cache.exists() and cache_meta.exists():
            info = read_json(cache_meta)
            if info["fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Stale functional reference")
            compatible_cache = info.get("implementation") == "dense-effective-flow-v1"
        if compatible_cache:
            reference = read_npz(cache)
        else:
            load_factors(backend.modules, full_values)
            apply_updates(backend.modules, full)
            full_flow, valid, full_loss = flow_probes(backend, episodes, probe_seed)
            load_factors(backend.modules, clean_values)
            apply_updates(backend.modules, effective_updates(clean_values))
            clean_flow, clean_valid, clean_loss = flow_probes(backend, episodes, probe_seed)
            if not np.array_equal(valid, clean_valid):
                raise ValueError("Probe windows were not paired")
            reference = {"full": full_flow, "clean": clean_flow, "valid": valid}
            write_npz(cache, reference)
            info = {"fingerprint": manifest["fingerprint"], "implementation": "dense-effective-flow-v1", "full_loss": full_loss, "clean_loss": clean_loss,
                    "original_discrepancy": functional_mse(full_flow, clean_flow, valid)}
            write_json(cache_meta, info)
        for prediction_id, row in sorted(by_pair[pair["id"]].items()):
            updates = {}
            for layer_index, name in enumerate(pair["layers"]):
                change = read_npz(root / "predictions" / prediction_id / f"layer-{layer_index}.npz")["change"]
                updates[name] = full[name] + change
            load_factors(backend.modules, full_values)
            apply_updates(backend.modules, updates)
            predicted_flow, valid, loss = flow_probes(backend, episodes, probe_seed)
            discrepancy = functional_mse(predicted_flow, reference["clean"], valid)
            original = info["original_discrepancy"]
            rows.append({"prediction_id": prediction_id, "pair": pair["id"], "task": task, "seed": seed, "request": request,
                         "mode": row["mode"], "method": row["method"], "discrepancy": discrepancy,
                         "original_discrepancy": original, "gain_over_full": 1 - discrepancy / original if original > 1e-20 else None,
                         "predicted_probe_loss": loss, "full_probe_loss": info["full_loss"], "clean_probe_loss": info["clean_loss"]})
        print(f"Functional checks: {pair['id']}", flush=True)
    if backend is not None:
        backend.close()
    result = {"fingerprint": manifest["fingerprint"], "synthetic": manifest["provenance"]["synthetic"], "rows": rows,
              "evaluation_seconds": time.perf_counter() - start_time,
              "note": "Held-out native-flow checks, not rollout success or a deletion certificate."}
    write_json(root / "functional" / "metrics.json", result)
    return result
