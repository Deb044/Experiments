"""Build actual paired effective-weight changes and loss-gradient features."""
from pathlib import Path
import time

import numpy as np
import torch

from .backends import create_backend
from .features import deterministic_seed, request_features
from .io import read_json, read_npz, write_json, write_npz
from .lora import effective_gradients, effective_updates, load_factors
from .runtime import precision_context
from .training import array_digest


def completed_factors(manifest, run_id):
    root = Path(manifest["config"]["output"]) / "runs" / run_id
    metadata = read_json(root / "complete.json")
    if metadata["fingerprint"] != manifest["fingerprint"]:
        raise ValueError(f"Stale training artifact: {run_id}")
    values = read_npz(root / "adapter.npz")
    if array_digest(values) != metadata["adapter_hash"]:
        raise ValueError(f"Damaged training artifact: {run_id}")
    return values, metadata


def gradient_summary(backend, episodes, seed):
    count = backend.cfg["probes"]["gradient_windows"]
    microbatch = backend.cfg["training"]["microbatch"]
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=backend.device).manual_seed(seed)
    total = {name: np.zeros(tuple(module.delta().shape), dtype=np.float32) for name, module in backend.modules.items()}
    for start in range(0, count, microbatch):
        n = min(microbatch, count - start)
        batch = backend.sample(episodes, n, rng)
        noise, time_value = backend.random_flow(batch, generator)
        with precision_context(backend.cfg, backend.device):
            gradients = effective_gradients(backend, batch, noise, time_value)
        for name in total:
            total[name] += gradients[name] * (n / count)
    return total


def extract(manifest):
    cfg = manifest["config"]
    root = Path(cfg["output"]) / "pairs"
    root.mkdir(parents=True, exist_ok=True)
    pairs = []
    start_time = time.perf_counter()
    for task, entry in manifest["tasks"].items():
        for seed in cfg["seeds"]:
            full_values, full_meta = completed_factors(manifest, f"{task}/seed-{seed}/full")
            full = effective_updates(full_values)
            backend = None
            for request, subsets in entry["requests"].items():
                pair_id = f"{task}__seed-{seed}__request-{request}"
                target_values, clean_meta = completed_factors(manifest, f"{task}/seed-{seed}/clean-{request}")
                if full_meta["initial_hash"] != clean_meta["initial_hash"]:
                    raise ValueError("Paired models did not start from identical adapters")
                if full_meta["steps"] != clean_meta["steps"]:
                    raise ValueError("Paired models have different training budgets")
                metadata_path = root / f"{pair_id}.json"
                npz_path = root / f"{pair_id}.npz"
                if metadata_path.exists():
                    record = read_json(metadata_path)
                    if record["fingerprint"] != manifest["fingerprint"] or record["full_hash"] != full_meta["adapter_hash"] or record["clean_hash"] != clean_meta["adapter_hash"] or not npz_path.exists():
                        raise ValueError("Stale/incomplete extracted pair")
                    pairs.append(record)
                    continue
                if backend is None:
                    backend = create_backend(manifest, task, seed)
                    load_factors(backend.modules, full_values)
                    backend.policy.eval()
                g_forget = gradient_summary(backend, subsets["forget"], deterministic_seed(cfg["probes"]["seed"], task, request, "forget"))
                g_retain = gradient_summary(backend, subsets["retain"], deterministic_seed(cfg["probes"]["seed"], task, request, "retain"))
                clean = effective_updates(target_values)
                if set(full) != set(clean):
                    raise ValueError("Paired models have different adapter layer layouts")
                arrays = {}
                layers = sorted(full)
                for i, name in enumerate(layers):
                    arrays[f"delta_{i}"] = clean[name] - full[name]
                    arrays[f"gradient_{i}"] = g_forget[name]
                    arrays[f"features_{i}"] = request_features(full[name], g_forget[name], g_retain[name])
                write_npz(npz_path, arrays)
                record = {"id": pair_id, "task": task, "seed": seed, "request": request, "layers": layers,
                          "fingerprint": manifest["fingerprint"], "full_hash": full_meta["adapter_hash"], "clean_hash": clean_meta["adapter_hash"],
                          "synthetic": manifest["provenance"]["synthetic"]}
                write_json(metadata_path, record)
                pairs.append(record)
                print(f"Extracted {pair_id}", flush=True)
            if backend is not None:
                backend.close()
                del backend
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_json(root / "index.json", {"fingerprint": manifest["fingerprint"], "pairs": pairs, "extraction_seconds": time.perf_counter() - start_time})
    return pairs

