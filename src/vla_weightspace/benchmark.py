"""Time complete updates on the actual configured backend, not theoretical FLOPs."""
from pathlib import Path
import time

import numpy as np
import torch

from .backends import create_backend
from .io import write_json
from .runtime import hardware_info, optimizer_for, seed_everything, timing_summary, update


def benchmark(manifest):
    cfg = manifest["config"]
    task = next(iter(manifest["tasks"]))
    backend = create_backend(manifest, task, cfg["seeds"][0])
    backend.policy.eval()
    seed_everything(cfg["seeds"][0])
    rng = np.random.default_rng(1234)
    optimizer = optimizer_for(backend)
    episodes = manifest["tasks"][task]["train"]
    start = time.perf_counter()
    for _ in range(cfg["benchmark"]["warmup"]):
        update(backend, optimizer, episodes, rng)
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    times = [update(backend, optimizer, episodes, rng)[1] for _ in range(cfg["benchmark"]["updates"])]
    summary = timing_summary(times)
    run_count = len(manifest["tasks"]) * len(cfg["seeds"]) * (1 + len(cfg["request_seeds"]))
    result = {"fingerprint": manifest["fingerprint"], "synthetic": manifest["provenance"]["synthetic"],
              "effective_batch": cfg["training"]["effective_batch"], "microbatch": cfg["training"]["microbatch"],
              "accumulation": cfg["training"]["effective_batch"] // cfg["training"]["microbatch"],
              "precision": cfg["training"]["precision"], "backend": cfg["backend"], "hardware": hardware_info(backend.device),
              "timing": summary, "benchmark_wall_seconds": time.perf_counter() - start,
              "samples_per_second": cfg["training"]["effective_batch"] / summary["mean_seconds"],
              "adapter_runs": run_count, "training_hours_estimate": run_count * cfg["training"]["steps"] * summary["mean_seconds"] / 3600,
              "note": "Estimate excludes loading, checkpoint I/O, extraction, prediction and evaluation; contention may change."}
    write_json(Path(cfg["output"]) / "benchmark.json", result)
    backend.close()
    return result
