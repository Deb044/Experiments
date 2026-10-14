"""Time complete updates on the actual configured backend, not theoretical FLOPs."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import time

import numpy as np
import torch

from .backends import create_backend
from .io import digest, write_json
from .runtime import hardware_info, optimizer_for, seed_everything, synchronize, timing_summary, update
from .telemetry import GpuTelemetry, job_environment


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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = Path(cfg["output"]) / "benchmarks" / stamp
    # MIG hides utilization, so time the input pipeline itself: a GPU that waits on data is idle.
    draw, sampling = backend.sample, []

    def timed_sample(*arguments):
        begin = time.perf_counter()
        batch = draw(*arguments)
        synchronize(backend.device)
        sampling.append(time.perf_counter() - begin)
        return batch

    backend.sample = timed_sample
    with GpuTelemetry(directory, backend.device, interval=2, scope="benchmark-measured-updates") as monitor:
        times = [update(backend, optimizer, episodes, rng)[1] for _ in range(cfg["benchmark"]["updates"])]
    backend.sample = draw
    summary = timing_summary(times)
    summary["sampling_mean_seconds"] = float(sum(sampling) / len(times))
    summary["sampling_fraction"] = float(sum(sampling) / sum(times))
    run_count = len(manifest["tasks"]) * len(cfg["seeds"]) * (1 + len(cfg["request_seeds"]))
    result = {"fingerprint": manifest["fingerprint"], "synthetic": manifest["provenance"]["synthetic"],
              "effective_batch": cfg["training"]["effective_batch"], "microbatch": cfg["training"]["microbatch"],
              "accumulation": cfg["training"]["effective_batch"] // cfg["training"]["microbatch"],
              "precision": cfg["training"]["precision"], "backend": cfg["backend"], "hardware": hardware_info(backend.device),
              "job": job_environment(), "gpu_telemetry": monitor.summary, "data_cache": getattr(backend, "cache_info", None),
              "timing": summary, "benchmark_wall_seconds": time.perf_counter() - start,
              "samples_per_second": cfg["training"]["effective_batch"] / summary["mean_seconds"],
              "adapter_runs": run_count, "training_hours_estimate": run_count * cfg["training"]["steps"] * summary["mean_seconds"] / 3600,
              "note": "Estimate excludes loading, checkpoint I/O, extraction, prediction and evaluation; contention may change."}
    # Hardware and output locations can differ, but compared workloads must match.
    workload = copy.deepcopy(cfg)
    workload.pop("output")
    workload["training"].pop("device")
    provenance = {key: value for key, value in manifest["provenance"].items() if key != "dataset_root"}
    result["workload_key"] = digest({"config": workload, "provenance": provenance,
                                     "task": task, "episodes": episodes})
    write_json(directory / "benchmark.json", result)
    write_json(Path(cfg["output"]) / "benchmark.json", result)
    backend.close()
    return result
