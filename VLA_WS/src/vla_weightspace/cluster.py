"""Generate immutable scratch configs and compare measured GPU job levels."""
import argparse
import csv
from pathlib import Path
import sys

from .io import read_json, write_json


def scratch_config(source, destination, output, microbatch=None):
    cfg = read_json(source)
    cfg["output"] = str(Path(output).resolve())
    if microbatch is not None:
        if microbatch < 1 or cfg["training"]["effective_batch"] % microbatch:
            raise ValueError("microbatch must be positive and divide effective_batch")
        cfg["training"]["microbatch"] = microbatch
    destination = Path(destination)
    if destination.exists():
        if read_json(destination) != cfg:
            raise ValueError("Existing scratch config differs; choose a new VLA_EXPERIMENT or restore the original settings")
    else:
        write_json(destination, cfg)
    return destination


def compare(paths, margin=0.2):
    if not 0 < margin < 1:
        raise ValueError("headroom must be between zero and one")
    rows = []
    workload = None
    for path in paths:
        record = read_json(path)
        if record["synthetic"]:
            raise ValueError("Synthetic CPU fixtures cannot guide GPU job selection")
        key = record.get("workload_key")
        if not key:
            raise ValueError(f"{path}: rerun benchmark with workload provenance")
        if workload is not None and key != workload:
            raise ValueError("Workloads differ (including microbatch or immutable revisions); compare matching protocols only")
        workload = key
        gpu = record["gpu_telemetry"]
        metrics = gpu["metrics"]
        capacity = record["hardware"].get("total_gb")
        reserved = metrics["torch_peak_reserved_gb"]["max"]
        device_used = metrics["device_used_gb"]["max"]
        peak = max(x for x in (reserved, device_used) if x is not None) if any(x is not None for x in (reserved, device_used)) else None
        fits = None if not capacity or peak is None else peak <= capacity * (1 - margin)
        rows.append({"file": str(path), "level": record["job"].get("VLA_JOB_LEVEL"),
                     "job_id": record["job"].get("SLURM_JOB_ID"), "capacity_gib": capacity,
                     "microbatch": record["microbatch"], "mean_update_s": record["timing"]["mean_seconds"],
                     "p95_update_s": record["timing"]["p95_seconds"], "sampling_fraction": record["timing"].get("sampling_fraction"),
                     "samples_per_s": record["samples_per_second"],
                     "training_hours": record["training_hours_estimate"], "torch_peak_reserved_gib": reserved,
                     "sampled_device_peak_gib": device_used,
                     "mean_gpu_util_percent": metrics["utilization_gpu_percent"]["mean"],
                     "util_samples": metrics["utilization_gpu_percent"]["samples"],
                     "vram_headroom_pass": fits, "required_headroom_fraction": margin})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    config = sub.add_parser("config")
    config.add_argument("--source", required=True)
    config.add_argument("--destination", required=True)
    config.add_argument("--output", required=True)
    config.add_argument("--microbatch", type=int)
    comparison = sub.add_parser("compare")
    comparison.add_argument("benchmarks", nargs="+")
    comparison.add_argument("--headroom", type=float, default=0.2)
    comparison.add_argument("--csv")
    args = parser.parse_args(argv)
    if args.command == "config":
        print(scratch_config(args.source, args.destination, args.output, args.microbatch))
        return
    rows = compare(args.benchmarks, args.headroom)
    stream = open(args.csv, "w", newline="", encoding="utf-8") if args.csv else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.csv:
            stream.close()
    print("Compare repeated measurements: choose the smallest measured level with VRAM headroom and acceptable runtime. "
          "Blank utilization means unavailable, not idle. These are benchmark-training measurements; "
          "check whole-job telemetry for extraction/evaluation peaks before the full pilot.", file=sys.stderr)


if __name__ == "__main__":
    main()
