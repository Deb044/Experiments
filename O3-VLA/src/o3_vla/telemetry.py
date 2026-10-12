"""Allocated-slice telemetry; unsupported MIG metrics stay null."""

import importlib.metadata
import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

from .io import write_json


def hardware():
    packages = {}
    for name in ["torch", "transformers", "bitsandbytes", "numpy", "h5py", "timm", "tensorflow-cpu"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "packages": packages,
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update(device_name=props.name, total_vram_gib=props.total_memory / 2**30)
    return result


class Telemetry:
    def __init__(self, cfg, phase):
        self.cfg, self.phase = cfg, phase
        self.stop = threading.Event()
        self.samples, self.errors = [], set()
        self.nvml, self.handle = None, None

    def __enter__(self):
        self.started = time.monotonic()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = self.cfg.root / "telemetry" / f"{self.phase}-{stamp}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            try:
                import pynvml

                pynvml.nvmlInit()
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
                cuda_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
                identity = visible if visible.startswith(("MIG-", "GPU-")) else cuda_uuid
                if not identity:
                    raise RuntimeError("no UUID mapping; refuse node-wide/index-ambiguous NVML readings")
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(identity))
                self.nvml = pynvml
            except Exception as error:
                self.errors.add(f"NVML resolution: {type(error).__name__}: {error}")
        self.sample()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        sample = {
            "seconds": time.monotonic() - self.started,
            "host_rss_gib": psutil.Process().memory_info().rss / 2**30,
            "device_used_gib": None,
            "gpu_utilization_percent": None,
            "memory_bandwidth_utilization_percent": None,
        }
        if torch.cuda.is_available():
            sample.update(
                torch_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                torch_reserved_gib=torch.cuda.memory_reserved() / 2**30,
            )
        if self.handle is not None:
            try:
                sample["device_used_gib"] = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**30
            except Exception as error:
                self.errors.add(f"NVML memory: {type(error).__name__}: {error}")
            try:
                usage = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                sample["gpu_utilization_percent"] = usage.gpu
                sample["memory_bandwidth_utilization_percent"] = usage.memory
            except Exception as error:
                self.errors.add(f"NVML utilization: {type(error).__name__}: {error}")
        self.samples.append(sample)

    def collect(self):
        while not self.stop.wait(self.cfg.telemetry_interval_seconds):
            self.sample()

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join()
        self.sample()
        summary = {}
        for key in self.samples[0]:
            values = [s[key] for s in self.samples if s.get(key) is not None]
            summary[key] = {
                "samples": len(values),
                "mean": float(np.mean(values)) if values else None,
                "max": float(np.max(values)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
            }
        payload = {
            "phase": self.phase,
            "wall_seconds": time.monotonic() - self.started,
            "exception": None if error is None else f"{kind.__name__}: {error}",
            "oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "metrics": summary,
            "errors": sorted(self.errors),
            "job": {
                key: os.environ.get(key)
                for key in [
                    "SLURM_JOB_ID",
                    "SLURM_JOB_PARTITION",
                    "SLURM_JOB_QOS",
                    "SLURM_CPUS_ON_NODE",
                    "CUDA_VISIBLE_DEVICES",
                    "O3_JOB_LEVEL",
                ]
            },
            "note": "NVML is allocated-device scope, not exclusively this process; MIG utilization may be unsupported",
        }
        if torch.cuda.is_available():
            payload.update(
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
        write_json(str(self.path) + ".json", payload)
        Path(str(self.path) + ".jsonl").write_text(
            "".join(json.dumps(s, allow_nan=False) + "\n" for s in self.samples), encoding="utf-8"
        )
        # NVML teardown after the sampler stops. Errors are diagnostics, not missing experiment failures.
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
