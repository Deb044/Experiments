"""Sample the CUDA device used by this process; never substitute a parent GPU.

NVML is optional. MIG utilization is often unsupported and stays null rather
than becoming a fabricated zero. PyTorch memory statistics remain available.
"""
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch

from .io import write_json


def job_environment():
    keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_QOS",
            "SLURM_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
            "CUDA_VISIBLE_DEVICES", "VLA_JOB_LEVEL")
    return {key: os.environ.get(key) for key in keys}


def statistics(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not values:
        return {"samples": 0, "mean": None, "p95": None, "max": None}
    return {"samples": len(values), "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)), "max": max(values)}


class GpuTelemetry(AbstractContextManager):
    def __init__(self, directory, device="cuda", interval=2.0, scope="command"):
        if interval <= 0:
            raise ValueError("Telemetry interval must be positive")
        self.directory = Path(directory) if directory is not None else None
        self.device = torch.device(device)
        self.interval = interval
        self.scope = scope
        self.rows = []
        self.errors = set()
        self.stop_event = threading.Event()
        self.nvml = self.handle = self.identity = None
        self.nvml_attempted = False
        self.thread = self.stream = None
        self.summary = None

    def __enter__(self):
        self.start = time.perf_counter()
        if self.directory is None:
            return self
        self.directory.mkdir(parents=True, exist_ok=True)
        # A job restart must not erase the previous segment's telemetry.
        self.segment = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
        self.samples_path = self.directory / f"gpu-{self.segment}.jsonl"
        self.summary_path = self.directory / f"gpu-{self.segment}-summary.json"
        self.stream = self.samples_path.open("x", encoding="utf-8", buffering=1)
        self._sample()
        self.thread = threading.Thread(target=self._loop, name="gpu-telemetry", daemon=True)
        self.thread.start()
        return self

    def _connect_nvml(self, props):
        if self.nvml_attempted:
            return
        self.nvml_attempted = True
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            uuid = getattr(props, "uuid", None)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            # CUDA ordinals are remapped by Slurm. A numeric visibility value is
            # insufficient evidence for selecting an NVML physical device.
            if index < len(visible) and visible[index].startswith("MIG-"):
                uuid = visible[index]
            elif not uuid and index < len(visible) and visible[index].startswith("GPU-"):
                uuid = visible[index]
            if not uuid or not str(uuid).startswith(("GPU-", "MIG-")):
                raise RuntimeError("CUDA device UUID unavailable; refusing a physical-index fallback")
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            identity = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(identity, bytes):
                identity = identity.decode()
            if not identity.startswith("MIG-"):
                # A CUDA MIG slice can expose the parent UUID on some stacks.
                # Its utilization/memory must not become this job's readings.
                try:
                    if pynvml.nvmlDeviceGetMigMode(handle)[0]:
                        raise RuntimeError("Resolved a MIG parent, not the allocated slice")
                except pynvml.NVMLError_NotSupported:
                    pass
            self.handle = handle
            self.identity = identity
            if isinstance(self.identity, bytes):
                self.identity = self.identity.decode()
        except Exception as error:
            self.errors.add(f"NVML identity: {type(error).__name__}: {error}")

    def _sample(self):
        row = {"utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.perf_counter() - self.start,
               "cuda_ready": False, "device_uuid": self.identity,
               "utilization_gpu_percent": None, "utilization_memory_percent": None,
               "device_used_gb": None, "device_total_gb": None,
               "torch_allocated_gb": None, "torch_reserved_gb": None,
               "torch_peak_allocated_gb": None, "torch_peak_reserved_gb": None}
        try:
            if self.device.type == "cuda" and torch.cuda.is_initialized():
                props = torch.cuda.get_device_properties(self.device)
                free, total = torch.cuda.mem_get_info(self.device)
                row.update(cuda_ready=True, device_name=props.name, device_total_gb=props.total_memory / 2**30,
                           device_used_gb=(total - free) / 2**30,
                           torch_allocated_gb=torch.cuda.memory_allocated(self.device) / 2**30,
                           torch_reserved_gb=torch.cuda.memory_reserved(self.device) / 2**30,
                           torch_peak_allocated_gb=torch.cuda.max_memory_allocated(self.device) / 2**30,
                           torch_peak_reserved_gb=torch.cuda.max_memory_reserved(self.device) / 2**30)
                self._connect_nvml(props)
                row["device_uuid"] = self.identity
                if self.handle is not None:
                    try:
                        memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                        row.update(device_used_gb=memory.used / 2**30)
                    except Exception as error:
                        self.errors.add(f"NVML memory: {type(error).__name__}: {error}")
                    try:
                        utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                        row.update(utilization_gpu_percent=float(utilization.gpu),
                                   utilization_memory_percent=float(utilization.memory))
                    except Exception as error:
                        self.errors.add(f"NVML utilization: {type(error).__name__}: {error}")
        except Exception as error:
            self.errors.add(f"CUDA telemetry: {type(error).__name__}: {error}")
        self.rows.append(row)
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self._sample()
            except Exception as error:
                self.errors.add(f"Telemetry write: {type(error).__name__}: {error}")
                break

    def __exit__(self, exc_type, exc_value, traceback):
        if self.directory is None:
            return False
        self.stop_event.set()
        self.thread.join()
        try:
            self._sample()
            metrics = {key: statistics([row[key] for row in self.rows]) for key in self.rows[0]
                       if key.endswith(("_gb", "_percent"))}
            self.summary = {"scope": self.scope, "interval_seconds": self.interval,
                            "wall_seconds": time.perf_counter() - self.start,
                            "sample_count": len(self.rows), "metrics": metrics,
                            "job": job_environment(), "device_uuid": self.identity,
                            "samples_file": str(self.samples_path), "errors": sorted(self.errors),
                            "exit": "success" if exc_type is None else "error",
                            "exception_type": exc_type.__name__ if exc_type else None,
                            "oom": bool(exc_type and issubclass(exc_type, torch.cuda.OutOfMemoryError)),
                            "note": "NVML readings belong to the resolved device, not exclusively this process. "
                                    "Unsupported readings are null. Memory utilization is bandwidth activity, not VRAM occupancy. "
                                    "Sampled device memory can miss transient peaks; PyTorch allocator peaks cover tensor allocations."}
            write_json(self.summary_path, self.summary)
        finally:
            self.stream.close()
            if self.nvml is not None:
                try:
                    self.nvml.nvmlShutdown()
                except Exception:
                    pass
        return False
