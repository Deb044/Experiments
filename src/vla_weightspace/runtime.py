"""Device-aware timing and bounded training helpers."""
from contextlib import nullcontext
import platform
import subprocess
import time

import numpy as np
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_context(cfg, device):
    if cfg["training"]["precision"] == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported on this CUDA device")
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def hardware_info(device):
    result = {"platform": platform.platform(), "torch": torch.__version__, "device": str(device), "cuda_runtime": torch.version.cuda}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(name=props.name, total_gb=props.total_memory / 2**30,
                      free_gb=torch.cuda.mem_get_info(device)[0] / 2**30,
                      peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30,
                      peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 2**30)
        try:
            result["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.free,power.limit,driver_version", "--format=csv"], text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            result["nvidia_smi"] = "unavailable"
    return result


def optimizer_for(backend):
    cfg = backend.cfg["training"]
    return torch.optim.AdamW([p for p in backend.policy.parameters() if p.requires_grad], lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)


def update(backend, optimizer, episodes, rng):
    cfg = backend.cfg["training"]
    accumulation = cfg["effective_batch"] // cfg["microbatch"]
    synchronize(backend.device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for _ in range(accumulation):
        batch = backend.sample(episodes, cfg["microbatch"], rng)
        with precision_context(backend.cfg, backend.device):
            loss = backend.loss(batch) / accumulation
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite native training loss")
        loss.backward()
        total_loss += float(loss.detach())
    torch.nn.utils.clip_grad_norm_([p for p in backend.policy.parameters() if p.requires_grad], 10.0)
    optimizer.step()
    synchronize(backend.device)
    return total_loss, time.perf_counter() - start


def timing_summary(values):
    return {"updates": len(values), "mean_seconds": float(np.mean(values)), "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95)), "total_seconds": float(sum(values))}

