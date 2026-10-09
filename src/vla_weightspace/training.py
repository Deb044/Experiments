"""Paired training with exact optimizer, sampler, and Torch RNG resume."""
import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from .backends import create_backend
from .io import read_json, read_npz, write_json, write_npz
from .lora import factors, load_factors
from .manifest import run_specs
from .runtime import hardware_info, optimizer_for, seed_everything, timing_summary, update


def array_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def save_resume(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train_one(backend, spec, manifest, initial, max_updates=None):
    path = Path(backend.cfg["output"]) / "runs" / spec["id"]
    path.mkdir(parents=True, exist_ok=True)
    fingerprint = manifest["fingerprint"]
    initial_hash = array_digest(initial)
    done = path / "complete.json"
    if done.exists():
        metadata = read_json(done)
        if metadata["fingerprint"] != fingerprint or metadata["initial_hash"] != initial_hash:
            raise ValueError("Completed run has mismatched protocol or initialization")
        if array_digest(read_npz(path / "adapter.npz")) != metadata["adapter_hash"]:
            raise ValueError("Completed adapter artifact is damaged")
        return metadata
    load_factors(backend.modules, initial)
    # Adapter optimization does not require dropout in the frozen foundation model.
    backend.policy.eval()
    seed_everything(spec["seed"])
    rng = np.random.default_rng(spec["seed"])
    optimizer = optimizer_for(backend)
    step, losses, timings = 0, [], []
    resume = path / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint or state["initial_hash"] != initial_hash:
            raise ValueError("Resume checkpoint has mismatched protocol or initialization")
        load_factors(backend.modules, {key: value.numpy() for key, value in state["factors"].items()})
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["sampler_rng"]
        torch.set_rng_state(state["torch_rng"])
        if backend.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, losses, timings = state["step"], state["losses"], state["timings"]
    stop = backend.cfg["training"]["steps"]
    if max_updates is not None:
        stop = min(stop, step + max_updates)
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    while step < stop:
        loss, seconds = update(backend, optimizer, spec["episodes"], rng)
        step += 1
        losses.append(loss)
        timings.append(seconds)
        if step % backend.cfg["training"]["checkpoint_every"] == 0 or step == stop:
            save_resume(resume, {"fingerprint": fingerprint, "initial_hash": initial_hash, "step": step,
                                "factors": {key: torch.from_numpy(value) for key, value in factors(backend.modules).items()},
                                "optimizer": optimizer.state_dict(), "sampler_rng": rng.bit_generator.state,
                                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if backend.device.type == "cuda" else [],
                                "losses": losses, "timings": timings})
        if step % max(1, backend.cfg["training"]["checkpoint_every"]) == 0 or step == stop:
            print(f"{spec['id']}: {step}/{backend.cfg['training']['steps']} loss={loss:.5f}", flush=True)
    metadata = {"fingerprint": fingerprint, "initial_hash": initial_hash, "spec": spec, "steps": step,
                "losses": losses, "timing": timing_summary(timings), "hardware": hardware_info(backend.device),
                "synthetic": manifest["provenance"]["synthetic"]}
    if step == backend.cfg["training"]["steps"]:
        values = factors(backend.modules)
        write_npz(path / "adapter.npz", values)
        metadata["adapter_hash"] = array_digest(values)
        write_json(done, metadata)
    return metadata


def train(manifest, task_filter=None, run_filter=None, max_updates=None):
    specs = [spec for spec in run_specs(manifest) if (task_filter is None or spec["task"] == task_filter)
             and (run_filter is None or spec["id"] == run_filter)]
    if not specs:
        raise ValueError("No matching training runs")
    current = None
    backend = None
    for spec in specs:
        group = (spec["task"], spec["seed"])
        if group != current:
            if backend is not None:
                backend.close()
                del backend
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            backend = create_backend(manifest, *group)
            initial = factors(backend.modules)
            current = group
        train_one(backend, spec, manifest, initial, max_updates=max_updates)
    if backend is not None:
        backend.close()

