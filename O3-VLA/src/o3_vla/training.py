"""Matched sequential/orthogonal training, measured budgets and exact update-boundary resume."""

import time

import numpy as np
import torch

from .actions import hold_action
from .data import Windows
from .io import load_torch, read_json, save_torch, write_json
from .telemetry import hardware

ARMS = ("sequential", "orthogonal")


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pool(manifest, stage):
    forget = [r for r in manifest["records"] if r["split"] == "train" and r["stage"] == stage]
    retain = [r for r in manifest["records"] if r["split"] == "train" and r["role"] == "retain"]
    if not forget or not retain:
        raise ValueError("empty current-request or allowed-task training pool")
    return forget, retain


def sample_batch(rng, forget, retain, size):
    # Each paired arm receives the same deterministic sample sequence.
    result = []
    for _ in range(size // 2):
        result.extend([forget[int(rng.integers(len(forget)))], retain[int(rng.integers(len(retain)))]])
    return result


def optimizer(backend, cfg):
    return torch.optim.AdamW(backend.adapters.parameters(), lr=cfg.learning_rate, weight_decay=0.0)


def update(backend, windows, batch, cfg, optim, previous=None, orthogonal=False):
    backend.training(True)
    optim.zero_grad(set_to_none=True)
    losses = {"forget_ce": [], "retain_ce": []}
    for record in batch:
        window = windows.get(record)
        target = (
            hold_action(window["gripper_qpos"], cfg.gripper_open_threshold)
            if (record["role"] == "forget")
            else window["action"]
        )
        loss, _ = backend.supervised(window, target)
        # Equal sampling produces the average of the two loss means. A global factor of 2
        # makes this exactly CE_forget_mean + CE_retain_mean, as specified in the protocol.
        (loss * 2 / len(batch)).backward()
        losses["forget_ce" if record["role"] == "forget" else "retain_ce"].append(float(loss.detach()))
    penalty, penalty_gradient_norm = 0.0, 0.0
    if orthogonal and previous is not None:
        raw = backend.adapters.orthogonal_loss(previous)
        gradients = torch.autograd.grad(
            cfg.orthogonal_weight * raw, backend.adapters.parameters(), retain_graph=True, allow_unused=True
        )
        penalty_gradient_norm = float(torch.sqrt(sum(g.square().sum() for g in gradients if g is not None)))
        (cfg.orthogonal_weight * raw).backward()
        penalty = float(raw.detach())
    norm = torch.nn.utils.clip_grad_norm_(backend.adapters.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite adapter gradient")
    optim.step()
    return {key: float(np.mean(values)) for key, values in losses.items()} | {
        "orthogonal_penalty": penalty,
        "orthogonal_gradient_norm": penalty_gradient_norm,
        "gradient_norm_before_clip": float(norm),
    }


def checkpoint_path(cfg, arm, stage):
    return cfg.root / "adapters" / arm / f"stage-{stage}.pt"


def checkpoint(cfg, backend, optim, rng, manifest, arm, stage, completed, previous, total):
    return {
        "fingerprint": cfg.fingerprint,
        "manifest_digest": manifest["digest"],
        "arm": arm,
        "stage": stage,
        "completed_updates": completed,
        "target_updates": total,
        "complete": completed == total,
        "adapters": backend.adapters.state(),
        "previous": previous,
        "optimizer": optim.state_dict(),
        "sample_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def validate_checkpoint(cfg, manifest, payload):
    if payload["fingerprint"] != cfg.fingerprint or payload["manifest_digest"] != manifest["digest"]:
        raise ValueError("checkpoint does not belong to this configuration/dataset")


def benchmark(cfg, backend, manifest, deadline):
    path = cfg.root / "benchmark.json"
    if path.exists():
        result = read_json(path)
        if result["fingerprint"] != cfg.fingerprint or result["manifest_digest"] != manifest["digest"]:
            raise ValueError("stale benchmark")
        return result
    initial = backend.adapters.state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rng = np.random.default_rng(cfg.seed)
    forget, retain = pool(manifest, 1)
    windows = Windows(cfg)
    optim = optimizer(backend, cfg)
    started = time.monotonic()
    try:
        for _ in range(cfg.benchmark_warmup):
            deadline.check()
            update(backend, windows, sample_batch(rng, forget, retain, cfg.effective_batch), cfg, optim)
        # A nonzero previous-stage factor exercises the actual orthogonal graph.
        previous = backend.adapters.state()
        timings = []
        for _ in range(cfg.benchmark_updates):
            deadline.check()
            synchronize()
            then = time.monotonic()
            update(
                backend,
                windows,
                sample_batch(rng, forget, retain, cfg.effective_batch),
                cfg,
                optim,
                previous,
                orthogonal=True,
            )
            synchronize()
            timings.append(time.monotonic() - then)
        mean = float(np.mean(timings))
        choices = [n for n in [cfg.updates_per_stage, 100, 50] if n <= cfg.updates_per_stage]
        choices = list(dict.fromkeys(choices))
        selected = next((n for n in choices if mean * n * 6 <= cfg.train_budget_hours * 3600), None)
        if selected is None:
            raise TimeoutError("even 50 matched updates/stage do not fit training budget")
        result = {
            "fingerprint": cfg.fingerprint,
            "manifest_digest": manifest["digest"],
            "synthetic": cfg.backend == "synthetic",
            "hardware": hardware(),
            "mean_update_seconds": mean,
            "p95_update_seconds": float(np.percentile(timings, 95)),
            "update_seconds": timings,
            "measured_updates": len(timings),
            "effective_batch": cfg.effective_batch,
            "actual_microbatch": 1,
            "selected_updates_per_stage": selected,
            "train_only_hours_estimate": mean * selected * 6 / 3600,
            "benchmark_wall_seconds": time.monotonic() - started,
            "note": "excludes model loading, features, offline evaluation, simulator rollouts, checkpoint I/O",
        }
        write_json(path, result)
        print(f"Measured {mean:.3f}s/update; matched budget={selected}/stage", flush=True)
        return result
    finally:
        backend.adapters.load(initial)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        windows.close()


def train(cfg, backend, manifest, budget, deadline):
    total = budget["selected_updates_per_stage"]
    windows = Windows(cfg)
    stage_times = []
    try:
        for arm in ARMS:
            backend.adapters.load(backend.initial)
            for stage in range(1, 4):
                deadline.check()
                path = checkpoint_path(cfg, arm, stage)
                previous = backend.adapters.state() if stage > 1 else None
                optim = optimizer(backend, cfg)
                rng = np.random.default_rng(cfg.seed + stage)
                completed = 0
                if path.exists():
                    state = load_torch(path)
                    validate_checkpoint(cfg, manifest, state)
                    if state["target_updates"] != total:
                        raise ValueError("matched training budget changed")
                    backend.adapters.load(state["adapters"])
                    if state["complete"]:
                        continue
                    optim.load_state_dict(state["optimizer"])
                    rng.bit_generator.state = state["sample_rng"]
                    torch.set_rng_state(state["torch_rng"])
                    if state["cuda_rng"] is not None:
                        torch.cuda.set_rng_state_all(state["cuda_rng"])
                    previous, completed = state["previous"], state["completed_updates"]
                forget, retain = pool(manifest, stage)
                log = cfg.root / "training" / f"{arm}-stage-{stage}.jsonl"
                log.parent.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                try:
                    with log.open("a", encoding="utf-8") as stream:
                        while completed < total:
                            deadline.check()
                            batch = sample_batch(rng, forget, retain, cfg.effective_batch)
                            then = time.monotonic()
                            metrics = update(
                                backend, windows, batch, cfg, optim, previous, orthogonal=arm == "orthogonal"
                            )
                            completed += 1
                            import json

                            stream.write(
                                json.dumps(
                                    metrics | {"update": completed, "seconds": time.monotonic() - then}
                                )
                                + "\n"
                            )
                            stream.flush()
                            if completed % cfg.checkpoint_every == 0:
                                save_torch(
                                    path,
                                    checkpoint(
                                        cfg,
                                        backend,
                                        optim,
                                        rng,
                                        manifest,
                                        arm,
                                        stage,
                                        completed,
                                        previous,
                                        total,
                                    ),
                                )
                                print(f"{arm} stage {stage}: {completed}/{total}", flush=True)
                finally:
                    save_torch(
                        path,
                        checkpoint(
                            cfg, backend, optim, rng, manifest, arm, stage, completed, previous, total
                        ),
                    )
                stage_times.append(
                    {
                        "arm": arm,
                        "stage": stage,
                        "updates": completed,
                        "wall_seconds_this_invocation": time.monotonic() - started,
                        "effective_norms": backend.adapters.effective_norms(),
                    }
                )
        write_json(
            cfg.root / "training-summary.json", {"stages": stage_times, "selected_updates_per_stage": total}
        )
    finally:
        windows.close()


def load_adapter(cfg, backend, manifest, arm, stage):
    payload = load_torch(checkpoint_path(cfg, arm, stage))
    validate_checkpoint(cfg, manifest, payload)
    if not payload["complete"]:
        raise ValueError("cannot evaluate incomplete stage checkpoint")
    backend.adapters.load(payload["adapters"])
    return payload
