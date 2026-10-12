"""Validated configuration and stable experiment identity."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    backend: str = "openvla"
    output: str = "outputs/pilot"
    data_root: str = ""
    libero_root: str = ""
    seed: int = 7
    suite: str = "libero_object"
    task_order_index: int = 0
    model_id: str = "openvla/openvla-7b-finetuned-libero-object"
    model_revision: str = "287d6cfdf12d07b1449505f66d9bf3550257e9b3"
    code_revision: str = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
    libero_revision: str = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    quantization: str = "nf4"
    attention: str = "sdpa"
    rank: int = 8
    alpha: float = 8
    learning_rate: float = 1e-4
    orthogonal_weight: float = 0.1
    updates_per_stage: int = 200
    effective_batch: int = 8
    microbatch: int = 1
    checkpoint_every: int = 25
    trajectories_per_task: int = 30
    windows_per_trajectory: int = 32
    eval_windows_per_trajectory: int = 8
    validation_false_positive_rate: float = 0.05
    gate_smoothing: float = 0.8
    gripper_open_threshold: float = 0.04
    hold_position_tolerance: float = 0.01
    hold_rotation_tolerance: float = 0.1
    hold_gripper_tolerance: float = 0.01
    smoke_trials: int = 2
    rollout_trials: int = 4
    rollout_min_trials: int = 2
    rollout_optional_gates: bool = False
    settle_steps: int = 10
    benchmark_warmup: int = 10
    benchmark_updates: int = 20
    train_budget_hours: float = 6
    rollout_budget_hours: float = 9
    total_budget_hours: float = 24
    video_episodes_per_configuration: int = 1
    telemetry_interval_seconds: float = 2

    def validate(self):
        if self.backend not in {"openvla", "synthetic"}:
            raise ValueError("backend must be openvla or synthetic")
        if self.quantization not in {"nf4", "bf16"}:
            raise ValueError("quantization must be nf4 or bf16")
        if self.attention not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("unsupported attention implementation")
        if self.microbatch != 1:
            raise ValueError("this implementation uses microbatch 1 with gradient accumulation")
        if self.effective_batch < 2 or self.effective_batch % 2:
            raise ValueError("effective batch must be positive and even")
        if self.orthogonal_weight < 0 or self.settle_steps < 0 or self.benchmark_warmup < 0:
            raise ValueError("penalty, settling steps and warmup must be nonnegative")
        if not 15 <= self.trajectories_per_task <= 30:
            raise ValueError("this pilot uses between 15 and 30 trajectories per task")
        for name in [
            "rank",
            "alpha",
            "learning_rate",
            "updates_per_stage",
            "microbatch",
            "checkpoint_every",
            "windows_per_trajectory",
            "eval_windows_per_trajectory",
            "benchmark_updates",
            "smoke_trials",
            "telemetry_interval_seconds",
            "train_budget_hours",
            "rollout_budget_hours",
            "total_budget_hours",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rollout_trials < self.rollout_min_trials or self.rollout_min_trials < 2:
            raise ValueError("rollout trials must be >= min_trials >= 2")
        if not 0 <= self.gate_smoothing < 1 or not 0 <= self.validation_false_positive_rate < 1:
            raise ValueError("invalid gate or calibration probability")
        if self.backend == "openvla":
            if self.suite != "libero_object":
                raise ValueError("this verified pilot supports libero_object (280-step horizon) only")
            for name in ["model_revision", "code_revision", "libero_revision"]:
                if not re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)):
                    raise ValueError(f"{name} must be an immutable commit SHA")
        return self

    @property
    def root(self):
        return Path(self.output).expanduser().resolve()

    @property
    def fingerprint(self):
        # Data/output paths are still recorded, but relocating a cache is not a scientific change.
        payload = asdict(self)
        for name in ["output", "data_root", "libero_root"]:
            payload.pop(name)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load(path, **overrides):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(payload) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    payload.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**payload).validate()
