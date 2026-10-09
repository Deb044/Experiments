"""Load and validate the fixed experiment protocol."""
from pathlib import Path

from .io import read_json


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    cfg["output"] = str((path.parent / cfg["output"]).resolve())
    if cfg["backend"] not in {"synthetic", "smolvla"}:
        raise ValueError("backend must be synthetic or smolvla")
    for key in ("seeds", "request_seeds"):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"{key} must be nonempty and unique")
    ids = [task["id"] for task in cfg["tasks"]]
    if len(ids) < 3 or len(set(ids)) != len(ids):
        raise ValueError("At least three unique task IDs are required for grouped validation")
    if len(cfg["request_seeds"]) < 3:
        raise ValueError("At least three deletion requests are required")
    if not 0 < cfg["forget_fraction"] < 1:
        raise ValueError("forget_fraction must lie strictly between zero and one")
    train = cfg["training"]
    if any(train[k] < 1 for k in ("steps", "effective_batch", "microbatch", "checkpoint_every")):
        raise ValueError("Training counts must be positive")
    if train["effective_batch"] % train["microbatch"]:
        raise ValueError("effective_batch must be divisible by microbatch")
    if train["precision"] not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if cfg["model"]["rank"] < 1:
        raise ValueError("LoRA rank must be positive")
    if any(cfg["probes"][k] < 1 for k in ("windows", "draws", "gradient_windows")):
        raise ValueError("Probe counts must be positive")
    if any(x <= 0 for x in cfg["prediction"]["ridge_grid"]):
        raise ValueError("Ridge values must be positive")
    return cfg
