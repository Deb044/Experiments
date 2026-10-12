"""Atomic artifacts, provenance and deadline handling."""

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_torch(path):
    # Only load checkpoints produced by this experiment, never untrusted external pickle files.
    return torch.load(path, map_location="cpu", weights_only=False)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def bind_run(cfg):
    cfg.root.mkdir(parents=True, exist_ok=True)
    path = cfg.root / "run.json"
    source_digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_digest.update(source.name.encode())
        source_digest.update(source.read_bytes())
    source_hash = source_digest.hexdigest()
    if path.exists():
        current = read_json(path)
        if current["fingerprint"] != cfg.fingerprint:
            raise ValueError("configuration changed; use a new output directory")
        if current.get("source_hash") != source_hash:
            raise ValueError("experiment source changed; use a new output directory")
    else:
        try:
            commit = git_revision(Path(__file__).resolve().parents[3])
        except (subprocess.SubprocessError, FileNotFoundError):
            commit = None
        write_json(
            path,
            {
                "fingerprint": cfg.fingerprint,
                "synthetic": cfg.backend == "synthetic",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit,
                "source_hash": source_hash,
                "scope": "conditional behavior suppression; not data erasure",
                "config": asdict(cfg),
            },
        )


class Deadline:
    def __init__(self, hours):
        self.started = time.monotonic()
        self.end = self.started + hours * 3600

    @property
    def remaining(self):
        return max(0.0, self.end - time.monotonic())

    def check(self):
        if self.remaining == 0:
            raise TimeoutError("configured wall-time budget exhausted; checkpoints preserved")
