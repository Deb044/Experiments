"""Trajectory-level manifests; lazy HDF5 reads with explicit simulator coordinates."""

import hashlib
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .io import hash_file, read_json, write_json


def split_trajectories(names, seed, maximum):
    if len(names) < 15:
        raise ValueError("need at least 15 complete trajectories")
    names = sorted(names)
    np.random.default_rng(seed).shuffle(names)
    names = names[:maximum]
    if len(names) >= 30:
        counts = (20, 5)
    else:
        counts = (int(len(names) * 0.6), int(len(names) * 0.2))
    a, b = counts
    return {"train": names[:a], "validation": names[a : a + b], "test": names[a + b :]}


def frame_indices(length, maximum):
    if length < 1:
        raise ValueError("empty trajectory")
    return np.unique(np.linspace(0, length - 1, min(length, maximum), dtype=int)).tolist()


def build_manifest(cfg, tasks, selection):
    if len(tasks) != 6 or len({t["id"] for t in tasks}) != 6:
        raise ValueError("six distinct tasks are required")
    records, files = [], {}
    for position, task in enumerate(tasks):
        role = "forget" if position < 3 else "retain"
        stage = position + 1 if role == "forget" else 0
        if cfg.backend == "synthetic":
            names = [f"demo_{i}" for i in range(cfg.trajectories_per_task)]
            lengths = {name: 12 for name in names}
        else:
            path = Path(task["file"])
            if not path.is_file():
                raise FileNotFoundError(f"missing LIBERO demonstration file: {path}")
            files[str(path)] = hash_file(path)
            with h5py.File(path, "r") as source:
                lengths = {}
                for name, group in source["data"].items():
                    required = ["actions", "obs/agentview_rgb", "obs/gripper_states", "dones"]
                    if not all(key in group for key in required):
                        raise ValueError(f"incomplete {path}:{name}; need {required}")
                    count = len(group["actions"])
                    if any(len(group[key]) != count for key in required):
                        raise ValueError(f"unaligned {path}:{name}")
                    if group["actions"].shape[1:] != (7,):
                        raise ValueError("expected raw seven-dimensional simulator actions")
                    if count < 2 or not bool(group["dones"][-1]):
                        continue
                    # Upstream create_dataset.py records obs after env.step(action_j).
                    # The causal next-action target for obs_j is action_{j+1}.
                    lengths[name] = count - 1
            names = list(lengths)
        splits = split_trajectories(names, cfg.seed + task["id"], cfg.trajectories_per_task)
        for split, trajectory_names in splits.items():
            for trajectory in trajectory_names:
                maximum = cfg.windows_per_trajectory if split == "train" else cfg.eval_windows_per_trajectory
                for frame in frame_indices(lengths[trajectory], maximum):
                    records.append(
                        {
                            "task_id": task["id"],
                            "instruction": task["instruction"],
                            "role": role,
                            "stage": stage,
                            "split": split,
                            "file": task.get("file"),
                            "trajectory": trajectory,
                            "frame": frame,
                            "key": f"{task['id']}/{trajectory}/{frame}",
                        }
                    )
    payload = {
        "fingerprint": cfg.fingerprint,
        "synthetic": cfg.backend == "synthetic",
        "tasks": tasks,
        "selection": selection,
        "files": files,
        "records": records,
        "observation_action_alignment": "raw HDF5 post-action obs[j] -> actions[j+1]"
        if cfg.backend == "openvla"
        else "synthetic",
    }
    payload["digest"] = hashlib.sha256(
        str(sorted(files.items())).encode() + str([(r["key"], r["split"]) for r in records]).encode()
    ).hexdigest()
    write_json(cfg.root / "manifest.json", payload)
    return payload


def load_manifest(cfg, verify_files=False):
    manifest = read_json(cfg.root / "manifest.json")
    if manifest["fingerprint"] != cfg.fingerprint:
        raise ValueError("manifest/config identity mismatch")
    if verify_files:
        for path, expected in manifest["files"].items():
            if hash_file(path) != expected:
                raise ValueError(f"dataset changed after preparation: {path}")
    return manifest


class Windows:
    def __init__(self, cfg):
        self.cfg = cfg
        self.files = {}

    def get(self, record):
        if self.cfg.backend == "synthetic":
            return self.synthetic(record)
        path = record["file"]
        if path not in self.files:
            self.files[path] = h5py.File(path, "r")
        group = self.files[path]["data"][record["trajectory"]]
        index = record["frame"]
        # Original LIBERO HDF5 stores the upside-down renderer images. Rotate like official eval.
        image = np.asarray(group["obs/agentview_rgb"][index])[::-1, ::-1].copy()
        return {
            "image": Image.fromarray(image),
            "action": np.asarray(group["actions"][index + 1]),
            "gripper_qpos": np.asarray(group["obs/gripper_states"][index]),
            "instruction": record["instruction"],
        }

    def synthetic(self, record):
        seed = int(hashlib.sha256(record["key"].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        image[:, :, record["task_id"] % 3] = record["task_id"] * 25
        return {
            "image": Image.fromarray(image),
            "action": np.array([0.2, 0, 0, 0, 0, 0, -1.0]),
            "gripper_qpos": np.array([0.03, -0.03]),
            "instruction": record["instruction"],
        }

    def close(self):
        for stream in self.files.values():
            stream.close()
        self.files.clear()
