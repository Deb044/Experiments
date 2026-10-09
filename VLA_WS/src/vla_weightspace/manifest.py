"""Trajectory-level splits shared by paired full and clean training."""
import math

import numpy as np

from .io import digest


def make_manifest(cfg, episodes, provenance):
    tasks = {}
    used = set()
    for task in cfg["tasks"]:
        ids = sorted(episodes[task["id"]])
        if len(ids) < 10 or len(set(ids)) != len(ids):
            raise ValueError(f"{task['id']}: need at least ten distinct complete trajectories")
        if used.intersection(ids):
            raise ValueError("A trajectory cannot belong to multiple selected tasks")
        used.update(ids)
        rng = np.random.default_rng(cfg["split_seed"])
        shuffled = rng.permutation(ids).tolist()
        n_train, n_cal = math.floor(0.7 * len(ids)), math.floor(0.1 * len(ids))
        train, calibration = shuffled[:n_train], shuffled[n_train:n_train + n_cal]
        heldout = shuffled[n_train + n_cal:]
        count = math.ceil(cfg["forget_fraction"] * n_train)
        if count >= n_train:
            raise ValueError("Deletion would leave no retained data")
        requests = {}
        subsets = set()
        for seed in cfg["request_seeds"]:
            forget = sorted(np.random.default_rng(seed).choice(train, count, replace=False).tolist())
            signature = tuple(forget)
            if signature in subsets:
                raise ValueError("Duplicate deletion subsets; change request seeds or add trajectories")
            subsets.add(signature)
            requests[str(seed)] = {"forget": forget, "retain": sorted(set(train) - set(forget))}
        tasks[task["id"]] = {"train": sorted(train), "calibration": sorted(calibration), "heldout": sorted(heldout), "requests": requests, "instruction": task.get("instruction", task["id"])}
    result = {"schema": 1, "config": cfg, "provenance": provenance, "tasks": tasks}
    result["fingerprint"] = digest(result)
    return result


def run_specs(manifest):
    for task, entry in manifest["tasks"].items():
        for seed in manifest["config"]["seeds"]:
            yield {"id": f"{task}/seed-{seed}/full", "task": task, "seed": seed, "request": None, "episodes": entry["train"]}
            for request, subsets in entry["requests"].items():
                yield {"id": f"{task}/seed-{seed}/clean-{request}", "task": task, "seed": seed, "request": request, "episodes": subsets["retain"]}
