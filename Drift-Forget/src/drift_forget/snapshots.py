"""Snapshots of the trainable parameters, taken inside continual-openpi's training loop.

`scripts/train.py` calls `on_start` once before training and `maybe_save` after every step. Both only copy arrays
to the host: they never touch the RNG, the data iterators or the optimizer state.

Configuration is a single environment variable, required so that a run can never lose its snapshots silently:

    DF_SNAPSHOT_DIR=<root>   snapshots go to <root>/<config name>/<exp name>/ (the checkpoint directory layout)
    DF_SNAPSHOT_DIR=off      no snapshots

Layout of one run's directory:

    run.json                 run settings and the snapshot offsets
    buffer_lst.json          replayed sample indices per task, fixed for the whole run (ER runs only: without
                             replay, train.py still draws a buffer, but never uses it)
    init.npz / init.json     the state before the first update
    step_NNNNNN.npz / .json  the state after the update with index NNNNNN

Step labels follow openpi's checkpoints: label g is the state after the update with index g. The end of a
10,000-step task is label 9999 of that task, which is the full orbax checkpoint; it is not duplicated here.
Each .npz maps a "/"-joined parameter path to its array. The .json sidecar holds the typed key paths, original
dtypes and shapes, and the SHA-256 of the .npz. It is written last, so a snapshot without its sidecar is
incomplete and must be ignored.
"""

import datetime
import hashlib
import json
import logging
import os
import pathlib
import shutil
from typing import Any

import flax.traverse_util as traverse_util
import jax
import ml_dtypes
import numpy as np

# Steps from the start of a task at which a snapshot is taken (decision D13: dense just after a task switch).
OFFSETS = (49, 99, 249, 499, 999, 1999, 3999, 5999, 7999)

_ENV = "DF_SNAPSHOT_DIR"


def run_dir(config) -> pathlib.Path | None:
    value = os.environ.get(_ENV)
    if value is None:
        raise RuntimeError(f"{_ENV} is not set. Set it to a snapshot root directory, or to 'off' to disable snapshots.")
    if value == "off":
        return None
    return pathlib.Path(value).resolve() / config.name / config.exp_name


def on_start(config, train_state, buffer_lst, *, resumed_step: int | None) -> None:
    """Prepare the run directory and save the initial state (fresh run) or discard stale snapshots (resume)."""
    directory = run_dir(config)
    if directory is None:
        logging.warning(f"[snapshots] disabled ({_ENV}=off)")
        return

    if resumed_step is None:
        if directory.exists() and any(directory.iterdir()):
            if not config.overwrite:
                raise FileExistsError(f"Snapshot directory {directory} is not empty. Use --overwrite or --resume.")
            shutil.rmtree(directory)
            logging.info(f"[snapshots] wiped {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "run.json", _run_info(config))
        if config.cl == "er":
            _write_json(directory / "buffer_lst.json", _plain(buffer_lst))
        _save(directory, "init", config, train_state, step=None, task_idx=None, offset=None)
        return

    # Resume: training restarts from `resumed_step`, but the data order after it will differ from the dead attempt,
    # so every later snapshot belongs to a history that no longer exists.
    for sidecar in sorted(directory.glob("step_*.json")):
        if int(sidecar.stem.removeprefix("step_")) > resumed_step:
            for path in (sidecar, sidecar.with_suffix(".npz")):
                path.unlink(missing_ok=True)
            logging.info(f"[snapshots] deleted {sidecar.stem}: after the resumed step {resumed_step}")
    for partial in directory.glob(".*.tmp"):
        partial.unlink()
    if config.cl == "er" and json.loads((directory / "buffer_lst.json").read_text()) != _plain(buffer_lst):
        raise RuntimeError(f"Replay buffer of the resumed run differs from {directory / 'buffer_lst.json'}")


def maybe_save(config, train_state, *, global_step: int, global_start: int, task_idx: int) -> None:
    offset = global_step - global_start
    if offset not in OFFSETS or offset >= config.steps_per_task - 1:
        return
    directory = run_dir(config)
    if directory is not None:
        _save(directory, f"step_{global_step:06d}", config, train_state, step=global_step, task_idx=task_idx, offset=offset)


def _save(directory: pathlib.Path, name: str, config, train_state, *, step, task_idx, offset) -> None:
    flat = traverse_util.flatten_dict(train_state.params.filter(config.trainable_filter).to_pure_dict())
    flat = jax.device_get(flat)  # host copy; `train_state` itself is untouched

    arrays, keys, dtypes, shapes = {}, [], {}, {}
    for key, value in flat.items():
        value = np.asarray(value)
        path = "/".join(str(k) for k in key)
        dtypes[path] = str(value.dtype)
        shapes[path] = list(value.shape)
        if value.dtype == ml_dtypes.bfloat16:
            value = value.astype(np.float32)  # exact; .npz cannot store bfloat16
        arrays[path] = value
        keys.append(list(key))
    if len(arrays) != len(flat):
        raise RuntimeError("Two parameter paths map to the same snapshot key")

    tmp = directory / f".{name}.npz.tmp"
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    sidecar = {
        "name": name,
        "step": step,
        "n_updates": 0 if step is None else step + 1,
        "task_idx": task_idx,
        "offset": offset,
        "config_name": config.name,
        "exp_name": config.exp_name,
        "seed": config.seed,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_params": int(sum(a.size for a in arrays.values())),
        "sha256": _sha256(tmp),
        "keys": keys,
        "dtypes": dtypes,
        "shapes": shapes,
    }
    os.replace(tmp, directory / f"{name}.npz")
    _write_json(directory / f"{name}.json", sidecar)
    logging.info(f"[snapshots] saved {name} ({sidecar['n_params']:,} trainable params)")


def _run_info(config) -> dict[str, Any]:
    return {
        "config_name": config.name,
        "exp_name": config.exp_name,
        "seed": config.seed,
        "cl": config.cl,
        "n_memories": config.n_memories,
        "n_tasks": config.n_tasks,
        "steps_per_task": config.steps_per_task,
        "batch_size": config.batch_size,
        "offsets": list(OFFSETS),
    }


def _plain(buffer_lst) -> list[list[int]]:
    return [[int(i) for i in buffer] for buffer in buffer_lst]


def _write_json(path: pathlib.Path, data) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()
