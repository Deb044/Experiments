"""Raw LIBERO HDF5 uses post-action observations: require causal next-action alignment."""

from dataclasses import replace

import h5py
import numpy as np
import pytest
from test_protocol import small_config

from o3_vla.data import Windows, build_manifest, load_manifest


def test_raw_hdf5_alignment_split_and_hash_guard(tmp_path):
    cfg = replace(small_config(tmp_path / "run"), backend="openvla")
    tasks = []
    for task_id in range(6):
        path = tmp_path / f"task-{task_id}.hdf5"
        with h5py.File(path, "w") as stream:
            data = stream.create_group("data")
            for demo in range(15):
                group = data.create_group(f"demo_{demo}")
                actions = np.zeros((3, 7))
                actions[:, 0] = [0.1, 0.2, 0.3]
                actions[:, 6] = -1
                group.create_dataset("actions", data=actions)
                group.create_dataset("dones", data=[0, 0, 1])
                images = np.zeros((3, 2, 2, 3), dtype=np.uint8)
                images[:, 0, 0] = 255
                group.create_dataset("obs/agentview_rgb", data=images)
                group.create_dataset("obs/gripper_states", data=np.ones((3, 2)) * 0.03)
        tasks.append({"id": task_id, "instruction": f"task {task_id}", "file": str(path)})
    manifest = build_manifest(cfg, tasks, [])
    assert max(r["frame"] for r in manifest["records"]) == 1
    record = next(r for r in manifest["records"] if r["frame"] == 0)
    reader = Windows(cfg)
    try:
        window = reader.get(record)
        assert window["action"][0] == pytest.approx(0.2)
        assert np.all(np.asarray(window["image"])[1, 1] == 255)
    finally:
        reader.close()
    load_manifest(cfg, verify_files=True)
    with h5py.File(tasks[0]["file"], "r+") as stream:
        stream["data/demo_0/actions"][0, 0] = 0.5
    with pytest.raises(ValueError, match="dataset changed"):
        load_manifest(cfg, verify_files=True)
