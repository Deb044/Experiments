"""Regression for LIBERO v3 metadata with no per-episode tasks column."""
import pytest

from vla_weightspace.dataset_tasks import select_episodes


def fixture():
    tasks = [{"id": name, "instruction": f"pick up {name}"} for name in ("a", "b", "c")]
    table = {i: task["instruction"] for i, task in enumerate(tasks)}
    metadata, frames = [], []
    for eid in range(36):
        metadata.append({"episode_index": eid, "length": 2, "data/chunk_index": 0, "data/file_index": 0})
        frames.extend([{"episode_index": eid, "task_index": eid // 12}] * 2)
    return tasks, table, metadata, frames


def test_missing_tasks_column_resolves_frames_and_complete_episodes():
    tasks, table, metadata, frames = fixture()
    calls = []
    def read(path):
        calls.append(path)
        return frames
    episodes, audit = select_episodes(tasks, metadata, table, "data/{chunk_index}/{file_index}.parquet", read)
    assert episodes == {"a": list(range(12)), "b": list(range(12, 24)), "c": list(range(24, 36))}
    assert calls == ["data/0/0.parquet"]
    assert audit == {"source": "frame_task_index", "frame_files": 1, "counts": {"a": 12, "b": 12, "c": 12}}


def test_legacy_labels_do_not_download_frames():
    tasks, table, metadata, frames = fixture()
    for row in metadata:
        row["tasks"] = table[row["episode_index"] // 12].upper() + "."
    episodes, audit = select_episodes(tasks, metadata, table, "unused", lambda _: pytest.fail("Unexpected frame download"))
    assert len(episodes["a"]) == 12
    assert audit["source"] == "episode_tasks"


@pytest.mark.parametrize("problem", ["unknown_task", "mixed_tasks", "incomplete", "unknown_episode"])
def test_rejects_invalid_frame_mapping(problem):
    tasks, table, metadata, frames = fixture()
    frames = [dict(row) for row in frames]
    if problem == "unknown_task":
        frames[0]["task_index"] = 99
    elif problem == "mixed_tasks":
        frames[0]["task_index"] = 1
    elif problem == "incomplete":
        frames.pop(0)
    else:
        frames[0]["episode_index"] = 99
    with pytest.raises(ValueError):
        select_episodes(tasks, metadata, table, "data", lambda _: frames)


def test_zero_matches_reports_counts_and_available_labels():
    tasks, table, metadata, frames = fixture()
    tasks[0]["instruction"] = "missing instruction"
    with pytest.raises(ValueError, match="Available task labels"):
        select_episodes(tasks, metadata, table, "data", lambda _: frames)
