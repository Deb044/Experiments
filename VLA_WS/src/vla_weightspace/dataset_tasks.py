"""Resolve trajectory labels when LeRobot v3 episode metadata omits tasks."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import re


def canonical(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def select_episodes(tasks, episode_rows, task_table, data_path, read_file):
    expected = {canonical(task["instruction"]): task["id"] for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError("Selected task instructions are ambiguous after normalization")
    by_id = {int(row["episode_index"]): row for row in episode_rows}
    if len(by_id) != len(episode_rows):
        raise ValueError("Duplicate episode IDs in dataset metadata")
    labels = {}
    for eid, row in by_id.items():
        value = row.get("tasks")
        if isinstance(value, str):
            value = [value]
        if value is not None and len(value):
            labels[eid] = list(value)
    missing = set(by_id) - set(labels)
    files = []
    if missing:
        # Read only frame metadata, without loading the model or any videos.
        files = sorted({data_path.format(chunk_index=row["data/chunk_index"],
                                         file_index=row["data/file_index"]) for row in episode_rows})
        print(f"Resolving episode task labels from {len(files)} frame parquet files (no videos).", flush=True)
        indices, counts = defaultdict(set), Counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for number, rows in enumerate(pool.map(read_file, files), 1):
                for row in rows:
                    eid = int(row["episode_index"])
                    if eid not in by_id:
                        raise ValueError(f"Frame data contains unknown episode {eid}")
                    indices[eid].add(int(row["task_index"]))
                    counts[eid] += 1
                if number % 25 == 0 or number == len(files):
                    print(f"Task-label scan: {number}/{len(files)} files", flush=True)
        for eid in missing:
            if counts[eid] != int(by_id[eid]["length"]):
                raise ValueError(f"Episode {eid}: frame count {counts[eid]} differs from metadata length {by_id[eid]['length']}")
            unknown = indices[eid] - set(task_table)
            if unknown:
                raise ValueError(f"Episode {eid}: unknown task indices {sorted(unknown)}")
            labels[eid] = [task_table[index] for index in sorted(indices[eid])]
    episodes = {task["id"]: [] for task in tasks}
    for eid in sorted(by_id):
        matches = {expected[canonical(label)] for label in labels[eid] if canonical(label) in expected}
        if len(matches) > 1:
            raise ValueError(f"Episode {eid} belongs to multiple selected tasks")
        if matches:
            episodes[matches.pop()].append(eid)
    counts = {task: len(ids) for task, ids in episodes.items()}
    print(f"Selected trajectory counts: {counts}", flush=True)
    if any(count < 10 for count in counts.values()):
        raise ValueError(f"Too few matched trajectories: {counts}. Available task labels: {sorted(set(task_table.values()))}. "
                         "Check the dataset revision and exact instructions; do not lower the trajectory requirement.")
    return episodes, {"source": "frame_task_index" if missing else "episode_tasks", "frame_files": len(files),
                      "counts": counts}
