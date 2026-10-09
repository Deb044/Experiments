"""Resolve immutable Hub revisions and select actual dataset trajectories."""
from pathlib import Path
import re

from .io import read_json, write_json
from .manifest import make_manifest


def canonical(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def prepare(cfg):
    if cfg["backend"] == "synthetic":
        n = cfg["synthetic"]["episodes_per_task"]
        episodes = {task["id"]: list(range(i * n, (i + 1) * n)) for i, task in enumerate(cfg["tasks"])}
        provenance = {"synthetic": True, "dataset_revision": "fixture-v1", "model_revision": "fixture-v1"}
    else:
        from huggingface_hub import HfApi, hf_hub_download
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        api = HfApi()
        ds = cfg["dataset"]
        model = cfg["model"]
        dataset_revision = api.dataset_info(ds["repo_id"], revision=ds["revision"]).sha
        model_revision = api.model_info(model["repo_id"], revision=model["revision"]).sha
        model_config = read_json(hf_hub_download(model["repo_id"], "config.json", revision=model_revision))
        vlm_id = model_config["vlm_model_name"]
        vlm_revision = api.model_info(vlm_id).sha
        meta = LeRobotDatasetMetadata(ds["repo_id"], revision=dataset_revision)
        episodes = {task["id"]: [] for task in cfg["tasks"]}
        expected = {canonical(task["instruction"]): task["id"] for task in cfg["tasks"]}
        for episode in meta.episodes:
            labels = episode.get("tasks", [])
            if isinstance(labels, str):
                labels = [labels]
            matches = {expected[canonical(label)] for label in labels if canonical(label) in expected}
            if len(matches) > 1:
                raise ValueError("Episode belongs to multiple selected tasks")
            if matches:
                episodes[matches.pop()].append(int(episode["episode_index"]))
        provenance = {"synthetic": False, "dataset_revision": dataset_revision, "model_revision": model_revision,
                      "vlm_id": vlm_id, "vlm_revision": vlm_revision, "dataset_root": str(meta.root),
                      "lerobot_commit": "b9cb121cb7d4e3e26ec5c906d08dda68d151ba52"}
    manifest = make_manifest(cfg, episodes, provenance)
    path = Path(cfg["output"]) / "manifest.json"
    if path.exists() and read_json(path)["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Output already contains a different manifest; use a new output directory")
    write_json(path, manifest)
    return manifest

