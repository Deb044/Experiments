"""Resolve immutable Hub revisions and select actual dataset trajectories."""
from pathlib import Path

from .io import digest, read_json, write_json
from .manifest import make_manifest
from .dataset_tasks import canonical, select_episodes


def prepare(cfg):
    path = Path(cfg["output"]) / "manifest.json"
    if path.exists():
        existing = read_json(path)
        if existing["config"] != cfg:
            raise ValueError("Output already contains a different protocol; use a new output directory")
        content = {key: value for key, value in existing.items() if key != "fingerprint"}
        if digest(content) != existing["fingerprint"]:
            raise ValueError("Existing manifest failed its integrity check")
        # Resume immutable revisions instead of re-resolving mutable main refs.
        return existing
    if cfg["backend"] == "synthetic":
        n = cfg["synthetic"]["episodes_per_task"]
        episodes = {task["id"]: list(range(i * n, (i + 1) * n)) for i, task in enumerate(cfg["tasks"])}
        provenance = {"synthetic": True, "dataset_revision": "fixture-v1", "model_revision": "fixture-v1"}
    else:
        from .provenance import verify_lerobot
        installed_lerobot = verify_lerobot()
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
        import pyarrow.parquet as pq
        task_table = {int(row["task_index"]): str(label) for label, row in meta.tasks.iterrows()}

        def read_file(filename):
            # Cache into the same LeRobot root that training uses. Every file is
            # resolved at the immutable dataset commit, not a mutable main ref.
            local = hf_hub_download(ds["repo_id"], filename, repo_type="dataset",
                                    revision=dataset_revision, local_dir=meta.root)
            return pq.read_table(local, columns=["episode_index", "task_index"], use_threads=False).to_pylist()

        episodes, selection = select_episodes(cfg["tasks"], list(meta.episodes), task_table, meta.data_path, read_file)
        provenance = {"synthetic": False, "dataset_revision": dataset_revision, "model_revision": model_revision,
                      "vlm_id": vlm_id, "vlm_revision": vlm_revision, "dataset_root": str(meta.root),
                      "lerobot_commit": installed_lerobot["commit"], "lerobot_version": installed_lerobot["version"],
                      "task_selection": selection}
    manifest = make_manifest(cfg, episodes, provenance)
    if path.exists() and read_json(path)["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Output already contains a different manifest; use a new output directory")
    write_json(path, manifest)
    return manifest
