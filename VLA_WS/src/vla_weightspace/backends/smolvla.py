"""Native SmolVLA flow matching, trajectory-only sampling, calibration-only stats.

The action-output hook observes the native v_t without copying the model's
attention or flow implementation. This deliberately fails on incompatible APIs.

Updates never wait on video decoding: every frame a task can sample is read once
through the dataset's own item access and kept in a per-task cache. Cached pixels
reproduce the native items bit for bit. `model.cache_frozen_vision` keeps the
frozen vision encoder's output for those frames instead of the pixels.
"""
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from ..io import read_json, temporary_path, write_json
from ..lora import attach_lora
from ..runtime import precision_context

CACHE_IMPLEMENTATION = "task-cache-v1"
ITEM_KEYS = ("observation.state", "action", "action_is_pad", "task")


class SmolVLABackend:
    def __init__(self, manifest, task, seed):
        from ..provenance import verify_lerobot
        verify_lerobot()
        from huggingface_hub import snapshot_download
        from lerobot.configs import FeatureType, PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.feature_utils import dataset_to_policy_features

        self.manifest, self.task = manifest, task
        self.cfg = manifest["config"]
        self.device = torch.device(self.cfg["training"]["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run the real study on the research GPU, or use the explicit synthetic config")
        provenance = manifest["provenance"]
        config = PreTrainedConfig.from_pretrained(self.cfg["model"]["repo_id"], revision=provenance["model_revision"])
        self.chunk_size, self.max_action_dim = config.chunk_size, config.max_action_dim
        config.device = str(self.device)
        config.load_vlm_weights = False  # Full SmolVLA weights are loaded below, not a second VLM checkpoint.
        config.train_expert_only = False
        config.train_state_proj = False
        config.vlm_model_name = snapshot_download(provenance["vlm_id"], revision=provenance["vlm_revision"],
                                                 allow_patterns=["*.json", "*.txt", "*.model"])
        entry = manifest["tasks"][task]
        episodes = sorted(entry["train"] + entry["calibration"] + entry["heldout"])
        # The calibration/evaluation episodes are loaded but never sampled by training.
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata(self.cfg["dataset"]["repo_id"], root=provenance["dataset_root"], revision=provenance["dataset_revision"])
        self.dataset = LeRobotDataset(self.cfg["dataset"]["repo_id"], root=meta.root, revision=provenance["dataset_revision"],
                                      episodes=episodes, delta_timestamps={"action": [i / meta.fps for i in range(self.chunk_size)]},
                                      video_backend=self.cfg["dataset"].get("video_backend", "pyav"))
        features = dataset_to_policy_features(meta.features)
        config.input_features = {key: value for key, value in features.items() if value.type in {FeatureType.VISUAL, FeatureType.STATE} and key != "observation.environment_state"}
        config.output_features = {"action": features["action"]}
        self.action_dim = features["action"].shape[0]
        self.policy = SmolVLAPolicy.from_pretrained(self.cfg["model"]["repo_id"], revision=provenance["model_revision"], config=config, strict=True)
        self.policy.float()  # BF16 is applied through autocast; fp32 also remains a valid configuration.
        torch.manual_seed(seed)
        self.modules = attach_lora(self.policy, self.cfg["model"]["rank"])
        names = list(self.modules)
        if not any("lm_expert" in name for name in names) or not any("vlm." in name for name in names):
            raise RuntimeError("Expected LoRA targets in both language and action transformer stacks")
        self.tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
        self.image_keys = list(config.image_features)
        ep_column = np.asarray(self.dataset.hf_dataset["episode_index"])
        self.indices = {episode: np.flatnonzero(ep_column == episode) for episode in episodes}
        if any(len(value) == 0 for value in self.indices.values()):
            raise ValueError("Selected metadata episodes have no loaded frames")
        self.stats = self._calibration_stats()
        self._load_cache()

    def _calibration_stats(self):
        path = Path(self.cfg["output"]) / "calibration" / f"{self.task}.json"
        if path.exists():
            saved = read_json(path)
            if saved["fingerprint"] != self.manifest["fingerprint"]:
                raise ValueError("Calibration fingerprint mismatch")
            return saved["stats"]
        sums, squares, count = {}, {}, 0
        # No image decoding is needed to estimate proprioception/action normalization.
        table = self.dataset.hf_dataset
        for episode in self.manifest["tasks"][self.task]["calibration"]:
            rows = table.select(self.indices[episode].tolist())
            for key in ("observation.state", "action"):
                values = np.asarray(rows[key], dtype=np.float64)
                sums[key] = sums.get(key, 0) + values.sum(axis=0)
                squares[key] = squares.get(key, 0) + (values ** 2).sum(axis=0)
            count += len(rows)
        stats = {}
        for key in sums:
            mean = sums[key] / count
            std = np.sqrt(np.maximum(squares[key] / count - mean ** 2, 1e-12))
            stats[key] = {"mean": mean.tolist(), "std": std.tolist()}
        write_json(path, {"fingerprint": self.manifest["fingerprint"], "episodes": self.manifest["tasks"][self.task]["calibration"], "stats": stats})
        return stats

    def _item(self, row):
        item = self.dataset[row]
        return {key: item[key] for key in [*self.image_keys, *ITEM_KEYS] if key in item}

    def _tokens(self, texts):
        return self.tokenizer([text.rstrip() + "\n" for text in texts], padding="max_length", truncation=True,
                              max_length=self.policy.config.tokenizer_max_length, return_tensors="pt")

    def _normalized(self, key, value):
        stats = self.stats[key]
        return (value.float() - torch.as_tensor(stats["mean"], dtype=torch.float32)) / torch.as_tensor(stats["std"], dtype=torch.float32)

    def _collate(self, examples):
        """Reference path without a cache: native items read at every step."""
        batch = default_collate(examples)
        tokens = self._tokens(batch.pop("task"))
        batch["observation.language.tokens"] = tokens["input_ids"]
        batch["observation.language.attention_mask"] = tokens["attention_mask"].bool()
        for key in self.stats:
            batch[key] = self._normalized(key, batch[key])
        return {key: value.to(self.device) for key, value in batch.items()}

    def _embed(self, frames):
        """Native image preprocessing and frozen vision encoder, at the training precision."""
        batch = {key: value.to(self.device) for key, value in frames.items()}
        with torch.no_grad(), precision_context(self.cfg, self.device):
            images, _ = self.policy.prepare_images(batch)
            return [self.policy.model.vlm_with_expert.embed_image(image).cpu() for image in images]

    def _build_cache(self, vision):
        """Read every loaded frame once through the dataset, exactly as a training step would."""
        rows = len(self.dataset)
        tasks, task_ids = [], torch.empty(rows, dtype=torch.long)
        columns, frames, embeddings, pending = {}, {}, {}, {}
        for row in range(rows):
            item = self._item(row)
            if item["task"] not in tasks:
                tasks.append(item["task"])
            task_ids[row] = tasks.index(item["task"])
            for key in ITEM_KEYS[:-1]:
                if key in item:
                    if key not in columns:
                        columns[key] = torch.empty((rows, *item[key].shape), dtype=item[key].dtype)
                    columns[key][row] = item[key]
            for key in self.image_keys:
                if vision:
                    pending.setdefault(key, []).append(item[key])
                    continue
                exact = (item[key] * 255.0).round().to(torch.uint8)
                if not torch.equal(exact.to(torch.float32) / 255.0, item[key]):
                    raise RuntimeError("Decoded frames are not exact 8-bit images, so cached pixels would change the training "
                                       "inputs; set dataset.frame_cache to false")
                if key not in frames:
                    frames[key] = torch.empty((rows, *exact.shape), dtype=torch.uint8)
                frames[key][row] = exact
            if pending and (len(pending[self.image_keys[0]]) == 32 or row == rows - 1):
                first = row + 1 - len(pending[self.image_keys[0]])
                for key, output in zip(self.image_keys, self._embed({key: torch.stack(value) for key, value in pending.items()})):
                    if key not in embeddings:
                        embeddings[key] = torch.empty((rows, *output.shape[1:]), dtype=output.dtype)
                    embeddings[key][first:row + 1] = output
                pending = {}
            if (row + 1) % 1000 == 0 or row + 1 == rows:
                print(f"{self.task}: cached {row + 1}/{rows} frames", flush=True)
        return {"implementation": CACHE_IMPLEMENTATION, "fingerprint": self.manifest["fingerprint"], "rows": rows,
                "tasks": tasks, "task_ids": task_ids, "columns": columns, "frames": frames, "embeddings": embeddings}

    def _load_cache(self):
        self.cache, self.cache_info = None, None
        vision = bool(self.cfg["model"].get("cache_frozen_vision", False))
        if not vision and not self.cfg["dataset"].get("frame_cache", True):
            return
        kind = "vision" if vision else "frames"
        path = Path(self.cfg["output"]) / "cache" / f"{self.task}-{kind}.pt"
        start = time.perf_counter()
        stored = None
        if path.exists():
            stored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if (stored.get("implementation"), stored.get("fingerprint"), stored.get("rows")) != (
                    CACHE_IMPLEMENTATION, self.manifest["fingerprint"], len(self.dataset)):
                print(f"{self.task}: discarding a {kind} cache written for another protocol", flush=True)
                stored = None
        built = stored is None
        if built:
            print(f"{self.task}: reading {len(self.dataset)} frames once to build the {kind} cache", flush=True)
            stored = self._build_cache(vision)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = temporary_path(path)
            torch.save(stored, temporary)
            os.replace(temporary, path)
        tokens = self._tokens(stored["tasks"])
        self.cache = {"task_ids": stored["task_ids"].to(self.device), "tokens": tokens["input_ids"].to(self.device),
                      "token_mask": tokens["attention_mask"].bool().to(self.device), "frames": stored["frames"],
                      "columns": {key: (self._normalized(key, value) if key in self.stats else value).to(self.device)
                                  for key, value in stored["columns"].items()}}
        size = sum(value.numel() * value.element_size() for value in (*stored["frames"].values(), *stored["embeddings"].values()))
        # Embeddings are only gathered, so they can live beside the model when there is ample room.
        on_device = vision and self.device.type == "cuda" and size <= 0.25 * torch.cuda.mem_get_info(self.device)[0]
        self.cache["embeddings"] = {key: value.to(self.device) if on_device else value for key, value in stored["embeddings"].items()}
        if vision:
            self._serve_cached_vision()
        self.cache_info = {"kind": kind, "rows": stored["rows"], "gib": size / 2**30, "built": built,
                           "seconds": time.perf_counter() - start, "device": str(self.device) if on_device else "cpu"}
        print(f"{self.task}: {kind} cache ready ({self.cache_info['gib']:.2f} GiB, "
              f"{'built' if built else 'loaded'} in {self.cache_info['seconds']:.1f}s)", flush=True)

    def _serve_cached_vision(self):
        """Let the native forward take cached vision embeddings where it would take pixels."""
        expert, keys = self.policy.model.vlm_with_expert, self.image_keys
        prepare, embed = self.policy.prepare_images, expert.embed_image

        def prepare_images(batch):
            if batch[keys[0]].ndim != 3:
                return prepare(batch)
            embeddings = [batch[key] for key in keys]
            return embeddings, [torch.ones(len(value), dtype=torch.bool, device=value.device) for value in embeddings]

        self.policy.prepare_images = prepare_images
        expert.embed_image = lambda image: image if image.ndim == 3 else embed(image)

    def sample(self, episodes, count, rng):
        rows = []
        for _ in range(count):
            episode = int(rng.choice(episodes))
            rows.append(int(rng.choice(self.indices[episode])))
        if self.cache is None:
            return self._collate([self._item(row) for row in rows])
        index = torch.as_tensor(rows)
        on_device = index.to(self.device)
        # The dataset's own uint8 -> [0, 1] conversion, on the CPU where it was first applied.
        batch = {key: (value[index].to(torch.float32) / 255.0).to(self.device) for key, value in self.cache["frames"].items()}
        for key, value in self.cache["embeddings"].items():
            batch[key] = value[index.to(value.device)].to(self.device)
        for key, value in self.cache["columns"].items():
            batch[key] = value[on_device]
        task = self.cache["task_ids"][on_device]
        batch["observation.language.tokens"] = self.cache["tokens"][task]
        batch["observation.language.attention_mask"] = self.cache["token_mask"][task]
        return batch

    def random_flow(self, batch, generator=None):
        shape = (len(batch["action"]), self.chunk_size, self.max_action_dim)
        noise = torch.randn(shape, generator=generator, device=self.device)
        # Native SmolVLA training uses Beta(1.5,1) (inverse CDF), scaled away from endpoints.
        time = torch.rand(shape[0], generator=generator, device=self.device).pow(1 / 1.5) * 0.999 + 0.001
        return noise, time

    def loss(self, batch, noise=None, time=None):
        if noise is None:
            noise, time = self.random_flow(batch)
        return self.policy.forward(batch, noise=noise, time=time)[0]

    def flow_and_loss(self, batch, noise, time):
        """One native forward pass yielding both the predicted flow and its training loss."""
        captured = []
        hook = self.policy.model.action_out_proj.register_forward_hook(lambda module, args, output: captured.append(output))
        try:
            loss = self.policy.forward(batch, noise=noise, time=time)[0]
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Native action projection did not expose exactly one flow prediction")
        return captured[0][:, :, :self.action_dim], loss

    def flow(self, batch, noise, time):
        return self.flow_and_loss(batch, noise, time)[0]

    def valid_mse(self, left, right, batch):
        error = (left - right).square()
        mask = batch.get("action_is_pad")
        if mask is None:
            return error.mean()
        valid = (~mask).unsqueeze(-1)
        return (error * valid).sum() / (valid.sum() * error.shape[-1]).clamp_min(1)

    def close(self):
        del self.dataset
        self.cache = None
