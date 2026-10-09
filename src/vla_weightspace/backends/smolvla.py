"""Native SmolVLA flow matching, trajectory-only sampling, calibration-only stats.

The action-output hook observes the native v_t without copying the model's
attention or flow implementation. This deliberately fails on incompatible APIs.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from ..io import read_json, write_json
from ..lora import attach_lora


class SmolVLABackend:
    def __init__(self, manifest, task, seed):
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

    def sample(self, episodes, count, rng):
        examples = []
        for _ in range(count):
            episode = int(rng.choice(episodes))
            index = int(rng.choice(self.indices[episode]))
            item = self.dataset[index]
            examples.append({key: item[key] for key in [*self.image_keys, "observation.state", "action", "action_is_pad", "task"] if key in item})
        batch = default_collate(examples)
        tokens = self.tokenizer([text.rstrip() + "\n" for text in batch.pop("task")], padding="max_length", truncation=True,
                                max_length=self.policy.config.tokenizer_max_length, return_tensors="pt")
        batch["observation.language.tokens"] = tokens["input_ids"]
        batch["observation.language.attention_mask"] = tokens["attention_mask"].bool()
        for key, stats in self.stats.items():
            mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
            std = torch.as_tensor(stats["std"], dtype=torch.float32)
            batch[key] = (batch[key].float() - mean) / std
        return {key: value.to(self.device) for key, value in batch.items()}

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

    def flow(self, batch, noise, time):
        captured = []
        hook = self.policy.model.action_out_proj.register_forward_hook(lambda module, args, output: captured.append(output))
        try:
            self.policy.forward(batch, noise=noise, time=time)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Native action projection did not expose exactly one flow prediction")
        return captured[0][:, :, :self.action_dim]

    def valid_mse(self, left, right, batch):
        error = (left - right).square()
        mask = batch.get("action_is_pad")
        if mask is None:
            return error.mean()
        valid = (~mask).unsqueeze(-1)
        return (error * valid).sum() / (valid.sum() * error.shape[-1]).clamp_min(1)

    def close(self):
        del self.dataset
