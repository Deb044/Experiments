"""Explicit CPU-only fixture, never evidence about real VLA unlearning."""
import numpy as np
import torch
from torch import nn

from ..lora import attach_lora


class TinyFlow(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q_proj = nn.Linear(state_dim + action_dim + 1, 12)
        self.v_proj = nn.Linear(12, action_dim)

    def forward(self, state, noisy_actions, time):
        x = torch.cat([state, noisy_actions[:, 0], time[:, None]], dim=-1)
        return self.v_proj(torch.tanh(self.q_proj(x)))[:, None, :]


class SyntheticBackend:
    def __init__(self, manifest, task, seed):
        self.manifest = manifest
        self.cfg = manifest["config"]
        self.task = task
        self.device = torch.device(self.cfg["training"]["device"])
        spec = self.cfg["synthetic"]
        self.action_dim, self.chunk_size, self.max_action_dim = spec["action_dim"], 1, spec["action_dim"]
        with torch.random.fork_rng():
            torch.manual_seed(1000)
            self.policy = TinyFlow(spec["state_dim"], self.action_dim).to(self.device)
        torch.manual_seed(seed)
        self.modules = attach_lora(self.policy, self.cfg["model"]["rank"])
        rng = np.random.default_rng(100)
        teacher = rng.normal(size=(spec["state_dim"], self.action_dim)).astype(np.float32)
        self.data = {}
        for i, task_id in enumerate(manifest["tasks"]):
            ids = set(manifest["tasks"][task_id]["train"] + manifest["tasks"][task_id]["calibration"] + manifest["tasks"][task_id]["heldout"])
            for episode in sorted(ids):
                local = np.random.default_rng(episode + 314)
                state = local.normal(size=(spec["frames_per_episode"], spec["state_dim"])).astype(np.float32)
                action = np.tanh(state @ teacher + 0.15 * i + local.normal(0, 0.05, size=(len(state), self.action_dim))).astype(np.float32)
                self.data[episode] = {"state": state, "action": action[:, None, :]}
        calibration = manifest["tasks"][task]["calibration"]
        self.stats = {}
        for key in ("state", "action"):
            values = np.concatenate([self.data[ep][key] for ep in calibration])
            self.stats[key] = (values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6))

    def sample(self, episodes, count, rng):
        batch = {"state": [], "action": []}
        for _ in range(count):
            episode = int(rng.choice(episodes))
            frame = int(rng.integers(len(self.data[episode]["state"])))
            for key in batch:
                mean, std = self.stats[key]
                batch[key].append((self.data[episode][key][frame] - mean) / std)
        return {key: torch.tensor(np.stack(value), device=self.device) for key, value in batch.items()}

    def random_flow(self, batch, generator=None):
        noise = torch.randn(batch["action"].shape, generator=generator, device=self.device)
        time = torch.rand(len(noise), generator=generator, device=self.device) * 0.998 + 0.001
        return noise, time

    def flow(self, batch, noise, time):
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * batch["action"]
        return self.policy(batch["state"], noisy, time)

    def loss(self, batch, noise=None, time=None):
        if noise is None:
            noise, time = self.random_flow(batch)
        return ((self.flow(batch, noise, time) - (noise - batch["action"])) ** 2).mean()

    def valid_mse(self, left, right, batch):
        return ((left - right) ** 2).mean()

    def close(self):
        self.data.clear()

