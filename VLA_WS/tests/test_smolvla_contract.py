"""Exercise the native-flow hook and frame-cache contracts without downloading a foundation model."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from vla_weightspace.backends.smolvla import SmolVLABackend

KEYS = ["observation.images.image", "observation.images.image2"]


class FrameFixture:
    """Stands in for the episode-filtered dataset: deterministic 8-bit frames, counted reads."""

    def __init__(self, rows=12):
        self.rows, self.reads = rows, 0

    def __len__(self):
        return self.rows

    def __getitem__(self, row):
        self.reads += 1
        generator = torch.Generator().manual_seed(row)
        frames = [torch.randint(0, 256, (3, 8, 8), generator=generator, dtype=torch.uint8).to(torch.float32) / 255.0 for _ in KEYS]
        return {KEYS[0]: frames[0], KEYS[1]: frames[1], "observation.state": torch.randn(4, generator=generator),
                "action": torch.randn(5, 3, generator=generator), "action_is_pad": torch.arange(5) >= 5 - row % 3,
                "task": "open the drawer" if row < 6 else "close the drawer ", "timestamp": torch.tensor(row / 10)}


def tokenizer(texts, padding, truncation, max_length, return_tensors):
    ids = torch.zeros(len(texts), max_length, dtype=torch.long)
    for row, text in enumerate(texts):
        codes = [ord(character) for character in text][:max_length]
        ids[row, :len(codes)] = torch.tensor(codes)
    return {"input_ids": ids, "attention_mask": (ids > 0).long()}


class VisionTowerFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 6)
        self.calls = 0

    def embed_image(self, image):
        self.calls += 1
        return self.proj(image.flatten(2).transpose(1, 2))


class CachedPolicyFixture(nn.Module):
    """Pixels -> preprocessing -> frozen encoder -> action projection, like the native forward."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tokenizer_max_length=6)
        self.model = nn.Module()
        self.model.vlm_with_expert = VisionTowerFixture()
        self.model.action_out_proj = nn.Linear(6, 3)

    def prepare_images(self, batch):
        images = [batch[key] * 2.0 - 1.0 for key in KEYS]
        return images, [torch.ones(len(image), dtype=torch.bool) for image in images]

    def forward(self, batch, noise=None, time=None):
        images, _ = self.prepare_images(batch)
        hidden = sum(self.model.vlm_with_expert.embed_image(image).mean(dim=1) for image in images)
        flow = self.model.action_out_proj(hidden[:, None, :] + batch["action"].sum(dim=-1, keepdim=True))
        return (flow - noise).square().mean(), {}


def cached_backend(output, dataset, vision=False, fingerprint="protocol-a"):
    backend = SmolVLABackend.__new__(SmolVLABackend)
    backend.manifest, backend.task = {"fingerprint": fingerprint}, "goal-0"
    backend.cfg = {"output": str(output), "dataset": {}, "model": {"cache_frozen_vision": vision}, "training": {"precision": "fp32"}}
    backend.device = torch.device("cpu")
    backend.dataset, backend.image_keys, backend.action_dim = dataset, list(KEYS), 3
    torch.manual_seed(0)
    backend.policy = CachedPolicyFixture()
    backend.tokenizer = tokenizer
    backend.indices = {episode: np.arange(3 * episode, 3 * episode + 3) for episode in range(4)}
    backend.stats = {"observation.state": {"mean": [0.1, 0.2, 0.3, 0.4], "std": [1.0, 2.0, 0.5, 1.5]},
                     "action": {"mean": [0.0, 1.0, -1.0], "std": [0.5, 1.0, 2.0]}}
    backend._load_cache()
    return backend


class NativePolicyFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.action_out_proj = nn.Linear(3, 4)

    def forward(self, batch, noise=None, time=None):
        flow = self.model.action_out_proj(batch["hidden"])
        return (flow - noise).square().mean(), {}


class SmolVLAContractTests(unittest.TestCase):
    def test_hook_returns_native_flow_not_squared_training_loss(self):
        backend = SmolVLABackend.__new__(SmolVLABackend)
        backend.policy = NativePolicyFixture()
        backend.action_dim = 2
        batch = {"hidden": torch.randn(2, 3, 3)}
        noise = torch.randn(2, 3, 4)
        time = torch.rand(2)
        actual = backend.flow(batch, noise, time)
        expected = backend.policy.model.action_out_proj(batch["hidden"])[:, :, :2]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(backend.policy.model.action_out_proj._forward_hooks), 0)

    def test_native_validity_mask_excludes_padded_actions(self):
        backend = SmolVLABackend.__new__(SmolVLABackend)
        left = torch.tensor([[[1., 1.], [100., 100.]]])
        right = torch.zeros_like(left)
        batch = {"action_is_pad": torch.tensor([[False, True]])}
        self.assertEqual(float(backend.valid_mse(left, right, batch)), 1)

    def test_one_forward_returns_the_flow_and_loss_of_separate_passes(self):
        backend = SmolVLABackend.__new__(SmolVLABackend)
        backend.policy = NativePolicyFixture()
        backend.action_dim = 2
        batch, noise = {"hidden": torch.randn(2, 3, 3)}, torch.randn(2, 3, 4)
        flow, loss = backend.flow_and_loss(batch, noise, torch.rand(2))
        self.assertTrue(torch.equal(flow, backend.flow(batch, noise, None)))
        self.assertTrue(torch.equal(loss, backend.loss(batch, noise, None)))
        self.assertEqual(len(backend.policy.model.action_out_proj._forward_hooks), 0)


class SmolVLACacheTests(unittest.TestCase):
    def test_cached_pixels_reproduce_native_items_and_sampler_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = FrameFixture()
            backend = cached_backend(directory, dataset)
            self.assertEqual(dataset.reads, len(dataset))  # every frame is read exactly once
            cached_rng, native_rng = np.random.default_rng(7), np.random.default_rng(7)
            for _ in range(5):
                cached = backend.sample([0, 2, 3], 4, cached_rng)
                cache, backend.cache = backend.cache, None  # reference: native items at every step
                native = backend.sample([0, 2, 3], 4, native_rng)
                backend.cache = cache
                self.assertEqual(set(cached), set(native))
                for key in native:
                    self.assertEqual(cached[key].dtype, native[key].dtype, key)
                    self.assertTrue(torch.equal(cached[key], native[key]), key)
            self.assertEqual(cached_rng.bit_generator.state, native_rng.bit_generator.state)
            self.assertEqual(dataset.reads, len(dataset) + 20)

    def test_cache_is_reused_and_rebuilt_for_another_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = cached_backend(directory, FrameFixture()).sample([1, 2], 3, np.random.default_rng(3))
            unread = FrameFixture()
            actual = cached_backend(directory, unread).sample([1, 2], 3, np.random.default_rng(3))
            self.assertEqual(unread.reads, 0)
            for key in expected:
                self.assertTrue(torch.equal(actual[key], expected[key]), key)
            other = FrameFixture()
            cached_backend(directory, other, fingerprint="protocol-b")
            self.assertEqual(other.reads, len(other))

    def test_inexact_frames_are_refused_and_the_cache_can_be_disabled(self):
        class Smooth(FrameFixture):
            def __getitem__(self, row):
                item = super().__getitem__(row)
                item[KEYS[0]] = item[KEYS[0]] * 0.999
                return item

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "exact 8-bit"):
                cached_backend(directory, Smooth())
            backend = SmolVLABackend.__new__(SmolVLABackend)
            backend.cfg = {"output": directory, "dataset": {"frame_cache": False}, "model": {}}
            backend._load_cache()
            self.assertIsNone(backend.cache)

    def test_cached_frozen_vision_matches_the_native_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            pixels = cached_backend(Path(directory) / "pixels", FrameFixture())
            vision = cached_backend(Path(directory) / "vision", FrameFixture(), vision=True)
            tower = vision.policy.model.vlm_with_expert
            calls = tower.calls
            for seed in range(3):
                native = pixels.sample([0, 1, 3], 4, np.random.default_rng(seed))
                cached = vision.sample([0, 1, 3], 4, np.random.default_rng(seed))
                self.assertEqual(cached[KEYS[0]].shape, (4, 64, 6))
                noise = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(seed))
                expected_flow, expected_loss = pixels.flow_and_loss(native, noise, None)
                flow, loss = vision.flow_and_loss(cached, noise, None)
                torch.testing.assert_close(flow, expected_flow)
                torch.testing.assert_close(loss, expected_loss)
            self.assertEqual(tower.calls, calls)  # the frozen encoder no longer runs during updates
            torch.testing.assert_close(vision.flow_and_loss(native, noise, None)[0], expected_flow)  # pixels still work


if __name__ == "__main__":
    unittest.main()
