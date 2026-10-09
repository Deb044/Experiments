"""CPU training contracts: real gradients, frozen bases, and exact resume."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from vla_weightspace.backends import create_backend
from vla_weightspace.config import load_config
from vla_weightspace.io import read_json, read_npz
from vla_weightspace.lora import LoRALinear, effective_gradients, effective_updates, factors
from vla_weightspace.manifest import make_manifest, run_specs
from vla_weightspace.training import train_one


class TrainingTests(unittest.TestCase):
    def fixture(self, output):
        cfg = load_config(Path(__file__).parents[1] / "configs/smoke.json")
        cfg["output"] = str(output)
        cfg["training"]["steps"] = 4
        episodes = {task["id"]: list(range(i * 20, (i + 1) * 20)) for i, task in enumerate(cfg["tasks"])}
        return make_manifest(cfg, episodes, {"synthetic": True})

    def test_effective_update_is_gauge_invariant(self):
        a = np.array([[1., 2., 3.], [2., 1., 0.]], dtype=np.float32)
        b = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        change = np.array([[2., 0.], [0., 0.5]], dtype=np.float32)
        left = effective_updates({"x.A": a, "x.B": b})["x"]
        right = effective_updates({"x.A": change @ a, "x.B": b @ np.linalg.inv(change)})["x"]
        np.testing.assert_allclose(left, right)

    def test_effective_gradient_matches_dense_finite_difference(self):
        base = nn.Linear(3, 2, bias=False)
        wrapper = LoRALinear(base, 2)
        x = torch.randn(4, 3)
        wrapper.capture_gradient = True
        loss = wrapper(x).square().mean()
        gradient = torch.autograd.grad(loss, wrapper.gradient_leaves)[0]
        direction = torch.randn_like(gradient)
        epsilon = 1e-3
        dense = base.weight.detach() + wrapper.delta().detach()
        plus = torch.nn.functional.linear(x, dense + epsilon * direction).square().mean()
        minus = torch.nn.functional.linear(x, dense - epsilon * direction).square().mean()
        self.assertAlmostEqual(float((plus - minus) / (2 * epsilon)), float((gradient * direction).sum()), places=3)

    def test_resume_matches_uninterrupted_training_and_freezes_base(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.fixture(Path(directory) / "first")
            backend = create_backend(first, "goal-0", 0)
            initial = factors(backend.modules)
            base = {name: module.base.weight.detach().clone() for name, module in backend.modules.items()}
            spec = next(run_specs(first))
            train_one(backend, spec, first, initial)
            expected = factors(backend.modules)
            for name, module in backend.modules.items():
                torch.testing.assert_close(module.base.weight, base[name], rtol=0, atol=0)
            second = self.fixture(Path(directory) / "second")
            resumed = create_backend(second, "goal-0", 0)
            initial_resumed = factors(resumed.modules)
            train_one(resumed, spec, second, initial_resumed, max_updates=2)
            train_one(resumed, spec, second, initial_resumed)
            for name in expected:
                np.testing.assert_array_equal(expected[name], factors(resumed.modules)[name])

    def test_paired_runs_share_initialization_and_exclude_forget(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.fixture(Path(directory))
            backend = create_backend(manifest, "goal-0", 0)
            initial = factors(backend.modules)
            specs = list(run_specs(manifest))[:2]
            results = [train_one(backend, spec, manifest, initial) for spec in specs]
            self.assertEqual(results[0]["initial_hash"], results[1]["initial_hash"])
            deleted = manifest["tasks"]["goal-0"]["requests"][specs[1]["request"]]["forget"]
            self.assertFalse(set(specs[1]["episodes"]) & set(deleted))

    def test_gradient_capture_does_not_update_factors(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.fixture(Path(directory))
            backend = create_backend(manifest, "goal-0", 0)
            initial = factors(backend.modules)
            batch = backend.sample(manifest["tasks"]["goal-0"]["train"], 3, np.random.default_rng(5))
            noise, time = backend.random_flow(batch)
            gradients = effective_gradients(backend, batch, noise, time)
            self.assertGreater(sum(np.linalg.norm(value) for value in gradients.values()), 0)
            for name, value in initial.items():
                np.testing.assert_array_equal(value, factors(backend.modules)[name])


if __name__ == "__main__":
    unittest.main()
