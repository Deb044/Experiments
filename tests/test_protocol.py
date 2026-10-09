"""Scientific protocol tests runnable with only Python and NumPy."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vla_weightspace.config import load_config
from vla_weightspace.io import read_json, read_npz, write_json, write_npz
from vla_weightspace.manifest import make_manifest, run_specs
from vla_weightspace.prediction import folds, layer_metrics, pca_ridge, select_alpha, shuffle_within_models


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).parents[1] / "configs/smoke.json")
        self.episodes = {task["id"]: list(range(i * 30, (i + 1) * 30)) for i, task in enumerate(self.cfg["tasks"])}

    def test_complete_trajectory_partition_and_shared_requests(self):
        manifest = make_manifest(self.cfg, self.episodes, {"synthetic": True})
        self.assertEqual(manifest, make_manifest(self.cfg, self.episodes, {"synthetic": True}))
        self.assertEqual(len(list(run_specs(manifest))), 24)
        for task, entry in manifest["tasks"].items():
            all_ids = entry["train"] + entry["calibration"] + entry["heldout"]
            self.assertEqual(set(all_ids), set(self.episodes[task]))
            self.assertEqual(len(all_ids), len(set(all_ids)))
            for request in entry["requests"].values():
                self.assertFalse(set(request["forget"]) & set(request["retain"]))
                self.assertEqual(set(request["forget"] + request["retain"]), set(entry["train"]))
                self.assertFalse(set(request["forget"]) & set(entry["heldout"]))

    def test_rejects_cross_task_trajectory_leakage(self):
        episodes = copy.deepcopy(self.episodes)
        episodes["object-0"][0] = 0
        with self.assertRaises(ValueError):
            make_manifest(self.cfg, episodes, {})

    def test_outer_folds_exclude_all_heldout_task_or_request(self):
        records = [{"task": task, "seed": seed, "request": request} for task in ("a", "b", "c")
                   for seed in (0, 1) for request in ("1", "2", "3")]
        for mode, heldout, train, test in folds(records):
            key = "task" if mode == "task" else "request"
            self.assertFalse(set(train) & set(test))
            self.assertTrue(all(records[i][key] != heldout for i in train))
            self.assertTrue(all(records[i][key] == heldout for i in test))

    def test_snapshot_pca_predicts_a_known_linear_change(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(12, 3))
        matrix = rng.normal(size=(3, 20))
        y = x @ matrix
        xt = rng.normal(size=(4, 3))
        prediction = pca_ridge(x, y, xt, 1e-8, 8)
        np.testing.assert_allclose(prediction, xt @ matrix, atol=1e-6)

    def test_pca_rank_deficiency_and_constant_features(self):
        x = np.ones((4, 2))
        y = np.ones((4, 50)) * 3
        prediction = pca_ridge(x, y, np.zeros((2, 2)), 1.0, 64)
        np.testing.assert_allclose(prediction, np.ones((2, 50)) * 3)

    def test_shuffled_control_stays_within_checkpoint(self):
        records = [{"task": task, "seed": 0, "request": request} for task in ("a", "b") for request in ("1", "2", "3")]
        x = np.arange(6)[:, None]
        shuffled = shuffle_within_models(x, records)
        self.assertTrue(np.all(shuffled != x))
        self.assertEqual(set(shuffled[:3, 0]), {0, 1, 2})
        self.assertEqual(set(shuffled[3:, 0]), {3, 4, 5})

    def test_nested_ridge_selection_uses_checkpoint_groups(self):
        records = [{"task": task, "seed": seed, "request": request} for task in ("a", "b", "c") for seed in (0, 1) for request in ("1", "2")]
        x = np.arange(len(records), dtype=float)[:, None]
        y = np.column_stack([x[:, 0], -x[:, 0]])
        alpha, scores = select_alpha(x, y, records, "request", [0.01, 1.0], 2)
        self.assertIn(alpha, scores)
        self.assertTrue(all(np.isfinite(v) for v in scores.values()))

    def test_zero_changes_have_undefined_normalized_metrics(self):
        result = layer_metrics(np.zeros((3, 4)), np.zeros((3, 4)))
        self.assertIsNone(result["gain_over_zero"])
        self.assertIsNone(result["cosine"])
        self.assertEqual(layer_metrics(np.ones((2, 2)), np.ones((2, 2)))["gain_over_zero"], 1)

    def test_atomic_artifacts_preserve_array_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_json(path / "a.json", {"value": 2})
            self.assertEqual(read_json(path / "a.json"), {"value": 2})
            write_npz(path / "a.npz", {"x": np.arange(12).reshape(3, 4)})
            np.testing.assert_array_equal(read_npz(path / "a.npz")["x"], np.arange(12).reshape(3, 4))


if __name__ == "__main__":
    unittest.main()
