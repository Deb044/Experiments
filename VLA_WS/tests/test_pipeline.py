"""End-to-end artifact and provenance verification on the explicit fixture."""
from pathlib import Path
import tempfile
import unittest

from vla_weightspace.cli import main
from vla_weightspace.io import read_json, write_json


class PipelineTests(unittest.TestCase):
    def test_all_stages_and_expected_scientific_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = read_json(Path(__file__).parents[1] / "configs/smoke.json")
            cfg["output"] = str(root / "artifacts")
            cfg["training"]["steps"] = 2
            cfg["training"]["checkpoint_every"] = 2
            cfg["probes"].update(windows=2, draws=1, gradient_windows=2)
            cfg["benchmark"].update(warmup=1, updates=2)
            config = root / "fixture.json"
            write_json(config, cfg)
            main(["--config", str(config), "run"])
            out = Path(cfg["output"])
            self.assertEqual(len(list((out / "runs").rglob("complete.json"))), 24)
            pairs = read_json(out / "pairs/index.json")
            self.assertEqual(len(pairs["pairs"]), 18)
            metrics = read_json(out / "predictions/metrics.json")
            self.assertEqual(len(metrics["rows"]), 360)
            self.assertTrue(metrics["synthetic"])
            for audit in metrics["audits"]:
                self.assertFalse(set(audit["train_ids"]) & set(audit["test_ids"]))
                self.assertTrue(audit["validation_sse"])
            functional = read_json(out / "functional/metrics.json")
            self.assertEqual(len(functional["rows"]), 180)
            for row in functional["rows"]:
                if row["method"] == "zero":
                    self.assertAlmostEqual(row["discrepancy"], row["original_discrepancy"], delta=1e-8)
            summary = read_json(out / "report/summary.json")
            self.assertTrue(summary["functional_evaluated"])
            self.assertTrue(summary["synthetic"])
            for name in ("magnitude.png", "layer_predictability.png", "heldout_tasks.png", "prediction_summary.csv", "variability.csv"):
                self.assertGreater((out / "report" / name).stat().st_size, 100)
            self.assertIn("NOT VLA RESEARCH RESULTS", (out / "report/results.md").read_text())
            # Reusing completed stages must be safe and preserve data provenance.
            main(["--config", str(config), "train"])
            cfg["training"]["steps"] = 3
            write_json(config, cfg)
            with self.assertRaises(ValueError):
                main(["--config", str(config), "train"])


if __name__ == "__main__":
    unittest.main()
