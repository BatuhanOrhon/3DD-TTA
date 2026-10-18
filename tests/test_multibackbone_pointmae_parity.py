"""Native-free checks for the prepared parity invocation and comparison logic."""

import importlib.util
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_multibackbone_pointmae_parity.py"


class ParityPreparationTests(unittest.TestCase):
    def test_bad_manifest_stops_before_native_imports_and_seals_failed_bundle_once(self):
        spec = importlib.util.spec_from_file_location("pointmae_parity_failure", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "manifest.json").write_text("{}")
            arguments = ["--asset-root", str(root), "--result-root", str(root / "results"),
                         "--asset-manifest-sha256", "0" * 64]
            with patch.object(module.source_runner, "command_output", return_value="CPU failure test"), redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    module.main(arguments)
            self.assertNotEqual(failure.exception.code, 0)
            archives = list((root / "results").rglob("*.zip"))
            self.assertEqual(len(archives), 1)
            with zipfile.ZipFile(archives[0]) as archive:
                config_path = next(name for name in archive.namelist() if name.endswith("/config.json"))
                config = json.loads(archive.read(config_path))
                self.assertEqual(config["status"], "failed")
                self.assertIn("Asset manifest SHA-256 mismatch", config["failure"])
                self.assertNotIn("runtime", config)
                self.assertNotIn("model_state_before_sha256", config)
                self.assertTrue(any(name.endswith("/bundle_manifest.json") for name in archive.namelist()))

    def test_help_and_preview_need_no_heavy_dependencies_or_existing_assets(self):
        for arguments in (["--help"], ["--asset-root", "/missing/assets", "--result-root",
                                      "/missing/results", "--preview"]):
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, "-S", str(SCRIPT), *arguments],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("32", result.stdout)

    def test_pair_comparison_restores_all_rngs_and_detects_changed_logits(self):
        import numpy as np
        import torch
        from torch import nn
        from classifier_backbones import FrozenClassifier

        spec = importlib.util.spec_from_file_location("pointmae_parity", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class RandomClassifier(nn.Module):
            def classification_only(self, points, *, only_unmasked):
                self.asserted_unmasked = only_unmasked
                noise = module.random.random() + np.random.random() + torch.rand(())
                return points.mean(1) + noise

        model = RandomClassifier().eval()
        classifier = FrozenClassifier(model, "pointmae", 3, input_layout="bnc")
        points = torch.ones(1, 5, 3)
        direct, adapted, checks = module.compare_calls(model, classifier, points)
        self.assertTrue(torch.equal(direct, adapted))
        self.assertTrue(checks["logits_equal"])
        self.assertTrue(checks["predictions_equal"])
        self.assertTrue(checks["rng_equal"])
        self.assertEqual(checks["max_abs_diff"], 0.)
        self.assertEqual(checks["n_unequal_logits"], 0)
        self.assertFalse(model.asserted_unmasked)
        _, _, mismatch = module.compare_calls(model, lambda xyz: classifier(xyz) + 1., points)
        self.assertFalse(mismatch["logits_equal"])
        self.assertEqual(mismatch["n_unequal_logits"], 3)
        self.assertAlmostEqual(mismatch["max_abs_diff"], 1., places=6)


if __name__ == "__main__":
    unittest.main()
