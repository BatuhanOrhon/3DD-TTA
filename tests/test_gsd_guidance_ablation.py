"""CPU contracts for missing-ablation launches and prediction-bearing artifacts."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from research_artifacts import CORRUPTIONS
from scripts.analyze_gsd_full_dataset_screen import read_full_run
from scripts.run_gsd_guidance_ablation import build_commands, run_ablation, _matching_runs
from tests.test_analyze_gsd_full_dataset_screen import FullDatasetAnalyzerTests


class AblationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = FullDatasetAnalyzerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def _bundle(self, arm="unguided"):
        path = self.fixture._bundle("scd" if arm == "unguided" else "beta05", 0,
                                    counts={c: 2468 for c in CORRUPTIONS})
        def change(raw):
            c = json.loads(raw)
            c["stage"] = c["cli_args"]["gsd_stage"] = "full_dataset_ablation"
            c["spectral"]["scd_weight"] = c["cli_args"]["gsd_scd_weight"] = 0.
            c["guidance_ablation"] = arm
            c["calibration_source_compatibility"]["allowed_source_extensions"].append("tta_gsd.py")
            c["observed_batch_sizes"] = {name: [32] * 77 + [4] for name in CORRUPTIONS}
            c["per_example_predictions"] = {
                name: {"sample_indices": list(range(2468)), "labels": [0] * 2468,
                       "predictions": [0] * i + [1] * (2468-i)}
                for i, name in enumerate(CORRUPTIONS)}
            return json.dumps(c).encode()
        self.fixture._rewrite(path, "config.json", change)
        self.fixture._rewrite(path, "command.txt", lambda raw: raw.replace(
            b"full_dataset_development", b"full_dataset_ablation"))
        return path

    def test_both_full_artifacts_and_tampered_predictions(self):
        for arm in ("unguided", "smooth_only"):
            path = self._bundle(arm)
            self.assertEqual(read_full_run(path, guidance_ablation=True)["arm"], arm)
            with self.assertRaises(ValueError):
                read_full_run(path)
        def corrupt(raw):
            c = json.loads(raw)
            c["per_example_predictions"][CORRUPTIONS[0]]["predictions"][0] = 0
            return json.dumps(c).encode()
        self.fixture._rewrite(path, "config.json", corrupt)
        with self.assertRaisesRegex(ValueError, "predictions disagree"):
            read_full_run(path, guidance_ablation=True)

    def test_commands_are_six_missing_conditions_with_no_scd(self):
        with patch("scripts.run_gsd_full_dataset_screen.load_pinned_calibration",
                   return_value=self.fixture._calibration()):
            for arm, weight in (("unguided", "0.0"), ("smooth_only", "8.140161356429882")):
                commands = build_commands(arm, Path("config.json"), "./result")
                self.assertEqual(len(commands), 3)
                for seed, cmd in enumerate(commands):
                    get = lambda flag: cmd[cmd.index(flag)+1]
                    self.assertEqual(get("--seed"), str(seed))
                    self.assertEqual(get("--gsd-weight"), weight)
                    self.assertEqual(get("--gsd-scd-weight"), "0")
                    self.assertEqual(get("--max-batches"), "0")
                    self.assertEqual(get("--gsd-stage"), "full_dataset_ablation")

    def test_first_failure_stops_and_resume_skips_all_completed(self):
        module = "scripts.run_gsd_guidance_ablation."
        with patch("scripts.run_gsd_full_dataset_screen.load_pinned_calibration",
                   return_value=self.fixture._calibration()), \
                patch(module + "run_streamed", return_value=(1, "failed", None)) as child:
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                run_ablation("unguided", Path("config.json"), str(self.fixture.root))
            self.assertEqual(child.call_count, 1)
        with patch("scripts.run_gsd_full_dataset_screen.load_pinned_calibration",
                   return_value=self.fixture._calibration()), \
                patch(module + "_matching_runs", return_value=[{"path": "completed.zip"}]), \
                patch(module + "_check_sources"), patch(module + "summarize_runs", return_value={}), \
                patch(module + "run_streamed", side_effect=AssertionError("must skip")):
            run_ablation("unguided", Path("config.json"), str(self.fixture.root))

    def test_successful_attempt_suffix_is_resumed(self):
        name = "gsd-ablation-unguided-seed0"
        path = self.fixture.root / ("20300101-000000_" + name + "-attempt02.zip")
        path.touch()
        run = {"arm": "unguided", "seed": 0, "config": {"cli_args": {"run_name": name + "-attempt02"}}}
        with patch("scripts.run_gsd_guidance_ablation.read_full_run", return_value=run):
            self.assertEqual(_matching_runs(self.fixture.root, name, "unguided", 0), [run])

    def test_existing_calibrated_stage_accepts_only_declared_branch_extension(self):
        from gsd_calibration import verify_calibration_inputs
        path = self.fixture._bundle("scd", 0)
        def changed(raw):
            c = json.loads(raw)
            c["runtime_source_manifest"]["tta_gsd.py"] = {"sha256": "c" * 64}
            c["calibration_reference"]["expected_manifests"]["runtime_source_manifest"]["tta_gsd.py"] = {"sha256": "d" * 64}
            return json.dumps(c).encode()
        self.fixture._rewrite(path, "config.json", changed)
        with self.assertRaisesRegex(ValueError, "source identity mismatch"):
            read_full_run(path)
        def declare(raw):
            c = json.loads(raw)
            c["calibration_source_compatibility"]["allowed_source_extensions"].append("tta_gsd.py")
            return json.dumps(c).encode()
        self.fixture._rewrite(path, "config.json", declare)
        config = read_full_run(path)["config"]
        verify_calibration_inputs(config)
        config["calibration_source_compatibility"]["allowed_source_extensions"].remove("tta_gsd.py")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            verify_calibration_inputs(config)


if __name__ == "__main__":
    unittest.main()
