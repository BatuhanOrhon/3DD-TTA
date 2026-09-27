"""Compact JSON and development-beta selection checks."""
import csv
import json
from pathlib import Path
import shutil
import unittest
import uuid

from scripts.analyze_gsd_calibration import calibration_summary, screen_ranking


class AnalyzeCalibrationTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "tmp" / ("analyze-cal-" + uuid.uuid4().hex)

    def setUp(self):
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_phase_report_reduces_to_beta_weights_and_state_scales(self):
        metric = dict(count=192, missing=0, median=.001, p90=.002, min=0., max=.003)
        candidate = dict(local_ratio=metric, style_ratio=metric,
                         local_cosine=metric, weights={"0.0001": .1, "0.001": 1., "0.01": 10.})
        beta_keys = {"hard": candidate, "0.5": candidate, "2.0": candidate, "8.0": candidate}
        per_corruption = {name: beta_keys for name in ("gaussian", "impulse")}
        state_metrics = {"local_state_rms": metric, "local_scd_update_rms": metric,
                         "local_scd_update_state_ratio": metric, "local_ddim_displacement_norm": metric,
                         "local_scd_ddim_ratio": metric}
        report = dict(run_id="calibration", candidates=beta_keys,
                      by_corruption=per_corruption,
                      state_by_step={name: {str(step): state_metrics for step in range(5)}
                                     for name in per_corruption})
        source = self.root / "report.json"
        source.write_text(json.dumps(report), encoding="utf-8")
        summary = calibration_summary(source)
        self.assertEqual(summary["examples_per_corruption"], 64)
        self.assertEqual([row["beta"] for row in summary["smooth_candidates"]], [.5, 2., 8.])
        self.assertEqual(summary["smooth_candidates"][1]["alpha_by_rho"]["0.001"], 1.)
        self.assertNotIn("state_by_step", summary)

    def test_screen_ranking_requires_matched_runs_and_finds_development_winner(self):
        calibration_id = "calibration-run"
        candidates = [
            ("scd", 0., "hard", None, None, .50),
            ("beta2", 1., "smooth", 2., .001, .55),
            ("beta05", 1., "smooth", .5, .001, .70),
            ("beta8", 1., "smooth", 8., .001, .60),
            ("hard", 1., "hard", None, .001, .65),
        ]
        paths = []
        for name, weight, profile, beta, rho, accuracy in candidates:
            directory = self.root / name
            directory.mkdir()
            split = {corruption: {"indices": [4, 2], "total_examples": 20}
                     for corruption in ("gaussian", "impulse")}
            config = dict(stage="development", execution_status="complete", run_id=name,
                          completed_corruptions=["gaussian", "impulse"], seed=0,
                          calibration_reference={"sha256": "same-hash", "run_id": calibration_id,
                                                 "target_rho": rho},
                          development_split=split,
                          cli_args={"gsd_weight": weight, "gsd_profile": profile,
                                    "gsd_beta": beta, "gsd_development_count": 32})
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with (directory / "per_corruption.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("corruption", "status", "n_examples", "accuracy"))
                writer.writeheader()
                for corruption in ("gaussian", "impulse"):
                    writer.writerow(dict(corruption=corruption, status="partial", n_examples=2,
                                         accuracy=accuracy))
            paths.append(directory)
        weights = {"hard": {"0.001": 1.}, "beta_0.5": {"0.001": 1.},
                   "beta_2.0": {"0.001": 1.}, "beta_8.0": {"0.001": 1.}}
        result = screen_ranking(paths, calibration_id, weights)
        self.assertEqual(result["best_observed_candidates"], ["beta_0.5"])
        self.assertAlmostEqual(result["ranking"][0]["macro_accuracy"], .70)
        paths[1].joinpath("config.json").write_text(
            paths[1].joinpath("config.json").read_text(encoding="utf-8").replace("same-hash", "other"),
            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different calibration"):
            screen_ranking(paths, calibration_id, weights)


if __name__ == "__main__":
    unittest.main()
