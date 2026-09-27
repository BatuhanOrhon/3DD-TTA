"""Compact JSON and development-beta selection checks."""
import csv
import json
from pathlib import Path
import shutil
import unittest
import uuid
from zipfile import ZipFile

from scripts.analyze_gsd_calibration import calibration_summary, screen_ranking, screen_weight_ranking


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
            split = {corruption: {"indices": list(range(100)), "total_examples": 100}
                     for corruption in ("gaussian", "impulse")}
            config = dict(stage="development", execution_status="complete", run_id=name,
                          completed_corruptions=["gaussian", "impulse"], seed=0,
                          calibration_reference={"sha256": "same-hash", "run_id": calibration_id,
                                                 "target_rho": rho},
                          development_split=split,
                          cli_args={"gsd_weight": weight, "gsd_profile": profile,
                                    "gsd_beta": beta, "gsd_development_count": 100})
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with (directory / "per_corruption.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("run_id", "corruption", "status", "n_examples", "n_correct", "accuracy"))
                writer.writeheader()
                for corruption in ("gaussian", "impulse"):
                    writer.writerow(dict(run_id=name, corruption=corruption, status="partial", n_examples=100,
                                         n_correct=round(accuracy * 100), accuracy=accuracy))
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

    def test_weight_screen_ranks_four_complete_zip_bundles(self):
        weights = {"0.0001": 1., "0.001": 2., "0.01": 3.}
        report = {"run_id": "calibration-run", "candidates": {"2.0": {"weights": weights}}}
        conditions = [("baseline", 0., None, .5),
                      ("rho0001", 1., .0001, .55),
                      ("rho001", 2., .001, .6),
                      ("rho01", 3., .01, .595)]
        split = {corruption: {"indices": list(range(200)), "total_examples": 200}
                 for corruption in ("gaussian", "impulse")}
        paths = []
        for name, weight, rho, accuracy in conditions:
            config = dict(stage="development", execution_status="complete", status="partial",
                          run_id=name, completed_corruptions=["gaussian", "impulse"], seed=0,
                          calibration_reference={"sha256": "same-hash", "run_id": "calibration-run",
                                                 "target_rho": rho}, development_split=split,
                          asset_manifest={"checkpoint": {"sha256": "asset"}},
                          dataset_hash_manifest={"gaussian": {"sha256": "data"}},
                          runtime_source_manifest={"source.py": {"sha256": "source"}},
                          cli_args={"gsd_weight": weight, "gsd_profile": "hard" if rho is None else "smooth",
                                    "gsd_beta": None if rho is None else 2., "gsd_development_count": 32})
            stream = __import__("io").StringIO()
            writer = csv.DictWriter(stream, fieldnames=("run_id", "corruption", "status", "n_examples", "n_correct", "accuracy"))
            writer.writeheader()
            correct = round(accuracy * 200)
            for corruption in ("gaussian", "impulse"):
                writer.writerow(dict(run_id=name, corruption=corruption, status="partial",
                                     n_examples=200, n_correct=correct, accuracy=accuracy))
            archive_path = self.root / (name + ".zip")
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(name + "/config.json", json.dumps(config))
                archive.writestr(name + "/per_corruption.csv", stream.getvalue())
                for filename in ("command.txt", "environment.txt", "stdout.log", "notes.md", "summary.csv"):
                    archive.writestr(name + "/" + filename, "fixture")
            paths.append(archive_path)
        result = screen_weight_ranking(paths, report)
        self.assertEqual(result["best_observed_rho_candidates"], ["beta2_rho_0.001"])
        self.assertEqual(result["ranking"][0]["weight"], 2.)


if __name__ == "__main__":
    unittest.main()
