"""Compact JSON and development-beta selection checks."""
import csv
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid
from zipfile import ZipFile

from gsd_calibration import development_indices
from scripts.analyze_gsd_calibration import (
    calibration_summary,
    interaction_ranking,
    screen_ranking,
    screen_weight_ranking,
)


PILOT_CORRUPTIONS = ("gaussian", "impulse")
MANIFESTS = {
    "asset_manifest": {"asset": {"path": "/content/asset", "bytes": 1, "sha256": "a" * 64}},
    "dataset_hash_manifest": {"gaussian": {"path": "/content/gaussian", "bytes": 1, "sha256": "b" * 64},
                              "impulse": {"path": "/content/impulse", "bytes": 1, "sha256": "c" * 64}},
    "runtime_source_manifest": {"run_baseline.py": {"path": "/content/run_baseline.py", "bytes": 1,
                                                       "sha256": "d" * 64}},
}


class AnalyzeCalibrationTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "tmp" / ("analyze-cal-" + uuid.uuid4().hex)

    def setUp(self):
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    @staticmethod
    def _config(name, weight, profile, beta, rho, count=32):
        indices = development_indices(64, count, 20260927)
        split = {corruption: {"indices": indices, "total_examples": 64, "split_seed": 20260927,
                              "purpose": "exploratory target development; not held-out confirmation"}
                 for corruption in PILOT_CORRUPTIONS}
        cli = {
            "batch_size": 32, "corruptions": list(PILOT_CORRUPTIONS), "dataset_name": "modelnet-c",
            "eta": .01, "gamma": .01, "lambdaa": .95, "lion_eval_mode": True,
            "lion_ema_mode": False, "max_batches": 0, "method": "gsd_latent_spectral_smooth_v2",
            "seed": 0, "severity": 5, "gsd_stage": "development", "gsd_weight": weight,
            "gsd_scd_weight": 1., "gsd_profile": profile, "gsd_beta": beta,
            "gsd_development_count": count, "gsd_split_seed": 20260927,
            "gsd_k": 10, "gsd_delta": .1, "gsd_graph_gamma": .6, "gsd_modes": 100,
            "gsd_target_rho": rho, "run_name": name,
        }
        config = {
            "stage": "development", "execution_status": "complete", "status": "partial", "run_id": name,
            "completed_corruptions": list(PILOT_CORRUPTIONS), "seed": 0, "batch_size": 32,
            "dataset": "modelnet40_c", "severity": 5, "method": cli["method"],
            "corruptions": list(PILOT_CORRUPTIONS), "num_classes": 40, "num_input_points": 2048,
            "num_classifier_points": 1024, "scale_factor": 3.3885, "ddim_total_steps": 100,
            "normal_reverse_steps": 5, "background_reverse_steps": 35, "gamma": .01, "eta": .01,
            "lambda_cd": .95, "guidance_mapping": {"gamma": "local latent", "eta": "style condition"},
            "final_decode_style": "original shape_latent", "lion_mode_policy": "raw LION eval; EMA disabled",
            "lion_loaded": True, "scd_normalization": {"enabled": False, "reduction": "legacy sum"},
            "scheduler_class": "DDIMScheduler",
            "scheduler_config": {"beta_start": .0001, "beta_end": .02, "beta_schedule": "linear",
                                 "num_train_timesteps": 1000, "clip_sample": False,
                                 "set_alpha_to_one": True, "steps_offset": 0, "prediction_type": "epsilon",
                                 "trained_betas": None},
            "scheduler_timesteps": {corruption: list(range(990, -1, -10)) for corruption in PILOT_CORRUPTIONS},
            "observed_batch_sizes": {corruption: [32] for corruption in PILOT_CORRUPTIONS},
            "randomness": {"seed": 0, "num_workers": 0, "drop_last": False,
                           "data_order": "fixed shuffled development indices; loader shuffle=False"},
            "development_split": split, "cli_args": cli,
            "spectral": {"weight": weight, "scd_weight": 1., "profile": profile, "beta": beta,
                         "k": 10, "delta": .1, "graph_gamma": .6, "hard_requested_modes": 100 if beta is None else None},
            "calibration_reference": {"sha256": "e" * 64, "run_id": "calibration-run", "target_rho": rho,
                                      "development_split": split, "expected_manifests": MANIFESTS},
        }
        config.update(MANIFESTS)
        return config

    def _write_bundle(self, name, weight, profile, beta, rho, correct, count=32, as_zip=False):
        config = self._config(name, weight, profile, beta, rho, count)
        rows = []
        for corruption in PILOT_CORRUPTIONS:
            rows.append({"run_id": name, "dataset": "modelnet40_c", "severity": 5,
                         "method": "gsd_latent_spectral_smooth_v2", "seed": 0, "corruption": corruption,
                         "n_examples": count, "n_correct": correct, "accuracy": correct / count,
                         "runtime_seconds": 1., "peak_gpu_memory_mb": 2., "status": "partial"})
        per_corruption = io.StringIO()
        fieldnames = ("run_id", "dataset", "severity", "method", "seed", "corruption", "n_examples",
                      "n_correct", "accuracy", "runtime_seconds", "peak_gpu_memory_mb", "status")
        writer = csv.DictWriter(per_corruption, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        summary = io.StringIO()
        summary_fields = ("run_id", "dataset", "severity", "method", "seed", "n_corruptions",
                          "macro_accuracy", "total_examples", "total_correct", "micro_accuracy",
                          "total_runtime_seconds", "status")
        writer = csv.DictWriter(summary, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({"run_id": name, "dataset": "modelnet40_c", "severity": 5,
                         "method": "gsd_latent_spectral_smooth_v2", "seed": 0, "n_corruptions": 2,
                         "macro_accuracy": correct / count, "total_examples": 2 * count,
                         "total_correct": 2 * correct, "micro_accuracy": correct / count,
                         "total_runtime_seconds": 2., "status": "partial"})
        files = {"config.json": json.dumps(config), "per_corruption.csv": per_corruption.getvalue(),
                 "summary.csv": summary.getvalue(), "command.txt": "command", "environment.txt": "environment",
                 "stdout.log": "stdout", "notes.md": "notes"}
        if as_zip:
            path = self.root / (name + ".zip")
            with ZipFile(path, "w") as archive:
                for filename, contents in files.items():
                    archive.writestr(name + "/" + filename, contents)
            return path
        path = self.root / name
        path.mkdir()
        for filename, contents in files.items():
            (path / filename).write_text(contents, encoding="utf-8")
        return path

    def test_phase_report_reduces_to_beta_weights_and_state_scales(self):
        metric = dict(count=192, missing=0, median=.001, p90=.002, min=0., max=.003)
        candidate = dict(local_ratio=metric, style_ratio=metric,
                         local_cosine=metric, weights={"0.0001": .1, "0.001": 1., "0.01": 10.})
        beta_keys = {"hard": candidate, "0.5": candidate, "2.0": candidate, "8.0": candidate}
        per_corruption = {name: beta_keys for name in PILOT_CORRUPTIONS}
        state_metrics = {"local_state_rms": metric, "local_scd_update_rms": metric,
                         "local_scd_update_state_ratio": metric, "local_ddim_displacement_norm": metric,
                         "local_scd_ddim_ratio": metric}
        report = dict(run_id="calibration", candidates=beta_keys, by_corruption=per_corruption,
                      state_by_step={name: {str(step): state_metrics for step in range(5)} for name in per_corruption})
        source = self.root / "report.json"
        source.write_text(json.dumps(report), encoding="utf-8")
        summary = calibration_summary(source)
        self.assertEqual(summary["examples_per_corruption"], 64)
        self.assertEqual([row["beta"] for row in summary["smooth_candidates"]], [.5, 2., 8.])
        self.assertEqual(summary["smooth_candidates"][1]["alpha_by_rho"]["0.001"], 1.)
        self.assertNotIn("state_by_step", summary)

    def test_screen_ranking_rejects_reviewed_protocol_and_artifact_mismatches(self):
        candidates = [("scd", 0., "hard", None, None, 16), ("beta2", 1., "smooth", 2., .001, 18),
                      ("beta05", 1., "smooth", .5, .001, 22), ("beta8", 1., "smooth", 8., .001, 20),
                      ("hard", 1., "hard", None, .001, 21)]
        paths = [self._write_bundle(*candidate) for candidate in candidates]
        weights = {"hard": {"0.001": 1.}, "beta_0.5": {"0.001": 1.},
                   "beta_2.0": {"0.001": 1.}, "beta_8.0": {"0.001": 1.}}
        result = screen_ranking(paths, "calibration-run", weights)
        self.assertEqual(result["best_observed_candidates"], ["beta_0.5"])
        for field, value, message in (("gsd_scd_weight", .5, "SCD weight"), ("gsd_modes", 400, "graph modes")):
            config_path = paths[1] / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["cli_args"][field] = value
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                screen_ranking(paths, "calibration-run", weights)
            config["cli_args"][field] = 1. if field == "gsd_scd_weight" else 100
            config_path.write_text(json.dumps(config), encoding="utf-8")
        config_path = paths[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        del config["runtime_source_manifest"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "runtime_source_manifest"):
            screen_ranking(paths, "calibration-run", weights)

    def test_screen_ranking_rejects_scheduler_host_mismatch(self):
        candidates = [("scd", 0., "hard", None, None, 16), ("beta2", 1., "smooth", 2., .001, 18),
                      ("beta05", 1., "smooth", .5, .001, 22), ("beta8", 1., "smooth", 8., .001, 20),
                      ("hard", 1., "hard", None, .001, 21)]
        paths = [self._write_bundle(*candidate) for candidate in candidates]
        weights = {"hard": {"0.001": 1.}, "beta_0.5": {"0.001": 1.},
                   "beta_2.0": {"0.001": 1.}, "beta_8.0": {"0.001": 1.}}
        config_path = paths[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["scheduler_class"] = "DifferentScheduler"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "scheduler"):
            screen_ranking(paths, "calibration-run", weights)

    def test_screen_ranking_rejects_count_and_summary_disagreements(self):
        candidates = [("scd", 0., "hard", None, None, 16), ("beta2", 1., "smooth", 2., .001, 18),
                      ("beta05", 1., "smooth", .5, .001, 22), ("beta8", 1., "smooth", 8., .001, 20),
                      ("hard", 1., "hard", None, .001, 21)]
        paths = [self._write_bundle(*candidate) for candidate in candidates]
        weights = {"hard": {"0.001": 1.}, "beta_0.5": {"0.001": 1.},
                   "beta_2.0": {"0.001": 1.}, "beta_8.0": {"0.001": 1.}}
        config_path = paths[1] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["cli_args"]["gsd_development_count"] = 16
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "declared count"):
            screen_ranking(paths, "calibration-run", weights)
        config["cli_args"]["gsd_development_count"] = 32
        config_path.write_text(json.dumps(config), encoding="utf-8")
        summary_path = paths[1] / "summary.csv"
        summary_path.write_text(summary_path.read_text(encoding="utf-8").replace(",64,36,", ",64,35,"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "summary CSV"):
            screen_ranking(paths, "calibration-run", weights)

    def test_weight_screen_and_interaction_report_match_validated_cells(self):
        baseline = self._write_bundle("baseline", 0., "hard", None, None, 16, as_zip=True)
        beta2_low = self._write_bundle("beta2-low", 1., "smooth", 2., .0001, 17, as_zip=True)
        beta2_mid = self._write_bundle("beta2-mid", 2., "smooth", 2., .001, 19, as_zip=True)
        beta2_high = self._write_bundle("beta2-high", 3., "smooth", 2., .01, 18, as_zip=True)
        report = {"run_id": "calibration-run", "candidates": {"2.0": {"weights": {"0.0001": 1., "0.001": 2., "0.01": 3.}}}}
        result = screen_weight_ranking([baseline, beta2_low, beta2_mid, beta2_high], report)
        self.assertEqual(result["best_observed_rho_candidates"], ["beta2_rho_0.001"])
        beta05_mid = self._write_bundle("beta05-mid", 4., "smooth", .5, .001, 18)
        beta05_high = self._write_bundle("beta05-high", 5., "smooth", .5, .01, 20)
        weights = {"beta_0.5": {"0.001": 4., "0.01": 5.}, "beta_2.0": {"0.001": 2., "0.01": 3.}}
        interaction = interaction_ranking([baseline, beta05_mid, beta2_mid, beta2_high, beta05_high],
                                          "calibration-run", weights)
        self.assertEqual(interaction["beta_rho_cells"]["beta_0.5"]["0.01"]["total_correct"], 40)
        self.assertEqual(interaction["scd_only"]["total_correct"], 32)

    def test_weight_screen_rejects_non_hard_scd_only_placeholder(self):
        baseline = self._write_bundle("baseline", 0., "hard", None, None, 16)
        beta2_low = self._write_bundle("beta2-low", 1., "smooth", 2., .0001, 17)
        beta2_mid = self._write_bundle("beta2-mid", 2., "smooth", 2., .001, 19)
        beta2_high = self._write_bundle("beta2-high", 3., "smooth", 2., .01, 18)
        config_path = baseline / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["cli_args"].update(gsd_profile="smooth", gsd_beta=.5)
        config["spectral"].update(profile="smooth", beta=.5, hard_requested_modes=None)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report = {"run_id": "calibration-run", "candidates": {"2.0": {"weights": {"0.0001": 1., "0.001": 2., "0.01": 3.}}}}
        with self.assertRaisesRegex(ValueError, "invalid spectral settings"):
            screen_weight_ranking([baseline, beta2_low, beta2_mid, beta2_high], report)


if __name__ == "__main__":
    unittest.main()
