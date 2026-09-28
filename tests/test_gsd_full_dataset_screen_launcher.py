"""Fixed-matrix and safe-resume contracts for the Colab full-suite launcher."""
import unittest
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

from research_artifacts import CORRUPTIONS
from scripts.run_gsd_full_dataset_screen import build_screen_commands, run_screen
from run_baseline import REPO


class FullDatasetLauncherTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "tmp" / ("gsd-full-launcher-" + uuid.uuid4().hex)

    def setUp(self):
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    @staticmethod
    def _calibration():
        return {"run_id": "20260928-113047_gsd-cal-diagnose-reference-seed0-n64",
                "sha256": "550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6",
                "report": {"candidates": {"0.5": {"weights": {"0.001": 8.140161356429882}},
                                             "2.0": {"weights": {"0.01": 113.10823980075075}}}}}

    def test_builds_exact_three_arms_by_three_seeds(self):
        with patch("scripts.run_gsd_full_dataset_screen.load_pinned_calibration",
                   return_value=self._calibration()):
            commands = build_screen_commands(Path("calibration.json"), "./result")
        self.assertEqual(len(commands), 9)
        parsed = []
        for command in commands:
            parsed.append({key: command[command.index(flag) + 1] for key, flag in (
                ("seed", "--seed"), ("stage", "--gsd-stage"), ("weight", "--gsd-weight"),
                ("profile", "--gsd-profile"), ("run_name", "--run-name"))})
            self.assertEqual(command[command.index("--corruptions") + 1:
                                     command.index("--gsd-stage")], list(CORRUPTIONS))
            self.assertEqual(command[command.index("--max-batches") + 1], "0")
            self.assertIn("--gsd-calibration-reference", command)
        self.assertEqual({(row["seed"], row["weight"]) for row in parsed},
                         {(str(seed), weight) for seed in range(3) for weight in
                          ("0.0", "8.140161356429882", "113.10823980075075")})
        self.assertTrue(all(row["stage"] == "full_dataset_development" for row in parsed))

    def test_rejects_bad_calibration_before_building_commands(self):
        with patch("scripts.run_gsd_full_dataset_screen.load_pinned_calibration",
                   side_effect=ValueError("calibration hash mismatch")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_screen_commands(Path("bad.json"), "./result")
        altered = self._calibration()
        altered["report"]["candidates"]["2.0"]["weights"]["0.01"] += 1
        with patch("scripts.run_gsd_full_dataset_screen.load_pinned_calibration", return_value=altered):
            with self.assertRaisesRegex(ValueError, "coefficient changed"):
                build_screen_commands(Path("altered.json"), "./result")

    def test_resume_skips_only_complete_matching_archives(self):
        result_root = self.root / "result"
        method_root = result_root / "modelnet40_c" / "gsd_latent_spectral_smooth_v2"
        method_root.mkdir(parents=True)
        with patch("scripts.run_gsd_full_dataset_screen.load_pinned_calibration",
                   return_value=self._calibration()), \
                patch("scripts.run_gsd_full_dataset_screen.analyze_runs", return_value={"valid": True}), \
                patch("scripts.run_gsd_full_dataset_screen.run_streamed",
                      side_effect=AssertionError("validated completions should be skipped")), \
                patch("scripts.run_gsd_full_dataset_screen.read_full_run") as reader:
            commands = build_screen_commands(Path("calibration.json"), str(result_root))
            expected = []
            def read_matching(path):
                run_name = Path(path).stem.split("_", 1)[1]
                arm = "scd_only" if "-scd-" in run_name else (
                    "beta_0.5_rho_0.001" if "beta0p5" in run_name else "beta_2_rho_0.01")
                seed = int(run_name.rsplit("seed", 1)[1])
                return {"arm": arm, "seed": seed, "config": {"cli_args": {"run_name": run_name}}}
            reader.side_effect = read_matching
            for command in commands:
                run_name = command[command.index("--run-name") + 1]
                archive = method_root / ("20300101-000000_" + run_name + ".zip")
                archive.write_bytes(b"test placeholder")
                expected.append(archive)
            output = run_screen(Path("calibration.json"), str(result_root), resume=True,
                                summary_output=self.root / "summary.json")
        self.assertEqual(output, (self.root / "summary.json").resolve())
        self.assertEqual(len(expected), 9)

    def test_first_child_failure_stops_the_matrix(self):
        result_root = self.root / "result"
        with patch("scripts.run_gsd_full_dataset_screen.load_pinned_calibration",
                   return_value=self._calibration()), \
                patch("scripts.run_gsd_full_dataset_screen.run_streamed", return_value=(1, "failed-run", None)) as run:
            with self.assertRaisesRegex(RuntimeError, "command failed"):
                run_screen(Path("calibration.json"), str(result_root))
        self.assertEqual(run.call_count, 1)

    def test_corrupt_resume_archive_is_preserved_and_retry_gets_unique_attempt_name(self):
        result_root = self.root / "result"
        method_root = result_root / "modelnet40_c" / "gsd_latent_spectral_smooth_v2"
        method_root.mkdir(parents=True)
        base_name = "gsd-full-screen-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-scd-seed0"
        stale = method_root / ("20300101-000000_" + base_name + ".zip")
        stale.write_bytes(b"corrupt; preserve")

        def fake_read(path):
            run_name = Path(path).stem.split("_", 1)[1]
            if run_name == base_name:
                raise ValueError("invalid ZIP")
            arm = "scd_only" if "-scd-" in run_name else (
                "beta_0.5_rho_0.001" if "beta0p5" in run_name else "beta_2_rho_0.01")
            return {"arm": arm, "seed": int(run_name.rsplit("seed", 1)[1].split("-", 1)[0]),
                    "config": {"cli_args": {"run_name": run_name}}}

        launched_names = []
        def fake_run(command):
            run_name = command[command.index("--run-name") + 1]
            launched_names.append(run_name)
            archive = method_root / ("20300101-000000_" + run_name + ".zip")
            archive.write_bytes(b"synthetic complete archive")
            return 0, "run-dir", str(archive)

        with patch("scripts.run_gsd_full_dataset_screen.load_pinned_calibration",
                   return_value=self._calibration()), \
                patch("scripts.run_gsd_full_dataset_screen.read_full_run", side_effect=fake_read), \
                patch("scripts.run_gsd_full_dataset_screen.run_streamed", side_effect=fake_run), \
                patch("scripts.run_gsd_full_dataset_screen.analyze_runs", return_value={"ok": True}):
            run_screen(Path("calibration.json"), str(result_root), resume=True,
                       summary_output=self.root / "summary.json")
        self.assertEqual(stale.read_bytes(), b"corrupt; preserve")
        self.assertEqual(launched_names[0], base_name + "-attempt02")
        self.assertEqual(len(launched_names), 9)


if __name__ == "__main__":
    unittest.main()
