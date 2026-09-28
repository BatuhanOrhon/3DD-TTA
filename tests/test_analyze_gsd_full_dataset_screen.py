"""Synthetic CPU contracts for the complete all-corruptions GSD screen analyzer."""
import csv
import io
import json
from pathlib import Path
import shlex
import shutil
import unittest
import uuid
from zipfile import ZipFile

from research_artifacts import CORRUPTIONS, PER_COLUMNS, SUMMARY_COLUMNS, summarize
from scripts.analyze_gsd_full_dataset_screen import analyze_runs, read_full_run


class FullDatasetAnalyzerTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "tmp" / ("analyze-gsd-full-" + uuid.uuid4().hex)

    def setUp(self):
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def _bundle(self, arm, seed, counts=None, row_count_overrides=None):
        from gsd_calibration import PINNED_REFERENCE_CONFIG_SHA256, PINNED_REFERENCE_RUN_ID
        run_name = f"gsd-full-{arm}-seed{seed}"
        name = "20300101-000000_" + run_name
        weights = {"scd": (0., "hard", None, None),
                   "beta05": (8.140161356429882, "smooth", .5, .001),
                   "beta2": (113.10823980075075, "smooth", 2., .01)}
        weight, profile, beta, rho = weights[arm]
        count_by_name = counts or {corruption: 2 for corruption in CORRUPTIONS}
        config = {
            "run_id": name, "stage": "full_dataset_development", "execution_status": "complete",
            "status": "complete", "completed_corruptions": list(CORRUPTIONS),
            "git_branch": "gsd-smooth-spectrum", "git_commit": "fixture-commit",
            "dataset": "modelnet40_c", "severity": 5, "method": "gsd_latent_spectral_smooth_v2",
            "seed": seed, "batch_size": 32, "corruptions": list(CORRUPTIONS),
            "dataset_inventory": {c: {"total_examples": count_by_name[c]} for c in CORRUPTIONS},
            "asset_manifest": {"checkpoint": {"sha256": "a" * 64}},
            "dataset_hash_manifest": {c: {"sha256": (str(i % 10) * 64)} for i, c in enumerate(CORRUPTIONS)},
            "runtime_source_manifest": {"source.py": {"sha256": "b" * 64}},
            "extension_inventory": {"op": {"loaded": True}},
            "calibration_source_compatibility": {
                "allowed_source_extensions": ["gsd_calibration.py", "gsd_protocol.py", "run_baseline.py"]},
            "randomness": {"seed": seed, "torch_version": "2.1", "numpy_version": "1.26", "gpu": "fixture"},
            "observed_batch_sizes": {c: [count_by_name[c]] for c in CORRUPTIONS},
            "scheduler_timesteps": {c: list(range(990, -1, -10)) for c in CORRUPTIONS},
            "calibration_reference": {"run_id": PINNED_REFERENCE_RUN_ID,
                                      "sha256": PINNED_REFERENCE_CONFIG_SHA256,
                                      "expected_manifests": {
                                          "asset_manifest": {"checkpoint": {"sha256": "a" * 64}},
                                          "dataset_hash_manifest": {
                                              "gaussian": {"sha256": "1" * 64},
                                              "impulse": {"sha256": "3" * 64}},
                                          "runtime_source_manifest": {"source.py": {"sha256": "b" * 64}}}},
            "spectral": {"weight": weight, "scd_weight": 1., "profile": profile, "beta": beta,
                         "k": 10, "delta": .1, "graph_gamma": .6, "requested_modes": 100},
            "cli_args": {"gsd_stage": "full_dataset_development", "gsd_weight": weight,
                         "gsd_scd_weight": 1., "gsd_profile": profile, "gsd_beta": beta,
                         "gsd_target_rho": rho, "gsd_k": 10, "gsd_delta": .1,
                         "gsd_graph_gamma": .6, "gsd_modes": 100, "batch_size": 32, "seed": seed,
                         "dataset_name": "modelnet-c", "severity": 5, "corruptions": list(CORRUPTIONS),
                         "max_batches": 0, "gsd_calibration_reference": "config.json",
                         "run_name": run_name,
                         "lion_eval_mode": True, "lion_ema_mode": False, "lambdaa": .95,
                         "gamma": .01, "eta": .01}}
        config["randomness"].update(data_order="file order; shuffle=False", num_workers=0, drop_last=False)
        rows = []
        for i, corruption in enumerate(CORRUPTIONS):
            n = (row_count_overrides or {}).get(corruption, count_by_name[corruption])
            correct = (i + seed) % (n + 1)
            rows.append({"run_id": name, "dataset": "modelnet40_c", "severity": 5,
                         "method": config["method"], "seed": seed, "corruption": corruption,
                         "n_examples": n, "n_correct": correct, "accuracy": correct / n,
                         "runtime_seconds": 1. + i, "peak_gpu_memory_mb": 10. + i, "status": "complete"})
        summary = summarize(rows)
        config_text = json.dumps(config)
        def csv_text(fields, data):
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
            return out.getvalue()
        command = shlex.join(["python", "run_baseline.py", "--run-name", run_name, "--seed", str(seed),
                              "--gsd-stage", "full_dataset_development", "--max-batches", "0"])
        environment = ("Python: Python 3.8 fixture\nPlatform: Linux fixture\n"
                       "Git branch: gsd-smooth-spectrum\nGit commit: fixture-commit\n\n"
                       "pip freeze:\nnumpy==1.26\ntorch==2.1\n\nAsset hashes:\nfixture\n")
        files = {"config.json": config_text, "per_corruption.csv": csv_text(PER_COLUMNS, rows),
                 "summary.csv": csv_text(SUMMARY_COLUMNS, [summary]), "command.txt": "cwd: fixture\n" + command,
                 "environment.txt": environment, "stdout.log": "Run directory: fixture\n", "notes.md": "notes\n"}
        path = self.root / (name + ".zip")
        with ZipFile(path, "w") as archive:
            for filename, data in files.items():
                archive.writestr(name + "/" + filename, data)
        return path

    def _nine(self):
        return [self._bundle(arm, seed) for arm in ("scd", "beta05", "beta2") for seed in range(3)]

    @staticmethod
    def _rewrite(path, member, transform):
        with ZipFile(path) as archive:
            root = Path(archive.namelist()[0]).parts[0]
            contents = {Path(name).name: archive.read(name) for name in archive.namelist()}
        contents[member] = transform(contents[member])
        replacement = path.with_suffix(".replacement.zip")
        with ZipFile(replacement, "w") as archive:
            for filename, data in contents.items():
                archive.writestr(root + "/" + filename, data)
        replacement.replace(path)

    @staticmethod
    def _calibration():
        from gsd_calibration import PINNED_REFERENCE_CONFIG_SHA256, PINNED_REFERENCE_RUN_ID
        return {"run_id": PINNED_REFERENCE_RUN_ID, "sha256": PINNED_REFERENCE_CONFIG_SHA256,
                "expected_manifests": {
                    "asset_manifest": {"checkpoint": {"sha256": "a" * 64}},
                    "dataset_hash_manifest": {"gaussian": {"sha256": "1" * 64},
                                              "impulse": {"sha256": "3" * 64}},
                    "runtime_source_manifest": {"source.py": {"sha256": "b" * 64}}},
                "report": {"candidates": {"0.5": {"weights": {"0.001": 8.140161356429882}},
                                             "2.0": {"weights": {"0.01": 113.10823980075075}}}}}

    def test_analyzes_exact_nine_runs_with_equal_corruption_macro_and_seed_deltas(self):
        result = analyze_runs(self._nine(), self._calibration())
        self.assertEqual(len(result["runs"]), 9)
        self.assertEqual(result["protocol"]["n_corruptions"], 15)
        self.assertEqual(result["input_identity"]["full_examples_by_corruption"][CORRUPTIONS[0]], 2)
        self.assertIn("packages", result["environment"]["python_platform_packages"])
        self.assertEqual(result["arms"]["scd_only"]["seeds"]["0"]["macro_accuracy"],
                         sum((i % 3) / 2 for i in range(15)) / 15)
        self.assertIn("mean_pp", result["candidate_deltas"]["beta_0.5_rho_0.001"])
        self.assertIn("sample_sd_pp", result["candidate_deltas"]["beta_2_rho_0.01"])
        self.assertTrue(result["evidence_status"].startswith("full-test-set development"))

    def test_rejects_incomplete_counts_duplicate_pairs_and_archive_path_traversal(self):
        paths = self._nine()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            analyze_runs(paths[:-1], self._calibration())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_runs(paths[:-1] + [paths[0]], self._calibration())
        broken = self._bundle("scd", 0, row_count_overrides={CORRUPTIONS[0]: 1})
        with self.assertRaisesRegex(ValueError, "full file/config"):
            read_full_run(broken)
        unsafe = self._bundle("beta05", 2)
        with ZipFile(unsafe) as original:
            root = Path(original.namelist()[0]).parts[0]
            contents = {Path(name).name: original.read(name) for name in original.namelist()}
        replacement = unsafe.with_suffix(".unsafe.zip")
        with ZipFile(replacement, "w") as archive:
            for filename, data in contents.items():
                member = "../outside/command.txt" if filename == "command.txt" else root + "/" + filename
                archive.writestr(member, data)
        replacement.replace(unsafe)
        with self.assertRaisesRegex(ValueError, "unsafe archive member"):
            read_full_run(unsafe)

    def test_rejects_stale_calibration_source_and_command_or_corrupt_zip(self):
        paths = self._nine()
        first = paths[0]
        self._rewrite(first, "config.json", lambda raw: json.dumps(
            dict(json.loads(raw), runtime_source_manifest={"source.py": {"sha256": "c" * 64}})).encode())
        with self.assertRaisesRegex(ValueError, "calibration source identity"):
            read_full_run(first)
        paths = self._nine()
        first = paths[0]
        self._rewrite(first, "command.txt", lambda raw: raw.replace(b"--seed 0", b"--seed 1"))
        with self.assertRaisesRegex(ValueError, "command/config mismatch"):
            read_full_run(first)
        paths = self._nine()
        first = paths[0]
        first.write_bytes(first.read_bytes()[:24])
        with self.assertRaisesRegex(ValueError, "invalid ZIP"):
            read_full_run(first)

    def test_rejects_calibration_coefficient_drift(self):
        payload = self._calibration()
        payload["report"]["candidates"]["0.5"]["weights"]["0.001"] += 1
        with self.assertRaisesRegex(ValueError, "locked coefficient"):
            analyze_runs(self._nine(), payload)


if __name__ == "__main__":
    unittest.main()
