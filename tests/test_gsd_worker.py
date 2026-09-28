"""CPU worker integration: real dispatch/counting/artifacts, fake model/GPU ops."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

import gsd_protocol
import run_baseline
from research_artifacts import RunBundle


@contextmanager
def artifact_directory():
    root = (Path(__file__).resolve().parents[1] / "tmp").resolve()
    directory = root / ("gsd-worker-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        if not directory.resolve().is_relative_to(root):
            raise RuntimeError("Worker-test cleanup escaped workspace tmp")
        shutil.rmtree(directory)


class CPUTorch:
    """Expose real CPU tensors while replacing every worker CUDA operation."""
    def __init__(self):
        self.cuda = SimpleNamespace(
            is_available=lambda: True, manual_seed_all=lambda seed: None,
            synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 0, get_device_name=lambda: "CPU test double")
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(
            benchmark=False, deterministic=False, version=lambda: None))
        self.version = SimpleNamespace(cuda=None)

    def manual_seed(self, seed):
        # torch.manual_seed also calls the real CUDA surface; use CPU generator only.
        torch.random.default_generator.manual_seed(seed)

    def __getattr__(self, name):
        return getattr(torch, name)


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def classification_only(self, points, only_unmasked):
        if self.training or self.weight.requires_grad or torch.is_grad_enabled():
            raise AssertionError("Classifier must be frozen and evaluated under no_grad")
        return torch.tensor([[1., 0.]]).repeat(len(points), 1)


class Dataset:
    def __init__(self, root, labels, corruption, severity):
        self.data = np.load(run_baseline.selected_data_path(root, corruption, severity))
        self.labels = np.load(labels)

    def __len__(self):
        return len(self.data)


def batches(dataset, batch_size, shuffle):
    if shuffle:
        raise AssertionError("The dataset order must remain fixed")
    for index in range(0, len(dataset), batch_size):
        yield (torch.from_numpy(dataset.data[index:index+batch_size]),
               torch.from_numpy(dataset.labels[index:index+batch_size]))


class GSDWorkerTests(unittest.TestCase):
    def run_fixture(self, directory, corruption, *, fail_second_batch=False, calibration=False,
                    full_dataset=False, invalid_dataset=False):
        data_root = directory / "data"
        data_root.mkdir()
        selected = run_baseline.CORRUPTIONS if full_dataset else [corruption]
        for name in selected:
            count = 0 if invalid_dataset and name == corruption else 33
            np.save(data_root / ("data_" + name + "_5.npy"), np.ones((count, 8, 3), dtype=np.float32))
        if calibration:
            np.save(data_root / "data_impulse_5.npy", np.ones((33, 8, 3), dtype=np.float32))
        np.save(data_root / "label.npy", np.arange(33, dtype=np.int64) % 2)
        asset_paths = {}
        for name in ("pointmae_ckpt", "pointmae_config", "diff_ckpt", "diff_config"):
            asset_paths[name] = directory / (name + ".fixture")
            asset_paths[name].write_text("CPU fixture " + name, encoding="utf-8")
        cli = ["--method", gsd_protocol.METHOD, "--batch_size", "32", "--seed", "0",
               "--gsd-stage", "smoke", "--max-batches", "2", "--corruptions", corruption,
               "--dataset_root", str(data_root), "--label_path", str(data_root / "label.npy")]
        if calibration:
            cli += ["--method", gsd_protocol.SMOOTH_METHOD, "--gsd-stage", "calibrate",
                    "--gsd-weight", "0", "--gsd-profile", "hard", "--max-batches", "0",
                    "--corruptions", "gaussian", "impulse", "--gsd-development-count", "32"]
        if full_dataset:
            cli = ["--method", gsd_protocol.SMOOTH_METHOD, "--batch_size", "32", "--seed", "0",
                   "--gsd-stage", "full_dataset_development", "--gsd-weight", "0",
                   "--gsd-profile", "hard", "--gsd-calibration-reference", "fixture-calibration.json",
                   "--max-batches", "0", "--corruptions", *run_baseline.CORRUPTIONS,
                   "--dataset_root", str(data_root), "--label_path", str(data_root / "label.npy")]
        for name, path in asset_paths.items():
            cli.extend(["--" + name, str(path)])
        args = run_baseline.parse_arguments(cli)
        provenance = {"path": "fixture", "sha256": "a" * 64, "run_id": "fixture-calibration",
                      "target_rho": None, "development_split": {}, "expected_manifests": {}}
        with patch.object(run_baseline, "command_output", return_value="CPU fixture"), \
                patch("gsd_calibration.calibration_provenance", return_value=provenance):
            config = run_baseline.build_config(args)
        bundle = RunBundle.create(directory / "runs", "worker", config, "CPU worker integration")
        classifier = torch.nn.Module()
        classifier.add_module("module", Classifier())
        lion = SimpleNamespace(vae=torch.nn.Linear(1, 1), priors=torch.nn.ModuleList([torch.nn.Linear(1, 1)]))

        def configure(arguments, checkpoint_observer):
            self.assertTrue(arguments.lion_eval_mode)
            self.assertFalse(arguments.lion_ema_mode)
            classifier.eval()  # Existing configure_model owns classifier eval mode.
            for name in ("pointmae", "lion_vae", "lion_priors"):
                checkpoint_observer(name, SimpleNamespace(missing_keys=[], unexpected_keys=[]))
            return classifier, lion

        host = SimpleNamespace(
            configure_model=configure, diff_config=SimpleNamespace(dump=lambda: "CPU LION fixture"),
            PointDataset=Dataset, DataLoader=batches, tqdm=lambda iterable, **kw: iterable)
        calls = []

        def reconstruct(points, model, steps, gamma, eta, retained, total, **kwargs):
            calls.append((steps, gamma, eta, retained, total))
            for module in (classifier, model.vae, model.priors):
                self.assertFalse(module.training)
                self.assertFalse(any(p.requires_grad for p in module.parameters()))
            if fail_second_batch and len(calls) == 2:
                raise FloatingPointError("synthetic GSD gradient failure")
            kwargs["scheduler_observer"](SimpleNamespace(
                config={"prediction_type": "epsilon"}, timesteps=torch.arange(steps-1, -1, -1)))
            kwargs["diagnostics_observer"]({"kind": "graph", "samples": [{"actual_rank": 3}] * len(points)})
            kwargs["diagnostics_observer"]({"kind": "step", "batch_size": len(points), "spectral_loss": .5})
            if calibration:
                for step in range(5):
                    kwargs["sample_observer"](dict(step_index=step, timestep=40 - step * 10,
                                                   samples=[dict(local_state_rms=.1)] * len(points)))
                    if step in (0, 2, 4):
                        from gsd_calibration import CANDIDATES
                        for profile, beta in CANDIDATES:
                            kwargs["probe_observer"](dict(step_index=step, timestep=40 - step * 10,
                                profile=profile, beta=beta,
                                samples=[dict(local_ratio=.001, style_ratio=.002)] * len(points)))
            return points

        with patch.dict(sys.modules, {"torch": CPUTorch(), "main_3dd_tta": host,
                                     "tta_gsd": SimpleNamespace(tta_gsd_reconstruct=reconstruct)}), \
                patch.object(run_baseline, "command_output", return_value="CPU fixture"), \
                patch.object(run_baseline, "tta_preprocess_points", side_effect=lambda data, *a: (data, None, None)), \
                patch.object(run_baseline, "tta_postprocess_points", side_effect=lambda points, *a: points), \
                patch.object(gsd_protocol, "process_batches", wraps=gsd_protocol.process_batches) as dispatch, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            if invalid_dataset:
                with self.assertRaisesRegex(ValueError, "Empty data or data/label count mismatch"):
                    run_baseline.run_worker(str(bundle.path))
            elif fail_second_batch:
                with self.assertRaisesRegex(FloatingPointError, "synthetic GSD"):
                    run_baseline.run_worker(str(bundle.path))
            else:
                run_baseline.run_worker(str(bundle.path))
            expected_dispatch = 0 if invalid_dataset else (2 if calibration else (len(selected) if full_dataset else 1))
            self.assertEqual(dispatch.call_count, expected_dispatch)
            if not invalid_dataset:
                self.assertEqual(dispatch.call_args.args[-1], 35 if corruption == "background" else 5)
        saved = json.loads((bundle.path / "config.json").read_text(encoding="utf-8"))
        with (bundle.path / "per_corruption.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        with (bundle.path / "summary.csv").open(newline="", encoding="utf-8") as stream:
            summary, = list(csv.DictReader(stream))
        self.assertEqual({path.name for path in bundle.path.iterdir()}, {
            "config.json", "command.txt", "environment.txt", "stdout.log", "notes.md",
            "per_corruption.csv", "summary.csv"})
        self.assertEqual(saved["asset_manifest"]["lion_checkpoint"],
                         run_baseline.file_identity(asset_paths["diff_ckpt"]))
        self.assertEqual(saved["asset_manifest"]["lion_config"],
                         run_baseline.file_identity(asset_paths["diff_config"]))
        for item in saved["module_inventory_before"].values():
            self.assertFalse(item["training"])
            self.assertEqual(item["trainable_parameters"], 0)
        self.assertEqual(saved["lion_mode_policy"], "raw LION eval; EMA disabled")
        self.assertEqual(saved["final_decode_style"], "original shape_latent")
        expected_corruptions = selected if full_dataset else (["gaussian", "impulse"] if calibration else [corruption])
        batches_per_corruption = 1 if calibration else 2
        expected_calls = [] if invalid_dataset else [
            (35 if name == "background" else 5, .01, .01, .95, 100)
            for name in expected_corruptions for _ in range(batches_per_corruption)]
        self.assertEqual(calls, expected_calls)
        return saved, rows, summary

    def test_calibration_worker_records_original_indices_and_partial_coverage(self):
        from gsd_calibration import summarize_calibration, development_indices
        with artifact_directory() as directory:
            saved, rows, summary = self.run_fixture(directory, "gaussian", calibration=True)
        self.assertEqual(saved["execution_status"], "complete")
        self.assertEqual(saved["status"], "partial")
        self.assertEqual([row["n_examples"] for row in rows], ["32", "32"])
        self.assertEqual([row["status"] for row in rows], ["partial", "partial"])
        for corruption in ("gaussian", "impulse"):
            indices = development_indices(33, 32, 20260927)
            self.assertEqual(saved["development_split"][corruption]["indices"], indices)
            self.assertEqual({r["sample_index"] for r in saved["gsd_sample_diagnostics"][corruption]["probe"]}, set(indices))
        report = summarize_calibration(saved)
        self.assertEqual(report["candidates"]["2.0"]["weights"]["0.001"], 1.)
        saved["gsd_sample_diagnostics"]["impulse"]["probe"].pop()
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            summarize_calibration(saved)

    def test_staged_launcher_binds_weights_and_provenance_and_rejects_failed_evidence(self):
        from eval_gsd_calibration import build_commands
        from gsd_calibration import verify_calibration_inputs
        with artifact_directory() as directory:
            saved, _, _ = self.run_fixture(directory, "gaussian", calibration=True)
            reference = directory / "reference.json"
            reference.write_text(json.dumps(saved), encoding="utf-8")
            commands = build_commands("screen-weight", calibration=reference, count=32)
            self.assertEqual(len(commands), 4)
            parsed = [run_baseline.parse_arguments(command[3:]) for command in commands]
            self.assertEqual([args.gsd_weight for args in parsed], [0., .1, 1., 10.])
            self.assertTrue(all("." not in args.run_name for args in parsed))
            with patch.object(run_baseline, "command_output", return_value="CPU fixture"):
                config = run_baseline.build_config(parsed[2])
            self.assertEqual(config["calibration_reference"]["target_rho"], .001)
            config.update(asset_manifest=saved["asset_manifest"], dataset_hash_manifest=saved["dataset_hash_manifest"])
            verify_calibration_inputs(config)
            config["dataset_hash_manifest"] = {"gaussian": {"sha256": "different"}}
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_calibration_inputs(config)
            parsed[2].gsd_weight = 123.
            with self.assertRaisesRegex(ValueError, "does not match"), \
                    patch.object(run_baseline, "command_output", return_value="CPU fixture"):
                run_baseline.build_config(parsed[2])
            with self.assertRaisesRegex(ValueError, "explicit --rho"):
                build_commands("screen-beta", calibration=reference)
            beta_commands = build_commands("screen-beta", calibration=reference, rho=.001)
            self.assertEqual(len(beta_commands), 3)
            for command in beta_commands:
                run_baseline.parse_arguments(command[3:])
            saved["execution_status"] = "failed"
            reference.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed seed-0"):
                build_commands("screen-weight", calibration=reference)

    def test_worker_gsd_dispatch_seals_complete_bundle_for_both_schedules(self):
        for corruption in ("gaussian", "background"):
            with self.subTest(corruption=corruption), artifact_directory() as directory:
                saved, rows, summary = self.run_fixture(directory, corruption)
                self.assertEqual(saved["status"], "complete")
                self.assertEqual(saved["execution_status"], "complete")
                self.assertEqual(saved["completed_corruptions"], [corruption])
                self.assertEqual(saved["observed_batch_sizes"][corruption], [32, 1])
                self.assertEqual(saved["gsd_diagnostics"][corruption]["graph"]["records"], 33)
                self.assertEqual(saved["gsd_diagnostics"][corruption]["step"]["records"], 2)
                self.assertEqual((rows[0]["n_examples"], rows[0]["n_correct"], rows[0]["status"]),
                                 ("33", "17", "complete"))
                self.assertEqual((summary["total_examples"], summary["total_correct"], summary["status"]),
                                 ("33", "17", "complete"))

    def test_worker_gsd_failure_preserves_completed_batch_counters_and_diagnostics(self):
        with artifact_directory() as directory:
            saved, rows, summary = self.run_fixture(directory, "gaussian", fail_second_batch=True)
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["execution_status"], "failed")
            self.assertEqual(saved["failed_corruption"], "gaussian")
            self.assertEqual(saved["completed_corruptions"], [])
            self.assertEqual(saved["observed_batch_sizes"]["gaussian"], [32])
            self.assertEqual(saved["gsd_diagnostics"]["gaussian"]["graph"]["records"], 32)
            self.assertEqual((rows[0]["n_examples"], rows[0]["n_correct"], rows[0]["status"]),
                             ("32", "16", "failed"))
            self.assertEqual((summary["total_examples"], summary["total_correct"], summary["status"]),
                             ("32", "16", "failed"))

    def test_worker_full_dataset_stage_records_all15_full_file_rows_without_sample_traces(self):
        with artifact_directory() as directory:
            saved, rows, summary = self.run_fixture(directory, "gaussian", full_dataset=True)
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["execution_status"], "complete")
        self.assertEqual(saved["completed_corruptions"], list(run_baseline.CORRUPTIONS))
        self.assertEqual(saved["evaluation_scope"], "full ModelNet40-C severity-5 test files; all 15 corruptions")
        self.assertEqual(set(saved["dataset_inventory"]), set(run_baseline.CORRUPTIONS))
        self.assertNotIn("gsd_sample_diagnostics", saved)
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["n_examples"] == "33" and row["status"] == "complete" for row in rows))
        self.assertEqual(summary["total_examples"], "495")
        self.assertTrue(all(saved["observed_batch_sizes"][name] == [32, 1]
                            for name in run_baseline.CORRUPTIONS))

    def test_worker_full_dataset_rejects_empty_or_count_mismatched_corruption_file(self):
        with artifact_directory() as directory:
            saved, rows, _ = self.run_fixture(directory, "uniform", full_dataset=True,
                                               invalid_dataset=True)
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["execution_status"], "failed")
        self.assertEqual(saved["completed_corruptions"], [])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
