"""CPU contracts: GSD is opt-in and cannot silently change baseline factors."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import shutil
import types
import unittest
import uuid
from unittest.mock import patch

import gsd_protocol
import run_baseline
from research_artifacts import RunBundle, corruption_row


METHOD = "gsd_latent_spectral_v1"


@contextmanager
def artifact_directory():
    # Plain mkdir avoids Windows restricted-temp ACLs set by mkdtemp(mode=0700).
    root = (Path(__file__).resolve().parents[1] / "tmp").resolve()
    directory = root / ("gsd-protocol-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        if not directory.resolve().is_relative_to(root):
            raise RuntimeError("Test artifact cleanup escaped workspace tmp")
        shutil.rmtree(directory)


class GSDProtocolTests(unittest.TestCase):
    def args(self, *extra):
        return run_baseline.parse_arguments([
            "--method", METHOD, "--batch_size", "32", "--max-batches", "0",
            "--corruptions", "gaussian", "impulse", *extra])

    def test_opt_in_pilot_enforces_operational_contract_and_metadata(self):
        args = self.args()
        self.assertTrue(args.lion_eval_mode)
        self.assertFalse(args.lion_ema_mode)
        config = run_baseline.build_config(args)
        self.assertEqual(config["stage"], "pilot")
        self.assertEqual(config["spectral"]["weight"], 1.0)
        self.assertEqual(config["spectral"]["reduction"], "sum_samples_mean_modes_xyz")
        self.assertEqual(config["final_decode_style"], "original shape_latent")
        self.assertEqual(config["projection"], {})
        self.assertIn("tta_gsd.py", config["runtime_source_manifest"])
        self.assertIn("graph_spectral.py", config["runtime_source_manifest"])
        json.dumps(config, allow_nan=False)
        self.assertIn("GSD-inspired latent spectral guidance", run_baseline.notes_for_run(args))

    def test_original_defaults_and_no_spectral_side_effect(self):
        args = run_baseline.parse_arguments([])
        self.assertEqual(args.method, "3dd_original")
        self.assertFalse(args.lion_eval_mode)
        self.assertEqual(args.batch_size, 40)
        self.assertEqual(run_baseline.build_config(args)["spectral"], {})

    def test_protocol_rejects_changes_to_controlled_factors(self):
        for extra in (("--lion-ema-mode",), ("--batch_size", "16"),
                      ("--gamma", ".02"), ("--eta", ".02"),
                      ("--lambdaa", ".96"), ("--severity", "4"),
                      ("--dataset-name", "scanobjectnn-c"), ("--seed", "3"),
                      ("--gsd-weight", "nan"), ("--gsd-weight", "-1"),
                      ("--gsd-modes", "0"), ("--gsd-k", "2048"),
                      ("--gsd-delta", "0"), ("--gsd-graph-gamma", "inf"),
                      ("--corruptions", "background")):
            with self.subTest(extra=extra), redirect_stderr(StringIO()):
                with self.assertRaises((SystemExit, ValueError)):
                    self.args(*extra)

    def test_spectral_options_rejected_for_other_methods(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run_baseline.parse_arguments(["--method", "source_only", "--gsd-weight", "1"])

    def test_zero_weight_and_smoke_are_explicit(self):
        args = self.args("--gsd-weight", "0", "--gsd-stage", "smoke",
                         "--max-batches", "1", "--corruptions", "gaussian")
        config = run_baseline.build_config(args)
        self.assertEqual(config["stage"], "smoke")
        self.assertEqual(config["spectral"]["weight"], 0)
        self.assertIn("baseline", config["spectral"]["zero_weight_behavior"])

    def test_benchmark_requires_complete_canonical_all15(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.args("--gsd-stage", "benchmark")
        args = self.args("--gsd-stage", "benchmark", "--corruptions", *run_baseline.CORRUPTIONS)
        self.assertEqual(run_baseline.build_config(args)["stage"], "benchmark")

    def test_benchmark_rejects_undeclared_retuning(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.args("--gsd-stage", "benchmark", "--gsd-weight", "3",
                      "--corruptions", *run_baseline.CORRUPTIONS)

    def test_background_excluded_confirmation_scope_is_explicit(self):
        corruptions = [name for name in run_baseline.CORRUPTIONS if name != "background"]
        args = self.args("--gsd-stage", "benchmark_no_background",
                         "--corruptions", *corruptions)
        self.assertEqual(run_baseline.build_config(args)["stage"], "benchmark_no_background")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.args("--gsd-stage", "benchmark_no_background",
                      "--corruptions", *run_baseline.CORRUPTIONS)

    def test_diagnostics_aggregate_is_bounded_and_json_serializable(self):
        config = {}
        for _ in range(4):
            gsd_protocol.merge_diagnostics(config, "gaussian", {
                "kind": "graph", "samples": [{"actual_rank": 2, "eigengap": None},
                                              {"actual_rank": 4, "eigengap": .2}]})
        graph = config["gsd_diagnostics"]["gaussian"]["graph"]
        self.assertEqual(graph["records"], 8)
        self.assertEqual(graph["scalars"]["actual_rank"]["mean"], 3)
        self.assertEqual(graph["scalars"]["eigengap"]["missing"], 4)
        json.dumps(config, allow_nan=False)
        with self.assertRaises(ValueError):
            gsd_protocol.merge_diagnostics(config, "gaussian", {"kind": "step", "loss": float("nan")})

    def test_missing_checkpoint_seals_failed_seven_file_bundle_without_accuracy(self):
        config = run_baseline.build_config(self.args())
        config["cli_args"]["pointmae_ckpt"] = "missing-gsd-test-checkpoint-20260922.pth"
        with artifact_directory() as directory:
            bundle = RunBundle.create(Path(directory), "gsd-failure-test", config, "unit test")
            with patch.object(run_baseline, "command_output", return_value="CPU protocol test"), \
                    redirect_stderr(StringIO()), redirect_stdout(StringIO()), self.assertRaises(FileNotFoundError):
                run_baseline.run_worker(str(bundle.path))
            saved = json.loads((bundle.path / "config.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["completed_corruptions"], [])
            self.assertEqual(len(list(bundle.path.iterdir())), 7)
            self.assertIn("failed", (bundle.path / "summary.csv").read_text())
            with self.assertRaises(FileExistsError):
                RunBundle.create(Path(directory), "gsd-failure-test", config, "unit test",
                                 timestamp=bundle.path.name[:15])

    def test_successful_artifact_preserves_method_counts_and_diagnostics(self):
        config = run_baseline.build_config(self.args())
        gsd_protocol.merge_diagnostics(config, "gaussian", {"kind": "step", "spectral_loss": 2.0})
        with artifact_directory() as directory:
            bundle = RunBundle.create(Path(directory), "gsd-counts-test", config, "unit test")
            row = corruption_row(bundle.path.name, 0, "gaussian", 10, 7, 2.5, 0,
                                 "complete", method=METHOD)
            bundle.write_results([row], "complete")
            saved = json.loads((bundle.path / "config.json").read_text())
            self.assertEqual(saved["method"], METHOD)
            self.assertEqual(saved["gsd_diagnostics"]["gaussian"]["step"]["records"], 1)
            self.assertIn("0.7", (bundle.path / "summary.csv").read_text())

    def test_batch_adapter_preserves_counting_postprocess_and_frozen_classifier(self):
        import torch
        calls, observed = [], []
        args = self.args()
        def reconstruct(points, lion, steps, gamma, eta, retained, total, **kw):
            calls.append((steps, gamma, eta, retained, total, kw["spectral_weight"]))
            kw["diagnostics_observer"]({"kind": "step", "spectral_loss": 1.0})
            return points + 1
        def classify(points, only_unmasked):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(only_unmasked)
            # Postprocess sentinel changes the class decision.
            return torch.stack((points[:, 0, 0], -points[:, 0, 0]), dim=1)
        base = types.SimpleNamespace(module=types.SimpleNamespace(classification_only=classify))
        host = types.SimpleNamespace(tqdm=lambda iterable, **kw: iterable)
        batches = [(torch.zeros(2, 4, 3), torch.tensor([1, 0])),
                   (torch.zeros(1, 4, 3), torch.tensor([1]))]
        diagnostic_config = {}
        with patch.dict(sys.modules, {"tta_gsd": types.SimpleNamespace(tta_gsd_reconstruct=reconstruct)}), \
                patch.object(run_baseline, "tta_preprocess_points", side_effect=lambda x, *a: (x, None, None)), \
                patch.object(run_baseline, "tta_postprocess_points", side_effect=lambda x, *a: -x):
            targets, predictions = gsd_protocol.process_batches(
                batches, base, None, args, host, torch, 35,
                scheduler_observer=lambda scheduler: None,
                batch_observer=lambda target, pred: observed.append(target.numel()),
                diagnostics_observer=lambda event: gsd_protocol.merge_diagnostics(diagnostic_config, "background", event))
        self.assertEqual(calls, [(35, .01, .01, .95, 100, 1.0)] * 2)
        self.assertEqual(observed, [2, 1])
        self.assertEqual(targets.tolist(), [1, 0, 1])
        self.assertEqual(predictions.tolist(), [1, 1, 1])
        self.assertEqual(diagnostic_config["gsd_diagnostics"]["background"]["step"]["records"], 2)

    def test_old_methods_parse_and_config_match_pinned_baseline(self):
        # Compare actual existing APIs against the immutable starting revision.
        source = subprocess.check_output(["git", "show", "79cc027:run_baseline.py"], text=True)
        old = types.ModuleType("baseline_reference")
        old.__file__ = run_baseline.__file__
        exec(compile(source, "baseline_reference", "exec"), old.__dict__)
        scopes = [[], ["--method", "source_only"],
                  ["--method", "preprocessing_identity", "--batch_size", "32", "--max-batches", "0",
                   "--corruptions", *run_baseline.CORRUPTIONS]]
        for method in (run_baseline.SHARED_DECODER_METHOD, run_baseline.SCD_NORMALIZATION_METHOD,
                       run_baseline.SCD_LAMBDA96_METHOD):
            scopes.append(["--method", method, "--batch_size", "32", "--max-batches", "0",
                           "--lambdaa", ".96" if method == run_baseline.SCD_LAMBDA96_METHOD else ".95",
                           "--corruptions", "gaussian", "impulse"])
        for cli in scopes:
            with self.subTest(cli=cli):
                old_args, new_args = old.parse_arguments(cli), run_baseline.parse_arguments(cli)
                self.assertEqual(vars(old_args), vars(new_args))
                with patch.object(old, "command_output", return_value="same environment"), \
                        patch.object(run_baseline, "command_output", return_value="same environment"):
                    self.assertEqual(old.build_config(old_args), run_baseline.build_config(new_args))
                self.assertEqual(old.notes_for_run(old_args), run_baseline.notes_for_run(new_args))


if __name__ == "__main__":
    unittest.main()
