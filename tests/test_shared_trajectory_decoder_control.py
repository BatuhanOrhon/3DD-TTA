from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import sys
import types
import unittest

import numpy as np
import torch

import run_baseline
from research_artifacts import summarize


def _install_chamfer_import_stub():
    module_names = [
        "third_party.ChamferDistancePytorch",
        "third_party.ChamferDistancePytorch.chamfer3D",
        "third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D",
    ]
    for name in module_names:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[module_names[-1]].chamfer_3DDist = lambda: None
    diffusers = types.ModuleType("diffusers")
    diffusers.DDIMScheduler = object
    sys.modules.setdefault("diffusers", diffusers)
    utilities = types.ModuleType("utilities_3dd_tta")
    utilities.grad_freeze = lambda module: None
    sys.modules.setdefault("utilities_3dd_tta", utilities)


_install_chamfer_import_stub()
from tta import SharedTrajectory, decode_shared_trajectory


class SharedTrajectoryDecoderControlTests(unittest.TestCase):
    def test_decoder_variants_share_one_final_local_latent_and_use_both_styles(self):
        final_local = torch.randn(2, 4, 1, 1)
        original_style = torch.randn(2, 3)
        updated_style = torch.randn(2, 3)
        trajectory = SharedTrajectory(final_local, original_style, updated_style)
        calls = []

        class Decoder:
            def __call__(self, latent, beta, context, style):
                calls.append((context, style))
                return style.unsqueeze(1).expand(-1, 4, -1)

        lion = SimpleNamespace(vae=SimpleNamespace(decoder=Decoder()))
        original_points, updated_points = decode_shared_trajectory(lion, trajectory)

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], calls[1][0])
        self.assertIs(calls[0][1], original_style)
        self.assertIs(calls[1][1], updated_style)
        self.assertFalse(torch.equal(original_points, updated_points))

    def test_decoder_control_metrics_report_paired_values(self):
        target = torch.tensor([0, 1, 2, 3])
        original_pred = torch.tensor([0, 0, 2, 1])
        updated_pred = torch.tensor([0, 1, 0, 1])
        metrics = run_baseline.decoder_control_batch_metrics(
            target,
            original_pred,
            updated_pred,
            torch.zeros(4, 2, 3),
            torch.ones(4, 2, 3),
            torch.zeros(4, 5),
            torch.full((4, 5), 2.0),
        )

        self.assertEqual(metrics["n_examples"], 4)
        self.assertEqual(metrics["original_style_n_correct"], 2)
        self.assertEqual(metrics["updated_style_n_correct"], 2)
        self.assertEqual(metrics["disagreement_n"], 2)
        self.assertAlmostEqual(metrics["decoder_output_difference"], 1.0)
        self.assertAlmostEqual(metrics["style_displacement"], 2.0 * np.sqrt(5.0), places=6)

    def test_second_classifier_call_restores_numpy_and_torch_rng(self):
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        with run_baseline.preserve_classifier_rng(np, torch):
            np.random.rand(7)
            torch.rand(7)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_cli_accepts_exact_shared_decoder_pilot_scope_and_forces_eval_mode(self):
        args = run_baseline.parse_arguments([
            "--method", "shared_trajectory_decoder_control",
            "--dataset-name", "modelnet-c",
            "--severity", "5",
            "--batch_size", "32",
            "--seed", "1",
            "--max-batches", "0",
            "--corruptions", "gaussian", "impulse",
        ])

        self.assertEqual(args.method, "shared_trajectory_decoder_control")
        self.assertTrue(args.lion_eval_mode)
        self.assertFalse(args.lion_ema_mode)
        self.assertEqual(args.corruptions, ["gaussian", "impulse"])

    def test_cli_rejects_shared_decoder_scope_change(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                run_baseline.parse_arguments([
                    "--method", "shared_trajectory_decoder_control",
                    "--dataset-name", "modelnet-c",
                    "--severity", "5",
                    "--batch_size", "16",
                    "--seed", "0",
                    "--max-batches", "0",
                    "--corruptions", "gaussian", "impulse",
                ])

    def test_default_original_path_remains_original_style(self):
        args = run_baseline.parse_arguments([])
        config = run_baseline.build_config(args)
        self.assertEqual(args.method, "3dd_original")
        self.assertFalse(args.lion_eval_mode)
        self.assertEqual(config["final_decode_style"], "original shape_latent")

    def test_decoder_control_summary_has_explicit_macro_and_micro_fields(self):
        row = run_baseline.corruption_row(
            "decoder-control-test", 0, "gaussian", 4, 2, 1.0, 2.0,
            "complete", method=run_baseline.SHARED_DECODER_METHOD, severity=5)
        row.update(run_baseline.decoder_control_row(4, 1, 2, 2, 0.5, 1.5))
        summary = summarize([row], "complete")

        self.assertAlmostEqual(summary["original_style_macro_accuracy"], 0.25)
        self.assertAlmostEqual(summary["updated_style_micro_accuracy"], 0.5)
        self.assertAlmostEqual(summary["paired_delta_pp_micro"], 25.0)


if __name__ == "__main__":
    unittest.main()
