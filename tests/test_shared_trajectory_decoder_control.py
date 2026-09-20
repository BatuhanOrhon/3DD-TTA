from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import patch

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
from tta import SharedTrajectory, decode_shared_trajectory, selective_chamfer_loss


class SharedTrajectoryDecoderControlTests(unittest.TestCase):
    def test_actual_trajectory_updates_style_once_per_step_and_preserves_default(self):
        import tta
        counts = {"encode": 0, "prior": 0}
        class CPU:
            def __getattr__(self, name):
                return getattr(torch, name)
            def ones(self, *a, **kw):
                kw["device"] = "cpu"
                return torch.ones(*a, **kw)
        class Scheduler:
            def __init__(self, **kw):
                pass
            def set_timesteps(self, total, device):
                self.timesteps = torch.arange(total - 1, -1, -1)
                self.alphas_cumprod = torch.full((total,), .8)
            def step(self, noise, t, x):
                return SimpleNamespace(pred_original_sample=x-noise, prev_sample=x-.1*noise)
        class VAE:
            def encode(self, x):
                counts["encode"] += 1
                return None, None, [[torch.ones(1, 128)], [torch.ones(1, 8192)]]
            def global2style(self, z):
                return z
            def decoder(self, unused, beta, context, style):
                return context.reshape(1, 2048, 4)[:, :, :3] + .01*style.mean()
        def prior(x, t, condition_input, clip_feat):
            counts["prior"] += 1
            return .1*x + condition_input.mean(dim=1, keepdim=True)
        def chamfer(a, b):
            d = (a-b).square().sum(-1)
            return d, d, None, None
        lion = SimpleNamespace(vae=VAE(), priors=[None, prior])
        with patch.multiple(tta, torch=CPU(), DDIMScheduler=Scheduler,
                            chamfer_grad=lambda: chamfer, grad_freeze=lambda m: None):
            torch.manual_seed(7)
            trajectory = tta.tta_reconstruct(
                torch.ones(1, 2048, 3), lion, 5, .01, .01, .95, return_trajectory=True)
            self.assertEqual(counts, {"encode": 1, "prior": 5})
            self.assertTrue(torch.equal(trajectory.original_style, torch.ones(1, 128, 1, 1)))
            self.assertFalse(torch.equal(trajectory.original_style, trajectory.updated_style))
            original, updated = tta.decode_shared_trajectory(lion, trajectory)
            self.assertEqual(counts, {"encode": 1, "prior": 5})
            torch.manual_seed(7)
            default = tta.tta_reconstruct(torch.ones(1, 2048, 3), lion, 5, .01, .01, .95)
            self.assertTrue(torch.equal(original, default))
            self.assertFalse(torch.equal(original, updated))

    def test_control_enables_guidance_but_disables_decoder_gradients(self):
        calls = []
        def reconstruct(*args, **kwargs):
            calls.append("trajectory")
            x = torch.ones(1, requires_grad=True)
            (x * 2).sum().backward()
            self.assertEqual(x.grad.item(), 2)
            return SharedTrajectory(x.detach(), x.detach(), x.detach())
        def decode(*args):
            self.assertFalse(torch.is_grad_enabled())
            calls.append("decode")
            return torch.ones(1, 2, 3), torch.ones(1, 2, 3)
        baseline = SimpleNamespace(tta_reconstruct=reconstruct, decode_shared_trajectory=decode)
        args = SimpleNamespace(gamma=.01, eta=.01, lambdaa=.95)
        with patch.object(run_baseline, "tta_preprocess_points", return_value=(torch.ones(1), None, None)):
            with patch.object(run_baseline, "tta_postprocess_points", side_effect=lambda x, *a: x):
                with torch.no_grad():
                    run_baseline.shared_trajectory_decoder_points(None, baseline, None, args, torch, 5)
        self.assertEqual(calls, ["trajectory", "decode"])

    def test_control_metadata_accepts_corruption_metrics(self):
        args = run_baseline.parse_arguments([
            "--method", run_baseline.SHARED_DECODER_METHOD, "--batch_size", "32",
            "--max-batches", "0", "--corruptions", "gaussian", "impulse"])
        config = run_baseline.build_config(args)
        config.setdefault("decoder_control", {})["gaussian"] = {"paired_delta_pp": 0}
        self.assertIn("gaussian", config["decoder_control"])

    def test_empty_failed_decoder_row_has_no_fabricated_accuracy(self):
        failed = run_baseline.corruption_row("audit", 0, "impulse", 0, 0, 1, 0, "failed")
        failed.update(run_baseline.decoder_control_row(0, 0, 0, 0, 0, 0))
        self.assertEqual(summarize([failed], "failed")["updated_style_macro_accuracy"], "")
        complete = run_baseline.corruption_row("audit", 0, "gaussian", 4, 2, 1, 0, "complete")
        complete.update(run_baseline.decoder_control_row(4, 1, 2, 2, .5, 1.5))
        summary = summarize([complete, failed], "failed")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["updated_style_micro_accuracy"], .5)

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

    def test_cli_accepts_shared_decoder_all15_confirmation_scope(self):
        args = run_baseline.parse_arguments([
            "--method", "shared_trajectory_decoder_control",
            "--dataset-name", "modelnet-c",
            "--severity", "5",
            "--batch_size", "32",
            "--seed", "2",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        self.assertEqual(args.corruptions, list(run_baseline.CORRUPTIONS))
        self.assertTrue(args.lion_eval_mode)
        self.assertFalse(args.lion_ema_mode)

    def test_cli_rejects_shared_decoder_partial_nonpilot_scope(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                run_baseline.parse_arguments([
                    "--method", "shared_trajectory_decoder_control",
                    "--dataset-name", "modelnet-c",
                    "--severity", "5",
                    "--batch_size", "32",
                    "--seed", "0",
                    "--max-batches", "0",
                    "--corruptions", "gaussian", "impulse", "uniform",
                ])

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

    def test_scd_normalization_divides_directed_sum_by_original_point_count(self):
        dists1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        dists2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        legacy = selective_chamfer_loss(dists1, dists2, 4, normalize=False)
        normalized = selective_chamfer_loss(dists1, dists2, 4, normalize=True)

        self.assertAlmostEqual(legacy.item(), 36.0)
        self.assertAlmostEqual(normalized.item(), 9.0)
        self.assertAlmostEqual(normalized.item(), legacy.item() / 4.0)

    def test_scd_normalization_control_locks_eval_and_scope(self):
        args = run_baseline.parse_arguments([
            "--method", run_baseline.SCD_NORMALIZATION_METHOD,
            "--dataset-name", "modelnet-c",
            "--severity", "5",
            "--batch_size", "32",
            "--seed", "0",
            "--max-batches", "0",
            "--corruptions", "gaussian", "impulse",
        ])

        self.assertTrue(args.lion_eval_mode)
        self.assertFalse(args.lion_ema_mode)
        config = run_baseline.build_config(args)
        self.assertEqual(config["method"], run_baseline.SCD_NORMALIZATION_METHOD)
        self.assertTrue(config["scd_normalization"]["enabled"])
        self.assertEqual(
            config["scd_normalization"]["denominator"],
            "original point-set cardinality")

    def test_scd_config_records_colab_runtime_contract(self):
        args = run_baseline.parse_arguments([
            "--method", run_baseline.SCD_NORMALIZATION_METHOD,
            "--dataset-name", "modelnet-c",
            "--severity", "5",
            "--batch_size", "32",
            "--seed", "0",
            "--max-batches", "0",
            "--corruptions", "gaussian", "impulse",
        ])

        config = run_baseline.build_config(args)
        self.assertIn("VAE encode", config["preprocessing"])
        self.assertIn("DDIM reverse", config["preprocessing"])
        self.assertIn("original shape_latent", config["preprocessing"])
        contract = config["scd_normalization"]
        self.assertEqual(contract["denominator_value"], 2048)
        self.assertEqual(contract["retained_fraction"], 0.95)
        self.assertEqual(contract["retained_count"], 1945)
        self.assertIn("Colab GPU", contract["integration_check"])

    def test_scd_normalization_control_accepts_all15_scope(self):
        args = run_baseline.parse_arguments([
            "--method", run_baseline.SCD_NORMALIZATION_METHOD,
            "--dataset-name", "modelnet-c",
            "--severity", "5",
            "--batch_size", "32",
            "--seed", "2",
            "--max-batches", "0",
            "--corruptions", *run_baseline.CORRUPTIONS,
        ])

        self.assertEqual(args.corruptions, list(run_baseline.CORRUPTIONS))
        self.assertTrue(args.lion_eval_mode)


if __name__ == "__main__":
    unittest.main()
