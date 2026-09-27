"""CPU contracts for label-free common-state calibration."""
import json
import random
import unittest
from unittest.mock import patch

import torch

import graph_spectral as spectral


class CalibrationTests(unittest.TestCase):
    def test_diagnostic_protocol_is_explicit_scd_only_and_blocks_scope_leaks(self):
        import run_baseline
        import gsd_protocol
        from contextlib import redirect_stderr
        from io import StringIO
        cli = ["--method", gsd_protocol.SMOOTH_METHOD, "--batch_size", "32",
               "--gsd-stage", "calibrate", "--gsd-weight", "0", "--gsd-profile", "hard",
               "--max-batches", "0", "--corruptions", "gaussian", "impulse"]
        args = run_baseline.parse_arguments(cli)
        self.assertEqual(args.gsd_development_count, 64)
        self.assertEqual(args.gsd_split_seed, 20260927)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run_baseline.parse_arguments(cli + ["--gsd-weight", "1"])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run_baseline.parse_arguments(cli + ["--gsd-stage", "development"])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run_baseline.parse_arguments(["--method", "source_only", "--gsd-development-count", "128"])

    def test_shared_spectrum_matches_dense_losses_gradients_with_one_eigh(self):
        reference = torch.arange(24, dtype=torch.float64).reshape(1, 8, 3) / 30
        config = spectral.SpectralConfig(k=3, delta=.4, modes=3)
        with patch.object(torch.linalg, "eigh", wraps=torch.linalg.eigh) as eigh:
            target = spectral.build_probe_spectral_target(reference, config)
        self.assertEqual(eigh.call_count, 1)
        prediction = (reference + .03 * torch.cos(reference * 10)).requires_grad_()
        for profile, beta in (("hard", None), ("smooth", .5), ("smooth", 2.), ("smooth", 8.)):
            dense = spectral.build_smooth_spectral_target(reference, config, profile=profile, beta=beta)
            actual = target.loss(prediction, profile=profile, beta=beta)
            expected = dense.loss(prediction)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(torch.autograd.grad(actual, prediction)[0],
                                       torch.autograd.grad(expected, prediction)[0],
                                       rtol=1e-10, atol=1e-12)

    def test_empty_spectrum_has_connected_zero_loss(self):
        reference = torch.arange(24, dtype=torch.float64).reshape(1, 8, 3)
        target = spectral.build_probe_spectral_target(
            reference, spectral.SpectralConfig(k=2, graph_gamma=1e9))
        prediction = reference.clone().requires_grad_()
        loss = target.loss(prediction, profile="smooth", beta=2.)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(torch.autograd.grad(loss, prediction)[0].abs().sum().item(), 0)

    def test_split_is_nested_unique_and_does_not_consume_rng(self):
        from gsd_calibration import development_indices
        before = random.getstate()
        short = development_indices(200, 64, 20260927)
        long = development_indices(200, 128, 20260927)
        self.assertEqual(short, long[:64])
        self.assertEqual(len(set(long)), 128)
        self.assertEqual(before, random.getstate())
        self.assertNotEqual(short, list(range(64)))
        with self.assertRaises(ValueError):
            development_indices(32, 64, 0)

    def test_per_sample_metrics_preserve_zero_denominators_and_cosine(self):
        from gsd_calibration import gradient_rows
        scd = torch.tensor([[3., 4.], [0., 0.]])
        spec = torch.tensor([[-.3, -.4], [1., 0.]])
        rows = gradient_rows(scd, scd, spec, spec)
        self.assertAlmostEqual(rows[0]["local_ratio"], .1, places=6)
        self.assertAlmostEqual(rows[0]["local_cosine"], -1, places=6)
        self.assertIsNone(rows[1]["local_ratio"])
        self.assertEqual(rows[1]["local_scd_norm"], 0)
        json.dumps(rows, allow_nan=False)

    def test_coefficient_uses_median_of_ratios_and_keeps_zero_numerators(self):
        from gsd_calibration import coefficient_summary
        rows = [dict(local_ratio=value, style_ratio=2 * value) for value in (0., .1, .2)]
        result = coefficient_summary(rows)
        self.assertAlmostEqual(result["local_ratio"]["median"], .1)
        self.assertAlmostEqual(result["weights"]["0.001"], .01)
        self.assertIsNone(coefficient_summary([dict(local_ratio=0., style_ratio=0.)])["weights"])
        self.assertIsNone(coefficient_summary(rows + [dict(local_ratio=None, style_ratio=0.)])["weights"])

    def test_state_metrics_separate_scd_spectral_and_total_updates(self):
        from gsd_calibration import state_rows
        state = torch.tensor([[3., 4.], [0., 0.]])
        grad = torch.tensor([[6., 8.], [0., 0.]])
        spectral_grad = -grad * .5
        rows = state_rows(state, state, grad, grad, state + state, .1, .1,
                          spectral_grad, spectral_grad)
        self.assertAlmostEqual(rows[0]["local_scd_update_state_ratio"], .2)
        self.assertAlmostEqual(rows[0]["local_spectral_update_state_ratio"], .1)
        self.assertAlmostEqual(rows[0]["local_total_update_state_ratio"], .1)
        self.assertAlmostEqual(rows[0]["local_total_ddim_ratio"], .1)
        self.assertIsNone(rows[1]["local_total_update_state_ratio"])
        self.assertIsNone(rows[1]["local_scd_ddim_ratio"])
        json.dumps(rows, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
