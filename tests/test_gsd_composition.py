"""CPU contracts for per-example GSD block composition."""
import json
import math
import unittest

import torch

from gsd_composition import CompositionConfig, compose_block


class ComposeBlockTests(unittest.TestCase):
    def test_agreeing_vectors_keep_their_ordinary_sum_under_pcgrad(self):
        scd = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        spectral = torch.tensor([[2.0, 0.0]], dtype=torch.float64)

        direction, diagnostics = compose_block(
            scd, spectral, scd_weight=1.0, spectral_weight=1.0, mode="pcgrad"
        )

        torch.testing.assert_close(direction, torch.tensor([[3.0, 0.0]], dtype=torch.float64))
        self.assertFalse(diagnostics[0]["projection_skipped"])
        self.assertAlmostEqual(diagnostics[0]["cap_factor"], 1.0)

    def test_conflicting_pair_uses_symmetric_projection_then_caps_to_sum_norm(self):
        scd = torch.tensor([[1.0, 0.0]])
        spectral = torch.tensor([[-1.0, 1.0]])

        direction, diagnostics = compose_block(
            scd, spectral, scd_weight=1.0, spectral_weight=1.0, mode="pcgrad"
        )

        torch.testing.assert_close(
            direction, torch.tensor([[0.5, 1.5]]) / math.sqrt(2.5)
        )
        self.assertAlmostEqual(diagnostics[0]["raw_projected_norm"], math.sqrt(2.5))
        self.assertAlmostEqual(diagnostics[0]["capped_projected_norm"], 1.0)
        self.assertAlmostEqual(diagnostics[0]["cap_factor"], 1.0 / math.sqrt(2.5))
        self.assertGreater(diagnostics[0]["scd_direction_dot"], 0.0)
        self.assertGreater(diagnostics[0]["spectral_direction_dot"], 0.0)
        torch.testing.assert_close(
            compose_block(scd, spectral, scd_weight=1.0, spectral_weight=1.0, mode="sum")[0],
            torch.tensor([[0.0, 1.0]]),
        )

    def test_opposite_vectors_return_zero_without_nan_diagnostics(self):
        direction, diagnostics = compose_block(
            torch.tensor([[1.0]]), torch.tensor([[-1.0]]),
            scd_weight=1.0, spectral_weight=1.0, mode="pcgrad",
        )

        torch.testing.assert_close(direction, torch.zeros_like(direction))
        self.assertEqual(diagnostics[0]["cosine"], -1.0)
        self.assertEqual(diagnostics[0]["angle_degrees"], 180.0)
        json.dumps(diagnostics, allow_nan=False)

    def test_conflicts_are_projected_per_example_and_permutation_equivariantly(self):
        scd = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        spectral = torch.tensor([[-1.0, 1.0], [1.0, 1.0]])

        direction, rows = compose_block(
            scd, spectral, scd_weight=1.0, spectral_weight=1.0, mode="pcgrad"
        )
        reverse, reverse_rows = compose_block(
            scd.flip(0), spectral.flip(0), scd_weight=1.0,
            spectral_weight=1.0, mode="pcgrad",
        )

        torch.testing.assert_close(direction[0], torch.tensor([0.5, 1.5]) / math.sqrt(2.5))
        torch.testing.assert_close(direction[1], torch.tensor([2.0, 1.0]))
        torch.testing.assert_close(reverse, direction.flip(0))
        self.assertEqual(reverse_rows, list(reversed(rows)))

    def test_scd_priority_can_preserve_negative_spectral_first_order_dot(self):
        direction, rows = compose_block(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[-2.0, 1.0]]),
            scd_weight=1.0, spectral_weight=1.0, mode="scd_priority",
        )

        torch.testing.assert_close(direction, torch.tensor([[1.0, 1.0]]))
        self.assertLess(torch.dot(direction[0], torch.tensor([-2.0, 1.0])).item(), 0.0)
        self.assertLess(rows[0]["spectral_direction_dot"], 0.0)

    def test_sum_norm_pcgrad_records_projection_cap_and_matching_scale_separately(self):
        direction, rows = compose_block(
            torch.tensor([[10.0, 0.0]]), torch.tensor([[-1.0, 1.0]]),
            scd_weight=1.0, spectral_weight=1.0, mode="sum_norm_pcgrad",
        )

        self.assertEqual(rows[0]["cap_factor"], 1.0)
        self.assertLess(rows[0]["applied_scale"], 1.0)
        self.assertAlmostEqual(rows[0]["capped_projected_norm"], rows[0]["raw_projected_norm"])
        self.assertAlmostEqual(rows[0]["applied_norm"], rows[0]["capped_projected_norm"])
        torch.testing.assert_close(direction, torch.tensor([[9.0, 1.0]]) * rows[0]["applied_scale"])

    def test_all_routes_emit_json_safe_diagnostics(self):
        scd = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        spectral = torch.tensor([[-1.0, 1.0], [0.0, 0.0]])
        for mode in ("off", "scd", "spectral", "sum", "pcgrad", "scd_priority", "sum_norm_pcgrad"):
            _, rows = compose_block(
                scd, spectral, scd_weight=1.0, spectral_weight=1.0, mode=mode,
            )
            json.dumps(rows, allow_nan=False)

    def test_weights_are_applied_once_and_inputs_are_not_mutated(self):
        scd = torch.tensor([[2.0, 0.0]])
        spectral = torch.tensor([[0.0, 3.0]])
        original_scd = scd.clone()
        original_spectral = spectral.clone()

        direction, _ = compose_block(
            scd, spectral, scd_weight=0.5, spectral_weight=2.0, mode="sum"
        )

        torch.testing.assert_close(direction, torch.tensor([[1.0, 6.0]]))
        torch.testing.assert_close(scd, original_scd)
        torch.testing.assert_close(spectral, original_spectral)

    def test_missing_gradient_is_a_zero_direction_with_presence_diagnostic(self):
        direction, rows = compose_block(
            None, torch.tensor([[2.0, -1.0]]), scd_weight=1.0,
            spectral_weight=0.0, mode="sum",
        )

        torch.testing.assert_close(direction, torch.zeros((1, 2)))
        self.assertFalse(rows[0]["scd_present"])
        self.assertTrue(rows[0]["spectral_present"])

    def test_zero_and_tiny_norms_skip_projection_without_amplification(self):
        zero_direction, zero_rows = compose_block(
            torch.zeros((1, 2)), torch.zeros((1, 2)),
            scd_weight=1.0, spectral_weight=1.0, mode="pcgrad",
        )
        torch.testing.assert_close(zero_direction, torch.zeros((1, 2)))
        self.assertIsNone(zero_rows[0]["cosine"])
        self.assertIsNone(zero_rows[0]["angle_degrees"])

        for scd, spectral in ((
            torch.tensor([[1e-14, 0.0]]), torch.tensor([[-1e-14, 0.0]])
        ),):
            direction, rows = compose_block(
                scd, spectral, scd_weight=1.0, spectral_weight=1.0,
                mode="pcgrad", norm_floor=1e-12,
            )
            self.assertTrue(rows[0]["projection_skipped"])
            torch.testing.assert_close(direction, scd + spectral)

    def test_output_preserves_input_dtype_and_fails_on_nonfinite_gradients(self):
        direction, _ = compose_block(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            scd_weight=1.0, spectral_weight=1.0, mode="sum",
        )
        self.assertEqual(direction.dtype, torch.float32)
        with self.assertRaises(ValueError):
            compose_block(
                torch.tensor([[float("inf")]]), torch.tensor([[1.0]]),
                scd_weight=1.0, spectral_weight=1.0, mode="sum",
            )
        with self.assertRaises(ValueError):
            compose_block(
                torch.tensor([[1e308]], dtype=torch.float64), torch.tensor([[0.0]], dtype=torch.float64),
                scd_weight=1e308, spectral_weight=1.0, mode="sum",
            )
        with self.assertRaises(ValueError):
            compose_block(
                torch.ones((1, 2)), torch.ones((2, 2)),
                scd_weight=1.0, spectral_weight=1.0, mode="sum",
            )
        with self.assertRaises(ValueError):
            compose_block(
                torch.ones((1, 2)), torch.ones((1, 2)),
                scd_weight=float("nan"), spectral_weight=1.0, mode="sum",
            )

    def test_config_defaults_and_block_route_validation(self):
        config = CompositionConfig()
        self.assertEqual(config.local_mode, "off")
        self.assertEqual(config.style_mode, "off")
        self.assertEqual(config.local_scd_weight, 1.0)
        self.assertEqual(config.local_spectral_weight, 1.0)
        self.assertEqual(config.style_scd_weight, 1.0)
        self.assertEqual(config.style_spectral_weight, 1.0)
        self.assertEqual(config.schema_version, 1)
        with self.assertRaises(ValueError):
            CompositionConfig(local_mode="pcgrad")
        with self.assertRaises(ValueError):
            CompositionConfig(style_mode="unknown")


if __name__ == "__main__":
    unittest.main()
