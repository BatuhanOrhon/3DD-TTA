"""[Code] CPU analytic/protocol checks for the static latent spectral objective."""

from dataclasses import replace
import json
import math
import unittest
from unittest.mock import patch

import torch

from graph_spectral import SpectralConfig, build_spectral_target


class GraphSpectralTests(unittest.TestCase):
    def setUp(self):
        self.reference = torch.tensor([
            [0.0, 0.0, 0.0], [0.2, 0.1, 0.0],
            [0.7, 0.3, 0.1], [1.4, -0.1, 0.3],
        ], dtype=torch.float64).unsqueeze(0)
        self.config = SpectralConfig(k=2, delta=1.0, graph_gamma=0.0, modes=2)

    def test_reference_and_basis_detached_and_rng_untouched(self):
        reference = self.reference.clone().requires_grad_()
        state = torch.random.get_rng_state().clone()
        target = build_spectral_target(reference, self.config)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertFalse(target.reference_xyz.requires_grad)
        self.assertTrue(all(not basis.requires_grad for basis in target.bases))
        with torch.no_grad():
            reference.add_(1)
        torch.testing.assert_close(target.reference_xyz, self.reference)
        json.dumps(target.diagnostics, allow_nan=False)

    def test_analytic_gradient_and_no_reference_gradient(self):
        reference = self.reference.clone().requires_grad_()
        target = build_spectral_target(reference, self.config)
        prediction = (reference.detach() + torch.tensor([0.2, -0.3, 0.1])).requires_grad_()
        loss = target.loss(prediction)
        gradient, = torch.autograd.grad(loss, prediction)
        basis = target.bases[0]
        residual = prediction.detach()[0] - reference.detach()[0]
        expected = 2 * basis @ basis.T @ residual / (3 * basis.shape[1])
        torch.testing.assert_close(gradient[0], expected)
        self.assertIsNone(reference.grad)
        self.assertGreater(float(gradient.norm()), 0)

    def test_finite_difference_gradient(self):
        target = build_spectral_target(self.reference, self.config)
        prediction = (self.reference + 0.2).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(target.loss, (prediction,)))

    def test_zero_reference_loss(self):
        target = build_spectral_target(self.reference, self.config)
        self.assertEqual(float(target.loss(self.reference)), 0.0)

    def test_full_band_is_normalized_coordinate_error(self):
        target = build_spectral_target(self.reference, replace(self.config, modes=20))
        prediction = self.reference + torch.tensor([0.3, 0.2, -0.1])
        torch.testing.assert_close(target.loss(prediction), (prediction - self.reference).square().mean())
        self.assertEqual(target.diagnostics[0]["actual_rank"], 4)
        self.assertIsNone(target.diagnostics[0]["eigengap"])

    def test_sign_and_selected_basis_rotation_invariance(self):
        target = build_spectral_target(self.reference, self.config)
        prediction = self.reference + torch.tensor([0.1, 0.7, -0.2])
        rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64)
        rotated = replace(target, bases=(target.bases[0] @ rotation,))
        signed = replace(target, bases=(-target.bases[0],))
        torch.testing.assert_close(target.loss(prediction), rotated.loss(prediction))
        torch.testing.assert_close(target.loss(prediction), signed.loss(prediction))

    def test_vertex_permutation_preserves_loss_and_gradient(self):
        target = build_spectral_target(self.reference, self.config)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = build_spectral_target(self.reference[:, permutation], self.config)
        perturbation = torch.arange(12, dtype=torch.float64).reshape(1, 4, 3) / 20
        prediction = (self.reference + perturbation).requires_grad_()
        perm_prediction = prediction.detach()[:, permutation].requires_grad_()
        torch.testing.assert_close(target.loss(prediction), permuted.loss(perm_prediction))
        gradient, = torch.autograd.grad(target.loss(prediction), prediction)
        perm_gradient, = torch.autograd.grad(permuted.loss(perm_prediction), perm_prediction)
        torch.testing.assert_close(gradient[:, permutation], perm_gradient)

    def test_batch_sum_matches_individual_targets_and_gradients(self):
        batch = torch.cat((self.reference, self.reference * 0.8), dim=0)
        target = build_spectral_target(batch, self.config)
        prediction = (batch + 0.2).requires_grad_()
        individual = [build_spectral_target(sample[None], self.config) for sample in batch]
        separate_loss = sum(item.loss(prediction[i:i + 1]) for i, item in enumerate(individual))
        torch.testing.assert_close(target.loss(prediction), separate_loss)
        batch_grad, = torch.autograd.grad(target.loss(prediction), prediction)
        separate_grad, = torch.autograd.grad(separate_loss, prediction)
        torch.testing.assert_close(batch_grad, separate_grad)
        self.assertEqual(len(target.diagnostics), 2)

    def test_threshold_uses_directed_sum_divided_by_n_times_k(self):
        config = replace(self.config, graph_gamma=0.6)
        target = build_spectral_target(self.reference, config)
        xyz = self.reference[0]
        expected_sum = 0.0
        for i in range(len(xyz)):
            distances = sorted(float((xyz[i] - xyz[j]).square().sum())
                               for j in range(len(xyz)) if j != i)
            expected_sum += sum(math.exp(-value / (2 * config.delta ** 2))
                                for value in distances[:config.k])
        self.assertAlmostEqual(target.diagnostics[0]["threshold"], 0.6 * expected_sum / (4 * 2))
        self.assertAlmostEqual(target.diagnostics[0]["directed_degree_mean"], expected_sum / 4)

    def test_two_vertices_exclude_self_and_use_standard_rbf(self):
        reference = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]], dtype=torch.float64)
        config = SpectralConfig(k=1, delta=1, graph_gamma=0, modes=1)
        target = build_spectral_target(reference, config)
        self.assertAlmostEqual(target.diagnostics[0]["degree_mean"], math.exp(-2))
        self.assertAlmostEqual(target.diagnostics[0]["eigenvalue_max"], 2 * math.exp(-2))
        expected_projector = torch.full((2, 2), 0.5, dtype=torch.float64)
        torch.testing.assert_close(target.bases[0] @ target.bases[0].T, expected_projector)

    def test_max_union_symmetry_and_both_endpoint_threshold_mask(self):
        reference = torch.tensor([[[0., 0., 0.], [1., 0., 0.],
                                   [1.2, 0., 0.], [3., 0., 0.]]], dtype=torch.float64)
        config = SpectralConfig(k=1, delta=1, graph_gamma=0, modes=4)
        target = build_spectral_target(reference, config)
        # Directed edges 0->1, 1->2, 2->1, 3->2 become exactly three edges.
        expected_adjacency = torch.zeros(4, 4, dtype=torch.float64)
        for i, j in ((0, 1), (1, 2), (2, 3)):
            weight = math.exp(-float((reference[0, i] - reference[0, j]).square().sum()) / 2)
            expected_adjacency[i, j] = expected_adjacency[j, i] = weight
        expected_laplacian = torch.diag(expected_adjacency.sum(1)) - expected_adjacency
        self.assertAlmostEqual(target.diagnostics[0]["degree_mean"], float(expected_adjacency.sum(1).mean()))
        self.assertAlmostEqual(target.diagnostics[0]["eigenvalue_max"], float(torch.linalg.eigvalsh(expected_laplacian)[-1]))
        masked = build_spectral_target(reference, replace(config, graph_gamma=1.1))
        self.assertEqual(masked.diagnostics[0]["isolation_count"], 2)
        self.assertTrue(torch.equal(masked.bases[0][[0, 3]], torch.zeros(2, 2, dtype=torch.float64)))
        self.assertAlmostEqual(masked.diagnostics[0]["degree_mean"], math.exp(-0.02) / 2)

    def test_duplicate_vertices_count_as_nonself_neighbors(self):
        reference = torch.zeros((1, 4, 3), dtype=torch.float64)
        config = SpectralConfig(k=3, delta=1, graph_gamma=0.6, modes=2)
        target = build_spectral_target(reference, config)
        self.assertEqual(target.diagnostics[0]["directed_degree_mean"], 3)
        self.assertEqual(target.diagnostics[0]["threshold"], 0.6)
        # The complete graph eigenvalue 4 has multiplicity three: include all.
        self.assertEqual(target.diagnostics[0]["actual_rank"], 4)
        self.assertEqual(target.diagnostics[0]["low_energy_fraction"], 0)
        torch.testing.assert_close(target.bases[0] @ target.bases[0].T, torch.eye(4, dtype=torch.float64))

    def test_disconnected_zero_eigenspace_boundary_expands(self):
        reference = torch.tensor([[[0., 0., 0.], [0.1, 0., 0.],
                                   [3., 0., 0.], [3.2, 0., 0.]]], dtype=torch.float64)
        target = build_spectral_target(reference, SpectralConfig(k=1, delta=1, modes=1))
        self.assertEqual(target.diagnostics[0]["actual_rank"], 2)
        self.assertEqual(target.diagnostics[0]["zero_mode_count"], 2)
        self.assertGreater(target.diagnostics[0]["eigendecomposition_seconds"], 0)
        self.assertGreater(target.diagnostics[0]["eigengap"], 0)

    def test_final_isolate_has_zero_basis_and_gradient(self):
        reference = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]]], dtype=torch.float64)
        target = build_spectral_target(reference, SpectralConfig(k=1, delta=1, modes=5))
        self.assertEqual(target.diagnostics[0]["isolation_count"], 1)
        self.assertEqual(target.diagnostics[0]["actual_rank"], 2)
        # Full active band retains the energy at vertices 0/1; vertex 2 is
        # outside the active graph and contributes to remaining energy.
        self.assertAlmostEqual(target.diagnostics[0]["low_energy_fraction"], 1 / 17)
        self.assertAlmostEqual(target.diagnostics[0]["remaining_energy_fraction"], 16 / 17)
        self.assertTrue(torch.equal(target.bases[0][2], torch.zeros(2, dtype=torch.float64)))
        prediction = (reference + 0.2).requires_grad_()
        gradient, = torch.autograd.grad(target.loss(prediction), prediction)
        self.assertTrue(torch.equal(gradient[0, 2], torch.zeros(3, dtype=torch.float64)))

    def test_empty_underflow_graph_has_connected_zero_loss(self):
        target = build_spectral_target(self.reference, replace(self.config, delta=1e-6))
        self.assertEqual(target.diagnostics[0]["actual_rank"], 0)
        self.assertEqual(target.diagnostics[0]["isolation_count"], 4)
        self.assertEqual(target.diagnostics[0]["zero_mode_count"], 0)
        self.assertEqual(target.diagnostics[0]["eigendecomposition_seconds"], 0)
        self.assertEqual(target.diagnostics[0]["remaining_energy_fraction"], 1)
        prediction = self.reference.clone().requires_grad_()
        loss = target.loss(prediction)
        gradient, = torch.autograd.grad(loss, prediction)
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(prediction)))

    def test_float32_supported_and_deterministic(self):
        reference = self.reference.float()
        first = build_spectral_target(reference, self.config)
        second = build_spectral_target(reference, self.config)
        torch.testing.assert_close(first.bases[0], second.bases[0], rtol=0, atol=0)
        self.assertEqual(first.loss(reference + 1).dtype, torch.float32)

    @staticmethod
    def _disconnected_float32_reference():
        generator = torch.Generator().manual_seed(0)
        return torch.cat((torch.randn(16, 3, generator=generator) * 0.01,
                          torch.randn(16, 3, generator=generator) * 0.01 + 10))[None]

    def test_float32_disconnected_roundoff_preserves_complete_zero_band(self):
        reference = self._disconnected_float32_reference()
        config = SpectralConfig(k=10, modes=1)
        target = build_spectral_target(reference, config)
        double_target = build_spectral_target(reference.double(), config)
        self.assertEqual(target.diagnostics[0]["actual_rank"], 2)
        self.assertEqual(target.diagnostics[0]["zero_mode_count"], 2)
        self.assertEqual(double_target.diagnostics[0]["actual_rank"], 2)
        self.assertEqual(double_target.diagnostics[0]["zero_mode_count"], 2)
        prediction = reference.clone()
        prediction[:, :16, 0] += 1
        torch.testing.assert_close(target.loss(prediction), torch.tensor(8 / 3), rtol=1e-5, atol=1e-5)
        diagnostics = target.diagnostics[0]
        self.assertGreater(diagnostics["roundoff_floor"], config.eigenspace_atol)
        self.assertAlmostEqual(diagnostics["effective_boundary_atol"], 2 * diagnostics["roundoff_floor"])
        self.assertAlmostEqual(diagnostics["effective_zero_atol"], diagnostics["roundoff_floor"])

    def test_float32_disconnected_permutation_preserves_loss_and_gradient(self):
        reference = self._disconnected_float32_reference()
        config = SpectralConfig(k=10, modes=1)
        permutation = torch.randperm(32, generator=torch.Generator().manual_seed(0))
        target = build_spectral_target(reference, config)
        permuted_target = build_spectral_target(reference[:, permutation], config)
        prediction = reference.clone()
        prediction[:, :16, 0] += 1
        prediction.requires_grad_()
        permuted_prediction = prediction.detach()[:, permutation].requires_grad_()
        loss = target.loss(prediction)
        permuted_loss = permuted_target.loss(permuted_prediction)
        torch.testing.assert_close(loss, permuted_loss, rtol=1e-5, atol=1e-5)
        gradient, = torch.autograd.grad(loss, prediction)
        permuted_gradient, = torch.autograd.grad(permuted_loss, permuted_prediction)
        torch.testing.assert_close(gradient[:, permutation], permuted_gradient, rtol=1e-5, atol=1e-5)
        expected_gradient = torch.zeros_like(prediction)
        expected_gradient[:, :16, 0] = 1 / 3
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-5, atol=1e-5)

    def test_float32_synthetic_signed_zero_roundoff_keeps_both_modes(self):
        real_eigh = torch.linalg.eigh
        expected_floor = []

        def perturbed_eigh(matrix):
            values, vectors = real_eigh(matrix)
            values[:2] = values.new_tensor([-1e-6, 1e-6])
            expected_floor.append(torch.finfo(matrix.dtype).eps * len(matrix)
                                  * float(matrix.abs().sum(dim=1).max()))
            return values, vectors

        with patch("graph_spectral.torch.linalg.eigh", side_effect=perturbed_eigh):
            target = build_spectral_target(self._disconnected_float32_reference(), SpectralConfig(k=10, modes=1))
        self.assertEqual(target.diagnostics[0]["actual_rank"], 2)
        self.assertEqual(target.diagnostics[0]["zero_mode_count"], 2)
        self.assertEqual(target.diagnostics[0]["roundoff_floor"], expected_floor[0])

    def test_roundoff_boundary_expansion_uses_fixed_requested_anchor(self):
        real_eigh = torch.linalg.eigh

        def nearby_eigh(matrix):
            values, vectors = real_eigh(matrix)
            floor = torch.finfo(matrix.dtype).eps * len(matrix) * float(matrix.abs().sum(dim=1).max())
            # Adjacent separations each fit the 2*floor allowance; the third
            # eigenvalue is outside the allowance of the fixed first anchor.
            values[:3] = values.new_tensor([-0.75, 0.75, 2.25]) * floor
            return values, vectors

        with patch("graph_spectral.torch.linalg.eigh", side_effect=nearby_eigh):
            target = build_spectral_target(self._disconnected_float32_reference(), SpectralConfig(k=10, modes=1))
        self.assertEqual(target.diagnostics[0]["actual_rank"], 2)
        self.assertEqual(target.diagnostics[0]["zero_mode_count"], 2)

    def test_bad_config_rejected(self):
        for kwargs in ({"k": 0}, {"k": True}, {"modes": 1.2}, {"delta": 0},
                       {"delta": -1}, {"delta": float("nan")}, {"graph_gamma": -1},
                       {"graph_gamma": float("inf")}, {"eigenspace_rtol": -1},
                       {"eigenspace_atol": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpectralConfig(**kwargs)

    def test_bad_reference_rejected(self):
        for reference in (torch.zeros(4, 3), torch.zeros(1, 1, 3), torch.zeros(0, 4, 3),
                          torch.zeros(1, 4, 4), self.reference.long(), self.reference.half(),
                          self.reference * float("nan")):
            with self.subTest(shape=reference.shape, dtype=reference.dtype), self.assertRaises(ValueError):
                build_spectral_target(reference, self.config)
        with self.assertRaises(ValueError):
            build_spectral_target(self.reference, SpectralConfig(k=4))
        with self.assertRaises(TypeError):
            build_spectral_target([], self.config)
        with self.assertRaises(TypeError):
            build_spectral_target(self.reference, {})

    def test_prediction_contract_rejected(self):
        target = build_spectral_target(self.reference, self.config)
        for prediction in (self.reference.float(), self.reference.repeat(2, 1, 1),
                           self.reference * float("inf")):
            with self.subTest(dtype=prediction.dtype), self.assertRaises(ValueError):
                target.loss(prediction)

    def test_unrepresentable_rbf_denominator_rejected(self):
        for dtype, delta in ((torch.float32, 1e-30), (torch.float64, 1e-200),
                             (torch.float32, 1e30), (torch.float64, 1e200)):
            with self.subTest(dtype=dtype, delta=delta), self.assertRaisesRegex(ValueError, "delta squared"):
                build_spectral_target(self.reference.to(dtype), replace(self.config, delta=delta))

    def test_finite_coordinates_with_overflowing_distances_rejected(self):
        for dtype, scale in ((torch.float32, 1e30), (torch.float64, 1e200)):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "squared distances"):
                build_spectral_target(self.reference.to(dtype) * scale, self.config)

    def test_finite_config_with_overflowing_threshold_rejected(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            build_spectral_target(self.reference.float(), replace(self.config, graph_gamma=1e300))

    def test_nonfinite_eigenvalues_or_eigenvectors_rejected(self):
        real_eigh = torch.linalg.eigh
        for corrupt_values in (True, False):
            def corrupt_eigh(matrix):
                values, vectors = real_eigh(matrix)
                if corrupt_values:
                    values[0] = float("nan")
                else:
                    vectors[0, 0] = float("inf")
                return values, vectors
            with self.subTest(corrupt_values=corrupt_values), patch(
                "graph_spectral.torch.linalg.eigh", side_effect=corrupt_eigh
            ), self.assertRaisesRegex(FloatingPointError, "eigendecomposition"):
                build_spectral_target(self.reference, self.config)

    def test_nonfinite_loss_rejected(self):
        target = build_spectral_target(self.reference.float(), self.config)
        with self.assertRaisesRegex(FloatingPointError, "spectral loss"):
            target.loss(torch.full_like(target.reference_xyz, 1e30))

    def test_energy_diagnostics_finite_for_large_constant_reference(self):
        # Finite graph distances even though unscaled float32 energy overflows.
        reference = torch.full((1, 4, 3), 1e30, dtype=torch.float32)
        target = build_spectral_target(reference, replace(self.config, k=3, modes=4))
        self.assertAlmostEqual(target.diagnostics[0]["low_energy_fraction"], 1, places=6)
        self.assertAlmostEqual(target.diagnostics[0]["remaining_energy_fraction"], 0, places=6)
        json.dumps(target.diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
