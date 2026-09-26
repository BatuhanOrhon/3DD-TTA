"""Independent CPU checks of the v2 metric and numerical edge cases."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch

from graph_spectral import SpectralConfig, build_spectral_target, build_smooth_spectral_target


class SmoothSpectralTests(unittest.TestCase):
    def setUp(self):
        self.reference = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]]],
                                      dtype=torch.float64)
        self.config = SpectralConfig(k=1, delta=1, modes=1)

    def test_hard_matches_v1_with_actual_rank_over_original_point_count(self):
        # Disconnected pairs expand requested rank 1 to actual rank 2.
        reference = torch.tensor([[[0., 0., 0.], [.1, 0., 0.],
                                   [3., 0., 0.], [3.2, 0., 0.]]], dtype=torch.float64)
        for ref in (self.reference, reference):
            with self.subTest(vertices=ref.shape[1]):
                v1 = build_spectral_target(ref, self.config)
                hard = build_smooth_spectral_target(ref, self.config, profile="hard")
                residual = torch.arange(ref.numel(), dtype=ref.dtype).reshape_as(ref) / 10
                prediction = (ref + residual).requires_grad_()
                ratio = v1.bases[0].shape[1] / ref.shape[1]
                torch.testing.assert_close(hard.loss(prediction), ratio * v1.loss(prediction))
                grad1, = torch.autograd.grad(v1.loss(prediction), prediction)
                grad2, = torch.autograd.grad(hard.loss(prediction), prediction)
                torch.testing.assert_close(grad2, ratio * grad1)

    def test_isolates_and_empty_graphs_have_zero_rows_and_correct_storage(self):
        for profile in ("hard", "smooth"):
            for empty in (False, True):
                with self.subTest(profile=profile, empty=empty):
                    config = replace(self.config, delta=1e-6) if empty else self.config
                    target = build_smooth_spectral_target(
                        self.reference, config, profile=profile,
                        beta=2. if profile == "smooth" else None)
                    operator = target.filters[0]
                    self.assertEqual(target.diagnostics[0]["operator_storage_bytes"],
                                     operator.numel() * operator.element_size())
                    torch.testing.assert_close(operator[2], torch.zeros(3, dtype=operator.dtype))
                    pred = (self.reference + .2).requires_grad_()
                    grad, = torch.autograd.grad(target.loss(pred), pred)
                    torch.testing.assert_close(grad[0, 2], torch.zeros(3, dtype=grad.dtype))
                    if empty:
                        self.assertEqual(float(target.loss(pred)), 0)
                        torch.testing.assert_close(grad, torch.zeros_like(grad))

    def test_smooth_finite_difference_detachment_batch_sum_and_rng(self):
        reference = self.reference.repeat(2, 1, 1).requires_grad_()
        state = torch.get_rng_state().clone()
        target = build_smooth_spectral_target(reference, self.config, profile="smooth", beta=2.)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(target.reference_xyz.requires_grad)
        self.assertTrue(all(not op.requires_grad for op in target.filters))
        prediction = (reference.detach() + .2).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(target.loss, (prediction,)))
        single = build_smooth_spectral_target(self.reference, self.config, profile="smooth", beta=2.)
        torch.testing.assert_close(target.loss(prediction), 2 * single.loss(prediction[:1]))
        self.assertIsNone(reference.grad)

    def test_heat_operator_matches_matrix_exponential_and_is_independent_of_modes(self):
        # Square with k=2 has the four sides, all of identical edge weight.
        ref = torch.tensor([[[0., 0., 0.], [.1, 0., 0.],
                             [0., .1, 0.], [.1, .1, 0.]]], dtype=torch.float64)
        adjacency = math.exp(-.01 / (2 * .2**2)) * torch.tensor(
            [[0., 1., 1., 0.], [1., 0., 0., 1.],
             [1., 0., 0., 1.], [0., 1., 1., 0.]], dtype=ref.dtype)
        laplacian = torch.diag(adjacency.sum(1)) - adjacency
        expected = torch.matrix_exp(-1.3 * laplacian / laplacian.diagonal().mean())
        for modes in (1, 2, 4):
            config = SpectralConfig(k=2, delta=.2, modes=modes)
            target = build_smooth_spectral_target(ref, config, profile="smooth", beta=1.3)
            torch.testing.assert_close(target.filters[0], expected, atol=1e-12, rtol=1e-12)

    def test_sign_and_repeated_eigenspace_rotation_preserve_operator(self):
        ref = torch.zeros(1, 4, 3, dtype=torch.float64)
        config = SpectralConfig(k=3, modes=1)
        expected = build_smooth_spectral_target(ref, config, profile="smooth", beta=2.)
        eigh = torch.linalg.eigh

        def rotated_eigh(laplacian):
            values, vectors = eigh(laplacian)
            # Complete graph: eigenvalue 4 has multiplicity 3.
            rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=vectors.dtype)
            vectors[:, 1:3] = vectors[:, 1:3] @ rotation
            return values, -vectors

        with patch("torch.linalg.eigh", side_effect=rotated_eigh):
            actual = build_smooth_spectral_target(ref, config, profile="smooth", beta=2.)
        torch.testing.assert_close(actual.filters[0], expected.filters[0])

    def test_permutation_preserves_loss_and_gradient(self):
        perm = torch.tensor([2, 0, 1])
        target = build_smooth_spectral_target(self.reference, self.config, profile="smooth", beta=2.)
        other = build_smooth_spectral_target(self.reference[:, perm], self.config, profile="smooth", beta=2.)
        prediction = (self.reference + torch.arange(9).reshape(1, 3, 3) / 10).requires_grad_()
        permuted = prediction.detach()[:, perm].requires_grad_()
        torch.testing.assert_close(target.loss(prediction), other.loss(permuted))
        grad, = torch.autograd.grad(target.loss(prediction), prediction)
        perm_grad, = torch.autograd.grad(other.loss(permuted), permuted)
        torch.testing.assert_close(grad[:, perm], perm_grad)

    def test_large_finite_beta_preserves_exact_zero_mode_in_float32(self):
        ref = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], dtype=torch.float32)
        target = build_smooth_spectral_target(ref, self.config, profile="smooth", beta=1e40)
        torch.testing.assert_close(target.filters[0], torch.full((2, 2), .5))

    def test_invalid_profiles_beta_and_materially_negative_eigenvalues_rejected(self):
        for profile, beta in (("unknown", None), ("hard", 1.), ("smooth", None),
                              ("smooth", 0.), ("smooth", -1.), ("smooth", True),
                              ("smooth", float("inf")), ("smooth", float("nan"))):
            with self.subTest(profile=profile, beta=beta), self.assertRaises(ValueError):
                build_smooth_spectral_target(self.reference, self.config, profile=profile, beta=beta)
        eigh = torch.linalg.eigh

        def negative_eigh(laplacian):
            values, vectors = eigh(laplacian)
            values[0] = -1e-3
            return values, vectors

        for profile in ("hard", "smooth"):
            with self.subTest(profile=profile), patch("torch.linalg.eigh", side_effect=negative_eigh), \
                    self.assertRaisesRegex(FloatingPointError, "materially negative"):
                build_smooth_spectral_target(self.reference, self.config, profile=profile,
                                             beta=1. if profile == "smooth" else None)
