"""CPU witnesses for the standalone H4 heat approximation."""
import unittest
from unittest.mock import patch

import torch

from graph_spectral import SpectralConfig, build_probe_spectral_target
import sparse_heat


class SparseHeatTests(unittest.TestCase):
    def setUp(self):
        # Duplicate/tied points, a separate component, and one RBF-underflow isolate.
        self.x = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [.1, 0., 0.],
                                [1., 0., 0.], [1.1, 0., 0.], [100., 0., 0.]]],
                              dtype=torch.float64)
        self.config = SpectralConfig(k=2, delta=.15, graph_gamma=.6)

    def test_graph_matches_existing_eigensystem_without_calling_eigh(self):
        probe = build_probe_spectral_target(self.x, self.config)
        with patch('torch.linalg.eigh', side_effect=AssertionError('unexpected eigh')):
            target = sparse_heat.build_sparse_heat_target(self.x, self.config)
        indices, u, q, _ = probe.spectra[0]
        graph = target.graphs[0]
        torch.testing.assert_close(graph.active_indices, indices, atol=0, rtol=0)
        # U/q come from the existing builder, not a repeated graph recipe in this test.
        expected = (u * q) @ u.T
        torch.testing.assert_close(graph.laplacian.to_dense(), expected, atol=2e-14, rtol=2e-14)
        self.assertLessEqual(float(q.max()), graph.upper_bound + 1e-14)
        self.assertEqual(graph.laplacian.layout, torch.sparse_coo)
        self.assertFalse(graph.laplacian.requires_grad)
        self.assertEqual(graph.mean_active_degree, probe.diagnostics[0]['mean_active_degree'])

    def test_graph_parity_across_dtype_pruning_and_distance_ties(self):
        for dtype in (torch.float32, torch.float64):
            for gamma in (0., .6, 3., 100.):
                x = self.x.to(dtype)
                config = SpectralConfig(k=2, delta=.15, graph_gamma=gamma)
                probe = build_probe_spectral_target(x, config)
                target = sparse_heat.build_sparse_heat_target(x, config)
                indices, u, q, _ = probe.spectra[0]
                graph = target.graphs[0]
                torch.testing.assert_close(indices, graph.active_indices, atol=0, rtol=0)
                torch.testing.assert_close(graph.laplacian.to_dense(), (u * q) @ u.T,
                                           atol=2e-6 if dtype == torch.float32 else 2e-14,
                                           rtol=2e-6 if dtype == torch.float32 else 2e-14)

    def test_square_root_loss_gradient_bounds_and_order_sensitivity(self):
        target = sparse_heat.build_sparse_heat_target(self.x, self.config)
        graph = target.graphs[0]
        values, basis = torch.linalg.eigh(graph.laplacian.to_dense())
        residual = torch.arange(self.x.numel(), dtype=self.x.dtype).reshape_as(self.x) / 7
        residual.requires_grad_()
        active = residual[0, graph.active_indices]
        exact = (active * (torch.matrix_exp(-2 * graph.laplacian.to_dense()) @ active)).sum() / 18
        exact_gradient, = torch.autograd.grad(exact, residual, retain_graph=True)
        errors = []
        for order in (8, 16, 32):
            approximation = target.prepare(order=order)
            loss = approximation.loss(residual)
            gradient, = torch.autograd.grad(loss, residual, retain_graph=True)
            coefficients = approximation.coefficients[0]
            scaled = 2 * values / graph.upper_bound - 1
            terms = [torch.ones_like(scaled), scaled]
            for _ in range(2, order + 1):
                terms.append(2 * scaled * terms[-1] - terms[-2])
            polynomial = sum(c * t for c, t in zip(coefficients, terms))
            epsilon = float((polynomial - torch.exp(-values)).abs().max())
            metric_bound = epsilon * (2 + epsilon)
            loss_bound = metric_bound * float(active.detach().square().sum()) / 18
            gradient_bound = 2 * metric_bound * float(active.detach().norm()) / 18
            self.assertLessEqual(abs(float((loss - exact).detach())), loss_bound + 2e-13)
            error = float((gradient - exact_gradient).norm())
            self.assertLessEqual(error, gradient_bound + 2e-13)
            self.assertGreaterEqual(float(loss.detach()), 0)
            torch.testing.assert_close(gradient[0, -1], torch.zeros(3, dtype=self.x.dtype), atol=0, rtol=0)
            errors.append(error)
        self.assertLess(errors[1], errors[0] / 100)
        self.assertLess(errors[2], 1e-12)

    def test_empty_support_is_connected_zero_and_batch_sum_uses_original_n(self):
        target = sparse_heat.build_sparse_heat_target(self.x.repeat(2, 1, 1), self.config)
        residual = torch.ones_like(self.x).repeat(2, 1, 1).requires_grad_()
        one = sparse_heat.build_sparse_heat_target(self.x, self.config).prepare(order=32).loss(residual[:1])
        torch.testing.assert_close(target.prepare(order=32).loss(residual), 2 * one)
        isolated = sparse_heat.build_sparse_heat_target(self.x, SpectralConfig(k=2, graph_gamma=100))
        loss = isolated.prepare(order=8).loss(residual[:1])
        gradient, = torch.autograd.grad(loss, residual)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(int(torch.count_nonzero(gradient)), 0)

    def test_measurements_preserve_rng_inputs_and_validate_contract(self):
        residual = torch.ones_like(self.x, requires_grad=True)
        before = torch.get_rng_state().clone()
        report = sparse_heat.measure_sparse_heat(self.x, residual, self.config)
        self.assertEqual([row['order'] for row in report['orders']], [8, 16, 32])
        for row in report['orders']:
            for key in ('coefficient_seconds', 'loss_seconds', 'backward_seconds'):
                self.assertGreaterEqual(row[key], 0)
            self.assertIsNone(row['peak_allocated_bytes'])
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
        self.assertIsNone(residual.grad)
        target = sparse_heat.build_sparse_heat_target(self.x, self.config)
        for order in (True, 0, 9, 8.):
            with self.assertRaises(ValueError):
                target.prepare(order=order)
        with self.assertRaises(ValueError):
            target.prepare(order=8, beta=0)
        with self.assertRaises(ValueError):
            target.prepare(order=8).loss(residual.float())


if __name__ == '__main__':
    unittest.main()
