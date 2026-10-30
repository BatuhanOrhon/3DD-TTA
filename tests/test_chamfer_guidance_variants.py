"""CPU mathematical contracts; no CUDA kernels or pretrained models execute."""
import inspect
import types
import unittest
from unittest.mock import patch

import torch

from legacy_trajectory import rng_hash
from tests.test_legacy_gsd_numerics import load_source, chamfer, Scheduler, VAE, Prior


class Observer:
    def __init__(self, loss=None):
        self.loss = loss
        self.seen = []

    def mark(self, name):
        pass

    def encode_rng_start(self):
        return None

    def initial(self, **values):
        self.timesteps = values['timesteps'].clone()

    def clean(self, latent, style, index, timestep):
        self.seen.append((index + 1, torch.is_grad_enabled(), latent.requires_grad))
        return None if self.loss is None else self.loss(latent, style, index)

    def final_state(self, *values):
        pass


class ChamferVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def sampler(self):
        native = types.ModuleType('third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D')
        native.chamfer_3DDist = lambda: chamfer
        ops = types.ModuleType('utilities_3dd_tta')
        ops.grad_freeze = lambda model: model.requires_grad_(False)
        return load_source('tta_gsd_v2', {native.__name__: native,
                           'diffusers': types.SimpleNamespace(DDIMScheduler=Scheduler),
                           'utilities_3dd_tta': ops})

    def run_sampler(self, experiment=None, steps=10, observer=None, chamfer_impl=None, graph=None, **overrides):
        module = self.sampler()
        if chamfer_impl is not None:
            module.chamfer_grad = lambda: chamfer_impl
        class RecordingPrior(Prior):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, x, t, condition_input, clip_feat):
                self.calls.append((t.detach().clone(), torch.is_grad_enabled()))
                return super().forward(x, t, condition_input, clip_feat)
        vae, prior = VAE(), RecordingPrior()
        kwargs = dict(loss_weights={'chamfer': 1., 'spectral': 0., 'invariant': 0.},
                      skip_inactive_components=True, trajectory_observer=observer)
        if experiment is not None:
            kwargs['chamfer_experiment'] = experiment
        kwargs.update(overrides)
        torch.manual_seed(73)
        with patch.object(module, 'spectral_guidance_loss', side_effect=AssertionError('spectral called')):
            points, metrics = module.tta_gsd_reconstruct(
                torch.zeros(1, 2048, 3), types.SimpleNamespace(vae=vae, priors=[None, prior]),
                graph, steps, .01, .01, .95, **kwargs)
        return points.detach(), vae.final_style, metrics, prior.calls, rng_hash()

    def test_sampler_exposes_explicit_opt_in(self):
        self.assertIn('chamfer_experiment', inspect.signature(self.sampler().tta_gsd_reconstruct).parameters)

    def test_interval_one_preserves_default_c_points_style_and_rng(self):
        baseline = self.run_sampler()
        candidate = self.run_sampler({'variant': 'sparse', 'interval': 1})
        torch.testing.assert_close(candidate[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(candidate[1], baseline[1], rtol=0, atol=0)
        self.assertEqual(candidate[-1], baseline[-1])
        self.assertEqual(baseline[2]['spectral_loss_calls'], 0)
        self.assertEqual(candidate[2]['graph_calls'], 0)
        self.assertEqual(candidate[2]['spectral_loss_calls'], 0)

    def test_active_spectral_metrics_count_actual_loss_calls(self):
        module = self.sampler()
        def spectral(*args):
            return args[0].square().sum()
        class Graph:
            use_4d_gft = True
            M = 1200

            def __call__(self, h):
                basis = torch.eye(h.shape[1]).unsqueeze(0)
                return h, basis, torch.ones(h.shape[1])
        with patch.object(module, 'spectral_guidance_loss', side_effect=spectral) as loss:
            _, metrics = module.tta_gsd_reconstruct(
                torch.zeros(1, 2048, 3), types.SimpleNamespace(vae=VAE(), priors=[None, Prior()]),
                Graph(), 2, .01, .01, .95,
                loss_weights={'chamfer': 0., 'spectral': 1., 'invariant': 0.},
                skip_inactive_components=True)
        self.assertEqual(loss.call_count, 4)
        self.assertEqual(metrics['spectral_loss_calls'], 4)

    def test_sparse_uses_same_timesteps_with_four_or_eleven_backwards(self):
        for steps, expected in ((10, [1, 4, 7, 10]), (30, list(range(1, 31, 3)) + [30])):
            observer = Observer()
            _, _, metrics, calls, _ = self.run_sampler({'variant': 'sparse', 'interval': 3}, steps, observer)
            self.assertEqual(len(calls), steps)
            self.assertEqual([int(t[0]) for t, _ in calls], list(range(steps, 0, -1)))
            self.assertEqual([i + 1 for i, (_, grad) in enumerate(calls) if grad], expected)
            self.assertEqual(metrics['guidance_backward_calls'], len(expected))
            self.assertEqual(metrics['prior_forward_calls'], steps)
            self.assertEqual(metrics['chamfer_calls'], 2 * len(expected))
            self.assertEqual(metrics['graph_calls'], 0)

    def test_semantic_zero_weight_preserves_baseline_and_active_loss_changes_update(self):
        semantic = lambda h, s, i: h.square().sum() + s.square().sum() if i == 4 else None
        baseline = self.run_sampler()
        zero = self.run_sampler({'variant': 'semantic', 'semantic_weight': 0.}, observer=Observer(semantic))
        changed = self.run_sampler({'variant': 'semantic', 'semantic_weight': .1}, observer=Observer(semantic))
        torch.testing.assert_close(zero[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(zero[1], baseline[1], rtol=0, atol=0)
        self.assertEqual(zero[-1], baseline[-1])
        self.assertFalse(torch.equal(changed[0], baseline[0]))
        self.assertFalse(torch.equal(changed[1], baseline[1]))
        self.assertEqual(changed[2]['semantic_loss_calls'], 1)
        self.assertEqual(changed[2]['semantic_callback_calls'], 10)
        self.assertEqual(changed[2]['guidance_backward_calls'], 10)

    def test_semantic_rejects_invalid_callback_losses(self):
        for callback in (lambda h, s, i: h.detach().square().sum(),
                         lambda h, s, i: h[:, 0],
                         lambda h, s, i: h.sum() * float('nan'),
                         lambda h, s, i: 2.):
            with self.subTest(callback=callback), self.assertRaises(ValueError):
                self.run_sampler({'variant': 'semantic', 'semantic_weight': .1}, observer=Observer(callback))

    def test_experiment_requires_inactive_spectral_graph_and_single_variant(self):
        invalid = [({}, {}), ({'variant': 'bad'}, {}), ({'variant': []}, {}),
                   ({'variant': 'sparse', 'interval': 0}, {}),
                   ({'variant': 'sparse', 'interval': True}, {}),
                   ({'variant': 'sparse', 'semantic_weight': 1}, {}),
                   ({'variant': 'semantic'}, {}),
                   ({'variant': 'semantic', 'semantic_weight': float('inf')}, {}),
                   ({'variant': 'weighted', 'density_k': 0}, {}),
                   ({'variant': 'sparse'}, {'skip_inactive_components': False}),
                   ({'variant': 'sparse'}, {'loss_weights': {'chamfer': 1., 'spectral': 1.}}),
                   ({'variant': 'sparse'}, {'dynamic_graph': True})]
        for experiment, overrides in invalid:
            with self.subTest(experiment=experiment, overrides=overrides), self.assertRaises(ValueError):
                self.run_sampler(experiment, **overrides)

    def test_experiment_rejects_graph_before_any_graph_calculation(self):
        graph = unittest.mock.Mock(side_effect=AssertionError('graph called'))
        with self.assertRaises(ValueError):
            self.run_sampler({'variant': 'sparse'}, graph=graph)
        graph.assert_not_called()

    def test_weighted_sampler_uses_one_static_density_without_spectral(self):
        def indexed_chamfer(left, right):
            first, second, _, _ = chamfer(left, right)
            identity = torch.arange(left.shape[1], dtype=torch.int32).expand(left.shape[0], -1)
            return first, second, identity, identity
        points, style, metrics, calls, _ = self.run_sampler(
            {'variant': 'weighted', 'density_k': 10}, steps=2, chamfer_impl=indexed_chamfer)
        self.assertTrue(torch.isfinite(points).all())
        self.assertTrue(torch.isfinite(style).all())
        self.assertEqual(metrics['reference_density_calls'], 1)
        self.assertEqual(metrics['guidance_backward_calls'], 2)
        self.assertEqual(metrics['chamfer_calls'], 4)
        self.assertEqual(metrics['spectral_loss_calls'], 0)
        self.assertEqual(metrics['graph_calls'], 0)
        self.assertEqual(len(calls), 2)

    def test_reference_density_weights_are_static_finite_and_permutation_equivariant(self):
        from chamfer_guidance_variants import reference_density_weights
        points = torch.tensor([[[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.], [5., 0., 0.]]], requires_grad=True)
        weights = reference_density_weights(points, 2)
        self.assertFalse(weights.requires_grad)
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(weights.mean(-1), torch.ones(1))
        self.assertLess(weights[0, -1], weights[0, 1])
        order = torch.tensor([3, 1, 0, 2])
        torch.testing.assert_close(reference_density_weights(points[:, order], 2), weights[:, order])
        duplicate = reference_density_weights(torch.zeros(1, 4, 3), 2)
        torch.testing.assert_close(duplicate, torch.ones_like(duplicate))

    def test_weighted_chamfer_routes_reference_weights_before_distance_trim(self):
        from chamfer_guidance_variants import weighted_selective_chamfer
        pred = torch.tensor([[4., 1., 9.]], requires_grad=True)
        ref = torch.tensor([[9., 4., 1.]], requires_grad=True)
        nearest = torch.tensor([[2, 0, 1]], dtype=torch.int32)
        weights = torch.tensor([[.5, 1., 1.5]])
        value = weighted_selective_chamfer(pred, ref, nearest, weights, 2 / 3)
        # Pred keeps4,1 with weights1.5,.5; reference keeps4,1 with weights1,1.5.
        torch.testing.assert_close(value, torch.tensor(12.))
        value.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[1.5, .5, 0.]]))
        torch.testing.assert_close(ref.grad, torch.tensor([[0., 1., 1.5]]))
        ordinary = weighted_selective_chamfer(pred, ref, nearest, torch.ones_like(weights), 2 / 3)
        torch.testing.assert_close(ordinary, pred.sort().values[:, :2].sum() + ref.sort().values[:, :2].sum())
        ref_order = torch.tensor([2, 0, 1])
        inverse = torch.argsort(ref_order)
        permuted = weighted_selective_chamfer(
            pred, ref[:, ref_order], inverse[nearest.long()], weights[:, ref_order], 2 / 3)
        torch.testing.assert_close(permuted, value)


if __name__ == '__main__':
    unittest.main()
