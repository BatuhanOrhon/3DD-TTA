"""CPU loss/gradient and opt-in protocol contracts; no real model or GPU run."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from legacy_trajectory import rng_hash
from spectral_guidance_losses import SPECTRAL_LOSS_KINDS, spectral_guidance_loss
from scripts import eval_multibackbone_guidance as runner
from tests import test_guidance_components as components
from tests.test_legacy_gsd_numerics import VAE, Prior


def legacy_inline(predicted, reference, eigenvalues, beta, maximum, kind='legacy_l2'):
    weights = torch.exp(-beta * eigenvalues)
    weights[:, maximum:] = 0.0
    weights = weights.unsqueeze(-1)
    residual = (predicted - reference)**2
    weighted = weights * residual
    return weighted.sum() / (3 * predicted.shape[1])


class SpectralLossTests(unittest.TestCase):
    def test_legacy_values_gradients_cutoff_and_rng_are_exact(self):
        torch.manual_seed(31)
        for dtype in (torch.float32, torch.float64):
            for channels in (3, 4):
                predicted = torch.randn(2, 7, channels, dtype=dtype, requires_grad=True)
                target = torch.randn(2, 7, channels, dtype=dtype)
                eigenvalues = torch.rand(2, 7, dtype=dtype)
                for maximum in (0, 3, 7, 11):
                    before = rng_hash()
                    expected = legacy_inline(predicted, target, eigenvalues, 2., maximum)
                    actual = spectral_guidance_loss(predicted, target, eigenvalues, 2., maximum)
                    self.assertTrue(torch.equal(actual, expected))
                    expected_gradient, = torch.autograd.grad(expected, predicted)
                    gradient, = torch.autograd.grad(actual, predicted)
                    self.assertTrue(torch.equal(gradient, expected_gradient))
                    self.assertEqual(rng_hash(), before)

    def test_all_objectives_pass_finite_difference_with_a_masked_mode(self):
        predicted = torch.tensor([[[.2, -.3, .4], [1.5, .1, -.2]]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[[.1, .2, -.1], [-.4, .3, .2]]], dtype=torch.float64, requires_grad=True)
        eigenvalues = torch.tensor([[.1, .7]], dtype=torch.float64)
        for kind in SPECTRAL_LOSS_KINDS:
            with self.subTest(kind=kind):
                self.assertTrue(torch.autograd.gradcheck(lambda p, r: spectral_guidance_loss(p, r, eigenvalues, 2., 1, kind),
                                                        (predicted, target), eps=1e-6, atol=1e-6, rtol=1e-5))
                gradient, = torch.autograd.grad(spectral_guidance_loss(predicted, target, eigenvalues, 2., 1, kind), predicted)
                self.assertTrue(torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1])))

    def test_huber_delta_one_bounds_outlier_gradient(self):
        predicted = torch.tensor([[[20., .2, -.3], [.1, -.2, .3]]], dtype=torch.float64, requires_grad=True)
        target, eigenvalues = torch.zeros_like(predicted), torch.zeros(1, 2, dtype=torch.float64)
        actual = spectral_guidance_loss(predicted, target, eigenvalues, 2., 2, 'huber')
        expected = torch.where(predicted.abs() < 1., .5 * predicted.square(), predicted.abs() - .5).sum() / 6
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient, = torch.autograd.grad(actual, predicted)
        self.assertAlmostEqual(gradient[0, 0, 0].item(), 1. / 6)
        legacy_gradient, = torch.autograd.grad(spectral_guidance_loss(predicted, target, eigenvalues, 2., 2), predicted)
        self.assertLess(abs(gradient[0, 0, 0]), abs(legacy_gradient[0, 0, 0]) / 10)

    def test_power_is_energy_difference_with_xyz_rotation_and_row_sign_invariance(self):
        torch.manual_seed(5)
        predicted = torch.randn(2, 5, 3, dtype=torch.float64)
        target = torch.randn_like(predicted)
        eigenvalues = torch.rand(2, 5, dtype=torch.float64)
        weights = torch.exp(-2. * eigenvalues); weights[:, 3:] = 0.
        expected = (weights * (predicted.square().sum(-1) - target.square().sum(-1)).square()).sum() / 15
        actual = spectral_guidance_loss(predicted, target, eigenvalues, 2., 3, 'power')
        self.assertTrue(torch.equal(actual, expected))
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        rotated = spectral_guidance_loss(predicted @ rotation, target @ rotation, eigenvalues, 2., 3, 'power')
        torch.testing.assert_close(actual, rotated, rtol=1e-12, atol=1e-12)
        signs = torch.tensor([1., -1., 1., -1., 1.], dtype=torch.float64).view(1, 5, 1)
        torch.testing.assert_close(actual, spectral_guidance_loss(predicted * signs, target, eigenvalues, 2., 3, 'power'), rtol=0, atol=0)

    def test_per_mode_power_is_not_a_degenerate_subspace_invariant(self):
        predicted = torch.tensor([[[2., 0., 0.], [0., 0., 0.]]], dtype=torch.float64)
        target = torch.tensor([[[1., 0., 0.], [1., 0., 0.]]], dtype=torch.float64)
        eigenvalues = torch.zeros(1, 2, dtype=torch.float64)
        rotation = torch.tensor([[1., 1.], [-1., 1.]], dtype=torch.float64) / (2. ** .5)
        old = spectral_guidance_loss(predicted, target, eigenvalues, 2., 2, 'power')
        changed = spectral_guidance_loss(rotation @ predicted, rotation @ target, eigenvalues, 2., 2, 'power')
        self.assertNotAlmostEqual(old.item(), changed.item())

    def test_invalid_kind_or_broadcast_shapes_fail(self):
        predicted = torch.zeros(2, 5, 3)
        for target, eigenvalues, kind in ((predicted, torch.zeros(2, 5), 'amplitude'),
                                          (predicted[:1], torch.zeros(2, 5), 'legacy_l2'),
                                          (predicted, torch.zeros(1, 5), 'power')):
            with self.assertRaises(ValueError):
                spectral_guidance_loss(predicted, target, eigenvalues, 2., 3, kind)

    def test_default_reconstruction_matches_old_inline_loss_and_state_rng(self):
        module = components.ComponentTests().reconstruction()
        basis = torch.eye(2048).unsqueeze(0)
        eigenvalues = torch.linspace(0, 4, 2048).unsqueeze(0)
        class Graph:
            use_4d_gft = False
            def __call__(self, local):
                return local[:, :, :3].detach(), basis, eigenvalues
        results = []
        for objective in (legacy_inline, spectral_guidance_loss):
            vae, prior = VAE(), Prior()
            torch.manual_seed(13)
            with patch.object(module, 'spectral_guidance_loss', objective):
                points, metrics = module.tta_gsd_reconstruct(torch.zeros(1, 2048, 3), types.SimpleNamespace(vae=vae, priors=[None, prior]),
                    Graph(), 3, .03, .02, .95, loss_weights=dict(spectral=1.17, chamfer=1.), M_max=800)
            results.append((points.detach(), vae.final_style, metrics, rng_hash()))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertEqual(results[0][2:], results[1][2:])

    def test_logging_and_applied_loss_use_the_selected_objective(self):
        module = components.ComponentTests().reconstruction()
        basis = torch.eye(2048).unsqueeze(0)
        eigenvalues = torch.linspace(0, 4, 2048).unsqueeze(0)
        class Graph:
            use_4d_gft = False
            def __call__(self, local):
                return local[:, :, :3].detach(), basis, eigenvalues
        for kind in SPECTRAL_LOSS_KINDS:
            calls = []
            def record(*args):
                value = spectral_guidance_loss(*args)
                calls.append((args[-1], torch.is_grad_enabled(), value.detach().item()))
                return value
            with patch.object(module, 'spectral_guidance_loss', record):
                _, metrics = module.tta_gsd_reconstruct(torch.zeros(1, 2048, 3), types.SimpleNamespace(vae=VAE(), priors=[None, Prior()]),
                    Graph(), 1, .01, .01, .95, loss_weights=dict(spectral=1.17, chamfer=0.), M_max=800, spectral_loss_kind=kind)
            self.assertEqual([value[:2] for value in calls], [(kind, False), (kind, True)])
            self.assertEqual(calls[0][2], calls[1][2])
            self.assertEqual(metrics['mean_raw_spectral'], calls[0][2])


class SpectralRunnerTests(unittest.TestCase):
    def args(self, dataset='modelnet40_c', *extra):
        return ['--preview', '--dataset', dataset, '--dataset-root', '/data', '--label-path', '/data/label.npy',
                '--asset-root', '/assets', '--result-root', '/results', *extra]

    def test_loss_kind_is_opt_in_spectral_only_and_preserves_default_dicts(self):
        protocol_before = json.dumps(runner.PROTOCOL, sort_keys=True)
        default = runner.configurations(self.args())[0]
        self.assertNotIn('spectral_loss_kind', default)
        self.assertNotIn('spectral_loss_kind', default['guidance_protocol'])
        for arm in ('combined', 'spectral'):
            baseline = runner.configurations(self.args('modelnet40_c', '--trajectory-observe', '--component-arm', arm))[0]
            self.assertFalse(baseline['component_parameters_modified'])
            self.assertNotIn('spectral_loss_kind', baseline['guidance_protocol'])
            for kind in SPECTRAL_LOSS_KINDS:
                config = runner.configurations(self.args('modelnet40_c', '--trajectory-observe', '--component-arm', arm, '--spectral-loss-kind', kind))[0]
                self.assertEqual(config['spectral_loss_kind'], kind)
                self.assertEqual(config['guidance_protocol']['spectral_loss_kind'], kind)
                self.assertTrue(config['component_parameters_modified'])
        for extra in (('--spectral-loss-kind', 'legacy_l2'),
                      ('--trajectory-observe', '--component-arm', 'chamfer', '--spectral-loss-kind', 'huber'),
                      ('--trajectory-observe', '--component-arm', 'unguided', '--spectral-loss-kind', 'power'),
                      ('--method', 'original3dd', '--spectral-loss-kind', 'legacy_l2')):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                runner.configurations(self.args('modelnet40_c', *extra))
        self.assertEqual(json.dumps(runner.PROTOCOL, sort_keys=True), protocol_before)

    def test_original_pointmae_requires_paths_and_retains_actual_original_contract(self):
        for dataset in ('modelnet40_c', 'scanobjectnn_c'):
            paths = ('--pointmae-config', 'source_snapshot/cfgs/tta.yaml', '--pointmae-checkpoint', 'checkpoints/pointmae.pth')
            prefix = ('--method', 'original3dd', '--backbones', 'pointmae')
            config, = runner.configurations(self.args(dataset, *prefix, *paths))
            self.assertEqual(config['method'], runner.ORIGINAL_METHOD)
            self.assertEqual(config['batch_size'], 32)
            self.assertEqual((config['normal_reverse_steps'], config['background_reverse_steps']), (5, 35))
            self.assertEqual(config['decoder_style'], 'original')
            self.assertTrue(config['scheduler_final_alpha_to_one'])
            self.assertFalse(config.get('trajectory_observation'))
            self.assertIsNone(config['graph'])
            self.assertEqual(config['guidance_protocol'], runner.ORIGINAL_PROTOCOL)
            self.assertEqual(runner.matched_source(config, {}, {})['status'], 'unavailable')
            for extra in (prefix, (*prefix, '--pointmae-config', 'source_snapshot/cfgs/tta.yaml'),
                          (*prefix, *paths, '--trajectory-observe'), (*prefix, *paths, '--batch-size', '70')):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    runner.configurations(self.args(dataset, *extra))
            with self.assertRaises(ValueError):
                runner.configurations(self.args(dataset, '--backbones', 'pointmae', *paths))

    def test_executed_source_snapshot_binds_the_loss_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            assets = root / 'assets'; assets.mkdir()
            bundle = types.SimpleNamespace(path=root / 'run')
            config = dict(asset_root=str(assets))
            runner.source.snapshot_sources(bundle, config, {})
            helper = runner.source.REPO / 'spectral_guidance_losses.py'
            self.assertEqual(config['executed_source_files']['repo/spectral_guidance_losses.py'], runner.source.file_identity(helper))
            self.assertEqual((bundle.path / 'source_snapshot/spectral_guidance_losses.py').read_bytes(), helper.read_bytes())


if __name__ == '__main__':
    unittest.main()
