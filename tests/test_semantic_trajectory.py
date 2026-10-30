"""CPU contracts for opt-in Chamfer-only semantic probes."""
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from legacy_trajectory import rng_hash
from scripts import eval_multibackbone_guidance as runner
from shared_trajectory import SharedTrajectoryObserver


class SemanticContracts(unittest.TestCase):
    def args(self, *extra):
        return ['--dataset', 'modelnet40_c', '--dataset-root', '/data', '--label-path', '/data/label.npy',
                '--asset-root', '/assets', '--result-root', '/results', '--preview', *extra]

    def test_opt_in_configuration_is_chamfer_only_and_defaults_are_unchanged(self):
        self.assertTrue(hasattr(runner, 'experiment_main'))
        original = runner.configurations(self.args())[0]
        self.assertNotIn('chamfer_experiment', original)
        sparse = runner.configurations(self.args('--chamfer-experiment', 'sparse'))[0]
        self.assertEqual(sparse['guidance_protocol']['chamfer_experiment'], {'variant': 'sparse', 'interval': 3})
        self.assertEqual(sparse['guidance_protocol']['loss_weights'], dict(chamfer=1., spectral=0., invariant=0.))
        self.assertIsNone(sparse['graph'])
        self.assertTrue(sparse['trajectory_observation'])
        for extra in (['--component-arm', 'spectral'], ['--spectral-weight', '1'],
                      ['--semantic-kind', 'kl'], ['--sparse-interval', '0'],
                      ['--chamfer-experiment', 'weighted']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                runner.configurations(self.args('--chamfer-experiment', 'sparse', *extra))

    def test_semantic_requires_pointmae_and_explicit_execution_weight(self):
        self.assertTrue(hasattr(runner, 'experiment_main'))
        with self.assertRaises(ValueError):
            runner.configurations(self.args('--chamfer-experiment', 'semantic'))
        args = self.args('--chamfer-experiment', 'semantic', '--backbones', 'pointmae',
                         '--pointmae-config', '/p.yaml', '--pointmae-checkpoint', '/p.pth')
        for kind in ('kl', 'marginal_entropy'):
            config = runner.configurations([*args, '--semantic-kind', kind, '--semantic-weight', '0'])[0]
            self.assertEqual(config['semantic_guidance']['kind'], kind)
            self.assertEqual(config['guidance_protocol']['chamfer_experiment']['semantic_weight'], 0.)
        with self.assertRaises(ValueError):
            runner.configurations([arg for arg in args if arg != '--preview'])

    def test_actual_semantic_run_names_pass_bundle_validation_for_frozen_weights(self):
        from unittest.mock import patch
        class NameAccepted(Exception):
            pass
        create = runner.source.RunBundle.create
        with tempfile.TemporaryDirectory() as directory:
            names = []
            def checked_create(root, name, config, command):
                create(root, name, config, command)
                names.append(name)
                raise NameAccepted()
            for kind in ('kl', 'marginal_entropy'):
                for weight in (0., .1, 1e-9, 1e20):
                    config = runner.configurations(self.args('--chamfer-experiment', 'semantic',
                        '--backbones', 'pointmae', '--pointmae-config', '/p.yaml',
                        '--pointmae-checkpoint', '/p.pth', '--semantic-kind', kind,
                        '--semantic-weight', str(weight)))[0]
                    config.update(preview=False, result_root=directory)
                    with self.subTest(kind=kind, weight=weight), \
                            patch.object(runner, 'configurations', return_value=[config]), \
                            patch.object(runner.source.RunBundle, 'create', side_effect=checked_create), \
                            self.assertRaises(NameAccepted):
                        runner.main([])
            self.assertEqual(len(set(names)), 8)

    def test_calibration_report_binds_coefficients_and_weighted_update_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'calibration.json'
            report.write_text(json.dumps(self.calibration_report()))
            identity = runner.source.file_identity(report)
            args = self.args('--chamfer-experiment', 'weighted', '--calibration-report', str(report),
                             '--calibration-report-sha256', identity['sha256'])
            config = runner.configurations(args)[0]
            json.dumps(config)
            self.assertEqual(config['guidance_protocol']['eta'], .02)
            self.assertEqual(config['guidance_protocol']['gamma'], .02)
            self.assertEqual(config['calibration_reference']['resolved_scalar'], 2.)
            with self.assertRaises(ValueError):
                runner.configurations([*args, '--eta', '.01'])
            with self.assertRaises(ValueError):
                runner.configurations([*args, '--calibration-report-sha256', '0' * 64])

    def calibration_report(self):
        medians = dict(semantic_kl=.5, semantic_h=1. / 3., weighted=.5)
        return dict(schema_version=1, purpose='label_free_chamfer_calibration',
            status='verified_label_free_gpu', target_semantic_update_ratio=.1,
            loss_reduction='sum for Chamfer, weighted Chamfer, and per-object semantic losses',
            rule='semantic=0.1/median(candidate/C combined update norm); weighted_scale=1/median(weighted/C)',
            coefficients=dict(semantic_kl=.2, semantic_h=.3, weighted_scale=2.),
            ratios={name: dict(count=4200, missing=0, median=median, min=median, max=median, p90=median)
                    for name, median in medians.items()},
            calibration_bundles={name: {'config_identity': {'bytes': 1, 'sha256': 'a' * 64},
                'executed_source_files': {'a': {'bytes': 1, 'sha256': 'a' * 64}},
                'native_extensions': {'b': {'bytes': 1, 'sha256': 'a' * 64}}, 'asset_manifest_sha256': 'b' * 64}
                for name in ('modelnet40_c', 'scanobjectnn_c')})

    def test_runtime_rejects_rehashed_but_invalid_calibration_math_before_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'calibration.json'
            changes = [('rule', None, 'other formula'), ('loss_reduction', None, 'mean'),
                       ('coefficients', 'semantic_kl', .7), ('coefficients', 'semantic_h', float('nan')),
                       ('ratios', 'weighted', dict(count=4199, missing=1, median=.5, min=.5, max=.5, p90=.5)),
                       ('ratios', 'semantic_kl', dict(count=4200, missing=0, median=.7, min=.7, max=.7, p90=.7))]
            for field, key, value in changes:
                data = copy.deepcopy(self.calibration_report())
                if key is None:
                    data[field] = value
                else:
                    data[field][key] = value
                report.write_text(json.dumps(data))
                with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                    runner.fixed_calibration_reference(report, runner.source.file_identity(report)['sha256'], 'weighted_scale')

    def observer_class(self):
        self.assertIsNotNone(importlib.util.find_spec('semantic_trajectory'))
        from semantic_trajectory import SemanticTrajectoryObserver
        return SemanticTrajectoryObserver

    def fixture(self, classes):
        class VAE(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.calls = 0
            def decoder(self, unused, beta, context, style):
                self.calls += 1
                torch.rand(2)
                return context.reshape(-1, 4, 3) + style[:, :3].unsqueeze(1)
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, classes).requires_grad_(False)
                self.calls = 0
            def forward(self, points, *, features=None):
                self.calls += 1
                torch.rand(1)
                return self.linear(points.square().mean(1))
        vae, head = VAE().eval(), Head().eval()
        counts = {'plain': 0, 'semantic': 0}
        def plain(decoded):
            counts['plain'] += 1
            return {'pointmae': head(decoded[:, :2])}
        def semantic(decoded):
            counts['semantic'] += 1
            return {'pointmae': head(decoded[:, :2])}
        latent = torch.linspace(-1, 2, 24).reshape(2, 12, 1, 1).requires_grad_()
        style = torch.tensor([[.1, -.2, .3], [.2, .3, -.1]]).reshape(2, 3, 1, 1).requires_grad_()
        return vae, head, counts, plain, semantic, latent, style

    def reference(self, classes):
        return torch.linspace(-1., 1., classes).repeat(2, 1)

    def test_losses_reuse_decode_logits_and_differentiate_both_latents_without_weight_gradients(self):
        Observer = self.observer_class()
        for classes in (15, 40):
            for kind in ('kl', 'marginal_entropy'):
                vae, head, calls, plain, semantic, latent, style = self.fixture(classes)
                observer = Observer(vae, {'pointmae': head}, plain, 10, classes,
                                    classify_semantic_decoded=semantic, kind=kind,
                                    reference_logits=self.reference(classes),
                                    reference_record={'geometry_sha256': 'fixed-X', 'forward_calls': 1})
                before = rng_hash()
                self.assertIsNone(observer.clean(latent, style, 2, 930))
                loss = observer.clean(latent, style, 4, 910)
                self.assertEqual(rng_hash(), before)
                self.assertEqual((vae.calls, head.calls), (2, 2))
                self.assertEqual(calls, {'plain': 1, 'semantic': 1})
                local_grad, style_grad = torch.autograd.grad(loss, (latent, style))
                self.assertTrue(torch.isfinite(loss))
                for grad in (local_grad, style_grad):
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(float(grad.abs().sum()), 0.)
                self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_zero_weight_sham_preserves_candidate_logits_and_rng(self):
        Observer = self.observer_class()
        vae, head, calls, plain, semantic, latent, style = self.fixture(40)
        baseline = SharedTrajectoryObserver(vae, {'pointmae': head}, plain, 10, 40)
        observer = Observer(vae, {'pointmae': head}, plain, 10, 40,
                            classify_semantic_decoded=semantic, kind='kl', reference_logits=self.reference(40),
                            reference_record={'geometry_sha256': 'fixed-X', 'forward_calls': 1})
        before = rng_hash()
        for current in (baseline, observer):
            current.initial(local=latent, style=style)
            for count in (3, 5, 8):
                current.clean(latent, style, count - 1, 990 - count)
            current.final_state(latent, style, 0)
            current.terminal({'pointmae': torch.zeros(2, 40)})
        self.assertEqual(before, rng_hash())
        np.testing.assert_array_equal(baseline.arrays('pointmae')['trajectory_logits'],
                                      observer.arrays('pointmae')['trajectory_logits'])
        self.assertEqual(observer.diagnostics()['operation_counts']['probe_decode_forward'], 3)
        self.assertEqual(observer.diagnostics()['operation_counts']['semantic_reference_forward'], 1)
        self.assertEqual(observer.diagnostics()['operation_counts']['probe_classifier_forward'], 3)
        self.assertEqual(observer.diagnostics()['semantic_guidance']['reference']['geometry_sha256'], 'fixed-X')

    def test_real_preprocess_path_preserves_baseline_view_and_restores_rng(self):
        self.observer_class()
        vae, head, _, _, _, _, _ = self.fixture(40)
        geometry = torch.linspace(-1, 2, 2 * 2048 * 3).reshape(2, 2048, 3).requires_grad_()
        configs = {'pointmae': {'backbone': 'pointmae', 'dataset': 'modelnet40_c'}}
        calls = []
        def fps(points, count):
            calls.append(points[:, 0].detach().clone())
            torch.rand(1)
            return points[:, :count]
        before = rng_hash()
        expected, _ = runner.classify_geometry(geometry, {'pointmae': head}, configs, fps)
        reference = self.reference(40)
        baseline = runner.classify_semantic_geometry(geometry, {'pointmae': head}, configs, fps)
        self.assertEqual(before, rng_hash())
        torch.testing.assert_close(expected['pointmae'], baseline['pointmae'], rtol=0, atol=0)
        torch.testing.assert_close(calls[0], calls[1], rtol=0, atol=0)
        self.assertEqual(len(calls), 2)
        from semantic_trajectory import semantic_loss
        grad, = torch.autograd.grad(semantic_loss(baseline['pointmae'], reference, 'kl'), (geometry,))
        self.assertGreater(float(grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_gradient_guard_restores_rng_on_failure_and_rejects_trainable_weights(self):
        self.observer_class()
        from semantic_trajectory import gradient_probe
        model = torch.nn.Linear(3, 2).eval().requires_grad_(False)
        before = rng_hash()
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with gradient_probe([model]):
                torch.rand(1)
                raise RuntimeError('failure')
        self.assertEqual(before, rng_hash())
        model.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            with gradient_probe([model]):
                pass

    def test_corrupted_x_reference_is_one_detached_rng_isolated_forward(self):
        self.assertTrue(hasattr(runner, 'semantic_reference'))
        _, head, _, _, _, _, _ = self.fixture(40)
        geometry = torch.linspace(-1, 2, 2 * 1024 * 3).reshape(2, 1024, 3).requires_grad_()
        config = {'pointmae': {'backbone': 'pointmae', 'dataset': 'modelnet40_c'}}
        before = rng_hash()
        reference, record = runner.semantic_reference(geometry, {'pointmae': head}, config,
                                                      lambda points, count: points[:, :count])
        self.assertEqual(head.calls, 1)
        self.assertFalse(reference.requires_grad)
        self.assertEqual(before, rng_hash())
        self.assertEqual(record['forward_calls'], 1)
        self.assertEqual(len(record['geometry_sha256']), 64)
        self.assertTrue(record['rng_guard']['rng_unchanged'])

    def test_background_semantic_checkpoints_and_reference_gradient_contract(self):
        Observer = self.observer_class()
        vae, head, calls, plain, semantic, latent, style = self.fixture(15)
        observer = Observer(vae, {'pointmae': head}, plain, 30, 15,
                            classify_semantic_decoded=semantic, kind='marginal_entropy',
                            reference_logits=self.reference(15), reference_record={'forward_calls': 1})
        losses = [observer.clean(latent, style, count - 1, 990 - count) for count in (8, 15, 23)]
        self.assertIsNone(losses[0])
        self.assertTrue(all(loss.requires_grad for loss in losses[1:]))
        self.assertEqual([record['loop_count'] for record in observer.semantic_records], [15, 23])
        from semantic_trajectory import semantic_loss
        with self.assertRaisesRegex(ValueError, 'detached'):
            semantic_loss(torch.zeros(2, 15), torch.zeros(2, 15, requires_grad=True), 'kl')

    def test_semantic_sum_reduction_preserves_per_object_gradient_when_batch_is_duplicated(self):
        self.observer_class()
        from semantic_trajectory import semantic_loss
        for kind in ('kl', 'marginal_entropy'):
            logits = torch.linspace(-1., 2., 30).reshape(2, 15).requires_grad_()
            reference = torch.linspace(1., -1., 30).reshape(2, 15)
            loss = semantic_loss(logits, reference, kind)
            grad, = torch.autograd.grad(loss, (logits,))
            duplicated = torch.cat([logits.detach(), logits.detach()]).requires_grad_()
            double_loss = semantic_loss(duplicated, torch.cat([reference, reference]), kind)
            double_grad, = torch.autograd.grad(double_loss, (duplicated,))
            torch.testing.assert_close(double_loss, 2 * loss)
            torch.testing.assert_close(double_grad[:2], grad)
            torch.testing.assert_close(double_grad[2:], grad)


if __name__ == '__main__':
    unittest.main()
