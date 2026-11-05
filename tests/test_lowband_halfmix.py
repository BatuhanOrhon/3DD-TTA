"""CPU checks for checkpoint-only mixing and label-free update calibration."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_chamfer_depth import CPU, Scheduler, VAE, Prior, chamfer, load_sampler, seeded
from tests.test_decoded_spectral import target_for
from decoded_spectral import DecodedSpectralObserver


class HalfmixTests(unittest.TestCase):
    def test_half_scd_changes_only_pm_backward_with_one_joint_backward(self):
        module = load_sampler()
        reference = torch.ones(1, 2048, 3)
        lion = SimpleNamespace(vae=VAE().eval(), priors=[None, Prior().eval()])
        head = torch.nn.Linear(3, 40).eval().requires_grad_(False)
        observer = DecodedSpectralObserver(lion.vae, {'pointmae': head},
            lambda x: {'pointmae': head(x.mean(1))}, 8,
            target=target_for(reference), checkpoint_steps=[2, 4, 6])
        backward_losses, original = [], torch.Tensor.backward
        def backward(loss, *args, **kwargs):
            backward_losses.append(float(loss.detach()))
            return original(loss, *args, **kwargs)
        scd_losses = []
        def distance(a, b):
            result = chamfer(a, b)
            keep = int(2048*.95)
            scd_losses.append(float(sum(d.sort(dim=1).values[:, :keep].sum() for d in result[:2]).detach()))
            return result
        seeded()
        with patch.multiple(module, torch=CPU(), DDIMScheduler=Scheduler,
                            chamfer_grad=lambda: distance, grad_freeze=lambda m: None), \
                patch.object(torch.Tensor, 'backward', backward):
            result = module.tta_reconstruct(reference, lion, 8, .03, .003, .95,
                trajectory_observer=observer, decoded_scd_weight=.5, return_trajectory=True)
        self.assertEqual(len(backward_losses), 8)
        self.assertEqual(len(observer.probes), 3)
        self.assertEqual(len(lion.vae.styles), 3)
        for count, (loss, scd) in enumerate(zip(backward_losses, scd_losses), 1):
            self.assertAlmostEqual(loss, scd*(.5 if count in (2,4,6) else 1), places=4)
        self.assertTrue(torch.isfinite(result.final_local_latent).all())

    def test_combined_mix_probes_preserve_updates_for_normal_and_background(self):
        module = load_sampler()
        reference = torch.ones(1, 2048, 3)
        for horizon, checkpoints in ((8, [2,4,6]), (35, [9,18,27])):
            outputs = []
            for probes in (False, True):
                lion = SimpleNamespace(vae=VAE().eval(), priors=[None, Prior().eval()])
                head = torch.nn.Linear(3, 40).eval().requires_grad_(False)
                observer = DecodedSpectralObserver(lion.vae, {'pointmae': head},
                    lambda x: {'pointmae': head(x.mean(1))}, horizon,
                    target=target_for(reference), checkpoint_steps=checkpoints, record_updates=probes)
                backward_losses, scd_losses = [], []
                original = torch.Tensor.backward
                def backward(loss, *args, **kwargs):
                    backward_losses.append(float(loss.detach()))
                    return original(loss, *args, **kwargs)
                def distance(a, b):
                    result = chamfer(a, b)
                    keep = int(2048*.95)
                    scd_losses.append(float(sum(d.sort(dim=1).values[:, :keep].sum() for d in result[:2]).detach()))
                    return result
                seeded()
                with patch.multiple(module, torch=CPU(), DDIMScheduler=Scheduler,
                                    chamfer_grad=lambda: distance, grad_freeze=lambda m: None), \
                        patch.object(torch.Tensor, 'backward', backward):
                    outputs.append(module.tta_reconstruct(reference, lion, horizon, .03, .003, .95,
                        trajectory_observer=observer, decoded_scd_weight=.5,
                        decoded_spectral_weight=7., return_trajectory=True))
                self.assertEqual(len(backward_losses), horizon)
                self.assertEqual(len(lion.vae.styles), 3)
                records = {r['loop_count']: r for r in observer.spectral_records}
                for count, (loss, scd) in enumerate(zip(backward_losses, scd_losses), 1):
                    expected = .5*scd+7*records[count]['raw_loss'] if count in checkpoints else scd
                    self.assertAlmostEqual(loss, expected, delta=1e-3)
                self.assertEqual(sum('update_norms' in r for r in records.values()), 3 if probes else 0)
            torch.testing.assert_close(outputs[0].final_local_latent, outputs[1].final_local_latent, rtol=0, atol=0)
            torch.testing.assert_close(outputs[0].updated_style, outputs[1].updated_style, rtol=0, atol=0)

    def test_retained_hard_basis_reconstructs_every_filter_and_gradient(self):
        from graph_spectral import SpectralConfig, build_smooth_spectral_target
        reference = torch.tensor([[[0.,0.,0.], [1.,0.,0.], [.5,.8660254037844386,0.], [1e6,0.,0.]]], dtype=torch.float64)
        target = build_smooth_spectral_target(reference, SpectralConfig(k=2,delta=1.,graph_gamma=0.,modes=2),
                                              profile='hard', retain_bases=True)
        basis = target.bases[0]
        self.assertEqual(basis.shape, (4,3))  # complete repeated boundary retained
        self.assertTrue(torch.equal(basis[-1], torch.zeros(3, dtype=basis.dtype)))
        prediction = (reference+.1).requires_grad_()
        expected = (basis.T@(prediction[0]-reference[0])).square().sum()/12
        torch.testing.assert_close(target.loss(prediction), expected)
        actual_grad, = torch.autograd.grad(target.loss(prediction), prediction, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, prediction)
        torch.testing.assert_close(actual_grad, expected_grad)
        torch.testing.assert_close(target.filters[0], basis@basis.T, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            build_smooth_spectral_target(reference, SpectralConfig(k=2), profile='smooth', beta=2., retain_bases=True)

    def test_weights_rejected_before_setup_and_half_requires_live_observer(self):
        module = load_sampler()
        for weight in (-1, float('nan'), float('inf'), True, torch.tensor(.5)):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, 'decoded_scd_weight'):
                module.tta_reconstruct(None, None, 8, .03, .003, .95, decoded_scd_weight=weight)
        with self.assertRaisesRegex(ValueError, 'observer'):
            module.tta_reconstruct(None, None, 8, .03, .003, .95, decoded_scd_weight=.5)

    def test_update_probes_use_actual_local_style_steps_and_no_grad_accumulation(self):
        from decoded_spectral import update_norm_diagnostics, freeze_update_alpha
        local = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        style = torch.tensor([[2.], [1.]], requires_grad=True)
        scd = (local.square().sum()+style.square().sum())
        spectral = (2*local.square().sum()-3*style.square().sum())
        record = update_norm_diagnostics(scd, spectral, local, style, .03, .003, .5, 7.)
        dc = torch.cat((.03*2*local.detach(), .003*2*style.detach()), dim=1).double()
        dg = torch.cat((.03*4*local.detach(), -.003*6*style.detach()), dim=1).double()
        np.testing.assert_allclose(record['scd_update_norm'], dc.norm(dim=1), rtol=1e-6)
        np.testing.assert_allclose(record['spectral_update_norm'], dg.norm(dim=1), rtol=1e-6)
        np.testing.assert_allclose(record['cosine'], (dc*dg).sum(1)/(dc.norm(dim=1)*dg.norm(dim=1)), rtol=1e-6)
        np.testing.assert_allclose(record['mixed_update_norm'], (.5*dc+7*dg).norm(dim=1), rtol=1e-6)
        self.assertIsNone(local.grad)
        self.assertIsNone(style.grad)
        alpha = freeze_update_alpha([record])
        self.assertAlmostEqual(alpha, float(np.median((dc.norm(dim=1)/dg.norm(dim=1)).numpy())))
        # A separate joint backward remains possible after the optional VJPs.
        (.5*scd+7*spectral).backward()
        torch.testing.assert_close(local.grad, 29*local)
        for invalid in ([], [{'scd_over_spectral_update_ratio': [None]}],
                        [{'scd_over_spectral_update_ratio': [float('inf')]}]):
            with self.assertRaises(ValueError):
                freeze_update_alpha(invalid)

    def test_zero_update_norm_is_explicitly_inactive_and_cannot_calibrate(self):
        from decoded_spectral import update_norm_diagnostics, freeze_update_alpha
        local = torch.ones(2, 2, requires_grad=True)
        style = torch.ones(2, 1, requires_grad=True)
        scd = local.square().sum()+style.square().sum()
        zero = 0*scd
        record = update_norm_diagnostics(scd, zero, local, style, .03, .003, .5, 1.)
        self.assertEqual(record['status'], ['inactive', 'inactive'])
        self.assertEqual(record['scd_over_spectral_update_ratio'], [None, None])
        with self.assertRaises(ValueError):
            freeze_update_alpha([record])

    def test_driver_hard_profile_is_opt_in_and_half_control_needs_no_spectral_weight(self):
        from scripts.diagnose_decoded_spectral import configuration
        args = ['--asset-root','/a','--asset-manifest-sha256','a'*64,
                '--result-root','/r','--paired-state-root','/p']
        default = configuration(args)
        self.assertEqual(default['physical_graph']['profile'], 'smooth')
        for m in (100, 200):
            c = configuration(args+['--arm','live_control','--lowband-modes',str(m),
                                    '--checkpoint-scd-weight','.5','--update-norm-probes'])
            self.assertEqual(c['physical_graph']['profile'], 'hard')
            self.assertEqual(c['physical_graph']['modes'], m)
            self.assertIsNone(c['physical_graph']['beta'])
            self.assertEqual(c['checkpoint_scd_weight'], .5)
            self.assertEqual(c['weight'], 0)
        with self.assertRaises(ValueError):
            configuration(args+['--arm','control','--checkpoint-scd-weight','.5'])


if __name__ == '__main__':
    unittest.main()
