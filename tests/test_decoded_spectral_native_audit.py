"""Native-free gates for the actual decoder directional-derivative audit."""
import importlib.util
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph_spectral import SmoothSpectralTarget
from legacy_trajectory import model_buffer_hashes, rng_hash


class Decoder(torch.nn.Module):
    def __init__(self, *, wrong_backward=False):
        super().__init__()
        self.register_buffer('counter', torch.zeros(1, dtype=torch.float64))
        self.wrong_backward = wrong_backward
        self.grad_flags = []

    def decoder(self, unused, beta, context, style):
        self.grad_flags.append((torch.is_grad_enabled(), context.requires_grad, style.requires_grad))
        self.counter.add_(1)
        noise = .01 * (torch.rand_like(context) + random.random() + np.random.rand())
        class WrongSquare(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.save_for_backward(value)
                return value.square()
            @staticmethod
            def backward(ctx, gradient):
                value, = ctx.saved_tensors
                return -2 * value * gradient
        squared = WrongSquare.apply(context) if self.wrong_backward else context.square()
        return (squared + .1*style[:, :1] + noise + self.counter.clone()*.001).reshape(2, 4, 3)


class NativeAuditTests(unittest.TestCase):
    def audit(self):
        self.assertIsNotNone(importlib.util.find_spec('decoded_spectral_native_audit'))
        from decoded_spectral_native_audit import audit_decoded_gradient
        return audit_decoded_gradient

    def inputs(self):
        latent = torch.linspace(.2, .8, 24, dtype=torch.float64).reshape(2, 12, 1, 1).requires_grad_()
        style = torch.ones(2, 3, 1, 1, dtype=torch.float64, requires_grad=True)
        reference = torch.zeros(2, 4, 3, dtype=torch.float64)
        target = SmoothSpectralTarget(reference, (torch.eye(4, dtype=torch.float64),)*2, [])
        return target, latent, style

    def test_nonlinear_decoder_matches_analytic_gradient_and_preserves_live_state(self):
        target, latent, style = self.inputs()
        vae = Decoder().eval()
        before = rng_hash(), model_buffer_hashes({'vae': vae})
        copies = latent.detach().clone(), style.detach().clone(), target.reference_xyz.clone()
        with torch.no_grad():
            report, arrays = self.audit()(vae, target, latent, style)
        self.assertTrue(report['passed'], report)
        self.assertEqual(len(report['finite_differences']), 2)
        self.assertTrue(all(record['passed'] for record in report['finite_differences']))
        expected_gradient = 4 * latent.detach().numpy().reshape(2, 4, 3) * arrays['baseline_decoded'] / 12
        np.testing.assert_allclose(arrays['gradient'].reshape(2, 4, 3), expected_gradient)
        self.assertTrue(np.all(arrays['direction'][1] == 0))
        self.assertAlmostEqual(float(np.linalg.norm(arrays['direction'][0])), 1.)
        np.testing.assert_array_equal(arrays['baseline_decoded'], arrays['repeat_decoded'])
        self.assertEqual(before, (rng_hash(), model_buffer_hashes({'vae': vae})))
        for value, copy in zip((latent, style, target.reference_xyz), copies):
            torch.testing.assert_close(value, copy, rtol=0, atol=0)
        self.assertIsNone(latent.grad)
        self.assertIsNone(style.grad)
        self.assertEqual(vae.grad_flags, [(True, True, True)]*6)
        self.assertTrue(all(np.isfinite(value).all() for value in arrays.values()))
        self.assertEqual(report['relative_tolerance'], .1)
        self.assertEqual(report['absolute_tolerance'], 1e-5)

    def test_return_annotation_is_deferred_for_python38_runtime(self):
        self.assertIsInstance(self.audit().__annotations__['return'], str)

    def test_wrong_native_backward_fails_while_retaining_fd_vectors(self):
        target, latent, style = self.inputs()
        report, arrays = self.audit()(Decoder(wrong_backward=True).eval(), target, latent, style)
        self.assertFalse(report['passed'])
        self.assertTrue(all(not record['passed'] for record in report['finite_differences']))
        self.assertTrue(all(record['finite_difference'] < 0 for record in report['finite_differences']))
        self.assertIn('fd_0_plus_decoded', arrays)
        self.assertIn('fd_1_minus_decoded', arrays)

    def test_zero_derivative_and_invalid_reference_return_archivable_failures(self):
        target, latent, style = self.inputs()
        class Zero(Decoder):
            def decoder(self, unused, beta, context, style):
                return (context*0).reshape(2, 4, 3)
        report, arrays = self.audit()(Zero().eval(), target, latent, style)
        self.assertFalse(report['passed'])
        self.assertIn('gradient', arrays)
        self.assertTrue(report['errors'])
        target.reference_xyz.requires_grad_()
        report, arrays = self.audit()(Decoder().eval(), target, latent, style)
        self.assertFalse(report['passed'])
        self.assertTrue(report['errors'])
        self.assertIn('latent', arrays)

    def test_unsupported_backward_returns_finite_baseline_without_mutating_live_state(self):
        target, latent, style = self.inputs()
        class Unsupported(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.square()
            @staticmethod
            def backward(ctx, gradient):
                raise RuntimeError('toy native backward unsupported')
        class UnsupportedDecoder(Decoder):
            def decoder(self, unused, beta, context, style):
                self.counter.add_(1)
                torch.rand(1)
                return Unsupported.apply(context).reshape(2, 4, 3)
        vae = UnsupportedDecoder().eval()
        before = rng_hash(), model_buffer_hashes({'vae': vae})
        report, arrays = self.audit()(vae, target, latent, style)
        self.assertFalse(report['passed'])
        self.assertTrue(any('unsupported' in error for error in report['errors']))
        self.assertIn('baseline_decoded', arrays)
        self.assertTrue(all(np.isfinite(value).all() for value in arrays.values()))
        self.assertEqual(before, (rng_hash(), model_buffer_hashes({'vae': vae})))

    def test_unstable_repeat_fails_even_when_fd_happens_to_match_autograd(self):
        target, latent, style = self.inputs()
        class Unstable(Decoder):
            def decoder(self, unused, beta, context, style):
                decoded = super().decoder(unused, beta, context, style)
                return decoded + (1. if len(self.grad_flags) == 2 else 0.)
        report, arrays = self.audit()(Unstable().eval(), target, latent, style)
        self.assertFalse(report['passed'])
        self.assertGreater(report['repeat_loss_absolute_difference'], 0)
        self.assertTrue(all(record['absolute_error'] <= record['allowed_absolute_error']
                            for record in report['finite_differences']))

    def test_post_backward_guard_error_still_returns_archivable_vectors(self):
        target, latent, style = self.inputs()
        audit = self.audit()
        with patch('decoded_spectral_native_audit.model_buffer_hashes',
                   side_effect=[{'vae': 'before'}, RuntimeError('toy native context unavailable')]):
            report, arrays = audit(Decoder().eval(), target, latent, style)
        self.assertFalse(report['passed'])
        self.assertIn('baseline_decoded', arrays)
        self.assertTrue(any('unavailable' in error for error in report['errors']))


if __name__ == '__main__':
    unittest.main()
