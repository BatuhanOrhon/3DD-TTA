"""CPU contracts for physical decoded guidance; no LION weights/native CUDA."""
import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from graph_spectral import SmoothSpectralTarget
from legacy_trajectory import model_buffer_hashes, rng_hash
from semantic_trajectory import gradient_probe
from tests.test_chamfer_depth import CPU, Scheduler, VAE, Prior, chamfer, load_sampler, seeded


def target_for(reference):
    return SmoothSpectralTarget(reference.detach().clone(),
                                tuple(torch.eye(reference.shape[1], dtype=reference.dtype)
                                      for _ in reference), [])


class DecodedSpectralTests(unittest.TestCase):
    def observer_class(self):
        self.assertIsNotNone(importlib.util.find_spec('decoded_spectral'),
                             'Physical decoded spectral observer is missing')
        from decoded_spectral import DecodedSpectralObserver
        return DecodedSpectralObserver

    def fixture(self, *, dtype=torch.float64, num_classes=40, **observer_options):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.register_buffer('counter', torch.zeros(1))
            def decoder(self, unused, beta, context, style):
                self.calls += 1
                self.counter.add_(1)
                torch.rand(1)
                return context.reshape(-1, 2048, 4)[:, :, :3].square() + .02*style[:, :3, None].transpose(1, 2)
        vae = Decoder().eval()
        head = torch.nn.Linear(3, num_classes).to(dtype).eval().requires_grad_(False)
        def classify(points):
            self.assertFalse(points.requires_grad)
            self.assertFalse(torch.is_grad_enabled())
            torch.rand(1)
            return {'pointmae': head(points.mean(1))}
        reference = torch.zeros(1, 2048, 3, dtype=dtype)
        weights = torch.linspace(.25, .75, 2048, dtype=dtype)
        target = SmoothSpectralTarget(reference, (torch.diag(weights),), [])
        observer = self.observer_class()(vae, {'pointmae': head}, classify, 2, num_classes, target=target,
                                        checkpoint_steps=[1], record_gradients=True, **observer_options)
        latent = torch.linspace(.1, .9, 8192, dtype=dtype).reshape(1, -1, 1, 1).requires_grad_()
        style = torch.ones(1, 128, 1, 1, dtype=dtype, requires_grad=True)
        observer.initial(input_points=reference)
        return observer, latent, style

    def test_scan_singleton_live_gradient_and_readout_have_15_classes(self):
        observer, latent, style = self.fixture(num_classes=15)
        loss = observer.clean(latent, style, 0, 10)
        local_gradient, style_gradient = torch.autograd.grad(loss, (latent, style))
        self.assertGreater(float(local_gradient.norm()), 0)
        self.assertGreater(float(style_gradient.norm()), 0)
        observer.final_state(latent, style, 0)
        observer.terminal({'pointmae': torch.zeros(1, 15)})
        self.assertEqual(observer.arrays('pointmae')['trajectory_logits'].shape, (1, 2, 15))

    def test_one_live_decode_matches_independent_loss_gradient_and_directional_fd(self):
        observer, latent, style = self.fixture()
        before = rng_hash(), model_buffer_hashes(observer.models)
        loss = observer.clean(latent, style, 0, 10)
        xyz = latent.reshape(1, 2048, 4)[:, :, :3]
        residual = xyz.square() + .02
        weights = torch.linspace(.25, .75, 2048, dtype=latent.dtype).reshape(1, -1, 1)
        expected = (weights * residual.square()).sum() / (3 * 2048)
        torch.testing.assert_close(loss, expected)
        local_grad, style_grad = torch.autograd.grad(loss, (latent, style))
        expected_local = torch.zeros_like(latent).reshape(1, 2048, 4)
        expected_local[:, :, :3] = 4 * weights * xyz * residual / (3 * 2048)
        torch.testing.assert_close(local_grad, expected_local.reshape_as(latent))
        expected_style = torch.zeros_like(style)
        expected_style[0, :3, 0, 0] = .04 * (weights * residual).sum(1)[0] / (3 * 2048)
        torch.testing.assert_close(style_grad, expected_style)
        self.assertEqual(observer.vae.calls, 1)
        direction = torch.ones_like(latent) / latent.numel()**.5
        epsilon = 1e-5
        def decoder_loss(value):
            with gradient_probe(observer.models.values()):
                decoded = observer.vae.decoder(None, beta=None, context=value.flatten(1), style=style.flatten(1))
                return (weights * decoded.square()).sum() / (3 * 2048)
        finite_difference = (decoder_loss(latent + epsilon*direction)
                             - decoder_loss(latent - epsilon*direction)) / (2*epsilon)
        torch.testing.assert_close((local_grad * direction).sum(), finite_difference)
        self.assertEqual(before, (rng_hash(), model_buffer_hashes(observer.models)))
        self.assertEqual(len(observer.probes), 1)
        record = observer.spectral_records[0]
        self.assertGreater(record['decoded_gradient_norm'], 0)
        self.assertGreater(record['clean_latent_gradient_norm'], 0)
        self.assertGreater(record['decoder_style_gradient_norm'], 0)
        self.assertIsNone(latent.grad)
        self.assertIsNone(style.grad)
        observer.final_state(latent, style, 0)
        observer.terminal({'pointmae': torch.zeros(1, 40)})
        self.assertEqual(observer.arrays('pointmae')['trajectory_logits'].shape, (1, 2, 40))
        json.dumps(observer.diagnostics(), allow_nan=False)

    def test_original_decoder_style_can_be_detached_without_disconnect(self):
        observer, latent, style = self.fixture()
        loss = observer.clean(latent, style.detach(), 0, 10)
        gradient, = torch.autograd.grad(loss, (latent,))
        self.assertGreater(gradient.norm().item(), 0)
        self.assertIsNone(observer.spectral_records[0]['decoder_style_gradient_norm'])
        self.assertFalse(observer.spectral_records[0]['decoder_style_connected'])

    def test_frozen_residual_preserves_exact_forward_loss_and_uses_identity_xyz_gradient(self):
        full, latent, style = self.fixture()
        surrogate = self.observer_class()(full.vae, full.heads, full.classify_decoded, 2,
            target=full.target, checkpoint_steps=[1], record_gradients=True, freeze_decoder_residual=True)
        before = rng_hash(), model_buffer_hashes(full.models)
        latent_before, style_before = latent.detach().clone(), style.detach().clone()
        full_loss = full.clean(latent, style, 0, 10)
        loss = surrogate.clean(latent, style, 0, 10)
        torch.testing.assert_close(loss, full_loss, rtol=0, atol=0)
        torch.testing.assert_close(torch.from_numpy(surrogate.saved['pointmae'][0]),
                                   torch.from_numpy(full.saved['pointmae'][0]), rtol=0, atol=0)
        gradient, style_gradient = torch.autograd.grad(loss, (latent, style), allow_unused=True)
        xyz = latent.reshape(1, 2048, 4)[:, :, :3]
        weights = torch.linspace(.25, .75, 2048, dtype=latent.dtype).reshape(1, -1, 1)
        expected = torch.zeros_like(latent).reshape(1, 2048, 4)
        expected[:, :, :3] = 2 * weights * (xyz.square() + .02) / (3 * 2048)
        torch.testing.assert_close(gradient, expected.reshape_as(latent))
        self.assertIsNone(style_gradient)
        self.assertEqual(full.vae.calls, 2)
        self.assertEqual(before, (rng_hash(), model_buffer_hashes(full.models)))
        torch.testing.assert_close(latent, latent_before, rtol=0, atol=0)
        torch.testing.assert_close(style, style_before, rtol=0, atol=0)
        record = surrogate.spectral_records[0]
        self.assertFalse(record['decoded_connected'])
        self.assertIsNone(record['decoded_gradient_norm'])
        self.assertTrue(record['clean_latent_connected'])
        diagnostics = surrogate.diagnostics()['decoded_spectral']
        self.assertEqual(diagnostics['loss_gradient_mode'], 'frozen_residual_surrogate')
        self.assertFalse(diagnostics['decoder_jacobian_in_loss'])
        self.assertTrue(full.diagnostics()['decoded_spectral']['decoder_jacobian_in_loss'])

    def test_frozen_residual_bypasses_unsupported_decoder_backward_but_reaches_upstream_style(self):
        class Unsupported(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.square()
            @staticmethod
            def backward(ctx, gradient):
                raise RuntimeError('native decoder backward unsupported')
        for freeze in (False, True):
            observer, local, original_style = self.fixture(freeze_decoder_residual=freeze)
            observer.vae.decoder = lambda unused, beta, context, style: Unsupported.apply(
                context.reshape(1, 2048, 4)[:, :, :3])
            condition = torch.ones(1, 128, 1, 1, dtype=local.dtype, requires_grad=True)
            predicted_clean = .9*local + .01*condition.mean()
            if not freeze:
                with self.assertRaisesRegex(RuntimeError, 'backward unsupported'):
                    observer.clean(predicted_clean, original_style.detach(), 0, 10)
            else:
                loss = observer.clean(predicted_clean, original_style.detach(), 0, 10)
                local_gradient, condition_gradient = torch.autograd.grad(loss, (local, condition))
                self.assertGreater(local_gradient.norm().item(), 0)
                self.assertGreater(condition_gradient.norm().item(), 0)

    def test_frozen_residual_flag_requires_bool_and_explicit_false_matches_default(self):
        for invalid in (0, 1, None, 'false'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'freeze_decoder_residual'):
                self.fixture(freeze_decoder_residual=invalid)
        results = []
        for options in ({}, {'freeze_decoder_residual': False}):
            torch.manual_seed(7)
            observer, latent, style = self.fixture(**options)
            loss = observer.clean(latent, style, 0, 10)
            gradient, = torch.autograd.grad(loss, (latent,))
            results.append((loss.detach(), gradient, observer.saved['pointmae'][0], rng_hash()))
        for index in (0, 1):
            torch.testing.assert_close(results[0][index], results[1][index], rtol=0, atol=0)
        self.assertTrue((results[0][2] == results[1][2]).all())
        self.assertEqual(results[0][3], results[1][3])

    def test_unselected_steps_skip_decoder_and_duplicate_checkpoints_fail(self):
        observer, latent, style = self.fixture()
        self.assertIsNone(observer.clean(latent, style, 1, 0))
        self.assertEqual(observer.vae.calls, 0)
        observer.clean(latent, style, 0, 10)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            observer.clean(latent, style, 0, 10)
        self.assertEqual(observer.vae.calls, 1)

    def test_reference_must_be_exact_prepared_physical_xyz(self):
        observer, _, _ = self.fixture()
        vae, heads, callback = observer.vae, observer.heads, observer.classify_decoded
        Observer = self.observer_class()
        with self.assertRaisesRegex(ValueError, '2048'):
            Observer(vae, heads, callback, 2, target=target_for(torch.zeros(1, 4, 3)), checkpoint_steps=[1])
        reference = observer.target.reference_xyz.clone().requires_grad_()
        with self.assertRaisesRegex(ValueError, 'detached'):
            Observer(vae, heads, callback, 2,
                     target=SmoothSpectralTarget(reference, observer.target.filters, []), checkpoint_steps=[1])
        observer, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, 'prepared'):
            observer.initial(input_points=torch.ones(1, 2048, 3))

    def test_sampler_default_zero_parity_and_positive_weight_updates_live_local_and_style(self):
        Observer = self.observer_class()
        module = load_sampler()
        reference = torch.ones(1, 2048, 3)
        target = target_for(reference)
        def run(weight=None, observe=True):
            lion = SimpleNamespace(vae=VAE().eval(), priors=[None, Prior().eval()])
            head = torch.nn.Linear(3, 40).eval().requires_grad_(False)
            observer = Observer(lion.vae, {'pointmae': head},
                                lambda x: {'pointmae': head(x.mean(1))}, 2,
                                target=target, checkpoint_steps=[1]) if observe else None
            seeded()
            kwargs = {} if weight is None else {'decoded_spectral_weight': weight}
            with patch.multiple(module, torch=CPU(), DDIMScheduler=Scheduler,
                                chamfer_grad=lambda: chamfer, grad_freeze=lambda model: None):
                result = module.tta_reconstruct(reference, lion, 2, .01, .01, .95,
                                                trajectory_observer=observer, return_trajectory=True, **kwargs)
            return result, observer, rng_hash()
        baseline, _, baseline_rng = run(observe=False)
        default, _, default_rng = run()
        zero, observer, zero_rng = run(0.)
        guided, _, _ = run(100.)
        for actual in (default, zero):
            torch.testing.assert_close(actual.final_local_latent, baseline.final_local_latent, rtol=0, atol=0)
            torch.testing.assert_close(actual.updated_style, baseline.updated_style, rtol=0, atol=0)
        self.assertEqual((default_rng, zero_rng), (baseline_rng, baseline_rng))
        self.assertFalse(torch.equal(guided.final_local_latent, baseline.final_local_latent))
        self.assertFalse(torch.equal(guided.updated_style, baseline.updated_style))
        self.assertTrue(torch.equal(guided.original_style, baseline.original_style))
        self.assertEqual(observer.vae.encode_calls, 1)
        self.assertTrue(torch.equal(observer.vae.styles[0], zero.original_style.squeeze(3).squeeze(2)))
        with self.assertRaisesRegex(FloatingPointError, 'Weighted decoded spectral'):
            run(1e100)

    def test_sampler_rejects_invalid_weights_and_missing_observer_before_setup(self):
        module = load_sampler()
        for weight in (-1., float('nan'), float('inf'), True, '1'):
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, 'decoded_spectral_weight'):
                module.tta_reconstruct(None, None, 2, .01, .01, .95, decoded_spectral_weight=weight)
        with self.assertRaisesRegex(ValueError, 'observer'):
            module.tta_reconstruct(None, None, 2, .01, .01, .95, decoded_spectral_weight=1.)


if __name__ == '__main__':
    unittest.main()
