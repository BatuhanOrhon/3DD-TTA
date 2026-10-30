"""CPU fixture: actual autograd path, fake prior/scheduler/graph/native kernel."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path.cwd()))
from legacy_trajectory import _rng_state, tensor_hash
import spectral_guidance_losses

spec = importlib.util.spec_from_file_location('calibration', Path(__file__).with_name('calibrate_spectral_losses.py'))
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class Scheduler:
    def __init__(self, **options):
        assert options == dict(beta_end=.02, beta_schedule='linear', beta_start=.0001,
            clip_sample=False, num_train_timesteps=1000, prediction_type='epsilon', set_alpha_to_one=False)
        self.alphas_cumprod = torch.linspace(.9999, .2, 1000)

    def set_timesteps(self, count, device):
        assert count == 100
        self.timesteps = torch.arange(990, -1, -10, device=device)

    def step(self, prediction, timestep, local):
        alpha = self.alphas_cumprod[timestep]
        return SimpleNamespace(pred_original_sample=(local - torch.sqrt(1 - alpha) * prediction) / torch.sqrt(alpha))


class Graph:
    use_4d_gft = False

    def __call__(self, local):
        return local[:, :, :3], torch.zeros(1, 1, 1), torch.linspace(0., 4., 2048).expand(70, -1)


def prior(*, x, t, condition_input, clip_feat):
    assert t.shape == (70,) and clip_feat is None
    return .2 * x + .1 * condition_input.mean(1, keepdim=True)


def chamfer(left, right):
    distances = (left - right).square().sum(-1)
    return distances, distances, None, None


class FirstStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, horizon):
        h0 = torch.zeros(70, 8192, 1, 1)
        z0 = torch.ones(70, 128, 1, 1)
        noise = torch.ones_like(h0)
        scheduler = Scheduler(beta_end=.02, beta_schedule='linear', beta_start=.0001,
            clip_sample=False, num_train_timesteps=1000, prediction_type='epsilon', set_alpha_to_one=False)
        scheduler.set_timesteps(100, device='cpu')
        times = scheduler.timesteps[-horizon:]
        alpha = scheduler.alphas_cumprod[times[0]]
        initial = {key: tensor_hash(value) for key, value in dict(style=z0,
            noisy_local=torch.sqrt(alpha) * h0 + noise * torch.sqrt(1 - alpha), timesteps=times, alpha_bar=alpha).items()}
        lion = SimpleNamespace(vae=SimpleNamespace(global2style=lambda value: value), priors=[None, prior])
        return lion, dict(local_latent=h0, shape_latent=z0, noise=noise), initial

    def test_first_step_gradients_include_prior_style_and_both_horizons(self):
        for horizon in (10, 30):
            lion, tensors, initial = self.fixture(horizon)
            with patch.dict(sys.modules, {'diffusers': SimpleNamespace(DDIMScheduler=Scheduler)}), \
                 patch.object(torch, 'bmm', lambda basis, signal: signal):
                norms, parity = calibration.measure_first_step(lion, Graph(), tensors, _rng_state(), initial, horizon, chamfer)
            self.assertTrue(parity['scalar_exact'] and parity['direct_coefficient_gradient_exact'])
            self.assertEqual(parity['rng_before_sha256'], parity['rng_after_sha256'])
            for blocks in norms.values():
                for values in blocks.values():
                    self.assertEqual(len(values), 70)
                    self.assertTrue(all(value > 0 for value in values))
            for block in ('local', 'style'):
                self.assertEqual(parity['native_gradient_differences'][block]['inline_repeat_relative_l2'], 0.)
                self.assertEqual(parity['native_gradient_differences'][block]['helper_vs_inline_relative_l2'], 0.)

    def test_changed_helper_arithmetic_fails_real_first_step_parity(self):
        lion, tensors, initial = self.fixture(10)
        actual = spectral_guidance_losses.spectral_guidance_loss
        def changed(*args):
            result = actual(*args)
            return result * .5 if args[-1] == 'legacy_l2' else result
        with patch.dict(sys.modules, {'diffusers': SimpleNamespace(DDIMScheduler=Scheduler)}), \
             patch.object(torch, 'bmm', lambda basis, signal: signal), \
             patch.object(spectral_guidance_losses, 'spectral_guidance_loss', changed), \
             self.assertRaisesRegex(ValueError, 'scalar differs'):
            calibration.measure_first_step(lion, Graph(), tensors, _rng_state(), initial, 10, chamfer)


if __name__ == '__main__':
    unittest.main()
