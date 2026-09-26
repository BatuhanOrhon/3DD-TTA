"""CPU trajectory checks; native Chamfer/DDIM adapters are replaced only here."""
from contextlib import ExitStack
import importlib
import importlib.util
import json
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import patch

import torch


def load_baseline():
    names = ["third_party.ChamferDistancePytorch",
             "third_party.ChamferDistancePytorch.chamfer3D",
             "third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D"]
    replacements = {name: types.ModuleType(name) for name in names}
    replacements[names[-1]].chamfer_3DDist = lambda: None
    replacements["diffusers"] = types.ModuleType("diffusers")
    replacements["diffusers"].DDIMScheduler = object
    replacements["utilities_3dd_tta"] = types.ModuleType("utilities_3dd_tta")
    replacements["utilities_3dd_tta"].grad_freeze = lambda module: None
    with patch.dict(sys.modules, replacements):
        module = importlib.import_module("tta")
    sys.modules.setdefault("tta", module)
    return module


class CPU:
    def __getattr__(self, name):
        return getattr(torch, name)

    def ones(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.ones(*args, **kwargs)


class Scheduler:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def set_timesteps(self, total, device):
        self.timesteps = torch.arange(total - 1, -1, -1)
        self.alphas_cumprod = torch.full((total,), .8)

    def step(self, noise, timestep, latent):
        return SimpleNamespace(pred_original_sample=latent-noise,
                               prev_sample=latent-.1*noise)


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.02))
        self.decode_styles = []
        self.encode_calls = 0

    def encode(self, points):
        self.encode_calls += 1
        return None, None, [[torch.ones(points.shape[0], 8)],
                            [torch.linspace(-.2, .2, 8192).repeat(points.shape[0], 1)]]

    def global2style(self, style):
        return style * 2

    def decoder(self, unused, beta, context, style):
        self.decode_styles.append(style.detach().clone())
        return context.reshape(-1, 2048, 4)[:, :, :3] + self.weight*style.mean()


class Prior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self.timesteps = []
        self.styles = []

    def forward(self, x, t, condition_input, clip_feat):
        self.timesteps.append(t.detach().clone())
        self.styles.append(condition_input.detach().clone())
        return self.weight*x + .01*condition_input.mean(dim=1, keepdim=True)


def chamfer(left, right):
    distance = (left-right).square().sum(-1)
    return distance, distance, None, None


def freeze(module):
    module.requires_grad_(False)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("tta_gsd"), "GSD trajectory module missing")
        self.baseline = load_baseline()
        self.gsd = importlib.import_module("tta_gsd")
        self.lion = SimpleNamespace(vae=VAE(), priors=[None, Prior()])
        self.points = torch.ones(1, 2048, 3)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.multiple(
            self.baseline, torch=CPU(), DDIMScheduler=Scheduler,
            chamfer_grad=lambda: chamfer, grad_freeze=freeze))
        Scheduler.instances.clear()

    def run_gsd(self, weight=1., **kwargs):
        return self.gsd.tta_gsd_reconstruct(
            self.points, self.lion, 100, .017, .023, .95, total=2,
            spectral_weight=weight, **kwargs)

    def test_zero_weight_exact_original_output_rng_and_graph_bypass_under_no_grad(self):
        torch.manual_seed(23)
        expected = self.baseline.tta_reconstruct(
            self.points, self.lion, 100, .017, .023, .95, total=2)
        expected_rng = torch.get_rng_state().clone()
        # A sentinel module proves that zero-weight execution needs no graph import.
        for profile, beta in ((None, None), ("hard", None), ("smooth", 2.)):
            with self.subTest(profile=profile), \
                    patch.dict(sys.modules, {"graph_spectral": None}), torch.no_grad():
                torch.manual_seed(23)
                actual = self.run_gsd(0., spectral_profile=profile, spectral_beta=beta)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_active_zero_objective_preserves_original_unequal_rate_math_and_scheduler(self):
        target = SimpleNamespace(diagnostics=[], loss=lambda predicted: predicted.sum()*0)
        graph = SimpleNamespace(SpectralConfig=lambda: None,
                                build_spectral_target=lambda *args: target)
        torch.manual_seed(3)
        expected = self.baseline.tta_reconstruct(
            self.points, self.lion, 100, .017, .023, .95, total=2)
        expected_styles = [s.clone() for s in self.lion.priors[1].styles]
        self.lion.priors[1].styles.clear()
        observed = []
        with patch.dict(sys.modules, {"graph_spectral": graph}), torch.no_grad():
            torch.manual_seed(3)
            actual = self.run_gsd(scheduler_observer=observed.append)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_style, expected_style in zip(self.lion.priors[1].styles, expected_styles):
            torch.testing.assert_close(actual_style, expected_style, rtol=0, atol=0)
        self.assertEqual(len(observed), 1)
        self.assertEqual(Scheduler.instances[0].kwargs, Scheduler.instances[1].kwargs)
        self.assertNotIn("set_alpha_to_one", observed[0].kwargs)
        self.assertEqual([t.item() for t in self.lion.priors[1].timesteps], [2, 1, 2, 1])

    def test_real_spectral_objective_updates_both_inputs_and_keeps_original_decoder(self):
        import graph_spectral
        # Exercise the real graph loss on 8 vertices; production uses all 2048.
        build = graph_spectral.build_spectral_target
        def small_graph(reference, config):
            target = build(reference[:, :8], config)
            return SimpleNamespace(diagnostics=target.diagnostics,
                                   loss=lambda predicted: target.loss(predicted[:, :8]))
        config = graph_spectral.SpectralConfig(k=3, modes=3, delta=.1)
        events = []
        before = [p.detach().clone() for m in [self.lion.vae, self.lion.priors[1]]
                  for p in m.parameters()]
        torch.manual_seed(5)
        original = self.run_gsd(0)
        with patch.object(graph_spectral, "build_spectral_target", side_effect=small_graph):
            with torch.no_grad():
                torch.manual_seed(5)
                guided = self.run_gsd(2., spectral_config=config,
                                      diagnostics_observer=events.append)
        self.assertFalse(torch.equal(original, guided))
        self.assertEqual([e["kind"] for e in events], ["graph", "step", "step"])
        json.dumps(events, allow_nan=False)
        for event in events[1:]:
            self.assertGreater(event["local_spectral_grad_norm"], 0)
            self.assertGreater(event["style_spectral_grad_norm"], 0)
            self.assertAlmostEqual(event["local_weighted_spectral_grad_norm"],
                                   2*event["local_spectral_grad_norm"], places=6)
            self.assertGreater(event["style_update_norm"], 0)
            self.assertEqual(event["batch_size"], 1)
        for style in self.lion.vae.decode_styles:
            self.assertTrue(torch.equal(style, torch.ones(1, 8)))
        after = [p for m in [self.lion.vae, self.lion.priors[1]] for p in m.parameters()]
        for old, current in zip(before, after):
            self.assertTrue(torch.equal(old, current))
            self.assertFalse(current.requires_grad)
            self.assertIsNone(current.grad)

    def test_smooth_profile_flows_through_both_guidance_states(self):
        import graph_spectral
        build = graph_spectral.build_smooth_spectral_target

        def small_smooth_graph(reference, config, *, profile, beta):
            target = build(reference[:, :8], config, profile=profile, beta=beta)
            return SimpleNamespace(diagnostics=target.diagnostics,
                                   loss=lambda predicted: target.loss(predicted[:, :8]))

        config = graph_spectral.SpectralConfig(k=3, modes=3, delta=.1)
        events = []
        with patch.object(graph_spectral, "build_smooth_spectral_target",
                          side_effect=small_smooth_graph), torch.no_grad():
            result = self.run_gsd(2., spectral_config=config,
                                  spectral_profile="smooth", spectral_beta=1.7,
                                  diagnostics_observer=events.append)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual([event["kind"] for event in events], ["graph", "step", "step"])
        for event in events[1:]:
            self.assertEqual(event["spectral_profile"], "smooth")
            self.assertEqual(event["spectral_beta"], 1.7)
            self.assertGreater(event["local_spectral_grad_norm"], 0)
            self.assertGreater(event["style_spectral_grad_norm"], 0)

    def test_nonfinite_loss_is_rejected(self):
        target = SimpleNamespace(diagnostics=[], loss=lambda predicted: predicted.sum()*float("nan"))
        graph = SimpleNamespace(SpectralConfig=lambda: None,
                                build_spectral_target=lambda *args: target)
        with patch.dict(sys.modules, {"graph_spectral": graph}):
            with self.assertRaisesRegex(FloatingPointError, "loss"):
                self.run_gsd()

    def test_nonfinite_output_is_rejected(self):
        with patch.object(self.lion.vae, "decoder", return_value=torch.tensor(float("inf"))):
            with self.assertRaisesRegex(FloatingPointError, "output"):
                self.run_gsd(0)

    def test_finite_loss_with_nonfinite_gradient_is_rejected(self):
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.save_for_backward(value)
                return value.sum()*0

            @staticmethod
            def backward(ctx, gradient):
                value, = ctx.saved_tensors
                return torch.full_like(value, float("inf"))

        target = SimpleNamespace(diagnostics=[], loss=BadGradient.apply)
        graph = SimpleNamespace(SpectralConfig=lambda: None,
                                build_spectral_target=lambda *args: target)
        with patch.dict(sys.modules, {"graph_spectral": graph}):
            with self.assertRaisesRegex(FloatingPointError, "gradient"):
                self.run_gsd()

    def test_invalid_weight_is_rejected_before_encoding(self):
        for weight in [-1., float("nan"), float("inf")]:
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                self.run_gsd(weight)
        self.assertEqual(self.lion.vae.encode_calls, 0)


if __name__ == "__main__":
    unittest.main()
