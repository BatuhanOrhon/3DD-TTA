"""CPU-mocked checks for paired GSD routing on shared sampler inputs."""
from contextlib import ExitStack
from dataclasses import replace
import importlib
import json
import math
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import patch

import torch


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

    def set_timesteps(self, total, device=None):
        self.timesteps = torch.arange(total - 1, -1, -1)
        self.alphas_cumprod = torch.full((total,), .8)

    def step(self, noise, timestep, latent):
        return SimpleNamespace(pred_original_sample=latent - noise,
                               prev_sample=latent - .1 * noise)


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.02))
        self.decode_styles = []
        self.encode_calls = 0

    def encode(self, points):
        self.encode_calls += 1
        batch = points.shape[0]
        return None, None, [[torch.full((batch, 8), .2)],
                            [torch.linspace(-.2, .2, 8192).repeat(batch, 1)]]

    def global2style(self, shape):
        return shape * 3 + 1

    def decoder(self, unused, beta, context, style):
        self.decode_styles.append(style.detach().clone())
        return context.reshape(-1, 2048, 4)[:, :, :3] + self.weight * style.mean()


class Prior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self.styles = []

    def forward(self, x, t, condition_input, clip_feat):
        self.styles.append(condition_input.detach().clone())
        return self.weight * x + .01 * condition_input.mean(dim=1, keepdim=True)


def chamfer(left, right):
    distance = (left - right).square().sum(-1)
    return distance, distance, None, None


def freeze(module):
    module.requires_grad_(False)


class GraphConfig:
    k = 10
    delta = .1
    graph_gamma = .6
    modes = 100
    eigenspace_rtol = 1e-5
    eigenspace_atol = 1e-7


class CompositionTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules, {
            "third_party.ChamferDistancePytorch": types.ModuleType("third_party.ChamferDistancePytorch"),
            "third_party.ChamferDistancePytorch.chamfer3D": types.ModuleType("third_party.ChamferDistancePytorch.chamfer3D"),
            "third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D": types.ModuleType("third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D"),
            "diffusers": types.ModuleType("diffusers"),
            "utilities_3dd_tta": types.ModuleType("utilities_3dd_tta"),
        }))
        sys.modules["third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D"].chamfer_3DDist = lambda: None
        sys.modules["diffusers"].DDIMScheduler = object
        sys.modules["utilities_3dd_tta"].grad_freeze = freeze
        self.baseline = importlib.import_module("tta")
        self.stack.enter_context(patch.multiple(
            self.baseline, torch=CPU(), DDIMScheduler=Scheduler,
            chamfer_grad=lambda: chamfer, grad_freeze=freeze))
        self.gsd = importlib.import_module("tta_gsd")
        self.inputs = torch.ones(1, 2048, 3)
        self.lion = SimpleNamespace(vae=VAE(), priors=[None, Prior()])
        Scheduler.instances.clear()

    def config(self, local="off", style="off", **weights):
        from gsd_composition import CompositionConfig
        return CompositionConfig(local_mode=local, style_mode=style, **weights)

    def run_route(self, cfg, *, prepared=None, total=10, steps_back=50, observer=None):
        return self.gsd.tta_gsd_reconstruct(
            self.inputs, self.lion, steps_back, .017, .023, .95, total=total,
            spectral_weight=1., scd_weight=1., composition_config=cfg,
            prepared_inputs=prepared, composition_observer=observer)

    def test_preparation_encodes_once_and_replays_detached_copies_with_stable_hashes(self):
        from gsd_paired_inputs import prepare_guidance_inputs
        torch.manual_seed(17)
        prepared = prepare_guidance_inputs(self.inputs, self.lion, total=10, steps_back_local=50)
        self.assertEqual(self.lion.vae.encode_calls, 1)
        self.assertFalse(prepared.shape_latent.requires_grad)
        self.assertFalse(prepared.local_latent.requires_grad)
        self.assertFalse(prepared.style_conditioning.requires_grad)
        self.assertFalse(prepared.local_noise.requires_grad)
        self.assertEqual(prepared.timesteps.tolist(), [4, 3, 2, 1, 0])
        self.assertEqual(len(prepared.input_sha256), 64)
        self.assertEqual(len(prepared.config_sha256), 64)
        original = prepared.local_latent.clone()
        self.run_route(self.config("scd", "scd"), prepared=prepared)
        self.assertTrue(torch.equal(original, prepared.local_latent))
        with self.assertRaisesRegex(ValueError, "point cloud"):
            self.gsd.tta_gsd_reconstruct(
                self.inputs + 1, self.lion, 50, .017, .023, .95, total=10,
                composition_config=self.config("scd", "scd"),
                prepared_inputs=prepared)

    def test_prepared_scheduler_scalar_may_live_off_the_point_cloud_device(self):
        from gsd_paired_inputs import prepare_guidance_inputs
        prepared = prepare_guidance_inputs(
            self.inputs, self.lion, total=10, steps_back_local=50)
        prepared = replace(prepared, alpha_bar=torch.empty((), device="meta"))
        with patch("gsd_paired_inputs._config_hash", return_value=prepared.config_sha256):
            prepared.validate_for(self.inputs, total=10, steps_back_local=50)

    def test_all_legacy_composition_routes_match_corresponding_sampler(self):
        # Same CPU fixture, encoded inputs and noise are regenerated from the
        # same seed to provide a concrete legacy-path parity check.
        cases = (("scd", "scd", 0., 1.), ("spectral", "spectral", 1., 0.),
                 ("sum", "sum", 1., 1.))
        for local_mode, style_mode, spectral_weight, scd_weight in cases:
            with self.subTest(local=local_mode, style=style_mode):
                torch.manual_seed(23)
                expected = self.gsd.tta_gsd_reconstruct(
                    self.inputs, self.lion, 50, .017, .023, .95, total=10,
                    spectral_weight=spectral_weight, scd_weight=scd_weight)
                torch.manual_seed(23)
                actual = self.run_route(self.config(local_mode, style_mode))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hard_v1_probe_runs_even_when_spectral_route_is_masked(self):
        target = SimpleNamespace(diagnostics=[{"actual_rank": 2}], calls=0)

        def loss(predicted):
            target.calls += 1
            return predicted.square().mean()

        target.loss = loss
        graph = types.ModuleType("graph_spectral")
        graph.SpectralConfig = GraphConfig
        graph.build_spectral_target = lambda reference, config: target
        events = []
        with patch.dict(sys.modules, {"graph_spectral": graph}):
            self.run_route(self.config("scd", "off"), observer=events.append)
        self.assertEqual(target.calls, 5)
        step_events = [event for event in events if event.get("kind") == "step"]
        self.assertEqual(len(step_events), 5)
        self.assertTrue(all(event["local_spectral_grad_norm"] > 0 for event in step_events))
        self.assertTrue(all(event["style_spectral_grad_norm"] > 0 for event in step_events))
        json.dumps(events, allow_nan=False)

    def test_style_off_local_off_observers_and_reversed_arm_order_do_not_mutate_preparation(self):
        from gsd_paired_inputs import prepare_guidance_inputs
        torch.manual_seed(31)
        prepared = prepare_guidance_inputs(self.inputs, self.lion, total=10, steps_back_local=50)
        bundle = {name: value.clone() for name, value in prepared.tensor_items()}
        torch.manual_seed(99)
        local_off = self.run_route(self.config("off", "off"), prepared=prepared)
        torch.manual_seed(99)
        observer_events = []
        local_off_observed = self.run_route(self.config("off", "off"), prepared=prepared,
                                            observer=observer_events.append)
        torch.testing.assert_close(local_off, local_off_observed, rtol=0, atol=0)
        self.assertEqual(len([event for event in observer_events
                              if event.get("kind") == "step"]), 5)
        torch.manual_seed(99)
        other_arm = self.run_route(self.config("spectral", "scd"), prepared=prepared)
        torch.manual_seed(99)
        replay = self.run_route(self.config("off", "off"), prepared=prepared)
        torch.testing.assert_close(local_off, replay, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(other_arm).all())
        step_events = [event for event in observer_events if event.get("kind") == "step"]
        self.assertTrue(all(event["local_routed_direction_norm"] == 0
                            and event["style_update_norm"] == 0
                            and event["style_drift_norm"] == 0
                            for event in step_events))
        for name, value in prepared.tensor_items():
            self.assertTrue(torch.equal(bundle[name], value))
        self.assertTrue(all(torch.equal(s, torch.full_like(s, 0.2))
                            for s in self.lion.vae.decode_styles[-1:]))

    def test_rejects_ambiguous_legacy_weights_and_calibration_probes(self):
        cfg = self.config("sum", "sum")
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.gsd.tta_gsd_reconstruct(
                self.inputs, self.lion, 50, .017, .023, .95, total=10,
                spectral_weight=2., composition_config=cfg)
        with self.assertRaisesRegex(ValueError, "probe"):
            self.gsd.tta_gsd_reconstruct(
                self.inputs, self.lion, 50, .017, .023, .95, total=10,
                composition_config=cfg, probe_observer=lambda event: None)

    def test_original_shape_latent_decode_contract_and_step_counts(self):
        from gsd_paired_inputs import prepare_guidance_inputs
        for total, steps_back, expected_count in ((100, 5, 5), (100, 35, 35)):
            self.lion.vae.decode_styles.clear()
            torch.manual_seed(total + steps_back)
            prepared = prepare_guidance_inputs(
                self.inputs, self.lion, total=total, steps_back_local=steps_back)
            events = []
            self.run_route(self.config("sum", "sum"), prepared=prepared,
                           total=total, steps_back=steps_back, observer=events.append)
            self.assertEqual(len([event for event in events if event.get("kind") == "step"]),
                             expected_count)
            self.assertEqual(len(self.lion.vae.decode_styles), 1)
            torch.testing.assert_close(self.lion.vae.decode_styles[0],
                                       torch.full((1, 8), .2), rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad and p.grad is None
                                for module in (self.lion.vae, self.lion.priors[1])
                                for p in module.parameters()))

    def test_last_step_style_only_perturbation_is_logged_but_not_decoded(self):
        from gsd_paired_inputs import prepare_guidance_inputs
        graph = types.ModuleType("graph_spectral")
        graph.SpectralConfig = GraphConfig
        graph.build_spectral_target = lambda reference, config: SimpleNamespace(
            diagnostics=[], loss=lambda predicted: predicted.sum() * 0.)
        prepared = prepare_guidance_inputs(self.inputs, self.lion, total=10, steps_back_local=50)
        cfg = self.config("off", "off")
        with patch.dict(sys.modules, {"graph_spectral": graph}):
            expected = self.run_route(cfg, prepared=prepared)
            events = []
            original_compose = self.gsd.compose_block
            call_count = [0]

            def perturb_final_style(scd_grad, spectral_grad, **kwargs):
                call_count[0] += 1
                direction, rows = original_compose(scd_grad, spectral_grad, **kwargs)
                if call_count[0] == 10:
                    direction = torch.ones_like(direction)
                return direction, rows

            with patch.object(self.gsd, "compose_block", side_effect=perturb_final_style):
                actual = self.run_route(cfg, prepared=prepared, observer=events.append)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        step_events = [event for event in events if event.get("kind") == "step"]
        self.assertEqual(len(step_events), 5)
        self.assertAlmostEqual(step_events[-1]["style_update_norm"], .023 * math.sqrt(8), places=7)
        self.assertAlmostEqual(step_events[-1]["style_drift_norm"], .023 * math.sqrt(8), places=7)

    def test_observer_exposes_pcgrad_and_hard_v1_graph_scalar_diagnostics(self):
        class Config:
            k = 2
            delta = .25
            graph_gamma = .5
            modes = 3
            eigenspace_rtol = 1e-5
            eigenspace_atol = 1e-7

        target = SimpleNamespace(
            diagnostics=[{"actual_rank": 2, "k": 2, "threshold": .25,
                          "runtime_seconds": .125,
                          "eigengap": None}],
            loss=lambda predicted: predicted.square().mean())
        graph = types.ModuleType("graph_spectral")
        graph.SpectralConfig = Config
        graph.build_spectral_target = lambda reference, config: target
        events, returned_rows = [], []
        original_compose = self.gsd.compose_block

        def capture_compose(*args, **kwargs):
            direction, rows = original_compose(*args, **kwargs)
            returned_rows.append((kwargs["mode"], rows[0]))
            return direction, rows

        with patch.dict(sys.modules, {"graph_spectral": graph}), \
                patch.object(self.gsd, "compose_block", side_effect=capture_compose):
            self.run_route(self.config("sum", "pcgrad"), observer=events.append)

        graph_rows = [event for event in events if event.get("kind") == "graph"]
        step_rows = [event for event in events if "step_index" in event]
        self.assertEqual(len(graph_rows), 1)
        self.assertEqual(graph_rows[0]["graph_actual_rank"], 2)
        self.assertEqual(graph_rows[0]["graph_k"], 2)
        self.assertEqual(graph_rows[0]["graph_runtime_seconds"], .125)
        self.assertEqual(graph_rows[0]["graph_config_delta"], .25)
        self.assertEqual(len(step_rows), 5)
        for step_index, event in enumerate(step_rows):
            for field in ("style_raw_projected_norm", "style_capped_projected_norm",
                          "style_cap_factor", "style_applied_scale",
                          "style_applied_norm", "style_routed_scd_dot",
                          "style_routed_spectral_dot"):
                self.assertIn(field, event)
            self.assertAlmostEqual(event["style_raw_projected_norm"],
                                   returned_rows[2 * step_index + 1][1]["raw_projected_norm"])
            self.assertAlmostEqual(event["style_applied_norm"],
                                   returned_rows[2 * step_index + 1][1]["applied_norm"])
        self.assertTrue(all(not torch.is_tensor(value) for event in events
                            for value in event.values()))
        json.dumps(events, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
