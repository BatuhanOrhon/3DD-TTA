"""Native-free contracts for the bounded shared-state mechanism probes."""
import ast
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from graph_spectral import SpectralConfig, build_probe_spectral_target
from legacy_trajectory import _rng_state, rng_hash, tensor_hash
from scripts import probe_gsd_shared_states as probe
from sparse_heat import build_sparse_heat_target


class SharedProbeTests(unittest.TestCase):
    def setUp(self):
        self.config = SpectralConfig(k=2, delta=1., graph_gamma=0, modes=2)
        self.x = torch.tensor([[[0., 0., 0.], [.2, 0., .1], [.4, .1, .2], [.8, .2, .3], [1.1, .1, .4], [1.4, .2, .6]]], dtype=torch.float64)
        self.target = build_probe_spectral_target(self.x, self.config)
        self.sparse = build_sparse_heat_target(self.x, self.config)

    def test_repeated_boundary_expansion_is_anchored_and_empty_safe(self):
        q = torch.tensor([0., 1., 1., 1., 2., 3.], dtype=torch.float64)
        config = SpectralConfig(modes=4, eigenspace_atol=1e-7)
        row = probe.band_bounds(q, 4, torch.eye(6, dtype=q.dtype), 1., config)
        self.assertEqual(row['high_start'], 1)
        self.assertEqual(row['high_rank'], 5)
        self.assertEqual(row['overlap_rank'], 3)
        # The tolerance stays anchored at q[2], without transitive chaining.
        q = torch.tensor([0., 1., 1.000009, 1.000018, 2.], dtype=torch.float64)
        config = SpectralConfig(modes=2, eigenspace_atol=0, eigenspace_rtol=1e-5)
        self.assertEqual(probe.band_bounds(q, 1, torch.eye(5, dtype=q.dtype), 1., config)['high_start'], 2)
        empty = probe.band_bounds(torch.empty(0), 0, torch.empty(0, 0), 0., config)
        self.assertEqual(empty['high_rank'], 0)
        with self.assertRaises(ValueError):
            probe.band_bounds(torch.tensor([1., 0.]), 1, torch.eye(2), 1., config)

    def test_rotation_sign_invariant_band_energy_and_deterministic_direction(self):
        indices = torch.arange(6)
        basis = torch.eye(6, dtype=torch.float64)
        q = torch.tensor([0., 1., 1., 2., 3., 4.], dtype=torch.float64)
        spectrum = (indices, basis, q, 3)
        angle = .71
        rotated = basis.clone()
        rotated[:, 1] = math.cos(angle)*basis[:, 1] + math.sin(angle)*basis[:, 2]
        rotated[:, 2] = -math.sin(angle)*basis[:, 1] + math.cos(angle)*basis[:, 2]
        rotated[:, 0] *= -1
        other = (indices, rotated, q, 3)
        delta = torch.arange(18, dtype=torch.float64).reshape(6, 3)/17
        first, second = probe.modal_energy(delta, spectrum), probe.modal_energy(delta, other)
        for key in ('low', 'other', 'total'):
            self.assertAlmostEqual(first[key], second[key], places=12)
        before = rng_hash()
        a = probe.deterministic_band_direction(self.x[0], spectrum, 0, 3)
        b = probe.deterministic_band_direction(self.x[0], other, 0, 3)
        torch.testing.assert_close(a, b)
        self.assertAlmostEqual(float(a.norm()), 1.)
        self.assertEqual(rng_hash(), before)
        isolated = (indices[:4], basis[:4, :4], q[:4], 2)
        self.assertAlmostEqual(probe.modal_energy(delta, isolated)['isolate'], float(delta[4:].square().sum()))

    def test_h6_preserves_feature_style_xyz_and_joint_norm(self):
        local = torch.arange(1, 25, dtype=torch.float64).reshape(1, 24, 1, 1)
        style = torch.tensor([[3., 4.]], dtype=torch.float64)
        original, style_copy = local.clone(), style.clone()
        gh, gs, records = probe.project_scd_xyz(local, style, self.target.spectra)
        torch.testing.assert_close(probe.xyz(gh).norm(), probe.xyz(local).norm())
        torch.testing.assert_close(gh.reshape(1, 6, 4)[:, :, 3], local.reshape(1, 6, 4)[:, :, 3], rtol=0, atol=0)
        torch.testing.assert_close(gs, style, rtol=0, atol=0)
        torch.testing.assert_close(local, original, rtol=0, atol=0)
        torch.testing.assert_close(style, style_copy, rtol=0, atol=0)
        torch.testing.assert_close(probe.joint_norm(.03*gh, .003*gs), probe.joint_norm(.03*local, .003*style))
        projected = probe.project_band(probe.xyz(gh)[0], self.target.spectra[0], 0, self.target.spectra[0][3])
        torch.testing.assert_close(projected, probe.xyz(gh)[0])
        self.assertGreater(records[0]['projected_xyz_norm'], 0)
        zero_spectrum = (torch.arange(6), torch.eye(6, dtype=torch.float64), torch.arange(6, dtype=torch.float64), 1)
        tiny = torch.zeros_like(local)
        probe.xyz(tiny)[0, 3, 0] = 1
        with self.assertRaisesRegex(ValueError, 'tiny'):
            probe.project_scd_xyz(tiny, style, (zero_spectrum,))

    def test_joint_calibration_combination_and_actual_next_step_arithmetic(self):
        h = torch.arange(24, dtype=torch.float64).reshape(1, 24, 1, 1)
        style = torch.ones(1, 2, dtype=torch.float64)
        c = (.03*(h+1), .003*style)
        s = probe.matched_update((torch.ones_like(h)*2, style*3), c)
        cs = probe.combined_update(c, s)
        torch.testing.assert_close(probe.joint_norm(*s), probe.joint_norm(*c))
        torch.testing.assert_close(probe.joint_norm(*cs), probe.joint_norm(*c))
        prev_sample = h + 19
        hn, sn = probe.sampler_branch(prev_sample, style, s, .5)
        torch.testing.assert_close(hn, prev_sample-.5*s[0])
        torch.testing.assert_close(sn, style-.5*s[1])
        self.assertFalse(torch.equal(hn, h-.5*s[0]))
        self.assertEqual(probe.next_timestep([70, 60, 50, 40, 30, 20, 10, 0], 6, 20), 10)
        for args in (([70, 60], 2, 60), ([70, 60], 1, 60), ([60, 70], 1, 60)):
            with self.assertRaises(ValueError):
                probe.next_timestep(*args)
        with self.assertRaisesRegex(ValueError, 'tiny'):
            probe.combined_update(c, tuple(-v for v in c))
        with self.assertRaises(ValueError):
            probe.sampler_branch(prev_sample, style, c, .1)

    def test_operator_comparison_same_graph_and_joint_permutation_action(self):
        rows = probe.operator_comparison(self.target, self.target, self.sparse, self.sparse, self.x*.1)
        self.assertEqual(rows[0]['edge_overlap_jaccard'], 1.)
        self.assertEqual(rows[0]['support_jaccard'], 1.)
        self.assertEqual(rows[0]['heat_F']['relative_error'], 0.)
        self.assertEqual(rows[0]['heat_action']['relative_error'], 0.)
        self.assertLess(rows[0]['exact_U_L_vs_sparse'][0]['relative_error'], 1e-12)
        f = probe.embedded_operator(self.target.spectra[0], 6, heat=True)
        order = torch.tensor([5, 3, 1, 4, 2, 0])
        r = self.x[0]
        torch.testing.assert_close((f@r)[order], f[order][:, order]@r[order])

    def test_immutable_numeric_artifacts_and_input_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = torch.tensor([2., 3.])
            probe.write_arrays(root/'state.npz', dict(tensor=value))
            original_bytes = (root/'state.npz').read_bytes()
            value.zero_()
            with np.load(root/'state.npz', allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['tensor'], [2., 3.])
            with self.assertRaises(FileExistsError):
                probe.write_arrays(root/'state.npz', dict(tensor=value))
            self.assertEqual(original_bytes, (root/'state.npz').read_bytes())
            for value in (np.asarray([float('nan')]), np.asarray([object()]), np.asarray([1+2j])):
                with self.assertRaises(ValueError):
                    probe.write_arrays(root/'bad.npz', dict(tensor=value))
            probe.write_json(root/'status.json', dict(status='complete'))
            with self.assertRaises(FileExistsError):
                probe.write_json(root/'status.json', {})
        with self.assertRaises(ValueError):
            probe.validate_capture_config({})
        with self.assertRaises(ValueError):
            probe.forward_gate(torch.ones(2, 3), torch.ones(2, 4))
        self.assertFalse(probe.forward_gate(torch.zeros(2, 3), torch.ones(2, 3))['passed'])
        for filename in ('scripts/probe_gsd_shared_states.py', 'tests/test_gsd_shared_state_probes.py'):
            ast.parse((probe.REPO/filename).read_text(), feature_version=(3, 8))

    def test_toy_h3_native_floor_and_all_frozen_perturbations(self):
        class Decoder(torch.nn.Module):
            def decoder(self, _unused, *, beta, context, style):
                return probe.xyz(context)*2 + style[:, :3, None].transpose(1, 2)*.01
        vae = Decoder().eval().requires_grad_(False)
        local = torch.cat([self.x, torch.ones_like(self.x[:, :, :1])], -1).reshape(1, 24)
        style = torch.ones(1, 3, dtype=local.dtype)
        archived = probe.decode(vae, local, style)
        graph = self.sparse.graphs[0]
        bands = [probe.band_bounds(self.target.spectra[0][2], self.target.spectra[0][3],
                   graph.laplacian.to_dense(), graph.mean_active_degree, self.config)]
        before_rng, before_local = rng_hash(), tensor_hash(local)
        with tempfile.TemporaryDirectory() as temporary:
            result = probe.run_h3(Path(temporary), vae, local, style, archived, _rng_state(), self.target, self.target, bands)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['perturbations']), 8)
            self.assertEqual(len(result['repeat_gates']), 3)
            self.assertEqual(result['same_state_native_floor'][0]['max_delta_norm'], 0.)
            for row in result['perturbations']:
                self.assertTrue(row['objects'][0]['resolved_above_native_floor'])
                self.assertAlmostEqual(row['objects'][0]['output_energy']['total'], 4*row['objects'][0]['identity_xyz_control_energy']['total'], places=12)
            self.assertEqual(len(list(Path(temporary).glob('decoder_*.npz'))), 11)
        self.assertEqual(rng_hash(), before_rng)
        self.assertEqual(tensor_hash(local), before_local)

    def test_toy_native_fd_keeps_failed_witness(self):
        h, style = torch.ones(1, 8, dtype=torch.float64), torch.ones(1, 2, dtype=torch.float64)
        def evaluate(a, b, _kind):
            return float(a.square().sum()+b.square().sum()), a, a[:, :6].reshape(1, 2, 3)
        with tempfile.TemporaryDirectory() as temporary:
            good = probe.direction_fd(Path(temporary), 'C', h, style, (2*h, 2*style), evaluate)
            self.assertTrue(good['passed'])
            bad = probe.direction_fd(Path(temporary), 'S_lat', h, style, (-2*h, -2*style), evaluate)
            self.assertFalse(bad['passed'])
            self.assertEqual(len(list(Path(temporary).glob('fd_*.npz'))), 8)

    def test_toy_prior_scheduler_and_decoder_branch_replays(self):
        next_prior_flags, next_decoder_flags, next_rng_draws = [], [], []
        in_next_forward = [False]
        class Prior(torch.nn.Module):
            def forward(self, *, x, t, condition_input, clip_feat):
                in_next_forward[0] = int(t[0]) == 61
                if in_next_forward[0]:
                    next_prior_flags.append((torch.is_grad_enabled(), x.requires_grad,
                                             condition_input.requires_grad, x.is_leaf, condition_input.is_leaf))
                    next_rng_draws.append([float(torch.rand(()))])
                return .2*x + .03*condition_input.mean(1, keepdim=True)

        class VAE(torch.nn.Module):
            def decoder(self, _unused, *, beta, context, style):
                if in_next_forward[0]:
                    next_decoder_flags.append((torch.is_grad_enabled(), context.requires_grad, style.requires_grad))
                    next_rng_draws[-1].append(float(torch.rand(())))
                return 1.1*probe.xyz(context) + .01*style[:, None, :3]

        class Scheduler:
            def step(self, epsilon, _t, local):
                return SimpleNamespace(pred_original_sample=local-.5*epsilon, prev_sample=local-.1*epsilon)

        lion = SimpleNamespace(vae=VAE().eval().requires_grad_(False),
                               priors=torch.nn.ModuleList([torch.nn.Identity(), Prior()]).eval().requires_grad_(False))
        h = torch.cat([self.x+.13, torch.ones_like(self.x[:, :, :1])], -1).reshape(1, 24)
        style = torch.ones(1, 3, dtype=h.dtype)*.7
        z0 = torch.ones_like(style)*.3
        prior_t = torch.tensor([71.], dtype=h.dtype)
        scheduler = Scheduler()
        initial = scheduler.step(lion.priors[1](x=h, t=prior_t, condition_input=style, clip_feat=None), 70, h)
        decoded = probe.decode(lion.vae, initial.pred_original_sample, z0)

        def chamfer(left, right):
            distance = torch.cdist(left, right).square()
            a, b = distance.min(2), distance.min(1)
            return a.values, b.values, a.indices, b.indices

        before_rng, before_h, before_style = rng_hash(), tensor_hash(h), tensor_hash(style)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = probe.run_directions(directory, lion, scheduler, [70, 60, 50, 40, 30, 20, 10, 0],
                1, 70, h, style, prior_t, initial.pred_original_sample, decoded, z0, _rng_state(), _rng_state(),
                self.x, self.target, self.target, chamfer)
            self.assertEqual(result['status'], 'complete', result)
            self.assertEqual(set(r['name'] for r in result['branches']),
                             {'zero', 'C', 'S_lat', 'S_phys', 'P100_C', 'CS_lat', 'CS_phys'})
            self.assertEqual(len(result['branches']), 14)
            self.assertEqual(next_prior_flags, [(True, True, True, True, True)]*14)
            self.assertEqual(next_decoder_flags, [(False, False, False)]*14)
            expected_rng = torch.Generator().manual_seed(1729)
            expected_draws = [float(torch.rand((), generator=expected_rng)) for _ in range(2)]
            self.assertEqual(next_rng_draws, [expected_draws]*14)
            self.assertTrue(all(row['finite_difference']['passed'] for row in result['objectives'].values()))
            with np.load(directory/'branch_C_0p5_state.npz', allow_pickle=False) as archive:
                np.testing.assert_allclose(archive['noisy_local_next'], initial.prev_sample.numpy()-.5*archive['update_local'])
            with np.load(directory/'branch_zero_0p5_prediction.npz', allow_pickle=False) as first, np.load(directory/'branch_zero_1p0_prediction.npz', allow_pickle=False) as second:
                np.testing.assert_array_equal(first['decoded_xyz'], second['decoded_xyz'])
        self.assertEqual(rng_hash(), before_rng)
        self.assertEqual(tensor_hash(h), before_h)
        self.assertEqual(tensor_hash(style), before_style)


if __name__ == '__main__':
    unittest.main()
