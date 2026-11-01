"""CPU witnesses for routed SCD-h/spectral-z; trained/native inference is external."""
import json
import hashlib
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from legacy_trajectory import rng_hash
from tests.test_chamfer_depth import load_sampler, run, seeded


class IdentityGraph:
    def __init__(self):
        self.calls = 0

    def __call__(self, reference):
        self.calls += 1
        batch, count, _ = reference.shape
        basis = torch.eye(count, dtype=reference.dtype).repeat(batch, 1, 1)
        eigenvalues = torch.arange(count, dtype=reference.dtype).repeat(batch, 1)
        return reference.detach().clone(), basis, eigenvalues


class SpectralStyleTests(unittest.TestCase):
    def route(self, modes=2, kappa=.7, **kwargs):
        from scdh_spectralz import SpectralStyleRoute
        return SpectralStyleRoute(IdentityGraph(), modes, kappa, **kwargs)

    def states(self, batch=2):
        q = torch.zeros(batch, 6, 3, dtype=torch.float64)
        local = torch.ones_like(q).requires_grad_()
        style = torch.ones(batch, 1, 3, dtype=torch.float64, requires_grad=True)
        clean = .9 * local + .2 * style
        return q, local, style, clean, 2 * (clean - q).square().sum()

    def test_applied_h_is_only_scd_and_z_only_weighted_spectral(self):
        q, local, style, clean, scd = self.states()
        route = self.route()
        route.initialize(q)
        h_gradient, z_gradient = route.gradients(clean, scd, local, style, 0, 70)
        torch.testing.assert_close(h_gradient, torch.full_like(local, 3.96))
        expected_z = 2 * .2 * 1.1 * (1 + torch.exp(torch.tensor(-2., dtype=torch.float64))) / 18
        torch.testing.assert_close(z_gradient, torch.full_like(style, expected_z))
        self.assertEqual(route.style_step, .7)
        record = route.diagnostics()['steps'][0]
        self.assertEqual(len(record['applied_local_spectral_norm']), 2)
        self.assertEqual(record['applied_local_spectral_norm'], [0., 0.])
        self.assertEqual(record['applied_style_scd_norm'], [0., 0.])

    def test_loss_has_actual_M_cutoff_and_correct_style_finite_difference(self):
        q, local, style, clean, _ = self.states()
        one, two = self.route(modes=1), self.route(modes=2)
        one.initialize(q); two.initialize(q)
        self.assertGreater(float(two.loss(clean).detach()), float(one.loss(clean).detach()))
        self.assertTrue(torch.autograd.gradcheck(lambda z: two.loss(.9 * local.detach() + .2 * z), (style,)))

    def test_batch_sum_preserves_each_sample_style_gradient(self):
        gradients = []
        for batch in (1, 2):
            q, local, style, clean, scd = self.states(batch)
            route = self.route(); route.initialize(q)
            gradients.append(route.gradients(clean, scd, local, style, 0, 70)[1][0])
        torch.testing.assert_close(*gradients)

    def test_cache_replays_same_physical_basis_without_rng_or_rebuild_and_rejects_tamper(self):
        q = self.states()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'basis.npz'
            first = self.route(cache_path=path)
            before = rng_hash(); first.initialize(q)
            self.assertEqual(before, rng_hash())
            second = self.route(modes=4, cache_path=path); second.initialize(q)
            self.assertEqual(second.graph.calls, 0)
            self.assertEqual(first.diagnostics()['graph']['basis_sha256'], second.diagnostics()['graph']['basis_sha256'])
            with np.load(path, allow_pickle=False) as archive:
                values = {k: archive[k].copy() for k in archive.files}
            values['eigenvalues'][0, 0] += .1
            with path.open('wb') as stream: np.savez(stream, **values)
            with self.assertRaisesRegex(ValueError, 'identity|hash'):
                self.route(cache_path=path).initialize(q)

    def test_zero_kappa_matches_actual_style_off_sampler_and_original_z_decode(self):
        module = load_sampler()
        seeded(); expected, expected_lion = run(module, depth=8, gamma=.03, eta=0., return_trajectory=True)
        expected_rng = rng_hash()
        seeded(); actual, lion = run(module, depth=8, gamma=.03, eta=.003,
                                     style_spectral=self.route(kappa=0.), return_trajectory=True)
        torch.testing.assert_close(actual.final_local_latent, expected.final_local_latent, atol=0, rtol=0)
        torch.testing.assert_close(actual.updated_style, expected.updated_style, atol=0, rtol=0)
        torch.testing.assert_close(actual.original_style, expected.original_style, atol=0, rtol=0)
        self.assertEqual(rng_hash(), expected_rng)

    def test_bad_configuration_and_nonfinite_gradients_stop(self):
        for modes, kappa in ((0, 1.), (True, 1.), (800, -1.), (800, float('nan'))):
            with self.subTest(modes=modes, kappa=kappa), self.assertRaises(ValueError):
                self.route(modes=modes, kappa=kappa)
        q, local, style, clean, scd = self.states()
        route = self.route(); route.initialize(q)
        with self.assertRaisesRegex((ValueError, FloatingPointError), 'finite'):
            route.gradients(clean, scd * float('inf'), local, style, 0, 70)

    def test_calibration_observes_all_M_only_at_registered_states_without_changing_C_base(self):
        from scdh_spectralz import SpectralCalibrationRoute
        module = load_sampler()
        seeded(); expected, _ = run(module, depth=8, gamma=.03, eta=.003, return_trajectory=True)
        expected_rng = rng_hash()
        route = SpectralCalibrationRoute(IdentityGraph(), (2, 4, 6), modes=(200, 400, 800, 1200, 1800))
        seeded(); actual, _ = run(module, depth=8, gamma=.03, eta=.003,
                                 style_spectral=route, return_trajectory=True)
        torch.testing.assert_close(actual.final_local_latent, expected.final_local_latent, rtol=0, atol=0)
        torch.testing.assert_close(actual.updated_style, expected.updated_style, rtol=0, atol=0)
        self.assertEqual(rng_hash(), expected_rng)
        records=route.diagnostics()['calibration']
        self.assertEqual({r['loop_count'] for r in records}, {2,4,6})
        self.assertEqual({r['M_max'] for r in records}, {200,400,800,1200,1800})
        self.assertTrue(all(len(r['ratios'])==len(actual.updated_style) for r in records))

    def test_physical_two_example_two_scale_audit_saves_vectors_and_rejects_wrong_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'fd.npz'
            route=self.route(audit_path=path)
            q,local,style,clean,scd=self.states()
            route.initialize(q)
            hgrad,zgrad=route.gradients(clean,scd,local,style,0,70)
            route.audit(clean,local,style,hgrad,zgrad,lambda z:.9*local.detach()+.2*z)
            self.assertEqual(len(route.diagnostics()['native_finite_difference']['checks']),4)
            with np.load(path,allow_pickle=False) as data:
                self.assertEqual(data['basis'].shape,(2,6,6))
                np.testing.assert_array_equal(data['style_gradient'],zgrad.detach().numpy())
            bad=self.route(audit_path=Path(directory).resolve()/'bad.npz');bad.initialize(q)
            with self.assertRaisesRegex(RuntimeError,'finite-difference'):
                bad.audit(clean,local,style,hgrad,zgrad*1000,lambda z:.9*local.detach()+.2*z)


class SpectralStudyConfigTests(unittest.TestCase):
    def args(self, root, count=64, stage='development'):
        ids = root/'ids.json'
        ids.write_text(json.dumps(dict(dataset='modelnet40_c', n_examples=2468,
                                      stage='heldout' if stage == 'confirmation' else 'development',
                                      indices=list(range(count)))))
        return ['--preview', '--stage', stage, '--dataset-root', str(root), '--label-path', str(root/'label.npy'),
                '--asset-root', str(root), '--result-root', str(root/'results'),
                '--pointmae-config', 'model.yml', '--pointmae-checkpoint', 'model.pth',
                '--index-manifest', str(ids), '--paired-state-root', str(root/'paired')]

    def test_controls_use_same_N8_host_and_primary_probability_readout(self):
        from scripts import eval_scdh_spectralz as study
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            for arm, eta in [('scd', .003), ('off', 0.)]:
                config, = study.configurations(args + ['--arm', arm, '--alpha-h', '.05'])
                self.assertEqual(config['guidance_protocol'], dict(gamma=.05, eta=eta, p=.95, total=100, scd_normalize=False))
                self.assertEqual(config['csv_policy'], 'probability_mean')
                self.assertEqual(config['normal_reverse_steps'], 8)
                self.assertEqual(config['background_reverse_steps'], 35)
                self.assertIsNone(config['graph'])
                self.assertFalse(config['scdh_spectralz']['global_diffusion'])

    def test_spectral_step_comes_only_from_pinned_M_specific_calibration(self):
        from scripts import eval_scdh_spectralz as study
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = self.args(root)
            report = root/'calibration.json'
            report.write_text(json.dumps(dict(schema_version=1,status='frozen',
                coefficients={'400':dict(status='available', b_M=2., n_valid=384)},
                binding=dict(source_snapshot_sha256='b'*64,asset_manifest_sha256='a'*64))))
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            flags = ['--arm','spectral','--modes','400','--rho','.3',
                     '--calibration-report',str(report),'--calibration-report-sha256',digest]
            config, = study.configurations(args+flags)
            self.assertEqual(config['scdh_spectralz']['kappa_z_spectral_effective'], .6)
            self.assertEqual(config['scdh_spectralz']['spectral_weight'],1.)
            self.assertEqual(config['graph']['isolate_policy'],'offdiag')
            self.assertTrue(config['scheduler_final_alpha_to_one'])
            config['graph']['isolate_policy']='legacy'
            with self.assertRaises(ValueError): study.validate_config(config)
            report.write_text('{}')
            with self.assertRaises(ValueError): study.configurations(args+flags)

    def test_unregistered_scope_or_uncalibrated_research_cannot_run(self):
        from scripts import eval_scdh_spectralz as study
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); args=self.args(root)
            for flags in (['--arm','spectral','--modes','800','--rho','1'],
                          ['--arm','scd','--alpha-h','.02'], ['--arm','scd','--seed','1']):
                with self.subTest(flags=flags), self.assertRaises((ValueError,SystemExit)):
                    study.configurations(args+flags)
            with self.assertRaises(ValueError):
                study.configurations(self.args(root,count=2468)+['--arm','scd'])

    def test_worker_rejects_tampered_readout_subset_and_scope(self):
        from scripts import eval_scdh_spectralz as study
        with tempfile.TemporaryDirectory() as directory:
            config,=study.configurations(self.args(Path(directory))+['--arm','off'])
            for key,value in [('corruptions',['gaussian']),('max_batches',1),('severity',3),
                              ('primary_policy','terminal'),('trajectory_observation',False)]:
                changed=copy.deepcopy(config); changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError): study.validate_config(changed)
            for key,value in [('temperature',2.),('labels_used',True),('normal_candidate_names',['terminal'])]:
                changed=copy.deepcopy(config); changed['trajectory_selection'][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError): study.validate_config(changed)
            changed=copy.deepcopy(config);changed['selection_manifest']['stage']='heldout'
            with self.assertRaises(ValueError): study.validate_config(changed)

    def test_worker_bootstraps_audited_native_modules_before_importing_graph(self):
        from scripts import eval_scdh_spectralz as study
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,=study.configurations(self.args(root)+['--arm','off'])
            (root/'config.json').write_text(json.dumps(config))
            ready=[];module=types.ModuleType('graph_spectral_v2_offdiag')
            def symbol(name):
                if name=='GraphSpectralDNA':
                    self.assertTrue(ready,'Graph imported before pinned native bootstrap')
                    return IdentityGraph
                raise AttributeError(name)
            module.__getattr__=symbol
            def load(common,manifest):ready.append(True);return None
            def execute(directories,**hooks):return hooks['runtime_loader'](config,{})
            with patch.dict('sys.modules',{'graph_spectral_v2_offdiag':module}),patch.object(study.shared,'worker',side_effect=execute),patch.object(study.shared,'load_guidance_runtime',side_effect=load):
                study.worker([root])

    def test_native_gradient_gate_does_not_apply_an_arbitrary_uncalibrated_style_step(self):
        from scripts import eval_scdh_spectralz as study
        with tempfile.TemporaryDirectory() as directory:
            config,=study.configurations(self.args(Path(directory),count=32,stage='technical')+
                ['--arm','spectral','--modes','800','--rho','1'])
            self.assertEqual(config['scdh_spectralz']['kappa_z_spectral_effective'],0.)
            self.assertIsNone(config['scdh_calibration'])


if __name__ == '__main__':
    unittest.main()
