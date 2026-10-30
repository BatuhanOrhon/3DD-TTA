"""Label-free calibration math, graph retention and coverage contracts."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from legacy_trajectory import rng_hash
from research_artifacts import CORRUPTIONS
from tests import test_chamfer_guidance_variants as sampler_fixtures
from tests import test_semantic_trajectory as semantic_fixtures


class CalibrationContracts(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.calibrate_chamfer_variants'))
        from scripts import calibrate_chamfer_variants
        return calibrate_chamfer_variants

    def test_combined_norm_uses_actual_eta_local_gamma_style_and_neutral_names(self):
        calibration = self.module()
        rows = calibration.update_gradient_rows((torch.tensor([[3.]]), torch.tensor([[4.]])),
                                                (torch.tensor([[6.]]), torch.tensor([[8.]])), eta=.1, gamma=.3)
        self.assertAlmostEqual(rows[0]['combined_update_ratio'], 2.)
        self.assertAlmostEqual(rows[0]['chamfer_update_norm'], (0.3 ** 2 + 1.2 ** 2) ** .5)
        self.assertFalse(any('spectral' in name for name in rows[0]))

    def rows(self):
        return {dataset: [dict(corruption=corruption, original_id=index, loop_count=step, candidate=candidate,
                              combined_update_ratio=ratio, chamfer_update_norm=1., candidate_update_norm=ratio,
                              local_cosine=.5, style_cosine=.5)
                          for corruption in CORRUPTIONS for index in range(70)
                          for step in ((15, 23) if corruption == 'background' else (5, 8))
                          for candidate, ratio in (('semantic_kl', 2.), ('semantic_h', 4.), ('weighted', .5))]
                for dataset in ('modelnet40_c', 'scanobjectnn_c')}

    def test_coefficient_pool_rejects_missing_coverage_and_zero_norms(self):
        calibration = self.module()
        rows = self.rows()
        report = calibration.coefficient_report(rows, {name: list(range(70)) for name in rows})
        self.assertEqual(report['coefficients'], dict(semantic_kl=.05, semantic_h=.025, weighted_scale=2.))
        rows['modelnet40_c'].pop()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            calibration.coefficient_report(rows, {name: list(range(70)) for name in rows})
        rows = self.rows()
        rows['modelnet40_c'][0]['chamfer_update_norm'] = 0.
        with self.assertRaisesRegex(ValueError, 'near-zero'):
            calibration.coefficient_report(rows, {name: list(range(70)) for name in rows})

    def test_observer_keeps_both_semantic_graphs_and_never_assigns_grad(self):
        calibration = self.module()
        fixture = semantic_fixtures.SemanticContracts()
        vae, head, _, plain, semantic, local, style = fixture.fixture(15)
        def native(left, right):
            distance = torch.cdist(left, right).square()
            first, nearest = distance.min(-1)
            second, reverse = distance.min(-2)
            return first, second, nearest, reverse
        observer = calibration.CalibrationTrajectoryObserver(
            vae, {'pointmae': head}, plain, 10, 15,
            classify_semantic_decoded=semantic, kind='kl', reference_logits=fixture.reference(15),
            reference_record={'forward_calls': 1}, original_ids=[9, 17], corruption='gaussian',
            chamfer=native, density_k=2)
        before = rng_hash()
        observer.clean(local, style, 2, 970)
        predicted = 2 * local + .3 * style.mean(1, keepdim=True)
        semantic_scalar = observer.clean(predicted, style, 4, 950)
        reference = torch.linspace(-.7, .9, 24).reshape(2, 4, 3)
        ch_loss = (predicted.reshape(2, 4, 3) - reference).square().sum()
        observer.guidance_components(ch_loss=ch_loss, semantic_loss=semantic_scalar, noisy_local=local,
                                     style=style, pred_latent=predicted.reshape(2, 4, 3), h0=reference,
                                     loop_count=5, timestep=950, eta=.01, gamma=.02, p=.95)
        self.assertEqual(before, rng_hash())
        self.assertIsNone(local.grad); self.assertIsNone(style.grad)
        self.assertEqual({row['candidate'] for row in observer.calibration_rows}, {'semantic_kl', 'semantic_h', 'weighted'})
        self.assertEqual(len(observer.calibration_rows), 6)
        self.assertTrue(all(row['candidate_update_norm'] > 0 for row in observer.calibration_rows))
        (ch_loss + 0 * semantic_scalar).backward()
        self.assertIsNotNone(local.grad); self.assertIsNotNone(style.grad)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))

    def test_diagnostic_sampler_hook_preserves_zero_semantic_c_trajectory(self):
        self.module()
        class Diagnostic(sampler_fixtures.Observer):
            def guidance_components(self, **values):
                if values['loop_count'] in (5, 8):
                    torch.autograd.grad(values['ch_loss'], (values['noisy_local'], values['style']), retain_graph=True)
        fixture = sampler_fixtures.ChamferVariantTests()
        baseline = fixture.run_sampler()
        observed = fixture.run_sampler({'variant': 'semantic', 'semantic_weight': 0.}, observer=Diagnostic())
        torch.testing.assert_close(observed[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(observed[1], baseline[1], rtol=0, atol=0)
        self.assertEqual(observed[-1], baseline[-1])

    def bundle(self, root, dataset, rows):
        path = root / dataset
        path.mkdir()
        selection = dict(dataset=dataset, n_examples=2468 if dataset == 'modelnet40_c' else 581,
                         stage='development', indices=list(range(70)))
        raw = json.dumps(selection).encode()
        (path / 'selection_manifest.json').write_bytes(raw)
        identity = dict(bytes=1, sha256=hashlib.sha256(b'x').hexdigest())
        sources = {}
        for name in ('scripts/calibrate_chamfer_variants.py', 'tta_gsd_v2.py', 'semantic_trajectory.py'):
            target = path / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'x')
            sources['repo/' + name] = identity
        config = dict(dataset=dataset, seed=0, batch_size=70, backbone='pointmae', backbones=['pointmae'],
            corruptions=list(CORRUPTIONS), max_batches=0, selection_manifest=selection,
            selected_original_ids=list(range(70)), status='partial', completed_batches={name: 1 for name in CORRUPTIONS},
            guidance_protocol=dict(chamfer_experiment=dict(variant='semantic', semantic_weight=0.), eta=.01, gamma=.01, p=.95),
            semantic_guidance={'loss_reduction': 'sum'},
            chamfer_calibration={'purpose': 'label_free_chamfer_calibration'},
            classifier_state_before_sha256='a'*64, classifier_state_after_sha256='a'*64,
            lion_state_before={'vae':'b'*64}, lion_state_after={'vae':'b'*64}, executed_source_files=sources,
            native_extensions={'chamfer': identity}, asset_manifest={'files': {'asset': identity}},
            asset_manifest_sha256='c'*64, runtime={'cuda': '12.1', 'gpu': 'synthetic test record'},
            run_id='synthetic-calibration', selection_manifest_identity={'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()})
        (path / 'config.json').write_text(json.dumps(config))
        for corruption in CORRUPTIONS:
            target = path / 'batches' / corruption / 'batch0000'
            target.mkdir(parents=True)
            diagnostic = dict(guidance_metrics={'graph_calls': 0, 'spectral_loss_calls': 0}, trajectory=dict(
                initial_rng_sha256='d'*64, probe_rng_unchanged=True, probe_buffers_unchanged=True,
                calibration=dict(probe_calls=2, autograd_grad_calls=8, labels_used_for_selection=False,
                                 rows=[row for row in rows if row['corruption'] == corruption])))
            (target / 'diagnostics.json').write_text(json.dumps(diagnostic))
        manifest = {str(file.relative_to(path)): {'bytes': file.stat().st_size, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
                    for file in path.rglob('*') if file.is_file()}
        (path / 'bundle_manifest.json').write_text(json.dumps(manifest))
        return path

    def test_sealed_summary_reads_gradients_without_any_prediction_or_label_file(self):
        calibration = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [self.bundle(root, dataset, rows) for dataset, rows in self.rows().items()]
            report = calibration.summarize_bundles(paths)
            self.assertEqual(report['coefficients']['semantic_kl'], .05)
            self.assertEqual(report['status'], 'verified_label_free_gpu')
            self.assertEqual(set(report['calibration_bundles']), {'modelnet40_c', 'scanobjectnn_c'})


if __name__ == '__main__':
    unittest.main()
