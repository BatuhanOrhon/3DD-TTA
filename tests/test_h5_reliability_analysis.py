"""Synthetic leakage, eigenspace and oracle witnesses for frozen H5 diagnostics."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

FILE = Path(__file__).resolve().parents[1]/'scripts/analyze_h5_reliability.py'


class H5AnalysisTests(unittest.TestCase):
    def analysis(self):
        self.assertTrue(FILE.is_file(), 'H5 CPU analyzer missing')
        spec = importlib.util.spec_from_file_location('h5_analysis', FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self):
        analysis = self.analysis()
        q = np.array([.24999999, .25000001, .6])
        diagnostic = dict(mean_active_degree=1., effective_boundary_atol=1e-6)
        bands = analysis.make_bands(q, diagnostic, dict(eigenspace_rtol=1e-5))
        reference = np.array([[1., 2., 3.], [2., 1., 0.], [0., 3., 1.]])
        history = np.stack([reference+np.array([.1+t*.03, .2-t*.02, 0.]) for t in range(4)])
        return analysis, bands, reference, history

    def test_native_float32_reversal_stays_in_existing_block_without_reordering(self):
        analysis = self.analysis()
        q = np.array([0., .9999, 1.1159626245498657, 1.1159625053405762, 2.1], np.float32)
        original = q.copy()
        diagnostic = dict(mean_active_degree=1., effective_boundary_atol=.0028)
        bands = analysis.make_bands(q, diagnostic, dict(eigenspace_rtol=1e-5))
        np.testing.assert_array_equal(q, original)
        band = next(b for b in bands if 2 in b['mode_indices'])
        self.assertIn([2, 4], band['repeated_blocks'])
        self.assertEqual(band['mode_indices'], [2, 3])
        for invalid in ([1.1159626, 1.1159625, .25], [0., .1, .6, .59999, .2]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'anchored eigenspace block'):
                analysis.make_bands(np.asarray(invalid, np.float32), diagnostic, dict(eigenspace_rtol=1e-5))

    def test_sorted_frozen_band_memberships_do_not_require_git_history(self):
        analysis = self.analysis()
        diagnostic, config = dict(mean_active_degree=1., effective_boundary_atol=1e-6), dict(eigenspace_rtol=1e-5)
        cases = (([], [[], [], [], [], [], []]),
                 ([.24999999, .25000001, .6], [[0, 1], [], [2], [], [], []]),
                 ([0., .25, .5, 1., 2., 4., 5.], [[0], [1], [2], [3], [4], [5, 6]]))
        for values, expected in cases:
            with self.subTest(values=values):
                bands = analysis.make_bands(np.asarray(values), diagnostic, config)
                self.assertEqual([b['mode_indices'] for b in bands], expected)
                self.assertEqual([b['mode_count'] for b in bands], [len(indices) for indices in expected])
                if len(values) == 3:
                    self.assertEqual(bands[0]['repeated_blocks'], [[0, 2]])
                self.assertTrue(all(b['normalized_atol'] == (1e-6 if values else 0.) and b['eigenspace_rtol'] == 1e-5 for b in bands))

    def test_future_geometry_cannot_change_earlier_clean_free_features(self):
        analysis, bands, reference, history = self.fixture()
        before = analysis.clean_free_features(history, reference, np.arange(3), np.eye(3), bands)
        changed = history.copy(); changed[3] += 1000.
        after = analysis.clean_free_features(changed, reference, np.arange(3), np.eye(3), bands)
        self.assertEqual(before[:18], after[:18])
        first = next(row for row in before if row['checkpoint_index'] == 0 and row['band_index'] == 0)
        self.assertIsNotNone(first['r'])
        self.assertIsNone(first['v'])
        self.assertIsNone(first['s'])
        self.assertEqual(first['v_status'], 'NO_PREVIOUS_CHECKPOINT_UNAVAILABLE')
        self.assertEqual(bands[0]['mode_indices'], [0, 1])
        self.assertTrue(bands[1]['empty'])

    def test_whole_repeated_block_is_orthogonal_rotation_invariant(self):
        analysis, bands, reference, history = self.fixture()
        rotated = np.eye(3)
        theta = .73
        rotated[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        plain = analysis.clean_free_features(history, reference, np.arange(3), np.eye(3), bands)
        other = analysis.clean_free_features(history, reference, np.arange(3), rotated, bands)
        for left, right in zip(plain, other):
            for key in ('r', 'v', 's'):
                if left[key] is not None:
                    self.assertAlmostEqual(left[key], right[key], places=13)
        a, _ = analysis.band_step(history[0], reference, np.arange(3), np.eye(3), bands[0])
        b, _ = analysis.band_step(history[0], reference, np.arange(3), rotated, bands[0])
        self.assertAlmostEqual(a['spectral_before'], b['spectral_before'], places=13)
        self.assertAlmostEqual(a['predicted_quadratic_delta'], b['predicted_quadratic_delta'], places=13)

    def test_spectral_overshoot_is_unavailable_before_oracle_and_clean_nn_matches_brute(self):
        analysis = self.analysis()
        band = dict(mode_indices=[0, 1], empty=False)
        points = np.array([[1000., 0., 0.], [1002., 0., 0.]])
        step, after = analysis.band_step(points, points+.01, np.arange(2), np.eye(2), band)
        self.assertEqual(step['spectral_status'], 'OVERSHOOT_UNAVAILABLE')
        self.assertFalse(step['oracle_eligible'])
        self.assertIsNone(after)
        self.assertGreater(step['predicted_quadratic_delta'], 0)
        clean = np.array([[0., 0., 0.], [3., 0., 0.]])
        decoded = np.array([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]])
        squared = ((decoded[:, None]-clean[None])**2).sum(-1)
        expected = squared.min(1).mean()+squared.min(0).mean()
        self.assertAlmostEqual(analysis.oracle.squared_chamfer(decoded, clean), expected)

    def test_scalar_rank_hash_preserves_zero_dimensional_tensor_shape(self):
        import torch
        from legacy_trajectory import tensor_hash
        analysis = self.analysis()
        scalar = np.asarray(3, dtype=np.int64)
        self.assertEqual(analysis.tensor_hash(scalar), tensor_hash(torch.from_numpy(scalar)))
        self.assertNotEqual(analysis.tensor_hash(scalar), analysis.tensor_hash(scalar.reshape(1)))

    def test_serialized_capture_roundtrip_and_eigensystem_hash_tampering(self):
        import torch
        import graph_spectral
        from legacy_trajectory import tensor_hash
        from scripts.diagnose_decoded_spectral import record_h5_capture
        analysis = self.analysis()
        prepared = torch.zeros(2, 2048, 3)
        center, scale = torch.zeros(2, 1, 3), torch.ones(2, 1, 1)
        snapshots = [prepared+t*.01 for t in range(4)]
        hashes = [tensor_hash(value) for value in snapshots]
        target = graph_spectral.ProbeSpectralTarget(prepared.clone(),
            ((torch.tensor([0, 1]), torch.eye(2), torch.tensor([0., .5]), 2),)*2,
            [dict(active_vertices=2, actual_rank=2, mean_active_degree=1., effective_boundary_atol=1e-7)]*2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.object(graph_spectral, 'build_probe_spectral_target', return_value=target):
                metadata = record_h5_capture(path, [459, 545], prepared, center, scale, snapshots,
                    ['clean_02', 'clean_04', 'clean_06', 'terminal'], [2, 4, 6, 8], hashes)
            files = {p.name: p.read_bytes() for p in path.iterdir()}
            terminal = io.BytesIO()
            np.savez_compressed(terminal, terminal_decoded_xyz=snapshots[-1].numpy(),
                                normalization_center=center.numpy(), normalization_scale=scale.numpy())
            files['terminal_geometry.npz'] = terminal.getvalue()
            diagnostic = dict(trajectory=dict(probes=[dict(decoded_geometry_sha256=value) for value in hashes[:3]]))
            def seal_metadata():
                raw = json.dumps(metadata).encode()
                files['h5_capture.json'] = raw
                diagnostic['h5_capture'] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                files['diagnostics.json'] = json.dumps(diagnostic).encode()
            seal_metadata()
            class Bundle:
                def read(self, name):
                    return files[Path(name).name]
                def json(self, name):
                    return json.loads(self.read(name))
            basic = dict(corruption='gaussian', batch=0, index=np.array([459, 545]), first=snapshots[0].numpy(),
                         identity=dict(prepared=metadata['reference_xyz_sha256'], output_rng=metadata['rng_after_sha256']))
            loaded = analysis.load_capture(Bundle(), 'gaussian/0', basic)
            self.assertEqual(loaded['objects'][0]['rank'], 2)
            self.assertEqual(loaded['geometry']['decoded_xyz'].shape, (2, 4, 2048, 3))
            metadata['spectra_tensor_sha256'][0]['rank'] = '0'*64
            seal_metadata()
            with self.assertRaisesRegex(ValueError, 'eigensystem tensor hash'):
                analysis.load_capture(Bundle(), 'gaussian/0', basic)

    def test_associations_and_zero_temporal_drift_do_not_invent_defined_statistics(self):
        analysis = self.analysis()
        rows = [dict(scores={'score': value}, oracle_benefit=bool(value), clean_cd_delta=1.-2*value)
                for value in (0., 1.)]
        metric = analysis.association(rows, 'score')
        self.assertAlmostEqual(metric['auroc'], 1.)
        self.assertAlmostEqual(metric['spearman'], 1.)
        self.assertIsNone(analysis.association(rows[:1], 'score')['auroc'])
        features = [dict(object_id=459, corruption='gaussian', band_index=0, repeat=repeat,
                         checkpoint_index=t, r=repeat*.1, v=None, s=None)
                    for repeat in range(3) for t in range(4)]
        variation = next(r for r in analysis.feature_variation(features) if r['feature'] == 'r')
        self.assertTrue(variation['native_dominates_temporal'])
        self.assertIsNone(variation['native_over_temporal_median'])

    def test_missing_input_is_archived_and_existing_output_is_never_overwritten(self):
        analysis = self.analysis()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'attempt1'
            args = ([root/'missing1.zip', root/'missing2.zip', root/'missing3.zip'],
                    root/'clean.npz', root/'receipt.json', output)
            with self.assertRaises(FileNotFoundError):
                analysis.analyze(*args)
            self.assertEqual(json.loads((output/'failure.json').read_text())['status'], 'failed')
            before = (output/'failure.json').read_bytes()
            with self.assertRaises(FileExistsError):
                analysis.analyze(*args)
            self.assertEqual(before, (output/'failure.json').read_bytes())

    def test_score_comparisons_use_one_common_feature_and_oracle_mask(self):
        analysis = self.analysis()
        rows = [dict(scores=dict(frequency_reliability=1., residual_reliability=1., combined_reliability=None),
                     oracle_benefit=True, clean_cd_delta=-1.),
                dict(scores=dict(frequency_reliability=0., residual_reliability=0., combined_reliability=0.),
                     oracle_benefit=False, clean_cd_delta=1.),
                dict(scores=dict(frequency_reliability=2., residual_reliability=2., combined_reliability=2.),
                     oracle_benefit=True, clean_cd_delta=-2.)]
        compared = analysis.score_comparison(rows)
        self.assertEqual(compared['common_eligible_rows'], 2)
        self.assertTrue(all(result['n'] == 2 for result in compared['scores'].values()))
        self.assertEqual(analysis.association(rows, 'frequency_reliability')['n'], 3)


if __name__ == '__main__':
    unittest.main()
