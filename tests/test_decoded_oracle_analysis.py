"""Small independent witnesses for posthoc physical-space oracle analysis."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

FILE = Path(__file__).resolve().parents[1]/'knowledge/evidence/decoded_physical_followup_20261104/analyze_oracle.py'


class OracleAnalysisTests(unittest.TestCase):
    def analysis(self):
        self.assertTrue(FILE.is_file(), 'CPU oracle analyzer missing')
        spec = importlib.util.spec_from_file_location('oracle_analysis', FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_frame_and_unequal_count_squared_set_distance(self):
        analysis = self.analysis()
        physical = analysis.physical_clean(np.array([[4., 8., 10.]]), np.array([[3., 6., 8.]]), np.array([[2.]]))
        np.testing.assert_allclose(physical, [[-3.3885, 3.3885, 1.69425]])
        left = np.array([[0., 0., 0.], [2., 0., 0.]])
        right = np.zeros((1, 3))
        self.assertEqual(analysis.squared_chamfer(left, right), 2.)
        np.testing.assert_array_equal(analysis.nearest_squared(left, right), [0., 4.])
        with patch.object(analysis, 'cKDTree', None):
            np.testing.assert_array_equal(analysis.nearest_squared(left, right), [0., 4.])

    def test_quadratic_step_matches_fd_and_exhibits_objective_bias(self):
        analysis = self.analysis()
        clean = np.array([[0., 0., 0.], [2., 0., 0.]])
        reference = clean + np.array([1., 0., 0.])
        metric = np.diag([1., .5])
        direction = np.array([[1., .2, 0.], [-.5, 0., 0.]])
        step = 1e-5
        fd = (analysis.quadratic_loss(clean+step*direction, reference, metric)
              - analysis.quadratic_loss(clean-step*direction, reference, metric))/(2*step)
        self.assertAlmostEqual(fd, -1./4., places=9)
        row = analysis.analyze_direction(clean, reference, metric, clean)
        self.assertTrue(row['resolved'])
        self.assertTrue(row['spectral_decrease_clean_cd_increase'])
        self.assertGreater(row['clean_cd_delta'], 0)
        self.assertLess(row['spectral_delta'], 0)
        self.assertAlmostEqual(row['spectral_delta'], row['predicted_quadratic_delta'], places=12)
        self.assertTrue(row['permutation']['joint_invariant'])
        self.assertTrue(row['permutation']['clean_cd_invariant'])
        self.assertGreater(abs(row['permutation']['y_only_spectral_delta']), .1)
        zero = analysis.analyze_direction(reference, reference, metric, clean)
        self.assertFalse(zero['resolved'])
        self.assertFalse(zero['spectral_decrease_clean_cd_increase'])
        far = np.array([[1000., 0., 0.], [1002., 0., 0.]])
        overshoot = analysis.analyze_direction(far, far+1e-5, metric, clean)
        self.assertTrue(overshoot['finite_step_overshoot'])

    def test_saved_first_probe_or_filter_mismatch_aborts_comparison(self):
        analysis = self.analysis()
        reference = np.array([[1., 0., 0.], [3., 0., 0.]])
        decoded = reference-np.array([1., 0., 0.])
        metric = np.diag([1., .5])
        check = analysis.validate_metric(metric, decoded, reference, .25, metric.copy())
        self.assertTrue(check['passed'])
        with self.assertRaisesRegex(ValueError, 'loss mismatch'):
            analysis.validate_metric(metric, decoded, reference, .250002)
        with self.assertRaisesRegex(ValueError, 'filter mismatch'):
            analysis.validate_metric(metric, decoded, reference, .25, metric+np.eye(2)*2e-6)

    def test_clean_receipt_binds_exact_subset_and_source_pin(self):
        analysis = self.analysis()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset, receipt_path = root/'oracle_clean.npz', root/'oracle_clean_receipt.json'
            np.savez_compressed(subset, index=np.asarray(analysis.IDS, dtype=np.int64),
                                clean_xyz=np.zeros((8, 1024, 3), dtype=np.float32))
            receipt = dict(schema_version=1, status='CPU_oracle_only', full_clean_asset=analysis.CLEAN_IDENTITY,
                           subset=dict(path='/remote/oracle_clean.npz', **analysis.identity(subset)),
                           indices=analysis.IDS, selection_manifest_sha256=analysis.SELECTION_SHA256,
                           usage='posthoc CPU oracle only; never guidance or selection')
            receipt_path.write_text(json.dumps(receipt))
            clean, _ = analysis.load_oracle(subset, receipt_path)
            self.assertEqual(clean.shape, (8, 1024, 3))
            receipt['full_clean_asset'] = dict(analysis.CLEAN_IDENTITY, sha256='0'*64)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'clean asset'):
                analysis.load_oracle(subset, receipt_path)

    def test_archive_row_accounting_and_metric_mismatch_do_not_silently_drop_cases(self):
        import torch
        import graph_spectral
        from legacy_trajectory import tensor_hash
        from types import SimpleNamespace
        analysis = self.analysis()
        selection_bytes = FILE.with_name('development_ids.json').read_bytes()
        points = np.zeros((2, 2048, 3), dtype=np.float32)
        config = dict(dataset='modelnet40_c', selected_original_ids=analysis.IDS, count=8,
            backend_policy='controlled', severity=5, seed=0,
            corruptions=['gaussian', 'background', 'shear', 'lidar'],
            index_manifest_sha256=analysis.SELECTION_SHA256, selection_manifest=json.loads(selection_bytes),
            guidance_protocol=dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False),
            normal_reverse_steps=8, background_reverse_steps=35, freeze_decoder_residual=False,
            physical_graph=dict(k=10, delta=.1, graph_gamma=.6, modes=100, profile='smooth', beta=2.),
            setup_verification=dict(report=dict(assets={'data/modelnet40_c/data_original.npy': analysis.CLEAN_IDENTITY})),
            executed_source_files={'repo/graph_spectral.py': analysis.identity(graph_spectral.__file__)})
        mismatch = False
        class Bundle:
            names = set()
            def read(self, name):
                return selection_bytes
            def json(self, name):
                if name.endswith('probe_records.json'):
                    loss = .1 if mismatch and name.startswith('batches/gaussian/batch0000/') else 0.
                    return [dict(loop_count=9 if '/background/' in name else 2,
                                 spectral_loss_per_object=[loss, 0.])]*3
                return dict(prepared_x_sha256=tensor_hash(torch.from_numpy(points)))
            def close(self):
                pass
        def saved_arrays(bundle, name):
            batch = int(name.split('/')[2][5:])
            ids = np.asarray(analysis.IDS[batch*2:batch*2+2])
            if name.endswith('reference.npz'):
                return dict(prepared_x=points, initial_h0_xyz=points)
            if name.endswith('terminal_geometry.npz'):
                return dict(index=ids, terminal_decoded_xyz=points, terminal_classifier_geometry=points,
                            raw_corrupted_xyz=points, normalization_center=np.zeros((2, 1, 3)),
                            normalization_scale=np.ones((2, 1, 1)))
            return dict(index=ids, decoded_xyz=points)
        # Expensive dense graph construction and per-row distances have separate numerical witnesses above.
        metric = torch.zeros(2048, 2048)
        direction_result = dict(resolved=False, spectral_decrease=False, clean_cd_increase=False,
                                spectral_decrease_clean_cd_increase=False, finite_step_overshoot=None)
        with tempfile.TemporaryDirectory() as directory:
            inputs = [Path(directory)/str(i) for i in range(3)]
            for path in inputs:
                path.write_text('fixed input')
            with patch.object(analysis, 'open_bundle', return_value=(Bundle(), config)), \
                 patch.object(analysis, 'arrays', side_effect=saved_arrays), \
                 patch.object(analysis, 'load_oracle', return_value=(np.zeros((8, 1024, 3)), {})), \
                 patch.object(graph_spectral, 'build_smooth_spectral_target', return_value=SimpleNamespace(filters=[metric, metric])), \
                 patch.object(analysis, 'analyze_direction', return_value=direction_result):
                result = analysis.summarize(*inputs)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['total_rows'], 64)
                self.assertEqual(result['metric_provenance']['saved_gpu_filter_conditions'], 0)
                self.assertTrue(all(not r['operator_comparison_available'] for r in result['rows']))
                mismatch = True
                failed = analysis.summarize(*inputs)
                self.assertEqual(failed['status'], 'analysis_mismatch')
                self.assertEqual(failed['total_rows'], 60)
                self.assertEqual(len(failed['mismatches']), 1)
                self.assertTrue(failed['mismatches'][0]['comparison_aborted'])
                self.assertEqual(failed['aggregates']['gaussian']['first']['total'], 6)
                json.dumps(failed, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
