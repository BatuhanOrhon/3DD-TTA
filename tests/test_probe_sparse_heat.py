"""Primitive CPU artifact/numerical contracts; no model initialization."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph_spectral import SpectralConfig
from scripts import probe_sparse_heat as probe


class SparseHeatProbeTests(unittest.TestCase):
    def test_cuda_backend_recipe_precedes_input_read_and_rejects_initialized_cuda(self):
        before = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32,
                      benchmark=torch.backends.cudnn.benchmark, deterministic=torch.backends.cudnn.deterministic,
                      algorithms=torch.are_deterministic_algorithms_enabled())
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                geometry, assets = self.fixture(root)
                original_identity = probe.identity
                checks = []
                def inspect_identity(path):
                    if Path(path) == geometry:
                        checks.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                            torch.are_deterministic_algorithms_enabled()))
                    return original_identity(path)
                # Wrong geometry SHA stops before any device allocation.
                with patch.object(torch.cuda, 'is_initialized', return_value=False), \
                        patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('unexpected CUDA init')), \
                        patch.object(probe, 'identity', side_effect=inspect_identity):
                    with self.assertRaisesRegex(ValueError, 'geometry identity'):
                        probe.main(['--geometry', str(geometry), '--geometry-sha256', '0'*64,
                                    '--asset-manifest', str(assets), '--asset-manifest-sha256', original_identity(assets)['sha256'],
                                    '--output', str(root/'cuda_config'), '--device', 'cuda'])
                self.assertEqual(checks, [(False, False, False, True, False)])
                report = json.loads((root/'cuda_config'/'report.json').read_text())
                self.assertTrue(report['backend']['configured_before_cuda_initialization'])
            with patch.object(torch.cuda, 'is_initialized', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'before CUDA initialization'):
                    probe.apply_controlled_backend()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = before['matmul']
            torch.backends.cudnn.allow_tf32 = before['cudnn']
            torch.backends.cudnn.benchmark = before['benchmark']
            torch.backends.cudnn.deterministic = before['deterministic']
            torch.use_deterministic_algorithms(before['algorithms'])

    def test_same_graph_all_orders_bounds_isolates_and_saved_native_arrays(self):
        x = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [.1, 0., 0.],
                           [1., 0., 0.], [1.1, 0., 0.], [100., 0., 0.]]], dtype=torch.float64)
        e = torch.arange(x.numel(), dtype=x.dtype).reshape_as(x)/7
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rows = probe.check_equivalence(x, e[:, None], SpectralConfig(k=2, delta=.15), path)
            self.assertEqual([r['order'] for r in rows['orders']], [8, 16, 32])
            self.assertTrue(rows['same_graph_passed'])
            self.assertTrue(rows['equivalence_passed'])
            self.assertIn(rows['first_acceptable_order'], (8, 16))
            for row in rows['orders']:
                self.assertEqual(row['objects'][0]['isolate_gradient_max_abs'], 0)
                self.assertLessEqual(row['objects'][0]['absolute_loss_error'],
                                     row['objects'][0]['ideal_loss_error_bound']+1e-12)
            with np.load(path/'native_graph_and_residuals.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['basis_0'].dtype, np.float64)
                np.testing.assert_array_equal(saved['residuals'], e[:, None].numpy())
                raw = saved['raw_laplacian_0']
                np.testing.assert_allclose(raw/saved['mean_degree_0'], saved['normalized_laplacian_0'], atol=0, rtol=0)
            self.assertEqual(rows['numeric_gates']['relative_loss_tolerance'], 1e-6)

    def test_no_order_passes_when_native_quadratic_is_perturbed_and_partial_graph_survives_failure(self):
        x = torch.arange(36, dtype=torch.float64).reshape(1, 12, 3)/40
        residuals = torch.ones_like(x)[:, None]/10
        original = probe.sparse_heat.SparseHeatFilter.loss
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed, partial = root/'failed', root/'partial'
            failed.mkdir(); partial.mkdir()
            with patch.object(probe.sparse_heat.SparseHeatFilter, 'loss', lambda self, e: original(self, e)*1.01):
                report = probe.check_equivalence(x, residuals, SpectralConfig(), failed)
            self.assertFalse(report['equivalence_passed'])
            self.assertIsNone(report['first_acceptable_order'])
            self.assertEqual(len(report['orders']), 3)
            with patch.object(probe.sparse_heat.SparseHeatFilter, 'loss', side_effect=RuntimeError('native failed')):
                with self.assertRaisesRegex(RuntimeError, 'native failed'):
                    probe.check_equivalence(x, residuals, SpectralConfig(), partial)
            self.assertTrue((partial/'native_graph_and_residuals.npz').is_file())
            self.assertEqual(json.loads((partial/'equivalence.json').read_text())['orders'], [])

    def test_strict_hash_failure_retains_protocol_and_failure_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry, assets = self.fixture(root)
            output = root/'failed'
            with self.assertRaisesRegex(ValueError, 'geometry identity'):
                probe.main(['--geometry', str(geometry), '--geometry-sha256', '0'*64,
                            '--asset-manifest', str(assets), '--asset-manifest-sha256', probe.identity(assets)['sha256'],
                            '--output', str(output), '--device', 'cpu'])
            self.assertEqual(json.loads((output/'report.json').read_text())['status'], 'failed')
            self.assertTrue((output/'protocol.json').is_file())
            with self.assertRaises(FileExistsError):
                probe.main(['--geometry', str(geometry), '--geometry-sha256', '0'*64,
                            '--asset-manifest', str(assets), '--asset-manifest-sha256', '0'*64,
                            '--output', str(output), '--device', 'cpu'])

    def test_cpu_toy_cli_records_replication_and_predeclared_timing_scopes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry, assets = self.fixture(root)
            output = root/'complete'
            # A deployed overlay need not be a git checkout.
            with patch.object(probe, 'REPO', root):
                probe.main(['--geometry', str(geometry), '--geometry-sha256', probe.identity(geometry)['sha256'],
                            '--asset-manifest', str(assets), '--asset-manifest-sha256', probe.identity(assets)['sha256'],
                            '--output', str(output), '--device', 'cpu'])
            report = json.loads((output/'report.json').read_text())
            self.assertIn(report['status'], ('complete', 'equivalence_gate_failed'))
            self.assertFalse(report['gpu_executed'])
            self.assertIsNone(report['checkout_head'])
            self.assertEqual([b['batch_size'] for b in report['timings']], [1, 32])
            for batch in report['timings']:
                self.assertEqual(len(batch['replicated_input_positions']), batch['batch_size'])
                self.assertEqual(len(batch['exact_setup']['samples']), 3)
                for arm in batch['filtering']:
                    self.assertEqual(len(arm['samples']), 10)
                    self.assertIsNone(arm['samples'][0]['peak_allocated_bytes'])
            self.assertEqual(len(report['equivalence']['orders'][0]['objects']), 8)
            self.assertTrue((output/'artifact_manifest.json').is_file())

    def test_second_timing_batch_failure_preserves_completed_first_batch_scopes(self):
        original = probe.build_probe_spectral_target
        def fail_b32(x, config):
            if len(x) == 32:
                raise RuntimeError('simulated B32 OOM')
            return original(x, config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry, assets = self.fixture(root)
            output = root/'partial_timing'
            with patch.object(probe, 'build_probe_spectral_target', side_effect=fail_b32):
                with self.assertRaisesRegex(RuntimeError, 'simulated B32 OOM'):
                    probe.main(['--geometry', str(geometry), '--geometry-sha256', probe.identity(geometry)['sha256'],
                                '--asset-manifest', str(assets), '--asset-manifest-sha256', probe.identity(assets)['sha256'],
                                '--output', str(output), '--device', 'cpu'])
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            scopes = report['completed_timing_scopes']
            self.assertEqual(len(scopes), 7)  # two setups, four filtering arms, batch summary
            self.assertTrue(all(v['batch_size'] == 1 for v in scopes))
            for entry in scopes:
                self.assertEqual(probe.identity(output/entry['file']), entry['identity'])
            batch = json.loads((output/'timing_b1_batch.json').read_text())
            self.assertEqual(len(batch['filtering']), 4)
            self.assertEqual(len(batch['exact_setup']['samples']), 3)

    @staticmethod
    def fixture(root):
        # Default k=10 requires N>10; all four captured checkpoint geometries.
        x = np.arange(72, dtype=np.float64).reshape(2, 12, 3)/80
        y = np.stack([x+.01*(i+1) for i in range(4)], axis=1)
        geometry = root/'h5_geometry.npz'
        np.savez_compressed(geometry, index=np.array([44, 91], dtype=np.int64),
                            prepared_x=x, decoded_xyz=y,
                            candidate_names=np.array(['clean_02', 'clean_04', 'clean_06', 'terminal']),
                            candidate_loop_counts=np.array([2, 4, 6, 8], dtype=np.int64))
        assets = root/'asset_manifest.json'
        assets.write_text(json.dumps({'source_overlay_receipt': {'sha256': 'test'}}))
        return geometry, assets


if __name__ == '__main__':
    unittest.main()
