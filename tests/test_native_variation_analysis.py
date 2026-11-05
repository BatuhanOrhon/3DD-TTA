"""Small sealed-ZIP and numeric localization witnesses; no native inference."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from legacy_trajectory import tensor_hash

FILE = Path(__file__).resolve().parents[1]/'scripts/analyze_native_variation.py'


class NativeVariationAnalysisTests(unittest.TestCase):
    def analysis(self):
        self.assertTrue(FILE.is_file(), 'Native variation analyzer missing')
        spec = importlib.util.spec_from_file_location('native_variation_analysis', FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self):
        analysis = self.analysis()
        n, ids = 4, [44, 91]
        reference = dict(prepared_x=np.zeros((2, n, 3), np.float32), initial_h0_xyz=np.zeros((2, n, 3), np.float32))
        noisy = np.ones((2, 4*n, 1, 1), np.float32)
        style = np.ones((2, 128, 1, 1), np.float32)
        initial = dict(noisy_local=analysis.tensor_sha(noisy), style=analysis.tensor_sha(style), shape_latent=analysis.tensor_sha(style))
        case = dict(index=np.asarray(ids), identity=dict(scheduler=dict(active_timesteps=[20, 10]), initial=initial, initial_rng='a'*64))
        report = dict(schema_version=1, original_ids=ids, loop_limit=2, complete=True,
                      extra_model_forward_calls=0, extra_chamfer_forward_calls=0, extra_backward_calls=0, snapshots=[])
        files = {}
        for loop, timestep in enumerate([20, 10], 1):
            raw = np.tile(np.arange(n, dtype=np.float32), (2, 1))
            indices = np.tile(np.arange(n), (2, 1))
            fields = dict(
                prior_input=dict(noisy_local=noisy, conditioning_style=style, original_decoder_style=style,
                                 original_h0_xyz=reference['initial_h0_xyz'], prior_t=np.full(2, timestep+1, np.float32)),
                scheduler=dict(noise_pred=noisy*.1, pred_original_sample=noisy*.9, prev_sample=noisy*.99),
                scd=dict(scd_loss=np.asarray(6., np.float32)),
                gradients=dict(noisy_local_gradient=noisy*.1, conditioning_style_gradient=style*.1),
                post_update=dict(noisy_local=noisy*.987, conditioning_style=style*.9997))
            for direction in (1, 2):
                fields['scd'].update({
                    'distances_%d'%direction: raw, 'nearest_indices_%d'%direction: indices.astype(np.int32),
                    'sorted_distances_%d'%direction: raw, 'retained_indices_%d'%direction: indices[:, :3],
                    'cutoff_distance_%d'%direction: raw[:, 2], 'next_distance_%d'%direction: raw[:, 3]})
            for boundary in analysis.BOUNDARIES:
                name = 'native_trace_loop%02d_%s.npz'%(loop, boundary)
                arrays = dict(index=np.asarray(ids, np.int64), **fields[boundary])
                stream = io.BytesIO(); np.savez_compressed(stream, **arrays); raw_bytes = stream.getvalue()
                files[name] = raw_bytes
                metadata = {}
                for key, value in fields[boundary].items():
                    tensor = torch.from_numpy(value)
                    entry = dict(available=True, shape=list(value.shape), dtype=str(tensor.dtype), sha256=tensor_hash(tensor),
                                 finite=True, norm=float(tensor.double().norm()))
                    if value.ndim and value.shape[0] == 2:
                        entry['norm_per_object'] = tensor.double().reshape(2, -1).norm(dim=1).tolist()
                    metadata[key] = entry
                report['snapshots'].append(dict(loop_count=loop, timestep=timestep, boundary=boundary, file=name,
                    identity=dict(bytes=len(raw_bytes), sha256=hashlib.sha256(raw_bytes).hexdigest()), tensors=metadata,
                    rng_before_sha256='a'*64, rng_after_sha256='a'*64, rng_unchanged=True, copy_and_serialization_seconds=0.))
        files['native_trace.json'] = json.dumps(report).encode()
        return analysis, case, reference, files

    def bundle(self, root, files, name):
        path = root/name
        manifest = {key: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) for key, value in files.items()}
        with zipfile.ZipFile(path, 'x') as archive:
            for key, value in files.items():
                archive.writestr('run/'+key, value)
            archive.writestr('run/bundle_manifest.json', json.dumps(manifest))
        return path

    def test_small_sealed_trace_and_resealed_tensor_tampering(self):
        analysis, case, reference, files = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = analysis.accepted.open_bundle(self.bundle(root, files, 'valid.zip'))
            try:
                trace = analysis.load_trace(bundle, '', case, reference)
                self.assertEqual(len(trace), 10)
            finally:
                bundle.close()
            altered = dict(files)
            key = 'native_trace_loop01_scheduler.npz'
            with np.load(io.BytesIO(altered[key]), allow_pickle=False) as saved:
                values = {name: saved[name] for name in saved.files}
            values['noise_pred'][0, 0] += .1
            stream = io.BytesIO(); np.savez_compressed(stream, **values); altered[key] = stream.getvalue()
            report = json.loads(altered['native_trace.json'])
            report['snapshots'][1]['identity'] = dict(bytes=len(altered[key]), sha256=hashlib.sha256(altered[key]).hexdigest())
            altered['native_trace.json'] = json.dumps(report).encode()
            bundle = analysis.accepted.open_bundle(self.bundle(root, altered, 'tampered.zip'))
            try:
                with self.assertRaisesRegex(ValueError, 'Tensor hash'):
                    analysis.load_trace(bundle, '', case, reference)
            finally:
                bundle.close()

    def test_first_numeric_boundary_exact_input_candidate_and_propagated_input(self):
        analysis, case, reference, files = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            bundle = analysis.accepted.open_bundle(self.bundle(Path(temporary), files, 'valid.zip'))
            try:
                left = analysis.load_trace(bundle, '', case, reference)
            finally:
                bundle.close()
        right = copy.deepcopy(left)
        right[1]['arrays']['noise_pred'][0, 0] += np.float32(.001)
        report, rows = analysis.compare_traces(left, right, case['index'].tolist())
        self.assertEqual(report['earliest_numeric_difference']['boundary'], 'scheduler')
        self.assertEqual(report['earliest_exact_input_candidate']['operator'], 'local_prior_forward')
        self.assertTrue(report['earliest_exact_input_candidate']['inputs_bit_exact'])
        right[0]['arrays']['noisy_local'][0, 0] += np.float32(.001)
        report, _ = analysis.compare_traces(left, right, case['index'].tolist())
        self.assertEqual(report['earliest_numeric_difference']['boundary'], 'prior_input')
        self.assertFalse(next(row for row in report['operator_observations'] if row['operator'] == 'local_prior_forward')['inputs_bit_exact'])

    def test_discrete_counts_and_signed_zero_hash_are_separate_from_numeric_difference(self):
        analysis = self.analysis()
        result = analysis.compare_arrays(np.asarray([[0, 1], [2, 3]], np.int32), np.asarray([[1, 1], [2, 4]], np.int32))
        self.assertEqual(result['discrete_changed_count'], 2)
        self.assertEqual(result['changed_count_per_object'], [1, 1])
        signed = analysis.compare_arrays(np.asarray([0.], np.float32), np.asarray([-0.], np.float32))
        self.assertFalse(signed['hash_equal'])
        self.assertFalse(signed['numeric_difference'])

    def test_missing_reordered_boundary_and_rng_guard_fail_before_comparison(self):
        analysis, case, reference, files = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in ('missing', 'reordered', 'rng'):
                changed = dict(files)
                report = json.loads(changed['native_trace.json'])
                if label == 'missing':
                    report['snapshots'].pop()
                elif label == 'reordered':
                    report['snapshots'][0], report['snapshots'][1] = report['snapshots'][1], report['snapshots'][0]
                else:
                    report['snapshots'][0]['rng_after_sha256'] = 'b'*64
                changed['native_trace.json'] = json.dumps(report).encode()
                bundle = analysis.accepted.open_bundle(self.bundle(root, changed, label+'.zip'))
                try:
                    with self.subTest(label=label), self.assertRaises(ValueError):
                        analysis.load_trace(bundle, '', case, reference)
                finally:
                    bundle.close()

    def test_fresh_output_failure_preserves_evidence_and_refuses_overwrite(self):
        analysis = self.analysis()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'analysis'
            with self.assertRaises(FileNotFoundError):
                analysis.analyze(['missing.zip']*6, output)
            failure = (output/'failure.json').read_bytes()
            self.assertEqual(json.loads(failure)['status'], 'failed')
            with self.assertRaises(FileExistsError):
                analysis.analyze(['missing.zip']*6, output)
            self.assertEqual((output/'failure.json').read_bytes(), failure)

    def test_producer_source_aliases_bind_bytes_and_reject_tamper_or_unsafe_paths(self):
        analysis = self.analysis()
        raw = b'value = 1\n'
        with tempfile.TemporaryDirectory() as temporary:
            bundle = analysis.accepted.open_bundle(self.bundle(Path(temporary), {'source_snapshot/probe.py': raw}, 'source.zip'))
            try:
                sources = {name: analysis.byte_identity(raw) for name in ('repo/probe.py', 'assets/source_snapshot/probe.py')}
                analysis.verify_executed_sources(bundle, sources)
                with self.assertRaisesRegex(ValueError, 'snapshot differs'):
                    analysis.verify_executed_sources(bundle, {'repo/probe.py': analysis.byte_identity(b'tampered')})
                for name in ('arbitrary/source_snapshot/probe.py', 'repo/../probe.py', 'repo//probe.py',
                             'repo/a/./probe.py', 'repo/a\\probe.py', 'repo/C:probe.py', 'repo/'):
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        analysis.verify_executed_sources(bundle, {name: analysis.byte_identity(raw)})
            finally:
                bundle.close()


if __name__ == '__main__':
    unittest.main()
