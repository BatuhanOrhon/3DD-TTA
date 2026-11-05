"""CPU posthoc analysis contracts; no native model or synthetic research result."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from scripts import analyze_gsd_shared_probes as analysis


class SharedAnalysisTests(unittest.TestCase):
    def test_cluster_means_count_objects_not_dependent_rows(self):
        rows = [dict(object_id=459, delta=1.)]*9+[dict(object_id=545, delta=-3.)]
        result = analysis.cluster_summary(rows, 'delta')
        self.assertEqual(result['n_original_objects'], 2)
        self.assertEqual(result['equal_object_mean'], -1.)
        self.assertEqual(result['positive_objects'], 1)
        self.assertIsNone(analysis.cluster_summary([], 'delta')['equal_object_mean'])

    def test_accepted_physical_frame_and_set_distance_are_permutation_invariant(self):
        clean = np.asarray([[0., 1., 2.], [2., 3., 4.]])
        center, scale = np.asarray([[1., 1., 1.]]), np.asarray([[2.]])
        physical = analysis.oracle.physical_clean(clean, center, scale)
        np.testing.assert_allclose(physical[0], [-1.69425, 0., -1.69425])
        self.assertEqual(analysis.oracle.squared_chamfer(physical, physical[::-1]), 0.)
        with self.assertRaises(ValueError):
            analysis.oracle.physical_clean(clean, center, np.zeros((1, 1)))

    def test_zip_pin_manifest_path_tamper_and_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            raw = b'{"preserved":true}\n'
            archive_path = root/'good.zip'
            digest = __import__('hashlib').sha256(raw).hexdigest()
            with zipfile.ZipFile(archive_path, 'x') as archive:
                archive.writestr('payload.json', raw)
                archive.writestr('evidence_manifest.json', json.dumps({'payload.json': dict(bytes=len(raw), sha256=digest)}))
            expected = analysis.source.file_identity(archive_path)['sha256']
            out, observed = analysis.checked_extract(archive_path, expected, root/'checked', 'evidence_manifest.json')
            self.assertEqual((out/'payload.json').read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                analysis.checked_extract(archive_path, expected, root/'checked', 'evidence_manifest.json')
            with self.assertRaises(ValueError):
                analysis.checked_extract(archive_path, '0'*64, root/'wrong', 'evidence_manifest.json')
            bad = root/'bad.zip'
            with zipfile.ZipFile(bad, 'x') as archive:
                archive.writestr('../escape', b'bad')
            with self.assertRaises(ValueError):
                analysis.checked_extract(bad, analysis.source.file_identity(bad)['sha256'], root/'unsafe', 'evidence_manifest.json')
            analysis.write(root/'immutable.json', {})
            with self.assertRaises(FileExistsError):
                analysis.write(root/'immutable.json', {})
        ast.parse(Path(analysis.__file__).read_text(), feature_version=(3, 8))
        ast.parse(Path(__file__).read_text(), feature_version=(3, 8))

    def test_readout_raw_keys_and_true_margin_join_without_unpaired_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            capture, readouts = root/'capture', root/'readouts'
            capture.mkdir(); readouts.mkdir()
            analysis.write(capture/'config.json', {})
            for corr in analysis.CORRUPTIONS:
                for batch in range(2):
                    ids = analysis.IDS[batch*2:batch*2+2]
                    path = capture/'batches'/corr/('batch%04d' % batch)
                    path.mkdir(parents=True)
                    np.savez(path/'predictions.npz', index=np.asarray(ids), label=np.asarray(ids)%40)
            ids = analysis.IDS[:2]
            logits = np.zeros((2, 40), dtype=np.float32)
            for i, object_id in enumerate(ids):
                logits[i, object_id%40] = 5
            payload = dict(index=np.asarray(ids), logits=logits, prediction=logits.argmax(1),
                corruption=np.asarray(['gaussian']*2), loop_count=np.asarray([2]*2),
                branch=np.asarray(['C']*2), fraction=np.asarray([.5]*2), next_timestep=np.asarray([50]*2))
            np.savez(readouts/'readout.npz', **payload)
            cloud_id = dict(bytes=12, sha256='a'*64)
            key = 'gaussian/batch0000/loop02/branch_C_0p5'
            receipt = dict(status='complete', readout_file='readout.npz', key=key, cloud_identity=cloud_id,
                original_ids=ids, corruption='gaussian', loop_count=2, arm='C', fraction=.5, next_timestep=50)
            report = dict(status='complete_with_unavailable', inputs=dict(mechanism_report={'sha256': 'pinned'},
                capture_config=analysis.source.file_identity(capture/'config.json')), readouts=[receipt],
                artifacts={'readout.npz': analysis.source.file_identity(readouts/'readout.npz')})
            analysis.write(readouts/'complete.json', report)
            coverage = [dict(key=key, status='available', cloud_sha256=cloud_id)]
            result = analysis.analyze_readouts(readouts, {'sha256': 'pinned'}, capture, coverage)
            self.assertEqual(result['summaries']['C_0.5']['correct'], 2)
            self.assertEqual(result['summaries']['C_0.5']['paired_C_n'], 2)
            self.assertEqual(result['summaries']['C_0.5']['clusters']['margin_delta_vs_C']['equal_object_mean'], 0.)
            payload['branch'] = np.asarray(['S_lat']*2)
            np.savez(readouts/'readout.npz', **payload)
            report['artifacts']['readout.npz'] = analysis.source.file_identity(readouts/'readout.npz')
            (readouts/'complete.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'payload/receipt key'):
                analysis.analyze_readouts(readouts, {'sha256': 'pinned'}, capture, coverage)


if __name__ == '__main__':
    unittest.main()
