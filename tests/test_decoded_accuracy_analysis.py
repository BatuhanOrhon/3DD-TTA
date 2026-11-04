"""Native-free acceptance witnesses for paired accuracy evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np

FILE = Path(__file__).resolve().parents[1]/'knowledge/evidence/decoded_physical_accuracy_20261104/analyze_accuracy.py'


class AccuracyAnalysisTests(unittest.TestCase):
    def analysis(self):
        self.assertTrue(FILE.is_file(), 'Accuracy artifact analyzer missing')
        spec = importlib.util.spec_from_file_location('accuracy_analysis', FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_independent_stable_pm_recount_rejects_false_stored_prediction(self):
        analysis = self.analysis()
        logits = np.array([[[1000., 0.]]*3+[[0., 1000.]], [[0., 1000.]]*4])
        pm, terminal = analysis.recount(logits)
        np.testing.assert_array_equal(pm, [0, 1])
        np.testing.assert_array_equal(terminal, [1, 1])
        payload = dict(trajectory_logits=logits, prediction=pm, probability_mean=pm,
                       terminal_prediction=terminal, logits=logits[:, -1])
        analysis.validate_predictions(payload)
        payload['prediction'] = np.array([1, 1])
        with self.assertRaisesRegex(ValueError, 'PM'):
            analysis.validate_predictions(payload)

    def test_real_sealed_bundle_sha_tampering_is_rejected(self):
        analysis = self.analysis()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tampered.zip'
            original = b'original'
            manifest = {'config.json': dict(bytes=len(original), sha256=hashlib.sha256(original).hexdigest())}
            with zipfile.ZipFile(path, 'x') as archive:
                archive.writestr('run/config.json', b'tampered')
                archive.writestr('run/bundle_manifest.json', json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'SHA'):
                analysis.open_bundle(path)

    def test_pairing_and_per_checkpoint_repeat_gate_detect_distinct_failures(self):
        import copy
        analysis = self.analysis()
        logits = np.zeros((2, 4, 40))
        case = dict(identity={'prepared': 'same', 'rng': 'fixed'}, index=np.array([44, 91]),
                    label=np.array([0, 0]), logits=logits, first=np.zeros((2, 2, 3)),
                    pm=np.zeros(2, dtype=np.int64), terminal=np.zeros(2, dtype=np.int64))
        control = dict(control_identity={'source': 'fixed'}, cases={'gaussian/0': case})
        other = copy.deepcopy(control)
        analysis.check_pairing(control, other)
        self.assertTrue(analysis.repeat_gate(control, other)['passed'])
        other['cases']['gaussian/0']['logits'][0, 2] += .011
        gate = analysis.repeat_gate(control, other)
        self.assertFalse(gate['passed'])
        self.assertLess(gate['logit_max_absolute'], .05)
        self.assertGreater(gate['logit_per_object_checkpoint_rms_max'], .01)
        other['cases']['gaussian/0']['identity']['prepared'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'identity'):
            analysis.check_pairing(control, other)

    def test_bootstrap_resamples_two_objects_with_all_corruptions_and_one_real_seed(self):
        analysis = self.analysis()
        winner = np.ones((15, 2), dtype=bool)
        baseline = np.ones((15, 2), dtype=bool)
        baseline[:, 0] = False
        result = analysis.paired_metrics(winner, baseline)
        self.assertEqual(result['delta_pp'], 50.)
        self.assertEqual(result['object_cluster_ci95_pp'], [0., 100.])
        self.assertEqual(result['corrected'], 15)
        self.assertEqual(result['harmed'], 0)
        self.assertEqual(result['seeds'], [0])
        self.assertEqual(result['bootstrap_repetitions'], 10000)
        with self.assertRaises(ValueError):
            analysis.paired_metrics(np.tile(winner, (3, 1, 1)), np.tile(baseline, (3, 1, 1)))

    def test_physical_metric_hash_drift_is_reported_as_failed_gate(self):
        analysis = self.analysis()
        metric = dict(reference_xyz_sha256='0'*64, filter_sha256=['1'*64, '2'*64])
        runs = {arm: dict(cases={'gaussian/0': dict(physical_metric_identity=dict(metric))})
                for arm in ('live0', 'full', 'h8')}
        self.assertTrue(analysis.physical_metric_gate(runs)['passed'])
        runs['h8']['cases']['gaussian/0']['physical_metric_identity']['filter_sha256'] = ['3'*64, '2'*64]
        result = analysis.physical_metric_gate(runs)
        self.assertFalse(result['passed'])
        self.assertEqual(result['mismatches'][0]['arm'], 'h8')

    def test_main_uses_disjoint32_objects_and_one_seed_accuracy_matrices(self):
        import copy
        analysis = self.analysis()
        runs, mapping = {}, {}
        for shard in range(4):
            name = 'shard%d'%shard
            mapping[name] = {}
            ids = analysis.COHORT_IDS[8*shard:8*shard+8]
            for arm in analysis.ARMS:
                key = name+'-'+arm
                mapping[name][arm] = key
                cases = {}
                for corruption in analysis.CORRUPTIONS:
                    for batch in range(4):
                        index = np.asarray(ids[2*batch:2*batch+2])
                        prediction = np.asarray([int(i == analysis.COHORT_IDS[0] and arm in ('full', 'h8')) for i in index])
                        cases[corruption+'/'+str(batch)] = dict(
                            identity={'prepared': index.tolist()}, index=index, label=np.zeros(2, dtype=int),
                            logits=np.zeros((2, 4, 40)), first=np.zeros((2, 2, 3)), pm=prediction, terminal=prediction,
                            corruption=corruption, batch=batch, physical_metric_identity=None if arm == 'control' else
                                dict(reference_xyz_sha256='0'*64, filter_sha256=['1'*64, '2'*64]))
                runs[key] = dict(path=key, archive_identity={}, arm=arm, ids=ids, corruptions=list(analysis.CORRUPTIONS),
                    selection_manifest=dict(parent_manifest_sha256=analysis.COHORT_SHA256, shard=shard),
                    control_identity={'source': 'fixed'}, cases=cases,
                    counts={c: dict(reconstruction_seconds=1., peak_gpu_memory_mb=100.) for c in analysis.CORRUPTIONS},
                    costs=[dict(case_elapsed_seconds=1., graph_diagnostic_seconds=0.)], worker_wall_seconds=20.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stage_paths.json'
            path.write_text(json.dumps(mapping))
            with patch.object(analysis, 'load_run', side_effect=lambda p: runs[p]):
                report = analysis.main_analysis(path)
                self.assertEqual(report['status'], 'complete')
                self.assertEqual(report['object_conditions_per_arm'], 480)
                self.assertEqual(report['seeds'], [0])
                self.assertEqual(report['comparisons']['full_vs_live0']['pm']['delta_pp'], -3.125)
                self.assertEqual(report['comparisons']['full_vs_live0']['pm']['harmed'], 15)
                self.assertEqual(report['comparisons']['h8_vs_full']['pm']['delta_pp'], 0.)
                self.assertTrue(report['numerical_attribution_gate_passed'])
                json.dumps(report, allow_nan=False)
                runs['shard1-control'] = copy.deepcopy(runs['shard1-control'])
                runs['shard1-control']['ids'] = runs['shard0-control']['ids']
                with self.assertRaisesRegex(ValueError, 'scope differs'):
                    analysis.main_analysis(path)


if __name__ == '__main__':
    unittest.main()
