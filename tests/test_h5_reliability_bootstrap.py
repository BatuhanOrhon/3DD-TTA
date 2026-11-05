"""Whole-object paired nonlinear bootstrap and pinned input witnesses."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class H5BootstrapTests(unittest.TestCase):
    def module(self):
        from scripts import bootstrap_h5_reliability
        return bootstrap_h5_reliability

    def row(self, identity, number, label, combined=.8):
        module = self.module()
        return dict(object_id=identity, repeat=0, case='gaussian/0', corruption='gaussian',
                    checkpoint_index=number, band_index=0, oracle_benefit=label,
                    clean_cd_delta=-1. if label else 1.,
                    scores=dict(zip(module.h5.SCORES, [.5, .6, combined])))

    def test_whole_clusters_duplicate_and_all_scores_share_draw_mask_and_order(self):
        module = self.module()
        rows = [self.row(1, 0, True, None), self.row(1, 1, True, .9), self.row(1, 2, False, .1),
                self.row(2, 1, False, .8)]
        draws = np.array([[0, 0], [1, 0]], dtype=np.int64)
        report, raw = module.cluster_bootstrap(rows, [1, 2], draws)
        common = rows[1:]
        self.assertEqual(report['common_eligible_rows'], 3)
        np.testing.assert_array_equal(raw['sampled_object_ids'], [[1, 1], [2, 1]])
        np.testing.assert_array_equal(raw['sampled_common_rows'], [4, 3])
        for draw, sampled in enumerate(([rows[1], rows[2]]*2, [rows[3], rows[1], rows[2]])):
            expected = module.h5.score_comparison(sampled)
            self.assertEqual(raw['draw_common_ordered_row_keys_sha256'][draw], expected['common_ordered_row_keys_sha256'])
            for column, score in enumerate(module.h5.SCORES):
                self.assertEqual(raw['auroc'][draw, column], expected['scores'][score]['auroc'])
            self.assertAlmostEqual(raw['paired_auroc_differences'][draw, 0], raw['auroc'][draw, 2]-raw['auroc'][draw, 0])
        self.assertEqual(report['common_ordered_row_keys_sha256'], module.h5.score_comparison(common)['common_ordered_row_keys_sha256'])

    def test_one_class_draws_are_unavailable_and_cannot_advance(self):
        module = self.module()
        rows = [self.row(1, 1, True), self.row(2, 1, True)]
        report, raw = module.cluster_bootstrap(rows, [1, 2], np.array([[0, 0], [0, 1]], np.int64))
        self.assertTrue(np.isnan(raw['auroc']).all())
        self.assertEqual(report['one_class_replicates'], 2)
        self.assertIsNone(report['combined_auroc_ci95'])
        self.assertFalse(module.advance_rule(True, report))

    def test_coverage_pins_and_fresh_failure_directory(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features, rows = [], []
            for repeat in range(3):
                for position, identity in enumerate(module.h5.IDS):
                    for corruption in module.h5.CORRUPTIONS:
                        for checkpoint in range(4):
                            for band in range(6):
                                row = self.row(identity, checkpoint, bool(position % 2), None if checkpoint == 0 else .8)
                                row.update(repeat=repeat, corruption=corruption, case=corruption+'/'+str(position//2), band_index=band,
                                           loop_count=([9, 18, 27, 35] if corruption == 'background' else [2, 4, 6, 8])[checkpoint])
                                features.append(copy.deepcopy(row)); rows.append(copy.deepcopy(row))
            (root/'features.json').write_text(json.dumps(features))
            (root/'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            summary = dict(status='complete', rows=2304, ids=module.h5.IDS, corruptions=list(module.h5.CORRUPTIONS),
                repeats=3, seed=0, native_repeat_gates_all_passed=True,
                feature_identity=module.h5.accepted.identity(root/'features.json'),
                pooled_descriptive_comparison=module.h5.score_comparison(rows))
            (root/'summary.json').write_text(json.dumps(summary))
            pins = {name: module.h5.accepted.identity(root/name)['sha256'] for name in module.INPUT_NAMES}
            loaded, _ = module.load_analysis(root, pins)
            self.assertEqual(len(loaded), 2304)
            rows[-1] = rows[-2]
            (root/'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'SHA'):
                module.load_analysis(root, pins)
            changed = dict(pins, **{'rows.jsonl': module.h5.accepted.identity(root/'rows.jsonl')['sha256']})
            with self.assertRaisesRegex(ValueError, 'coverage'):
                module.load_analysis(root, changed)
            output = root/'failure'
            with self.assertRaisesRegex(ValueError, 'SHA'):
                module.analyze(root, pins, output)
            failure = (output/'failure.json').read_bytes()
            with self.assertRaises(FileExistsError):
                module.analyze(root, pins, output)
            self.assertEqual((output/'failure.json').read_bytes(), failure)


if __name__ == '__main__':
    unittest.main()
