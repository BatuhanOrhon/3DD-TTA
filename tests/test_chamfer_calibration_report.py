"""Dependency-free shared acceptance math; no GPU or label data."""
import copy
import unittest

from chamfer_calibration_report import (CALIBRATION_RULE, LOSS_REDUCTION,
                                        ROWS_PER_CANDIDATE, validate_calibration_report)


def valid_report():
    medians = dict(semantic_kl=.5, semantic_h=1. / 3., weighted=2.)
    return dict(schema_version=1, purpose='label_free_chamfer_calibration', status='verified_label_free_gpu',
                target_semantic_update_ratio=.1, rule=CALIBRATION_RULE, loss_reduction=LOSS_REDUCTION,
                coefficients=dict(semantic_kl=.2, semantic_h=.3, weighted_scale=.5),
                ratios={name: dict(count=ROWS_PER_CANDIDATE, missing=0, min=value, median=value, max=value, p90=value)
                        for name, value in medians.items()},
                calibration_bundles={name: {} for name in ('modelnet40_c', 'scanobjectnn_c')})


class CalibrationReportContracts(unittest.TestCase):
    def test_validated_report_is_returned_without_mutation(self):
        report = valid_report()
        before = copy.deepcopy(report)
        self.assertIs(validate_calibration_report(report), report)
        self.assertEqual(report, before)

    def test_all_candidates_require_complete_ordered_finite_summary(self):
        for candidate in ('semantic_kl', 'semantic_h', 'weighted'):
            for key, value in (('count', 4199), ('count', 4200.), ('missing', 1), ('median', None),
                               ('median', 0.), ('median', float('nan')), ('median', 1e-15),
                               ('p90', float('inf')), ('min', 100.)):
                report = valid_report()
                report['ratios'][candidate][key] = value
                with self.subTest(candidate=candidate, key=key, value=value), self.assertRaises(ValueError):
                    validate_calibration_report(report)

    def test_global_formula_reduction_target_and_every_coefficient_are_frozen(self):
        for key, value in (('rule', 'changed'), ('loss_reduction', 'mean'), ('target_semantic_update_ratio', .2),
                           ('schema_version', True)):
            report = valid_report()
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_calibration_report(report)
        for key in ('semantic_kl', 'semantic_h', 'weighted_scale'):
            for value in (0., True, float('inf'), float('nan'), .71):
                report = valid_report()
                report['coefficients'][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_calibration_report(report)
        for group, key in (('coefficients', 'semantic_h'), ('ratios', 'semantic_h')):
            report = valid_report()
            report[group].pop(key)
            with self.assertRaises(ValueError):
                validate_calibration_report(report)


if __name__ == '__main__':
    unittest.main()
