"""Shared, dependency-free acceptance rule for frozen Chamfer calibration math."""
import math


CALIBRATION_RULE = 'semantic=0.1/median(candidate/C combined update norm); weighted_scale=1/median(weighted/C)'
LOSS_REDUCTION = 'sum for Chamfer, weighted Chamfer, and per-object semantic losses'
CANDIDATES = ('semantic_kl', 'semantic_h', 'weighted')
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
# Two datasets,70 objects,15 corruptions and two existing clean probes each.
ROWS_PER_CANDIDATE = len(DATASETS) * 70 * 15 * 2
RATIO_EPS = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value, minimum=0.):
    return type(value) in (int, float) and math.isfinite(value) and value > minimum


def validate_calibration_report(report: dict) -> dict:
    """Reject changed formulas, missing probes or inconsistent global coefficients.

    Callers additionally verify file hashes and source/native/assets/model proof.
    This function never mutates the report and never imports numerical/GPU code.
    """
    _require(isinstance(report, dict), 'Calibration report must be a mapping')
    _require(type(report.get('schema_version')) is int and report['schema_version'] == 1
             and report.get('purpose') == 'label_free_chamfer_calibration'
             and report.get('status') == 'verified_label_free_gpu'
             and type(report.get('target_semantic_update_ratio')) in (int, float)
             and report['target_semantic_update_ratio'] == .1,
             'Actual frozen-rule label-free GPU calibration is required')
    _require(report.get('rule') == CALIBRATION_RULE, 'Calibration formula changed')
    _require(report.get('loss_reduction') == LOSS_REDUCTION, 'Calibration summed loss reduction changed')
    proofs = report.get('calibration_bundles')
    _require(isinstance(proofs, dict) and set(proofs) == set(DATASETS), 'Calibration must cover both datasets')
    coefficients, ratios = report.get('coefficients'), report.get('ratios')
    _require(isinstance(coefficients, dict) and set(coefficients) == {'semantic_kl', 'semantic_h', 'weighted_scale'}
             and all(_positive(value) for value in coefficients.values()), 'Missing/nonpositive frozen calibration coefficients')
    _require(isinstance(ratios, dict) and set(ratios) == set(CANDIDATES), 'Calibration ratio candidates missing')
    for candidate in CANDIDATES:
        summary = ratios[candidate]
        _require(isinstance(summary, dict) and type(summary.get('count')) is int
                 and summary['count'] == ROWS_PER_CANDIDATE and type(summary.get('missing')) is int
                 and summary['missing'] == 0, 'Calibration ratio count/coverage mismatch: ' + candidate)
        _require(all(_positive(summary.get(key), RATIO_EPS) for key in ('min', 'median', 'p90', 'max')),
                 'Calibration ratio summary is nonfinite or zero/near-zero: ' + candidate)
        _require(summary['min'] <= summary['median'] <= summary['p90'] <= summary['max'],
                 'Calibration ratio summary order mismatch: ' + candidate)
        key, target = ('weighted_scale', 1.) if candidate == 'weighted' else (candidate, .1)
        _require(math.isclose(coefficients[key], target / summary['median'], rel_tol=1e-12, abs_tol=0.),
                 'Calibration coefficient violates frozen median rule: ' + candidate)
    return report
