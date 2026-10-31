"""Synthetic checks for unsealed local analysis adapter; no GPU/raw run claims."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(np.__version__.split('.')[0] != '1', reason='Pinned physical RNG/object analysis requires NumPy1')

spec = importlib.util.spec_from_file_location('retrieved_analysis_tested', Path(__file__).with_name('analyze_retrieved.py'))
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def fixture():
    checkpoint = 'a' * 64
    candidate = dict(dataset='scanobjectnn_c', backbone='curvenet', seed=0, technical_smoke=False,
                     object_ids=[2, 5], labels=[0, 1], corruptions=['gaussian', 'background'],
                     checkpoint_load=dict(checkpoint_sha256=checkpoint), outcomes={'probability_mean': [[True, False], [True, False]]})
    run = dict(dataset='scanobjectnn_c', backbone='curvenet', seed=0, outcome_prefix='r0', phase='variants_pointmae_teacher', arm='baseline',
               provenance=dict(protocol=dict(checkpoint_load=dict(checkpoint_sha256=checkpoint))),
               metrics=dict(probability_mean=dict(correct=2, per_corruption_correct=[1, 1])), archive=dict(sha256='c' * 64))
    arrays = dict(r0_index=np.array([[2, 5], [2, 5]]), r0_label=np.array([[0, 1], [0, 1]]),
                  r0_probability_mean_correct=np.array([[False, True], [False, True]]))
    return candidate, run, arrays


@pytest.mark.parametrize('fault', ['ids', 'labels', 'checkpoint', 'seed', 'corruption'])
def test_alignment_rejects_unmatched_historical_inputs(fault):
    candidate, run, arrays = fixture()
    if fault == 'ids':
        arrays['r0_index'][0, 0] = 3
    elif fault == 'labels':
        arrays['r0_label'][0, 0] = 1
    elif fault == 'checkpoint':
        candidate['checkpoint_load']['checkpoint_sha256'] = 'b' * 64
    elif fault == 'seed':
        candidate['seed'] = 1
    else:
        candidate['corruptions'].reverse()
    with pytest.raises(ValueError):
        ops.aligned_history(run, arrays, candidate, ['gaussian', 'background'], 'probability_mean')


def test_missing_seed_is_unavailable_instead_of_repeated_seed0():
    candidate, run, arrays = fixture()
    historical = dict(historical_runs=[run], corruptions=candidate['corruptions'], cohort={'scanobjectnn_c': {}})
    result = ops.historical_contrasts([candidate], historical, arrays)
    assert result[0]['status'] == 'unavailable'
    assert 'seed' in result[0]['reason']


def test_three_seed_comparison_resamples_objects_and_retains_historical_provenance():
    candidate, run, arrays = fixture()
    candidates, runs = [], []
    for seed in (0, 1, 2):
        candidates.append(dict(copy.deepcopy(candidate), seed=seed))
        runs.append(dict(copy.deepcopy(run), seed=seed))
    historical = dict(historical_runs=runs, corruptions=candidate['corruptions'], cohort={'scanobjectnn_c': {}})
    rows = ops.historical_contrasts(candidates, historical, arrays)
    complete = [r for r in rows if r['status'] == 'complete']
    assert len(complete) == 8
    joint = complete[0]
    assert joint['fixed_seeds'] == [0, 1, 2] and joint['seed'] is None
    assert joint['delta_pp'] == 0 and joint['n_original_object_clusters'] == 2
    assert joint['conditional_object_bootstrap_low_pp'] == -100
    assert joint['conditional_object_bootstrap_high_pp'] == 100
    assert joint['historical_raw_provenance'][0]['archive']['sha256'] == 'c' * 64


def test_remote_path_mapping_refuses_other_campaign_and_traversal(tmp_path):
    assert ops.mapped_raw_path('/remote/campaign/slots/a/raw/head.zip', '/remote/campaign', tmp_path) == tmp_path / 'slots/a/raw/head.zip'
    with pytest.raises(ValueError):
        ops.mapped_raw_path('/remote/other/head.zip', '/remote/campaign', tmp_path)
    with pytest.raises(ValueError):
        ops.mapped_raw_path('/remote/campaign/../head.zip', '/remote/campaign', tmp_path)


def test_original3dd_uses_only_candidate_seed0(monkeypatch):
    candidate, run, arrays = fixture()
    run.update(arm='original3dd', phase='historical_original_full_subset')
    original = np.array([[False, True], [False, True]])
    called = []
    def aligned(*args):
        called.append(args[2]['seed'])
        return original
    monkeypatch.setattr(ops, 'aligned_history', aligned)
    candidates = [dict(copy.deepcopy(candidate), seed=s) for s in (0, 1, 2)]
    historical = dict(historical_runs=[run], corruptions=candidate['corruptions'], cohort={'scanobjectnn_c': {}})
    complete = [r for r in ops.historical_contrasts(candidates, historical, arrays) if r['status'] == 'complete']
    assert called == [0]
    assert len(complete) == 2 and all(r['fixed_seeds'] == [0] and r['seed'] == 0 and r['seeds'] == 1 for r in complete)
    assert all('fixed seeds [0]' in r['interval_scope'] for r in complete)


def test_candidate_summary_keeps_incomplete_three_seed_mean_unavailable():
    candidate, _, _ = fixture()
    candidate['outcomes']['terminal'] = [[False, True], [False, True]]
    rows = ops.candidate_summary([candidate])
    current = next(r for r in rows if r['dataset'] == candidate['dataset'] and r['backbone'] == candidate['backbone'])
    assert current['status'] == 'unavailable' and current['mean_accuracy_pct'] is None
    assert current['seed0_accuracy_pct'] == 50 and current['seed1_accuracy_pct'] is None
    complete = ops.candidate_summary([dict(copy.deepcopy(candidate), seed=s) for s in (0, 1, 2)])
    current = next(r for r in complete if r['dataset'] == candidate['dataset'] and r['backbone'] == candidate['backbone'])
    assert current['status'] == 'complete' and current['mean_accuracy_pct'] == 50
    assert current['sample_sd_pp'] == 0


def test_coherently_altered_registered_plan_rejected_before_other_evidence(tmp_path):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    altered = dict(campaign_root='/other/coherent/campaign', jobs=[])
    raw = ops.encoded(altered)
    (tmp_path / 'registered_plan.json').write_bytes(raw)
    (campaign / 'plan.json').write_bytes(raw)
    ledger = dict(plan_sha256=ops.hashlib.sha256(raw).hexdigest(), status='complete', latest_event=dict(event='controller_stopped'))
    (campaign / 'registered_latest_ledger.json').write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='frozen actual plan SHA'):
        ops.validate_snapshot(tmp_path)
