"""Recount the baseline decision from accepted CSV counts; no model execution."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.run_cloudfixer_point2vec import POLICIES, registration


def recount():
    source = ROOT / 'knowledge/evidence/cloudfixer_two_colab_20261118/a_full_point2vec_per_corruption.csv'
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    methods = ['source', 'proximal_pm', 'proximal_terminal', 'cloudfixer']
    assert len(rows) == 60
    assert len({(r['method'], r['corruption']) for r in rows}) == 60
    assert all(int(r['n_examples']) == 2468 and int(r['seed']) == 2
               and int(r['severity']) == 5 and r['status'] == 'complete' for r in rows)
    assert all(abs(float(r['accuracy']) - int(r['n_correct']) / 2468) < 1e-12 for r in rows)
    totals = {m: sum(int(r['n_correct']) for r in rows if r['method'] == m) for m in methods}
    assert totals == dict(source=22947, proximal_pm=27748, proximal_terminal=27111, cloudfixer=28596)
    n = 2468 * 15
    focus = {c: {m: next(int(r['n_correct']) for r in rows if r['method'] == m and r['corruption'] == c)
                 for m in methods} for c in ['rotation', 'background']}
    focused_gap = sum(v['cloudfixer'] - v['proximal_pm'] for v in focus.values())
    total_gap = totals['cloudfixer'] - totals['proximal_pm']
    required_gain = total_gap + 1
    focused_baseline = sum(v['proximal_pm'] for v in focus.values())
    assert (total_gap, focused_gap, total_gap - focused_gap, required_gain) == (848, 795, 53, 849)
    assert focused_gap / total_gap == .9375
    policy = dict(steps=8, start=80, tau=.1, retained=.95, probe_indices=[20, 10, 0])
    assert POLICIES['proximal_pm'] == policy
    reg = registration()
    assert (reg['objects'], reg['seed'], reg['batch_size']) == (2468, 2, 1)
    assert POLICIES['proximal_terminal'] == dict(steps=32, start=160, tau=.03, retained=.95, probe_indices=[0])
    return dict(
        evidence='accepted A counts; descriptive conditional arithmetic, no new inference',
        baseline_id='cloudfixer_xyz_proximal_pm_n8_k80_late_v1',
        existing_runner_arm='proximal_pm', policy=policy,
        denoiser_calls=9, classifier_calls=3,
        pm='equal float64 softmax probabilities; pre-clean20/10 plus true terminal0',
        primary_comparator='original plain CloudFixer, first logits',
        tested_protocol=dict(classifier='point2vec', objects=2468, corruptions=15,
                             severity=5, seed=2, batch_size=1, observations_per_arm=n),
        source_csv=dict(path=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        totals_correct=totals, accuracy_percent={m: count / n * 100 for m, count in totals.items()},
        focus_correct=focus, total_net_deficit=total_gap, focus_net_deficit=focused_gap,
        focus_fraction_of_net_deficit=focused_gap / total_gap, other13_net_deficit=total_gap-focused_gap,
        hypothetical_cf_parity_on_focus=dict(correct=totals['proximal_pm']+focused_gap,
            accuracy_percent=(totals['proximal_pm']+focused_gap)/n*100,
            remaining_deficit_pp=(total_gap-focused_gap)/n*100),
        required_to_strictly_beat_observed_cf_if_other13_unchanged=dict(
            net_extra_correct=required_gain, focus_correct=focused_baseline+required_gain,
            focus_observations=4936, focus_accuracy_percent=(focused_baseline+required_gain)/4936*100,
            focus_mean_gain_pp=required_gain/4936*100,
            rotation_only_required_correct=focus['rotation']['proximal_pm']+required_gain,
            background_only_required_correct=focus['background']['proximal_pm']+required_gain),
        limits=['fixed seed/native realization; not seed variance',
                'N8 versus N32 changes N/K/tau/readout; not isolated PM',
                'no corruption-name routing or test-label retuning',
                'arithmetic target is not evidence of achievable or statistical superiority'])


if __name__ == '__main__':
    result = recount()
    existing = Path(__file__).with_name('decision.json')
    if existing.exists():
        assert json.loads(existing.read_text()) == result, 'Saved decision differs from current counts/policy'
    print(json.dumps(result, indent=2) + '\n', end='')
