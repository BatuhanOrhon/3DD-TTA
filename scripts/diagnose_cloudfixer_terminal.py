"""Posthoc saved-logit diagnostics; no model inference or deployed policy selection."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    'entropy': ('20261115-114749_entropy_run.zip', '509ea9699cd2fcde8542fc52564c03146a1f2f30db0d4e20d176e1aa479506df'),
    'nstep': ('20261115-110019_nstep_run.zip', '2a4c544f2f08de453205f556689a47a1aedabf37be4b87e07d8c9348e68d58f7'),
}


def load(role, steps):
    name, digest = INPUTS[role]
    path = ROOT / 'result/modelnet40_c/cloudfixer_exploration' / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    records = []
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if '/pending_unit_' not in name or not name.endswith('.npz'):
                continue
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as payload:
                unit = json.loads(str(payload['unit']))
                if unit['config']['steps'] == steps:
                    records.append((unit, {k: payload[k].copy() for k in payload.files}))
    records.sort(key=lambda pair: (pair[0]['corruption'], pair[0]['ids']))
    assert len(records) == 60
    times = [json.loads(str(p['sampler']))['candidate_indices'] for _, p in records]
    assert all(t == times[0] for t in times)
    out = {k: np.concatenate([p[k] for _, p in records]) for k in ('ids', 'labels', 'probabilities', 'entropy')}
    out['times'] = times[0]
    out['corruptions'] = np.array([u['corruption'] for u, p in records for _ in p['ids']])
    labels_path = ROOT / 'result/preparation/cloudfixer_transfer_20261114/verified_label.npy'
    assert hashlib.sha256(labels_path.read_bytes()).hexdigest() == '97b6e660820103074ab192f10b6d9c33aa0446f1547ddb6ad2ca35a029a63b31'
    assert np.array_equal(out['labels'], np.load(labels_path).reshape(-1)[out['ids']])
    assert len(out['ids']) == 1920 and len(set(zip(out['corruptions'], out['ids']))) == 1920
    assert len(set(out['ids'])) == 128 and len(set(out['corruptions'])) == 15
    return out


def paired_counts(correct, terminal):
    return dict(correct=int(correct.sum()), repair=int((correct & ~terminal).sum()),
                harm=int((~correct & terminal).sum()), delta_pp=float(100 * (correct.mean() - terminal.mean())))


def diagnose(view):
    p, h, y = view['probabilities'], view['entropy'], view['labels']
    assert np.isfinite(p).all() and np.isfinite(h).all() and np.allclose(p.sum(-1), 1)
    assert np.allclose(h, -(p * np.log(np.maximum(p, 1e-300))).sum(-1), atol=2e-6)
    n, k, _ = p.shape
    correct = p.argmax(-1) == y[:, None]
    terminal = correct[:, -1]
    equal = p.mean(1).argmax(-1) == y
    mixed = correct.any(1) & ~correct.all(1)
    rank_sum, pair_count, object_scores = 0., 0, []
    for hi, ci in zip(h[mixed], correct[mixed]):
        difference = hi[~ci][None, :] - hi[ci][:, None]
        score = (difference > 0) + .5 * (difference == 0)
        rank_sum += score.sum()
        pair_count += score.size
        object_scores.append(float(score.mean()))
    pooled_right, pooled_wrong = h[correct], h[~correct]
    pooled_score = sum(float(((pooled_wrong > value) + .5 * (pooled_wrong == value)).sum()) for value in pooled_right)
    subsets = {'equal': list(range(k))}
    subsets.update({'suffix_' + str(count): list(range(k-count, k)) for count in range(1, k)})
    subsets.update({'prefix_' + str(count): list(range(count)) for count in range(1, k)})
    subsets.update({'omit_' + str(t): [i for i in range(k) if i != j] for j, t in enumerate(view['times'])})
    rows = {}
    object_ids = sorted(set(view['ids']))
    draws = np.random.default_rng(20261115).integers(0, len(object_ids), (10000, len(object_ids)))
    for name, slots in subsets.items():
        outcome = p[:, slots].mean(1).argmax(-1) == y
        row = paired_counts(outcome, terminal)
        row['times'] = [view['times'][i] for i in slots]
        cluster_delta = np.array([(outcome[view['ids'] == i].astype(float) - terminal[view['ids'] == i]).mean() for i in object_ids])
        row['conditional_object_bootstrap_ci95_pp'] = (100 * np.quantile(cluster_delta[draws].mean(1), [.025, .975])).tolist()
        row['per_corruption'] = {c: paired_counts(outcome[view['corruptions'] == c], terminal[view['corruptions'] == c])
                                 for c in sorted(set(view['corruptions']))}
        assert row['correct'] - int(terminal.sum()) == row['repair'] - row['harm']
        rows[name] = row
    harmed, repaired = terminal & ~equal, ~terminal & equal
    # Compare truth with the wrong prediction: PM winner on harms, terminal winner on repairs.
    competitor = np.where(equal, p[:, -1].argmax(-1), p.mean(1).argmax(-1))
    margin = p[np.arange(n)[:, None], np.arange(k)[None, :], y[:, None]] - p[np.arange(n)[:, None], np.arange(k)[None, :], competitor[:, None]]
    assert (margin[harmed].sum(1) <= 0).all()
    assert (margin[repaired].sum(1) >= 0).all()
    def group(mask):
        return dict(n=int(mask.sum()), candidate_correct_counts=correct[mask].sum(0).tolist(),
                    mean_true_vs_wrong_winner_margin=margin[mask].mean(0).tolist(),
                    mean_entropy=h[mask].mean(0).tolist(),
                    at_least_half_candidates_correct=int((correct[mask].sum(1) >= k/2).sum()))
    return dict(times=view['times'], terminal=int(terminal.sum()),
                probes=[dict(time=t, **paired_counts(correct[:, j], terminal)) for j, t in enumerate(view['times'])],
                mixed_correctness_objects=int(mixed.sum()), pooled_entropy_correct_vs_wrong_auc=float(pooled_score / (len(pooled_right)*len(pooled_wrong))),
                within_object_entropy_pair_auc=float(rank_sum / pair_count), within_object_entropy_mean_auc=float(np.mean(object_scores)),
                within_object_correct_wrong_pairs=pair_count,
                min_entropy=paired_counts(correct[np.arange(n), h.argmin(1)], terminal),
                pm_harms=group(harmed), pm_repairs=group(repaired), subsets=rows)


if __name__ == '__main__':
    dense, n12 = load('entropy', 8), load('nstep', 12)
    sparse = {**dense, 'probabilities': dense['probabilities'][:, [1, 3, 5, 8]],
              'entropy': dense['entropy'][:, [1, 3, 5, 8]], 'times': [70, 50, 30, 0]}
    results = {name: diagnose(view) for name, view in [('n8_four', sparse), ('n8_nine', dense), ('n12_four', n12)]}
    for name, t, pm in [('n8_four', 1269, 1250), ('n8_nine', 1269, 1256), ('n12_four', 1235, 1257)]:
        assert results[name]['terminal'] == t and results[name]['subsets']['equal']['correct'] == pm
    output = ROOT / 'knowledge/evidence/cloudfixer_terminal_diagnosis_20261115/diagnostics.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(dict(inputs=INPUTS, bootstrap_seed=20261115, bootstrap_replicates=10000,
                       scope='Posthoc explanatory diagnostics; all subsets retained; no inference or policy promotion', views=results), stream, indent=2)
    print(output)
