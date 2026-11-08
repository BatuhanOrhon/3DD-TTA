"""Paired operational accuracy and batch-level latency; repeats are not votes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def choose_profile(profiles):
    """Select a common batch by complete, memory-safe measured throughput."""
    if sorted(p['batch_size'] for p in profiles) != [32, 64, 128]:
        raise ValueError('Incomplete batch profile grid')
    valid, identities = [], set()
    for p in profiles:
        identities.add((p['source_overlay_sha256'], p['asset_manifest_sha256']))
        if p['status'] == 'accepted_failed_attempt' and 'out of memory' in str(p.get('failure', '')).lower():
            continue
        if p['status'] != 'accepted' or p['phase'] != 'profile' or p['rows'] != 18 * p['batch_size']:
            raise ValueError('Invalid profile coverage or non-OOM failure')
        m = p['profile_metrics']
        values = [m[k] for k in ('objects_per_second','max_allocated_mib','max_reserved_mib','gpu_total_mib')]
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Invalid profile measurements')
        if max(m['max_allocated_mib'], m['max_reserved_mib']) <= .8 * m['gpu_total_mib']:
            valid.append(p)
    if len(identities) != 1 or not valid:
        raise ValueError('No common-source memory-safe profile')
    best = max(valid, key=lambda p: (p['profile_metrics']['objects_per_second'], -p['batch_size']))
    return dict(status='selected', batch_size=best['batch_size'],
                criterion='Maximum aggregate timed throughput, allocated/reserved <=80%VRAM; smaller batch on exact tie',
                profiles=profiles)


def cross_control_differences(case, reference, candidate, repetitions):
    """All cross-repeat pairs, never select the most favorable native match."""
    values = {}
    for arm in (reference, candidate):
        for repeat in range(repetitions):
            with np.load(Path(case) / ('%s_r%d.npz' % (arm, repeat)), allow_pickle=False) as data:
                values[arm, repeat] = {k: data[k].copy() for k in ('latents', 'logits', 'prediction', 'index')}
    result = []
    for a in range(repetitions):
        for b in range(repetitions):
            left, right = values[reference, a], values[candidate, b]
            if not np.array_equal(left['index'], right['index']):
                raise ValueError('Cross-control IDs differ')
            result.append(dict(reference=reference, candidate=candidate, reference_repeat=a, candidate_repeat=b,
                latent_rms=float(np.sqrt(np.mean((left['latents'].astype(np.float64) - right['latents']) ** 2))),
                logit_rms=float(np.sqrt(np.mean((left['logits'].astype(np.float64) - right['logits']) ** 2))),
                prediction_changes=int((left['prediction'] != right['prediction']).sum())))
    return result


def paired_summary(rows, reference, candidate, draws=10000):
    selected = {}
    timings = {}
    memberships = {}
    for arm in (reference, candidate):
        own, times, members = {}, {}, {}
        for row in rows:
            if row['arm'] != arm:
                continue
            key = (int(row['object_id']), row['corruption'], int(row['seed']), int(row['repeat']))
            if key in own:
                raise ValueError('duplicate paired inference')
            correct = int(row['correct'])
            seconds, size = float(row['seconds']), int(row['batch_size'])
            if correct not in (0, 1) or not np.isfinite(seconds) or seconds <= 0 or size < 1:
                raise ValueError('Invalid outcome/timing')
            own[key] = (correct, int(row['label']))
            batch = (row['corruption'], int(row['seed']), int(row['repeat']), int(row['batch']))
            members.setdefault(batch, set()).add(int(row['object_id']))
            if batch in times and times[batch][:2] != (seconds, size):
                raise ValueError('Inconsistent batch timing')
            old = times.get(batch, (seconds, size, 0))
            times[batch] = (seconds, size, old[2] + 1)
        if not own or any(size != count for _, size, count in times.values()):
            raise ValueError('Incomplete paired batch')
        selected[arm], timings[arm] = own, times
        memberships[arm] = members
    a, b = selected[reference], selected[candidate]
    if a.keys() != b.keys() or timings[reference].keys() != timings[candidate].keys():
        raise ValueError('Unmatched pairs')
    if any(a[k][1] != b[k][1] for k in a):
        raise ValueError('Mismatched paired label')
    if memberships[reference] != memberships[candidate]:
        raise ValueError('Mismatched paired batch membership')
    ids = sorted({k[0] for k in a})
    strata = sorted({k[1:] for k in a})
    if len(a) != len(ids) * len(strata):
        raise ValueError('Incomplete object/stratum paired grid')
    delta = np.array([[(b[(i, *s)][0] - a[(i, *s)][0]) * 100 for s in strata] for i in ids])
    object_delta = delta.mean(1)
    rng = np.random.RandomState(20261107)
    boot = object_delta[rng.randint(len(ids), size=(draws, len(ids)))].mean(1)
    at = sum(t[0] for t in timings[reference].values())
    bt = sum(t[0] for t in timings[candidate].values())
    by_seed_repeat = []
    for seed, repeat in sorted({s[1:] for s in strata}):
        keys = [k for k in a if k[2:] == (seed, repeat)]
        ta = sum(v[0] for k, v in timings[reference].items() if k[1:3] == (seed, repeat))
        tb = sum(v[0] for k, v in timings[candidate].items() if k[1:3] == (seed, repeat))
        by_seed_repeat.append(dict(seed=seed, repeat=repeat,
            delta_pp=100 * np.mean([b[k][0] - a[k][0] for k in keys]), saving_percent=100 * (1 - tb / ta)))
    by_corruption = {c: 100 * np.mean([b[k][0] - a[k][0] for k in a if k[1] == c])
                     for c in sorted({k[1] for k in a})}
    # Batch clusters retain all repeated timings for one corruption/object batch.
    clusters = sorted({(k[0], k[3]) for k in timings[reference]})
    ta = np.array([sum(v[0] for k, v in timings[reference].items() if (k[0], k[3]) == c) for c in clusters])
    tb = np.array([sum(v[0] for k, v in timings[candidate].items() if (k[0], k[3]) == c) for c in clusters])
    picks = rng.randint(len(clusters), size=(draws, len(clusters)))
    savings = 100 * (1 - tb[picks].sum(1) / ta[picks].sum(1))
    return dict(reference=reference, candidate=candidate, objects=len(ids), paired_inferences=len(a),
        reference_accuracy_percent=100 * np.mean([x[0] for x in a.values()]),
        candidate_accuracy_percent=100 * np.mean([x[0] for x in b.values()]),
        delta_pp=float(object_delta.mean()), delta_ci95_pp=np.quantile(boot, [.025, .975]).tolist(),
        degenerate_object_differences=bool(np.ptp(object_delta) == 0),
        reference_seconds=at, candidate_seconds=bt, saving_percent=100 * (1 - bt / at),
        saving_batch_cluster_ci95_percent=np.quantile(savings, [.025, .975]).tolist(),
        timed_batches_per_arm=len(timings[reference]), timing_clusters=len(clusters),
        per_seed_repeat=by_seed_repeat, per_corruption_delta_pp=by_corruption,
        uncertainty='Accuracy CI clusters original objects, conditional on observed seeds/repeats/runtime; '
                    'timing CI clusters corruption/batch and is descriptive for this fixed suite. '
                    'Degenerate zero-difference CI does not prove population equivalence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open() as stream:
        rows = list(csv.DictReader(stream))
    result = [paired_summary(rows, 'original', arm) for arm in sorted({r['arm'] for r in rows} - {'original'})]
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
