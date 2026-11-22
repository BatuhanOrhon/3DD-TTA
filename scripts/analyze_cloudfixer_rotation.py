#!/usr/bin/env python3
"""Validate paired raw records and report accuracy, harms, timings and continuation."""
import argparse
import csv
import io
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from cloudfixer_store import canonical, sha


def softmax(logits):
    scores = np.asarray(logits, dtype=np.float64)
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)


def read_records(root):
    config = json.loads((root / 'config.json').read_text())
    if config['execution_status'] != 'complete':
        raise ValueError('Analysis requires complete paired coverage')
    rows = {}
    for path in sorted((root / 'raw').glob('*.json')):
        receipt = json.loads(path.read_text())
        unit = receipt['unit']
        data = path.with_suffix('.npz').read_bytes()
        if (unit['campaign'] != config['campaign'] or len(data) != receipt['bytes']
                or sha(data) != receipt['sha256']):
            raise ValueError(f'Block receipt mismatch: {path}')
        with np.load(io.BytesIO(data), allow_pickle=False) as payload:
            block = json.loads(payload['records'].tobytes())
        if [r['id'] for r in block] != unit['ids']:
            raise ValueError(f'Block ID mismatch: {path}')
        for row in block:
            key = row['condition'], row['seed'], row['id']
            if key in rows or key[:2] != (unit['condition'], unit['seed']):
                raise ValueError(f'Duplicate or mismatched row: {key}')
            rows[key] = row
            for arm, entry in row['arms'].items():
                probabilities = softmax(entry['logits'])
                if arm == 'view_mean':
                    probabilities[0] = softmax(entry['diagnostics']['option_logits']).mean(0)
                expected = probabilities.mean(0)
                if (expected.shape != (40,) or not np.isfinite(expected).all()
                        or not np.allclose(expected, entry['probabilities'], atol=1e-12, rtol=1e-10)
                        or expected.argmax() != entry['prediction']
                        or not np.isclose(expected.max(), entry['confidence'])):
                    raise ValueError(f'Invalid probabilities in {key}, {arm}')
    required = set(itertools.product(config['conditions'], config['seeds'], config['ids']))
    if set(rows) != required or (len(list((root / 'raw').glob('*.json'))) != config['blocks']):
        raise ValueError('Paired verification did not observe complete registered coverage')
    arm_sets = {tuple(sorted(row['arms'])) for row in rows.values()}
    label_sets = {ident: {rows[c, s, ident]['label'] for c in config['conditions'] for s in config['seeds']}
                  for ident in config['ids']}
    if len(arm_sets) != 1 or any(len(x) != 1 for x in label_sets.values()):
        raise ValueError('Arm coverage or object label varies across paired records')
    if config['phase'] != 'development' and set(config['params']) != set(next(iter(arm_sets))):
        raise ValueError('Observed arms differ from the registered phase arms')
    return config, rows


def csv_write(path, rows):
    with path.open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def joint_radius(values, *, draws, seed):
    """Maximum absolute deviation of the paired effects in each object-cluster draw."""
    rng = np.random.default_rng(seed)
    centered = values - values.mean(0)
    objects = len(centered)
    weights = np.full(objects, 1 / objects)
    maxima = []
    for start in range(0, draws, 128):
        counts = rng.multinomial(objects, weights, size=min(128, draws-start))
        maxima.extend(np.abs(counts @ centered / objects).max(1))
    return float(np.quantile(maxima, .95))


def paired_counts(base, candidate):
    return dict(baseline_correct=int(base.sum()), candidate_correct=int(candidate.sum()),
                total=base.size, baseline_accuracy=float(base.mean()), candidate_accuracy=float(candidate.mean()),
                delta_pp=100 * float(candidate.mean()-base.mean()), repairs=int((~base & candidate).sum()),
                harms=int((base & ~candidate).sum()))


def bootstrap_groups(conditions):
    groups = {name: [i] for i, name in enumerate(conditions)}
    if len([name for name in conditions if name != 'clean']) == 15:
        groups.update(
            macro14=[i for i,c in enumerate(conditions) if c not in ('clean', 'rotation')],
            macro13=[i for i,c in enumerate(conditions) if c not in ('clean', 'rotation', 'background')],
            all15=[i for i,c in enumerate(conditions) if c != 'clean'],
        )
    return groups


def per_arm_metrics(config, rows):
    """Build arm × corruption × seed × object arrays and verify raw call counts."""
    arms = sorted(next(iter(rows.values()))['arms'])
    labels = np.array([rows[k]['label'] for k in itertools.product(
        config['conditions'], config['seeds'], config['ids'])])
    tensors = {}
    details = {}
    for arm in arms:
        entries = [rows[k]['arms'][arm] for k in itertools.product(
            config['conditions'], config['seeds'], config['ids'])]
        for entry in entries:
            call = entry['diagnostics']
            expected = 3 if arm in ('identity', 'scd_search') or arm.startswith(('kabsch', 'bg_')) else 9
            if (call['denoiser_calls'] != 9 or call['classifier_calls'] != expected
                    or not np.isfinite(entry['seconds']) or entry['seconds'] <= 0
                    or not np.isfinite(entry['peak_mib']) or entry['peak_mib'] <= 0
                    or any(not np.isfinite(v) or v < 0 for v in entry['cuda_ms'].values())):
                raise ValueError(f'Invalid query or timing receipt for {arm}')
        correct = (np.array([entry['prediction'] for entry in entries]) == labels)
        tensors[arm] = correct.reshape(len(config['conditions']), len(config['seeds']), len(config['ids']))
        details[arm] = entries
    return tensors, details


def report(config, rows, *, raw_output, derived_output):
    tensors, entries = per_arm_metrics(config, rows)
    baseline = tensors['identity']
    contrasts = [(arm, 'identity') for arm in tensors if arm != 'identity']
    contrasts.extend((arm, control) for arm in ('output', 'feedback') for control in ('random', 'view_mean')
                     if arm in tensors and control in tensors)
    contrasts.extend((arm, reference) for arm, reference in (
        ('bg_weighted', 'bg_scalar'), ('bg_view', 'bg_random_view'),
        ('bg_weighted_pose', 'bg_weighted'), ('bg_weighted_pose', 'kabsch_progressive'),
        ('kabsch_fixed', 'kabsch_single'), ('kabsch_progressive', 'kabsch_single'),
        ('kabsch_progressive', 'kabsch_fixed'), ('feedback', 'output'),
        ('kabsch_spectral_single', 'kabsch_single'), ('kabsch_spectral_fixed', 'kabsch_fixed'),
        ('kabsch_spectral_progressive', 'kabsch_progressive'), ('scd_search', 'kabsch_fixed'),
    ) if arm in tensors and reference in tensors)
    conditions, seeds = config['conditions'], config['seeds']
    groups = bootstrap_groups(conditions)
    metrics = []
    values = []
    for arm, reference in contrasts:
        left, right = tensors[reference], tensors[arm]
        for group, ix in groups.items():
            counts = paired_counts(left[ix], right[ix])
            counts['seed_delta_pp'] = {s: paired_counts(left[ix, j], right[ix, j])['delta_pp']
                                       for j, s in enumerate(seeds)}
            counts['seed_std_pp'] = (float(np.std(list(counts['seed_delta_pp'].values()), ddof=1))
                                     if len(seeds) > 1 else None)
            metrics.append(dict(arm=arm, reference=reference, group=group, **counts))
            values.append(100 * (right[ix].astype(float)-left[ix]).mean(axis=(0,1)))
    reg = config['spec']['registration']
    radius = joint_radius(np.column_stack(values), draws=reg['bootstrap_draws'], seed=reg['bootstrap_seed'])
    per_seed = []
    table = []
    worst = {}
    profile = []
    failures = []
    for arm, tensor in tensors.items():
        for i, name in enumerate(conditions):
            for j, seed in enumerate(seeds):
                n = len(config['ids'])
                records = entries[arm][(i*len(seeds)+j)*n:(i*len(seeds)+j+1)*n]
                per_seed.append(dict(run_id=config['run_id'], dataset='modelnet40_c',
                    severity=0 if name=='clean' else 5, method=arm, seed=seed, corruption=name,
                    correct=int(tensor[i,j].sum()), total=n, accuracy=float(tensor[i,j].mean()),
                    elapsed_seconds=sum(e['seconds'] for e in records), peak_gpu_memory_mb=max(e['peak_mib'] for e in records),
                    status='complete' if n==2468 else 'partial'))
            packed = [rows[name, s, ident] for s in seeds for ident in config['ids']]
            corrections = [v['arms'][arm] for v in packed]
            denoms = [r['arms']['identity'] for r in packed]
            proposed = [u for e in corrections for u in e['diagnostics'].get('updates', [])]
            final = [e['diagnostics'].get('updates', [])[-1] for e in corrections if e['diagnostics'].get('updates')]
            angles = [v['total_degrees'] for v in final] or [e['diagnostics'].get('angle_degrees', 0.) for e in corrections]
            travels = [v['travel_degrees'] for v in final] or angles
            profile.append(dict(arm=arm, corruption=name,
                mean_ms=1000*np.mean([e['seconds'] for e in corrections]),
                median_ms=1000*np.median([e['seconds'] for e in corrections]),
                p95_ms=1000*np.quantile([e['seconds'] for e in corrections], .95),
                peak_allocated_mib=max(e['peak_mib'] for e in corrections),
                denoiser_calls=corrections[0]['diagnostics']['denoiser_calls'],
                classifier_calls=corrections[0]['diagnostics']['classifier_calls'],
                **{c+'_ms': np.mean([e['cuda_ms'][c] for e in corrections]) for c in corrections[0]['cuda_ms']},
                mean_total_degrees=float(np.mean(angles)), mean_travel_degrees=float(np.mean(travels)),
                mean_applied_step_degrees=float(np.mean([u['applied_degrees'] for u in proposed])) if proposed else 0.,
                mean_anchor_support=float(np.mean([u['anchors'] for u in proposed])) if proposed else None,
                pose_attempts=len(proposed), pose_rejections=sum(u['reason']!='accepted' for u in proposed),
                graph_builds=sum(e.get('graph_builds', 0) for e in corrections),
                rematches=sum(u['matching'] for u in proposed), svds=sum(u['svd'] for u in proposed),
                search_nonidentity=sum(e['diagnostics'].get('choice',0)!=0 for e in corrections)))
            for row, src, cand in zip(packed, denoms, corrections):
                if (src['prediction'] == row['label'] != cand['prediction']
                        and cand['confidence'] >= .8 and cand['confidence'] > src['confidence']):
                    failures.append(dict(arm=arm, condition=name, id=row['id'], seed=row['seed'],
                        label=row['label'], baseline_confidence=src['confidence'],
                        candidate_confidence=cand['confidence'], candidate_prediction=cand['prediction']))
        if arm != 'identity':
            corrs = [m for m in metrics if m['arm']==arm and m['reference']=='identity'
                     and m['group'] in conditions and m['group']!='clean']
            worst[arm] = min(corrs, key=lambda x: x['delta_pp'])['group'] if corrs else None
    for m in metrics:
        m['simultaneous_lower_pp'] = max(-100., m['delta_pp']-radius)
        m['simultaneous_upper_pp'] = min(100., m['delta_pp']+radius)
        m['worst_for_arm'] = m['reference']=='identity' and worst.get(m['arm'])==m['group']
        table.append(dict(m, seed_delta_pp=json.dumps(m['seed_delta_pp'], sort_keys=True)))
    csv_write(derived_output/'paired.csv', table)
    csv_write(derived_output/'gpu_cost.csv', profile)
    (derived_output/'confident_harms.json').write_bytes(canonical(failures))
    csv_write(raw_output/'per_corruption.csv', per_seed)
    summary = []
    for method, seed in itertools.product(tensors, seeds):
        subset = [r for r in per_seed if r['method']==method and r['seed']==seed and r['corruption']!='clean']
        if not subset:
            continue
        total = sum(r['total'] for r in subset)
        correct = sum(r['correct'] for r in subset)
        summary.append(dict(run_id=config['run_id'], method=method, seed=seed, observations=total,
                            correct=correct, accuracy=correct/total, n_corruptions=len(subset),
                            elapsed_seconds=sum(r['elapsed_seconds'] for r in subset),
                            peak_gpu_memory_mb=max(r['peak_gpu_memory_mb'] for r in subset),
                            status='complete' if total==37020 else 'partial'))
    csv_write(raw_output/'summary.csv', summary)
    return metrics, entries


def synthetic_summary(rows):
    nested = {}
    for row in rows.values():
        for scenario, arms in row.get('synthetic', {}).items():
            for arm, x in arms.items():
                key = row['condition'], scenario, arm
                nested.setdefault(key, []).append((x['before_degrees']-x['residual_degrees'],
                                                   x['before_scd']-x['scd']))
    return [dict(condition=c, scenario=s, arm=a, objects=len(v), mean_degree_recovery=np.mean(v,axis=0)[0],
                 mean_chamfer_reduction=np.mean(v,axis=0)[1]) for (c,s,a),v in sorted(nested.items())]


def decide(config, metrics, entries, synthetic):
    reg = config['spec']['registration']
    lookup = {(m['arm'],m['reference'],m['group']):m for m in metrics}
    mean = {a: float(np.mean([e['seconds'] for e in observations]))
            for a, observations in entries.items()}
    seen_synthetic = {x['arm']:x['mean_degree_recovery'] for x in synthetic
                      if x['condition']=='clean' and x['scenario']=='compound'}
    decisions = {}
    for arm in ('kabsch_single', 'kabsch_fixed', 'kabsch_progressive', 'output', 'feedback'):
        def effect(group, reference='identity'):
            return lookup[arm,reference,group]
        rotation = effect('rotation')
        ratio = mean[arm]/mean['identity']
        gates = dict(rotation=(rotation['delta_pp'] >= reg['min_rotation_pp']
                               and rotation['simultaneous_lower_pp'] > 0),
            clean=effect('clean')['simultaneous_lower_pp'] >= reg['clean_lower_pp'],
            macro14=effect('macro14')['simultaneous_lower_pp'] >= reg['macro14_lower_pp'],
            all15=effect('all15')['delta_pp'] >= 0,
            tail=min(effect(c)['delta_pp'] for c in config['conditions'] if c!='clean') >= reg['worst_corruption_point_pp'],
            time=ratio <= reg['max_time_ratio'])
        if arm.startswith('kabsch'):
            gates['pose'] = bool(seen_synthetic.get(arm,-1.) > 0)
        else:
            gates['development'] = config['freeze']['development_passed']
            gates['query_controls'] = all(effect('rotation', c)['delta_pp'] > 0 for c in ('random', 'view_mean'))
        decisions[arm] = dict(gates=gates, rotation_pp=rotation['delta_pp'], time_ratio=ratio,
                              full_pair_hours=(mean['identity']+mean[arm])*37020/3600,
                              candidate_only_hours=mean[arm]*37020/3600)
    eligible = [arm for arm,value in decisions.items() if all(value['gates'].values())]
    chosen = max(eligible, key=lambda arm:(decisions[arm]['rotation_pp'], -mean[arm])) if eligible else None
    return dict(campaign=config['campaign'], next_arm=chosen, arms=decisions)


def noise_pm_interactions(config, rows, output):
    """Cluster by object for the paired change in PM's gain over its own terminal."""
    conditions, seeds, ids = (config[key] for key in ('conditions', 'seeds', 'ids'))
    schedule_arms = ('identity', 'bg_mid', 'bg_high', 'bg_start_only')
    if not all(name in config['params'] for name in schedule_arms):
        return
    keys = list(itertools.product(conditions, seeds, ids))
    shape = (len(conditions), len(seeds), len(ids))
    truth = np.array([rows[k]['label'] for k in keys])
    windows = {'t20': [0], 't10': [1], 't0': [2], 't20_t10': [0, 1],
               't20_t0': [0, 2], 't10_t0': [1, 2], 'pm': [0, 1, 2]}
    correct = {}
    descriptive = []
    groups = bootstrap_groups(conditions)
    for arm in schedule_arms:
        p = softmax([rows[k]['arms'][arm]['logits'] for k in keys])
        each = {name: (p[:, ix].mean(1).argmax(1) == truth).reshape(shape) for name, ix in windows.items()}
        correct[arm] = each
        votes = p.argmax(2)
        disagree = (votes != votes[:, :1]).any(1).reshape(shape)
        for group, ix in groups.items():
            for name, hits in each.items():
                comparison = paired_counts(each['t0'][ix], hits[ix])
                descriptive.append(dict(arm=arm, group=group, window=name,
                                        disagree=int(disagree[ix].sum()), **comparison))
    csv_write(output/'noise_pm_windows.csv', descriptive)
    draws = config['spec']['registration']['bootstrap_draws']
    rng = np.random.default_rng(20261122)
    weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws) / len(ids)
    interactions, vectors = [], []
    for arm, reference in [('bg_mid', 'identity'), ('bg_high', 'identity'),
                           ('bg_start_only', 'identity'), ('bg_high', 'bg_start_only')]:
        for group, ix in groups.items():
            base = correct[reference]
            target = correct[arm]
            base_gain = base['pm'][ix].astype(int) - base['t0'][ix]
            target_gain = target['pm'][ix].astype(int) - target['t0'][ix]
            difference = 100 * (target_gain - base_gain)
            by_object = difference.mean(axis=(0, 1))
            bootstrap = weights @ by_object
            lo, hi = np.quantile(bootstrap, [.025, .975])
            vectors.append(by_object)
            interactions.append(dict(candidate=arm, reference=reference, group=group,
                n_original_ids=len(ids), paired_events=difference.size,
                candidate_pm_minus_t0_pp=100*target_gain.mean(), reference_pm_minus_t0_pp=100*base_gain.mean(),
                interaction_pp=by_object.mean(), nominal_lower_pp=lo, nominal_upper_pp=hi,
                by_seed_pp=json.dumps(difference.mean(axis=(0, 2)).tolist()),
                pm_change_pp=100*(target['pm'][ix].mean()-base['pm'][ix].mean()),
                terminal_change_pp=100*(target['t0'][ix].mean()-base['t0'][ix].mean())))
    radius = joint_radius(np.column_stack(vectors), draws=draws, seed=20261122)
    for row in interactions:
        row['simultaneous_lower_pp'] = row['interaction_pp'] - radius
        row['simultaneous_upper_pp'] = row['interaction_pp'] + radius
    csv_write(output/'noise_pm_interaction.csv', interactions)


def background_report(config, rows, metrics, output):
    readouts = []
    gates = {}
    arms = config['params']
    for arm in arms:
        for cond in config['conditions']:
            subset = [r for r in rows.values() if r['condition'] == cond]
            labels = np.array([r['label'] for r in subset])
            entries = [r['arms'][arm] for r in subset]
            logits = np.array([e['logits'] for e in entries])
            pm = np.array([e['prediction'] for e in entries]) == labels
            terminal = logits[:, -1].argmax(-1) == labels
            truth = np.take_along_axis(logits, labels[:, None, None], axis=2)
            readouts.append(dict(arm=arm, condition=cond, total=len(labels), pm=int(pm.sum()),
                terminal=int(terminal.sum()), pm_repairs=int((pm & ~terminal).sum()),
                pm_harms=int((~pm & terminal).sum()),
                common_rival=int((logits > truth).all(1).any(1).sum())))
        if arm == 'identity' or config['phase'] != 'background':
            continue
        selected = [r for r in rows.values() if r['condition'] != 'clean']
        costs = np.array([[r['arms'][name]['seconds'] for r in selected] for name in ('identity', arm)])
        pct = np.quantile(costs, [.5, .95], axis=1)
        ratios = pct[:, 1] / pct[:, 0]
        effects = {m['group']: m for m in metrics if m['arm'] == arm and m['reference'] == 'identity'}
        limits = {'background': 3., 'clean': -1., 'rotation': -1., 'macro13': -.5, 'all15': 0.}
        point_gate = all(effects[group]['delta_pp'] + 1e-10 >= limit for group, limit in limits.items())
        point_gate &= effects['all15']['delta_pp'] > 0 and bool(np.all(ratios <= 1.10))
        interval_gate = all(effects[group]['simultaneous_lower_pp'] > (0 if group == 'background' else limit)
                            for group, limit in limits.items())
        gates[arm] = dict(p50_ratio=float(ratios[0]), p95_ratio=float(ratios[1]),
                          point_gate=bool(point_gate), interval_gate=bool(interval_gate),
                          passes=bool(point_gate and interval_gate))
    csv_write(output/'pm_vs_terminal.csv', readouts)
    (output/'background_gates.json').write_bytes(canonical(gates))
    if config['phase'] == 'background_dev':
        diagnostics = [dict(id=r['id'], **r['support_diagnostic'])
                       for r in rows.values() if 'support_diagnostic' in r]
        (output/'support_diagnostic.json').write_bytes(canonical(diagnostics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--derived', type=Path, required=True)
    parser.add_argument('--development', type=Path)
    args = parser.parse_args()
    config, rows = read_records(args.run)
    args.derived.mkdir(parents=True, exist_ok=False)
    metrics, entries = report(config, rows, raw_output=args.run, derived_output=args.derived)
    if config['phase'].startswith('background'):
        noise_pm_interactions(config, rows, args.derived)
        background_report(config, rows, metrics, args.derived)
    if config['phase'] in ('development', 'spectral', 'scd'):
        (args.derived/'synthetic.json').write_bytes(canonical(synthetic_summary(rows)))
    if config['phase'] == 'validation':
        if args.development is None:
            raise ValueError('Validation decisions require the sealed development run')
        dev_config, dev_rows = read_records(args.development)
        if (dev_config['campaign'] != config['freeze']['campaign']
                or json.loads((args.development/'search_lock.json').read_text()) != config['freeze']):
            raise ValueError('Validation used a mismatched development lock')
        outcome = decide(config, metrics, entries, synthetic_summary(dev_rows))
        (args.derived/'decision.json').write_bytes(canonical(outcome))
    print('CLOUDFIXER_ROTATION_ANALYSIS_DONE', config['phase'], flush=True)


if __name__ == '__main__':
    main()
