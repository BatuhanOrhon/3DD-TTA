"""Join the follow-up seeds by original ID and test the registered adoption margins."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_artifacts import CORRUPTIONS
from scripts.analyze_cloudfixer_rotation import joint_radius, paired_counts, read_records, softmax


def read_archive(path, phase, registration):
    with tempfile.TemporaryDirectory() as tmp, ZipFile(path) as z:
        run = path.stem
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None:
            raise ValueError('Invalid run ZIP')
        if json.loads(z.read(run + '/exit_status.json'))['returncode']:
            raise ValueError('Failed run')
        for name in names:
            parts = Path(name).parts
            if len(parts) in (2, 3) and parts[0] == run and (
                    parts[1:] == ('config.json',) or len(parts) == 3 and parts[1] == 'raw'
                    and parts[2].startswith('block_') and Path(parts[2]).suffix in ('.npz', '.json')):
                z.extract(name, tmp)
        config, raw = read_records(Path(tmp)/run)
    if config['phase'] != phase or config['spec']['background_followup'] != registration:
        raise ValueError('Archive does not belong to the locked follow-up')
    data = {}
    view_diagnostics = {}
    for key, row in raw.items():
        arms = row['arms']
        if 'bg_view' in arms:
            left, right = arms['bg_view'], arms['bg_random_view']
            assert left['diagnostics']['t10_view_points'] == right['diagnostics']['t10_view_points']
            for i in (0, 2):
                assert left['logits'][i] == right['logits'][i] == arms['identity']['logits'][i]
            stats = view_diagnostics.setdefault(key[0], dict(events=0, points=[], arms={}))
            stats['events'] += 1
            stats['points'].append(left['diagnostics']['t10_view_points'])
            truth = row['label']
            for name in ('identity', 'bg_view', 'bg_random_view'):
                e = arms[name]
                tally = stats['arms'].setdefault(name, dict(t10_correct=0, pm_correct=0, terminal_correct=0,
                                                           t10_changed=0, pm_changed=0))
                t10 = int(np.argmax(e['logits'][1]))
                tally['t10_correct'] += int(t10 == truth)
                tally['terminal_correct'] += int(np.argmax(e['logits'][2]) == truth)
                tally['pm_correct'] += int(e['prediction'] == truth)
                tally['t10_changed'] += int(t10 != np.argmax(arms['identity']['logits'][1]))
                tally['pm_changed'] += int(e['prediction'] != arms['identity']['prediction'])
        data[key] = {}
        for name, entry in arms.items():
            probs = softmax(entry['logits'])
            data[key][name] = (entry['prediction'] == row['label'], entry['seconds'], entry['peak_mib'],
                               probs[1].argmax() == row['label'], probs[1:].mean(0).argmax() == row['label'])
    for stats in view_diagnostics.values():
        points = stats.pop('points')
        stats['kept_points_p10_p50_p90'] = np.quantile(points, [.1, .5, .9]).tolist()
    return config, data, view_diagnostics


def compare(data, ids, contrasts, timing_conditions, timing_seeds, draws):
    weights = np.random.default_rng(20261122).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=draws)/len(ids)
    rows, clusters = [], []
    for group, conditions, seeds, arm, control in contrasts:
        keys = list(itertools.product(conditions, seeds, ids))
        shape = (len(conditions), len(seeds), len(ids))
        base, candidate = [np.array([data[k][a][0] for k in keys]).reshape(shape) for a in (control, arm)]
        diff = 100 * (candidate.astype(int) - base)
        cluster = diff.mean(axis=(0, 1))
        rows.append(dict(group=group, arm=arm, control=control, seeds=seeds, **paired_counts(base, candidate),
                         nominal_95_pp=np.quantile(weights @ cluster, [.025, .975]).tolist(),
                         per_seed_delta_pp=diff.mean(axis=(0, 2)).tolist()))
        clusters.append(cluster)
    radius = joint_radius(np.column_stack(clusters), draws=draws, seed=20261122)
    for row in rows:
        row['joint_95_pp'] = [row['delta_pp']-radius, row['delta_pp']+radius]
    keys = list(itertools.product(timing_conditions, timing_seeds, ids))
    profile = {}
    for arm in data[keys[0]]:
        values = np.array([data[k][arm][1:3] for k in keys])
        assert np.isfinite(values).all() and (values>0).all()
        p50, p95 = 1000*np.quantile(values[:, 0], [.5, .95])
        profile[arm] = dict(p50_ms=p50, p95_ms=p95, peak_allocated_mib=values[:, 1].max())
    for v in profile.values():
        for metric in ('p50','p95'):
            v[metric+'_ratio'] = v[metric+'_ms']/profile['identity'][metric+'_ms']
    return dict(comparisons=rows,timing=profile)


def exploratory_windows(data, ids, groups, draws):
    weights = np.random.default_rng(20261122).multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=draws)/len(ids)
    result = []
    for group, conditions, seeds in groups:
        keys = list(itertools.product(conditions, seeds, ids))
        for window, column in [('t10', 3), ('t10_t0', 4), ('pm', 0)]:
            candidate = np.array([data[k]['bg_view'][column] for k in keys])
            row = dict(group=group, window=window, events=len(keys), filtered_correct=int(candidate.sum()))
            for field, arm, index in [('baseline_pm', 'identity', 0), ('unfiltered_window', 'identity', column),
                                      ('random_window', 'bg_random_view', column)]:
                ref = np.array([data[k][arm][index] for k in keys])
                cluster = 100*(candidate.astype(int)-ref).reshape(-1,len(ids)).mean(axis=0)
                row[field] = dict(correct=int(ref.sum()), delta_pp=float(cluster.mean()),
                                  nominal_95_pp=np.quantile(weights @ cluster,[.025,.975]).tolist())
            result.append(row)
    return result


def analyze(focus, other, scd):
    reg = json.loads((ROOT/'cfgs/cloudfixer_background_followup.json').read_text())
    names = ('focus','other','scd')
    paths = (focus, other, scd)
    runs = [read_archive(p, 'background_follow_'+name, reg) for name,p in zip(names,paths)]
    configs = [r[0] for r in runs]
    assert len({c['spec']['source_commit'] for c in configs}) == 1
    ids, seeds = reg['ids'], reg['seeds']
    other13 = sorted(set(CORRUPTIONS)-{'background','rotation'})
    assert len(other13)==13
    for c, expected_ids, conds, rep in zip(configs,[ids,ids,reg['scd_ids']],
            [['background','rotation','clean'],other13,['background','rotation','clean']],[seeds,[0],seeds]):
        assert c['ids']==expected_ids and set(c['conditions'])==set(conds) and c['seeds']==rep
    pm_data = dict(runs[0][1])
    assert pm_data.keys().isdisjoint(runs[1][1])
    pm_data.update(runs[1][1])
    view = 'bg_view'
    design = [(c,[c],seeds,view,'identity') for c in ('background','rotation','clean')]
    design += [('background_random',['background'],seeds,view,'bg_random_view'),
               ('macro13',other13,[0],view,'identity'),('all15',sorted(CORRUPTIONS),[0],view,'identity')]
    design_pm = [entry for entry in design if entry[0]!='background_random']
    pm = compare(pm_data,ids,design,sorted(CORRUPTIONS),[0],reg['bootstrap_draws'])
    limits = dict(background=3.,rotation=-1.,clean=-1.,macro13=-.5,all15=0.,background_random=0.)
    points, intervals = [], []
    for r in pm['comparisons']:
        group = r['group']
        points.append(r['delta_pp']>0 if group in ('all15','background_random') else r['delta_pp']+1e-9>=limits[group])
        intervals.append(r['joint_95_pp'][0]>(0 if group=='background' else limits[group]))
    pm['decision'] = dict(point=all(points),joint_interval=all(intervals),
                          speed=max(pm['timing'][view][k] for k in ('p50_ratio','p95_ratio'))<=1.10)
    pm['decision']['adopt'] = all(pm['decision'].values())
    design = [(c,[c],seeds,'bg_multiscale',control) for c in ('background','rotation','clean')
              for control in ('identity','bg_weighted','bg_multiscale_scalar')]
    scd_result = compare(runs[2][1],reg['scd_ids'],design,['background','rotation','clean'],seeds,reg['bootstrap_draws'])
    return dict(source_commit=configs[0]['spec']['source_commit'], archives={n:dict(file=p.name,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in zip(names,paths)},
                pm=pm,scd=scd_result,
                exploratory_windows=exploratory_windows(pm_data, ids, [(g,cs,ss) for g,cs,ss,_,_ in design_pm], reg['bootstrap_draws']),
                view_diagnostics={**runs[0][2], **runs[1][2]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('focus','other','scd','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    result = analyze(args.focus,args.other,args.scd)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
