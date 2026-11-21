#!/usr/bin/env python3
"""Paired, original-object-clustered confirmation readouts from B's immutable logits."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_store import canonical, sha, unit_key
from scripts.analyze_cloudfixer_exploration import correctness_auc
from scripts.analyze_cloudfixer_weighted_pm import entropy_readouts
from scripts.run_cloudfixer_exploration import unpack
from scripts.run_cloudfixer_hypotheses import GEOMETRY, LATE, read_pilot_environment, registration, units, validate


def entropy_diagnostics(probability, entropy, labels, threshold):
    correct = probability.argmax(-1) == labels[:, None]
    mixed = correct.any(1) & ~correct.all(1)
    total_score = 0.
    pair_count = 0
    example_scores = []
    for row, right in zip(entropy[mixed], correct[mixed]):
        d = row[~right][None, :] - row[right][:, None]
        score = (d>0) + .5*(d==0)
        total_score += score.sum()
        pair_count += score.size
        example_scores.append(score.mean())
    confident = probability.max(-1) >= threshold
    return dict(pooled_correct_vs_wrong_auc=correctness_auc(-entropy, correct),
        within_example_pair_auc=float(total_score/pair_count) if pair_count else None,
        within_example_mean_auc=float(np.mean(example_scores)) if example_scores else None,
        within_example_pairs=pair_count, mixed_examples=int(mixed.sum()),
        wrong_confident=int((confident & ~correct).sum()), confident=int(confident.sum()),
        confidence_threshold=threshold, wrong=int((~correct).sum()), total_candidates=int(correct.size))


def paired(correct, reference, ids, draws):
    delta = correct.astype(np.int8)-reference.astype(np.int8)
    unique, inverse = np.unique(ids, return_inverse=True)
    per_object = np.bincount(inverse, weights=delta)/np.bincount(inverse)
    if len(unique) != draws.shape[1]:
        raise ValueError('Bootstrap lost an original-object cluster')
    interval = np.quantile(per_object[draws].mean(1)*100, [.025, .975])
    return dict(n=len(delta), correct=int(correct.sum()), reference_correct=int(reference.sum()),
        repair=int((delta==1).sum()), harm=int((delta==-1).sum()), delta_pp=float(100*delta.mean()),
        ci95_low_pp=float(interval[0]), ci95_high_pp=float(interval[1]))


def true_margin(scores, labels):
    """True class minus largest other class, allowing negative margins on wrong predictions."""
    truth = scores[np.arange(len(labels)), labels]
    ordered = np.partition(scores, -2, axis=-1)
    rival = np.where(scores.argmax(-1)==labels, ordered[:, -2], ordered[:, -1])
    return truth-rival


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output: Path, records: list, wanted: list, reg: dict, campaign: str):
    expected = {unit_key(u) for u in wanted}
    actual = {unit_key(r['unit']) for r in records}
    if expected != actual or len(records) != len(expected) or len(wanted) != len(expected):
        raise ValueError('Complete registered unit coverage required')
    draws = np.random.default_rng(reg['bootstrap_seed']).integers(
        len(reg['confirmation_ids']), size=(reg['bootstrap_draws'], len(reg['confirmation_ids'])))
    summary, per_corruption, pairs, diagnostics, terminal_cases = [], [], [], {}, []
    arm_outcomes, common_native, pairing = {}, {}, {}
    for config in reg['schedules']:
        group = [(r['unit'], unpack(r['data'])) for r in records if r['unit']['config'] == config]
        group.sort(key=lambda pair: (pair[0]['seed'], pair[0]['corruption'], pair[0]['ids']))
        def cat(name):
            return np.concatenate([p[name] for u, p in group])
        tokens = [(u['seed'], u['corruption'], tuple(u['ids'])) for u, p in group]
        provenance = {key: (json.loads(str(p['sampler']))['input_rng_sha256'], p['model_hashes'].tolist(),
                            p['labels'].tolist()) for key, (u, p) in zip(tokens, group)}
        if pairing and pairing != provenance:
            raise ValueError('N8/N12 strength comparisons used different inputs, RNG, models or labels')
        pairing = provenance
        seed = np.array([u['seed'] for u, p in group for _ in u['ids']])
        corruption = np.array([u['corruption'] for u, p in group for _ in u['ids']])
        ids, labels, p, h = (cat(k) for k in ('ids','labels','probabilities','entropy'))
        times = json.loads(str(group[0][1]['sampler']))['candidate_indices']
        n = config['steps']
        name = 'N'+str(n)+'_tau'+str(config['tau'])
        windows = dict(canonical=[n//4-1, n//2-1, 3*n//4-1,n], all=list(range(n+1)),
                       late=[times.index(t) for t in LATE[n]])
        terminal = p[:, -1].argmax(-1)==labels
        post = np.concatenate((cat('post_probabilities'), p[:, -1:]), axis=1)
        pre_extra = cat('pre_extra_t0_probabilities')
        pre_logits, post_logits = cat('logits'), cat('post_logits')
        geometry = {field: cat(field) for field in GEOMETRY}
        cumulative = geometry['correction_rms'].cumsum(1)
        dg = dict(config=config, entropy={}, positions=[], terminal_transitions={})
        diagnostics[name] = dg
        policies = dict(terminal=terminal, pre_extra_t0_state=pre_extra.argmax(-1)==labels,
                        last_post_clean=post[:, -2].argmax(-1)==labels)
        for window, slots in windows.items():
            prob, entropy = p[:, slots], h[:, slots]
            dg['entropy'][window] = entropy_diagnostics(prob, entropy, labels, reg['confidence_threshold'])
            for rule, (means, weights) in entropy_readouts(prob, entropy).items():
                if rule != 'terminal':
                    policies[window+'_'+rule] = means.argmax(-1)==labels
            policies[window+'_post_equal_pm'] = post[:, slots].mean(1).argmax(-1)==labels

        late_single = windows['late'][0]
        late_ok = p[:, late_single].argmax(-1)==labels
        common_native[name] = {native:p[:,i].argmax(-1)==labels for i,native in enumerate(times)}
        policies['first_locked_late_probe'] = late_ok
        for idx, native in enumerate(times):
            pre_ok = p[:,idx].argmax(-1)==labels
            pos = dict(native=native, pre_correct=int(pre_ok.sum()), post_correct=int((post[:,idx].argmax(-1)==labels).sum()),
                mean_pre_logit_margin=float(true_margin(pre_logits[:,idx],labels).mean()),
                mean_post_logit_margin=float(true_margin(post_logits[:,idx] if idx<n else pre_logits[:,idx], labels).mean()),
                high_confidence_wrong=int(((p[:,idx].max(-1)>=reg['confidence_threshold']) & ~pre_ok).sum()),
                mean_cumulative_correction_rms=float(cumulative[:,idx].mean()))
            pos.update({key:float(value[:,idx].mean()) for key,value in geometry.items()})
            dg['positions'].append(pos)
        dg['endpoint'] = dict(pre_extra_t0_state_correct=int(policies['pre_extra_t0_state'].sum()),
            last_post_clean_correct=int(policies['last_post_clean'].sum()), true_terminal_correct=int(terminal.sum()),
            pre_extra_t0_to_terminal_mean_rms=float(cat('pre_extra_t0_to_terminal_rms').mean()),
            mean_pre_extra_t0_logit_margin=float(true_margin(cat('pre_extra_t0_logits'),labels).mean()))
        dg['late_to_terminal_groups'] = {}
        endpoint_displacement = cat('pre_extra_t0_to_terminal_rms')
        late_margin = true_margin(pre_logits[:,late_single], labels)
        terminal_margin = true_margin(pre_logits[:,-1], labels)
        for transition,take in (('late_correct_terminal_wrong', late_ok & ~terminal),
                                ('late_wrong_terminal_correct', ~late_ok & terminal),
                                ('both_correct', late_ok & terminal), ('both_wrong', ~late_ok & ~terminal)):
            metrics = {'late_true_class_margin':late_margin,'terminal_true_class_margin':terminal_margin,
                       'correction_rms_after_late':geometry['correction_rms'][:,late_single:].sum(1),
                       'pre_extra_t0_to_terminal_rms':endpoint_displacement,
                       'terminal_drift_from_input_rms':geometry['pre_to_input_rms'][:,-1]}
            dg['late_to_terminal_groups'][transition] = dict(
                n=int(take.sum()), **{field:float(a[take].mean()) if take.any() else None for field,a in metrics.items()})
        for c in reg['corruptions']:
            select = corruption==c
            dg['terminal_transitions'][c] = dict(
                first_locked_late_correct_to_terminal_wrong=int((select & late_ok & ~terminal).sum()),
                first_locked_late_wrong_to_terminal_correct=int((select & ~late_ok & terminal).sum()))
        for i in np.flatnonzero(late_ok != terminal):
            terminal_cases.append(dict(schedule=name, seed=int(seed[i]), corruption=corruption[i],
                original_id=int(ids[i]), label=int(labels[i]), native=times[late_single],
                late_prediction=int(p[i,late_single].argmax()), terminal_prediction=int(p[i,-1].argmax()),
                late_correct=bool(late_ok[i]), terminal_correct=bool(terminal[i])))
        row_base = dict(run_id=campaign,dataset=reg['dataset'],severity=reg['severity'],schedule=name)
        costs = {(s,c): (sum(float(v['seconds']) for u,v in group if u['seed']==s and u['corruption']==c),
                         max(float(v['peak_mb']) for u,v in group if u['seed']==s and u['corruption']==c))
                 for s in reg['seeds'] for c in reg['corruptions']}
        for method, correct in policies.items():
            row = dict(row_base, method=method)
            ref = 'terminal'
            pairs.append(dict(row, seeds='pooled', reference=ref, **paired(correct, terminal, ids, draws)))
            for window in windows:
                if method.startswith(window+'_') and method != window+'_equal_pm':
                    ref = window+'_equal_pm'
                    pairs.append(dict(row, seeds='pooled', reference=ref, **paired(correct, policies[ref], ids, draws)))
                    break
            if method == 'late_equal_pm':
                ref = 'canonical_equal_pm'
                pairs.append(dict(row, seeds='pooled',reference=ref, **paired(correct, policies[ref], ids, draws)))
            for s in reg['seeds']:
                take = seed==s
                row_seed = dict(row,seed=s,status='complete')
                pairs.append(dict(row,seeds=str(s),reference='terminal', **paired(correct[take],terminal[take],ids[take],draws)))
                accru = []
                for c in reg['corruptions']:
                    m = take & (corruption==c)
                    if not np.array_equal(ids[m], reg['confirmation_ids']):
                        raise ValueError('Original object order/coverage differs across corruptions or seeds')
                    accuracy = float(correct[m].mean())
                    accru.append(accuracy)
                    secs,peak = costs[s,c]
                    per_corruption.append(dict(row_seed,corruption=c,n_examples=int(m.sum()),n_correct=int(correct[m].sum()),
                        accuracy=accuracy,repair=int((correct[m] & ~terminal[m]).sum()),harm=int((~correct[m] & terminal[m]).sum()),
                        runtime_seconds=secs,peak_gpu_memory_mb=peak))
                summary.append(dict(row_seed,n_corruptions=len(accru),macro_accuracy=float(np.mean(accru)),
                    total_examples=int(take.sum()),total_correct=int(correct[take].sum()),micro_accuracy=float(correct[take].mean()),
                    total_runtime_seconds=sum(costs[s,c][0] for c in reg['corruptions'])))
        arm_outcomes[name] = {key:policies[key] for key in ('terminal','first_locked_late_probe','late_equal_pm')}
    n8, n12, matched = arm_outcomes
    for target, reference in ((n12,n8),(matched,n8),(matched,n12)):
        for method in arm_outcomes[target]:
            pairs.append(dict(row_base,schedule=target,method=method,seeds='pooled',reference=reference+'/'+method,
                **paired(arm_outcomes[target][method],arm_outcomes[reference][method],ids,draws)))
        for native in sorted(common_native[target].keys() & common_native[reference].keys(),reverse=True):
            method = 'pre_clean_native_'+str(native) if native else 'true_terminal_native_0'
            pairs.append(dict(row_base,schedule=target,method=method,seeds='pooled',reference=reference+'/'+method,
                **paired(common_native[target][native],common_native[reference][native],ids,draws)))
    for name, rows in (('summary',summary),('per_corruption',per_corruption),('paired',pairs)):
        write_csv(output/(name+'.csv'), rows)
    fields = ('schedule','seed','corruption','original_id','label','native','late_prediction','terminal_prediction',
              'late_correct','terminal_correct')
    with (output/'terminal_transitions.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(terminal_cases)
    (output/'diagnostics.json').write_bytes(canonical(dict(campaign=campaign,configurations=diagnostics,
        confidence_threshold=reg['confidence_threshold'],bootstrap_seed=reg['bootstrap_seed'],
        bootstrap_draws=reg['bootstrap_draws'],n_object_clusters=len(reg['confirmation_ids']),
        n_units=len(records),timing='Diagnostic pipeline shared by all readouts of each unit; do not add across readouts.')))


def same_report(left, right):
    # Accommodate float64 reductions on different BLAS/CPU builds while requiring exact counts and identities.
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys()==right.keys() and all(same_report(v,right[k]) for k,v in left.items())
    if isinstance(left, list):
        return isinstance(right,list) and len(left)==len(right) and all(same_report(a,b) for a,b in zip(left,right))
    if isinstance(left,float):
        return isinstance(right,(int,float)) and math.isclose(left,right,abs_tol=1e-10,rel_tol=1e-12)
    if isinstance(left,str) and left!=right:
        try:
            if any(c in left for c in '.eE') and any(c in right for c in '.eE'):
                return math.isclose(float(left),float(right),abs_tol=1e-10,rel_tol=1e-12)
        except (TypeError,ValueError):
            pass
    return left==right


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from scripts.run_cloudfixer_transfer import file_hash
    if file_hash(args.archive) != args.sha256:
        raise ValueError('Raw archive SHA mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist()
        if (len(names)!=len(set(names)) or z.testzip() or
                any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names)):
            raise ValueError('Corrupt or ambiguous raw archive')
        configs = [n for n in names if n == 'config.json' or n.endswith('/config.json')]
        if len(configs)!=1:
            raise ValueError('Expected one configuration')
        prefix = configs[0][:-len('config.json')]
        cfg = json.loads(z.read(configs[0]))
        reg = registration()
        manifest = cfg['manifest']
        campaign = sha(canonical(dict(manifest=manifest,role=reg['role'])))
        if cfg['execution_status']!='complete' or cfg['campaign']!=campaign or manifest['registration']!=reg:
            raise ValueError('Incomplete or wrong confirmation archive')
        wanted = units(campaign,reg)
        expected = {prefix+'pending_'+unit_key(u)+'.npz' for u in wanted}
        if expected != {n for n in names if PurePosixPath(n).name.startswith('pending_unit_')}:
            raise ValueError('Raw NPZ unit coverage differs')
        label_data = z.read(prefix+'verified_label.npy')
        if sha(label_data) != manifest['dataset_files']['label.npy']:
            raise ValueError('Archive original label SHA mismatch')
        labels = np.load(io.BytesIO(label_data), allow_pickle=False).reshape(-1)
        environments = {}
        records = []
        for u in wanted:
            key = unit_key(u)
            data = z.read(prefix+'pending_'+key+'.npz')
            receipt = json.loads(z.read(prefix+key))
            if receipt != dict(schema=1,unit=u,payload='data_'+sha(data)+'.npz',sha256=sha(data),bytes=len(data)):
                raise ValueError('Raw unit receipt mismatch')
            env_sha = str(unpack(data)['environment_sha256'])
            if env_sha not in environments:
                environments[env_sha] = read_pilot_environment(lambda name: z.read(prefix+name), env_sha, campaign, reg)
            validate(data,u,labels,manifest,environments[env_sha])
            records.append(dict(unit=u,data=data))
        write_reports(args.output,records,wanted,reg,campaign)
        for name in ('summary.csv','per_corruption.csv','paired.csv','diagnostics.json','terminal_transitions.csv'):
            local, archived = (args.output/name).read_text(), z.read(prefix+name).decode()
            if name.endswith('.json'):
                a, b = json.loads(local), json.loads(archived)
            else:
                a, b = (list(csv.DictReader(io.StringIO(s))) for s in (local,archived))
            if not same_report(a,b):
                raise ValueError('Recount differs: '+name)
        print('CF_HYPOTHESES_RECOUNT_PASS',campaign,len(records))


if __name__=='__main__':
    main()
