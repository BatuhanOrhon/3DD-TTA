"""Independently validate an extracted transfer snapshot; write fresh derived outputs.

Uses frozen per-head raw ZIPs and actual paired NPZs, never controller accuracy
as input. Historical contrasts consume the separately pinned 78-run recount.
No GPU, runtime, Drive, authentication or Git operations are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts import analyze_chamfer_depth_transfer as analysis
from scripts.prepare_chamfer_variants_assets import identity

HISTORY = REPO / 'result/preparation/chamfer_depth_transfer_20261031/cohort140'
ACTUAL_PLAN_SHA = '9154038be293c5ec9482a7010c4c7e0f1777763ab9421e58ab40d1ee54793554'
HISTORY_SHA = '531570b25d7e5bc0b2d6bb1b04a970d486ed0095da8462e4c64a38d02328499b'
OUTCOMES_SHA = '564cc8cc1862607d6ef4c366a8b859f440e449a39a14f0548c23368627f92081'
ERRATA_SHA = '641666ed30c471774ae9e23ba2eccb2d12334892797ab9775ab7cdf0c5732b99'
HEADS = ('pointmae', 'dgcnn', 'curvenet', 'pointnext')
require = analysis.require
np = analysis.np


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def mapped_raw_path(remote, campaign_root, local_campaign):
    """Only an exact registered campaign-relative path can address retrieved raw."""
    path, root = Path(remote), Path(campaign_root)
    require(path.is_absolute() and root.is_absolute() and '..' not in path.parts, 'Unsafe remote raw path')
    relative = path.relative_to(root)
    target = Path(local_campaign) / relative
    require(target.resolve().is_relative_to(Path(local_campaign).resolve()) and not target.is_symlink(), 'Raw path escapes snapshot campaign')
    return target


def compact_acceptance(job, reports, remote_raw_paths):
    require(len(reports) == 4 and {r['backbone'] for r in reports} == set(HEADS), 'Four independently validated heads required')
    analysis.analyze_reports(reports, repetitions=1)
    signature = {k: reports[0][k] for k in ('object_ids', 'labels', 'corruptions', 'reconstruction_group_id',
                                          'source_snapshot_sha256', 'asset_manifest_sha256', 'batch_guards')}
    return dict(status='accepted', slot_id=job['slot_id'], dataset=job['dataset'], seed=job['seed'], stage=job['stage'],
                shared_proof_sha256=hashlib.sha256(encoded(signature)).hexdigest(),
                heads={r['backbone']: dict(raw_zip=remote_raw_paths[r['backbone']],
                    checkpoint_load=job['bindings'][r['backbone']]['expected_config']['checkpoint_load'],
                    **{k: r[k] for k in ('bundle_identity', 'sealed_manifest_identity', 'aggregate')}) for r in reports},
                cache_proofs=reports[0]['cache_proofs'])


def validate_snapshot(snapshot):
    snapshot = Path(snapshot)
    campaign = snapshot / 'campaign'
    plan_path = snapshot / 'registered_plan.json'
    plan = read(plan_path)
    require((campaign / 'plan.json').read_bytes() == plan_path.read_bytes(), 'Registered/controller plan bytes differ')
    plan_sha = identity(plan_path)['sha256']
    require(plan_sha == ACTUAL_PLAN_SHA, 'Registered plan differs from frozen actual plan SHA')
    ledger = read(campaign / 'registered_latest_ledger.json')
    require(ledger.get('plan_sha256') == plan_sha and ledger.get('status') in ('complete', 'failed') and
            ledger.get('latest_event', {}).get('event') == 'controller_stopped', 'Expected stopped controller bound to registered plan')
    cpu_path = snapshot / 'cpu_preflight_receipt.json'
    cpu = read(cpu_path)
    require(identity(cpu_path) == plan['cpu_preflight_receipt_identity'] and
            (campaign / 'registered_cpu_preflight_receipt.json').read_bytes() == cpu_path.read_bytes(), 'CPU receipt identity differs')
    require(cpu.get('status') == cpu.get('report', {}).get('status') == 'complete' and
            cpu['report'].get('cuda_initialized') is False and cpu['report'].get('forward_executed') is False and
            cpu['report'].get('source_snapshot_sha256') == plan['source_snapshot_sha256'] and
            cpu['report'].get('manifest_sha256') == plan['asset_manifest_sha256'], 'Unavailable CPU source/assets proof')
    require(identity(snapshot / 'preflight_evidence/cpu_preflight.zip') == {k: cpu['archive'][k] for k in ('bytes', 'sha256')}, 'Physical CPU evidence ZIP differs')
    # Reused validator/helper bytes must still be the registered ones.
    sources = {}
    for name, expected in plan['phase_sources'].items():
        actual = identity(REPO / name)
        require(actual == expected, 'Local analysis dependency differs from frozen source: ' + name)
        sources[name] = actual
    jobs = plan['jobs']
    require(len(jobs) == 8 and len({j['slot_id'] for j in jobs}) == 8, 'Expected eight unique frozen jobs')
    require({(j['stage'], j['dataset'], j['seed']) for j in jobs} ==
            {('smoke', d, 0) for d in ('modelnet40_c', 'scanobjectnn_c')} |
            {('research', d, s) for d in ('modelnet40_c', 'scanobjectnn_c') for s in (0, 1, 2)}, 'Changed fixed job/seed scope')
    reports, unavailable, compacts = [], [], []
    require({p.name for p in (campaign / 'slots').iterdir()} <= {j['slot_id'] for j in jobs}, 'Undeclared slot present')
    for job in jobs:
        slot = campaign / 'slots' / job['slot_id']
        if (slot / 'failure.json').exists() or not (slot / 'acceptance.json').is_file():
            unavailable.append(dict(slot_id=job['slot_id'], dataset=job['dataset'], seed=job['seed'], stage=job['stage'],
                status='unavailable', reason=read(slot / 'failure.json') if (slot / 'failure.json').exists() else 'unattempted or no accepted complete endpoint'))
            continue
        try:
            require(read(slot / 'job.json') == job, 'Slot job differs from frozen plan')
            accepted = read(slot / 'acceptance.json')
            require(set(accepted['heads']) == set(HEADS), 'Controller acceptance lacks exact four heads')
            local_reports, remote_paths = [], {}
            for head in HEADS:
                remote_paths[head] = accepted['heads'][head]['raw_zip']
                raw = mapped_raw_path(remote_paths[head], plan['campaign_root'], campaign)
                require(raw.is_relative_to(slot / 'raw') and raw.suffix == '.zip', 'Raw ZIP outside registered slot')
                report = analysis.analyze_bundle(raw, job['bindings'][head], campaign)
                require((report['dataset'], report['backbone'], report['seed'], report['technical_smoke']) ==
                        (job['dataset'], head, job['seed'], job['stage'] == 'smoke'), 'Independent raw endpoint identity differs')
                report['checkpoint_load'] = job['bindings'][head]['expected_config']['checkpoint_load']
                local_reports.append(report)
            declared_raw = {mapped_raw_path(path, plan['campaign_root'], campaign) for path in remote_paths.values()}
            actual_raw = {Path(str(p.parent) + '.zip') for p in (slot / 'raw').rglob('bundle_manifest.json')}
            require(actual_raw == declared_raw, 'Accepted slot raw bundle inventory differs')
            compact = compact_acceptance(job, local_reports, remote_paths)
            require(compact == accepted, 'Controller compact identities/aggregates/shared proof differ from independent raw/cache analysis')
            compacts.append(compact)
            reports.extend(local_reports)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
            unavailable.append(dict(slot_id=job['slot_id'], dataset=job['dataset'], seed=job['seed'], stage=job['stage'],
                                    status='unavailable', reason='independent rejection: ' + str(error)))
    result = analysis.analyze_reports(reports, repetitions=10000)
    for dataset in ('modelnet40_c', 'scanobjectnn_c'):
        for head in HEADS:
            if not any(r['dataset'] == dataset and r['backbone'] == head and not r['technical_smoke'] for r in reports):
                result['comparisons'].append(dict(dataset=dataset, backbone=head, technical_smoke=False, status='unavailable',
                    reason='no independently accepted research seeds; no replacement endpoint'))
    result.update(status='independently_accepted_complete' if not unavailable and ledger['status'] == 'complete' else 'independent_partial_unavailable',
                  unavailable=unavailable, controller_compacts_reproduced=compacts, plan_identity=identity(plan_path),
                  cpu_receipt_identity=identity(cpu_path), controller_status=ledger['status'], reused_analysis_sources=sources)
    return result


def load_history(history_root):
    history_root = Path(history_root)
    files = {'historical_baselines.json': HISTORY_SHA, 'historical_baseline_outcomes.npz': OUTCOMES_SHA, 'errata.json': ERRATA_SHA}
    for name, sha in files.items():
        require(identity(history_root / name)['sha256'] == sha, 'Historical accepted identity changed: ' + name)
    historical = read(history_root / 'historical_baselines.json')
    errata = read(history_root / 'errata.json')
    plan = read(history_root / 'analysis_plan_v2.json')
    require(plan.get('registered_before_new_research_GPU') is True and plan['historical_receipt_sha256'] == HISTORY_SHA and
            plan['historical_errata_sha256'] == ERRATA_SHA and plan['uncertainty']['repetitions'] == 10000 and
            plan['uncertainty']['bootstrap_seed'] == 20261030, 'Historical analysis plan binding changed')
    require(errata['applies_to']['sha256'] == HISTORY_SHA and len(historical['historical_runs']) == 78, 'Historical receipt/errata coverage differs')
    require(identity(history_root / 'historical_baseline_outcomes.npz') == {k: historical['outcomes'][k] for k in ('bytes', 'sha256')}, 'Historical outcomes identity differs')
    for dataset, entry in historical['cohort'].items():
        require(identity(history_root / Path(entry['path']).name) == {k: entry[k] for k in ('bytes', 'sha256')}, 'Historical cohort file differs')
    with np.load(history_root / 'historical_baseline_outcomes.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return historical, arrays, errata, plan


def aligned_history(run, arrays, candidate, corruption_order, historical_policy):
    """Never silently align different objects, labels, checkpoints or seed identities."""
    require(run['dataset'] == candidate['dataset'] and run['backbone'] == candidate['backbone'] and run['seed'] == candidate['seed'], 'Historical dataset/head/seed mismatch')
    require(candidate['corruptions'] == corruption_order, 'Historical corruption ordering mismatch')
    prefix = run['outcome_prefix']
    index, labels = arrays[prefix + '_index'], arrays[prefix + '_label']
    expected_ids = np.broadcast_to(candidate['object_ids'], index.shape)
    expected_labels = np.broadcast_to(candidate['labels'], labels.shape)
    require(index.shape == labels.shape == (len(corruption_order), len(candidate['object_ids'])) and
            np.array_equal(index, expected_ids) and np.array_equal(labels, expected_labels), 'Historical original IDs/labels mismatch')
    protocol = run['provenance']['protocol']
    if isinstance(protocol, dict):
        checkpoint = protocol['checkpoint_load']['checkpoint_sha256']
    else:
        # The original-subset receipt retained raw SHA/provenance but omitted load metadata.
        archive_ref = run['archive']
        path = Path(archive_ref['path'])
        require(identity(path) == {k: archive_ref[k] for k in ('bytes', 'sha256')}, 'Historical original outer ZIP identity differs')
        with zipfile.ZipFile(path) as archive:
            require(archive.namelist().count(archive_ref['config_member']) == 1 and archive.testzip() is None, 'Historical original config/CRC mismatch')
            cfg = json.loads(archive.read(archive_ref['config_member']))
        require(cfg.get('backbone') == run['backbone'] and cfg.get('dataset') == run['dataset'] and cfg.get('seed') == 0 and
                cfg['checkpoint_load'].get('strict') is True, 'Historical original checkpoint config differs')
        checkpoint = cfg['checkpoint_load']['checkpoint_sha256']
    require(checkpoint == candidate['checkpoint_load']['checkpoint_sha256'], 'Historical classifier checkpoint mismatch')
    outcome = arrays[prefix + '_' + historical_policy + '_correct']
    require(outcome.dtype == np.bool_ and outcome.shape == index.shape, 'Historical outcome shape/type mismatch')
    metric = run['metrics'][historical_policy]
    require(int(outcome.sum()) == metric['correct'] and outcome.sum(1).tolist() == metric['per_corruption_correct'], 'Historical outcome recount differs')
    return outcome


def historical_contrasts(reports, historical, arrays):
    candidates = {}
    for report in reports:
        if report['technical_smoke']:
            continue
        key = (report['dataset'], report['backbone'], report['seed'])
        require(key not in candidates, 'Duplicate candidate seed; replacements forbidden')
        candidates[key] = report
    grouped = {}
    for run in historical['historical_runs']:
        key = (run['phase'], run['arm'], run['dataset'], run['backbone'])
        seeds = grouped.setdefault(key, {})
        require(run['seed'] not in seeds, 'Duplicate historical phase/arm/seed')
        seeds[run['seed']] = run
    contrasts = []
    for (phase, arm, dataset, head), runs in sorted(grouped.items()):
        original = arm == 'original3dd'
        seeds = [0] if original else [0, 1, 2]
        policy = 'terminal' if original else 'probability_mean'
        context = dict(phase=phase, arm=arm, dataset=dataset, backbone=head, historical_policy=policy,
                       candidate_policy='probability_mean', comparison='historical whole-policy; not isolated depth/rate/loss causality')
        if set(runs) != set(seeds) or any((dataset, head, seed) not in candidates for seed in seeds):
            contrasts.append(dict(**context, status='unavailable', reason='fixed required seed endpoint incomplete; no surviving-seed substitution'))
            continue
        selected = [candidates[dataset, head, seed] for seed in seeds]
        require(all(r['object_ids'] == selected[0]['object_ids'] and r['labels'] == selected[0]['labels'] for r in selected), 'Candidate IDs/labels differ across seeds')
        try:
            old = np.asarray([aligned_history(runs[seed], arrays, candidate, historical['corruptions'], policy) for seed, candidate in zip(seeds, selected)])
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
            contrasts.append(dict(**context, status='unavailable', reason='historical alignment rejected: ' + str(error)))
            continue
        new = np.asarray([candidate['outcomes']['probability_mean'] for candidate in selected], dtype=bool)
        provenance = [dict(seed=seed, archive=runs[seed]['archive'], provenance=runs[seed]['provenance']) for seed in seeds]
        for scope in ('all15', 'macro14'):
            mask = [i for i, c in enumerate(historical['corruptions']) if scope == 'all15' or c != 'background']
            x, y = new[:, mask], old[:, mask]
            for seed in ([0] if original else [None, 0, 1, 2]):
                a, b = (x, y) if seed is None or original else (x[seed:seed + 1], y[seed:seed + 1])
                comparison = analysis._depth_validators().compare(a, b, repetitions=10000)
                comparison['interval_scope'] = 'conditional on frozen policies/model/corruptions and fixed seeds ' + str(seeds if seed is None else [seed]) + '; original objects jointly resampled'
                contrasts.append(dict(**context, status='complete', scope=scope, seed=seed, fixed_seeds=seeds,
                    checkpoint_sha256=selected[0]['checkpoint_load']['checkpoint_sha256'],
                    candidate_accuracy_pct=float(a.mean() * 100), historical_accuracy_pct=float(b.mean() * 100),
                    historical_raw_provenance=provenance,
                    **comparison))
    # Point-MAE originals have no eligible indexed current-cohort endpoint, per preserved errata.
    for dataset in historical['cohort']:
        contrasts.append(dict(phase='historical_original_full_subset', arm='original3dd', dataset=dataset, backbone='pointmae',
                              status='unavailable', reason='No eligible current140-ID indexed original Point-MAE endpoint; see errata'))
    return contrasts


def candidate_summary(reports):
    rows = []
    for dataset in ('modelnet40_c', 'scanobjectnn_c'):
        for head in HEADS:
            seeds = {r['seed']: r for r in reports if r['dataset'] == dataset and r['backbone'] == head and not r['technical_smoke']}
            for scope in ('all15', 'macro14'):
                for policy in ('probability_mean', 'terminal'):
                    values = {}
                    for seed, report in seeds.items():
                        mask = [i for i, c in enumerate(report['corruptions']) if scope == 'all15' or c != 'background']
                        values[seed] = float(np.asarray(report['outcomes'][policy], dtype=bool)[mask].mean() * 100)
                    complete = set(values) == {0, 1, 2}
                    rows.append(dict(dataset=dataset, backbone=head, scope=scope, policy=policy,
                        status='complete' if complete else 'unavailable',
                        seed0_accuracy_pct=values.get(0), seed1_accuracy_pct=values.get(1), seed2_accuracy_pct=values.get(2),
                        mean_accuracy_pct=float(np.mean(list(values.values()))) if complete else None,
                        sample_sd_pp=float(np.std(list(values.values()), ddof=1)) if complete else None))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_snapshot(snapshot_dir, output_dir, history_root=HISTORY):
    require(np.__version__.split('.')[0] == '1', 'Independent physical RNG acceptance requires pinned NumPy1')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = validate_snapshot(snapshot_dir)
        historical, arrays, errata, historical_plan = load_history(history_root)
        result.update(historical_contrasts=historical_contrasts(result['bundles'], historical, arrays), historical_errata=errata,
                      historical_analysis_plan=historical_plan, historical_protocol_confounds=historical['protocol_confounds'],
                      analysis_operator_identity=identity(Path(__file__)),
                      historical_inputs={n: identity(Path(history_root) / n) for n in
                          ('historical_baselines.json', 'historical_baseline_outcomes.npz', 'errata.json', 'analysis_plan_v2.json')},
                      limitations=['conditional uncorrected per-comparison object intervals; historically exposed140-object cohorts',
                                   'separate dataset/head endpoints; no model ensemble or global optimum claim',
                                   'historical whole-policy differences do not isolate depth/rate/style/loss effects',
                                   'geometry is sealed hash evidence, without CPU reconstruction replay'])
        write_new(output / 'independent_analysis.json', result)
        per_head = [dict(dataset=r['dataset'], backbone=r['backbone'], seed=r['seed'], technical_smoke=r['technical_smoke'], **row)
                    for r in result['bundles'] for row in r['aggregate']]
        write_csv(output / 'candidate_per_head.csv', candidate_summary(result['bundles']))
        write_csv(output / 'candidate_per_head_seed.csv', per_head)
        write_csv(output / 'candidate_per_corruption_seed.csv', [dict(technical_smoke=r['technical_smoke'], **row) for r in result['bundles'] for row in r['per_corruption']])
        # Full provenance remains in JSON; compact CSV keeps comparisons readable.
        cols = ('phase', 'arm', 'dataset', 'backbone', 'status', 'scope', 'seed', 'candidate_accuracy_pct', 'historical_accuracy_pct',
                'delta_pp', 'conditional_object_bootstrap_low_pp', 'conditional_object_bootstrap_high_pp', 'reason')
        write_csv(output / 'historical_contrasts.csv', [{k: row.get(k) for k in cols} for row in result['historical_contrasts']])
        for name in ('historical_baselines.json', 'errata.json', 'analysis_plan_v2.json'):
            shutil.copyfile(Path(history_root) / name, output / name)
        return result
    except BaseException:
        import traceback
        write_new(output / 'analysis_failure.json', dict(status='failed', traceback=traceback.format_exc()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--history-root', type=Path, default=HISTORY)
    args = parser.parse_args(argv)
    result = analyze_snapshot(args.snapshot, args.output, args.history_root)
    print(json.dumps(dict(status=result['status'], independently_validated_heads=len(result['bundles']), unavailable=result['unavailable'])))


if __name__ == '__main__':
    main()
