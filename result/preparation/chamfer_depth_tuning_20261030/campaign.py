"""CPU registration and approval-gated bounded Chamfer depth campaign.

No default invocation runs research. Plans, stages, approvals and run slots are
immutable; an interrupted ledger requires explicit human-reviewed reconciliation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PREP = Path(__file__).resolve().parent
REPO = PREP.parents[2]
sys.path.insert(0, str(REPO))
RATES = (.003, .01, .03)
SEED = 20261030
RUNTIME = 'tta-chamfer-depth-tuning-a100-20261030-r1'
PHASE_SOURCES = [PREP / 'campaign.py', PREP / 'analyze.py', PREP / 'make_binding.py', REPO / 'scripts/analyze_scan_trajectory.py',
                 PREP / 'operations.py', REPO / 'scripts/analyze_legacy_trajectory.py', REPO / 'research_artifacts.py']


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(value))


def config(depth, eta=.01, gamma=.01, p=.95):
    require(type(depth) is int and 1 <= depth <= 10, 'Normal depth must be 1..10')
    require(eta in RATES and gamma in RATES and p in (.9, .95, .96, .99), 'Unregistered rates or p')
    return dict(normal_reverse_steps=depth, eta=float(eta), gamma=float(gamma), p=float(p))


def config_id(value):
    return 'n%d-e%s-g%s-p%s' % (value['normal_reverse_steps'], value['eta'], value['gamma'], value['p'])


def default_configs():
    return [config(n) for n in range(1, 11)]


def reserve(destination):
    """SHA-ranked original IDs: independent of NumPy/RNG implementation and labels."""
    previous = [PREP.parent / 'guidance_component_study_20261025' / ('modelnet40_c_' + s + '_ids.json')
                for s in ('development', 'heldout')]
    previous.append(PREP.parent / 'guidance_component_sensitivity_20261025/modelnet40_c_sensitivity_confirmation_ids.json')
    excluded = set()
    for path, count in zip(previous, (70, 70, 140)):
        manifest = read(path)
        ids = manifest['indices']
        require(manifest['dataset'] == 'modelnet40_c' and manifest['n_examples'] == 2468 and
                len(ids) == count and ids == sorted(set(ids)), 'Invalid prior cohort: ' + str(path))
        require(not excluded.intersection(ids), 'Prior cohorts overlap')
        excluded.update(ids)
    ordered = sorted(set(range(2468)) - excluded,
                     key=lambda i: hashlib.sha256(('%d:%d' % (SEED, i)).encode()).digest())
    output = Path(destination)
    require(not output.exists(), 'Reservation directory already exists')
    output.mkdir(parents=True)
    for stage, ids in [('development', sorted(ordered[:128])), ('heldout', sorted(ordered[128:384]))]:
        manifest = dict(dataset='modelnet40_c', n_examples=2468, stage=stage, indices=ids,
                        selection_seed=SEED, selection_algorithm='ascending SHA256(UTF8(seed:original_ID))',
                        selected_without_labels=True, split_unit='original_object_id_across_all_corruptions',
                        prior_benchmark_exposure=True, purpose='stage-reserved; not externally unseen',
                        excluded_prior_cohorts=[dict(path=str(p.relative_to(REPO)), **identity(p)) for p in previous])
        write_new(output / ('modelnet40_c_' + stage + '_ids.json'), manifest)
    write_new(output / 'reservation.json', dict(excluded_prior_ids=sorted(excluded),
              development=identity(output / 'modelnet40_c_development_ids.json'),
              confirmation=identity(output / 'modelnet40_c_heldout_ids.json'), unique_reserved_objects=384))


def validate_binding(binding):
    """Binding is produced only after final worker/source/assets exist, never placeholders."""
    expected = binding['expected_config']
    for key in ('source_snapshot_sha256', 'asset_manifest_sha256'):
        require(is_sha(expected.get(key)), 'Unbound ' + key)
    require(is_sha(expected.get('checkpoint_load', {}).get('checkpoint_sha256')), 'Unbound Point-MAE checkpoint')
    dataset = expected.get('dataset_provenance', {})
    require(dataset.get('n_examples') == 2468 and len(dataset.get('files', {})) >= 16 and
            all(is_sha(v.get('sha256')) and type(v.get('bytes')) is int and v['bytes'] > 0
                for v in dataset['files'].values()), 'Unbound dataset provenance/identities')
    require(expected['checkpoint_load'].get('strict') is True, 'Strict classifier checkpoint contract missing')
    for key in ('dataset_root', 'label_path', 'asset_root', 'pointmae_config', 'pointmae_checkpoint', 'repo_root', 'python'):
        require(isinstance(binding.get(key), str) and binding[key].startswith('/'), 'Unbound absolute worker path: ' + key)
    files = binding.get('required_source_files', {})
    require(all(any(prefix + name in files for prefix in ('repo/', 'assets/source_snapshot/'))
                for name in ('tta.py', 'shared_trajectory.py', 'scripts/eval_chamfer_depth.py')),
            'Required source coverage missing')
    require(all(is_sha(v.get('sha256')) and type(v.get('bytes')) is int and v['bytes'] > 0 for v in files.values()),
            'Invalid required source identities')


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def register(binding_path, selections, destination, cap, estimate):
    binding = read(binding_path)
    validate_binding(binding)
    require(math.isfinite(cap) and math.isfinite(estimate) and 0 < estimate <= cap, 'Invalid whole-phase duration')
    selections = Path(selections)
    ids = {}
    for name, count in [('development', 128), ('heldout', 256)]:
        path = selections / ('modelnet40_c_' + name + '_ids.json')
        value = read(path)
        require(len(value['indices']) == count and value['indices'] == sorted(set(value['indices'])), 'Invalid IDs')
        ids[name] = dict(manifest=value, identity=identity(path))
    require(not set(ids['development']['manifest']['indices']) & set(ids['heldout']['manifest']['indices']), 'Split overlap')
    # Recompute source-independent reservation; an edited split cannot be silently sealed.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        fresh = Path(temporary) / 'ids'
        reserve(fresh)
        for name in ids:
            require(ids[name]['manifest'] == read(fresh / ('modelnet40_c_' + name + '_ids.json')), 'Reservation changed')
    plan = dict(schema_version=1, status='registered_CPU_only', runtime=RUNTIME,
                phase_sources={str(p.relative_to(REPO)): identity(p) for p in PHASE_SOURCES},
                remote_root='/content/studies/chamfer-depth-tuning',
                cpu_preflight_receipt_path='/content/studies/chamfer-depth-tuning/cpu_preflight_receipt.json',
                drive_root='/content/drive/MyDrive/thesis/result/chamfer_depth_tuning_20261030',
                binding=binding, selections=ids, max_dev_configs=32, confirmation_seeds=[0, 1, 2],
                background_reverse_steps=35, batch_size=32, primary_readout='probability_mean',
                default=dict(p=.95, gamma=.01, eta=.01), rate_mapping=dict(gamma='local', eta='style'),
                depth_values=list(range(1, 11)), rate_values=list(RATES), p_values=[.9, .95, .96, .99],
                selection='all15 probability_mean; exact ties lower depth, then eta,gamma,p ascending',
                stage_a=default_configs(), top_depths=2,
                confirmation_roles=['depth5-default', 'stageA-best-default', 'best-tuned'],
                duration=dict(whole_phase_estimate_seconds=estimate, cumulative_wall_cap_seconds=cap,
                              final_ledger_sealing_reserve_seconds=60,
                              per_job_acceptance_copy_reserve_seconds=180,
                              budget_policy='measured stop-before-next-worker budget; not a strict clock guarantee; retain acceptance/sealing overruns and stop without more GPU',
                              includes='research worker, probes, failed work, validation, sealing, transfer; charge actual phase wall plus60s final ledger-copy reserve'),
                technical_smoke=dict(depths=[1, 5, 10], corruptions=['gaussian', 'background'],
                                     indices=ids['development']['manifest']['indices'][:32], seed=0,
                                     defaults=dict(p=.95, eta=.01, gamma=.01), selects_settings=False),
                no_failure_replacement=True, historical_benchmark_exposure=True,
                limitations=['finite sequential search, no global optimum', 'stage-reserved objects, not externally unseen'])
    write_new(destination, plan)
    return identity(destination)


def validate_plan(plan):
    validate_binding(plan['binding'])
    require(plan['runtime'] == RUNTIME and plan['stage_a'] == default_configs() and plan['max_dev_configs'] == 32 and
            plan['confirmation_seeds'] == [0, 1, 2] and plan['background_reverse_steps'] == 35 and
            plan['p_values'] == [.9, .95, .96, .99] and plan['rate_values'] == list(RATES) and
            plan['technical_smoke'] == dict(depths=[1, 5, 10], corruptions=['gaussian', 'background'],
                indices=plan['selections']['development']['manifest']['indices'][:32], seed=0,
                defaults=dict(p=.95, eta=.01, gamma=.01), selects_settings=False),
            'Changed scientific registry')
    require(plan.get('phase_sources') == {str(p.relative_to(REPO)): identity(p) for p in PHASE_SOURCES},
            'Controller/analyzer dependency source changed after registration')


def ranked(configs, scores):
    lookup = {row['config_id']: row for row in scores}
    require(len(lookup) == len(scores), 'Duplicate score configurations')
    for value in configs:
        row = lookup.get(config_id(value), {})
        require(row.get('status') == 'accepted' and row.get('readout') == 'probability_mean' and
                row.get('seed') == 0 and row.get('stage') == 'development' and
                math.isfinite(row.get('accuracy_pct', float('nan'))), 'Required selection endpoint unavailable: ' + config_id(value))
        require(is_sha(row.get('bundle_sha256')), 'Score has no accepted raw bundle SHA')
    return sorted(configs, key=lambda c: (-lookup[config_id(c)]['accuracy_pct'], c['normal_reverse_steps'], c['eta'], c['gamma'], c['p']))


def exact_score_scope(scores, configs):
    require({r['config_id'] for r in scores} == {config_id(c) for c in configs},
            'Scores must cover exactly the registered completed development slots; no extra search')


def stage(plan_path, scores_path, phase):
    plan = read(plan_path)
    validate_plan(plan)
    scores = read(scores_path) if scores_path else []
    require(all(r.get('plan_sha256') == identity(plan_path)['sha256'] for r in scores), 'Scores from another plan')
    stage_a = plan['stage_a']
    prior = []
    if phase == 'smoke':
        values, seeds = [config(n) for n in (1, 5, 10)], [0]
    elif phase == 'depth':
        values, seeds = stage_a, [0]
    else:
        best_a = ranked(stage_a, scores)
        depths = [c['normal_reverse_steps'] for c in best_a[:2]]
        grid = [config(n, e, g) for n in depths for e in RATES for g in RATES]
        prior = stage_a
        if phase == 'rates':
            exact_score_scope(scores, stage_a)
            values, seeds = grid, [0]
        else:
            best_rates = [ranked([c for c in grid if c['normal_reverse_steps'] == n], scores)[0] for n in depths]
            fraction = [config(c['normal_reverse_steps'], c['eta'], c['gamma'], p)
                        for c in best_rates for p in (.9, .95, .96, .99)]
            prior = stage_a + grid
            if phase == 'fraction':
                exact_score_scope(scores, prior)
                values, seeds = fraction, [0]
            elif phase == 'confirmation':
                best = ranked(fraction, scores)[0]
                values, seeds = [config(5), best_a[0], best], [0, 1, 2]
                ranked(list({config_id(c): c for c in stage_a + grid + fraction}.values()), scores)
                exact_score_scope(scores, stage_a + grid + fraction)
            else:
                raise ValueError('Unknown phase')
    seen, jobs, reused = set(), [], []
    prior_ids = {config_id(c) for c in prior}
    for value in values:
        key = config_id(value)
        if key in seen:
            continue
        seen.add(key)
        if phase != 'confirmation' and key in prior_ids:
            reused.append(key)
            continue
        for seed in seeds:
            jobs.append(dict(config=value, config_id=key, seed=seed,
                             split='heldout' if phase == 'confirmation' else 'development',
                             technical_smoke=phase == 'smoke',
                             slot_id=phase + '-' + key + '-s' + str(seed)))
    return dict(schema_version=1, phase=phase, plan_sha256=identity(plan_path)['sha256'],
                score_input_sha256=identity(scores_path)['sha256'] if scores_path else None,
                frozen_conditions=list({config_id(c): c for c in values}.values()) if phase == 'confirmation' else None,
                condition_roles=[dict(role=r, config_id=config_id(c)) for r, c in zip(plan['confirmation_roles'], values)]
                    if phase == 'confirmation' else None,
                jobs=jobs, reused_config_ids=reused)


def check_approval(plan_path, approval):
    plan = read(plan_path)
    validate_plan(plan)
    require(approval.get('approved') is True and approval.get('user_statement') and approval.get('approved_at_utc'),
            'Explicit whole-phase user approval missing')
    require(approval.get('plan_sha256') == identity(plan_path)['sha256'] and
            approval.get('source_snapshot_sha256') == plan['binding']['expected_config']['source_snapshot_sha256'] and
            approval.get('cumulative_wall_cap_seconds') == plan['duration']['cumulative_wall_cap_seconds'],
            'Approval not bound to exact final plan/source/duration')
    return plan


def worker_command(plan, job, result_root, manifest_path, paired_root):
    b, c = plan['binding'], job['config']
    argv = [b['python'], '-u', str(Path(b['repo_root']) / 'scripts/eval_chamfer_depth.py')]
    for option, value in [('dataset-root', b['dataset_root']), ('label-path', b['label_path']),
                          ('asset-root', b['asset_root']), ('asset-manifest-sha256', b['expected_config']['asset_manifest_sha256']),
                          ('result-root', result_root), ('drive-root', plan['drive_root']), ('seed', job['seed']),
                          ('normal-reverse-steps', c['normal_reverse_steps']), ('p', c['p']), ('gamma', c['gamma']), ('eta', c['eta']),
                          ('index-manifest', manifest_path), ('paired-state-root', Path(paired_root) / ('seed%d' % job['seed'])),
                          ('pointmae-config', b['pointmae_config']), ('pointmae-checkpoint', b['pointmae_checkpoint'])]:
        argv += ['--' + option, str(value)]
    if job.get('technical_smoke'):
        argv += ['--technical-smoke', '--corruptions', 'gaussian', 'background']
    return argv


def ledger_events(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []
    previous = None
    for row in rows:
        require(row.get('previous_sha256') == previous, 'Ledger chain mismatch')
        previous = digest(row)
    starts = [r for r in rows if r['event'] == 'phase_started']
    ends = [r for r in rows if r['event'] == 'phase_finished']
    require(len(starts) == len(ends), 'Unresolved interrupted/active phase; preserve partials and reconcile explicitly')
    return rows, sum(r['elapsed_seconds'] for r in ends) + sum(r['elapsed_seconds'] for r in rows if r['event'] == 'sealing_overrun')


def append_event(path, rows, **event):
    row = dict(previous_sha256=digest(rows[-1]) if rows else None, utc_epoch=time.time(), **event)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    rows.append(row)


def run_phase(plan_path, stage_path, scores_path, approval_path, root):
    """Sequential execution; any failure stops, preserves its slot, and is never retried."""
    plan = check_approval(plan_path, read(approval_path))
    from operations import verify_preflight
    verify_preflight(plan)
    registered = read(stage_path)
    require(registered == stage(plan_path, scores_path, registered['phase']), 'Stage differs from registered selection')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'controller.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, encoded(dict(pid=os.getpid(), plan_sha256=identity(plan_path)['sha256'])))
    os.close(descriptor)
    started = None
    rows = []
    ledger = root / 'wall_ledger.jsonl'
    try:
        rows, spent = ledger_events(ledger)
        require(all(r.get('plan_sha256') == registered['plan_sha256'] for r in rows), 'Ledger belongs to another plan')
        slots = root / 'slots'
        slots.mkdir(exist_ok=True)
        for job in registered['jobs']:
            require(not (slots / job['slot_id']).exists(), 'Slot already attempted; no automatic failure replacement')
        if registered['phase'] != 'smoke':
            smoke_stage = stage(plan_path, None, 'smoke')
            for job in smoke_stage['jobs']:
                receipt = slots / job['slot_id'] / 'acceptance.json'
                require(receipt.is_file() and read(receipt).get('status') == 'accepted' and
                        read(receipt).get('plan_sha256') == registered['plan_sha256'],
                        'Accepted exact technical smoke required before development')
        write_new(root / ('registered_' + registered['phase'] + '.json'), registered)
        if scores_path:
            write_new(root / ('registered_' + registered['phase'] + '_scores.json'), read(scores_path))
        started = time.monotonic()
        append_event(ledger, rows, event='phase_started', phase=registered['phase'], plan_sha256=registered['plan_sha256'])
        from analyze import analyze_bundle
        from operations import persist_checkpoint, run_worker
        for name, value in [('plan.json', plan), ('approval.json', read(approval_path))]:
            target = root / name
            if target.exists():
                require(read(target) == value, 'Immutable campaign identity/approval changed')
            else:
                write_new(target, value)
        reserve = plan['duration']['final_ledger_sealing_reserve_seconds']
        for job in registered['jobs']:
            remaining = (plan['duration']['cumulative_wall_cap_seconds'] - spent - (time.monotonic() - started)
                         - reserve - plan['duration']['per_job_acceptance_copy_reserve_seconds'])
            require(remaining > 0, 'Cumulative wall cap exhausted')
            slot = slots / job['slot_id']
            slot.mkdir()
            write_new(slot / 'job.json', job)
            manifest = slot / 'selection_manifest.json'
            selection = plan['selections'][job['split']]['manifest']
            if job.get('technical_smoke'):
                selection = dict(selection, stage='smoke', indices=plan['technical_smoke']['indices'])
            write_new(manifest, selection)
            paired = root / ('paired-smoke' if job.get('technical_smoke') else 'paired')
            command = worker_command(plan, job, slot / 'raw', manifest, paired)
            write_new(slot / 'command.json', command)
            append_event(ledger, rows, event='job_started', slot_id=job['slot_id'], plan_sha256=registered['plan_sha256'])
            try:
                returncode = run_worker(command, plan['binding']['repo_root'], slot / 'console.log', remaining, job['slot_id'])
                require(returncode == 0, 'Worker failed: %s' % returncode)
                bundles = list((slot / 'raw').rglob('bundle_manifest.json'))
                require(len(bundles) == 1, 'Expected one sealed Point-MAE bundle')
                raw = Path(str(bundles[0].parent) + '.zip')
                require(raw.is_file(), 'Worker sealed ZIP missing')
                report = analyze_bundle(raw, plan, job, cache_root=root)
                write_new(slot / 'acceptance.json', report)
                append_event(ledger, rows, event='job_accepted', slot_id=job['slot_id'], bundle_sha256=identity(raw)['sha256'],
                             plan_sha256=registered['plan_sha256'])
                published = persist_checkpoint(root, plan['drive_root'], registered['plan_sha256'], len(rows), slot, report['cache_proofs'])
                append_event(ledger, rows, event='drive_snapshot_verified', snapshot=published, plan_sha256=registered['plan_sha256'])
            except BaseException as error:
                write_new(slot / 'failure.json', dict(status='unavailable', error=repr(error)))
                append_event(ledger, rows, event='job_failed', slot_id=job['slot_id'], error=repr(error),
                             plan_sha256=registered['plan_sha256'])
                published = persist_checkpoint(root, plan['drive_root'], registered['plan_sha256'], len(rows), slot, preserve_raw=True)
                append_event(ledger, rows, event='drive_snapshot_verified', snapshot=published, plan_sha256=registered['plan_sha256'])
                raise
            require(spent + time.monotonic() - started + reserve <= plan['duration']['cumulative_wall_cap_seconds'],
                    'Cap exceeded during sealing/acceptance; stop before any further work')
    finally:
        if started is not None:
            reserve = plan['duration']['final_ledger_sealing_reserve_seconds']
            append_event(ledger, rows, event='phase_finished', phase=registered['phase'],
                         elapsed_seconds=time.monotonic() - started + reserve,
                         final_ledger_reserved_seconds=reserve, plan_sha256=registered['plan_sha256'])
            sealed_at = time.monotonic()
            persist_checkpoint(root, plan['drive_root'], registered['plan_sha256'], len(rows))
            sealing = time.monotonic() - sealed_at
            if sealing > reserve:
                append_event(ledger, rows, event='sealing_overrun', elapsed_seconds=sealing - reserve,
                             plan_sha256=registered['plan_sha256'])
                raise RuntimeError('Final ledger persistence exceeded reserved wall; preserve state and review before further work')
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    reserve_cli = sub.add_parser('reserve')
    reserve_cli.add_argument('--output', type=Path, required=True)
    register_cli = sub.add_parser('register')
    for option in ('binding', 'selections', 'output'):
        register_cli.add_argument('--' + option, type=Path, required=True)
    register_cli.add_argument('--cap-seconds', type=float, required=True)
    register_cli.add_argument('--estimate-seconds', type=float, required=True)
    stage_cli = sub.add_parser('stage')
    stage_cli.add_argument('--plan', type=Path, required=True)
    stage_cli.add_argument('--scores', type=Path)
    stage_cli.add_argument('--phase', choices=('smoke', 'depth', 'rates', 'fraction', 'confirmation'), required=True)
    stage_cli.add_argument('--output', type=Path, required=True)
    run_cli = sub.add_parser('run')
    for option in ('plan', 'stage', 'approval', 'root'):
        run_cli.add_argument('--' + option, type=Path, required=True)
    run_cli.add_argument('--scores', type=Path)
    args = parser.parse_args()
    if args.action == 'reserve':
        reserve(args.output)
    elif args.action == 'register':
        print(json.dumps(register(args.binding, args.selections, args.output, args.cap_seconds, args.estimate_seconds)))
    elif args.action == 'stage':
        write_new(args.output, stage(args.plan, args.scores, args.phase))
    else:
        run_phase(args.plan, args.stage, args.scores, args.approval, args.root)


if __name__ == '__main__':
    main()
