"""Run one immutable, explicitly approved eight-job frozen-N8 transfer campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import analyze_chamfer_depth_transfer as analysis
from scripts import eval_chamfer_depth as depth

PREP = 'result/preparation/chamfer_depth_tuning_20261030/'
PHASE_SOURCES = tuple(dict.fromkeys((*analysis.CRITICAL_SOURCES,
    'scripts/run_chamfer_depth_transfer.py', 'scripts/analyze_chamfer_depth_transfer.py',
    'scripts/analyze_scan_trajectory.py', 'spectral_guidance_losses.py', 'chamfer_calibration_report.py',
    PREP + 'operations.py', PREP + 'analyze.py', PREP + 'campaign.py')))
spec = importlib.util.spec_from_file_location('_depth_transfer_operations', REPO / PREP / 'operations.py')
operations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operations)

DATASETS = ('modelnet40_c', 'scanobjectnn_c')
HEADS = depth.TRANSFER_BACKBONES
MAX_WALL_CAP_SECONDS = 28800
DEFAULT_FINAL_ARCHIVE_RESERVE_SECONDS = 60
DEFAULT_PER_JOB_ACCEPTANCE_ARCHIVE_RESERVE_SECONDS = 180


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def verify_sources(plan: dict) -> None:
    require(plan.get('phase_sources') == {name: operations.file_identity(REPO / name) for name in PHASE_SOURCES},
            'Controller/analyzer/worker source changed after plan sealing')


def verify_preflight(plan: dict) -> None:
    """Read actual pinned CPU proof and its physical local/Drive archive copies."""
    path = Path(plan['cpu_preflight_receipt_path'])
    require(operations.file_identity(path) == plan['cpu_preflight_receipt_identity'], 'CPU receipt identity mismatch')
    receipt = read(path)
    report = receipt.get('report', {})
    require(analysis.np.__version__.split('.')[0] == '1', 'Physical CPU acceptance requires pinned NumPy1')
    require(receipt.get('status') == report.get('status') == 'complete'
            and report.get('cuda_initialized') is False and report.get('forward_executed') is False
            and report.get('manifest_sha256') == plan['asset_manifest_sha256']
            and report.get('source_snapshot_sha256') == plan['source_snapshot_sha256'],
            'CPU source/assets/no-forward preflight receipt invalid')
    require(set(report.get('heads', {})) == set(DATASETS), 'CPU receipt requires both datasets')
    for dataset in DATASETS:
        require(set(report['heads'][dataset]) == set(HEADS), 'CPU receipt requires all four frozen heads')
    for job in plan['jobs']:
        for head, binding in job['bindings'].items():
            proof = report['heads'][job['dataset']][head]
            expected = binding['expected_config']
            analysis._subset(expected['checkpoint_load'], proof.get('checkpoint_load'), 'CPU checkpoint')
            require(proof.get('checkpoint_load', {}).get('strict') is True
                    and proof.get('dataset_provenance') == expected.get('dataset_provenance')
                    and isinstance(expected.get('dataset_provenance'), dict), 'CPU dataset/checkpoint binding mismatch')
            if 'classifier_state_sha256' in proof:
                require(proof['classifier_state_sha256'] == expected.get('classifier_state_before_sha256'),
                        'CPU frozen classifier state binding mismatch')
    require(os.path.ismount('/content/drive') and Path(plan['drive_root']).is_dir(),
            'Actual Google Drive mount missing; no local directory fallback')
    archive = receipt['archive']
    expected = {k: archive[k] for k in ('bytes', 'sha256')}
    for key in ('local_path', 'drive_path'):
        require(operations.file_identity(archive[key]) == expected, 'CPU archive local/Drive readback mismatch')
    require(Path(plan['drive_root']).resolve() in Path(archive['drive_path']).resolve().parents,
            'CPU archive not preserved beneath this campaign Drive root')


def validate_plan(plan: dict, root: Path) -> dict:
    cap = plan.get('cumulative_wall_cap_seconds')
    require(type(plan.get('schema_version')) is int and plan['schema_version'] == 1 and type(cap) in (int, float)
            and math.isfinite(cap) and 0 < cap <= MAX_WALL_CAP_SECONDS,
            'Campaign schema/cap must be positive and at most eight hours')
    for key in ('source_snapshot_sha256', 'asset_manifest_sha256'):
        require(analysis._is_sha(plan.get(key)), 'Unpinned ' + key)
    reserves = [plan.get(key) for key in ('final_archive_reserve_seconds', 'per_job_acceptance_archive_reserve_seconds')]
    require(all(type(value) is int and value > 0 for value in reserves) and sum(reserves) < cap,
            'Explicit archive/acceptance reserves must be positive integers with sum below the cap')
    verify_sources(plan)
    jobs = plan.get('jobs', [])
    require(isinstance(jobs, list) and len(jobs) == 8, 'Exactly eight predeclared jobs required')
    expected_scopes = {('smoke', d, 0) for d in DATASETS} | {('research', d, s) for d in DATASETS for s in (0, 1, 2)}
    require({(j['stage'], j['dataset'], j['seed']) for j in jobs} == expected_scopes,
            'Duplicate or changed eight-job scope')
    require([j['stage'] for j in jobs] == ['smoke'] * 2 + ['research'] * 6,
            'Both technical smokes must precede research')
    names = [j['slot_id'] for j in jobs]
    require(len(set(names)) == 8 and all(re.fullmatch('[a-z0-9][a-z0-9_-]{0,79}', n) for n in names),
            'Duplicate or unsafe job slots')
    cohorts, resolved = {}, {}
    for job in jobs:
        require(type(job['seed']) is int and set(job['bindings']) == set(HEADS), 'Job requires four distinct head bindings')
        command = job['command']
        require(isinstance(command, list) and all(isinstance(x, str) for x in command) and len(command) > 3,
                'Worker command must be an explicit argument list')
        require(Path(command[0]).is_absolute() and Path(command[0]).is_file() and os.access(command[0], os.X_OK)
                and command[1] == '-u' and Path(command[2]).resolve() == REPO / 'scripts/eval_chamfer_depth.py'
                and Path(job['cwd']).resolve() == REPO, 'Worker must use the sealed depth CLI and checkout')
        require('--preview' not in command and '--help' not in command and '--transfer-study' in command,
                'Actual transfer worker command required')
        configs = depth.configurations(command[3:])
        resolved[job['slot_id']] = configs
        require(len(configs) == 4 and {c['backbone'] for c in configs} == set(HEADS), 'Worker must share all four heads')
        for config in configs:
            require((Path(config['asset_root']) / 'source_snapshot').resolve() == REPO
                    and not (Path(config['asset_root']) / 'source_snapshot').is_symlink(),
                    'Worker checkout must be the declared asset source_snapshot')
            require(not config['preview'] and config['batch_size'] == 32 and config['dataset'] == job['dataset']
                    and config['seed'] == job['seed'] and config.get('technical_smoke') == (job['stage'] == 'smoke')
                    and config['asset_manifest_sha256'] == plan['asset_manifest_sha256']
                    and Path(config['result_root']).resolve() == root / 'slots' / job['slot_id'] / 'raw'
                    and Path(config['paired_state_root']).resolve() == root / ('paired-smoke' if job['stage'] == 'smoke' else 'paired') / ('seed%d' % job['seed'])
                    and Path(config['drive_root']).resolve() == Path(plan['drive_root']).resolve(),
                    'Worker command differs from frozen job/seed/path bindings')
            binding = job['bindings'][config['backbone']]
            expected = binding['expected_config']
            for key in ('dataset', 'backbone', 'seed', 'selected_original_ids'):
                require(expected.get(key) == config[key], 'Expected config/command mismatch: ' + key)
            require(expected.get('source_snapshot_sha256') == plan['source_snapshot_sha256']
                    and expected.get('asset_manifest_sha256') == plan['asset_manifest_sha256']
                    and expected.get('dataset_provenance', {}).get('n_examples') == config['n_examples']
                    and analysis._is_sha(expected.get('checkpoint_load', {}).get('checkpoint_sha256'))
                    and expected.get('checkpoint_load', {}).get('strict') is True, 'Unbound frozen source/assets/checkpoint')
            required = binding.get('required_source_files', {})
            for source in analysis.CRITICAL_SOURCES:
                identity = operations.file_identity(REPO / source)
                require(required.get('repo/' + source) == identity or required.get('assets/source_snapshot/' + source) == identity,
                        'Required executed source binding differs: ' + source)
        key = (job['dataset'], job['stage'])
        ids = configs[0]['selected_original_ids']
        require(key not in cohorts or cohorts[key] == ids, 'Cross-seed original object IDs differ')
        cohorts[key] = ids
    verify_preflight(plan)
    return resolved


def accept_job(slot: Path, job: dict, root: Path) -> dict:
    manifests = sorted((slot / 'raw').rglob('bundle_manifest.json'))
    require(len(manifests) == 4, 'Expected exactly four sealed head bundles')
    reports, seen, raw_paths = [], set(), {}
    for manifest in manifests:
        head = read(manifest.parent / 'config.json')['backbone']
        require(head in HEADS and head not in seen, 'Missing/duplicate/unregistered frozen head')
        seen.add(head)
        raw = Path(str(manifest.parent) + '.zip')
        require(raw.is_file() and not raw.is_symlink(), 'Worker sealed raw ZIP missing')
        raw_paths[head] = str(raw)
        report = analysis.analyze_bundle(raw, job['bindings'][head], root)
        require(report.get('status') == 'accepted' and report.get('backbone') == head
                and report.get('dataset') == job['dataset'] and report.get('seed') == job['seed']
                and report.get('technical_smoke') == (job['stage'] == 'smoke'), 'CPU endpoint identity unavailable')
        reports.append(report)
    # The independent analyzer compares IDs/labels, physical cache, geometry, RNG and source guards.
    analysis.analyze_reports(reports, repetitions=1)
    signature = {k: reports[0][k] for k in ('object_ids', 'labels', 'corruptions', 'reconstruction_group_id',
                                          'source_snapshot_sha256', 'asset_manifest_sha256', 'batch_guards')}
    compact = dict(status='accepted', slot_id=job['slot_id'], dataset=job['dataset'], seed=job['seed'], stage=job['stage'],
                   shared_proof_sha256=hashlib.sha256(encoded(signature)).hexdigest(),
                   heads={r['backbone']: dict(raw_zip=raw_paths[r['backbone']],
                       checkpoint_load=job['bindings'][r['backbone']]['expected_config']['checkpoint_load'],
                       **{k: r[k] for k in ('bundle_identity', 'sealed_manifest_identity', 'aggregate')}) for r in reports},
                   cache_proofs=reports[0]['cache_proofs'])
    return compact


def run(plan_path: Path, approval_path: Path, root: Path) -> None:
    """Run once; failed/finished roots and immutable Drive namespaces cannot replay."""
    started = time.monotonic()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = os.open(root / 'controller.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(lock, 'w') as stream:
        stream.write('single immutable attempt; PID=%d\n' % os.getpid())
    require(not any((root / name).exists() for name in ('wall_ledger.jsonl', 'slots', 'paired', 'paired-smoke', 'failure.json')),
            'Campaign root is not fresh; no resume/replacement')
    rows, sequence, accepted_smokes, accepted = [], 0, set(), []
    plan, slot, plan_sha, can_archive = None, None, None, False
    status = 'running'
    cap, final_reserve, job_reserve, reserve_active = MAX_WALL_CAP_SECONDS, 0, 0, True

    def event(name, **details):
        elapsed = time.monotonic() - started
        row = dict(event=name, actual_elapsed_seconds=elapsed, **details)
        with (root / 'wall_ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        rows.append(row)
        latest = dict(status=status, plan_sha256=plan_sha, actual_elapsed_seconds=elapsed,
                      charged_seconds=elapsed + (final_reserve if reserve_active else 0),
                      final_archive_reserved_seconds=final_reserve,
                      cumulative_wall_cap_seconds=cap, accepted_slots=accepted, latest_event=row)
        temporary = root / 'registered_latest_ledger.pending.json'
        temporary.write_bytes(encoded(latest))
        temporary.replace(root / 'registered_latest_ledger.json')

    def publish(current=None, proofs=(), failure=False):
        nonlocal sequence
        sequence += 1
        destination = operations.persist_checkpoint(root, plan['drive_root'], plan_sha, sequence,
                                                    current, proofs, preserve_raw=failure)
        event('drive_snapshot_verified', snapshot=destination)

    event('campaign_started')
    try:
        plan_raw = Path(plan_path).read_bytes()
        plan = json.loads(plan_raw)
        plan_sha = hashlib.sha256(plan_raw).hexdigest()
        approval_raw = Path(approval_path).read_bytes()
        approval = json.loads(approval_raw)
        approval_identity = dict(bytes=len(approval_raw), sha256=hashlib.sha256(approval_raw).hexdigest())
        require(approval.get('approved') is True and approval.get('plan_sha256') == plan_sha
                and approval.get('source_snapshot_sha256') == plan.get('source_snapshot_sha256')
                and approval.get('cumulative_wall_cap_seconds') == plan.get('cumulative_wall_cap_seconds'),
                'Approval not bound to actual final plan/source/whole-phase cap')
        resolved = validate_plan(plan, root)
        cap = plan['cumulative_wall_cap_seconds']
        final_reserve = plan['final_archive_reserve_seconds']
        job_reserve = plan['per_job_acceptance_archive_reserve_seconds']
        require(not (Path(plan['drive_root']) / 'campaign' / plan_sha).exists(), 'Immutable Drive campaign already exists')
        for source, name in ((plan_path, 'plan.json'), (approval_path, 'approval.json'),
                             (plan['cpu_preflight_receipt_path'], 'registered_cpu_preflight_receipt.json')):
            operations.copy_verified(source, root / name)
        can_archive = True
        (root / 'slots').mkdir()
        event('preflight_accepted')
        publish()
        for job in plan['jobs']:
            slot = None
            if job['stage'] == 'research':
                require(accepted_smokes == set(DATASETS), 'Research gate requires both accepted four-head technical smokes')
            require(operations.file_identity(plan_path)['sha256'] == plan_sha
                    and operations.file_identity(root / 'plan.json')['sha256'] == plan_sha
                    and operations.file_identity(approval_path) == operations.file_identity(root / 'approval.json') == approval_identity,
                    'Immutable plan/approval changed during campaign')
            require(operations.file_identity(plan['cpu_preflight_receipt_path']) == plan['cpu_preflight_receipt_identity']
                    == operations.file_identity(root / 'registered_cpu_preflight_receipt.json'), 'CPU receipt changed during campaign')
            require(depth.configurations(job['command'][3:]) == resolved[job['slot_id']],
                    'Actual worker IDs/configuration changed after registration')
            verify_sources(plan)
            remaining = cap - (time.monotonic() - started) - final_reserve - job_reserve
            require(remaining > 0, 'Cumulative wall budget exhausted before next worker')
            slot = root / 'slots' / job['slot_id']
            slot.mkdir()
            write_new(slot / 'job.json', job)
            event('job_started', slot_id=job['slot_id'], worker_timeout_seconds=remaining)
            code = operations.run_worker(job['command'], job['cwd'], slot / 'console.log', remaining, job['slot_id'])
            require(code == 0, 'Worker failed with exit code %s' % code)
            report = accept_job(slot, job, root)
            write_new(slot / 'acceptance.json', report)
            event('job_cpu_accepted', slot_id=job['slot_id'])
            publish(slot, report['cache_proofs'])
            accepted.append(job['slot_id'])
            if job['stage'] == 'smoke':
                accepted_smokes.add(job['dataset'])
            event('job_accepted', slot_id=job['slot_id'])
            require(time.monotonic() - started + final_reserve <= cap,
                    'Cumulative wall cap exceeded during CPU acceptance/archive')
        require(len(accepted) == 8, 'All eight endpoints must be accepted')
        status = 'finalizing'
    except BaseException as error:
        status = 'failed'
        failure = dict(status='unavailable', error=repr(error), slot_id=slot.name if slot else None)
        write_new(root / 'failure.json', failure)
        if slot is not None:
            write_new(slot / 'failure.json', failure)
        event('campaign_failed', **failure)
        if can_archive:
            try:
                publish(slot, failure=True)
            except BaseException as archive_error:
                event('failure_archive_failed', error=repr(archive_error))
        raise
    finally:
        reserve_active = False
        event('campaign_finished', outcome=status)
        if can_archive:
            try:
                publish()
                require(time.monotonic() - started <= cap, 'Cumulative wall cap exceeded during final archive')
                if status == 'finalizing':
                    status = 'complete'
                event('final_archive_verified')
                # This small immutable readback follows the last full metadata snapshot.
                operations.copy_verified(root / 'registered_latest_ledger.json',
                    Path(plan['drive_root']) / 'campaign' / plan_sha / 'final_ledger.json')
                require(time.monotonic() - started <= cap, 'Cumulative wall cap exceeded during final ledger readback')
            except BaseException as archive_error:
                status = 'failed'
                if not (root / 'failure.json').exists():
                    write_new(root / 'failure.json', dict(status='unavailable', error=repr(archive_error)))
                event('final_archive_failed', error=repr(archive_error))
                try:
                    for name, target in (('failure.json', 'final_failure.json'),
                                         ('registered_latest_ledger.json', 'failed_final_ledger.json')):
                        operations.copy_verified(root / name, Path(plan['drive_root']) / 'campaign' / plan_sha / target)
                except BaseException as failure_copy_error:
                    event('final_failure_copy_failed', error=repr(failure_copy_error))
                raise
        event('controller_stopped')
    print('DEPTH_TRANSFER_CAMPAIGN_COMPLETE', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('plan', 'approval', 'root'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.plan, args.approval, args.root)


if __name__ == '__main__':
    main()
