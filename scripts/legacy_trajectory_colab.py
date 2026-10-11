"""CPU orchestration for user-run Colab smoke/pilot; never launches on preview.

All evaluation subprocesses explicitly use conda's 3dd_tta_env. No Git writes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research_artifacts import CORRUPTIONS

CONDA = ['conda', 'run', '--no-capture-output', '-n', '3dd_tta_env', 'python']
PILOT_SEEDS = (0,)
BASE_SHA = '534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968'
PROVENANCE_PATH = 'cfgs/legacy_trajectory_provenance.json'
SMOKE_ARMS = tuple('smoke_%s_%s.zip' % (corruption, mode)
                   for corruption in ('gaussian', 'background')
                   for mode in ('off', 'audit', 'observe'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def smoke_arm_key(corruption, mode):
    """Build a JSON-safe validation key for one smoke arm."""
    return '%s_%s' % (corruption, mode)


def verify_sources():
    """Verify inference sources against one resolved Git commit, without writes.

    Ignore unrelated worktree/build output. Normalize checkout line endings so
    Windows autocrlf does not masquerade as a source modification.
    """
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=REPO)

    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    files = {}

    def checked_source(name):
        path = REPO / name
        if path.is_symlink() or REPO.resolve() not in path.resolve().parents:
            raise ValueError('Invalid source path: ' + name)
        actual = path.read_bytes().replace(b'\r\n', b'\n')
        expected = git('show', commit + ':' + name).replace(b'\r\n', b'\n')
        if actual != expected:
            raise ValueError('Runtime source differs from Git commit: ' + name)
        files[name] = dict(sha256=sha(actual), bytes=len(actual))
        return actual

    spec = json.loads(checked_source(PROVENANCE_PATH))
    if spec['schema_version'] != 1 or spec['baseline_archive_sha256'] != BASE_SHA:
        raise ValueError('Invalid baseline provenance')
    if branch != spec['branch']:
        raise ValueError('Expected branch ' + spec['branch'])
    for name in spec['source_paths']:
        checked_source(name)
    return dict(schema_version=2, resolved_commit=commit, branch=branch,
                baseline_archive_sha256=BASE_SHA, files=files,
                baseline_assets=spec['baseline_assets'], source_delivery='git')


def initialize(session):
    manifest = verify_sources()
    session.mkdir(parents=True, exist_ok=False)
    ids = list(range(2468))
    random.Random(20261011).shuffle(ids)
    ids = sorted(ids[:280])
    for stage, indices in [('smoke', ids[:140]), ('pilot', ids)]:
        write_json(session / (stage + '_indices.json'), dict(schema_version=1, dataset='modelnet40_c',
                   selection_seed=20261011, selection='label-free shuffled subset, sorted for stable order',
                   indices={c: indices for c in CORRUPTIONS}))
    write_json(session / 'session.json', dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
               source_manifest=manifest, baseline_accuracy_reference=64.570502,
               baseline_archive_sha256=BASE_SHA, pilot_seeds=list(PILOT_SEEDS), deferred_seeds=[1, 2], status='prepared',
               stage='development', checkpoints={'10': [3, 5, 8], '30': [8, 15, 23]}))


def commands(session, stage):
    common = CONDA + ['eval_gsd_tta_v2.py', '--batch_size', '70', '--M_max', '800', '--beta', '2',
                     '--weight_spectral', '1.17', '--weight_chamfer', '1', '--gamma', '.01', '--eta', '.01',
                     '--lambdaa', '.95', '--denoising_step_normal', '10', '--denoising_step_bg', '30',
                     '--isolate_policy', 'legacy', '--trajectory_steps_normal', '3,5,8', '--trajectory_steps_bg', '8,15,23',
                     '--indices_manifest', str(session / (stage + '_indices.json'))]
    if stage == 'smoke':
        for corruption in ('gaussian', 'background'):
            for mode in ('off', 'audit', 'observe'):
                yield common + ['--seed', '0', '--corruption', corruption, '--trajectory_mode', mode,
                                '--trajectory_geometry_samples', '1', '--output_dir', str(session / ('smoke_%s_%s' % (corruption, mode)))]
    else:
        for seed in PILOT_SEEDS:
            yield common + ['--seed', str(seed), '--trajectory_mode', 'observe', '--output_dir', str(session / ('pilot_seed%d' % seed))]


def verify_assets(manifest, session):
    # Match data/checkpoint/config bytes of accepted Legacy, without importing models.
    for old_path, identity in manifest['baseline_assets'].items():
        path = Path(old_path)
        if not path.is_absolute():
            path = REPO / path
        if not path.is_file():
            raise FileNotFoundError(str(path))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != identity['sha256']:
            raise ValueError('Baseline asset mismatch: ' + str(path))


def backup_zip(path, directory):
    """Copy sealed evidence without replacing any differing Drive artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / path.name
    digest = sha(path.read_bytes())
    if destination.exists():
        if sha(destination.read_bytes()) != digest:
            raise FileExistsError('Different Drive artifact exists: ' + str(destination))
    else:
        with path.open('rb') as source, destination.open('xb') as target:
            shutil.copyfileobj(source, target)
    if sha(destination.read_bytes()) != digest:
        raise ValueError('Drive copy checksum mismatch')


def execute(session, stage, drive_dir=None):
    manifest = verify_sources()
    if json.loads((session / 'session.json').read_text())['source_manifest'] != manifest:
        raise ValueError('Git source identity changed since session initialization')
    if stage == 'pilot':
        gate = json.loads((session / 'smoke_validation.json').read_text())
        if not gate.get('passed'):
            raise ValueError('Smoke invariants must pass first')
    verify_assets(manifest, session)
    jobs = list(commands(session, stage))
    for cmd in jobs:
        destination = Path(cmd[-1])
        if destination.exists() or destination.with_suffix('.zip').exists():
            raise FileExistsError('Preserve existing attempt; no automatic rerun: ' + str(destination))
    write_json(session / (stage + '_launch.json'), dict(commands=jobs, status='launching'))
    for cmd in jobs:
        print('RUN:', ' '.join(cmd), flush=True)
        try:
            subprocess.run(cmd, cwd=REPO, check=True)
        finally:
            sealed = Path(cmd[-1]).with_suffix('.zip')
            if drive_dir is not None and sealed.exists():
                backup_zip(sealed, drive_dir / session.name)
    if stage == 'smoke':
        from scripts.analyze_legacy_trajectory import compare, analyze, validate_smoke_bundle
        results, arms = [], {}
        for corruption in ('gaussian', 'background'):
            prefix = session / ('smoke_' + corruption)
            for mode in ('off', 'audit', 'observe'):
                arm = Path(str(prefix) + '_' + mode + '.zip')
                arms[smoke_arm_key(corruption, mode)] = validate_smoke_bundle(
                    arm, mode, expected_count=140)
            analyze([Path(str(prefix)+'_observe.zip')], session / ('smoke_analysis_' + corruption))
            results.append(compare(str(prefix)+'_off.zip', str(prefix)+'_audit.zip'))
            results.append(compare(str(prefix)+'_audit.zip', str(prefix)+'_observe.zip'))
        write_json(session / 'smoke_validation.json', dict(passed=True, arms=arms,
                   cross_process_variability=results,
                   limitation='Smoke checks artifacts and within-run RNG/buffer invariants. Cross-process logits may differ; variability is descriptive, not a pass/fail criterion. No accuracy or latency evidence. Stop for review.'))
    else:
        from scripts.analyze_legacy_trajectory import analyze
        analyze([session / ('pilot_seed%d.zip' % seed) for seed in PILOT_SEEDS], session / 'pilot_analysis')


def recover_smoke(session, backup_dir):
    """Recover completed Drive smoke arms into a fresh, current-source session.

    This is CPU-only. It validates existing ZIPs and never reruns inference or
    overwrites a session, arm archive, analysis, or smoke gate.
    """
    from scripts.analyze_legacy_trajectory import Bundle, analyze, compare, validate_smoke_bundle

    session, backup_dir = Path(session).resolve(), Path(backup_dir).resolve()
    manifest = verify_sources()
    session_record = json.loads((session / 'session.json').read_text(encoding='utf-8'))
    if session_record.get('source_manifest') != manifest:
        raise ValueError('Recovery requires a freshly initialized current-source session')
    if not (session / 'smoke_indices.json').is_file():
        raise FileNotFoundError('Fresh session is missing smoke_indices.json')
    if (session / 'smoke_validation.json').exists():
        raise FileExistsError('Preserve existing smoke gate; refusing recovery')

    sources = {name: backup_dir / name for name in SMOKE_ARMS}
    missing = [str(path) for path in sources.values() if not path.is_file() or path.is_symlink()]
    if missing:
        raise ValueError('Missing smoke archives: ' + ', '.join(missing))
    destinations = {name: session / name for name in SMOKE_ARMS}
    collisions = [str(path) for path in destinations.values() if path.exists()]
    if collisions:
        raise FileExistsError('Preserve existing smoke output: ' + ', '.join(collisions))
    for corruption in ('gaussian', 'background'):
        output = session / ('smoke_analysis_' + corruption)
        if output.exists():
            raise FileExistsError('Preserve existing smoke analysis: ' + str(output))

    expected_indices = json.loads((session / 'smoke_indices.json').read_text(encoding='utf-8'))['indices']
    source_commit = None
    source_reports = {}
    for name, path in sources.items():
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError('Smoke ZIP CRC failure: %s (%s)' % (path, bad_member))
        mode = name.rsplit('_', 1)[1][:-4]
        report = validate_smoke_bundle(path, mode, expected_count=140)
        bundle = Bundle(path)
        try:
            config = bundle.config
            corruption = config['corruptions'][0]
            if corruption not in expected_indices or config['selected_indices'][corruption] != expected_indices[corruption]:
                raise ValueError('Smoke IDs do not match fresh session manifest: ' + name)
            if config.get('seed') != 0:
                raise ValueError('Only the completed seed-0 smoke can be recovered: ' + name)
            if source_commit is None:
                source_commit = config['resolved_commit']
            elif source_commit != config['resolved_commit']:
                raise ValueError('Smoke arms come from different source commits')
        finally:
            bundle.close()
        source_reports[name] = report

    # Permit only post-smoke report/orchestration changes; any model/evaluation
    # source change requires a fresh smoke instead of this recovery path.
    subprocess.check_output(['git', 'merge-base', '--is-ancestor', source_commit,
                             manifest['resolved_commit']], cwd=REPO)
    changed = set(subprocess.check_output(
        ['git', 'diff', '--name-only', source_commit + '..' + manifest['resolved_commit']],
        cwd=REPO).decode().splitlines())
    allowed = {'scripts/analyze_legacy_trajectory.py', 'scripts/legacy_trajectory_colab.py',
               'tests/test_legacy_trajectory_observer.py',
               'knowledge/gsd_trajectory_implementation_20261011.md'}
    if changed - allowed:
        raise ValueError('Model/evaluation sources changed since smoke: ' + ', '.join(sorted(changed - allowed)))

    for name, source in sources.items():
        with source.open('rb') as src, destinations[name].open('xb') as dst:
            shutil.copyfileobj(src, dst)

    results, arms = [], {}
    for corruption in ('gaussian', 'background'):
        prefix = session / ('smoke_' + corruption)
        for mode in ('off', 'audit', 'observe'):
            name = 'smoke_%s_%s.zip' % (corruption, mode)
            arms[smoke_arm_key(corruption, mode)] = validate_smoke_bundle(
                destinations[name], mode, expected_count=140)
        analyze([destinations['smoke_%s_observe.zip' % corruption]],
                session / ('smoke_analysis_' + corruption))
        results.append(compare(destinations['smoke_%s_off.zip' % corruption],
                               destinations['smoke_%s_audit.zip' % corruption]))
        results.append(compare(destinations['smoke_%s_audit.zip' % corruption],
                               destinations['smoke_%s_observe.zip' % corruption]))
    gate = dict(passed=True, recovered=True, recovered_from=str(backup_dir),
                recovery_source_commit=source_commit, current_source_commit=manifest['resolved_commit'],
                arms=arms, cross_process_variability=results,
                limitation='Recovered completed seed-0 smoke; no inference was rerun. Cross-process variability is descriptive, not a pass/fail criterion. No accuracy or latency evidence.')
    write_json(session / 'smoke_validation.json', gate)
    write_json(session / 'smoke_recovery.json', dict(source_commit=source_commit,
               current_source_commit=manifest['resolved_commit'],
               archives={name: sha(path.read_bytes()) for name, path in sources.items()},
               copied_archives={name: sha(path.read_bytes()) for name, path in destinations.items()}))
    return gate


def pack_results(session, output):
    if session.resolve() in output.resolve().parents:
        raise ValueError('Handoff ZIP must be outside the session directory')
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        # Include logs/partial failed directories as well as sealed arm ZIPs.
        for path in sorted(session.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(session).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Handoff CRC failure')
    print(json.dumps(dict(zip=str(output), sha256=sha(output.read_bytes())), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'init', 'preview', 'smoke', 'recover-smoke', 'pilot', 'pack'))
    parser.add_argument('--session', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke-dir', type=Path, help='Drive directory containing all six completed smoke ZIPs')
    parser.add_argument('--drive-dir', type=Path, help='Optional mounted Drive directory; back up each sealed arm, even failed.')
    args = parser.parse_args()
    if args.action == 'verify':
        print(json.dumps(verify_sources(), indent=2))
        return
    if not args.session: parser.error('--session required')
    session = args.session.resolve()
    if args.action == 'init': initialize(session)
    elif args.action == 'preview':
        print(json.dumps({stage: list(commands(session, stage)) for stage in ('smoke', 'pilot')}, indent=2))
        print('Smoke: 6 seed0 arms x 140 = 840 classifications; pilot: 15 x 280 x 1 seed0 = 4200. Seeds1/2 deferred. No inference launched.')
    elif args.action == 'pack':
        if not args.output: parser.error('--output required')
        pack_results(session, args.output.resolve())
    elif args.action == 'recover-smoke':
        if not args.smoke_dir: parser.error('--smoke-dir required')
        report = recover_smoke(session, args.smoke_dir)
        print(json.dumps(dict(passed=report['passed'], recovered=report['recovered'],
                              source_commit=report['recovery_source_commit'],
                              current_commit=report['current_source_commit']), indent=2))
    else:
        status = session / (args.action + '_status.json')
        try:
            execute(session, args.action, args.drive_dir)
        except BaseException as error:
            if session.is_dir() and not status.exists():
                write_json(status, dict(execution_status='failed', error=type(error).__name__ + ': ' + str(error)))
            raise
        else:
            write_json(status, dict(execution_status='complete', next_action='stop_and_review_complete_zip'))


if __name__ == '__main__':
    main()
