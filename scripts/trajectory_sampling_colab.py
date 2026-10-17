"""CPU preview/delivery and user-operated Colab sampling smoke/pilot launcher.

Pack reads Git without staging/committing. A sealed source snapshot is extracted
into a fresh Colab directory; assets are linked from the prepared Legacy repo.
GPU subprocesses are launched only by the explicit 'run' command in Colab.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
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

BASE = 'a351e66ec58f08f3a2104e46cab2cb8b20cdf961'
BASE_ZIP_SHA = '534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968'
MANIFEST = 'sampling_delivery.json'
CONDA = ['conda', 'run', '--no-capture-output', '-n', '3dd_tta_env', 'python']
MACRO14 = [c for c in CORRUPTIONS if c != 'background']
OVERRIDES = ('eval_gsd_tta_v2.py', 'legacy_gsd_artifacts.py', 'legacy_trajectory.py',
             'trajectory_sampling.py', 'scripts/analyze_trajectory_sampling.py',
             'scripts/trajectory_sampling_colab.py', 'tests/test_trajectory_sampling.py',
             'tests/test_legacy_gsd_restore.py', 'tests/test_legacy_trajectory_observer.py',
             'knowledge/gsd_sampling_consistency_20261011.md')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pack(output: Path) -> dict:
    """Deliver pinned tracked code + ONLY the declared working-tree overrides."""
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
    if commit != BASE:
        raise ValueError('Reassess delivery for a different base commit')
    spec = json.loads((REPO / 'cfgs/legacy_trajectory_provenance.json').read_text())
    base_bytes = subprocess.check_output(['git', 'archive', '--format=zip', BASE], cwd=REPO)
    contents = {}
    suffixes = {'.py', '.json', '.yaml', '.yml', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.c', '.txt', '.sh'}
    with zipfile.ZipFile(io.BytesIO(base_bytes)) as archive:
        for name in archive.namelist():
            if (Path(name).suffix in suffixes and not name.startswith(('result/', 'docs/', 'knowledge/', '.'))
                    and '/build/' not in name and '/tmp/' not in name):
                contents[name] = archive.read(name).replace(b'\r\n', b'\n')
        for name in spec['source_paths'] + ['cfgs/legacy_trajectory_provenance.json']:
            if name not in OVERRIDES and (REPO / name).read_bytes().replace(b'\r\n', b'\n') != archive.read(name).replace(b'\r\n', b'\n'):
                raise ValueError('Unapproved runtime source modification: ' + name)
    for name in OVERRIDES:
        path = REPO / name
        if path.is_symlink():
            raise ValueError('Source symlink refused: ' + name)
        contents[name] = path.read_bytes().replace(b'\r\n', b'\n')
    manifest = dict(schema_version=1, base_commit=BASE,
        source_delivery='pinned Git snapshot plus explicitly hashed working-tree overrides',
        overrides=list(OVERRIDES), baseline_archive_sha256=BASE_ZIP_SHA,
        files={name: dict(bytes=len(data), sha256=sha(data)) for name, data in sorted(contents.items())})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            archive.writestr(name, data)
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2))
    return dict(path=str(output), sha256=file_sha(output), files=len(contents), base_commit=BASE)


def verify_delivery(root: Path = None) -> dict:
    root = REPO if root is None else Path(root)
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 1 or manifest.get('base_commit') != BASE
            or manifest.get('baseline_archive_sha256') != BASE_ZIP_SHA
            or not set(OVERRIDES).issubset(manifest.get('files', {}))):
        raise ValueError('Invalid sampling source delivery manifest')
    for name, identity in manifest['files'].items():
        path = root / name
        if ('\\' in name or ':' in name or Path(name).is_absolute() or '..' in Path(name).parts
                or not path.is_file() or path.is_symlink() or root.resolve() not in path.resolve().parents):
            raise ValueError('Unsafe/missing delivery file: ' + name)
        if path.stat().st_size != identity['bytes'] or file_sha(path) != identity['sha256']:
            raise ValueError('Source delivery hash mismatch: ' + name)
    return dict(base_commit=BASE, manifest_sha256=file_sha(manifest_path),
                kind=manifest['source_delivery'], files=manifest['files'])


def initialize(session: Path, prepared_repo: Path) -> None:
    delivery = verify_delivery()
    if REPO.resolve() == Path(prepared_repo).resolve():
        raise ValueError('Extract the source snapshot to a NEW directory first')
    session = Path(session).resolve()
    if session.exists():
        raise FileExistsError('Preserve existing sampling session')
    for name in ('data', 'lion_ckpts', 'pointnet_ckpts'):
        source, target = Path(prepared_repo).resolve() / name, REPO / name
        if not source.is_dir():
            raise FileNotFoundError('Prepared assets missing: ' + str(source))
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise FileExistsError('Different asset path exists: ' + str(target))
        else:
            target.symlink_to(source, target_is_directory=True)
    verify_assets()
    session.mkdir(parents=True, exist_ok=False)
    ids = list(range(2468))
    random.Random(20261011).shuffle(ids)
    pilot_ids = sorted(ids[:280])
    for stage, names, selected in [('smoke', ['gaussian'], pilot_ids[:140]),
                                   ('pilot', MACRO14, pilot_ids)]:
        write_json(session / (stage + '_indices.json'), dict(schema_version=1,
            dataset='modelnet40_c', selection_seed=20261011,
            indices={name: selected for name in names}))
    write_json(session / 'session.json', dict(schema_version=1, delivery=delivery,
        baseline_archive_sha256=BASE_ZIP_SHA, seed=0, stage='development',
        index_manifest_sha256={stage: file_sha(session / (stage + '_indices.json'))
                               for stage in ('smoke', 'pilot')},
        created_utc=datetime.now(timezone.utc).isoformat(), status='prepared'))
    print('Sampling preflight passed. Session:', session)


def verify_assets() -> None:
    spec = json.loads((REPO / 'cfgs/legacy_trajectory_provenance.json').read_text())
    for name, identity in spec['baseline_assets'].items():
        if name.endswith('data_background_5.npy'):
            continue
        path = REPO / name
        if not path.is_file() or file_sha(path) != identity['sha256']:
            raise ValueError('Baseline asset hash mismatch: ' + name)


def command(session: Path, stage: str) -> list:
    if stage not in ('smoke', 'pilot'):
        raise ValueError('Stage must be smoke or pilot')
    cmd = CONDA + ['eval_gsd_tta_v2.py', '--batch_size', '70', '--M_max', '800',
        '--beta', '2', '--weight_spectral', '1.17', '--weight_chamfer', '1',
        '--gamma', '.01', '--eta', '.01', '--lambdaa', '.95', '--seed', '0',
        '--isolate_policy', 'legacy', '--denoising_step_normal', '10', '--denoising_step_bg', '30',
        '--trajectory_mode', 'observe', '--trajectory_sampling', 'paired',
        '--corruption_scope', 'macro14', '--trajectory_steps_normal', '3,5,8',
        '--indices_manifest', str(Path(session) / (stage + '_indices.json'))]
    if stage == 'smoke':
        cmd += ['--corruption', 'gaussian']
    return cmd + ['--output_dir', str(Path(session) / (stage + '_seed0'))]


def backup(path: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.name
    if target.exists():
        if file_sha(target) != file_sha(path):
            raise FileExistsError('Different backup exists: ' + str(target))
    else:
        with path.open('rb') as source, target.open('xb') as destination:
            shutil.copyfileobj(source, destination)
    if file_sha(target) != file_sha(path):
        raise ValueError('Drive backup SHA mismatch')
    return target


def package(session: Path, drive_dir: Path) -> dict:
    """Preserve partial files after errors without labeling them complete."""
    session = Path(session).resolve()
    if not (session / 'session.json').is_file():
        raise ValueError('Not an initialized sampling session')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = session.parent / (session.name + '_handoff_' + stamp + '.zip')
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(session.rglob('*')):
            if item.is_symlink():
                raise ValueError('Symlink in sampling artifacts')
            if item.is_file():
                archive.write(item, item.relative_to(session).as_posix())
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Handoff CRC failure')
    destination = backup(path, Path(drive_dir))
    receipt = dict(path=str(path), sha256=file_sha(path), drive_path=str(destination),
                   note='May include partial runs; analysis gates determine completion')
    write_json(session / ('handoff_' + stamp + '.json'), receipt)
    return receipt


def run(session: Path, stage: str, drive_dir: Path) -> None:
    session = Path(session).resolve()
    state = json.loads((session / 'session.json').read_text())
    delivery = verify_delivery()
    if delivery != state['delivery']:
        raise ValueError('Source delivery changed since initialization')
    if file_sha(session / (stage + '_indices.json')) != state['index_manifest_sha256'][stage]:
        raise ValueError('Predeclared original IDs changed since initialization')
    verify_assets()
    if stage == 'pilot':
        gate = json.loads((session / 'smoke_gate.json').read_text())
        smoke_zip = session / 'smoke_seed0.zip'
        if (not gate['passed'] or gate['delivery'] != delivery['manifest_sha256']
                or file_sha(smoke_zip) != gate['archive_sha256']):
            raise ValueError('Valid smoke gate required before pilot')
    cmd = command(session, stage)
    out = Path(cmd[-1])
    if out.exists() or out.with_suffix('.zip').exists():
        raise FileExistsError('Preserve previous attempt; automatic rerun refused')
    write_json(session / (stage + '_launch.json'), dict(command=cmd, delivery=delivery['manifest_sha256']))
    try:
        with (session / (stage + '_launcher.log')).open('x', encoding='utf-8') as log:
            process = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, cmd)
        verify_delivery()
        from scripts.analyze_trajectory_sampling import analyze
        report = analyze([out.with_suffix('.zip')], session / (stage + '_analysis'))
        expected = 140 if stage == 'smoke' else 14 * 280
        if (report['summary'][0]['n'] != expected
                or report['corruptions'] != (['gaussian'] if stage == 'smoke' else MACRO14)
                or report['source_delivery'] != delivery):
            raise ValueError('Stage coverage mismatch')
        if stage == 'smoke':
            write_json(session / 'smoke_gate.json', dict(passed=True,
                delivery=delivery['manifest_sha256'], archive_sha256=file_sha(out.with_suffix('.zip')),
                limitation='CPU/artifact/native-start gate, not accuracy or cross-process CUDA parity. Review ZIP before pilot.'))
        print('Completed', stage, '; inspect the paired report before the next stage.')
    finally:
        print(json.dumps(package(session, drive_dir), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('pack'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('init'); p.add_argument('--session', type=Path, required=True)
    p.add_argument('--prepared-repo', type=Path, required=True)
    for name in ('preview', 'run'):
        p = sub.add_parser(name); p.add_argument('--session', type=Path, required=True)
        p.add_argument('--stage', choices=('smoke', 'pilot'), required=True)
        if name == 'run':
            p.add_argument('--drive-dir', type=Path, required=True)
    p = sub.add_parser('package'); p.add_argument('--session', type=Path, required=True)
    p.add_argument('--drive-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'pack':
        print(json.dumps(pack(args.output), indent=2))
    elif args.action == 'init':
        initialize(args.session, args.prepared_repo)
    elif args.action == 'preview':
        print(json.dumps(dict(command=command(args.session, args.stage), launches_gpu=False), indent=2))
    elif args.action == 'package':
        print(json.dumps(package(args.session, args.drive_dir), indent=2))
    else:
        run(args.session, args.stage, args.drive_dir)


if __name__ == '__main__':
    main()
