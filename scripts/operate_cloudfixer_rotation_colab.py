"""Deploy or execute one CloudFixer rotation phase on the already assigned Colab VM."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from time import time
import zipfile


WORK = Path('/content/cloudfixer_rotation_20261121')
SOURCE = WORK/'source'
OFFICIAL = Path('/content/CloudFixer')
DRIVE = Path('/content/drive/MyDrive/thesis/cloudfixer_rotation_20261121')
CONDA = ['conda', 'run', '--no-capture-output', '-n', 'cf_point2vec', 'python', '-u']


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_copy(source, target):
    if target.exists():
        if target.stat().st_size != source.stat().st_size or digest(source) != digest(target):
            raise ValueError(f'Existing archive differs: {target}')
        return
    pending = target.with_name('.' + target.name + '.pending')
    with source.open('rb') as original, pending.open('xb') as file:
        shutil.copyfileobj(original, file, length=4 << 20)
        file.flush()
        os.fsync(file.fileno())
    if digest(source) != digest(pending):
        raise ValueError(f'Drive copy failed verification: {target}')
    pending.rename(target)
    if digest(source) != digest(target):
        raise ValueError(f'Drive archive failed readback: {target}')


def stream(command, path):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as child, path.open('a') as file:
        for line in child.stdout:
            file.write(line)
            file.flush()
            print(line, end='', flush=True)
        return child.wait()


def assert_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe archive member {name}')


def deploy(*, reuse_verified_environment=False, session=None):
    verified = Path('/content/cli_setup_verification.json')
    if json.loads(verified.read_text())['status'] != 'passed':
        raise RuntimeError('Canonical environment verification must pass before deployment')
    receipt = json.loads((WORK/'receipt.json').read_text())
    for name, identity in receipt.items():
        path = WORK/name
        if path.stat().st_size != identity['bytes'] or digest(path) != identity['sha256']:
            raise ValueError(f'Invalid source upload: {name}')
    if not OFFICIAL.exists():
        with tarfile.open(WORK/'official.tar') as archive:
            for member in archive.getmembers():
                assert_path(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unsafe author member {member.name}')
            archive.extractall(OFFICIAL)
    if not SOURCE.exists():
        with zipfile.ZipFile(WORK/'research.zip') as archive:
            for member in archive.infolist():
                assert_path(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f'Unsafe research member {member.filename}')
            archive.extractall(SOURCE)
    spec = json.loads((SOURCE/'manifest.json').read_text())
    for folder, files in [(OFFICIAL, spec['official_files']), (SOURCE, spec['overlay_files'])]:
        for name, expected in files.items():
            if digest(folder/name) != expected:
                raise ValueError(f'Source hash mismatch in {folder/name}')
    assets = json.loads((SOURCE/'cfgs/cloudfixer_transfer_assets.json').read_text())
    checkpoints = OFFICIAL/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    weights = {
        spec['classifier']['classifier_filename']:
            (spec['classifier']['classifier_drive_id'], spec['classifier_sha256']),
        'diffusion_model_transformer_modelnet40.npy': (assets['diffusion_drive_id'], spec['diffusion_sha256']),
    }
    log = WORK/'deployment.log'
    for name, (drive_id, expected) in weights.items():
        path = checkpoints/name
        if not path.exists():
            pending = path.with_suffix('.download')
            if stream(['gdown', drive_id, '-O', str(pending)], log) != 0 or digest(pending) != expected:
                raise RuntimeError(f'Checkpoint download/verification failed: {name}')
            pending.rename(path)
        if digest(path) != expected:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    command = [sys.executable, '-u', str(SOURCE/'scripts/setup_cloudfixer_point2vec_colab.py'),
               '--official-root', str(OFFICIAL), '--manifest', str(SOURCE/'manifest.json'),
               '--classifier-checkpoint', str(checkpoints/spec['classifier']['classifier_filename'])]
    if not reuse_verified_environment and stream(command, log) != 0:
        raise RuntimeError('Supplemental native/head verification failed')
    if not Path('/content/.cf_point2vec_setup/verified.complete').is_file():
        raise RuntimeError('Supplemental native/head verification failed')
    proof = DRIVE/'setup'
    if 'background' in WORK.name:
        proof = proof/(session or digest(verified)[:16])
    proof.mkdir(parents=True, exist_ok=True)
    for path in (verified, WORK/'official.tar', WORK/'research.zip', WORK/'receipt.json',
                 Path('/content/.cf_point2vec_setup/verified.complete')):
        write_copy(path, proof/path.name)
    (WORK/'deployment.complete').write_text('verified\n')
    print('CLOUDFIXER_ROTATION_DEPLOYMENT_DONE', flush=True)


def execute(phase, session, allocation_epoch, dev_run):
    if (WORK/'deployment.complete').read_text() != 'verified\n':
        raise RuntimeError('Supplemental setup has not completed')
    sys.path.insert(0, str(SOURCE))
    from cloudfixer_store import MountStore, canonical, record_setup

    store = MountStore(DRIVE)
    if not phase.startswith(('spectral', 'scd')):
        misc = WORK/'pilot_setup.json'
        if not misc.exists():
            start = datetime.fromtimestamp(allocation_epoch, timezone.utc).isoformat()
            misc.write_bytes(canonical(dict(attempts=[dict(session=session, started_utc=start,
                                                           seconds=time()-allocation_epoch)])))
        is_bg = phase.startswith('background')
        record_setup(MountStore(DRIVE/('background_budget' if is_bg else 'pilot_budget')),
                     json.loads(misc.read_text()), cap=(json.loads((SOURCE/'cfgs/cloudfixer_background_pilot.json').read_text())['budget_seconds']
                          if is_bg else 4*3600))
    now = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    output = WORK/f'{now}_{phase}'
    derived = WORK/f'{output.name}_derived'
    log = output.with_suffix('.log')
    command = [*CONDA, str(SOURCE/'scripts/run_cloudfixer_rotation.py'), '--phase', phase,
               '--manifest', str(SOURCE/'manifest.json'), '--output', str(output), '--drive-folder', str(DRIVE)]
    result = stream(command, log)
    if result == 0:
        command = [*CONDA, str(SOURCE/'scripts/analyze_cloudfixer_rotation.py'), str(output),
                   '--derived', str(derived)]
        if phase == 'validation':
            if not dev_run:
                raise ValueError('Validation requires the prior development run directory name')
            command.extend(('--development', str(WORK/Path(dev_run).name)))
        result = stream(command, log)
    output.mkdir(exist_ok=True)
    shutil.copyfile(log, output/'stdout.log')
    (output/'exit_status.json').write_bytes(canonical(dict(returncode=result)))
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as zipped:
        for folder in (output, derived):
            for path in sorted(folder.rglob('*')):
                if path.is_symlink():
                    raise ValueError(f'Unexpected symlink {path}')
                if path.is_file():
                    zipped.write(path, path.relative_to(WORK))
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
    archive_folder = DRIVE/'archives'
    archive_folder.mkdir(parents=True, exist_ok=True)
    write_copy(archive, archive_folder/archive.name)
    print('CLOUDFIXER_ROTATION_ARCHIVE', json.dumps(dict(run=output.name, archive=str(archive),
          sha256=digest(archive), bytes=archive.stat().st_size, returncode=result)), flush=True)
    if result != 0:
        raise RuntimeError(f'Failed {phase} attempt archived as {archive.name}')
    if phase == 'validation':
        store.put('validation_decision.json', (derived/'decision.json').read_bytes())
    print('CLOUDFIXER_ROTATION_OPERATOR_DONE', phase, flush=True)


def main(argv=None):
    global WORK, SOURCE, DRIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('deploy', 'smoke', 'development', 'validation', 'full',
                                          'spectral_deploy', 'spectral_smoke', 'spectral', 'scd_deploy', 'scd_smoke', 'scd',
                                          'background_deploy', 'background_smoke', 'background_pm', 'background_dev', 'background'))
    parser.add_argument('--session')
    parser.add_argument('--allocation-epoch', type=float)
    parser.add_argument('--development')
    args = parser.parse_args(argv)
    if not Path('/content/drive').is_mount():
        raise RuntimeError('Drive mount required')
    if args.phase.startswith('background'):
        WORK = Path('/content/cloudfixer_background_20261122')
        SOURCE = WORK/'source'
        DRIVE = Path('/content/drive/MyDrive/thesis')/WORK.name
    elif args.phase.startswith(('scd', 'spectral')):
        folder = 'cloudfixer_' + args.phase.split('_')[0] + '_rotation_20261121'
        WORK = Path('/content')/folder
        SOURCE = WORK/'source'
        DRIVE = Path('/content/drive/MyDrive/thesis')/folder
    if args.phase in ('deploy', 'spectral_deploy', 'scd_deploy', 'background_deploy'):
        deploy(reuse_verified_environment=args.phase not in ('deploy', 'background_deploy'), session=args.session)
    else:
        if not args.session or args.allocation_epoch is None:
            parser.error('Execution requires the session identity and allocation epoch')
        execute(args.phase, args.session, args.allocation_epoch, args.development)


if __name__ == '__main__':
    main()
