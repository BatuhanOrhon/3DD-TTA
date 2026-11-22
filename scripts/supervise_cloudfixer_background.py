"""Host CLI supervisor for an owned, canonically prepared Background A100."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FOLDER = 'cloudfixer_background_20261122'
REMOTE = '/content/' + FOLDER
OPERATOR = ROOT / 'scripts/operate_cloudfixer_rotation_colab.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pull(session, remote, expected, output):
    target = output / Path(remote).name
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f'Existing downloaded file differs: {target}')
        return
    with tempfile.TemporaryDirectory(dir=output) as temp:
        staged = Path(temp)/target.name
        for attempt in range(3):
            result = subprocess.run(['colab', '--auth', 'oauth2', 'download', '-s', session, remote, str(staged)],
                                    capture_output=True, timeout=300)
            if result.returncode == 0 and staged.is_file() and digest(staged) == expected:
                break
            staged.unlink(missing_ok=True)
        else:
            raise RuntimeError('Mirror download failed after three attempts: ' + remote)
        with zipfile.ZipFile(staged) as archive:
            if archive.testzip() is not None:
                raise ValueError('Downloaded ZIP failed CRC')
        staged.rename(target)


def run(args):
    if not re.fullmatch(r'tta-cloudfixer-background-a100-20261122-r[0-9]+', args.session):
        raise ValueError('Expected an explicitly owned Background assignment')
    output = args.output.resolve()
    output.mkdir(exist_ok=True, parents=True)
    log_dir = output/'logs'
    log_dir.mkdir(exist_ok=True)
    host_mirrors = output/'mirrors'
    host_mirrors.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    log = log_dir / (stamp + '_' + args.phase)
    cli = ['colab', '--auth', 'oauth2']
    if args.phase == 'background_deploy':
        if args.package is None:
            raise ValueError('Deployment needs the original sealed source package')
        for name in ('official.tar', 'research.zip', 'receipt.json'):
            subprocess.run([*cli, 'upload', '-s', args.session, str(args.package/name),
                            '/content/' + FOLDER + '_' + name], check=True)
        code = f"""from pathlib import Path
work = Path({REMOTE!r})
work.mkdir(exist_ok=True)
for name in ('official.tar','research.zip','receipt.json'):
    incoming = Path('/content') / ({FOLDER!r} + '_' + name)
    target = work / name
    if target.exists():
        assert target.read_bytes() == incoming.read_bytes(), 'Source differs from prior deployment'
    else:
        incoming.rename(target)
"""
    else:
        code = ''
    if args.phase == 'restore':
        mirrors = sorted(host_mirrors.glob('*.zip'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'host.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                for mirror in mirrors:
                    archive.write(mirror, mirror.name)
            checksum = digest(path)
            subprocess.run([*cli, 'upload', '-s', args.session, str(path), REMOTE+'/host.zip'], check=True)
        code = f"""from pathlib import Path
import hashlib, io, sys, zipfile
work=Path({REMOTE!r})
assert Path('/content/drive').is_mount()
assert (work/'deployment.complete').read_text().strip() == 'verified'
sys.path.insert(0,str(work/'source'))
from cloudfixer_background_checkpoint import import_mirror
path=work/'host.zip'
assert hashlib.sha256(path.read_bytes()).hexdigest() == {checksum!r}
with zipfile.ZipFile(path) as archive:
    for member in archive.infolist():
        import_mirror(Path('/content/drive/MyDrive/thesis') / {FOLDER!r}, io.BytesIO(archive.read(member)))
print('BACKGROUND_HOST_RESTORE_DONE', flush=True)
"""
        marker = 'BACKGROUND_HOST_RESTORE_DONE'
    else:
        source = OPERATOR.read_text().split("if __name__ == '__main__':")[0]
        # Run the exact packaged operator on all phases after deployment.
        if args.phase == 'background_deploy':
            with zipfile.ZipFile(args.package/'research.zip') as archive:
                if archive.read('scripts/operate_cloudfixer_rotation_colab.py') != OPERATOR.read_bytes():
                    raise ValueError('Deployment operator differs from sealed package')
            code += source + '\n'
        else:
            code += f"import sys\nsys.path.insert(0,{(REMOTE+'/source/scripts')!r})\nfrom operate_cloudfixer_rotation_colab import main\n"
        code += 'main(' + repr([args.phase, '--allocation-epoch', str(args.epoch), '--session', args.session]) + ')\n'
        marker = ('CLOUDFIXER_ROTATION_DEPLOYMENT_DONE' if args.phase == 'background_deploy'
                  else 'CLOUDFIXER_ROTATION_OPERATOR_DONE ' + args.phase)
    script = log.with_suffix('.py')
    script.write_text(code)
    futures = []
    passed = False
    with ThreadPoolExecutor(max_workers=1) as pool, log.with_suffix('.log').open('x') as stream:
        process = subprocess.Popen([*cli, 'exec', '-s', args.session, '-f', str(script), '--timeout', '18000'],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            value = re.sub(r'\x1b\[[0-9;]*[mK]', '', line).strip()
            if value.startswith('BACKGROUND_MIRROR '):
                _, remote, expected = value.split()
                futures.append(pool.submit(pull, args.session, remote, expected, host_mirrors))
            if value.startswith('CLOUDFIXER_ROTATION_ARCHIVE '):
                receipt = json.loads(value.split(' ', 1)[1])
                futures.append(pool.submit(pull, args.session, receipt['archive'], receipt['sha256'], output))
            passed |= value == marker
            if value.startswith(('BACKGROUND_', 'CLOUDFIXER_', 'ROTATION_BLOCK', 'Traceback')):
                print(value, flush=True)
        status = process.wait()
        for future in futures:
            future.result()
    if status or not passed:
        raise RuntimeError(f'Phase did not complete. Inspect {log}.log and remote state before resubmission.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['background_deploy', 'restore', 'background_smoke',
                                          'background_pm', 'background_dev', 'background'])
    parser.add_argument('--session', required=True)
    parser.add_argument('--epoch', type=float, required=True)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'result/modelnet40_c'/FOLDER)
    run(parser.parse_args())
