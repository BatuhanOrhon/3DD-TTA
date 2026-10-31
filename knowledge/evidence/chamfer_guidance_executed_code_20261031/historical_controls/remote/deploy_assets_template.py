"""Agent B deployment only; run after the ten-cell setup on the owned VM."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
import zipfile

ROOT = Path('/content/studies/guidance-components')
PREPARED = ROOT / 'prepared_v1'
ASSETS = PREPARED / 'assets'
DRIVE = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
PYTHON = '/usr/local/envs/3dd_tta_env/bin/python'
SESSION = 'tta-guidance-components-a100-20261025-r1'
MANIFEST_SHA = '7012c30ad88222bbc5435d8508f781fc33385ed3976951c45f48da67f625ed75'
SOURCE_SHA = 'f5bca11a543fd5463ef5a09b26cf216b2bd049523d61a97da24e184e5576d0c6'
PAYLOAD = dict(bytes=5497794, sha256='1646533c54cd4a1d8b8ab921813cd333ade99c470297f5334658ec7007e5065c')
BASE = Path('/content/drive/MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261019T110110Z/r8_assets_v7_full.zip')
BASE_ID = dict(bytes=247002524, sha256='ce9c797ccacb4eeffcc0a2836c65109213c9c7920a40a5145ef598b2adac1333')
DELTA = Path('/content/guidance_components_v1_delta.zip')
MODELNET = Path('/content/3DD-TTA/data/modelnet40_c')
NATIVE_PATHS = {
    '_pvcnn_backend': '/root/.cache/torch_extensions/py38_cu121/_pvcnn_backend/_pvcnn_backend.so',
    'chamfer_3D': '/content/3DD-TTA/third_party/ChamferDistancePytorch/tmp/chamfer_3D.so',
    'emd_ext': '/root/.cache/torch_extensions/py38_cu121/emd_ext/emd_ext.so',
    'knn': '/usr/local/envs/3dd_tta_env/lib/python3.8/site-packages/knn_cuda/csrc/_ext/knn/knn.so'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(path, expected):
    observed = identity(path)
    require(observed == {key: expected[key] for key in ('bytes', 'sha256')}, 'Identity mismatch: ' + str(path))
    return observed


def checked_members(archive):
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    require(len(entries) <= 20000 and len(names) == len(set(names)), 'Duplicate/excessive archive members')
    require(sum(entry.file_size for entry in entries) <= 4 * 1024 ** 3, 'Excessive extracted archive size')
    for entry in entries:
        path = PurePosixPath(entry.filename)
        require(bool(path.parts) and not path.is_absolute() and '..' not in path.parts
                and path.as_posix() == entry.filename.rstrip('/') and '\\' not in entry.filename
                and ':' not in entry.filename and '\x00' not in entry.filename, 'Unsafe archive path')
        mode = stat.S_IFMT(entry.external_attr >> 16)
        require(mode in (0, stat.S_IFREG, stat.S_IFDIR) and entry.file_size <= 2 * 1024 ** 3, 'Unsafe archive type/size')
    require(archive.testzip() is None, 'Archive CRC mismatch')
    return entries


def extract_checked(path, destination, overlay=False, declared=None):
    destination = Path(destination)
    require(not destination.is_symlink() and not any(p.is_symlink() for p in destination.parents), 'Symlink extraction destination')
    with zipfile.ZipFile(path) as archive:
        entries = checked_members(archive)
        if declared is not None:
            require(all(entry.is_dir() or entry.filename in declared for entry in entries), 'Undeclared source archive member')
        destination.mkdir(parents=True, exist_ok=overlay)
        for entry in entries:
            target = destination / entry.filename
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as incoming, target.open('wb' if overlay else 'xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def copy_checked(source, target, expected):
    verify(source, expected)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as incoming, target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    return verify(target, expected)


def seal(directory, drive_target):
    directory, drive_target = Path(directory), Path(drive_target)
    files = {str(path.relative_to(directory)): identity(path) for path in sorted(directory.rglob('*')) if path.is_file()}
    (directory / 'evidence_manifest.json').write_text(json.dumps(files, indent=2, sort_keys=True) + '\n')
    archive = directory.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                require(not path.is_symlink(), 'Evidence symlink forbidden')
                stream.write(path, str(path.relative_to(directory)))
    with zipfile.ZipFile(archive) as stream:
        checked_members(stream)
    expected = identity(archive)
    copy_checked(archive, drive_target, expected)
    return dict(expected, local_path=str(archive), drive_path=str(drive_target))


def preserve_phase_sources(output):
    phases = Path(__file__).resolve().parent
    for name in ('deploy_assets_template.py', 'preflight_components_cpu.py', 'gpu_components_template.py'):
        source = phases / name
        require(source.is_file(), 'Upload all three reviewed phase templates before execution: ' + name)
        shutil.copyfile(source, output / name)


def restore_native(manifest, output, drive_output):
    native = manifest['guidance']['native_archive']
    archive_path = ROOT / 'native' / 'pinned_native.zip'
    if not archive_path.exists():
        copy_checked(native['drive_archive'], archive_path, native)
    verify(archive_path, native)
    audit = json.loads((ASSETS / manifest['guidance']['native_audit']).read_text())
    require(audit.get('status') == 'complete' and set(audit['backends']) == set(NATIVE_PATHS), 'Require complete four-backend audit')
    before = output / 'before_native_restore'
    before.mkdir()
    records, payloads = {}, {}
    with zipfile.ZipFile(archive_path) as stream:
        checked_members(stream)
        for name, allowed in NATIVE_PATHS.items():
            entry, = audit['backends'][name]
            require(entry['path'] == allowed, 'Unaudited native target')
            target = Path(allowed)
            require(not target.is_symlink() and not any(p.is_symlink() for p in target.parents), 'Native target symlink')
            members = [info.filename for info in stream.infolist() if info.filename.endswith('/payload/' + allowed.lstrip('/'))]
            require(len(members) == 1, 'Require one exact pinned native member')
            data = stream.read(members[0])
            pinned = {key: entry[key] for key in ('bytes', 'sha256')}
            require(dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) == pinned, 'Pinned native bytes mismatch')
            current = identity(target) if target.is_file() else None
            if current is not None:
                copy_checked(target, before / (name + '.so'), current)
            records[name] = dict(target=allowed, before=current, pinned=pinned)
            payloads[name] = data
    (before / 'restore_plan.json').write_text(json.dumps(records, indent=2) + '\n')
    preservation = seal(before, drive_output / 'before_native_restore.zip')
    print('FRESH_NATIVE_PRESERVATION_DRIVE_VERIFIED', json.dumps(preservation), flush=True)
    for name, record in records.items():
        target = Path(record['target'])
        if record['before'] != record['pinned']:
            if target.exists():
                adjacent = target.with_name(target.name + '.before_agent_b_' + output.name)
                require(not adjacent.exists(), 'Native backup collision')
                target.rename(adjacent)
                verify(adjacent, record['before'])
                record['preserved_adjacent_path'] = str(adjacent)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(payloads[name])
        verify(target, record['pinned'])
    (output / 'native_restore.json').write_text(json.dumps(dict(records=records, preservation=preservation), indent=2) + '\n')
    return records


def dependency_check(output):
    before = subprocess.check_output([PYTHON, '-m', 'pip', 'freeze'], text=True)
    (output / 'pip_freeze_before.txt').write_text(before)
    query = "import importlib.util,importlib.metadata,json; s=importlib.util.find_spec('multimethod'); print(json.dumps(None if s is None else importlib.metadata.version('multimethod')))"
    version = json.loads(subprocess.check_output([PYTHON, '-c', query], text=True))
    require(version in (None, '1.9.1'), 'Unexpected multimethod version; do not rewrite the environment')
    if version is None:
        command = [PYTHON, '-m', 'pip', 'install', '--no-deps', 'multimethod==1.9.1']
        (output / 'dependency_install_command.json').write_text(json.dumps(command) + '\n')
        with (output / 'dependency_install.log').open('x') as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in child.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            require(child.wait() == 0, 'Pinned multimethod installation failed')
    after = subprocess.check_output([PYTHON, '-m', 'pip', 'freeze'], text=True)
    (output / 'pip_freeze_after.txt').write_text(after)
    require(not set(before.splitlines()) - set(after.splitlines()), 'Dependency restore removed existing packages')
    require(set(after.splitlines()) - set(before.splitlines()) == ({'multimethod==1.9.1'} if version is None else set()), 'Unexpected dependency changes')
    subprocess.run([PYTHON, '-c', "from multimethod import multimethod; import importlib.metadata; assert importlib.metadata.version('multimethod')=='1.9.1'"], check=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    stamp = 'deployment_' + utc()
    output = ROOT / stamp; output.mkdir()
    drive_output = DRIVE / 'phase-evidence' / stamp; drive_output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', session=SESSION, manifest_sha256=MANIFEST_SHA, research_gpu_launched=False)
    try:
        preserve_phase_sources(output)
        verify(BASE, BASE_ID); verify(DELTA, PAYLOAD)
        require(not PREPARED.exists(), 'Fresh prepared_v1 destination required; inspect any partial deployment before retry')
        extract_checked(BASE, PREPARED)
        verify(ASSETS / 'manifest.json', dict(bytes=123403, sha256='78cdb53891bc0714dacd100eb220a5f6deb26bfe36f4e10b0aaa16f9aced60f9'))
        extract_checked(DELTA, PREPARED, overlay=True)
        require(identity(ASSETS / 'manifest.json')['sha256'] == MANIFEST_SHA, 'Agent B manifest SHA mismatch')
        manifest = json.loads((ASSETS / 'manifest.json').read_text())
        verify(ASSETS / 'source_snapshot.zip', dict(bytes=5524113, sha256=SOURCE_SHA))
        extract_checked(ASSETS / 'source_snapshot.zip', ASSETS, overlay=True, declared=manifest['files'])
        for name, source in [('modelnet_jt.pth', '/content/3DD-TTA/pointnet_ckpts/modelnet_jt.pth'),
                             ('scanobject_jt.pth', '/content/drive/MyDrive/thesis/weights/scanobject_jt.pth')]:
            target = ASSETS / 'checkpoints' / name
            expected = manifest['files']['checkpoints/' + name]
            if target.exists():
                verify(target, expected)
            else:
                copy_checked(source, target, expected)
        for name, expected in manifest['files'].items():
            verify(ASSETS / name, expected)
        for entry in (manifest['guidance']['lion_checkpoint'], manifest['guidance']['lion_config']):
            verify(entry['path'], entry)
        state['native_restore'] = restore_native(manifest, output, drive_output)
        dependency_check(output)
        state.update(status='complete', declared_files=len(manifest['files']), source_snapshot_sha256=SOURCE_SHA)
    except BaseException:
        state.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        receipt = dict(state=state, archive=seal(output, drive_output / 'deployment_evidence.zip'))
        (ROOT / 'deployment_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (drive_output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('AGENT_B_DEPLOYMENT_RECEIPT', json.dumps(receipt), flush=True)
    (PREPARED / '.deployed').write_text(MANIFEST_SHA + '\n')
    print('AGENT_B_COMPONENT_ASSETS_DEPLOYED', flush=True)


def self_check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        good = root / 'good.zip'
        with zipfile.ZipFile(good, 'x') as archive:
            archive.writestr('sub/value.txt', 'preserved')
        extract_checked(good, root / 'valid')
        require((root / 'valid/sub/value.txt').read_text() == 'preserved', 'Valid extraction failed')
        for number, name in enumerate(('../escape', '/absolute', 'a\\b', 'a//b')):
            path = root / ('bad%d.zip' % number)
            with zipfile.ZipFile(path, 'x') as archive:
                archive.writestr(name, 'blocked')
            try:
                extract_checked(path, root / ('bad%d' % number))
            except ValueError:
                pass
            else:
                raise AssertionError('Unsafe extraction accepted')
        evidence = root / 'evidence'; evidence.mkdir()
        (evidence / 'record.json').write_text('{}')
        receipt = seal(evidence, root / 'drive.zip')
        require(identity(receipt['local_path']) == identity(receipt['drive_path']), 'Roundtrip self-check failed')
    print('AGENT_B_DEPLOYMENT_TEMPLATE_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
