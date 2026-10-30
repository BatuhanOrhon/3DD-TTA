"""Seal a source/metadata overlay over pinned sensitivity assets; CPU only.

Large checkpoint/dataset entries remain inherited manifest identities. Deploy
over the exact full base, extract this delta, then extract source_snapshot.zip
and verify every declared file before execution. This module never deploys.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'result/preparation/chamfer_guidance_variants_20261030'
BASE = ROOT.parent / 'guidance-components/result/preparation/guidance_component_sensitivity_20261025/assets_v2'
BASE_SHA = '599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef'
BULK = ('checkpoints', 'datasets')
ID_NAMES = tuple(dataset + '_' + stage + '_ids.json'
                 for dataset in ('modelnet40_c', 'scanobjectnn_c')
                 for stage in ('smoke', 'development', 'heldout'))


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def _relative(name):
    path = Path(name)
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in str(name):
        raise ValueError('Unsafe asset/source path: ' + str(name))
    return path


def _verify(path, expected):
    actual = identity(path)
    if any(actual[key] != expected[key] for key in ('bytes', 'sha256')):
        raise ValueError('Asset identity mismatch: ' + str(path))
    return actual


def discover_sources(repository, manifest):
    """Refresh the complete prior runtime source inventory and include new code."""
    names = {name[len('source_snapshot/'):] for name in manifest['files']
             if name.startswith('source_snapshot/')
             and not name.startswith(('source_snapshot/tests/', 'source_snapshot/result/'))}
    directories = {Path(name).parts[0] for name in names if len(Path(name).parts) > 1}
    inventory = subprocess.check_output(
        ['git', '-C', str(repository), 'ls-files', '-co', '--exclude-standard', '-z']).decode().split('\0')
    extensions = {'.py', '.yaml', '.yml', '.json', '.sh', '.cpp', '.cu', '.cuh', '.h'}
    for name in inventory:
        path = Path(name)
        if (name and path.suffix in extensions
                and (len(path.parts) == 1 or path.parts[0] in directories)
                and path.parts[0] not in ('result', 'tests')
                and not any(part.startswith('.') or part in ('build', 'dist', '__pycache__') for part in path.parts)):
            names.add(name)
    return sorted(names)


def _archive(path, root, names):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (root / name).read_bytes())
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC failed: ' + str(path))


def build_assets(repository, base, prep, expected_base_sha256, source_names=None, source_commit=None):
    """Refuse overwrite; never read/copy bulk files; publish only a complete seal."""
    repository, base, prep = map(Path, (repository, base, prep))
    destination = prep / 'assets_v1'
    delta = prep / 'chamfer_guidance_variants_v1_delta.zip'
    receipt_path = prep / 'preparation_identity.json'
    if any(path.exists() for path in (destination, delta, receipt_path)):
        raise FileExistsError('Sealed outputs already exist; use a new version destination')
    if identity(base / 'manifest.json')['sha256'] != expected_base_sha256:
        raise ValueError('Pinned base manifest SHA-256 mismatch')
    manifest = json.loads((base / 'manifest.json').read_text())
    source_names = discover_sources(repository, manifest) if source_names is None else sorted(set(source_names))
    inputs = {}
    for name in source_names:
        source = repository / _relative(name)
        source.resolve().relative_to(repository.resolve())
        if source.suffix == '.py':
            ast.parse(source.read_text(), feature_version=(3, 8))
        inputs[source] = identity(source)
    scope = json.loads((prep / 'study_scope.json').read_text())
    selection = json.loads((prep / 'selection_receipt.json').read_text())
    study_names = ('study_scope.json', 'selection_receipt.json') + ID_NAMES
    for name in study_names:
        inputs[prep / name] = identity(prep / name)
    for dataset in ('modelnet40_c', 'scanobjectnn_c'):
        for stage in ('smoke', 'development', 'heldout'):
            name = dataset + '_' + stage + '_ids.json'
            if inputs[prep / name]['sha256'] != selection['datasets'][dataset]['selected'][stage]['sha256']:
                raise ValueError('Task ID manifest differs from selection receipt: ' + name)
    commit = source_commit or subprocess.check_output(
        ['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    source_git_status = subprocess.check_output(
        ['git', '-C', str(repository), 'status', '--porcelain', '--untracked-files=all', '--', *source_names], text=True)
    source_dirty = bool(source_git_status.strip())
    with tempfile.TemporaryDirectory(prefix='.chamfer-seal-', dir=str(prep)) as temporary:
        stage = Path(temporary)
        assets = stage / 'assets'
        assets.mkdir()
        for name, expected in manifest['files'].items():
            relative = _relative(name)
            if relative.parts[0] in BULK or name == 'source_snapshot.zip':
                continue
            _verify(base / relative, expected)
            target = assets / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / relative, target)
            _verify(target, expected)
        manifest.setdefault('source_overlay', {})
        for name in source_names:
            target = assets / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / name, target)
            observed = _verify(target, inputs[repository / name])
            manifest['files']['source_snapshot/' + name] = observed
            manifest['source_overlay'][name] = observed
        for name in study_names:
            target = assets / 'study' / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(prep / name, target)
            manifest['files']['study/' + name] = _verify(target, inputs[prep / name])
        archive = assets / 'source_snapshot.zip'
        names = [name for name in manifest['files']
                 if name != 'source_snapshot.zip' and Path(name).parts[0] not in BULK]
        _archive(archive, assets, names)
        manifest['files']['source_snapshot.zip'] = identity(archive)
        manifest.update(manifest_revision='chamfer_guidance_variants_v1',
                        supersedes_manifest_sha256=expected_base_sha256,
                        source_commit=commit, source_dirty=source_dirty,
                        source_snapshot_archive='source_snapshot.zip',
                        source_snapshot_sha256=identity(archive)['sha256'],
                        gpu_inference_executed=False,
                        revision_reason='Independent Chamfer-only sparse, semantic KL/H and density-confidence SCD; frozen original-ID campaign')
        manifest['guidance'].update(extension_scope='study/study_scope.json', active_backbones=scope['backbones'])
        (assets / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        payload = stage / delta.name
        _archive(payload, stage, ['assets/manifest.json', 'assets/source_snapshot.zip'])
        for path, expected in inputs.items():
            _verify(path, expected)
        receipt = dict(status='sealed_CPU_only', asset_layout='overlay_only_reuses_exact_base_bulk',
                       project_source_commit=commit, source_dirty=source_dirty, source_git_status=source_git_status,
                       source_bytes_recoverable=True,
                       recoverable_overlay_base_manifest_sha256=expected_base_sha256,
                       manifest=identity(assets / 'manifest.json'), source_snapshot=identity(archive),
                       payload=identity(payload), source_files={name: inputs[repository / name] for name in source_names},
                       inherited_bulk_files={name: value for name, value in manifest['files'].items()
                                             if Path(name).parts[0] in BULK})
        assets.rename(destination)
        payload.rename(delta)
        with receipt_path.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--base-manifest-sha256', default=BASE_SHA)
    parser.add_argument('--prep', type=Path, default=PREP)
    parser.add_argument('--seal', action='store_true', help='Use only after the coordinated source freeze')
    args = parser.parse_args(argv)
    if not args.seal:
        print(json.dumps({'status': 'preview_no_files_written', 'base_manifest_sha256': args.base_manifest_sha256,
                          'destination': str(args.prep / 'assets_v1'), 'requires_source_freeze': True}, indent=2))
        return
    print(json.dumps(build_assets(ROOT, args.base, args.prep, args.base_manifest_sha256), indent=2))


if __name__ == '__main__':
    main()
