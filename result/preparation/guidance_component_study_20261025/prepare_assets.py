"""Seal Agent B's overlay over accepted immutable v8 assets; no remote work."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
PREP = Path(__file__).resolve().parent
BASE = Path('/Users/batuhan/Desktop/projects/3DD-TTA/result/preparation/scan_trajectory_20261019/assets_v8')
DEST = PREP / 'assets_v1'
OVERLAY = ['tta_gsd_v2.py', 'shared_trajectory.py', 'scripts/eval_multibackbone_guidance.py',
           'scripts/analyze_scan_trajectory.py', 'tests/test_guidance_components.py',
           'tests/test_shared_trajectory.py', 'tests/test_multibackbone_scan_trajectory.py']


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def main():
    assert ROOT.name == 'guidance-components' and not DEST.exists()
    assert subprocess.check_output(['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip() == 'gsd-guidance-components'
    expected = '7c3b55fe7cf179a63ee47c4c62a1a02cacd68372a645b534dd31f4d8e616806c'
    assert identity(BASE / 'manifest.json')['sha256'] == expected
    scope = json.loads((PREP / 'study_scope.json').read_text())
    assert scope['initial_gpu_phase']['cap_seconds'] == 3300
    for name in OVERLAY:
        ast.parse((ROOT / name).read_text(), feature_version=(3, 8))
    shutil.copytree(BASE, DEST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = json.loads((DEST / 'manifest.json').read_text())
    for name in OVERLAY:
        target = DEST / 'source_snapshot' / name
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(ROOT / name, target)
        manifest['files']['source_snapshot/' + name] = identity(target)
        manifest['source_overlay'][name] = identity(target)
    manifest['files']['checkpoints/modelnet_jt.pth'] = dict(bytes=351656113, sha256='507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75')
    for path in [PREP / 'study_scope.json', *sorted(PREP.glob('*_ids.json'))]:
        target = DEST / 'study' / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        manifest['files']['study/' + path.name] = identity(target)
    # Preserve audited native consumer sources; the changed guidance consumer is declared above.
    for name in ['tta.py', 'graph_spectral_v2.py', 'utilities_3dd_tta.py', 'scripts/multibackbone_native_bootstrap.py']:
        assert identity(DEST / 'source_snapshot' / name) == identity(BASE / 'source_snapshot' / name)
    archive = DEST / 'source_snapshot.zip'
    archive.unlink()
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for name in sorted(manifest['files']):
            if name != 'source_snapshot.zip' and Path(name).parts[0] not in ('checkpoints', 'datasets'):
                stream.write(DEST / name, name)
    manifest['files']['source_snapshot.zip'] = identity(archive)
    manifest.update(manifest_revision='agent_b_components_v1', supersedes_manifest_sha256=expected,
                    source_snapshot_sha256=identity(archive)['sha256'], source_dirty=True,
                    source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                    revision_reason='Agent B opt-in four-arm paired-state comparison, strict 40/15-class observation and fixed original-ID subsets')
    manifest['guidance'].update(extension_scope='study/study_scope.json', active_backbones=scope['backbones'])
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    delta = PREP / 'guidance_components_v1_delta.zip'
    with zipfile.ZipFile(delta, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for name in OVERLAY:
            stream.write(DEST / 'source_snapshot' / name, 'assets/source_snapshot/' + name)
        for name in ['manifest.json', 'source_snapshot.zip', *sorted('study/' + p.name for p in (DEST / 'study').iterdir())]:
            stream.write(DEST / name, 'assets/' + name)
    with zipfile.ZipFile(delta) as stream:
        assert stream.testzip() is None
        assert len(stream.namelist()) == len(set(stream.namelist()))
    receipt = dict(payload=identity(delta), manifest=identity(DEST / 'manifest.json'), source_snapshot=identity(archive),
                   overlay={name: identity(ROOT / name) for name in OVERLAY}, source_commit=manifest['source_commit'],
                   source_dirty=True, source_bytes_recoverable=True, historical_base_manifest_sha256=expected)
    (PREP / 'preparation_identity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
