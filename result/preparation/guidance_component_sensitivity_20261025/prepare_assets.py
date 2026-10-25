"""Seal the sensitivity overlay over accepted v1 assets; CPU only."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

PREP = Path(__file__).resolve().parent
REPOSITORY = PREP.parents[2]
BASE = PREP.parent / 'guidance_component_study_20261025/assets_v1'
DEST = PREP / 'assets_v2'
BASE_SHA = '7012c30ad88222bbc5435d8508f781fc33385ed3976951c45f48da67f625ed75'
OVERLAY = ('tta_gsd_v2.py', 'scripts/eval_multibackbone_guidance.py',
           'spectral_guidance_losses.py', 'tests/test_spectral_guidance_losses.py')


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def main():
    assert REPOSITORY.name == 'guidance-components' and not DEST.exists()
    assert identity(BASE / 'manifest.json')['sha256'] == BASE_SHA
    proof = json.loads((PREP / 'baseline_equivalence_cpu.json').read_text())
    assert proof['status'] == 'passed' and proof['tests_passed'] == 54
    for name in OVERLAY:
        ast.parse((REPOSITORY / name).read_text(), feature_version=(3, 8))
        assert identity(REPOSITORY / name)['sha256'] == proof['source_hashes'][name]
    shutil.copytree(BASE, DEST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = json.loads((DEST / 'manifest.json').read_text())
    for name in OVERLAY:
        target = DEST / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / name, target)
        manifest['files']['source_snapshot/' + name] = identity(target)
        manifest['source_overlay'][name] = identity(target)
    study_files = [PREP / 'trial_registry.json', PREP / 'reused_baselines.json',
                   PREP / 'registration_validation.json', PREP / 'baseline_equivalence_cpu.json',
                   *sorted(PREP.glob('*_ids.json'))]
    for path in study_files:
        target = DEST / 'study' / path.name
        shutil.copyfile(path, target)
        manifest['files']['study/' + path.name] = identity(target)
    for name in ('tta.py', 'graph_spectral_v2.py', 'utilities_3dd_tta.py', 'scripts/multibackbone_native_bootstrap.py'):
        assert identity(DEST / 'source_snapshot' / name) == identity(BASE / 'source_snapshot' / name)
    archive = DEST / 'source_snapshot.zip'
    archive.unlink()
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for name in sorted(manifest['files']):
            if name != 'source_snapshot.zip' and Path(name).parts[0] not in ('checkpoints', 'datasets'):
                stream.write(DEST / name, name)
    manifest['files']['source_snapshot.zip'] = identity(archive)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPOSITORY), text=True).strip()
    manifest.update(manifest_revision='agent_b_sensitivity_v2', supersedes_manifest_sha256=BASE_SHA,
                    source_commit=commit, source_dirty=True, source_snapshot_sha256=identity(archive)['sha256'],
                    revision_reason='Declared L2/Huber/power sensitivity and original Point-MAE terminal boundary; legacy graph/native/original math preserved')
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    delta = PREP / 'sensitivity_v2_delta.zip'
    with zipfile.ZipFile(delta, 'x', zipfile.ZIP_DEFLATED) as stream:
        names = ['manifest.json', 'source_snapshot.zip', *('source_snapshot/' + name for name in OVERLAY),
                 *('study/' + path.name for path in study_files)]
        assert len(names) == len(set(names))
        for name in names:
            stream.write(DEST / name, 'assets/' + name)
    with zipfile.ZipFile(delta) as stream:
        assert stream.testzip() is None
    receipt = dict(status='sealed_CPU_only', project_source_commit=commit, source_dirty=True,
                   recoverable_overlay_base_manifest_sha256=BASE_SHA, manifest=identity(DEST / 'manifest.json'),
                   source_snapshot=identity(archive), payload=identity(delta),
                   overlay={name: identity(REPOSITORY / name) for name in OVERLAY})
    (PREP / 'preparation_identity.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
