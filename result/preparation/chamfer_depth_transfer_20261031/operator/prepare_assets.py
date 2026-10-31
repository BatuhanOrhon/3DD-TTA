"""Seal only after coordinated source freeze; source/metadata overlay, no bulk copies."""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.prepare_chamfer_variants_assets import discover_sources, _archive, _relative, _verify, identity

PREP = REPO / 'result/preparation/chamfer_depth_transfer_20261031'
BASE = Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025/assets_v2')
BASE_SHA = '599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef'
BULK = ('checkpoints', 'datasets')
BACKBONES = ['pointmae', 'dgcnn', 'curvenet', 'pointnext']
EXTRA_SOURCES = tuple('result/preparation/chamfer_depth_tuning_20261030/' + name for name in ('analyze.py', 'campaign.py', 'operations.py')) + tuple(
    'result/preparation/chamfer_depth_transfer_20261031/operator/' + name for name in ('prepare_assets.py', 'deploy_and_preflight.py', 'deploy_assets_template.py')) + (
    'scripts/analyze_chamfer_depth_transfer.py', 'scripts/run_chamfer_depth_transfer.py',
    'scripts/prepare_chamfer_variants_assets.py')


def build_assets(repository, base, prep, expected_base_sha256, source_names=None):
    """Create fresh assets_v1, payload and receipt; inherited bulk is never opened."""
    repository, base, prep = map(Path, (repository, base, prep))
    destination, payload, receipt_path = prep / 'assets_v1', prep / 'depth_transfer_assets_v1.zip', prep / 'preparation_identity.json'
    if any(p.exists() for p in (destination, payload, receipt_path)):
        raise FileExistsError('Sealed outputs already exist; use a fresh version destination')
    if identity(base / 'manifest.json')['sha256'] != expected_base_sha256:
        raise ValueError('Pinned base manifest SHA mismatch')
    manifest = json.loads((base / 'manifest.json').read_text())
    bulk = {k: copy.deepcopy(v) for k, v in manifest['files'].items() if Path(k).parts[0] in BULK}
    names = sorted(set(discover_sources(repository, manifest) if source_names is None else source_names) | set(EXTRA_SOURCES))
    inputs = {}
    for name in names:
        path = repository / _relative(name)
        path.resolve().relative_to(repository.resolve())
        if path.is_symlink():
            raise ValueError('Source symlink forbidden: ' + name)
        if path.suffix == '.py':
            ast.parse(path.read_text(), feature_version=(3, 8))
        inputs[path] = identity(path)
    id_names = []
    for dataset, count in (('modelnet40_c', 2468), ('scanobjectnn_c', 581)):
        for stage in ('full', 'smoke'):
            name = dataset + '_' + stage + '_ids.json'
            path = prep / 'ids' / name
            value = json.loads(path.read_text())
            expected_ids = list(range(count if stage == 'full' else 32))
            if (value.get('dataset') != dataset or value.get('n_examples') != count or value.get('indices') != expected_ids
                    or value.get('stage') != ('heldout' if stage == 'full' else 'smoke')):
                raise ValueError('Fixed full/smoke ID contract mismatch: ' + name)
            inputs[path] = identity(path)
            id_names.append(name)
    commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Actual source Git commit missing')
    with tempfile.TemporaryDirectory(prefix='.depth-transfer-seal-', dir=str(prep)) as temporary:
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
        for name in names:
            target = assets / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / name, target)
            observed = _verify(target, inputs[repository / name])
            manifest['files']['source_snapshot/' + name] = observed
            manifest['source_overlay'][name] = observed
        for name in id_names:
            target = assets / 'study' / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(prep / 'ids' / name, target)
            manifest['files']['study/' + name] = _verify(target, inputs[prep / 'ids' / name])
        scope = dict(study='frozen_n8_chamfer_depth_transfer', datasets=['modelnet40_c', 'scanobjectnn_c'],
                     backbones=BACKBONES, seeds=[0, 1, 2], batch_size=32, normal_reverse_steps=8,
                     background_reverse_steps=35, eta=.003, gamma=.03, p=.95, temperature=1.,
                     readout='equal probability mean', normal_candidates=[2, 4, 6, 8], background_candidates=[9, 18, 27, 35],
                     original_style=True, final_alpha_to_one=True, lion_ema=False, spectral=False, semantic=False,
                     calibration_required=False, parameter_search=False, no_model_ensemble=True)
        scope_path = assets / 'study/depth_transfer_scope.json'
        scope_path.write_text(json.dumps(scope, indent=2, sort_keys=True) + '\n')
        manifest['files']['study/depth_transfer_scope.json'] = identity(scope_path)
        snapshot = assets / 'source_snapshot.zip'
        snapshot_names = [n for n in manifest['files'] if n != 'source_snapshot.zip' and Path(n).parts[0] not in BULK]
        _archive(snapshot, assets, snapshot_names)
        manifest['files']['source_snapshot.zip'] = identity(snapshot)
        manifest.update(manifest_revision='chamfer_depth_transfer_v1', source_commit=commit, source_dirty=bool(status.strip()),
                        supersedes_manifest_sha256=expected_base_sha256, source_snapshot_archive='source_snapshot.zip',
                        source_snapshot_sha256=identity(snapshot)['sha256'], gpu_inference_executed=False,
                        revision_reason='Only frozen N8 original Chamfer transfer, all four frozen heads; no search or guidance combinations')
        manifest['guidance'].update(active_backbones=BACKBONES, parked_backbones=[], extension_scope='study/depth_transfer_scope.json')
        if bulk != {k: v for k, v in manifest['files'].items() if Path(k).parts[0] in BULK}:
            raise ValueError('Inherited checkpoint/dataset identities changed')
        (assets / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        staged_payload = stage / payload.name
        _archive(staged_payload, stage, ['assets/manifest.json', 'assets/source_snapshot.zip'])
        for path, expected in inputs.items():
            _verify(path, expected)
        receipt = dict(status='sealed_CPU_only', asset_layout='overlay_only_reuses_exact_base_bulk',
                       project_source_commit=commit, source_dirty=bool(status.strip()), source_git_status=status,
                       source_bytes_recoverable=True, recoverable_overlay_base_manifest_sha256=expected_base_sha256,
                       manifest=identity(assets / 'manifest.json'), source_snapshot=identity(snapshot), payload=identity(staged_payload),
                       source_files={n: inputs[repository / n] for n in names}, inherited_bulk_files=bulk,
                       id_files={n: inputs[prep / 'ids' / n] for n in id_names})
        assets.rename(destination)
        staged_payload.rename(payload)
        with receipt_path.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--prep', type=Path, default=PREP)
    parser.add_argument('--base-manifest-sha256', default=BASE_SHA)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args(argv)
    if not args.seal:
        return dict(status='preview_no_files_written', requires_coordinated_source_freeze=True,
                    payload=str(args.prep / 'depth_transfer_assets_v1.zip'), base_manifest_sha256=args.base_manifest_sha256)
    return build_assets(REPO, args.base, args.prep, args.base_manifest_sha256)


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, sort_keys=True))
