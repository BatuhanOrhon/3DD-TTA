"""Bind actual CPU-sealed asset/source bytes; creates no approval or GPU work."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from campaign import REPO, identity, require, validate_binding, write_new

CRITICAL_SOURCES = ('tta.py', 'shared_trajectory.py', 'legacy_trajectory.py', 'research_artifacts.py',
                    'classifier_backbones.py', 'scripts/eval_chamfer_depth.py',
                    'scripts/eval_multibackbone_guidance.py', 'scripts/eval_multibackbone_source.py',
                    'scripts/analyze_legacy_trajectory.py')


def build(payload, manifest_sha, asset_root):
    with zipfile.ZipFile(payload) as archive:
        raw = archive.read('manifest.json')
        require(hashlib.sha256(raw).hexdigest() == manifest_sha, 'Actual manifest SHA differs')
        manifest = json.loads(raw)
        snapshot = archive.read(manifest['source_snapshot_archive'])
        require(hashlib.sha256(snapshot).hexdigest() == manifest['source_snapshot_sha256'], 'Source snapshot SHA differs')
        sources = {}
        with zipfile.ZipFile(io.BytesIO(snapshot)) as source:
            for name in CRITICAL_SOURCES:
                target = 'source_snapshot/' + name
                raw = source.read(target)
                observed = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                require(observed == manifest['files'][target] == identity(REPO / name), 'Source payload differs from current checkout: ' + name)
                sources['assets/' + target] = observed
    assets = Path(asset_root)
    pointmae_checkpoint = assets / 'checkpoints/modelnet_jt.pth'
    pointmae_config = assets / 'source_snapshot/cfgs/tta_modelnet.yaml'
    result = dict(python='/usr/local/envs/3dd_tta_env/bin/python', repo_root=str(assets / 'source_snapshot'),
        dataset_root='/content/3DD-TTA/data/modelnet40_c', label_path='/content/3DD-TTA/data/modelnet40_c/label.npy',
        asset_root=str(assets), pointmae_config=str(pointmae_config), pointmae_checkpoint=str(pointmae_checkpoint),
        required_source_files=sources, expected_config=dict(
            source_snapshot_sha256=manifest['source_snapshot_sha256'], asset_manifest_sha256=manifest_sha,
            checkpoint_load=dict(strict=True, checkpoint_sha256=manifest['files']['checkpoints/modelnet_jt.pth']['sha256']),
            dataset_provenance=manifest['datasets']['modelnet40_c'], pointmae_config=str(pointmae_config),
            pointmae_checkpoint=str(pointmae_checkpoint), asset_manifest=dict(guidance={k: manifest['guidance'][k]
                for k in ('lion_checkpoint', 'lion_config')})))
    validate_binding(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--asset-root', default='/content/studies/chamfer-depth-tuning/prepared/assets')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_new(args.output, build(args.payload, args.manifest_sha256, args.asset_root))
