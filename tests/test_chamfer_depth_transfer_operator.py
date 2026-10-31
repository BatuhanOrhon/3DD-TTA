"""No runtime/Git calls: synthetic operator sealing and safe overlay boundaries."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
OPERATOR = REPO / 'result/preparation/chamfer_depth_transfer_20261031/operator'


def load(name):
    spec = importlib.util.spec_from_file_location('depth_transfer_operator_' + name, OPERATOR / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(tmp_path, monkeypatch):
    builder = load('prepare_assets')
    repo, base, prep = [tmp_path / n for n in ('repo', 'base', 'prep')]
    base.mkdir()
    prep.mkdir()
    for name in builder.EXTRA_SOURCES + ('demo.py',):
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('"""synthetic source"""\n')
    (base / 'metadata.json').write_text('{}')
    manifest = dict(files={'metadata.json': builder.identity(base / 'metadata.json'),
                           'checkpoints/absent.pth': dict(bytes=100, sha256='a' * 64),
                           'datasets/absent.npy': dict(bytes=200, sha256='b' * 64)}, guidance={})
    (base / 'manifest.json').write_text(json.dumps(manifest))
    for dataset, count in (('modelnet40_c', 2468), ('scanobjectnn_c', 581)):
        for stage in ('full', 'smoke'):
            path = prep / 'ids' / (dataset + '_' + stage + '_ids.json')
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(dict(dataset=dataset, n_examples=count, stage='heldout' if stage == 'full' else 'smoke',
                                           indices=list(range(count if stage == 'full' else 32)))))
    monkeypatch.setattr(builder.subprocess, 'check_output', lambda command, **kwargs: 'c' * 40 + '\n' if 'rev-parse' in command else ' M demo.py\n')
    return builder, repo, base, prep


def test_seal_inherits_absent_bulk_and_refuses_overwrite(tmp_path, monkeypatch):
    builder, repo, base, prep = synthetic(tmp_path, monkeypatch)
    receipt = builder.build_assets(repo, base, prep, builder.identity(base / 'manifest.json')['sha256'], source_names=['demo.py'])
    assert receipt['source_dirty']
    assert len(receipt['inherited_bulk_files']) == 2
    assert not (prep / 'assets_v1/checkpoints').exists()
    manifest = json.loads((prep / 'assets_v1/manifest.json').read_text())
    assert manifest['guidance']['active_backbones'] == ['pointmae', 'dgcnn', 'curvenet', 'pointnext']
    with zipfile.ZipFile(prep / 'depth_transfer_assets_v1.zip') as archive:
        assert set(archive.namelist()) == {'assets/manifest.json', 'assets/source_snapshot.zip'}
    with pytest.raises(FileExistsError):
        builder.build_assets(repo, base, prep, receipt['recoverable_overlay_base_manifest_sha256'], source_names=['demo.py'])


def test_wrong_base_hash_has_no_sealed_output(tmp_path, monkeypatch):
    builder, repo, base, prep = synthetic(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='base manifest'):
        builder.build_assets(repo, base, prep, '0' * 64, source_names=['demo.py'])
    assert not (prep / 'assets_v1').exists()


def test_safe_extract_and_real_archive_roundtrip(tmp_path):
    helper = load('deploy_assets_template')
    for number, name in enumerate(('../escape', '/absolute', 'a\\b', 'a//b')):
        archive = tmp_path / ('unsafe%d.zip' % number)
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr(name, 'blocked')
        with pytest.raises(ValueError):
            helper.extract_checked(archive, tmp_path / ('out%d' % number))
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence / 'cpu_report.json').write_text('{}')
    archive = helper.seal(evidence, tmp_path / 'drive.zip')
    assert helper.identity(archive['local_path']) == helper.identity(archive['drive_path'])
    assert set(archive) == {'local_path', 'drive_path', 'bytes', 'sha256'}


def test_source_changes_during_seal_leave_no_published_outputs(tmp_path, monkeypatch):
    builder, repo, base, prep = synthetic(tmp_path, monkeypatch)
    original_archive = builder._archive
    def changing_archive(*args):
        original_archive(*args)
        (repo / 'demo.py').write_text('changed = True\n')
    monkeypatch.setattr(builder, '_archive', changing_archive)
    with pytest.raises(ValueError, match='identity'):
        builder.build_assets(repo, base, prep, builder.identity(base / 'manifest.json')['sha256'], source_names=['demo.py'])
    assert not (prep / 'assets_v1').exists()
    assert not (prep / 'depth_transfer_assets_v1.zip').exists()


def test_deploy_rejects_bulk_replacement_before_overlay_extraction(tmp_path, monkeypatch):
    deployer = load('deploy_and_preflight')
    helper = deployer.tools
    prepared = tmp_path / 'prepared'
    assets = prepared / 'assets'
    base = tmp_path / 'base.zip'
    base_manifest = json.dumps({'original': True}).encode()
    with zipfile.ZipFile(base, 'w') as archive:
        archive.writestr('assets/manifest.json', base_manifest)
        archive.writestr('assets/datasets/original.txt', 'preserved')
    source = tmp_path / 'source.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('datasets/original.txt', 'forbidden replacement')
    manifest = dict(manifest_revision='chamfer_depth_transfer_v1', guidance=dict(active_backbones=['pointmae', 'dgcnn', 'curvenet', 'pointnext']),
                    source_snapshot_sha256=helper.identity(source)['sha256'], files={})
    manifest_file = tmp_path / 'manifest.json'
    manifest_file.write_text(json.dumps(manifest))
    payload = tmp_path / 'payload.zip'
    with zipfile.ZipFile(payload, 'w') as archive:
        archive.write(manifest_file, 'assets/manifest.json')
        archive.write(source, 'assets/source_snapshot.zip')
    monkeypatch.setattr(deployer, 'BASE', base)
    monkeypatch.setattr(deployer, 'BASE_ID', helper.identity(base))
    monkeypatch.setattr(deployer, 'BASE_MANIFEST_SHA', __import__('hashlib').sha256(base_manifest).hexdigest())
    monkeypatch.setattr(deployer, 'PAYLOAD', payload)
    monkeypatch.setattr(deployer, 'ASSETS', assets)
    receipt = dict(payload=helper.identity(payload), manifest=helper.identity(manifest_file), source_snapshot=helper.identity(source))
    with pytest.raises(ValueError, match='Bulk overwrite forbidden'):
        deployer.deploy(receipt, tmp_path / 'output', tmp_path / 'drive')
    assert (assets / 'datasets/original.txt').read_text() == 'preserved'
    with pytest.raises(ValueError, match='Fresh prepared'):
        deployer.deploy(receipt, tmp_path / 'output', tmp_path / 'drive')
