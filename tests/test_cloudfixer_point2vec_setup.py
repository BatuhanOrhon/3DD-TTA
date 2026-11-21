import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import setup_cloudfixer_point2vec_colab as setup


def test_cuda_headers_and_library_can_use_different_toolkit_layouts(tmp_path):
    header = tmp_path / 'targets/x86_64-linux/include/cuda_runtime_api.h'
    library = tmp_path / 'lib/libcudart.so'
    for item in (header, library):
        item.parent.mkdir(parents=True)
        item.touch()
    assert setup.cuda_paths(tmp_path) == [str(header.parent), str(library.parent)]
    library.unlink()
    with pytest.raises(FileNotFoundError, match='libcudart'):
        setup.cuda_paths(tmp_path)


def test_completed_install_requires_success_and_matching_identity(tmp_path):
    marker = tmp_path / 'pytorch3d.complete'
    counter = tmp_path / 'installs'
    write = [sys.executable, '-c',
             "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
             "p.open('a').write('built\\n')", str(counter)]
    with pytest.raises(subprocess.CalledProcessError):
        setup.stage(tmp_path, 'pytorch3d', 'source1', [sys.executable, '-c', 'raise SystemExit(9)'])
    assert not marker.exists()
    setup.stage(tmp_path, 'pytorch3d', 'source1', write)
    setup.stage(tmp_path, 'pytorch3d', 'source1', write)
    setup.stage(tmp_path, 'pytorch3d', 'source2', write)
    assert counter.read_text().splitlines() == ['built', 'built']
    assert marker.read_text() == 'source2'
    with pytest.raises(subprocess.CalledProcessError):
        setup.stage(tmp_path, 'pytorch3d', 'source3', [sys.executable, '-c', 'raise SystemExit(9)'])
    assert not marker.exists()


def test_missing_official_manifest_files_are_rejected_before_install(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(author_commit=setup.AUTHOR, classifier_sha256=setup.CHECKPOINT,
                                        official_files={})))
    with pytest.raises(ValueError, match='official source manifest'):
        setup.check_inputs(tmp_path, manifest, tmp_path / 'missing')


@pytest.fixture
def worker(tmp_path, monkeypatch):
    prefix = tmp_path / setup.ENV
    binary = prefix / 'lib/python3.10/site-packages/pytorch3d/_C.so'
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'\x7fELFverified-build')
    setup_script = tmp_path / 'setup.py'
    setup_script.write_text('setup revision')
    freeze = tmp_path / 'pip-freeze.txt'
    freeze.write_bytes(b'example==1\n')
    live = dict(freeze=freeze.read_bytes())
    versions = dict(req.split('==') for req in setup.TORCH + setup.PACKAGES)
    versions.update({'pytorch3d': '0.7.7', 'chamfer-distance': '0.1'})
    sources = {name: dict(url=f'https://github.com/{repo}.git', vcs_info=dict(vcs='git', commit_id=commit))
               for name, (repo, commit) in setup.NATIVE.items()}
    monkeypatch.setattr(setup, 'STATE', tmp_path)
    monkeypatch.setattr(setup, 'SCRIPT', setup_script)
    monkeypatch.setattr(setup, 'sys', SimpleNamespace(prefix=str(prefix), version_info=(3, 10, 14),
                                                      executable=str(prefix/'bin/python'), version='3.10.14'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(version=SimpleNamespace(cuda='12.1')))
    monkeypatch.setitem(sys.modules, 'pytorch3d', SimpleNamespace(_C=SimpleNamespace(__file__=str(binary))))
    monkeypatch.setattr(setup.metadata, 'version', versions.__getitem__)
    monkeypatch.setattr(setup.metadata, 'distribution', lambda name: SimpleNamespace(
        read_text=lambda path: json.dumps(sources[name]) if path == 'direct_url.json' else None))

    def pip_freeze(command):
        assert command == [setup.sys.executable, '-m', 'pip', 'freeze', '--all']
        return live['freeze']

    monkeypatch.setattr(setup.subprocess, 'check_output', pip_freeze)
    identity = dict(script_sha256=setup.sha(setup_script), python_prefix=str(prefix),
                    toolchain=dict(TORCH_CUDA_ARCH_LIST='8.0'), official=setup.AUTHOR_FILES)
    receipt = dict(identity=identity, classifier_sha256=setup.CHECKPOINT, pip_freeze_sha256=setup.sha(freeze),
                   native_extensions={binary.relative_to(prefix).as_posix(): setup.sha(binary)})
    (tmp_path / 'verified.complete').write_text(json.dumps(receipt))
    return SimpleNamespace(live=live, versions=versions, sources=sources, binary=binary, receipt=receipt)


def test_runtime_identity_binds_recoverable_native_bytes_and_live_pins(worker):
    result = setup.runtime_identity()
    assert json.loads(json.dumps(result)) == setup.runtime_identity()
    assert result['native_extensions'] == worker.receipt['native_extensions']
    assert result['package_versions'] == worker.versions
    for name, (repo, commit) in setup.NATIVE.items():
        assert result['native_sources'][name] == dict(url=f'https://github.com/{repo}.git', commit_id=commit)
    worker.binary.write_bytes(b'\x7fELFrebuild')
    with pytest.raises(ValueError, match='native extension SHA'):
        setup.runtime_identity()


def test_supplemental_smoke_reports_exact_worker_freeze_and_native_bytes(worker, monkeypatch):
    report = setup.STATE / 'verified.tmp'
    monkeypatch.setattr(sys, 'argv', ['setup', '--smoke', '--smoke-report', str(report), '--official-root',
                                     str(setup.STATE), '--manifest', str(setup.SCRIPT),
                                     '--classifier-checkpoint', str(setup.SCRIPT)])
    monkeypatch.setattr(setup, 'smoke', lambda official, checkpoint: setup.native_extensions())
    worker.live['freeze'] = b'actual==3\n'
    setup.main()
    frozen = setup.STATE / 'pip-freeze.txt'
    assert frozen.read_bytes() == worker.live['freeze']
    assert json.loads(report.read_text()) == dict(native_extensions=worker.receipt['native_extensions'],
                                                  pip_freeze_sha256=setup.sha(frozen))


def test_runtime_identity_rejects_changed_freeze_or_setup_script(worker):
    worker.live['freeze'] += b'extra==1\n'
    with pytest.raises(ValueError, match='Live pip freeze'):
        setup.runtime_identity()
    (setup.STATE / 'pip-freeze.txt').write_bytes(worker.live['freeze'])
    with pytest.raises(ValueError, match='Saved pip freeze'):
        setup.runtime_identity()
    setup.SCRIPT.write_text('changed')
    with pytest.raises(ValueError, match='setup script'):
        setup.runtime_identity()


@pytest.mark.parametrize('name', setup.NATIVE)
def test_runtime_identity_rejects_wrong_native_commit(worker, name):
    worker.sources[name]['vcs_info']['commit_id'] = 'bad-revision'
    with pytest.raises(ValueError, match=f'{name} source mismatch'):
        setup.runtime_identity()


def test_runtime_identity_rejects_wrong_head_package_or_interpreter(worker):
    worker.versions['timm'] = '0.0'
    with pytest.raises(RuntimeError, match='Expected timm=='):
        setup.runtime_identity()
    setup.sys.version_info = (3, 10, 13)
    with pytest.raises(RuntimeError, match='Expected cf_point2vec Python 3.10.14'):
        setup.runtime_identity()


def test_official_config_import_dependency_is_installed():
    assert 'easydict==1.13' in setup.PACKAGES
