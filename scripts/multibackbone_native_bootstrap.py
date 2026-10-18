"""Opt-in bindings to audited r7 binaries; no compilation fallback is permitted."""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys

from scripts.eval_multibackbone_source import asset_path, file_identity, verify_identity

ORIGINAL_REPO = Path('/content/3DD-TTA')
KNN_ROOT = Path('/usr/local/envs/3dd_tta_env/lib/python3.8/site-packages/knn_cuda')
GROUPS = {'_pvcnn_backend': Path('third_party/pvcnn/functional'),
          'chamfer_3D': Path('third_party/ChamferDistancePytorch/chamfer3D'),
          'emd_ext': Path('third_party/PyTorchEMD')}
BACKENDS = set(GROUPS) | {'knn'}


def _load_binary(name: str, path: Path):
    if name in sys.modules:
        module = sys.modules[name]
        if Path(getattr(module, '__file__', '')).resolve() != path.resolve():
            raise ValueError('Native module already loaded from a different path: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def bootstrap(asset_root: Path) -> dict:
    """Call in a fresh worker after the caller verifies the pinned asset manifest."""
    root = Path(asset_root).resolve()
    source = Path(__file__).resolve().parents[1]
    if source != root / 'source_snapshot':
        raise ValueError('Native bootstrap must execute from asset-root/source_snapshot')
    manifest = json.loads(asset_path(root, 'manifest.json').read_text())
    def declared(relative):
        if relative not in manifest['files']:
            raise ValueError('Native bootstrap asset is absent from pinned manifest: ' + relative)
        path = asset_path(root, relative)
        verify_identity(path, manifest['files'][relative])
        return path
    declared(str(Path(__file__).resolve().relative_to(root)))
    audit_path = declared(manifest['guidance']['native_audit'])
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'complete' or set(audit.get('backends', {})) != BACKENDS:
        raise ValueError('A complete four-backend native audit is required')
    source_files, expected_calls = {}, {name: set() for name in BACKENDS}
    project_count = 0
    for original, expected in audit['files'].items():
        if expected.get('kind') != 'native_source':
            continue
        original = Path(original)
        backend = None
        for name, relative in GROUPS.items():
            if ORIGINAL_REPO / relative in original.parents:
                backend = name
                path = declared('source_snapshot/' + str(original.relative_to(ORIGINAL_REPO)))
                project_count += 1
                break
        if KNN_ROOT in original.parents:
            backend, path = 'knn', original
        if backend is None:
            continue
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError('Native source symlinks are forbidden')
        verify_identity(path, expected)
        source_files[path.resolve()] = expected
        if path.suffix in ('.cpp', '.cu'):
            expected_calls[backend].add(path.resolve())
    if project_count != 43 or any(not values for values in expected_calls.values()):
        raise ValueError('Incomplete native wrapper/source/header audit')
    bindings = {}
    for name in BACKENDS:
        entries = audit['backends'][name]
        if len(entries) != 1:
            raise ValueError('Native backend requires one unambiguous binary: ' + name)
        entry = entries[0]
        path = Path(entry['path'])
        if path.name != name + '.so' or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError('Unexpected native binary path/name: ' + str(path))
        symbols = [line.split()[-1] for line in entry.get('nm', {}).get('lines', []) if line.split()]
        if 'PyInit_' + name not in symbols:
            raise ValueError('Native binary initialization symbol mismatch: ' + name)
        verify_identity(path, entry)
        bindings[name] = dict(path=str(path), **file_identity(path))
    targets = {'knn_cuda': KNN_ROOT / '__init__.py'}
    targets.update({name: declared('source_snapshot/' + name.replace('.', '/') + '.py')
                    for name in ('models.lion', 'tta_gsd_v2', 'graph_spectral_v2', 'utilities_3dd_tta')})
    import torch
    import torch.utils.cpp_extension as cpp
    python_version = tuple(int(value) for value in audit['runtime']['python'].split()[0].split('.')[:2])
    if sys.version_info[:2] != python_version or torch.__version__ != audit['runtime']['packages']['torch']:
        raise ValueError('Audited native binaries require the recorded Python/PyTorch ABI')
    observed, loaded = [], {}
    def pinned_load(name, sources, *args, **kwargs):
        if name not in BACKENDS:
            raise ValueError('Unaudited JIT request blocked: ' + str(name))
        requested = [Path(path).resolve() for path in sources]
        if len(requested) != len(expected_calls[name]) or set(requested) != expected_calls[name]:
            raise ValueError('Native JIT source list differs from audit: ' + name)
        for path, expected in source_files.items():
            verify_identity(path, expected)
        path = Path(bindings[name]['path'])
        verify_identity(path, bindings[name])
        if name not in loaded:
            loaded[name] = _load_binary(name, path)
        observed.append(name)
        return loaded[name]
    previous = cpp.load
    try:
        cpp.load = pinned_load
        for name, expected_path in targets.items():
            module = importlib.import_module(name)
            if Path(module.__file__).resolve() != expected_path.resolve():
                raise ValueError('Native consumer imported from a different source: ' + name)
    finally:
        cpp.load = previous
    if set(loaded) != BACKENDS:
        raise RuntimeError('Bootstrap requires fresh imports of all four audited native consumers')
    return dict(no_build=True, native_audit_sha256=file_identity(audit_path)['sha256'], bindings=bindings,
                source_files_verified=len(source_files), jit_requests=observed, imported_modules=list(targets))
