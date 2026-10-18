"""Native binding validation uses real file identities and a fake ELF loader."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/multibackbone_native_bootstrap.py'
GROUPS = {'_pvcnn_backend': 'third_party/pvcnn/functional',
          'chamfer_3D': 'third_party/ChamferDistancePytorch/chamfer3D',
          'emd_ext': 'third_party/PyTorchEMD'}


def identity(path):
    payload = path.read_bytes()
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


@contextmanager
def assets():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        source = root / 'source_snapshot'
        script = source / 'scripts/multibackbone_native_bootstrap.py'
        script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, script)
        files, audited = {}, {}
        for relative in GROUPS.values():
            for path in (REPO / relative).rglob('*'):
                if (path.is_file() and path.suffix in ('.py', '.cpp', '.cu', '.cuh', '.h')
                        and not {'tmp', 'build', 'build_dynamic', '__pycache__'} & set(path.parts)):
                    target = source / path.relative_to(REPO)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    audited['/content/3DD-TTA/' + str(path.relative_to(REPO))] = dict(identity(target), kind='native_source')
        knn = root / 'site/knn_cuda'
        for relative in ('__init__.py', 'csrc/cuda/knn.cpp', 'csrc/cuda/knn.cu'):
            target = knn / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('/* pinned fixture */')
            audited[str(target)] = dict(identity(target), kind='native_source')
        backends = {}
        for name in ('_pvcnn_backend', 'chamfer_3D', 'emd_ext', 'knn'):
            binary = root / 'native' / (name + '.so')
            binary.parent.mkdir(exist_ok=True)
            binary.write_bytes(('fake ELF ' + name).encode())
            backends[name] = [dict(identity(binary), path=str(binary),
                                   nm=dict(returncode=0, lines=['0 T PyInit_' + name]))]
            audited[str(binary)] = dict(identity(binary), kind='native_binary')
        for relative in ('models/lion.py', 'tta_gsd_v2.py', 'graph_spectral_v2.py', 'utilities_3dd_tta.py'):
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# pinned target fixture\n')
        import importlib.metadata
        audit = dict(status='complete', original_repo='/content/3DD-TTA', files=audited, backends=backends,
                     runtime=dict(python=sys.version, packages=dict(torch=importlib.metadata.version('torch'))))
        audit_path = root / 'gsd_native_audit.json'
        audit_path.write_text(json.dumps(audit))
        for path in source.rglob('*'):
            if path.is_file():
                files[str(path.relative_to(root))] = identity(path)
        files[audit_path.name] = identity(audit_path)
        (root / 'manifest.json').write_text(json.dumps(dict(files=files, guidance=dict(native_audit=audit_path.name))))
        spec = importlib.util.spec_from_file_location('native_bootstrap_fixture', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.KNN_ROOT = knn
        yield root, module, audit


class NativeBootstrapTests(unittest.TestCase):
    def test_exact_bindings_reuse_audited_modules_without_calling_original_jit(self):
        import torch.utils.cpp_extension as cpp
        with assets() as (root, module, audit):
            imports = []
            def fake_import(name):
                if name == 'knn_cuda':
                    cpp.load('knn', [str(module.KNN_ROOT / 'csrc/cuda/knn.cpp'), str(module.KNN_ROOT / 'csrc/cuda/knn.cu')])
                    return SimpleNamespace(__file__=str(module.KNN_ROOT / '__init__.py'))
                if name == 'models.lion':
                    for backend, relative in GROUPS.items():
                        sources = [str(path) for path in sorted((root / 'source_snapshot' / relative).rglob('*')) if path.suffix in ('.cpp', '.cu')]
                        cpp.load(backend, sources)
                return SimpleNamespace(__file__=str(root / 'source_snapshot' / (name.replace('.', '/') + '.py')))
            def fake_binary(name, path):
                imports.append((name, str(path)))
                return SimpleNamespace(__file__=str(path))
            with patch.object(cpp, 'load', side_effect=AssertionError('JIT must never run')) as original:
                with patch.object(module.importlib, 'import_module', side_effect=fake_import), patch.object(module, '_load_binary', side_effect=fake_binary):
                    report = module.bootstrap(root)
                self.assertIs(cpp.load, original)
                original.assert_not_called()
            self.assertTrue(report['no_build'])
            self.assertEqual({name for name, _ in imports}, set(audit['backends']))
            self.assertEqual(set(report['bindings']), set(audit['backends']))
            self.assertGreater(report['source_files_verified'], 43)

    def test_tampered_binary_header_or_audit_rejects_before_any_native_import(self):
        for target in ('binary', 'header', 'audit'):
            with self.subTest(target=target), assets() as (root, module, audit):
                path = (Path(audit['backends']['emd_ext'][0]['path']) if target == 'binary' else
                        next((root / 'source_snapshot/third_party/pvcnn').rglob('*.cuh')) if target == 'header' else
                        root / 'gsd_native_audit.json')
                path.write_bytes(path.read_bytes() + b'changed')
                with patch.object(module.importlib, 'import_module') as imports, patch.object(module, '_load_binary') as binaries:
                    with self.assertRaises(ValueError):
                        module.bootstrap(root)
                    imports.assert_not_called()
                    binaries.assert_not_called()

    def test_unknown_jit_name_rejects_before_binary_import_and_restores_loader(self):
        import torch.utils.cpp_extension as cpp
        with assets() as (root, module, _):
            before = cpp.load
            with patch.object(module.importlib, 'import_module', side_effect=lambda name: cpp.load('unreviewed', [])), patch.object(module, '_load_binary') as binary:
                with self.assertRaises(ValueError):
                    module.bootstrap(root)
                binary.assert_not_called()
            self.assertIs(cpp.load, before)

    def test_pinned_audit_with_wrong_python_abi_rejects_before_binary_import(self):
        with assets() as (root, module, audit):
            audit['runtime']['python'] = '0.0.0 incompatible Python'
            audit_path = root / 'gsd_native_audit.json'
            audit_path.write_text(json.dumps(audit))
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['files'][audit_path.name] = identity(audit_path)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(module.importlib, 'import_module') as imports, patch.object(module, '_load_binary') as binary:
                with self.assertRaises(ValueError):
                    module.bootstrap(root)
                imports.assert_not_called()
                binary.assert_not_called()

    def test_wrong_source_list_and_import_failure_restore_loader(self):
        import torch.utils.cpp_extension as cpp
        for error in ('sources', 'import'):
            with self.subTest(error=error), assets() as (root, module, _):
                before = cpp.load
                def fail_import(name):
                    if error == 'sources':
                        return cpp.load('knn', [])
                    raise RuntimeError('import failed')
                with patch.object(module.importlib, 'import_module', side_effect=fail_import), patch.object(module, '_load_binary') as binary:
                    with self.assertRaises((ValueError, RuntimeError)):
                        module.bootstrap(root)
                    binary.assert_not_called()
                self.assertIs(cpp.load, before)


if __name__ == '__main__':
    unittest.main()
