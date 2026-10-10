"""CPU checks for the separately restored smooth-integration entry point."""
import ast
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
import importlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import uuid
import unittest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '5c3ed88314f1c3d50ab77d17f7c6ba22da59c441'


@contextmanager
def workspace_temp():
    # Ordinary workspace mkdir avoids this Windows sandbox's temp ACL issue.
    path = ROOT / ('tmp_legacy_test_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != ROOT.resolve() or not path.name.startswith('tmp_legacy_test_'):
            raise RuntimeError('Unsafe cleanup path')
        shutil.rmtree(path)


class RestoreContractTests(unittest.TestCase):
    def test_offdiag_policy_is_explicit_in_preview(self):
        import eval_gsd_tta_v2 as entry
        import legacy_gsd_artifacts as artifacts
        args = entry.parse_arguments(['--isolate_policy', 'offdiag', '--M_max', '800'])
        self.assertEqual(artifacts.protocol(args)['graph']['isolate_policy'], 'offdiag')
        self.assertEqual(entry.parse_arguments([]).isolate_policy, 'legacy')

    def test_preview_exact_reported_command_without_model_imports(self):
        result = subprocess.run([
            sys.executable, str(ROOT / 'eval_gsd_tta_v2.py'), '--preview',
            '--batch_size', '70', '--weight_spectral', '1.17',
            '--weight_chamfer', '1', '--M_max', '800', '--beta', '2',
            '--gamma', '.01', '--eta', '.01', '--lambdaa', '.95',
            '--denoising_step_bg', '30', '--denoising_step_normal', '10',
        ], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        contract = json.loads(result.stdout)
        self.assertEqual(contract['reference_commit'], REFERENCE)
        self.assertEqual(contract['expected_classifications'], 37020)
        self.assertFalse(contract['scheduler']['set_alpha_to_one'])
        self.assertEqual(contract['decoder_style'], 'updated')
        self.assertEqual(contract['cli']['M_max'], 800)
        self.assertIsNone(contract['cli']['seed'])

    def test_core_sources_equal_pinned_branch(self):
        for name in ('tta_gsd_v2.py', 'graph_spectral_v2.py'):
            reference = subprocess.check_output(['git', 'show', REFERENCE + ':' + name], cwd=ROOT)
            self.assertEqual((ROOT / name).read_bytes().replace(b'\r\n', b'\n'), reference.replace(b'\r\n', b'\n'))

    def test_fresh_output_refuses_existing_directory(self):
        module = importlib.import_module('legacy_gsd_artifacts')
        with workspace_temp() as directory:
            path = Path(directory)
            marker = path / 'existing.txt'
            marker.write_text('preserve', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                module.create_output(path)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'preserve')

    def test_invalid_scope_rejected_before_runtime_import(self):
        for arguments in (['--resume'], ['--M_max', '0'], ['--beta', 'nan'],
                          ['--weight_invariant', '1'], ['--batch_size', '0'],
                          ['--denoising_step_normal', '101'], ['--csv_name', '../evil.csv']):
            result = subprocess.run([sys.executable, 'eval_gsd_tta_v2.py', '--preview', *arguments],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(result.returncode, 0, arguments)
            self.assertNotIn('ModuleNotFoundError', result.stderr)
            self.assertIn('error:', result.stderr)

    def test_inference_batch_body_preserved_except_observer(self):
        reference = ast.parse(subprocess.check_output(['git', 'show', REFERENCE + ':eval_gsd_tta_v2.py'], cwd=ROOT))
        current = ast.parse((ROOT / 'eval_gsd_tta_v2.py').read_text(encoding='utf-8'))
        old = next(n for n in reference.body if isinstance(n, ast.FunctionDef) and n.name == 'process_batches')
        new = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == 'process_batches')
        new.args = old.args
        loop = next(n for n in new.body if isinstance(n, ast.For))
        loop.body = [n for n in loop.body if not (isinstance(n, ast.If) and 'batch_observer' in ast.unparse(n.test))]
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_runtime_import_failure_is_sealed_as_failed_not_complete(self):
        import zipfile
        from unittest.mock import Mock
        import eval_gsd_tta_v2 as entry
        import legacy_gsd_artifacts as artifacts
        with workspace_temp() as root:
            output = root / 'failed'
            args = entry.parse_arguments(['--output_dir', str(output), '--M_max', '800'])
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()), self.assertRaisesRegex(RuntimeError, 'simulated missing dependency'):
                artifacts.run(args, Mock(side_effect=RuntimeError('simulated missing dependency')), Mock(), Mock())
            config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['execution_status'], 'failed')
            self.assertEqual(config['completed_examples'], 0)
            with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
                self.assertIsNone(archive.testzip())
                status = json.loads(archive.read('status.json'))
                self.assertEqual(status['status'], 'failed')

    def test_configure_restores_legacy_eval_modes(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import eval_gsd_tta_v2 as entry
        args = entry.parse_arguments([])
        lion, classifier = Mock(), Mock()
        patches = dict(cfg_from_yaml_file=Mock(return_value=SimpleNamespace(model=SimpleNamespace())),
                       builder=SimpleNamespace(model_builder=Mock(return_value=classifier), load_model=Mock()),
                       diff_config=Mock(), LION=Mock(return_value=lion), GraphSpectralDNA=Mock())
        with patch.multiple(entry, create=True, **patches):
            entry.configure_model(args)
        lion.load_model.assert_called_once_with(args.diff_ckpt)
        lion.vae.eval.assert_called_once_with()
        lion.priors.eval.assert_called_once_with()
        classifier.eval.assert_called_once_with()

    def test_cpu_simulated_all15_and_midrun_failure_preserve_counts(self):
        import numpy as np
        import torch
        import types
        from unittest.mock import patch
        import eval_gsd_tta_v2 as entry
        import legacy_gsd_artifacts as artifacts

        class Dataset:
            def __init__(self, *args):
                self.labels = np.zeros((2468, 1), dtype=np.int64)
            def __len__(self):
                return 2468
            def __getitem__(self, index):
                return torch.zeros(4, 3), torch.tensor([0])

        utilities = types.ModuleType('utilities_3dd_tta')
        utilities.PointDataset = Dataset
        def models(args):
            return (torch.nn.Identity(), types.SimpleNamespace(vae=torch.nn.Identity(),
                    priors=torch.nn.ModuleList([torch.nn.Identity()])), None)
        fail = False
        def batches(loader, base, lion, graph, args, steps, *, batch_observer):
            targets, predictions = [], []
            for data, labels in loader:
                target = labels.view(-1)
                prediction = torch.zeros_like(target)
                batch_observer(target, prediction, torch.zeros(len(target), 40), {'loss': 0.})
                if fail:
                    raise RuntimeError('simulated midrun failure')
                targets.append(target)
                predictions.append(prediction)
            return torch.cat(targets), torch.cat(predictions)

        with workspace_temp() as root:
            fixture = root / 'asset'
            fixture.write_text('CPU fixture only', encoding='utf-8')
            for c in artifacts.CORRUPTIONS:
                (root / ('data_' + c + '_5.npy')).write_bytes(b'CPU fixture only')
            for failed, capped_smoke in ((False, False), (True, False), (False, True)):
                fail = failed
                output = root / ('failed' if failed else 'smoke' if capped_smoke else 'complete')
                arguments = ['--output_dir', str(output), '--batch_size', '2468', '--dataset_root', str(root)]
                if capped_smoke:
                    arguments += ['--max_batches', '1']
                for key in ('pointmae_config','pointmae_ckpt','diff_config','diff_ckpt','label_path'):
                    arguments += ['--' + key, str(fixture)]
                args = entry.parse_arguments(arguments)
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()), \
                     patch.dict(sys.modules, {'utilities_3dd_tta': utilities}), \
                     patch.multiple(torch.cuda, is_available=lambda: True, get_device_name=lambda: 'CPU MOCK',
                                    synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                                    max_memory_allocated=lambda: 0), \
                     patch.object(artifacts, 'native_inventory', return_value={}):
                    if failed:
                        with self.assertRaisesRegex(RuntimeError, 'simulated midrun failure'):
                            artifacts.run(args, lambda: None, models, batches)
                    else:
                        artifacts.run(args, lambda: None, models, batches)
                config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
                self.assertEqual(config['status'], 'failed' if failed else 'partial' if capped_smoke else 'complete')
                self.assertEqual(config['completed_examples'], 2468 if failed else 37020)
                saved = np.load(output / 'predictions' / 'uniform_0000.npz')
                np.testing.assert_array_equal(saved['indices'], np.arange(2468))
                saved.close()


if __name__ == '__main__':
    unittest.main()
