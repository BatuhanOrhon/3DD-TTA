"""Native-free guards for the bounded physical diagnostic invocation."""
import importlib.util
import unittest


class DriverContractTests(unittest.TestCase):
    def test_surrogate_is_explicit_and_lightweight_controls_have_zero_weight(self):
        from scripts.diagnose_decoded_spectral import configuration
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        self.assertFalse(configuration(args)['freeze_decoder_residual'])
        live = configuration([*args, '--arm', 'live_control'])
        self.assertEqual(live['weight'], 0.)
        self.assertTrue(configuration([*args, '--arm', 'physical', '--weight', '1',
                                       '--freeze-decoder-residual'])['freeze_decoder_residual'])
        for arm in ('control', 'diagnostic'):
            with self.assertRaises(ValueError):
                configuration([*args, '--arm', arm, '--freeze-decoder-residual'])
        with self.assertRaises(ValueError):
            configuration([*args, '--arm', 'live_control', '--weight', '1'])

    def test_native_backward_failure_is_archived_before_observer_backward(self):
        import ast
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import numpy as np
        from scripts import diagnose_decoded_spectral as driver
        import decoded_spectral_native_audit as audit
        tree = ast.parse(Path(driver.__file__).read_text())
        observer = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                        and node.name == 'DiagnosticObserver')
        base = type('Base', (), {'clean': Mock(side_effect=AssertionError('premature backward'))})
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            scope = dict(DecodedSpectralObserver=base, config={'native_decoder_audit': True},
                         capture={}, lion=SimpleNamespace(vae=None), target=None, np=np, case=case, json=json)
            exec(compile(ast.Module(body=[observer], type_ignores=[]), driver.__file__, 'exec'), scope)
            instance = scope['DiagnosticObserver']()
            instance.steps = [2, 4, 6]
            with patch.object(audit, 'audit_decoded_gradient', return_value=(
                    {'passed': False, 'errors': ['unsupported native backward']}, {'baseline_decoded': np.zeros((2, 3))})):
                with self.assertRaisesRegex(RuntimeError, 'evidence retained'):
                    instance.clean(None, None, 1, 90)
            base.clean.assert_not_called()
            self.assertFalse(json.loads((case / 'decoder_native_audit.json').read_text())['passed'])
            with np.load(case / 'decoder_native_audit.npz', allow_pickle=False) as saved:
                self.assertIn('baseline_decoded', saved)

    def test_state_capture_supplies_the_same_h0_used_by_scd_diagnostics(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        import torch
        from scripts import diagnose_decoded_spectral as driver
        prepared = torch.zeros(2, 2048, 3)
        local = torch.arange(2 * 8192, dtype=torch.float32).reshape(2, 8192, 1, 1).requires_grad_()
        capture = {}
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            driver.record_reference(capture, case, prepared, {'local_latent': local})
            with np.load(case / 'reference.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['initial_h0_xyz'], capture['initial_h0_xyz'].numpy())
                np.testing.assert_array_equal(saved['prepared_x'], prepared.numpy())
            self.assertFalse(capture['initial_h0_xyz'].requires_grad)
            torch.testing.assert_close(capture['initial_h0_xyz'][0, 0], torch.tensor([0., 1., 2.]))
            torch.testing.assert_close(capture['initial_h0_xyz'][1, 0], torch.tensor([8192., 8193., 8194.]))

    def test_native_decoder_audit_is_explicit_and_diagnostic_only(self):
        from scripts.diagnose_decoded_spectral import configuration
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        self.assertFalse(configuration(args)['native_decoder_audit'])
        self.assertTrue(configuration([*args, '--native-decoder-audit'])['native_decoder_audit'])
        for extra in (['--arm', 'control'], ['--arm', 'physical', '--weight', '1']):
            with self.assertRaises(ValueError):
                configuration([*args, *extra, '--native-decoder-audit'])

    def test_cli_locks_protocol_selection_and_requires_explicit_physical_weight(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.diagnose_decoded_spectral'))
        from scripts.diagnose_decoded_spectral import configuration
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        control = configuration([*args, '--arm', 'control'])
        self.assertEqual(control['selected_original_ids'], [44, 91])
        self.assertEqual(control['guidance_protocol'],
                         dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False))
        self.assertEqual(control['trajectory_selection'],
                         dict(normal=[2, 4, 6], background=[9, 18, 27]))
        self.assertEqual(control['pointmae_checkpoint'], 'checkpoints/modelnet_jt.pth')
        self.assertEqual(control['pointmae_config'], 'source_snapshot/cfgs/tta_modelnet.yaml')
        diagnostic = configuration([*args, '--arm', 'diagnostic', '--count', '8'])
        self.assertEqual(diagnostic['selected_original_ids'], [44, 91, 103, 137, 158, 223, 236, 265])
        self.assertEqual(diagnostic['weight'], 0)
        physical = configuration([*args, '--arm', 'physical', '--weight', '.5'])
        self.assertEqual(physical['weight'], .5)
        for arm, weight in [('physical', '0'), ('physical', '-1'), ('physical', 'nan'),
                            ('physical', 'inf'), ('control', '.1'), ('diagnostic', '1')]:
            with self.subTest(arm=arm, weight=weight), self.assertRaises(ValueError):
                configuration([*args, '--arm', arm, '--weight', weight])
        with self.assertRaises(ValueError):
            configuration([*args, '--asset-manifest-sha256', 'unpinned'])


if __name__ == '__main__':
    unittest.main()
