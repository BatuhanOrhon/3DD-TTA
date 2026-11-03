"""Native-free guards for the bounded physical diagnostic invocation."""
import importlib.util
import unittest


class DriverContractTests(unittest.TestCase):
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
