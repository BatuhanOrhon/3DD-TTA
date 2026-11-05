"""Native-free guards for the bounded physical diagnostic invocation."""
import importlib.util
import unittest


class DriverContractTests(unittest.TestCase):
    def test_h5_capture_is_explicit_control_only_and_zero_weight(self):
        from scripts.diagnose_decoded_spectral import configuration
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        self.assertFalse(configuration(args)['capture_h5'])
        self.assertTrue(configuration([*args, '--arm', 'control', '--capture-h5'])['capture_h5'])
        for extra in ([], ['--arm', 'live_control'], ['--arm', 'physical', '--weight', '1'],
                      ['--arm', 'control', '--weight', '1']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                configuration([*args, *extra, '--capture-h5'])

    def test_h5_snapshot_and_spectrum_archive_preserve_geometry_rng_and_hashes(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        import torch
        import graph_spectral
        from legacy_trajectory import rng_hash, tensor_hash
        from scripts import diagnose_decoded_spectral as driver
        prepared = torch.linspace(-.2, .2, 2*12*3).reshape(2, 12, 3)
        center, scale = torch.zeros(2, 1, 3), torch.ones(2, 1, 1)
        capture, expected_hashes = {}, []
        before = rng_hash()
        for checkpoint in range(4):
            decoded = (prepared+checkpoint*.01).clone().requires_grad_()
            expected_hashes.append(tensor_hash(decoded))
            driver.capture_h5_snapshot(capture, decoded)
            with torch.no_grad():
                decoded.add_(10.)
        self.assertTrue(all(not value.requires_grad and value.device.type == 'cpu' for value in capture['h5_decoded']))
        builder = graph_spectral.build_probe_spectral_target
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(graph_spectral, 'build_probe_spectral_target', wraps=builder) as build:
                report = driver.record_h5_capture(directory, [459, 545], prepared, center, scale,
                    capture['h5_decoded'], ['clean_02', 'clean_04', 'clean_06', 'terminal'], [2, 4, 6, 8], expected_hashes)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(report['extra_decoder_forward_calls'], 0)
            self.assertEqual(report['decoded_tensor_sha256'], expected_hashes)
            self.assertEqual(report['reference_xyz_sha256'], tensor_hash(prepared))
            with np.load(directory/'h5_geometry.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['decoded_xyz'].shape, (2, 4, 12, 3))
                np.testing.assert_array_equal(saved['candidate_loop_counts'], [2, 4, 6, 8])
                np.testing.assert_array_equal(saved['prepared_x'], prepared.numpy())
                np.testing.assert_array_equal(saved['decoded_xyz'][:, 0], prepared.numpy())
            with np.load(directory/'h5_spectra.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['index'], [459, 545])
                for index in range(2):
                    for name in ('active_indices', 'basis', 'q', 'rank'):
                        values = saved['%s_%d'%(name, index)]
                        self.assertEqual(report['spectra_tensor_sha256'][index][name], tensor_hash(torch.from_numpy(values)))
                    self.assertEqual(saved['basis_%d'%index].shape[0], len(saved['active_indices_%d'%index]))
            for filename in ('h5_geometry.npz', 'h5_spectra.npz'):
                self.assertEqual(report['files'][filename], driver.source.file_identity(directory/filename))
            json.dumps(report, allow_nan=False)
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                driver.record_h5_capture(directory, [459, 545], prepared, center, scale,
                    capture['h5_decoded'][:3], ['a', 'b', 'c', 'd'], [2, 4, 6, 8], expected_hashes)
            failed = directory/'failed'
            failed.mkdir()
            with patch.object(graph_spectral, 'build_probe_spectral_target', side_effect=RuntimeError('probe failed')):
                with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                    driver.record_h5_capture(failed, [459, 545], prepared, center, scale,
                        capture['h5_decoded'], ['clean_02', 'clean_04', 'clean_06', 'terminal'], [2, 4, 6, 8], expected_hashes)
            self.assertTrue((failed/'h5_geometry.npz').is_file())
        self.assertEqual(before, rng_hash())

    def test_all15_corruptions_are_explicitly_supported_without_changing_defaults(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from research_artifacts import CORRUPTIONS
        from scripts.diagnose_decoded_spectral import configuration
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        self.assertEqual(configuration(args)['corruptions'], ['gaussian', 'background'])
        selected = configuration([*args, '--corruptions', *CORRUPTIONS])
        self.assertEqual(selected['corruptions'], list(CORRUPTIONS))
        with self.assertRaises(ValueError):
            configuration([*args, '--corruptions', 'gaussian', 'gaussian'])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            configuration([*args, '--corruptions', 'original'])

    def test_index_manifest_is_pinned_sorted_unique_and_count_matched(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts import diagnose_decoded_spectral as driver
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        self.assertEqual(driver.configuration(args)['selected_original_ids'], [44, 91])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'selection.json'
            good = dict(schema_version=1, dataset='modelnet40_c', indices=[103, 137])
            path.write_text(json.dumps(good))
            identity = driver.source.file_identity(path)
            flags = ['--index-manifest', str(path), '--index-manifest-sha256', identity['sha256']]
            config = driver.configuration([*args, *flags])
            self.assertEqual(config['selected_original_ids'], [103, 137])
            self.assertEqual(config['selection_manifest'], good)
            self.assertEqual(config['selection_manifest_identity'], identity)
            for extra in (['--index-manifest', str(path)], ['--index-manifest-sha256', identity['sha256']],
                          ['--index-manifest', str(path), '--index-manifest-sha256', 'b'*64],
                          [*flags, '--count', '8']):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    driver.configuration([*args, *extra])
            for patch in (dict(schema_version=True), dict(dataset='scanobjectnn_c'),
                          dict(indices=[137, 103]), dict(indices=[103, 103]), dict(indices=[True, 137]),
                          dict(indices=[-1, 137]), dict(indices=[103, 2468]), dict(indices=[103])):
                path.write_text(json.dumps(dict(good, **patch)))
                sha = driver.source.file_identity(path)['sha256']
                with self.subTest(patch=patch), self.assertRaises(ValueError):
                    driver.configuration([*args, '--index-manifest', str(path), '--index-manifest-sha256', sha])
            # The worker's shared validator must reject a changed stored manifest too.
            with self.assertRaisesRegex(ValueError, 'SHA'):
                driver.read_index_manifest(path, identity['sha256'], 2)

    def test_backend_policy_preserves_legacy_tf32_and_controls_flags_without_cuda_init(self):
        import torch
        from scripts import diagnose_decoded_spectral as driver
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        self.assertEqual(driver.configuration(args)['backend_policy'], 'legacy')
        self.assertEqual(driver.configuration([*args, '--backend-policy', 'controlled'])['backend_policy'], 'controlled')
        names = ('benchmark', 'deterministic', 'allow_tf32')
        original = {name: getattr(torch.backends.cudnn, name) for name in names}
        matmul = torch.backends.cuda.matmul.allow_tf32
        algorithms = torch.are_deterministic_algorithms_enabled()
        initialized = torch.cuda.is_initialized()
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            legacy = driver.apply_backend_policy('legacy')
            self.assertTrue(legacy['cudnn_benchmark'])
            self.assertFalse(legacy['cudnn_deterministic'])
            self.assertTrue(legacy['cudnn_allow_tf32'])
            self.assertTrue(legacy['cuda_matmul_allow_tf32'])
            controlled = driver.apply_backend_policy('controlled')
            self.assertFalse(controlled['cudnn_benchmark'])
            self.assertTrue(controlled['cudnn_deterministic'])
            self.assertFalse(controlled['cudnn_allow_tf32'])
            self.assertFalse(controlled['cuda_matmul_allow_tf32'])
            self.assertFalse(controlled['deterministic_algorithms'])
            self.assertFalse(controlled['backend_determinism_guaranteed'])
            self.assertEqual(controlled['cudnn'], torch.backends.cudnn.version())
            self.assertEqual(initialized, torch.cuda.is_initialized())
        finally:
            for name, value in original.items():
                setattr(torch.backends.cudnn, name, value)
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.use_deterministic_algorithms(algorithms)

    def test_live_full_repeat_replays_rng_and_restores_buffers_without_detaching_inputs(self):
        import torch
        from legacy_trajectory import _rng_state, rng_hash, model_buffer_hashes
        from semantic_trajectory import gradient_probe
        from scripts import diagnose_decoded_spectral as driver
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('counter', torch.zeros(1))
                self.flags = []
            def decoder(self, unused, beta, context, style):
                self.flags.append((torch.is_grad_enabled(), context.requires_grad, style.requires_grad))
                self.counter.add_(1)
                return context.square() + style + torch.rand_like(context) + self.counter.clone()
        vae = Decoder().eval()
        latent = torch.ones(2, 3, 1, 1, requires_grad=True)
        style = torch.full_like(latent, .2, requires_grad=True)
        initial_rng = _rng_state()
        with gradient_probe([vae]):
            decoded = vae.decoder(None, beta=None, context=latent.flatten(1), style=style.flatten(1))
        torch.rand(1)
        before = rng_hash(), model_buffer_hashes({'vae': vae})
        report, repeated = driver.repeat_full_decoder(vae, latent, style, decoded.detach(), initial_rng)
        self.assertEqual(report['full_repeat_decoder_max_absolute_difference'], 0.)
        self.assertEqual(report['full_repeat_decoder_rms_difference'], 0.)
        self.assertTrue(report['full_repeat_decoder_output_requires_grad'])
        self.assertEqual(report['extra_full_repeat_decoder_forward_calls'], 1)
        torch.testing.assert_close(decoded, repeated, rtol=0, atol=0)
        self.assertEqual(vae.flags, [(True, True, True)]*2)
        self.assertEqual(before, (rng_hash(), model_buffer_hashes({'vae': vae})))
        self.assertIsNone(latent.grad)
        self.assertIsNone(style.grad)
        torch.testing.assert_close(latent, torch.ones_like(latent), rtol=0, atol=0)
        shifted = decoded.detach() + torch.tensor([[1.], [2.]])
        drift, _ = driver.repeat_full_decoder(vae, latent, style, shifted, initial_rng)
        torch.testing.assert_close(torch.tensor(drift['full_repeat_decoder_max_absolute_difference_per_object']), torch.tensor([1., 2.]))
        torch.testing.assert_close(torch.tensor(drift['full_repeat_decoder_rms_difference_per_object']), torch.tensor([1., 2.]))

    def test_terminal_geometry_retains_raw_frame_for_independent_clean_oracle(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        import torch
        from scripts import diagnose_decoded_spectral as driver
        raw = np.asarray([[[2., 4., 6.], [4., 8., 10.]]], dtype=np.float64)
        raw_copy = raw.copy()
        center = torch.tensor([[[3., 6., 8.]]])
        scale = torch.tensor([[[2.]]])
        clean = torch.tensor([[[4., 8., 10.]]])
        normalized = (clean-center)/scale*3.3885
        decoded = normalized[:, :, [2, 1, 0]].clone()
        decoded[:, :, 0].neg_()
        geometry = torch.ones(1, 2048, 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            driver.record_terminal_geometry(path, [103], raw, center, scale, decoded, geometry)
            with np.load(path/'terminal_geometry.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['index'], [103])
                np.testing.assert_array_equal(saved['raw_corrupted_xyz'], raw_copy)
                recovered = (clean.numpy()-saved['normalization_center'])/saved['normalization_scale']*3.3885
                physical = recovered[:, :, [2, 1, 0]].copy(); physical[:, :, 0] *= -1
                np.testing.assert_array_equal(saved['terminal_decoded_xyz'], physical)
                np.testing.assert_array_equal(saved['terminal_classifier_geometry'], geometry.numpy())
        np.testing.assert_array_equal(raw, raw_copy)

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
