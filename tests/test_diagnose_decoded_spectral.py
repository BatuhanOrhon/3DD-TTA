"""Native-free guards for the bounded physical diagnostic invocation."""
import importlib.util
import unittest


class DriverContractTests(unittest.TestCase):
    def test_scan_worker_counts_real_odd_tail_and_writes_15_class_predictions(self):
        import csv
        import json
        import tempfile
        import types
        from contextlib import ExitStack
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        import torch
        from scripts import diagnose_decoded_spectral as driver
        from tests.test_chamfer_depth import installed_modules
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selection = dict(schema_version=1, dataset='scanobjectnn_c', indices=[578, 579, 580])
            selected = root/'ids.json'
            selected.write_text(json.dumps(selection))
            config = driver.configuration(['--asset-root', str(root), '--asset-manifest-sha256', 'a'*64,
                '--result-root', str(root/'results'), '--paired-state-root', str(root/'paired'),
                '--dataset', 'scanobjectnn_c', '--benchmark-shard', '--arm', 'control', '--count', '3',
                '--corruptions', 'gaussian', '--index-manifest', str(selected),
                '--index-manifest-sha256', driver.source.file_identity(selected)['sha256']])
            bundle = driver.source.RunBundle.create(root/'results', 'scan-tail', config, 'CPU native stubs')
            (bundle.path/'selection_manifest.json').write_bytes(selected.read_bytes())
            class VAE(torch.nn.Module):
                def decoder(self, unused, beta, context, style):
                    return context.reshape(len(context), 2048, 4)[:, :, :3]
            lion = types.SimpleNamespace(vae=VAE().eval(), priors=torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]))
            calls = []
            def reconstruct(prepared, lion, horizon, **options):
                calls.append(len(prepared))
                local = torch.zeros(len(prepared), 8192, 1, 1)
                style = torch.zeros(len(prepared), 128, 1, 1)
                options['state_observer'](dict(local_latent=local, shape_latent=style, noise=local))
                observer = options['trajectory_observer']
                observer.initial(local_latent=local, shape_latent=style, noise=local)
                for step in range(horizon):
                    observer.clean(local, style, step, step)
                observer.final_state(local, style, 0)
                options['scheduler_observer'](types.SimpleNamespace(config={}, timesteps=torch.arange(100), final_alpha_cumprod=torch.tensor(1.)))
                return prepared
            ops = types.SimpleNamespace(normalize=lambda raw: (raw, torch.zeros(len(raw), 1, 3), torch.ones(len(raw), 1, 1)),
                upsample_all=lambda raw, count: raw, rotate_pointcloud=lambda raw: raw)
            dataset = dict(root=str(root/'data'), files={'label.npy': {}, 'data_gaussian_5.npy': {}})
            native = types.ModuleType('pointnet2_ops')
            native.pointnet2_utils = object()
            chamfer = types.ModuleType('third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D')
            chamfer.chamfer_3DDist = lambda: None
            with ExitStack() as stack:
                stack.enter_context(installed_modules({'pointnet2_ops': native, chamfer.__name__: chamfer}))
                stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor))
                for name, value in dict(is_available=True, max_memory_allocated=0, get_device_name='CPU native stub').items():
                    stack.enter_context(patch.object(torch.cuda, name, return_value=value))
                for name in ('manual_seed_all', 'synchronize', 'reset_peak_memory_stats'):
                    stack.enter_context(patch.object(torch.cuda, name))
                stack.enter_context(patch.object(driver, 'verify_setup', return_value={}))
                stack.enter_context(patch.object(driver, 'apply_backend_policy', return_value={}))
                stack.enter_context(patch.multiple(driver.source, read_manifest=lambda *a: {}, dataset_manifest=lambda *a: dataset,
                    load_array=lambda *a, **kw: np.zeros(581, dtype=np.int64) if kw.get('labels') else np.zeros((581, 2048, 3), dtype=np.float32),
                    snapshot_sources=lambda *a: None))
                stack.enter_context(patch.object(driver.shared, 'load_guidance_runtime', return_value=(lion, None, reconstruct, ops, {}, 'fixture')))
                stack.enter_context(patch.object(driver.shared, 'initialize_heads', return_value=({'pointmae': torch.nn.Linear(3, 15).eval()}, {})))
                stack.enter_context(patch.object(driver.shared, 'classify_geometry', side_effect=lambda geometry, *a: ({'pointmae': torch.zeros(len(geometry), 15)}, {})))
                stack.enter_context(patch.object(driver.shared, 'decoded_to_geometry', side_effect=lambda decoded, *a: decoded))
                stack.enter_context(patch.object(driver.shared, 'copy_batch_shard', return_value={}))
                stack.enter_context(patch('scripts.eval_chamfer_depth.scheduler_record', return_value={}))
                driver.worker(str(bundle.path))
            self.assertEqual(calls, [2, 1])
            finished = json.loads((bundle.path/'config.json').read_text())
            self.assertTrue(finished['diagnostic_complete'])
            self.assertEqual(finished['completed_batches'], {'gaussian': 2})
            with (bundle.path/'per_corruption.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual((int(row['n_examples']), int(row['n_correct'])), (3, 3))
            with np.load(bundle.path/'gaussian_predictions.npz', allow_pickle=False) as result:
                self.assertEqual(result['index'].tolist(), [578, 579, 580])
                self.assertEqual(result['trajectory_logits'].shape, (3, 4, 15))
            with np.load(bundle.path/'batches/gaussian/batch0001/predictions.npz', allow_pickle=False) as tail:
                self.assertEqual(tail['trajectory_logits'].shape, (1, 4, 15))

    def test_scan_selection_binds_15_class_source_and_preserves_singleton_tail(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts import diagnose_decoded_spectral as driver
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired',
                '--dataset', 'scanobjectnn_c', '--benchmark-shard']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'selection.json'
            for ids in ([580], [576, 577, 578, 579, 580], list(range(581))):
                path.write_text(json.dumps(dict(schema_version=1, dataset='scanobjectnn_c', indices=ids)))
                config = driver.configuration([*args, '--count', str(len(ids)), '--index-manifest', str(path),
                    '--index-manifest-sha256', driver.source.file_identity(path)['sha256']])
                self.assertEqual(config['selected_original_ids'], ids)
                self.assertEqual((config['n_examples'], config['num_classes'], config['split']), (581, 15, 'main_split'))
                self.assertEqual(config['checkpoint'], 'checkpoints/scanobject_jt.pth')
                self.assertEqual(config['pointmae_config'], 'source_snapshot/cfgs/tta_scanobj.yaml')
            for selection in (dict(dataset='modelnet40_c', indices=[580]),
                              dict(dataset='scanobjectnn_c', indices=[581]),
                              dict(dataset='scanobjectnn_c', indices=[True])):
                path.write_text(json.dumps(dict(schema_version=1, **selection)))
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    driver.configuration([*args, '--count', '1', '--index-manifest', str(path),
                        '--index-manifest-sha256', driver.source.file_identity(path)['sha256']])
        with self.assertRaises(ValueError):
            driver.configuration(args)

    def test_scan_setup_requires_separate_pinned_accepted_assets_receipt(self):
        import copy
        import json
        import tempfile
        from pathlib import Path
        from scripts import diagnose_decoded_spectral as driver
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            def write(name, value):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return driver.source.file_identity(path)
            modelnet = dict(files={'label.npy': dict(bytes=2, sha256='a'*64),
                                  'data_gaussian_5.npy': dict(bytes=3, sha256='b'*64)})
            checkpoint = dict(bytes=351525407, sha256='53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac')
            write('checkpoints/scanobject_jt.pth', {'fixture': 'pinned checkpoint identity supplied below'})
            files = {'checkpoints/modelnet_jt.pth': dict(bytes=1, sha256='c'*64),
                     'checkpoints/scanobject_jt.pth': checkpoint}
            scan = dict(n_examples=581, split='main_split', files={})
            for name in ['label.npy', 'data_original.npy', *('data_'+c+'_5.npy' for c in driver.CORRUPTIONS)]:
                entry = write('datasets/scanobjectnn_c/'+name, {'fixture': name})
                scan['files'][name] = entry
                files['datasets/scanobjectnn_c/'+name] = entry
            lion = dict(bytes=4, sha256='d'*64)
            setup = dict(status='passed', assets={'pointnet_ckpts/modelnet_jt.pth': files['checkpoints/modelnet_jt.pth'],
                'lion_ckpts/epoch_10999_iters_2100999.pt': lion,
                **{'data/modelnet40_c/'+name: entry for name, entry in modelnet['files'].items()}})
            files['setup.json'] = write('setup.json', setup)
            manifest = dict(setup_verification='setup.json', files=files,
                guidance=dict(lion_checkpoint=lion), datasets=dict(modelnet40_c=modelnet, scanobjectnn_c=scan))
            config = dict(dataset='scanobjectnn_c', n_examples=581, num_classes=15,
                          pointmae_checkpoint='checkpoints/scanobject_jt.pth', corruptions=['gaussian'])
            with self.assertRaises(ValueError):
                driver.verify_setup(root, manifest, config)
            receipt = dict(schema_version=1, status='passed', dataset='scanobjectnn_c', split='main_split',
                severity=5, n_examples=581, canonical_setup_sha256=files['setup.json']['sha256'],
                dataset_archive=dict(bytes=488341020, sha256='e236151c9f0ad805f59d3571a956b75ada2e1208b030f56f84546cc993b38eb2'),
                files={name: entry for name, entry in files.items() if name.startswith('datasets/scanobjectnn_c/') or name=='checkpoints/scanobject_jt.pth'},
                checkpoint_load=dict(strict=True, missing_keys=[], unexpected_keys=[], num_classes=15,
                                     checkpoint_sha256=checkpoint['sha256']))
            manifest['scan_asset_verification'] = 'scan.json'
            files['scan.json'] = write('scan.json', receipt)
            # Large real checkpoints are represented by their pinned identity at this I/O boundary.
            from unittest.mock import patch
            real_verify = driver.source.verify_identity
            def verify(path, expected):
                if Path(path).name != 'scanobject_jt.pth':
                    real_verify(path, expected)
            with patch.object(driver.source, 'verify_identity', side_effect=verify):
                accepted = driver.verify_setup(root, manifest, config)
                self.assertEqual(accepted['scan_assets']['identity'], files['scan.json'])
                for changes in (dict(status='failed'), dict(canonical_setup_sha256='f'*64),
                                dict(dataset_archive=dict(bytes=1, sha256='e'*64)), dict(files={}),
                                dict(checkpoint_load=dict(receipt['checkpoint_load'], strict=False))):
                    bad = dict(copy.deepcopy(receipt), **changes)
                    files['scan.json'] = write('scan.json', bad)
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        driver.verify_setup(root, manifest, config)
                files['scan.json'] = write('scan.json', receipt)
                (root/'datasets/scanobjectnn_c/label.npy').write_bytes(b'tampered')
                with self.assertRaises(ValueError):
                    driver.verify_setup(root, manifest, config)

    def test_benchmark_shards_require_explicit_opt_in_and_hashed_id_selection(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts import diagnose_decoded_spectral as driver
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        self.assertFalse(driver.configuration(args).get('benchmark_shard', False))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'selection.json'
            for count in (2, 4, 6, 8, 36, 64, 2468):
                indices = list(range(2432, 2468)) if count == 36 else list(range(count))
                selection = dict(schema_version=1, dataset='modelnet40_c', indices=indices)
                path.write_text(json.dumps(selection))
                identity = driver.source.file_identity(path)
                flags = ['--count', str(count), '--index-manifest', str(path),
                         '--index-manifest-sha256', identity['sha256']]
                with self.subTest(count=count):
                    config = driver.configuration([*args, '--benchmark-shard', *flags])
                    self.assertTrue(config['benchmark_shard'])
                    self.assertEqual(config['selected_original_ids'], indices)
                    self.assertEqual(config['selection_manifest_identity'], identity)
                    if count not in (2, 8):
                        with self.assertRaises(ValueError):
                            driver.configuration([*args, *flags])
            for count in (-2, 0, 1, 3, 2469, 2470):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'Benchmark shard count'):
                    driver.configuration([*args, '--benchmark-shard', '--count', str(count)])
            for extra in ([], ['--index-manifest', str(path)],
                          ['--index-manifest-sha256', identity['sha256']]):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    driver.configuration([*args, '--benchmark-shard', *extra])

    def test_worker_revalidates_shard_count_and_pinned_manifest_before_loading_assets(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import diagnose_decoded_spectral as driver
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            manifest = dict(schema_version=1, dataset='modelnet40_c', indices=[0, 1, 2, 3])
            (path/'selection_manifest.json').write_text(json.dumps(manifest))
            identity = driver.source.file_identity(path/'selection_manifest.json')
            base = dict(driver.configuration(args), run_id='test', count=4, benchmark_shard=True,
                        index_manifest='/original/selection.json', index_manifest_sha256=identity['sha256'],
                        selection_manifest=manifest, selection_manifest_identity=identity,
                        selected_original_ids=[0, 1, 2, 3])
            cases = [(dict(), RuntimeError, 'Reached asset loading'),
                     (dict(benchmark_shard=False), ValueError, 'Count4'),
                     (dict(count=0), ValueError, 'Benchmark shard count'),
                     (dict(count=3), ValueError, 'Benchmark shard count'),
                     (dict(count=2470), ValueError, 'Benchmark shard count'),
                     (dict(index_manifest=None), ValueError, 'manifest'),
                     (dict(index_manifest_sha256=None), ValueError, 'manifest'),
                     (dict(index_manifest_sha256='b'*64), ValueError, 'SHA'),
                     (dict(selected_original_ids=[0, 1, 2, 4]), ValueError, 'worker IDs')]
            for changes, error, message in cases:
                (path/'config.json').write_text(json.dumps(dict(base, **changes)))
                with self.subTest(changes=changes), patch.object(driver, 'apply_backend_policy', return_value={}), \
                        patch.object(driver.source, 'read_manifest', side_effect=RuntimeError('Reached asset loading')), \
                        self.assertRaisesRegex(error, message):
                    driver.worker(str(path))
                self.assertEqual(json.loads((path/'config.json').read_text())['status'], 'failed')

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
                for mode in ([], ['--benchmark-shard']):
                    with self.subTest(patch=patch, mode=mode), self.assertRaises(ValueError):
                        driver.configuration([*args, *mode, '--index-manifest', str(path), '--index-manifest-sha256', sha])
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
