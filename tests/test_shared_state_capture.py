"""CPU contracts for reusable baseline states; no native model inference."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from legacy_trajectory import _rng_state, _rng_state_hash, _restore_rng, rng_hash, tensor_hash
from shared_trajectory import SharedTrajectoryObserver
from tests.test_chamfer_depth import CPU, Scheduler, VAE, Prior, chamfer, load_sampler, seeded


class SharedStateCaptureTests(unittest.TestCase):
    def test_opt_in_is_separate_control_only(self):
        from scripts.diagnose_decoded_spectral import configuration
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        self.assertFalse(configuration(args)['capture_shared_states'])
        self.assertTrue(configuration([*args, '--arm', 'control', '--capture-shared-states'])['capture_shared_states'])
        for extra in ([], ['--arm', 'live_control'], ['--arm', 'physical', '--weight', '1'],
                      ['--arm', 'control', '--capture-h5'], ['--arm', 'control', '--capture-native-trace']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                configuration([*args, *extra, '--capture-shared-states'])

    def test_count4_is_shared_capture_only_with_pinned_sorted_four_id_selection(self):
        from scripts.diagnose_decoded_spectral import configuration
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired', '--count', '4']
        for extra in ([], ['--arm', 'control'], ['--arm', 'control', '--capture-h5'],
                      ['--arm', 'control', '--capture-native-trace']):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'Count4'):
                configuration([*args, *extra])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'selection.json'
            selected = dict(schema_version=1, dataset='modelnet40_c', indices=[459, 545, 1229, 1435])
            path.write_text(json.dumps(selected))
            flags = ['--index-manifest', str(path), '--index-manifest-sha256', hashlib.sha256(path.read_bytes()).hexdigest()]
            config = configuration([*args, '--arm', 'control', '--capture-shared-states', *flags])
            self.assertEqual(config['selected_original_ids'], selected['indices'])
            self.assertEqual(config['count'], 4)
            selected['indices'].reverse(); path.write_text(json.dumps(selected))
            flags[-1] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'sorted unique'):
                configuration([*args, '--arm', 'control', '--capture-shared-states', *flags])

    def test_existing_votes_states_rng_calls_and_immutable_snapshots_match(self):
        from scripts.diagnose_decoded_spectral import SharedStateObserver, load_shared_state_rng, record_reference
        module = load_sampler()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def run(capture):
                directory = root/str(capture)
                directory.mkdir()
                prior_rng, decoder_rng = [], []
                class RecordingPrior(Prior):
                    def forward(self, *args, **kwargs):
                        prior_rng.append(rng_hash())
                        torch.rand(1)
                        return super().forward(*args, **kwargs)
                class RecordingVAE(VAE):
                    def decoder(self, *args, **kwargs):
                        decoder_rng.append(rng_hash())
                        return super().decoder(*args, **kwargs)
                lion = SimpleNamespace(vae=RecordingVAE().eval(), priors=[None, RecordingPrior().eval()])
                head = torch.nn.Linear(3, 40).eval().requires_grad_(False)
                classify = lambda value: {'pointmae': head(value.mean(1))}
                observer = SharedStateObserver(lion.vae, {'pointmae': head}, classify, 8,
                    directory=directory, ids=[44, 91], center=torch.zeros(2, 1, 3), scale=torch.ones(2, 1, 1),
                    checkpoint_steps=[2, 4, 6]) if capture else SharedTrajectoryObserver(
                        lion.vae, {'pointmae': head}, classify, 8, 40, checkpoint_steps=[2, 4, 6])
                prior_calls = []
                hook = lion.priors[1].register_forward_hook(lambda *a: prior_calls.append(1))
                prehook = lion.priors[1].register_forward_pre_hook(observer.capture_prior, with_kwargs=True) if capture else None
                seeded()
                reference = torch.ones(2, 2048, 3)
                def state_observer(values):
                    record_reference({}, directory, reference, values)
                with patch.multiple(module, torch=CPU(), DDIMScheduler=Scheduler,
                                    chamfer_grad=lambda: chamfer, grad_freeze=lambda model: None):
                    result = module.tta_reconstruct(reference, lion, 8, .03, .003, .95,
                        trajectory_observer=observer, state_observer=state_observer, return_trajectory=True)
                if prehook:
                    prehook.remove()
                hook.remove()
                return result, observer, rng_hash(), len(prior_calls), lion, prior_rng, decoder_rng
            seeded()
            plain = run(False)
            seeded()
            captured = run(True)
            self.assertEqual(plain[2:4], captured[2:4])
            for key in ('final_local_latent', 'original_style', 'updated_style'):
                self.assertTrue(torch.equal(getattr(plain[0], key), getattr(captured[0], key)))
            np.testing.assert_array_equal(np.stack(plain[1].saved['pointmae']), np.stack(captured[1].saved['pointmae']))
            self.assertEqual(len(plain[4].vae.styles), len(captured[4].vae.styles))
            report = json.loads((root/'True'/'shared_states.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['checkpoint_steps'], [2, 6])
            self.assertEqual(len(report['snapshots']), 7)
            for probe in report['paired_probe_hashes']:
                entries = {v['stage']: v for v in report['snapshots'] if v['loop_count'] == probe['loop_count']}
                self.assertEqual(entries['pre_decoder']['tensor_sha256']['predicted_clean_latent'], probe['clean_latent_sha256'])
                self.assertEqual(entries['decoded']['tensor_sha256']['decoded_xyz'], probe['decoded_geometry_sha256'])
            before = rng_hash()
            for entry in report['snapshots']:
                with np.load(root/'True'/entry['file'], allow_pickle=False) as saved:
                    for key, expected in entry['tensor_sha256'].items():
                        self.assertEqual(tensor_hash(torch.from_numpy(saved[key])), expected)
                    if entry['stage'] in ('pre_prior', 'pre_decoder'):
                        state = load_shared_state_rng(saved)
                        self.assertEqual(_rng_state_hash(state), entry['rng_state_sha256'])
                        live = _rng_state()
                        _restore_rng(state)
                        self.assertEqual(rng_hash(), entry['rng_state_sha256'])
                        _restore_rng(live)
                        expected_rng = captured[5][entry['loop_count']-1] if entry['stage'] == 'pre_prior' else captured[6][[2, 4, 6].index(entry['loop_count'])]
                        self.assertEqual(entry['rng_state_sha256'], expected_rng)
                    if entry['stage'] == 'pre_decoder':
                        self.assertEqual(saved['predicted_clean_latent'].shape, (2, 8192, 1, 1))
                    if entry['stage'] == 'pre_prior':
                        self.assertEqual(saved['conditioning_style'].shape, (2, 128, 1, 1))
            for count in (2, 6):
                pair = {v['stage']: v['rng_state_sha256'] for v in report['snapshots'] if v['loop_count'] == count}
                self.assertNotEqual(pair['pre_prior'], pair['pre_decoder'])
            self.assertEqual(before, rng_hash())

    def test_rng_codec_preserves_cached_gaussians_cuda_byte_arrays_and_rejects_tampering(self):
        from scripts.diagnose_decoded_spectral import SharedStateObserver, load_shared_state_rng
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            observer = SharedStateObserver(VAE().eval(), {'pointmae': torch.nn.Linear(3, 40).eval()},
                lambda x: {}, 8, directory=directory, ids=[44, 91], center=torch.zeros(2, 1, 3),
                scale=torch.ones(2, 1, 1), checkpoint_steps=[2, 4, 6])
            state = _rng_state()
            py, numpy_rng = state[:2]
            synthetic = ((py[0], py[1], .75), (numpy_rng[0], numpy_rng[1], numpy_rng[2], 1, 1.25),
                         state[2], [torch.arange(32, dtype=torch.uint8)])
            observer._save('pre_prior', 2, 70, {}, synthetic)
            with np.load(directory/'shared_state_loop02_pre_prior.npz', allow_pickle=False) as saved:
                loaded = load_shared_state_rng(saved)
                self.assertEqual(_rng_state_hash(loaded), _rng_state_hash(synthetic))
                corrupted = {key: saved[key].copy() for key in saved.files}
            corrupted['torch_rng_cuda_0'][0] ^= 1
            with self.assertRaisesRegex(ValueError, 'RNG hash mismatch'):
                load_shared_state_rng(corrupted)

    def test_decoder_failure_preserves_pre_prior_and_pre_decoder_states(self):
        from scripts.diagnose_decoded_spectral import SharedStateObserver, record_reference
        module = load_sampler()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            vae, prior = VAE().eval(), Prior().eval()
            vae.decoder = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('decoder failed'))
            head = torch.nn.Linear(3, 40).eval()
            observer = SharedStateObserver(vae, {'pointmae': head}, lambda x: {}, 8,
                directory=directory, ids=[44, 91], center=torch.zeros(2, 1, 3), scale=torch.ones(2, 1, 1),
                checkpoint_steps=[2, 4, 6])
            hook = prior.register_forward_pre_hook(observer.capture_prior, with_kwargs=True)
            reference = torch.ones(2, 2048, 3)
            try:
                with patch.multiple(module, torch=CPU(), DDIMScheduler=Scheduler,
                                    chamfer_grad=lambda: chamfer, grad_freeze=lambda model: None):
                    with self.assertRaisesRegex(RuntimeError, 'decoder failed'):
                        module.tta_reconstruct(reference, SimpleNamespace(vae=vae, priors=[None, prior]),
                            8, .03, .003, .95, trajectory_observer=observer,
                            state_observer=lambda values: record_reference({}, directory, reference, values))
            finally:
                hook.remove()
            report = json.loads((directory/'shared_states.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual([v['stage'] for v in report['snapshots']], ['frame', 'pre_prior', 'pre_decoder'])
            self.assertTrue(all((directory/v['file']).is_file() for v in report['snapshots']))


if __name__ == '__main__':
    unittest.main()
