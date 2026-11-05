"""Native-free boundary witnesses for the opt-in baseline trace."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from legacy_trajectory import model_buffer_hashes, rng_hash, tensor_hash
from tests.test_chamfer_depth import CPU, Scheduler, VAE, Prior, load_sampler, seeded


class NativeTraceTests(unittest.TestCase):
    def test_trace_preserves_actual_updates_rng_state_and_call_counts(self):
        from scripts.diagnose_decoded_spectral import NativeVariationTrace
        module = load_sampler()
        def run(trace=None):
            lion = SimpleNamespace(vae=VAE().eval(), priors=[None, Prior().eval()])
            prior_calls, chamfer_calls = [], []
            hook = lion.priors[1].register_forward_hook(lambda *args: prior_calls.append(1))
            def distance(a, b):
                chamfer_calls.append(1)
                values = (a-b).square().sum(-1)
                indices = torch.arange(a.shape[1]).expand(a.shape[0], -1).contiguous()
                return values, values, indices, indices
            seeded()
            before = model_buffer_hashes({'vae': lion.vae, 'prior': lion.priors[1]})
            with patch.multiple(module, torch=CPU(), DDIMScheduler=Scheduler,
                                chamfer_grad=lambda: distance, grad_freeze=lambda model: None):
                result = module.tta_reconstruct(torch.ones(2, 2048, 3), lion, 11, .03, .003, .95,
                                                return_trajectory=True, native_trace=trace)
            hook.remove()
            self.assertEqual(before, model_buffer_hashes({'vae': lion.vae, 'prior': lion.priors[1]}))
            return result, rng_hash(), len(prior_calls), len(chamfer_calls)
        expected = run()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace = NativeVariationTrace(directory, [44, 91], 11)
            actual = run(trace)
            self.assertEqual(expected[1:], actual[1:])
            self.assertEqual(actual[2:], (11, 11))
            for name in ('final_local_latent', 'updated_style', 'original_style'):
                self.assertTrue(torch.equal(getattr(expected[0], name), getattr(actual[0], name)))
            report = json.loads((directory/'native_trace.json').read_text())
            self.assertEqual(len(report['snapshots']), 45)
            self.assertTrue(report['complete'])
            self.assertEqual(sorted({v['loop_count'] for v in report['snapshots']}), list(range(1, 10)))
            def arrays(boundary):
                with np.load(directory/('native_trace_loop01_'+boundary+'.npz'), allow_pickle=False) as saved:
                    return {key: saved[key] for key in saved.files}
            inputs, schedule, scd, grads, after = map(arrays, ['prior_input', 'scheduler', 'scd', 'gradients', 'post_update'])
            np.testing.assert_array_equal(inputs['conditioning_style'], 2*inputs['original_decoder_style'])
            np.testing.assert_array_equal(after['noisy_local'], schedule['prev_sample']-.03*grads['noisy_local_gradient'])
            np.testing.assert_array_equal(after['conditioning_style'], inputs['conditioning_style']-.003*grads['conditioning_style_gradient'])
            self.assertEqual(scd['retained_indices_1'].shape, (2, int(2048*.95)))
            np.testing.assert_array_equal(scd['cutoff_distance_1'], scd['sorted_distances_1'][:, int(2048*.95)-1])
            np.testing.assert_array_equal(scd['next_distance_1'], scd['sorted_distances_1'][:, int(2048*.95)])
            for snapshot in report['snapshots']:
                with np.load(directory/snapshot['file'], allow_pickle=False) as saved:
                    for key, metadata in snapshot['tensors'].items():
                        self.assertEqual(metadata['sha256'], tensor_hash(torch.from_numpy(saved[key])))
                        self.assertTrue(metadata['finite'])
                self.assertEqual(snapshot['rng_before_sha256'], snapshot['rng_after_sha256'])

    def test_driver_trace_requires_control_and_is_separate_from_h5(self):
        from scripts.diagnose_decoded_spectral import configuration
        args = ['--asset-root', '/assets', '--asset-manifest-sha256', 'a'*64,
                '--result-root', '/results', '--paired-state-root', '/paired']
        self.assertFalse(configuration(args)['capture_native_trace'])
        config = configuration([*args, '--arm', 'control', '--capture-native-trace'])
        self.assertTrue(config['capture_native_trace'])
        for extra in ([], ['--arm', 'live_control'], ['--arm', 'physical', '--weight', '1'],
                      ['--arm', 'control', '--capture-h5'],
                      ['--arm', 'control', '--checkpoint-scd-weight', '.5'],
                      ['--arm', 'control', '--update-norm-probes'],
                      ['--arm', 'control', '--lowband-modes', '100']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                configuration([*args, *extra, '--capture-native-trace'])

    def test_trace_api_rejects_nonbaseline_or_noncallable_before_model_use(self):
        module = load_sampler()
        for options in ({'native_trace': True}, {'native_trace': lambda *a, **kw: None, 'scd_normalize': True},
                        {'native_trace': lambda *a, **kw: None, 'shape_guidance': object()},
                        {'native_trace': lambda *a, **kw: None, 'decoded_scd_weight': .5}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'native_trace'):
                module.tta_reconstruct(None, None, 8, .03, .003, .95, **options)

    def test_partial_trace_preserves_failed_numeric_evidence_without_json_nan(self):
        from scripts.diagnose_decoded_spectral import NativeVariationTrace
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trace = NativeVariationTrace(directory, [44, 91], 8)
            value = torch.tensor([[float('nan')], [1.]])
            before = rng_hash()
            trace(1, 70, 'gradients', dict(noisy_local_gradient=value, conditioning_style_gradient=None))
            report = json.loads((directory/'native_trace.json').read_text())
            self.assertFalse(report['complete'])
            metadata = report['snapshots'][0]['tensors']
            self.assertFalse(metadata['noisy_local_gradient']['finite'])
            self.assertIsNone(metadata['noisy_local_gradient']['norm'])
            self.assertFalse(metadata['conditioning_style_gradient']['available'])
            with np.load(directory/report['snapshots'][0]['file'], allow_pickle=False) as saved:
                self.assertTrue(np.isnan(saved['noisy_local_gradient'][0, 0]))
            self.assertEqual(before, rng_hash())


if __name__ == '__main__':
    unittest.main()
