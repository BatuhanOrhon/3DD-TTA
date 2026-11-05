"""Native-free contracts for exact-RNG prior repeats; no LION substitution."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from legacy_trajectory import _rng_state, _rng_state_hash, rng_hash, model_buffer_hashes


class PriorRepeatTests(unittest.TestCase):
    def test_numeric_paired_rng_chain_requires_all_hashes_and_cache_bytes(self):
        from scripts import probe_local_prior_repeat as probe
        state = _rng_state(); py, n, cpu, cuda = state; sha = _rng_state_hash(state)
        metadata = dict(python_rng=py, numpy_rng=[n[0], *n[2:]], cuda_devices=len(cuda), rng_after_sha256=sha)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'paired.npz'
            np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)), numpy_rng_keys=n[1], torch_rng_cpu=cpu.numpy(),
                                **{'torch_rng_cuda_%d'%i: v.numpy() for i, v in enumerate(cuda)})
            loaded = probe.paired_rng(path, probe.identity(path), sha, sha)
            self.assertEqual(_rng_state_hash(loaded), sha)
            with self.assertRaisesRegex(ValueError, 'RNG chain'):
                probe.paired_rng(path, probe.identity(path), '0'*64, sha)
            with self.assertRaisesRegex(ValueError, 'cache identity'):
                probe.paired_rng(path, dict(bytes=0, sha256='0'*64), sha, sha)

    def test_three_actual_forward_calls_keep_grad_semantics_inputs_rng_buffers(self):
        from scripts import probe_local_prior_repeat as probe
        class Prior(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.child = torch.nn.Identity(); self.calls = 0
                self.register_buffer('counter', torch.zeros(1))
            def forward(self, x, t, condition_input, clip_feat):
                self.calls += 1
                assert torch.is_grad_enabled() and x.is_leaf and condition_input.is_leaf
                assert x.requires_grad and condition_input.requires_grad
                self.counter.add_(1)
                return self.child(x*.1+condition_input.mean()*.01+torch.rand(1))
        prior = Prior().eval(); h, style, t = torch.ones(2, 8, 1, 1), torch.ones(2, 2, 1, 1), torch.ones(2)
        state = _rng_state(); before = rng_hash(), model_buffer_hashes({'prior': prior})
        with tempfile.TemporaryDirectory() as temporary:
            report = probe.repeat_prior(prior, h, style, t, state, _rng_state_hash(state), Path(temporary))
            self.assertEqual(prior.calls, 3)
            self.assertEqual(len(report['pair_differences']), 3)
            self.assertTrue(all(row['max_absolute'] == 0 for row in report['pair_differences']))
            self.assertEqual(before, (rng_hash(), model_buffer_hashes({'prior': prior})))
            self.assertTrue(torch.equal(h, torch.ones_like(h)))
            with np.load(Path(temporary)/'repeat0_output.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['noise_pred'].shape, tuple(h.shape))

    def test_leaf_pre_hooks_handle_inplace_input_and_nondeterministic_toy_counter(self):
        from scripts import probe_local_prior_repeat as probe
        class Drifting(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.calls = 0
            def forward(self, x):
                self.calls += 1; return x+self.calls*.001
        class Prior(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.relu = torch.nn.ReLU(inplace=True); self.drift = Drifting()
            def forward(self, x, t, condition_input, clip_feat):
                return self.drift(self.relu(x+condition_input.mean()*0))
        prior=Prior().eval(); state=_rng_state()
        with tempfile.TemporaryDirectory() as temporary:
            report=probe.repeat_prior(prior, torch.ones(2, 8, 1, 1), torch.ones(2, 2, 1, 1), torch.ones(2), state, _rng_state_hash(state), Path(temporary))
            self.assertGreater(report['pair_differences'][0]['max_absolute'], 0)
            for repeat in (1, 2):
                trace=json.loads((Path(temporary)/('repeat%d_leaf_trace.json'%repeat)).read_text())
                candidate=next(row for row in trace if row.get('exact_input_numeric_output_drift'))
                self.assertEqual(candidate['module'], 'drift')
                self.assertTrue(candidate['inputs_exact'])


if __name__ == '__main__':
    unittest.main()
