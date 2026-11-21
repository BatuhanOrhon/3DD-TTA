"""Exercise the real loader/inference closure with CPU author/CUDA boundary fixtures."""
import io
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

from tests.test_cloudfixer_diffusion import ConstantEpsilon


def test_runtime_dense_trace_classifies_after_transport_and_preserves_canonical_logits(tmp_path, monkeypatch):
    from scripts import run_cloudfixer_transfer as transfer
    events = []

    class Frozen:
        def cuda(self): return self
        def eval(self): return self
        def requires_grad_(self, value): return self
        def state_dict(self): return {'weight': torch.tensor([1.])}
        def load_state_dict(self, state, strict): assert strict and 'weight' in state

    class Gamma:
        gamma = torch.zeros(501)
        def __call__(self, t): return t

    class Prior(ConstantEpsilon, Frozen):
        def __init__(self):
            super().__init__()
            self.gamma = Gamma()

        def phi(self, x, t, node_mask):
            events.append(('denoise', int(round(t[0, 0].item() * 500))))
            super().phi(x, t, node_mask)
            return x.sin() * .1 + t[:, :, None]

    class Classifier(Frozen):
        def __call__(self, x):
            events.append(('classify', None))
            return x.mean((1, 2))[:, None] * torch.arange(40)[None, :] + torch.rand(40)

    class Dataset:
        pc_list = np.zeros((2, 2, 3), dtype=np.float32)
        def __init__(self, options, partition): assert partition == 'test'
        def __getitem__(self, index):
            return torch.full((2, 3), float(index + 1)), index, torch.ones(2), torch.arange(2)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    modules = {
        'adapt': SimpleNamespace(get_model=lambda *a: Prior(), parse_arguments=lambda: SimpleNamespace(),
                                 ModelNet40C=Dataset),
        'classifier.pointMAE': SimpleNamespace(build_model_from_cfg=lambda cfg: Classifier()),
        'utils.config': SimpleNamespace(cfg_from_yaml_file=lambda path: SimpleNamespace(model={})),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(torch, 'load', lambda path, **kw:
                        {'base_model': {'weight': torch.tensor([1.])}} if path == 'classifier' else {'weight': torch.tensor([1.])})
    for name in ('manual_seed_all', 'synchronize', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda *a: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state, *a: torch.set_rng_state(state))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU fixture')
    monkeypatch.setattr(transfer, 'file_hash', lambda path: 'fixture')
    monkeypatch.setattr(transfer.os, 'environ', dict(transfer.os.environ))
    for backend, attribute in ((torch.backends.cudnn, 'benchmark'), (torch.backends.cudnn, 'deterministic'),
                               (torch.backends.cudnn, 'allow_tf32'), (torch.backends.cuda.matmul, 'allow_tf32')):
        monkeypatch.setattr(backend, attribute, getattr(backend, attribute))
    args = SimpleNamespace(official_root=str(tmp_path), dataset_root=str(tmp_path),
                           classifier='classifier', diffusion='diffusion', output=str(tmp_path))
    manifest = dict(official_files={}, overlay_files={}, dataset_files={},
                    classifier_sha256='fixture', diffusion_sha256='fixture')
    unit = dict(campaign='fixture', phase='development', config_id='fixture', corruption='gaussian',
                seed=0, ids=[0, 1], config=dict(method='proximal', steps=8, start=80, tau=.1, retained=1.))
    original_rng, deterministic = torch.get_rng_state(), torch.are_deterministic_algorithms_enabled()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        sparse_infer, _, _ = transfer.runtime(args, manifest)
        sparse = sparse_infer(unit)
        sparse_rng = torch.get_rng_state()
        events.clear()
        dense_infer, verify, _ = transfer.runtime(args, manifest, trace_all=True)
        dense = dense_infer(unit)
        assert torch.equal(torch.get_rng_state(), sparse_rng)
        verify()
    finally:
        torch.set_rng_state(original_rng)
        torch.use_deterministic_algorithms(deterministic)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    with np.load(io.BytesIO(sparse), allow_pickle=False) as old, np.load(io.BytesIO(dense), allow_pickle=False) as new:
        assert old['logits'].shape == (2, 4, 40) and new['logits'].shape == (2, 9, 40)
        assert np.array_equal(old['logits'], new['logits'][:, [1, 3, 5, 8]])
        assert old['denoiser_calls'] == new['denoiser_calls'] == 9
        assert json.loads(old['sampler'].item())['input_rng_sha256'] == json.loads(new['sampler'].item())['input_rng_sha256']
    assert events == ([('denoise', i) for i in (40, 30, 20, 10, 0)] + [('classify', None)] * 5) * 2 + [
        ('denoise', i) for i in (80, 70, 60, 50, 40, 30, 20, 10, 0)] + [('classify', None)] * 9
