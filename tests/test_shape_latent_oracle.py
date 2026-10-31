"""Paired oracle preparation contracts; no pretrained model or native kernels."""
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from legacy_trajectory import _rng_state, _rng_state_hash, rng_hash
from models.distributions import Normal


class VAE(torch.nn.Module):
    def encode_global(self, x):
        return Normal(x.mean((1, 2))[:, None].expand(-1, 128).clone(),
                      torch.full((len(x), 128), -.7))

    def global2style(self, z):
        return z

    def encode_local(self, x, style):
        mean = x.mean((1, 2)) + .2 * style.mean(1)
        return Normal(mean[:, None].expand(-1, 8192).clone(),
                      (-1 + .01 * style.mean(1))[:, None].expand(-1, 8192).clone())

    def encode(self, x):
        z, rz = self.encode_global(x).sample()
        h, rh = self.encode_local(x, self.global2style(z)).sample()
        return None, None, [[z], [h]]


def feature():
    return importlib.import_module('shape_latent_oracle')


def inputs():
    return torch.ones(2, 2048, 3), torch.full((2, 2048, 3), .4), VAE().eval()


def test_reference_matches_existing_encoder_draws_and_rng_exactly():
    x, clean, vae = inputs()
    torch.manual_seed(43)
    old = vae.encode(x)
    noise = torch.randn(2, 8192, 1, 1)
    expected_rng = rng_hash()
    torch.manual_seed(43)
    states, proof = feature().prepare_states(x, clean, vae)
    assert torch.equal(states['baseline']['shape_latent'].flatten(1), old[2][0][0])
    assert torch.equal(states['baseline']['local_latent'].flatten(1), old[2][1][0])
    assert torch.equal(states['baseline']['noise'], noise)
    assert rng_hash() == expected_rng
    assert all(_rng_state_hash(s['rng_after_prepare']) == expected_rng for s in states.values())


def test_clean_z_and_local_condition_use_exact_common_posterior_draws():
    x, clean, vae = inputs()
    states, proof = feature().prepare_states(x, clean, vae)
    a, b, c, d = [states[k] for k in feature().ARMS]
    assert torch.equal(a['local_latent'], b['local_latent'])
    assert torch.equal(b['shape_latent'], c['shape_latent'])
    assert torch.equal(c['shape_latent'], d['shape_latent'])
    z = vae.encode_global(clean).sample_given_rho(proof['eps_z'])
    assert torch.equal(z, c['shape_latent'].flatten(1))
    for state, source in [(c, x), (d, clean)]:
        expected = vae.encode_local(source, vae.global2style(z)).sample_given_rho(proof['eps_h'])
        assert torch.equal(expected, state['local_latent'].flatten(1))
    assert not torch.equal(c['local_latent'], d['local_latent'])
    assert all(torch.equal(a['noise'], s['noise']) for s in states.values())


def test_recorded_common_eps_are_the_actual_standard_normal_draws():
    x, clean, vae = inputs()
    torch.manual_seed(91)
    eps_z = torch.empty(2, 128).normal_()
    eps_h = torch.empty(2, 8192).normal_()
    torch.manual_seed(91)
    _, proof = feature().prepare_states(x, clean, vae)
    assert torch.equal(proof['eps_z'], eps_z)
    assert torch.equal(proof['eps_h'], eps_h)


def test_physical_proof_keeps_both_prepared_inputs_for_frame_audit():
    x, clean, vae = inputs()
    _, proof = feature().prepare_states(x, clean, vae)
    assert torch.equal(proof['corrupted_input'], x)
    assert torch.equal(proof['clean_input'], clean)


@pytest.mark.parametrize('bad', ['nan', 'batch', 'shape', 'train'])
def test_invalid_inputs_and_unfrozen_encoder_reject(bad):
    x, clean, vae = inputs()
    if bad == 'nan': clean[0, 0, 0] = float('nan')
    if bad == 'batch': clean = clean[:1]
    if bad == 'shape': clean = clean[:, :12]
    if bad == 'train': vae.train()
    with pytest.raises(ValueError): feature().prepare_states(x, clean, vae)


def test_common_cache_roundtrip_checks_corrupt_clean_and_rng(tmp_path):
    x, clean, vae = inputs()
    path = tmp_path/'common.npz'
    torch.manual_seed(3)
    states, record = feature().paired_states(path, x, clean, vae)
    torch.manual_seed(3)
    loaded, rerecord = feature().paired_states(path, x, clean, vae)
    assert rerecord == record
    for arm in states:
        for key in ['shape_latent', 'local_latent', 'noise']:
            assert torch.equal(states[arm][key], loaded[arm][key])
    torch.manual_seed(3)
    with pytest.raises(ValueError): feature().paired_states(path, x, clean + 1, vae)
    torch.manual_seed(4)
    with pytest.raises(ValueError): feature().paired_states(path, x, clean, vae)


def test_tampered_oracle_tensor_rejected(tmp_path):
    x, clean, vae = inputs()
    torch.manual_seed(3)
    feature().paired_states(tmp_path/'common.npz', x, clean, vae)
    path = tmp_path/'common.oracle.npz'
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays['clean_z'][0, 0] += 1
    with path.open('wb') as stream: np.savez_compressed(stream, **arrays)
    torch.manual_seed(3)
    with pytest.raises(ValueError): feature().paired_states(tmp_path/'common.npz', x, clean, vae)


def test_sampler_keeps_reference_chamfer_target_when_initial_h_changes():
    from unittest.mock import patch
    import test_chamfer_depth as host
    module = host.load_sampler()
    _, _, vae = inputs()
    host.seeded(8)
    states, _ = feature().prepare_states(torch.ones(2, 2048, 3), torch.full((2, 2048, 3), .4), vae)
    target = states['baseline']['local_latent'].view(2, 2048, 4)[:, :, :3].clone()
    seen = []
    def distance(a, b):
        seen.append(b.detach().clone())
        return host.chamfer(a, b)
    with patch.multiple(module, torch=host.CPU(), DDIMScheduler=host.Scheduler,
                        chamfer_grad=lambda: distance, grad_freeze=lambda model: None):
        lion = SimpleNamespace(vae=host.VAE().eval(), priors=[None, host.Prior().eval()])
        module.tta_reconstruct(torch.ones(2, 2048, 3), lion, 2, .01, .01, .95,
                               prepared_state=states['clean_pair'], chamfer_reference=target)
    assert len(seen) == 2 and all(torch.equal(v, target) for v in seen)
