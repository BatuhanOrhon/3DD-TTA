"""Posterior refresh contracts; no pretrained model or CUDA."""
import importlib
import json

import pytest
import torch

from legacy_trajectory import rng_hash
from models.distributions import Normal
from test_shape_latent_oracle import VAE


class RefreshVAE(VAE):
    def decoder(self, unused, *, beta, context, style):
        return context.view(len(style), 2048, 4)[:, :, :3] + .01 * style.mean(1)[:, None, None]


def feature():
    return importlib.import_module('shape_latent_refresh')


def test_prepare_matches_original_encoder_noise_and_rng():
    vae = RefreshVAE().eval(); x = torch.ones(2, 2048, 3)
    torch.manual_seed(31); expected = vae.encode(x)
    expected_noise = torch.randn(2, 8192, 1, 1); boundary = rng_hash()
    torch.manual_seed(31); state, proof = feature().prepare_reference(x, vae)
    assert torch.equal(state['shape_latent'].flatten(1), expected[2][0][0])
    assert torch.equal(state['local_latent'].flatten(1), expected[2][1][0])
    assert torch.equal(state['noise'], expected_noise) and rng_hash() == boundary
    torch.manual_seed(31)
    assert torch.equal(proof['eps_z'], torch.empty(2, 128).normal_())
    assert torch.equal(proof['prepared_input'], x)


def test_new_mean_keeps_old_noise_and_full_posterior_uses_same_eps():
    m = feature(); vae = RefreshVAE().eval(); x = torch.ones(2, 2048, 3)
    state, proof = m.prepare_reference(x, vae)
    # Different variance is necessary to detect using the old sigma in both arms.
    def global_encoder(points):
        torch.randn(3)
        return Normal(torch.full((len(points),128), 3.), torch.full((len(points),128), -1.4))
    vae.encode_global = global_encoder
    boundary = rng_hash(); decoded = torch.full_like(x, .2)
    result = m.refresh_proof(decoded, state, proof, vae)
    assert rng_hash() == boundary
    assert torch.equal(result['mean_z'], result['new_mu'] + proof['sigma'] * proof['eps_z'])
    assert torch.equal(result['posterior_z'], result['new_mu'] + result['new_sigma'] * proof['eps_z'])
    assert not torch.equal(result['mean_z'], result['posterior_z'])
    assert torch.equal(result['decoded_points'], decoded)
    assert 'ae_mu' in result and 'ae_points' in result


def test_same_sigma_gives_identical_candidates_and_preserves_buffers():
    m = feature(); vae = RefreshVAE().eval(); x = torch.ones(2, 2048, 3)
    state, proof = m.prepare_reference(x, vae)
    vae.register_buffer('counter', torch.zeros(()))
    def global_encoder(points):
        vae.counter.add_(1)
        return Normal(torch.full((len(points),128), 3.), proof['sigma'].log())
    vae.encode_global = global_encoder
    result=m.refresh_proof(x, state, proof, vae)
    assert torch.equal(result['mean_z'], result['posterior_z'])
    assert vae.counter.item() == 0


@pytest.mark.parametrize('case', ['nan','shape','train','grad'])
def test_invalid_reference_rejects(case):
    x=torch.ones(2,2048,3);vae=RefreshVAE().eval()
    if case=='nan':x[0,0,0]=float('nan')
    if case=='shape':x=x[:,:10]
    if case=='train':vae.train()
    if case=='grad':vae.register_parameter('weight',torch.nn.Parameter(torch.ones(())))
    with pytest.raises(ValueError):feature().prepare_reference(x,vae)


def test_physical_proof_roundtrip_binding_and_no_overwrite(tmp_path):
    m=feature();path=tmp_path/'proof.npz';proof={'z':torch.ones(2,128)};binding={'input':'a','source':'b'}
    m.save_proof(path,proof,binding)
    assert torch.equal(m.load_proof(path,'cpu',binding)['z'],proof['z'])
    with pytest.raises(FileExistsError):m.save_proof(path,proof,binding)
    with pytest.raises(ValueError):m.load_proof(path,'cpu',dict(binding,source='changed'))
    import numpy as np
    with np.load(path,allow_pickle=False) as saved:fields={k:saved[k].copy() for k in saved.files}
    fields['z'][0,0]=4
    with path.open('wb') as stream:np.savez_compressed(stream,**fields)
    with pytest.raises(ValueError):m.load_proof(path,'cpu',binding)
