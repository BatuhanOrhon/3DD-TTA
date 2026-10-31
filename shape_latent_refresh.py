"""Frozen global posterior refresh from one shared local-denoised observation."""
import json
from pathlib import Path

import numpy as np
import torch

from legacy_trajectory import _rng_state, isolated_probe, tensor_hash

ARMS = ('baseline', 'mean_refresh', 'posterior_refresh')


def validate_input(x, vae):
    if (x.ndim != 3 or x.shape[1:] != (2048, 3) or x.dtype != torch.float32
            or not torch.isfinite(x).all() or any(m.training for m in vae.modules())
            or any(p.requires_grad for p in vae.parameters())):
        raise ValueError('Refresh requires finite float32 Bx2048x3 and frozen eval VAE')


@torch.no_grad()
def prepare_reference(x, vae):
    validate_input(x, vae)
    q_z = vae.encode_global(x)
    # Normal.sample() aliases its returned rho; keep the actual Gaussian explicitly.
    eps_z = torch.empty_like(q_z.mu).normal_()
    z = q_z.sample_given_rho(eps_z)
    q_h = vae.encode_local(x, vae.global2style(z))
    eps_h = torch.empty_like(q_h.mu).normal_()
    h = q_h.sample_given_rho(eps_h)
    base = dict(shape_latent=z[:, :, None, None], local_latent=h[:, :, None, None],
                noise=torch.randn_like(h[:, :, None, None]), rng_after_prepare=_rng_state())
    proof = dict(mu=q_z.mu, sigma=q_z.sigma, eps_z=eps_z, eps_h=eps_h,
                 h_mu=q_h.mu, h_sigma=q_h.sigma, prepared_input=x)
    return base, {k: v.detach().clone() for k, v in proof.items()}


@torch.no_grad()
def refresh_proof(decoded, base, initial, vae):
    validate_input(decoded, vae)
    with isolated_probe([vae]):
        new = vae.encode_global(decoded)
        ae = vae.decoder(None, beta=None, context=base['local_latent'].flatten(1),
                         style=base['shape_latent'].flatten(1))
        ae_q = vae.encode_global(ae)
        proof = dict(initial, decoded_points=decoded, new_mu=new.mu, new_sigma=new.sigma,
                     mean_z=new.mu + initial['sigma'] * initial['eps_z'],
                     posterior_z=new.sample_given_rho(initial['eps_z']),
                     ae_points=ae, ae_mu=ae_q.mu, ae_sigma=ae_q.sigma)
        expected = dict(prepared_input=(len(decoded),2048,3), decoded_points=(len(decoded),2048,3),
                        ae_points=(len(decoded),2048,3))
        for key, value in proof.items():
            shape = expected.get(key, (len(decoded),8192 if key in ('h_mu','h_sigma','eps_h') else 128))
            if (tuple(value.shape) != shape or value.dtype != torch.float32
                    or not torch.isfinite(value).all()
                    or ('sigma' in key and not (value > 0).all())):
                raise ValueError('Invalid physical refresh tensor: '+key)
        return {k: v.detach().clone() for k, v in proof.items()}


def save_proof(path, proof, binding):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Proof symlinks forbidden')
    metadata = dict(schema_version=1, binding=binding,
                    tensor_hashes={k: tensor_hash(v) for k,v in proof.items()})
    with path.open('xb') as stream:
        np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)),
                            **{k:v.detach().cpu().numpy() for k,v in proof.items()})


def load_proof(path, device, binding):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Proof symlinks forbidden')
    with np.load(path, allow_pickle=False) as saved:
        metadata=json.loads(saved['metadata'].item())
        proof={k:torch.from_numpy(saved[k].copy()).to(device) for k in saved.files if k != 'metadata'}
    if (metadata.get('schema_version') != 1 or metadata.get('binding') != binding
            or set(proof) != set(metadata.get('tensor_hashes',{}))
            or any(v.dtype != torch.float32 or not torch.isfinite(v).all()
                   or tensor_hash(v) != metadata['tensor_hashes'][k] for k,v in proof.items())):
        raise ValueError('Refresh proof binding/tensors changed')
    return proof
