"""Fixed clean-z diagnostic states with original posterior draws and RNG."""
import json
from pathlib import Path

import numpy as np
import torch

from legacy_trajectory import _rng_state, isolated_probe, rng_hash, tensor_hash

ARMS = ('baseline', 'clean_z_fixed_h', 'clean_z_reencode_h', 'clean_pair')


def _states(base, proof):
    states = {}
    for arm in ARMS:
        states[arm] = dict(base)
        if arm != 'baseline':
            states[arm]['shape_latent'] = proof['clean_z'].unsqueeze(2).unsqueeze(3)
        if arm in ('clean_z_reencode_h', 'clean_pair'):
            key = 'h_reencoded' if arm == 'clean_z_reencode_h' else 'h_clean'
            states[arm]['local_latent'] = proof[key].unsqueeze(2).unsqueeze(3)
    return states


@torch.no_grad()
def prepare_states(corrupted: torch.Tensor, clean: torch.Tensor, vae):
    if (corrupted.ndim != 3 or corrupted.shape[1:] != (2048, 3)
            or clean.shape != corrupted.shape or clean.device != corrupted.device
            or clean.dtype != corrupted.dtype or corrupted.dtype != torch.float32
            or not torch.isfinite(corrupted).all() or not torch.isfinite(clean).all()
            or vae.training or any(p.requires_grad for p in vae.parameters())):
        raise ValueError('Oracle requires paired finite float32 Bx2048x3 inputs and frozen eval VAE')
    q_z = vae.encode_global(corrupted)
    # LION's sample() mutates/aliases its returned rho; retain explicit Gaussian draws.
    eps_z = torch.empty_like(q_z.mu).normal_()
    z = q_z.sample_given_rho(eps_z)
    q_h = vae.encode_local(corrupted, vae.global2style(z))
    eps_h = torch.empty_like(q_h.mu).normal_()
    h = q_h.sample_given_rho(eps_h)
    base = dict(shape_latent=z.unsqueeze(2).unsqueeze(3),
                local_latent=h.unsqueeze(2).unsqueeze(3),
                noise=torch.randn_like(h.unsqueeze(2).unsqueeze(3)))
    base['rng_after_prepare'] = _rng_state()
    boundary = rng_hash()
    with isolated_probe([vae]):
        q_clean_z = vae.encode_global(clean)
        clean_z = q_clean_z.sample_given_rho(eps_z)
        style = vae.global2style(clean_z)
        q_reencoded = vae.encode_local(corrupted, style)
        q_clean_h = vae.encode_local(clean, style)
        proof = dict(eps_z=eps_z, eps_h=eps_h, clean_z=clean_z,
                     h_reencoded=q_reencoded.sample_given_rho(eps_h),
                     h_clean=q_clean_h.sample_given_rho(eps_h), clean_input=clean,
                     corrupted_input=corrupted)
        for name, distribution in [('z_corrupt', q_z), ('z_clean', q_clean_z),
                                   ('h_baseline', q_h), ('h_reencoded', q_reencoded),
                                   ('h_clean', q_clean_h)]:
            proof[name+'_mu'] = distribution.mu
            proof[name+'_sigma'] = distribution.sigma
    if rng_hash() != boundary:
        raise RuntimeError('Oracle inference changed the reference post-preparation RNG')
    expected = dict(eps_z=(len(clean), 128), eps_h=(len(clean), 8192),
                    clean_z=(len(clean), 128), h_reencoded=(len(clean), 8192), h_clean=(len(clean), 8192))
    for name, value in proof.items():
        if (not torch.isfinite(value).all() or value.dtype != torch.float32
                or (name in expected and tuple(value.shape) != expected[name])
                or (name.endswith('_sigma') and not (value > 0).all())):
            raise ValueError('Invalid oracle posterior tensor: '+name)
    return _states(base, proof), {k: v.detach().clone() for k, v in proof.items()}


def paired_states(path: Path, corrupted: torch.Tensor, clean: torch.Tensor, vae):
    """Retain a standard reference cache plus complete posterior/noise proof."""
    from scripts.eval_multibackbone_guidance import paired_preparation
    from scripts.eval_multibackbone_source import file_identity
    path = Path(path)
    proof_path = path.with_suffix('.oracle.npz')
    if proof_path.is_symlink():
        raise ValueError('Oracle proof symlinks forbidden')
    base, record = paired_preparation(path, corrupted)
    if base is None:
        if proof_path.exists():
            raise ValueError('Orphan oracle proof; inspect failed phase, never overwrite')
        states, proof = prepare_states(corrupted, clean, vae)
        base = states['baseline']
        record({k: base[k] for k in ('shape_latent', 'local_latent', 'noise')})
        metadata = dict(schema_version=1, input_sha256=tensor_hash(corrupted),
                        clean_input_sha256=tensor_hash(clean), reference_cache=file_identity(path),
                        tensor_hashes={k: tensor_hash(v) for k, v in proof.items()})
        with proof_path.open('xb') as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)),
                                **{k: v.cpu().numpy() for k, v in proof.items()})
    else:
        if not proof_path.is_file():
            raise ValueError('Missing oracle proof; inspect failed phase')
        with np.load(proof_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata'].item()))
            if (metadata.get('schema_version') != 1 or metadata.get('input_sha256') != tensor_hash(corrupted)
                    or metadata.get('clean_input_sha256') != tensor_hash(clean)
                    or metadata.get('reference_cache') != file_identity(path)):
                raise ValueError('Oracle input/reference-cache identity changed')
            proof = {k: torch.from_numpy(saved[k].copy()).to(corrupted.device)
                     for k in saved.files if k != 'metadata'}
            if (set(proof) != set(metadata['tensor_hashes'])
                    or any(tensor_hash(v) != metadata['tensor_hashes'][k] for k, v in proof.items())):
                raise ValueError('Oracle posterior proof tensor changed')
        states = _states(base, proof)
    report = dict(common_state=dict(path=str(path), **file_identity(path)),
                  posterior_proof=dict(path=str(proof_path), **file_identity(proof_path)),
                  clean_input_sha256=tensor_hash(clean), corrupted_input_sha256=tensor_hash(corrupted),
                  global_prior_calls=0)
    return states, report
