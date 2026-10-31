"""Fixed three-step global DDIM with a quadratic clean-estimate correction."""
import json
from pathlib import Path

import numpy as np
import torch

from legacy_trajectory import _rng_state_hash, rng_hash, tensor_hash

ARMS=('original','mean','prior3','proxy3','oracle3')


@torch.no_grad()
def propose(original,target,arm,prior,scheduler,*,noise_seed,oracle_target=None):
    if (arm not in ARMS or original.ndim!=4 or original.shape[1:]!=(128,1,1)
            or original.dtype!=torch.float32 or target.shape!=original.shape
            or target.dtype!=original.dtype or target.device!=original.device
            or not torch.isfinite(original).all() or not torch.isfinite(target).all()
            or type(noise_seed) is not int or not 0<=noise_seed<2**63
            or prior.training or any(p.requires_grad for p in prior.parameters())
            or getattr(prior,'mixed_prediction',False)):
        raise ValueError('Expected fixed arm, finite float32 global tensors and frozen eval prior')
    if (arm=='oracle3')!=(oracle_target is not None):
        raise ValueError('Only explicit oracle3 accepts/requires a clean target')
    if oracle_target is not None and (oracle_target.shape!=original.shape
            or oracle_target.dtype!=original.dtype or oracle_target.device!=original.device
            or not torch.isfinite(oracle_target).all()):
        raise ValueError('Invalid oracle target')
    if arm in ('original','mean'):
        value=original if arm=='original' else target
        return value.detach().clone(),dict(final_z=value.detach().clone())
    if len(scheduler.alphas_cumprod)!=1000 or float(scheduler.final_alpha_cumprod)!=1.:
        raise ValueError('Global DDIM requires1000 training steps and final alpha1')
    before=rng_hash()
    generator=torch.Generator(device=original.device).manual_seed(noise_seed)
    noise=torch.randn(original.shape,dtype=original.dtype,device=original.device,generator=generator)
    alpha=scheduler.alphas_cumprod[20].to(original)
    noisy=alpha.sqrt()*original+(1-alpha).sqrt()*noise
    proof=dict(global_noise=noise.detach().clone())
    guide=oracle_target if arm=='oracle3' else target
    for i,t in enumerate((20,10,0)):
        prediction=prior(x=noisy,t=original.new_full((len(original),),t+1),
                         condition_input=None,clip_feat=None)
        output=scheduler.step(prediction,t,noisy)
        clean=output.pred_original_sample
        guided=clean if arm=='prior3' else (clean+guide)/2
        previous_alpha=(scheduler.alphas_cumprod[t-10] if t>=10
                        else scheduler.final_alpha_cumprod).to(original)
        previous=output.prev_sample+previous_alpha.sqrt()*(guided-clean)
        for name,value in [('noisy_before',noisy),('prior_clean',clean),('guided_clean',guided),('noisy_after',previous)]:
            if not torch.isfinite(value).all():raise ValueError('Nonfinite global transition: '+name)
            proof[name+'_%d'%i]=value.detach().clone()
        noisy=previous
    if rng_hash()!=before:raise RuntimeError('Global proposal advanced common RNG')
    proof['final_z']=noisy.detach().clone()
    return noisy.detach().clone(),proof


def load_reference(path,expected,device):
    """Read an immutable prior-study state; replay its explicit postprepare boundary."""
    from scripts.eval_multibackbone_source import verify_identity
    path=Path(path)
    if path.is_symlink():raise ValueError('Reference symlinks forbidden')
    verify_identity(path,expected)
    with np.load(path,allow_pickle=False) as saved:
        meta=json.loads(saved['metadata'].item())
        state={k:torch.from_numpy(saved[k].copy()).to(device) for k in ('shape_latent','local_latent','noise')}
        py,nr=meta['python_rng'],meta['numpy_rng']
        rng=((py[0],tuple(py[1]),py[2]),(nr[0],saved['numpy_rng_keys'].copy(),*nr[1:]),
             torch.from_numpy(saved['torch_rng_cpu'].copy()),
             [torch.from_numpy(saved['torch_rng_cuda_%d'%i].copy()) for i in range(meta['cuda_devices'])])
    count=len(state['shape_latent'])
    if meta.get('schema_version')!=1 or _rng_state_hash(rng)!=meta['rng_after_sha256']:
        raise ValueError('Imported reference schema/RNG differs')
    for k,width in [('shape_latent',128),('local_latent',8192),('noise',8192)]:
        value=state[k]
        if (value.shape!=(count,width,1,1) or value.dtype!=torch.float32
                or not torch.isfinite(value).all() or tensor_hash(value)!=meta['tensor_hashes'][k]):
            raise ValueError('Imported reference tensor differs: '+k)
    state['rng_after_prepare']=rng
    return state,meta
