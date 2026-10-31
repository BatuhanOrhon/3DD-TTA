"""Closed-form global guidance contracts, no pretrained model or CUDA."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from legacy_trajectory import rng_hash


class Scheduler:
    def __init__(self):
        self.alphas_cumprod=torch.cumprod(1-torch.linspace(.0001,.02,1000),0)
        self.final_alpha_cumprod=torch.tensor(1.)

    def step(self,noise,t,x):
        alpha=self.alphas_cumprod[t];prev=self.alphas_cumprod[t-10] if t>=10 else self.final_alpha_cumprod
        clean=(x-(1-alpha).sqrt()*noise)/alpha.sqrt()
        return SimpleNamespace(pred_original_sample=clean,prev_sample=prev.sqrt()*clean+(1-prev).sqrt()*noise)


class Prior(torch.nn.Module):
    def __init__(self):super().__init__();self.times=[]
    def forward(self,*,x,t,condition_input,clip_feat):
        self.times.append(t.clone());return torch.zeros_like(x)


def feature():return importlib.import_module('shape_latent_proximal')


def test_three_half_corrections_have_exact_analytic_solution_and_same_noise():
    m=feature();scheduler=Scheduler();prior=Prior().eval();z=torch.ones(2,128,1,1);target=torch.full_like(z,2.)
    before=rng_hash()
    plain,p=m.propose(z,target,'prior3',prior,scheduler,noise_seed=91)
    guided,g=m.propose(z,target,'proxy3',prior,scheduler,noise_seed=91)
    oracle,o=m.propose(z,target,'oracle3',prior,scheduler,noise_seed=91,oracle_target=target)
    assert rng_hash()==before and torch.equal(p['global_noise'],g['global_noise'])
    expected=z+((1-scheduler.alphas_cumprod[20])/scheduler.alphas_cumprod[20]).sqrt()*p['global_noise']
    torch.testing.assert_close(plain,expected)
    torch.testing.assert_close(guided,target+(plain-target)/8)
    assert torch.equal(guided,oracle)
    assert [int(t[0]) for t in prior.times]==[21,11,1]*3
    assert not guided.requires_grad
    for i in range(3):
        torch.testing.assert_close(g['guided_clean_%d'%i],(g['prior_clean_%d'%i]+target)/2)


def test_original_mean_controls_have_no_prior_calls_and_no_projection():
    m=feature();z=torch.ones(2,128,1,1);target=torch.full_like(z,3.);prior=Prior().eval()
    a,_=m.propose(z,target,'original',prior,Scheduler(),noise_seed=1)
    b,_=m.propose(z,target,'mean',prior,Scheduler(),noise_seed=1)
    assert torch.equal(a,z) and torch.equal(b,target) and not prior.times
    assert (b-z).norm()>.05*z.norm()


@pytest.mark.parametrize('arm',['original','mean','prior3','proxy3'])
def test_clean_target_rejected_outside_explicit_oracle_arm(arm):
    x=torch.ones(2,128,1,1)
    with pytest.raises(ValueError):feature().propose(x,x,arm,Prior().eval(),Scheduler(),noise_seed=1,oracle_target=x)


@pytest.mark.parametrize('case',['nan','shape','dtype','train','missing_oracle'])
def test_invalid_model_or_input_rejects(case):
    x=torch.ones(2,128,1,1);target=x.clone();prior=Prior().eval();arm='proxy3'
    if case=='nan':target[0,0]=float('nan')
    if case=='shape':target=target[:,:10]
    if case=='dtype':target=target.double()
    if case=='train':prior.train()
    if case=='missing_oracle':arm='oracle3'
    with pytest.raises(ValueError):feature().propose(x,target,arm,prior,Scheduler(),noise_seed=1)


def test_imported_reference_checks_file_tensors_and_rng_without_consuming_it(tmp_path):
    from scripts.eval_multibackbone_guidance import paired_preparation
    from scripts.eval_multibackbone_source import file_identity
    from legacy_trajectory import _rng_state_hash
    path=tmp_path/'reference.npz';x=torch.ones(2,2048,3)
    _,record=paired_preparation(path,x)
    data=dict(shape_latent=torch.ones(2,128,1,1),local_latent=torch.ones(2,8192,1,1),noise=torch.zeros(2,8192,1,1))
    record(data);boundary=rng_hash()
    state,metadata=feature().load_reference(path,file_identity(path),'cpu')
    assert rng_hash()==boundary==_rng_state_hash(state['rng_after_prepare'])
    assert torch.equal(state['local_latent'],data['local_latent'])
    with pytest.raises(ValueError):feature().load_reference(path,dict(file_identity(path),sha256='bad'),'cpu')
