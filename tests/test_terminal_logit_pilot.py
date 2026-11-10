"""Frozen pilot contracts; native/GPU validation is a separate smoke phase."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize('steps', [5, 8, 35])
def test_terminal_sampler_skips_all_intermediate_readouts_and_snapshots(monkeypatch, steps):
    from legacy_trajectory import _rng_state
    from lion_guidance_pilot import sample
    updates, modes, readouts = [], [], []
    class Scheduler:
        def __init__(self, **kw):
            self.alphas_cumprod = torch.linspace(.99, .5, 1000)
            self.final_alpha_cumprod = torch.tensor(1.)
        def set_timesteps(self, count, device):
            self.timesteps = torch.arange(990, -1, -10, device=device)
        def step(self, eps, t, x):
            updates.append(int(t))
            return SimpleNamespace(pred_original_sample=x-.2*eps, prev_sample=x-.1*eps)
    class Chamfer:
        def __call__(self, q, reference):
            d = (q-reference).square().sum(-1)
            i = torch.arange(q.shape[1]).expand(q.shape[:2])
            return d, d, i, i
    monkeypatch.setitem(sys.modules, 'diffusers', SimpleNamespace(DDIMScheduler=Scheduler))
    monkeypatch.setitem(sys.modules, 'third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D',
                        SimpleNamespace(chamfer_3DDist=Chamfer))
    def prior(x, **kw):
        modes.append(torch.is_grad_enabled())
        return .1*x
    lion = SimpleNamespace(vae=SimpleNamespace(global2style=lambda z:z), priors=[None, prior])
    h = torch.arange(16).float().reshape(1,16,1,1)/10
    state = dict(local_latent=h, shape_latent=torch.ones(1,2,1,1),
                 noise=torch.zeros_like(h), rng_after_prepare=_rng_state())
    snapshots, records = sample(lion, state, steps, 'cheap', tau=.1, terminal_only=True,
        callback=lambda h,z,i:readouts.append((i,h)))
    assert len(updates) == len(records) == steps
    assert updates == list(range((steps-1)*10, -1, -10))
    assert modes == [False]*steps
    assert len(snapshots) == len(readouts) == 1 and readouts[0][0] == steps
    assert snapshots[0] is readouts[0][1]
    assert all(not r['full_backward'] for r in records)
    if steps != 5:
        old, _ = sample(lion, state, steps, 'cheap', tau=.1)
        torch.testing.assert_close(old[-1], snapshots[0], rtol=0, atol=0)
        assert len(old) == 4


def test_registration_and_arm_scope_are_locked():
    from scripts.run_terminal_logit_pilot import registration, validate_registration, ARMS
    reg = registration()
    permutation = np.random.default_rng(20261108).permutation(2468).tolist()
    assert reg['smoke_ids'] == sorted(permutation[:4])
    assert reg['pilot_ids'] == sorted(permutation[4:36])
    assert list(ARMS) == ['T5','P5','T8','P8','P8-PM']
    assert [v['steps'] for v in ARMS.values()] == [5,5,8,8,8]
    assert reg['inference_seed'] == 0 and reg['operational_repeats'] == 3
    validate_registration(reg)
    reg['pilot_ids'][0] = reg['smoke_ids'][0]
    with pytest.raises(ValueError):validate_registration(reg)


def test_equal_precision_pm_lm_can_agree_or_disagree_and_reject_bad_scores():
    from scripts.analyze_terminal_logit_pilot import reductions
    scores = np.zeros((2,4,40), dtype=np.float32)
    scores[:,:,2:] = -100
    scores[0,:,0] = 2
    scores[1,:,0] = [10,-2,-2,-2]
    pm, lm = reductions(scores)
    assert pm.dtype == lm.dtype == np.float64
    assert pm.argmax(-1).tolist() == [0,1]
    assert lm.argmax(-1).tolist() == [0,0]
    for bad in (scores[:,:3], np.full_like(scores,np.nan)):
        with pytest.raises(ValueError):reductions(bad)


@pytest.mark.parametrize('arm',['T5','P5','T8','P8','P8-PM'])
def test_host_routes_real_classic_and_preserves_decoder_grad_style(monkeypatch,arm):
    from legacy_trajectory import _rng_state
    from scripts.run_terminal_logit_pilot import infer, ARMS
    import lion_guidance_pilot
    from scripts import eval_multibackbone_guidance as shared
    decoder_calls,head_calls,classic_calls=[],[],[]
    class Decoder(torch.nn.Module):
        def forward(self,unused,beta,context,style):
            decoder_calls.append((torch.is_grad_enabled(),context.requires_grad,style.clone()))
            return context.reshape(len(context),2048,4)[...,:3]
    class Head(torch.nn.Module):
        def forward(self,x):
            head_calls.append(torch.is_grad_enabled())
            return x.mean((1,2))[:,None].expand(-1,40)
    decoder,head=Decoder(),Head()
    decoder.eval();head.eval()
    vae=torch.nn.Module();vae.decoder=decoder;vae.eval()
    lion=SimpleNamespace(vae=vae)
    state=dict(local_latent=torch.ones(2,8192,1,1),shape_latent=torch.full((2,128,1,1),7.),
               noise=torch.zeros(2,8192,1,1),rng_after_prepare=_rng_state())
    cached=(torch.zeros(2,2048,3),None,None,state)
    monkeypatch.setattr(shared,'decoded_to_geometry',lambda decoded,*a:decoded)
    def original(prepared,lion,steps,gamma,eta,p,**kw):
        classic_calls.append((steps,gamma,eta,p,kw))
        return decoder(None,beta=None,context=state['local_latent'].flatten(1).requires_grad_(True),
                       style=state['shape_latent'].flatten(1))
    def sample(lion,state,steps,method,*,tau,terminal_only,callback,scheduler_observer):
        assert steps==ARMS[arm]['steps'] and method=='cheap' and tau==.1
        for loop in ([steps] if terminal_only else [2,4,6,8]):
            callback(state['local_latent'],state['shape_latent'],loop)
        return [state['local_latent']],[]
    monkeypatch.setattr(lion_guidance_pilot,'sample',sample)
    logits,pm=infer(np.zeros((2,2048,3)),arm,lion,original,None,head,[vae,head],cached=cached,device='cpu')
    expected=4 if arm=='P8-PM' else 1
    assert logits.shape==(2,expected,40) and pm.shape==(2,40)
    assert len(decoder_calls)==len(head_calls)==expected
    assert head_calls==[False]*expected
    assert [r[:2] for r in decoder_calls]==([(False,False)]*3+[(True,True)] if expected==4 else [(True,True)])
    assert all(torch.equal(r[2],state['shape_latent'].flatten(1)) for r in decoder_calls)
    if arm.startswith('T'):
        p=ARMS[arm]
        assert classic_calls[0][:4]==(p['steps'],p['gamma'],p['eta'],.95)
        assert set(classic_calls[0][4])=={'prepared_state'}
    else:assert classic_calls==[]


def test_paired_cluster_keeps_corruptions_together_and_counts_changes():
    from scripts.analyze_terminal_logit_pilot import paired
    a=np.zeros((32,15),dtype=int);b=a.copy();b[0,:]=1
    result=paired(a,b)
    assert result['corrected']==result['net']==15 and result['harmed']==0
    assert result['delta_pp']==3.125 and result['ci95_pp'][0]==0
    assert result['ci95_pp'][1]>3.125
    with pytest.raises(ValueError):paired(a,b[:,:14])


def test_source_seal_rejects_changed_repo_code_even_with_same_archive_hash():
    from scripts.run_terminal_logit_pilot import validate_executed_source
    expected=dict(bytes=4,sha256='a'*64)
    manifest=dict(files={'source_snapshot/lion_guidance_pilot.py':expected})
    validate_executed_source({'repo/lion_guidance_pilot.py':expected},manifest)
    with pytest.raises(ValueError):
        validate_executed_source({'repo/lion_guidance_pilot.py':dict(expected,sha256='b'*64)},manifest)


def test_complete_synthetic_grid_recounts_same_logit_lm_and_rejects_changed_ids(tmp_path):
    import json
    from scripts.run_terminal_logit_pilot import registration,ARMS,CANDIDATES
    from scripts.analyze_terminal_logit_pilot import analyze,softmax
    reg=registration()
    def write(path,value):path.write_text(json.dumps(value))
    write(tmp_path/'config.json',dict(phase='pilot',protocol=reg))
    write(tmp_path/'completion.json',dict(status='complete',cases=225))
    write(tmp_path/'environment.json',dict(model_hashes=['frozen']))
    for ci,c in enumerate(reg['corruptions']):
        case=tmp_path/c;case.mkdir()
        integrity=tmp_path/'integrity'/c;integrity.mkdir(parents=True)
        write(integrity/'integrity.json',dict(status='verified',before=['frozen'],after=['frozen']))
        for repeat in range(3):
            for arm,p in ARMS.items():
                count=4 if p['pm'] else 1
                logits=np.zeros((32,count,40),dtype=np.float32);logits[:,:,2:]=-100;logits[:,:,0]=2
                if p['pm']:logits[0,:,0]=[10,-2,-2,-2]
                probs=softmax(logits).mean(1)
                stem='%s_b0_r%d'%(arm,repeat)
                np.savez(case/(stem+'.npz'),logits=logits,probability_mean=probs,prediction=probs.argmax(-1),
                    index=reg['pilot_ids'],label=np.zeros(32,dtype=int),candidates=CANDIDATES if p['pm'] else ['terminal'])
                write(case/(stem+'.json'),dict(seconds=1,peak_mib=1,reserved_mib=2,draw_seed=ci*1000,
                    inference_seed=0,operational_repeat=repeat,ids=reg['pilot_ids']))
    result=analyze(tmp_path,reg)
    assert len(result['per_object'])==8640
    pm=next(r for r in result['summary'] if r['method']=='P8-PM' and r['repeat']==0)
    lm=next(r for r in result['summary'] if r['method']=='P8-LM' and r['repeat']==0)
    assert pm['total_correct']==465 and lm['total_correct']==480
    difference=next(r for r in result['comparisons'] if r['candidate']=='P8-LM' and r['repeat']==0 and r['group']=='all15')
    assert difference['delta_pp']==3.125 and difference['corrected']==15 and difference['harmed']==0
    meta=tmp_path/reg['corruptions'][0]/'T5_b0_r0.json'
    values=json.loads(meta.read_text());values['ids'][0]=-1;write(meta,values)
    with pytest.raises(ValueError):analyze(tmp_path,reg)
