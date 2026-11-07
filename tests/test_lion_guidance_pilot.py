import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'lion_guidance_pilot.py'


def module():
    assert PATH.exists(), 'Missing opt-in pilot kernels'
    spec = importlib.util.spec_from_file_location('lion_guidance_pilot', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_proximal_counts_reverse_edges_and_preserves_unobserved_point():
    m = module()
    q = torch.tensor([[[0., 0., 0.], [9., 0., 0.]]], dtype=torch.float64)
    r = torch.tensor([[[2., 0., 0.], [4., 0., 0.]]], dtype=torch.float64)
    # One outgoing plus two incoming observations on point0; point1 is trimmed.
    result = m.proximal_chamfer(q, r, torch.tensor([[0, 1]]), torch.tensor([[0, 0]]),
                              torch.tensor([[True, False]]), torch.tensor([[True, True]]), .5)
    torch.testing.assert_close(result[0, 0], torch.tensor([1.6, 0., 0.], dtype=q.dtype))
    torch.testing.assert_close(result[0, 1], q[0, 1])
    torch.testing.assert_close(m.proximal_chamfer(q,r,torch.tensor([[0,1]]),torch.tensor([[0,0]]),
        torch.tensor([[True,False]]),torch.tensor([[True,True]]),0.),q)


def test_feature_matching_disambiguates_without_feature_gradient():
    m = module()
    q = torch.tensor([[[.49, 0., 0.]]], requires_grad=True)
    r = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    f = torch.tensor([[[1.]]], requires_grad=True)
    rf = torch.tensor([[[0.], [1.]]])
    plain = m.feature_matches(q,r,f,rf,k=2,beta=0.)
    aware = m.feature_matches(q,r,f,rf,k=2,beta=1.)
    assert plain.tolist()==[[0]] and aware.tolist()==[[1]]
    loss=(q-r.gather(1,aware[...,None].expand(-1,-1,3))).square().sum()
    loss.backward()
    assert q.grad is not None and f.grad is None
    assert m.feature_matches(q,r,f,rf.flip(1),k=2,beta=1.).tolist()==[[0]]


def test_feature_matching_constant_feature_is_finite_and_geometric():
    m=module()
    q=torch.zeros(1,2,3); r=torch.tensor([[[1.,0.,0.],[2.,0.,0.]]])
    f=torch.ones(1,2,1)
    assert m.feature_matches(q,r,f,f,k=2,beta=.1).tolist()==[[0,0]]


def test_causal_teacher_empty_and_detached_history():
    m=module()
    assert m.causal_teacher([]) is None
    a=torch.tensor([[.8,.2]],requires_grad=True);b=torch.tensor([[.2,.8]])
    teacher=m.causal_teacher([a,b])
    assert not teacher.requires_grad
    torch.testing.assert_close(teacher,torch.tensor([[.5,.5]]))


def test_feature_step_freezes_xyz_and_is_bounded():
    m=module()
    h=torch.randn(2,10,4,dtype=torch.float64)
    g=torch.randn_like(h)
    corrected=m.feature_step(h,g,.01)
    torch.testing.assert_close(corrected[...,:3],h[...,:3],rtol=0,atol=0)
    scale=h[...,3:].square().mean((1,2)).sqrt().clamp_min(1e-6)
    norm=(corrected-h).square().sum((1,2)).sqrt()
    assert torch.all(norm<=.01*scale*10**.5+1e-12)
    torch.testing.assert_close(m.feature_step(h,torch.zeros_like(h),.01),h)
    with pytest.raises(ValueError): m.feature_step(h,g,-1.)


def test_bridge_uses_final_alpha_and_leaves_feature_unchanged():
    m=module()
    prev=torch.randn(2,12,1,1,dtype=torch.float64)
    delta=torch.ones(2,3,3,dtype=torch.float64)
    result=m.bridge_clean_delta(prev,delta,.25)
    expected=prev.reshape(2,3,4).clone();expected[...,:3]+=.5
    torch.testing.assert_close(result.reshape(2,3,4),expected)
    final=m.bridge_clean_delta(prev,delta,1.)
    torch.testing.assert_close(final.reshape(2,3,4)[...,:3],prev.reshape(2,3,4)[...,:3]+1)


def test_repair_uses_past_only_and_rejects_geometric_damage():
    m=module()
    latents=[torch.tensor([[[0.,0.,0.,v],[1.,0.,0.,v]]]) for v in (1.,-1.,-.5,0.)]
    def decode(h): return h[...,:3]+h[...,3:]*.1
    def classify(x):
        v=x[...,1].mean(1)*10
        return torch.stack([v,-v],-1)
    diagnostics=[]
    result,records=m.repair_candidates(latents,decode,classify,fraction=.01,geometry_limit=0.,diagnostic=diagnostics.append)
    assert records[0]['teacher_count']==0
    assert [r['teacher_count'] for r in records]==[0,1,2,3]
    expected=torch.stack([classify(decode(h)) for h in latents],1)
    torch.testing.assert_close(result,expected)
    assert all(not any(r['accepted']) for r in records[1:])
    assert all(torch.equal(h[...,:3],latents[0][...,:3]) for h in latents)
    for h,d in zip(latents,diagnostics):
        torch.testing.assert_close(d['chosen_latent'],h)
        torch.testing.assert_close(d['chosen_decoded'],decode(h))
        assert not d['chosen_latent'].requires_grad
    assert any(not torch.equal(d['attempted_latent'],d['latent']) for d in diagnostics[1:])


def test_repair_zero_control_has_identical_before_after_and_repeats():
    m=module()
    latents=[torch.randn(2,5,4,dtype=torch.float64) for _ in range(4)]
    def decode(h):return h[...,:3]+h[...,3:]
    def classify(x):return x.mean(1)
    recorded=[]
    first,_=m.repair_candidates(latents,decode,classify,fraction=0.,diagnostic=recorded.append)
    second,_=m.repair_candidates(latents,decode,classify,fraction=0.)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    for d in recorded:
        for a,b in (('before','after'),('latent','attempted_latent'),('before_logits','after_logits')):
            torch.testing.assert_close(d[a],d[b],rtol=0,atol=0)
