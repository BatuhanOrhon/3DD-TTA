"""CPU contracts for bounded global proposals; no native/model inference."""
import importlib.util
from pathlib import Path
import types
from unittest.mock import patch

import pytest
import torch

from legacy_trajectory import rng_hash
from test_chamfer_depth import CPU, Prior, Scheduler, VAE, chamfer, load_sampler, seeded


def feature():
    # Missing implementation must fail as an assertion before production edits.
    assert importlib.util.find_spec('shape_latent_guidance') is not None, 'bounded shape feature is missing'
    import shape_latent_guidance
    return shape_latent_guidance


class GlobalPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2), requires_grad=False)
        self.calls = []

    def forward(self, x, t, **kwargs):
        self.calls.append(t.detach().clone())
        return self.weight * x


class IdentityVAE(VAE):
    def global2style(self, style):
        return style


def runtime():
    return types.SimpleNamespace(vae=IdentityVAE().eval(),
                                 priors=[GlobalPrior().eval(), Prior().eval()])


def scheduler():
    result = Scheduler()
    result.set_timesteps(100, 'cpu')
    result.alphas_cumprod = torch.cumprod(1-torch.linspace(.0001, .02, 1000), 0)
    # Same deterministic DDIM clean estimate and transition as the native protocol.
    def step(eps, t, x):
        a = result.alphas_cumprod[t]
        previous = result.alphas_cumprod[t-10] if t >= 10 else torch.tensor(1.)
        clean = (x-(1-a).sqrt()*eps)/a.sqrt()
        return types.SimpleNamespace(pred_original_sample=clean,
                                     prev_sample=previous.sqrt()*clean+(1-previous).sqrt()*eps)
    result.step = step
    return result


def test_projection_is_per_object_bounded_and_zero_radius_freezes_style():
    project = feature().project_to_anchor
    anchor = torch.tensor([[1., 0.], [100., 0.]]).reshape(2, 2, 1, 1)
    value = anchor + 20
    actual = project(value, anchor, .05)
    distance = (actual-anchor).flatten(1).norm(dim=1)
    torch.testing.assert_close(distance, torch.tensor([.05, 5.]), rtol=2e-5, atol=1e-6)
    assert torch.equal(project(value, anchor, 0), anchor)
    assert torch.equal(project(anchor, anchor, .05), anchor)
    for radius in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError): project(value, anchor, radius)
    with pytest.raises(ValueError): project(value*float('nan'), anchor, .05)
    with pytest.raises(ValueError): project(value[:1], anchor, .05)


@pytest.mark.parametrize('arm,count', [('updated', 0), ('anchor', 0), ('proposal', 1), ('guided3', 3)])
def test_global_calls_time_contract_clean_output_and_rng_isolation(arm, count):
    cls = feature().ShapeLatentGuidance
    lion, sched = runtime(), scheduler()
    anchor = torch.ones(2, 128, 1, 1)
    adapter = cls(arm, noise_seed=19)
    before = rng_hash()
    actual = adapter.initialize(anchor, sched, lion)
    assert rng_hash() == before
    assert torch.equal(anchor, torch.ones_like(anchor))
    assert len(lion.priors[0].calls) == count
    assert [int(v[0]) for v in lion.priors[0].calls] == {'updated': [], 'anchor': [], 'proposal': [21], 'guided3': [21, 11, 1]}[arm]
    if arm != 'updated':
        assert torch.all((actual-anchor).flatten(1).norm(dim=1) <= .05*anchor.flatten(1).norm(dim=1)+1e-5)
    report = adapter.diagnostics()
    assert report['global_forward_calls'] == count
    assert report['global_backward_calls'] == (3 if arm == 'guided3' else 0)
    assert report['global_rng_unchanged']
    assert all(p.grad is None for p in lion.priors[0].parameters())


def test_proposal_is_a_clean_prediction_at_t20_and_global_noise_is_shared():
    cls = feature().ShapeLatentGuidance
    anchor = torch.ones(2, 128, 1, 1)
    lion, sched = runtime(), scheduler()
    adapter = cls('proposal', noise_seed=13)
    actual = adapter.initialize(anchor, sched, lion)
    generator = torch.Generator().manual_seed(13)
    noise = torch.randn(anchor.shape, generator=generator)
    a = sched.alphas_cumprod[20]
    noisy = a.sqrt()*anchor+(1-a).sqrt()*noise
    clean = sched.step(.2*noisy, 20, noisy).pred_original_sample
    expected = feature().project_to_anchor(clean, anchor, .05)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, feature().project_to_anchor(noisy, anchor, .05))
    other = cls('guided3', noise_seed=13)
    other.initialize(anchor, scheduler(), runtime())
    assert adapter.diagnostics()['global_noise_sha256'] == other.diagnostics()['global_noise_sha256']


def test_guidance_differentiates_clean_estimate_through_the_global_prior():
    cls = feature().ShapeLatentGuidance
    anchor = torch.ones(2, 128, 1, 1)
    sched = scheduler()
    adapter = cls('guided3', noise_seed=13)
    adapter.initialize(anchor, sched, runtime())
    a = sched.alphas_cumprod[20]
    noise = torch.randn(anchor.shape, generator=torch.Generator().manual_seed(13))
    noisy = a.sqrt()*anchor+(1-a).sqrt()*noise
    jacobian = (1-(1-a).sqrt()*.2)/a.sqrt()
    expected_gradient = (2/128)*(jacobian*noisy-anchor)*jacobian
    actual_norms = adapter.diagnostics()['global_steps'][0]['anchor_gradient_norms']
    torch.testing.assert_close(torch.tensor(actual_norms), expected_gradient.flatten(1).norm(dim=1))
    assert adapter.diagnostics()['global_steps'][0]['timestep'] == 20


@pytest.mark.parametrize('weight', [.2, 20.])
def test_three_guided_transitions_match_independent_signed_and_capped_oracle(weight):
    cls = feature().ShapeLatentGuidance
    anchor = torch.stack([torch.ones(128), torch.linspace(.1, 3., 128)]).reshape(2, 128, 1, 1)
    lion, sched = runtime(), scheduler()
    lion.priors[0].weight.fill_(weight)
    adapter = cls('guided3', noise_seed=17)
    actual = adapter.initialize(anchor, sched, lion)
    noise = torch.randn(anchor.shape, generator=torch.Generator().manual_seed(17))
    a = sched.alphas_cumprod[20]
    initial = a.sqrt()*anchor+(1-a).sqrt()*noise
    expected_norms, active = [], False
    def oracle(sign):
        nonlocal active
        current = initial.clone()
        for t in [20, 10, 0]:
            alpha = sched.alphas_cumprod[t]
            previous = sched.alphas_cumprod[t-10] if t >= 10 else torch.tensor(1.)
            jacobian = (1-(1-alpha).sqrt()*weight)/alpha.sqrt()
            clean = jacobian*current
            grad = (2/128)*(clean-anchor)*jacobian
            grad_norm = grad.flatten(1).norm(dim=1)
            cap = .01*anchor.flatten(1).norm(dim=1)
            factor = (cap/grad_norm.clamp_min(1e-30)).clamp(max=1)
            correction = grad*factor.reshape(-1, 1, 1, 1)
            if sign == -1:
                expected_norms.append(correction.flatten(1).norm(dim=1))
                active |= bool(torch.any(grad_norm > cap))
            current = previous.sqrt()*clean+(1-previous).sqrt()*weight*current+sign*correction
        delta = current-anchor
        fraction = (.05*anchor.flatten(1).norm(dim=1)/delta.flatten(1).norm(dim=1)).clamp(max=1)
        return anchor+delta*fraction.reshape(-1, 1, 1, 1)
    expected, omitted, reversed_sign = oracle(-1), oracle(0), oracle(1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)
    assert not torch.allclose(actual, omitted, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(actual, reversed_sign, rtol=1e-6, atol=1e-7)
    for record, norms in zip(adapter.diagnostics()['global_steps'], expected_norms):
        torch.testing.assert_close(torch.tensor(record['correction_norms']), norms)
    assert active == (weight == 20.)


def test_rejects_nonidentity_style_training_mode_and_unfrozen_global_prior():
    cls = feature().ShapeLatentGuidance
    anchor = torch.ones(2, 128, 1, 1)
    lion = runtime()
    lion.vae = VAE().eval()
    with pytest.raises(ValueError, match='identity'):
        cls('proposal', noise_seed=1).initialize(anchor, scheduler(), lion)
    for change in ('train', 'weights'):
        lion = runtime()
        if change == 'train': lion.priors[0].train()
        else: lion.priors[0].weight.requires_grad_(True)
        with pytest.raises(ValueError):
            cls('guided3', noise_seed=1).initialize(anchor, scheduler(), lion)


def test_adapter_is_applied_to_every_style_update_and_clean_final_decode():
    cls = feature().ShapeLatentGuidance
    module, lion = load_sampler(), runtime()
    observer = types.SimpleNamespace(mark=lambda *a: None, initial=lambda **k: None,
                                     styles=[], final_styles=[])
    observer.final_state = lambda local, style, *a: observer.final_styles.append(style.detach().clone())
    observer.clean = lambda local, style, *args: observer.styles.append(style.detach().clone())
    prepared = dict(shape_latent=torch.ones(2, 128, 1, 1),
                    local_latent=torch.ones(2, 8192, 1, 1), noise=torch.ones(2, 8192, 1, 1))
    from legacy_trajectory import _rng_state
    prepared['rng_after_prepare'] = _rng_state()
    adapter = cls('anchor', noise_seed=1)
    with patch.multiple(module, torch=CPU(), DDIMScheduler=Scheduler,
                        chamfer_grad=lambda: chamfer, grad_freeze=lambda m: None):
        trajectory = module.tta_reconstruct(torch.ones(2, 2048, 3), lion, 8, .03, 10., .95,
            prepared_state=prepared, trajectory_observer=observer,
            shape_guidance=adapter, return_trajectory=True)
    anchor = prepared['shape_latent']
    assert len(observer.styles) == 8
    for style in observer.styles+[trajectory.updated_style]:
        assert torch.all((style-anchor).flatten(1).norm(dim=1) <= .05*anchor.flatten(1).norm(dim=1)+1e-5)
    assert not torch.equal(trajectory.updated_style, anchor)
    assert not torch.equal(observer.styles[-1], anchor)
    torch.testing.assert_close(observer.final_styles[0], trajectory.updated_style)

