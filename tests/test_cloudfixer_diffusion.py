"""Physical sampler contracts; fixtures do not claim author-model accuracy."""
import sys

import pytest
import torch


def implementation():
    from cloudfixer_diffusion import sample, correct, grid
    return sample, correct, grid


class ConstantEpsilon:
    T = 500

    def __init__(self):
        self.calls = []

    def gamma(self, t):
        return t

    def alpha(self, g, target_tensor):
        return torch.full((len(g), 1, 1), .8, dtype=target_tensor.dtype)

    def sigma(self, g, target_tensor):
        return torch.full((len(g), 1, 1), .6, dtype=target_tensor.dtype)

    def phi(self, x, t, node_mask):
        assert not torch.is_grad_enabled()
        self.calls.append(int(round(t[0, 0].item() * 500)))
        return torch.full_like(x, 2.)


def test_zero_tau_never_matches_and_terminal_removes_residual_noise(monkeypatch):
    sample, _, _ = implementation()
    import cloudfixer_diffusion as module
    monkeypatch.setattr(module, 'correct', lambda *a, **k: pytest.fail('tau0 constructed Chamfer'))
    model = ConstantEpsilon()
    x = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], requires_grad=True)
    candidates, metadata = sample(model, x, torch.ones(1, 2), torch.full_like(x, 2.),
                                  steps=4, start=40, tau=0)
    assert model.calls == [40, 30, 20, 10, 0]
    assert candidates.shape == (1, 4, 2, 3)
    assert torch.allclose(candidates, x[:, None].expand_as(candidates), atol=2e-6)
    assert not candidates.requires_grad and x.grad is None
    assert metadata['denoiser_calls'] == 5
    assert metadata['candidate_loops'] == [1, 2, 3, 4]
    assert 'models.lion' not in sys.modules


def test_reference_duplicates_do_not_multiply_proximal_weight():
    _, correct, _ = implementation()
    q = torch.zeros(1, 1, 3)
    reference = torch.tensor([[[2., 0, 0], [2., 0, 0], [100., 0, 0]]])
    value = correct(q, reference, torch.tensor([[True, False, False]]), tau=1., retained=1.)
    assert torch.allclose(value, torch.tensor([[[4/3, 0., 0.]]]))


def test_sampling_positive_tau_changes_clean_state_without_any_backward():
    sample, _, _ = implementation()
    model = ConstantEpsilon()
    x = torch.tensor([[[2., 0., 0.], [4., 0., 0.]]])
    candidates, metadata = sample(model, x, torch.ones(1, 2), torch.zeros_like(x),
                                  steps=4, start=40, tau=.1, retained=1.)
    assert torch.isfinite(candidates).all() and metadata['denoiser_calls'] == 5
    assert len(model.calls) == 5


@pytest.mark.parametrize(('steps', 'loops', 'indices'), [
    (6, [1, 3, 4, 6], [80, 53, 40, 0]),
    (7, [1, 3, 5, 7], [80, 57, 34, 0]),
    (8, [2, 4, 6, 8], [70, 50, 30, 0]),
    (9, [2, 4, 6, 9], [71, 53, 36, 0]),
    (10, [2, 5, 7, 10], [72, 48, 32, 0]),
    (11, [2, 5, 8, 11], [73, 51, 29, 0]),
    (12, [3, 6, 9, 12], [67, 47, 27, 0]),
])
@pytest.mark.parametrize('tau', [0., .1])
def test_dense_trace_preserves_sparse_clean_estimates_transport_and_rng(steps, loops, indices, tau, monkeypatch):
    sample, _, _ = implementation()
    if not tau:
        import cloudfixer_diffusion
        monkeypatch.setattr(cloudfixer_diffusion, 'correct',
                            lambda *a, **k: pytest.fail('tau0 constructed Chamfer'))

    class StatefulEpsilon(ConstantEpsilon):
        def __init__(self):
            super().__init__()
            self.states = []

        def phi(self, x, t, node_mask):
            super().phi(x, t, node_mask)
            self.states.append(x.clone())
            return x.sin() + t[:, :, None] + torch.rand_like(x) * .01

    x = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], dtype=torch.float64)
    mask, noise = torch.tensor([[1., 0.]]), torch.full_like(x, .5)
    sparse_model, dense_model = StatefulEpsilon(), StatefulEpsilon()
    original_rng = torch.get_rng_state()
    try:
        sparse, sparse_meta = sample(sparse_model, x, mask, noise, steps=steps, start=80,
                                     tau=tau, retained=1.)
        sparse_rng = torch.get_rng_state()
        torch.set_rng_state(original_rng)
        dense, dense_meta = sample(dense_model, x, mask, noise, steps=steps, start=80,
                                   tau=tau, retained=1., trace_all=True)
        assert torch.equal(sparse_rng, torch.get_rng_state())
    finally:
        torch.set_rng_state(original_rng)
    assert sparse.shape == (1, 4, 2, 3) and dense.shape == (1, steps + 1, 2, 3)
    assert sparse_meta['candidate_loops'] == loops
    assert sparse_meta['candidate_indices'] == indices
    assert dense_meta['candidate_loops'] == list(range(1, steps + 1)) + [steps]
    assert dense_meta['candidate_indices'] == dense_meta['indices'][:-1] + [0]
    assert torch.equal(sparse, dense[:, [i - 1 for i in loops[:-1]] + [steps]])
    assert sparse_model.calls == dense_model.calls == dense_meta['indices']
    assert sparse_meta['denoiser_calls'] == dense_meta['denoiser_calls'] == steps + 1
    assert all(torch.equal(a, b) for a, b in zip(sparse_model.states, dense_model.states))
    assert torch.equal(sparse[:, -1], dense[:, -1])


def test_dense_candidates_are_clean_estimates_before_each_proximal_correction():
    sample, _, _ = implementation()
    x = torch.zeros(1, 1, 3, dtype=torch.float64)
    candidates, _ = sample(ConstantEpsilon(), x, torch.ones(1, 1), torch.ones_like(x),
                            steps=4, start=40, tau=1., retained=1., trace_all=True)
    # Each one-point correction divides by three; t0 removes the residual epsilon.
    expected = torch.tensor([-.75, -.25, -1/12, -1/36, -1/108], dtype=x.dtype)
    assert torch.allclose(candidates[0, :, 0, 0], expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize('kwargs', [dict(steps=3, start=40), dict(steps=32, start=20),
    dict(steps=8, start=501), dict(steps=True, start=40)])
def test_invalid_or_duplicate_timestep_grids_are_rejected(kwargs):
    _, _, grid = implementation()
    with pytest.raises(ValueError):
        grid(**kwargs)


def test_grid_preserves_training_indices_instead_of_changing_trained_schedule():
    _, _, grid = implementation()
    assert grid(steps=4, start=40) == [40, 30, 20, 10, 0]
    assert grid(steps=8, start=40) == [40, 35, 30, 25, 20, 15, 10, 5, 0]


@pytest.mark.parametrize('mask', [torch.zeros(1, 2), torch.tensor([[1., float('nan')]])])
def test_empty_or_nonfinite_reference_mask_is_rejected(mask):
    sample, _, _ = implementation()
    x = torch.ones(1, 2, 3)
    with pytest.raises(ValueError):
        sample(ConstantEpsilon(), x, mask, x, steps=4, start=40, tau=.1)


def test_varying_polynomial_transport_matches_frozen_quadratic_witness():
    sample,_,_=implementation()
    class Polynomial(ConstantEpsilon):
        def alpha(self,g,target_tensor):
            return (((1-g.square()).square()*(1-2e-5)+1e-5).sqrt())[:,:,None]
        def sigma(self,g,target_tensor):
            return ((1-((1-g.square()).square()*(1-2e-5)+1e-5)).sqrt())[:,:,None]
        def phi(self,x,t,node_mask):
            return torch.zeros_like(x)
    x=torch.zeros(1,1,3,dtype=torch.float64)
    candidates,metadata=sample(Polynomial(),x,torch.ones(1,1),torch.ones_like(x),
        steps=4,start=40,tau=1,retained=1)
    # alpha40²=.9872312151808; four q/(1+2*tau) solves; epsilon remains zero.
    assert torch.allclose(candidates[:,-1],torch.full_like(x,.0014040427323258984),atol=1e-14,rtol=0)
    assert metadata['valid_reference_counts']==[1]
    assert metadata['retained_forward_count']==1
    assert metadata['retained_reference_counts']==[1]


def test_incompatible_checkpoint_gamma_table_is_rejected():
    from cloudfixer_diffusion import verify_gamma
    intended=torch.zeros(501)
    changed=intended.clone();changed[40]=1
    with pytest.raises(ValueError):verify_gamma(changed,intended)
    verify_gamma(intended.clone(),intended)


def test_author_ema_gamma_is_accepted_without_accepting_other_schedule_changes():
    from cloudfixer_diffusion import verify_gamma
    native=torch.zeros(501)
    native[34]=-4.675320148468018
    native[468]=4.160384178161621
    released=native.clone()
    released[34]=-4.674612522125244
    released[468]=4.1597137451171875
    proof=verify_gamma(released,native)
    assert proof['mode']=='author_ema_0.9999_fixed_point'
    assert proof['ema_iterations']>0
    changed=released.clone();changed[34]+=.002
    with pytest.raises(ValueError):verify_gamma(changed,native)
