"""Minimal synthetic gradient/transport/scope contracts, not native validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

REPO = Path('/Users/batuhan/.codex/worktrees/7bf0/3DD-TTA')
sys.path[:0] = [str(REPO), str(REPO/'tests'), str(Path(__file__).parent)]
import shape_latent_conditional as module


class Local(torch.nn.Module):
    def forward(self, x, t, condition_input, **kwargs):
        return .2*x + .4*condition_input.repeat(1, 64, 1, 1)


class Global(torch.nn.Module):
    def forward(self, x, **kwargs):
        return .05*x


class Scheduler:
    alphas_cumprod = torch.linspace(.9999, .001, 1000)
    def step(self, pred, t, x):
        a = self.alphas_cumprod[t]
        clean = (x-(1-a).sqrt()*pred)/a.sqrt()
        prev_a = self.alphas_cumprod[t-10] if t >= 10 else x.new_tensor(1.)
        return SimpleNamespace(pred_original_sample=clean,
                               prev_sample=prev_a.sqrt()*clean+(1-prev_a).sqrt()*pred)


def main():
    local, prior, scheduler = Local().eval(), Global().eval(), Scheduler()
    z = torch.linspace(-1, 1, 256).reshape(2, 128, 1, 1)
    h = torch.linspace(-.5, .5, 16384).reshape(2, 8192, 1, 1)
    rng = torch.get_rng_state().clone()
    bank = module.noise_bank(h, 17)
    values, grad = module.energy(local, z, h, scheduler.alphas_cumprod, bank, gradient=True)
    leaf = z.clone().requires_grad_()
    losses = []
    for t, eps in zip(module.LOCAL_TIMES, bank):
        a = scheduler.alphas_cumprod[t]
        pred = local(a.sqrt()*h+(1-a).sqrt()*eps, None, leaf)
        losses.append((pred-eps).flatten(1).square().mean(1))
    expected = torch.autograd.grad(torch.stack(losses).mean(0).sum(), leaf)[0]
    assert torch.allclose(grad, expected, atol=1e-8, rtol=1e-6)
    _, one = module.energy(local, z[:1], h[:1], scheduler.alphas_cumprod, bank[:, :1], gradient=True)
    assert torch.allclose(one, grad[:1], atol=1e-8, rtol=1e-6)
    direction = grad/grad.norm()
    plus, _ = module.energy(local, z+.01*direction, h, scheduler.alphas_cumprod, bank)
    minus, _ = module.energy(local, z-.01*direction, h, scheduler.alphas_cumprod, bank)
    fd = (plus.mean(1).sum()-minus.mean(1).sum())/.02
    assert torch.allclose(fd, (grad*direction).sum(), atol=2e-5, rtol=.01)
    for mode in ('none', 'prior3', 'guided3'):
        result, proof = module.propose(z, z, h, prior, local, scheduler, mode, 11)
        assert torch.isfinite(result).all() and not result.requires_grad
        if mode == 'none':
            assert torch.equal(z, result)
        for i in range(3 if mode != 'none' else 0):
            assert torch.equal(proof['noisy_after_%d'%i],
                               proof['ddim_previous_%d'%i]-proof['guidance_update_%d'%i])
            if mode == 'guided3':
                assert torch.allclose(proof['guidance_update_%d'%i].flatten(1).norm(dim=1),
                                      .02*z.flatten(1).norm(dim=1), atol=1e-6)
                # Exact dense clean-estimate derivative of this synthetic global prior.
                t = module.GLOBAL_TIMES[i];a = scheduler.alphas_cumprod[t]
                jac = (1-.05*(1-a).sqrt())/a.sqrt()
                assert torch.allclose(proof['grad_noisy_%d'%i],
                                      jac*proof['grad_clean_%d'%i], atol=1e-7)
    assert torch.equal(rng, torch.get_rng_state())
    local.train()
    try:
        module.energy(local, z, h, scheduler.alphas_cumprod, bank)
    except ValueError:
        pass
    else:
        raise AssertionError('Training state accepted')
    spec = importlib.util.spec_from_file_location('conditional_runner', Path(__file__).with_name('eval_shape_conditional.py'))
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    from test_eval_shape_proximal import args
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for arm in runner.ARMS:
            options = args(root)
            index = options.index('--global-state-root');del options[index:index+2]
            config, = runner.configurations(options+['--conditional-arm', arm,
                '--conditional-state-root', str(root/'conditional')])
            assert config['conditional']['clean_information_used'] is False
            assert config['corruptions'] == ['gaussian', 'background', 'rotation', 'lidar']
            bad = dict(config, decoder_style='updated')
            try:runner.validate(bad)
            except ValueError:pass
            else:raise AssertionError('Changed decoder accepted')
    report = dict(status='passed', tests=['condition gradient vs dense reference',
        'per-object batch scale', 'finite difference', 'global clean-estimate VJP',
        'DDIM correction equation', 'fixed update norms', 'private RNG',
        'frozen/eval guard', 'seven-arm scope/decoder guard'], native_gpu_tested=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
