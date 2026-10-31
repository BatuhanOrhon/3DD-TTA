"""Exploratory frozen conditional-energy guidance; no production default."""
import torch

LOCAL_TIMES = (99, 299, 599, 899)
GLOBAL_TIMES = (20, 10, 0)
STEP_FRACTION = .02


def noise_bank(h, seed):
    generator = torch.Generator(device=h.device).manual_seed(seed)
    return torch.randn((len(LOCAL_TIMES),) + tuple(h.shape), device=h.device,
                       dtype=h.dtype, generator=generator)


def energy(prior, z, h, alphas, bank, *, gradient=False):
    """Per-object MSE and optional condition VJP, one graph at a time."""
    if (h.requires_grad or bank.requires_grad or z.shape[1:] != (128, 1, 1)
            or h.shape != (len(z), 8192, 1, 1)
            or bank.shape != (4,) + h.shape
            or any(p.requires_grad for p in prior.parameters())
            or any(m.training for m in prior.modules())):
        raise ValueError('Energy requires fixed h/noise and frozen eval prior')
    values, terms = [], []
    for tau, eps in zip(LOCAL_TIMES, bank):
        a = alphas[tau].to(h)
        noisy = a.sqrt() * h + (1-a).sqrt() * eps
        with torch.set_grad_enabled(gradient):
            leaf = z.detach().requires_grad_(gradient)
            pred = prior(x=noisy, t=z.new_full((len(z),), tau+1),
                         condition_input=leaf, clip_feat=None)
            per_object = (pred-eps).flatten(1).square().mean(1)
            if gradient:
                terms.append(torch.autograd.grad(per_object.sum()/len(LOCAL_TIMES), leaf)[0].detach())
        values.append(per_object.detach())
    result = torch.stack(values, dim=1)
    if not torch.isfinite(result).all():
        raise RuntimeError('Nonfinite conditional energy')
    grad = torch.stack(terms).sum(0) if gradient else None
    if gradient and not torch.isfinite(grad).all():
        raise RuntimeError('Nonfinite conditional gradient')
    return result, grad


def propose(start, encoder, h, global_prior, local_prior, scheduler, mode, seed):
    """Three-step DDIM with an explicit local-condition/global-clean VJP."""
    if mode not in ('none', 'prior3', 'guided3'):
        raise ValueError('Unknown fixed proposal')
    if any(p.requires_grad for p in global_prior.parameters()) or global_prior.training:
        raise ValueError('Frozen global prior required')
    bank = noise_bank(h, seed+1)
    heldout = noise_bank(h, seed+2)
    proof = dict(start_z=start.detach().clone(), encoder_z=encoder.detach().clone(),
                 h_star=h.detach().clone(), train_noise=bank, heldout_noise=heldout)
    proof['start_energy'], _ = energy(local_prior, start, h, scheduler.alphas_cumprod, heldout)
    if mode == 'none':
        proof['final_z'] = start.detach().clone()
        proof['final_energy'] = proof['start_energy'].clone()
        return proof['final_z'], proof
    generator = torch.Generator(device=start.device).manual_seed(seed)
    eps = torch.randn(start.shape, generator=generator, device=start.device, dtype=start.dtype)
    a = scheduler.alphas_cumprod[GLOBAL_TIMES[0]].to(start)
    noisy = a.sqrt()*start + (1-a).sqrt()*eps
    proof['global_noise'] = eps
    scale = STEP_FRACTION * encoder.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    for i, t in enumerate(GLOBAL_TIMES):
        before = noisy.detach().requires_grad_(mode == 'guided3')
        with torch.set_grad_enabled(mode == 'guided3'):
            pred = global_prior(x=before, t=start.new_full((len(start),), t+1),
                                condition_input=None, clip_feat=None)
            step = scheduler.step(pred, t, before)
            clean = step.pred_original_sample
        update = torch.zeros_like(before)
        if mode == 'guided3':
            losses, g_clean = energy(local_prior, clean, h, scheduler.alphas_cumprod,
                                     bank, gradient=True)
            g_noisy = torch.autograd.grad(clean, before, grad_outputs=g_clean)[0]
            norms = g_noisy.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            update = scale*g_noisy/norms.clamp_min(1e-12)
            proof['energy_%d'%i] = losses
            proof['grad_clean_%d'%i] = g_clean
            proof['grad_noisy_%d'%i] = g_noisy.detach()
        noisy = (step.prev_sample.detach()-update.detach()).detach()
        for key, value in [('noisy_before', before), ('prior_clean', clean),
                           ('ddim_previous', step.prev_sample), ('guidance_update', update),
                           ('noisy_after', noisy)]:
            if not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite global transition: '+key)
            proof[key+'_%d'%i] = value.detach().clone()
    proof['final_z'] = noisy
    proof['final_energy'], _ = energy(local_prior, noisy, h, scheduler.alphas_cumprod, heldout)
    return noisy, proof
