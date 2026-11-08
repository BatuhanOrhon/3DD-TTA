"""Reusable LION guidance kernels; the named baseline selects only frozen cheap guidance."""
import math

import torch


def proximal_chamfer(q, reference, forward, reverse, keep_forward, keep_reverse, tau):
    """Solve the frozen, bidirectional trimmed Chamfer quadratic exactly."""
    if not math.isfinite(tau) or tau < 0:
        raise ValueError('tau must be finite and nonnegative')
    forward, reverse = forward.long(), reverse.long()
    targets = reference.gather(1, forward[..., None].expand(-1, -1, 3))
    counts = keep_forward.to(q.dtype)[..., None]
    sums = targets * counts
    reverse_weight = keep_reverse.to(q.dtype)[..., None]
    sums = sums.scatter_add(1, reverse[..., None].expand(-1, -1, 3), reference * reverse_weight)
    counts = counts.scatter_add(1, reverse[..., None], reverse_weight)
    return (q + tau * sums) / (1 + tau * counts)


@torch.no_grad()
def feature_matches(q, reference, feature, reference_feature, *, k=8, beta=.1, scale_feature=None, scales=None):
    """Choose detached matches inside a geometric shortlist; no feature fidelity loss."""
    if not math.isfinite(beta) or beta < 0 or type(k) is not int or k < 1:
        raise ValueError('Invalid feature matching settings')
    distance = torch.cdist(q, reference).square()
    values, indices = distance.topk(min(k, reference.shape[1]), largest=False)
    candidates = reference_feature.squeeze(-1).gather(1, indices.flatten(1)).reshape(indices.shape)
    scale_source = reference_feature if scale_feature is None else scale_feature
    centered = scale_source - scale_source.median(dim=1, keepdim=True).values
    feature_scale = centered.abs().median(dim=1, keepdim=True).values.clamp_min(1e-6)
    # A two-point/sparse feature distribution can have zero MAD; RMS fallback
    # preserves feature units without independently rescaling the two clouds.
    rms = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    feature_scale = torch.where(feature_scale <= 1e-6, rms, feature_scale)
    spatial_scale = values.flatten(1).median(dim=1).values.clamp_min(1e-12)[:, None, None]
    if scales is not None:
        spatial_scale, feature_scale = scales
    score = values / spatial_scale + beta * ((feature - candidates) / feature_scale).square()
    return indices.gather(2, score.argmin(-1, keepdim=True)).squeeze(-1)


def causal_teacher(previous_probabilities):
    """Only explicitly supplied past observations can enter the teacher."""
    if not previous_probabilities:
        return None
    values = torch.stack([p.detach() for p in previous_probabilities])
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite teacher')
    return values.mean(0)


def feature_step(latent, gradient, fraction):
    """One RMS-relative feature-only step; zero gradient means no update."""
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError('fraction must be finite and nonnegative')
    if latent.shape != gradient.shape or latent.ndim != 3 or latent.shape[-1] != 4:
        raise ValueError('Expected matching B x N x4 tensors')
    if not torch.isfinite(latent).all() or not torch.isfinite(gradient).all():
        raise ValueError('Nonfinite latent or gradient')
    g = gradient[..., 3:]
    scale = latent[..., 3:].square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    rms = g.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    result = latent.detach().clone()
    result[..., 3:] -= fraction * scale * g / rms
    return result


def bridge_clean_delta(previous, xyz_delta, alpha_previous):
    """DDIM transport while keeping the already predicted epsilon fixed."""
    alpha = torch.as_tensor(alpha_previous, device=previous.device, dtype=previous.dtype)
    if not torch.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError('Expected actual previous cumulative alpha in (0,1]')
    correction = torch.zeros_like(previous).reshape(previous.shape[0], -1, 4)
    correction[..., :3] = xyz_delta
    return previous + alpha.sqrt() * correction.reshape_as(previous)


def sample(lion, state, steps, arm, *, tau=.06, beta=.1, callback=None, diagnostic=None,
           scheduler_observer=None):
    """Isolated experimental N8/B35 sampler; full-control parity is a GPU gate."""
    from diffusers import DDIMScheduler
    from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
    from legacy_trajectory import _restore_rng
    allowed = ('full', 'style_frozen', 'sparse', 'cheap', 'hybrid', 'feature', 'shuffled', 'metric4d', 'geometric', 'unguided')
    if arm not in allowed or steps not in (8, 35):
        raise ValueError('Invalid frozen pilot arm/horizon')
    scheduler = DDIMScheduler(beta_end=.02, beta_schedule='linear', beta_start=.0001,
        clip_sample=False, num_train_timesteps=1000, prediction_type='epsilon')
    h0, z0 = (state[key].detach().clone() for key in ('local_latent','shape_latent'))
    scheduler.set_timesteps(100, device=h0.device)
    if scheduler_observer is not None:
        scheduler_observer(scheduler)
    times = scheduler.timesteps[-steps:]
    style = lion.vae.global2style(z0)
    alpha = scheduler.alphas_cumprod[times[0]]
    noisy = torch.sqrt(alpha)*h0 + state['noise']*torch.sqrt(1-alpha)
    _restore_rng(state['rng_after_prepare'])
    reference = h0.reshape(len(h0), -1, 4)
    xyz_reference = reference[..., :3].contiguous()
    permutation = torch.randperm(reference.shape[1], generator=torch.Generator().manual_seed(20261107)).to(h0.device) if arm=='shuffled' else None
    matching_scales=None
    if arm in ('feature','shuffled','metric4d','geometric'):
        with torch.no_grad():
            ref_distance=torch.cdist(xyz_reference,xyz_reference).square()
            neighbors=ref_distance.topk(9,largest=False).values[:,:,1:]
            xyz_scale=neighbors.flatten(1).median(1).values.clamp_min(1e-12)[:,None,None]
            centered=reference[...,3:]-reference[...,3:].median(1,keepdim=True).values
            feature_scale=centered.abs().median(1,keepdim=True).values
            rms=centered.square().mean(1,keepdim=True).sqrt().clamp_min(1e-6)
            feature_scale=torch.where(feature_scale<=1e-6,rms,feature_scale)
            matching_scales=(xyz_scale,feature_scale)
    chamfer = chamfer_3DDist()
    snapshots, records = [], []
    checkpoints = [2,4,6] if steps==8 else [9,18,27]
    for i,t in enumerate(times):
        full = arm not in ('cheap','sparse','hybrid','unguided') or (arm in ('sparse','hybrid') and i%3==0)
        noisy = noisy.detach().requires_grad_(full)
        style = style.detach().requires_grad_(full)
        with torch.set_grad_enabled(full):
            eps = lion.priors[1](x=noisy,t=torch.ones(len(h0),dtype=torch.int64,device=h0.device).float()*(t+1),
                condition_input=style,clip_feat=None)
            out = scheduler.step(eps,t,noisy)
            clean = out.pred_original_sample
        if i+1 in checkpoints:
            snapshots.append(clean.detach().clone())
            if callback is not None: callback(clean.detach(),z0,i+1)
        if not full and arm in ('sparse','unguided'):
            noisy,style=out.prev_sample.detach(),style.detach()
            records.append(dict(loop=i+1,timestep=int(t),full_backward=False))
            continue
        points = clean.reshape(len(h0),-1,4)
        q = points[...,:3].contiguous()
        d1,d2,idx1,idx2 = chamfer(q,xyz_reference)
        retained=int(q.shape[1]*.95)
        keep1=keep2=None
        if not full or diagnostic is not None:
            keep1=torch.zeros_like(d1,dtype=torch.bool).scatter_(1,d1.argsort(1)[:,:retained],True)
            keep2=torch.zeros_like(d2,dtype=torch.bool).scatter_(1,d2.argsort(1)[:,:retained],True)
        if full:
            if arm in ('feature','shuffled','geometric'):
                rf=reference[...,3:]
                if arm=='shuffled': rf=rf[:,permutation]
                idx1=feature_matches(q,xyz_reference,points[...,3:],rf,k=8,beta=0. if arm=='geometric' else beta,scales=matching_scales)
                idx2=feature_matches(xyz_reference,q,rf,points[...,3:],k=8,beta=0. if arm=='geometric' else beta,scales=matching_scales)
                d1=(q-xyz_reference.gather(1,idx1[...,None].expand(-1,-1,3))).square().sum(-1)
                d2=(xyz_reference-q.gather(1,idx2[...,None].expand(-1,-1,3))).square().sum(-1)
            if arm=='metric4d':
                xyz_scale,feature_scale=matching_scales
                f_weight=(beta*xyz_scale).sqrt()/feature_scale
                live4=torch.cat((points[...,:3],points[...,3:]*f_weight),-1)
                ref4=torch.cat((reference[...,:3],reference[...,3:]*f_weight),-1)
                distances=torch.cdist(live4,ref4).square()
                d1,d2=distances.min(2).values,distances.min(1).values
            loss=d1.sort(1).values[:,:retained].sum()+d2.sort(1).values[:,:retained].sum()
            loss.backward()
            delta=-.03*noisy.grad
            previous=out.prev_sample+delta
            next_style=style if arm=='style_frozen' else style-.003*style.grad
        else:
            delta=torch.zeros_like(noisy)
            if arm=='hybrid' or arm=='cheap':
                corrected=proximal_chamfer(q,xyz_reference,idx1,idx2,keep1,keep2,tau)
                previous_t=int(t)-1000//100
                previous_alpha=scheduler.alphas_cumprod[previous_t] if previous_t>=0 else scheduler.final_alpha_cumprod
                previous=bridge_clean_delta(out.prev_sample,corrected-q,previous_alpha)
                delta=previous-out.prev_sample
            else: previous=out.prev_sample
            next_style=style
        if diagnostic is not None:
            diagnostic(dict(loop=i+1,timestep=int(t),clean=clean.detach(),reference=reference,
                delta=delta.detach(),idx1=idx1.detach(),idx2=idx2.detach(),keep1=keep1,keep2=keep2,
                full_backward=full))
        records.append(dict(loop=i+1,timestep=int(t),full_backward=full))
        noisy,style=previous.detach(),next_style.detach()
    snapshots.append(noisy.clone())
    if callback is not None: callback(noisy,z0,steps)
    return snapshots, records


def repair_candidates(latents, decode, classify, *, fraction=.01, geometry_limit=.001,
                      fixed_teacher=None, feature_only=True, diagnostic=None):
    """Readout-only repair: never uses future predictions or changes sampler state."""
    if not math.isfinite(geometry_limit) or geometry_limit < 0:
        raise ValueError('Invalid geometry trust limit')
    history, saved, records = [], [], []
    for position,h in enumerate(latents):
        live=h.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            before=decode(live)
            logits=classify(before)
            teacher=causal_teacher(history) if fixed_teacher is None else fixed_teacher.detach()
            # Keep first candidate unchanged in both teacher arms for a matched budget.
            if position==0:
                chosen=logits.detach()
                corrected,after,changed_logits=live,before,logits
                gradient=torch.zeros_like(live)
                accepted=torch.zeros(len(h),dtype=torch.bool,device=h.device)
                record=dict(teacher_count=0,accepted=[False]*len(h),geometry_ratio=[0.]*len(h))
            else:
                loss=-(teacher*logits.log_softmax(-1)).sum()
                gradient,=torch.autograd.grad(loss,live)
                corrected=feature_step(live,gradient,fraction)
                if not feature_only:
                    limit=fraction*live[...,3:].square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-6)*live.shape[1]**.5
                    norm=gradient.square().sum((1,2),keepdim=True).sqrt().clamp_min(1e-12)
                    corrected=live.detach()-limit*gradient/norm
                # Same gradient-enabled forward branch avoids an eval/no-grad confound.
                corrected=corrected.detach().requires_grad_(True)
                after=decode(corrected)
                changed_logits=classify(after)
                ratio=(after-before.detach()).square().mean((1,2)).sqrt()/before.detach().square().mean((1,2)).sqrt().clamp_min(1e-6)
                accepted=torch.isfinite(ratio)&(ratio<=geometry_limit)&torch.isfinite(changed_logits).all(1)
                chosen=torch.where(accepted[:,None],changed_logits,logits).detach()
                record=dict(teacher_count=len(history) if fixed_teacher is None else 1,
                    teacher_source='past_corrected_predictions' if fixed_teacher is None else 'corrupted_input',
                    accepted=accepted.detach().cpu().tolist(),geometry_ratio=ratio.detach().cpu().tolist(),
                    teacher=teacher.detach().cpu().tolist(),before_logits=logits.detach().cpu().tolist(),
                    attempted_logits=changed_logits.detach().cpu().tolist(),
                    feature_only=feature_only,fraction=fraction)
        if diagnostic is not None:
            diagnostic(dict(position=position,before=before.detach(),after=after.detach(),
                latent=live.detach(),attempted_latent=corrected.detach(),gradient=gradient.detach(),
                chosen_latent=torch.where(accepted[:,None,None],corrected,live).detach(),
                chosen_decoded=torch.where(accepted[:,None,None],after,before).detach(),
                before_logits=logits.detach(),after_logits=changed_logits.detach()))
        history.append(chosen.softmax(-1))
        saved.append(chosen)
        records.append(record)
    return torch.stack(saved,1),records
