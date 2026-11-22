"""XYZ proximal sampling around the unchanged official CloudFixer denoiser."""
from __future__ import annotations

import hashlib
import math

import torch

from pointcloud_proximal import proximal_chamfer


def verify_gamma(actual, intended) -> dict:
    if (actual.shape != (501,) or actual.shape != intended.shape or actual.dtype != intended.dtype
            or not torch.isfinite(actual).all() or not torch.isfinite(intended).all()):
        raise ValueError('Invalid native polynomial500 gamma table')
    mode,iterations='native_polynomial',0
    if not torch.allclose(actual,intended,atol=1e-6,rtol=1e-6):
        # Author EMA also averages frozen gamma; all501 release values match its fixed point.
        averaged=intended.detach().clone();beta=.9999
        for iterations in range(10000):
            updated=averaged*beta+(1-beta)*intended
            if torch.equal(updated,averaged):break
            averaged=updated
        else:raise ValueError('Author EMA gamma failed to converge')
        if not torch.allclose(actual,averaged,atol=1e-6,rtol=1e-6):
            raise ValueError('Checkpoint gamma differs from native polynomial500 and its author EMA')
        mode='author_ema_0.9999_fixed_point'
    return dict(mode=mode,ema_iterations=iterations,max_abs_native=float((actual-intended).abs().max()))


def grid(*, steps: int, start: int) -> list:
    if type(steps) is not int or type(start) is not int or not 4 <= steps <= start <= 500:
        raise ValueError('Require integer 4 <= reverse updates <= initial index <= 500')
    result = [round(start * (steps-i) / steps) for i in range(steps+1)]
    if any(a <= b for a, b in zip(result, result[1:])):
        raise ValueError('Diffusion indices must strictly decrease')
    return result


@torch.no_grad()
def _trimmed_matches(q, reference, mask, retained):
    keep_n = int(q.shape[1] * retained)
    keep_r = (mask.sum(1) * retained).long()
    if keep_n < 1 or (keep_r < 1).any():
        raise ValueError('Trimming removed all points')
    distance = torch.cdist(q, reference).square().masked_fill(~mask[:, None, :], float('inf'))
    d1, forward = distance.min(2)
    d2, reverse = distance.min(1)
    keep1 = torch.zeros_like(d1, dtype=torch.bool).scatter_(1, d1.argsort(1)[:, :keep_n], True)
    keep2 = (d2.argsort(1).argsort(1) < keep_r[:, None]) & mask
    return forward, reverse, keep1, keep2, d1, d2


def trimmed_neighbors(q, reference, mask, retained):
    """Return bidirectional nearest neighbors and the two retained-pair masks."""
    return _trimmed_matches(q, reference, mask, retained)[:4]


def trimmed_scd(q, reference, mask, retained):
    """Sum the two directional means of retained squared distances per candidate."""
    _, _, keep1, keep2, d1, d2 = _trimmed_matches(q, reference, mask, retained)
    return (d1.masked_fill(~keep1, 0).sum(1) / keep1.sum(1)
            + d2.masked_fill(~keep2, 0).sum(1) / keep2.sum(1))


@torch.no_grad()
def correct(q, reference, mask, *, tau: float, retained: float, measure=None, index=None, pair_transform=None):
    """Full predicted set versus valid observed anchors, with bidirectional trimming."""
    if not math.isfinite(tau) or tau < 0 or not 0 < retained <= 1:
        raise ValueError('Invalid proximal strength or retained fraction')
    if measure is None:
        pairs = trimmed_neighbors(q, reference, mask, retained)
    else:
        pairs = measure("local_matching", trimmed_neighbors, q, reference, mask, retained)
    if pair_transform is not None:
        pairs = pair_transform(index, *pairs)
    if measure is None:
        return proximal_chamfer(q, reference, *pairs, tau)
    return measure("proximal", proximal_chamfer, q, reference, *pairs, tau)


@torch.no_grad()
def sample(model, x, mask, noise, *, steps: int, start: int, tau: float, retained: float = .95,
           trace_all: bool = False, probe_indices=None, observe=None, fingerprint=False,
           pose_reference=None, feedback_rotation=None, measure=None, schedule=None, pair_transform=None):
    """Return pre-correction clean estimates plus a native deterministic t0 terminal."""
    indices = grid(steps=steps, start=start)
    if schedule is not None:
        if (len(schedule) != len(indices) or schedule[0] != start or schedule[-1] != 0
                or any(type(i) is not int for i in schedule)
                or any(a <= b for a, b in zip(schedule, schedule[1:]))):
            raise ValueError('Invalid user-supplied native timestep grid')
        indices = list(schedule)
    reference = x
    predict = model.phi
    correction = correct
    if measure is not None:
        from functools import partial
        predict = partial(measure, "denoiser", predict)
        correction = partial(correction, measure=measure)
    if (x.ndim != 3 or x.shape[-1] != 3 or not x.is_floating_point()
            or x.shape != noise.shape or x.device != noise.device or x.dtype != noise.dtype
            or not torch.isfinite(x).all() or not torch.isfinite(noise).all()):
        raise ValueError('Expected finite matching XYZ/noise tensors')
    if mask.shape == x.shape[:2] + (1,):
        mask = mask.squeeze(-1)
    if (mask.shape != x.shape[:2] or mask.device != x.device or not torch.isfinite(mask).all()
            or not ((mask == 0) | (mask == 1)).all() or not mask.bool().any(1).all()):
        raise ValueError('Expected a binary mask with at least one valid anchor per object')
    if model.T != 500 or not math.isfinite(tau) or tau < 0 or not 0 < retained <= 1:
        raise ValueError('Invalid native model/sampling configuration')
    mask = mask.bool()
    node_mask = torch.ones_like(x[..., :1])  # Author phi uses the complete XYZ array.

    def coefficients(index):
        t = torch.full((len(x), 1), index / 500, device=x.device, dtype=x.dtype)
        if not ((t * 500).long() == index).all():
            raise ValueError('Normalized timestep truncation disagrees with the integer noise index')
        gamma = model.gamma(t)
        return t, model.alpha(gamma, target_tensor=x), model.sigma(gamma, target_tensor=x)

    _, alpha, sigma = coefficients(start)
    state = alpha * x + sigma * noise
    candidates = []
    loops = list(range(1, steps+1)) if trace_all else [steps//4, steps//2, 3*steps//4]
    if probe_indices is not None:
        probes = list(probe_indices)
        if (trace_all or not probes or probes[-1] != 0
                or any(type(i) is not int for i in probes)
                or probes != [i for i in indices if i in probes]):
            raise ValueError('Expected descending native probe indices ending in zero')
        loops = [i + 1 for i, t in enumerate(indices[:-1]) if t in probes]
    trajectory = hashlib.sha256() if fingerprint else None
    for loop, (index, previous) in enumerate(zip(indices, indices[1:]), 1):
        t, alpha, sigma = coefficients(index)
        if fingerprint:
            trajectory.update(state.detach().cpu().contiguous().numpy().tobytes())
        eps = predict(state, t, node_mask)
        clean = (state - sigma * eps) / alpha
        if not torch.isfinite(clean).all():
            raise FloatingPointError('Nonfinite diffusion clean estimate')
        if loop in loops:
            candidates.append(clean.clone())
        pre = clean
        if pose_reference is not None:
            reference = pose_reference(index, pre)
        if tau:
            clean = correction(clean, reference, mask, tau=tau, retained=retained,
                               index=index, pair_transform=pair_transform)
        if index == 20 and feedback_rotation is not None:
            rotation = feedback_rotation(pre)
            if rotation is not None:
                clean, eps, reference = clean @ rotation, eps @ rotation, x @ rotation
        if observe is not None:
            observe(index, state.clone(), pre.clone(), clean.clone(), alpha.clone(), sigma.clone())
        _, alpha_s, sigma_s = coefficients(previous)
        state = alpha_s * clean + sigma_s * eps
    t, alpha, sigma = coefficients(0)
    terminal = (state - sigma * predict(state, t, node_mask)) / alpha
    if not torch.isfinite(terminal).all():
        raise FloatingPointError('Nonfinite terminal estimate')
    if observe is not None:
        observe(0, state.clone(), terminal.clone(), terminal.clone(), alpha.clone(), sigma.clone())
    candidates.append(terminal)
    metadata = dict(indices=indices, candidate_loops=loops+[steps],
        candidate_indices=[indices[i-1] for i in loops]+[0], denoiser_calls=steps+1,
        valid_reference_counts=mask.sum(1).cpu().tolist(),retained_forward_count=int(x.shape[1]*retained),
        retained_reference_counts=(mask.sum(1)*retained).long().cpu().tolist())
    if fingerprint:
        trajectory.update(state.detach().cpu().contiguous().numpy().tobytes())
        metadata['denoiser_input_trajectory_sha256'] = trajectory.hexdigest()
    return torch.stack(candidates, 1), metadata
