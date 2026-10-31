"""Fixed research-only global proposals and per-object clean-latent bounds."""
import math

import torch

from legacy_trajectory import rng_hash, tensor_hash

ARMS = ('baseline', 'updated', 'anchor', 'proposal', 'guided3')


def project_to_anchor(value: torch.Tensor, anchor: torch.Tensor, fraction: float) -> torch.Tensor:
    """Project each object into a ball of radius fraction*||original z||."""
    if (not math.isfinite(fraction) or fraction < 0 or value.shape != anchor.shape
            or value.ndim < 2 or value.dtype != anchor.dtype or value.device != anchor.device
            or not value.is_floating_point() or not torch.isfinite(value).all()
            or not torch.isfinite(anchor).all()):
        raise ValueError('Invalid finite shape latent/anchor/radius')
    delta = value-anchor
    radius = fraction*anchor.flatten(1).norm(dim=1)
    norm = delta.flatten(1).norm(dim=1)
    factor = (radius/norm.clamp_min(torch.finfo(value.dtype).tiny)).clamp(max=1)
    return anchor+delta*factor.reshape((-1,)+(1,)*(value.ndim-1))


class ShapeLatentGuidance:
    """One batch; raw z and style must coincide for the pinned all55 checkpoint."""

    def __init__(self, arm: str, *, noise_seed: int):
        if arm not in ARMS[1:] or type(noise_seed) is not int or not 0 <= noise_seed < 2**63:
            raise ValueError('Expected a non-baseline shape arm and valid independent seed')
        self.arm, self.noise_seed = arm, noise_seed
        self.anchor = None
        self.report = dict(arm=arm, radius_fraction=None if arm == 'updated' else .05,
                           global_forward_calls=0, global_backward_calls=0,
                           global_steps=[], local_style_projections=0,
                           global_noise_sha256=None, global_rng_unchanged=True)

    def initialize(self, shape: torch.Tensor, scheduler, lion) -> torch.Tensor:
        if self.anchor is not None:
            raise ValueError('Shape adapter is single-batch; duplicate initialization')
        if (shape.ndim != 4 or shape.shape[1:] != (128, 1, 1)
                or not torch.isfinite(shape).all() or not shape.is_floating_point()):
            raise ValueError('Expected finite B x128 x1 x1 global latent')
        self.anchor = shape.detach().clone()
        if not torch.equal(lion.vae.global2style(self.anchor), self.anchor):
            raise ValueError('Bounded raw z requires identity global2style')
        self.report['initial_z_sha256'] = tensor_hash(self.anchor)
        style = self.anchor
        if self.arm in ('proposal', 'guided3'):
            prior = lion.priors[0]
            if prior.training or any(p.requires_grad for p in prior.parameters()):
                raise ValueError('Global prior must be eval with frozen parameters')
            before = rng_hash()
            generator = torch.Generator(device=shape.device).manual_seed(self.noise_seed)
            noise = torch.randn(shape.shape, dtype=shape.dtype, device=shape.device, generator=generator)
            self.report['global_noise_sha256'] = tensor_hash(noise)
            alpha = scheduler.alphas_cumprod[20].to(shape)
            noisy = alpha.sqrt()*self.anchor+(1-alpha).sqrt()*noise
            for timestep in ([20] if self.arm == 'proposal' else [20, 10, 0]):
                noisy = noisy.detach().requires_grad_(self.arm == 'guided3')
                with torch.set_grad_enabled(self.arm == 'guided3'):
                    prediction = prior(x=noisy, t=shape.new_full((len(shape),), timestep+1),
                                       condition_input=None, clip_feat=None)
                    output = scheduler.step(prediction, timestep, noisy)
                    clean = output.pred_original_sample
                    if not torch.isfinite(clean).all():
                        raise ValueError('Nonfinite global clean prediction')
                    self.report['global_forward_calls'] += 1
                    record = dict(timestep=timestep, clean_displacement_norms=
                                  (clean-self.anchor).flatten(1).norm(dim=1).detach().cpu().tolist())
                    if self.arm == 'guided3':
                        loss = (clean-self.anchor).flatten(1).square().mean(dim=1).sum()
                        gradient, = torch.autograd.grad(loss, noisy)
                        if not torch.isfinite(gradient).all():
                            raise ValueError('Nonfinite global anchor gradient')
                        norms = gradient.flatten(1).norm(dim=1)
                        limit = .01*self.anchor.flatten(1).norm(dim=1)
                        factor = (limit/norms.clamp_min(torch.finfo(shape.dtype).tiny)).clamp(max=1)
                        correction = gradient*factor.reshape(-1, 1, 1, 1)
                        noisy = output.prev_sample.detach()-correction.detach()
                        self.report['global_backward_calls'] += 1
                        record.update(anchor_loss=float(loss.detach()),
                                      anchor_gradient_norms=norms.detach().cpu().tolist(),
                                      correction_norms=correction.flatten(1).norm(dim=1).detach().cpu().tolist())
                        style = noisy
                    else:
                        style = clean.detach()
                    self.report['global_steps'].append(record)
            self.report['global_rng_unchanged'] = rng_hash() == before
            if not self.report['global_rng_unchanged']:
                raise RuntimeError('Global proposal advanced the common RNG stream')
        result = self._project(style.detach())
        self.report['global_result_z_sha256'] = tensor_hash(result)
        return result

    def _project(self, style: torch.Tensor) -> torch.Tensor:
        if self.anchor is None or not torch.isfinite(style).all():
            raise ValueError('Uninitialized or nonfinite style')
        if self.arm == 'updated':
            return style
        return project_to_anchor(style, self.anchor, .05)

    def project(self, style: torch.Tensor) -> torch.Tensor:
        result = self._project(style.detach())
        self.report['local_style_projections'] += 1
        self.report['final_displacement_norms'] = (result-self.anchor).flatten(1).norm(dim=1).cpu().tolist()
        self.report['anchor_norms'] = self.anchor.flatten(1).norm(dim=1).cpu().tolist()
        return result

    def diagnostics(self) -> dict:
        return dict(self.report)
