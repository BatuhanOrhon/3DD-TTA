"""Local decoder finite differences on one frozen physical reference graph."""
from __future__ import annotations

import math

import torch

from graph_spectral import SmoothSpectralTarget
from legacy_trajectory import (_restore_rng, _rng_state, model_buffer_hashes,
                               rng_hash, tensor_hash)
from semantic_trajectory import gradient_probe


def audit_decoded_gradient(vae, target, latent, original_style, *,
                           epsilon_scales=(.001, .01)) -> tuple[dict, dict]:
    """Return a JSON report and finite NumPy arrays, including failed evidence.

    All forwards use the gradient-enabled decoder branch and fresh live clones.
    The direction perturbs only sample zero; each epsilon is its latent norm
    times the requested scale. This checks the local decoder at a fixed graph,
    not gradients through the denoiser, graph construction, or point matching.
    """
    relative_tolerance, absolute_tolerance = .1, 1e-5
    report = dict(passed=False, relative_tolerance=relative_tolerance,
                  absolute_tolerance=absolute_tolerance, derivative_resolution_floor=absolute_tolerance,
                  finite_differences=[], errors=[], omitted_nonfinite_arrays=[],
                  scope='Local decoder at frozen physical graph; not a whole-denoiser certificate',
                  perturbed_sample=0, all_forwards_gradient_enabled=True)
    arrays, tracked = {}, {}
    state = _rng_state()
    before_rng, before_buffers = rng_hash(), None

    def save(name, value):
        if isinstance(value, torch.Tensor):
            if torch.isfinite(value).all():
                arrays[name] = value.detach().cpu().numpy().copy()
            else:
                report['omitted_nonfinite_arrays'].append(name)

    def decoder_shape(value):
        return value.squeeze(3).squeeze(2) if value.ndim == 4 else value

    try:
        for name, value in (('latent', latent), ('original_style', original_style)):
            save(name, value)
            if (not isinstance(value, torch.Tensor) or value.ndim not in (2, 4)
                    or (value.ndim == 4 and value.shape[2:] != (1, 1))
                    or not 1 <= value.shape[0] <= 64 or value.shape[0] != latent.shape[0]
                    or value.dtype not in (torch.float32, torch.float64)
                    or not torch.isfinite(value).all()):
                raise ValueError('Expected matching finite B1..64 floating latent/style in decoder-compatible shape')
            tracked[name] = (value, tensor_hash(value))
        if original_style.device != latent.device or original_style.dtype != latent.dtype:
            raise ValueError('Latent and original style device/dtype must match')
        if not isinstance(target, SmoothSpectralTarget):
            raise ValueError('Expected a frozen SmoothSpectralTarget')
        reference = target.reference_xyz
        save('reference_xyz', reference)
        if (reference.ndim != 3 or reference.shape[0] != latent.shape[0] or reference.shape[1] < 2
                or reference.shape[2] != 3 or reference.requires_grad
                or reference.device != latent.device or reference.dtype != latent.dtype
                or not torch.isfinite(reference).all()):
            raise ValueError('Reference must be finite detached B x N x 3 matching the latent batch/device/dtype')
        tracked['reference_xyz'] = (reference, tensor_hash(reference))
        if len(target.filters) != len(reference):
            raise ValueError('Expected one detached physical filter per sample')
        for index, spectral_filter in enumerate(target.filters):
            if (spectral_filter.shape != (reference.shape[1], reference.shape[1])
                    or spectral_filter.requires_grad or spectral_filter.device != reference.device
                    or spectral_filter.dtype != reference.dtype or not torch.isfinite(spectral_filter).all()):
                raise ValueError('Physical filter shape/device/dtype must match the detached reference')
            tracked['filter_%d' % index] = (spectral_filter, tensor_hash(spectral_filter))
        save('filters_first_two', torch.stack(target.filters))
        scales = tuple(epsilon_scales)
        if not scales or any(isinstance(scale, bool) or not isinstance(scale, (int, float))
                             or not math.isfinite(scale) or scale <= 0 for scale in scales):
            raise ValueError('epsilon_scales must contain finite positive scales')
        before_buffers = model_buffer_hashes({'vae': vae})

        def forward(name, perturbation=None, *, gradient=False):
            # Reset each forward independently, including stateful/stochastic decoders.
            with gradient_probe([vae]):
                _restore_rng(state)
                local = latent.detach().clone()
                if perturbation is not None:
                    local = local + perturbation
                local.requires_grad_(True)
                style = original_style.detach().clone().requires_grad_(True)
                decoded = vae.decoder(None, beta=None, context=decoder_shape(local),
                                      style=decoder_shape(style))
                save(name + '_decoded', decoded)
                loss = target.loss(decoded)
                if (not isinstance(loss, torch.Tensor) or loss.ndim != 0
                        or not loss.requires_grad or not torch.isfinite(loss)):
                    raise ValueError('Expected finite scalar loss connected to live decoder inputs')
                derivative = None
                if gradient:
                    derivative, = torch.autograd.grad(loss, (local,), allow_unused=True)
                    if derivative is None:
                        raise ValueError('Decoder loss is disconnected from the local latent')
                    save('gradient', derivative)
                return decoded.detach(), float(loss.detach()), derivative

        baseline, baseline_loss, gradient = forward('baseline', gradient=True)
        report['baseline_loss'] = baseline_loss
        repeat, repeat_loss, _ = forward('repeat')
        difference = repeat - baseline
        repeat_metrics = dict(repeat_loss=repeat_loss, repeat_loss_absolute_difference=abs(repeat_loss-baseline_loss),
                              repeat_decoded_max_absolute_difference=float(difference.abs().max()),
                              repeat_decoded_rms_difference=float(difference.square().mean().sqrt()))
        if not all(math.isfinite(value) for value in repeat_metrics.values()):
            raise ValueError('Repeat decoder comparison is nonfinite')
        report.update(repeat_metrics)
        if not torch.isfinite(gradient).all():
            raise ValueError('Autograd returned a nonfinite decoder derivative')
        first_norm = float(gradient[0].norm())
        if not math.isfinite(first_norm) or first_norm <= absolute_tolerance:
            raise ValueError('First-sample decoder derivative is zero or unresolved at the absolute tolerance')
        direction = torch.zeros_like(gradient)
        direction[0] = gradient[0] / first_norm
        save('direction', direction)
        autodiff = float((gradient * direction).sum())
        latent_norm = float(latent[0].detach().norm())
        if not math.isfinite(autodiff) or not math.isfinite(latent_norm):
            raise ValueError('Directional derivative or latent norm is nonfinite')
        report.update(autograd_directional_derivative=autodiff, first_sample_latent_norm=latent_norm,
                      first_sample_gradient_norm=first_norm)
        for index, scale in enumerate(scales):
            record = dict(scale=float(scale), passed=False)
            report['finite_differences'].append(record)
            try:
                epsilon = latent_norm * scale
                if not math.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError('Relative latent perturbation has zero or nonfinite epsilon')
                record['epsilon'] = epsilon
                _, plus, _ = forward('fd_%d_plus' % index, epsilon * direction)
                _, minus, _ = forward('fd_%d_minus' % index, -epsilon * direction)
                finite_difference = (plus-minus)/(2*epsilon)
                if not math.isfinite(finite_difference):
                    raise ValueError('Central finite difference is nonfinite')
                error = abs(finite_difference-autodiff)
                tolerance = absolute_tolerance + relative_tolerance*abs(autodiff)
                repeat_slope_uncertainty = abs(repeat_loss-baseline_loss)/epsilon
                if not all(math.isfinite(value) for value in (error, tolerance, repeat_slope_uncertainty)):
                    raise ValueError('Finite-difference comparison exceeds finite numeric range')
                record.update(plus_loss=plus, minus_loss=minus, finite_difference=finite_difference,
                              autograd_directional_derivative=autodiff, absolute_error=error,
                              relative_error=error/abs(autodiff), allowed_absolute_error=tolerance,
                              repeat_loss_slope_uncertainty=repeat_slope_uncertainty,
                              passed=abs(finite_difference) > absolute_tolerance and error <= tolerance
                              and repeat_slope_uncertainty <= tolerance)
            except Exception as error:
                record['error'] = type(error).__name__ + ': ' + str(error)
                report['errors'].append(record['error'])
        report['passed'] = all(record['passed'] for record in report['finite_differences'])
    except Exception as error:
        report['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        try:
            _restore_rng(state)
        except Exception as error:
            report['errors'].append('RNG restoration failed: ' + type(error).__name__ + ': ' + str(error))
        checks = [('rng_unchanged', lambda: before_rng == rng_hash()),
                  ('inputs_unchanged', lambda: all(expected == tensor_hash(value) for value, expected in tracked.values())),
                  ('buffers_unchanged', lambda: before_buffers is None or before_buffers == model_buffer_hashes({'vae': vae}))]
        for name, check in checks:
            try:
                report[name] = check()
            except Exception as error:
                report[name] = False
                report['errors'].append(name + ': ' + type(error).__name__ + ': ' + str(error))
        if not (report['rng_unchanged'] and report['inputs_unchanged'] and report['buffers_unchanged']):
            report['errors'].append('Audit changed RNG, model buffers or live inputs')
        report['passed'] = report['passed'] and not report['errors'] and not report['omitted_nonfinite_arrays']
    return report, arrays
