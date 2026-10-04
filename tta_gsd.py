"""[Code] Opt-in GSD-inspired latent spectral guidance on the original trajectory.

The mathematical contract and paper/code distinctions are recorded in
knowledge/gsd_integration_20260922.md. This is not a GSDTTA reproduction.
"""
import math
from typing import Callable, Optional

import torch

import tta as baseline
from gsd_composition import CompositionConfig, compose_block
from gsd_paired_inputs import PreparedGuidanceInputs, prepare_guidance_inputs


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"GSD nonfinite {name}")


def _loss_gradients(loss, local, style, *, retain_graph):
    gradients = torch.autograd.grad(
        loss, (local, style), retain_graph=retain_graph, allow_unused=True)
    gradients = tuple(torch.zeros_like(state) if grad is None else grad
                      for grad, state in zip(gradients, (local, style)))
    for gradient in gradients:
        _require_finite(gradient, "gradient")
    return gradients


def _number(value: torch.Tensor, name: str) -> float:
    _require_finite(value, name)
    return float(value.detach().item())


def _paired_loss_gradients(loss, local, style, *, retain_graph):
    gradients = torch.autograd.grad(
        loss, (local, style), retain_graph=retain_graph, allow_unused=True)
    unused = tuple(gradient is None for gradient in gradients)
    values = tuple(torch.zeros_like(state) if gradient is None else gradient.detach()
                   for gradient, state in zip(gradients, (local, style)))
    for gradient in values:
        _require_finite(gradient, "gradient")
    return values, unused


def _sample_norm(value: torch.Tensor) -> torch.Tensor:
    return value.detach().reshape(value.shape[0], -1).double().norm(dim=1)


def _sample_cosine(left: torch.Tensor, right: torch.Tensor):
    left_flat = left.detach().reshape(left.shape[0], -1).double()
    right_flat = right.detach().reshape(right.shape[0], -1).double()
    left_norm = torch.linalg.vector_norm(left_flat, dim=1)
    right_norm = torch.linalg.vector_norm(right_flat, dim=1)
    valid = (left_norm > 1e-12) & (right_norm > 1e-12)
    result = [None] * left.shape[0]
    if bool(valid.any()):
        values = (left_flat[valid] * right_flat[valid]).sum(dim=1) / (left_norm[valid] * right_norm[valid])
        for index, value in zip(valid.nonzero(as_tuple=False).flatten().tolist(), values.tolist()):
            result[index] = max(-1.0, min(1.0, float(value)))
    return result


def _observer_scalar(value, name):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("composition observer diagnostic must be scalar: " + name)
        value = value.detach().item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError("composition observer diagnostic is not JSON scalar: " + name)


def _composition_reconstruct(x, lion, steps_back_local, gamma, eta, p, total, *,
                             spectral_config, scheduler_observer,
                             composition_config, prepared_inputs,
                             composition_observer):
    if not isinstance(composition_config, CompositionConfig):
        raise TypeError("composition_config must be a CompositionConfig")
    if total <= 0 or not 0 < steps_back_local <= 100:
        raise ValueError("GSD requires total > 0 and steps_back_local in (0, 100]")
    reverse_steps = (total * steps_back_local) // 100
    if reverse_steps < 1:
        raise ValueError("GSD requires at least one reverse step")
    if not 0 < p <= 1 or not all(math.isfinite(v) for v in (gamma, eta)):
        raise ValueError("GSD requires p in (0, 1] and finite update rates")
    baseline.grad_freeze(lion.vae)
    baseline.grad_freeze(lion.priors[1])
    lion.vae.eval()
    lion.priors[1].eval()
    if prepared_inputs is None:
        prepared_inputs = prepare_guidance_inputs(
            x, lion, total=total, steps_back_local=steps_back_local)
    elif not isinstance(prepared_inputs, PreparedGuidanceInputs):
        raise TypeError("prepared_inputs must be PreparedGuidanceInputs")
    else:
        prepared_inputs.validate_for(x, total, steps_back_local)

    from graph_spectral import SpectralConfig, build_spectral_target

    shape_latent, original_local, original_style, noise, timesteps, alpha_bar = \
        prepared_inputs.clone_states()
    reference_xyz = original_local.view(len(x), 2048, 4)[:, :, :3].detach()
    graph_config = SpectralConfig() if spectral_config is None else spectral_config
    target = build_spectral_target(reference_xyz, graph_config)
    if composition_observer is not None:
        graph_config_values = {
            "k": graph_config.k,
            "delta": graph_config.delta,
            "gamma": graph_config.graph_gamma,
            "modes": graph_config.modes,
            "eigenspace_rtol": graph_config.eigenspace_rtol,
            "eigenspace_atol": graph_config.eigenspace_atol,
        }
        for sample_index, diagnostics in enumerate(target.diagnostics):
            graph_event = {"kind": "graph", "sample_index": int(sample_index)}
            graph_event.update({
                "graph_config_" + key: _observer_scalar(value, "config/" + key)
                for key, value in graph_config_values.items()
            })
            graph_event.update({
                "graph_" + key: _observer_scalar(value, "diagnostics/" + key)
                for key, value in diagnostics.items()
            })
            graph_event["graph_runtime_seconds"] = _observer_scalar(
                diagnostics.get("runtime_seconds", 0.0), "diagnostics/runtime_seconds")
            composition_observer(graph_event)
    scheduler = baseline.DDIMScheduler(**dict(prepared_inputs.scheduler_options))
    scheduler.set_timesteps(total, device=x.device)
    if not torch.equal(scheduler.timesteps[-reverse_steps:].to(timesteps.device), timesteps):
        raise ValueError("prepared timesteps do not match the active scheduler")
    current_alpha = scheduler.alphas_cumprod[timesteps[0].to(scheduler.alphas_cumprod.device)]
    if not torch.equal(current_alpha.to(alpha_bar.device), alpha_bar):
        raise ValueError("prepared alpha does not match the active scheduler")
    if scheduler_observer is not None:
        scheduler_observer(scheduler)
    alpha = alpha_bar.to(device=original_local.device, dtype=original_local.dtype)
    local = torch.sqrt(alpha) * original_local + noise * torch.sqrt(1 - alpha)
    style = original_style.detach().clone()
    chamfer_dist = baseline.chamfer_grad()
    num_samples, num_points = x.shape[:2]
    retained = int(num_points * p)

    for step_index, timestep in enumerate(timesteps):
        t_tensor = torch.ones(num_samples, dtype=torch.int64, device=x.device) * (timestep + 1)
        local = local.detach().requires_grad_(True)
        style = style.detach().requires_grad_(True)
        noise_pred = lion.priors[1](
            x=local, t=t_tensor.float(), condition_input=style, clip_feat=None)
        scheduler_output = scheduler.step(noise_pred, timestep, local)
        predicted_xyz = scheduler_output.pred_original_sample.view(num_samples, 2048, 4)[:, :, :3]
        distances1, distances2, _, _ = chamfer_dist(predicted_xyz, reference_xyz)
        distances1 = torch.sort(distances1, dim=1).values[:, :retained]
        distances2 = torch.sort(distances2, dim=1).values[:, :retained]
        scd_loss = baseline.selective_chamfer_loss(distances1, distances2, num_points)
        # Hard-v1 is evaluated and differentiated on every route, including a
        # zero spectral coefficient, so masked mechanism probes remain live.
        spectral_loss = target.loss(predicted_xyz)
        _require_finite(scd_loss, "SCD loss")
        _require_finite(spectral_loss, "hard-v1 spectral loss")
        (local_scd, style_scd), local_scd_unused = _paired_loss_gradients(
            scd_loss, local, style, retain_graph=True)
        (local_spectral, style_spectral), local_spectral_unused = _paired_loss_gradients(
            spectral_loss, local, style, retain_graph=False)
        local_direction, local_rows = compose_block(
            local_scd, local_spectral,
            scd_weight=composition_config.local_scd_weight,
            spectral_weight=composition_config.local_spectral_weight,
            mode=composition_config.local_mode,
            norm_floor=composition_config.norm_floor)
        style_direction, style_rows = compose_block(
            style_scd, style_spectral,
            scd_weight=composition_config.style_scd_weight,
            spectral_weight=composition_config.style_spectral_weight,
            mode=composition_config.style_mode,
            norm_floor=composition_config.norm_floor)
        local_update = gamma * local_direction
        style_update = eta * style_direction
        _require_finite(local_update, "local update")
        _require_finite(style_update, "style update")
        local_guidance_displacement = -local_update
        local_ddim_displacement = scheduler_output.prev_sample - local
        guidance_cosine = _sample_cosine(local_guidance_displacement, local_ddim_displacement)
        guidance_norm = _sample_norm(local_guidance_displacement)
        ddim_norm = _sample_norm(local_ddim_displacement)
        guidance_dot = (local_guidance_displacement.detach().reshape(num_samples, -1).double() *
                        local_ddim_displacement.detach().reshape(num_samples, -1).double()).sum(dim=1)
        style_after = (style - style_update).detach()
        style_drift_norm = _sample_norm(style_after - original_style)
        if composition_observer is not None:
            local_scd_norm, local_spectral_norm = _sample_norm(local_scd), _sample_norm(local_spectral)
            style_scd_norm, style_spectral_norm = _sample_norm(style_scd), _sample_norm(style_spectral)
            local_applied_norm, style_applied_norm = _sample_norm(local_direction), _sample_norm(style_direction)
            local_actual_update_norm = _sample_norm(local_update)
            style_actual_update_norm = _sample_norm(style_update)
            for sample_index, (local_diag, style_diag) in enumerate(zip(local_rows, style_rows)):
                ddim = float(ddim_norm[sample_index].item())
                composition_observer({
                    "kind": "step",
                    "step_index": int(step_index),
                    "timestep": int(timestep.item()),
                    "sample_index": int(sample_index),
                    "local_mode": composition_config.local_mode,
                    "style_mode": composition_config.style_mode,
                    "local_scd_grad_norm": float(local_scd_norm[sample_index].item()),
                    "local_spectral_grad_norm": float(local_spectral_norm[sample_index].item()),
                    "style_scd_grad_norm": float(style_scd_norm[sample_index].item()),
                    "style_spectral_grad_norm": float(style_spectral_norm[sample_index].item()),
                    "local_weighted_scd_grad_norm": local_diag["scd_norm"],
                    "local_weighted_spectral_grad_norm": local_diag["spectral_norm"],
                    "style_weighted_scd_grad_norm": style_diag["scd_norm"],
                    "style_weighted_spectral_grad_norm": style_diag["spectral_norm"],
                    "local_scd_unused": bool(local_scd_unused[0]),
                    "local_spectral_unused": bool(local_spectral_unused[0]),
                    "style_scd_unused": bool(local_scd_unused[1]),
                    "style_spectral_unused": bool(local_spectral_unused[1]),
                    "local_gradient_cosine": local_diag["cosine"],
                    "style_gradient_cosine": style_diag["cosine"],
                    "local_routed_scd_dot": local_diag["scd_direction_dot"],
                    "local_routed_spectral_dot": local_diag["spectral_direction_dot"],
                    "style_routed_scd_dot": style_diag["scd_direction_dot"],
                    "style_routed_spectral_dot": style_diag["spectral_direction_dot"],
                    "local_raw_projected_norm": local_diag["raw_projected_norm"],
                    "local_capped_projected_norm": local_diag["capped_projected_norm"],
                    "local_cap_factor": local_diag["cap_factor"],
                    "local_applied_scale": local_diag["applied_scale"],
                    "local_applied_norm": local_diag["applied_norm"],
                    "style_raw_projected_norm": style_diag["raw_projected_norm"],
                    "style_capped_projected_norm": style_diag["capped_projected_norm"],
                    "style_cap_factor": style_diag["cap_factor"],
                    "style_applied_scale": style_diag["applied_scale"],
                    "style_applied_norm": style_diag["applied_norm"],
                    "local_routed_direction_norm": float(local_applied_norm[sample_index].item()),
                    "style_routed_direction_norm": float(style_applied_norm[sample_index].item()),
                    "local_update_norm": float(local_actual_update_norm[sample_index].item()),
                    "style_update_norm": float(style_actual_update_norm[sample_index].item()),
                    "style_drift_norm": float(style_drift_norm[sample_index].item()),
                    "local_guidance_ddim_dot": float(guidance_dot[sample_index].item()),
                    "local_guidance_ddim_cosine": guidance_cosine[sample_index],
                    "local_guidance_ddim_norm_ratio": (
                        float(guidance_norm[sample_index].item() / ddim) if ddim > 1e-12 else None),
                    "local_projection_skipped": bool(local_diag["projection_skipped"]),
                    "style_projection_skipped": bool(style_diag["projection_skipped"]),
                })
        local = (scheduler_output.prev_sample - local_update).detach()
        style = style_after
        _require_finite(local, "local state")
        _require_finite(style, "style conditioning state")
        del (noise_pred, scheduler_output, predicted_xyz, distances1, distances2,
             scd_loss, spectral_loss, local_scd, style_scd, local_spectral,
             style_spectral, local_direction, style_direction, local_update,
             style_update, local_rows, style_rows)

    with torch.no_grad():
        result = lion.vae.decoder(
            None, beta=None, context=local.squeeze(3).squeeze(2),
            style=shape_latent.squeeze(3).squeeze(2))
    _require_finite(result, "output")
    return result


@torch.no_grad()
def _unguided_diffusion(x, lion, steps_back_local, total, scheduler_observer,
                        diagnostics_observer):
    """Same encode/noise/DDIM/decode host, with no losses or guidance gradients."""
    if total <= 0 or not 0 < steps_back_local <= 100:
        raise ValueError("Unguided diffusion requires valid total/reverse steps")
    reverse_steps = (total * steps_back_local) // 100
    if reverse_steps < 1:
        raise ValueError("Unguided diffusion requires at least one reverse step")
    scheduler = baseline.DDIMScheduler(
        beta_end=.02, beta_schedule="linear", beta_start=.0001,
        clip_sample=False, num_train_timesteps=1000, prediction_type="epsilon")
    scheduler.set_timesteps(total, device=x.device)
    if scheduler_observer is not None:
        scheduler_observer(scheduler)
    timesteps = scheduler.timesteps[-reverse_steps:]
    vae, prior = lion.vae, lion.priors[1]
    baseline.grad_freeze(vae)
    baseline.grad_freeze(prior)
    latents = vae.encode(x)
    shape = latents[2][0][0].unsqueeze(2).unsqueeze(3)
    local = latents[2][1][0].unsqueeze(2).unsqueeze(3)
    if local.shape != (len(x), 8192, 1, 1):
        raise ValueError("Unguided diffusion expects B x 8192 x 1 x 1 local encoding")
    style = vae.global2style(shape)
    noise = torch.randn_like(local)
    alpha = scheduler.alphas_cumprod[timesteps[0]]
    local = torch.sqrt(alpha) * local + torch.sqrt(1 - alpha) * noise
    for index, timestep in enumerate(timesteps):
        t = torch.ones(len(x), dtype=torch.int64, device=x.device) * (timestep + 1)
        prediction = prior(x=local, t=t.float(), condition_input=style, clip_feat=None)
        previous = scheduler.step(prediction, timestep, local).prev_sample
        _require_finite(previous, "unguided local state")
        if diagnostics_observer is not None:
            diagnostics_observer(dict(kind="step", step_index=index, timestep=int(timestep.item()),
                                      batch_size=len(x), local_update_norm=0., style_update_norm=0.,
                                      local_ddim_displacement_norm=_number((previous-local).norm(), "DDIM step")))
        local = previous
    result = vae.decoder(None, beta=None, context=local.squeeze(3).squeeze(2),
                         style=shape.squeeze(3).squeeze(2))
    _require_finite(result, "unguided output")
    return result


@torch.enable_grad()
def tta_gsd_reconstruct(
    x: torch.Tensor, lion, steps_back_local: int, gamma: float, eta: float,
    p: float, total: int = 100, *, spectral_weight: float = 1.0,
    scd_weight: float = 1.0,
    spectral_config=None, scheduler_observer: Optional[Callable] = None,
    diagnostics_observer: Optional[Callable] = None,
    spectral_profile: Optional[str] = None, spectral_beta: Optional[float] = None,
    probe_observer: Optional[Callable] = None,
    sample_observer: Optional[Callable] = None,
    allow_unguided: bool = False,
    composition_config: Optional[CompositionConfig] = None,
    prepared_inputs: Optional[PreparedGuidanceInputs] = None,
    composition_observer: Optional[Callable] = None,
) -> torch.Tensor:
    """Guide local and conditioning states; decode with the encoded global state.

    Weight zero directly delegates to the unchanged original implementation,
    bypassing graph construction and preserving its random number consumption.
    Positive weights add a fixed latent-XYZ graph objective to legacy summed
    SCD. Gradients pass through the frozen prior to both optimization variables.
    Explicit probe/sample observers opt into an instrumented reference path;
    probes never apply their candidate gradients and consume no random draws.
    An outer ``no_grad`` is supported; ``inference_mode`` is not supported.
    """
    if composition_config is not None:
        if spectral_weight != 1.0 or scd_weight != 1.0:
            raise ValueError("composition_config cannot be combined with legacy scalar weights")
        if spectral_profile is not None or spectral_beta is not None:
            raise ValueError("composition_config uses hard-v1 and does not accept legacy profiles")
        if probe_observer is not None or sample_observer is not None:
            raise ValueError("composition_config cannot be combined with legacy calibration probes")
        if diagnostics_observer is not None:
            raise ValueError("composition_config uses composition_observer for scalar diagnostics")
        if allow_unguided:
            raise ValueError("composition_config selects the instrumented route directly")
        return _composition_reconstruct(
            x, lion, steps_back_local, gamma, eta, p, total,
            spectral_config=spectral_config,
            scheduler_observer=scheduler_observer,
            composition_config=composition_config,
            prepared_inputs=prepared_inputs,
            composition_observer=composition_observer)
    if prepared_inputs is not None or composition_observer is not None:
        raise ValueError("prepared_inputs and composition_observer require composition_config")
    if (not math.isfinite(spectral_weight) or spectral_weight < 0 or
            not math.isfinite(scd_weight) or scd_weight < 0):
        raise ValueError("spectral_weight and scd_weight must be finite and nonnegative")
    if allow_unguided and spectral_weight == 0 and scd_weight == 0:
        if probe_observer is not None or sample_observer is not None:
            raise ValueError("Unguided diffusion does not support calibration gradient probes")
        return _unguided_diffusion(x, lion, steps_back_local, total,
                                  scheduler_observer, diagnostics_observer)
    if spectral_weight == 0 and scd_weight != 1.0:
        raise ValueError(
            "A zero spectral weight with a non-default SCD weight has no "
            "baseline-compatible trajectory; use spectral_weight=1 for the "
            "spectral-only ablation"
        )
    if probe_observer is not None and (spectral_weight != 0 or scd_weight != 1):
        raise ValueError("calibration probes require SCD-only weight 1, spectral weight 0")
    if spectral_weight == 0 and probe_observer is None and sample_observer is None:
        result = baseline.tta_reconstruct(
            x, lion, steps_back_local, gamma, eta, p, total,
            scheduler_observer=scheduler_observer)
        _require_finite(result, "output")
        return result

    from graph_spectral import SpectralConfig, build_spectral_target

    if total <= 0 or not 0 < steps_back_local <= 100:
        raise ValueError("GSD requires total > 0 and steps_back_local in (0, 100]")
    reverse_steps = (total * steps_back_local) // 100
    if reverse_steps < 1:
        raise ValueError("GSD requires at least one reverse step")
    if not 0 < p <= 1 or not all(math.isfinite(v) for v in (gamma, eta)):
        raise ValueError("GSD requires p in (0, 1] and finite update rates")

    chamfer_dist = baseline.chamfer_grad()
    num_samples, num_points = x.shape[:2]
    scheduler = baseline.DDIMScheduler(
        beta_end=.02, beta_schedule="linear", beta_start=.0001,
        clip_sample=False, num_train_timesteps=1000, prediction_type="epsilon")
    scheduler.set_timesteps(total, device=x.device)
    if scheduler_observer is not None:
        scheduler_observer(scheduler)
    timesteps_local = scheduler.timesteps[-reverse_steps:]
    alpha_bar_local = scheduler.alphas_cumprod[timesteps_local[0]]

    vae, local_prior = lion.vae, lion.priors[1]
    baseline.grad_freeze(local_prior)
    baseline.grad_freeze(vae)
    with torch.no_grad():
        latents = vae.encode(x)
        shape_latent = latents[2][0][0].unsqueeze(2).unsqueeze(3)
        local_latent = latents[2][1][0].unsqueeze(2).unsqueeze(3)
        if local_latent.shape != (num_samples, 8192, 1, 1):
            raise ValueError("GSD expects LION local encoding B x 8192 x 1 x 1")
        reference_xyz = local_latent.view(num_samples, 2048, 4)[:, :, :3]
        graph_config = SpectralConfig() if spectral_config is None else spectral_config
        if probe_observer is not None:
            from graph_spectral import build_probe_spectral_target
            target = build_probe_spectral_target(reference_xyz, graph_config)
        elif spectral_weight == 0:
            target = None
        elif spectral_profile is None:
            target = build_spectral_target(reference_xyz, graph_config)
        else:
            from graph_spectral import build_smooth_spectral_target
            target = build_smooth_spectral_target(
                reference_xyz, graph_config, profile=spectral_profile, beta=spectral_beta)
    if diagnostics_observer is not None and target is not None:
        diagnostics_observer({"kind": "graph", "samples": target.diagnostics})

    style_cond = vae.global2style(shape_latent)
    noise = torch.randn_like(local_latent)
    noisy_local = (torch.sqrt(alpha_bar_local) * local_latent
                   + noise * torch.sqrt(1 - alpha_bar_local))
    probe_steps = {0, reverse_steps // 2, reverse_steps - 1}
    for step_index, timestep in enumerate(timesteps_local):
        t_tensor = torch.ones(num_samples, dtype=torch.int64, device=x.device) * (timestep + 1)
        noisy_local = noisy_local.detach().requires_grad_(True)
        style_cond = style_cond.detach().requires_grad_(True)
        noise_pred = local_prior(
            x=noisy_local, t=t_tensor.float(), condition_input=style_cond, clip_feat=None)
        scheduler_output = scheduler.step(noise_pred, timestep, noisy_local)
        predicted_xyz = scheduler_output.pred_original_sample.view(num_samples, 2048, 4)[:, :, :3]
        distances1, distances2, _, _ = chamfer_dist(predicted_xyz, reference_xyz)
        retained = int(num_points * p)
        distances1 = torch.sort(distances1, dim=1).values[:, :retained]
        distances2 = torch.sort(distances2, dim=1).values[:, :retained]
        scd_loss = baseline.selective_chamfer_loss(distances1, distances2, num_points)
        spectral_loss = target.loss(predicted_xyz) if spectral_weight else scd_loss.detach() * 0
        weighted_scd_loss = scd_weight * scd_loss
        weighted_spectral_loss = spectral_weight * spectral_loss
        total_loss = weighted_scd_loss + weighted_spectral_loss
        for name, loss in (("SCD loss", scd_loss), ("spectral loss", spectral_loss),
                           ("weighted SCD loss", weighted_scd_loss),
                           ("weighted spectral loss", weighted_spectral_loss),
                           ("total loss", total_loss)):
            _require_finite(loss, name)

        probing = probe_observer is not None and step_index in probe_steps
        local_scd, style_scd = _loss_gradients(
            scd_loss, noisy_local, style_cond, retain_graph=bool(spectral_weight) or probing)
        if spectral_weight:
            local_spectral, style_spectral = _loss_gradients(
                spectral_loss, noisy_local, style_cond, retain_graph=False)
        else:
            local_spectral, style_spectral = torch.zeros_like(noisy_local), torch.zeros_like(style_cond)
        if probing:
            from gsd_calibration import CANDIDATES, gradient_rows
            for candidate_index, (profile, beta) in enumerate(CANDIDATES):
                loss = target.loss(predicted_xyz, profile=profile, beta=beta)
                local_probe, style_probe = _loss_gradients(
                    loss, noisy_local, style_cond, retain_graph=candidate_index < len(CANDIDATES) - 1)
                probe_observer(dict(step_index=step_index, timestep=int(timestep.item()),
                                    profile=profile, beta=beta,
                                    samples=gradient_rows(local_scd, style_scd, local_probe, style_probe)))
                del loss, local_probe, style_probe
        local_weighted = spectral_weight * local_spectral
        style_weighted = spectral_weight * style_spectral
        local_scd_weighted = scd_weight * local_scd
        style_scd_weighted = scd_weight * style_scd
        local_update = gamma * (local_scd_weighted + local_weighted)
        style_update = eta * (style_scd_weighted + style_weighted)
        _require_finite(local_update, "local update")
        _require_finite(style_update, "style update")

        if sample_observer is not None:
            from gsd_calibration import state_rows, gradient_rows
            rows = state_rows(noisy_local, style_cond, local_scd_weighted, style_scd_weighted,
                              scheduler_output.prev_sample, gamma, eta, local_weighted, style_weighted)
            if spectral_weight:
                for row, gradients in zip(rows, gradient_rows(
                        local_scd_weighted, style_scd_weighted, local_weighted, style_weighted)):
                    row.update(gradients)
            sample_observer(dict(step_index=step_index, timestep=int(timestep.item()), samples=rows))

        if diagnostics_observer is not None:
            diagnostics = {"kind": "step", "step_index": step_index,
                           "timestep": int(timestep.item()), "batch_size": num_samples,
                           "scd_loss": _number(scd_loss, "SCD loss"),
                           "spectral_loss": _number(spectral_loss, "spectral loss"),
                           "scd_weight": scd_weight,
                           "spectral_weight": spectral_weight,
                           "weighted_spectral_loss": _number(weighted_spectral_loss, "weighted loss")}
            if spectral_profile is not None:
                diagnostics["spectral_profile"] = spectral_profile
                diagnostics["spectral_beta"] = spectral_beta
            for name, value in (
                ("local_scd_grad_norm", local_scd), ("local_spectral_grad_norm", local_spectral),
                ("style_scd_grad_norm", style_scd), ("style_spectral_grad_norm", style_spectral),
                ("local_weighted_spectral_grad_norm", local_weighted),
                ("style_weighted_spectral_grad_norm", style_weighted),
                ("local_update_norm", local_update), ("style_update_norm", style_update),
            ):
                diagnostics[name] = _number(value.norm(), name)
            diagnostics_observer(diagnostics)

        noisy_local = (scheduler_output.prev_sample - local_update).detach()
        style_cond = (style_cond - style_update).detach()
        _require_finite(noisy_local, "local state")
        _require_finite(style_cond, "conditioning state")

    with torch.no_grad():
        result = vae.decoder(
            None, beta=None, context=noisy_local.squeeze(3).squeeze(2),
            style=shape_latent.squeeze(3).squeeze(2))
    _require_finite(result, "output")
    return result
