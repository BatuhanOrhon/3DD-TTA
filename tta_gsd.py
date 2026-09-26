"""[Code] Opt-in GSD-inspired latent spectral guidance on the original trajectory.

The mathematical contract and paper/code distinctions are recorded in
knowledge/gsd_integration_20260922.md. This is not a GSDTTA reproduction.
"""
import math
from typing import Callable, Optional

import torch

import tta as baseline


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


@torch.enable_grad()
def tta_gsd_reconstruct(
    x: torch.Tensor, lion, steps_back_local: int, gamma: float, eta: float,
    p: float, total: int = 100, *, spectral_weight: float = 1.0,
    scd_weight: float = 1.0,
    spectral_config=None, scheduler_observer: Optional[Callable] = None,
    diagnostics_observer: Optional[Callable] = None,
    spectral_profile: Optional[str] = None, spectral_beta: Optional[float] = None,
) -> torch.Tensor:
    """Guide local and conditioning states; decode with the encoded global state.

    Weight zero directly delegates to the unchanged original implementation,
    bypassing graph construction and preserving its random number consumption.
    Positive weights add a fixed latent-XYZ graph objective to legacy summed
    SCD. Gradients pass through the frozen prior to both optimization variables.
    An outer ``no_grad`` is supported; ``inference_mode`` is not supported.
    """
    if (not math.isfinite(spectral_weight) or spectral_weight < 0 or
            not math.isfinite(scd_weight) or scd_weight < 0):
        raise ValueError("spectral_weight and scd_weight must be finite and nonnegative")
    if spectral_weight == 0 and scd_weight != 1.0:
        raise ValueError(
            "A zero spectral weight with a non-default SCD weight has no "
            "baseline-compatible trajectory; use spectral_weight=1 for the "
            "spectral-only ablation"
        )
    if spectral_weight == 0:
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
        if spectral_profile is None:
            target = build_spectral_target(reference_xyz, graph_config)
        else:
            from graph_spectral import build_smooth_spectral_target
            target = build_smooth_spectral_target(
                reference_xyz, graph_config, profile=spectral_profile, beta=spectral_beta)
    if diagnostics_observer is not None:
        diagnostics_observer({"kind": "graph", "samples": target.diagnostics})

    style_cond = vae.global2style(shape_latent)
    noise = torch.randn_like(local_latent)
    noisy_local = (torch.sqrt(alpha_bar_local) * local_latent
                   + noise * torch.sqrt(1 - alpha_bar_local))
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
        spectral_loss = target.loss(predicted_xyz)
        weighted_scd_loss = scd_weight * scd_loss
        weighted_spectral_loss = spectral_weight * spectral_loss
        total_loss = weighted_scd_loss + weighted_spectral_loss
        for name, loss in (("SCD loss", scd_loss), ("spectral loss", spectral_loss),
                           ("weighted SCD loss", weighted_scd_loss),
                           ("weighted spectral loss", weighted_spectral_loss),
                           ("total loss", total_loss)):
            _require_finite(loss, name)

        local_scd, style_scd = _loss_gradients(scd_loss, noisy_local, style_cond, retain_graph=True)
        local_spectral, style_spectral = _loss_gradients(
            spectral_loss, noisy_local, style_cond, retain_graph=False)
        local_weighted = spectral_weight * local_spectral
        style_weighted = spectral_weight * style_spectral
        local_scd_weighted = scd_weight * local_scd
        style_scd_weighted = scd_weight * style_scd
        local_update = gamma * (local_scd_weighted + local_weighted)
        style_update = eta * (style_scd_weighted + style_weighted)
        _require_finite(local_update, "local update")
        _require_finite(style_update, "style update")

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
