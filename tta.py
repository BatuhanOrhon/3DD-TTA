from dataclasses import dataclass
import math

import torch
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist as chamfer_grad
from diffusers import DDIMScheduler
from utilities_3dd_tta import grad_freeze


@dataclass(frozen=True)
class SharedTrajectory:
    """Final local state and both style states from one TTA trajectory."""

    final_local_latent: torch.Tensor
    original_style: torch.Tensor
    updated_style: torch.Tensor


def decode_shared_trajectory(lion, trajectory: SharedTrajectory):
    """Decode one final local latent with original and updated styles."""
    def decoder_shape(latent):
        return latent.squeeze(3).squeeze(2) if latent.dim() == 4 else latent

    local_latent = decoder_shape(trajectory.final_local_latent)
    original_points = lion.vae.decoder(
        None, beta=None, context=local_latent, style=decoder_shape(trajectory.original_style)
    )
    updated_points = lion.vae.decoder(
        None, beta=None, context=local_latent, style=decoder_shape(trajectory.updated_style)
    )
    return original_points, updated_points


def selective_chamfer_loss(dists1, dists2, num_points, *, normalize=False):
    """Return the legacy SCD sum or its Eq. 11 point-count normalization."""
    loss = dists1.sum() + dists2.sum()
    return loss / num_points if normalize else loss


def tta_reconstruct(x, lion, steps_back_local, gamma, eta, p, total=100, *,
                    scheduler_observer=None, return_trajectory=False,
                    scd_normalize=False, prepared_state=None, state_observer=None,
                    trajectory_observer=None, shape_guidance=None, chamfer_reference=None,
                    style_spectral=None, decoded_spectral_weight=0., decoded_scd_weight=1., native_trace=None):
    """
    Test-Time Adaptation (TTA) reconstruction using DDIMScheduler and Chamfer Distance.

    Args:
    - x: Input point cloud data.
    - lion: Model instance containing VAE and local prior.
    - steps_back_local: Percentage of total steps to use in reverse scheduling.
    - gamma: Step size for updating noisy_local.
    - eta: Step size for updating style_cond.
    - p: Proportion of points to consider in Chamfer Distance.
    - total: Total number of diffusion steps (default: 100).
    - scd_normalize: Opt-in Eq. 11 point-count normalization for SCD.
    - decoded_spectral_weight: Opt-in physical decoded-X loss at observer checkpoints.
    - decoded_scd_weight: SCD coefficient only at live decoded clean checkpoints.
    - native_trace: Read-only baseline boundary callback, limited to loops 1 through 9.

    Returns:
    - pred_points: Reconstructed point cloud.
    """
    if native_trace is not None and (not callable(native_trace) or shape_guidance is not None
            or style_spectral is not None or chamfer_reference is not None or scd_normalize
            or decoded_spectral_weight != 0 or decoded_scd_weight != 1):
        raise ValueError('native_trace requires a callable and the original zero-weight joint SCD baseline')
    if (isinstance(decoded_spectral_weight, bool) or not isinstance(decoded_spectral_weight, (int, float))
            or not math.isfinite(decoded_spectral_weight) or decoded_spectral_weight < 0):
        raise ValueError('decoded_spectral_weight must be finite and nonnegative')
    if (isinstance(decoded_scd_weight, bool) or not isinstance(decoded_scd_weight, (int, float))
            or not math.isfinite(decoded_scd_weight) or decoded_scd_weight < 0):
        raise ValueError('decoded_scd_weight must be finite and nonnegative')
    if decoded_spectral_weight > 0 or decoded_scd_weight != 1:
        from decoded_spectral import DecodedSpectralObserver
        if not isinstance(trajectory_observer, DecodedSpectralObserver):
            raise ValueError('Decoded spectral mixing requires a decoded spectral observer')
        if shape_guidance is not None or style_spectral is not None:
            raise ValueError('Decoded spectral guidance requires the original joint local/style update')
    # Initialize chamfer distance and scheduler
    chamfer_dist = chamfer_grad()
    num_samples, num_points = x.size()[0], x.size()[1]
    
    scheduler = DDIMScheduler(
        beta_end=0.02, beta_schedule="linear", beta_start=0.0001, 
        clip_sample=False, num_train_timesteps=1000, prediction_type="epsilon"
    )
    scheduler.set_timesteps(total, device='cuda')
    if scheduler_observer is not None:
        scheduler_observer(scheduler)
    
    steps_back_local = (total * steps_back_local) // 100
    timesteps_local = scheduler.timesteps[-steps_back_local:]
    alpha_bar_local = scheduler.alphas_cumprod[timesteps_local[0]]

    # Freeze gradients for VAE and local prior
    vae = lion.vae
    local_prior = lion.priors[1]
    grad_freeze(local_prior)
    grad_freeze(vae)
    
    # The optional cache pairs z0/h0/epsilon, not the depth-dependent noisy h_t.
    if trajectory_observer is not None:
        trajectory_observer.mark('encode')
    with torch.no_grad():
        if prepared_state is None:
            latents = vae.encode(x)
            shape_latent = latents[2][0][0].unsqueeze(2).unsqueeze(3)
            latent_point = latents[2][1][0].unsqueeze(2).unsqueeze(3)
        else:
            shape_latent, latent_point = (prepared_state[key].detach().clone()
                                         for key in ('shape_latent', 'local_latent'))
            for value in (shape_latent, latent_point, prepared_state['noise']):
                if (value.ndim != 4 or value.shape[0] != num_samples or value.shape[2:] != (1, 1)
                        or value.dtype != x.dtype or value.device != x.device or not torch.isfinite(value).all()):
                    raise ValueError('Invalid paired latent/noise tensor')
            if latent_point.shape != (num_samples, 8192, 1, 1) or prepared_state['noise'].shape != latent_point.shape:
                raise ValueError('Paired local latent/noise must be B x 8192 x 1 x 1')
        latent_point_reshaped = latent_point.view(num_samples, 2048, -1)[:, :, :3]
        if chamfer_reference is not None:
            if (chamfer_reference.shape != (num_samples, 2048, 3)
                    or chamfer_reference.dtype != x.dtype or chamfer_reference.device != x.device
                    or not torch.isfinite(chamfer_reference).all()):
                raise ValueError('Invalid paired Chamfer reference')
            latent_point_reshaped = chamfer_reference.detach()
        # latent_point_std = latent_point.std(dim=1).view(num_samples, 1, -1)
    
    # Global style conditioning
    if trajectory_observer is not None:
        trajectory_observer.mark('style_noise')
    style_cond = vae.global2style(shape_latent)
    if style_spectral is not None:
        if shape_guidance is not None or chamfer_reference is not None or scd_normalize:
            raise ValueError('Spectral style routing requires the original fixed-anchor SCD host')
        style_spectral.initialize(latent_point_reshaped)
    if shape_guidance is not None:
        style_cond = shape_guidance.initialize(shape_latent, scheduler, lion)
    
    # Add noise to the local latent vector
    noise = torch.randn_like(latent_point) if prepared_state is None else prepared_state['noise'].detach().clone()
    if prepared_state is not None:
        from legacy_trajectory import _restore_rng
        _restore_rng(prepared_state['rng_after_prepare'])
    if state_observer is not None:
        state_observer(dict(shape_latent=shape_latent, local_latent=latent_point, noise=noise))
    noisy_latent_point = torch.sqrt(alpha_bar_local) * latent_point + noise * torch.sqrt(1 - alpha_bar_local)
    if trajectory_observer is not None:
        trajectory_observer.initial(input_points=x, shape_latent=shape_latent,
                                    local_latent=latent_point, style=style_cond, noise=noise,
                                    noisy_local=noisy_latent_point, timesteps=timesteps_local,
                                    alpha_bar=alpha_bar_local)
 
    # Reverse diffusion process using DDIMScheduler
    for i, t in enumerate(timesteps_local):
        if trajectory_observer is not None:
            trajectory_observer.mark('prior_scheduler')
        t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t + 1)
        noisy_latent_point = noisy_latent_point.detach()
        noisy_latent_point.requires_grad = True
        style_cond = style_cond.detach()
        style_cond.requires_grad = True
        trace_loop = native_trace is not None and i < 9
        if trace_loop:
            prior_t = t_tensor.float()
            native_trace(i + 1, t, 'prior_input', dict(noisy_local=noisy_latent_point,
                conditioning_style=style_cond, original_decoder_style=shape_latent,
                original_h0_xyz=latent_point_reshaped, prior_t=prior_t))

        if style_spectral is not None and style_spectral.audit_path is not None:
            from legacy_trajectory import _rng_state, _restore_rng, isolated_probe
            audit_rng = _rng_state()

        # Predict noise
        noise_pred = local_prior(x=noisy_latent_point, t=prior_t if trace_loop else t_tensor.float(), condition_input=style_cond, clip_feat=None)
        scheduler_output = scheduler.step(noise_pred, t, noisy_latent_point)
        pred_latent_point = scheduler_output.pred_original_sample
        if trace_loop:
            native_trace(i + 1, t, 'scheduler', dict(noise_pred=noise_pred,
                pred_original_sample=pred_latent_point, prev_sample=scheduler_output.prev_sample))
        if trajectory_observer is not None:
            # Clean predictions and terminal both use the original encoded style.
            decoded_spectral_loss = trajectory_observer.clean(
                pred_latent_point, shape_latent if shape_guidance is None else style_cond, i, t)
            trajectory_observer.mark('chamfer_backward_update')
        
        # Compute Selective Chamfer Distance
        pred_latent_point_reshaped = pred_latent_point.view(num_samples, 2048, -1)[:, :, :3]
        if trace_loop:
            dists1, dists2, nn1, nn2 = chamfer_dist(pred_latent_point_reshaped, latent_point_reshaped)
            raw_dists1, raw_dists2 = dists1, dists2
            sorted1, sorted2 = torch.sort(dists1, dim=1), torch.sort(dists2, dim=1)
            retained = int(num_points * p)
            dists1, dists2 = sorted1.values[:, :retained], sorted2.values[:, :retained]
        else:
            dists1, dists2, _, _ = chamfer_dist(pred_latent_point_reshaped, latent_point_reshaped)
            dists1 = torch.sort(dists1, dim=1).values[:, :int(num_points * p)]
            dists2 = torch.sort(dists2, dim=1).values[:, :int(num_points * p)]
        ch_loss = selective_chamfer_loss(
            dists1, dists2, num_points, normalize=scd_normalize)
        if trajectory_observer is not None and decoded_spectral_loss is not None:
            if getattr(trajectory_observer, 'record_updates', False):
                trajectory_observer.record_update_norms(ch_loss, decoded_spectral_loss,
                    noisy_latent_point, style_cond, gamma, eta, decoded_scd_weight, decoded_spectral_weight)
            if decoded_scd_weight != 1:
                ch_loss = decoded_scd_weight * ch_loss
        if trace_loop:
            scd_tensors = dict(scd_loss=ch_loss, distances_1=raw_dists1, distances_2=raw_dists2,
                               nearest_indices_1=nn1, nearest_indices_2=nn2)
            for direction, ordered in enumerate((sorted1, sorted2), 1):
                scd_tensors['sorted_distances_%d' % direction] = ordered.values
                scd_tensors['retained_indices_%d' % direction] = ordered.indices[:, :retained]
                if retained > 0:
                    scd_tensors['cutoff_distance_%d' % direction] = ordered.values[:, retained-1]
                if retained < ordered.values.shape[1]:
                    scd_tensors['next_distance_%d' % direction] = ordered.values[:, retained]
            native_trace(i + 1, t, 'scd', scd_tensors)
        if decoded_spectral_weight > 0 and decoded_spectral_loss is not None:
            if (not isinstance(decoded_spectral_loss, torch.Tensor) or decoded_spectral_loss.ndim != 0
                    or not decoded_spectral_loss.requires_grad or not torch.isfinite(decoded_spectral_loss)):
                raise ValueError('Expected finite live scalar decoded spectral loss')
            ch_loss = ch_loss + decoded_spectral_weight * decoded_spectral_loss
            if not torch.isfinite(ch_loss):
                raise FloatingPointError('Weighted decoded spectral objective is nonfinite')

        # Zero out gradients and backpropagate Chamfer loss
        if noisy_latent_point.grad is not None:
            noisy_latent_point.grad.zero_()
        if style_spectral is None:
            ch_loss.backward()
            if trace_loop:
                native_trace(i + 1, t, 'gradients', dict(noisy_local_gradient=noisy_latent_point.grad,
                    conditioning_style_gradient=style_cond.grad))
            noisy_latent_point = scheduler_output.prev_sample - gamma * noisy_latent_point.grad
            style_cond = style_cond - eta * style_cond.grad
            if trace_loop:
                native_trace(i + 1, t, 'post_update', dict(noisy_local=noisy_latent_point,
                    conditioning_style=style_cond))
        else:
            local_scd, style_spec = style_spectral.gradients(
                pred_latent_point_reshaped, ch_loss, noisy_latent_point, style_cond, i, t)
            if style_spectral.audit_path is not None and style_spectral.audit_record is None:
                def audit_clean(style):
                    with isolated_probe([local_prior]), torch.enable_grad():
                        _restore_rng(audit_rng)
                        physical_local = noisy_latent_point.detach().clone().requires_grad_(True)
                        predicted = local_prior(x=physical_local, t=t_tensor.float(),
                                                condition_input=style.detach().clone().requires_grad_(True), clip_feat=None)
                        return scheduler.step(predicted, t, physical_local).pred_original_sample.view(num_samples, 2048, -1)[:, :, :3].detach()
                style_spectral.audit(pred_latent_point_reshaped, noisy_latent_point, style_cond,
                                     local_scd, style_spec, audit_clean)
            noisy_latent_point = scheduler_output.prev_sample - gamma * local_scd
            style_cond = style_cond - style_spectral.style_step * style_spec
        if shape_guidance is not None:
            style_cond = shape_guidance.project(style_cond)

    decoder_style = shape_latent if shape_guidance is None else style_cond
    if trajectory_observer is not None:
        trajectory_observer.final_state(noisy_latent_point, decoder_style, timesteps_local[-1])
        trajectory_observer.mark('terminal_decode')

    if return_trajectory:
        return SharedTrajectory(
            final_local_latent=noisy_latent_point.detach(),
            original_style=shape_latent.detach(),
            updated_style=style_cond.detach(),
        )

    # Decode the predicted points from VAE decoder
    pred_points = vae.decoder(
        None, beta=None, context=noisy_latent_point.squeeze(3).squeeze(2), 
        style=decoder_style.squeeze(3).squeeze(2)
    )
    
    return pred_points
