import torch
import torch.nn.functional as F
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist as chamfer_grad
from diffusers import DDIMScheduler
from utilities_3dd_tta import grad_freeze

def tta_gsd_reconstruct_sync(x, lion, graph_spectral_module, steps_back_local, steps_back_global, gamma, eta, p, loss_weights=None, total=100, use_static_style=False):
    """
    Test-Time Adaptation (TTA) using a Synchronous Dual Diffusion approach.
    Runs both Global Prior and Local Prior synchronously, updating both shape_latent and latent_point via GSD guidance.

    Args:
    - x: Input point cloud data (B, N, 3).
    - lion: Model instance containing VAE and priors.
    - graph_spectral_module: Instance of GraphSpectralDNA.
    - steps_back_local: Percentage of total steps to use in reverse scheduling for local points (e.g., 5).
    - steps_back_global: Percentage of total steps to use in reverse scheduling for global shape (e.g., 20).
    - gamma: Step size for updating global shape latent (next_noisy_z).
    - eta: Step size for updating local latent points (noisy_h).
    - p: Proportion of points to consider in Chamfer Distance.
    - loss_weights: Dictionary of loss weights.
    - total: Total number of diffusion steps (default: 100).
    - use_static_style: If true, uses the original shape_latent for the final decode instead of the updated one.
    """
    if loss_weights is None:
        loss_weights = {"spectral_low": 1.0, "spectral_mid": 0.0, "spectral_high": 0.0, "chamfer": 0.0}
        
    weight_spectral_low = loss_weights.get("spectral_low", loss_weights.get("spectral", 1.0))
    weight_spectral_mid = loss_weights.get("spectral_mid", 0.0)
    weight_spectral_high = loss_weights.get("spectral_high", 0.0)
    weight_chamfer = loss_weights.get("chamfer", 0.0)
    weight_invariant = loss_weights.get("invariant", 0.0)

    # Initialize chamfer distance only if needed
    chamfer_dist = None
    if weight_chamfer > 0.0:
        chamfer_dist = chamfer_grad()
        
    num_samples, num_points = x.size()[0], x.size()[1]
    num_latent_points = 2048  # VAE latent space fixed at 2048 points
    
    scheduler = DDIMScheduler(
        beta_end=0.02, beta_schedule="linear", beta_start=0.0001, 
        clip_sample=False, num_train_timesteps=1000, prediction_type="epsilon"
    )
    scheduler.set_timesteps(total, device=x.device)
    
    steps_back_local = int((total * steps_back_local) // 100)
    steps_back_global = int((total * steps_back_global) // 100)
    max_steps = max(steps_back_local, steps_back_global)
    
    timesteps_sync = scheduler.timesteps[-max_steps:] if max_steps > 0 else []
    
    timesteps_local = scheduler.timesteps[-steps_back_local:] if steps_back_local > 0 else []
    alpha_bar_local = scheduler.alphas_cumprod[timesteps_local[0]] if steps_back_local > 0 else 1.0

    timesteps_global = scheduler.timesteps[-steps_back_global:] if steps_back_global > 0 else []
    alpha_bar_global = scheduler.alphas_cumprod[timesteps_global[0]] if steps_back_global > 0 else 1.0

    # Freeze gradients for VAE and priors
    vae = lion.vae
    global_prior = lion.priors[0]
    local_prior = lion.priors[1]
    grad_freeze(global_prior)
    grad_freeze(local_prior)
    grad_freeze(vae)
    
    # Latent encoding (STEP 1)
    with torch.no_grad():
        latents = vae.encode(x)
        shape_latent = latents[2][0][0].unsqueeze(2).unsqueeze(3)  # z_0 abstract
        latent_point = latents[2][1][0].unsqueeze(2).unsqueeze(3)  # h_0 abstract
        # Reshape to (B, N, 4) - assuming 2048 points
        h_0 = latent_point.view(num_samples, num_latent_points, -1)
        
        # Pre-compute original full spectral components (H_orig) and eigenvectors (U_o)
        H_orig, U_o = graph_spectral_module(h_0)
    
    # Add initial noise
    if steps_back_global > 0:
        noise_global = torch.randn_like(shape_latent)
        noisy_z = torch.sqrt(alpha_bar_global) * shape_latent + noise_global * torch.sqrt(1 - alpha_bar_global)
    else:
        noisy_z = shape_latent
        
    if steps_back_local > 0:
        noise_local = torch.randn_like(latent_point)
        noisy_h = torch.sqrt(alpha_bar_local) * latent_point + noise_local * torch.sqrt(1 - alpha_bar_local)
    else:
        noisy_h = latent_point
        
    # Reverse diffusion process using DDIMScheduler
    history = {'raw_loss_spectral_low': [], 'raw_loss_spectral_mid': [], 'raw_loss_spectral_high': [], 'raw_loss_invariant': [], 'raw_loss_chamfer': []}
    
    for i, t in enumerate(timesteps_sync):
        t_tensor = torch.ones(num_samples, dtype=torch.int64, device=x.device) * (t + 1)
        
        # --- GLOBAL PRIOR STEP ---
        if steps_back_global > 0 and t <= timesteps_global[0]:
            noise_pred_global = global_prior(x=noisy_z, t=t_tensor.float(), condition_input=None, clip_feat=None)
            scheduler_out_global = scheduler.step(noise_pred_global, t, noisy_z)
            next_noisy_z = scheduler_out_global.prev_sample
        else:
            next_noisy_z = noisy_z
            
        # We want to guide next_noisy_z using the gradients from local prior
        next_noisy_z = next_noisy_z.detach()
        next_noisy_z.requires_grad = True
        
        # Global to Style
        style_cond = vae.global2style(next_noisy_z)
        
        # --- LOCAL PRIOR STEP ---
        if steps_back_local > 0 and t <= timesteps_local[0]:
            noisy_h = noisy_h.detach()
            noisy_h.requires_grad = True
            
            noise_pred_local = local_prior(x=noisy_h, t=t_tensor.float(), condition_input=style_cond, clip_feat=None)
            scheduler_out_local = scheduler.step(noise_pred_local, t, noisy_h)
            next_noisy_h = scheduler_out_local.prev_sample
            h_bar_0 = scheduler_out_local.pred_original_sample.view(num_samples, num_latent_points, -1)
        
        # This is h_bar_0 in the abstract space (if local step ran, otherwise use dummy or skip)
        if steps_back_local > 0 and t <= timesteps_local[0]:
            pass # h_bar_0 already defined
        else:
            # If local step didn't run yet, we don't calculate loss
            h_bar_0 = None
        
        total_loss = 0.0
        
        # STEP 4: Spectral Guidance Loss
        if h_bar_0 is not None:
            with torch.no_grad():
                signal = h_bar_0 if graph_spectral_module.use_4d_gft else h_bar_0[:, :, :3]
            H_pred = torch.bmm(U_o.transpose(1, 2), signal)
            
            raw_loss_spectral_low = F.mse_loss(H_pred[:, :graph_spectral_module.M, :], H_orig[:, :graph_spectral_module.M, :], reduction='mean')
            raw_loss_spectral_mid = F.mse_loss(H_pred[:, graph_spectral_module.M:graph_spectral_module.M_mid, :], H_orig[:, graph_spectral_module.M:graph_spectral_module.M_mid, :], reduction='mean') if graph_spectral_module.M < graph_spectral_module.M_mid else torch.tensor(0.0)
            raw_loss_spectral_high = F.mse_loss(H_pred[:, graph_spectral_module.M_mid:, :], H_orig[:, graph_spectral_module.M_mid:, :], reduction='mean') if graph_spectral_module.M_mid < num_latent_points else torch.tensor(0.0)
            
            power_pred_raw = torch.norm(H_pred[:, :graph_spectral_module.M, :], dim=-1)
            power_orig_raw = torch.norm(H_orig[:, :graph_spectral_module.M, :], dim=-1)
            raw_loss_invariant = F.mse_loss(power_pred_raw, power_orig_raw, reduction='mean')
            
            history['raw_loss_spectral_low'].append(raw_loss_spectral_low.item())
            history['raw_loss_spectral_mid'].append(raw_loss_spectral_mid.item())
            history['raw_loss_spectral_high'].append(raw_loss_spectral_high.item())
            history['raw_loss_invariant'].append(raw_loss_invariant.item())
            
            if chamfer_dist is not None:
                pred_latent_point_reshaped = h_bar_0[:, :, :3]
                h_0_spatial = h_0[:, :, :3]
                dists1, dists2, _, _ = chamfer_dist(pred_latent_point_reshaped, h_0_spatial)
                dists1 = torch.sort(dists1, dim=1).values[:, :int(num_latent_points * p)]
                dists2 = torch.sort(dists2, dim=1).values[:, :int(num_latent_points * p)]
                raw_loss_chamfer = dists1.sum() + dists2.sum()
                history['raw_loss_chamfer'].append(raw_loss_chamfer.item())
            
        if h_bar_0 is not None and (weight_spectral_low > 0.0 or weight_spectral_mid > 0.0 or weight_spectral_high > 0.0 or weight_invariant > 0.0):
            signal = h_bar_0 if graph_spectral_module.use_4d_gft else h_bar_0[:, :, :3]
            H_pred = torch.bmm(U_o.transpose(1, 2), signal)
            
            if weight_invariant > 0.0:
                power_pred = torch.norm(H_pred[:, :graph_spectral_module.M, :], dim=-1)
                power_orig = torch.norm(H_orig[:, :graph_spectral_module.M, :], dim=-1)
                loss_invariant = F.mse_loss(power_pred, power_orig, reduction='mean')
                total_loss = total_loss + weight_invariant * loss_invariant
                
            if weight_spectral_low > 0.0:
                loss_spectral_low = F.mse_loss(H_pred[:, :graph_spectral_module.M, :], H_orig[:, :graph_spectral_module.M, :], reduction='mean')
                total_loss = total_loss + weight_spectral_low * loss_spectral_low
                
            if weight_spectral_mid > 0.0 and graph_spectral_module.M < graph_spectral_module.M_mid:
                loss_spectral_mid = F.mse_loss(H_pred[:, graph_spectral_module.M:graph_spectral_module.M_mid, :], H_orig[:, graph_spectral_module.M:graph_spectral_module.M_mid, :], reduction='mean')
                total_loss = total_loss + weight_spectral_mid * loss_spectral_mid
                
            if weight_spectral_high > 0.0 and graph_spectral_module.M_mid < graph_spectral_module.M_high:
                loss_spectral_high = F.mse_loss(H_pred[:, graph_spectral_module.M_mid:graph_spectral_module.M_high, :], H_orig[:, graph_spectral_module.M_mid:graph_spectral_module.M_high, :], reduction='mean')
                total_loss = total_loss + weight_spectral_high * loss_spectral_high
            
            if weight_chamfer > 0.0:
                pred_latent_point_reshaped = h_bar_0[:, :, :3]
                h_0_spatial = h_0[:, :, :3]
                dists1, dists2, _, _ = chamfer_dist(pred_latent_point_reshaped, h_0_spatial)
                dists1 = torch.sort(dists1, dim=1).values[:, :int(num_latent_points * p)]
                dists2 = torch.sort(dists2, dim=1).values[:, :int(num_latent_points * p)]
                ch_loss = dists1.sum() + dists2.sum()
                total_loss = total_loss + weight_chamfer * ch_loss

        # Gradient Update (Guidance)
        if steps_back_local > 0 and t <= timesteps_local[0]:
            if noisy_h.grad is not None:
                noisy_h.grad.zero_()
            if next_noisy_z.grad is not None:
                next_noisy_z.grad.zero_()
                
            if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                total_loss.backward()
                
                # Update local points
                next_noisy_h = next_noisy_h - eta * noisy_h.grad
                
                # Update global shape (if it was active)
                if steps_back_global > 0 and t <= timesteps_global[0]:
                    next_noisy_z = next_noisy_z - gamma * next_noisy_z.grad
            
            noisy_h = next_noisy_h
            
        noisy_z = next_noisy_z

    # Final Decoding
    final_style = shape_latent if use_static_style else noisy_z
    pred_points = vae.decoder(
        None, beta=None, context=noisy_h.squeeze(3).squeeze(2), 
        style=vae.global2style(final_style).squeeze(3).squeeze(2)
    )
    
    # Calculate means
    mean_spec_low = sum(history['raw_loss_spectral_low']) / len(history['raw_loss_spectral_low']) if history['raw_loss_spectral_low'] else 0.0
    mean_spec_mid = sum(history['raw_loss_spectral_mid']) / len(history['raw_loss_spectral_mid']) if history['raw_loss_spectral_mid'] else 0.0
    mean_spec_high = sum(history['raw_loss_spectral_high']) / len(history['raw_loss_spectral_high']) if history['raw_loss_spectral_high'] else 0.0
    mean_invariant = sum(history['raw_loss_invariant']) / len(history['raw_loss_invariant']) if history['raw_loss_invariant'] else 0.0
    mean_chamfer = sum(history['raw_loss_chamfer']) / len(history['raw_loss_chamfer']) if history['raw_loss_chamfer'] else 0.0
    
    metrics = {
        'mean_raw_spectral_low': mean_spec_low,
        'mean_raw_spectral_mid': mean_spec_mid,
        'mean_raw_spectral_high': mean_spec_high,
        'mean_raw_invariant': mean_invariant,
        'mean_raw_chamfer': mean_chamfer
    }
    
    return pred_points, metrics
