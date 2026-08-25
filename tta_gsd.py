import torch
import torch.nn.functional as F
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist as chamfer_grad
from diffusers import DDIMScheduler
from utilities_3dd_tta import grad_freeze

def tta_gsd_reconstruct(x, lion, graph_spectral_module, steps_back_local, gamma, eta, p, loss_weights=None, total=100, use_static_style=False, dynamic_graph=False):
    """
    Test-Time Adaptation (TTA) reconstruction using DDIMScheduler with Graph Spectral (and optional Chamfer) guidance.

    Args:
    - x: Input point cloud data (B, N, 3).
    - lion: Model instance containing VAE and local prior.
    - graph_spectral_module: Instance of GraphSpectralDNA for spectral computations.
    - steps_back_local: Percentage of total steps to use in reverse scheduling (e.g., 5).
    - gamma: Step size for updating latent points (noisy_local / h_t).
    - eta: Step size for updating shape latent (style_cond / z_0).
    - p: Proportion of points to consider in Chamfer Distance.
    - loss_weights: Dictionary of loss weights. Default: {"spectral": 1.0, "chamfer": 0.0}.
    - total: Total number of diffusion steps (default: 100).

    Returns:
    - pred_points: Reconstructed point cloud.
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
    timesteps_local = scheduler.timesteps[-steps_back_local:]
    alpha_bar_local = scheduler.alphas_cumprod[timesteps_local[0]]

    # Freeze gradients for VAE and local prior
    vae = lion.vae
    local_prior = lion.priors[1]
    grad_freeze(local_prior)
    grad_freeze(vae)
    
    # Latent encoding (STEP 1)
    with torch.no_grad():
        latents = vae.encode(x)
        shape_latent = latents[2][0][0].unsqueeze(2).unsqueeze(3)  # z_0 abstract
        latent_point = latents[2][1][0].unsqueeze(2).unsqueeze(3)  # h_0 abstract
        # Reshape to (B, N, 4) safely
        latent_point_squeezed = latent_point.squeeze(3).squeeze(2) # (B, 4, 2048)
        if latent_point_squeezed.shape[-1] == num_latent_points:
            h_0 = latent_point_squeezed.transpose(1, 2).contiguous() # (B, 2048, 4)
        else:
            h_0 = latent_point.view(num_samples, num_latent_points, -1)
        
        # Pre-compute original full spectral components (H_orig) and eigenvectors (U_o)
        H_orig, U_o = graph_spectral_module(h_0)
    
    # Global style conditioning
    style_cond = vae.global2style(shape_latent)
    
    # STEP 2: Noise Perturbation
    noise = torch.randn_like(latent_point)
    noisy_latent_point = torch.sqrt(alpha_bar_local) * latent_point + noise * torch.sqrt(1 - alpha_bar_local)
 
    # Reverse diffusion process using DDIMScheduler (STEP 3)
    history = {'raw_loss_spectral_low': [], 'raw_loss_spectral_mid': [], 'raw_loss_spectral_high': [], 'raw_loss_invariant': [], 'raw_loss_chamfer': []}
    for i, t in enumerate(timesteps_local):
        t_tensor = torch.ones(num_samples, dtype=torch.int64, device=x.device) * (t + 1)
        
        noisy_latent_point = noisy_latent_point.detach()
        noisy_latent_point.requires_grad = True
        
        style_cond = style_cond.detach()
        style_cond.requires_grad = True

        # Predict noise
        noise_pred = local_prior(x=noisy_latent_point, t=t_tensor.float(), condition_input=style_cond, clip_feat=None)
        scheduler_output = scheduler.step(noise_pred, t, noisy_latent_point)
        
        # This is h_bar_0 in the abstract space
        pred_latent_point = scheduler_output.pred_original_sample
        h_bar_0 = pred_latent_point.view(num_samples, num_latent_points, -1)
        
        total_loss = 0.0
        
        # STEP 4: Spectral Guidance Loss
        with torch.no_grad():
            if dynamic_graph:
                # Dinamik U_current hesapla
                _, U_current = graph_spectral_module(h_bar_0)
                # Orijinal gürültülü datayı (h_0), YENİ eksenlere (U_current) izdüşür
                signal_orig = h_0 if graph_spectral_module.use_4d_gft else h_0[:, :, :3]
                H_orig_target = torch.bmm(U_current.transpose(1, 2), signal_orig).detach()
                U_active = U_current
            else:
                H_orig_target = H_orig
                U_active = U_o

            signal = h_bar_0 if graph_spectral_module.use_4d_gft else h_bar_0[:, :, :3]
            H_pred = torch.bmm(U_active.transpose(1, 2), signal)
            
            raw_loss_spectral_low = F.mse_loss(H_pred[:, :graph_spectral_module.M, :], H_orig_target[:, :graph_spectral_module.M, :], reduction='sum')
            raw_loss_spectral_mid = F.mse_loss(H_pred[:, graph_spectral_module.M:graph_spectral_module.M_mid, :], H_orig_target[:, graph_spectral_module.M:graph_spectral_module.M_mid, :], reduction='sum') if graph_spectral_module.M < graph_spectral_module.M_mid else torch.tensor(0.0)
            raw_loss_spectral_high = F.mse_loss(H_pred[:, graph_spectral_module.M_mid:graph_spectral_module.M_high, :], H_orig_target[:, graph_spectral_module.M_mid:graph_spectral_module.M_high, :], reduction='sum') if graph_spectral_module.M_mid < graph_spectral_module.M_high else torch.tensor(0.0)
            
            power_pred_raw = torch.norm(H_pred[:, :graph_spectral_module.M, :], dim=-1)
            power_orig_raw = torch.norm(H_orig_target[:, :graph_spectral_module.M, :], dim=-1)
            raw_loss_invariant = F.mse_loss(power_pred_raw, power_orig_raw, reduction='sum')
            
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
            
        if weight_spectral_low > 0.0 or weight_spectral_mid > 0.0 or weight_spectral_high > 0.0 or weight_invariant > 0.0:
            if dynamic_graph:
                _, U_current = graph_spectral_module(h_bar_0)
                signal_orig = h_0 if graph_spectral_module.use_4d_gft else h_0[:, :, :3]
                H_orig_target = torch.bmm(U_current.transpose(1, 2), signal_orig).detach()
                U_active = U_current
            else:
                H_orig_target = H_orig
                U_active = U_o

            signal = h_bar_0 if graph_spectral_module.use_4d_gft else h_bar_0[:, :, :3]
            H_pred = torch.bmm(U_active.transpose(1, 2), signal)
            
            if weight_invariant > 0.0:
                power_pred = torch.norm(H_pred[:, :graph_spectral_module.M, :], dim=-1)
                power_orig = torch.norm(H_orig_target[:, :graph_spectral_module.M, :], dim=-1)
                loss_invariant = F.mse_loss(power_pred, power_orig, reduction='sum')
                total_loss = total_loss + weight_invariant * loss_invariant
                
            if weight_spectral_low > 0.0:
                loss_spectral_low = F.mse_loss(H_pred[:, :graph_spectral_module.M, :], H_orig_target[:, :graph_spectral_module.M, :], reduction='sum')
                total_loss = total_loss + weight_spectral_low * loss_spectral_low
                
            if weight_spectral_mid > 0.0 and graph_spectral_module.M < graph_spectral_module.M_mid:
                loss_spectral_mid = F.mse_loss(H_pred[:, graph_spectral_module.M:graph_spectral_module.M_mid, :], H_orig_target[:, graph_spectral_module.M:graph_spectral_module.M_mid, :], reduction='sum')
                total_loss = total_loss + weight_spectral_mid * loss_spectral_mid
                
            if weight_spectral_high > 0.0 and graph_spectral_module.M_mid < graph_spectral_module.M_high:
                loss_spectral_high = F.mse_loss(H_pred[:, graph_spectral_module.M_mid:graph_spectral_module.M_high, :], H_orig_target[:, graph_spectral_module.M_mid:graph_spectral_module.M_high, :], reduction='sum')
                total_loss = total_loss + weight_spectral_high * loss_spectral_high
            
        if weight_chamfer > 0.0:
            pred_latent_point_reshaped = h_bar_0[:, :, :3]
            h_0_spatial = h_0[:, :, :3]
            dists1, dists2, _, _ = chamfer_dist(pred_latent_point_reshaped, h_0_spatial)
            dists1 = torch.sort(dists1, dim=1).values[:, :int(num_latent_points * p)]
            dists2 = torch.sort(dists2, dim=1).values[:, :int(num_latent_points * p)]
            ch_loss = dists1.sum() + dists2.sum()
            total_loss = total_loss + weight_chamfer * ch_loss

        # STEP 5: Gradient Update (Guidance)
        if noisy_latent_point.grad is not None:
            noisy_latent_point.grad.zero_()
        if style_cond.grad is not None:
            style_cond.grad.zero_()
            
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            
            # Update latent variables with gradient step
            noisy_latent_point = scheduler_output.prev_sample - eta * noisy_latent_point.grad
            style_cond = style_cond - gamma * style_cond.grad
        else:
            # If no loss was computed or loss is 0 (both weights 0), just proceed with normal DDIM step
            noisy_latent_point = scheduler_output.prev_sample

    # STEP 6: Final Decoding
    final_style = shape_latent if use_static_style else style_cond
    pred_points = vae.decoder(
        None, beta=None, context=noisy_latent_point.squeeze(3).squeeze(2), 
        style=final_style.squeeze(3).squeeze(2)
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
