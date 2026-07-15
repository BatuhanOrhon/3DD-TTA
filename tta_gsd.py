import torch
import torch.nn.functional as F
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist as chamfer_grad
from diffusers import DDIMScheduler
from utilities_3dd_tta import grad_freeze

def tta_gsd_reconstruct(x, lion, graph_spectral_module, steps_back_local, gamma, eta, p, loss_weights=None, total=100):
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
        loss_weights = {"spectral": 1.0, "chamfer": 0.0}
        
    weight_spectral = loss_weights.get("spectral", 1.0)
    weight_chamfer = loss_weights.get("chamfer", 0.0)

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
        # Reshape to (B, N, 4) - assuming 2048 points
        h_0 = latent_point.view(num_samples, num_latent_points, -1)
        
        # Pre-compute original low-frequency spectral components (H_orig_low) and eigenvectors (U_o)
        H_orig_low, U_o = graph_spectral_module(h_0)
    
    # Global style conditioning
    style_cond = vae.global2style(shape_latent)
    
    # STEP 2: Noise Perturbation
    noise = torch.randn_like(latent_point)
    noisy_latent_point = torch.sqrt(alpha_bar_local) * latent_point + noise * torch.sqrt(1 - alpha_bar_local)
 
    # Reverse diffusion process using DDIMScheduler (STEP 3)
    history = {'raw_loss_spectral': [], 'raw_loss_chamfer': []}
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
            signal = h_bar_0 if graph_spectral_module.use_4d_gft else h_bar_0[:, :, :3]
            H_pred = torch.bmm(U_o.transpose(1, 2), signal)
            H_pred_low = H_pred[:, :graph_spectral_module.M, :]
            raw_loss_spectral = F.mse_loss(H_pred_low, H_orig_low, reduction='mean')
            
            pred_latent_point_reshaped = h_bar_0[:, :, :3]
            h_0_spatial = h_0[:, :, :3]
            dists1, dists2, _, _ = chamfer_dist(pred_latent_point_reshaped, h_0_spatial)
            dists1 = torch.sort(dists1, dim=1).values[:, :int(num_latent_points * p)]
            dists2 = torch.sort(dists2, dim=1).values[:, :int(num_latent_points * p)]
            raw_loss_chamfer = dists1.sum() + dists2.sum()
            
            # Keep track for logging
            history['raw_loss_spectral'].append(raw_loss_spectral.item())
            history['raw_loss_chamfer'].append(raw_loss_chamfer.item())
            
        if weight_spectral > 0.0:
            signal = h_bar_0 if graph_spectral_module.use_4d_gft else h_bar_0[:, :, :3]
            H_pred = torch.bmm(U_o.transpose(1, 2), signal)
            H_pred_low = H_pred[:, :graph_spectral_module.M, :]
            loss_spectral = F.mse_loss(H_pred_low, H_orig_low, reduction='mean')
            total_loss = total_loss + weight_spectral * loss_spectral
            
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
    # TODO: Bu mantık incelenecek. Yazarlar burada shape_latent kullanmış ancak optimizasyon döngüsünde style_cond (z_0) güncelleniyor. shape_latent yerine style_cond mu gelmeli kontrol edilecek.
    pred_points = vae.decoder(
        None, beta=None, context=noisy_latent_point.squeeze(3).squeeze(2), 
        style=shape_latent.squeeze(3).squeeze(2)
    )
    
    # Calculate means
    mean_spec = sum(history['raw_loss_spectral']) / len(history['raw_loss_spectral']) if history['raw_loss_spectral'] else 0.0
    mean_chamfer = sum(history['raw_loss_chamfer']) / len(history['raw_loss_chamfer']) if history['raw_loss_chamfer'] else 0.0
    
    metrics = {
        'mean_raw_spectral': mean_spec,
        'mean_raw_chamfer': mean_chamfer
    }
    
    return pred_points, metrics
