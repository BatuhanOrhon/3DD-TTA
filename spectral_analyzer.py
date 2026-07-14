import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import DDIMScheduler
from graph_spectral import GraphSpectralDNA
from utilities_3dd_tta import grad_freeze
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist as chamfer_grad


class GraphSpectralAnalyzer:
    def __init__(self, diff_model, 
                 # GFT Parameters
                 k=10, delta=0.1, gamma_outlier=0.6, M=100, use_4d_gft=False,
                 # Diffusion/TTA Parameters
                 denoising_step=30, gamma=0.01, eta=0.01, lambdaa=0.95,
                 # Loss Weights
                 weight_spectral=1.0, weight_chamfer=0.0,
                 device="cuda"):
        
        self.diff_model = diff_model
        self.device = device
        
        # Hyperparameters
        self.k = k
        self.delta = delta
        self.gamma_outlier = gamma_outlier
        self.M = M
        self.use_4d_gft = use_4d_gft
        
        self.denoising_step = denoising_step
        self.gamma = gamma
        self.eta = eta
        self.lambdaa = lambdaa
        
        self.weight_spectral = weight_spectral
        self.weight_chamfer = weight_chamfer
        
        # Initialize modules
        self.graph_spectral_module = GraphSpectralDNA(
            k=self.k, delta=self.delta, gamma=self.gamma_outlier, M=self.M, 
            use_4d_gft=self.use_4d_gft, device=self.device
        )
        self.chamfer_dist = chamfer_grad() if chamfer_grad is not None else None
        
        # Logging history
        self.history = {
            'step': [],
            'h_mse': [],
            'h_chamfer': [],
            'spatial_chamfer': []
        }
        
    def run_analysis(self, x):
        """
        Runs the TTA loop on a single sample and logs H matrix distances.
        Uses exact same logic as tta_gsd.py -> tta_gsd_reconstruct.
        """
        self.history = {k: [] for k in self.history} # Reset history
        
        # Initialize chamfer distance
        chamfer_dist = chamfer_grad()
            
        x = x.to(self.device) # [B, 2048, 3]
        num_samples, num_points = x.size()[0], x.size()[1]
        num_latent_points = 2048  # VAE latent space fixed at 2048 points
        
        scheduler = DDIMScheduler(
            beta_end=0.02, beta_schedule="linear", beta_start=0.0001, 
            clip_sample=False, num_train_timesteps=1000, prediction_type="epsilon"
        )
        
        # In tta_gsd.py, total is hardcoded to 100 for set_timesteps.
        total = 100
        scheduler.set_timesteps(total, device=x.device)
        
        # steps_back_local is a percentage of total=100.
        steps_back_local = int((total * self.denoising_step) // 100)
        timesteps_local = scheduler.timesteps[-steps_back_local:]
        alpha_bar_local = scheduler.alphas_cumprod[timesteps_local[0]]

        # Freeze gradients for VAE and local prior
        vae = self.diff_model.vae
        local_prior = self.diff_model.priors[1]
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
            H_orig_low, U_o = self.graph_spectral_module(h_0)
        
        # Global style conditioning
        style_cond = vae.global2style(shape_latent)
        
        # STEP 2: Noise Perturbation
        noise = torch.randn_like(latent_point)
        noisy_latent_point = torch.sqrt(alpha_bar_local) * latent_point + noise * torch.sqrt(1 - alpha_bar_local)
     
        # Reverse diffusion process using DDIMScheduler (STEP 3)
        
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
            
            total_loss = 0.0
            h_bar_0 = pred_latent_point.view(num_samples, num_latent_points, -1)
            
            # STEP 4: Spectral Guidance Loss
            if self.weight_spectral > 0.0:
                signal = h_bar_0 if self.use_4d_gft else h_bar_0[:, :, :3]
                H_pred = torch.bmm(U_o.transpose(1, 2), signal)
                H_pred_low = H_pred[:, :self.M, :]
                
                loss_spectral = F.mse_loss(H_pred_low, H_orig_low, reduction='mean')
                total_loss = total_loss + self.weight_spectral * loss_spectral
                
                # --- LOGGING ---
                with torch.no_grad():
                    d1_h, d2_h, _, _ = chamfer_dist(H_pred_low, H_orig_low)
                    h_chamfer_val = (d1_h.mean() + d2_h.mean()).item()
                    
                    pred_spatial = h_bar_0[:, :, :3]
                    orig_spatial = h_0[:, :, :3]
                    sd1, sd2, _, _ = chamfer_dist(pred_spatial, orig_spatial)
                    sd1 = torch.sort(sd1, dim=1).values[:, :int(num_latent_points * self.lambdaa)]
                    sd2 = torch.sort(sd2, dim=1).values[:, :int(num_latent_points * self.lambdaa)]
                    spatial_ch_val = (sd1.mean() + sd2.mean()).item()
                    
                    self.history['step'].append(i)
                    self.history['h_mse'].append(loss_spectral.item())
                    self.history['h_chamfer'].append(h_chamfer_val)
                    self.history['spatial_chamfer'].append(spatial_ch_val)
                
            # Optional: Original Selective Chamfer Distance (only computed if weight > 0)
            if self.weight_chamfer > 0.0:
                pred_spatial = h_bar_0[:, :, :3]
                orig_spatial = h_0[:, :, :3]
                dists1, dists2, _, _ = chamfer_dist(pred_spatial, orig_spatial)
                dists1 = torch.sort(dists1, dim=1).values[:, :int(num_latent_points * self.lambdaa)]
                dists2 = torch.sort(dists2, dim=1).values[:, :int(num_latent_points * self.lambdaa)]
                ch_loss = dists1.sum() + dists2.sum()
                total_loss = total_loss + self.weight_chamfer * ch_loss

            # STEP 5: Gradient Update (Guidance)
            if noisy_latent_point.grad is not None:
                noisy_latent_point.grad.zero_()
            if style_cond.grad is not None:
                style_cond.grad.zero_()
                
            if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                total_loss.backward()
                
                # Update latent variables with gradient step
                noisy_latent_point = scheduler_output.prev_sample - self.gamma * noisy_latent_point.grad
                style_cond = style_cond - self.eta * style_cond.grad
            else:
                # If no loss was computed or loss is 0 (both weights 0), just proceed with normal DDIM step
                noisy_latent_point = scheduler_output.prev_sample

        return self.history

    def plot_history(self):
        """
        Plots the recorded distances across diffusion steps.
        """
        if not self.history['step']:
            print("No history to plot. Run analysis first with weight_spectral > 0.")
            return
            
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        axs[0].plot(self.history['step'], self.history['h_mse'], marker='o', color='b')
        axs[0].set_title('MSE of H_pred vs H_orig')
        axs[0].set_xlabel('Diffusion Step')
        axs[0].set_ylabel('MSE Loss (Mean)')
        axs[0].grid(True)
        
        axs[1].plot(self.history['step'], self.history['h_chamfer'], marker='o', color='r')
        axs[1].set_title('Chamfer Dist of H_pred vs H_orig')
        axs[1].set_xlabel('Diffusion Step')
        axs[1].set_ylabel('Chamfer Distance (Mean)')
        axs[1].grid(True)
        
        axs[2].plot(self.history['step'], self.history['spatial_chamfer'], marker='o', color='g')
        axs[2].set_title('Spatial Chamfer of Pred vs Orig')
        axs[2].set_xlabel('Diffusion Step')
        axs[2].set_ylabel('Selective Chamfer Dist (Mean)')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()

