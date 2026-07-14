import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import DDIMScheduler
from graph_spectral import GraphSpectralDNA
from utilities_3dd_tta import grad_freeze

try:
    from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist as chamfer_grad
except ImportError:
    print("Warning: ChamferDistancePytorch not found.")
    chamfer_grad = None


class GraphSpectralAnalyzer:
    def __init__(self, diff_model, 
                 # GFT Parameters
                 k=10, delta=0.1, gamma_outlier=0.6, M=100, use_4d_gft=False,
                 # Diffusion/TTA Parameters
                 denoising_step=30, gamma_lr=1000, eta_lr=20, lambdaa=0.95,
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
        self.gamma_lr = gamma_lr
        self.eta_lr = eta_lr
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
        """
        self.history = {k: [] for k in self.history} # Reset history
        
        vae = self.diff_model.vae
        global_prior = self.diff_model.priors[0]
        local_prior = self.diff_model.priors[1]
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        scheduler.set_timesteps(1000)
        
        grad_freeze(vae)
        grad_freeze(local_prior)
        
        x = x.to(self.device).unsqueeze(0)  # [1, 2048, 3]
        
        with torch.no_grad():
            latents = vae.encode(x)
            latent_point = latents[2][1][0]
            shape_latent = latents[0]
            
            # TODO: Original author's logic for shape_latent vs style_cond
            style_cond = vae.global2style(shape_latent)
            
            num_samples = latent_point.shape[0]
            latent_point_reshaped = latent_point.view(num_samples, 2048, -1)
            h_0_spatial = latent_point_reshaped[:, :, :3]
            
            # Precompute GFT on corrupted input
            H_orig_low, U_o = self.graph_spectral_module(latent_point_reshaped)
            
        t = scheduler.timesteps[1000 - self.denoising_step]
        t_tensor = torch.tensor([t], device=self.device)
        alpha_bar_local = scheduler.alphas_cumprod[t].to(self.device)
        
        noise = torch.randn_like(latent_point)
        noisy_latent_point = torch.sqrt(alpha_bar_local) * latent_point + noise * torch.sqrt(1 - alpha_bar_local)
        
        noisy_latent_point = noisy_latent_point.detach()
        style_cond = style_cond.detach()
        
        # Diffusion Loop
        pbar = tqdm(range(self.denoising_step), desc="GSDTTA Analysis")
        for step in pbar:
            noisy_latent_point.requires_grad = True
            style_cond.requires_grad = True
            
            noise_pred = local_prior(x=noisy_latent_point, t=t_tensor.float(), condition_input=style_cond, clip_feat=None)
            scheduler_output = scheduler.step(noise_pred, t, noisy_latent_point, eta=0.0)
            pred_latent_point = scheduler_output.pred_original_sample
            
            total_loss = torch.tensor(0.0, device=self.device)
            
            h_bar_0 = pred_latent_point.view(num_samples, 2048, -1)
            pred_spatial = h_bar_0[:, :, :3]
            
            # --- SPECTRAL LOSS & LOGGING ---
            if self.weight_spectral > 0.0:
                signal = h_bar_0 if self.use_4d_gft else pred_spatial
                H_pred = torch.bmm(U_o.transpose(1, 2), signal)
                H_pred_low = H_pred[:, :self.M, :]
                
                loss_spectral = F.mse_loss(H_pred_low, H_orig_low, reduction='mean')
                total_loss = total_loss + self.weight_spectral * loss_spectral
                
                # LOGGING: Record MSE and Chamfer on H matrices
                with torch.no_grad():
                    mse_val = loss_spectral.item()
                    
                    if self.chamfer_dist is not None:
                        d1, d2, _, _ = self.chamfer_dist(H_pred_low, H_orig_low)
                        ch_h_val = (d1.mean() + d2.mean()).item()
                        
                        # Also track Spatial Chamfer
                        sd1, sd2, _, _ = self.chamfer_dist(pred_spatial, h_0_spatial)
                        num_points = 2048
                        sd1 = torch.sort(sd1, dim=1).values[:, :int(num_points * self.lambdaa)]
                        sd2 = torch.sort(sd2, dim=1).values[:, :int(num_points * self.lambdaa)]
                        spatial_ch_val = (sd1.mean() + sd2.mean()).item()
                    else:
                        ch_h_val = 0.0
                        spatial_ch_val = 0.0
                        
                    self.history['step'].append(step)
                    self.history['h_mse'].append(mse_val)
                    self.history['h_chamfer'].append(ch_h_val)
                    self.history['spatial_chamfer'].append(spatial_ch_val)
            
            # --- OPTIONAL CHAMFER LOSS ---
            if self.weight_chamfer > 0.0 and self.chamfer_dist is not None:
                d1, d2, _, _ = self.chamfer_dist(pred_spatial, h_0_spatial)
                num_points = 2048
                d1 = torch.sort(d1, dim=1).values[:, :int(num_points * self.lambdaa)]
                d2 = torch.sort(d2, dim=1).values[:, :int(num_points * self.lambdaa)]
                ch_loss = d1.sum() + d2.sum()
                total_loss = total_loss + self.weight_chamfer * ch_loss
            
            # --- GRADIENT UPDATE ---
            if total_loss > 0.0:
                if noisy_latent_point.grad is not None:
                    noisy_latent_point.grad.zero_()
                if style_cond.grad is not None:
                    style_cond.grad.zero_()
                    
                total_loss.backward()
                
                noisy_latent_point = scheduler_output.prev_sample - self.gamma_lr * noisy_latent_point.grad
                style_cond = style_cond - self.eta_lr * style_cond.grad
            else:
                noisy_latent_point = scheduler_output.prev_sample
                
            noisy_latent_point = noisy_latent_point.detach()
            style_cond = style_cond.detach()
            
            t -= 1000 // self.denoising_step
            t_tensor = torch.tensor([t], device=self.device)
            
        with torch.no_grad():
            final_pred_points = vae.decoder(pred_latent_point, style_cond)
            
        return final_pred_points[0]

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

