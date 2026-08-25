import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lion import LION
from default_config import cfg as configs
from utilities_3dd_tta import PointDataset, normalize, upsample_all

try:
    from knn_cuda import KNN
except ImportError:
    print("Warning: knn_cuda not installed. Using PyTorch fallback.")
    class KNN(torch.nn.Module):
        def __init__(self, k, transpose_mode=True):
            super().__init__()
            self.k = k
            self.transpose_mode = transpose_mode
        def forward(self, ref, query):
            if self.transpose_mode:
                ref = ref.transpose(1, 2)
                query = query.transpose(1, 2)
            dists = torch.cdist(query, ref)
            topk_dists, topk_idx = torch.topk(dists, self.k, dim=-1, largest=False)
            return topk_dists, topk_idx

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load diffusion model (LION)
    configs.merge_from_file("./lion_ckpts/unconditional_all55_cfg.yml")
    diff_model = LION(configs)
    
    # Check if checkpoint exists
    import os
    ckpt_path = "./lion_ckpts/epoch_10999_iters_2100999.pt"
    if os.path.exists(ckpt_path):
        diff_model.load_model(ckpt_path)
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Using untrained VAE weights for testing shape issues.")
        
    vae = diff_model.vae
    vae.eval()
    diff_model.priors.eval()
    
    # Dataset (we'll use uniform noise for testing)
    corruption = "uniform"
    try:
        dataset = PointDataset("./data/scanobjectnn_c", "./data/scanobjectnn_c/label.npy", corruption)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    except Exception as e:
        print(f"Could not load dataset. Make sure paths are correct. Error: {e}")
        return
    
    total_iou = 0.0
    total_samples = 0
    shape_printed = False
    
    print("Testing KNN overlap between Spatial Space and Latent Space...")
    for data, label in tqdm(dataloader):
        # Normalization steps as in main
        data_sample, _, _ = normalize(data)
        data_sample = upsample_all(data_sample.numpy(), 2048)
        x = torch.from_numpy(data_sample).float().to(device)
        
        B, N, _ = x.shape
        k = 10
        
        with torch.no_grad():
            # Spatial KNN
            dists_spatial = torch.cdist(x, x)
            _, idx_spatial = torch.topk(dists_spatial, k, dim=-1, largest=False)
            
            # Encode to get latent
            latents = vae.encode(x)
            
            if not shape_printed:
                print(f"\n[DEBUG] Input x shape: {x.shape}")
                print(f"[DEBUG] latents[2][1][0] shape: {latents[2][1][0].shape}")
                shape_printed = True
                
            latent_point = latents[2][1][0].unsqueeze(2).unsqueeze(3)
            
            # This is how tta_gsd.py currently reshapes:
            h_0_bad_view = latent_point.view(B, N, -1)
            h_0_spatial_bad = h_0_bad_view[:, :, :3].contiguous()
            
            # This is the mathematically correct way to reshape [B, C, N] to [B, N, C]:
            if latents[2][1][0].shape[-1] == N:
                h_0_correct = latents[2][1][0].transpose(1, 2).contiguous() # [B, N, C]
                h_0_spatial_correct = h_0_correct[:, :, :3].contiguous()
            else:
                h_0_correct = h_0_bad_view
                h_0_spatial_correct = h_0_spatial_bad

            # Latent KNN with the current method
            dists_latent = torch.cdist(h_0_spatial_bad, h_0_spatial_bad)
            _, idx_latent = torch.topk(dists_latent, k, dim=-1, largest=False)
            
            # Latent KNN with correct transpose
            dists_latent_corr = torch.cdist(h_0_spatial_correct, h_0_spatial_correct)
            _, idx_latent_corr = torch.topk(dists_latent_corr, k, dim=-1, largest=False)
            
        # Compute IoU for each point
        for b in range(B):
            for n in range(N):
                set_spatial = set(idx_spatial[b, n].tolist())
                set_latent = set(idx_latent[b, n].tolist())
                set_latent_corr = set(idx_latent_corr[b, n].tolist())
                
                # Check IoU with the 'bad view' method currently in tta_gsd.py
                intersection = len(set_spatial.intersection(set_latent))
                union = len(set_spatial.union(set_latent))
                iou = intersection / union if union > 0 else 0
                
                total_iou += iou
                total_samples += 1
                
        # Optional: Run for 2 batches to get an estimate
        if total_samples >= B * N * 2: 
            break
            
    avg_iou = total_iou / total_samples
    print(f"\nAverage IoU between Physical Space and Latent Space (Current Code): {avg_iou:.4f}")
    
    if avg_iou < 0.5:
        print("\n[WARNING] The overlap is very low. This means the KNN graph built in the latent space does NOT represent the actual geometric structure of the point cloud. The Graph Laplacian is analyzing a scrambled geometry.")
        
    print("\nNote: Please run this script using `python test_latent_knn.py` to see the results.")

if __name__ == "__main__":
    main()
