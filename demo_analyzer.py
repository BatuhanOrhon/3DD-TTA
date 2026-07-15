import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py

from utils_mate.config import *
from default_config import cfg as diff_config
from models.lion import LION
from spectral_analyzer import GraphSpectralAnalyzer

def main():
    parser = argparse.ArgumentParser(description='GSDTTA Analyzer')
    parser.add_argument('--diff_config', type=str, default="./lion_ckpts/unconditional_all55_cfg.yml")
    parser.add_argument('--diff_ckpt', type=str, default="./lion_ckpts/epoch_10999_iters_2100999.pt")
    parser.add_argument('--dataset_root', type=str, default="./data/modelnet40_c")
    parser.add_argument('--corruption', type=str, default="background")
    parser.add_argument('--sample_id', type=int, default=11)
    parser.add_argument('--pc_path', type=str, default=None, help="Direct path to a .npy point cloud file (overrides dataset_root)")
    
    # Analyzer params
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--gamma_outlier', type=float, default=0.6)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--use_4d_gft', action='store_true')
    parser.add_argument('--denoising_step', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--lambdaa', type=float, default=0.95)
    parser.add_argument('--weight_spectral', type=float, default=1.0)
    parser.add_argument('--weight_chamfer', type=float, default=0.0)
    
    parser.add_argument('--output_img', type=str, default="analyzer_plot.png")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    diff_config.merge_from_file(args.diff_config)
    diff_model = LION(diff_config)
    diff_model.load_model(args.diff_ckpt)
    
    # 2. Load Data
    if args.pc_path:
        data = np.load(args.pc_path)
        if len(data.shape) == 3: # If [B, N, 3], take first
            data = data[0]
    else:
        data_path = os.path.join(args.dataset_root, f"data_{args.corruption}_5.npy")
        points_array = np.load(data_path)
        data = points_array[args.sample_id]
        
    data_sample = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    
    from utilities_3dd_tta import upsample_all, rotate_pointcloud, normalize
    data_sample, _, _ = normalize(data_sample)
        
    data_sample = upsample_all(data_sample.numpy(), 2048)
    data_sample = torch.from_numpy(data_sample).float().to(device)
    data_sample = 3.3885 * data_sample # normalization
    data_sample = rotate_pointcloud(data_sample)
    
    # 3. Setup Analyzer
    analyzer = GraphSpectralAnalyzer(
        diff_model=diff_model,
        k=args.k,
        delta=args.delta,
        gamma_outlier=args.gamma_outlier,
        M=args.M,
        use_4d_gft=args.use_4d_gft,
        denoising_step=args.denoising_step,
        gamma=args.gamma,
        eta=args.eta,
        lambdaa=args.lambdaa,
        weight_spectral=args.weight_spectral,
        weight_chamfer=args.weight_chamfer,
        device=device
    )
    
    # 4. Run Analysis
    print(f"Running Spectral Analysis on sample {args.sample_id}...")
    _, pred_points = analyzer.run_analysis(data_sample)
    
    # Post-process the final point cloud to real space
    from utilities_3dd_tta import rotateback_pointcloud
    pred_points = rotateback_pointcloud(pred_points)
    pred_points, _, _ = normalize(pred_points)
    final_points = pred_points.cpu().squeeze().detach().numpy()
    
    # 5. Loss Scale Logging
    if analyzer.history['raw_loss_spectral']:
        mean_spec = sum(analyzer.history['raw_loss_spectral']) / len(analyzer.history['raw_loss_spectral'])
        mean_chamfer = sum(analyzer.history['raw_loss_chamfer']) / len(analyzer.history['raw_loss_chamfer'])
        print("\n--- Loss Scale Analysis ---")
        print(f"Average Raw Spectral Loss: {mean_spec:.6f}")
        print(f"Average Raw Chamfer Loss:  {mean_chamfer:.6f}")
        print(f"Suggested Weight Ratio (Chamfer/Spectral) to balance: {(mean_spec / (mean_chamfer + 1e-8)):.4f}")
        print("---------------------------\n")
    
    # 6. Plot and Save
    if not analyzer.history['step']:
        print("No history to plot.")
        return
        
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(24, 10))
    gs = GridSpec(2, 4, figure=fig)
    
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    axs[0].plot(analyzer.history['step'], analyzer.history['h_mse'], marker='o', color='b')
    axs[0].set_title('MSE of H_pred vs H_orig (Spectral)')
    axs[0].set_xlabel('Diffusion Step')
    axs[0].set_ylabel('MSE Loss (Mean)')
    axs[0].grid(True)
    
    axs[1].plot(analyzer.history['step'], analyzer.history['h_chamfer'], marker='o', color='r')
    axs[1].set_title('Chamfer Dist of H_pred vs H_orig')
    axs[1].set_xlabel('Diffusion Step')
    axs[1].set_ylabel('Chamfer Distance (Mean)')
    axs[1].grid(True)
    
    axs[2].plot(analyzer.history['step'], analyzer.history['spatial_chamfer'], marker='o', color='g')
    axs[2].set_title('Spatial Chamfer (Selective)')
    axs[2].set_xlabel('Diffusion Step')
    axs[2].set_ylabel('Chamfer Dist (Mean)')
    axs[2].grid(True)
    
    axs[3].plot(analyzer.history['step'], analyzer.history['f_score'], marker='o', color='purple')
    axs[3].set_title('F-Score (Threshold=0.05)')
    axs[3].set_xlabel('Diffusion Step')
    axs[3].set_ylabel('F-Score')
    axs[3].grid(True)
    
    # 3D plot on bottom row centered
    ax_3d = fig.add_subplot(gs[1, 1:3], projection='3d')
    ax_3d.scatter(final_points[:, 0], final_points[:, 2], final_points[:, 1], s=2, c='b', alpha=0.6)
    ax_3d.set_title("Final Reconstructed Point Cloud")
    ax_3d.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(args.output_img)
    print(f"Plot saved to {args.output_img}")

if __name__ == "__main__":
    main()
