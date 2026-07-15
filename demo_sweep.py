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
    parser = argparse.ArgumentParser(description='GSDTTA Sweep Analyzer')
    parser.add_argument('--diff_config', type=str, default="./lion_ckpts/unconditional_all55_cfg.yml")
    parser.add_argument('--diff_ckpt', type=str, default="./lion_ckpts/epoch_10999_iters_2100999.pt")
    parser.add_argument('--dataset_root', type=str, default="./data/modelnet40_c")
    parser.add_argument('--corruption', type=str, default="background")
    parser.add_argument('--sample_id', type=int, default=11)
    parser.add_argument('--pc_path', type=str, default=None, help="Direct path to a .npy point cloud file (overrides dataset_root)")
    
    # Sweep setup
    parser.add_argument('--sweep_param', type=str, default="weight_spectral", help="Which parameter to sweep")
    parser.add_argument('--sweep_values', type=float, nargs='+', default=[0.1, 1.0, 10.0, 50.0], help="Values to test")
    
    # Base Analyzer params
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
    
    parser.add_argument('--output_img', type=str, default="sweep_plot.png")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    diff_config.merge_from_file(args.diff_config)
    diff_model = LION(diff_config)
    diff_model.load_model(args.diff_ckpt)
    
    if args.pc_path:
        data = np.load(args.pc_path)
        if len(data.shape) == 3:
            data = data[0]
    else:
        data_path = os.path.join(args.dataset_root, f"data_{args.corruption}_5.npy")
        points_array = np.load(data_path)
        data = points_array[args.sample_id]
        
    data_sample = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    
    from utilities_3dd_tta import upsample_all, rotate_pointcloud, normalize, rotateback_pointcloud
    data_sample, _, _ = normalize(data_sample)
        
    data_sample = upsample_all(data_sample.numpy(), 2048)
    data_sample = torch.from_numpy(data_sample).float().to(device)
    data_sample = 3.3885 * data_sample
    data_sample = rotate_pointcloud(data_sample)
    
    base_params = {
        "k": args.k,
        "delta": args.delta,
        "gamma_outlier": args.gamma_outlier,
        "M": args.M,
        "use_4d_gft": args.use_4d_gft,
        "denoising_step": args.denoising_step,
        "gamma": args.gamma,
        "eta": args.eta,
        "lambdaa": args.lambdaa,
        "weight_spectral": args.weight_spectral,
        "weight_chamfer": args.weight_chamfer,
        "device": device
    }
    
    results = {}
    print(f"Starting Sweep for parameter '{args.sweep_param}' with values: {args.sweep_values}")
    
    for val in args.sweep_values:
        current_params = base_params.copy()
        
        # Cast to int if needed (e.g. M, k)
        if args.sweep_param in ["M", "k", "denoising_step"]:
            val = int(val)
            
        current_params[args.sweep_param] = val
        print(f"\n--- Testing: {args.sweep_param} = {val} ---")
        
        analyzer = GraphSpectralAnalyzer(diff_model=diff_model, **current_params)
        _, pred_points = analyzer.run_analysis(data_sample)
        
        # Post-process the final point cloud to real space
        pred_points = rotateback_pointcloud(pred_points)
        pred_points, _, _ = normalize(pred_points)
        final_points = pred_points.cpu().squeeze().detach().numpy()
        
        results[val] = {
            "step": analyzer.history["step"],
            "spatial_chamfer": analyzer.history["spatial_chamfer"],
            "f_score": analyzer.history["f_score"],
            "points": final_points
        }
        
        # Loss Scale Logging for this sweep value
        if analyzer.history['raw_loss_spectral']:
            mean_spec = sum(analyzer.history['raw_loss_spectral']) / len(analyzer.history['raw_loss_spectral'])
            mean_chamfer = sum(analyzer.history['raw_loss_chamfer']) / len(analyzer.history['raw_loss_chamfer'])
            print(f"   -> Avg Raw Spectral Loss: {mean_spec:.6f} | Avg Raw Chamfer Loss: {mean_chamfer:.6f}")
            print(f"   -> Suggested Weight Ratio (Chamfer/Spectral): {(mean_spec / (mean_chamfer + 1e-8)):.4f}")
        
    num_vals = len(results)
    fig = plt.figure(figsize=(6 * num_vals, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, num_vals * 2, figure=fig)
    
    # 1. Plot the 2D Line Graphs (Top row)
    # Left: Spatial Chamfer
    ax_chamfer = fig.add_subplot(gs[0, :num_vals])
    for val, hist in results.items():
        ax_chamfer.plot(hist["step"], hist["spatial_chamfer"], marker="o", label=f"{args.sweep_param}={val}")

    ax_chamfer.set_title(f"Effect of {args.sweep_param} on Spatial Chamfer Distance")
    ax_chamfer.set_xlabel("Diffusion Step")
    ax_chamfer.set_ylabel("Selective Chamfer Distance")
    ax_chamfer.legend()
    ax_chamfer.grid(True)
    
    # Right: F-Score
    ax_fscore = fig.add_subplot(gs[0, num_vals:])
    for val, hist in results.items():
        ax_fscore.plot(hist["step"], hist["f_score"], marker="o", label=f"{args.sweep_param}={val}")

    ax_fscore.set_title(f"Effect of {args.sweep_param} on F-Score")
    ax_fscore.set_xlabel("Diffusion Step")
    ax_fscore.set_ylabel("F-Score (Threshold=0.05)")
    ax_fscore.legend()
    ax_fscore.grid(True)
    
    # 2. Plot the 3D Point Clouds (Bottom row)
    for i, (val, hist) in enumerate(results.items()):
        ax_3d = fig.add_subplot(gs[1, i*2 : (i+1)*2], projection='3d')
        pts = hist["points"]
        ax_3d.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=2, c='b', alpha=0.6)
        ax_3d.set_title(f"val: {val}")
        ax_3d.set_axis_off()
        
    plt.tight_layout()
    plt.savefig(args.output_img)
    print(f"\nPlot saved to {args.output_img}")

if __name__ == "__main__":
    main()
