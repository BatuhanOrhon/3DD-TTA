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
    parser.add_argument('--gamma_lr', type=float, default=1000.0)
    parser.add_argument('--eta_lr', type=float, default=20.0)
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
        
    data_sample = torch.tensor(data, dtype=torch.float32)
    
    from utils.vis_helper import upsample_all
    def rotate_pointcloud(pointcloud):
        theta = np.pi / 2
        rot_matrix = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta), np.cos(theta), 0],
                                   [0, 0, 1]], dtype=torch.float32).to(pointcloud.device)
        return torch.matmul(pointcloud, rot_matrix)
        
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
        "gamma_lr": args.gamma_lr,
        "eta_lr": args.eta_lr,
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
        _ = analyzer.run_analysis(data_sample)
        
        results[val] = {
            "step": analyzer.history["step"],
            "spatial_chamfer": analyzer.history["spatial_chamfer"]
        }
        
    plt.figure(figsize=(10, 6))
    for val, hist in results.items():
        plt.plot(hist["step"], hist["spatial_chamfer"], marker="o", label=f"{args.sweep_param}={val}")

    plt.title(f"Effect of {args.sweep_param} on Spatial Chamfer Distance")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Selective Chamfer Distance")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.output_img)
    print(f"Plot saved to {args.output_img}")

if __name__ == "__main__":
    main()
