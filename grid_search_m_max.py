import argparse
import torch
from eval_gsd_tta_v2 import configure_model, process_batches
from utilities_3dd_tta import PointDataset
from torch.utils.data import DataLoader
from default_config import cfg as configs
import json
import os
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--pointmae_config', type=str, default="./cfgs/tta_modelnet.yaml")
    parser.add_argument('--pointmae_ckpt', type=str, default="./pointnet_ckpts/modelnet_jt.pth")
    parser.add_argument('--diff_config', type=str, default="./lion_ckpts/unconditional_all55_cfg.yml")
    parser.add_argument('--diff_ckpt', type=str, default="./lion_ckpts/epoch_10999_iters_2100999.pt")
    parser.add_argument('--dataset_name', type=str, default="modelnet-c")
    parser.add_argument('--dataset_root', type=str, default="./data/modelnet40_c")
    parser.add_argument('--label_path', type=str, default="./data/modelnet40_c/label.npy")
    parser.add_argument('--output_dir', type=str, default="./result/modelnet40_c/gsd_smooth_grid_search")
    
    # Base TTA parameters
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--lambdaa', type=float, default=0.95)
    parser.add_argument('--weight_spectral', type=float, default=1.17)
    parser.add_argument('--weight_invariant', type=float, default=0.0)
    parser.add_argument('--weight_chamfer', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--denoising_step_normal', type=int, default=10)
    
    parser.add_argument('--use_4d_gft', action='store_true')
    parser.add_argument('--dynamic_graph', action='store_true')
    parser.add_argument('--graph_update_interval', type=int, default=3)
    parser.add_argument('--use_static_style', action='store_true')
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_model, diff_model, graph_spectral_module = configure_model(args)
    
    corruptions = [
        "uniform", "gaussian", "background", "impulse", "upsampling", 
        "distortion_rbf", "distortion_rbf_inv", "density", "density_inc", 
        "shear", "rotation", "cutout", "distortion", "occlusion", "lidar"
    ]
    
    m_max_candidates = [400, 600, 800, 1000, 1200]
    
    results = {}
    
    for m_max in m_max_candidates:
        print(f"\\n=== Testing M_max = {m_max} ===")
        args.M_max = m_max
        results[m_max] = {}
        
        for corruption in corruptions:
            print(f"  Evaluating {corruption}...")
            
            # Create a dataset for just this corruption, only 120 items
            dataset = PointDataset(args.dataset_root, args.label_path, corruption)
            
            # Subset to exactly 120 samples
            from torch.utils.data import Subset
            subset_indices = list(range(120))
            if len(dataset) > 120:
                dataset = Subset(dataset, subset_indices)
            
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            
            # Standard background uses 30 steps, normal uses 10
            num_steps = 30 if corruption == "background" else args.denoising_step_normal
            
            targets, preds = process_batches(dataloader, base_model, diff_model, graph_spectral_module, args, num_steps)
            
            # Calculate accuracy
            targets_tensor = torch.tensor(targets)
            preds_tensor = torch.tensor(preds)
            acc = (preds_tensor == targets_tensor).float().mean().item()
            print(f"  {corruption} Acc: {acc*100:.2f}%")
            
            results[m_max][corruption] = acc
            
        # Save intermediate results
        with open(os.path.join(args.output_dir, "grid_search_m_max.json"), "w") as f:
            json.dump(results, f, indent=4)
            
if __name__ == "__main__":
    main()
