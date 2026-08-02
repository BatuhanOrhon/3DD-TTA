import argparse
import torch
import os
import csv
import itertools
from torch.utils.data import DataLoader, Subset
from utils_mate.config import *
from default_config import cfg as diff_config
from models.lion import LION
from utils_mate import misc
from default_config import cfg as configs
from utilities_3dd_tta import *
from tta_gsd import tta_gsd_reconstruct
from graph_spectral import GraphSpectralDNA

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Batch size
    parser.add_argument('--batch_size', type=int, default=70)
    
    # Configuration and checkpoint paths
    parser.add_argument('--pointmae_config', type=str, default="./cfgs/tta_modelnet.yaml")
    parser.add_argument('--pointmae_ckpt', type=str, default="./pointnet_ckpts/modelnet_jt.pth")
    parser.add_argument('--diff_config', type=str, default="./lion_ckpts/unconditional_all55_cfg.yml")
    parser.add_argument('--diff_ckpt', type=str, default="./lion_ckpts/epoch_10999_iters_2100999.pt")
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default="modelnet-c")
    parser.add_argument('--dataset_root', type=str, default="./data/modelnet40_c")
    parser.add_argument('--label_path', type=str, default="./data/modelnet40_c/label.npy")
    
    # Outputs
    parser.add_argument('--output_dir', type=str, default="./outputs/quantitative")
    parser.add_argument('--csv_name', type=str, default="grid_search_mid_high_results.csv")
    
    # Fixed Custom Parameters for this specific search
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--lambdaa', type=float, default=0.95)
    parser.add_argument('--weight_chamfer', type=float, default=1.0)
    parser.add_argument('--weight_spectral_low', type=float, default=16.0)
    parser.add_argument('--M', type=int, default=400)
    parser.add_argument('--m_mids', nargs='+', type=int, default=[512, 1024, 1536], help="List of M_mid values")
    parser.add_argument('--m_highs', nargs='+', type=int, default=[2048], help="List of M_high values")
    parser.add_argument('--mid_weights', nargs='+', type=float, default=[0.1, 1.0, 4.0], help="List of weight_mid values")
    parser.add_argument('--high_weights', nargs='+', type=float, default=[0.1, 1.0, 4.0], help="List of weight_high values")
    parser.add_argument('--denoising_steps', nargs='+', type=int, default=[-1], help="List of denoising steps. -1 uses Oracle baseline (35 for background, 5 for others)")
    parser.add_argument('--use_static_style', action='store_true', help="Use static shape_latent at final decode")
    parser.add_argument('--samples_per_noise', type=int, default=-1, help="Number of samples to evaluate per noise type. If not provided (-1), uses the entire dataset.")
    
    return parser.parse_args()

def configure_model(args):
    config = cfg_from_yaml_file(args.pointmae_config)
    if args.dataset_name == "modelnet-c":
        config.model.cls_dim = 40
    elif args.dataset_name == "shapenet-c":
        config.model.cls_dim = 55
    elif args.dataset_name == "scanobjectnn-c":
        config.model.cls_dim = 15

    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.pointmae_ckpt, logger=None)
    base_model.cuda()
    base_model.eval()

    diff_config.merge_from_file(args.diff_config)
    diff_model = LION(diff_config)
    diff_model.load_model(args.diff_ckpt)

    return base_model, diff_model

def process_batches(dataloader, base_model, diff_model, args, num_steps, m_mid, m_high, weight_mid, weight_high):
    graph_spectral_module = GraphSpectralDNA(
        k=10, delta=0.1, gamma=0.6, 
        M=args.M, M_mid=m_mid, M_high=m_high,
        use_4d_gft=False, device='cuda'
    )
    
    loss_weights = {
        "spectral_low": args.weight_spectral_low,
        "spectral_mid": weight_mid,
        "spectral_high": weight_high,
        "chamfer": args.weight_chamfer
    }
    
    preds, targets = [], []

    for data, label in dataloader:
        data_sample, data_center, data_max = normalize(data)
        data_sample = upsample_all(data_sample.numpy(), 2048)
        data_sample = torch.from_numpy(data_sample).float().cuda()
        label = label.cuda()

        data_sample *= 3.3885
        data_sample = rotate_pointcloud(data_sample)

        pred_points, _ = tta_gsd_reconstruct(
            x=data_sample, 
            lion=diff_model, 
            graph_spectral_module=graph_spectral_module, 
            steps_back_local=num_steps, 
            gamma=args.gamma, 
            eta=args.eta, 
            p=args.lambdaa, 
            loss_weights=loss_weights,
            total=100,
            use_static_style=args.use_static_style
        )
        pred_points = rotateback_pointcloud(pred_points)

        if args.dataset_name == "scanobjectnn-c":
            pred_points /= 3.3885
            pred_points = unnormalize_data(pred_points, data_max, data_center)
        else:
            pred_points, _, _ = normalize(pred_points)

        pred_points = misc.fps(pred_points, 1024)

        with torch.no_grad():
            logits = base_model.classification_only(pred_points, only_unmasked=False)
            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)

        preds.append(pred)
        targets.append(target)

    return torch.cat(targets), torch.cat(preds)

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.csv_name)

    base_model, diff_model = configure_model(args)

    target_noises = [
        'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
        'distortion_rbf', 'distortion_rbf_inv', 'density', 'density_inc',
        'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'
    ]
    
    # Grid Search Parameters will be retrieved from args
    m_mids = args.m_mids
    m_highs = args.m_highs
    mid_weights = args.mid_weights
    high_weights = args.high_weights
    denoising_steps = args.denoising_steps
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "M_mid", "M_high", "Weight_Low", "Weight_Mid", "Weight_High", "Denoising_Step", "Use_Static_Style"] + target_noises + ["Mean_Accuracy"])

    sample_count_str = str(args.samples_per_noise) if args.samples_per_noise > 0 else "ALL"
    print(f"Starting Mid/High Frequency Grid Search. 15 Noises ({sample_count_str} samples each). Total Combinations: {len(m_mids) * len(m_highs) * len(mid_weights) * len(high_weights) * len(denoising_steps)}")
    print(f"Fixed Params: M={args.M}, Weight_Low={args.weight_spectral_low}, Static_Style={args.use_static_style}")

    for m_mid, m_high, w_mid, w_high, test_step in itertools.product(m_mids, m_highs, mid_weights, high_weights, denoising_steps):
        print(f"\n--- Testing Combo: M_mid={m_mid}, M_high={m_high}, Weight_Mid={w_mid}, Weight_High={w_high}, Step={test_step} ---")
        combo_accuracies = []
        
        for corruption in target_noises:
            dataset = PointDataset(args.dataset_root, args.label_path, corruption)
            if args.samples_per_noise > 0:
                subset_dataset = Subset(dataset, range(args.samples_per_noise))
                dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)
            else:
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            
            # Oracle steps logic natively applied if test_step == -1
            if test_step == -1:
                step = 35 if corruption == "background" else 5
            else:
                step = test_step
            
            targets, preds = process_batches(dataloader, base_model, diff_model, args, num_steps=step, m_mid=m_mid, m_high=m_high, weight_mid=w_mid, weight_high=w_high)
            
            acc = (preds == targets).float().mean().item()
            print(f"  {corruption} -> {acc*100:.2f}%")
            combo_accuracies.append(acc)
            
        mean_acc = sum(combo_accuracies) / len(combo_accuracies)
        print(f"--- Combo Mean Accuracy: {mean_acc*100:.2f}% ---")
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.M, m_mid, m_high, args.weight_spectral_low, w_mid, w_high, test_step, args.use_static_style] + combo_accuracies + [mean_acc])
            
    print(f"\nGrid Search Finished. Results saved to {csv_path}")

if __name__ == "__main__":
    main()
