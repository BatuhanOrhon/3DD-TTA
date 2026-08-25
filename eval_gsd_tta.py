import argparse
import torch
import os
import csv
from torch.utils.data import DataLoader, Subset
from utils_mate.config import *
from default_config import cfg as diff_config
from models.lion import LION
from utils_mate import misc
from default_config import cfg as configs
from utilities_3dd_tta import *
from tta_gsd import tta_gsd_reconstruct
from graph_spectral import GraphSpectralDNA
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Batch size
    parser.add_argument('--batch_size', type=int, default=70, help='Batch size for processing data')

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
    parser.add_argument('--csv_name', type=str, default="eval_results.csv")

    # Fixed Custom Parameters
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--lambdaa', type=float, default=0.95)
    parser.add_argument('--M', type=int, default=400, help="Number of low-frequency components to preserve")
    parser.add_argument('--M_mid', type=int, default=600, help="Boundary for mid-frequency components")
    parser.add_argument('--M_high', type=int, default=1300, help="End index for the high frequency band")
    parser.add_argument('--weight_spectral', type=float, default=16.0, help="Weight for low-band Spectral guidance loss")
    parser.add_argument('--weight_spectral_mid', type=float, default=2.0, help="Weight for mid-band Spectral guidance loss")
    parser.add_argument('--weight_spectral_high', type=float, default=0.0, help="Weight for high-band Spectral guidance loss")
    parser.add_argument('--weight_invariant', type=float, default=0.0, help="Weight for rotation-invariant spectral power loss")
    parser.add_argument('--weight_chamfer', type=float, default=1.0, help="Weight for Chamfer guidance loss")
    parser.add_argument('--use_4d_gft', action='store_true')
    parser.add_argument('--dynamic_graph', action='store_true', help="Recompute graph dynamically")
    parser.add_argument('--denoising_step', type=int, default=None, help="If set, overrides the dynamic steps globally")
    parser.add_argument('--denoising_step_bg', type=int, default=30, help="Denoising step for background corruption")
    parser.add_argument('--denoising_step_normal', type=int, default=10, help="Denoising step for non-background corruptions")
    parser.add_argument('--corruption', type=str, default=None, help="Evaluate a specific noise type only")
    parser.add_argument('--use_static_style', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, help="Ignore updated style_cond at final decode (True/False)")
    parser.add_argument('--resume', action='store_true', help='Resume from an existing CSV file')
    
    return parser.parse_args()

def configure_model(args):
    # Same as main_gsd_tta.py
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

    graph_spectral_module = GraphSpectralDNA(k=10, delta=0.1, gamma=0.6, M=args.M, M_mid=args.M_mid, M_high=args.M_high, use_4d_gft=args.use_4d_gft, device='cuda')

    return base_model, diff_model, graph_spectral_module

def process_batches(dataloader, base_model, diff_model, graph_spectral_module, args, num_steps):
    loss_weights = {
        "spectral_low": args.weight_spectral,
        "spectral_mid": args.weight_spectral_mid,
        "spectral_high": args.weight_spectral_high,
        "invariant": args.weight_invariant,
        "chamfer": args.weight_chamfer
    }
    preds, targets = [], []

    for data, label in tqdm(dataloader, desc="Processing Batches"):
        # Normalize and upsample the point cloud data
        data_sample, data_center, data_max = normalize(data)
        data_sample = upsample_all(data_sample.numpy(), 2048)
        data_sample = torch.from_numpy(data_sample).float().cuda()
        label = label.cuda()

        # Scale and rotate the point cloud data
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
            use_static_style=args.use_static_style,
            dynamic_graph=args.dynamic_graph
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

    base_model, diff_model, graph_spectral_module = configure_model(args)

    if args.corruption:
        noises = [args.corruption]
    else:
        noises = [
            'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
            'distortion_rbf', 'distortion_rbf_inv', 'density', 'density_inc',
            'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'
        ]
    completed_noises = set()
    total_acc = 0.0

    if args.resume and os.path.exists(csv_path):
        print(f"Resuming from existing file: {csv_path}")
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) > 1:
                    completed_noises.add(row[1])
                    total_acc += float(row[-1])
    else:
        # Setup CSV Writer
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Corruption", "M", "M_mid", "M_high", "Weight_Spectral", "Weight_Spectral_Mid", "Weight_Spectral_High", "Weight_Invariant", "Weight_Chamfer", "Accuracy"])

    for corruption in noises:
        if corruption in completed_noises:
            print(f"\nSkipping Corruption: {corruption} (already evaluated).")
            continue

        print(f"\nEvaluating Corruption: {corruption}")
        dataset = PointDataset(args.dataset_root, args.label_path, corruption)
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        if args.denoising_step is not None:
            num_steps = args.denoising_step
        else:
            num_steps = args.denoising_step_bg if corruption == "background" else args.denoising_step_normal
            
        targets, preds = process_batches(dataloader, base_model, diff_model, graph_spectral_module, args, num_steps)

        acc = (preds == targets).float().mean().item()
        print(f"Accuracy for {corruption}: {acc * 100:.2f}%")
        total_acc += acc

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.dataset_name, corruption, args.M, args.M_mid, args.M_high, args.weight_spectral, args.weight_spectral_mid, args.weight_spectral_high, args.weight_invariant, args.weight_chamfer, acc])

    mean_acc = total_acc / len(noises)
    print(f"\n--- FULL EVALUATION FINISHED ---")
    print(f"Mean Accuracy (across {len(noises)} noises): {mean_acc * 100:.2f}%")
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
