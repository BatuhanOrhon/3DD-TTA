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
from tta_pxp_sym import tta_gsd_reconstruct as tta_pxp_reconstruct
from graph_spectral import GraphSpectralDNA

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Batch size
    parser.add_argument('--batch_size', type=int, default=40)
    
    # Configuration and checkpoint paths
    parser.add_argument('--pointmae_config', type=str, default="./cfgs/tta_modelnet.yaml")
    parser.add_argument('--pointmae_ckpt', type=str, default="./pointnet_ckpts/modelnet_jt.pth")
    parser.add_argument('--diff_config', type=str, default="./lion_ckpts/unconditional_all55_cfg.yml")
    parser.add_argument('--diff_ckpt', type=str, default="./lion_ckpts/epoch_10999_iters_2100999.pt")
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default="modelnet-c")
    parser.add_argument('--dataset_root', type=str, default="./data/modelnet40_c")
    parser.add_argument('--label_path', type=str, default="./data/modelnet40_c/label.npy")
    
    # Target Noise
    parser.add_argument('--corruption', type=str, default="uniform", help="The specific noise to test on")
    parser.add_argument('--num_samples', type=int, default=50, help="Number of samples to evaluate")
    
    # Outputs
    parser.add_argument('--output_dir', type=str, default="./outputs/quantitative")
    parser.add_argument('--csv_name', type=str, default="grid_search_gamma_eta_results.csv")
    parser.add_argument('--log_name', type=str, default="grid_search_gamma_eta_log.txt")
    
    # Fixed Custom Parameters
    parser.add_argument('--lambdaa', type=float, default=0.95)
    parser.add_argument('--denoising_steps', type=int, default=5)
    parser.add_argument('--weight_chamfer', type=float, default=1.0)
    parser.add_argument('--weight_spectral_low', type=float, default=0.0)
    parser.add_argument('--weight_spectral_mid', type=float, default=0.0)
    parser.add_argument('--weight_spectral_high', type=float, default=0.0)
    parser.add_argument('--use_static_style', action='store_true', default=True, help="Use static style for evaluation")
    parser.add_argument('--spectral_reduction', type=str, default="mean", choices=["mean", "sum"], help="Reduction method for spectral MSE loss")
    
    # Grid Search Parameters
    parser.add_argument('--gammas', nargs='+', type=float, default=[0.01, 0.005, 0.001, 0.0005, 0.0001], help="List of gammas")
    parser.add_argument('--etas', nargs='+', type=float, default=[0.01, 0.005], help="List of eta values to test")
    parser.add_argument('--delta1', type=float, default=1.0, help="Projection weight for Chamfer onto Spectral")
    parser.add_argument('--delta2', type=float, default=1.0, help="Projection weight for Spectral onto Chamfer")
    
    return parser.parse_args()

def configure_model(args):
    config = cfg_from_yaml_file(args.pointmae_config)
    if args.dataset_name == "modelnet-c":
        config.model.cls_dim = 40
    elif args.dataset_name == "shapenet-c":
        config.model.cls_dim = 50
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

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.csv_name)

    base_model, diff_model = configure_model(args)
    
    graph_spectral_module = GraphSpectralDNA(
        k=10, delta=0.1, gamma=0.6, M=400, M_mid=600, use_4d_gft=False, device='cuda'
    ).to('cuda')
    
    loss_weights = {
        "spectral_low": args.weight_spectral_low,
        "spectral_mid": args.weight_spectral_mid,
        "spectral_high": args.weight_spectral_high,
        "invariant": 0.0,
        "chamfer": args.weight_chamfer
    }

    # Prepare Dataset (Full dataset, no subset)
    dataset = PointDataset(args.dataset_root, args.label_path, args.corruption)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.csv_name)
    log_path = os.path.join(args.output_dir, args.log_name)
    
    # Initialize CSV if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Corruption", "Static_Style", "Gamma", "Eta", "Weight_Chamfer", "Delta1", "Delta2", "Accuracy"])
            
    # Write to log file header
    with open(log_path, "a") as f:
        f.write("\n=========================================\n")
        f.write(f"Starting Grid Search (PxP Symmetric) for Corruption: {args.corruption}\n")
        f.write("=========================================\n")

    print(f"Starting Gamma/Eta Grid Search for corruption: {args.corruption}")
    print(f"Total Combinations: {len(args.gammas) * len(args.etas)}")
    print(f"Total samples per combo: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    for gamma, eta in itertools.product(args.gammas, args.etas):
        print(f"\n--- Testing Combo: Gamma={gamma}, Eta={eta} ---")
        
        correct = 0
        total = 0
        
        for batch_idx, (data, label) in enumerate(dataloader):
            data_sample, data_center, data_max = normalize(data)
            data_sample = upsample_all(data_sample.numpy(), 2048)
            data_sample = torch.from_numpy(data_sample).float().cuda()
            label = label.cuda()

            data_sample *= 3.3885
            data_sample = rotate_pointcloud(data_sample)

            pred_points, _ = tta_pxp_reconstruct(
                x=data_sample, 
                lion=diff_model, 
                graph_spectral_module=graph_spectral_module, 
                steps_back_local=args.denoising_steps, 
                gamma=gamma, 
                eta=eta, 
                p=args.lambdaa, 
                loss_weights=loss_weights,
                total=100,
                use_static_style=args.use_static_style,
                delta1=args.delta1,
                delta2=args.delta2,
                spectral_reduction=args.spectral_reduction,
                dynamic_graph=False
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

            correct += (pred == target).sum().item()
            total += target.size(0)
            

        final_acc = correct / total
        print_msg = f"==> Final Accuracy for Combo (Gamma={gamma}, Eta={eta}): {final_acc * 100:.2f}%\n"
        print(print_msg)
        with open(log_path, "a") as f:
            f.write(print_msg)
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.corruption, args.use_static_style, gamma, eta, args.weight_chamfer, args.delta1, args.delta2, final_acc])
            
    finish_msg = f"\nGrid Search Finished! Results saved to {csv_path} and {log_path}\n"
    print(finish_msg)
    with open(log_path, "a") as f:
        f.write(finish_msg)

if __name__ == "__main__":
    main()
