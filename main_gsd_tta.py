import argparse
import torch
from torch.utils.data import DataLoader
from utils_mate.config import *
from default_config import cfg as diff_config
from models.lion import LION
from utils_mate import misc
from default_config import cfg as configs
from utilities_3dd_tta import *
from tta_gsd import tta_gsd_reconstruct
from graph_spectral import GraphSpectralDNA
from tqdm import tqdm
import os

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()

    # Batch size
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size for processing data')

    # Configuration and checkpoint paths
    parser.add_argument('--pointmae_config', type=str, default="./cfgs/tta_scanobjectnn.yaml", 
                        help='Path to the YAML config file for PointMAE')
    parser.add_argument('--pointmae_ckpt', type=str, default="./pointnet_ckpts/scanobjectnn_jt.pth", 
                        help='Path to the PointMAE checkpoint')
    parser.add_argument('--diff_config', type=str, default="./lion_ckpts/unconditional_all55_cfg.yml", 
                        help='Path to the diffusion model config file')
    parser.add_argument('--diff_ckpt', type=str, default="./lion_ckpts/epoch_10999_iters_2100999.pt", 
                        help='Path to the diffusion model checkpoint')

    # Dataset-related arguments
    parser.add_argument('--dataset_name', type=str, default="scanobjectnn-c", 
                        choices=["modelnet-c", "shapenet-c", "scanobjectnn-c"], 
                        help="Dataset name (options: modelnet-c, shapenet-c, scanobjectnn-c)")
    parser.add_argument('--dataset_root', type=str, default="./data/scanobjectnn_c", help='Root directory of the dataset')
    parser.add_argument('--label_path', type=str, default="./data/scanobjectnn_c/label.npy", help='Path to the dataset labels')

    # Shape latent and point updating factors
    parser.add_argument('--gamma', type=float, default=0.01, help='Shape latent updating factor')
    parser.add_argument('--eta', type=float, default=0.01, help='Latent point updating factor')
    parser.add_argument('--lambdaa', type=float, default=0.95, help='SCD distance percentile')

    # Device configuration
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the computations on (e.g., cuda, cpu)')

    parser.add_argument('--weight_spectral', type=float, default=1.0, help='Weight for Spectral Loss')
    parser.add_argument('--weight_chamfer', type=float, default=0.0, help='Weight for Chamfer Distance Loss')
    parser.add_argument('--use_4d_gft', action='store_true', help='Use 4D GFT instead of 3D')
    parser.add_argument('--dynamic_graph', action='store_true', help='Recompute graph and U_o dynamically at each step')
    parser.add_argument('--use_static_style', action='store_true', help='Keep the style latent code static (z_c) through diffusion')
    parser.add_argument('--M', type=int, default=100, help='Number of eigenvectors/frequencies to use for spectral matching')

    return parser.parse_args()


def configure_model(args):
    """Load and configure the base and diffusion models, and the graph spectral module."""
    config = cfg_from_yaml_file(args.pointmae_config)

    # Set classification dimensions based on dataset
    if args.dataset_name == "modelnet-c":
        config.model.cls_dim = 40
    elif args.dataset_name == "shapenet-c":
        config.model.cls_dim = 55
    elif args.dataset_name == "scanobjectnn-c":
        config.model.cls_dim = 15
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

    # Load base model
    base_model = load_base_model(args, config, None)
    base_model.eval()
    print('Base model loaded successfully.')

    # Load diffusion model
    diff_config.merge_from_file(args.diff_config)
    diff_model = LION(configs)
    diff_model.load_model(args.diff_ckpt)
    print('Diffusion model loaded successfully.')

    # Initialize Graph Spectral Module
    graph_spectral_module = GraphSpectralDNA(k=10, delta=0.1, gamma=0.6, M=args.M, use_4d_gft=args.use_4d_gft, device=args.device)

    return base_model, diff_model, graph_spectral_module


def process_batches(dataloader, base_model, diff_model, graph_spectral_module, args, num_steps):
    """Process batches of data and compute predictions using GSDTTA."""
    preds, targets = [], []
    loss_weights = {
        "spectral_low": args.weight_spectral,
        "chamfer": args.weight_chamfer
    }

    for data, label in tqdm(dataloader, desc="Processing Batches"):
        # Normalize and upsample the point cloud data
        data_sample, data_center, data_max = normalize(data)
        data_sample = upsample_all(data_sample.numpy(), 2048)
        data_sample = torch.from_numpy(data_sample).float().to(args.device)

        # Scale and rotate the point cloud data
        data_sample *= 3.3885
        data_sample = rotate_pointcloud(data_sample)

        # Perform GSD Test-Time Adaptation
        pred_points, metrics = tta_gsd_reconstruct(
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

        # Undo normalization based on dataset
        if args.dataset_name == "scanobjectnn-c":
            pred_points /= 3.3885
            pred_points = unnormalize_data(pred_points, data_max, data_center)
        else:
            pred_points, _, _ = normalize(pred_points)

        # Apply farthest point sampling (FPS)
        pred_points = misc.fps(pred_points, 1024)

        # Perform classification using the base model
        with torch.no_grad():
            logits = base_model.module.classification_only(pred_points, only_unmasked=False)
            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)

        # Store predictions and targets
        preds.append(pred)
        targets.append(target)

    # Concatenate predictions and targets for accuracy computation
    return torch.cat(targets), torch.cat(preds).cpu()


def main():
    """Main function to execute the GSDTTA pipeline."""
    args = parse_arguments()
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    args.distributed = False

    base_model, diff_model, graph_spectral_module = configure_model(args)

    os.makedirs("./outputs/quantitative", exist_ok=True)
    
    # We use corruptions from utilities_3dd_tta
    for corruption in corruptions:
        # Set reconstruction steps based on corruption type
        num_steps = 35 if corruption == "background" else 5

        # Load dataset and dataloader
        dataset = PointDataset(args.dataset_root, args.label_path, corruption)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # Process the batches
        targets, preds = process_batches(dataloader, base_model, diff_model, graph_spectral_module, args, num_steps)

        # Compute accuracy
        acc = (preds == targets).float().mean().item()
        print(f"Corruption: {corruption}, 4D_GFT: {args.use_4d_gft}, Accuracy: {acc}")

        # Save results to a file
        with open("./outputs/quantitative/gsd_results.txt", "a") as result_file:
            result_file.write(f"Dataset: {args.dataset_name}, Corruption: {corruption}, 4D_GFT: {args.use_4d_gft}, Spectral_W: {args.weight_spectral}, Chamfer_W: {args.weight_chamfer}, Accuracy: {acc}\n")


if __name__ == "__main__":
    main()
