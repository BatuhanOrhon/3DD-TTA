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
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing data')

    # Configuration and checkpoint paths
    parser.add_argument('--pointmae_config', type=str, default="./cfgs/tta_scanobjectnn.yaml")
    parser.add_argument('--pointmae_ckpt', type=str, default="./pointnet_ckpts/modelnet40_jt.pth")
    parser.add_argument('--diff_config', type=str, default="./lion_ckpts/unconditional_all55_cfg.yml")
    parser.add_argument('--diff_ckpt', type=str, default="./lion_ckpts/epoch_10999_iters_2100999.pt")

    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default="modelnet-c")
    parser.add_argument('--dataset_root', type=str, default="./data/modelnet40_c")
    parser.add_argument('--label_path', type=str, default="./data/modelnet40_c/label.npy")

    # Outputs
    parser.add_argument('--output_dir', type=str, default="./outputs/quantitative")
    parser.add_argument('--csv_name', type=str, default="eval_fast_results.csv")

    # Fixed Custom Parameters
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--lambdaa', type=float, default=0.95)
    parser.add_argument('--weight_spectral', type=float, default=16.0)
    parser.add_argument('--weight_chamfer', type=float, default=1.0)
    parser.add_argument('--use_4d_gft', action='store_true')
    parser.add_argument('--M', type=int, default=240)
    parser.add_argument('--denoising_step', type=int, default=35)
    
    return parser.parse_args()

def configure_model(args):
    # Same as main_gsd_tta.py
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
    diff_model.cuda()
    diff_model.eval()

    graph_spectral_module = GraphSpectralDNA(k=10, delta=0.1, gamma=0.6, M=args.M, use_4d_gft=args.use_4d_gft).cuda()

    return base_model, diff_model, graph_spectral_module

def process_batches(dataloader, base_model, diff_model, graph_spectral_module, args):
    loss_weights = (args.weight_spectral, args.weight_chamfer)
    preds, targets = [], []

    for label, data_sample in tqdm(dataloader, desc="Processing Batches"):
        data_sample = data_sample.cuda()
        label = label.cuda()

        if args.dataset_name == "scanobjectnn-c":
            data_sample, data_center, data_max = normalize_data(data_sample)
            data_sample *= 3.3885

        data_sample = rotate_pointcloud(data_sample)

        pred_points, _ = tta_gsd_reconstruct(
            x=data_sample, 
            lion=diff_model, 
            graph_spectral_module=graph_spectral_module, 
            steps_back_local=args.denoising_step, 
            gamma=args.gamma, 
            eta=args.eta, 
            p=args.lambdaa, 
            loss_weights=loss_weights,
            total=100
        )
        pred_points = rotateback_pointcloud(pred_points)

        if args.dataset_name == "scanobjectnn-c":
            pred_points /= 3.3885
            pred_points = unnormalize_data(pred_points, data_max, data_center)
        else:
            pred_points, _, _ = normalize(pred_points)

        pred_points = misc.fps(pred_points, 1024)

        with torch.no_grad():
            logits = base_model.module.classification_only(pred_points, only_unmasked=False)
            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)

        preds.append(pred)
        targets.append(target)

    return torch.cat(targets), torch.cat(preds).cpu()

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.csv_name)

    base_model, diff_model, graph_spectral_module = configure_model(args)

    noises = ["uni", "gauss", "back", "impu", "ups", "rbf", "rbf-i", "den-d", "den-i", "shear", "rot", "cut", "dist", "occ", "lidar"]
    
    # Setup CSV Writer
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Corruption", "M", "Weight_Spectral", "Weight_Chamfer", "Accuracy"])
    
    total_acc = 0.0

    for corruption in noises:
        print(f"\nEvaluating Corruption: {corruption}")
        dataset = PointDataset(args.dataset_root, args.label_path, corruption)
        
        # PROTOTYPE MODE: Just use the first 2 samples!
        subset_dataset = Subset(dataset, [0, 1])
        dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)

        targets, preds = process_batches(dataloader, base_model, diff_model, graph_spectral_module, args)

        acc = (preds == targets).float().mean().item()
        print(f"Accuracy for {corruption}: {acc * 100:.2f}%")
        total_acc += acc

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.dataset_name, corruption, args.M, args.weight_spectral, args.weight_chamfer, acc])

    mean_acc = total_acc / len(noises)
    print(f"\n--- FAST EVALUATION FINISHED ---")
    print(f"Mean Accuracy (across 2 samples of {len(noises)} noises): {mean_acc * 100:.2f}%")
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
