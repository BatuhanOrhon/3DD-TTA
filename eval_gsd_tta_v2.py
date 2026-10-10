import argparse
import os
import csv
import json


def _load_runtime():
    """Keep preview/argument validation independent of CUDA and model imports."""
    global torch, DataLoader, Subset, cfg_from_yaml_file, diff_config, LION
    global misc, builder, normalize, upsample_all, rotate_pointcloud
    global rotateback_pointcloud, unnormalize_data, PointDataset
    global tta_gsd_reconstruct, GraphSpectralDNA, tqdm
    import torch
    from torch.utils.data import DataLoader, Subset
    from utils_mate.config import cfg_from_yaml_file
    from default_config import cfg as diff_config
    from models.lion import LION
    from utils_mate import misc
    from tools import builder
    from utilities_3dd_tta import (normalize, upsample_all, rotate_pointcloud,
                                   rotateback_pointcloud, unnormalize_data, PointDataset)
    from tta_gsd_v2 import tta_gsd_reconstruct
    from graph_spectral_v2 import GraphSpectralDNA
    from tqdm import tqdm

def parse_arguments(argv=None):
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
    parser.add_argument('--M_max', type=int, default=1200, help="Maximum number of frequency components to keep")
    parser.add_argument('--beta', type=float, default=2.0, help="Exponential decay beta for smooth filtering")
    parser.add_argument('--weight_spectral', type=float, default=1.17, help="Weight for smooth Spectral guidance loss (adjusted for batch invariance)")
    parser.add_argument('--weight_invariant', type=float, default=0.0, help="Weight for rotation-invariant spectral power loss")
    parser.add_argument('--weight_chamfer', type=float, default=1.0, help="Weight for Chamfer guidance loss")
    parser.add_argument('--use_4d_gft', action='store_true')
    parser.add_argument('--dynamic_graph', action='store_true', help="Recompute graph dynamically")
    parser.add_argument('--graph_update_interval', type=int, default=3, help="Interval for dynamic graph updates")
    parser.add_argument('--denoising_step', type=int, default=None, help="If set, overrides the dynamic steps globally")
    parser.add_argument('--denoising_step_bg', type=int, default=30, help="Denoising step for background corruption")
    parser.add_argument('--denoising_step_normal', type=int, default=10, help="Denoising step for non-background corruptions")
    parser.add_argument('--corruption', type=str, default=None, help="Evaluate a specific noise type only")
    parser.add_argument('--use_static_style', action='store_true', help="Ignore updated style_cond at final decode (True/False)")
    parser.add_argument('--resume', action='store_true', help='Resume from an existing CSV file')
    parser.add_argument('--seed', type=int, default=None, help='Optional explicit seed; omitted preserves legacy unseeded execution.')
    parser.add_argument('--max_batches', type=int, default=0, help='Positive value runs a partial smoke prefix, never all15 accuracy.')
    parser.add_argument('--preview', action='store_true', help='Print protocol without loading models or creating output.')
    parser.add_argument('--isolate_policy', choices=('legacy', 'offdiag'), default='legacy',
                        help='Keep the historical isolate mask or correct only nonself isolation detection.')
    
    args = parser.parse_args(argv)
    from legacy_gsd_artifacts import validate_arguments
    validate_arguments(args, parser)
    return args

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
    # The source branch sets these modes in LION.__init__. Keep this opt-in
    # behavior here so historical runners retain their existing mode contract.
    diff_model.vae.eval()
    diff_model.priors.eval()

    graph_class = GraphSpectralDNA
    if args.isolate_policy == 'offdiag':
        from graph_spectral_v2_offdiag import GraphSpectralDNA as graph_class
    graph_spectral_module = graph_class(k=10, delta=0.1, gamma=0.6, use_4d_gft=args.use_4d_gft, device='cuda')

    return base_model, diff_model, graph_spectral_module

def process_batches(dataloader, base_model, diff_model, graph_spectral_module, args, num_steps, *, batch_observer=None):
    loss_weights = {
        "spectral": args.weight_spectral,
        "invariant": args.weight_invariant,
        "chamfer": args.weight_chamfer
    }
    preds, targets = [], []
    batch_metrics = []

    for data, label in tqdm(dataloader, desc="Processing Batches"):
        # Normalize and upsample the point cloud data
        data_sample, data_center, data_max = normalize(data)
        data_sample = upsample_all(data_sample.numpy(), 2048)
        data_sample = torch.from_numpy(data_sample).float().cuda()
        label = label.cuda()

        # Scale and rotate the point cloud data
        data_sample *= 3.3885
        data_sample = rotate_pointcloud(data_sample)

        pred_points, metrics = tta_gsd_reconstruct(
            x=data_sample, 
            lion=diff_model, 
            graph_spectral_module=graph_spectral_module, 
            steps_back_local=num_steps, 
            gamma=args.gamma, 
            eta=args.eta, 
            p=args.lambdaa, 
            loss_weights=loss_weights,
            beta=args.beta,
            M_max=args.M_max,
            total=100,
            use_static_style=args.use_static_style,
            dynamic_graph=args.dynamic_graph,
            graph_update_interval=args.graph_update_interval
        )
        batch_metrics.append(metrics)
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
        if batch_observer is not None:
            batch_observer(target, pred, logits, metrics)

    # Calculate average raw losses
    avg_spec = sum(m['mean_raw_spectral'] for m in batch_metrics) / len(batch_metrics) if batch_metrics else 0.0
    avg_chamfer = sum(m['mean_raw_chamfer'] for m in batch_metrics) / len(batch_metrics) if batch_metrics else 0.0
    
    print(f"\n--- Raw Loss Diagnostics ---")
    print(f"Average Raw Spectral Loss: {avg_spec:.6f}  (Current Weight: {args.weight_spectral})")
    print(f"Average Raw Chamfer Loss (Sum)        : {avg_chamfer:.6f}  (Current Weight: {args.weight_chamfer})")
    print(f"----------------------------\n")

    return torch.cat(targets), torch.cat(preds)

def main():
    args = parse_arguments()
    from legacy_gsd_artifacts import protocol, run
    if args.preview:
        print(json.dumps(protocol(args), indent=2, allow_nan=False))
        return
    # Fresh-directory and failure handling cover dependency/model loading too.
    run(args, _load_runtime, configure_model, process_batches)


if __name__ == '__main__':
    main()
