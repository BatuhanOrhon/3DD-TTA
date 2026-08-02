import argparse
import torch
import os
import csv
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler
from tqdm import tqdm

from utils_mate.config import *
from default_config import cfg as diff_config
from models.lion import LION
from utils_mate import misc
from utilities_3dd_tta import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Batch size
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size for processing data')

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
    parser.add_argument('--csv_name', type=str, default="eval_lion_only.csv")

    parser.add_argument('--denoising_step', type=int, default=None, help="If set, overrides the dynamic steps globally")
    parser.add_argument('--denoising_step_bg', type=int, default=35, help="Denoising step for background corruption")
    parser.add_argument('--denoising_step_normal', type=int, default=5, help="Denoising step for non-background corruptions")
    parser.add_argument('--corruption', type=str, default=None, help="Evaluate a specific noise type only")
    parser.add_argument('--resume', action='store_true', help='Resume from an existing CSV file')
    
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

def tta_lion_only(x, lion, steps_back_local=35):
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(100)
    
    alpha_bars = scheduler.alphas_cumprod
    
    timesteps_local = scheduler.timesteps[-steps_back_local:]
    alpha_bar_local = alpha_bars[timesteps_local[0]]
    
    num_samples = x.size(0)

    vae = lion.vae
    local_prior = lion.priors[1]
    
    with torch.no_grad():
        latents = vae.encode(x)
        shape_latent = latents[2][0][0].unsqueeze(2).unsqueeze(3)
        latent_point = latents[2][1][0].unsqueeze(2).unsqueeze(3)
        
        style_cond = vae.global2style(shape_latent)
        
        # Add noise
        noise = torch.randn_like(latent_point)
        noisy_latent_point = torch.sqrt(alpha_bar_local) * latent_point + noise * torch.sqrt(1 - alpha_bar_local)
     
        # Reverse diffusion process (PURE DDIM, NO GUIDANCE)
        for i, t in enumerate(timesteps_local):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device=x.device) * (t + 1)
            
            noise_pred = local_prior(x=noisy_latent_point, t=t_tensor.float(), condition_input=style_cond, clip_feat=None)
            scheduler_output = scheduler.step(noise_pred, t, noisy_latent_point)
            
            noisy_latent_point = scheduler_output.prev_sample

        # Final Decoding
        pred_points = vae.decoder(
            None, beta=None, context=noisy_latent_point.squeeze(3).squeeze(2), 
            style=style_cond.squeeze(3).squeeze(2)
        )
    
    return pred_points

def process_batches(dataloader, base_model, diff_model, args, num_steps):
    preds, targets = [], []

    for data, label in tqdm(dataloader, desc="Processing Batches"):
        data_sample, data_center, data_max = normalize(data)
        data_sample = upsample_all(data_sample.numpy(), 2048)
        data_sample = torch.from_numpy(data_sample).float().cuda()
        label = label.cuda()

        data_sample *= 3.3885
        data_sample = rotate_pointcloud(data_sample)

        pred_points = tta_lion_only(
            x=data_sample, 
            lion=diff_model, 
            steps_back_local=num_steps
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
                    total_acc += float(row[2])
    else:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Corruption", "Accuracy"])

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
            
        targets, preds = process_batches(dataloader, base_model, diff_model, args, num_steps)

        acc = (preds == targets).float().mean().item()
        print(f"Accuracy for {corruption}: {acc * 100:.2f}%")
        total_acc += acc

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.dataset_name, corruption, acc])

    mean_acc = total_acc / len(noises)
    print(f"\n--- FULL EVALUATION FINISHED ---")
    print(f"Mean Accuracy (across {len(noises)} noises): {mean_acc * 100:.2f}%")
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
