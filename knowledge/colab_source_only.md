# Batch 2: source-only identity evaluation

**[Code]** The `source_only` path gives corrupted ModelNet40-C points directly to FPS(1024) and the frozen Point-MAE classifier. It does not load LION or apply normalization, rotation, diffusion, SCD, guidance, GSD or dropout-mode changes.

Run all 15 severity-5 corruptions in the existing Colab environment:

```bash
%cd /content/3DD-TTA
!git pull --ff-only
!conda run --no-capture-output -n 3dd_tta_env python -u run_baseline.py --method source_only --corruptions uniform gaussian background impulse upsampling distortion_rbf distortion_rbf_inv density density_inc shear rotation cutout distortion occlusion lidar --max-batches 0 --batch_size 32 --seed 0 --run-name source-only_seed0 --pointmae_ckpt ./pointnet_ckpts/modelnet_jt.pth --dataset_root ./data/modelnet40_c --label_path ./data/modelnet40_c/label.npy
```

The run creates `result/modelnet40_c/source_only/<timestamp>_source-only_seed0/` and a sibling ZIP. Supply the complete ZIP even if it fails or OOMs. Compare the complete macro accuracy to the README source-only 57.6 only as a protocol reference; do not tune to force that value.
