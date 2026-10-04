#!/bin/bash
# Colab Run Script for GSD Smoothing (Optimal M_max=800)

conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta_v2.py \
    --batch_size 70 \
    --weight_spectral 1.17 \
    --weight_chamfer 1.0 \
    --M_max 800 \
    --beta 2.0 \
    --gamma 0.01 \
    --eta 0.01 \
    --lambdaa 0.95 \
    --denoising_step_bg 30 \
    --denoising_step_normal 10 \
    --dataset_root ./data/modelnet40_c \
    --label_path ./data/modelnet40_c/label.npy \
    --output_dir ./result/modelnet40_c/gsd_smooth_optimal_800 \
    --csv_name eval_results.csv
