# Colab clean Point-MAE control

## Purpose

Measure frozen Point-MAE on the `data_original.npy` member of the same Zenodo ModelNet40-C archive used for the corruption run. This is a diagnostic clean control, not a corruption benchmark and not proof of exact author asset identity.

## Preconditions

- Checkout is `baseline-repro-clean` at or after the commit that introduces this handoff.
- `data/modelnet40_c/data_original.npy` and `label.npy` exist. The Zenodo archive listing supplied on 2026-09-12 confirms that the archive contains both.
- Use the same `modelnet_jt.pth` checkpoint as the archived source-only run.

## Run

```bash
%cd /content/3DD-TTA
!git pull --ff-only
!conda run --no-capture-output -n 3dd_tta_env python -u run_baseline.py --method source_only --corruptions original --max-batches 0 --batch_size 32 --seed 0 --run-name clean-control_seed0 --pointmae_ckpt ./pointnet_ckpts/modelnet_jt.pth --dataset_root ./data/modelnet40_c --label_path ./data/modelnet40_c/label.npy
```

Provide the emitted ZIP from `result/modelnet40_c/source_only/`. Do not rename it or alter its raw files.

## Interpretation rule

Compare the artifact's clean result only with a documented checkpoint-appropriate reference. A low clean value implicates checkpoint/data/classifier/FPS compatibility; a plausible clean value with low corrupted values narrows the remaining issue to corruption assets/semantics or corrupted-input runtime behavior. Neither outcome alone proves a causal source without the corresponding reference evidence.
