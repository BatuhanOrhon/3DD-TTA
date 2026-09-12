# Paper Note: 3DD-TTA

## Reference

A. Dastmalchi et al., “Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models,” WACV 2025. Local PDF: `Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models kopyası.pdf`. Official page: <https://openaccess.thecvf.com/content/WACV2025/html/Dastmalchi_Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models_WACV_2025_paper.html>.

## Core idea

3DD-TTA performs source-free, training-free input adaptation. A corrupted test point cloud is encoded into LION's hierarchical latent representation: a global shape latent and a local point latent. The local latent is perturbed to an intermediate diffusion timestep and denoised with DDIM. During reverse sampling, gradients from a robust reconstruction loss guide the local noisy state and the global conditioning representation. The reconstructed point cloud is then fed to a frozen classifier.

The method targets a central trade-off: remove corruption using a learned shape prior without destroying instance identity. It avoids classifier adaptation and does not require source samples at test time.

## Selective Chamfer Distance

Ordinary Chamfer Distance can force a reconstruction to match corruption-induced outliers. SCD forms the two directed nearest-neighbor squared-distance sets, sorts them, retains the smallest fraction controlled by `lambda_cd`, and sums the retained values. This focuses guidance on geometrically plausible correspondences while suppressing the largest residuals.

Important consequences:

- `lambda_cd` is a retained fraction, not a loss weight.
- A smaller value rejects more correspondences but can also remove valid structure.
- Reduction (`sum` versus `mean`) changes gradient scale and cannot be compared without retuning guidance rates.

## Paper protocol relevant to this repo

- LION is pretrained and frozen.
- Classifier is frozen during test-time adaptation.
- Total DDIM schedule length: 100 steps.
- Most corruptions start from 5 reverse steps; background corruption uses up to 35.
- The paper reports `gamma = eta = 0.01` and `lambda = 0.96`.
- ModelNet40-C is evaluated at corruption severity 5.
- Hardware reported for the experiments: NVIDIA A6000.

The current repository default `lambda_cd=0.95` is therefore close but not identical to the paper value.

## Reported ModelNet40-C result

Table 2 of the published PDF reports the Point-MAE source model at **57.6%**, MATE-S at **64.3%**, and 3DD-TTA at **65.7%** mean accuracy. The 3DD-TTA per-corruption row is:

| scale | jitter | dropout-global | dropout-local | rotate | add-global | add-local | density | density-inc | cutout | distortion | occlusion | lidar | shear | uniform | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 77.5 | 79.1 | 49.9 | 80.3 | 81.8 | 63.8 | 66.9 | 79.3 | 84.7 | 63.7 | 33.4 | 74.7 | 68.2 | 39.9 | 42.2 | **65.7** |

The repo `README.md` reports **66.1%**, with at least some different cells (for example density decrease and distortion). Treat 65.7 as the published target and 66.1 as the repository target; do not average or silently substitute them.

## Current code correspondence

`tta.py` implements local latent perturbation, DDIM reverse sampling, SCD guidance, and conditioning updates. `main_3dd_tta.py` normalizes the input, randomly expands it to 2048 points when needed, multiplies by `3.3885`, rotates axes for LION, reconstructs, rotates back, normalizes again, farthest-point-samples 1024 points, and invokes Point-MAE.

**[Code, 2026-09-12]** SCD is computed on the first three channels of predicted/original local latents (`tta.py:68-75`); the decoder is only called at the end. Do not confuse paper-level geometric motivation with a physical decoder-space loss implementation. Original LION trainer inference disables dropout, but the inherited wrapper/current TTA setup does not. Accuracy impact and the runtime modes behind the published 3DD-TTA table are unknown; see [code audit](../code_audit_20260912.md).

Observed code/paper details that require controlled validation:

- `tta.py` uses the local-state update rate named `gamma` and the conditioning update rate named `eta`; the prose naming around global/local rates has changed across fork commits. Because both default to 0.01, the issue is hidden until unequal-rate sweeps.
- The final baseline decoder receives the original `shape_latent`, while the updated `style_cond` influences the local prior during sampling. The paper's notation can be read as using an updated global latent, so the exact LION style/decoder contract needs an explicit ablation.
- No global random seed is established. Both point interpolation and diffusion noise are stochastic.
- Result logging appends a sparse line and does not record environment, hashes, seed, or full arguments.

## Strengths

- No source-data access at test time.
- No classifier training/adaptation.
- Uses a learned generative prior while anchoring to the observed instance.
- SCD is explicitly designed for corrupted/outlier points.

## Limitations relevant to the thesis

- Rotation and shear remain difficult because pointwise geometric matching can conflict with semantic correction.
- Background corruption needs substantially more reverse steps.
- The method is stochastic and computationally expensive per input.
- Reproduction is sensitive to pretrained assets, preprocessing, diffusion implementation, and environment.

## Thesis use

3DD-TTA is the baseline and host algorithm. Any new spectral or conflict-aware method must be compared against a matched run of this implementation, with the same data, checkpoint, batch size, seed set, scheduler, steps, preprocessing, and metric aggregation.
