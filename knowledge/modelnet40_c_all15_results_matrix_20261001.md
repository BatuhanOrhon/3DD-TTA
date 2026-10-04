# ModelNet40-C all-15 corruption result matrix

> [Run] Later2026-10-04 addendum: six further complete full15 archives have
> arrived for unguided diffusion and fixed beta .5 smooth-only, seeds0-2.
> Rows are added below; the original40-archive inventory remains a historical
> count. The new arms have differing native binary hashes. Their full
> corruption/seed results, historical-control comparisons and predictions are
> in [the accepted ablation report](gsd_guidance_ablation_results_20261004.md).

> [Run/Inference] 2026-10-04 addendum: v1 spectral-only M100 seed0 now has
> a separately recorded Background result, 23.1767% (572/2468). The old14
> plus new1 descriptive composite is 61.1967% (22655/37020). It remains
> outside the homogeneous full-run table and the historical 40-archive count
> below because revisions/native binaries and execution scopes differ.
> Against the full-screen SCD-only seed0 reference (63.7520%), the composite
> delta is -2.5554 pp, of which Background contributes -2.5149 pp.
> [Complete validation and 15-row comparison](../result/modelnet40_c/gsd_latent_spectral_v1/analysis_20261004_background/validation.md).

[Run/Verification/Inference] Compiled on 2026-10-01 from the full knowledge
record and raw archives. Git context: `gsd-smooth-spectrum@4be6afd86629f3d52648fb543fc2eb350a5f9f1d`.
The question answered is descriptive: how did each completed ModelNet40-C
severity-5 test behave across all 15 corruption types? Results from different
commits/configurations are not automatically causal comparisons.

## Inclusion and interpretation

- Audited 141 local ModelNet40-C ZIPs. Forty have complete canonical rows for
  all 15 corruptions, 2,468 examples per row, seven unique bundle files, one
  safe archive root, good CRC, complete statuses, consistent CSV accuracy and
  macro summary. The remaining ZIPs use another protocol scope or do not meet these full-coverage
  criteria and are excluded from this matrix. Some full-coverage legacy
  `3dd_original` bundles retain a generic `stage=smoke` config label despite
  `max_batches=0`; inclusion here follows the complete 15-row, 2,468-example
  artifacts and their CSV totals, not that stale stage label.
- Severity-5 means one full 2,468-example test file for each canonical
  corruption. Macro is the equal-weight arithmetic mean of the 15 corruption
  accuracies. `mean +/- SD` below is across run seeds, using sample SD; it is not
  a confidence interval. Per-corruption cells in the next tables are the mean
  accuracy across the listed seeds, rounded to 2 decimals; paired effect cells
  are mean percentage-point differences across the paired seeds, rounded to
  3 decimals.
- Source-only severity 1-5 uses one seed per severity. The seed-stability
  source-only severity-5 runs are deterministic and have identical scores.
  Preprocessing identity, pure VAE, 3DD-TTA, SCD controls and GSD candidates
  are separate experiment families. Absolute values across these families
  should not be read as controlled treatment effects unless a paired test is
  named in the effect table.
- Seed-controlled comparisons are generally not common-random-number paired.
  The shared-decoder original/updated comparison is paired inside each run's
  shared trajectory. SCD normalization and lambda=.96 use the repository's logged
  seed-matched original-style comparator; their source revisions differ from
  that comparator, and lambda=.96 also records different native extension hashes.
  Treat those deltas as historical matched-setting evidence with that caveat.
- The three GSD full-test conditions were evaluated on the ModelNet40-C test
  set itself. Their table is descriptive development evidence, not independent
  confirmation. Only the locked conditions tested here are represented.

## Macro results across all 15 corruptions

| Test / condition | Scope | Seeds | Macro mean +/- sample SD (%) | Paired result (pp) | Note |
|---|---|---|---:|---:|---|
| Source-only severity 1 | 15 x 2,468 | 0 | 75.8806 +/- 0.0000 | n/a | S5 source-only is the comparator below. |
| Source-only severity 2 | 15 x 2,468 | 0 | 73.2739 +/- 0.0000 | n/a |  |
| Source-only severity 3 | 15 x 2,468 | 0 | 68.5062 +/- 0.0000 | n/a |  |
| Source-only severity 4 | 15 x 2,468 | 0 | 62.0205 +/- 0.0000 | n/a |  |
| Source-only severity 5 | 15 x 2,468 | 0/1/2 | 53.6899 +/- 0.0000 | n/a | Three seed archives have identical scores. |
| Preprocessing identity | 15 x 2,468 | 0/1/2 | 55.0135 +/- 0.0602 | +1.3236 pp vs source-only mean | LION bypassed. |
| Pure VAE encode/decode | 15 x 2,468 | 0/1/2 | 54.8469 +/- 0.0790 | +1.1570 pp vs source-only; -0.1666 pp vs identity | Macro mean difference from identity is calculated from raw; not paired. |
| 3DD original, eval + raw | 15 x 2,468 | 0/1/2 | 63.8052 +/- 0.0973 | +/- | Operational TTA reference for later ablations. |
| 3DD original, legacy + raw | 15 x 2,468 | 1/2 | 63.0970 +/- 0.0554 | Eval minus legacy +0.7577 +/- 0.1203 pp | Seed-matched, not common-draw paired. |
| 3DD original, eval + EMA | 15 x 2,468 | 0/1/2 | 63.9087 +/- 0.1268 | EMA minus raw +0.1035 +/- 0.1639 pp | Small, seed/corruption dependent; ablation only. |
| Shared decoder, original style | 15 x 2,468 | 0/1/2 | 63.7484 +/- 0.1399 | n/a | Comparator for the updated style. |
| Shared decoder, updated style | 15 x 2,468 | 0/1/2 | 63.7169 +/- 0.1480 | Updated minus original -0.0315 pp | Within-run shared-trajectory pairing. |
| SCD point-count normalized | 15 x 2,468 | 0/1/2 | 61.2678 +/- 0.0979 | -2.4806 pp vs shared original-style | Fixed rates; do not infer scale-matched normalization. |
| SCD lambda=.96 | 15 x 2,468 | 0/1/2 | 63.9339 +/- 0.1878 | +0.1855 +/- 0.0543 pp vs lambda=.95 comparator | Data/assets match; source and extension identities differ. |
| GSD smooth-v2 SCD-only | 15 x 2,468 | 0/1/2 | 63.8799 +/- 0.1326 | n/a |  |
| GSD smooth-v2 beta=.5/rho=.001 | 15 x 2,468 | 0/1/2 | 63.8520 +/- 0.0353 | -0.0279  +/- 0.1048 pp vs matched GSD SCD | One positive, two negative seed deltas. |
| GSD smooth-v2 beta=2/rho=.01 | 15 x 2,468 | 0/1/2 | 63.8214 +/- 0.1184 | -0.0585  +/- 0.0473 pp vs matched GSD SCD | All three seed deltas negative. |
| Unguided diffusion, SCD0/spectral0 | 15 x 2,468 | 0/1/2 | 61.2021 +/- 0.1274 | n/a | Ingested2026-10-04; five/35 reverse steps. |
| Smooth-only beta=.5, alpha8.140161, SCD0 | 15 x 2,468 | 0/1/2 | 61.3596 +/- 0.0721 | +0.1576 +/- 0.1032 pp vs unguided | Descriptive same-seed delta; all three positive, native binaries differ. |

## Source-only severity curve, per corruption

Accuracies are percentages, seed 0. This is a different comparison axis from
method ablations at fixed severity 5.

| Corruption | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|
| uniform | 89.10 | 85.05 | 79.50 | 69.41 | 59.68 |
| gaussian | 88.17 | 81.65 | 74.07 | 63.41 | 51.30 |
| background | 48.70 | 45.50 | 40.76 | 38.33 | 28.16 |
| impulse | 72.85 | 71.92 | 67.83 | 64.63 | 55.35 |
| upsampling | 83.14 | 81.40 | 80.67 | 77.39 | 71.52 |
| distortion_rbf | 89.30 | 86.22 | 78.20 | 68.76 | 57.41 |
| distortion_rbf_inv | 89.63 | 87.12 | 79.70 | 69.85 | 60.70 |
| density | 83.63 | 79.62 | 76.54 | 70.75 | 65.24 |
| density_inc | 82.33 | 81.60 | 81.28 | 79.17 | 77.35 |
| shear | 89.38 | 87.20 | 80.63 | 73.66 | 67.18 |
| rotation | 88.70 | 80.55 | 67.06 | 51.22 | 30.19 |
| cutout | 81.52 | 78.77 | 74.76 | 69.29 | 62.28 |
| distortion | 89.75 | 86.71 | 79.13 | 71.43 | 61.83 |
| occlusion | 41.77 | 43.64 | 43.72 | 41.17 | 37.24 |
| lidar | 20.22 | 22.16 | 23.74 | 21.84 | 19.94 |
| **Macro mean** | 75.88 | 73.27 | 68.51 | 62.02 | 53.69 |

## Severity-5 accuracy by corruption: input, reconstruction and 3DD-TTA

Each cell is the mean percentage accuracy across its listed full runs. `legacy`
uses only seeds 1/2; other TTA rows use seeds 0/1/2.

| Corruption | Source-only S5 | Preprocess identity | Pure VAE | 3DD eval/raw | 3DD legacy/raw |
|---|---:|---:|---:|---:|---:|
| uniform | 59.68 | 61.63 | 61.74 | 76.96 | 76.76 |
| gaussian | 51.30 | 52.49 | 53.88 | 74.66 | 73.91 |
| background | 28.16 | 23.76 | 23.28 | 60.45 | 60.41 |
| impulse | 55.35 | 54.92 | 55.77 | 70.41 | 69.12 |
| upsampling | 71.52 | 71.52 | 73.08 | 82.02 | 81.69 |
| distortion_rbf | 57.41 | 59.83 | 59.32 | 62.76 | 62.70 |
| distortion_rbf_inv | 60.70 | 62.84 | 62.55 | 65.15 | 64.36 |
| density | 65.24 | 69.14 | 66.88 | 72.49 | 70.77 |
| density_inc | 77.35 | 85.32 | 84.09 | 86.09 | 85.31 |
| shear | 67.18 | 66.76 | 66.52 | 66.17 | 64.75 |
| rotation | 30.19 | 30.36 | 30.96 | 33.20 | 32.82 |
| cutout | 62.28 | 68.45 | 67.14 | 70.57 | 69.67 |
| distortion | 61.83 | 63.14 | 62.86 | 65.05 | 64.81 |
| occlusion | 37.24 | 40.05 | 37.67 | 40.30 | 39.59 |
| lidar | 19.94 | 15.01 | 16.98 | 30.79 | 29.78 |
| **Macro mean** | 53.69 | 55.01 | 54.85 | 63.81 | 63.10 |

## Severity-5 accuracy by corruption: decoder, SCD and GSD variants

Shared-decoder columns are means across the three seeds. GSD columns are means
across seeds 0/1/2 under one calibration reference.

| Corruption | Shared original-style | Shared updated-style | SCD normalized | SCD lambda=.96 | GSD SCD-only | GSD beta=.5/rho=.001 | GSD beta=2/rho=.01 |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 76.86 | 76.89 | 77.26 | 76.93 | 77.05 | 76.77 | 76.88 |
| gaussian | 74.50 | 74.47 | 74.26 | 74.84 | 74.74 | 74.62 | 74.31 |
| background | 60.29 | 59.95 | 23.20 | 61.62 | 60.67 | 60.91 | 60.80 |
| impulse | 70.31 | 70.45 | 69.77 | 70.00 | 70.29 | 70.30 | 70.42 |
| upsampling | 81.90 | 81.87 | 81.67 | 82.16 | 82.01 | 82.04 | 82.08 |
| distortion_rbf | 62.72 | 62.76 | 63.10 | 63.16 | 62.97 | 63.05 | 62.82 |
| distortion_rbf_inv | 65.41 | 65.24 | 65.18 | 65.46 | 65.41 | 65.13 | 65.52 |
| density | 72.54 | 72.51 | 72.53 | 72.68 | 72.60 | 72.31 | 72.58 |
| density_inc | 85.99 | 86.03 | 86.05 | 86.08 | 86.10 | 86.02 | 85.97 |
| shear | 66.36 | 66.29 | 66.06 | 66.15 | 66.14 | 66.59 | 65.98 |
| rotation | 32.94 | 33.04 | 32.79 | 33.18 | 33.45 | 33.01 | 33.17 |
| cutout | 70.38 | 70.37 | 70.57 | 70.64 | 70.69 | 70.31 | 70.57 |
| distortion | 65.34 | 65.21 | 65.01 | 65.14 | 65.13 | 65.48 | 64.99 |
| occlusion | 40.02 | 40.21 | 40.13 | 40.33 | 40.22 | 40.41 | 40.44 |
| lidar | 30.65 | 30.46 | 31.44 | 30.65 | 30.73 | 30.83 | 30.79 |
| **Macro mean** | 63.75 | 63.72 | 61.27 | 63.93 | 63.88 | 63.85 | 63.82 |

## Paired effect by corruption

Positive values mean the left-hand condition is more accurate. All values are
percentage points. `SCD norm minus SCD .95*` and `lambda=.96 minus lambda=.95` use the logged
seed-matched original-style comparator, not an exact same-runtime new run.

| Corruption | Eval minus legacy (2 seeds) | EMA minus raw (3) | Updated minus original style (3) | SCD norm minus SCD .95* | lambda=.96 minus lambda=.95 | GSD beta .5 minus SCD | GSD beta 2 minus SCD |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | -0.122 | +0.230 | +0.027 | +0.392 | +0.068 | -0.284 | -0.176 |
| gaussian | +0.729 | +0.270 | -0.027 | -0.243 | +0.338 | -0.122 | -0.432 |
| background | -0.101 | +0.581 | -0.338 | -37.088 | +1.324 | +0.243 | +0.135 |
| impulse | +1.479 | -0.324 | +0.135 | -0.540 | -0.311 | +0.014 | +0.135 |
| upsampling | +0.628 | +0.297 | -0.027 | -0.230 | +0.257 | +0.027 | +0.068 |
| distortion_rbf | +0.162 | +0.068 | +0.041 | +0.378 | +0.432 | +0.081 | -0.149 |
| distortion_rbf_inv | +0.851 | +0.068 | -0.176 | -0.230 | +0.054 | -0.284 | +0.108 |
| density | +1.742 | -0.014 | -0.027 | -0.014 | +0.135 | -0.284 | -0.014 |
| density_inc | +0.750 | +0.230 | +0.041 | +0.054 | +0.081 | -0.081 | -0.135 |
| shear | +1.682 | +0.041 | -0.068 | -0.297 | -0.203 | +0.446 | -0.162 |
| rotation | +0.304 | +0.176 | +0.095 | -0.149 | +0.243 | -0.446 | -0.284 |
| cutout | +1.074 | +0.000 | -0.014 | +0.189 | +0.257 | -0.378 | -0.122 |
| distortion | +0.405 | +0.365 | -0.135 | -0.338 | -0.203 | +0.351 | -0.135 |
| occlusion | +0.750 | -0.189 | +0.189 | +0.108 | +0.311 | +0.189 | +0.216 |
| lidar | +1.033 | -0.243 | -0.189 | +0.797 | +0.000 | +0.108 | +0.068 |
| **Macro mean** | +0.758 | +0.104 | -0.032 | -2.481 | +0.185 | -0.028 | -0.059 |

## Context not in the 15-corruption matrix

The local GSD v1 `ablation_no_background` ZIP is complete on 14/15 corruption
files at seed 0, with 2,468 examples each and a 63.9124% equal-weight macro:
`result/modelnet40_c/gsd_latent_spectral_v1/20260926-141029_gsd-v1-ablation14-on-seed0-spectral-only.zip`.
It is a spectral-only arm without Background and no matched baseline ZIP was
found in the local archive set, so it is excluded from the all-15 matrices and
does not support a paired method effect.

The earlier rebuilt beta-rho interaction used only Gaussian and Impulse, 128
examples each, seed 0: beta .5/rho .001 was +0.7813 pp and beta 2/rho .01 was
+1.1719 pp versus its matched SCD-only control. Those two-corruption results
are not all-15 evidence and are superseded for full-benchmark generalization by
the complete GSD screen above. The other local GSD v1 archives are smoke/pilot
or selected-corruption coverage. No complete 15-corruption GSD v1 score is
present locally.

The WACV paper's 65.7% and the upstream README's 66.1% are external references,
not locally rerun conditions; keep them separate from the run table.

## Source artifacts and knowledge records


- Source-only severities and seeds: `result/modelnet40_c/source_only/`
- Preprocessing identity and pure VAE: `result/modelnet40_c/preprocessing_identity/`, `preprocessing_identity_seed_stability/`, `pure_vae_encode_decode/`, `pure_vae_seed_stability/`
- 3DD-TTA mode and EMA: `result/modelnet40_c/3dd_original/`; full tables in `knowledge/dropout_eval_mode_20260913.md` and `knowledge/ema_inventory_20260913.md`
- Decoder style, SCD normalization, lambda=.96: `result/modelnet40_c/shared_trajectory_decoder_control/`, `scd_normalization_control/`, `scd_lambda96_control/`
- Full GSD screen: `result/modelnet40_c/gsd_latent_spectral_smooth_v2/`; nine ZIPs plus `full_test_screen_summary_20260928-113047_gsd-cal-diagnose-reference-seed0-n64.json`


Raw ZIPs were read only. The matrix is derived documentation; it does not alter
or replace any archive or supplied summary.
