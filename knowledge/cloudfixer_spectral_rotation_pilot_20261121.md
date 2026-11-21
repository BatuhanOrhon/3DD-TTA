# CloudFixer XYZ spectral prefilter for rotation fitting

[Open/Code plan] 2026-11-21. The user authorizes a small, paired A100 pilot after the current rotation run. Hypothesis: a graph low-pass proxy can improve rigid correspondences on Background while preserving Rotation benefit. Sparse outliers and disconnected components can survive low-pass filtering. Measure geometric and classification effects independently.

- Build a mutual 12-nearest-neighbor graph over valid observed points and, independently, each pre-correction denoiser clean estimate used for fitting. Use self-tuned symmetric Gaussian weights, omit self edges, and apply three steps of `H = I - L/(2d_max)` for the combinatorial Laplacian. The polynomial response `(1 - λ/(2d_max))^3` attenuates high frequencies without a dense eigenbasis. Graph discovery still uses pairwise distances. Reuse the filtered unrotated input throughout a run, rotate it by the accumulated pose for matching, and filter the same target only once when t80 is revisited.
- Fit Kabsch on the two filtered proxies. Apply only the resulting rotation to the original physical reference. The proximal correction, diffusion states and classifier use the existing raw clouds. Compare single t80, fixed-target three t80 updates and progressive t80/t50/t20 with and without spectral filtering, plus identity N8. Fix 25° travel and N8/K80/tau=.1/95%/PM[20,10,0], 9/3 model queries for every arm. Record graph construction, filter, matching, SVD, denoiser and classifier CUDA time, wall time and peak allocated MiB.
- Select 12 IDs disjoint from both existing development and validation lists by SHA-256 ordering of `cloudfixer-spectral-rotation-20261121:<ID>`. Reuse them at clean and all 15 severity-5 corruptions across seeds 0/1/2, B1. Test known small/compound/large transforms on clean and Background at seed 0. Freeze graph and schedule choices before looking at spectral outcomes. The historical full benchmark has prior exposure. The 12 IDs are held out from the gradient-free selection and validation.
- Descriptive continuation screen vs the matched unfiltered schedule: Background improvement of at least +2 pp, Rotation change ≥0 pp, clean and macro14 point losses ≤2 pp, and mean synchronized latency ratio ≤1.20. Report object-clustered simultaneous intervals and poor individual corruptions. Treat a pass as pilot evidence for larger validation, not a default promotion.
- Same recorded A10040 runtime and verified pinned assets. Separate working source snapshot and Drive root `MyDrive/thesis/cloudfixer_spectral_rotation_20261121`. Seven arms ×576 benchmark inputs plus synthetic diagnostics. Expected wall time 18–25 min plus packaging, roughly 2–3 CU at 6.73 CU/h. Cap at 35 min of scientific work, about 4 CU. Preserve and verify every block and phase archive, audit before same-VM recovery, stop on source/configuration or matched-control failure. This authorization applies to the small follow-up. Runtime teardown requires the separate completion and close instruction.

[Paper] Cite Gabriel Taubin for polynomial Laplacian low-pass smoothing of geometric coordinates, and Hammond, Vandergheynst and Gribonval for spectral graph filtering without explicitly computing the eigenbasis. The fixed three-step mutual-kNN proxy and its integration are the evaluated choices. Retain CloudFixer, 3DD-TTA, Kabsch, Trimmed ICP and Lawrence et al. attribution from the gradient-free pilot. [Taubin 1995](https://doi.org/10.1145/218380.218473) · [Hammond et al. 2011](https://arxiv.org/abs/0912.3848)

[Paper] The local Gaussian scale follows the k-th-neighbor bandwidth idea from Zelnik-Manor and Perona. Here it is used on a mutual-neighbor graph for coordinate smoothing. [Self-Tuning Spectral Clustering, §2](https://proceedings.neurips.cc/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html)

## Completed pilot

[Run] Source `78850f5` on the existing A100-SXM4-40GB, with the same pinned Point2Vec and XYZ prior. Both smoke objects passed exact identity equivalence. The pilot completed 48 blocks, 576 paired inputs and 4,032 benchmark policy evaluations, plus 672 synthetic policy evaluations. SHA-256 and CRC of the 7,369,701-byte archive passed. The independent local recount verified the recorded logits, probabilities, accuracy, repairs, harms, seed effects, proper rotations and query/graph counts. [Run receipt](evidence/cloudfixer_spectral_rotation_pilot_20261121/archive_receipt.json)

Each column compares the spectral variant with the same unfiltered schedule in percentage points. The [paired table](evidence/cloudfixer_spectral_rotation_pilot_20261121/paired.csv) contains both correct/total counts, accuracy, repairs and harms, per-seed differences and simultaneous 95% intervals. Each individual condition has 36 predictions over 12 object clusters. The simultaneous radius is 13.89 pp.

| Condition | Single t80 | Fixed three t80 | Progressive t80/t50/t20 |
|---|---:|---:|---:|
| clean | +0.00 | -2.78 | +0.00 |
| background | +0.00 | +0.00 | +0.00 |
| cutout | -2.78 | +0.00 | -2.78 |
| density | +2.78 | -2.78 | +0.00 |
| density_inc | -2.78 | +8.33 | +0.00 |
| distortion | +0.00 | +0.00 | +0.00 |
| distortion_rbf | +0.00 | +2.78 | -5.56 |
| distortion_rbf_inv | +5.56 | +0.00 | +2.78 |
| gaussian | -2.78 | +0.00 | +0.00 |
| impulse | -2.78 | +2.78 | +0.00 |
| lidar | +0.00 | +0.00 | +2.78 |
| occlusion | -2.78 | -2.78 | -2.78 |
| rotation | -2.78 | +2.78 | +0.00 |
| shear | +0.00 | -2.78 | +0.00 |
| uniform | +0.00 | +5.56 | -2.78 |
| upsampling | +0.00 | +0.00 | +0.00 |
| macro14 | -0.40 | +0.79 | -0.60 |
| macro13 | -0.43 | +0.85 | -0.64 |
| all15 | -0.56 | +0.93 | -0.56 |

| Schedule | Unfiltered → filtered B1 ms | Relative wall increase | Filtered graph + sweeps (ms) | Peak allocated before → after (MiB) | Filtered model time projected full seed 2 (h) |
|---|---:|---:|---:|---:|---:|
| single | 252.75 → 256.07 | 1.31% | 2.62 | 554.35 → 554.39 | 2.64 |
| fixed | 260.77 → 264.50 | 1.43% | 2.61 | 554.35 → 554.39 | 2.72 |
| progressive | 261.10 → 267.33 | 2.39% | 5.21 | 554.35 → 554.39 | 2.75 |

[Run] Single and fixed used exactly two graph builds per input. Progressive used four. Each arm kept nine denoiser and three classifier calls. The pilot took about 21 min 38 s from its run ID to verified archive. Displayed account balance moved 67.15 → 64.35 CU across the follow-up window including smoke and deployment. Baseline session rate 6.73 CU/h. [Component costs](evidence/cloudfixer_spectral_rotation_pilot_20261121/gpu_cost.csv) · [Counts by seed](evidence/cloudfixer_spectral_rotation_pilot_20261121/per_corruption.csv)

[Run/Inference] On the compound synthetic transform with Background, inverse-angle error improved 0.590° with fixed unfiltered Kabsch and 0.895° with the filter. The filter therefore added 0.305° in this diagnostic. With a 60° rotation and Background, angular error increased by 0.259° without the filter and 0.482° with it. All three spectral schedules left Background accuracy exactly at 18/36, with no repairs or harms against their schedule-matched controls. This pilot establishes a small measured cost for spectral matching and no observed Background accuracy benefit. High-frequency attenuation need not remove a disconnected outlier. We did not measure an outlier-removal fraction. [Pose diagnostic](evidence/cloudfixer_spectral_rotation_pilot_20261121/synthetic.json)

[Decision] No schedule passed. Fixed gained one Rotation prediction and five across all15, while losing one clean prediction. Its Background matched control is unchanged and the simultaneous intervals accommodate gains and harms. The other placements also failed the Background screen. No larger run was launched. The same r9 A100 remains live. [Pilot decision](evidence/cloudfixer_spectral_rotation_pilot_20261121/decision.json)

## Integration scope

[Code/Run] Branch `experiment/cloudfixer-rotation-pilot`, starting from `6e812b400d7f27451256a1d2ae680344e8b7f604`. Scientific source snapshots `1367d39` (original comparison) and `78850f5` (spectral follow-up). Integration includes opt-in sampler/runner changes, the two frozen cohort registries, regression tests, citations and compact evidence. Raw ZIPs, checkpoints, extracted dataset and laptop operator files stay in local/Drive storage. Focused CPU suite 47 passed, both GPU audits passed, and every required corruption row and seed was independently counted. See the [original pilot](cloudfixer_rotation_pilot_20261121.md) for baseline-suite limitations and the recovered JSON decision.
