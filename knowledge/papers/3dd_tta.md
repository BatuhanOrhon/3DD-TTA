# Paper Note: 3DD-TTA

## ScanObjectNN severity attribution update - 2026-10-17

[Code/Paper/Inference] The released evaluation loader's `_5.npy` selection
also applies to ScanObjectNN: official `main_3dd_tta.py` selects15 classes
and invokes the same `PointDataset` in `utilities_3dd_tta.py`. Both official
raw files were checked online. The local upstream generator nevertheless
hardcodes8, present since initial author commit `e5b7278`; fork `1f28b29`
kept8 as the new CLI default. Prior ScanObjectNN18.59% was indeed severity8.

Official supplement Appendix A provides stronger numeric parameter evidence
than the old main-text-only audit below: Gaussian sigma0.03, uniform+/-0.05,
background5%, impulse10%, shear+/-0.25 and rotation+/-15degrees map to the
generator's severity5 entries. The main text does not explicitly label the
severity number. This supports severity5 for released ScanObjectNN evaluation
and our next Gaussian dataset; it does not prove authors' raw asset identity.
Some appendix prose/code details differ (cut-out centers/neighbors), so avoid
claiming complete paper corruption parity from the severity label alone.

Source: [official Appendix A](https://openaccess.thecvf.com/content/WACV2025/supplemental/Dastmalchi_Test-Time_Adaptation_of_WACV_2025_supplemental.pdf).
See [the audit, exact history and Colab plan](../colab_scanobjectnn_severity5_20261017.md#primary-source-follow-up-why8-and-evidence-for5).
No inference run was performed for this update.

### ScanObjectNN reproduction result - 2026-10-17

[Run] The released-code-compatible 581-example main split, supplied Point-MAE
checkpoint and exact author-order Gaussian generator give clean71.43%,
severity8 18.59%, and severity5 (`sigma=0.03`)31.50%. Matched controls show
point permutation and clipping each affect about1pp or less. The 2882-example
PB_T50_RS substitution is worse (clean54.79%, Gaussian-s521.51%).

[Run/Inference] A controlled Gaussian severity1–5 sweep yields60.59%,50.95%,
42.69%,36.14%,31.50%. Severity1 happens to approximate Table3's59.0%, but the
paper supplement maps Gaussian to sigma0.03/severity5 and the released loader
selects `_5.npy`. Therefore no honest severity-only change reproduces the
published row. Current evidence supports an unpublished array, checkpoint or
evaluation-protocol difference; it does not identify which one. Exact author
asset hashes or clarification are required before claiming paper parity.

[Code] Changing the local generator default8->5 prevents recurrence of the
known severity8 footgun. It fixes the released-code protocol mismatch, not the
remaining publication gap. Full evidence and archive identities are in the
[dated Colab record](../colab_scanobjectnn_severity5_20261017.md#validated-colab-results-and-pause-handoff).

### ScanObjectNN full severity1 profile - 2026-10-18

[Run/Inference] The13 non-rendered corruptions at seed0/actual severity1 give
49.2784% macro versus69.0% for the same13 Table3 source columns. Gaussian is
60.5852% versus59.0%; every other row is6.36–36.00pp lower. The result rejects
a global severity1 explanation while retaining the Gaussian-specific clue.

[Correction] Table3 orders RBF and inverse-RBF before density/shear/rotation,
while the generator emits shear/rotation/cutout before density/RBF. The first
diagnostic state zipped values positionally and therefore mislabeled several
per-corruption differences, though raw accuracies, macro and MAE were unchanged.
The corrected analysis maps values by corruption name and preserves the raw
artifact. See
[`corrected_table3_comparison.md`](../../result/preparation/cli_scanobjectnn_s1_20261018/corrected_table3_comparison.md).

## GSD host contract audit - 2026-09-23

[Paper] Rechecked Sec. 3.3/Eqs. 9--12 and Algorithm 1, visually including
PDF p. 5: predicted-clean local-latent guidance and updates to conditioning.
[Code] The GSD extension preserves operational unnormalized SCD, retention
.95, gamma-local/eta-style mapping and original-style decode. Paper-normalized
SCD, .96 retention or updated final style are not combined with GSD.
[Inference] A static spectral objective on the same predicted-clean latent
XYZ can compose with SCD by scalar addition and the ordinary chain rule;
see [design](../gsd_integration_20260922.md). This supplies no evidence of
accuracy benefit or paper-level reproduction parity.

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
- **[Paper]** Table 2 and Section 4 identify ModelNet40-C but do not state a severity number. **[Code]** The released upstream evaluation loader hard-codes data_<corruption>_5.npy, and its data guide documents the same _5 layout. Therefore severity 5 is the released-code protocol; attributing it to the paper table itself remains an inference pending an independent primary-source statement.
- Hardware reported for the experiments: NVIDIA A6000.

The current repository default `lambda_cd=0.95` is therefore close but not identical to the paper value.

## Reported ModelNet40-C result

Table 2 of the published PDF reports the Point-MAE source model at **57.6%**, MATE-S at **64.3%**, and 3DD-TTA at **65.7%** mean accuracy. The 3DD-TTA per-corruption row is:

| uniform | gaussian | background | impulse | upsampling | distortion_rbf | distortion_rbf_inv | density | density_inc | shear | rotation | cutout | distortion | occlusion | lidar | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 77.5 | 79.1 | 49.9 | 80.3 | 81.8 | 63.8 | 66.9 | 79.3 | 84.7 | 63.7 | 33.4 | 74.7 | 68.2 | 39.9 | 42.2 | **65.7** |

The repo `README.md` reports **66.1%**, with at least some different cells (for example density decrease and distortion). Its displayed cells average **65.44%**. Treat 65.7 as the published target and 66.1 as the repository's separate, arithmetically inconsistent claim; do not silently substitute them.

**[Paper, correction verified visually 2026-09-15]** The previous headings in
this note incorrectly named/reordered corruptions; the values were unchanged.
The corrected order above follows Table 2, PDF page 7 / printed page 1572.
The 2026-09-13 audit had recorded the correct mapping but had not actually
replaced this table's headings. Table 2 also explicitly reports Point-MAE
source **57.6%**; it is not solely a README reference.

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

### Severity attribution audit (2026-09-12)

- **[Paper]** The 3DD-TTA PDF, Table 2 (p. 7) and Section 4.1 (p. 6), name ModelNet40-C but do not specify a severity level or an aggregation across levels.
- **[Paper]** The cited ModelNet40-C benchmark has 15 corruption types with five severity levels (75 corruption settings): https://arxiv.org/abs/2201.12296 . Its official eval_cor.sh loops severity 1 2 3 4 5: https://github.com/jiachens/ModelNet40-C/blob/master/eval_cor.sh .
- **[Code]** The 3DD-TTA upstream utility loader has selected data_<corruption>_5.npy since its initial public commit e5b7278 (2024-12-05), before the WACV paper publication. README/data instructions also use _5 filenames. This is strong evidence for the released 3DD-TTA evaluation path, but it cannot prove the unpublished run that generated Table 2 used identical code/assets.
- **[Inference]** A lower-severity explanation remains logically possible only through an unpublished evaluation path. It currently has no positive evidence. A source-only severity sweep can identify which local severity best matches the paper's Point-MAE row, but cannot establish publication provenance if checkpoint/data identities differ.

### Current implementation audit (2026-09-15)

See [the main/LION audit](../code_audit_20260915.md) for visually verified
Algorithm-1 updated-style decoding, Eq.-11 point-count normalization, inherited
FPS edge cases, unused mask RNG consumption and staged Colab diagnostics.
Earlier statements that eval-mode accuracy impact is unmeasured are superseded
by the matched all-15 seed-1/2 result in `../dropout_eval_mode_20260913.md`:
eval/raw is the selected operational baseline; EMA remains an ablation.
