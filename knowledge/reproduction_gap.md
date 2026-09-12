# Reproduction Gap: Current Diagnosis

## The three numbers that must remain separate

| Value | Status | Meaning |
|---:|---|---|
| 65.7% | **[Paper]** | Published WACV 2025 ModelNet40-C mean for 3DD-TTA with Point-MAE |
| 66.1% | **[Code]** | Mean shown in the repository README's 3DD-TTA table |
| ~63% | **[User report]** | Local Colab run of the original code; no archived run bundle yet |
| ~63.5% | **[User report]** | Local best GSD-script observation; no archived run bundle yet |

The local gap is therefore approximately 2.7 points from the published paper or 3.1 points from the README, pending exact artifacts. The approximately 0.5-point GSD improvement is not yet statistically or procedurally verified.

## What has already been ruled down

- **[Code]** Current `tta.py` is almost the upstream baseline; method additions are primarily in separate scripts. The low baseline is unlikely to be caused simply by GSD logic contaminating the original path.
- **[Code]** ModelNet40-C loader selects severity 5, matching the stated benchmark severity.
- **[Code]** The main path performs the LION scale/rotation and Point-MAE resampling operations expected by the repository.

These observations narrow the search but do not prove parity.

## High-priority explanations

### P0 — Data and checkpoint identity

Different downloads, corruption-generation versions, incomplete files, or checkpoints can shift all methods. Record file paths, sizes, and SHA-256 hashes for:

- Point-MAE checkpoint;
- LION checkpoint(s);
- ModelNet40-C archive or every evaluated corruption file;
- any normalization/config assets.

### P0 — Environment drift

`requirements.txt`, `env.yaml`, and the Colab notebook do not define one identical environment. PyTorch, CUDA, `diffusers`, kNN/FPS extensions, NumPy, and compiled point-cloud ops may differ. The original requirements pin an old `diffusers`, while the Colab path may resolve a newer scheduler implementation.

### P0 — Stochasticity and aggregation

Random interpolation and diffusion noise are unseeded; CUDA settings may also be nondeterministic. Batch size changes the order/association of random draws. A single run can therefore differ from another even with identical arguments. Existing logs do not report variance.

Confirm that the reported mean is the macro-average of all 15 per-corruption accuracies and that resumed CSVs do not duplicate or omit corruptions.

### P1 — LION inference mode: confirmed discrepancy

**[Code]** Original LION trainer evaluation calls VAE/prior eval, while the fork inherits its byte-identical demo wrapper with no eval calls. Baseline/GSD model setup puts only Point-MAE in eval. The supplied config activates 0.1 dropout in the local prior and local encoder/decoder. Frozen weights/no_grad do not disable dropout. Original LION revision: `7711b3d185752eeb632d095494876e4de15f3195`; exact sources and wrapper hash are in [the follow-up audit](code_audit_20260912.md).

**[Inference]** Accuracy/variance impact is not measured and published 3DD-TTA runtime modes are unknown. Prioritize isolated legacy/eval A/B after artifact/source-only identity controls, without changing scheduler/rates/style/lambda. Preserve guidance input gradients and explicitly pair interpolation/VAE/noise draws.

### P1 — Scheduler semantics

The baseline and GSD paths do not currently instantiate DDIMScheduler identically. `set_alpha_to_one`, timestep spacing, prediction type, clipping, and library version can alter reconstruction. Compare scheduler configs serialized at runtime, not source defaults alone.

### P1 — Hyperparameter/protocol mismatch

The paper describes `lambda=0.96`, while repository defaults use 0.95. Background versus normal reverse steps, batch size, guidance rates, sampling points, and classifier voting/augmentation must be matched exactly. Even small differences can interact with loss sums.

### P1 — Output/resume contamination

Baseline output is append-oriented and GSD CSV resume logic can combine incompatible configurations if filenames are reused. Existing CSVs omit important configuration fields. Every new run must use a fresh immutable directory.

## Lower-priority but important code questions

- Which global/style representation should be used for final decoding?
- Are gamma and eta semantically mapped as intended when they are unequal?
- Does the classifier checkpoint load strictly and completely in both baseline and GSD paths?
- Do `knn_cuda`, FPS, and custom extensions behave identically across Colab GPU/runtime versions?
- Is the data order/class label map identical to the reported setup?

## Controlled diagnosis sequence

1. Preserve/archive the existing failing run if available; establish a reviewed clean main-based branch without losing knowledge/user work.
2. Add artifact/seeding/count controls while preserving original adaptation mathematics; inspect data/checkpoint hashes and environment.
3. Verify the complete source-only corruption row before interpreting LION effects.
4. Test original TTA legacy versus LION eval mode on Gaussian/background, seeds 0,1,2 and explicitly controlled common draws.
5. Archive direct LION encode/decode reconstruction (distinct from unguided perturbed DDIM), then full original 3DD-TTA under the selected mode.
6. Serialize schedulers; investigate dependency, lambda, steps or batch one factor at a time if the gap persists.
7. Require spectral-off GSD tensor/prediction parity with the selected baseline, then turn on one spectral band.
8. Defer sum/small-step, dynamic and PxP trials until these gates pass. See [execution batches](clean_restart_batches.md).

## Decision rule

A cause is accepted only if changing that single factor produces a repeatable, directionally consistent change and restoring it restores the old behavior. Correlation with a branch or environment label is not sufficient.

## Current conclusion

**[Inference]** The verified LION mode discrepancy is now a high-priority one-factor test alongside data/checkpoint/environment controls; no accuracy root cause is confirmed. This refines the earlier environment/stochasticity-first hypothesis rather than disproving it. Required evidence is a complete baseline bundle and controlled dropout A/B, not another unmatched method sweep.
