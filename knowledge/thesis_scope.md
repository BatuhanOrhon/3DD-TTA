# Thesis Scope and Research Questions

## Working title

**Graph-Spectral and Conflict-Aware Test-Time Adaptation of Corrupted 3D Point Clouds with Denoising Diffusion Models**

This is a working title, not a finalized thesis claim.

## Problem statement

Point-cloud classifiers trained on clean source data degrade under unseen corruptions. Source data may be unavailable at deployment, and retraining or changing the classifier may be undesirable. The thesis investigates input-level, training-free adaptation that uses a pretrained 3D diffusion prior to transform each corrupted point cloud into a representation that the frozen source classifier can classify more reliably.

The base method, 3DD-TTA, guides LION's local latent denoising with a robust geometric reconstruction objective. The fork adds graph-spectral guidance intended to preserve global/mesoscopic structure and explores gradient-conflict handling when geometric and spectral objectives disagree.

## Primary research question

Can graph-spectral structural guidance, alone or with conflict-aware gradient composition, improve the corruption robustness of 3DD-TTA on ModelNet40-C while retaining its training-free, frozen-classifier setting?

## Secondary questions

1. Which part of the current accuracy gap is caused by environment, data/checkpoint identity, preprocessing, stochasticity, scheduler configuration, or metric aggregation?
2. Is a static spectral basis computed from the corrupted local latent useful, or must the basis be recomputed during denoising?
3. Do low-, mid-, or high-frequency constraints carry useful signal in LION's local latent space?
4. Does a graph built on physical coordinates define a meaningful basis for LION latent features when point correspondence is retained?
5. When SCD and spectral gradients conflict, does one-way or symmetric projection improve accuracy, stability, or both?
6. Does adapting/denoising the global latent help, and what is the correct LION decoder/style contract?

## Scope boundaries

### Implementation policy agreed on 2026-09-12

Start the new branch directly from main, not a GSD/PxP development branch. Future legacy branches are references for ideas and code inspection, not implementations to copy wholesale. Prefer simpler, smaller code; independently test inherited assumptions and repair confirmed omissions/errors before using a component. Introduce one reviewed increment at a time and preserve original baseline math as the comparator. GSD work is explicitly deferred while the baseline/artifact/dropout controls are established.

### In scope

Research workflow preference (2026-09-12): do not add unit-test files by default. Verify proportionately through syntax, baseline/protocol checks and controlled archived Colab experiments; local checks are not accuracy evidence.

- ModelNet40-C, severity 5, initially with the repository's Point-MAE classifier and LION checkpoints.
- Frozen classifier and training-free test-time input adaptation unless explicitly declared otherwise.
- Original 3DD-TTA reproduction and controlled ablations.
- Static/dynamic latent spectral guidance, physical-basis experiments, dual global/local denoising, and PxP-inspired gradient projection.
- Accuracy, per-corruption behavior, variance, runtime, memory, and mechanism diagnostics.

### Out of scope unless separately approved

- Claiming a reproduction of full GSDTTA while omitting its learned spectral shift and model-adaptation stage.
- Comparing numbers across different classifiers/backbones as if protocols were identical.
- Hyperparameter search on the final test set followed by an unbiased generalization claim.
- Treating a single stochastic run or a partial 25-sample grid search as a definitive improvement.
- Rewriting the classifier or diffusion model training pipeline before the baseline gap is characterized.

## Candidate contributions, not yet established

- A graph-spectral regularizer applied inside a hierarchical 3D latent diffusion TTA trajectory.
- An analysis of static versus dynamic graph bases in a changing latent space.
- Conflict-aware composition of SCD and spectral guidance gradients for 3D diffusion TTA.
- A reproducibility and diagnostic study of 3DD-TTA on ModelNet40-C.

Each item becomes a thesis contribution only after controlled evidence, ablations, statistical reporting, and comparison against a verified baseline.

## Success criteria

Minimum scientific success is a defensible explanation of the reproduction gap and a reproducible baseline. A method improvement should additionally show:

- repeated-seed improvement over the matched baseline;
- per-corruption results, not only a mean;
- uncertainty or variance reporting;
- a mechanistic diagnostic consistent with the proposed explanation;
- compute/memory costs;
- no hidden change in data, checkpoint, classifier, preprocessing, or evaluation set.

## Current empirical status

- **[User report]** Original code: approximately 63% mean accuracy.
- **[User report]** Best current GSD variant: approximately 63.5%.
- **[Open]** Exact commands, seeds, environment, per-corruption values, checkpoint hashes, and raw logs have not yet been archived under `result/`.

Therefore no gain is yet treated as confirmed.
