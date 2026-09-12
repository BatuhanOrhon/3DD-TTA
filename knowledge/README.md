# 3DD-TTA Thesis Knowledge Base

This directory is the canonical, repository-local memory for the thesis. A new research session must start here instead of reconstructing the project from chat history.

## Current thesis in one paragraph

The project studies training-free test-time input adaptation for corrupted 3D point clouds. The base system is 3DD-TTA, which uses a pretrained LION latent diffusion model and Selective Chamfer Distance (SCD) guidance to reconstruct a classifier-friendly point cloud while keeping the source classifier frozen. This fork investigates whether graph-spectral structure constraints inspired by GSDTTA, dynamic spectral bases, global/local diffusion variants, and PixelAsParam-inspired conflicting-gradient projection can improve ModelNet40-C robustness. The immediate scientific priority is not another variant: it is first closing or explaining the gap between the published/repository baseline and locally observed Colab results under a controlled, reproducible protocol.

## Required reading order

1. [Thesis scope](thesis_scope.md)
2. [Reproduction gap](reproduction_gap.md)
3. [Experiment protocol](experiment_protocol.md)
4. [Repository map](repository_map.md)
5. [Method synthesis](method_synthesis.md)
6. Paper notes: [3DD-TTA](papers/3dd_tta.md), [GSDTTA](papers/gsdtta.md), [PixelAsParam](papers/pixelasparam.md)
7. [Development history](development_history.md)
8. [Findings log](findings_log.md) and [open questions](open_questions.md)
9. [Source index](source_index.md)

Before clean-restart implementation also read the [follow-up code audit](code_audit_20260912.md) and [small implementation batches](clean_restart_batches.md). These refine earlier informal next-test ordering.

Current next action: [Batch 1 Colab smoke instructions](colab_baseline_smoke.md). Research preference: no unit-test files by default; proportionate local structural/protocol checks and archived Colab experiments.

## Evidence labels

Use these labels in all future updates:

- **[Paper]** Directly reported in a cited paper.
- **[Code]** Verified in the named repository revision/file.
- **[Run]** Supported by an archived artifact under `result/`.
- **[User report]** Reported conversationally but not yet backed by an archived artifact.
- **[Inference]** A reasoned interpretation that still needs a controlled test.
- **[Open]** Unknown, ambiguous, or awaiting evidence.

Never silently promote an inference or user report into a run-backed finding. If sources disagree, retain both values and describe the disagreement.

## Headline state as of 2026-09-12

- **[Paper]** The WACV 2025 3DD-TTA paper reports **65.7%** ModelNet40-C mean accuracy for its Point-MAE setting.
- **[Code]** The repository README reports **66.1%**; several per-corruption cells differ from the published table. Both references must remain visible.
- **[User report]** Local original-code runs are approximately **63%**, while current GSD variants reach approximately **63.5%**. No complete run bundle is currently archived, so these numbers are provisional.
- **[Code]** Active branch is `baseline-repro-clean`, created from main `107305f` in the same repository folder. Batch 0 docs commit is `b31fd23`. Batch 1 adds a smoke runner/artifact module and optional observers; observer-stripped ASTs of the three changed baseline modules match main. LION/dependency files remain unchanged. Legacy GSD/PxP code remains on `pxp-gradient-projection` at `53ba252`; historical method notes refer to that audited revision.
- **[Inference]** Reproducibility risks include unseeded stochastic interpolation/noise, environment drift, scheduler details, checkpoint/data identity, batch-size sensitivity, and incomplete result logging.
- **[Code]** Original LION trainer inference disables dropout; the identical demo wrapper used here does not, and current baseline/GSD setup leaves LION in training mode. **[Inference]** Accuracy impact is unmeasured; isolated dropout A/B is the first adaptation-behavior test.
- **[Code]** Mean spectral loss plus summed SCD changes relative guidance with batch size. **[User report]** Mean settings previously outperformed sum; sum/smaller-step tuning is deferred until baseline and spectral-off parity.
- The user selected same-folder development and requested skill/knowledge preservation. No new stash was needed; existing dev stash is untouched. Research memory/skill/result protocol were committed in `b31fd23`; source PDFs remain local/untracked. Batch 1 local checks passed but Colab smoke acceptance is pending. No accuracy improvement or dropout effect is established.
- **[Code]** The fork's spectral method is not a reproduction of full GSDTTA: it regularizes a LION local latent and keeps the classifier frozen instead of learning a physical-coordinate spectral shift and alternating input/model adaptation.
- **[Code]** The fork's PxP variants are PixelAsParam-inspired gradient-conflict methods between spectral and Chamfer guidance; they do not implement PixelAsParam's denoising/diversity/classification decomposition.

## Update rule

After any material experiment, diagnosis, code change, or paper review:

1. archive the immutable output under `result/` using [the experiment protocol](experiment_protocol.md);
2. append the outcome—positive, negative, or inconclusive—to [findings_log.md](findings_log.md);
3. update affected method/reproduction notes and [open_questions.md](open_questions.md);
4. include date, Git commit, run path, evidence label, and what would falsify the conclusion;
5. do not erase superseded findings—mark them superseded and link the newer evidence.

## Terminology

- `z` / global latent: LION global shape/style representation.
- `h` / local latent: per-point LION latent, represented in this repo as `B × 2048 × 4` for spectral work.
- `U`: graph Laplacian eigenvector matrix (a basis), not a single vector. `U_current` is recomputed in dynamic variants.
- SCD: Selective Chamfer Distance.
- TTA: test-time adaptation.
- PxP-inspired projection: this fork's use of projection for conflicting loss gradients; not an exact implementation of PixelAsParam.

## First action for the next session

Request the Batch 1 smoke ZIP following [the Colab handoff](colab_baseline_smoke.md), or inspect a supplied bundle. An older approximately-63% run bundle is also valuable if available. Batch 1 local implementation exists; do not reimplement it or advance to source-only before reviewing the smoke. Then follow the source-only/dropout/baseline ladder in [clean restart batches](clean_restart_batches.md). Numerical experiments remain Colab-only.
