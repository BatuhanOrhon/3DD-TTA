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

Current next action: the source-only severity 1--5 probe is complete and shows a 22.1907 pp severity-1-to-5 drop, but no tested severity matches the paper's 57.6% source row. Internal artifact manifests are consistent across the complete severity-5 source-only runs. The canonical Zenodo archive is now identified by the reported API metadata (`modelnet40_c.zip`, 1,970,686,633 bytes, MD5 `c4a7fffaa52c80b33f7b3a0ac7782d3b`), but per-file byte identity with the Colab assets and checkpoint provenance remain unresolved because the workspace contains only data/checkpoint readmes, not the Colab assets. The preprocessing identity control is stable across seeds 0/1/2 at 55.0135% +/- 0.0602 pp sample SD (+1.3236 pp versus deterministic source-only). Pure VAE encode/decode is also stable across seeds 0/1/2 at 54.8469% +/- 0.0790 pp sample SD (+1.1570 pp versus source-only), but remains 0.1297--0.2296 pp below the matched preprocessing identity result at every seed. The common-draw control is deferred because it is not expected to change accuracy and would only strengthen a causal diffusion/guidance claim; that claim remains explicitly open. EMA, GSD/PxP and other datasets remain parked.

After the accepted Batch 1 smoke, follow [Batch 2 source-only instructions](colab_source_only.md).

After the source-only protocol audit, use the [clean Point-MAE control](colab_clean_control.md) before interpreting the remaining corrupted-source gap.

Historical/parked task: [ShapeNet Audit Instructions](colab_shapenet_audit.md).

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
- **[Run]** Batch 1 Gaussian smoke is archived at `result/modelnet40_c/3dd_original/20260912-110541_baseline-smoke_seed0/`: 47/64 on a two-batch prefix, with validated files/config/counts. It is not a full baseline or accuracy comparison.
- **[Run]** Full source-only identity evaluation is archived at `result/modelnet40_c/source_only/20260912-111822_source-only_seed0/`: 53.69% macro over all 15 corruptions, 3.91 points below README's 57.6%. The reproduction gap therefore precedes LION/TTA under this recorded protocol.
- **[Run]** The locked pure VAE control is archived at `result/modelnet40_c/pure_vae_encode_decode/20260919-144513_pure-vae-s5-all15-seed0.zip`: 54.7947% (20,285/37,020), +1.1048 points over source-only but 0.2296 points below preprocessing identity. This places the positive one-seed delta primarily in preprocessing rather than VAE reconstruction; see the findings log.
- **[Run]** Pure VAE seed-stability is archived at `result/modelnet40_c/pure_vae_seed_stability/`: seed 0/1/2 are 54.7947%, 54.9379%, and 54.8082%, respectively; mean 54.8469% with 0.0790 pp sample SD and +1.1570 pp mean over source-only. The early/late timestamp archives are the seed-1/seed-2 runs despite both filenames saying seed1; configs and commands record the true seeds.
- **[Run]** Preprocessing identity seed-stability is archived at `result/modelnet40_c/preprocessing_identity_seed_stability/`: seeds 0/1/2 are 55.0243%, 55.0675%, and 54.9487%, respectively; mean 55.0135% with 0.0602 pp sample SD and +1.3236 pp mean over source-only. The early/late timestamp archives are the seed-1/seed-2 runs despite both filenames saying seed1; configs and commands record the true seeds.
- **[Run]** The paired clean input control at `result/modelnet40_c/source_only/20260912-115021_clean-control_seed0/` obtains **90.64%** (2237/2468) using the same frozen Point-MAE checkpoint, labels and FPS(1024) path. This rules down an obvious clean classifier/data failure, but author-published clean checkpoint parity remains unavailable.
- **[Code]** Active branch is `baseline-repro-clean`, created from main `107305f` in the same repository folder. Batch 0 docs commit is `b31fd23`. Batch 1 adds a smoke runner/artifact module and optional observers; observer-stripped ASTs of the three changed baseline modules match main. LION/dependency files remain unchanged. Legacy GSD/PxP code remains on `pxp-gradient-projection` at `53ba252`; historical method notes refer to that audited revision.
- **[Inference]** Reproducibility risks include unseeded stochastic interpolation/noise, environment drift, scheduler details, checkpoint/data identity, batch-size sensitivity, and incomplete result logging.
- **[Code]** Original LION trainer inference disables dropout; the inherited demo wrapper leaves LION in training mode. **[Run]** A same-commit all-15 seed-1/2 screen favors eval mode by **+.7577 +/- .1203 pp** and 13/15 corruption means. It is seed-controlled rather than common-draw paired, so eval is a provisional operational baseline rather than an isolated causal dropout claim; see [the full screen](dropout_eval_mode_20260913.md).
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

Read the [current session handoff](session_handoff_20260915.md). It supersedes historical smoke/ShapeNet-next-action statements above and records the selected eval/raw baseline, EMA decision, current blockers, and immediate Colab experiment. Numerical experiments remain Colab-only.
