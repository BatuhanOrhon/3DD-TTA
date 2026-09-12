# Clean Restart: Small Implementation Batches

## Latest progress - Batch 1 local implementation, Colab gate pending

2026-09-12 **[Code]**. Batch 0 documentation was committed as b31fd23193bbcb9a5c189cfb4118be41506f9333; user accepted Batch 1. Added run_baseline.py and research_artifacts.py plus optional observers in three baseline modules. No GSD, LION eval, scheduler/style/rate/reduction changes or dependency changes. Observer-stripped ASTs of all three modified baseline files match main. Syntax/CLI and temporary count/schema/collision/failure-state checks passed; no unit-test files were added per user preference.

This is implementation evidence, not a Colab run. [Smoke handoff](colab_baseline_smoke.md): user runs Gaussian/two batches/seed0 with explicitly declared batch32/lambda0.95 and supplies the seven-file ZIP. Batch 1 is not accepted end-to-end until that bundle is validated. Do not implement Batch 2 yet. Older Batch 0 status paragraphs and the previous implementation handoff below are historical snapshots, superseded by this progress note.

## Status, assumptions and invariants

Recorded 2026-09-12; legacy audit revision `53ba252519c7cf65f836a9c1c564027142ab1573`. The user chose the same repository folder and approved preserving skill/knowledge. `baseline-repro-clean` is now checked out, based on main `107305fd7baf40b359f31c07d235599198be7324` (also origin/main and upstream/main). No worktree, Python change or push. Only curated research documentation/skill/result protocol are selected for the local documentation commit.

### Batch 0 progress — 2026-09-12

**[Code]** `git branch baseline-repro-clean main` succeeded; branch/main refs match. `git diff --exit-code` confirms no tracked working-tree change, and the pre-existing untracked files remain. Normal checkout detected (`git-dir == git-common-dir == .git`); no linked worktree created. Switching in place would remove legacy tracked experimental scripts/PDFs from the working folder, although Git retains them at `53ba252`. A user preference question was issued for isolated worktree versus same-folder checkout before that transition. No destructive cleanup or dependency installation was performed; all numerical runs remain Colab-only.

The initial partial status is superseded by the following same-folder transition. The next code work must use the clean branch, not the legacy checkout.

### Batch 0 placement and preservation — 2026-09-12

**[Code]** User chose in-place branch switch. Tracked/index state was clean, so no new stash was made; existing dev stash `02ddaf533c93164d69643e43c55ad37df6fa0343` is unchanged. `git switch baseline-repro-clean` succeeded. Four absent root scholarly PDFs were restored to the working folder from the legacy branch without staging/overwriting; the user's untracked PixelAsParam PDF and unrelated notes/scripts were untouched. SHA-256 checks verified all 23 selected memory/skill/protocol/PDF files were identical across the transition before status-note updates.

Tracked Python/requirements/environment match main. Legacy experiments remain on `pxp-gradient-projection` (`53ba252`); no cherry-pick or algorithm migration. Select only `knowledge/`, the two iterative-thesis-researcher skill files, and `result/README.md` for the documentation commit. Exclude PDFs, docs/superpowers and tmp/inspect_gsd_pdf.py. Verify staged paths, links and reference parity before commit; record the resulting commit in the handoff. Batch 0 has no Colab/numerical test; Batch 1 follows user review of this handoff.

Read the knowledge index, `code_audit_20260912.md`, `reproduction_gap.md`, `experiment_protocol.md` and `open_questions.md` first. `knowledge/` is canonical; `.codex/knowledge/INDEX.md` is absent. Implementation is handed to `$development-agent` one selected batch at a time. Local pure tests/compilation are not accuracy evidence. All evaluations/training run in the user's Colab.

Preserve legacy work. PDF sources and unrelated docs/tmp files remain untracked; do not broad-stage them. Memory/skill/result protocol are deliberately tracked through the documentation commit. No reset/clean or wholesale cherry-picking. Main ancestry does not guarantee correct inference behavior.

## Batch 0 - Preserve context and create clean ancestry

- Goal: safe, auditable baseline branch with research memory retained.
- Scope: ancestry/status checks, branch creation from verified main, selected documentation preservation; no mathematics changes.
- Touched areas: Git metadata; curated knowledge, existing researcher skill and result/README.md.
- Stack context: Windows workspace, Python/Colab repository.
- Dependencies: confirm main hash, branch-name/path collisions, tracked edits and user files.
- Implementation notes for $development-agent: create baseline-repro-clean when safe; preserve unrelated work, document starting hash/reference-file differences, selectively review staging. No dependency changes.
- Verification: git status --short, git rev-parse main HEAD, git merge-base HEAD main, explicit baseline-file diff and memory existence checks.
- Knowledge artifact to update: this file, findings_log.md, development_history.md.
- User review gate: review ancestry/status/selected diff before code work.
- Out of scope: legacy method migration, training, destructive cleanup.

## Batch 1 - Reproducible runner and output contract

- Goal: instrument baseline behavior without changing adaptation equations.
- Scope: seeds/corruption selection, actual config/environment/hash/mode logging, raw logs, count-based CSVs and fresh immutable directories. No resume initially.
- Touched areas: small new wrapper/helpers and pure-logic tests; minimal entry-point wiring only if needed.
- Stack context: Colab CUDA, seven-file bundle specified in result/README.md.
- Dependencies: Batch 0 accepted.
- Implementation notes for $development-agent: retain original scheduler, 5/35 steps, static final style, gamma/eta mapping, lambda and chosen fixed batch. Log resolved scheduler, module modes, checkpoint keys/hashes and actual CLI values. Do not silently change CuDNN/determinism between experiment arms.
- Verification: compile touched files; pure schema/count/macro/micro/duplicate/path-collision tests; Colab 1-2 batch smoke produces seven files with consistent configs/counts.
- Knowledge artifact to update: experiment_protocol.md, findings_log.md, this file.
- User review gate: user supplies smoke ZIP; validate before advancing. Smoke accuracy is not benchmark evidence.
- Out of scope: dropout/reduction/style/rate changes, classifier adaptation.

## Batch 2 - Source-only identity gate

- Goal: locate divergence before LION.
- Scope: source Point-MAE on all 15 severity-5 corruptions, with existing tools/tta_abl.py:136-216 as preprocessing reference.
- Touched areas: source-only adapter/runner and count checks; no TTA equations.
- Stack context: same Colab checkpoint/data/FPS setup.
- Dependencies: validated artifact smoke.
- Implementation notes for $development-agent: do not apply LION's scale/rotation to source-only input. Record FPS, point counts, label order and complete source row. Compare 57.6% as a protocol-specific reference, not a value to force by tuning. Local hashes identify assets but cannot prove parity with unavailable reference hashes.
- Verification: label/count/key reports, all 15 rows and recomputed macro/micro; user full source-only run.
- Knowledge artifact to update: reproduction_gap.md, findings_log.md, open_questions.md.
- User review gate: inspect ZIP and per-corruption row; investigate material unexplained mismatch before interpreting diffusion effects.
- Out of scope: classifier training or preprocessing tuning to match the target number.

## Batch 3 - Only LION dropout A/B

- Goal: measure accuracy/variance effect of original-LION-compatible evaluation mode.
- Scope: explicit legacy/eval switch after load; eval arm uses lion.vae.eval() and lion.priors.eval(). Only behavioral factor is module mode.
- Touched areas: setup/runner and mode assertions; reference TTA math remains unchanged.
- Stack context: Gaussian/background, complete selected-corruption data, seeds 0,1,2 and fixed batch.
- Dependencies: output/source-only gates reviewed; data/checkpoint/environment/config locked.
- Implementation notes for $development-agent: keep input/style guidance gradients enabled. Replay interpolation choices, VAE standard-normal draws and initial diffusion noise; same initial seeds alone do not ensure pairing because dropout consumes RNG. Identical encoded latents alone would mask encoder dropout, so use them only for supplementary fixed-state prior/decoder diagnostics. If exact draws are not controlled, label the run seed-controlled, not fully paired. Log actual modes/dropout inventory, noise identity, finite gradients, per-corruption counts, runtime/memory.
- Verification: mode tests and fixed-state repeated-forward diagnostic; finite/nonzero guidance gradients; user three-seed A/B Colab bundles and independent metric recomputation. No simultaneous scheduler/lambda/style/rate changes.
- Knowledge artifact to update: audit, reproduction_gap.md, findings_log.md, open_questions.md.
- User review gate: evaluate paired changes/variance, retain null/negative results; no guaranteed gain.
- Out of scope: spectra, sum loss, unequal eta/gamma, global-prior denoising.

## Batch 4 - Lock the baseline ladder

- Goal: archive source-only, direct LION reconstruction and original SCD-guided TTA under the selected mode.
- Scope: direct encode/decode without guidance (distinct from unguided perturbed DDIM), all-15 original TTA; separate one-factor diagnostics if gap persists.
- Touched areas: reconstruction adapter and runner, not new research guidance.
- Stack context: full severity-5 benchmark, fixed seeds/config; repeated seeds subject to declared compute budget.
- Dependencies: Batch 3 interpreted.
- Implementation notes for $development-agent: retain paper lambda 0.96 and repository lambda 0.95 as separately named protocols. Scheduler/lambda/batch-size ablations are distinct sub-batches. Select baseline from evidence, not its usefulness as a comparator.
- Verification: complete ladder rows/counts/means/variance, serialized scheduler/config/environment and checkpoint-key agreement; user Colab tests.
- Knowledge artifact to update: reproduction_gap.md, experiment_protocol.md, findings_log.md.
- User review gate: gap closed or explicitly characterized with reproducible evidence before method claims.
- Out of scope: tuning GSD or mixing comparator protocols.

## Batch 5 - Minimal static spectrum, off before on

- Goal: isolate spectral contribution and avoid inherited defects.
- Scope: small tested graph/static-guidance module; first zero spectral weights, then one low band, then separately reviewed extra bands.
- Touched areas: new graph/guidance/config adapter/tests; no wholesale old eval/analyzer/resume copy.
- Stack context: fixed mode/scheduler/steps/style/batch/data/common draws; initial mean spectral loss and verified baseline SCD reduction.
- Dependencies: baseline ladder accepted; visually verify threshold symbols and test self-neighbors/distance units/isolation choices first.
- Implementation notes for $development-agent: validate default/partial dictionaries and band boundaries; document paper deviations. Preserve latent-first-three-channel SCD. Require spectral-off tensor/prediction parity, not only rounded mean accuracy. Log band energy/gradient/update scales. Dynamic/PxP stay disabled.
- Verification: known tiny graphs, H/U shapes, finite gradients, caller tests, declared numerical off-path parity tolerance; paired low-band Colab pilot.
- Knowledge artifact to update: method_synthesis.md, papers/gsdtta.md, findings_log.md, open_questions.md.
- User review gate: accept spectral-off parity before enabling spectra; review low-band evidence before expansion.
- Out of scope: sum/small-step sweep, dynamic/physical/dual/PxP variants.

## Deferred experiments and result handoff

User-requested future sum plus smaller eta/gamma trials follow baseline/spectral-off parity. Separate exact reduction-equivalence controls from weight/rate sweeps; smaller shared rates change SCD too. Each band has its own element count. Mean-versus-sum accuracy advantage remains User report until archived. Compare normalized diagnostics and actual update norms, not raw loss numbers across reductions.

Dynamic projector diagnostics, per-sample PxP with conflict logging, style/global ablations and other datasets are separate future batches. Declare validation/test-set tuning and lock final settings before benchmark claims.

After every run request the complete ZIP with command.txt, config.json, environment.txt, stdout.log, summary.csv, per_corruption.csv and notes.md, remove credentials, validate/archive under result/<dataset>/<method>/<timestamp>_<short-name>/, update knowledge, then decide the next batch. Do not proceed automatically past an unreviewed Colab gate.

## Next handoff — Batch 1 after branch/documentation review

> $development-agent: After the Batch 0 handoff is reviewed, execute Batch 1 only on baseline-repro-clean in the existing folder. Read knowledge/README.md, audit and experiment protocol. Add minimal reproducible runner/artifact helpers and pure schema/count tests without changing original TTA math, dropout mode, scheduler, steps, style, rates or lambda. Record actual configuration and environment, output seven-file immutable bundles, start without resume. Verify locally only in proportion to available dependencies; user performs CUDA smoke in Colab. Update findings/protocol/this note and request the smoke ZIP. Do not start source-only/full accuracy or dropout changes in this batch.

## Current handoff - validate the Colab smoke ZIP

The implementation handoff above is superseded by the local Batch 1 progress note. No unit-test files were added per user request. Follow colab_baseline_smoke.md, request/validate the seven-file bundle, preserve raw artifacts under result/ and update findings/open questions. Do not reimplement Batch 1 or start Batch 2 before this review.
