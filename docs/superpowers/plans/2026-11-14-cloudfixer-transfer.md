# CloudFixer-only transfer implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Run the approved XYZ proximal-SCD/PM search using copied official CloudFixer models, with verified per-batch Drive recovery.
**Architecture:** Archive the pinned official checkout without rewriting models/data/classifiers. Add a small sampler, immutable Drive unit store and campaign CLI; reuse the existing proximal quadratic through a model-independent module.
**Tech stack:** Python3.8+ in Colab, PyTorch, NumPy, official CloudFixer, existing Google Drive mount, Colab CLI/monitor (user revision).
**Spec:** `docs/superpowers/specs/2026-11-14-cloudfixer-only-proximal-pm-design.md` (approved by user on2026-11-14; user additionally directs copy/reuse of official source).

## Global constraints

- Author source `f043e24414fcf4165fdc892abff21215c9989141`; only known PointNet++ import repair.
- No LION imports/models in the new inference worker; old baseline behavior preserved.
- N4/8/16/32 × K40/80/160 × tau0/.03/.1/.3, retained.95; native polynomial500 and deterministic terminal cleanup.
- B32;8smoke/128development/256confirmation object-disjoint IDs; all15/s5; no automatic full benchmark.
- DriveFS fsync/rename/SHA-readback batch commits (server sync unproven), immutable source/assets/config identity, no duplicated saved outcomes; cap24hours persists across recovery (user duration revision; previous10h superseded).
- User has approved implementation and Colab execution; no further design/method/duration approval requested. Google interactive consent remains user-operated; no runtime teardown authorization.

## Review focus

- Incomplete upload/commit acknowledgement: adopt valid committed batches once and never count conflicting data.
- Loaded checkpoint scheduler state: reject gamma inconsistent with declared polynomial500.
- Masked duplicates/variable cardinality: never produce invalid nearest matches or double reverse weighting.
- Restarted worker/new VM: restore the same unit identity and accumulated budget, reject changed assets.
- Tau0 and terminal timing: no Chamfer computation, N+1 actual calls, no false terminal-only latency claim.

### Task 1: model-independent correction and XYZ sampler

**Files:** `pointcloud_proximal.py`, `lion_guidance_pilot.py` (move/re-export only), `cloudfixer_diffusion.py`, `tests/test_cloudfixer_diffusion.py`.
**Interfaces:** `sample(model, x, mask, noise, *, steps, start, tau, retained=.95)` returns four clean XYZ candidates and actual integer indices; model is the official DiffusionModel, never LION.

- [x] Write failing hand-derived tests for fixed-noise transitions, clean terminal, N+1 calls, tau0 no-CD, mask-aware quadratic and invalid grids.
- [x] Run RED with CPU torch/NumPy/pytest in isolated temporary environment.
- [x] Move the existing quadratic unchanged; implement the sampler using official model coefficients/phi.
- [x] Run new tests and existing proximal/LION kernel tests; commit the verified batch.

### Task 2: immutable cloud units and campaign registration

**Files:** `cloudfixer_store.py`, `tests/test_cloudfixer_store.py`, `scripts/run_cloudfixer_transfer.py`, `tests/test_cloudfixer_transfer.py`.
**Interfaces:** store `put(key, bytes)`/`get(key)`/`keys(prefix)` verifies cloud bytes; `commit_unit`/`load_units` enforce immutable content. Registration fixes grid/cohorts; campaign units identify phase/config/corruption/seed/original IDs.

- [x] Write RED tests for cloud-boundary failure, orphan payload/receipt, duplicate acknowledgements, corrupt content, changed identity, cohort separation and independent batch RNG.
- [x] Implement cloud storage through existing Colab Google credentials and Drive API; use local disk only as the real transport fixture in tests.
- [x] Implement register/preview/run/resume/report around copied official model/dataset/classifier entrypoints and shared sampler.
- [x] Verify CPU end-to-end interrupted fixture resumes missing units, aggregates saved logits correctly and respects persisted budget; commit.

### Task 3: copied-source deployment and technical Colab smoke

**Files:** `scripts/prepare_cloudfixer_transfer.py`, `knowledge/cloudfixer_transfer_execution_20261114.md`, task-specific `result/preparation/cloudfixer_transfer_20261114/` evidence.
**Interfaces:** prepare emits exact official source tar, overlay/source hashes, registration, asset expectations and selected canonical setup phases. Runtime scripts use absolute paths and a distinct named owned session.

- [x] Test archive/source/patch identity and Python3.8 syntax; retain original source license.
- [x] Obtain one fresh whole-branch reviewer, fix material findings with regression tests; run relevant suite and record other full-suite limitations.
- [x] Record authorization/cost/cap/recovery; allocate distinct A100 High-RAM; own monitor; canonical10cells in order; user-operated account consent if needed.
- [x] Load author weights strictly; verify gamma, data/source identities and frozen state. Real smoke checks repeatability, variable point counts, tau0, PM, mounted Drive write/readback and missing-batch recovery. Archive failed/successful evidence.

### Task 4: bounded development search and research record

**Files:** existing campaign CLI, task-specific raw/derived results and knowledge updates.
**Interfaces:** accepted smoke is required before grid; report48-cell coverage and per-corruption counts from verified units only.

- [x] Project duration from actual smoke per-point-count timings, include API/archival overhead and remaining campaign budget; execute coarse grid within cap.
- [x] Persist each closed batch before advancing; preserve failures; recover only when prior worker state is known.
- [x] Retrieve and independently recount accepted outputs, rank completed grid only, preserve tau0 winner/edge flags and negative results. Do not launch refinement/confirmation automatically.
- [x] Update knowledge/status/source identities and report owned runtime state; preserve runtime absent separate completion-and-close authorization.
