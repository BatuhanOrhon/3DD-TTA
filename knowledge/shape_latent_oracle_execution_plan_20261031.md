# Clean-z Oracle Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Execute only the user-approved small four-arm clean-z diagnosis.
**Architecture:** Reuse depth/shared runner and immutable paired-state serialization. Add oracle preparation/entrypoint and one optional fixed Chamfer reference; default host unchanged.
**Tech Stack:** Python3.8/Torch2.1.2/CUDA12.1 on Colab; local CPU tests and independent NumPy1 recount.
**Spec:** knowledge/shape_latent_oracle_pilot_20261031.md, approved chat four-arm table and 31Oct user instruction to save knowledge before execution.

## Global Constraints

- Current worktree/branch and previous edits preserved; global diffusion OFF.
- Fixed64 exposed IDs/all15/severity5/seed0/Point-MAE/N8/B35;32GB technical gate.
- Shared posterior/diffusion noise, post-prepare RNG, A-h XYZ Chamfer anchor.
- Record clean oracle leakage; D is not a pure compatibility control or upper bound.
- No full data/grid/extra seed/dataset; inspect results before continuation.
- No VM teardown without explicit task-complete+close permission.

## Review Focus

- Conditional h input/order and exact clean/corrupt original-ID mapping.
- Existing Normal.sample returns aliased rho; store explicit actual Gaussian draws.
- Replacing h must not replace the SCD target; default source parity retained.
- Independent normalization, actual ModelNet output frame, D interpretation.
- Recoverable source/physical posterior/cache/RNG evidence and failed-phase preservation.

### Task 1: Fixed oracle preparation and runner

**Files:** shape_latent_oracle.py, scripts/eval_shape_oracle.py, tta.py; tests/test_shape_latent_oracle.py and tests/test_eval_shape_oracle.py.
**Interfaces:** prepare_states(corrupted,clean,vae) -> states/proof; paired_states(path,corrupted,clean,vae) -> states/record; fixed CLI and optional tta_reconstruct(chamfer_reference=None).
- [x] Tests fail before feature exists; actual standard-normal regression and common-anchor test fail before fixes.
- [x] Implement minimal paired states/worker; no unrelated production changes.
- [x] Final CPU suite: python -m pytest tests -q. Expected all executed tests pass, historical sealed-fixture skip disclosed.
- [x] Fresh independent final review; no blocking mathematical bug, explicit frame ruling and two evidence/metadata findings recorded.
- [x] One test-backed fix pass, seal exact scientific files, record review acceptance.

### Task 2: Setup, immutable deployment and small research phase

**Files:** result/preparation/shape_latent_oracle_pilot_20261031/{setup,operator}/ (ignored temporary drivers).
**Interfaces:** Task1 exact source/CLI -> reviewed source/asset/command/ID seal; controller -> raw/paired/receipt archives.
- [x] Verify own existing runtime; if lost record failure and run canonical10cells in unique new A100.
- [x] Verify real Drive/native/data/checkpoints/CPU load/source; actual receipt required.
- [x] Technical gate then four-arm pilot; whole research cap3300s, expected15–25min.
- [x] Preserve statuses/errors, actual sources, logs, posterior/cache identities and artifacts.

### Task 3: Independent acceptance and result handoff

**Files:** knowledge/shape_latent_oracle_results_20261031.md; compact evidence; findings/open_questions/method/README.
**Interfaces:** Task2 immutable raw and physical caches -> NumPy1 PM/posterior/RNG/source recount.
- [x] Verify safe ZIP extraction/CRC/SHA and actual Drive/local copies; recount PM/all15/macro14/B and paired object intervals.
- [x] Record negative/null/inconclusive outcomes, timing/preprocessing/oracle limits; no accuracy-driven continuation.
- [ ] Integrate necessary completed code/knowledge normally to latest origin/thesis-integration; preserve original branch and raw files.

## Execution rulings

- User explicitly authorized execution of the already reviewed four-arm design; no duplicate design/executor permission request.
- Source branch contains preserved dirty prior work; selective publication supplies the commit boundary, no whole-branch merge or intermediate commit of others' edits.
- Physical prepared input and correct method metadata are required scientific audit evidence; review's two Minor labels are regraded Important for this artifact contract, fixed together with RED→GREEN tests.
- Keep independent input normalization. D−C includes clean normalization/interpolation and remains under corrupted A-h anchor; weak D is not a universal clean-pair failure.
