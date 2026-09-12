# Findings Log

Append entries chronologically. Never delete negative or superseded results. Use exact run paths for **[Run]** claims.

## 2026-09-12 — Initial repository and literature audit

**Evidence:** paper PDFs, current branch source, full Git history; no archived Colab artifacts.  
**Git branch/commit:** `pxp-gradient-projection` / `53ba252` at audit start.  
**Status:** mixed **[Paper]**, **[Code]**, **[User report]**, and **[Inference]** evidence.

### Confirmed from sources

- **[Paper]** Published 3DD-TTA ModelNet40-C mean is 65.7%; repository README says 66.1%.
- **[Code]** The original baseline path remains separated from GSD/PxP scripts.
- **[Code]** The main GSD path is a latent spectral regularizer, not full GSDTTA.
- **[Code]** Dynamic mode recomputes an eigenvector matrix/basis, not a single `U_0` vector.
- **[Code]** PxP variants project spectral and SCD gradients and therefore are PixelAsParam-inspired rather than direct implementations.
- **[Code]** Runs are stochastic and current output schemas do not capture enough provenance for definitive comparison.
- **[Code]** Loss reductions and scheduler/style choices have changed across commits, creating confounds.

### Provisional observations

- **[User report]** Original baseline is approximately 63% locally.
- **[User report]** A GSD variant reaches approximately 63.5%.
- **[Inference]** The first investigation should prioritize environment/checkpoint/data identity, seed variance, scheduler serialization, and metric aggregation before further method search.

### What would update this conclusion

A complete archived baseline run with per-corruption results, exact command/config, package/GPU environment, data/checkpoint hashes, and repeated seeds.

## 2026-09-12 — Follow-up code audit and clean restart decision

**Evidence:** [Code], [Paper], [User report], [Inference]; no new [Run].  
**Fork commit:** `53ba252519c7cf65f836a9c1c564027142ab1573`  
**Original LION commit:** `7711b3d185752eeb632d095494876e4de15f3195`  
**Run paths:** none; static audit only, not an accuracy experiment.

### Findings

- [Code] Original LION trainer inference disables VAE/prior dropout. Its demo wrapper does not; the fork wrapper is byte-identical. Current baseline/GSD setup sets only Point-MAE to eval. This verified discrepancy is inherited, not caused by GSD additions; accuracy impact remains [Inference].
- [Code] Scheduler, reverse steps, final style, batch and unequal gamma/eta semantics confound current baseline/GSD comparisons.
- [Code] Mean spectral MSE plus summed SCD changes relative guidance with actual batch/band/channel counts.
- [User report] Mean settings previously gave smaller reported spectral losses and higher accuracy than sum. Exact runs/configurations are not archived; a smaller mean number alone is not spectral-fidelity evidence.
- [Paper/Code/Open] Threshold normalization differs from GSDTTA Eq. (10). Keep visual/symbol/adjacency verification before correction; self-neighbors, distance units and isolated-node handling need small tests.
- [Code] Conditional missing-key failures, unsafe resume aggregation, symmetric CSV header/row mismatch, stale analyzer shapes and batch-flattened projection require targeted tests rather than wholesale legacy migration.
- [Code] SCD uses the first three local-latent channels; it is not computed after decoding. Earlier method/map wording is superseded and corrected.
- [Inference] Shared eigenvector sign flips/within-band rotations do not change complete-band same-basis MSE; changing projectors and band boundaries are the dynamic diagnostics.

Full source paths, wrapper hash, caveats and falsifiers: [code audit](code_audit_20260912.md).

### Decision and verification gates

The user endorsed an incremental clean restart from main. Local main/origin/main/upstream/main equal `107305fd7baf40b359f31c07d235599198be7324`. Proposed `baseline-repro-clean` is not created yet. Preserve legacy branches/user work; add artifact and source-only controls, then isolated dropout A/B, then lock the baseline ladder and require spectral-off parity. Sum plus smaller eta/gamma and dynamic/PxP tuning are deferred. [Small batches](clean_restart_batches.md) define scopes, tests, knowledge updates and Colab review gates.

This update changes documentation only. No branch switch or Python modification; no gain claimed. Gaussian/background seeds 0,1,2 under fixed configuration/common draws are next accuracy evidence. Repeated null/negative dropout effects weaken the gap hypothesis. A source-only mismatch redirects diagnosis to data/classifier; a failed spectral-off parity check exposes accidental method differences.

## 2026-09-12 — Batch 0 branch reference prepared, placement pending

**Evidence:** [Code] Git/ref/status checks; no numerical run.  
**Legacy checkout:** `pxp-gradient-projection` / `53ba252519c7cf65f836a9c1c564027142ab1573`  
**New branch/main:** `baseline-repro-clean` / `107305fd7baf40b359f31c07d235599198be7324`  
**Run paths:** none.

The user approved proceeding. Created the branch reference with `git branch baseline-repro-clean main`, without switching checkout, staging, commit or push. Ref equality and `git diff --exit-code` verified; pre-existing untracked knowledge/skill/protocol/PDF/notes remain. Worktree inspection found only the normal root checkout. Legacy tracked scripts and source PDFs would disappear from the folder on an in-place switch, though still retained on the old branch; asked the user for placement preference before this transition.

Batch 0 remains partial until safe placement, curated memory preservation and review. No Python changes, dependency install or local accuracy test. Keep Batch 1 on hold until the clean branch is actually checked out in its declared workspace. See `clean_restart_batches.md` for the next handoff. This supersedes the earlier status that the new branch did not exist; it does not supersede the audit's inference findings.

## 2026-09-12 — Batch 0 same-folder checkout and documentation preservation

**Evidence:** [Code] Git state, SHA-256 and parity checks; no [Run].  
**Active baseline source:** `baseline-repro-clean`, main `107305fd7baf40b359f31c07d235599198be7324`  
**Legacy experiments:** `pxp-gradient-projection` / `53ba252519c7cf65f836a9c1c564027142ab1573`  
**Run paths:** none.

User selected same-folder development and approved carrying skill/knowledge. Switched to the prepared clean branch. No tracked or staged code edits existed, so no new stash; existing dev stash `02ddaf533c93164d69643e43c55ad37df6fa0343` remains. Restored four absent scholarly PDFs from legacy Git history to local untracked files; did not overwrite the user's PixelAsParam PDF. All 23 selected memory/skill/protocol/PDF hashes matched across checkout before documentation updates. Tracked Python/requirements/environment match main; old variants are not imported.

Curated memory, researcher skill and result protocol are selected for a local documentation-only commit, with all PDFs and unrelated docs/tmp excluded. The containing documentation commit is recorded in the handoff rather than embedded as a self-referential hash. No push, dependency installation or numerical test. The prior placement-pending status is superseded. Next is Batch 1 after handoff review, not dropout modification yet.

## 2026-09-12 - Batch 1 baseline smoke artifact implementation

**Evidence:** [Code], not [Run].

**Base Git commit:** b31fd23193bbcb9a5c189cfb4118be41506f9333 on baseline-repro-clean; implementation commit reported in the handoff.

**Run paths:** none supplied yet. Expected result/modelnet40_c/3dd_original/<UTC-timestamp>_baseline-smoke_seed0/.

**Question:** can we produce an internally consistent, diagnosable original-TTA smoke bundle without changing adaptation math/modes?

Added run_baseline.py and research_artifacts.py, with optional read-only checkpoint/scheduler/batch observers in three baseline modules. The runner reuses the original preprocessing/TTA/classification path. Seeds and actual runtime flags/configs, checkpoint/data/source hashes, load incompatibilities, dropout/module modes, installed extension identities and count-based fraction CSVs are logged; subprocess capture retains Python/native stderr. Each invocation creates a fresh seven-file directory and a sibling ZIP, with partial/failed status rather than fabricated full accuracy. No resume, GSD, LION eval, scheduler/rate/style/reduction or dependency changes.

Local syntax/CLI and temporary artifact/count/macro-micro/duplicate/collision/failure-state checks passed. Observer-stripped ASTs of all three modified baseline files match main. The first AST comparison failed because its checker omitted a nested batch-observer branch; the corrected recursive checker passed. This was a verification-script limitation, not an adaptation-code change. No unit-test files were created per user request.

**Protocol:** pending Level 0 Colab smoke, Gaussian severity5, first two batches, seed0, explicit batch32 and repository lambda0.95; gamma/eta0.01, normal/background5/35, unchanged legacy modes. Runner default batch remains40. Same seed is not a guarantee of common-draw pairing.

**Decision:** request the complete smoke ZIP using colab_baseline_smoke.md; validate before Batch 2. No numerical result, dropout effect or accuracy-gap cause is inferred. A Colab runtime/import/schema/count failure or unexplained GPU behavior rejects the end-to-end handoff until resolved.

## 2026-09-12 - Chamfer import restoration for Batch 1 smoke

**Evidence:** [Code] plus [User report] Colab traceback; no successful run yet.

The Batch 1 smoke failed while constructing Point-MAE, before LION/TTA or classification. `models_mate/Point_MAE.py` referenced `ChamferDistanceL2` for `cdl2` but its import was commented out. The same import is active on the previously working `pxp-gradient-projection` branch. Git history identifies commit `8183863` as restoring it for Colab.

Decision: restore only the missing import. This is a baseline construction repair, not a TTA, dropout, scheduler or GSD change. Re-run the same Level 0 Gaussian two-batch command with seed 0. The new ZIP falsifies this diagnosis if it still reaches the same undefined-name error.

## Entry template

```markdown
## YYYY-MM-DD — Short finding title

**Evidence:** [Run]/[Paper]/[Code]/[User report]/[Inference]  
**Git commit:** full hash  
**Run paths:** `result/...`  
**Hypothesis:** falsifiable statement  
**Protocol:** level, dataset, corruptions, seeds, important config  

### Result

Exact values, uncertainty, failures, runtime/memory, and per-corruption pattern.

### Interpretation

What the evidence supports and what it does not support.

### Decision

Continue, modify, reject, reproduce, or escalate to full evaluation.

### Falsifier / next evidence

What result would overturn or materially revise the interpretation.
```
