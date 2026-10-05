# SDD ledger - plan: knowledge/gsd_block_routing_implementation_plan_20261004.md

## Setup
- Current base: `gsd-smooth-spectrum@aa8b725ba4daaf3448f0cda75bfc6aaec6c86d48`.
- Isolated `git worktree` and local clone attempts failed because the managed Windows shell denied Git metadata/process access. Continue only with scoped edits in the existing workspace; do not alter other dirty files.
- Baseline before Batch 1: `python -m unittest discover -s tests -p "test_gsd_trajectory.py"` ? 12 tests passed.
- User-requested hypothesis added to `knowledge/open_questions.md`: v1 spectral gradient scale may limit accuracy gain; separate coefficient pilot planned.

## Batch status
- Batch 1 (pure routing algebra): complete; implementer `/root/gsd_compose_core`; reviewer `/root/review_compose_core` approved spec and quality. 12 focused CPU tests passed after expected RED. No code committed due shared `.git` write denial.
- Batch 2 (paired sampler and hard-v1 diagnostics): complete; implemented by `/root/gsd_paired_sampler`, reviewed by `/root/review_compose_core` and approved after fixes. Added detached prepared inputs with hashes, opt-in local/style routing, and scalar hard-v1 diagnostics. Observer includes graph config/per-sample actual rank and PCGrad pre/post/cap metrics. Verification after fixes: composition algebra 12 passed; new trajectory 8 passed; existing trajectory 12 passed; `git diff --check` clean apart from pre-existing line-ending warnings.
- Batch 3 (runner/artifacts/analyzer): implementation and independent review complete. It includes phase runner, strict phase/selection/reference validation, exact ten-file arm bundles, partial-run preservation, analysis/paired transitions, timing and runtime/artifact identities. Final verification: composition/protocol/runner suite 48 passed; analyzer 5 passed; trajectory suite 20 passed; `py_compile` passed; `git diff --check` exit 0 with pre-existing line-ending warnings.
- Colab scale/routing experiments: not launched. The approved code remains uncommitted; an accessible code ref is required before Colab smoke.

## Decisions
- No code has been committed or staged. Existing user changes remain outside task scope.

## 2026-10-05 Git authorization update
- The earlier managed-shell Git metadata/write denial was an environment limitation at the time, not a standing project rule. The user has now explicitly authorized Git writes for relevant project code and knowledge documentation, including commit/push when appropriate.
- Preserve unrelated user changes. Exclude temporary files, redundant scripts, and generated `result/` outputs from commits and pushes.
