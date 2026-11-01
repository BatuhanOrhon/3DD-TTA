# SCD-h spectral-z Implementation Plan

> Executor: inline under superpowers:executing-plans; independent whole-branch code review before scientific GPU.

**Goal:** Implement the registered N8/B35 SCD-h/spectral-z finite study without changing baseline behavior.
**Architecture:** Opt-in router in original sampler, task-specific CLI on existing depth/shared worker, finite registry/controller and independent NumPy analysis.
**Tech Stack:** Existing Python3.8/PyTorch2.1.2 CUDA stack; standard library/local CPU fixtures.
**Spec:** knowledge/scdh_spectralz_search_plan_20261101.md and scdh_spectralz_agent_handoff_20261101.md.

## Global Constraints

ModelNet40-C/s5/Point-MAE/B32/N8-B35/raw-eval/noEMA/original-z decode and probes; no global prior. Offdiag static raw-eigenvalue exp(-2lambda), cutoff M, sum/(3N).20A candidates/2controls,8new B maximum; disjoint128ID/three seeds conditional confirmation,384timing events. No scientific GPU before explicit >60min whole-phase approval.

## Review Focus

- Native derivatives/zero spectral gradients must fail the declared gates without epsilon inflation.
- Inactive/zero-kappa routing must preserve baseline style-off state/RNG/readouts.
- Source/ID/asset/cache tampering and missing partial files must stop acceptance.
- Reused .03 trials and duplicate C comparators must never execute twice.
- Whole-phase deadline and ordinary-failure archival must survive separate stage calls.

### Task 1: Routed sampler and worker entrypoint

Files: scdh_spectralz.py, tta.py (optional hook), scripts/eval_scdh_spectralz.py, tests/test_scdh_spectralz.py.
- [ ] Write failing CPU equations/route-isolation/style-off/default-parity/finite-difference/cache and original-z PM tests.
- [ ] Run targeted test; observe missing behavior.
- [ ] Implement fixed offdiag/profile router and strict semantic CLI/worker guard using the existing original depth host.
- [ ] Run new and existing sampler/depth/trajectory regressions; expected all pass.
- [ ] Commit scoped batch with execution ledger.

### Task 2: Registered search, calibration and independent analysis

Files: scripts/scdh_spectralz_study.py, scripts/run_scdh_spectralz.py, scripts/analyze_scdh_spectralz.py and tests/test_scdh_spectralz_study.py.
- [ ] Test label-free ID ranking/exposure/disjointness; pooled medians/unavailable M; deterministic A/B/C bounds/ties/reuse;10k object-cluster/PM/CRC/hash/partial rejection; cumulative approval deadline/failure archival.
- [ ] Implement minimal registered configuration/state machine and strict independent NumPy analysis on standard worker artifacts.
- [ ] Run explicit CPU fixture suites; expected all pass.
- [ ] Register source/IDs/grid and commit.

### Task 3: Concrete preparation and whole-phase review

Files: task-owned result/preparation/scdh_spectralz_search_20261101 and compact knowledge evidence/notes.
- [ ] Prepare new named A100 using canonical10cells; verify each marker/native/data/checkpoint gate; preserve failure evidence.
- [ ] Reuse pinned bulk identities, create fresh source+metadata overlay; strict CPU/no-forward deployment receipt.
- [ ] Independent whole-branch review; fix important findings via RED/GREEN.
- [ ] Present actual sealed source/assets/IDs/executable phases and updated whole-phase budget for one scientific-GPU approval.
- [ ] After approval execute spec Batches2–4 and archive/recount; until approval keep research pending.

## Rulings

The user already supplied and explicitly invoked the complete design; implementation does not need another design/plan permission gate. Scientific >60min approval is still pending. Primary offdiag fixes the known isolate defect, while cutoff tie/scale diagnostics remain recorded limitations; no unregistered beta/graph sweep.
