# LION second-pass N5/N8 implementation plan

> Execute inline with superpowers:executing-plans; user explicitly authorized the proposed N5/N8 cached comparison.

**Goal:** Compare second local depths5/8 to archived N3 with exactly the same first output, z1/h1 and Gaussian.
**Architecture:** New opt-in replay entrypoint reuses shared worker/artifacts/preprocessing, original tta and SharedTrajectoryObserver. Original N3 runner and baseline stay unchanged.
**Tech Stack:** Existing Python3.8/PyTorch/NumPy and assigned Colab CLI A100.
**Spec:** knowledge/lion_reencode_depth_20261101.md

## Global Constraints

64 exposed ModelNet objects/all15/s5/seed0/B32/Point-MAE, gamma.03/eta.003/p.95; no entropy, global diffusion, extra seeds or tuning beyond requested5/8. Preserve same VM and raw archives. Whole scientific phase estimate10–20min,55min cap. No teardown authorization.

## Review Focus

Changed cache/source/native/model identity; stale label/ID association; normalized bridge or renewed encoder draws; wrong depth alpha or PM checkpoints; copied first-pass metrics misrepresented as newly executed.

### Task1 — physical replay and scoped depth conditions

Files: scripts/eval_lion_reencode_depth.py, tests/test_lion_reencode_depth.py, knowledge/evidence/lion_reencode_depth_20261101/analyze_phase.py.
- [x] RED contracts for explicit5/8 config/readout, tampered physical replay and exact preserved posterior/noise/RNG; GREEN minimal implementation.
- [x] Test existing impacted paths and full tests/; independent fresh review before GPU.
- [ ] Seal source/operator, bounded read-only same-VM audit and deployment; N3 technical replay parity plus N5/N8 technical gates.
- [ ] Run requested N5/N8 pilots from immutable64-ID source; archive each condition to new Drive paths, independent local SHA/member/readout/state recount.
- [ ] Report paired object-cluster intervals, per-corruption changes and limits; update knowledge and selectively publish necessary reviewed source/evidence.

Ledger: base6ff2e49; user approved N5/N8 in current message. Ruling: reuse existing isolated worktree and execute inline; approval of the already-proposed experiment supersedes another design/plan permission cycle. Saved first trajectories are references, not rerun baselines. Replay restores the recorded post-prepare RNG before each second chain. N3 replay is a32-ID technical check, not an extra selected accuracy candidate. Temporary operator bytes and raw artifacts remain outside Git.

Review ledger:7new contracts, tests/647passed+412subtests/1historical skip. Reviewer found actual-noisy hash and shared-source/resolved-config binding gaps; both fixed RED→GREEN before GPU. First source/CPU tests unchanged after this full run.
