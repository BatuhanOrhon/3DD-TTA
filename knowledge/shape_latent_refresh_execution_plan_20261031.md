# Shape latent refresh pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.

**Goal:** Run the user-approved paired baseline/mean_refresh/posterior_refresh small pilot.
**Architecture:** New opt-in module handles reference preparation and posterior
refresh proof; new runner composes the existing original N8/B35 paired host.
No tta/model/default edits. Baseline produces common proposals once; candidates replay.
**Tech Stack:** existing PyTorch/NumPy/diffusers, CPU pytest, Colab CLI0.7.4.
**Spec:** [approved design](shape_latent_refresh_pilot_20261031.md), expanded from approved chat.

## Global Constraints

Same64development IDs/all15/seed0/Point-MAE/B32/N8/B35/gamma.03/eta.003/p.95.
Same eps, h0, local noise, anchor and postprepare RNG; frozen raw/eval models.
Global diffusion0. Single common X* in LION frame. No clean data in proposals.
No existing branch/reset; preserve dirty state. Research cap3300s; exact setup10cells.

## Review Focus

- Sigma/epsilon alias: explicit Gaussian draws, actual posterior equations.
- Extra encoder/AE probes: restore RNG/buffers and preserve baseline readout.
- Candidate state: retain original h/noise/anchor, own z at all decoder probes.
- Replay: source/input/common-state binding, finite/dtype/shape checks, orphan rejection.
- Timing: first pass counted in deployment; raw replay timing labelled separately.

### Task1: Paired states and posterior refresh

Files: create `shape_latent_refresh.py`, `tests/test_shape_latent_refresh.py`.
- [x] Write failing native-encode draw/RNG parity, changed-sigma/same-eps formula,
  same-sigma equality, invalid inputs, physical proof/replay/no-overwrite tests.
- [x] Run RED; implement minimal `prepare_reference`, `refresh_proof`, proof persistence.
- [x] Run GREEN. No clean tensors accepted by proposal API.

### Task2: Fixed runner and acceptance/operator

Files: create `scripts/eval_shape_refresh.py`, `tests/test_eval_shape_refresh.py`;
reuse existing sealed asset/setup/operator workflows under new ignored root.
- [x] Write failing three-arm config/smoke/cache roots and paired reconstruction tests.
- [x] Implement original-host wrapper: baseline records common states/one X*;
  candidates load same proposal, run local chain with selected z; preserve observers.
- [x] CPU targeted + impacted suite; seal exact source/manifest/operator.
- [x] One fresh reviewer per executing-plans, resolve substantive findings with regression.

### Task3: Setup, bounded smoke/pilot, archive and interpret

- [x] Canonical10setup cells, actual CUDA/data/checkpoint/native markers.
- [x] Strict frozen CPU deployment, then32-ID GB technical gate.
- [x] Whole-phase estimate below60min; same cap/ledger for64-ID/all15 three arms.
- [x] Validate physical states/proposals/RNG/source/readout and independent local recount.
- [ ] Update findings/method/open/README and preserve failures complete; selective publication pending.
- [x] Report actual results/runtime/session and limitations; keep runtime absent close consent.

Ruling: user explicitly authorized execution of the three-arm design; native
execution continues without another design/permission loop. User worktree
commits are not needed; selective publication uses isolated latest-target checkout.

[Run] Task1/2 and GPU/analysis/archive gates complete; accepted result is linked in shape_latent_refresh_results_20261031.md. Freshreview fixes verified; runtime preserved. Publication receipt supersedes the final pending checkbox after normal push.
