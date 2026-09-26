# GSD Smooth Spectrum Implementation Plan

> **For agentic workers:** Implement the tasks in order. Preserve current GSD v1 and follow test-first development where automated tests are explicitly requested.

**Goal:** Add an opt-in v2 GSD latent smooth spectral fidelity method and a reproducible test-plan runner without changing v1 or the original baseline.

**Architecture:** Keep graph construction shared and fixed to v1 conventions. Add a separate full-spectrum heat-kernel operator target with fixed `3*N` reduction, select it only for a new method ID, and keep the existing v1 route/defaults stable. Add an experiment command builder for unchanged v1, new-host hard-M, and new-host smooth arms; parameter calibration remains explicit and external to the final test set.

**Tech Stack:** Python, PyTorch, argparse, existing `unittest` and `research_artifacts` infrastructure; Colab for model evaluations.

**Spec:** `knowledge/gsd_guidance_math_20260926.md`, especially section 9; `knowledge/open_questions.md`, `NEXT GSD TEST`.

## Global Constraints

- Existing `gsd_latent_spectral_v1` and weight-zero dispatch remain behaviorally unchanged.
- V2 graph construction inherits v1: static reference latent XYZ, nonself kNN RBF, directed degree threshold, max symmetrization, both-endpoint mask, active induced combinatorial Laplacian.
- V2 scales L by mean active degree and uses `exp(-beta*lambda_scaled)` over all active modes for smooth profile.
- V2 smooth and hard-M profiles both reduce per sample by `3*N`; sum across samples.
- No parameter is selected from the final all-15 benchmark; pilot is severity-5 Gaussian/Impulse, batch 32, seeds 0/1/2.
- Local work is CPU-only; model/runtime evaluation is Colab-only.

## Review Focus

- Isolated/empty active graphs must return a finite connected zero loss and zero direct gradient.
- Smooth weights must remain PSD and match the half-filter quadratic form and analytic gradient.
- Large beta/near-zero eigenvalues must not create nonfinite weights; invalid beta must be rejected.
- V1 arguments/config serialization and exact baseline dispatch must remain unchanged.
- V2 smooth and hard arms must use identical graph, host, scheduler, rates, decoder, preprocessing and fixed-point denominator.

---

### Task 1: Full-spectrum spectral targets

**Files:**
- Modify: `graph_spectral.py`
- Verification plan: `knowledge/gsd_smooth_spectrum_test_plan_20260926.md`

**Interface:** Add a v2 target builder that accepts existing `SpectralConfig`, profile `hard|smooth`, and a finite positive beta for smooth profile. It returns a detached reference plus a per-sample dense filter operator and diagnostics. Its loss is `sum_b <E,F_b E>/(3*N)`.

- [x] Define expected literal results and edge cases for two-node hard projection and beta-weighted two-node smooth loss in the test plan.
- [x] Implement graph-kernel target construction while retaining v1 `build_spectral_target` behavior.
- [x] Record `trace(F)`, active vertices, beta, scaled eigenvalue bounds, operator storage, and construction timing.
- [x] Execute formulas analytically and document the command for CPU-only verification.

### Task 2: V2 trajectory and protocol

**Files:**
- Modify: `tta_gsd.py`
- Modify: `gsd_protocol.py`
- Modify: `run_baseline.py`
- Verification plan: `knowledge/gsd_smooth_spectrum_test_plan_20260926.md`

**Interface:** New method ID `gsd_latent_spectral_smooth_v2`; new `--gsd-profile {hard,smooth}` and `--gsd-beta` options. V1 does not accept or serialize v2-only arguments. V2 requires explicit profile and, for smooth profile, explicit positive beta. The method supports smoke/pilot only until promotion evidence exists.

- [x] Add CLI/config contract tests to the documented plan, including rejection of v2 flags on other methods and unchanged v1 configs.
- [x] Dispatch v2 through the shared SCD+denoiser host, selecting fixed-`3*N` hard or smooth target; leave default/v1 code path unchanged.
- [x] Include beta/profile, normalization, effective spectral mass, and target cost diagnostics in artifact metadata.
- [x] Keep all-15 benchmark stage locked for v2 pending pilot review.

### Task 3: Reproducible pilot command matrix and knowledge

**Files:**
- Create: `eval_gsd_smooth.py`
- Create: `knowledge/gsd_smooth_spectrum_test_plan_20260926.md`
- Modify: `knowledge/gsd_guidance_math_20260926.md`
- Modify: `knowledge/open_questions.md`
- Modify: `knowledge/method_synthesis.md`
- Modify: `knowledge/README.md`
- Modify: `knowledge/findings_log.md`

**Interface:** Launcher builds seed-matched `v1`, v2 `hard`, and v2 `smooth` commands for the same seeds and corruption set. V1 retains alpha=1. New-host hard and smooth spectral coefficients must be supplied explicitly after calibration. Beta and alpha are required arguments for the smooth arm; no defaults imply an optimum. Common random draws still need runtime verification.

- [x] Create unit/protocol and CPU mathematical verification plans with pass criteria.
- [x] Create a pilot matrix for 3 arms x seeds 0/1/2, each evaluating Gaussian/Impulse (9 run bundles, 18 per-corruption/seed cells), raw LION eval, EMA off, batch 32, severity 5, lambda `.95`.
- [x] Require exact result bundles and paired deltas plus gradient/memory/runtime diagnostics before reading accuracy.
- [x] State that the pilot is exploratory and all-15 is a later, separately approved confirmation.

## Out of Scope

- Graph-rule, reference-domain, SCD, scheduler, lambda, guidance-rate, decoder, classifier, preprocessing or FPS changes.
- PxP, learned GSDPS point shifts, dynamic graphs, automatic beta/alpha tuning, and local model/GPU runs.
