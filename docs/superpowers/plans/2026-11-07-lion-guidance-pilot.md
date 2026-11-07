# LION guidance pilot implementation plan

> Execute sequentially in this session with executing-plans; user requested new Colab and testing all hypotheses after reviewing the candidate mechanisms.

**Goal:** Test A cheap latent Chamfer, B feature-assisted matching, C causal-PM feature repair without promoting any default.
**Architecture:** Reuse frozen original LION/Point-MAE, paired preparation, source/frame helpers, canonical setup and immutable artifacts. Keep experimental kernels/sampler and bounded pilot entrypoint separate from production sampler.
**Tech stack:** Existing Python3.8/Torch2.1.2/CUDA12.1 Colab; local existing CPU test venv; no new dependencies.
**Spec:** knowledge/guidance_lion_candidates_20261107.md

## Global constraints
- User authorization: “Tamam yeni bir Colab oturumu başlatıp bu hipotezleri tek tek test edelim.”
- Whole initial three-hypothesis GPU screening phase estimated30–50min, hard cap55min including technical/repeat/capture/calibration/timing/model checks. No automatic256-ID/full/three-seed expansion. Setup separate. If estimates exceed60min before completion, prepare exact remaining phase and obtain duration authorization.
- N8/Background35, gamma.03/eta.003/p.95, equal clean-quarter+terminal PM, original decoder style, EMA off, frozen eval models. Pilot B2 rather than deploymentB32; no B32 speed claim.
- ModelNet40-C s5 seed0;16 explicit development IDs,4 corruptions gaussian/background/density/occlusion (actual dataset keys validated); separate4 calibration IDs. Previously exposed status declared; no labels/clean data in adaptation/calibration.
- Source/data/checkpoint/native hashes and bytes, RNG state, intermediate corrections, logits and timing retained. Guard controls and empty/failed arms are not successes.
- No peer-runtime interaction, no teardown without explicit task-complete+close permission. Preserve original dirty checkout.

## Review focus
- Reverse-Chamfer incoming edges and duplicate bidirectional edges must contribute to proximal numerator/denominator.
- Feature scaling/shuffling must not mutate reference or leak labels; beta0 reduces to XYZ nearest matching.
- Teacher uses strictly previous predictions; first checkpoint cannot adapt. Fixed q/z does not imply fixed decoded XYZ.
- Same epsilon DDIM bridge must use actual previous alpha, including final alpha1; full backward control must match existing sampler within measured native floor.
- No-grad/prior and decoder backward differences, missing native gradient support and cost of extra forwards must remain visible.

## Tasks
1. Setup: canonical10cell extraction/hashes; cell1 restart/reconnect; cell2 known fastrlock exception only; user Drive consent; native markers/data/weights verifier. Record session ownership and logs.
2. Add `lion_guidance_pilot.py` and `tests/test_lion_guidance_pilot.py`: test proximal against explicit quadratic solve/descent, feature disambiguation/shuffle/beta0, causal teacher and feature-only bounded update. Implement minimum pure kernels; CPU tests before GPU.
3. Add isolated experimental sampler and `scripts/run_lion_guidance_pilot.py`: reuse actual scheduler/preparation/frames; full-control, style-frozen, sparse-zero, cheap-only, hybrid; B geometry/feature/shuffled/4D controls; C causal/fixed teacher bounded feature readout repair, all-channel diagnostic. No default production change.
4. Prepare source-bound operator/deployment and frozen selection. Native technical phase first, paired repeats; calibrate cheap step using unlabeled actual update norms. Stop affected hypothesis if native gradients or same-path controls fail; retain failure evidence.
5. Run sequential A/B/C pilots and matched total-time checks within whole55min budget; archive each phase to fresh Drive path and local SHA-verified immutable bundles. Analyze per-object paired outcomes, per-corruption harm and native floor. Small cohorts only reject hypotheses; never establish benchmark superiority.
6. Review affected source and CPU checks, validate actual artifacts, update knowledge and publish only accepted scoped code/evidence to thesis-integration. Keep own runtime live unless user separately authorizes closure.

## Progress
- [x] Current shared79445c0 and newly integrated prior physical-lift negative study reconciled.
- [x] Own A10080GB HighRAM allocated; cell1 and same-VM reconnect passed.
- [x] Cell2–10 setup, local implementation, native gates, narrowed A/B/C science and acceptance completed; scientific gate failures are retained.

## Execution amendment
User now explicitly authorizes all necessary GPU tests without repeat permission and requests continuous sequential work until results, with knowledge updates at intermediate steps. The55min initial pilot remains an estimate/operational checkpoint, not an approval cap. Extend to necessary validation only after mechanism/native gates; stop failed hypotheses with explicit evidence, not silent omission. No teardown authorization.

## Final execution decision
Technical sampler5/8pass and semantic8/8pass led to A/B Density+Occlusion only and C all4 conditional frozen readouts, recorded before pilot outcomes. A/B pilot repeats14/16 and11/16; C32/32. AllA/B arms22/32 andallC arms48/64. No case exclusion/threshold relaxation. No candidate qualifies for advancement; own-repeat/warmed deployment/full confirmation NOT_ADVANCED. Independent review and815CPUtests/2skip support the limited result. See knowledge/lion_guidance_pilot_results_20261107.md.
