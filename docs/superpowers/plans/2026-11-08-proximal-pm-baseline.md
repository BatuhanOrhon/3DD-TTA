# Promote the proximal N8+PM baseline

User instruction: promote the validated speed method as the operational baseline and register a future terminal-only comparison without PM. Implement inline; no new GPU run/allocation. Both Colab sessions remain closed.

## Fixed policy

- Profile `lion_proximal_pm_n8_v1`, method `3dd_proximal_pm_baseline`.
- Eight reverse updates for every corruption; equal probability mean at2/4/6/terminal, temperature1.
- Existing `lion_guidance_pilot.sample(..., arm='cheap', tau=.1)` kernel: predicted-clean latentXYZ, bidirectional95%-trimmed frozen-correspondence quadratic solve and DDIM clean-delta transport; no denoiser backward or direct style update. Original encoded decoder style; no spectral/feature/semantic term.
- Controlled backend, raw/eval frozen LION and source heads, EMAoff, DDIM100/finalalpha1.
- Explicit original IDs/assets/checkpoints retained. DefaultB128 for ModelNet40-C/Point-MAE alone; B32 for inherited multi-head/Scan execution. Other sizes are explicit and need hardware validation. Empirical evidence remains full2468/all15/seed0/Point-MAE/B128, not all heads/datasets.
- Preserve `chamfer_pm_n8_v1` and historical saved workers via explicit legacy profile; keep old depth/transfer/GSD entrypoints unchanged.

## Implementation

- [x] RED tests: default/legacy preview, frozen-profile validation before asset access, scope/batch/readout invariants, worker primaryPM+terminal diagnostics, direct cheap adapter state/RNG/decode semantics and scheduler observability.
- [x] Extend only the existing baseline entrypoint, depth geometry horizon selection/known profile, sampler optional scheduler observer and shared-worker optional backend callback. Reuse existing preparation/observer/artifact machinery; no new inference framework or dependency.
- [x] GREEN targeted and project tests; independent review and verification. Preserve all prior math/default paths and source snapshots.
- [x] Update current_baseline/README/protocol/findings/research status and add planned terminal-only experiment with explicit NOT_RUN status. Publish necessary code/tests/docs to shared branch normally; no new GPU benchmark claim.

## Future experiment (registration only)

Compare classic tunedN8 terminal SCD with cheapN8 terminal (tau=.1), both without the three intermediate decoder/classifier readouts, all corruptions atN8 and matched protocol/IDs/seeds/batch. Record terminal accuracy and actual end-to-end timing, corrected/harmed counts, per-corruption behavior and frozen-model/source guards. Saved terminal logits from PM acquisition provide accuracy context only, never terminal-only runtime. Start with a separately launched technical/pilot phase before estimating/launching full evaluation. No corruption-specific fallback or new tuning; no runtime allocation now.

## Verification and additional user instruction

875 CPU tests passed,2skipped,564subtests passed; full run50.36s. A cold-process source-freeze regression reproduced the lazy policy import blocker before repair. Native-free preview/legacy dispatch and both-dataset/four-head worker fixtures pass. Independent review and metadata distinguish reference rates, cache/audit/RNG overhead and float64 PM from the accepted float32 speed host. No GPU smoke/rebenchmark or runtime allocation.

The user also requests a future pre-softmax class-logit mean comparison. Registered at `knowledge/plans/logit_mean_readout_20261108.md`, refining the existing25October question with checked Li et al. arXivv3 B.4/D attribution. It remains NOT_RUN. The terminal-only plan now includes matched N5/N8 pairs, with future N5cheap implementation explicitly pending.

Implementation and knowledge published normally to `origin/thesis-integration` at `64e447f7ff85091bef51be75ba5b7dd1dc47684d`; follow-up metadata records this identity.
