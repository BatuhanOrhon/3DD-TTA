# CloudFixer XYZ baseline across three frozen heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. User opted out of permission/design check-ins; execute within the requested research scope.

**Goal:** Evaluate the established XYZ proximalN8 baseline with/without PM on full ModelNet40-C using the three matched heads, then compare with accepted original CF.
**Architecture:** Opt-in baseline profile in the existing multibackbone worker/package/analyzer; preserve old originalCF behavior and schemas. Reuse the exact `proximal_policy` sampler/readout with a multi-head callable and immutable per-batch receipts. No independent PM-off trajectory.
**Tech Stack:** Existing Python3.8/Torch2.1.2/NumPy/nativeFPS and Colab CLI; no new dependencies.
**Spec:** `knowledge/cloudfixer_baseline_multibackbone_20261123.md`.

## Global Constraints

- Work only `/Users/batuhan/.codex/worktrees/0e7f/3DD-TTA`, branch research/cloudfixer-transfer-20261114; never alter another checkout/runtime.
- Main baseline `cloudfixer_xyz_proximal_pm_n8_k80_late_v1`: steps8/start80/tau.1/retained.95/probes[20,10,0], pre-correction20/10 and true extra-call t0. Nine denoiser calls, float64 softmax/equal PM, frozen prior/heads.
- PM-off means the true t0 from that SAME N8 trajectory, not the old N32 `POLICIES['proximal_terminal']`.
- Head order dgcnn/curvenet/pointnext; same accepted ModelNet40 checkpoint/state identities, class40 and preprocessing as accepted originalCF campaign3596befa489b/source228555f.
- All2468 originalIDs/all15/s5/seed0/B64/final36. Batch is fixed after technical validation; any change creates a new manifest/campaign.
- Old originalCF units/validation/recovery/reporting/default CLI stay compatible. No mutable-global monkeypatch dispatch or relabelled logits/probabilities.
- New manifest/cache identity distinguishes profile, source, policy, exact head/prior/data identities and RNG policy; no mixing old CF/LION caches.
- Use unit-addressed fixed geometry RNG seed (corruption/seed/IDs) inside torch fork; reuse exact existing proximal_policy, whose index-addressed timestep seeds isolate classifier observations. Multi-head forwards must restore full Python/NumPy/CPU/CUDA RNG.
- Save all three candidate logits, true terminal logits, float64 PM probabilities, selected IDs/labels, input/output/trajectory identity, RNG evidence, timing and frozen-state checks. No test-label tuning or inference use of labels.
- One shared geometry trajectory for three heads; separate per-head/corruption terminal/PM reports and completion receipts. Save each B64 unit to Drive, per-corruption SHA ZIP/cursors last; host independent raw recount before full accuracy claims.
- Author checkpointSHA b64841d022e08b1a89f58dd9b16d9023e00be52fbca3dca735d5798362bb5a5a; officialcommitf043e24414fcf4165fdc892abff21215c9989141. Use existing verified assets/native setup.
- Accepted CF reference is full all15 `knowledge/evidence/cloudfixer_multibackbone_20261122/a100_full_recount.json` + original archive index/raws, source228555f, campaign3596befa489b. Legacy LION reference remains historical, not the current comparator.
- No new originalCF GPU benchmark, N32 arm, changed PM window/entropy/pose method or Scan GPU test in this task.

## Review Focus

- Terminal/PM deliberately disagree: CSV PM must use float64 probability mean, terminal exact last candidate; never label intermediate logits terminal.
- Corrupted policy/candidate index/count/shape or nonfinite probabilities must fail before acceptance; classifier/head/data drift fails closed.
- Old CF payloads and cache identities stay unchanged; new profiles cannot reuse old receipts.
- Resume must reject sparse/mixed IDs, wrong RNG/trajectory identity and incomplete scopes; completed corruption cursors skip only validated complete ZIPs.
- Comparing CF must match actual original IDs/labels/head states across different campaign hashes, not zip order; paired corrected/broken counts require raw saved predictions.

### Task 1: Opt-in baseline profile, raw analysis and compatibility

**Files:** Modify `scripts/run_cloudfixer_multibackbone.py`, `scripts/prepare_cloudfixer_multibackbone.py`, `scripts/analyze_cloudfixer_multibackbone.py`, `tests/test_cloudfixer_multibackbone.py`; modify setup helper only if the task-specific Drive namespace needs it. Do not edit the sampler mathematics or existing Point2Vec path. Append brief code notes to `knowledge/cloudfixer_baseline_multibackbone_20261123.md` under an implementer section.

**Interfaces:** Packager gets an explicit opt-in baseline profile argument (default originalCF unchanged), records exact policy and baseline profile in manifest, packages matching source. Runtime reads that manifest profile. Existing worker helper signatures remain backward compatible via optional keyword profile/method parameters where needed. `proximal_policy(model, classifier_callable, x, mask, POLICIES['proximal_pm'])` accepts callable returning `[B,H,40]` logits; output candidate logits `[B,3,H,40]`, mean `[B,H,40]`, clouds/indices/calls. Use returned true terminal candidate; do not call N32 policy. Existing `checked_runtime` can supply the unchanged prior, accepted heads/nativeFPS and geometry conversion. New report method names `source_raw`, `proximal_n8_terminal`, `proximal_n8_pm`.

- [ ] Write failing tests with intentionally incorrect terminal but correct PM logits/probabilities. Assert exact[20,10,0], actual9call contract, probability (not logit) mean, true-terminal equality and no N32 invocation. Use tiny fake CPU models, no real model/data inference on Mac.
- [ ] Test new-profile payload rejection for changed policy/probe/order/nonfinite/missing candidate/terminal mismatch and old-CF payload compatibility. Pin B64 all2468/585/final36 and independent profile/cache identity.
- [ ] Implement the smallest profile dispatch and multi-head callable; call `proximal_policy` unchanged. Archive explicit profile policy, unit-addressed fork seed, per-candidate/head logits/probabilities and timings. Full state hashes before each commit; no head adaptation.
- [ ] Preserve per-batch and per-corruption recovery/ZIP/head receipts and originalCF behavior. New-profile cache validation must explicitly know expected profile. Use distinct method fields, not `frame_logits` renamed as terminal.
- [ ] Extend CPU analyzer to recompute each PM from raw candidates, terminal and all standard CSVs; add optional `--cloudfixer-run` with the15 accepted originalCF extraction dirs. Strict same-label/ID/head-state/dataset checks, then CF-only/ours-only/net for each PM/terminal cell and PM-versus-terminal same-trajectory pairs. Separate single-seed descriptive results from statistical claims. No inference labels in runtime.
- [ ] Run focused test file, then the impacted existing CloudFixer/store/Point2Vec/transfer/source-head CPU suites. Document red/green evidence and exact commands; no redundant broad suite after checks pass.
- [ ] Commit only scoped files; report source commits/tests/concerns to the designated report file. No Colab calls or other agents from implementer.

### Task 2: Root operator — canonical setup, technical gates, full evaluation and publication

- [x] Allocate distinct owned A100 (r25 standardRAM,40GB GPU/83.47GiB host, within user allowance); cell1 alone/reconnect, user-operated Drive consent, canonical2–10 in order with physical markers. No peer execution.
- [x] Prepare exact source/asset manifest and immutable Drive/host output; technical smoke and fresh-process recovery replay, B64 Gaussian and max-point capacity; record measured whole-phase ETA/shared balance and enforce independent wall/CU stop limits.
- [x] Run all15 once, save every batch/corruption, download completed ZIPs and independently recount. Do not restart while active state is unknown.
- [x] Compare PM vs true N8 terminal and both vs saved originalCF; all15/eachhead/corruption tables, corrected/broken counts and one-seed limitations.
- [x] Archive exact source/log/config/environment/arrays, Drive upload→freshhost SHA/CRC and official cloudflush; update knowledge and curate necessary reviewed changes to latest thesis-integration preserving peers. User's standing instruction authorizes close after completed work; record explicit scope before teardown.

### User steering — measured memory/throughput batch selection

2026-11-23: user requests efficient memory use. B64 remains the accepted
current package; it is not a measured optimum. After native setup and realB64
Gaussian/Upsampling capacity, resource-onlyB96/B128 probes may inform a
pre-benchmark choice using throughput plus GPU/host memoryheadroom. The
currentGlobalConstraintB64 is superseded only if those realmeasurements
justify anotherbatch and a separately reviewed registration/package/host
validator/recovery supports it. Never bypassB64 checks or mix campaigns.
Keep one chosenbatch for all15; disclose oldCFreferenceB64 if changed; never
selectbatch fromaccuracy or labels. Policy, heads, split, corruption scope
and pairedPM/terminal trajectory remainfixed.

### Task 3: Conditional measured-batch registration (before Task 2 fulltest)

User's memory-efficiency steering and actual B64/B96 resource observations
justify baseline-only registration for the three predeclared tested batches.
Current paired resource helper finds B96 eligible with ~5.50% geometric
throughput gain; B128 is still pending. Implement only registration and dynamic
block/archive counts in the current worker/packager/analyzer when necessary,
plus covering fake CPU checks. Preserve sampler mathematics, old CF B32/B64
and old baseline B64 compatibility. B64/B96/B128 have39/26/20blocks per
corruption and585/390/300all15; final36/68/36. Fulltest uses one chosen batch,
new source/campaign/package/store and fresh technical gates; no labels select
batch. Exact implementation brief and tests/report contract are in the ignored
`batch-registration-brief.md`. Root selects and deploys only after independent
review and completed resource probes. Original CF remains the saved B64 run;
state this protocol and unit-RNG difference in the final comparison.
