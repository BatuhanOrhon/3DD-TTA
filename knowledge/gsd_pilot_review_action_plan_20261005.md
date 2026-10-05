# GSD pilot review and next action plan — 2026-10-05

Status: Batch1 runner implemented; Batch2 completed and validated from the
user's Colab attempt on 2026-10-05. P_PC passes the registered pilot gate;
Phase5 development replication is next.
Repository: `gsd-smooth-spectrum@5b21382a9c0cb5e3572d383f11a8446e45cb12f0`.
Evidence labels: [Run], [Code], [Inference], [Open]. Existing dirty research
notes, including the independent M_max=1800 study, remain separate.

## 1. Evidence and interpretation

All figures below concern the hard-v1 composition development pilot, seed0,
64 indices per corruption. They are not full-dataset or multi-seed results.

| Within-phase contrast | Gaussian delta pp | Impulse | Background | Shear | Equal-corruption mean |
|---|---:|---:|---:|---:|---:|
| R_SG minus R_S0 | +1.5625 | +3.1250 | -1.5625 | +1.5625 | +1.171875 |
| R_GS minus R_G0 | -1.5625 | -4.6875 | +14.0625 | 0 | +1.953125 |
| P_PC minus P_SUM | 0 | +1.5625 | +1.5625 | 0 | +0.781250 |
| P_PC minus P_NORM | -1.5625 | +3.1250 | +4.6875 | -1.5625 | +1.171875 |

- [Run] Routing has 16 complete arms; projection has 12 complete arms.
  Existing analyzer outputs report valid within-phase comparisons. P_PC has
  175/256 correct corruption-example rows, versus P_SUM173 and P_NORM172.
  Against P_SUM it corrects5 and breaks3; against P_NORM it corrects5 and
  breaks2. These are small, uncertain signals, not established improvements.
- [Run] P_PC's valid nonterminal style conflicts are100/256,115/256,
  1589/2176,71/256 in Gaussian/Impulse/Background/Shear:27.7344–73.0239%.
  The earlier report's31.25% lower endpoint was an arithmetic transcription
  error. Conflicts and prediction changes do not themselves prove benefit.
- [Run/Inference] Scale100/1000 offers no consistent benefit. Background100
  failed after32 complete examples; Background1000 also failed nonfinite,
  and attempt-0005 contains only a partial manifest, no accuracy bundle.
  The predeclared safety rule excludes1000 from replication. Retain both
  failures; do not retry these arms or claim an optimal coefficient.
- [Inference] R_SG remains an exploratory routing candidate. Reverse routing's
  gain over its weak style-off anchor is concentrated in Background and does
  not establish superiority over SCD. Neither routing candidate is promoted.

Sources (immutable raw ZIPs/manifests and separate derived reports):

- [Routing report](../result/modelnet40_c/gsd_guidance_composition_v1/routing/attempt-0001/validation.md)
- [Projection report](../result/modelnet40_c/gsd_guidance_composition_v1/projection/attempt-0001/validation.md)
- [Shear scale report](../result/modelnet40_c/gsd_guidance_composition_v1/analysis_20261005_shear/validation.md)
- Scale raw records: `result/modelnet40_c/gsd_guidance_composition_v1/scale/attempt-0001/`
  and `scale/attempt-0005/phase_manifest.json` under the same method root.

## 2. Correction to the earlier stopping decision

[Inference] The earlier ingestion recommendation to stop projection because
per-corruption intervals include/touch zero was stronger than the registered
Phase4 rule. That rule requires positive aggregate paired gain against C_SCD,
P_SUM and P_NORM; it does not require statistical significance at pilot size.
P_PC passes the two observed aggregate contrasts. The C_SCD contrast is
missing, so the gate is **unevaluated**, rather than demonstrated to fail.
This review supersedes the earlier permanent-stop recommendation. It does
not promote PCGrad or change the original numerical gates.

[Run] Routing and projection share source/runtime/native identities and draw
keys, but their actual prepared-input hashes differ for all four corruptions.
Diagnose also differs in native/runtime identity. Do not pair old C_SCD with
new candidates or rank routing versus projection using their absolute macros.
Hash inequality does not measure tensor difference or identify its cause.

[Code] `gsd_paired_inputs.py` hashes inputs, encoded latents, style and noise;
the runner stores aggregate hashes, not a replayable prepared tensor bundle.
Running C_SCD alone in another process cannot guarantee recovery of the old
candidate states. The existing projection phase selects only P_SUM/P_PC/P_NORM.
A complete new comparison requires explicit runner support.

## 3. Recommended bounded follow-up

[Inference] Prioritize completing the projection control comparison, because
both of its existing mechanism contrasts are positive. This is a workload
choice, not evidence that P_PC outperforms R_SG. Park routing pending this
answer; do not expand to a six-arm tournament or new algorithm grid.

Proposed block: **C_SCD + P_SUM + P_PC + P_NORM**, four corruptions, first64
registered indices, seed0: **1,024 classifications (16 arm archives)**.
All four arms must consume clones of the same preparation in one process.
This necessarily re-evaluates three projection conditions to obtain the
missing matched baseline; it is an explicitly scoped control-completion
study. Existing results remain valid and are not overwritten or pooled.

Preserve hard v1/weight1, SCD weight/reduction, graph, B32, DDIM5/35,
gamma/eta, frozen/eval models, raw weights and original decode. No clipping,
smoothing, normalization, model initialization policy or deterministic-kernel
change is bundled with this study. Investigate identity differences separately
if needed; finding their root cause is not required to share one preparation.

### Batch 1 - Prepare the complete control block (implemented)

- Goal: make the proposed four-arm pilot runnable and auditable.
- Scope: explicit opt-in control-completion mode, fixed arm set and pilot
  indices; shared preparation; separate phase/attempt output; analyzer support.
- Touched areas: `scripts/run_gsd_composition.py`, phase definitions in
  `gsd_composition_protocol.py`, `scripts/analyze_gsd_composition.py` as needed,
  `gsd_paired_inputs.py` only if component hashes are added; Colab handoff.
- Stack context: Python3.8 Colab/PyTorch; CPU orchestration locally.
- Dependencies: this review; inspect the existing phase contracts first.
- Implementation notes for $development-agent: reuse existing arm algebra and
  prepare-once/clone path. Log per-component input/latent/style/noise hashes
  and configuration/draw identities to localize future mismatches. Full tensor
  persistence/replay is a separate optional change, not a launch dependency.
  Do not repurpose diagnose or replicate to bypass their fixed scopes.
- Verification: local compile and a no-inference phase-plan preview confirm
  exactly16 planned arms,64 indices each,seed0 and fixed settings. Each batch
  identity records component tensor and scheduler hashes. No automated tests
  or GPU inference were run. The user should review Colab's preview before
  running the execution cell.
- Knowledge artifact to update: this plan, `findings_log.md`,
  `open_questions.md`, `colab_gsd_block_routing_20261004.md`.
- User review gate: source-file synchronization and separate preview/execution
  cells are provided in the Colab handoff below.
- Out of scope: smoke/diagnose repeats, failed scale retries, new algorithms.

### Batch 2 - User-run control completion and result decision (completed)

- Goal: evaluate the missing practical SCD comparison.
- Scope: the1,024-classification block above, one seed; sequential arms share
  preparation. Do not launch controls as independent parallel notebook jobs.
- Touched areas: new result attempt and derived analysis only.
- Stack context: Colab GPU operated by the user; local offline ingestion.
- Dependencies: Batch1 ready and its exact command/settings frozen.
- Implementation notes for $development-agent: no algorithm changes. Request
  the full attempt directory ZIP including manifest, all arm archives and
  console/error logs; preserve partial outputs and credentials-free metadata.
- Verification: complete manifest;16/16 bundle archives valid; four paired
  blocks have matching inputs/runtime/config and component hashes; all source
  hashes match the recorded commit; all64 examples per arm and 5/35 step
  coverage validate. P_PC macro is69.5313%, versus C_SCD67.1875%,
  P_SUM68.3594% and P_NORM67.5781%. All three registered aggregate contrasts
  are positive. Per-corruption paired intervals remain broad and touch/include
  zero.
- Knowledge artifact to update: this plan and dated findings/open questions.
- User review gate: P_PC is nominated for Phase5; evaluate that replication
  before any all15 expansion.
- Out of scope: partial-as-complete accuracy, cross-attempt pairing, broad tuning.

### Batch 3 - Conditional replication under the original Phase5 rule

- Goal: assess stability on more examples and seeds after Batch2 passes.
- Scope: freeze selection manifest; same four arms/corruptions, next256
  indices, seeds0/1/2: **12,288 classifications**. If Batch2 fails an accuracy
  contrast, close projection accuracy promotion; if technically incomplete,
  mark the gate unevaluated and inspect the failure rather than select a winner.
- Touched areas: selection manifest, new replicate attempt and analysis.
- Stack context: user-run Colab and offline paired analysis.
- Dependencies: complete valid Batch2 and all three positive contrasts.
- Implementation notes for $development-agent: use registered Phase5 support;
  preserve comparators and the original fixed settings.
- Verification: mean P_PC-minus-C_SCD >=0.2pp, positive in all three seeds;
  mean Background delta >=-1pp; positive three-seed mean against both P_SUM
  and P_NORM. Report sample SD and paired uncertainty. Without verified object
  correspondence, give per-corruption intervals, not a pooled object interval.
- Knowledge artifact to update: findings/open questions/README and this plan.
- User review gate: assess replication before any all15 expansion.
- Out of scope: calling this pristine held-out confirmation. These files were
  previously used in development; pilot-disjoint indices do not erase that.

## 4. Implementation handoff

Batch1 was implemented in `gsd_composition_protocol.py`,
`gsd_paired_inputs.py`, and `scripts/run_gsd_composition.py`. It adds an explicit
four-arm pilot control-completion phase using C_SCD/P_SUM/P_PC/P_NORM and the
existing first64/seed0/four-corruption manifest. Shared preparation and
inference algebra are unchanged. See the Colab handoff for source synchronization,
preview and execution cells. No local GPU run, diagnose/smoke/scale repeat,
replication change, stage or commit was performed.
