**16:07UTC amendment [Run/Open]:** the selected four-head Scan trajectory
study now independently accepts DGCNN Scan GSD terminal36.075731%,8715IDs.
[Raw hashes and analysis](gsd_scan_trajectory_selection_20261019.md).
The older59386-event scope below is now partly superseded: do not run its
Scan GSD component again. Residual50671events comprise ModelNet GSD
Occlusion/LiDAR4936, ModelNet original37020 and Scan original8715.
No revised controller/budget/approval is activated. Before a continuation,
prepare a revised scope and whole-phase duration estimate; preserve the
accepted smoke admission and already complete Scan trace. Earlier65–90min
estimate described the larger four-job scope, not this reduced residual.

**15:10UTC amendment [Run/Open]:** four r9 GPU smokes nowaccepted under
separate smoke-only user scope4c2c4854…a4b139;408predictions/8batches verified.
[Acceptance and raw evidence](gsd_multibackbone_dgcnn_smoke_20261019.md).
The remaining59386 full predictions stillawait explicitlongphaseapproval.
Earlier instructions below to runsmokes afterapproval are superseded: reuse
these identical acceptedtests with a transparentcross-scope admission; do not
relabel their authorization or repeat solely for scopehashidentity. Original
r9v2scope70b19135…e8aa6c and remainingcontroller e304d301…83da18 remain
prepared/unapproved, with its smoke-admission gate to resolve before launch.

# DGCNN continuation after r7 interruption — 2026-10-19

**[Code/Run/Open] Current r9 v2 scope: technically ready, awaiting explicit long-phase approval.**
All ten r9 setup cells/native GPU/data/checkpoint checks pass; full evidence
2984884bytes SHA82737111…d0601e verifies on actualDrive/locally. Four exact
pinned native binaries are restored with fresh bytes preserved. UnchangedCPU
proof passes eight strict model loads, six matched-source bindings, no-build,
CUDA-uninitialized and real originalDDIM5/35 checks. FinalCPU archive2978673bytes
SHAa06fbba2…7df72a3 verifies actualDrive/local allmembers. One r7-specific
Python dependency (multimethod1.9.1) and exact legacyScanPointMAE asset were
restored after failed checks; both failed attempts remain immutable evidence.
The v1 r8 controllers remain historical local files and must not be launched.
The latest user confirmations authorize Google Drive consent, not this new
GPU phase. Existing CN/PN and six source-only evaluations are retained.

Question: with the same audited alternative DGCNN checkpoints and frozen
classifiers, how do GSD-inspired latent spectral guidance and original3DD
compare with matched source on ModelNet40-C and ScanObjectNN-C?
These are alternative-checkpoint comparisons, not GSDTTA paper reproduction.

## Reuse and interruption policy

The verified raw recovery contains 13 complete ModelNet GSD corruptions plus
Occlusion910/2468:32994 predictions and481 batch receipts. Raw ZIP147769635bytes,
SHA256 `e411595190ec043f964f42d84e4aeb4e23312d7403b4735310933a6f3f48bc8a`;
[paths and independent checks](gsd_multibackbone_recovery_20261019.md).

Keep the13 completed corruptions. Evaluate complete Occlusion and LiDAR in a
fresh two-corruption invocation with seed0; the prior910 Occlusion predictions
remain partial evidence and are excluded from the composed result. The
existing runner has no exact mid-batch resume, and actual RNG state was not
saved. This repeats910 examples to obtain a complete, auditable corruption
through the existing runner, without changing its inference implementation.

Any composed all15 summary must identify both attempts and the new RNG
boundary. It is complete dataset coverage assembled from two attempts, not a
single uninterrupted seed0 execution. The interrupted old13 have verified
per-corruption classifier hashes but lack a final LION before/after hash;
retain that limitation. Do not fabricate a canonical full15 raw run or copy
the old run into a resumed mutable directory. Keep the new two-corruption
run's raw `partial` status and put composition in a separate derived bundle.

## Exact remaining scope

| Job | Corruptions | Examples per corruption | Batch | New prediction events |
|---|---|---:|---:|---:|
| ModelNet40-C / GSD-inspired + DGCNN | Occlusion, LiDAR |2468|70|4936|
| ScanObjectNN-C / GSD-inspired + DGCNN |all15|581|70|8715|
| ModelNet40-C / original3DD + DGCNN |all15|2468|32|37020|
| ScanObjectNN-C / original3DD + DGCNN |all15|581|32|8715|

Total59386 new full-evaluation prediction events. Scan uses official
`main_split`, never `main_split_nobg`. Severity5, seed0, original ordered IDs,
unchanged checkpoints/data/preprocessing, no target-label tuning. Source
ModelNet58.797947% and Scan32.782559% remain the matched DGCNN references;
report its weaker source caveat. Source comparisons for a subset must use
the same corruptions/IDs, not the full15 source macro.

Before the four remaining jobs, run Gaussian+Background one-batch smokes
for the same four dataset/method pairs:408 predictions total. Independently
review smokes before the remaining phase; both share one approved7200s cap.
These smokes check the fresh runtime; they do not replace previous results.

Frozen parameters: GSD B70/M800, beta2, spectral1.17, SCD1, horizons10/30,
updated decoder and final-alpha false. Original B32, p.95, gamma/eta.01,
unnormalized SCD sum, horizons5/35, original decoder and final-alpha true.
LION raw/eval, EMAoff; unchanged repository `tta.py` in the original arm.
This remains a whole-policy comparison, not isolated spectral attribution.

## Executable identities and guards

Runtime: `tta-multibackbone-dgcnn-a100-r9-20261019`, endpoint
`gpu-a100-hm-kkb-euw4a0-1h0qzqfkhb7tr`, A100 High-RAM.
Actual base `dev@a458cd46bba1288a3f41302548bc777700cbc7d4` is separate from
the frozen research source ZIP SHA256
`f82cf5052b8100591df679b159acec9f172144c5f6a0a916c5a5255cadeed4cf`.
v7 asset manifest SHA256
`78cdb53891bc0714dacd100eb220a5f6deb26bfe36f4e10b0aaa16f9aced60f9`.

Pinned checkpoint SHA256 values:

- ModelNet DGCNN:
  `e9eb118cd5adb42b7a564e2cc396ce698dee6cad42342d5e6c518e47ea152494`.
- Scan DGCNN:
  `7816a3ccf915406c5cc613883d8deb187d78e8f505df925d83cdcb8ac1343bfe`.
- LION:
  `807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`.

The existing frozen DGCNN contract uses FPS1024 and the checkpoint's audited
input layout, with no new normalization or augmentation. LION reconstruction
returns2048 points before that same classifier preprocessing. Dataset file
identities and original label maps remain bound by the unchanged asset manifest
and passed the actual r9 CPU proof; GPU smokes remain required.

Prepared local files under
`result/preparation/cli_multibackbone_dgcnn_20261019_r9/`:

- Scope `dgcnn_recovery_scope_20261019_r9_v2.json`, SHA256
  `70b19135cae29f8b4b8a2dc0cdde4231f15c93c5ef86ae8562a3d2c160e8aa6c`.
- Smoke controller `dgcnn_recovery_smoke_r9_v2.py`, SHA256
  `63ffe4ee11f6de10befddc8eed4b5f6b12225135ab4f69f6b11672a7b36d2e72`.
- Remaining controller `dgcnn_recovery_remaining_r9_v2.py`, SHA256
  `e304d301ad7cc9086853443e68661c40bf5f33387a504c8f94f792055d83da18`.

Python3.8 parsing and actual pure command-construction assertions pass: four
DGCNN-only jobs, correct selected corruptions/seed/batch/data paths. No GPU
execution occurred in those checks. Controllers reuse the existing reviewed
v7 phase flow, retain streaming, frozen-state/receipt checks and a shared
budget ledger, and reject missing or mismatched explicit approval.

After approval, persist the exact scope at the study root and the user's reply
in `dgcnn_recovery_approval_20261019.json`; never precreate an approved receipt.
Submit the two reviewed controllers, in order, through monitor70119/63241,
with their exact hashes and markers:
`MULTIBACKBONE_DGCNN_RECOVERY_SMOKE_COMPLETE` and
`MULTIBACKBONE_DGCNN_RECOVERY_REMAINING_COMPLETE`.
The remaining controller requires an independently accepted smoke receipt.

Each controller calls unchanged `scripts/eval_multibackbone_guidance.py`
with the table's method/dataset/batch and `--backbones dgcnn --seed 0`.
Full invocations use `--max-batches 0`; smokes use Gaussian+Background and
`--max-batches 1`. Fresh UTC directories under the VM study's `gpu_phases/`
and `MyDrive/thesis/result/gsd_multibackbone_study/<dataset>/<method>/`
preserve raw seven-file bundles, indexed logits, batch diagnostics/receipts,
corruption ZIPs and final ZIPs. Verify physical Drive bytes and download every
material completed model/stage before accepting it. Original source and raw
recovery bytes are immutable inputs.

## Duration, stopping and approval gate

Estimate65–90 additional GPU-phase minutes, including smokes/model loads and
in-job artifact persistence; hard cap120minutes across both controllers.
Previous accepted jobs consumed2782.578s plus approximately61min interrupted
DGCNN work, which must remain in the overall cost record. Setup/authorization
time is separate and must not be presented as model-computation time.
At13:10UTC the sole A100 assignment reports6.77CU/hour and449.47CU balance.
Expected research-phase cost is approximately7.33–10.16CU, max13.54CU under
the120min wall cap; download/analysis and continued allocation are additional.

This new phase exceeds60min and needs one explicit scoped approval under the
[Colab skill](../.codex/skills/colab-research-operator/SKILL.md), after the
runtime/setup/source proof and immutable readiness bundle are reviewable.
Allocation/setup/artifact recovery did not authorize a new long GPU retry.

Stop advancement for setup/proof/hash/native mismatch, failed smoke, OOM,
nonfinite tensors, changed frozen states, receipt failure or exhausted cap.
On disconnect audit the same endpoint/process/files before another cell;
preserve partial Drive receipts instead of duplicating work. No runtime
restart or shutdown is part of this plan. Keep r9 until the user explicitly
declares the task complete and authorizes closure.


## r9 readiness evidence — 13:10 UTC

Actual remote base/dirty setup bytes are archived separately from the frozen
v7 research source. Exact executed controls and unchanged manifest are sealed
in the r9 readiness bundle under the preparation result root. Main sources:

- Complete setup archive
  `result/preparation/cli_multibackbone_dgcnn_20261019_r9/cli_setup_r9_evidence_20261019T130224Z.zip`,
  SHA256 `82737111c6b919f24f63d640ecd5e30f37bce29f070fdec19f16762a60d0601e`.
- Final CPU proof
  `result/preparation/cli_multibackbone_dgcnn_20261019_r9/r9_cpu_proof_final_20261019T130828Z.zip`,
  SHA256 `a06fbba2cfe53489103205c4cd814d89d58a2cacde6d5ca5855de47357df72a3`.
- Prior failed CPU archives SHA86386489…515b8a and80d99eb7…89d83f
  remain nested in the final proof with exact helper/log/env change evidence.
- Fresh native-byte preservation1497859bytes SHA464d1b55…cfdb5a is onDrive
  before pinned replacements and is nested in the failed-proof chain.
- Exact extra asset: ScanPointMAE351525407bytes
  SHA53f6c2e4…535dac, physicalDriveweights/scanobject_jt.pth; CPU strictloadpass.

No activation/approved receipt exists. On user approval, record the exact reply
and scope SHA70b19135…e8aa6c, then run four smokes/independent verification,
followed by the four remaining jobs under the same7200s budget. Retain and
extend only bounded local sleep prevention for the approved supervision period;
Colab lifetime cannot be guaranteed. All existing runtime state is preserved.
