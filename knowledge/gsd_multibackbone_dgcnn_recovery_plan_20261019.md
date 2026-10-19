**11:35UTC [Run/Open] Execution binding superseded:** r8 vanished after a
native identity mismatch before successfulCPUproof; no new GPU evaluation.
Current runtime is r9 (`tta-multibackbone-dgcnn-a100-r9-20261019`), setup in
progress with monitor70119/63241 and bounded hostsleep prevention. Retain the
scientific scope below, but rebind session/controller/scope hashes to r9 and
restore the exact audited native binaries before proof/approval. The r8-specific
controllers below are historical prepared artifacts and must not be launched.
[Current state](gsd_multibackbone_recovery_20261019.md).

# DGCNN continuation after r7 interruption — 2026-10-19

**[Code/Run/Open] Prepared, not approved or launched.** All ten r8 setup cells
and native GPU/data/checkpoint checks pass; complete evidence verifies on
Drive/locally (SHA f1590dc9…be81f81). Unchanged v7 strict CPU proof is running
and must pass before launch.
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

Runtime: `tta-multibackbone-recovery-a100-r8-20261019`, endpoint
`gpu-a100-hm-kkb-ass1c0-xawnnd6jnsdw`, A100 High-RAM.
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
and must pass the actual r8 CPU proof before launch.

Prepared local files under
`result/preparation/cli_multibackbone_recovery_20261019_r8/`:

- Scope `dgcnn_recovery_scope_20261019_r8_v1.json`, SHA256
  `ccb2e6ebad63beb1766182f25a1e00665edf4f701bb3a205f723047a9fe4b562`.
- Smoke controller `dgcnn_recovery_smoke_r8_v1.py`, SHA256
  `a868485cc486f4be4ca202a0ee3eeef7328e38170f62707097ccaa1fdf14d580`.
- Remaining controller `dgcnn_recovery_remaining_r8_v1.py`, SHA256
  `7f6029a293df28ad6343148ba545068970c317a444114a0971210577fd476983`.

Python3.8 parsing and actual pure command-construction assertions pass: four
DGCNN-only jobs, correct selected corruptions/seed/batch/data paths. No GPU
execution occurred in those checks. Controllers reuse the existing reviewed
v7 phase flow, retain streaming, frozen-state/receipt checks and a shared
budget ledger, and reject missing or mismatched explicit approval.

After approval, persist the exact scope at the study root and the user's reply
in `dgcnn_recovery_approval_20261019.json`; never precreate an approved receipt.
Submit the two reviewed controllers, in order, through monitor63286/69569,
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

This new phase exceeds60min and needs one explicit scoped approval under the
[Colab skill](../.codex/skills/colab-research-operator/SKILL.md), after the
runtime/setup/source proof and immutable readiness bundle are reviewable.
Allocation/setup/artifact recovery did not authorize a new long GPU retry.

Stop advancement for setup/proof/hash/native mismatch, failed smoke, OOM,
nonfinite tensors, changed frozen states, receipt failure or exhausted cap.
On disconnect audit the same endpoint/process/files before another cell;
preserve partial Drive receipts instead of duplicating work. No runtime
restart or shutdown is part of this plan. Keep r8 until the user explicitly
declares the task complete and authorizes closure.
