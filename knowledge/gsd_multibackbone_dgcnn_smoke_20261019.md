# DGCNN r9 GPU smoke tests — 2026-10-19

**[Run/Code/Open] Accepted: four of four smoke jobs,408 predictions,8 batches.**
These are execution/state/artifact checks on short prefixes, not all15 accuracy
evidence. The remaining full DGCNN evaluations are not launched or authorized
by the user's smoke-only instruction.

## Scope and source

User: “Tamam bu testleri de çalıştır”. Recorded smoke-only authorization at
15:03:38UTC; existing A100 High-RAM session
`tta-multibackbone-dgcnn-a100-r9-20261019`, endpoint
`gpu-a100-hm-kkb-euw4a0-1h0qzqfkhb7tr`, same monitor70119/local63241.
All three existing remote kernels were independently observed idle before
launch. No setup replay, allocation or runtime restart.

Scope SHA256 `4c2c4854dcac8951f7a17934adc71dc219b65290e35ca2e8c7711014b2a4b139`;
executed controller `dgcnn_smoke_only_r9_v1.py` SHA256
`c6ea2f8929a2f906af0a020c06870052686c1a4513e3289b9454c85190dd4ba6`.
It permits only smoke mode/onebatch, uses a separate approval/ledger/lock and
1800s cap, and cannot activate the unapproved7200s full-phase proposal.
Estimate10–20min, cap30min; actual allfour job wall154.552092s/2.576min,
including model loads and in-job Drive writes. Reconstruction60.128219s.
Job-time cost estimate0.290644CU at6.77CU/hour; ongoing allocation/CPU
validation/transfer are additional. No runtime-lifetime guarantee.

Unchanged research source ZIP SHAf82cf505…deed4cf and assetmanifest
78cdb538…60f9, separately from remote dev@a458cd4 plus archived setup overrides.
Exact four r7-pinned native binaries restored, no-build bootstrap. Extra
PointNeXt dependency/legacyScanPointMAE asset preparation and completeCPU
proof were accepted before this phase. DGCNN alternative checkpoint identities
are unchanged; this is not GSDTTA paper reproduction.

## Four jobs

Each job uses Gaussian+Background, severity5,seed0, ordered original IDs.
Scan uses581-object main_split provenance, never main_split_nobg.

| Dataset | Method | Batch per corruption | Predictions | Peak allocated GPU MB |
|---|---|---:|---:|---:|
| ModelNet40-C | GSD-inspired latent spectral guidance + DGCNN |70|140|32366.3784|
| ScanObjectNN-C | GSD-inspired latent spectral guidance + DGCNN |70|140|32366.3477|
| ModelNet40-C | original3DD + DGCNN |32|64|14494.0068|
| ScanObjectNN-C | original3DD + DGCNN |32|64|14493.9795|

GSD retains M800/beta2/spectral1.17/SCD1/10-or30steps/updated-style/alphaFalse;
original retains B32/p.95/gamma-eta.01/unnormalizedSCD/5-or35steps/original-style/
alphaTrue. LION raw/eval/EMAoff; classifier and LION states frozen.

Phase VMpath `gsd_multibackbone_study_20261018/gpu_phases/20261019T150433Z_dgcnn_recovery_smoke`;
first job15:04:33, last accepted marker15:07:08UTC. All jobs exit0.

## Independent acceptance and immutable evidence

`analyze_dgcnn_smoke_only_v1.py` SHA1f95eb60…c0f3c56 reuses the prior
independent validator with fixed4job/2corruption expectations. No inference
implementation change. Offline isolated NumPy validation passes all408 indexed
logits/argmax/labels/counts, eight geometry/diagnostic records, finite output/
guidance metrics, classifier/LION before-after hashes, RNG observer guards,
no-build native bindings, strict checkpoint/preprocessing links to the same
source archives and original realDDIM5/35 schedules. All seven standard files
exist in all four final bundles. Prefix status remainspartial; source full15
macros are not compared with smoke averages. Derived matched prefix tables
are in `smoke_only_analysis_v1/`; no accuracy-improvement conclusion.

All four raw ZIPs were persisted by the runner to actualDrive after each model,
then downloaded into canonical local result paths and independently verified:

| Run | Canonical local ZIP | SHA256 |
|---|---|---|
| MN GSD | `result/modelnet40_c/gsd_inspired_latent_spectral_guidance/20261019-150435_dgcnn-seed0.zip` |c9d58983b883402f12c5897ae9015fe7c20b338ee73d9f9fc090d3fe6507884f|
| Scan GSD | `result/scanobjectnn_c/gsd_inspired_latent_spectral_guidance/20261019-150521_dgcnn-seed0.zip` |d07874e21938e365d473adf830ed092b21015186c6199b839756cabd5695f681|
| MN original | `result/modelnet40_c/3dd_original/20261019-150606_dgcnn-seed0.zip` |76fb728a112d8f0a192ba603b0e65771c5c284e7d00fff1dde968717cdacecf4|
| Scan original | `result/scanobjectnn_c/3dd_original/20261019-150637_dgcnn-seed0.zip` |3062f8bdb3d02fed60610b3c4da04b9e040d4a77efaa97983b444dd3a88cabb3|

Their Drive paths are under
`MyDrive/thesis/result/gsd_multibackbone_study/<dataset>/<method>/<run_id>/final.zip`,
with perbatch receipts and percorruption ZIPs alongside.

CPU finalizer edc3e452…6cbe30 physically rereads4finalarchives,8batch receipts/
16files and8corruption archives onDrive, then seals analysis/source/ledger/
phase/audit/admission. Evidence17855bytes SHA256
`eacc97ac11defcf72af87c418a6775e726e1c404b1cfd346fe2518cf3e07f4aa`;
local `result/preparation/cli_multibackbone_dgcnn_20261019_r9/dgcnn_smoke_only_evidence_20261019T150837Z.zip`,
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261019T150837Z/dgcnn_smoke_only_evidence_20261019T150837Z.zip`.
ActualDrive/local size/SHA/CRC/allmanifest member checks pass; exact executed
finalizer source matches. Nested analysis10357bytes SHA4cf1c601…aeee595.

Local receipt rebinding verifier8a76b586…fae2bed verifies all actualDrive
receipt-file identities equal members of the independently analyzed rawZIPs.
Binding bundle1631bytes SHA256
`c519af6cd376484a2da25b5a07fb68c2cc5badbd20019dfe57d0fece56c8d81f`,
local `result/preparation/cli_multibackbone_dgcnn_20261019_r9/dgcnn_smoke_sealed_member_binding_v1.zip`,
Drive same preparation root filename. Fresh Drive download/SHA/CRC/
verifierbytes/8batch count pass. Smoke-only authorization5915byte ZIP
SHAd54e1f95…e582ea also physically verifies onDrive/locally.

## Decision and continuation

The four fresh-runtime execution/state/persistence gates pass. No new all15
accuracy or statistical superiority is established. Reuse existing accepted
source/CNPN/13MN_GSD corruptions. Remaining59386 full DGCNN predictions are
a distinct65–90min proposal awaiting explicit scope approval. These four smoke
tests need not be repeated solely because the authorization identity differs:
if the full proposal is approved on this unchanged runtime, create an explicit
cross-scope admission comparing source/assets/dataset/method/batch/corruptions
and preserve the original smoke-only ledger. Do not silently relabel the
4c2c4854 test authorization as the70b19135 full-phase authorization.

r9 is still live per15:10UTC serverinventory; same monitor70119/63241 and
bounded local awake73723/64954 remain. No teardown authorization or command.
[Remaining plan](gsd_multibackbone_dgcnn_recovery_plan_20261019.md).
