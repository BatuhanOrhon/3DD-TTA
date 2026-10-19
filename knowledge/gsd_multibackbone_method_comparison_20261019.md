# Matched source, original 3DD-TTA and GSD comparison — 2026-10-19

[Run] Original3DD CurveNet+PointNeXt full15 independently accepted on both
datasets. The r7 runtime unexpectedly disappeared around08:05UTC during
DGCNN ModelNet GSD; remaining DGCNN results are incomplete. Existing four
CurveNet/PointNeXt GSD rows are reused, not rerun. All are published-alternative
checkpoints, not established paper reproductions. Scan uses581 main_split.

| Dataset | Backbone | Source % | Original3DD % | GSD-inspired % | GSD minus original pp |
|---|---|---:|---:|---:|---:|
| ModelNet40-C | CurveNet | 64.0113 | 67.0719 | 65.1864 | -1.8855 |
| ModelNet40-C | PointNeXt | 65.8320 | 68.2928 | 66.4722 | -1.8206 |
| ModelNet40-C | DGCNN | 58.7979 | pending | pending | pending |
| ScanObjectNN-C | CurveNet | 52.0711 | 55.2381 | 55.0086 | -0.2295 |
| ScanObjectNN-C | PointNeXt | 49.0878 | 49.6845 | 45.2094 | -4.4750 |
| ScanObjectNN-C | DGCNN | 32.7826 | pending | pending | pending |

## ModelNet original3DD accepted evidence

[Artifact] Two raw final ZIPs:74040 predictions,1170 shared geometry batches,
strict/frozen classifier and LION states, per-head geometry/RNG isolation,
actual original DDIM config/full100/active5-or35 schedule, pinned IDs/labels,
finite logits/argmax/counts, full CRC/safe paths/member manifests all pass.
Raw ZIPs physically verified on Drive before Scan started and downloaded to
canonical local paths:

| Backbone | Canonical local raw ZIP | SHA256 |
|---|---|---|
| CurveNet | `result/modelnet40_c/3dd_original/20261019-062149_curvenet-seed0.zip` | `21c644db9568c9c98a38c484950e214a00b9d5c2251b74d6946dc9040c77e10c` |
| PointNeXt | `result/modelnet40_c/3dd_original/20261019-062150_pointnext-seed0.zip` | `78ca52702d59c58f9418ef2146decb994af8838e8c2f3f18aae11c9b245cef38` |

Drive root `MyDrive/thesis/result/gsd_multibackbone_study/modelnet40_c/3dd_original/<run_id>/final.zip`,
adjacent per-batch, corruption and final receipts. Raw bytes18330116/18139081
(CurveNet/PointNeXt). Joint job2023.697s=33.728min; shared reconstruction
1637.122s appears in both configs and must not be summed twice. This is not
a matched pure-inference timing comparison with GSD.

The triplet analyzer independently joins the own source and previously
accepted GSD ZIPs by IDs/labels/checkpoint/preprocessing. CurveNet GSD corrects
2250 original errors but harms2948 original correct events (net-698); PointNeXt
corrects2273/harms2947 (net-674). These repeat the same objects across15
corruptions and are not independent binomial samples. GSD beats original on
4/15 CurveNet and3/15 PointNeXt corruptions, with no ties.

CurveNet GSD minus original: largest gains Impulse+6.726pp, LiDAR+1.094pp,
Cutout+0.446pp; largest losses inverseRBF-5.267pp, RBF-5.146pp, Rotation-4.984pp.
PointNeXt gains Impulse+4.295pp, LiDAR+3.647pp, Gaussian+0.446pp; largest
losses Density-5.673pp, Shear-5.267pp, Distortion-5.146pp.

[Inference] Both methods exceed their own measured source row on ModelNet,
but fixed historical GSD does not outperform this original3DD reference for
either head in seed0. Do not claim backbone-independent gain or isolated
spectral harm/benefit. OriginalB32/5–35/original decoder/alphaTrue/benchmarkTrue
versus GSDB70/10–30/updated decoder/alphaFalse/benchmarkFalse changes the whole
policy. Existing prior-optimal Point-MAE settings are not proven optimal for
these heads. Additional seeds and same-flow spectral0 remain unexecuted.

Immutable matched analysis16321bytes SHA
`85ac78483c9ad5eb32e41aa7a40d6b5c9d82e4676ae76b6d957681d26418c5a8`, local
`result/preparation/multibackbone_20261018/modelnet_original_matched_analysis_20261019T065632Z.zip`.
Drive same method root, same ZIP name and `.receipt.json`; actual upload/
download bytes, CRC and all manifest members match. Contains partial campaign
state, exact two independent analyzers, source/priorGSD indices and derived
macro/per-corruption triplets. This is complete ModelNet evidence within a
still-running six-job campaign. Current derived CSVs:
`result/preparation/multibackbone_20261018/extension_modelnet_triplets_final/`.

[Protocol/source/checkpoints/approval](gsd_multibackbone_3dd_extension_20261019.md).
[Earlier GSD-only evidence](gsd_multibackbone_guidance_results_20261018.md).

## Remaining work (updated after unexpected runtime loss)

[Run/Open] r7 disappeared from repeated server inventories around08:05UTC.
DGCNN ModelNet GSD is incomplete; Scan GSD and both originalDGCNN jobs did
not start. Preserve all four accepted CN/PN original results and prior source/
GSD rows. Recover interrupted-job Drive shards before designing a fresh
DGCNN attempt. Real RNG state/latents/geometry were not saved, so this is
completed-prediction recovery, not an exact mid-run resume. The full8-new-row,
182940-prediction and6-triplet gates remain open. No new GPU run launched.

## ScanObjectNN original3DD accepted evidence — 07:06 UTC

[Run/Artifact] All581 main_split objects x15,8715 predictions/head. Original
CurveNet55.2381% is+3.1670pp over own source; originalPointNeXt49.6845% is
+0.5967pp. GSD is0.2295/4.4750pp below original. CurveNet's small0.2295pp
one-seed ordering is not a seed-stable superiority claim. GSD corrects524/453
original errors but harms544/843 correct events (CurveNet/PointNeXt); improves
5/4 corruptions, worsens9/11 and ties1/0.

CurveNet GSD-minus-original largest gains Impulse+12.392pp, Uniform+1.033pp,
Gaussian+1.033pp; losses Background-4.819pp, Rotation-4.131pp, RBF-2.582pp.
PointNeXt gains Impulse+6.540pp, Gaussian+5.164pp, Uniform+1.033pp; losses
Density-11.876pp, Cutout-11.015pp, Rotation-8.778pp. Whole policy differs as
declared; no hyperparameters changed after seeing results.

Raw ZIPs under `result/scanobjectnn_c/3dd_original/`:

| Backbone / run ID | Bytes | SHA256 |
|---|---:|---|
| CurveNet / 20261019-065533_curvenet-seed0 | 7019419 | `641b55130a83e85a6e6f006b7229e72de3f67e60480177218679fc052efe46f5` |
| PointNeXt / 20261019-065534_pointnext-seed0 | 6967801 | `27344cbf4e23d8efedc6a79b8ab7fec9b5ada3fab331298de08a1c8330ee8e62` |

Drive uses the same dataset/method/run IDs with `final.zip` and receipts. Joint
Scan original job508.593s=8.477min, reconstruction391.108s shared, never
counted twice. Both original jobs combined2532.291s; new ledger after smoke
and both jobs2782.578s/14400s, with DGCNN currently active.

Independent combined original validation91470 predictions/1455 shared
geometry batches, labels/IDs/logits/argmax/counts, frozen states, strict
checkpoint/preprocessing/source matching, actual DDIM schedules and all ZIP
member hashes pass. The4 complete triplets bind12 raw source/original/GSD
ZIPs by their hashes, checkpoint identity and per-original-index labels.

Immutable combined matched analysis22758bytes SHA
`8f3be7464f78c12dcf4eb588019fa131ba41f9241980eda71a21d2b7ccd35d64`, local
`result/preparation/multibackbone_20261018/original_two_backbone_matched_analysis_20261019T070752Z.zip`.
Drive `MyDrive/thesis/result/gsd_multibackbone_study/scanobjectnn_c/3dd_original/`
with the same filename plus `.receipt.json`; actual roundtrip bytes, CRC and
all manifest members match. Derived CSVs:
`result/preparation/multibackbone_20261018/extension_original_both_datasets_triplets_final/`.

[Inference] The historical fixed GSD policy is below this original reference
in all4 currently complete matched cases, despite source improvements in3/4.
This does not isolate spectral effects, prove an optimum or generalize beyond
these fixed alternatives and seed0. DGCNN comparison remains pending.

## Unexpected r7 loss — 08:05 UTC

[Run/Code/Open] Last successful read-only snapshot08:03:20.885750UTC:
13 complete ModelNet corruptions and Occlusion batch6 (IDs420–490), at least
32574 prediction events/475 batch acknowledgements. The runner prints each
batch only after its Drive copy check; transition into Occlusion also implies
13 completed corruption checkpoints. These acknowledgements are not a current
physical Drive audit. The actual last persisted batch may be later. No full
DGCNN GSD accuracy is accepted, and no DGCNN original job ran.

The next file download timed out (request08:04:26.949426UTC,30s timeout);
repeated server inventories then returned no assignments. Backend reason is
unknown. No runtime stop/release was requested. At08:07:03UTC only stale local
CLI clientPID58698 was SIGTERM'd after absence was established; the sole local
monitorPID34633/exec-session97631 remains alive and reportsLOST. This local
client cleanup did not terminate an assigned runtime.

Local immutable loss evidence:
`result/preparation/multibackbone_20261018/r7_runtime_loss_20261019T080916Z.zip`,
62808bytes, SHA256
`68986b855cfc1e1d30633aa5ace7d84a81ab0e302ed02fd8a140d12aa6853b3c`.
CRC,16 safe members and all member hashes pass. It contains the last raw
log/state, completed-job ledger, source/previousGSD/accepted-original indices,
approval, exact executed controller/asset manifest and observed-loss report.
Its Drive backup and interrupted DGCNN shard recovery remain pending: browser
access is blocked by a saved site permission, and user has been asked to enable
it. Existing accepted raw/analysis bundles already have independent Drive/local
SHA verification and remain valid.

Last completed-job budget2782.578s excludes the interrupted approximately
61-minute DGCNN job. Do not report that stale ledger as total consumed time.
A fresh full DGCNN attempt would repeat completed reconstruction because actual
RNG state is absent. Reassess the whole remaining phase and existing4h cap
before launching; completed CN/PN/source/GSD work must not be repeated.

## r8 recovery route authorized — 08:33 UTC

[User decision/Run/Open] Browser tool's saved-block report disagrees with the
user's visible settings; user explicitly selects fresh Colab CLI/Drive as the
recovery route. A100r8 is allocated and Cell1/2 evidence locally archived;
official Google Drive consent is pending. No recovered DGCNN metric or new
GPU inference exists yet. This supersedes browser-unblocking as the immediate
recovery action, while all accepted CN/PN/source/GSD evidence stays unchanged.
[Exact session/monitor/consent handoff](gsd_multibackbone_recovery_20261019.md).
