# Matched source, original 3DD-TTA and GSD comparison — 2026-10-19

[Run] Original3DD CurveNet+PointNeXt ModelNet full15 independently accepted.
Scan original3DD is running; DGCNN GSD and original3DD follow. Existing four
CurveNet/PointNeXt GSD rows are reused, not rerun. All are published-alternative
checkpoints, not established paper reproductions. Scan uses581 main_split.

| Dataset | Backbone | Source % | Original3DD % | GSD-inspired % | GSD minus original pp |
|---|---|---:|---:|---:|---:|
| ModelNet40-C | CurveNet | 64.0113 | 67.0719 | 65.1864 | -1.8855 |
| ModelNet40-C | PointNeXt | 65.8320 | 68.2928 | 66.4722 | -1.8206 |
| ModelNet40-C | DGCNN | 58.7979 | pending | pending | pending |
| ScanObjectNN-C | CurveNet | 52.0711 | running | 55.0086 | pending |
| ScanObjectNN-C | PointNeXt | 49.0878 | running | 45.2094 | pending |
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

## Remaining work

[Run/running] Protectedr7 remains live. Scan original CN/PN run IDs
20261019-065533_curvenet-seed0 /20261019-065534_pointnext-seed0; after both
final ZIPs verify, the same approved controller advances to DGCNN GSD and
original3DD. Finish8new model rows, then sealed-member Drive verification of
5040 batch/120corruption receipts and full6-triplet comparison. No tuning.
