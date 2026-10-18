# Two-backbone guidance results — 2026-10-18 UTC / 2026-10-19 local

[Run] ModelNet40-C full15 complete; ScanObjectNN-C is running on the same r7.
These are **GSD-inspired latent spectral guidance + <backbone>**, using published
alternative checkpoints. Exact GSDTTA checkpoint/protocol equivalence remains
unverified. DGCNN is parked by explicit user decision. No new test-label tuning.

## ModelNet40-C accepted full15

Seed0,2468 objects x15 corruptions,37020 predictions per head, batch70.
Fixed legacy M800/beta2/spectral1.17/SCD1, normal10/Background30, raw/eval LION,
updated decoder, final DDIM alpha-one=False. Classifier-facing preprocessing
is each accepted checkpoint's own pinned source contract. Source-only B32/
benchmark=True versus guidance B70/benchmark=False is explicitly recorded;
there is no bitwise common-draw or pure inference timing claim.

| Backbone | Matched source (%) | Guidance (%) | Delta (pp) |
|---|---:|---:|---:|
| CurveNet | 64.0113 | 65.1864 | +1.1750 |
| PointNeXt | 65.8320 | 66.4722 | +0.6402 |

[Artifact] Independent verification covers74040 indexed predictions and540
shared geometry batches: full CRC/safe paths/all member hashes, pinned labels/
0:N IDs, finite logits/argmax/counts, exact batch-shard concatenation, identical
shared geometry/RNG records across heads, restored per-head RNG, strict weights
and unchanged full classifier/LION state hashes. Source baseline ZIP hashes and
per-object labels/IDs match before computing corrected/harmed transitions.
Both final ZIPs were Drive-verified before Scan started. Joint ModelNet elapsed
4165.243s=69.421min; reconstruction3934.254s is shared across the heads.

## Heterogeneity; descriptive only

Both heads improve6/15 corruptions and worsen9/15. CurveNet corrects3909 but
harms3474 classification events (net435); PointNeXt corrects4039/harms3802
(net237). These counts repeat the same objects under15 corruptions; they are
not independent samples for a naive confidence interval. One seed establishes
neither seed stability nor isolated spectral causality. The selected historical
setting is not demonstrated optimal for either new backbone.

CurveNet gains concentrate in Background+45.016pp, Impulse+24.190pp and
LiDAR+10.211pp; largest losses are Density-12.399pp, DensityIncrease-12.318pp
and Cutout-10.697pp. PointNeXt gains most on Uniform+30.916pp, Gaussian+20.867pp
and LiDAR+14.911pp, but loses Background-24.797pp, Density-12.156pp and
Shear-9.522pp. Do not select corruption-specific settings from these test results.

| Corruption | CN source | CN guidance | CN delta pp | PN source | PN guidance | PN delta pp |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 67.30 | 71.88 | +4.58 | 45.26 | 76.18 | +30.92 |
| gaussian | 77.07 | 71.72 | -5.35 | 54.94 | 75.81 | +20.87 |
| background | 7.58 | 52.59 | +45.02 | 85.21 | 60.41 | -24.80 |
| impulse | 47.65 | 71.84 | +24.19 | 72.37 | 80.11 | +7.74 |
| upsampling | 70.10 | 74.92 | +4.82 | 71.27 | 81.36 | +10.09 |
| distortion_rbf | 78.61 | 72.24 | -6.36 | 70.99 | 68.56 | -2.43 |
| distortion_rbf_inv | 80.51 | 73.46 | -7.05 | 74.35 | 69.98 | -4.38 |
| density | 79.13 | 66.73 | -12.40 | 85.62 | 73.46 | -12.16 |
| density_inc | 88.17 | 75.85 | -12.32 | 91.21 | 83.47 | -7.74 |
| shear | 76.99 | 70.75 | -6.24 | 78.81 | 69.29 | -9.52 |
| rotation | 68.84 | 62.07 | -6.77 | 39.06 | 40.07 | +1.01 |
| cutout | 78.57 | 67.87 | -10.70 | 85.74 | 76.70 | -9.04 |
| distortion | 77.59 | 73.30 | -4.29 | 73.95 | 69.04 | -4.90 |
| occlusion | 35.62 | 35.90 | +0.28 | 39.47 | 38.49 | -0.97 |
| lidar | 26.46 | 36.67 | +10.21 | 19.25 | 34.16 | +14.91 |

## Immutable evidence

| Backbone | Canonical local raw ZIP | SHA256 |
|---|---|---|
| CurveNet | `result/modelnet40_c/gsd_inspired_latent_spectral_guidance/20261018-205338_curvenet-seed0.zip` | `740a1d4b9c77aa658218d3df651648e1aa6694bfbf5292cafc3e46335b3b3c8f` |
| PointNeXt | `result/modelnet40_c/gsd_inspired_latent_spectral_guidance/20261018-205339_pointnext-seed0.zip` | `fd59c5242980bf28bd6f9bd87d8b98919e7076a9d1b263ea3e47719481eedc8a` |

Drive raw root:
`MyDrive/thesis/result/gsd_multibackbone_study/modelnet40_c/gsd_inspired_latent_spectral_guidance/<run_id>/final.zip`.
Each model also retains15 corruption ZIP receipts and540 batch receipts.
Derived analysis ZIP8837bytes SHA
`77ff619a008b881702358b83e4fbe54e6518bf3904cbdb17c13a6500389f1eff`, local
`result/preparation/multibackbone_20261018/modelnet_matched_analysis_20261018T220442Z.zip`.
It was uploaded to the same Drive method root and downloaded to a new path;
exact bytes/SHA, CRC and every analysis member match. Matching receipt is beside
that ZIP on Drive. The analysis keeps the exact independent validator source,
partial campaign state and source baseline index. Raw inference ZIPs are unchanged.

## Next gate

[Run/running] ScanObjectNN-C581 main_split full15 follows automatically in the
same approved phase; run IDs220304-CurveNet/220305-PointNeXt. Finish, retrieve
and independently verify its two results; then validate all1350 batch receipts
and60 corruption receipts, archive the complete analysis, and update this note.
Original3DD-TTA and same-flow spectral0 controls remain proposed subsequent work.
[Protocol and operational log](gsd_multibackbone_guidance_20261018.md).
