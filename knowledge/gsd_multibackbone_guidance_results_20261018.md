# Two-backbone guidance results — 2026-10-18 UTC / 2026-10-19 local

**2026-10-19 20:44UTC superseding status [Run/Inference/Open]:** this note
preserves the original two-head GSD attempts. Original3DD and subsequent DGCNN
GSD/original cases are now complete; DGCNN is no longer parked. All runtimes
and task workers are closed. The later Scan trajectory experiment has distinct
CN/PN terminals and must not overwrite the values below. See the
[final synthesis](gsd_multibackbone_final_findings_20261019.md) for18 method
values,12 readout values, updated comparisons and immutable evidence. The
same-flow spectral0/SCD1 arm and additional seeds remain unexecuted; the
historical next-study/live-r7 statements below are superseded accordingly.

[Run] Both datasets full15 complete and independently verified on protected r7.
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



## ScanObjectNN-C accepted full15

[Run] All581 main_split objects, severity5,15 corruptions,8715 predictions per
head. Same fixed seed0/B70 policy and revision6 snapshot as ModelNet; no
parameters changed after observing results.

| Backbone | Matched source (%) | Guidance (%) | Delta (pp) | Improved / worse / tied corruptions |
|---|---:|---:|---:|---|
| CurveNet | 52.0711 | 55.0086 | +2.9375 | 7 / 8 / 0 |
| PointNeXt | 49.0878 | 45.2094 | -3.8784 | 5 / 9 / 1 |

CurveNet corrects739 and harms483 events (net256); PointNeXt corrects751 and
harms1089 (net-338). CurveNet gains most on Impulse+20.998pp and
Background+17.728pp, with Rotation-4.991pp its largest loss. PointNeXt gains
Gaussian+27.194pp but loses Rotation-16.007pp and several structural
corruptions by roughly12–15pp. This is evidence against a universal backbone
benefit from the fixed historical guidance policy. The positive CurveNet delta
is descriptive for one seed; neither spectral causality nor an optimum is shown.

| Corruption | CN source | CN guidance | CN delta pp | PN source | PN guidance | PN delta pp |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 55.08 | 62.99 | +7.92 | 29.78 | 41.48 | +11.70 |
| gaussian | 51.29 | 58.86 | +7.57 | 34.60 | 61.79 | +27.19 |
| background | 14.80 | 32.53 | +17.73 | 43.37 | 40.79 | -2.58 |
| impulse | 31.50 | 52.50 | +21.00 | 35.11 | 46.99 | +11.88 |
| upsampling | 60.59 | 66.09 | +5.51 | 34.60 | 42.86 | +8.26 |
| distortion_rbf | 67.64 | 65.92 | -1.72 | 58.00 | 43.89 | -14.11 |
| distortion_rbf_inv | 68.16 | 66.44 | -1.72 | 58.00 | 43.03 | -14.97 |
| density | 68.67 | 67.30 | -1.38 | 84.68 | 70.22 | -14.46 |
| density_inc | 76.76 | 74.35 | -2.41 | 89.67 | 77.80 | -11.88 |
| shear | 66.78 | 65.06 | -1.72 | 61.27 | 46.64 | -14.63 |
| rotation | 64.03 | 59.04 | -4.99 | 49.05 | 33.05 | -16.01 |
| cutout | 69.54 | 69.71 | +0.17 | 85.89 | 70.91 | -14.97 |
| distortion | 69.02 | 68.16 | -0.86 | 58.52 | 44.06 | -14.46 |
| occlusion | 9.29 | 7.92 | -1.38 | 7.57 | 8.43 | +0.86 |
| lidar | 7.92 | 8.26 | +0.34 | 6.20 | 6.20 | +0.00 |

Scan raw ZIPs under `result/scanobjectnn_c/gsd_inspired_latent_spectral_guidance/`:

| Backbone / run ID | Bytes | SHA256 |
|---|---:|---|
| CurveNet / 20261018-220304_curvenet-seed0 | 6684020 | `8699efd791d5be265ee555b905ef60dbf78fecaa9aeb0f3b78bf904f23f7e4d7` |
| PointNeXt / 20261018-220305_pointnext-seed0 | 6633531 | `f4cbc882b326bfa5f6e87fceb9bd0a8d53b517db027cf12c7285632f5d705e39` |

Drive raw root is `MyDrive/thesis/result/gsd_multibackbone_study/scanobjectnn_c/gsd_inspired_latent_spectral_guidance/<run_id>/final.zip`.
Each Scan head also has135 batch receipts and15 corruption ZIP receipts.

## Complete campaign verification and recovery limits

[Artifact] All91470 predictions,675 shared geometry batches,4 final archives,
1350 batch receipts and60 corruption receipts independently pass. Batch files
match local and Drive bytes; corruption ZIPs are physically rehashed on Drive.
Full member identities, nested analysis, exact executed verification source,
CRC, labels/IDs, logits, counts, source links, native guards and unchanged
classifier/LION states pass.

Immutable final evidence: local
`result/preparation/multibackbone_20261018/guidance_all15_evidence_v1.zip`,
183296 bytes, SHA `ef34d122fcf38f36a9a85a497f3aee0fc1584d1876f797167f7ff310574f90a3`.
Actual Drive download verified from
`MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261018T222327Z/guidance_all15_evidence_20261018T222327Z.zip`,
with adjacent `receipt.json` and `.complete`. Nested complete-analysis ZIP is
45803 bytes, SHA `ea3133e018bb4ae1860f773cac4b99aebcf2618e936d82e95e8c7d0d91b4d692`.
It preserves60-row matched comparisons,4-row macro table, final phase state,
exact independent analyzer and source validation. Current derived directory:
`result/preparation/multibackbone_20261018/guidance_all15_analysis_final/`.

Scan joint job elapsed1004.737s=16.746min; full guidance phase5169.979s=86.166min.
Whole approved campaign GPU-job wall budget consumed5815.594s=96.927min of
28800s, including initialization/IO and failed first parity attempt. This is
job wall time, not accelerator kernel time, runtime lifetime or a billing total.
Shared reconstruction must not be counted twice between heads.

Batch backups contain indexed labels/predictions/all-class logits, loss and
timing diagnostics, geometry/RNG hashes, config/source/environment/logs and
SHA/byte receipts. Raw decoded coordinates, latents and actual RNG states are
not saved. Weights are frozen and identified by checkpoint hashes; result ZIPs
do not duplicate classifier weights. The artifacts preserve completed work;
they do not implement automatic resumption or exact random-stream continuation.

## Next study and remaining limits

[Decision/Open] Keep DGCNN parked. Original3DD-TTA with the same accepted heads
is the proposed next comparison, with a separate same-flow spectral0/SCD1 arm
needed for isolated spectral attribution. Neither was launched in this task.
No seed replication, new optimum or full GSDTTA reproduction is claimed.
Protected session `tta-scanobjectnn-nobg-a100-r7-20261018` and its sole monitor
remain live; task completion does not authorize runtime shutdown.
[Protocol and operational log](gsd_multibackbone_guidance_20261018.md).
