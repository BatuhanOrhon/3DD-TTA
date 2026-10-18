# Multibackbone source-only results — 2026-10-18

[Run] Six complete seed0/severity5 all15 rows:137205 indexed predictions.
ModelNet40-C2468 objects; ScanObjectNN-C581-example `main_split`.
Batch32,1024 classifier points, fixed published alternative weights and
predeclared backbone-specific preprocessing. No classifier adaptation or LION.

| Dataset | Backbone | Measured source (%) | GSDTTA paper (%) | Delta (pp) |
|---|---|---:|---:|---:|
| modelnet40_c | dgcnn | 58.7979 | 66.51 | -7.7121 |
| modelnet40_c | curvenet | 64.0113 | 71.38 | -7.3687 |
| modelnet40_c | pointnext | 65.8320 | 66.99 | -1.1580 |
| scanobjectnn_c | dgcnn | 32.7826 | 55.72 | -22.9374 |
| scanobjectnn_c | curvenet | 52.0711 | 53.06 | -0.9889 |
| scanobjectnn_c | pointnext | 49.0878 | 50.66 | -1.5722 |

[Paper] Comparison uses printed macros in GSDTTA Tables1–2 (main PDF pp6–7).
[Official paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf).
Rounded per-corruption row means differ slightly from printed macros; use
printed macro targets and corruption-name mappings, never column-order guesses.

## Verification

[Artifact/Run] Every archive passes remote/Drive/local SHA, byte size, safe
paths, CRC and all member manifests. Per-object IDs equal0:N, labels match
pinned original arrays exactly, logits are finite Nx40/15, predictions equal
argmax and counts independently reproduce every CSV row and macro. All
classifiers have identical full state hashes before/after inference; strict
loads have zero missing/unexpected keys. All15 means equal-weight corruption
macro; all files have equal N, so micro happens to match. No seed variance or
causal paper-checkpoint equivalence is established by this single seed.

Source manifestSHA:
`bf964525b6d3e1d747f7d6ba959c02bd7f7153b0b9d79725ba08007b049cb7f8`.
Source ZIP SHA:
`ec136350e58506e700fb30e35c3e33a39fcdc3cc2cabfc542d54f9f28e039183`.

## Discrepancy audit before guidance

[Run] Largest negative per-corruption deltas for each fixed alternative:

| Dataset/backbone | Three largest losses versus printed paper row (pp) |
|---|---|
| modelnet40_c/dgcnn | background -36.14, density -21.35, cutout -18.27 |
| modelnet40_c/curvenet | uniform -20.83, upsampling -19.00, impulse -17.91 |
| modelnet40_c/pointnext | uniform -23.86, lidar -8.71, upsampling -5.76 |
| scanobjectnn_c/dgcnn | impulse -43.20, distortion_rbf_inv -35.80, distortion_rbf -34.94 |
| scanobjectnn_c/curvenet | background -10.16, impulse -9.12, shear -7.75 |
| scanobjectnn_c/pointnext | distortion -12.91, impulse -11.71, distortion_rbf_inv -11.36 |

[Code] Checkpoint IDs and source revisions were chosen by published provenance,
class count and strict compatibility before corrupted accuracy was measured.
SVWA ModelNet pretrain and TTA YAMLs at pinned59935d6 confirm k20,
emb_dims1024, dropout.5 and1024 FPS. Source eval directly dispatches raw FPS
points to DGCNN/CurveNet. The reversed DGCNN head-key selector is explicitly
recorded and changes only which identical-shaped head attribute is called.
There is no evidence that the measured DGCNN gap comes from a k/config typo.

[Code/Open] Exact GSDTTA model variants, checkpoint bytes, corruption bytes and
FPS/normalization policy remain unidentified. Published alternatives are a
different experimental asset. Scan PointNeXt training is PB_T50_RS; the
alternative Scan CurveNet trainer is OBJ_ONLY with random sampling and
L-infinity normalization. Its classes0–14 match official ScanObjectNN semantic
indices. Exact SVWA Scan training-log linkage is not available. The selected
released Scan data are an independently verified1024-point subsample base of
the581 official `main_split` objects, not proof of GSDTTA corruption seed/bytes.

[Inference] Unequal per-corruption effects (including some positive cells)
and three macro gaps below1.6pp do not support a simple global percentage/unit
or class-count error. They also do not establish a unique causal explanation
for the larger DGCNN/ModelNet CurveNet gaps. Known asset/training/protocol
differences limit comparison; do not claim that checkpoint identity alone
quantitatively explains the observed losses. Numerical closeness is not a
proof of identical paper weights.

## Clean source controls and eligibility decision

[Run] All six clean controls completed without changing weights or preprocessing.
Independent indexed-logit validation covers9147 predictions, severity0, seed0,
batch32, all2468/581 original objects, strict loads and frozen state hashes.

| Dataset | Backbone | Correct / N | Clean accuracy (%) |
|---|---|---:|---:|
| modelnet40_c | dgcnn | 2214 / 2468 | 89.7083 |
| modelnet40_c | curvenet | 2219 / 2468 | 89.9109 |
| modelnet40_c | pointnext | 2283 / 2468 | 92.5041 |
| scanobjectnn_c | dgcnn | 399 / 581 | 68.6747 |
| scanobjectnn_c | curvenet | 450 / 581 | 77.4527 |
| scanobjectnn_c | pointnext | 535 / 581 | 92.0826 |

[Inference] Scan DGCNN already has substantially lower clean accuracy than the
other two alternatives. Therefore its large corrupted gap must not be presented
as a guidance failure or a reproduced paper baseline. Correct label ordering,
class heads, full strict state loading, plausible clean performance and finite
outputs provide no evidence of a broken classifier boundary. Clean accuracy
does not isolate training split, model strength and corruption sensitivity.

[Decision] Source-only discrepancy audit is complete for the **published
alternative experiment**: known checkpoint/training/protocol and released-data
provenance differences are enumerated above; a unique quantitative cause of
the paper gaps remains unknown. Exact paper reproduction remains unresolved
and is not an acceptance claim. Retain all six fixed alternatives for the
already-approved exploratory matched-checkpoint guidance study, with Scan
DGCNN explicitly a weaker clean/source model. Do not tune these contracts to
the paper table, pool different checkpoint baselines, or call numerical
closeness reproduction. The relevant hypothesis is whether unchanged latent
spectral reconstruction changes each alternative's own measured source row.
This decision precedes GSD integration. A later source/config mismatch would
invalidate the affected matched pair and reopen this gate.

[Artifact] Clean raw ZIP paths/hashes and counts are indexed in
`result/preparation/multibackbone_20261018/clean_analysis_final/validation.json`;
all six immutable final ZIPs and original-condition checkpoints have matching
Drive/local SHA, safe paths, CRC and full member manifests. Clean asset manifest
SHA is `34db8218f6eeebaa1eee3c81deebfa6132449a066a0261fc9f71caaf272efeba`;
source ZIP SHA is `0b311b65f2270cabab31740341bdbd96d01d1d27aead345aae969c75d626b279`.
The source runner's only semantic addition is an explicitly singleton clean
control; all15 defaults and classifier math remain unchanged. Eight strict CPU
loads passed again with CUDA uninitialized; CPU ZIP SHA
`317af60a94be4562fce7e499e402552e4bd7cd42a552eb9b9270ad8a3011a8c8`
is verified locally and on Drive.

## Raw artifact identities

| Pair | ZIP SHA-256 |
|---|---|
| modelnet40_c/dgcnn | b08e296f7ef2c5828ea051263f6672334ca8a904f34acb604995dfc072ca3bdd |
| modelnet40_c/curvenet | 25736accdd0074535de3fcf0c5fc45c75ae51138a68b81b20ce10ae4e84d47eb |
| modelnet40_c/pointnext | ea88626ea8fe6dbe01d658e95c7282142160ccc328570c7d4bc7800dcf47a36a |
| scanobjectnn_c/dgcnn | 73af725a13e3be0f53f0f0b2658cc40e1e04b327c4d22bb014da1e2e9197f5b1 |
| scanobjectnn_c/curvenet | 67c7eaf13ccdff6659cb1ff32cfcf781933b68eadb7d8b75ef7973a736944e14 |
| scanobjectnn_c/pointnext | 51c50a53d8cc6d6f45e58f03599d8cb8c8feab6257fac40846eea1cf67d3123e |

Raw ZIPs: `result/<dataset>/source_only/<run_id>.zip`. Exact paths, config
contracts and counts are in
`result/preparation/multibackbone_20261018/source_analysis_final/validation.json`.
Derived90-row comparison: `per_corruption_comparison.csv` in the same directory.
Drive final archive roots:
`MyDrive/thesis/result/gsd_multibackbone_study/<dataset>/source_only/<run_id>/`.
Every corruption has a separate immutable checkpoint receipt in that root.
[Study and detailed provenance](gsd_multibackbone_study_20261018.md).
