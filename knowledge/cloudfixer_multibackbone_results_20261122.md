# ModelNet40-C: original CloudFixer, DGCNN / CurveNet / PointNeXt

**[Run/Complete] 2026-11-22 23:01 UTC: all15 GPU evaluation and independent raw recount complete.**
585blocks/135cells,threeheads/threeoutputmodes,37020 predictions per head/mode.
Final bundle Drive-to-host readback passed; publication/cloudflush/runtime closure are recorded below when completed.
[Execution and recovery](cloudfixer_multibackbone_preparation_20261122.md).

## Complete all15 accuracy and counts

[Run] All15/s5/seed0,2468 original objects per corruption;37020 predictions per head/output mode. Macro accuracy equals pooled accuracy because all cells retain2468 objects. CSV accuracy is a fraction; tables below are percentages or percentage points.

| Head | Raw source (%) | CF-frame source (%) | Plain CF (%) | Old LION N8 PM (%) | CF−LION (pp) | Net correct CF−LION |
|---|---:|---:|---:|---:|---:|---:|
| DGCNN | 58.835764 | 63.633171 | 71.277688 | 63.941113 | +7.336575 | +2716 |
| CurveNet | 64.011345 | 68.927607 | 75.931929 | 65.678012 | +10.253917 | +3796 |
| PointNeXt | 65.837385 | 67.314965 | 75.018909 | 70.626688 | +4.392220 | +1626 |

Plain CF correct counts: DGCNN 26387/37020, CurveNet 28110/37020, PointNeXt 27772/37020.

| Corruption | DGCNN CF / LION (%) | CurveNet CF / LION (%) | PointNeXt CF / LION (%) |
|---|---:|---:|---:|
| uniform | 88.1686 / 78.4846 | 88.4522 / 76.7018 | 87.9660 / 80.6321 |
| gaussian | 87.7634 / 78.4036 | 88.4927 / 76.3371 | 87.8444 / 80.6726 |
| background | 55.3485 / 15.8023 | 49.9190 / 14.5867 | 75.6078 / 70.2998 |
| impulse | 87.9254 / 76.8639 | 88.2091 / 75.4052 | 89.0194 / 83.6710 |
| upsampling | 88.4522 / 82.6985 | 89.3031 / 79.2950 | 88.2901 / 86.8720 |
| distortion_rbf | 82.0097 / 72.9741 | 84.8055 / 76.7018 | 79.9838 / 72.0421 |
| distortion_rbf_inv | 83.5089 / 74.7164 | 86.4668 / 78.4846 | 81.6856 / 75.0405 |
| density | 50.5267 / 66.1669 | 74.0276 / 69.8541 | 77.4311 / 79.6191 |
| density_inc | 83.9951 / 81.2399 | 86.3452 / 79.1734 | 85.4943 / 87.7634 |
| shear | 77.7147 / 72.0016 | 81.4019 / 74.1896 | 75.5673 / 73.7844 |
| rotation | 75.9724 / 53.9303 | 83.3063 / 66.4506 | 63.0065 / 41.8152 |
| cutout | 59.2788 / 65.7212 | 79.8217 / 69.1653 | 81.8071 / 80.6321 |
| distortion | 80.6321 / 73.0146 | 83.0227 / 75.7293 | 77.1475 / 72.5284 |
| occlusion | 36.2642 / 36.3047 | 37.4392 / 36.3857 | 38.5737 / 39.6272 |
| lidar | 31.6045 / 30.7942 | 37.9660 / 36.7099 | 35.8590 / 34.4003 |

| Head | CF versus old LION: win / tie / loss | CF versus raw source: win / tie / loss | CF versus frame source: win / tie / loss |
|---|---:|---:|---:|
| DGCNN | 12 / 0 / 3 | 14 / 0 / 1 | 11 / 1 / 3 |
| CurveNet | 15 / 0 / 0 | 13 / 0 / 2 | 8 / 0 / 7 |
| PointNeXt | 12 / 0 / 3 | 9 / 0 / 6 | 10 / 0 / 5 |

[Inference] All three heads have higher observed all15 CF accuracy than the accepted old LION PM and raw/frame source controls. This is not per-corruption dominance:

- DGCNN: density -15.640194pp (-386 net correct), cutout -6.442464pp (-159 net correct), occlusion -0.040519pp (-1 net correct).
- PointNeXt: density -2.188006pp (-54 net correct), density_inc -2.269044pp (-56 net correct), occlusion -1.053485pp (-26 net correct).

CurveNet exceeds old LION PM in all15 measured corruption cells; this remains one seed/one checkpoint per head, not a statistical dominance claim. Background PointNeXt rawsource85.251216% exceeds CF75.607780%; Density rawsource exceeds CF for all three heads. Shear CF-frame source exceeds CF for all three heads. These negative controls rule out a claim that either the CF transform or CF optimization is uniformly beneficial.

Rotation CF gains over old LION are +22.042139/+16.855754/+21.191248pp (DGCNN/CurveNet/PointNeXt). CF versus the same frame control adds324/195/504 net correct on Rotation. Background gains over old LION are +39.546191/+35.332253/+5.307942pp, while PointNeXt still loses238 net correct to rawsource. Prior-domain/preprocessing effects and the original rotation+delta package remain confounded relative to LION.

The newer CF-XYZ N8/K80/tau.1 late-PM thesis baseline is not measured on these three heads here; do not rename the old LION comparator as that baseline. These results do not establish whether PM alone helps with the CF denoiser. No test-label tuning, routing or policy switch is adopted.

All135 cells,45 CF−LION comparisons and9 overall cells are available in [raw-recount evidence](evidence/cloudfixer_multibackbone_20261122/a100_full_recount.json), [per-corruption CSV](evidence/cloudfixer_multibackbone_20261122/a100_per_corruption.csv), [summary CSV](evidence/cloudfixer_multibackbone_20261122/a100_summary.csv) and [original archive index](evidence/cloudfixer_multibackbone_20261122/a100_original_archive_index.json).

## Final execution and preservation status

[Run] Actual worker+supervisor elapsed 10826.007684s (3.007224h),20:00:42.921245–23:01:08.928929UTC. Both real returncodes0; workerfullflagtrue/15, independent deadline not reached. The full-time estimate at8.88CU/h is 26.704152CU; setup/allocation and finalization are additional, not exact billed CU.

Host final bundle `result/modelnet40_c/cloudfixer_multibackbone/3596befa489b/final_20261122T230335Z.zip`: 65528957 bytes/SHA256 `bcb64ea0c5c0165e43c4498dfc428af6c1e4804f82889e17d4924b76e38cfe07`. Contains15 unchanged raw ZIPs, complete derived seven-file report, exact executed v5 overlay/official source, canonical/operational source supplement, exact host analysis/assembly source and actual exit receipts. Final independent Drive-to-fresh-host size/SHA/CRC readback passed23:05:56UTC (41members); [bundle receipt](evidence/cloudfixer_multibackbone_20261122/a100_final_bundle_receipt.json). Official cloud flush, publication and ownedr16closure remain pending until their receipts confirm them.

## Question and fixed protocol

[User report/Code] The user requested original plain CloudFixer with source
models other than Point-MAE/Point2Vec, compared against our previously accepted
results. This experiment evaluates DGCNN, CurveNet and PointNeXt with exactly
the frozen ModelNet40 head checkpoints/states used in the accepted LION study.
No classifier training or test-label parameter/window selection is performed.

[Code/Run] ModelNet40-C test original IDs0–2467, all15 corruptions, severity5,
seed0, B64 including the final36 IDs;585 atomic prediction/RNG blocks. Each
corruption starts its own continuous seed0 native stream. Head classification
uses an isolated RNG stream and is checked not to shift the adaptation stream.
This is not the author's shuffled full-loader RNG stream or a common-random-
numbers comparison with LION. Every observed object remains in the denominator.

Official CloudFixer source is pinned to
`f043e24414fcf4165fdc892abff21215c9989141`. The prior is its ModelNet40 XYZ
diffusion checkpoint,159801477 bytes, SHA-256
`b64841d022e08b1a89f58dd9b16d9023e00be52fbca3dca735d5798362bb5a5a`.
Original rotation/delta optimization uses30 updates, Adamax, input_lr.2,
rotation_lr.02, t_min.02, t_len.1, configured lam_h10/lam_l1, weighted regularization, knn5,
plainCF/vote1. With updates indexed0–29 the applied regularization weight runs
from10 to1.3 (the configured lower endpoint1 is not reached). No LION, proximal
SCD or probability mean is used in this CF arm.
All heads share each corrected geometry; this is not three separately sampled
CloudFixer trajectories. Model/head state identities are checked before each
block commit. The manifest and exact executed source package are authoritative.

Scientific source `228555f`, v5 campaign
`3596befa489bc8f05cd5f139a976c7a00b7429d390a19fec850595814cd3f224`.
Later source commits update documentation or future packaging; they do not
replace the active v5 payload. A future-package guard pins the accepted LION
reference to94668 bytes/SHA
`199e52a5d826efaa39e159fcc98c441d95ac463951a6538dada3a450866bcc48`.
Repackaging from a later integration checkout creates a new campaign hash;
it must not be represented as the exact source identity of this run.

## Three measured outputs and the accepted comparator

[Code] Each original object has three outputs from the same frozen head:

1. `source_raw`: original dataset coordinates followed by the accepted head's
   native FPS1024 preprocessing; no CF coordinate conversion or correction.
2. `source_cf_frame`: author rotateTrue/unitSTD input followed by author
   unit-sphere/inverse-axis output conversion and native FPS1024, without CF
   optimization. The geometry-only round-trip was checked in the technical smoke.
3. `cloudfixer_plain`: the same CF coordinate pipeline with original author
   rotation/delta optimization, followed by the same output/head preprocessing.

Input arrays retain their actual corruption point counts up to2048; Background
has1075 and Upsampling2048. Both XYZ prior and preprocessing differ from the old
LION pipeline, which uses ShapeNet-all55 latent diffusion/decoder,2048-point
resampling, bbox/longest-half-edge normalization, scale3.3885 and its coordinate
conversion. Source-head checkpoints/classes/order remain matched.

[Run] The accepted old LION comparator is **proximalN8, tau.1, retained.95,
equal probability mean of candidates2/4/6/8 at T1**, DDIM100/final-alpha1,
original encoded decoder style, raw/eval/EMAoff, no Background horizon override.
Its source is `b651e259db9ec6341f39e606b01eef5254d90456`, B32, fresh worker seed0,
all2468×15/s5, same frozen three ModelNet40 heads. Its all15 accuracies are
DGCNN63.941113%, CurveNet65.678012%, PointNeXt70.626688%.

This comparator is separate from the newer **CloudFixer-XYZ-prior N8/K80/
tau.1, equalPM t20/t10/t0** thesis baseline. The present task does not measure
that newer method with these three heads. It therefore cannot prove that the
newer baseline is worse than plain CF on these heads, nor isolate PM's effect.
The accepted old LION N5 terminal reference also differs in horizon/policy;
N5-versus-N8 comparisons do not isolate readout alone.

## Detailed Background control interpretation

[Run] All15 corruption scopes completed2468 objects per corruption/head. Only complete raw scopes are accepted; no partial-corruption accuracy was used to choose settings.

Background has a particularly important control result:

| Head | Source raw (%) | CF-frame source (%) | Plain CF (%) | Old LION PM (%) |
|---|---:|---:|---:|---:|
| DGCNN | 13.614263 | 24.473258 | 55.348460 | 15.802269 |
| CurveNet | 7.576985 | 4.538088 | 49.918963 | 14.586710 |
| PointNeXt | 85.251216 | 49.392220 | 75.607780 | 70.299838 |

[Inference] CF exceeds old LION PM in all three Background cells. It does not
exceed source-only in every cell: PointNeXt loses238 net correct versus raw
source (−9.643436pp). The CF-frame transform loses885 net correct, and CF
optimization adds647 relative to that frame control. These are net count
changes, not separately counted object-level win/loss pairs. The transform
control changes several geometry operations together; it is not a single-factor
causal ablation. We do not switch policies per corruption after seeing labels.

## Validation, resources and persistence

[Run/Code] Canonical ten-cell preparation and actual CUDA/data/checkpoint
checks passed, with only the previously authorized cell2 fastrlock failure
deferred. Python3.8.20/Torch2.1.2+cu121/NumPy1.21.2, CUDA12.1/GCC11.4;
A10080GB/High-RAM (~167GiB), architecture8.0. Technical smoke, new-process
cached RNG replay, Gaussian64 profile and actual2048-point/B64 all-head capacity
checks passed before full inference. Gaussian64 native15.779777s/peak5089.01MiB;
Upsampling64 native41.639718s/peak18163.37MiB (=17.74GiB). Those are technical
measurements, not independent benchmark repeats or accuracy-based choices.

`denoiser_calls=30` in payloads records the configured author update count;
it is not a dynamically instrumented invocation count. Peak memory is for the
whole block. Per-head report times combine shared native reconstruction and
synchronized head preprocessing/forward; they are not isolated end-to-end
latencies. Shared reconstruction time must not be summed over three heads to
estimate campaign wall time.

The conservative full estimate was4.392751h; admission balance489.36CU and
owned observed rate8.88CU/h. The immutable full window ends2026-11-23T00:21:31UTC.
New allocation8h/80CU and global task12h/120CU bounds retain earlier L4 accounting;
old55CU is reserved allowance, not an assertion of billed cost. Full duration is recorded above; allocation-through-close duration and final balance are recorded in the closure receipt when available. Per64 Drive commits preserve
Python/NumPy/CPU/CUDA RNG and predictions. Each finished corruption is saved
with a SHA-checked ZIP, three head receipts and a completion cursor published last.
An independent host collector downloads each original ZIP, validates paths/CRC,
checks IDs/labels/logits/RNG chains, recounts all three output modes and rejects
mixed/overlapping/incomplete scopes.

Original corruption ZIPs contain six standard files plus raw data. Their bytes
are preserved; the actual primary observer stdout is added only to a separate
extraction, with a provenance receipt. Raw local root:
`result/modelnet40_c/cloudfixer_multibackbone/3596befa489b/`;
Drive root:
`MyDrive/thesis/result/cloudfixer_multibackbone_20261122/a100_highram_b64`.
The canonical/operational source archive has independent Drive→host SHA/CRC
readback proof; the v5 manifest/package receipts and native identities preserve
actual executed attribution. [Compact evidence](evidence/cloudfixer_multibackbone_20261122).

The earlier L4/B32 pilot (Uniform2468 +Gaussian608,97 blocks) remains separate.
GPU/batch changes affect random draws and native loss/optimization behavior;
those blocks are not mixed into the new B64 all15 campaign. The current user
explicitly authorizes closing only the owned A100r16 after completed results
are preserved. No peer or old-account runtime teardown is authorized or attempted.

## Limits for thesis claims

[Inference/Open] This is a one-seed, matched-head **whole-pipeline comparison**.
Prior training domain, representation, preprocessing, optimizer, batch size and
random stream differ from the old LION study. Higher descriptive accuracy does
not establish which component causes the difference or multi-seed robustness.
The scalar accepted LION reference cannot support paired object-level CF-only/
LION-only counts, paired confidence intervals or paired significance; its raw
prediction vectors have not been retrieved for this task. No fulltest tuning,
new source-model training or ScanObjectNN GPU experiment is included here.
