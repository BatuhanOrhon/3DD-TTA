# Agent B frozen component confirmation — 2026-10-25

[Run/Inference] The frozen Chamfer + probability mean candidate is a supported
lower-cost option at the tested GSD settings. Its reserved-ID three-seed balanced
accuracy is56.8929%, versus56.7222% for C+S with the same readout. The primary
difference is+0.1706pp with conditional95% object-bootstrap interval
[-0.0397,+0.3651]pp. This does not establish superior or equivalent accuracy.
No tuning or policy reselection occurred after the development screen.

## Frozen scope and acceptance

The [development screen](guidance_component_development_results_20261025.md)
compared C+S/C/S/U on70 development objects/dataset, seed0. Before confirmation,
the [plan](../result/preparation/guidance_component_study_20261025/frozen_confirmation_plan.json)
SHA256`c6ea3e01a03061afd227c82eb30eb128c2a4f1f342dbf937732dd5e61b122d45`
froze C probability mean, primary same-readout C+S, and secondary C+S minimum
entropy. Both arms received zero parameter trials. Other saved readouts remain
diagnostics; higher individual heldout cells do not change the selected policy.

[Run] Confirmation covers ModelNet40-C40 classes and ScanObjectNN-C15 classes
main_split;70 reserved original objects/dataset, all15 severity5 corruptions,
DGCNN/CurveNet/PointNeXt/Point-MAE, seeds0/1/2. These objects are disjoint from
development across every corruption, but prior benchmark exposure prevents an
external unseen-test claim. The same70 objects recur across all three seeds.

Twelve dataset/arm/seed jobs yield48 immutable head ZIPs,144 case/policy rows,
2160 corruption rows,50400 terminal head-input events and201600 candidate
predictions. Independent local analysis rechecks exact source/native/checkpoint
identities, dataset classes/labels/IDs, frozen state and probe/RNG purity,
inactive computations, seed-specific preparation/noise pairing, readouts and
physical archive SHA/CRC. All90 exact cached-state NPZs were downloaded from
Drive, rehashed and joined to the raw phase's cache records:30 per seed.

## Accuracy under the predeclared comparisons

Each row averages all15 corruptions within each of eight dataset/backbone cases,
then gives the eight cases equal weight. Units are accuracy percent or percentage
points, not relative percent gains.

| Seed | Frozen C probability mean % | C+S probability mean % | Primary C−C+S pp | Frozen C+S minimum entropy % | Secondary pp |
|---|---:|---:|---:|---:|---:|
|0|57.0119|56.8333|+0.1786|56.6548|+0.3571|
|1|56.5714|56.4643|+0.1071|56.6310|−0.0595|
|2|57.0952|56.8690|+0.2262|56.8095|+0.2857|
|Three-seed mean|56.8929|56.7222|+0.1706|56.6984|+0.1944|

[Run/Inference] The primary difference is positive in all three seeds, range
+0.1071 to+0.2262pp. Averaging seeds, six cases improve and two worsen. Across
the25200 paired prediction events,436 are corrected and393 harmed. The frozen
secondary contrast averages+0.1944pp, interval[-0.0714,+0.4683]pp, with seven
positive/one negative case; its seed1 difference is negative. Both all15
intervals include zero. Three seeds provide limited replication, not broad
seed stability, noninferiority or equivalence evidence.

| Dataset / head | C probability mean % | C+S probability mean % | Three-seed difference pp |
|---|---:|---:|---:|
|ModelNet / DGCNN|66.9841|66.7302|+0.2540|
|ModelNet / CurveNet|69.2381|68.6667|+0.5714|
|ModelNet / PointNeXt|70.4762|70.3492|+0.1270|
|ModelNet / Point-MAE|64.2222|63.8095|+0.4127|
|Scan / DGCNN|41.1746|41.1429|+0.0317|
|Scan / CurveNet|58.3810|58.5714|−0.1905|
|Scan / PointNeXt|47.5238|47.6508|−0.1270|
|Scan / Point-MAE|37.1429|36.8571|+0.2857|

C's own terminal averages55.1984%; the frozen probability readout adds1.6944pp.
C+S terminal averages55.1111%, probability mean56.7222% and entropy56.6984%.
These are observed trajectories with three probes plus terminal, not early
stopping or a comparison with a terminal-only deployment without observer cost.

[Run] Supplemental macro14 excludes Background: C probability mean57.9124%,
C+S probability mean57.6956%, difference+0.2168pp, conditional interval
[+0.0126,+0.4082]pp. Against frozen C+S entropy57.7296%, the secondary difference
is+0.1828pp, interval[-0.0723,+0.4379]pp. This diagnostic does not replace the
predeclared all15 primary endpoint or justify policy reselection.

Uncertainty uses2000 dataset-stratified original-object cluster draws, seed
20261025. Each dataset's70 draws are reused across all15 corruptions, four heads,
two arms, policies and all three seeds; datasets are resampled independently.
There are140 object clusters, not420 independent objects. Intervals are
conditional on these datasets/objects/seeds and are not multiplicity adjusted.

## Cost, scope and decision

Fresh-encode Gaussian/Background smoke on the same A10080GB measured shared
reconstruction totals52.3376s C+S versus39.2842s C across the two datasets:
24.94% lower for C, with the same B70, probes and four heads. Complete smoke
job+validation totals are132.2487s versus119.2058s,9.86% lower. No explicit
warmup sample was discarded. These measurements include instrumentation and
do not establish standalone-head or B1 deployment latency.

Confirmation mean per dataset/seed job: shared reconstruction151.3127s C+S
versus98.0025s C; complete job+validation232.3374s versus178.6731s. C+S creates
the fresh cache and C replays it, so those differences are comparison costs,
not measured deployment speedups. Peak allocations are about35.5GiB C+S versus
34.4GiB C with all four heads loaded. Shared reconstruction is counted once;
head times are summed and peak memory is not multiplied by four. The
development report includes graph/encode/prior/backward/probe dispatch spans,
explicitly labeled unsynchronized rather than additive CUDA event timings.

[Decision/Inference] Retain C probability mean as the supported fixed-setting
accuracy/cost candidate. C+S offers no established accuracy advantage here;
spectral-only and unguided remain seed0 development evidence. Spectral's
standalone update norms were much smaller than Chamfer's at the transferred
coefficient1.17, so this is not a calibrated spectral-method comparison or a
universal claim that spectral guidance is unnecessary. Original3DD is a
different host and is not synonymous with this Chamfer-only arm.

The whole three-seed phase consumed2505.1130s/41.7519min, below its3300s cap;
expected41.10min,20% margin plus300s reserve54.32min. The completed initial
smoke+development phase separately consumed34.8947min. Setup, CPU analysis,
transfers, idle time and total VM billing lifetime are separate from these
active-phase durations. No greater-than60min research GPU phase was launched.

## Recoverable evidence and remaining work

Root:`result/preparation/guidance_component_study_20261025/`.

- `local_frozen_confirmation_analysis/validation.json` and CSVs retain all
  all15/macro14 rows, paired case/corruption results, seed ranges and cost.
- Raw phase Drive/local ZIP607716167bytes SHA256
  `f3f0171c5d06bb88c95f6bd55b91b2097cce2337e5bdefa23022335d282b9070`.
- Three-seed cache Drive/local ZIP387106113bytes SHA256
  `6710a4e22a05daa64602cb13c2c559b789bfd2a200b1306ad2d7ddfb154e9569`.
- Source manifest SHA256
  `b95170f32c73967dd97b55057e57e9107d037b0ca8b8bd7a2fdcd39ed2337229`
  binds archived reviewed helpers; project source remains sealed62e9a2b with
  manifest7012c30a...f625ed75 and source ZIPf5bca11a...576d0c6.
- Canonical48 raw ZIP copies remain under own dataset/method result roots.
  Runtime/Drive paths and per-file cache hashes are in adjacent receipts.

[Open] Full2468/581-object evaluation, calibrated spectral/parameter development,
additional independent datasets/seeds, fine-grained synchronized CUDA profiling,
standalone deployment latency and matched70-ID source/original3DD deltas remain
unmeasured. Historical full-dataset source rows do not supply those exact subset
deltas; the previously missing matched Point-MAE Scan reference remains missing.
This bounded study does not establish an optimal or general thesis contribution.

Own session `tta-guidance-components-a100-20261025-r1` and its foreground monitor
remain live. No runtime teardown is authorized; Agent A resources and original
workspace user files were preserved.
