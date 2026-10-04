# Unguided and smooth-only results - ingested 2026-10-04

[Run/Code/Inference] Six complete ModelNet40-C severity-5 runs from October 3
were supplied on October 4, together with both launcher summaries. Ingestion
context: `gsd-smooth-spectrum@f04bfbdb24d5e9617602b50f1711f835861b6eb9`.
This note supersedes the pending-output status in the
[handoff](gsd_guidance_ablation_handoff_20261003.md).

## Evidence and verification

Raw inputs retain the user's supplied paths, without moving, duplicating or
rewriting them:

- `result/modelnet40_c/gsd-guidance-ablations/unguided/20260928-113047_gsd-cal-diagnose-reference-seed0-n64/`
- `result/modelnet40_c/gsd-guidance-ablations/smooth_only/20260928-113047_gsd-cal-diagnose-reference-seed0-n64/`

Each contains three seed ZIPs and `guidance_ablation_summary_<arm>_<reference>.json`.
Exact filenames, SHA-256 hashes and Git revisions for all 12 comparison inputs
and the two supplied summaries are in the
[derived analysis JSON](../result/modelnet40_c/gsd-guidance-ablations/analysis_20261004/analysis.json).
Additional derived files are
[per-corruption results](../result/modelnet40_c/gsd-guidance-ablations/analysis_20261004/per_corruption.csv)
and [prediction transitions](../result/modelnet40_c/gsd-guidance-ablations/analysis_20261004/prediction_transitions.csv).

[Run/Verification] All six new ZIPs pass `read_full_run(..., guidance_ablation=True)`;
the six reused SCD-only/same-beta combined ZIPs also pass the strict reader.
Checks include seven safe unique members, CRC, complete status, canonical
15-row scope, 2,468 examples per row, seed/config/CLI identity, calibration
identity, prediction/count agreement, and 37,020 examples per run. Independent
checks reconcile stdout result dictionaries with every CSV row, model modes,
checkpoint loading and recorded source hashes against Git blobs. All relevant
modules are frozen/eval and checkpoint missing/unexpected-key lists are empty.

Both supplied summaries match recomputation. Only machine directory portions
of run paths differ; filenames are checked. Maximum numeric differences are
1.110223e-16 (unguided) and 5.551115e-17 (smooth-only).
All raw input hashes remain unchanged after analysis. Local checks use archive
manifests; actual Colab dataset/checkpoint bytes were not rehashed locally.

## Fixed protocol

- All 15 severity-5 files, 2,468 examples each, seeds 0/1/2, batch32.
- Raw/eval LION, no EMA, frozen Point-MAE, original final decoder style.
- Gamma=eta=.01, lambda=.95, 100 scheduled DDIM steps; actual reverse
  execution uses five steps normally and **35 for Background**.
- Unguided: SCD=0, spectral=0, retaining encoding/noising/DDIM/decoding.
- Smooth-only: SCD=0, beta=.5, alpha=8.140161356429882. Rho=.001 identifies
  the archived SCD calibration; it is not an active ratio with SCD disabled.
- Reused combined arm: SCD=1 with the identical beta/alpha.

[Run] For every new Background run, diagnostics record 2,730 batch-step
observations (78 batches x35), step indices0-34 and diffusion times340-0.
Other corruptions record 390 observations each (78 x5). Unguided local/style
guidance update norms are exactly zero and graph records are absent.
Smooth-only records 2,468 graph observations per corruption, SCD weight0,
the fixed spectral coefficient and nonzero applied spectral updates.
The low Background scores cannot be explained by accidentally using five steps.

## Main results

Values are accuracy percentages; +/- is **sample SD across three seeds**,
not a confidence interval. Old and new groups have provenance differences
described below, so cross-group contrasts remain descriptive.

| Condition | All15 mean +/- SD | Background mean +/- SD | Other14 mean +/- SD | Mean recorded runtime / seed |
|---|---:|---:|---:|---:|
| A: unguided | 61.2021 +/- 0.1274 | 23.6224 +/- 0.3714 | 63.8863 +/- 0.1127 | 16.96 min |
| B: archived SCD-only | 63.8799 +/- 0.1326 | 60.6699 +/- 0.1999 | 64.1092 +/- 0.1534 | 29.60 min |
| C: smooth-only | 61.3596 +/- 0.0721 | 23.8790 +/- 0.2476 | 64.0368 +/- 0.0949 | 75.64 min |
| D: archived SCD + same smooth | 63.8520 +/- 0.0353 | 60.9130 +/- 0.8541 | 64.0619 +/- 0.0986 | 76.62 min |

Recorded peak GPU memory means: A11,686.34 MB, B14,597.91 MB,
C21,008.48 MB, D21,008.99 MB. Runtime is the sum of recorded corruption
evaluation times, not complete Colab setup/checkpoint-loading time. These
observations come from the reported A100 runs, not a separately controlled
performance benchmark. C/A observed runtime ratio is about4.46.

| Seed | Unguided (%) | Smooth-only (%) | C-A (pp) | Net extra correct / 37,020 |
|---|---:|---:|---:|---:|
| 0 | 61.1696 | 61.2831 | +0.1135 | +42 |
| 1 | 61.3425 | 61.4263 | +0.0837 | +31 |
| 2 | 61.0940 | 61.3695 | +0.2755 | +102 |

| Contrast | Mean delta +/- seed SD (pp) | Positive seeds |
|---|---:|---:|
| C-A: smooth without SCD | +0.1576 +/- 0.1032 | 3/3 |
| B-A: historical SCD vs unguided | +2.6778 +/- 0.2138 | 3/3 |
| D-B: smooth added to SCD | -0.0279 +/- 0.1048 | 1/3 |
| C-B: smooth replacing SCD | -2.5203 +/- 0.1106 | 0/3 |
| D-C: historical combined vs smooth-only | +2.4923 +/- 0.0801 | 3/3 |

The descriptive interaction (D-B)-(C-A) is -0.1855 +/-0.1919 pp.
Because this mixes runtime/build blocks, it does not establish a causal
interaction, redundant information or destructive gradient conflict.

## Per-corruption and prediction findings

[Run] C-A is positive for12/15 corruption means. It is positive in every seed
for Gaussian, Impulse, distortion_rbf, distortion_rbf_inv and Cutout. The
largest positive means are Cutout+.3917 pp and Gaussian+.3512 pp. Mean deltas
are negative for Density(-.0540), Shear(-.0270) and Distortion(-.0810).
Background C-A is+.2566 pp, with mixed seed directions(+.4457,-.2836,+.6078).
The full15 mean increment is not solely a Background effect: its contribution
is+.0171 pp; the other14 contribute+.1405 pp.

The new runs record labels, predicted classes and indices, permitting aligned
outcome counts. This alignment does not establish shared interpolation or
diffusion-noise draws. The archived B/D controls lack this prediction record.

| Seed | Wrong in A, correct in C | Correct in A, wrong in C | Net | Different predicted class |
|---|---:|---:|---:|---:|
| 0 | 531 | 489 | +42 | 2354 |
| 1 | 497 | 466 | +31 | 2295 |
| 2 | 534 | 432 | +102 | 2297 |

These are aligned cross-run prediction transitions, not proven treatment-caused
corrections. No significance test assumes the 15 corruptions or repeated seeds
are independent examples.

[Run/Inference] Most of the descriptive B-A macro gap is concentrated in
Background: +37.0475 pp on that corruption contributes+2.4698 pp of the
overall+2.6778 pp; other14 contribute+.2080 pp. The earlier v1 Background
score23.1767% is also near the no-SCD groups, but it uses a different spectral
operator/scale and another build. Its 14+1 composite is not included as a
homogeneous three-seed arm. This pattern motivates studying Background
anchoring; it does not prove a unique mechanism.

## Provenance limits

[Run] The new six runs share Git commit
`f04bfbdb24d5e9617602b50f1711f835861b6eb9`, runtime source manifests, assets,
all15 data manifests, Python/platform/package inventory and nominal
PyTorch2.1.2+cu121/CUDA12.1/cuDNN8902/A100-SXM4-80GB metadata. Every source
manifest matches its recorded commit. Each arm's three seeds also share
identical compiled-extension inventories.

**Between unguided and smooth-only**, SHA-256 differs for `chamfer`,
`chamfer_3D` and `pointnet2_ops._ext`, despite identical recorded file sizes
and nominal versions. Binary hashes do not prove different numerical behavior,
but functional equivalence has not been established. Runtime/build is therefore
confounded with method, and three seeds do not remove that limitation.
No shared-random-draw evidence is available.

Historical B/D runs share their own block at
`4be6afd86629f3d52648fb543fc2eb350a5f9f1d`; their data/assets and fixed host
settings match the new block. Runtime sources, native binaries and package
inventory differ across old/new blocks. B/D remain internally matched under
the recorded protocol; their previously nonpositive smooth increment is unchanged.

## Decision and next evidence

[Decision] Accept all six completed runs and reuse the supplied summaries;
the two missing full-test conditions are complete. No accuracy rerun or new
parameter search is scheduled. The chosen smooth-only configuration has a
small positive observed increment over unguided across seeds, but does not
replace SCD on this all15 comparison and has much higher recorded cost.
This is full-test-set development evidence, not independent confirmation.

[Inference/Open] Prioritize existing common-state update/graph/style diagnostics
to explain why the smooth increment changes in the presence of SCD. Previously
measured local cosines did not support a blanket destructive-local-conflict
claim. Capture only missing observations if a specific mechanism question
remains; do not jump directly to PxP, dynamic graphs, or a larger alpha grid.
The frozen alpha was calibrated relative to SCD states, so this result does
not characterize every possible smooth-only strength. Any further parameter
selection needs a declared independent development/evaluation strategy.

[Falsifier] Counts/prediction disagreement, nonzero unguided updates, incorrect
35-step Background execution or asset/source mismatches would invalidate the
accepted protocol; none was found. A future justified comparison resolving
build/draw confounds that loses C-A's positive direction would weaken a causal
spectral-benefit interpretation. The present observations remain archived.
