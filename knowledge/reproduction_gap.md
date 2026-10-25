# Reproduction Gap: Current Diagnosis

## Current amendment — 2026-10-19

[Run/Inference/Open] The dated September diagnosis below is retained as
history. The multibackbone study now has complete source/original3DD/GSD
coverage for six dataset/backbone cases (18 accuracy rows, seed0/severity5),
plus four-head GSD-only Scan readouts. Exact GSDTTA reproduction is still
unresolved: published alternative weights, training splits/preprocessing and
corruption identities are not the authors' verified experimental assets.
ModelNet CurveNet is7.3687pp below its paper source target, despite the small
Scan CurveNet gap; numerical closeness for selected rows is not parity.
Scan DGCNN clean68.6747% and corrupted32.7826% already indicate weakness before
adaptation. Strict loads, labels and split membership pass; no single cause
has been identified for the paper deficits.

[Run/Inference] The earlier Point-MAE Scan severity diagnostic found severity1
Gaussian60.59% close to reported59.0%, but its complete13-corruption mean49.28%
remained below the corresponding reported69.0%. No-background only partly
closed that Gaussian gap. Thus a global severity swap is not an established
explanation; those Point-MAE tests are also not a three-backbone severity sweep.
Continue to report main_split581/severity5 and exact checkpoint provenance.

[Inference/Open] GSD terminal is below original3DD in all six current cases;
Scan probability mean exceeds original for CurveNet/DGCNN, not PointNeXt.
The flows differ in batch, horizons, decoder and scheduler, so none of these
differences isolates spectral guidance. Same-flow spectral0 controls and seed
replication remain unexecuted. ModelNet DGCNN GSD has full coverage via13+2
attempts but lacks the old13 final LION state/RNG continuation proof.
[Final tables, provenance, limitations and falsifiers](gsd_multibackbone_final_findings_20261019.md);
[source discrepancy audit](gsd_multibackbone_source_results_20261018.md);
[severity evidence](colab_scanobjectnn_severity5_20261017.md).

[Code] 2026-09-23: GSD-only implementation is now authorized on a separate
branch; see [the current audit/design](gsd_integration_20260922.md).
Baseline `tta.py`, model, dataset, preprocessing, FPS and control equations
are preserved. [Open] This development does not close the paper reproduction
gap or Point-MAE checkpoint provenance question and supplies no GPU accuracy
evidence. Compare only matched eval/raw original-style controls.

## The three numbers that must remain separate

| Value | Status | Meaning |
|---:|---|---|
| 65.7% | **[Paper]** | Published WACV 2025 ModelNet40-C mean for 3DD-TTA with Point-MAE |
| 66.1% | **[Code]** | Mean shown in the repository README's 3DD-TTA table |
| ~63% | **[User report]** | Local Colab run of the original code; no archived run bundle yet |
| ~63.5% | **[User report]** | Local best GSD-script observation; no archived run bundle yet |

The local gap is therefore approximately 2.7 points from the published paper or 3.1 points from the README, pending exact artifacts. The approximately 0.5-point GSD improvement is not yet statistically or procedurally verified.

## What has already been ruled down

- **[Run]** Complete source-only evaluation at `result/modelnet40_c/source_only/20260912-111822_source-only_seed0/` completed all 15 severity-5 corruptions with frozen Point-MAE and FPS(1024), but obtained 53.69% macro accuracy versus the README 57.6% reference. Thus the gap begins before LION/TTA under the recorded Colab protocol.

- **[Code]** The archived source-only evaluation matches upstream's own source path exactly: direct `data_<corruption>_5.npy` loading, FPS to 1024 points, and frozen Point-MAE `classification_only`. **[Run]** all arrays have 2,468 examples and labels span 0--39, consistent with the expected ModelNet40-C test layout. The README explicitly designates the Zenodo pre-corrupted download for ModelNet40-C; its local corruption-generation instruction concerns ShapeNetCore/ScanObjectNN. This makes a locally introduced source-only preprocessing change unlikely, but cannot establish byte-identical data/checkpoint assets because authors publish no hashes.

- **[Run]** The matching `data_original.npy` clean control achieves 90.64% (2237/2468) with the same label file, Point-MAE checkpoint SHA-256 `507e0bbf...aec75`, batch size, FPS(1024), frozen classifier path and Colab environment. Thus the 53.69% corrupted-source macro is not explained by an obvious failure of this classifier on the paired clean input. This internal control has no author-published checkpoint-appropriate clean reference, so it does not establish external reproduction parity.

- **[Code]** Current `tta.py` is almost the upstream baseline; method additions are primarily in separate scripts. The low baseline is unlikely to be caused simply by GSD logic contaminating the original path.
- **[Code]** ModelNet40-C loader selects severity 5. **[Paper]** Table 2/Section 4 do not themselves state a severity number, so equivalence between the paper table and this released-code protocol is plausible but not yet proven.
- **[Code]** The main path performs the LION scale/rotation and Point-MAE resampling operations expected by the repository.

These observations narrow the search but do not prove parity.

## High-priority explanations

### P0 — Data and checkpoint identity

Different downloads, corruption-generation versions, incomplete files, or checkpoints can shift all methods. Record file paths, sizes, and SHA-256 hashes for:

- Point-MAE checkpoint;
- LION checkpoint(s);
- ModelNet40-C archive or every evaluated corruption file;
- any normalization/config assets.

### P0 — Environment drift

`requirements.txt`, `env.yaml`, and the Colab notebook do not define one identical environment. PyTorch, CUDA, `diffusers`, kNN/FPS extensions, NumPy, and compiled point-cloud ops may differ. The original requirements pin an old `diffusers`, while the Colab path may resolve a newer scheduler implementation.

### P0 — Stochasticity and aggregation

Random interpolation and diffusion noise are unseeded; CUDA settings may also be nondeterministic. Batch size changes the order/association of random draws. A single run can therefore differ from another even with identical arguments. Existing logs do not report variance.

Confirm that the reported mean is the macro-average of all 15 per-corruption accuracies and that resumed CSVs do not duplicate or omit corruptions.

### P1 — LION inference mode: confirmed discrepancy

**[Code]** Original LION trainer evaluation calls VAE/prior eval, while the fork inherits its byte-identical demo wrapper with no eval calls. Baseline/GSD model setup puts only Point-MAE in eval. The supplied config activates 0.1 dropout in the local prior and local encoder/decoder. Frozen weights/no_grad do not disable dropout. Original LION revision: `7711b3d185752eeb632d095494876e4de15f3195`; exact sources and wrapper hash are in [the follow-up audit](code_audit_20260912.md).

**[Inference]** Accuracy/variance impact is not measured and published 3DD-TTA runtime modes are unknown. Prioritize isolated legacy/eval A/B after artifact/source-only identity controls, without changing scheduler/rates/style/lambda. Preserve guidance input gradients and explicitly pair interpolation/VAE/noise draws.

### P1 — Scheduler semantics

The baseline and GSD paths do not currently instantiate DDIMScheduler identically. `set_alpha_to_one`, timestep spacing, prediction type, clipping, and library version can alter reconstruction. Compare scheduler configs serialized at runtime, not source defaults alone.

### P1 — Hyperparameter/protocol mismatch

The paper describes `lambda=0.96`, while repository defaults use 0.95. Background versus normal reverse steps, batch size, guidance rates, sampling points, and classifier voting/augmentation must be matched exactly. Even small differences can interact with loss sums.

### P1 — Output/resume contamination

Baseline output is append-oriented and GSD CSV resume logic can combine incompatible configurations if filenames are reused. Existing CSVs omit important configuration fields. Every new run must use a fresh immutable directory.

## Lower-priority but important code questions

- Which global/style representation should be used for final decoding?
- Are gamma and eta semantically mapped as intended when they are unequal?
- Does the classifier checkpoint load strictly and completely in both baseline and GSD paths?
- Do `knn_cuda`, FPS, and custom extensions behave identically across Colab GPU/runtime versions?
- Is the data order/class label map identical to the reported setup?

## Controlled diagnosis sequence

1. Preserve/archive the existing failing run if available; establish a reviewed clean main-based branch without losing knowledge/user work.
2. Add artifact/seeding/count controls while preserving original adaptation mathematics; inspect data/checkpoint hashes and environment.
3. Verify the complete source-only corruption row before interpreting LION effects.
4. Test original TTA legacy versus LION eval mode on Gaussian/background, seeds 0,1,2 and explicitly controlled common draws.
5. Archive direct LION encode/decode reconstruction (distinct from unguided perturbed DDIM), then full original 3DD-TTA under the selected mode.
6. Serialize schedulers; investigate dependency, lambda, steps or batch one factor at a time if the gap persists.
7. Require spectral-off GSD tensor/prediction parity with the selected baseline, then turn on one spectral band.
8. Defer sum/small-step, dynamic and PxP trials until these gates pass. See [execution batches](clean_restart_batches.md).

## Decision rule

A cause is accepted only if changing that single factor produces a repeatable, directionally consistent change and restoring it restores the old behavior. Correlation with a branch or environment label is not sufficient.

## Current conclusion

**[Inference]** The verified LION mode discrepancy is now a high-priority one-factor test alongside data/checkpoint/environment controls; no accuracy root cause is confirmed. This refines the earlier environment/stochasticity-first hypothesis rather than disproving it. Required evidence is a complete baseline bundle and controlled dropout A/B, not another unmatched method sweep.

## Operational update - 2026-09-15

The historical next-action text above is superseded by
`session_handoff_20260915.md`: eval/raw is provisionally selected from complete
matched seed-1/2 runs, EMA remains an ablation, and source-only severity 1--5 is
the active diagnostic. No source-data root cause has been established.

**[Code/Run/Inference]** The new `code_audit_20260915.md` identifies a concrete
source-path candidate: FPS requests 1024 outputs from 649/724/768-point
Density/Cutout/LiDAR files, and its bundled kernel contains an origin filter.
Index repetition is unavoidable for those lengths; the actual binary's filtering
and classifier effect await Colab evidence. These corruptions already have
source deficits of about 9.87/8.12/9.16 pp against rounded paper cells.
Investigate this alongside data/checkpoint identity if severity does not resolve
the descriptive gap. A preprocessing-only identity control would separate that
effect from LION reconstruction. Updated final style and SCD scaling are queued
TTA candidates, with no newly measured accuracy gain.


## 2026-10-25 — later literature does not establish exact baseline reproduction

[Paper/Code/Run/Inference/Open] The [citation audit](3dd_tta_citation_reproduction_audit_20261025.md)
starts from study base`4c5be0e` on isolated`gsd-3dd-citation-audit`. It verifies15
primary/reference citations within a documented bounded search. GSDTTA reports
rerunning3DD on DGCNN (ModelNet71.69/source66.51; Scan57.08/source55.72).
Uni-Adapter reports implementing3DD on Uni3D-Large (ModelNet56.06/source59.15,
Scan30.78/source38.07, Shape55.30/source57.92), B1/s5 and a timestep search;
three authors overlap original3DD. Neither is an exact Point-MAE reproduction,
and their raw data/checkpoint/log identities are unavailable. No later accessible
study here establishes the original65.7/64.3/69.8 as an independently verified
exact rerun. Citation-only studies and identical printed cells cannot supply it.

[Paper/Code] Original main tables do not explicitly name a severity level;
AppendixA Gaussianσ.03 supports5, as does the released evaluator. Pinned
upstream`107305fd7baf40b359f31c07d235599198be7324` still has unseeded generator
severity8 versus evaluator5. The already-repaired local generator and existing
Scan s1full13/profile controls remain valid: randomness/severity is plausible
but not a demonstrated complete cause. Canonical ModelNet byte parity was
already established, so local random regeneration is not a sufficient explanation
there. Original README printed66.1 also conflicts with its own displayed-cell
mean65.44; preserve it separately from paper65.7 rather than choosing a target.

[Run/Limit] SVWA's Gaussian32.5 and mean33.1 are confirmed in its primary paper;
the existing checkpoint receipt shows different weights from our3DD asset.
This is protocol-family evidence, not exact reproduction. No fresh checkpoint
inference or stochasticity experiment was run. Exact authors' arrays, weights,
preprocessing/aggregation and logs or clarification remain the falsifier.
Raw literature and SHA/CRC receipt are linked from the audit; immutable historical
results in the original root were not modified.

## 2026-10-25 — source-family agreement and downstream reporting variation

[Paper/Run/Inference/Open] The [expanded reporting audit](pointcloud_tta_reporting_audit_20261025.md)
adds source references previously missing from the comparison: SVWA CurveNet
ModelNet64.0 (suppTable3p3) versus our64.0113, and CloudFixer full15Point-MAE
53.70 versus our53.6899. Our ModelNetDGC58.7979 also matches SVWA58.8;
PointNeXt65.8320 remains1.1580pp belowGSD66.99 and1.1980 belowCF67.03.
Thus all localbackbones must not be declared broken merely because selected
GSD/3DD target rows differ. SVWA-linked ModelNet checkpoint-family consistency
is supported; full paper/data/checkpoint identity is not established.

Source training matters before adaptation: originalMATE53.9/45.7/57.6 differs
from its joint-trained57.6/45.6/59.3. Original3DD Scan61.5 is not universally
repeated in laterPoint-MAE papers. ScanDGC32.7826 remains belowSVWA37.9;
its cause staysopen. Existing severity/split controls and canonicalModelNet
byte-parity evidence are unchanged, and no random-draw variability was measured.

The audit distinguishes explicit copied numbers, changed-protocol author
reruns and unknown provenance. The MATE64.3→SMART58.9→PG58.90 chain shows why
identical later methodnames need an immediate numerical donor, not just the
originalcitation. CloudFixer six/all15, CF/CF-O, and PointCache/BayesMM7/8term
means demonstrate non-random explanations for selected differences. These
findings do not isolate the cause of our residual gap or prove other papers'
results incorrect. Exact author assets/protocol/logs remain the falsifier.
