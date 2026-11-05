# Frozen H5 cluster bootstrap — 2026-11-05

[Code/CPU/Decision] Rules frozen at CPU preparation, 20:30:06 UTC, before any
actual H5 capture or clean-oracle result was available to this implementation
batch. Only synthetic labels are used in its tests. Root owns the execution
record and any actual GPU/CPU evidence acceptance; those records are unchanged.

`scripts/bootstrap_h5_reliability.py` consumes a completed H5 analysis:

```sh
python scripts/bootstrap_h5_reliability.py \
  --analysis-dir COMPLETED_H5_ANALYSIS \
  --summary-sha256 SUMMARY_SHA --features-sha256 FEATURES_SHA \
  --rows-sha256 ROWS_JSONL_SHA --output-dir FRESH_BOOTSTRAP_OUTPUT
```

All three file SHA pins are required and reverified at completion. The summary
must be complete or complete_with_repeat_floor_failure, with consistent native
gate status, feature bytes/SHA, original8 IDs/four corruptions/three repeats/
seed0. Both features and oracle rows must cover exactly2304 unique Cartesian
conditions:8 IDs ×4corruptions ×3repeats ×4checkpoints ×6bands. Case/B2 membership,
actual checkpoint counts, frozen score keys/finite values and identical feature
values/order between the two artifacts are checked. Original pooled common-mask
row-key SHA, AUROC and class/sample counts must reproduce the summary.

## Frozen nonlinear bootstrap

The original `analyze_h5_reliability.score_comparison` determines eligibility,
using resolved oracle labels/finite CD deltas and all three finite scores. Its
same mask excludes first-checkpoint combined-score unavailability. All three
rankings use exactly these rows; per-feature full-coverage rows are excluded
from this contrast. Neither raw clean geometry nor classifier labels are read.

There are exactly2000 draws from `numpy.random.default_rng(0)`. Each draw samples
eight original-object cluster positions with replacement. An object contributes
ALL its eligible corruption/checkpoint/repeat/band rows, in their original
within-cluster order. Sampling an object twice duplicates its entire cluster;
rows or repeated IDs are not deduplicated. Objects with no eligible rows remain
part of the original eight-cluster sampling frame.

For every draw the existing score_comparison/association functions recompute
the three pooled nonlinear AUROCs on the same duplicated rows. No linear mean
of object AUROCs substitutes for this statistic. Paired contrasts are combined
minus frequency and combined minus residual, computed from that same draw.
There is no score/feature selection, fitting, hyperparameter search, new q or
applied guidance. Spearman intervals are omitted in this compact batch.

Report observed pooled AUROCs/contrasts and percentile95 intervals from the
defined draw values. One-class or empty draws have unavailable AUROC for all
three rankings, remain NaN in numeric NPZ and are counted separately. They are
never zero-imputed. The JSON states that percentile intervals condition on
defined draws and reports valid, invalid, one-class and empty replicate counts.
No additional data-dependent minimum-valid-draw rule is introduced.

## Advance rule, frozen before actual data

Advance only if ALL hold:

1. `native_repeat_gates_all_passed` is true.
2. Combined AUROC lower95CI is strictly greater than0.5.
3. Both paired AUROC-difference lower95CIs are strictly greater than0.

Undefined intervals fail advancement. Equality fails the strict inequalities.
Otherwise `advanced=false` and `advance_status=not_advanced`; completed artifact
analysis itself can still have status=complete. This gate is descriptive
development triage on eight historically exposed clusters, not population,
heldout, accuracy or operational confirmation. Known native-floor failure is
not repaired by an apparently favorable bootstrap interval.

## Artifacts, checks and limits

Fresh output includes input/analyzer/association-helper identities, compact
summary and `bootstrap_values.npz`. Raw arrays retain sampled object IDs and
cluster indices, all three AUROCs, paired differences, sampled row/class counts,
score/difference column names and each draw's ordered common-row-key SHA.
The base common-mask ordered row-key SHA is also reported. These identities and
draws reconstruct identical masks for all rankings without a massive row dump.
Original analysis files are only read; failures retain a fresh failure.json and
partial outputs. Existing output directories are refused.

[CPU] Three focused tests cover cluster duplication, all-score paired masks and
ordered hashes, nonlinear AUROC recomputation, one-class unavailability/no
advancement, full2304-row coverage, source SHA changes, duplicate condition
rejection and failure-output overwrite refusal. Combined bootstrap/core checks
pass11tests. Python3.8 grammar and whitespace checks pass. No full suite or GPU
run is needed for this isolated CPU script. Actual completed H5 data and the
fixed2000-draw native-study result remain unexecuted in this implementation batch.
