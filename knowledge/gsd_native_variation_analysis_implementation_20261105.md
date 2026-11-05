# Native trace CPU analysis — 2026-11-05

[Code/CPU/Open] `scripts/analyze_native_variation.py` analyzes six fresh sealed
control/native-trace ZIPs: three Gaussian44/91 and three Background495/512,
severity5/seed0/controlled backend/N8-B35. No GPU or operator replay is performed.
Old same-observer repeat evidence motivates this new capture; it does not contain
the required native intermediate arrays and is not relabeled as new trace data.

```sh
python scripts/analyze_native_variation.py --bundles RUN1.zip RUN2.zip RUN3.zip RUN4.zip RUN5.zip RUN6.zip --output-dir FRESH_DIRECTORY
```

[Code] Reuses accepted accuracy `load_run`, `open_bundle`, `arrays`,
`check_pairing`, `repeat_gate`, identities and strict sealed Bundle checks.
This retains source/native/backend/model/cache/reference/RNG pairing, frozen
model-state guards, original protocol and independent PM/terminal/CSV recount.
Executed source snapshot bytes are additionally matched to their recorded
identities. Native ELF identities are the producer's recorded/config-bound
identities; this CPU analyzer neither reloads native libraries nor executes them.

Trace validation requires schema1/complete, exact IDs and loop limit, zero extra
model/Chamfer/backward calls, exact five ordered boundaries per captured loop,
and no missing/extra trace NPZ. The first9 loops are captured, or all8 for N8.
Every NPZ bytes/SHA and numeric tensor legacy hash/shape/dtype/finite/header/
index is checked without pickle loads. Native NN indices are int32 and retained
sort indices int64; original float32 state/feature tensors retain their dtype.
Scalar SCD shape() is preserved by the existing legacy Torch tensor-hash helper.

Writer RNG guards must hold, with syntactically valid SHA256. First prior h,
condition and RNG bind to the accepted initial hashes; original-z/h0/timestep
bind throughout. Sorted-distance/retained-membership/cutoff witnesses are
checked against the existing unsorted distances. No new forward/reduction is
used to replace the archived SCD scalar. Recorded norms are header-validated;
comparison sizes are independently recomputed in float64 from raw arrays.

## Numeric evidence and candidate boundaries

[Decision] No new boundary-specific materiality threshold is invented.
`earliest_hash_difference` records the first bit identity mismatch.
`earliest_numeric_difference` requires actual unequal finite numbers, so a
signed-zero-only hash mismatch is excluded. This is a descriptive nonzero size,
not a claim that the difference exceeds a native floor or matters scientifically.
Every float/discrete tensor reports max absolute difference, relativeL2 with
denominator max(left norm,right norm), per-object RMS/changed count when it has
a batch axis, and discrete changed count for integer tensors. Zero denominator
with two zero tensors gives relativeL2=0. Batch scalar SCD has no object RMS.
All per-boundary RNG hash mismatches are separately retained.

Each corruption has three choose two repeat comparisons. Existing logitmax.05,
per-object/checkpointRMS.01 and firstdecodedRMS1e-4 gates are reported unchanged,
descriptively; their failure does not erase a valid trace analysis. There is no
threshold retuning, new acceptance gate or accuracy arm.

Candidate operators are local-prior forward, scheduler step, Chamfer forward,
sort/truncation, SCD reduction, joint backward/accumulation and joint update
arithmetic. A candidate requires exactly matching archived tensor inputs AND
matching relevant RNG hashes. Otherwise it is explicitly propagated input/RNG
difference. The gradient candidate remains broad because internal prior
activations/Jacobians were not archived. A candidate never identifies a unique
native kernel. Initial-condition differences are propagated inputs, not a
new operator diagnosis.

Candidate input references preserve original IDs, both ZIP identities and full
ZIP member names, tensor keys and hashes. These can seed a separately selected
fixed-input repeat. The native loop trace contains RNG hashes, not full RNG
values; it cannot by itself restore arbitrary later stochastic boundaries.
The separate shared-state capture is the full numeric RNG source for replays.

## Outputs and limitations

Fresh output contains input/analyzer/helper identities, incremental validated-run
JSON, six pair-localization JSON files, six keyed tensor-difference JSON files,
`tensor_differences.csv` and `summary.json`. Input ZIP identities are reverified
at completion. On failure, `failure.json` and any earlier output remain; an
existing output directory is refused. Original raw ZIPs are only read.

[CPU] Tiny sealed ZIP tests cover trace round-trip, re-sealed NPZ tampering with
stale tensor hash, missing/reordered boundaries, failed RNG guard, exact-input
prior candidate versus propagated h mismatch, discrete changed counts,
signed-zero hash separation and fresh failure/overwrite refusal. Focused checks
include existing trace and accepted accuracy-helper tests. Python3.8 syntax is
required; no complete real six-ZIP trace package exists in this CPU batch.

[Limitation] Background loops10–35 remain outside this trace, including later
PM checkpoints18/27. Trace copies/serialization synchronize native work and can
alter allocator/history. This analysis neither certifies uninstrumented CUDA
noninterference nor estimates speed, accuracy gain, or population significance.
Source/cache/RNG equality does not imply deterministic native kernels, and hash
inequality alone is not a numerically meaningful causal explanation.
