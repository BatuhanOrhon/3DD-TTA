# H0 native variation — actual trace result, 2026-11-05/06

**[Run/CPU/Inference] First divergence is localized to local-prior forward,
not a unique native kernel.** All six fresh-process repeat pairs have their
first numeric AND bit-hash difference in loop1 `noise_pred`, from bitexact
archived h/optimized condition/t with equal RNG. This occurs before SCD,
backward/update or a PM decoder callback. A bounded exact-input prior repeat
is therefore supported; a particular CUDA kernel or complete amplification
mechanism has not been identified.

## Accepted raw scope and validation history

Six new operational control/weight0 traces: Gaussian44/91 and Background495/512,
three processes each, seed0/s5/B2/PointMAE/LION raw/eval/frozen, controlled
backend. Gaussian N8, Background B35; original .03/.003/.95/DDIM100 SCD host.
No spectral graph, additional model/Chamfer forward, extra backward, new
guidance or broad accuracy rerun. First8/9 loops have five existing boundaries.

[CPU] `capture_receipt_from_drive.json` and `received_captures_v2/download_verified.json`
match. All six original ZIP SHA/bytes, CRC, path/type/size, exact sealed manifest
coverage and extracted file hashes pass. Fresh extraction and outputs live in
`result/preparation/gsd_comprehensive_20261105/h0_native_trace_v1/`.
Source/native/backend/model/cache/prepared/reference/initial+output RNG pairing
and every trace NPZ/header/tensor hash/finite/dtype/shape/index/order/cutoff/RNG
guard pass. Total255 boundary snapshots; paired tensor-comparison rows1275.
Actual native ELF identities also match the six preserved H5 native payloads.

[Code/CPU] First `analysis_v1` failed before comparisons because its source
validator only accepted asset snapshot keys; the producer records88
`assets/source_snapshot/...` and88 `repo/...` aliases per run. All176 aliases
independently match the same canonical sealed source bytes. Parent approved a
validation-only mapping fix: exact known prefixes, canonical safe relative
PurePosix paths, unchanged byte/SHA enforcement, unknown/traversal rejection.
Sixteen focused tests pass; source patch/before-after hashes and failed V1 are
retained. `analysis_v2` succeeds without changing comparisons or frozen gates.
Actual GPU source72e0eb4/overlay8095 remains unchanged.

## Earliest numeric and hash differences

All first prior inputs are bitexact, including original h0/z0 and actual t;
all captured writer RNG boundaries also match across each pair. Gaussian
loop1 scheduler t70 uses prior t71; Background t340 uses prior t341.
The `scheduler` snapshot holds the already-computed prior output, so the
`noise_pred` divergence precedes scheduler output arithmetic.

| Corruption, repeat pair | First epsilon maxabs | Relative L2 | Per-object RMS range |
|---|---:|---:|---:|
|Gaussian0–1|1.907349e-6|4.213657e-7|1.702114e-7–1.817351e-7|
|Gaussian0–2|1.430511e-6|4.103894e-7|1.545222e-7–1.869076e-7|
|Gaussian1–2|2.264977e-6|4.283753e-7|1.593146e-7–1.967191e-7|
|Background0–1|1.266599e-6|1.970380e-7|1.595844e-7–1.645430e-7|
|Background0–2|1.430511e-6|1.998398e-7|1.580881e-7–1.704539e-7|
|Background1–2|1.430511e-6|1.972655e-7|1.547136e-7–1.694894e-7|

[CPU/Inference] These are finite nonzero numbers, not signed-zero-only hashes.
No boundary-specific materiality threshold was invented. Tiny first differences
do not alone establish their causal contribution to the much larger later drift.
Only the network-level forward operator is bracketed; its internal activations
and Jacobian are not certified by equal input tensors.

## Discrete order versus actual selection/routing

The first discrete difference is loop1 `retained_indices_1`: Gaussian changed
array positions6/4/2 by pair, Background8/10/6. These are permutations INSIDE
the retained prefix, not first-loop retained-set membership changes.

| Corruption/pair | First NN assignment change | First retained-set membership change |
|---|---|---|
|Gaussian all three pairs|None through loop8|None through loop8|
|Background0–1|loop3, direction1, counts[1,0]|loop5, direction1, symmetric difference[2,0]|
|Background0–2|loop3, direction1, counts[1,0]|loop5, direction1, symmetric difference[2,0]|
|Background1–2|loop7, direction2, counts[1,0]|loop8, direction2, symmetric difference[2,0]|

[Inference] Later changed NN/retained routing is downstream of already unequal
prior outputs/inputs; this trace does not isolate SCD as the first source.
Array-order differences alone must not be relabeled as a changed robust mask.

## Frozen whole-trajectory numerical gates

Original logitmax.05, per-object/checkpoint RMS.01 and firstdecoded RMS1e-4
are unchanged. Gaussian passes3/3 pairs (worst logit5.865097e-5).
Background fails3/3 (logit maxima2.667616/1.968346/1.924604).
These gates describe the complete original trajectory, not only first9 loops.
PM/terminal predictions and every candidate discrepancy remain in raw reports.
Passing Gaussian gates do not mean bitwise repeatability; failing Background
does not identify a unique native kernel or justify threshold relaxation.

## Exact-RNG feasibility for the separately selected follow-up

[CPU] The original accepted zero-repeat paired caches are byte-identical to
the current first Gaussian/Background H0 cache references. Their numeric
Python/NumPy/Torch CPU/one-CUDA-device RNG states load without pickle objects.
Computed state hash equals cache `rng_after_sha256`, actual trajectory initial
RNG and loop1 pre-prior RNG, in both cases:

- Gaussian cache135233bytes, SHA `e15137161981feabbbf0a8bb49f3c3a3e9c4d5cc053ba05cb0dc16ddc43814d6`.
- Background cache135046bytes, SHA `1a43e2bf9decc5d2dc07956a09b543b09517f9ca6e2389c9b4bcccc7010f43a8`.

The exact numeric RNG proof and full candidate ZIP member/input hashes are
saved in `acceptance/exact_first_prior_rng_proof.json` and
`acceptance/targeted_prior_followup_inputs.json`. Native trace NPZ itself only
contains RNG hashes; it is the separately bound paired cache that supplies
the complete restorable state. A replay must preserve actual grad-enabled
prior semantics with requires-grad h/style leaves and fixed eval/frozen model;
a no_grad/toy/seed-only substitute is not the matched native witness.

## Artifacts and limits

[CPU] Existing `scripts/analyze_native_variation.py` ran in the pinned persistent
CPU venv. Commands/stdout/stderr/source patch/identities are in `acceptance/`;
`analysis_v2` retains six pair-localization JSONs, six keyed tensor-difference
JSONs, CSV and full summary. Original ZIPs were only read. Compact acceptance:
[H0 evidence](evidence/h0_native_trace_20261105/cpu_acceptance_v2.json).

No GPU/Colab/commit/push was performed for this analysis. Source/native/model
equality is verified, but CUDA scheduling/internal module history is unobserved.
Trace copies synchronize execution and can affect native history; no equivalence
to an uninstrumented process, speed result, accuracy gain or production repair
is claimed. The parent separately authorized a bounded targeted prior repeat;
its outcome is not part of these completed trace results.
