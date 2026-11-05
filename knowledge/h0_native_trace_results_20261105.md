# H0 native variation — actual trace result, 2026-11-05/06

**[Run/CPU/Inference] First divergence is localized to local-prior forward,
not a unique native kernel.** All six fresh-process repeat pairs have their
first numeric AND bit-hash difference in loop1 `noise_pred`, from bitexact
archived h/optimized condition/t with equal RNG. This occurs before SCD,
backward/update or a PM decoder callback. A bounded exact-input prior repeat
is therefore supported; a particular CUDA kernel or complete amplification
mechanism has not been identified.

**[Run/CPU] Subsequent bounded exact-input/RNG repeat also completed:** six
actual native prior forwards reproduce within-process drift. The earliest
covered exact-input numeric leaf difference is
`sa_layers.0.0.voxelization`: voxelized FEATURE output differs while normalized
COORDINATE output is bitexact. This is an empirical module bracket, not a
uniquely established kernel defect or complete trajectory-amplification cause.

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

No GPU/Colab/commit/push was performed by this CPU acceptance agent. Parent
executed the separately authorized native phases. Source/native/model
equality is verified, but CUDA scheduling/internal module history is unobserved.
Trace copies synchronize execution and can affect native history; no equivalence
to an uninstrumented process, speed result, accuracy gain or production repair
is claimed. The parent separately authorized a bounded targeted prior repeat;
its actual outcome and independent CPU acceptance follow below.

## Actual targeted prior repeat and CPU acceptance

[Run] Parent launched the reviewed separate operator at21:53:33 UTC. Whole
phase including archival47.277s, below900s cap. Same first Gaussian/BG h/style/t
and full numeric paired-cache RNG, grad-enabled leaf inputs, eval/frozen raw
model, unchanged source/native/backend/hardware/config. Three network forwards
per case; no backward/decoder/classifier/Chamfer forward or trajectory update.

[CPU] Returned ZIP558,879bytes, SHA
`4e16df1347bf8a5d89d3285ec8919f86e584fe4d8729ca0fa64edf3f82eaadb5`.
Actual Drive/local receipts match;25members/24package identities and every
inner output-manifest identity, CRC/path/type/bounds pass. Archived operator
source matches reviewed canonical de4d2fc7…94096; parent launcher/source/selection
pins also match their receipt. All six saved noise_pred arrays are finite
float32(2,8192,1,1), hashes and pairwise scalar metrics independently recounted.
All1962 recorded leaf events have finite outputs. Source/native guards true,
before/after model hashes equal, entry RNG before/after equal/restored=true.

| Same-input repeat pairs | Noise_pred maxabs range | RelativeL2 range |
|---|---:|---:|
|Gaussian all3 pairs|1.430511e-6–1.668930e-6|3.961359e-7–4.185163e-7|
|Background all3 pairs|1.624227e-6–1.728535e-6|1.955590e-7–2.050342e-7|

[Run/CPU] Each forward has327 leaf events;74 events have full numeric first-
repeat references within the64MiB cap. Other events retain hashes only.
For BOTH repeats1/2 versus0, in BOTH cases, the first output-hash drift and
first covered exact-input numeric drift occur in the same voxelization leaf:

| Case | Feature-output maxabs | Feature relativeL2 range | Coordinate maxabs |
|---|---:|---:|---:|
|Gaussian|2.384186e-7|5.686479e-9–5.807718e-9|0 exactly|
|Background|1.192093e-7|3.178588e-9–3.239524e-9|0 exactly|

Pre-hook input hashes, RNG hashes and grad mode match. Pre-hooks measure inputs
before possible in-place mutation. This reproduces observable-state drift
within one process; a fresh-process mismatch alone is no longer the only
witness. Later output-hash changes mostly have already unequal module inputs.
The cap leaves253 events without full numeric reference, and hooks themselves
synchronize/copy/hash. Do not infer a complete per-kernel audit, uninstrumented
CUDA noninterference, deterministic repair, speed or accuracy result.

## Pinned source chain and FP32 arithmetic interpretation

[Code] Actual raw imported Python files and pinned asset declarations verify:
`models/lion.py` imports local `models.latent_points_ada_localprior.PVCNN2Prior`;
the prior uses `models/pvcnn2_ada.py` Voxelization (class166, return188).
It returns `(F.avg_voxelize(features, vox_coords, r), norm_coords)`. Thus the
observed tuple root/0 is voxel features; root/1 is normalized coordinates.

`third_party/pvcnn/functional/voxelization.py` AvgVoxelization.forward calls
`_backend.avg_voxelize_forward`. `src/bindings.cpp` binds that to
`src/voxelization/vox.cpp`, whose zero-initialized output/count/index tensors
feed `vox.cu` grid_stats_kernel and avg_voxelize_kernel. The latter adds
`feat/count` into a shared voxel output using FP32 atomicAdd.

[CPU] Current Python wrapper bytes match the actual H0 archived imports AND
the frozen asset manifest. Current bindings.cpp/vox.cpp/vox.cu bytes also match
the frozen asset manifest's declared native-source identities; vox.cu4358bytes,
SHA `12fc6d6fc3ba78c73c0eb5c0ab1871f5c505d71e4bf68dbd04f83381970a5b0c`.
The actual loaded native ELF identities were independently bound and checked.
This does NOT establish an independent binary-to-source compilation proof;
the relevant CUDA/C++ sources are declaration-bound, not recompiled here.

[Inference] The real-arithmetic operation is a valid voxel mean. Different
legal accumulation orders of FP32 contributions can produce slightly different
rounding because addition is non-associative. This source mechanism is
consistent with feature-only drift and stable coordinates, and is a plausible
explanation to investigate. The module witness does not uniquely prove that
specific atomic instruction, an incorrect kernel, or its contribution to later
Background trajectory amplification. No deterministic rewrite/new GPU repair
was performed or selected after this evidence.

Actual receipt/raw outputs:
`result/preparation/gsd_comprehensive_20261105/prior_repeat_receipt_from_drive.json`,
`received_prior_repeat_v1.zip`, `prior_repeat_cpu_v1/extracted/`.
Compact independent acceptance and source identities:
[Prior repeat evidence](evidence/h0_native_trace_20261105/prior_repeat_cpu_acceptance_v1.json).
Original H0 trace analysis, failed source-validator V1, H5 analyses and all
frozen scientific gates remain unchanged. CPU acceptance is complete; no
additional GPU execution is planned by this agent.
