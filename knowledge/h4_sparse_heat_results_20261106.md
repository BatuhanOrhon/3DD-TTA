# H4 sparse heat technical result — 2026-11-06

**[Run/CPU/Decision] Accepted negative result: NOT_ADVANCED.** All four frozen
corruption probes completed, and none of K8/16/32 passed the preregistered native
1e-6 objective-and-vertex-gradient equivalence gate. The reference graph agrees;
K16/K32 loss errors pass but their gradient errors plateau above the gate.
Faster graph setup coexists with slower cached polynomial loss/VJP evaluation.
No whole-pipeline, trajectory, accuracy or equivalence claim follows.

## Frozen scope and accepted artifacts

[Run] A model-free probe consumed **existing** first-H5-control prepared X and
four decoded geometries for IDs459/545: Gaussian, Background, Shear and LiDAR,
s5/seed0/B2. Native dtype is float32. Checkpoints are2/4/6/8, or9/18/27/35 for
Background. No encoder, prior, decoder or classifier was executed. B1/B32 cost
inputs select or deterministically cycle these same two objects; B32 is a timing
batch, not32 independent evaluation objects.

[Run] Observed hardware/software: **NVIDIA A100-SXM4-80GB**, Torch2.1.2+cu121,
CUDA12.1. Before CUDA initialization the frozen script explicitly set matmul
TF32=false, cuDNN TF32=false, benchmark=false, cuDNN deterministic=true, and
deterministic_algorithms=false. These flags do not make all CUDA operations
deterministic. Supplemental source overlay:
`809573e8351877b8de4408090cf64300de5696dcf3b12a18911c14eba594b713`.
The frozen parent launcher SHA is
`3080be7c338fd1b11b85856456080c8d6f008a33ebaadfb7fb17c5b1e417f342`.

[CPU] Four result ZIPs and metadata ZIP passed independent bytes/SHA, full CRC,
path/type/size checks and complete manifests:164 payload identities across the
five archives. The actual source copies, pinned asset manifest, original H5
archive/config/producer-metadata chain, all four geometry hashes and IDs match.
Native sparse COO reconstructs the archived normalized L exactly; all eight
graphs have the reported same support, mean degree and native normalized raw L.
The archived residuals equal the original float32 decoded-Y minus prepared-X.
All saved losses/gradients, numeric decisions, exact-zero isolate gradients,
timing replication, sample counts and timing/memory summaries were recounted.

| Corruption | Result ZIP bytes | SHA256 |
|---|---:|---|
|Gaussian|30,121,501|`984d13fd95e74a5bc37e744c1b10ddce008411fb05815617767aec270e847c8f`|
|Background|28,887,726|`b1850fe42b6af4663d6612f15817153425694363a1970715d12eba050081bdf5`|
|Shear|31,409,378|`c2eb1eb3636751afbe522becb61e39e8b24c236f99aa311d737563aebc44d965`|
|LiDAR|31,236,059|`46724daa95e8e774e88492d912b7a754f11fb4dad3b81e985df01c5737b2fed7`|

[Run] Metadata ZIP:1,044,055 bytes,
SHA `1bc3541c8ae8e23111474119482591a84cea5974c1bb37c3c36241178718ceeb`.
Parent phase wall time167.929s under the single1800s cap; stage return codes
were zero with explicit `equivalence_gate_failed` statuses. A successful process
exit is descriptive completion, not a positive scientific gate.

## Objective, gate and approximation order

[Code/Derivation] The active frozen graph uses `A=L/mean_active_degree`, beta2
and the original `3N` reduction with batch sum. Exact reference is the actual
native-U/q all-active quadratic. The polynomial computes
`||p_K(A)E_active||²/(3N)`, `p_K≈exp(-beta*A/2)`, with the degree-only upper
bound `2*dmax/mean_degree`; no eigensolve selects its interval. Isolates stay
zero. No clean labels choose K or an approximation tolerance.

[Run/CPU] Gates require per-object and batch relative loss errors <=1e-6 and
per-object vertex-gradient relative L2 error <=1e-6, with denominator floors
1e-12 and exact-zero isolate gradients. There are32 dependent object/checkpoint
pairs per K: two objects × four corruptions × four checkpoints.

| K | Maximum loss relative error | Maximum gradient relative error | Maximum scalar error on actual q | Loss passes | Gradient passes |
|---|---:|---:|---:|---:|---:|
|8|6.451e-5|1.611e-4|9.264e-5|2/32|0/32|
|16|4.947e-7|4.465e-6|1.808e-11|32/32|0/32|
|32|4.947e-7|4.465e-6|2.442e-15|32/32|0/32|

[Run] K16 gradient relative error ranges1.619e-6–4.465e-6; K32 gives almost
the same range. Every corruption has `first_acceptable_order=None`. The strict
gradient failure is retained even where the scalar objective agrees.

## Distinguishing truncation from native operator discrepancy

[Derivation] The epsilon bounds `epsilon*(2+epsilon)*||E||²/(3N)` and
`2*epsilon*(2+epsilon)*||E||/(3N)` apply to an ideal orthonormal eigensystem
for the same operator. They exclude native U nonorthogonality, eigenvalue
clamping, finite-precision normalization and sparse recurrence arithmetic.
Their failure to cover the **total native-reference discrepancy** is not a
contradiction of polynomial approximation theory.

[Run] Archived U has orthogonality max error1.614e-6–2.221e-6 and Frobenius
error6.342e-5–7.543e-5. `||A U-U q||_F` is6.057e-5–8.234e-5. Clamping changes
normalized eigenvalues by up to1.391e-6; normalized row sums have float32
roundoff. Thus the actual stored-L filter and actual finite-precision U/q
reference need not coincide to1e-6 gradient accuracy.

[CPU] Label-free FP64 **evaluations of the archived operators**, without a new
eigensolve or model run, distinguish these effects for K16/K32:

| Gradient comparison | Relative L2 range |
|---|---:|
|Native polynomial versus FP64 polynomial on stored L|8.774e-8–1.257e-7|
|Native exact reference versus FP64 evaluation of archived U/q|2.593e-7–3.456e-7|
|FP64 stored-L polynomial versus FP64 archived-U/q operator|1.595e-6–4.442e-6|
|Changing only archived-q clamping in the FP64 U evaluation|0–3.122e-7|

[Inference] The disagreement between the archived operators dominates each
path's native evaluation error here; truncation alone does not explain the
K16/K32 plateau. Clamping alone also does not explain it. This does not identify
an incorrect native gradient or certify a replacement high-precision heat
reference. The CPU replay changes no raw tensor, scientific tolerance or gate.
It cannot rescue the declared native equivalence failure.

[CPU] An initial analyst-only exact-division assertion exposed a CPU/CUDA
arithmetic difference: these CUDA float32 tensor/scalar divisions are exactly
reproduced by float32 reciprocal multiplication, whereas NumPy division can
differ by one ULP. The analyst check was corrected to reproduce that arithmetic;
its initial log is retained. This was not the scientific equivalence test or
the earlier failed CPU-filter-reconstruction gate, which remains unchanged.

## Component cost and memory

[Run] Each graph builder had two warmups and three measured fresh builds.
Each cached arm had three warmups and ten measured loss/residual-VJP pairs.
CUDA boundaries synchronize. Ranges below span the four corruption medians.

| Scope | B1 ms | B32 ms |
|---|---:|---:|
|Exact graph + U/q setup|50.10–58.35|1568.11–1646.35|
|Sparse graph setup|9.78–9.90|301.18–304.27|
|Cached exact-U loss + residual VJP|1.00–1.07|11.72–12.36|
|Cached K8 loss + residual VJP|6.76–6.89|195.68–208.15|
|Cached K16 loss + residual VJP|12.51–12.76|386.64–402.99|
|Cached K32 loss + residual VJP|23.87–24.71|755.85–800.10|

[Run] Coefficient preparation is separate: exact0.099–0.154ms versus
K16 1.189–1.429ms for B1; exact1.413–1.449ms versus K16 35.694–51.971ms for
B32. Sparse validation is separately recorded. Dense cdist/sort/adjacency
setup remains; this is not a sparse-kNN result. The Python per-object/per-order
sparse recurrence is the implementation tested, not a general sparse-kernel
performance theorem.

| Setup scope | B1 absolute / incremental peak MiB | B32 absolute / incremental peak MiB |
|---|---:|---:|
|Exact|178.15–191.79 /102.65–111.46|650.40–692.57 /573.07–610.79|
|Sparse|145.59–155.92 /70.09–75.59|202.40–210.16 /125.07–128.37|

[Run] Cached-filter peak increases are0.113–0.118MiB exact versus
1.635–1.789MiB polynomial for B1, and1.501MiB versus2.424–2.544MiB for B32.
The receipt also retains every absolute peak and starting allocation. Both
operators coexist during filtering; starting allocations vary with retained
allocations/workspaces, so absolute peaks are not isolated deployment footprints.
Input replication/cloning, decoder/sampler backward, encoder/PM and artifact
serialization are outside cached-filter latency. These component measurements
cannot establish an end-to-end speed gain.

## Reproduction and decision boundary

[CPU] Compact evidence: [independent acceptance receipt](evidence/gsd_sparse_heat_20261106/cpu_acceptance_v1.json).
Full acceptance, source and replay rows are under
`result/preparation/gsd_comprehensive_20261105/received_h4_v1/independent_v1/`.
Reproduction command and analyst source identity are in the compact receipt;
the isolated CPU helper uses NumPy/SciPy on native saved arrays, no CUDA/model.

[Decision] Do not advance a whole-pipeline or trajectory experiment from this
H4 probe. Do not relax the1e-6 gate, select a new degree using the oracle,
relabel old failed evidence, or infer accuracy from these costs. A later change
to the reference precision, numeric contract or implementation would require
its own prospectively frozen protocol; none is executed or promoted here.
