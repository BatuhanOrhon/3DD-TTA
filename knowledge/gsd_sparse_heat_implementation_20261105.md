# H4 standalone sparse square-root heat diagnostic — 2026-11-05

[Code] This CPU-only implementation batch adds `sparse_heat.py` and
`tests/test_sparse_heat.py`; it does not change `tta.py`, the diagnostic driver,
or `graph_spectral.py`. The checked-out source during verification was
`78658d07e168f337ab3472556f5e0ca849748ebe`; the production graph recipe is
`graph_spectral._sample_target`. The parent operator owns subsequent source
sealing, independent review and any actual-GPU measurements. No model or GPU
run was performed for this batch.

## Objective and graph contract

[Code] `build_sparse_heat_target(X, config)` repeats the current reference graph
recipe exactly: direct-difference cdist squared; diagonal infinity; stable
nonself kNN; RBF with denominator `2*delta²`; directed-degree retention threshold
`gamma*sum(directed)/(N*k)`; max symmetry; both-endpoint pruning; final positive
degree support. It forms the same active dense Laplacian, divides by the same
Python-float mean active degree, and stores only its coalesced sparse COO tensor
and original active vertex indices. It does **not** compute eigenvalues or a
spectral radius. Dense cdist, sorting, adjacency and active Laplacian setup remain.

[Derivation] Write the frozen active normalized Laplacian as `A=L/mean_degree`.
For residual `E=Y-X`, the target is the existing beta-2 smooth quadratic
`sum_b <E_active, exp(-2A) E_active>/(3N)`. The standalone approximation is
`sum_b ||p_K(A) E_active||²/(3N)` where `p_K(t)≈exp(-beta*t/2)` and beta defaults
to 2. This approximates the square root of the heat operator before taking a
squared norm. The original point count N remains in the denominator; objects
are summed, with no batch mean. Isolates contribute exactly zero loss/gradient.
The graph and support are detached and remain fixed during residual backward.

[Derivation] Gershgorin provides `upper=2*dmax/mean_degree`; Chebyshev uses
`2A/upper-I`. The bound is computed from degrees, never from an eigensolve.
`target.prepare(order=K, beta=2)` accepts the predeclared K values 8, 16, 32.
Coefficients use deterministic CPU scalar cosine quadrature and `math.fsum`,
with at least 256 nodes (more for unusually wide intervals); no RNG is consumed.
This quadrature is an approximation, not an interval-error certificate.

[Derivation] If the maximum square-root approximation error on the actual
eigenvalues is epsilon, the metric error is at most `epsilon*(2+epsilon)` because
the exact square-root operator has norm at most one. Thus absolute loss error
is bounded by `epsilon*(2+epsilon)*||E_active||²/(3N)` and gradient L2 error by
`2*epsilon*(2+epsilon)*||E_active||/(3N)`. Tests independently evaluate the scalar
polynomial on the toy exact spectrum and check both bounds, allowing floating
point roundoff. This is a graph-specific witness, not a certified uniform
interval bound or a theorem about accuracy.

## Minimal interface and measurement scopes

```python
from graph_spectral import SpectralConfig
from sparse_heat import build_sparse_heat_target, measure_sparse_heat

target = build_sparse_heat_target(X, SpectralConfig())
prepared = target.prepare(order=16)  # predeclared; no clean-label selection
loss = prepared.loss(Y - X)         # gradients through the live residual only
report = measure_sparse_heat(X, residual, SpectralConfig())
```

[Code] Graphs expose `active_indices`, sparse normalized `laplacian`,
`mean_active_degree`, and `upper_bound`. Prepared filters expose coefficients,
order and beta. An exact comparator can reuse the existing
`build_probe_spectral_target(X, config)` and its smooth beta-2 loss on the same
X/residual. Actual GPU parity must verify the supports/graph before interpreting
the scalar loss/gradient differences; independently rebuilding a graph is not
a substitute for this check under native numerical variation.

[Code] `measure_sparse_heat` evaluates each K once, using a detached residual
clone and `autograd.grad`, leaving caller gradients and RNG untouched. It
synchronizes CUDA around setup, loss and residual-VJP timing; coefficient
calculation is timed on CPU. Setup includes graph construction and conversion
to COO. Loss includes input validation/finite checks; backward includes only
the residual VJP, not decoder/sampler backward. CUDA memory reports absolute
peak allocated bytes and peak increase over that scope's starting allocated
bytes, resetting CUDA peak counters for setup and each K. CPU memory fields are
null. Target storage persists across K scopes; input cloning is outside loss
timing but inside each order's memory scope. These are cold diagnostic timings,
without warmup or repeated samples. They cannot establish throughput or an
end-to-end speedup.

## CPU evidence and remaining gate

[Observed] Five focused tests passed with
`/tmp/tta-lion-reencode-venv/bin/python -m unittest discover -s tests -p test_sparse_heat.py -v`.
The initial run failed because the new helper module was absent; an added
contract test subsequently caught noninteger `8.0` order validation before its
repair. Float32/float64 parity includes duplicate/tied points, gamma 0/.6/3/100,
and empty active support. Existing U and normalized q reconstruct the expected
Laplacian. A patched `torch.linalg.eigh` raises if the eigenfree builder calls it.
Tests also check detached graph state, beta/order/dtype validation, batch-sum
scaling, connected zero loss, exact zero isolate gradients, RNG preservation,
and the timing report's scopes. Both new Python files parse with Python 3.8
grammar; actual Python 3.8/PyTorch 2.1.2 execution remains a deployment check.
The complete first-party unittest discovery also passed: 530 tests in 45.339s,
one skipped. Its output includes the existing optional `knn_cuda` warning;
no CUDA execution is implied by this CPU run. `git diff --check` passed.

[Observed] The float64 six-vertex toy has five active vertices, 17 sparse
Laplacian entries and upper bound 2.646421334486625. Against the existing exact-U
beta-2 heat objective on that same graph:

| K | absolute loss error | gradient L2 error |
|---|---:|---:|
| 8 | 8.599709833e-8 | 3.895430732e-8 |
| 16 | 1.998401444e-15 | 9.317516143e-16 |
| 32 | 3.108624469e-15 | 1.454985290e-15 |

[Inference] Increasing K beyond the float64 roundoff floor need not reduce the
observed error monotonically. The meaningful order sensitivity is K8 to K16;
no K is selected by clean labels. This toy neither certifies K16 on real graphs
nor demonstrates a useful downstream guidance direction.

[Open] The parent must still run matched actual-GPU graph/objective/gradient
gates for all predeclared K values, check native sparse backward compatibility,
and measure warm loss/VJP and common graph setup costs separately. Any claimed
operational benefit additionally needs encoder, graph, decoder, PM and sampler
costs on matched workloads. No sparse-kNN, full-sampler, accuracy or end-to-end
speed claim follows from this implementation.

## Standalone H4 probe preparation — later CPU-only batch

[Code] `scripts/probe_sparse_heat.py` consumes one actual H5 `h5_geometry.npz`
through required geometry SHA and pinned asset-manifest SHA arguments, using a
fresh output directory. It reads prepared X and the four already-computed
decoded geometries; no model is initialized. Native CUDA input must retain
float32 N2048. An explicit CPU device supports small primitive tests and is
labeled `gpu_executed=false`. CUDA execution verifies the actual imported
graph/helper/probe bytes against the pinned asset manifest's file map. Source
hashes, sealed source commit/overlay receipt, available checkout HEAD, torch
native core identity, versions and actual backend flags are recorded. Deployment
overlays can lack `.git`; the pinned manifest remains the source binding.

[Code] `protocol.json` is written before input tensor inspection. All provided
checkpoint/object pairs are compared at beta2 for K8/16/32. Gates require both
per-object and batch-summed relative loss errors <=1e-6, and per-object vertex
gradient relative L2 error <=1e-6. Relative denominators are explicitly floored
at 1e-12 in both cases; isolate gradients must be exactly zero. Every K remains
in the report; the first passing K is only a technical cost candidate for this
input. No passing K yields `equivalence_gate_failed`; timing remains descriptive.
There is no tolerance adjustment, clean-label access, accuracy result, or
rewrite of the earlier failed CPU filter reconstruction evidence.

[Code] The exact comparator uses the production `build_probe_spectral_target`
U/q values and its beta2 weights, evaluating the quadratic directly on the
same residual to avoid an artificial `(X+E)-X` rounding discrepancy. The probe
captures the actual raw matrix passed to production `eigh` and its unclamped
eigenvalues in an untimed diagnostic build. It requires exact support, mean
degree and normalized raw-L equality against the sparse setup. It saves native
dtype U, q, unclamped eigenvalues, raw L, normalized L, sparse indices/values,
prepared X and residuals **before** polynomial evaluations. Each successful
checkpoint/K also saves exact/approximate losses and vertex gradients.
Orthogonality, `Lbar U-U q`, q-clamp differences and row-sum diagnostics remain
separate from polynomial-on-q epsilon and its ideal theoretical bounds. The
native numeric gate includes all these effects; the ideal bounds do not absorb
them or certify native float32 equivalence.

[Code] B1 and B32 timing inputs cycle deterministically through the provided
objects, recording input positions and original IDs; timing uses the first
captured residual. Each exact and sparse fresh setup uses two warmups and three
measured builds. Exact setup reports recorded eigensolve time and remaining
setup time; sparse setup reports an additional validation scope. Cached exact
U/q weights or sparse coefficients are prepared separately, then each arm has
three warmups and ten measured loss/residual-VJP pairs. CUDA scopes synchronize
and report both absolute peak allocated bytes and increase from starting
allocated bytes. Input replication/cloning and source/artifact serialization
are outside filtering latency. Both operators coexist in the filtering memory
baseline. Neither graph construction nor complete model inference is sparsified
or included in these filtering timings. An actual whole-pipeline/trajectory
stage remains conditional on the equivalence gate.

[CPU] Four focused probe tests passed after the initial missing-module failure;
combined helper/probe checks cover the graph/scalar/gradient/isolate contract,
all-K failure under a 1% quadratic perturbation, graph evidence survival after
a simulated native failure, strict input hash rejection, no output overwrite,
non-git deployment metadata, deterministic timing replication and sample counts.
The new script/test parse under Python3.8 grammar. No GPU/model execution was
performed, and the complete suite was not rerun for this isolated probe batch.
Combined helper/probe discovery passed nine tests in 1.184s; CLI `--help` and
`git diff --check` also passed.

[Code/CPU/Review] A later review caught that a B32 setup failure could discard
already-completed B1 timings because the benchmark returned both batches only
at the end. The probe now writes a distinct JSON receipt after each completed
exact/sparse setup, filtering arm and batch summary, immediately indexing its
bytes/SHA in the incremental report. A forced second-batch OOM regression first
failed on the missing progress index, then passed with all seven B1 receipts
preserved and hash-verified. Combined helper/probe checks now pass ten tests in
1.300s. Python3.8 grammar and whitespace checks pass. The capture sibling
independently accepted the small callback repair and reran that regression
(one test, 0.139s). No GPU or additional full-suite run was performed.

[Code/CPU] Final deployment review required explicit controlled native settings,
rather than recorded Torch defaults. The CUDA CLI now requires an uninitialized
CUDA context, then sets matmul TF32=false, cuDNN TF32=false, benchmark=false,
cuDNN deterministic=true and deterministic_algorithms=false before reading
geometry or allocating device tensors. Protocol/runtime JSON retain the recipe
and observed flags plus `configured_before_cuda_initialization=true`. CPU CLI
behavior is unchanged. A new ordering witness first failed on the former cuDNN
defaults, then passed; it forbids CUDA lazy initialization during configuration
and verifies rejection after initialization. Combined helper/probe checks now
pass eleven tests in 1.447s, with Python3.8 grammar and whitespace checks. No GPU,
production-source change, extra CLI flag or additional full-suite run occurred.

```sh
python scripts/probe_sparse_heat.py \
  --geometry /path/to/h5_geometry.npz --geometry-sha256 GEOMETRY_SHA \
  --asset-manifest /path/to/assets/manifest.json --asset-manifest-sha256 MANIFEST_SHA \
  --output /fresh/h4_probe --device cuda
```
