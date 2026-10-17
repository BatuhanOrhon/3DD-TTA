# Spectral and trajectory mathematics audit - 2026-10-08

## Scope and conclusion

[Code/CPU/Inference] Compare (A) established `graph_spectral.py` /
`tta_gsd.py`, including hard-v1 used in Phase6 and calibrated smooth-v2,
against (B) restored `graph_spectral_v2.py` / `tta_gsd_v2.py` from
smooth-integration5c3ed883. Main base remainsc18926f plus the uncommitted
restoration described in [the handoff](gsd_legacy_smooth_restore_20261008.md).

**A real legacy isolate-detection error is demonstrated.** The GFT direction,
latent layout, main weighted spectral objective and its gradient are correct.
Additional findings are hard-cutoff basis dependence, a conservative current
rank-expansion policy, a different DDIM terminal boundary, an inactive loss
logging mismatch, and an unguarded degenerate preprocessing case. These are
classified separately below; none demonstrates the cause of the accuracy gap.

No production source, source overlay, raw ZIP, original CSV or historical
result was modified in this audit. Only derived CPU probes and dated research
records were added. Restoring old code exactly can restore an existing defect:
the preceding51 port tests established their stated contracts, not universal
mathematical correctness. The new isolate witness is a deeper property check.

## 1. Confirmed active legacy defect: self-only vertices evade isolation penalty

[Code] `graph_spectral_v2.py:59,76,88,95` queries KNN against the same cloud,
keeps diagonal edges, computes D from the full row sum, and detects isolates
with `D_o_diag == 0`. A self-neighbor produces A_ii=1. For a retained vertex
with no off-diagonal edge, D_ii=1 while L_ii=D_ii-A_ii=0. Consequently its
penalty is absent and its eigenvalue stays near the1e-5 jitter.

This is not an invalid matrix multiplication or a loss of Laplacian symmetry.
It is an incorrect test for isolation from **other** vertices, defeating the
stated intention to move such vertices to high frequencies.

[CPU] Executed the actual graph forward method with a deterministic CPU KNN
adapter returning Euclidean distances, including self. N2048, k10, delta.1,
gamma.6, beta2, M800 match the requested structural settings. The deliberately
sparse cloud contains one connected pair and2046 off-diagonal isolates:

| Observation | Result |
|---|---:|
| True off-diagonal isolates |2046|
| Isolates receiving penalty1000 |0|
| Their operator diagonal |~0.00001|
| Isolates with spectral-filter diagonal >.5 |800|
| Maximum such spectral weight |0.99997997|
| Maximum weight after offline corrected isolate mask |0|

The offline counterfactual added1000 only to genuinely isolated vertices,
keeping all other operator settings. It is not a production fix. A small
four-node counterpart confirms that the current nonself graph excludes these
isolates (zero filter rows).

[Open] This establishes a reproducible mechanism, not its frequency in real
LION latents, installed CUDA behavior or accuracy effect. Old raw latent
graphs were not supplied. Do not claim this explains64.5327% or Phase6's null.

[Inference/fix direction] The minimal correction would detect zero
off-diagonal degree, e.g. `sum(A_o)-diag(A_o)`, with a declared numerical
tolerance. Removing self edges before all graph calculations and requesting
k actual nonself neighbors is a broader graph-policy change; isolate it from
that minimal correction. Keep the exact legacy reference separate from a
future corrected variant so historical reproduction remains interpretable.

## 2. Legacy cutoff can split a degenerate eigenspace

[Code/Math] Legacy weights are w_i=exp(-beta*lambda_i) for i<M_max and zero
afterward. If lambda_(M-1)=lambda_M, equal-eigenvalue vectors receive unequal
weights. A rotation of the eigensolver basis within that tied space can then
change the loss even though the graph itself is unchanged.

[CPU] With eigenvalues(0,0,2), truncating at1, two valid eigenbases of the
same matrix produce losses1/9 and0 for the same residual. A complete
equal-weight eigenspace rotation preserves the loss. In the isolate witness,
the numerous near-zero modes make this risk particularly relevant.

[Inference] This is ambiguity in an eigenvalue-defined cutoff, not a wrong
eigendecomposition. A fixed returned basis still defines a valid PSD quadratic
loss. Practical instability depends on actual boundary multiplicity/gap and
backend. Current boundary expansion addresses that ambiguity, with its own
trade-off below. Eigenvector sign flips alone are harmless in both losses.

## 3. Current hard-v1 expands beyond distinct nearby modes

[Code] `graph_spectral.py:163-184` uses roughly
2*eps*N_active*||L||_infinity as an absolute boundary allowance. This explicitly
allows nearby numerically unresolved modes; it is not a promise of exact M.

[CPU] Repeated the earlier seed314159 float32 witness, N2048, M100:

- Actual selected rank104; active vertices1921.
- Effective boundary absolute tolerance0.0071692473.
- Float64 eigenvalue100-to101 gap on the same float32 Laplacian0.000217711.

Thus the policy can include distinct nearby modes. It also changes hard loss
normalization from1/(3*100) to1/(3*104). This is a documented conservative
numerical policy, not proof of an algebraic error or inaccurate CUDA eigenpairs.
It affects hard-v1/Phase6 and hard comparators; the calibrated **smooth** filter
already uses all active modes and does not use M as its heat-filter cutoff.
`zero_mode_count` similarly counts numerical near-zero modes under a tolerance,
not proven connected components. Preserve earlier rank diagnostics accordingly.

## 4. Main spectral feature/loss equations are consistent

Let Q be the encoded latent XYZ, Y the denoised prediction, E=Y-Q, and U an
orthonormal eigenbasis of the fixed reference operator.

| Path | Loss per example | Prediction-space derivative |
|---|---|---|
| Current hard-v1 | ||U_r^T E||_F^2/(3r) |2 U_r U_r^T E/(3r)|
| Current smooth | tr(E^T F E)/(3N), F=U diag(exp(-beta*lambda/mean_degree)) U^T |2FE/(3N)|
| Legacy smooth | sum_i w_i ||(U^T E)_i||^2/(3N), w_i=exp(-beta*lambda_i) below M |2U diag(w) U^T E/(3N)|

[Code/CPU] `U.T @ Y - U.T @ Q` equals `U.T @ (Y-Q)`. Transpose orientation
is correct; there is no missing inverse transform required to compare GFT
coefficients. With identical U, weights and denominator, coefficient and
operator implementations agree: loss error0, maximum gradient error2.17e-18,
directional finite-difference error2.89e-13 (float64 probe).

Weights multiply **squared** coefficient residuals. This equals
||diag(sqrt(w)) U^T E||^2, not ||diag(w) U^T E||^2. The latter would double
the exponential decay rate; it is a different objective, not a required fix.
Reference/basis detachment is consistent with a fixed reference objective;
gradients through the frozen denoiser still reach local and style variables.
Summing over batch preserves each example's guidance magnitude when examples
are independent; diagnostics called mean_raw_* average batch losses, so their
numerical values still depend on batch cardinality.

[Inference] Both objectives penalize deviations from the **corrupted encoded
reference**, especially in low graph frequencies. They are not direct measures
of clean geometry or a universal instruction to smooth the predicted cloud.
Low frequency can describe a weak/disconnected corrupt component. This is an
objective-design limitation, distinct from the concrete isolate-mask bug.

## 5. Distance units and graph threshold

[Code/Open] Legacy squares KNN distances before the Gaussian kernel. The
accessible primary [KNN implementation fork](https://raw.githubusercontent.com/unique1i/knn_cuda/master/knn_cuda/csrc/cuda/knn.cu)
explicitly takes a square root before returning distances. For that
implementation, squaring here correctly produces squared Euclidean distance.
The old unlimblue URL returned404 during this audit and the exact historical
installed binary is unavailable. There is **no evidence for a fourth-power
distance bug**; verify the runtime wheel before making that claim. The current
cdist(...).square() path is unambiguous.

[Code/Math] Current threshold is gamma*sum(A_directed)/(N*k), compared to
the row degree. At gamma.6/k10 this is6% of the mean directed degree, not60%.
Legacy uses60% of mean symmetric degree, including self entries. Both are
well-defined thresholds, but the same gamma does not mean the same filter.
Calling current gamma.6 a60%-of-mean-degree cutoff would be incorrect.

## 6. Latent layout, style and gradient composition

[Code] `models/latent_points_ada.py:215` concatenates per-point XYZ and
feature channels before flattening. The actual YAML has input_dim3 and
latent_dim1. `view(B,2048,4)[:,:,:3]` is therefore correct, consistent with
the local prior and decoder reshapes. No transpose/channel-interleaving bug.

[Code] The YAML has `style_mlp: ''`, so global2style is identity for this
configuration. No double-MLP or wrong-space decode is established. Different
future checkpoints with a nonidentity style MLP require a separate contract.
Style is an optimized conditioning vector here; the global diffusion prior
is not executed. Final original versus updated style is an explicit choice.

[Code/Math] Current original-style decode makes the **last** style update
ineffective on the returned point cloud: there is no subsequent prior call to
consume it. Earlier style updates do affect later local predictions. Legacy
updated-style decode consumes the final update as well. This is an inert
terminal update/diagnostic detail, not proof that style guidance is useless.

[Code/CPU] SCD uses two sorted nearest-neighbor squared-distance lists,
retaining floor(.95*2048)=1945 in each direction, then sums. Native Chamfer
source explicitly computes squared distances. Nearest-neighbor/sort ties
are nonsmooth boundaries, not missing backpropagation by themselves.
The update subtracts gradients, with chain rule through predicted x0. It is
a guided DDIM heuristic, not guaranteed descent of classifier error.
Current symmetric projection is per example, uses both original gradients,
and a nonnegative norm cap; its algebra/tests do not show a sign or batch-axis
error. Exact opposing directions can cancel, as the formula implies.

[Code] Legacy docstrings swap gamma/eta descriptions relative to its actual
local=eta, style=gamma updates. With both=.01 this cannot affect the supplied
run. Unequal-rate future tests must follow the actual contract.

## 7. DDIM terminal boundary is different, not a sign error

[Code/Math] The [pinned Diffusers0.11.1 scheduler](https://raw.githubusercontent.com/huggingface/diffusers/v0.11.1/src/diffusers/schedulers/scheduling_ddim.py)
uses alpha_prev=alpha_0 at terminal timestep0 when set_alpha_to_one=False.
With deterministic eta0, no clipping, epsilon prediction:

`x0_hat=(x-sqrt(1-alpha0)*epsilon)/sqrt(alpha0)`

`prev=sqrt(alpha0)*x0_hat+sqrt(1-alpha0)*epsilon=x`.

The legacy last scheduler transition is therefore identity (up to roundoff),
though the final guidance update still runs. True advances to the clean
endpoint instead. The synthetic equation check returns exact identity.
False is an allowed boundary convention; calling it full terminal denoising
would be misleading. Its accuracy effect is unmeasured.

[Code] Both pass timestep+1 to LION. `utils/diffusion_pvd.py:44-59` uses
that1-based prior convention with0-based alpha lookup, so no off-by-one error
was found. YAML beta endpoints,1000 training steps and linear schedule match;
mixed_prediction is false. CLI eta is a guidance rate, not scheduler eta:
scheduler.step receives its default deterministic eta0 in both implementations.

## 8. Inactive and degenerate-input defects

- [Code] Legacy invariant diagnostic uses M_max, but differentiated invariant
  loss uses graph_spectral_module.M (default100). That is a real logging versus
  applied-objective mismatch when invariant weight>0. The supplied weight is0
  and the restored wrapper rejects enabling it; it does not explain this run.
- [Code/CPU] Shared normalize divides by max extent without a zero guard.
  A constant cloud produces NaNs. Nondegenerate normalization/inverse and
  rotate/rotateback round trips pass. Bounding-box centering means max positive
  coordinate equals the maximum absolute extent; absence of abs() there is
  not itself an error. Zero-extent prevalence in actual data/outputs is unknown.
  This probe is not a diagnosis of the recorded Background100/1000 failures.

## Verification and decision

[Code/CPU]85 existing/new CPU unit checks pass, plus the separate executable
counterexamples above. An independent reviewer confirmed the isolate defect,
basis-cutoff issue and mathematical equivalences, and clarified terminal style
and tolerance-policy interpretations. No GPU/model inference was performed.

Reproduce from the repo root:

```powershell
python result/modelnet40_c/diagnostics/spectral_math_audit_20261008/audit_math.py
python -m unittest tests.test_graph_spectral tests.test_graph_spectral_smooth tests.test_gsd_composition tests.test_gsd_composition_trajectory tests.test_gsd_trajectory tests.test_legacy_gsd_numerics tests.test_legacy_gsd_restore -v
```

Derived evidence: [results and source hashes](../result/modelnet40_c/diagnostics/spectral_math_audit_20261008/results.json),
[test log](../result/modelnet40_c/diagnostics/spectral_math_audit_20261008/cpu_tests.log),
[commands](../result/modelnet40_c/diagnostics/spectral_math_audit_20261008/commands.json).

[Inference/Decision] The isolate-mask defect deserves a separately named,
minimal corrected variant before broad tuning. It must not be silently folded
into the exact historical reproduction or combined with graph-neighbor,
threshold, band, decoder and scheduler changes. Current rank tolerance merits
residual-aware diagnosis if hard-v1 work resumes. Neither correction can be
promised to improve all15 accuracy, and no new GPU comparison is launched here.
