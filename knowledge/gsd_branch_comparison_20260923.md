# GSD branch comparison and null-pilot diagnosis — 2026-09-23

## Scope and evidence

[Code] Compared checked-out `gsd-development` at
`761f47fc6e927f4c71b3a1ea1cc97da0926ba065`, `dev` at
`a458cd46bba1288a3f41302548bc777700cbc7d4`, and
`pxp-gradient-projection` at `53ba252519c7cf65f836a9c1c564027142ab1573`.
Historical files were read using `git show`; no checkout or implementation
change was made. The project combines frozen-classifier 3DD-TTA/LION input
adaptation with GSD-inspired latent spectral fidelity, not full GSDTTA.

[User report] The old scripts improved accuracy with defaults such as
M=400 and mean spectral weight=16. Complete legacy run commands, per-corruption
counts, seeds and environment are not archived here. The table below describes
`eval_gsd_tta.py` defaults, not inferred historical runtime configurations.

## Verified differences

| Setting | Current GSD v1 | dev eval GSD | pxp branch eval GSD |
|---|---|---|---|
| Batch size | 32 | 70 | 70 |
| Normal/background reverse steps | 5 / 35 | 10 / 30 | 10 / 30 |
| Low band | requested M=100, weight=1 | M=400, weight=16 | M=400, weight=16 |
| Mid band | absent | indices 400:600, weight=2 | indices 400:600, weight=2 |
| Spectral reduction | per-sample mean over modes/XYZ, sum over batch | mean over batch/modes/channels | same as dev |
| SCD | sum, weight=1 by default | sum, weight=1 | sum, weight=1 |
| Retention lambda | .95 | .95 | .95 |
| Graph | static, explicit 10 nonself neighbors | static, KNN_CUDA self-query without explicit self exclusion | same graph as dev; optional dynamic updates |
| Threshold | .6 * sum(directed A)/(N*k) | .6 * sum(symmetric A)/N | same as dev |
| Isolates | removed from eigensystem, embedded with zero rows | diagonal penalty 1000, jitter 1e-5 | same as dev |
| Band boundary | expands unresolved boundary eigenspace | exact index slice | exact index slice |
| DDIM set_alpha_to_one | True (recorded in pilot) | default True | explicit False |
| Final decoder style | original encoded global latent | original by eval-script default | updated by eval-script default |
| Update rates | gamma local / eta style | eta local / gamma style | eta local / gamma style |
| LION mode | raw/eval, EMA off | no explicit VAE/prior eval call | no explicit VAE/prior eval call |
| Projection in tta_gsd.py | none | none | none |

[Code] Source anchors: current `graph_spectral.py:60-71,79-148`,
`tta_gsd.py:59-65,76-99,129-167`, `gsd_protocol.py:9-11,35-38,63`;
legacy `eval_gsd_tta.py:19,40-55`, `graph_spectral.py:60-103`;
dev `tta_gsd.py:135-172`, pxp `tta_gsd.py:45,122-126,186-223`.
Both legacy graph files are byte-identical. Both use the same correct
`B x 2048 x 4` reshape as current code. There is no evidence for a newly
introduced transpose error.

[Code] Entry point matters: legacy `main_gsd_tta.py` defaults to SCD weight=0,
M=100, spectral weight=1, batch=40 and 5/35 steps. Legacy
`eval_gsd_tta_fast.py` defaults to batch=2, M=240, spectral weight=16 and
actually chooses 5/35 steps in its loop. These cannot be treated as the
M400/batch70 eval protocol. PxP projection exists only in separate
`tta_pxp*.py` paths; the branch name does not activate it in `tta_gsd.py`.

## Archived current results

[Run] Recomputed directly from the 18 pilot ZIPs in
`result/modelnet40_c/gsd_latent_spectral_v1/`, at seeds 0/1/2, Gaussian and
Impulse severity 5, all 2,468 examples per corruption. Each comparison is
against its own spectral-weight-zero run. This is a two-corruption pilot,
not an all-15 benchmark or a common-draw paired experiment.

| Requested M | On-minus-off deltas at seeds 0/1/2 (pp) | Mean ± sample SD (pp) |
|---|---|---|
| 100 | -0.0608, +0.1418, -0.1216 | -0.0135 ± 0.1379 |
| 240 | -0.4660, +0.1013, -0.1418 | -0.1688 ± 0.2846 |
| 400 | +0.0608, -0.0810, -0.1418 | -0.0540 ± 0.1040 |

[Run] From `config.json.gsd_diagnostics`, seed 0 batch-step means:

| M | Corruption | Local SCD norm | Local spectral norm | Ratio of mean norms |
|---|---|---:|---:|---:|
| 100 | Gaussian | 113.4171 | .094994 | .0838% |
| 100 | Impulse | 104.1047 | .100761 | .0968% |
| 240 | Gaussian | 113.4178 | .064068 | .0565% |
| 240 | Impulse | 104.1063 | .063200 | .0607% |
| 400 | Gaussian | 113.4172 | .048003 | .0423% |
| 400 | Impulse | 104.1046 | .046847 | .0450% |

[Run] Seeds 1/2 closely reproduce these norm ratios. At M100, style-gradient
ratios are approximately .164% Gaussian and .0905% Impulse (seed 0).
These are ratios of aggregate mean norms, not mean per-step ratios, cosine
measurements or bounds on classifier sensitivity.

[Inference] The active spectral term is tiny relative to SCD; a near-null
incremental effect is plausible. This is positive evidence against a completely
disconnected spectral gradient, not proof that increasing its weight improves
classification. Larger M does not necessarily strengthen guidance: the
denominator grows, and the measured spectral norm falls by about half at M400.

[Run] M100 actually selects mean ranks 138.43/120.64 for Gaussian/Impulse
(seed 0); M400 selects 404.53/403.71. The graph excludes approximately
129/111 vertices. Reference energy retained is 91.98%/92.73% at M100 and
92.72%/93.44% at M400. A large reference energy fraction is not evidence that
the retained components are clean, or that the gradient helps classification.
The reported numerical zero-mode count is not an exact connected-component count.

## Loss-scale comparison

[Code/Inference] For identical graph, selected rank M, XYZ signal and batch B,
write `E=sum_b ||U_b.T (Qhat_b-Qref_b)||^2`. Legacy weighted low-band loss is
`w_old * E/(B*3*M)`; current weighted loss is `w_new * E/(3*M)`.
Consequently `w_new=w_old/B` matches the old low-band scale for a fixed batch.
Old weight16 maps to current .5 at B32 or approximately .2286 at B70;
current weight1 maps to old weight32 at B32. Therefore **old weight16 was not
automatically stronger than current weight1**. The graph, rank, mid band and
trajectory also differ, so these are algebraic scale equivalents, not full
algorithm equivalence. Legacy last partial batches have stronger per-sample
spectral guidance relative to summed SCD than full batches.

[Code/Inference] Replacing M400 by M600 in the current single-band objective
would not reproduce old low+mid behavior. Old low coefficients carry
16/(B*3*400), whereas mid coefficients carry 2/(B*3*200), i.e. one quarter
of the low coefficient weight. Current one-band loss weights all selected
modes equally before the projection.

## Ranked explanations and limitations

1. [Run/Inference] **Current weak incremental guidance:** norm measurements
   directly explain why adding weight1 hardly changes the update. They do
   not by themselves explain why legacy runs improved: legacy also used
   batch-mean spectra, and no legacy gradient diagnostics exist.
2. [Code/Inference] **Different denoising trajectory:** old default 10/30
   versus current 5/35 changes both initial noising strength and reverse-step
   count. A legacy on-versus-original gain may persist with all spectral
   weights zero. This is the first confound to isolate.
3. [Code/Inference] **Different graph and multiband objective:** old threshold
   is much stricter in its normalization; holding A fixed makes the factor
   k=10, but actual adjacency and degree conventions also differ, so the
   implemented thresholds are not exactly related by ten. A plausible
   legacy advantage is stronger rejection of corrupted anchors plus mid-band
   part structure. Neither mechanism is established without a matched graph
   ablation. KNN_CUDA distance units/self-neighbor behavior of the actual
   legacy wheel remain unverified; do not assert a fourth-power kernel bug.
4. [Run/Inference] **Stronger present comparator:** the prior controlled
   raw-LION train/eval screen improved baseline mean by .7577 pp at seeds
   1/2 (`dropout_eval_mode_20260913.md`). A method that helps a dropout-active
   baseline need not help the eval baseline. This is not a direct numerical
   comparison of the unarchived old GSD run with current GSD.
5. [Code/Run/Inference] **Secondary, branch-specific factors:** updated final
   style and `set_alpha_to_one=False` are pxp-eval differences, not shared
   dev-eval differences. Existing shared-trajectory style all-15 controls give
   -0.0315 pp mean, weakening a generic style-only explanation (interactions
   with legacy GSD remain open). The rate-name swap has no numeric effect
   when gamma=eta=.01. Lambda is .95 on all three compared GSD paths.
6. [Open] Legacy stochastic variation, small-prefix tuning and sparse/resumed
   CSVs can confound comparisons. The user's gain is retained as a report;
   it is neither disproved nor established as an isolated spectral benefit.

[Code] In archived diffusers 0.11.1 the alpha flag changes the final-step
previous-alpha convention; it does not change the whole beta schedule.
See [versioned DDIM source](https://raw.githubusercontent.com/huggingface/diffusers/v0.11.1/src/diffusers/schedulers/scheduling_ddim.py).

## Smallest discriminating Colab sequence

[Inference] First recover the successful legacy command. Under matched assets,
batch, LION mode, preprocessing, seeds 0/1/2 and common random draws, evaluate:

1. Original SCD host.
2. The successful legacy GSD host with **all** spectral low/mid/high/invariant
   weights zero and SCD weight retained at 1; preserve its steps/style/scheduler.
3. The same legacy host with its original M400/weight16 plus mid-band defaults.

`2-1` measures non-spectral host changes; `3-2` measures the legacy spectral
increment. If `3-2` is null, the original gain is not evidence for the spectral
loss alone. If positive, port one difference at a time into the current host:
graph policy first, then low/mid objective with batch-adjusted weights; retain
the original host control. Keep mode interactions separate.

[Inference] For current scale diagnosis, first evaluate frozen-state virtual
updates/norms and alignment under declared weights; then choose a limited
validation-only weight comparison. Do not jump to weight1000 or interpret a
loss-scale match as an optimal guidance setting. Raising M alone has already
failed to improve the pilot mean. The new spectral-only ablation has no
archived result and would remove SCD, so it answers a different question.

[Open] Gaussian/Impulse can reject a promising configuration locally but
cannot establish an all-corruption null result. Preserve the successful
legacy per-corruption pattern when choosing validation scope, and lock the
configuration before any final all-15 benchmark.

## Memory corrections and verification boundary

[Code] `3955f7d` briefly locked GSD lambda to .96; `509b901` restored .95.
Current source and pilot configs both confirm .95. Historical .96 and
GPU-pending headlines in knowledge are superseded by this dated audit and
the existing run entries. M400's -0.0540 pp is **more**, not less, negative
than M100's -0.0135 pp. Neither establishes a meaningful difference in benefit.

[Code/Run] This task reads branch sources and existing artifacts, recomputes
pilot deltas/diagnostics, and updates research memory only. No local GPU
evaluation, new accuracy experiment, source edit, commit or branch change.
Complete legacy seven-file run ZIPs remain the evidence needed to attribute
the historical improvement.

## Follow-up: spectral mathematics rather than gradient magnitude

[User report] Legacy spectral gradients were already small, yet accuracy
improved by almost one percentage point. PxP-inspired projection was introduced
to address gradient imbalance/conflicts. This strengthens the need to separate
direction/subspace quality from norm; small gradients alone are not an adequate
explanation of the difference between old and current results.

### Exact objective and what a weight can change

[Code/Inference] For one sample let `E=Qhat-Qref` and `P=U_m U_m.T`.
Current loss is `||U_m.T E||_F^2/(3*m)`, with clean-prediction derivative
`2*P*E/(3*m)`. Frozen-denoiser Jacobians then map this into local/style gradients.
The old same-basis coefficient subtraction is algebraically equivalent to
projecting the difference. No sign error or lost derivative was identified.
Shared eigenvector sign changes or rotations within a complete selected band
cancel in P; they do not by themselves break the loss. PyTorch's warning about
backpropagating through near-repeated eigenvectors is not the gradient route
here, since graph/basis construction is detached. See the
[PyTorch 2.7 eigh documentation](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.eigh.html).

[Inference] Changing a scalar weight scales P; it cannot generally replace
P with a different projector, delete newly admitted modes while retaining old
ones, or reproduce the legacy low/mid weighting. Consequently recovering an
accuracy gain by tuning is possible, but recovering the old objective using
only current M and weight is generally impossible.

### New numerical finding: conservative boundary expansion can mix distinct modes

[Code] Current `graph_spectral.py:123-144` sets the absolute boundary allowance
to at least `2*eps(dtype)*active_vertices*||L||_infinity`, anchored at the
requested boundary eigenvalue. This is a conservative heuristic, not a measured
eigenpair error. Lowering only `eigenspace_atol` does not bypass this floor.

[Code: CPU numerical diagnostic, not model evaluation] On PyTorch
2.7.1+cu126 using CPU only, seed314159, `x=.6*randn(1,2048,3)` in float32,
`SpectralConfig(modes=100)`, k10/delta.1/gamma.6 defaults and two CPU threads:

- active vertices=1921, excluded=127; actual rank=104;
- implemented boundary tolerance=.0071692473;
- float64 eigendecomposition of the **same float32 Laplacian** gives an
  eigenvalue-100-to-101 gap of .000217711 (one-based indices);
- eigenvalue104 minus eigenvalue100 is .005910085;
- maximum float32-versus-float64 eigenvalue difference is .0000100318;
- maximum float64-evaluated float32 eigenpair residual norm is .0000196301;
- the float64 reference with atol1e-7/rtol1e-5 retains 100 modes.

Thus this concrete case includes distinguishable additional modes rather than
only protecting an unresolved repeated eigenvalue. This demonstrates a
mathematical behavior of the heuristic, not a CUDA accuracy regression.
For a prediction error along added mode101, the unscaled `P*E/m` norm is
approximately 6.47e-9 under fixed M100 versus .0096154 under expanded M104.
A scalar loss weight cannot reconcile these projectors for all errors.
The analytic current-loss gradient agreed with autograd to max error 1.09e-10.

Reproduction uses the actual implementation: patch `torch.linalg.eigh` only
with a wrapper that captures its input and returned eigenpairs, then calls
the real function. Build the target once from the seeded CPU cloud. Apply
`torch.linalg.eigh(captured_L.double())` for the reference and compute
`||L.double()@U.double()-U.double()*eigenvalues.double()||` per column.
This avoids rebuilding a different graph in float64.

[Run] Real seed0 M100 artifacts select mean ranks 138.43 Gaussian and 120.64
Impulse, maxima 299 and 273. Their effective boundary tolerances average
.008270 and .007535. Different spectra change both the projector and the
sample coefficient `1/m`: rank299 receives about one third the per-mode
coefficient of rank100. These aggregates do not prove that every expansion
is excessive; actual latent/Laplacian captures and float64/residual checks are
needed. At M400 the mean ranks are only 404.53/403.71, so expansion alone is
a weak explanation of the persistent M400 null result.

### Low graph frequency need not mean reliable global shape

[Code/Inference] Current filtering is exactly
`directed_degree > (graph_gamma/k)*mean_directed_degree`, hence a **6%**
mean-degree cutoff at gamma.6/k10. Legacy is a 60% mean **symmetric** degree
cutoff with different self-neighbor conventions. A current gamma=6 diagnostic
would make it 60% of directed mean degree, but would not reproduce the legacy
graph exactly or establish an optimal parameter.

[Inference] With combinatorial `L=D-A`, a weakly connected small component
can occupy low frequencies. For a nonself graph the coordinate indicator has
Rayleigh quotient `e_i.T L e_i/(e_i.T e_i)=degree_i`; centering it against DC
gives `degree_i*N/(N-1)`. Therefore low energy can reflect weak connectivity,
not semantic importance. The new weak cutoff can admit corrupt sparse regions
whose low modes then receive fidelity guidance. Removing exactly isolated
vertices does not remove weakly connected groups or disconnected multi-node
components. A disconnected component has a constant zero mode that can anchor
its corrupted mean. This is a plausible direction-quality failure mechanism,
not a measured identification of outlier modes in the real pilot. Numerical
zero-mode counts of 78/55 must not be presented as exact component counts.

Falsifier/diagnostic: on the same encoded clouds, compare actual connected
components, boundary eigenpair residuals, projector diagonal versus degree,
and spectral-gradient directions under current versus legacy graph rules.
Then test graph-rule-only accuracy with host/bands/scales fixed.

### Why a small spectral gradient can have a large PxP effect

[Code/Inference] The legacy one-way projection uses, on negative dot product,
`g_c' = g_c - dot(g_c,g_s)/(||g_s||^2+1e-8)*g_s`.
Ignoring epsilon, rescaling nonzero g_s cancels from the removed component.
Thus a small spectral vector can substantially redirect a large SCD vector.
With epsilon, this approximate invariance requires `||g_s||^2 >> 1e-8`.
A CPU two-vector check with `g_c=(-1,1)` gives SCD changes of .999999 and
.990099 for spectral norms .1 and .001 respectively. This establishes scale
behavior, not that projection improved the archived benchmark. The direction
must be useful; amplifying a corrupted spectral direction can be harmful.

### Decision

[Inference] Do not diagnose a fundamental failure of spectral guidance from
the present null pilots or prescribe a blind large-weight sweep. Prioritize
same-latent spectral diagnostics and graph-only comparison; audit the band
expansion against a higher-precision reference while preserving genuinely
repeated eigenspaces. Separate any boundary-policy change from its `1/m`
normalization change. Restore/test the legacy two-band metric under matched
host conditions if the legacy spectral increment survives. Tune weight only
after specifying which projector/metric it weights. No implementation change
or new Colab/model accuracy experiment was performed in this follow-up.
