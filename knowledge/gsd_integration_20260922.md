# GSD-inspired latent spectral guidance: first integration

## Scope and evidence

[Code] Base: `79cc027` (`origin/baseline-repro-clean`, fetched 2026-09-22).
Work branch: `gsd-development`. Historical GSD inspected at `53ba252` and
`06561f3`; no historical implementation is imported wholesale.
[User report] This task authorizes GSD development and CPU unit/protocol tests;
it supersedes older notes parking GSD and discouraging new test files. GPU
evaluation remains Colab-only. PxP and gradient projection are outside scope.

## Paper/code audit

[Paper] 3DD-TTA Sec. 3.3, Eqs. 9--12 and Algorithm 1 (PDF pp. 4--5)
guide predicted clean local latents, update conditioning, normalize each
directed SCD sum by point count, and decode using the updated global state.
[Code] `tta.py` uses unnormalized summed SCD, gamma for local updates and eta
for style updates, and the original encoded global state for final decoding.
Operational retention remains .95 rather than the paper's .96 (Sec. 4.1).
These differences are preserved as explicit controls, not silently repaired.

[Paper] LION Sec. 3, Eqs. 5--7 (PDF pp. 3--4) defines a global vector and
point-structured local latent with XYZ and additional feature channels.
[Code] `models/latent_points_ada.py` concatenates point/feature channels before
flattening; its decoder and `latent_points_ada_localprior.py` reshape to
`B x N x (3+D)`. The spatial encoder mean has a residual input-coordinate
connection. The supplied contract is `B x 8192 -> B x 2048 x 4`.
[Inference] Latent XYZ is therefore the most direct graph domain for the
existing latent SCD. It is not identical to physical input coordinates.

[Paper] GSDTTA Sec. 3.1, Eqs. 7--13 (PDF p. 4) uses weighted graphs and
learnable low-frequency physical point shifts; Sec. 3.2 also adapts the model.
Eq. 10 visibly has gamma/(N*k). Eq. 7 displays d^2 while adjacent prose calls
d squared Euclidean distance; the distance convention is ambiguous in print.
[Code] Legacy graph code uses gamma/N after max symmetrization, undocumented
self-neighbor conventions, a fixed isolated-node diagonal penalty of 1000,
and a full dense eigendecomposition. Legacy GSD also changes scheduler
`set_alpha_to_one`, decoder style, update-rate mapping and batch loss scaling.
[Inference] A fixed graph fidelity objective fits frozen-classifier 3DD-TTA,
but does not implement GSDTTA's learned shift or alternating model adaptation.
The method is named **GSD-inspired latent spectral guidance**, ID
`gsd_latent_spectral_v1`.

## Selected mathematics

[Inference] Encode once: `(z_enc,h_ref)`, with `Q_ref = h_ref[..., :3]`.
Build a detached static graph per sample on Q_ref using exactly k non-self
neighbors, standard RBF `exp(-||qi-qj||^2/(2*delta^2))`, and explicit max
symmetrization. Compute the threshold from the directed adjacency as
`tau = graph_gamma * sum(A_directed)/(N*k)`; retain both endpoints only when
their directed weighted degree exceeds tau. These choices resolve the paper's
symmetry/distance ambiguities explicitly and are not a claimed reproduction.
Remove zero-degree vertices from the eigensystem and embed retained
eigenvectors back with zero rows there. Use the combinatorial Laplacian with
no arbitrary diagonal penalty. Keep M lowest modes, expanding the boundary
to include a numerically unresolved boundary eigenspace. Record actual ranks.
[Code] The absolute allowance is max(1e-7, 2*r), with
`r = eps(dtype)*active_vertices*||L||_infinity`, plus relative allowance
1e-5*abs(requested boundary eigenvalue). Zero-mode diagnostics use
max(1e-7,r). This conservative numerical heuristic can include nearby modes
and is logged; it is not a proof of exact eigenspace multiplicity.

[Inference] For sample b, selected orthonormal columns U_b and rank m_b:

`L_spec = sum_b ||U_b^T (Qhat_0,b - Q_ref,b)||_F^2 / (3*m_b)`.

An empty graph contributes zero and is logged. This sums over samples like
legacy SCD while normalizing by channel and selected mode count. The graph,
basis and target are detached; Qhat remains differentiable. Both signals
share the same basis, so sign flips and within-band orthogonal rotations
leave the objective unchanged. This retains DC as part of the selected band.

[Inference] At each original DDIM step, `L = L_SCD + w*L_spec` and

`h_prev = DDIM_prev(h_t, eps_theta(h_t,s_t,t)) - gamma*dL/dh_t`

`s_next = s_t - eta*dL/ds_t`.

Gradients traverse predicted-clean reconstruction and the frozen denoiser,
including its dependence on conditioning. No direct update to clean h0,
extra diffusion, classifier optimization or feature-channel metric is added.
Final decode uses `(h_final,z_enc)`. Weight zero dispatches the original
trajectory directly and performs no graph work/random draws.

## Alternatives and falsifiers

[Inference] Physical-coordinate graphs would add a cross-space correspondence
assumption. Dynamic graphs would add changing projectors and greater cost.
All-four-channel distances would combine geometry and an uncalibrated feature
metric. Defer each to an independent future ablation.
[Open] Low-frequency corrupted geometry may retain corruption or constrain
valid denoising; labels do not choose bands. No optimality or accuracy gain is
established. Test whether fixed low-frequency fidelity improves matched
three-seed results, and reject the hypothesis if gains are inconsistent,
gradients vanish, or runtime/memory is unacceptable.

## Ordered implementation batches

### Batch 1 - Static graph and objective

- Goal: deterministic detached graph and differentiable spectral fidelity.
- Scope/touched areas: `graph_spectral.py`, CPU graph tests only.
- Stack context: PyTorch CPU tests; real N=2048 graph execution in Colab.
- Dependencies: this design and baseline tensor audit.
- Implementation notes for $development-agent: explicit kNN/RBF units,
  isolated-node exclusion, degenerate-boundary expansion, per-sample sum.
- Verification: graph invariants, analytic gradients, basis invariance,
  degenerate/disconnected/duplicate cases, RNG and batch-scale checks.
- Knowledge artifact to update: this file, completion note.
- User review gate: existing task authorization; CPU tests before integration.
- Out of scope: dynamic graphs, PxP, GPU accuracy experiments.

### Batch 2 - Opt-in trajectory and artifact wiring

- Goal: GSD alone with unchanged existing method paths.
- Scope/touched areas: new `tta_gsd.py`, additive `run_baseline.py` dispatch,
  dedicated protocol and trajectory tests.
- Stack context: existing Colab runner and immutable seven-file bundles.
- Dependencies: Batch 1 contracts/tests.
- Implementation notes for $development-agent: preserve all baseline math;
  log graph diagnostics, both gradient norms and weighted guidance scale.
- Verification: zero-weight original parity, active-gradient execution under
  no_grad caller, frozen weights, original decoder, protocol rejection tests.
- Knowledge artifact to update: this file and experiment protocol.
- User review gate: existing authorization; local tests gate Colab handoff.
- Out of scope: changing source/original/control behavior or preprocessing.

### Batch 3 - Colab handoff and research record

- Goal: reproducible smoke, pilot and locked all-15 scripts.
- Scope/touched areas: `eval_gsd_tta.py`, `scripts/`, knowledge and result README.
- Stack context: unchanged installed Colab environment; no dependency upgrades.
- Dependencies: Batch 2 tests; all-15 promotion awaits actual pilot artifacts.
- Implementation notes for $development-agent: seeds 0/1/2, batch 32,
  severity 5, eval/raw, EMA off, original decode, fixed scheduler/rates/.95.
- Verification: dry-run command scopes, artifact serialization/failure paths,
  unchanged baseline file hashes, CPU regression suite, Git diff check.
- Knowledge artifact to update: findings, questions, method synthesis,
  repository map, paper notes and Colab instructions.
- User review gate: complete pilot ZIP review before all-15 execution;
  preparing the script is authorized now.
- Out of scope: running local GPU experiments or adding raw ZIPs to Git.

## Primary sources

[Paper] Local PDFs read without modification: 3DD-TTA WACV 2025,
GSDTTA ICCV 2025, and `lion.pdf` (NeurIPS 2022).
Official pages: [3DD-TTA](https://openaccess.thecvf.com/content/WACV2025/html/Dastmalchi_Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models_WACV_2025_paper.html),
[GSDTTA](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.html),
[LION](https://research.nvidia.com/labs/toronto-ai/LION/).

## Batch 1 completion (2026-09-22)

[Code] `graph_spectral.py` implements the selected detached static graph and
sample-summed normalized objective through `SpectralConfig`,
`build_spectral_target`, and `SpectralTarget.loss`. Dense graph construction
runs sequentially per sample; only the reference and selected bases persist.
Diagnostics record degrees, threshold, isolates, actual rank, eigenvalue
range, boundary gap, reference low-band energy fraction, and elapsed time.
Exact kNN distance ties use stable input-index ordering; permutation
invariance therefore assumes no tie is split at the neighbor boundary.

[Code] Initial CPU command `python -m unittest discover -s tests -p test_graph_spectral.py -v`
passed 19 tests (0.503 s); the final verification below supersedes that count.
Checks cover analytic and finite-difference
gradients, full-band normalization, sign/band rotation and vertex permutation
invariance, batch sum/gradients, nonself/duplicate neighbors, RBF values,
directed threshold normalization, max-union symmetry, both-endpoint masking,
disconnected degenerate bands, zero-isolate gradients, RNG preservation,
float32/float64 support, serialization, and rejected malformed inputs.

[Open] Dense N=2048 graph/eigendecomposition cost and CUDA numerical behavior
remain Colab measurements; local checks used only small CPU tensors. Graph
parameters and selected mode counts require the authorized pilot evaluation.

## Final completion - 2026-09-23

[Code] Batches 1--3 are implemented. The graph now has explicit overflow,
nonfinite, float32 eigenspace-boundary and fixed-anchor tests. The additional
eigendecomposition time, numerical zero-mode count, complementary energy
and effective tolerance diagnostics are serialized with the spectral contract.
Trajectory, protocol, worker success/failure and Colab command matrices are
covered; the full CPU suite passed 84 tests. Detailed commands, file list,
review outcome and limitations: [verification record](gsd_verification_20260923.md).

[Code] Independent review reproduced a float32 loss of 0 versus 5.3263 under
vertex permutation when a two-component zero eigenspace was truncated. After
the scale-aware tolerance fix, both retained rank 2, loss 8/3 and matching
gradients within float32 tolerance. The fixed-anchor policy also has regression
coverage. This is a numerical correctness result, not classification evidence.

[Open] Dense full-size GPU cost, real CUDA derivatives and benchmark accuracy
remain pending. Initial weight1 may be weak versus summed SCD; separate
gradient norms make this measurable without silently changing either rate.
The selected low modes can preserve corruption. Larger numerical bands and
exact-distance kNN ties remain declared limitations.

## Loss path and exploratory bands - 2026-09-23

[Code] For each batch, LION encodes the input into a global style latent and a
local latent of shape `B x 8192 x 1 x 1`. The local latent is reshaped to
`B x 2048 x 4`; its first three channels are detached reference XYZ. A static
non-self kNN/RBF graph, active combinatorial Laplacian, and detached low-mode
basis `U_b` are built once per batch.

[Code] At every reverse step, the frozen local prior predicts noise and DDIM
produces predicted clean latent XYZ `P_b`. Selective Chamfer retains the lowest
`int(2048*.95)=1945` distances in both directions and uses the unchanged
legacy sum:

`L_SCD = sum(retained d(P_b,R_b)) + sum(retained d(R_b,P_b))`.

The spectral term is the low-band projection error:

`L_spec = sum_b || U_b^T (P_b - R_b) ||_F^2 / (3 * rank_b)`.

The total objective is `L_total = L_SCD + w * L_spec`. SCD and spectral
gradients are computed separately with respect to the noisy local state and
style conditioning. The updates are `local -= gamma*(g_SCD + w*g_spec)` and
`style -= eta*(g_SCD + w*g_spec)`, followed by the original-style decoder.

[Inference] Increasing `gsd_modes` from 100 to 240 or 400 enlarges only the
selected low-frequency subspace in `L_spec`; it does not alter SCD or the
diffusion schedule. These are exploratory pilot variants, not the locked
benchmark configuration.

## Spectral-only ablation - 2026-09-23

[Code] `--gsd-scd-weight 0` is an explicit exploratory ablation. It keeps
`spectral_weight=1`, so the reverse-step objective is `L_total = L_spec` and
the SCD gradient is removed from both local and style updates. The operational
lambda remains `.95`; this option does not alter the decoder, scheduler,
preprocessing, or graph contract.

[Code] This ablation is launched with `--arms on` and compared with the
existing `3dd_original` baseline. A synthetic GSD-off arm is rejected because
the zero-spectral path is reserved for exact baseline delegation when the SCD
weight is the default `1`; it must not be mislabeled as a no-SCD control.

[Inference] The method name for this run is **spectral-only latent guidance
ablation**. It is not a GSDTTA reproduction and is not eligible for the locked
all-14/all-15 benchmark. Existing M=100/240/400 pilot artifacts use the
default SCD weight `1` and remain the combined SCD+spectral GSD-inspired
variant.

## Lambda baseline separation - 2026-09-23

[Run] The separate all-15 original-style lambda=.96 confirmation is positive
against the matched lambda=.95 comparator in all three seeds, with mean delta
+0.1855 pp.

[Inference] The separate `scd_lambda96_control` result does not authorize a
GSD change. `gsd_latent_spectral_v1` remains locked to the operational
lambda=.95 contract; a future GSD lambda=.96 ablation would be a separately
declared method with fresh matched weight-zero/weight-one artifacts.
