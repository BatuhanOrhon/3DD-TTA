# Spectral guidance for 3DD-TTA: mathematical decision, 2026-09-26

## Scope and decision

[User report] Re-read the local Wei ICCV2025 paper and decide how spectral
transformation/comparison should guide 3DD-TTA denoising. Research/design only.
[Code, original design snapshot] Branch `gsd-development`, commit
`9650770f75cf0c37e1e16b873bd6b8ddd0a4276e`. Existing modified README and
untracked branch-comparison note preserved. No Python implementation changes.

[Paper] Read local Wei PDF text pp.1-10 and visually inspected complete
pp.3-5 (Fig.2/3, Eqs.7-18) using Windows.Data.Pdf renders. PDF SHA256:
`7ba86ae3902628a7cae464008303342afecc17c61f3d09b7393049ced2ea978b`.
Re-read local 3DD-TTA PDF pp.4-5 (Eqs.8-12, Algorithm1) and traced `tta.py`,
`tta_gsd.py` and `models/latent_points_ada.py`. Source PDFs unchanged.

[Inference: selected proposal] Fixed reference graph on encoded latent XYZ;
compare reference and predicted-clean coordinate signals in its common basis;
use a smooth eigenvalue-weighted fidelity loss (heat-kernel quadratic metric),
fixed original point-count normalization, and sample sum. Differentiate through
predicted-clean reconstruction and the frozen denoiser, not through the graph.
Keep SCD in the initial comparison. This is proposed GSD-inspired guidance,
not implemented v1 behavior, full GSDTTA, or a demonstrated accuracy gain.

## 1. What the paper actually does

[Paper] Sec.3.1, p.4, Eqs.11-13 imply

`C=U_o^T X`, `X_s=U_o(C+[DeltaC;0])=X+U_o,M DeltaC`.

Low-frequency displacement is learned; high-frequency input coefficients stay
unchanged. Fig.2's low-pass reconstruction is motivation, not the Eq.12 update.
There is no loss directly matching two independently computed GFTs.

[Paper] Eq.17, p.5, uses pseudo-label cross entropy, entropy/diversity and
one-direction input-to-adapted Chamfer. Eq.18 additionally adapts the classifier.
The approximately95% low-band energy in Fig.2 is a chair example, not a
guarantee for corrupted inputs or LION latents.

[Inference] This differs fundamentally from v1: GSDPS allows low-band shifts;
low-band fidelity penalizes those shifts. For `X_s=X+U_M DeltaC`, a plain
gradient step on a scalar objective J gives
`grad_DeltaC J=U_M^T grad_Xs J` and induced
`Delta X_s=-eta U_M U_M^T grad_Xs J`. This is spectral preconditioning of a
driving loss, not an extra fidelity loss. Through a nonlinear LION denoiser
it is not equivalent to projecting the noisy-state gradient with U_M.
An explicit GSDPS parameterization is a separate method candidate.

## 2. Printed ambiguities and implementation decisions

[Paper] Eq.7 shows d squared, while prose calls d squared Euclidean distance.
Select standard Gaussian `exp(-||qi-qj||^2/(2*sigma^2))` explicitly.
Eqs.8-9 describe directed kNN and row masking, although symmetric L is assumed.
kNN and row masking do not guarantee symmetry. Specify max symmetrization and
both-endpoint masking rather than applying eigh to a nonsymmetric matrix.
Eq.10 is `gamma/(N*k)*sum(A)`; its distance wording conflicts with affinity
units in the equation. Gamma/N is a different cutoff, not an algebraic repair.
Eq.16 visibly says argmin of positive cosine similarities; do not import that
pseudo-label rule without resolving the convention. It is unused here.

## 3. Domain and reference graph

[Code] Define `R=h_ref[:,:,:3]` and
`Y_t=hat_h_0(h_t,s_t)[:,:,:3]`, both Bx2048x3. LION's fourth local channel is
a learned feature, not another spatial axis. Its spatial encoder mean has
an input-coordinate residual; this does not make latent XYZ equal physical XYZ.

[Inference] Use latent XYZ because it is the existing SCD domain and its slots
are tracked without resampling along the trajectory. Do not compare physical
input coordinates to latent predictions, or build on noisy h_t. A physical
decoder-space loss adds decoder Jacobians/cost and is a separate ablation.
Slot identity defines a displacement regularizer, not guaranteed geometric
nearest-neighbor correspondence. If independent clouds have unknown ordering,
a vertex map/transport T is needed before this loss; a shared basis alone does
not solve matching. Log slot exchange/correspondence diagnostics on real states.

Keep current graph conventions initially, so a new loss is not confounded by
simultaneously changing graph rules:

1. Directed nonself kNN affinities
   `B_ij=1[j in Nk(i)] exp(-||r_i-r_j||^2/(2*sigma^2))`, diagonal zero.
2. `d_dir=B*1`, `tau_g=gamma_g*sum(B)/(N*k)`, retain `d_dir>tau_g`.
3. `A=max(B,B^T)`; mask both endpoints by retention.
4. I contains vertices with positive final degree; S selects these rows;
   n=|I|; `L=diag(A_I*1)-A_I` on the induced active graph.
5. `mu=trace(L)/n`, `Lbar=L/mu`. For nonempty active graphs mu>0.

This scalar frequency normalization leaves eigenvectors unchanged; it is
not `D^-1/2 L D^-1/2`. Empty active graphs give zero spectral loss and are
logged. No diagonal1000 penalty. Removed vertices get zero direct clean-XYZ
spectral gradient, but the denoiser Jacobian can spread effects elsewhere.

k10/sigma.1/gamma_g.6 are inherited controls, not established optima. The
6%-of-mean-directed-degree cutoff can retain weak corrupt regions. Smoothing
the spectrum does not fix graph quality. Stronger filtering, bandwidth or
physical graphs are separate factors; do not silently replace Eq.10 by gamma/N.
Reference units/normalization stay fixed; do not independently rescale Y.

## 4. What two-graph comparison means

Independent `||U_Y^T Y-U_R^T R||` is generally not a valid aligned coefficient
comparison. Sorting eigenvalues does not align signs, eigenspaces or vertices;
even identical clouds can get positive loss from a sign change.

Use `Lbar=U Lambda U^T`, `C_R=U^T S R`, `C_Y=U^T S Y` instead.
Each coefficient now measures the same function on the same nodes. The selected
loss compares **graph signals on a fixed topology**, not two adjacency matrices.

If distinct graphs/bases are required, a vertex map T and functional map
`C=U_R^T T U_Y` are needed (plus truncation residuals for incomplete bases).
In full bases `C U_Y^T Y=U_R^T T Y`. This is unnecessary complexity for the
current same-trajectory fidelity objective.

## 5. Selected loss and its derivative

For eigenvalues `lambda_i>=0` of Lbar, choose
`w_i(beta)=exp(-beta*lambda_i)`, beta>0. For each sample:

`E=S(Y-R)`

`ell_spec = (1/(3*N)) sum_i w_i ||u_i^T E||_2^2`

`L_spec = sum_b ell_spec,b`, with original N=2048.

All active modes are included mathematically; high modes have small rather
than exactly zero weight. No unlogged truncation. Beta sets selectivity;
outer alpha_spec sets strength. Both are validation parameters, not optima
deducible from the paper. Keep them fixed before final benchmark reporting.

Equivalently, `F=U diag(w) U^T=exp(-beta Lbar)`:

`ell_spec=trace(E^T F E)/(3*N)=||exp(-beta Lbar/2) E||_F^2/(3*N)`

`grad_Y ell_spec = 2 S^T F S (Y-R)/(3*N)`.

The factor one-half in the filter matters: filtering with exp(-beta Lbar)
and then squaring its norm instead uses exp(-2*beta*lambda) weights.

[Inference] Why selected: no hard M boundary or rank expansion; no actual-rank
denominator; smooth weight changes; sign and rotations inside a repeated
eigenspace cancel because their weights are equal. Sample sum avoids batch-size
scaling; fixed original N avoids amplifying surviving vertices after masking.
This is a new smooth weighting proposal, not an equation from GSDTTA.

Normalization differs from v1's 1/(3*actual_rank) and legacy batch MSE;
alpha is not numerically transferable. Separate normalization and profile
changes in ablations; weight1-versus-weight1 does not isolate profile shape.

Limits: beta=0 gives active-vertex pointwise MSE; beta->infinity retains only
component-constant modes (only centroid discrepancy on a connected graph).
Large beta can therefore discard useful shape information. Low-frequency
corruption is still anchored. Tiny roundoff-negative eigenvalues may be clipped
under a logged tolerance; material PSD violations must fail.

## 6. Guidance in the host trajectory

`hat_h_0=(h_t-sqrt(1-alpha_bar_t)*epsilon_theta(h_t,s_t,t))/sqrt(alpha_bar_t)`

`L_t=L_SCD(Y_t,R)+alpha_spec*L_spec(Y_t,R)`

`h_prev=DDIM_prev-gamma*grad_h_t L_t`

`s_next=s_t-eta*grad_s_t L_t`.

These are repository gamma/eta names. For spectral loss,
`grad_h_t L_spec=J_h^T grad_Y L_spec`, likewise for style. J includes the
denoiser's input dependence and the direct h_t route. Freeze model weights,
not input gradients; detach graph/reference only. Do not detach Y or replace
this derivative with a direct clean-point update. Keep original final decode,
lambda.95, scheduler, rates, preprocessing and FPS controls initially.

[Inference] LION supplies denoising, SCD set-based anchoring, spectral loss
structural fidelity. R is a minimizer of spectral fidelity: it is not a
stand-alone denoiser and supplies no semantic target for rotation/shear
correction. Full GSDTTA's classifier adaptation/pseudo-labels and PxP are not
part of the first comparison. Small spectral norms do not establish uselessness.

## 7. Alternatives and limitations

| Candidate | Decision |
|---|---|
| Independent-basis coefficient MSE | Reject: coordinate systems are unaligned |
| Eigenvalue-only distance | Not primary: spectrum does not uniquely identify shape or orientation/correspondence |
| Per-mode power matching | Not primary: loses directional information; repeated modes complicate per-mode matching |
| `trace(Y^T L_R Y)` alone | Reject: constant signals minimize it; can shrink/collapse geometry |
| Aligned heat-kernel operator distance between graphs | Valid with correspondence, but graph matching and differentiating candidate geometry are additional factors |
| Shared hard low/mid coefficient loss | Historical control; meaningful with proper bands, but hard boundaries/scales remain factors |
| Explicit learned low-frequency displacement | Closest GSDPS transfer, but changes optimization parameterization and needs a driving objective |

Smooth spectral fidelity is jointly rotation/permutation consistent when both
signals and graph are transformed together; it is not invariant to independently
rotating only Y. Isolated removal does not remove weak multi-node components.
Repeated zero modes can anchor corrupted component means. No spectral-fidelity
formula alone guarantees reconstruction or classification improvement.

## 8. Mathematical checks and falsifiers

[Code: CPU checks] Float64 CPU, seed26, symmetric nonself 8-node random graph,
trace-normalized L, beta2: spectral/spatial loss error0; analytic/autograd
gradient error3.90e-18; directional finite-difference error1.14e-12; aligned
permutation error2.78e-17. Complete-graph repeated-space basis rotation changes
F by1.39e-16. An independent basis sign flip gives naive loss.66477 on the
same cloud; shared-basis self-comparison is0. No GPU/model/dataset evaluation.

Reproduction: one `torch.rand(8,8,dtype=float64)` seeded matrix T, set
`A=(T+T.T)/2`, diagonal zero; Gaussian R and perturbations; use eigh and
`w=exp(-2*lambda)`. Finite differences perturb Y by +/-1e-6 along a seeded
Gaussian direction. Repeated eigenspace uses ones-minus-identity adjacency.

[Open] On real latent captures inspect components, node correspondence,
operator localization, effective spectral mass, weighted local/style gradient
norms, alignment with SCD, update magnitude, runtime and memory. Full-spectrum
storage can exceed N-by-M storage; any approximation needs an error bound.

[Inference] Preserve baseline/v1. Compare hard versus smooth profiles with
the same reference/graph and controlled normalization, then graph rules alone,
then alpha on declared validation data. Use seeds0/1/2 and common random draws.
Final all15 cannot select parameters. If scale-matched variants do not help,
reconsider fidelity-to-corruption rather than continually tuning weights.
No new run directory or accuracy claim is created by this design.

## 9. Difference from the current GSD code and next test case

[Code] In `graph_spectral.py`, v1 already builds one detached graph and basis
from the encoded reference XYZ, then evaluates `U_m^T(Y_t-R)` in that same
basis. In `tta_gsd.py`, `Y_t` is the predicted-clean XYZ; SCD and spectral
losses are differentiated through the frozen denoiser to both local latent
and style, then combined in the existing guidance updates. These are inherited
v1 behaviors, not proposed changes. The graph remains fixed across denoising.

[Inference] The proposed code delta is limited to the spectral metric and its
scale/configuration: retain the active full spectrum and eigenvalues; scale
the combinatorial Laplacian by mean active degree; replace the hard first-M
projector/equal mode weights with `exp(-beta*lambda_scaled)` on all active
modes; and use `3*N` (`N=2048`) in the per-sample loss. Add beta as a separately
logged parameter. Do not change graph construction, reference/prediction
signals, SCD, scheduler, rates, decoder, classifier or data path in this test.
The `3*N` denominator changes strength as well as profile, so alpha from v1
cannot be copied and called scale matched.

**Next GSD test case: `gsd_smooth_spectrum_profile_pilot`** (code added on
`gsd-smooth-spectrum`; CPU-verified after review, no model run yet). Hypothesis: with the
reference graph, SCD host and effective spectral guidance scale controlled,
smooth eigenvalue weights
produce a more useful cleaning direction than the v1 hard low-mode cutoff.
The test is exploratory and does not presuppose an accuracy gain.

1. Preserve separately identified `gsd_latent_spectral_smooth_v2`; keep v1 and
   its weight-zero baseline dispatch unchanged. Save the equivalent dense
   spectral filter and log beta, active count, effective weight
   mass, alpha, local/style spectral and SCD gradient norms, update norms,
   runtime and peak memory.
2. On identical reference graphs and common random draws compare (A) unchanged
   v1 hard-M as the operational comparator, (B) a hard-M profile in the new
   `3*N` host, and (C) the smooth profile in that same host. Calibrate B and C
   to the same predeclared aggregate spectral-gradient scale on a declared
   calibration set; freeze both alphas before accuracy evaluation. B vs C
   compares profile shape at calibrated scale; A vs B reports the combined
   denominator/refactor and coefficient impact. A denominator-only comparison
   requires holding alpha fixed.
   Log local and style scales separately so an aggregate match cannot hide a
   route imbalance. Keep beta candidates and calibration rule fixed before
   looking at evaluation accuracy.
3. First run the exploratory ModelNet40-C severity-5 Gaussian/Impulse pilot
   with batch 32, seeds 0/1/2, raw LION eval, EMA off, frozen Point-MAE,
   lambda `.95`, and unchanged v1 scheduler/rates/decoder/preprocessing.
   Use paired seed/common draws, complete run bundles and per-corruption rows.
4. Read accuracy deltas with paired uncertainty alongside profile/gradient
   diagnostics, runtime and memory. A smooth-vs-hard accuracy comparison is
   interpretable only if the graphs, host trajectory and calibrated scale
   match. Null or negative evidence stops promotion; any parameter selection
   needs declared validation data, and a pilot result is not an all-15 claim.

[Open] Before implementation, settle the beta candidate grid, calibration-set
source and scale-matching statistic from `experiment_protocol.md`; the design
does not imply an optimal beta or alpha. Validate graph component quality and
slot correspondence on real latent captures. Numerical/model runs remain
Colab-only under the project workflow.
