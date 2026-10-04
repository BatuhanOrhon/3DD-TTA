# SCD / spectral guidance composition: evidence and research decision

[Paper/Code/Run/Inference/Open] 2026-10-04. Research requested by the user;
no adaptation method was implemented and no GPU experiment was launched.
Current branch `gsd-smooth-spectrum`, HEAD
`aa8b725ba4daaf3448f0cda75bfc6aaec6c86d48`; legacy PxP inspected with
`git show pxp-gradient-projection:<file>` at
`53ba252519c7cf65f836a9c1c564027142ab1573` without changing branches.

## Decision

[Inference] Investigate composition, but do not assume close standalone
accuracies prove complementary information or destructive cancellation.
The available smooth-profile probes indicate **local magnitude dominance**
and **some style conflicts**. These require different interventions.
There is not yet equivalent common-state evidence for current hard GSD v1.

First obtain missing v1 geometry and paired prediction complementarity on a
small, declared diagnostic subset. If its conflicts resemble the smooth
probes, the first direction-changing candidate should be **per-example,
style-only, conditional symmetric projection**, with a magnitude control.
Treat SCD-priority one-way projection as a conservative comparison, not as
a mechanism that guarantees retaining spectral progress. Broader conflict
would justify CAGrad; useful prediction complementarity without useful
gradient interaction would justify separate trajectories or staged guidance.

## 1. What close scores establish

[Run] Background-excluded macro accuracies from the
[accepted ablations](gsd_guidance_ablation_results_20261004.md):

| Method | Other 14 corruptions (%) | Scope |
|---|---:|---|
| Unguided DDIM | 63.8863 | Seeds 0/1/2 mean |
| Hard v1 spectral-only | 63.9124 | Archived seed 0 only |
| Smooth-only, beta .5 | 64.0368 | Seeds 0/1/2 mean |
| SCD-only | 64.1092 | Archived seeds 0/1/2 mean |
| SCD + same smooth | 64.0619 | Archived seeds 0/1/2 mean |

[Inference] The unguided control is also close. A strong shared diffusion
prior, overlapping constraints, small effective guidance, and different
errors with similar totals are all plausible. Source/native binary and
random-draw differences between historical blocks prevent a causal ranking.
Background is a major exception: archived SCD-only is 60.6699%, while
unguided/smooth-only are near 24%; the completed 5/10/15/20/25-step scan
does not remove this gap. See [step study](gsd_background_step_sensitivity_20261004.md).

[Code] `tta_gsd.py` differentiates the losses separately with respect to
noisy local latent and style conditioning, then adds their weighted gradients.
SCD uses trimmed nearest-neighbour geometry; `graph_spectral.py` uses spectral
fidelity of predicted latent XYZ to the corrupted encoded reference. Both
anchor to the same corrupted sample. Different losses need not carry
independent clean-shape information or have different preferred solutions.
The current method is GSD-inspired latent spectral guidance, not the full
GSDTTA point-shift method.

[Inference] For fixed scalar weights, addition is exactly the gradient of
the weighted sum objective. It does not intrinsically discard information.
However, its direction can nearly ignore a small gradient, or improve one
objective while worsening the other. If the two gradients are exactly
opposite, no single infinitesimal direction can strictly improve both.
Projection also removes components; it cannot universally preserve all
objectives and guarantee better classification.

## 2. New offline analysis of existing common-state probes

[Run/Inference] Reconstructed hypothetical weighted sums from:

`result/modelnet40_c/gsd_latent_spectral_smooth_v2/20260928-113047_gsd-cal-diagnose-reference-seed0-n64.zip`

SHA256: `363cd803b835a67546074cd0ac81b244ef57ec092b27b63f49060c1ad167d325`.
The archive was read only and its hash checked again after calculation.
Each corruption/profile/block has 192 observations: 64 examples x three
timesteps, seed 0, **SCD-only reference states**. These are repeated
observations, not 192 independent samples. All rows have usable gradients.
Use archived alpha 8.140161356429882 for beta .5 and
113.10823980075075 for beta 2, with SCD weight 1. This is not a new guided
trajectory or an accuracy experiment; the probes are smooth, not hard v1.

Let `c = weighted SCD gradient`, `s = weighted spectral gradient`,
`r = ||s||/||c||`, and `q = cosine(c,s)`. The angle of `c+s` from `c` is
`atan2(r*sqrt(1-q*q), 1+r*q)`. Positive constant weights do not change
pairwise cosine or conflict sign, but do change this combined direction.

| Corruption | Beta | Block | Median r (%) | Median angle from SCD | Negative cosine (%) |
|---|---:|---|---:|---:|---:|
| Gaussian | .5 | local | .09728 | .05430 degrees | 0 |
| Impulse | .5 | local | .10329 | .05791 degrees | 0 |
| Gaussian | 2 | local | .99142 | .55512 degrees | 0 |
| Impulse | 2 | local | 1.00943 | .56816 degrees | 0 |
| Gaussian | .5 | style | .08815 | .03429 degrees | 10.417 |
| Impulse | .5 | style | .06708 | .02348 degrees | 8.854 |
| Gaussian | 2 | style | 1.18082 | .58814 degrees | 38.021 |
| Impulse | 2 | style | .70762 | .34064 degrees | 36.979 |

[Inference] Local gradients are not identical: their median cosines are
about .17-.23, yet the much smaller spectral contribution barely rotates
the sum. Conditional conflict projection would do nothing on these local
observations. No local cancellation mechanism is demonstrated here.
This does not establish that increasing the spectral weight improves accuracy;
the existing full screens did not show such a gain.

[Inference] In the style block, a small guidance-only step along `-(c+s)`
has predicted spectral ascent whenever `r+q < 0`. This occurs in
10.417%/8.854% of beta .5 Gaussian/Impulse observations and
36.458%/36.979% of beta 2 observations. It predicts no SCD ascent in these
rows. These are **blockwise first-order signs**, excluding local-block
contributions, DDIM displacement, curvature and changing timestep; they
are not observations of the total spectral loss increasing after a real step.
Final decoding uses original shape latent, but updated style conditioning
still affects subsequent denoising steps.

Reproducible artifacts, including formulas, quantile convention and all rows:
[analysis.json](../result/modelnet40_c/gsd_latent_spectral_smooth_v2/analysis_20261004_gradient_composition/analysis.json),
[summary.csv](../result/modelnet40_c/gsd_latent_spectral_smooth_v2/analysis_20261004_gradient_composition/summary.csv),
[recompute.py](../result/modelnet40_c/gsd_latent_spectral_smooth_v2/analysis_20261004_gradient_composition/recompute.py).

## 3. Primary literature and transfer limits

| Source | Supported mechanism | Relevance here |
|---|---|---|
| [PCGrad, Yu et al., NeurIPS 2020](https://arxiv.org/abs/2001.06782), sections 2.2-2.3, Algorithm 1 | Project negative pairwise conflicts; leave nonconflicting gradients unchanged. The analysis considers conflict, magnitude imbalance and curvature together. | A simple conditional composition control. Negative cosine alone does not establish harm. |
| [PixelAsParam, Dinh et al., ICML 2023](https://proceedings.mlr.press/v202/dinh23a.html), sections 4.2-4.4 and 6.4, **Table 6** | Diffusion update decomposition and selective treatment of direction pairs; the pair ablation cautions against indiscriminate intervention. | Supports measuring which pairs matter. Its denoising/diversity/classifier decomposition is not SCD/spectral guidance. Table 4 is classifier-free guidance, not the pair ablation. |
| [CAGrad, Liu et al., NeurIPS 2021](https://arxiv.org/abs/2110.14048), sections 3.1-3.2, Algorithm 1 | Choose a direction near the average gradient while optimizing worst local objective improvement. | A compromise if neither objective should have unconditional priority. Sensitive to objective scaling; its fixed-objective convergence assumptions do not imply convergence or accuracy gains for our changing diffusion states. |

## 4. Candidate methods and what each can solve

### A. Bounded magnitude calibration: control for dominance

[Inference/Proposal] Measure local and style norms separately and compare
the ordinary sum with a predeclared, bounded spectral/SCD ratio policy.
Calibrate a small number of fixed scales on development data; do not
automatically force equal norms or amplify near-zero gradients. Keep total
guidance/DDIM displacement ratios under observation. Loss values alone are
not a useful scale match. This is a magnitude experiment; it cannot solve
negative alignment by itself. Reuse completed fixed-weight conditions.

### B. Conditional symmetric projection: first direction candidate

[Proposal/Derivation] For one example and one state block, if `c dot s < 0`
and both norms exceed a declared numerical threshold, compute using the
**original** pair:

```text
c_projected = c - (c dot s) / ||s||^2 * s
s_projected = s - (c dot s) / ||c||^2 * c
d = c_projected + s_projected
```

Otherwise use `d=c+s`. Skip ill-defined near-zero projections and record
the skip; do not claim exact orthogonality for an epsilon-altered formula.
For the ideal two-vector construction, `c dot d = ||c||^2*(1-q^2)` and
`s dot d = ||s||^2*(1-q^2)` on a conflict. Thus `-d` is nonascending to
first order for both fixed block objectives, and is zero at exact opposition.
This is not a finite-step, DDIM, or classification guarantee.

[Proposal] Start with style only **if v1 diagnostics confirm that location
of conflicts**. Keep local composition unchanged. Compare raw projected
updates with a control ordinary-sum update rescaled down to the projected
norm, to distinguish direction change from smaller step size. Avoid
renormalizing a near-zero projected direction to a large original norm.

[Code/Inference] Legacy `tta_pxp.py` instead protects spectral guidance by
projecting SCD; `tta_pxp_sym.py` supports both projections. Both make the
conflict decision after flattening the entire batch. This can hide individual
conflicts and must not be copied into a per-example method. Projection off
a tiny spectral vector can still substantially change SCD: positive scale
cancels from the ideal projection formula.

[Proposal] An SCD-priority control modifies only `s`, using `s_projected`
above. It protects first-order SCD progress, but **does not ensure spectral
progress**, because the unchanged `c` may still oppose `s`. Symmetric
projection addresses that distinction. Neither projection should be applied
unconditionally to agreeing gradients.

### C. CAGrad: a subsequent compromise

[Proposal] If conflicts are widespread or symmetric projection stalls,
compare a scale-controlled CAGrad composition. Retain the ordinary sum and
projection controls. The mean-loss preference is an explicit design choice;
lower surrogate losses need not imply better point-cloud classification.
Do not add another optimizer before geometry indicates a reason to do so.

### D. Separate time or trajectory: test prediction complementarity

[Proposal] If different examples benefit from the methods, retain separate
SCD-only and GSD-only trajectories and evaluate fixed probability averaging
using frozen-classifier outputs. Record its doubled adaptation cost and
compare against two same-method trajectories with equal compute. Ground
truth must never choose the deployed branch; an oracle selector is an offline
upper bound only. Confidence-based selection needs calibration evidence.

Alternatively use a predeclared timestep schedule or sequential substeps.
Two substeps with **stale gradients from the same state equal the original
sum**; genuinely alternating optimization requires recomputing after the
first update. Extra compute and order must be controlled. Existing normal
runs have only five actual reverse steps, making scheduling coarse.

[Open] Frequency routing (spectral guidance for low modes, SCD for a
complementary subspace) is a later hypothesis. Graph eigenvectors live in
predicted latent XYZ, while actual gradients live in noisy latent/style
coordinates. Projection and the denoiser Jacobian generally do not commute;
the current N-point basis cannot simply be applied to flattened state
gradients. Hard routing could remove useful SCD components, including ones
relevant to Background. It is less justified than the measured block split.

## 5. Smallest informative next experiment

1. **Missing evidence only:** on fixed indices in Gaussian, Impulse,
   Background and Shear, capture matched SCD-only/v1-only predictions and
   both gradients at common states. Use a small declared diagnostic sample,
   not another complete 15-corruption accuracy sweep. Keep the profile fixed
   while testing composition; do not combine this with the separate mean-scale
   experiment. The archived smooth probes already answer their scoped question.
2. **Complementarity:** tabulate both correct / SCD-only correct / GSD-only
   correct / both wrong and predicted-class disagreement. Report oracle
   accuracy `1 - both_wrong/N` and its gap above the stronger method as an
   offline potential, not an attainable deployment score. Existing A/C
   predictions compare unguided/smooth-only; archived B/D have no per-example
   predictions and cannot answer the SCD/v1 question.
3. **Geometry:** record per-example local/style norm ratio, cosine, angle of
   combined update, projection magnitude, and first-order loss change. Probe
   states from both standalone paths where feasible, avoiding conclusions
   based exclusively on the SCD reference trajectory. Compare signed local
   guidance displacements with the DDIM displacement; DDIM is not assumed
   to be the gradient of a fixed scalar objective. Measure same-timestep
   surrogate response separately from real next-timestep loss changes.
4. **Conditional pilot:** ordinary sum versus magnitude-matched sum versus
   style-only symmetric projection; add SCD-priority as an interpretation
   control if projection changes results. Add local projection only when
   local conflicts are observed. Promote CAGrad or separate trajectories
   only according to the diagnostic outcome above.
5. **Controls and metrics:** matched source, native extensions, checkpoints,
   graph/profile/rank, scheduler/5-or-35 actual steps, batch/order and decoder.
   Explicitly share per-example preprocessing, VAE/noising and FPS random
   draws; identical seeds alone do not establish this. Report per-corruption
   accuracy, paired transitions, runtime and memory. Candidate replication
   uses seeds 0/1/2, with fixed configuration before independent evaluation.
   Existing inspected examples are development evidence; labels are used
   only for offline evaluation, never to choose guidance at inference.

**Falsifiers:** negligible prediction complementarity limits the value of
ensembling; absent conflicts make PCGrad identical to the sum; successful
geometric repair without accuracy gain rejects conflict resolution as the
explanation of classifier performance at this scope; equal performance from
a norm-matched sum attributes any benefit to step scale rather than direction.

[Decision] Continue with this diagnostic decision tree. No claim of a new
method's accuracy gain is supported yet, and no completed full benchmark
needs to be repeated to establish the currently missing mechanism evidence.
