# Calibration review: update scales and a small staged search

Status: [Code/Run/Inference] review of `d75a32d`; user-approved experiment now
implemented in [the execution plan](gsd_calibration_execution_20260927.md),
with [Colab instructions](colab_gsd_calibration.md). No model run yet.
Supersedes the mandatory q95-derived grid, v1-scale target,
and blanket prohibition on unlabeled target-input statistics in section 10 of
`gsd_guidance_math_20260926.md`. Keep those earlier proposals as history.

## What the smoke establishes

[Run] Source artifact:
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/20260927-192156_gsd-smooth-v2-smoke-smooth-seed0-beta2p0.zip`.
This is one Gaussian batch, seed 0, beta 2, spectral/SCD weights 1/1;
26/32 accuracy is not comparative evidence.

[Code/Derivation] The reported local SCD gradient norm 113.0058 is over
`32 * 8192` coordinates. With gamma .01, its batch-wide update norm is
1.13006 and its coordinate RMS is about .002207. State RMS and the unguided
DDIM displacement are not logged, so neither overshooting nor negligible
physical effect follows from this value. Loss units/reductions and gradient
units are arbitrary without the associated update rate and state scale.
Small spectral/SCD norm ratios likewise do not establish useless guidance.

Measure per-example, per-probed-step quantities, separately for local/style:

- SCD and weighted spectral update RMS, and state RMS;
- update/state norm ratios;
- local SCD update versus `prev_sample - noisy_local` (DDIM displacement);
- SCD/spectral gradient cosine and stepwise tails (median and p90);
- raw denominators and near-zero flags; do not let an epsilon hide a nearly
  zero DDIM displacement or create an enormous calibration coefficient.

No universal threshold on these ratios proves that SCD is too large. A
consistent large relative displacement and unstable trajectory would justify
a separate SCD-only coefficient-.5 control. Preserve the original coefficient
1 reference. The archived SCD normalization control lost 2.4806 pp, but dividing
the loss by N with unchanged rates is not evidence about a modest .5 ablation
or an optimum SCD strength.

## Beta need not be data-calibrated initially

[Code/Derivation] On an active graph with n vertices,
`q_i=lambda_i/mean_degree` satisfies `sum(q_i)/n=1`, up to numerical roundoff:
`trace(L)=sum(degrees)=n*mean_degree`. Thus beta is already dimensionless.
Use a predeclared small logarithmic grid `{0.5, 2, 8}` as an exploratory
proposal, not an optimum. At q=1 these give weights .6065, .1353, .0003355.
Monitor effective mass, isolates and zero modes; equal mean q does not imply
equal spectral shape across graphs. q95 can be added as a diagnostic later,
but neither a new dataset nor q95 is necessary to define the initial grid.

## Match a chosen contribution, not an assumed optimum

[Inference] Retain v1 weight-1 scale matching as an optional implementation
control, not the primary optimum. It would inherit an unvalidated strength.
Avoid matching spectral and SCD norms to 100% merely because they are summed:
strong fidelity can retain corruption.

At identical SCD-only reference states, measure unit-weight spectral gradients
for hard-v2 and each beta. Define the primary calibration statistic as the
median per-example/per-probe local spectral-to-SCD gradient norm ratio R.
For a predeclared desired contribution rho, use a single fixed coefficient
`alpha=rho/R` for the subsequent actual guidance run. Propose the broad
small-contribution grid `rho={1e-4,1e-3,1e-2}` (0.01%, 0.1%, 1%), including
low doses because the user reported useful effects with small gradients.
These are exploratory test points, not empirical optima. If a boundary is
promising, preregister an extension rather than expanding repeatedly on test
accuracy. Treat degenerate/near-zero gradients explicitly.

Use local as the primary route and monitor style independently. Raw local and
style vectors have different dimensions and scales, so concatenating them can
hide an imbalance. A single alpha generally cannot match both routes exactly;
do not claim full two-route scale equivalence from a local match.

The alpha relation is exact on fixed states. Changing alpha changes subsequent
states, so it is only an initial calibration for a complete trajectory. Actual
guided runs and their diagnostics remain mandatory. Do not introduce online
per-step norm balancing in this initial test: that is a different method.

## Economical experiment sequence

[Proposal] Use a fixed development pool, initially 64 randomly selected
identities per corruption for diagnostics (Gaussian and Impulse), seed 0.
Record all SCD-only steps, probing extra spectral gradients at early/middle/late
steps. Use the same reference states and RNG draws for every candidate.
Reuse the reference-graph eigendecomposition across betas and reuse the
denoiser forward where memory permits; each beta still needs its own VJP
through the denoiser. Coefficient trials need no extra backward pass on fixed
states, because they only scale the gradient. Do not store all timestep graphs.

For actual guided screening on a fixed development subset, initially 128
examples per corruption, seed 0:

1. Baseline plus beta=2 at the three rho values: four conditions.
2. At the best supported rho, add beta=.5, beta=8 and matched-rho hard-v2:
   three additional conditions, reusing beta=2 and baseline.
3. Check an adjacent rho at a promising new beta to detect interaction; keep
   multiple candidates if the small sample is inconclusive.
4. Extend only promising candidates and matched controls to more development
   examples and seeds 0/1/2, then freeze before held-out confirmation.

This staged screen trades exhaustive coverage for cost; it can miss beta/rho
interactions. A small prefix rejects gross failures, not +/-1 pp effects.
Do not forecast runtime from Gaussian alone: the reverse-step schedule varies
across corruptions. Calibration and accuracy screening can share a development
pool; both must be distinguished from final confirmation data.

## Data-policy correction

[Paper] TTA can use unlabeled target inputs by definition; source-free does
not mean target-data-free. Tent uses target batches without source examples:
https://arxiv.org/abs/2006.10726 . Hyperparameter/model selection and online
batch dependency need explicit protocols, as discussed by TTAB:
https://proceedings.mlr.press/v202/zhao23d.html .

[Inference] A predeclared label-free normalization rule using target inputs
can be a valid source-free TTA variant. Fitting a pooled coefficient adds
cross-example dependence and must be identified as such; it is not automatically
equivalent to the current static, independently applied method. Accuracy-based
selection on final test labels remains a different issue. Merely viewing a
smoke result does not automatically invalidate all label-free target adaptation.

For the fastest exploratory work, use a fixed ModelNet40-C development subset;
verify identity correspondence across corruptions and exclude the same object
identities from held-out confirmation. Report this split explicitly; its score
is not the canonical full-test average. If an independent canonical full-test
claim is required, use separate validation data for accuracy-based selection.
Do not require source-training data or a new external dataset merely to read
label-free update statistics. A deterministic development index pool and
calibration runner are now implemented; cross-corruption object correspondence
is not verified. No new accuracy claim follows from this review.
