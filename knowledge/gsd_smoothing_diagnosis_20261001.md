# GSD smoothing and historical small-gradient gains: diagnosis

[Code/Run/Inference/Open] 2026-10-01, local branch `gsd-smooth-spectrum`,
HEAD `4be6afd86629f3d52648fb543fc2eb350a5f9f1d`. This is an offline
analysis of existing artifacts and current/legacy source. No model run,
trajectory modification, commit, push, or Colab restart was performed.

## What the completed experiment establishes

[Run] The nine full-file screen ZIPs show no positive mean increment from
the two selected smooth candidates over their SCD-only reference. They do
not include a matched all-15 hard-profile arm, so they do not identify the
causal effect of replacing hard weighting with smooth weighting.

| Scope | beta .5 / rho .001 minus SCD (pp) | beta 2 / rho .01 minus SCD (pp) |
|---|---:|---:|
| Rebuilt development, 128 Gaussian + 128 Impulse, seed 0 | +0.781250 | +1.171875 |
| Full Gaussian + Impulse, seeds 0/1/2 mean | -0.054025 | -0.148568 |
| Full all-15, seeds 0/1/2 mean | -0.027913 | -0.058527 |

[Run/Inference] The small-pool improvements were two and three net correct
predictions out of 256. Their disappearance cannot be attributed solely to
adding 13 other corruptions: even full Gaussian/Impulse is nonpositive.
Sample coverage, seed replication, batch/order/random-draw assignment and
runtime differ between subset and full screening; these factors have not
been isolated. The small-pool result is compatible with sampling/selection
variation, but the archives do not prove a single cause.

[Run] Historical v1 SCD-plus-spectral pilots already had mean on-minus-off
Gaussian/Impulse deltas -0.0135, -0.1688 and -0.0540 pp at M100/240/400.
The spectral-only pilot's +0.2296 pp also removes SCD, so it does not isolate
an additive spectral benefit. Its full 14-corruption seed-0 ablation lacks
Background; 63.9124% cannot be compared directly with an all-15 macro.
See `gsd_branch_comparison_20260923.md` and the all-15 matrix.

## Gradient magnitude is not the sufficient explanation

[Code] The implemented local update is
`gamma * (scd_weight * g_scd + spectral_weight * g_spec)`; the analogous
style update uses eta. Compare actual weighted contributions on a common
state, not raw loss, unweighted norm, or coefficient alone.

[Run] Recomputed from all three full-screen guided ZIPs. Values below are
ratios of recorded batch-step mean norms, then averaged over seeds. They
are not per-example medians or the calibrated target statistic.

| Condition | Gaussian local spectral/SCD | Impulse local spectral/SCD | Background local spectral/SCD | Gaussian effective spectral mass |
|---|---:|---:|---:|---:|
| beta .5 / rho .001, alpha 8.140161 | 0.096322% | 0.101999% | 0.428373% | 1259.76 |
| beta 2 / rho .01, alpha 113.108240 | 0.980377% | 1.001897% | 4.914854% | 646.17 |

[Run] For Gaussian, beta2's unweighted local spectral norm is about .009833,
but its weighted norm is about 1.112; SCD is about 113.440. At beta .5,
unweighted spectral norm is .013425 and SCD 113.452. Older v1 M100 seed0
recorded .094994 versus 113.4171 (0.0838%). Thus a smaller raw v2 gradient
does not imply a smaller applied contribution. Different observations and
operators remain unsuitable for a pure scale-effect claim.

[Inference] Weak incremental guidance can explain small changes, but cannot
by itself explain which direction improves classification. The two selected
full-screen candidates change beta and rho together; their roughly tenfold
scale difference is not a controlled weight-only ablation. No vanishing or
disconnected spectral route is indicated by these logs, nor is an optimal
larger weight established.

[Code] Legacy `dev:tta_gsd.py` reports `mean_raw_spectral_low` by averaging
MSE loss values. This is not a gradient norm. If the historical small numbers
refer to that column, comparing them with today's `local_spectral_grad_norm`
is a metric mismatch. Genuine historical norm logs must be assessed separately.

[Code/Derivation] For the same graph, selected rank M, XYZ signal and batch B,
legacy loss uses `w_old * E/(B*3*M)` and v1 `w_new * E/(3*M)`.
The low-band equivalent is `w_new=w_old/B`; old weight16 at B70 corresponds
to about .2286, not a stronger coefficient than current weight1. V2 uses
`3*N`, with N2048. The coefficients cannot be transferred numerically.

## The leading mechanism hypothesis: fidelity to an imperfect target

[Code/Derivation] On active reference graph vertices, smooth fidelity is
`L = tr(E^T F E)/(3*N)`, `E = Y-R`,
`F = U diag(exp(-beta*lambda/mean_active_degree)) U^T`.
Its clean-prediction gradient is `2*F*(Y-R)/(3*N)` and is backpropagated
through the frozen denoiser to local state and conditioning.

[Inference] This penalizes departures from the corrupted encoded reference;
it does not directly estimate the clean cloud or penalize the predicted
signal's roughness. If useful denoising changes a corrupted low-frequency
component, fidelity can resist that correction. Rotation/shear and incorrect
graph topology need not be confined to high frequencies. This failure
mechanism is plausible, but has not been measured by a clean-reference or
per-example outcome diagnostic.

[Code/Run] Smooth weights cover all active modes. Gaussian effective mass
`sum_i w_i` is about 1260 at beta .5 and 646 at beta2, compared with v1 M100's
roughly 138 selected modes. Effective mass is not a hard mode count or a
denoising-quality measure. Still, these are broad profiles, not a minor
rounding of the M100 boundary. At normalized eigenvalue 1, weights are
exp(-.5)=.607 and exp(-2)=.135. Total norm calibration does not preserve
per-mode emphasis or gradient direction.

[Code/Inference] Current graph rejection is a 6%-of-mean directed-degree
cutoff at gamma=.6/k10. The legacy graph uses 60% of mean symmetric degree,
with different neighbor/self conventions. Weak components can occupy low
frequencies; smoothing does not repair a graph built from corrupted anchors.
For a disconnected component, its constant mode has zero eigenvalue and
weight one. Current numerical zero-mode counts are not exact connectivity
counts and do not prove that the retained modes are outliers.

[Paper] GSDTTA learns low-frequency coordinate shifts using eigenmap-guided
self-training and also adapts the classifier. Our fixed-reference latent
fidelity has a different objective and cannot inherit the paper's mechanism
or success claim: [paper, sections 3.1-3.2](https://arxiv.org/html/2507.18225v1).

## Conflict and calibration transfer: measured limits

[Run] Recomputed from the raw 113047 reference, 192 shared-state probe
observations per corruption/profile (64 examples x 3 steps):

| Profile | Gaussian local median cosine | Impulse local median cosine | Gaussian style negative fraction | Impulse style negative fraction |
|---|---:|---:|---:|---:|
| beta .5 | .228501 | .202259 | 10.4167% | 8.8542% |
| beta 2 | .185580 | .170070 | 38.0208% | 36.9792% |

[Run/Inference] Both smooth profiles have zero negative local cosines in
these probes. A blanket explanation of destructive local SCD/spectral
conflict is therefore unsupported on the measured states. Positive cosine
does not establish semantic usefulness or complete redundancy. The style
route has more conflict, particularly at beta2, but negative cosine alone
does not establish accuracy harm. Final decoding uses original shape style;
updated conditioning still affects subsequent local denoising steps.

[Run/Inference] Calibration targets the local median on Gaussian/Impulse
reference states; it does not match style ratios, every corruption or every
reverse step. Background uses 35 steps versus five for the calibration
corruptions and has materially larger aggregate spectral/SCD ratios, as
shown above. Background improves slightly in both candidates, so this
transfer mismatch is not evidence that Background caused the overall loss.
Full-screen archives lack per-sample cosines/predictions; conflict/outcome
association cannot be recovered from their aggregate diagnostics.

## Why a genuinely small legacy gradient could help

[Inference] Class predictions depend on direction as well as displacement
size. A small step aligned with a useful classifier-margin direction can
change borderline predictions; a larger step in a different subspace may
not. Denoiser/decoder sensitivity and sampling can mediate the final effect.
The existing artifacts do not identify which samples changed or establish
this mechanism for the historical runs.

[Code/Inference] Legacy defaults also change the trajectory: normal/background
steps 10/30 versus 5/35, batch70 versus32, low400 plus mid400:600 weighting,
graph policy and LION mode. The standalone eval script settings are not proof
of the user's actual successful runtime settings. The eval-mode baseline
itself previously improved by .7577 pp in a seed-controlled all-15 comparison.
Any legacy benefit must be split into host changes and spectral increment.

[Code/Inference] If a historical run used the separate PxP projection path,
`g_c' = g_c - (g_c dot g_s)/(||g_s||^2+epsilon) * g_s` can change SCD strongly
even for a small spectral norm; scale approximately cancels when the norm
squared dominates epsilon. Ordinary GSD and the current smooth run do not
apply this projection. A branch name is not evidence it was enabled.

[Run/Open] The earlier beta2/rho .001 small-pool result changed from +1.171875
to -.781250 pp after rebuilding the reference/runtime. Coefficient drift
was only about .0051%, while package/binary identities and nondeterministic
execution also differed. This does not identify the cause. Matching seeds
alone does not ensure identical execution; see [PyTorch reproducibility](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md).

## Next evidence that distinguishes the hypotheses

[Inference/Proposal] Recover the actual successful legacy command and full
run bundle first. Compare the original host, that legacy host with every
spectral term disabled, and the same legacy host with its successful spectral
settings, using matched per-example random draws. This isolates host changes
from the spectral increment; treat `main_gsd_tta.py` SCD-zero defaults separately.

[Inference/Proposal] On identical frozen latent states, compare legacy and
current graph/profile directions at matched applied update norm. Record graph
components/degree, mode contributions, decoded displacement and classifier
margin changes, with example IDs. Then isolate the implicated graph, profile,
or conditioning route in a declared comparison. Store paired predictions so
corrected versus broken examples and uncertainty can be inspected. This is
a diagnostic proposal, not authorization to launch another run or select
parameters on the same test outcomes.

[Falsifier] A positive matched legacy-on minus legacy-off increment would
support a real historical spectral benefit. A host-off gain with no spectral
increment would favor the host explanation. A matched hard-versus-smooth
comparison that favors smooth would contradict attributing the null to the
profile alone. Clean-reference/margin diagnostics showing beneficial spectral
directions would weaken the corrupted-anchor explanation.

## Recomputable evidence

- Full input ZIP names, config SHA-256 hashes and source revisions are in
  `result/modelnet40_c/gsd_latent_spectral_smooth_v2/analysis_20261001_smoothing/diagnosis.json`.
- `audit.py` beside that JSON reads the nine full-screen bundles and the raw
  113047 calibration bundle, checks counts/accuracy, and recomputes the new
  scope, norm-ratio and cosine findings. Raw archives are not modified.
- Code anchors: `graph_spectral.py:88`, `graph_spectral.py:129`,
  `graph_spectral.py:199`, `tta_gsd.py:132`, `tta_gsd.py:169`,
  `gsd_calibration.py:34`, and `git show dev:tta_gsd.py`.
- Historical evidence: `gsd_branch_comparison_20260923.md`,
  `gsd_interaction_results_20260928.md`, `gsd_guidance_math_20260926.md`,
  and `modelnet40_c_all15_results_matrix_20261001.md`.
