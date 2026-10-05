# Method Synthesis and Hypotheses

## Unguided versus smooth-only - 2026-10-04

[Run/Inference] All15/seeds0-2 unguided61.2021% versus fixed beta .5,
alpha8.140161 smooth-only61.3596% gives an observed +0.1576 pp in all three
seeds. Native binary hashes differ between arms; repeated seeds do not remove
the method/build confound. This is a small positive development observation,
not a proven causal gain. Recorded smooth-only runtime is about4.46 times
unguided on the reported hardware. Historical same-smooth+SCD still has a
nonpositive increment versus SCD (-0.0279 pp).

[Run] Background is23.6224% unguided and23.8790% smooth-only despite35 actual
reverse steps, versus60.6699% in historical SCD-only. Other14 means are
63.8863/64.0368/64.1092%, respectively. Background dominates the descriptive
SCD-versus-no-SCD gap. Neither this pattern nor the negative descriptive
interaction proves harmful gradient conflict. Reuse common-state probes
before introducing projection or more parameters. See the
[accepted results and limits](gsd_guidance_ablation_results_20261004.md).

## V1 spectral-only Background evidence - 2026-10-04

[Run/Inference] The missing full Background seed0 run gives 23.1767%; the
archived 14 rows plus this run give a descriptive 61.1967% composite.
Against the archived full-screen SCD-only seed0, other-14 means differ by
-0.0434 pp but Background by -37.7229 pp. Thus the observed composite
deficit is concentrated in Background. Different revisions/native binaries,
one seed and unmatched random draws prevent a causal SCD-removal claim.
Wait for unguided and smooth-only Background outcomes before attributing
the failure to spectral fidelity, the graph, or absence of SCD. This does
not isolate smoothing. See the [validation report](../result/modelnet40_c/gsd_latent_spectral_v1/analysis_20261004_background/validation.md).

## Measured smoothing limits - 2026-10-01

[Run/Inference] The [smoothing diagnosis](gsd_smoothing_diagnosis_20261001.md)
supersedes the pending-v2-accuracy status below. Both selected smooth profiles
have nonpositive full-screen mean increments, including when restricted back
to full Gaussian/Impulse. Weighted spectral gradients are finite and applied;
small unweighted norms alone do not explain the null. Broad fidelity to a
corrupted latent reference and changed legacy graph/trajectory remain mechanism
hypotheses. Calibration has no negative local smooth/SCD cosines but measurable
style conflicts; accuracy causality is unproven. No all-15 matched hard-profile
arm exists, so a smoothing-specific regression has not been isolated.

## Proposed mathematics after paper reread - 2026-09-26

[Paper/Inference] GSDTTA's low-frequency shift and our low-frequency fidelity
have different roles. A source-faithful shift requires a driving adaptation
objective; it is not a spectral-matching loss. For scalar guidance in the
existing 3DD-TTA host, fixed-reference common-basis smooth spectral fidelity
is implemented as opt-in `gsd_latent_spectral_smooth_v2`; see the
[mathematics and test plan](gsd_guidance_math_20260926.md). It removes hard
rank boundaries but does not solve corrupted targets, wrong topology or slot
correspondence. The implementation is unverified on model data and has no
accuracy evidence; v1 and its archived results remain unchanged.

## Selected first GSD integration - 2026-09-23

[Code] The implemented opt-in `gsd_latent_spectral_v1` keeps the existing
SCD trajectory and adds static latent-XYZ spectral fidelity. It uses
`sum_b ||U_b^T(Qhat_b-Qref_b)||^2/(3*m_b)`, a detached common basis and
reference, and derivatives through the denoiser to local state and style.
Original encoded style is retained for final decode; weight zero delegates
directly to the original path. See [the mathematical design and audit](gsd_integration_20260922.md).
[Inference] H1 below is now testable under matched controls, not confirmed.
Low-band fidelity can also retain corruption. Physical graphs, dynamic bases,
all-feature distances, global diffusion and projection remain separate ideas.
[Open] Colab pilot and all-15 evidence are pending. Historical code descriptions
below refer to legacy branches unless explicitly superseded by this entry.

## Conceptual synthesis

3DD-TTA supplies a hierarchical generative prior and robust instance anchoring. GSDTTA supplies a graph-frequency view of point-cloud structure. PixelAsParam supplies a way to reason about competing guidance directions. The thesis synthesis is:

```text
corrupted point cloud
  → LION global/local latents
  → DDIM denoising trajectory
       ├─ SCD gradient: preserve reliable observed geometry
       ├─ spectral gradient: preserve/shape graph-frequency structure
       └─ conflict rule: combine or project gradients conditionally
  → decoded adapted point cloud
  → frozen Point-MAE classifier
```

This is a new composition. Evidence for each source method does not automatically validate the composition.

## Baseline objective

**[Code, corrected 2026-09-12]** At a reverse step, the baseline predicts a clean local latent and computes SCD between its first three channels and the original corrupted local latent's first three channels (`tta.py:68-75`). Decoder is called only after denoising (`:87-90`), not inside the loss loop. The gradient updates the noisy local state and conditioning representation. The hypothesis is that the LION prior removes corruption while latent-space SCD limits semantic drift. Earlier decoder-space wording is superseded; do not silently implement physical-space SCD during cleanup.

## Static latent spectral guidance

`tta_gsd.py` computes a graph eigenbasis from the original corrupted local latent and penalizes selected graph-frequency differences between the predicted clean latent and the original latent.

**Hypothesis H1:** low-frequency latent components contain corruption-resistant global shape information, so preserving them reduces semantic drift during SCD-guided reconstruction.

**Falsifiers:** no low-frequency energy concentration; no relationship between spectral loss and classification; matched runs perform no better or become less stable than 3DD-TTA.

**Risk:** preserving the corrupted latent spectrum can preserve corruption, especially if the graph itself is corrupted.

## Multiband guidance

The fork exposes low, mid, and high spectral bands.

**Hypothesis H2:** different corruptions require different bands—low frequencies for global structure, mid frequencies for parts, and carefully weighted high frequencies for local detail.

**Falsifiers:** band ablations show no consistent corruption-specific pattern after multiple seeds, or gains arise only from total gradient magnitude.

Use band-normalized losses or log band size and gradient norm. A sum over 400 coefficients and a mean over 16 coefficients are not comparable by their nominal weights.

**[Code]** Current mean spectra plus summed SCD have batch-dependent relative scale; changing batch size is a method factor as well as a random-number/compute factor. **[User report]** Mean trials gave smaller losses and higher accuracy than sum trials. Raw mean/sum values are not directly comparable. The user deferred sum plus smaller eta/gamma work until the baseline/spectral-off gates. Smaller shared rates change SCD too; later tests need individual band-count normalization, gradient/update logs and separate equivalence/weight/rate controls. See [audit section 3](code_audit_20260912.md).

## Dynamic eigenbasis

Dynamic mode periodically recomputes `U_current` from the changing predicted latent. Both the current prediction and the original target are projected into this current basis.

**Hypothesis H3:** an updated graph better tracks the denoised manifold and avoids anchoring optimization to a corrupted initial topology.

**Falsifiers:** eigenbasis instability dominates, runtime is prohibitive, or static basis is equally accurate under matched settings.

Mechanism diagnostics should use subspace distances/projectors for bands, not only raw eigenvector signs, because eigenvectors are sign-ambiguous and can rotate within near-degenerate eigenspaces.

**[Inference, clarification 2026-09-12]** Since original/predicted signals use the same current basis, shared sign flips and orthogonal rotations within a complete selected band preserve Frobenius/MSE loss. The concern is changing band projectors and near-degenerate spaces crossing a band boundary, not sign flips alone.

## Physical-basis-to-latent guidance

The physical variant builds `U` from XYZ but applies it to aligned local latent channels.

**Hypothesis H4:** LION retains point correspondence, allowing a physically meaningful topology to organize learned local features.

**Falsifiers:** correspondence tests fail; permuting latent point order does not behave as predicted; physical-basis loss does not correlate with decoded geometry or accuracy.

## Global/local diffusion

Sequential and synchronized variants denoise/adapt the global latent as well as the local latent.

**Hypothesis H5:** global latent correction helps with global corruptions such as scale/rotation/shear, where local SCD alone is insufficient.

**Precondition:** determine whether the decoder expects raw `shape_latent`, transformed `style_cond`, or another representation. Without this contract, a result cannot be interpreted as a clean global-latent ablation.

## Conflict-aware gradient composition

Let `g_s` be the weighted spectral gradient and `g_c` the weighted SCD gradient. The legacy `pxp-gradient-projection` branch's one-way mode preserves `g_s` and projects `g_c` when `g_s · g_c < 0`; its symmetric mode modifies both. Current `gsd-smooth-spectrum` uses their ordinary weighted sum and has no projection.

[Run/Inference] The [2026-10-04 composition review](gsd_scd_gradient_composition_review_20261004.md)
distinguishes local magnitude dominance from style conflicts using archived
common-state smooth probes. Weighted local sums rotate SCD by median
.054-.568 degrees, with no local conflicts in that scope. This is not v1
evidence. Obtain missing v1 geometry and prediction complementarity before
choosing per-example style projection, scale control, or separate trajectories.
SCD-priority projection protects SCD locally but need not protect spectral
progress; the proposed symmetric control explicitly tests both objectives.

**Hypothesis H6:** destructive SCD–spectral conflicts cause unstable or suboptimal steps; conditional projection improves final classification or reduces variance.

**Falsifiers:** conflict frequency is negligible; projection improves loss geometry but not accuracy; an equal-norm simple sum performs the same; benefits disappear with per-sample projection/common random numbers.

## Required ablation matrix

| Family | Minimum ablations |
|---|---|
| Baseline | source-only, LION reconstruction, original 3DD-TTA |
| Spectral | SCD only, low only, mid only, high only, low+mid+high, power off/on |
| Basis | static, dynamic with several intervals, physical basis |
| Scale | mean/sum or normalized equivalent, matched gradient norms |
| Style/global | static original decode, updated style decode, sequential, synchronized |
| Projection | none, spectral-priority, Chamfer-priority control, symmetric, per-sample versus batch |

Do not run the full matrix immediately. Use the staged protocol in [experiment_protocol.md](experiment_protocol.md) and promote only well-motivated configurations.

## Interpretation guardrails

- An accuracy gain with a different scheduler is not evidence for spectral guidance.
- A gain on a 25-sample prefix is a pilot signal, not a benchmark result.
- A lower spectral/SCD loss is not necessarily better classification.
- A negative gradient cosine is not automatically destructive; test the post-step outcome.
- ModelNet40-C results from GSDTTA's DGCNN/CurveNet/PointNeXt protocol are context, not a target for Point-MAE here.


## 2026-10-04 local/style objective routing

[User report/Code/Inference] New proposal: local SCD with spectral style,
plus its reverse as a control. Neither low-frequency guidance nor LION's
hierarchy establishes a one-to-one loss/block correspondence. Loss gradients
reach style through the local-prior Jacobian; block routing can still have
cross-block interference. Style-off anchors distinguish useful new guidance
from merely removing old guidance. SCD already adapts conditioning, while
global latent diffusion is inactive in the current GSD path.

[Planning] See [design](gsd_block_routing_design_20261004.md),
[implementation plan](gsd_block_routing_implementation_plan_20261004.md), and
[Colab phases](colab_gsd_block_routing_20261004.md). Initial style PCGrad is
per-example symmetric projection with an explicit cap at ordinary-sum norm,
and an ordinary-sum control at that applied norm. This cap is a documented
variant, not silently identical to standard PCGrad. Global diffusion,
CAGrad, norm calibration, ensembling and scheduling have separate evidence
gates; none is implemented by this planning record.
