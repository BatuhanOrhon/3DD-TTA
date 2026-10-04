# Local/style gradient routing: assessment and design

[User report/Code/Run/Inference/Open] 2026-10-04. Planning only, for execution
by another agent. Source inspected at `gsd-smooth-spectrum`
`aa8b725ba4daaf3448f0cda75bfc6aaec6c86d48`. This note extends the
[composition review](gsd_scd_gradient_composition_review_20261004.md).
No method code, GPU experiment or model training was added.

## User intent and interpretation

Evaluate local latent guidance by SCD with style guidance by spectral loss;
also consider the reverse assignment because the user separately mentions
SCD style gradients. Compare with style-only PCGrad and retain other
composition ideas as conditional alternatives. Provide implementation and
Colab handoff documents, without starting implementation here.

## Honest assessment

[Inference] Block routing is worth a small ablation. It removes direct
within-block competition by assigning each block one loss, and provides a
clearer mechanism test than another broad scalar-weight search. Its strongest
justification is experimental separability, not evidence that spectral loss
is inherently the correct objective for style.

[Code] With `h` the noisy local state and `s` conditioning, both objectives
are evaluated on `predicted_xyz = f_t(h,s)[:,:,:3]`. Therefore:

```text
g_spectral_style = J_style(f_t)^T * grad_xyz(L_spectral)
g_scd_local      = J_local(f_t)^T * grad_xyz(L_scd)
```

A low-frequency filter in latent XYZ does not guarantee a low-frequency,
semantic, or disentangled style update after this Jacobian. The two blocks
remain coupled through the local prior and all subsequent steps. Assigning
different objectives to different blocks generally is not gradient descent
on one common scalar objective. It can still be useful, but is not a proof
of conflict removal at trajectory level.

[Run] Existing smooth probes show local magnitude dominance and some style
conflicts; equivalent hard-v1 probes are missing. Spectral style gradients
are small at the archived weights. Replacing SCD style guidance with them
may behave almost like freezing style. A local-SCD/style-off control is
therefore essential, as is the reversed assignment's local-spectral/style-off
control. Retain original coefficients first; calibrate style scale separately
only if the native-scale result is inconclusive due to negligible updates.

## Three different meanings of 'update style'

1. **Conditioning guidance, already active:** `tta_gsd.py` updates
   `style_cond` using SCD and/or spectral gradients through the frozen local
   prior. SCD-only already updates both local and style states.
2. **Final decoder style:** current adaptation decodes with the original
   encoded `shape_latent`, not final `style_cond`. Updated conditioning still
   changes later local denoising. Its very last update has no downstream use
   under this decoder contract; a last-step-only style intervention should
   not change reconstructed points when all other states/draws are identical.
3. **Global latent diffusion:** `lion.priors[0]` is a pretrained diffusion
   model on raw global latent `z`; current GSD adaptation only calls
   `lion.priors[1]`. Running global diffusion adds a different prior, noise,
   state representation and trajectory. It is not enabling an unused SCD
   gradient. All model weights remain frozen; 'enable' means inference.

[Paper] LION trains the local prior conditioned on the clean global latent
and has a separate global diffusion prior (section 3, equations 6-7).
Its hierarchy motivates testing global/local roles, but does not establish
that spectral/SCD routing improves TTA. Feeding a noisy global state as if it
were clean conditioning changes the training-time conditioning distribution.
Source: [LION, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/40e56dabe12095a5fc44a6e4c3835948-Paper-Conference.pdf).

[Code] `models/vae_adain.py::global2style` can transform raw `z` through an
MLP. The recorded supplied configuration uses identity, but the implementation
must validate the active checkpoint configuration rather than assume identity
for every model. Preserve raw `z`, transformed conditioning and decoder style
as explicitly different variables. Do not apply the transform twice or infer
an inverse MLP. Set VAE/priors to eval in the runner, as current operational
`run_baseline.py` does; the wrapper alone does not establish eval mode.

[Run] The already completed shared-trajectory decoder test (all 15, seeds
0/1/2) gave original-style 63.7484%, updated-style 63.7169%, paired mean
delta -0.0315 pp; seed deltas +.0054/-.1270/+.0270 pp. Artifacts:
`result/modelnet40_c/shared_trajectory_decoder_control/`; detailed entry in
`findings_log.md`, 2026-09-20. This is evidence against expecting a generic
gain from updated final style. It does not test style-off or global diffusion.
Do not repeat that full evaluation or present it as evidence against all
conditioning adaptation.

**Decision:** test routing and conditional style projection first. A stable
style-guidance gain would motivate a global-prior study but would not predict
its result. Global diffusion could regularize useful style changes or move
the instance toward an incorrect plausible shape. It requires its own controls.

## Locked first implementation scope

- New opt-in method ID `gsd_guidance_composition_v1`; hard GSD v1 only initially.
- ModelNet40-C severity 5; frozen Point-MAE, raw/eval LION, EMA off, batch 32.
- Existing v1 graph: k=10, delta=.1, graph gamma=.6, modes=100 with recorded
  actual rank; preserve rank-normalized v1 loss, not hard-v2's normalization.
- SCD weight 1, spectral weight 1, local gamma=.01, style eta=.01,
  lambda=.95; DDIM grid 100, actual normal/background reverse steps 5/35.
- Original raw encoded global latent at final decode. Global prior unused.
- Preserve existing method defaults, zero-weight delegation, results and scripts.
- Routing, projection and scale must be independently serialized. Never rename
  a routed result as legacy 'SCD-only' or 'GSD-only'. No mean-reduction change.
- Claims require shared prepared inputs/noise, not just matching seed numbers.
- Future scripts/flags below are specifications, not currently runnable tools.

## Routing and composition contract

For each sample, compute original unweighted gradients for both losses and
both blocks, then weights exactly once: `c_b = 1*g_scd_b`,
`q_b = 1*g_spectral_b`, `b in {local,style}`. Block choices:

| Mode | Direction |
|---|---|
| `off` | zeros with the block's shape/device/dtype |
| `scd` | c |
| `spectral` | q |
| `sum` | c + q |
| `pcgrad` | Conditional symmetric projection of original c,q; cap summed norm at ordinary-sum norm |
| `scd_priority` | c unchanged; remove only opposing component of q |
| `sum_norm_pcgrad` | Ordinary sum scaled down to symmetric-PCGrad sum norm |

PCGrad projection occurs only if dot<0 and each norm exceeds `1e-12`;
dot/norm arithmetic is accumulated in float64 per sample, over nonbatch
dimensions only; cast the composed direction back to input dtype.
On near-zero inputs leave sum unchanged and log `projection_skipped`.
Use the exact original counterparts for the two projections. No additive
epsilon inside a denominator after a valid-norm check. Nonfinite inputs fail.
The raw symmetric sum can be larger than the ordinary sum. For `pcgrad`,
if raw projected norm exceeds `1e-12`, multiply by
`min(1, norm(c+q)/norm(raw_projected_sum))`; otherwise keep the tiny raw sum.
Log both raw and applied projected norms. This is explicitly a norm-capped
PCGrad variant; positive rescaling preserves its first-order sign property.
For numerical safety `sum_norm_pcgrad` uses scale
`min(1, norm(capped_pcgrad_sum)/norm(sum))`; if norm(sum)<=1e-12 return sum and log
the degeneracy. No-conflict output must equal the ordinary sum. This control
does not turn a tiny projected direction into a large step.

The norm control computes its hypothetical PCGrad direction at **its own
current state**. It matches directions' norms at that state, not necessarily
the actual norm sequence of a different PCGrad trajectory after divergence.
Report those trajectory norm distributions; a matched-state norm control
reduces a confound but does not make final accuracy a pure direction-only effect.

Local DDIM update remains `h_next = ddim_prev - gamma*d_local`;
style update `s_next = s - eta*d_style`. Route loss gradients, not graph
eigenvectors into style, and not total scalar losses into an optimizer.
Do not flatten a batch into one conflict decision. For per-example gradient
interpretation, assert operational eval mode and audit absence of cross-sample
coupling in the prior; batch-permutation checks supplement this audit.

## Required mechanism evidence

- Per-example predictions/logits and original indices; both-correct /
  SCD-only-correct / spectral-only-correct / both-wrong; oracle as an offline
  upper bound only, no labels in routing/calibration.
- Per step/block, raw and weighted norms, cosine, conflict, applied norm,
  angle from SCD, pre/post projection change, skipped/zero counts; style drift.
- Local signed guidance displacement versus DDIM displacement, including
  cosine and norm ratio. Do not call DDIM a fixed-objective gradient.
- A missing/unused gradient is explicitly flagged, not confused with a real
  connected zero gradient. Hard-v1 probes on SCD states must use v1 loss,
  not existing smooth-calibration hard candidates.
- Attribution compares intervention with the same local route and same
  magnitudes. `scd/pcgrad` versus `sum/sum` alone does not isolate projection.
- Any optional same-state loss response diagnostic freezes timestep and
  resets inputs; distinguish it from next-timestep loss and final accuracy.

## Alternatives and promotion boundaries

1. **Bounded magnitude calibration:** if style-spectral is effectively off,
   calibrate an explicit fixed style multiplier from label-free probe norms;
   a separate pilot tests it. Do not force equality across all samples.
2. **SCD-priority style:** interpretation control after nonzero symmetric
   projection effect; it can still sacrifice spectral progress.
3. **Local PCGrad:** only after actual hard-v1 local conflicts are observed;
   smooth local probes currently do not support it.
4. **CAGrad:** if two-objective conflicts are substantial and projection loses
   useful progress; compare at controlled norms. See primary
   [CAGrad sections 3.1-3.2](https://arxiv.org/abs/2110.14048).
5. **Separate-trajectory probability averaging:** only if paired predictions
   show useful complementarity, with equal-compute same-method ensemble controls.
6. **Timestep scheduling:** route changes only at existing reverse steps;
   make coarse five-step schedules explicit. Sequential same-step guidance
   needs recomputed gradients to differ from the sum; it is a separate cost.
7. **Frequency routing:** defer until an explicit Jacobian/subspace formulation
   exists; the N-point graph basis is not a basis of global style coordinates.
8. **Global diffusion:** separate gated study; begin with a frozen sequential
   global diffuse-denoise prior, not synchronized noisy-global conditioning
   copied from legacy dual code. Do not add training/unfreezing.

The literature basis and scope of
[PCGrad](https://arxiv.org/abs/2001.06782) and
[PixelAsParam](https://proceedings.mlr.press/v202/dinh23a.html) are recorded
in the preceding composition review. No cited method proves accuracy gains here.

## Execution documents

- [Implementation plan and agent handoff](gsd_block_routing_implementation_plan_20261004.md).
- [Colab scenarios, manifests, gates and future command contract](colab_gsd_block_routing_20261004.md).

[Open] No routing, style-only projection or global-prior accuracy result is
available. A routing gain no larger than style-off, a projection gain matched
by norm reduction, or unstable seed signs weakens the respective hypothesis.


## 2026-10-04 clarification: small relative gradients are not style-specific

[Code/Run] Small means spectral relative to SCD within the same state block,
not universally small absolute style gradients. Current SCD sums retained
Chamfer distances; v1 averages spectral coefficients over3*actual_rank,
smooth averages over3*N (N=2048); both current spectral losses sum samples.
There is no additional reduction applied only to style. Denoiser Jacobians
map the XYZ derivative into both blocks; filtering and residual orientation
also matter, so the measured cross-loss ratio is not simply1/(3*N).

[Code] `gsd_calibration.coefficient_summary` explicitly sets
alpha=rho/median(local spectral/SCD ratio). The discussed smooth beta.5 and
beta2 settings target rho=.001 and.01, respectively: small local contributions
were calibration choices. The same alpha is used for style without a separate
style-ratio constraint. Paired-probe median style ratios are about.067-.088%
and.708-1.181%; local about.097-.103% and.991-1.009%.

[Run] Existing hard-v1 M100/weight1 seed0 aggregate norm ratios already include
local Gaussian.0838%/Impulse.0968% and style approximately.164%/.0905%.
See gsd_branch_comparison_20260923.md. Thus v1 magnitude information exists;
missing v1 evidence is paired per-example geometry/complementarity, not all
style norm data. These ratios of mean norms must not be conflated with the
smooth per-example/probe medians. Style is not systematically relatively
smaller than local. Tiny relative norm neither proves a vanishing-gradient
fault nor predicts classification sensitivity or standalone effectiveness.


## 2026-10-04 implementation Batch 1

[Code/Verification] `gsd_composition.py` now provides per-example local/style
composition, weighted modes, JSON-safe norm/cosine/projection diagnostics, and
a capped symmetric-PCGrad update. CPU tests cover conflict/agreeing examples,
cap and same-state norm control, zero/tiny/absent gradients, batch permutation,
finite checks and input immutability. Test-first run observed the expected
missing-module failure, then `test_gsd_composition.py` passed 12 tests. A
separate task reviewer approved spec compliance and code quality. No sampler
integration or experiment support is complete yet.
