# LION: GSD integration notes

[Paper] LION, *Latent Point Diffusion Models for 3D Shape Generation*,
NeurIPS 2022, local `lion.pdf`, Sec. 3 (PDF pp. 3--4), Eqs. 5--7:
hierarchical global vector z and local point-structured h with three spatial
channels and extra features; the local diffusion prior conditions on z.
[Official paper/project](https://research.nvidia.com/labs/toronto-ai/LION/).

[Code] `models/latent_points_ada.py` concatenates point and extra-feature
channels per vertex and then flattens them. The local prior and decoder
reshape that layout back to B,N,C. For this checkpoint it is B,2048,4,
not B,4,2048. The spatial encoder mean has an input-coordinate residual;
latent XYZ is point-structured but not numerically identical to input XYZ.

[Code] The GSD graph and signal use only latent XYZ. The extra feature still
participates in the denoiser, so gradients may reach it indirectly. The graph
target is built once from the same sampled encoding used by baseline SCD.
No extra posterior or diffusion noise draw is introduced.

## Shape-latent follow-up — 2026-10-31

[Paper/Code] The hierarchy is q(z|X)q(h|X,z), p(z)p(h|z)p(X|h,z).
Official LION `7711b3d185752eeb632d095494876e4de15f3195`,
[`train_2prior.py:281–302`](https://github.com/nv-tlabs/LION/blob/7711b3d185752eeb632d095494876e4de15f3195/trainers/train_2prior.py#L281),
noises each prior's own latent but conditions the local prior on clean z.
App.C.1 of the [full paper](https://arxiv.org/abs/2210.06978v1), p.21,
also denoises global first, then local conditioned on the resulting clean z.
App.C.2 pp.21–23 fine-tunes encoders on corrupted/clean pairs with fixed
decoder/priors for noisy/voxel reconstruction; those demonstrations are not
evidence for our frozen-encoder source-free TTA protocol.

[Code] Current all55 config: z128; h2048×4; empty style MLP makes global2style
identity. Local encoder conditions on z and preserves input XYZ in its mean
via a residual. Decoder returns hXYZ plus `.01` times a style-conditioned
residual; this is not a derivative bound. Global prior is a residual 1×1
convolution network with SE/dropout; both priors are loaded in the current
common runner and explicitly put in eval/frozen mode. Switching z while
retaining old h changes a hierarchical pair; controlled reencoding is a
separate intervention, not a silent correction.

[Run/Inference/Open] The [focused report](../shape_latent_tta_research_20261031.md)
rehashes/recounts the old decoder control (−.031514pp three-seed macro),
recovers historical dual source but not the user's exact old run, and
analyzes180 physical cached z tensors. Missing clean z/posterior means
prevent a corruption-error ranking. Proposed bounded one-query global
clean estimate and later clean-estimate anchor guidance are unimplemented;
CPU algebra and artifact checks do not establish accuracy/runtime benefit.

[Inference] The direct host-compatible objective operates on predicted-clean
latent XYZ and backpropagates through the fixed local denoiser to its noisy
input and style. It does not require a physical-to-latent correspondence
model. See [selected design](../gsd_integration_20260922.md).

[Open] Inherited PVCNN coordinate/grid operators expose partial coordinate
derivatives, as recorded in `../code_audit_20260915.md`. CPU synthetic tests
cannot establish exact gradients through the installed CUDA extensions.

## 2026-10-31 — after clean-z oracle: mean refresh and conditional energy candidates

[Run/Inference] [New analysis/report](../shape_latent_clean_free_research_20261031.md):
97.8677% of aggregate diagonal posterior W2² clean/corrupt difference is mean
shift under the accepted64-object/all15/seed0 preparation. Not a causal
accuracy decomposition or an instruction to replace sampled z with mu.

[Code/Proposal] encode_global accepts decoded2048×3 LION-frame points and
returns mu/sigma without needing a new local encoder call. A proposed single
refresh uses z_d+(mu(E_z(D(h*,z_d)))−mu_d), preserving original posterior
sigma*eps. Must control against diffusion-off D(h_d,z_d) roundtrip and avoid
classifier-frame renormalization/FPS as encoder input. LION clean-encoder
fine-tuning results do not validate this frozen test-time cycle. Local
p(h|z) also supplies a possible noise-prediction surrogate energy direction
in z; not a calibrated conditional posterior with a few selected timesteps.
Global DDM stays off, no new implementation or GPU evaluation.

## 2026-10-31 — frozen global encoder refresh evaluated

[Code/Run/Inference] [Three-arm result](../shape_latent_refresh_results_20261031.md):
newmean+oldsigma*sameeps gives+.833333pp, full posterior+.3125pp on64all15
seed0Point-MAE; all conditional original-object intervals include0. Both use
LION-frame baseline decoded X*, fixed h0/local noise/anchor; global DDM0.
Clean-mean direction is mixed (median squared-error ratio .9535, mean ratio
1.0454); AE-only control has1.0076/1.1105. This is a frozen TTA extension
experiment, not validation of LION's fine-tuned denoising encoders or a proven
new generative prior. Two-pass deployment cost and native/seed variability
remain; no method/default/globaldiffusion activation follows automatically.

## 2026-10-31 — proximal global DDIM evaluated

[Run/Inference] [Fixed five-arm pilot](../shape_latent_proximal_results_20261031.md)
uses real global DDIM20/10/0 with half-target clean-estimate guidance. Proxy3
aggregate clean-z MSE falls10.4213% vs original, but accuracy+.78125pp has
conditional95%CI[−3.125,+4.6875]. Relative to prior3, both distance and point
accuracy improve; CI touches0. Direct mean has one more correct event.
Oracle3 nearly recovers the paired clean latent but remains diagnostic; no
formal upper-bound or compatibility-cause claim. Own-z decoder/h0/noise/RNG
and frozen model/source gates pass. Existing smoke/pilot predictions vary
despite matching states/RNG; no robust baseline gain or default promotion.

## 2026-10-31 — conditional local prior as a z-gradient source, untested

[Paper/Code/Inference/Open] [New review](../shape_latent_gradient_research_20261031.md)
source-binds the earlier conditional-energy proposal to Diffusion-TTA,
Diffusion Classifier and CWGF§3.2/Eq.21. Local ε_h(h_t,t;z) passes z as style
without detach; fixed detached h* permits real J_z ε_h-transpose residual
updates with frozen parameters. Stop-gradient on ε prediction would remove
this condition gradient. Full-channel MSE and native timestep indexing must
be preserved; finite-t weighted energy is not exact conditional likelihood.

[Proposal/Open] Baseline-derived h* can favor the old wrong z; evaluate
energy specificity/held-out noise, posthoc clean-direction alignment and
actual CUDA derivatives before global clean-estimate Jacobian transport.
Existing quadratic proximal path is no_grad and did not test this. LION's
native local DDPM score suffices to explore CWGF's condition-gradient idea;
LCM speed and known-operator convergence claims do not transfer. No new
implementation, native backward measurement, GPU result or default change.

## 2026-10-31 UTC / 2026-11-01 local — actual conditional z-gradient evaluated

[Code/Run/Inference] [Two-start result](../shape_latent_conditional_results_20261031.md)
uses original_style encoder z and last SCD updated_style as distinct global
initializations; current global2style identity makes this mapping explicit.
Shared detached terminal h* supplies full-channel four-time epsilon-MSE;
sequential condition VJPs plus global clean-estimate VJP preserve frozen
weights and paired state. Small actual CUDA finite differences pass, while
final guided clean-z MSE increases4.55%/4.58% versus respective starts.
First-step clean-condition negative-gradient cosine medians .141/.126 are
weak and decay; global guidance reduces drift vs prior but does not establish
clean recovery. Accuracy+.78125/−1.5625pp versus own controls, all conditional
intervals include0. SCD start gives no reliable advantage; no default change.
Native same-state/RNG prediction variation and h* self-confirming bias remain
limits. This does not invalidate the separate positive clean-z oracle or
establish a calibrated p(h|z) likelihood, novelty or high-accuracy method.
