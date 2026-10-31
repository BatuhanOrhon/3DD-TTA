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
