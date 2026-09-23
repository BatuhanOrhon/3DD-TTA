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

[Inference] The direct host-compatible objective operates on predicted-clean
latent XYZ and backpropagates through the fixed local denoiser to its noisy
input and style. It does not require a physical-to-latent correspondence
model. See [selected design](../gsd_integration_20260922.md).

[Open] Inherited PVCNN coordinate/grid operators expose partial coordinate
derivatives, as recorded in `../code_audit_20260915.md`. CPU synthetic tests
cannot establish exact gradients through the installed CUDA extensions.
