# Selected GSD design

[Code] Method: `gsd_latent_spectral_v1`, **GSD-inspired latent spectral guidance**.
Host: raw/eval LION, EMA off, frozen Point-MAE and original-style decoder.
Detailed source audit and implementation plan: [integration record](gsd_integration_20260922.md).

[Paper] LION supplies point-structured local latents; 3DD-TTA guides their
predicted clean XYZ during reverse diffusion. GSDTTA supplies the low-frequency
graph representation, but also learns physical point shifts and adapts a model.

[Inference] The compatible first extension is a scalar fidelity penalty on
predicted clean latent XYZ. Encode once to `(z_enc,h_ref)`. Build a detached,
static nonself kNN RBF graph on `Q_ref=h_ref[:,:,:3]`; use a symmetric
combinatorial Laplacian, exclude isolated vertices and retain the lowest M
modes, extending numerically unresolved boundary eigenspaces.

[Inference] For selected orthonormal U_b and actual rank m_b:

`L_spec = sum_b ||U_b^T(Qhat_0,b-Q_ref,b)||_F^2/(3*m_b)`

`L = L_SCD + w*L_spec`

`h_prev = DDIM_prev - gamma*grad_h_t(L)`

`s_next = s_t - eta*grad_s_t(L)`

[Inference] The clean-latent derivative is
`2*U_b*U_b^T*(Qhat_0,b-Q_ref,b)/(3*m_b)`. The chain rule passes it through
the denoiser to noisy local state and conditioning. The basis/reference stay
fixed. Summing per-sample losses avoids legacy batch-mean scaling; the
normalization makes the selected rank explicit. This is an additive scalar
objective, with no gradient projection.

[Code] Existing SCD, scheduler, 5/35 reverse steps, preprocessing and FPS are
preserved; final decoding uses `z_enc`. Weight zero calls baseline directly.
Initial graph values: k=10, delta=.1, graph gamma=.6, M=100; w=1.

[Open] A corrupted low-frequency target may retain corruption. Latent bandwidth
and guidance weight are not established optima; dense N=2048 eigendecomposition,
band expansion and inherited CUDA coordinate gradients need measurement.
CPU tests establish contracts, not accuracy benefit. Follow the
[Colab pilot and confirmation protocol](colab_gsd.md).

[Code, 2026-09-26] The separate `gsd_latent_spectral_smooth_v2` proposal adds
smooth all-active-mode weighting and fixed `3*N` reduction; it does not replace
or reinterpret v1. See the [v2 test plan](gsd_smooth_spectrum_test_plan_20260926.md).
