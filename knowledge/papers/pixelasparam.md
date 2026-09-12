# Paper Note: PixelAsParam

## Reference

T. M. Dinh et al., “Pixel-as-Param: A Gradient View on Diffusion Sampling with Guidance,” ICML 2023, PMLR 202. Local PDFs: `PixelAsParam_ A Gradient View on Diffusion Sampling with Guidance.pdf` and `dinh23a.pdf`. Official page: <https://proceedings.mlr.press/v202/dinh23a.html>.

## Core idea

PixelAsParam interprets the image being sampled as an optimization parameter and decomposes guided diffusion sampling into gradient-like directions. The paper identifies denoising, diversity, and classification-guidance directions, analyzes their pairwise interactions, and resolves empirically harmful conflicts with gradient projection.

For two directions `g_i` and `g_j`, a conflict exists when their dot product (or cosine similarity) is negative. The projected direction is of the form:

```text
g_i_projected = g_i - delta * (g_i · g_j / ||g_j||²) * g_j
```

Projection is applied selectively rather than to every pair. The paper argues that conflict matters through direction, magnitude, and local curvature; an indiscriminate projection can damage useful cooperation. Its ablation shows that projecting a theoretically non-conflicting denoising/diversity pair is harmful.

## What transfers to this thesis

The useful transferable principle is: when multiple guidance objectives update the same diffusion state, inspect their gradient geometry instead of blindly summing them. If SCD and spectral guidance are negatively aligned, one may project one direction away from the other or symmetrically modify both.

## What does not transfer directly

- PixelAsParam studies image diffusion and classifier guidance, not 3D point-cloud test-time adaptation.
- Its named directions are denoising, diversity, and classification. This fork projects spectral-loss and Chamfer-loss gradients.
- The paper's theoretical/empirical conflict analysis does not prove that SCD–spectral conflicts are harmful.
- The paper's best gradient definition and sampling details cannot be assumed optimal for LION latents.

The accurate name for the fork is therefore **PixelAsParam-inspired PCGrad-style guidance**, not a PixelAsParam implementation.

## Current fork mapping

- `tta_pxp.py` gives spectral guidance priority: if Chamfer conflicts with spectral guidance, the Chamfer gradient is projected away from the spectral direction before summation.
- `tta_pxp_sym.py` symmetrically projects both conflicting directions using the original counterpart gradient and configurable `delta1`/`delta2` values.
- Gradients are computed with respect to the noisy local latent and conditioning/style state.
- Loss weights are applied before gradient computation/projection, so weight scale changes both norm and geometry.

## Current diagnostic gaps

- Conflict is computed after flattening the full batch, so sample-level conflicts can cancel each other.
- Conflict rate, cosine distribution, gradient norms, projection frequency, and pre/post projection angles are not archived.
- The current evaluation CSV schema is incomplete; the symmetric script's header/row column count must be verified.
- **[Code, verified 2026-09-12]** Symmetric eval has 11 header fields but 13 row fields (`eval_pxp_sym_tta.py:198,224`), omitting delta1/delta2 from header and reduction entirely. The exposed spectral_reduction affects low-band MSE only; mid/high remain mean. See [audit](../code_audit_20260912.md).
- Denoising/diversity directions are not part of the projection, so the mechanism differs from the paper.

## Required experiments

1. Log per-sample and batch-level cosine similarities separately.
2. Report gradient norms before and after weighting/projection.
3. Compare no projection, one-way projection, symmetric projection, and an intentionally inappropriate control projection.
4. Hold diffusion noise fixed across paired runs using common random numbers.
5. Test whether projected steps actually improve the next-step losses and final classification, not only whether they remove negative dot products.

Only after these diagnostics should accuracy changes be attributed to conflict resolution.
