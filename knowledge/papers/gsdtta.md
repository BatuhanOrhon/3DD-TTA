# Paper Note: GSDTTA

## Reference

Y. Wei et al., “3D Test-time Adaptation via Graph Spectral Driven Point Shift,” ICCV 2025. Local PDF: `Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf`. Official page: <https://openaccess.thecvf.com/content/ICCV2025/html/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.html>.

## Core idea

GSDTTA treats each physical point cloud as an outlier-aware weighted k-nearest-neighbor graph. It computes the combinatorial graph Laplacian `L = D - A`, eigendecomposes it as `L = U Λ Uᵀ`, and maps coordinates to graph-frequency coefficients with `UᵀX`. Low-frequency components represent smooth/global shape variation; high-frequency components represent rapidly changing/local detail and noise.

The method learns an explicit point shift in graph spectral space and maps it back with the inverse graph Fourier transform. It also creates pseudo-labels by combining deep and graph-spectral/eigenmap information, and alternates input adaptation with model adaptation.

## Outlier-aware graph

The graph uses RBF-weighted kNN edges, degree statistics, and a threshold intended to suppress edges associated with outliers. This component is especially important for background corruption: the paper's ablation reports a large degradation when the outlier-aware construction is removed.

The exact threshold equation must be verified visually from the official PDF before further code changes. Text extraction can obscure whether the normalization contains `Nk`, and this fork's threshold formula changed during development.

**[Paper/Code, follow-up 2026-09-12]** Eq. (10), PDF page 4, has gamma/(N*k) times the adjacency sum, whereas current `graph_spectral.py:80` uses gamma/N. For identical A and k=10 this is a tenfold threshold difference. Keep visual/symbol/adjacency mapping as a correction gate: the code first max-symmetrizes A, and self-neighbor/distance conventions are untested. `09580b2` removed a previous extra k but left this normalization difference. See [audit](../code_audit_20260912.md); this does not explain the graph-free baseline gap.

## Spectral shift and adaptation loop

The paper's method is more than a spectral loss:

1. build a graph on the physical coordinates;
2. decompose the point cloud in the graph Fourier basis;
3. optimize an explicit low-frequency spectral displacement/shift;
4. reconstruct the shifted point cloud by inverse GFT;
5. infer eigenmap-guided pseudo-labels;
6. alternate several input-adaptation steps with a model-adaptation step.

Reported implementation settings include `k=10`, threshold factor `δ=0.1`, graph parameter `γ=0.6`, `M=100` low-frequency components, AdamW at `1e-4`, batch size 32, four input-adaptation steps followed by one model-adaptation step, and ten total adaptation steps. Consult the paper for the exact association of `α`, `β1`, `β2`, and `β3` with loss terms.

## Reported results and comparability

For the paper's ModelNet40-C protocol, Table 1 reports:

| Backbone | Source | 3DD-TTA | GSDTTA |
|---|---:|---:|---:|
| DGCNN | 66.51 | 71.69 | 79.07 |
| CurveNet | 71.38 | — | 82.63 |
| PointNeXt | 66.99 | — | 82.51 |

These values are **not directly comparable** with the WACV 3DD-TTA Point-MAE table. The backbone, source model, codebase, and evaluation protocol differ. They cannot be used as the expected accuracy for this repository.

The paper's ablation reports a full-method mean of 61.83 in its ablation setting, versus 57.28 without the spectral point-shift component, 58.35 without graph-spectral-guided model adaptation, and 61.20 without eigenmap-guided pseudo-labeling. These values show component relevance inside that protocol, not guaranteed gains in a LION latent.

## Difference from this fork

The fork transfers the graph-spectral intuition but not the complete algorithm:

- GSDTTA builds the graph on physical coordinates; the main fork path builds it on LION's local latent coordinates/features.
- GSDTTA learns an explicit spectral shift and applies inverse GFT; `tta_gsd.py` instead penalizes differences between spectral coefficients of the denoised estimate and the original corrupted latent.
- GSDTTA adapts the model in an alternating loop; the fork normally keeps Point-MAE frozen.
- GSDTTA focuses on a low-frequency shift; the fork exposes low/mid/high band losses and optional power terms.

Therefore use terms such as **GSD-inspired latent spectral guidance**, not “GSDTTA reproduction.”

## Risks and implications

- Dense eigendecomposition scales cubically with point count and may dominate runtime/memory at 2048 nodes.
- Eigenvectors have sign ambiguity and can rotate within near-repeated eigenspaces. **[Clarification 2026-09-12]** Shared sign flips/within-band orthogonal rotations do not themselves change complete-band same-basis Frobenius/MSE. Changing projectors and eigenspaces crossing a selected-band boundary are the dynamic risks; use subspace diagnostics. Earlier broad sign-instability wording is superseded.
- The spectral meaning of the first 4-dimensional LION local latent is not automatically the same as physical XYZ smoothness.
- Self-neighbor inclusion, distance units returned by `knn_cuda`, threshold normalization, graph symmetrization, and isolated-node handling can materially change the spectrum.

## Thesis use

GSDTTA supplies the structural prior and diagnostic vocabulary. A defensible synthesis must explicitly test whether LION latent spectra concentrate energy at low frequencies, whether spectral gradients correlate with accuracy gains, and whether the computational cost is justified.
