## Guidance-domain and computational reread — 2026-11-03

[Paper/Code/Run/Inference/Open] [Eleven-source focused review and derivations](../gsd_guidance_math_efficiency_20261103.md)
separate valid fixed-basis reconstruction from topology/decoder metric mismatch.
Explicit low-band shifts are a parameterization of a driving objective; low-band
corrupted-reference fidelity instead penalizes changes. Dynamic graph code uses
a frozen current metric, omitting its graph-Jacobian term intentionally; it is
not full differentiation of a geometry-dependent metric. Sparse square-root heat
Chebyshev loss is a prospective compute-equivalent approximation of smooth heat,
not of legacy rank-M truncation and not an established accuracy contribution.
Small legacy/component duplicate-computation and invariant-metric repairs preserve
toy guidance parity; no new GSDTTA reproduction/GPU inference or baseline promotion.

# Paper Note: GSDTTA

## Primary-paper/GFT retrospective reread — 2026-11-01

[Paper/Code/Run/Inference/Open] Venue PDFpp3–5/Eqs3,5–13,17–18/Fig2–3
visually checked again, plus experiments/ablation and supplement.
[Full audit and new asserted CPU witnesses](../graph_spectral_retrospective_audit_20261101.md): forwardU^TX,
layoutBx2048x4, common-basis loss and gradient are consistent; no IGFT call
is needed to evaluate that scalar coefficient loss. Paper optimizes explicit
low-frequency shifts in physical XYZ; our fidelity resists shifts from corrupt
latentQ. Legacy tau lacks Eq10's1/k and includes self edges; self-only isolation
bug and hard-cutoff tied-subspace ambiguity remain bounded issues. Neither
establishes the cause of accuracy changes. Mixed within-dataset head effects
and two-original-object/30cell CPU probes preclude dataset-only/general claims.
No production change or GPU inference; cite the paper for its mechanism and
our matched archived evidence for our method's behavior.

## Mathematical reread - 2026-09-26

[Paper] Visually rechecked local PDF pp.3-5, Fig.2/3 and Eqs.7-18. Eqs.11-13
give `X_s=X+U_M DeltaC`: low frequencies are adapted, high-frequency input
coefficients retained. Eq.17 is pseudo-label CE plus entropy/diversity and
one-direction Chamfer, not a two-spectrum matching loss.
[Inference] Low-band fidelity to corrupted input is a different hypothesis
from allowing low-band shifts. The chair energy plot does not establish latent
spectral-fidelity benefit. Eq.9 requires an explicit symmetric construction
for the stated eigensystem; Eq.10 sums affinities despite distance wording.
See the [reread and proposed guidance mathematics](../gsd_guidance_math_20260926.md).
Selected proposal: common-basis smooth fidelity in latent XYZ. Implemented
as v2 with CPU checks; not accuracy-validated. V1 remains the comparator.

## First GSD integration audit - 2026-09-23

[Paper] PDF p. 4 / Eqs. 7--13 was visually inspected: Eq. 10 uses
gamma/(N*k). Eq. 7 displays d squared in the exponent while nearby prose
calls d squared Euclidean distance; this is a printed ambiguity.
[Code] The new static latent graph explicitly selects standard Gaussian RBF
with squared Euclidean distance, excludes self-neighbors, computes threshold
from directed adjacency, max-symmetrizes, masks both endpoints and excludes
final isolates from the eigensystem. These are declared implementation
choices, replacing the legacy gamma/N and isolated-node 1000 penalty.
[Inference] The transferred idea is low-frequency structural fidelity in
LION latent space, not the paper's learned physical-coordinate shift or
alternating classifier adaptation. Use **GSD-inspired latent spectral
guidance**. See [design and limitations](../gsd_integration_20260922.md).

## Reference

Xin Wei et al., “3D Test-time Adaptation via Graph Spectral Driven Point Shift,” ICCV 2025. Local PDF: `Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf`. Official page: <https://openaccess.thecvf.com/content/ICCV2025/html/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.html>.

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


## Comparison-row provenance addendum — 2026-10-25

[Paper/Code/Open] [Fresh primary audit](../3dd_tta_citation_reproduction_audit_20261025.md):
§4.2 PDFp5 says common pretrained backbone weights across compared methods
and results from running published code. The3DD rows in Tables1/2 PDFpp6/7
are therefore **author-reported DGCNN reruns**, not originalPoint-MAE copies
or independently artifact-verified exact reproductions. They report71.69
ModelNet and57.08Scan; source66.51/55.72. CurveNet/PointNeXt3DD cells and
ShapeNet3DD are NA. The supplement omits a dedicated3DD configuration;
weights/data hashes/seeds and precise point/severity contracts remain open.
The actual ModelNet3DD PDF row hasRotation59.93/Cutout72.77; a web parser
returned incorrect alternatives. Use the archived PDF/fingerprintCSV.
Current local alternative-backbone studies remain unmatched to paper assets.

## Wider source-family and CloudFixer-variant context — 2026-10-25

[Paper/Code/Run/Inference/Open] [Expanded reporting audit](../pointcloud_tta_reporting_audit_20261025.md)
recovers SVWA supplementary CurveNet ModelNet source64.0, agreeing with our
64.0113. GSD71.38 is a different unresolved source family, not the only published
CurveNet reference. BFTT3D ordinary-error conversions yield source71.53CurveNet
and65.35DGCNN onModelNet, near GSD71.38/66.51 but without asset parity proof.
GSD's CloudFixer baseline description includes classifier/model updates, whereas
original CloudFixer distinguishes input-onlyCF fromCF-O. Do not automatically
map GSD's plain label to frozen-classifier input-onlyCF or compare their deltas
as matched. Exact baseline variant/assets/logs remain open.

## 2026-11-07 — Standalone GSD/LION literature and mechanism record integrated

[Paper/Run/Code/Inference/Open] The [Nov4primary-source reading log](../evidence/gsd_lion_literature_20261104/reading_log.md) and [synthesis](../gsd_lion_spectral_literature_20261104.md) are now published with the [Nov5four-hypothesis results](../gsd_hypotheses_results_20261105.md). Generic graph-posterior residual correction is acknowledged prior art; this is GSD-inspired LION guidance, not full GSDTTA replication.392frozen states/group do not establish classifier accuracy or SCD speed benefit. Uniform graph gain and identity-coordinate lift are unsupported; H1/H2superiority remains unestablished. [Inventory](../gsd_hypotheses_test_inventory_20261107.md) preserves failed/unfinished and unstarted tests and keeps later comprehensive H0–H8 naming separate.
