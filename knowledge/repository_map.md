# Repository Map

**Branch context, 2026-09-12:** active `baseline-repro-clean` contains main's baseline code. Variant paths and current-default comparisons in this historical map refer to audited legacy revision `53ba252` on `pxp-gradient-projection`; those variants were not migrated. See `clean_restart_batches.md` before implementation.

## Baseline execution path

```text
main_3dd_tta.py
  ├─ utilities_3dd_tta.py: ModelNet40-C loading and preprocessing
  ├─ models/lion.py: LION encode/decode and diffusion priors
  ├─ tta.py: latent perturbation, DDIM reverse process, SCD guidance
  └─ Point-MAE model: frozen downstream classification
```

### `main_3dd_tta.py`

Responsibilities:

- parse dataset/model/adaptation arguments;
- load ModelNet40-C and the source classifier;
- load LION;
- normalize each point cloud;
- interpolate to 2048 points when necessary;
- scale by `3.3885` and rotate axes for the LION convention;
- call `tta()`;
- undo rotation, normalize, farthest-point-sample 1024 points;
- classify and append an aggregate result.

Background corruption uses more denoising steps than other corruptions. The script does not establish NumPy/PyTorch/CUDA seeds and does not persist a complete run manifest.

### `tta.py`

The near-upstream baseline. It encodes the input, perturbs the local latent at a selected DDIM timestep, predicts a clean local latent, computes SCD on the first three local-latent channels against the original latent, and differentiates with respect to sampling state/conditioning. Decoding occurs only after the loop (`tta.py:68-90`); previous decoder-space SCD wording is superseded. The classifier remains frozen.

Current branch differences from upstream are operational rather than method-level: integer step handling and output-directory creation. This is important because the approximately 63% local baseline cannot be blamed on GSD code being injected into `tta.py`.

### `utilities_3dd_tta.py`

Contains ModelNet40-C loading, normalization, point interpolation, rotation, and sampling utilities. Severity is fixed to `_5` in the ModelNet-C path. Interpolation uses NumPy randomness when expanding point clouds, making runs stochastic unless seeded.

### `models/lion.py`

Defines hierarchical global/local latent encoding, diffusion priors, and decoder pathways. Different internal methods do not obviously use the same raw-versus-processed style representation at decode time. Any global-latent adaptation must first specify and test this contract.

The wrapper is byte-identical to original LION at revision `7711b3d`; neither wrapper calls eval. Original LION's trainer sampling does set VAE/prior eval, unlike current fork setup. This leaves active dropout in the current TTA path; accuracy impact is unmeasured. Current `style_mlp: ''` makes global2style identity, so original versus updated final style is the relevant difference for this supplied configuration. See the [audit](code_audit_20260912.md).

## Graph-spectral paths

### `graph_spectral.py`

Builds a dense outlier-aware RBF kNN graph and returns Laplacian eigenvalues/eigenvectors. Key implementation facts:

- main input is shaped as `B × N × 4` LION local latent;
- dense `B × N × N` matrices and `torch.linalg.eigh` imply high time/memory cost;
- the graph is symmetrized and thresholded;
- isolated nodes receive a large diagonal term intended to move them toward high frequencies;
- self-neighbor behavior and `knn_cuda` distance units still require direct tests.

### `tta_gsd.py`

Adds low/mid/high spectral coefficient losses and optional spectral-power loss to SCD. The static mode computes a basis once from the original corrupted local latent. The dynamic mode recomputes a current basis periodically and projects both the denoised estimate and the original latent into that basis.

This is a spectral fidelity regularizer, not an explicit inverse-GFT point-shift optimizer. It does not implement GSDTTA's classifier/model adaptation.

### `eval_gsd_tta.py`

Runs full evaluation and supports static/dynamic settings. Defaults differ materially from the original baseline and from the GSDTTA paper: spectral bands, steps, loss weights, and scheduler details must be included in every comparison. Resume logic and CSV metadata are not currently strong enough for cross-run aggregation without manual inspection.

### `tta_gsd.py` / `eval_gsd_tta.py`

The current branch integrates dynamic graph updates into these main GSD files. Use the term **dynamic eigenbasis** rather than “continuously updated `U_0` vector”: `U` is an eigenvector matrix, and it is recomputed only at the configured interval.

### `tta_gsd_physical.py` / `eval_gsd_tta_physical.py`

Build a graph basis from physical XYZ coordinates and apply it to corresponding local latent points. This assumes LION preserves point ordering/correspondence sufficiently for the physical basis to act on latent channels. That assumption is experimental and must be checked.

## Global/local diffusion variants

### `tta_gsd_dual_seq.py`

Sequentially denoises the global latent and then guides local latent denoising.

### `tta_gsd_dual_sync.py`

Denoises global and local states together. Later history changed it to update the global state continuously using gradients. The default evaluation path may use static style at final decode, so interpretation depends on the decoder contract.

These variants are research extensions, not components of the cited 3DD-TTA algorithm.

## PixelAsParam-inspired variants

### `tta_pxp.py`

Computes separate spectral and SCD gradients. On negative alignment, it projects the Chamfer gradient away from the spectral gradient, giving spectral guidance priority.

### `tta_pxp_sym.py`

Projects both gradients symmetrically when they conflict. Projection strength is controlled by two delta parameters.

### PxP evaluation/grid scripts

Provide pilot sweeps but do not yet archive mechanism diagnostics or a complete reproducibility manifest. The symmetric evaluation output schema appears to need correction because row fields and header fields can diverge.

## Analysis and test utilities

- `test_latent_knn.py`: documents the intended reshape of flat LION local latents to `B × 2048 × 4`; commit history reverted an incorrect transpose.
- `spectral_analyzer.py`: appears stale relative to the current `GraphSpectralDNA` return shape and must be validated before its outputs are trusted.
- `grid_search_tta.py`: evaluates a small leading subset of samples and is suitable only for pilot screening, not final claims.

## Configuration and environment

- `requirements.txt`: highly pinned legacy environment, including an older `diffusers` version.
- `env.yaml`: less constrained and uses a newer PyTorch/CUDA combination.
- The exact Colab setup is not committed: it must be archived with each run.

The difference between these environment definitions is a primary reproduction variable. Every archived run must state which path was used and include installed package versions.

## Safe method identifiers

Use these identifiers in output directories and tables:

| ID | Meaning |
|---|---|
| `source_only` | Classifier on corrupted input, no LION/TTA |
| `lion_recon` | LION reconstruction without guidance |
| `3dd_original` | Matched original `tta.py` path |
| `gsd_static` | Static latent spectral guidance |
| `gsd_dynamic` | Periodically recomputed latent eigenbasis |
| `gsd_physical` | Physical-coordinate basis applied to latent features |
| `dual_seq` | Sequential global/local diffusion |
| `dual_sync` | Synchronized global/local diffusion |
| `pxp_priority` | One-way spectral-priority projection |
| `pxp_symmetric` | Symmetric projection |

Never reuse a method ID when the semantic algorithm changes; add a suffix or version in `config.json`.
