# Method Synthesis and Hypotheses

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

Let `g_s` be the weighted spectral gradient and `g_c` the weighted SCD gradient. Current one-way mode preserves `g_s` and projects `g_c` when `g_s · g_c < 0`; symmetric mode modifies both.

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
