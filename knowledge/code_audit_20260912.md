# Follow-up Code Audit: Baseline and Research Variants

## Evidence, revisions, and limits

Date: 2026-09-12. Fork: `pxp-gradient-projection`, `53ba252519c7cf65f836a9c1c564027142ab1573`. Original LION: clean `7711b3d185752eeb632d095494876e4de15f3195` at `D:/Akademik/Okul/Thesis/code/LION`, origin `https://github.com/nv-tlabs/LION.git`.

Static source/history audit only; no local accuracy run or archived Colab evidence. Approximately 63% baseline / 63.5% GSD remain **[User report]**. Line references apply to the audited revisions. No Python behavior was changed.

## 1. LION dropout: confirmed mode discrepancy, unmeasured accuracy effect

**[Code]** Original LION's documented evaluation route is `README.md:89-94 -> script/eval.sh:1 -> train_dist.py:83-99 -> trainer sampling`. `trainers/train_prior.py:657` calls `self.model.eval()` on the VAE. `trainers/train_2prior.py:50-128` dispatches each global/local prior to samplers that disable dropout:

- `utils/diffusion_pvd.py:231`: DDPM `model.eval()`.
- `utils/diffusion_pvd.py:398`: DDIM `model.eval()`.
- `utils/diffusion_pvd.py:512`: denoising from a supplied timestep `model.eval()`.
- `utils/diffusion_continuous.py:188`: ODE `dae.eval()`.

Discrete samplers restore `train()` after the loop, not during inference. VAE sampling in `trainers/hvae_trainer.py:192` also calls eval.

**[Code]** Original `demo.py -> models/lion.py` does NOT call eval. The original and fork wrapper files are byte-identical, SHA-256 `A6FDC21B7A51CE52F6AA114E6481905319D309D3A9359D8FEA801BD2F3C737E4`. The omission is inherited, not introduced by GSD additions. Fork `main_3dd_tta.py:66` and `eval_gsd_tta.py:73` set only Point-MAE to eval. `tta.py:36-41` freezes VAE/local-prior weights, without changing module modes. `no_grad()` and loading state dicts do not disable dropout; see [PyTorch eval documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.eval).

**[Code]** Supplied `lion_ckpts/unconditional_all55_cfg.yml:63` has `ddpm.dropout=0.1`, reaching the local prior, encoder and decoder via `models/latent_points_ada_localprior.py:41-49`, `models/latent_points_ada.py:197,252`, and active `nn.Dropout` in `models/pvcnn2_ada.py:218,431`. The global encoder uses dropout-capable PVCNN components too. The global prior uses `sde.dropout=0.3`, but ordinary `tta.py` does not call that prior. Active BatchNorm effects have not been established.

**[Inference]** Dropout-off is consistent with original LION's trainer inference protocol and may change accuracy/variance; no improvement magnitude or direction is established. This audit does not prove which runtime modes generated the published 3DD-TTA results.

**Next test / falsifier:** paired original-TTA legacy versus VAE/priors eval, Gaussian/background, full selected-corruption data, seeds 0,1,2, fixed batch/configuration and explicitly controlled interpolation/VAE/noise draws. Keep sampling-state/style gradients enabled. Eval does not remove posterior sampling or initial diffusion noise. Repeated null/negative changes weaken dropout as the accuracy-gap explanation.

## 2. Baseline/GSD comparison is not currently isolated

**[Code]** Default differences:

| Factor | Baseline | GSD |
|---|---|---|
| DDIM set_alpha_to_one | Unspecified; installed default | Explicit False |
| Normal/background reverse steps | 5 / 35 | 10 / 30 |
| Final decoder style | Original shape_latent | Updated style_cond unless use_static_style |
| Batch size | CLI 40; README example 32 | CLI 70 |
| Local/style guidance rates | gamma / eta | eta / gamma |

Sources: `tta.py:26-30,83-89`, `tta_gsd.py:42-46,216-226`, `main_3dd_tta.py:19,129-135`, `eval_gsd_tta.py:19,51-55`, `README.md:84`. Paper lambda 0.96 and repo default 0.95 are separate configurations.

Commit `f63587b` introduced False under a native-LION alignment rationale, but the wrapper uses DDPM, not the original TTA DDIM. Serialize runtime configs; a commit title is not parity evidence. Commit `58247f7` swapped gamma/eta; current baseline and GSD no longer match. Equal 0.01 rates hide the difference; unequal sweeps need semantic correction/control.

**[Code]** Current LION config `style_mlp: ''` (`:130`) makes `global2style` identity (`models/vae_adain.py:120-127`). The important final-decode difference is original versus updated style; other checkpoint configs may differ. Updated style still influences baseline local-prior predictions even though final decoding uses original style.

**Decision:** require spectral-off GSD tensor/prediction parity under identical settings/noise, then enable only a spectral term. A reported approximately 0.5-point difference cannot currently be attributed specifically to spectra.

## 3. Spectral mean versus sum: scale is part of the algorithm

**[Code]** GSD mean spectral MSE (`tta_gsd.py:186-194`) averages over batch, band and channels; SCD is sum (`:203`). For identical tensors and K = B * band_width * C, L_sum = K * L_mean, including the gradient scale relation. Relative spectral/SCD per-sample guidance changes with B. Moving B from 70 to 32 increases that relative scale by about 70/32 (2.19) at fixed weights, independently of stochasticity. The last partial batch can differ too.

**[User report]** Earlier sum trials performed worse than mean settings; mean gave smaller reported spectral losses and higher accuracy. Exact runs/configurations are not archived. History confirms mean/sum changes (`8a48017`, `6712c11`), not their measured outcomes.

**[Inference]** Smaller mean loss values are expected normalization, not proof of a closer match across reductions. Sum might overpower SCD or preserve corrupted spectral features, but this is not a demonstrated cause. In legacy symmetric PxP, `spectral_reduction` changes low-band MSE only; mid/high and power remain mean (`tta_pxp_sym.py:186-194`). Do not call that an all-band sum trial.

**Decision:** defer sum plus smaller eta/gamma to future controlled batches, as requested by the user. Start the minimal new spectral path with mean, without claiming intrinsic superiority. Log comparable normalized losses, weighted gradients and actual update norms. Smaller shared rates weaken SCD as well as spectra; they are not an equivalent reduction conversion. A future equivalence control can scale each spectral band's weight by 1/K while retaining SCD/rates, with actual B and individual band counts, before independent weight/rate sweeps.

## 4. Graph construction: equation discrepancy and unvalidated choices

**[Paper/Code]** Local GSDTTA PDF page 4, Eq. (10), gives gamma/(N*k) times the adjacency sum; `graph_spectral.py:80` uses gamma/N. For the same A and k=10, current tau is ten times the equation's value. `09580b2` removed an earlier extra k but still did not divide by k. Mapping paper A to the fork's max-symmetrized A needs validation. Keep the visual equation/symbol gate before code corrections; extracted distance prose/notation is ambiguous. Primary source: [GSDTTA paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.html).

**[Code/Open]** Query/reference points are identical (`graph_spectral.py:60`) and no self-neighbors are explicitly removed. Check actual Colab indices. Self-edges would add w_ii=1 to degree/threshold but cancel in D-A; remove them before degrees if exclusion is selected. Test installed kNN distances on known coordinates; `dist ** 2` is not yet a confirmed double-square bug.

**[Code]** Max symmetrization, filtering both endpoints, and isolated-node diagonal 1000 (`:73-101`) are algorithm choices, not merely numerical fixes. Test tiny graphs, outlier fractions, Laplacian spectrum and jitter. Dense eigendecomposition at B-by-2048-by-2048 makes B=70 a resource risk. These cannot explain a graph-free baseline accuracy gap.

## 5. Defects and evidence-quality risks

- **[Code] Conditional missing-key failures:** `tta_gsd.py:72` indexes four spectral keys despite earlier get defaults. `any()` short-circuits: positive low weight can hide missing later keys, so default/old calls do not always fail. Missing-key failures arise when evaluation reaches an absent key; a demo using only the old spectral key fails at spectral_low. Check `main_gsd_tta.py:92`, `eval_gsd_tta_fast.py:74`, `demo_gsd_tta.py:98`, `grid_search_mid_high.py:80`, and PxP counterparts. Earlier unconditional-KeyError wording is superseded.
- **[Code] Resume:** `eval_gsd_tta.py:169-177,210` sums all existing rows without validating duplicates, requested corruptions or config, then divides by current requested count. Fresh full runs without resume are not thereby proven wrong. PxP/pure-LION evaluation has similar patterns.
- **[Code] CSV:** symmetric PxP header has 11 fields and rows 13 (`eval_pxp_sym_tta.py:198,224`); delta1/delta2 appear before accuracy but not in header. Reduction is not recorded.
- **[Code] Analyzer:** graph returns full H_orig; `spectral_analyzer.py:100,134-136` compares it to H_pred sliced to M. M<N gives incompatible MSE shapes; do not trust this analyzer.
- **[Code] Diagnostics:** eval averages batch-mean losses equally; use sample-weighted totals for run-wide interpretation.
- **[Code] Search:** `grid_search_tta.py:163` uses a first-25 prefix and uncontrolled randomness. This is a biased pilot/test-set tuning protocol, not confirmatory evidence. Its ShapeNet cls_dim=50 (`:55`) should be 55; unrelated to ModelNet observations.
- **[Code] PxP:** `tta_pxp.py:231-239` and symmetric counterpart flatten entire batch for conflict decisions. Sample conflicts can cancel and updates depend on batch composition. Conflict/norm/projection-outcome diagnostics are missing. The decomposition is spectral/SCD, not PixelAsParam's actual denoising/diversity/classification directions.

## 6. Correct method interpretation

**[Code]** Baseline `tta.py:68-75` computes SCD on the first three channels of predicted/original local latents, reshaped B-by-2048-by-C. Decoder is called only at `:87-90`. Prior notes describing decoder-space SCD are superseded; do not implement physical-space SCD as silent cleanup.

**[Code/Inference]** Dynamic mode projects both target and prediction into the same current U. Shared sign flips and orthogonal rotations inside a complete selected band do not change Frobenius/MSE loss. Eigenspace crossings at band boundaries and changing graph projectors remain risks. Track band projectors/subspaces, not raw signs; broad earlier sign-instability wording is superseded.

## Decision and next work

The user endorsed rebuilding incrementally from main. `main`, `origin/main`, `upstream/main` currently equal `107305fd7baf40b359f31c07d235599198be7324`. This clean ancestry still contains the dropout omission and sparse logs. Preserve old branches/PDFs/results; do not wholesale cherry-pick old variants. Follow [clean restart batches](clean_restart_batches.md). Branch creation and implementation are not part of this documentation update.
