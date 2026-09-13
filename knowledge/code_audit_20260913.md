# Reproduction audit: 2026-09-13

Scope: diagnose the remaining ModelNet40-C gap before GSD. Baseline revision `9ce5553de9277f9b9d7e26fc729e70b912a7c2d7`, branch `baseline-repro-clean`. No algorithm changes or GPU experiments in this audit. Existing dirty research documents and raw results are preserved. Evidence below distinguishes verified implementation differences from untested accuracy explanations.

## 1. Correct reference and location of the gap

**[Paper]** Published WACV PDF, Table 2, PDF page 7 / printed page 1572: column order is uniform, gaussian, background, impulse, upsampling, distortion_rbf, distortion_rbf_inv, density, density_inc, shear, rotation, cutout, distortion, occlusion, lidar. The older paper-note headings were incorrectly mapped; they are corrected by this audit. Text extraction resolves the headings and Eq. 11; local rendering succeeded but the image-view tool failed because of the Windows sandbox helper, so no visual-inspection claim is made.

| Corruption | Paper 3DD (%) | Local eval seed 0 (%) | Local minus paper (pp) |
|---|---:|---:|---:|
| uniform | 77.5 | 77.3501 | -0.1499 |
| gaussian | 79.1 | 74.6759 | -4.4241 |
| background | 49.9 | 60.8185 | +10.9185 |
| impulse | 80.3 | 69.8947 | -10.4053 |
| upsampling | 81.8 | 82.6175 | +0.8175 |
| distortion_rbf | 63.8 | 63.3712 | -0.4288 |
| distortion_rbf_inv | 66.9 | 65.1945 | -1.7055 |
| density | 79.3 | 71.7180 | -7.5820 |
| density_inc | 84.7 | 86.3452 | +1.6452 |
| shear | 63.7 | 65.8833 | +2.1833 |
| rotation | 33.4 | 32.9011 | -0.4989 |
| cutout | 74.7 | 71.0697 | -3.6303 |
| distortion | 68.2 | 64.9919 | -3.2081 |
| occlusion | 39.9 | 39.8298 | -0.0702 |
| lidar | 42.2 | 31.5640 | -10.6360 |
| Macro | 65.7 (reported) | 63.8817 | -1.8183 |

**[Run]** Local values are from the 30 full seed-0 legacy/eval archives indexed in the findings-log entry `15-corruption LION-mode screen at seed 0`, under `result/modelnet40_c/3dd_original/`. Every corruption has 2,468 examples, severity 5, batch 32, gamma=eta=.01, lambda=.95; 35 background / 5 other reverse steps, DDIM schedule 100. Macro legacy 63.0578%, eval 63.8817%, difference **0.8239 percentage points**, not 0.82% relative improvement. This remains exploratory, not a repeated-seed final benchmark.

**[Code/Paper]** The paper row sums to 985.4, mean **65.6933%**, consistent with reported 65.7. The repository README row substitutes density=78.5 and distortion=65.2; its 15 displayed values average **65.44%**, NOT its stated 66.1%. Preserve the README claim, but mark its arithmetic inconsistency instead of presenting 66.1 as an internally validated target. This does not establish why the mismatch occurred.

**[Run]** Source-only seed 0 (`result/modelnet40_c/source_only/20260912-111822_source-only_seed0/`) already averages 53.6899%, versus paper 57.6. Examples: density 65.2350 vs 75.1, lidar 19.9352 vs 29.1, gaussian 51.2966 vs 57.0. A LION-only issue cannot explain a gap that exists before LION runs. Impulse is particularly useful for adaptation diagnostics: source 55.3485 vs 58.8, but adapted 69.8947 vs 80.3.

## 2. Background reverse steps: hypothesis, not established error

**[Code]** Original upstream `main_3dd_tta.py` at commit `1e4815b7035f9b784bd991d62d1c000d467b8e09` already selects 35 steps for background and 5 otherwise. This is not a fork addition.

**[Paper]** Section 4.4, PDF page 7, explicitly describes up to 35 steps for background; page 8 discusses 35 reverse steps with a 100-step DDIM schedule for its Background schedule analysis. Table 2 alone does not identify its exact run configuration.

**[Inference]** Paper Background=49.9 does not demonstrate that Table 2 used 5 steps. An isolated background 5-vs-35 ablation can show sensitivity, not recover unpublished provenance. Do not degrade local background performance merely to match the paper. Matching severity-4 accuracy likewise would not prove the paper used severity 4.

## 3. EMA inference weights: high-priority unverified asset issue

**[Code]** `models/lion.py:33-34` loads only `dae_state_dict` and `vae_state_dict`. Original LION's demo wrapper does the same, but its trainer path differs:

- `../LION/trainers/train_prior.py:296-320`: resume loads model AND optimizer state.
- `:330-338`: saves prior model weights and `dae_optimizer` separately.
- `:647-699`: sampling swaps prior parameters with EMA when `cfg.ddpm.ema` is enabled, then swaps back.
- `../LION/utils/ema.py`: EMA tensors live in optimizer state entries under `ema`.
- `../LION/trainers/common_fun_prior_train.py:28-47`: prior optimizer uses `dae.parameters()` and is wrapped by EMA.
- Active config enables `ddpm.ema: 1`, `sde.ema_decay: .9999`.

**[Open]** Actual Colab checkpoint content must be inspected. It might already contain exported EMA model weights, lack optimizer state, or contain raw and EMA weights separately. The wrapper's omission alone does NOT prove we currently sample raw weights or explain any accuracy deficit. Original LION trainer code is evidence for LION inference practice, not proof of the 3DD-TTA authors' evaluation path.

Next diagnostic: CPU-load the existing checkpoint in Colab, archive checkpoint SHA256, top-level keys, optimizer parameter-group sizes, count/shapes of `state[*]['ema']`, and any export metadata. If present, validate parameter mapping before comparing raw vs EMA or installing EMA weights. Do not zip arbitrary `state_dict()` keys to optimizer entries; buffers/order can differ. VAE EMA swap is separately conditional on `eval.load_other_vae_ckpt` (false here), so do not blindly replace VAE weights.

Falsifier: checkpoint already contains the intended EMA parameters, or no distinct usable EMA weights exist. No EMA accuracy result exists yet.

## 4. Final decoder does not receive updated global conditioning

**[Code]** `tta.py:53,64-86` repeatedly updates `style_cond`, but `:89-92` passes original `shape_latent` to the decoder. Current `style_mlp` is empty, so `global2style` is identity and the updated representation is directly comparable after reshaping. LION VAE reconstruction/sampling uses processed style for decoding.

**[Paper]** Algorithm 1 / Eq. 12, PDF page 5, update global z0 and decode using z0. Code does not carry that final update into decoding. Updates still affect subsequent local-prior calls; do NOT say all style updates are unused. The final style update has no downstream consumer.

**[Code/history]** This was previously tried in fork commit `158eae8f5bc6385ee5b6944b21e387c35f5cede2`, then reverted in the author-baseline restoration `72335a3`. It is not a completely new idea. No matched artifact here isolates its effect.

Test: current eval baseline vs final updated-style decoding only, fixed assets/batch/rates/lambda/schedule/seed on Gaussian and Impulse (complete corruption files). Repeat only if a meaningful effect appears. A null/negative result would rule down this explanation, even if paper/code differ.

## 5. Eq. 11 SCD normalization is absent in released code

**[Paper]** Eq. 11 divides each directed retained-distance sum by that original point set's cardinality. **[Code]** `tta.py:75-77` retains the lowest fraction and sums both directions without those denominators. Both local point sets have 2,048 points here. Consequently at fixed rates, the released sum gives a per-example gradient 2,048 times the point-normalized formula (mathematical scale comparison, not a measured benefit).

This is **SCD**, not the deferred GSD spectral mean/sum experiment. Do not substitute `.mean()` over all batch entries: that introduces batch-size and retained-count denominators absent from Eq. 11. Summing separable per-example losses alone does not amplify a sample's gradient by batch size.

Equivalent isolated probe without changing the loss implementation: set both gamma and eta to `0.0000048828125` (= .01/2048), retaining lambda=.95 and all other settings. This matches the point-normalized guidance magnitude for this fixed point count; floating-point arithmetic need not be bitwise identical. It is a hypothesis test, NOT a recommended accuracy fix. Test separately from lambda=.96 (paper) vs .95 (code). With unequal rates, remember code gamma updates local state whereas paper gamma updates global state.

## 6. Checked paths that do not currently justify a fix

- **[Code]** Prior call at scheduler time t uses model time t+1, consistent with original LION's 1..T indexing; scheduler coefficients use zero-based t. Not an established off-by-one bug.
- **[Code]** DDIM 100-step schedule has [990,...,0]; selecting the last five means [40,30,20,10,0], and 35 starts at 340. `set_alpha_to_one=True` matches the final alpha=1 convention. CLI eta is a guidance rate, not DDIM stochastic eta.
- **[Code/Run]** Requirements and archived environment both use diffusers 0.11.1; generic diffusers-version drift is not supported for these runs. CUDA/FPS numerical sensitivity remains untested.
- **[Code]** Bbox-centering makes the positive max equal the absolute max; lack of an explicit abs here is not an established normalization bug. Axis rotations are inverse mappings. Classifier-training vs TTA normalization remains a protocol question, not a proven fork regression.
- **[Code]** Local latent layout is flattened (N,4) on both encoder and prior paths; first-three-channel SCD is not an apparent transpose error.
- **[Code]** LION posterior sampling remains stochastic in eval mode; eval does not replace posterior samples with means. This is consistent with the paper's sampling description.
- **[Code]** Active prior uses GroupNorm/AdaGN; a config string containing `bn` is insufficient evidence of active cross-example BatchNorm. Point-MAE is explicitly eval/frozen.
- **[Code]** The PVCNN repair's training flag gates backward index/weight storage, not the forward interpolation formula. Legacy train-mode behavior is unchanged by that conditional repair. Prior metadata caveat about different Gaussian seed-0 commits remains recorded, but inspection does not support treating this repair as a demonstrated legacy-forward confound.
- **[Code]** Counts, flattened labels and complete-run aggregation do not reveal a label-broadcasting or averaging error. Runner has stale generic smoke/legacy descriptions in some metadata; use actual arguments, mode inventory and counts. Those descriptions do not change predictions.

## 7. Next actions (supersedes immediate repeated full benchmark)

1. Inspect existing LION checkpoint EMA inventory in Colab; no inference or download required.
2. Run source-only severity **4**, all 15 corruptions, seed 0, batch 32, same Point-MAE checkpoint and label file. This cheaply probes the pre-LION gap. Compare the entire per-corruption pattern, not only the mean. Keep severity-5 reference results separate.
3. At severity 5, eval mode, pilot updated-style decoding and (if justified by inventory) EMA weights independently on complete Gaussian/Impulse files. Do not bundle changes. Normalized-SCD and lambda=.96 are separate later probes.
4. Background 5-vs-35 is optional provenance/sensitivity work, not the first action. Defer all-15 seeds 1/2 until selecting a candidate requiring confirmation; repeated benchmark is necessary for final claims, not for every diagnosis.
5. Keep ScanObjectNN/ShapeNet and GSD parked. Do not tune individual corruption settings to paper accuracies and call it reproduction. Label these tests exploratory; freeze any final protocol before reporting.

Each Colab evaluation must return the full seven-file ZIP under `result/modelnet40_c/<method>/`. Place lightweight checkpoint diagnostics separately under `result/modelnet40_c/diagnostics/<timestamp>_lion-ema-inventory/`, without credentials or checkpoint tensor dumps. Raw artifacts are never rewritten.
