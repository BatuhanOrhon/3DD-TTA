# Open Questions and Research Backlog

## EMA inventory result — 2026-09-13

- [x] Inspect checkpoint: 462/462 prior EMA entries, 0 VAE EMA entries.
- [x] Add opt-in, shape-validated `--lion-ema-mode`; default raw loading is unchanged.
- [x] Run eval+raw versus eval+EMA on complete Gaussian and Impulse severity 5, seed 0: EMA +.4862 pp Gaussian, +.4052 pp Impulse, +.4457 pp macro; see findings log.
- [x] Repeat EMA at seeds 1 and 2 on Gaussian+Impulse: three-seed macro +.2431 ± .3865 pp, Gaussian mean -.0540 pp and Impulse +.5402 pp.
- [x] Screen all 15 corruptions under matched eval+raw/eval+EMA conditions: completed at seeds 0/1/2; see EMA result below.
- [x] Complete all-15 EMA screen at seeds 0/1/2: +.1035 ± .1639 pp macro; EMA remains an ablation, not selected baseline. See `ema_inventory_20260913.md`.
- [x] Complete matched all-15 legacy+raw versus eval+raw screen at seeds 1/2: eval is +.7577 +/- .1203 pp macro and higher on 13/15 corruption means. See `dropout_eval_mode_20260913.md`; eval is the provisional baseline, not a causal/common-draw result.

Priority meanings: **P0** blocks trustworthy comparison; **P1** blocks method interpretation; **P2** is valuable after the foundation is stable.

## P0 â€” Reproduction blockers

- [ ] Obtain a complete archived Colab run for `3dd_original` at the user's approximately 63% result.
- [ ] Confirm exact ModelNet40-C files, severity, sample counts/order, and hashes.
- [ ] Confirm Point-MAE and LION checkpoint paths and SHA-256 hashes.
- [ ] Record the resolved environment and compare `requirements.txt`, `env.yaml`, and notebook installations.
- [x] Establish fixed seeds and measure baseline variance over at least three runs (completed for source-only: zero variance across seeds 0, 1, 2).
- [ ] Recompute both macro per-corruption mean and micro total accuracy from raw counts.
- [ ] Explain whether the intended reference is the published 65.7 or README 66.1 protocol and why the tables differ.
- [ ] Resolve Table 2 severity attribution: the paper omits it; released 3DD-TTA code hard-codes severity 5, whereas official ModelNet40-C evaluation spans levels 1--5. Run a labelled source-only severity 1--5 matching probe before treating a lower-severity explanation as plausible.

## P1 â€” Baseline implementation questions

- [x] Compare original LION inference modes: trainer evaluation disables dropout; the inherited demo wrapper and current TTA setup do not. Accuracy causality is still open; see `code_audit_20260912.md`.
- [x] Broaden the LION mode A/B beyond Gaussian/Background: matched-commit all-15 seed-1/2 screen favors eval by +.7577 +/- .1203 pp and 13/15 corruption means. `dropout_eval_mode_20260913.md` records the full evidence. A common-random-number pair remains open if a causal dropout estimate is required.
- [x] Rerun the eval-mode Gaussian full Gaussian evaluation after the PVCNN autograd repair: it completed at 74.68% versus legacy 73.99%, with priors/VAE eval confirmed. This is a one-seed, non-common-draw pilot only; repeat seeds 1--2 and background before selecting the mode.
- [x] Create clean main-based branch reference: baseline-repro-clean at 107305fd7baf40b359f31c07d235599198be7324; legacy checkout/files unchanged.
- [x] Finish same-folder clean-branch checkout and preserve curated skill/knowledge/protocol plus source PDFs; user selected in-place development. Tracked code matches main; existing stash untouched.
- [x] Restore missing Point-MAE ChamferDistance import exposed by Batch 1 Colab smoke; this prevents model construction before TTA. Await rerun artifact.
- [x] Implement Batch 1 original-TTA smoke artifact runner, without old-variant imports or mode/math changes; local structural/protocol checks passed. See `colab_baseline_smoke.md`.
- [x] Receive and validate the Batch 1 Colab smoke ZIP: safe seven-file archive, correct count-based CSVs and intended runtime modes; Level 0 only, not accuracy evidence.
- [x] Implement and archive Batch 2 source-only all-15-corruption identity evaluation: 53.69% macro, 3.91 pp below README 57.6%; see findings log.
- [x] Compare source-only FPS/classifier preprocessing against upstream reference: identical direct load -> FPS(1024) -> `classification_only` path; no added normalization/augmentation. The downloaded ModelNet40-C package also has expected array/label structure. Author asset hashes remain unavailable.
- [x] Run and archive the standalone `data_original.npy` clean Point-MAE control: 90.64% (2237/2468), same checkpoint/labels/FPS as the corruption run; see `20260912-115021_clean-control_seed0`. This rules down an obvious clean-path failure.
- [ ] Locate a checkpoint-appropriate author-published clean reference or independent canonical reproduction; the current clean control is internal evidence, not external parity proof.
- [ ] With clean-path failure ruled down, investigate the remaining corrupted-source gap through corruption-asset/version provenance and runtime/FPS extension sensitivity before treating an adaptation delta as causal.
- [ ] Serialize and compare the complete DDIM scheduler config between `tta.py` and `tta_gsd.py`, including `set_alpha_to_one` and installed `diffusers` behavior.
- [ ] Determine the correct final decode input: raw global `shape_latent`, processed/updated `style_cond`, or another LION representation.
- [ ] Verify gamma/eta global/local semantics with unequal values and gradient norms.
- [ ] Confirm that checkpoint loading reports no missing/unexpected keys in every entry point.
- [ ] Quantify batch-size effects under fixed seeds/common random numbers.
- [ ] Require spectral-off tensor/prediction parity with the selected baseline before enabling new guidance.
- [ ] Test complete/default/partial loss-weight dictionaries; short-circuit direct-key access can hide failures.
- [ ] Validate count-based aggregation and immutable fresh-run schemas; prevent duplicate/config-mixed resume (new runner starts without resume).

## P1 â€” Spectral correctness

- [ ] Visually verify the GSDTTA outlier-threshold equation and map every symbol to `graph_spectral.py`.
- [ ] Validate the recorded gamma/(N*k) versus gamma/N threshold discrepancy with the same adjacency before correction; see the audit.
- [ ] Log spectral/SCD gradient and update scales with actual batch/band/channel counts; raw mean/sum loss numbers are not comparable fidelity measures.
- [ ] Test whether `knn_cuda` returns Euclidean or squared distances in this installed build.
- [ ] Test and, if necessary, exclude self-neighbors.
- [ ] Validate graph symmetrization, degree thresholding, isolated-node treatment, and numerical jitter on small known graphs.
- [ ] Measure spectral energy concentration of LION local latents by corruption/class.
- [ ] Replace raw eigenvector tracking diagnostics with band-projector/subspace measures near degenerate eigenvalues.
- [ ] Validate or repair `spectral_analyzer.py` before trusting it.

## P1 â€” Gradient projection mechanism

- [ ] Log per-sample conflict rates; compare with current batch-flattened conflict decisions.
- [ ] Log raw/weighted/projected gradient norms and cosine distributions.
- [ ] Correct and validate symmetric PxP CSV schema, including delta fields and reduction.
- [ ] Compare no projection, spectral-priority, Chamfer-priority control, and symmetric projection with common random numbers.
- [ ] Test whether conflict predicts harmful next-step loss/accuracy behavior rather than assuming it.

## P2 â€” Research extensions

- [ ] Deferred by user: sum spectral loss plus smaller eta/gamma after baseline/spectral-off parity, including reduction-equivalence controls and explicit SCD effects. Earlier mean advantage is [User report], not [Run].
- [ ] Compare static versus dynamic eigenbasis at matched compute and several update intervals.
- [ ] Validate physical-to-latent point correspondence before interpreting physical-basis results.
- [ ] Evaluate global/local sequential and synchronized diffusion after resolving style decoding.
- [ ] Explore sparse/approximate eigensolvers if the dynamic graph is beneficial but too costly.
- [ ] Extend beyond ModelNet40-C only after the core protocol is locked.

## P2 â€” Thesis reporting

- [ ] Predefine final primary/secondary metrics and statistical tests.
- [ ] Separate hyperparameter-development corruptions/data from final reporting.
- [ ] Record compute budget and sustainability/feasibility constraints.
- [ ] Prepare an ablation table that distinguishes algorithmic effects from scheduler/reduction changes.

### 2026-09-12 batch-size audit note

- **[Code]** In the original tta.py, the SCD scalar is summed across examples (sum_i l_i), but the optimized local latent is per example. Absent batch-coupled operations, d(sum_i l_i)/d(h_i) = d(l_i)/d(h_i); therefore this reduction alone does not multiply an individual example's guidance update when batch size rises. This conclusion does not automatically extend to legacy GSD/spectral variants with different reductions.
- **[Inference]** Batch size can nevertheless change the realized TTA accuracy: it changes random-number consumption/grouping, can interact with legacy train-mode dropout or batch-stat layers, and may select different CUDA kernels. Evaluate batch size as a separately recorded protocol factor after the current fixed-batch dropout A/B, not as an unqualified method improvement.

### 2026-09-12 background dropout A/B status

- **[Run]** Complete Background severity-5 legacy/eval pairs at seeds 0, 1, 2 are archived under `result/modelnet40_c/3dd_original/20260912-171041...` through `20260912-175310...`; all six have the required seven files and complete 2,468-example records.
- **[Run]** Eval minus legacy is +0.1621, -1.1750, and +0.6888 pp (mean -0.1080 pp). This contrasts with Gaussian's +1.2966 pp mean and all-positive seed deltas.
- **[Decision]** Eval mode cannot yet be selected globally. Broaden the paired corruption coverage with batch 32 before choosing a baseline mode; retain the non-common-draw caveat.


### 2026-09-13 15-corruption seed-0 mode screen

- **[Run]** All 15 ModelNet40-C severity-5 corruption pairs are now complete at seed 0 and batch 32. The 30 selected full-run artifacts are valid seven-file ZIPs with complete 2,468-example CSV rows.
- **[Run]** Macro: legacy 63.0578% (23,344/37,020), eval 63.8817% (23,649/37,020), delta +0.8239 pp. Eval is higher for 13/15 seed-0 corruptions and lower for Rotation and Shear.
- **[Superseded by Run]** The seed-1/2 all-15 repeat is complete under one commit and selects eval provisionally; see `dropout_eval_mode_20260913.md`. It is still not common-draw paired and does not justify a paper-parity claim.

### 2026-09-15 main/LION audit - queued after severity results

Full evidence and staged tests: `code_audit_20260915.md`. No new accuracy run.

- [x] Recheck remote/local main refs and inherited LION sources: 99/100 overlapping model/utility/operator files are identical after line-ending normalization.
- [x] Visually verify 3DD-TTA Eq. 11, Algorithm 1 and Table 2; fix the still-stale corruption headings in `papers/3dd_tta.md`. Paper source mean is 57.6%.
- [ ] P0: inspect installed Pointnet2 FPS indices for Density (649 points), Cutout (724), LiDAR (768) and Gaussian (1024): repeated indices, origin filter, unique group centers and extension identity. The source path requests 1024 for all; runtime kernel behavior remains to be measured.
- [ ] P0: add a separately labelled preprocessing identity control (TTA preprocessing with LION bypassed), then pure VAE reconstruction if needed. Preserve the original direct-loading source comparator.
- [ ] P1: compare old versus updated final decoder style using one shared denoising trajectory per input, full Gaussian/Impulse at seeds 0/1/2 after source-data gate review. Historical trial/reversion has no matched archived effect.
- [ ] P1: test Eq.-11-equivalent guidance scale (.01/2048 for both existing rates), separately from lambda .95/.96. This concerns baseline SCD, not the deferred spectral mean/sum study.
- [ ] P1: isolate NumPy RNG when adding paired classifier calls: current all-token Point-MAE inference still generates unused random masks. Demonstrate fixed-input logit parity before any mask-removal optimization.
- [ ] P2: quantify missing coordinate derivatives in inherited PVCNN operators using fixed-draw directional finite differences; account for neighbor/grid discontinuities before proposing a kernel change.
- [ ] P1: establish checkpoint-specific provenance for scale 3.3885 and all55 normalization; public PointFlow loader alone does not establish the historical all55 training path.

Existing eval/raw baseline and EMA/GSD/other-dataset decisions remain in force.

### 2026-09-16 source-only severity probe

- [x] Complete all-15 source-only severity 1--5 at seed 0, batch 32, frozen Point-MAE and direct loading. Macro: 75.8806%, 73.2739%, 68.5062%, 62.0205%, 53.6899%.
- [x] Validate five ZIPs: seven files, 15 complete rows, 2,468 examples/corruption, 37,020 total, config/CSV/file severity agreement and no traceback.
- [x] Confirm severity 5 reproduces the prior 53.6899% source-only result and the same classifier/label hashes are used across levels.
- [x] Record metadata caveat: all five `notes.md` files retain old generic smoke wording. Raw artifacts are preserved and not rewritten; the results remain complete with documented metadata debt.
- [x] Descriptive conclusion: severity strongly affects source accuracy (-22.1907 pp from s1 to s5), but no tested severity is a provenance match for paper source 57.6% (s4 +4.4205 pp, s5 -3.9101 pp).
- [ ] P0 next: inspect installed Pointnet2 FPS extension indices for Density/Cutout/LiDAR and Gaussian, including unique counts, repeats, origin-filter candidates and extension identity.
- [ ] P0 next: identify corruption archive/generator version and checkpoint provenance; seek a paper-specific severity statement or canonical asset hashes.
- [ ] P0 next: run a preprocessing identity control that follows TTA normalization/interpolation/scale/rotation then bypasses LION, separating preprocessing from generative adaptation.

Severity 5 remains the operational benchmark. No lower severity may be reported as
the paper's benchmark without new provenance evidence.

### 2026-09-19 FPS diagnostic runner

- **[Code]** `run_baseline.py` now accepts opt-in `--fps-diagnostics` for
  `source_only`. It runs the same legacy PointNet2 FPS/gather path and records
  aggregate unique-index, duplicate-slot, near-origin and index-bound statistics
  in `config.json`; the default path and classifier input are unchanged.
- **[Open]** Colab evidence is still required. The first diagnostic is predeclared
  for severity 5, seed 0, batch 32, complete `density cutout lidar gaussian`.
No alternate resampling policy is implemented or benchmarked yet.

### 2026-09-19 rollback of diagnostic runner

- **[Code] Superseded:** The opt-in `--fps-diagnostics` implementation was
  reverted in `cc52437` before any valid diagnostic run. The source-only runner
  is intentionally back to its pre-diagnostic behavior while the Colab
  environment is rebuilt.
- **[Open]** Revisit FPS diagnostics only after the historical dependency pins
  and source-only import path are restored and an import smoke passes.

### 2026-09-19 dependency failure before FPS evaluation

- [x] Identify the failed run's import source: `diffusers` evaluates
  `torch.xpu.empty_cache` while the installed PyTorch lacks `torch.xpu`.
- [ ] Record Colab versions for `torch`, `diffusers`, and `huggingface-hub`.
- [ ] Restore the historical compatible pins (`torch==2.0.1+cu121`,
  `diffusers==0.11.1`, `huggingface-hub==0.11.1`) or isolate source-only imports
  from unused LION/Diffusers dependencies before rerunning FPS diagnostics.
