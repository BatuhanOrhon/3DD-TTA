# Open Questions and Research Backlog

## 2026-09-23 legacy/current GSD diagnosis update

- [x] Verify current spectral-loss derivative analytically and on a synthetic
  CPU graph; no detached-prediction/sign error found.
- [x] Demonstrate that boundary-expansion tolerance can include distinct
  eigenmodes in a CPU example; see the comparison's spectral-math follow-up.
- [ ] Capture real latent Laplacians and compare current boundary policy with
  float64/eigenpair-residual references; preserve true repeated eigenspaces.
- [ ] Separate projector changes from actual-rank normalization changes;
  inspect sample-dependent rank and graph-degree weighting.
- [ ] Inspect connected components and selected-projector localization versus
  vertex degree under matched current/legacy graphs; numerical zero-mode
  counts alone are not component counts.

- [x] Compare current `761f47f`, dev `a458cd4`, pxp `53ba252` source defaults;
  record [the comparison](gsd_branch_comparison_20260923.md).
- [x] Inspect all M100/240/400 three-seed pilot gradient aggregates: local
  spectral/SCD mean-norm ratios are approximately .04--.10% at weight1.
- [x] Recompute archived pilot deltas; none has a positive three-seed mean.
  This supersedes pilot/graph-diagnostics-pending statements below.
- [ ] Obtain successful legacy command and complete raw run ZIPs; distinguish
  eval GSD, fast, main/spectral-only, dual, dynamic and PxP entry points.
- [ ] Compare the successful legacy host with all spectral weights off/on
  under matched controls before assigning its improvement to spectral loss.
- [ ] If its spectral increment survives, isolate graph filtering and
  low/mid-band loss with batch-adjusted weights on the same host.
- [ ] Diagnose current weight sensitivity and gradient alignment on validation
  data; a tiny nonzero norm does not imply that a larger weight helps.
- [ ] Same-process common-draw GPU prediction parity and all-15 GSD evidence
  remain open. Current lambda is .95; .96 is a separate SCD control.

## GSD v1 current gates - 2026-09-23

- [x] [Code] Add opt-in GSD-only method with raw/eval, EMA-off, frozen
  classifier and original-style decoder contracts; preserve baseline paths.
- [x] [Paper] Visually check GSDTTA Eq. 10 and declare graph distance,
  symmetrization and isolation choices; see [design](gsd_design.md).
- [x] [Code] Cover tensor layout, both gradient routes, batch sum scaling,
  zero-weight parity, graph invariance and artifact/worker failure handling
  with CPU tests. Include the review's float32 eigenspace regression.
- [ ] [Open] Validate native CUDA Chamfer/DDIM/LION execution in Colab and
  compare spectral-off predictions with matched original eval/raw runs.
- [ ] [Open] Measure N=2048 graph/eigendecomposition runtime and GPU memory,
  isolates, actual ranks, effective tolerances and reference energy fraction.
- [ ] [Open] Inspect spectral/SCD gradient scales for both local and style
  inputs; initial weight1 and delta.1 are uncalibrated latent-space settings.
- [ ] [Open] Validate complete three-seed pilot ZIPs and apply the declared
  promotion rule before all-15; retain null/negative/OOM evidence.
- [ ] [Open] Full GSDTTA reproduction, physical/dynamic graph alternatives,
  and classifier adaptation are not implemented by this method.

[User report] Earlier GSD deferral is superseded by this task; PxP and
combined-method work remain out of scope. No GPU result is added here.

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
- [x] Audit internal source-only provenance manifests: complete severity-5 runs share the same 15 corruption hashes, Point-MAE checkpoint/config hashes, and label hash across their differing run commits. Canonical archive/checkpoint byte identity remains open because the Colab assets are not present locally.
- [x] Implement the opt-in preprocessing identity control: direct corruption loading -> TTA preprocessing/output contract -> FPS(1024) -> frozen Point-MAE, with LION bypassed and the source-only comparator preserved. No GPU result yet; see the 2026-09-19 findings entry.
- [ ] Run and validate the locked preprocessing identity control on Colab: ModelNet40-C severity 5, all 15 corruptions, batch 32, seed 0, direct files, frozen Point-MAE, exact seven-file ZIP.
- [x] Run and validate the locked preprocessing identity control on Colab: ModelNet40-C severity 5, all 15 corruptions, batch 32, seed 0, direct files, frozen Point-MAE, exact seven-file ZIP. Result: 55.0243%, +1.3344 pp versus source-only; see findings log and raw ZIP.
- [x] Implement the locked pure VAE encode/decode control: raw VAE `encode` -> `decompose_eps` -> `sample`, with priors, scheduler, and guidance bypassed; no GPU result yet.
- [x] Run and validate the locked pure VAE encode/decode control because the identity delta is positive but modest. Result: 54.7947% (+1.1048 pp versus source-only, -0.2296 pp versus preprocessing identity); see findings log and raw ZIP.
- [x] Implement the opt-in pure VAE seed-stability diagnostic; it accepts only seed 1 or 2 and preserves the seed-0 pure VAE method contract. No GPU result yet.
- [x] Run and validate pure VAE seed 1 and seed 2 as separate complete seven-file ZIPs, then combine with seed 0 for mean/std analysis. Seeds 0/1/2 mean 54.8469%, sample SD 0.0790 pp, +1.1570 pp versus source-only; pure VAE is below matched preprocessing identity by 0.2296/0.1297/0.1405 pp at seeds 0/1/2; see findings log.
- [x] Implement the opt-in preprocessing identity seed-stability diagnostic; it accepts only seed 1 or 2, fully bypasses LION, and preserves the seed-0 identity contract. No GPU result yet.
- [x] Run and validate preprocessing identity seed 1 and seed 2 as separate complete seven-file ZIPs before interpreting pure VAE seed variance. Seeds 0/1/2 mean 55.0135%, sample SD 0.0602 pp, +1.3236 pp versus source-only; see findings log.
- [x] Diagnose and fix the first preprocessing seed-stability Colab routing failure: the new method reached the old TTA `process_batches` branch with `lion=None`; the failed ZIP is preserved as non-evidence. Rerun is pending.
- [ ] Decide whether a matched common-draw or separately isolated decoder/protocol control is needed before attributing the remaining source-to-TTA gap to diffusion guidance. Keep GSD/PxP, EMA, and other datasets parked.
- [ ] Optional P1: implement and run a same-commit common-draw pure-VAE versus eval/raw 3DD-TTA control with preprocessing fixed only if a causal diffusion/guidance thesis claim is required. Deferred for now because existing results already establish the accuracy ordering and the test is not expected to change it.
- [ ] Serialize and compare the complete DDIM scheduler config between `tta.py` and `tta_gsd.py`, including `set_alpha_to_one` and installed `diffusers` behavior.
- [ ] Determine the correct final decode input: raw global `shape_latent`, processed/updated `style_cond`, or another LION representation.
- [x] Implement the opt-in shared-trajectory original-versus-updated final-style decoder control. It reuses one local trajectory and records both decoder branches plus paired diagnostics; the Gaussian/Impulse pilot result is validated in the findings log.
- [x] [Code] Repair shared decoder control gradient context, metadata type collision and failed paired artifact rows; add CPU regression coverage including the actual trajectory loop.
- [x] Run and validate the shared-trajectory decoder control on complete Gaussian/Impulse severity-5 files at seeds 0/1/2. All three seven-file ZIPs are complete, eval/dropout inventories are closed, and the updated-style arm is better in 3/6 paired rows and worse in 3/6; all-15 confirmation is not supported. See findings log.
- [x] [Code] Extend the shared decoder-control scope guard to accept the complete canonical all-15 list while rejecting arbitrary partial scopes; preserve the original pilot scope and all eval/raw controls.
- [x] Run and validate the user-requested all-15 shared-trajectory decoder confirmation at seeds 0/1/2: 15 complete rows per seed, 111,060 paired examples total, pooled updated-style delta -0.0315 pp, with mixed corruption/seed signs. See findings log.
- [x] Implement the opt-in `scd_normalization_control` path. It preserves the original-style decoder, raw LION/eval mode, EMA-off policy, existing scheduler and locked gamma/eta/lambda values; it normalizes SCD by the original point-set cardinality and rejects arbitrary partial scopes.
- [ ] Run the Gaussian/Impulse SCD-normalization pilot at seeds 0/1/2, then decide whether an all-15 confirmation is justified.
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
- [x] P0: add and run the separately labelled preprocessing identity control (TTA preprocessing with LION bypassed); it gives 55.0243% versus source-only 53.6899%. Preserve the original direct-loading source comparator.
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
- [x] P0 next: run a preprocessing identity control that follows TTA normalization/interpolation/scale/rotation then bypasses LION, separating preprocessing from generative adaptation. It gives 55.0243% and closes 13.32% of the source-to-TTA gap.

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

### 2026-09-19 re-enable after smoke

- **[Code]** The opt-in `--fps-diagnostics` flag was re-enabled after the user
  reported a successful environment-rebuilt source-only smoke. It still runs
  legacy FPS/gather and records only aggregate diagnostics.
- **[User report]** Smoke execution passed; the complete seven-file ZIP remains
  pending, so the environment gate is not yet archived as `[Run]` evidence.

### 2026-09-19 completed legacy FPS diagnostic

- [x] Validate the complete seven-file archive
  `20260919-122509_source-only-fpsdiag-s5-seed0.zip` without changing the raw
  artifact.
- [x] Confirm that diagnostic accuracies exactly match the prior source-only
  predictions for the four tested severity-5 corruptions.
- [x] Confirm the origin-filter/padding signature for Density, Cutout, and
  Gaussian; treat LiDAR as a separate unresolved case.
- [ ] **P0 next:** extend diagnostics with per-input finite/NaN/Inf counts and
  coordinate-unique counts, prioritizing LiDAR, before changing any resampling
  policy.
- [ ] Keep alternate resampling policies unimplemented until the LiDAR cause
  is localized and a predeclared comparison is approved.

### 2026-09-19 FPS diagnostic v2 implementation

- **[Code]** Added finite/non-finite point, scalar NaN/Inf, finite-coordinate-
  unique, and selected-point finite/non-finite counters to the opt-in legacy
  FPS diagnostic. Classifier inputs remain unchanged.
- **[Open]** Run the predeclared severity-5, seed-0, batch-32
  `density cutout lidar gaussian` Colab command and request the complete
  seven-file ZIP. Interpret LiDAR only after these counters are archived.

### 2026-09-19 stale-code v2-named archive

- **[Run]** `20260919-133003_source-only-fpsdiag-v2-s5-seed0.zip` is complete
  and reproduces v1 accuracy, but its recorded commit is `36a2d60`; it lacks
  the v2 schema and all new finite/NaN/Inf and coordinate-unique fields.
- [ ] In Colab fetch the remote branch at `ceb9576` or newer, verify
  `git rev-parse HEAD`, then verify `config.json` contains
  `legacy_fps_v2_finite_coordinate_unique` before uploading the next ZIP.

### 2026-09-19 valid FPS diagnostic v2 result

- [x] Verify provenance: the archive records commit `0003743` and the v2
  schema, so its new counters are admissible `[Run]` evidence.
- [x] Establish that LiDAR has zero NaN/Inf points but approximately 371.77
  exact coordinate-duplicate slots per input example; FPS-index uniqueness and
  coordinate uniqueness nearly coincide.
- [x] Locate a repository-level candidate mechanism: the LiDAR generator
  samples 768 rows with NumPy's default replacement behavior at
  `datasets_mate/create_corrupted_dataset.py:655` `[Code]`.
- [ ] Reconcile the historical `.npy` generation provenance before treating
  replacement sampling as the definitive source-data explanation.
- [ ] Keep inference resampling changes and contribution claims parked; this
  result diagnoses the input artifact, not an FPS implementation fix.

### 2026-09-19 upstream LiDAR generator cross-check

- [x] Compare the forked generator with the canonical ModelNet40-C
  `data/generate_c.py`: both sample 768 LiDAR rows with NumPy's default
  replacement behavior.
- [x] Downgrade the `replace=False` explanation from a suspected local bug to
  an upstream-consistent benchmark construction detail.
- [ ] Preserve archive/Zenodo byte identity as a provenance question, but do
  not regenerate LiDAR or change inference FPS based on duplicate counts alone.
- [ ] Resume the broader source-only gap investigation after this gate; LiDAR
  duplicates are not by themselves a proposed contribution.
- [x] Identify the canonical ModelNet40-C Zenodo archive from the reported API
  metadata: `modelnet40_c.zip`, 1,970,686,633 bytes, MD5
  `c4a7fffaa52c80b33f7b3a0ac7782d3b`.
- [x] Extract the canonical archive outside the repository and compare all 15
  corruption-file sizes/SHA-256 values with the Colab source-only manifest.
  The supplied provenance report confirms all 15 archive members and all 15
  current Colab files match.
- [ ] Compare the Colab Point-MAE checkpoint hash with an author/canonical
  checkpoint hash; keep this separate from the corruption-archive gate.

### 2026-09-20 SCD normalization CPU/GPU verification boundary

- **[Code]** Local tests cover the pure SCD reduction, locked CLI scope,
  numeric denominator/retained-count metadata, and method configuration.
- **[Open]** Validate the real CUDA/Chamfer trajectory in Colab with a complete
  `scd_normalization_control` run. Accept the integration gate only when the
  seven-file bundle is complete, traceback-free, records the new SCD contract,
  and reports raw LION eval mode with dropout disabled.
- **[Inference]** Do not treat the synthetic CPU trajectory test as evidence
  that the installed Colab CUDA extension produces the same gradients; it is
  only a control-flow regression check.

### 2026-09-22 SCD normalization all-15 confirmation

- [x] Validate the three all-15 SCD normalization ZIPs: seven-file contract,
  CRC, complete rows, traceback, commit and hashes.
- [x] Confirm raw LION eval mode and disabled VAE/prior dropout before and
  after every run.
- [x] Compare against the matched shared-trajectory original-style arms.
  The normalized control is lower by **2.4806 pp** mean across seeds.
- **[Inference]** Preserve the original unnormalized SCD baseline; do not
  retune normalized guidance on the final all-15 test set.
- [ ] Run the separately controlled `.96` lambda test with original summed
  SCD, if approved. Keep gamma, eta, decoder style, scheduler, eval mode,
  data, checkpoint and seed list fixed.

### 2026-09-22 SCD lambda=.96 control implementation

- [x] Add the isolated `scd_lambda96_control` method with legacy summed SCD,
  original-style decoder, raw LION eval mode and EMA disabled.
- [x] Lock the method to lambdaa=.96, batch 32, seeds 0/1/2, complete files,
  severity 5 and the Gaussian/Impulse or all-15 scope.
- [x] Verify locally with 33 tests, syntax compilation and diff checks.
- **[Open]** Run and validate the Colab Gaussian/Impulse pilot before any
  all-15 confirmation.

### 2026-09-22 SCD lambda=.96 pilot result

- [x] Validate the three lambda=.96 pilot ZIPs, including the seven-file
  contract, CRC, completion, traceback, commit, hashes and eval/dropout state.
- [x] Compare against the matched lambda=.95 original-style pilot. The mean
  delta is **+0.0068 pp** with mixed seed directions.
- **[Inference]** Reject lambda=.96 as an operational improvement and do not
  run all-15 confirmation under the current predeclared rule.
- **[Open]** Keep any scale-matched Eq. 11 test separate from lambda and do
  not interpret it as evidence for the paper's fixed-rate setting.

### 2026-09-22 SCD lambda=.96 all-15 confirmation decision

- [x] Approve all-15 confirmation as a predeclared paper-setting check,
  despite the null Gaussian/Impulse pilot.
- [ ] Run complete ModelNet40-C severity-5 evaluation for seeds 0/1/2.
- [ ] Compare macro mean, seed SD and all 15 per-corruption deltas against
  lambda=.95 original-style runs.
- **[Open]** Do not tune lambda or any other factor using the all-15 result;
  the confirmation answers whether the paper-reported fixed `.96` setting
  improves the general benchmark average under this repository protocol.

### 2026-09-23 SCD lambda=.96 all-15 confirmation result

- [x] Validate the three all-15 ZIPs: seven-file contract, CRC, completion,
  traceback state, 15/15 rows and 37,020 examples per seed.
- [x] Confirm shared commit, classifier/LION hashes, dataset manifests and
  raw/eval dropout-off metadata across seeds.
- [x] Compare against matched original-style lambda=.95 rows: all three
  seed deltas are positive and the mean delta is +0.1855 pp.
- **[Inference]** Keep lambda=.96 as the separate SCD control result only;
  retain lambda=.95 for the operational baseline and GSD method.
- **[Open]** The existing GSD `.95` pilot cannot be relabeled as `.96`; a
  `.96` GSD pilot requires a fresh matched weight-zero/weight-one run.

### 2026-09-23 GSD lambda baseline separation

- [x] Keep `gsd_latent_spectral_v1` locked to operational lambda `.95`.
- [x] Keep the separate `scd_lambda96_control` evidence isolated from GSD.
- [x] Reject implicit or explicit lambda `.96` for the current GSD method.
- [ ] If desired, declare and run a separate future GSD lambda `.96` ablation.
### 2026-09-26 spectral-only pilot

- [x] Validate the three spectral-only ZIPs: seven-file structure, CRC,
  completion, Gaussian/Impulse scope, commit, lambda `.95`, raw/eval LION,
  EMA off, and `gsd_scd_weight=0`.
- [x] Record the paired seed deltas against the supplied original-style
  baseline: mean **+0.2296 pp** over Gaussian/Impulse.
- **[Inference]** Keep this as an exploratory spectral-only latent guidance
  ablation. It is not a GSDTTA reproduction and does not qualify for all-14 or
  all-15 promotion.
- **[Open]** The current implementation intentionally has no no-SCD off arm;
  design a separately declared control before making a claim about removing
  SCD versus retaining it.
