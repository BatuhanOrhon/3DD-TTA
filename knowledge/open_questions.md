# Open Questions and Research Backlog

## EMA inventory result — 2026-09-13

- [x] Inspect checkpoint: 462/462 prior EMA entries, 0 VAE EMA entries.
- [x] Add opt-in, shape-validated `--lion-ema-mode`; default raw loading is unchanged.
- [x] Run eval+raw versus eval+EMA on complete Gaussian and Impulse severity 5, seed 0: EMA +.4862 pp Gaussian, +.4052 pp Impulse, +.4457 pp macro; see findings log.
- [ ] Repeat the promising EMA effect at seeds 1 and 2 before changing the selected baseline.

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
- [ ] Measure isolated dropout A/B after source-only identity checks: Gaussian is complete at seeds 0,1,2 and favors eval by +1.30 pp mean; run background at seeds 0,1,2 (35 reverse steps), then decide whether the effect generalizes. The runner is seed-controlled, not fully common-draw-paired.
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
- **[Open]** This remains an exploratory screen: Gaussian legacy seed 0 predates `9ce5553`, the 13 newly screened corruptions lack seeds 1--2, and pairing is not common-draw. Repeat the predeclared all-15 evaluation at seeds 1 and 2 under one commit before locking a global LION mode or making a benchmark claim.
