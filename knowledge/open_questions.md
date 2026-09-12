# Open Questions and Research Backlog

Priority meanings: **P0** blocks trustworthy comparison; **P1** blocks method interpretation; **P2** is valuable after the foundation is stable.

## P0 — Reproduction blockers

- [ ] Obtain a complete archived Colab run for `3dd_original` at the user's approximately 63% result.
- [ ] Confirm exact ModelNet40-C files, severity, sample counts/order, and hashes.
- [ ] Confirm Point-MAE and LION checkpoint paths and SHA-256 hashes.
- [ ] Record the resolved environment and compare `requirements.txt`, `env.yaml`, and notebook installations.
- [ ] Establish fixed seeds and measure baseline variance over at least three runs.
- [ ] Recompute both macro per-corruption mean and micro total accuracy from raw counts.
- [ ] Explain whether the intended reference is the published 65.7 or README 66.1 protocol and why the tables differ.

## P1 — Baseline implementation questions

- [x] Compare original LION inference modes: trainer evaluation disables dropout; the inherited demo wrapper and current TTA setup do not. Accuracy causality is still open; see `code_audit_20260912.md`.
- [ ] Measure isolated dropout A/B after source-only identity checks: fixed configuration/common draws, Gaussian/background and seeds 0,1,2; log module modes and guidance input gradients.
- [x] Create clean main-based branch reference: baseline-repro-clean at 107305fd7baf40b359f31c07d235599198be7324; legacy checkout/files unchanged.
- [x] Finish same-folder clean-branch checkout and preserve curated skill/knowledge/protocol plus source PDFs; user selected in-place development. Tracked code matches main; existing stash untouched.
- [ ] Implement Batch 1 artifact runner after branch/documentation handoff review; do not import old variants. See `clean_restart_batches.md`.
- [ ] Serialize and compare the complete DDIM scheduler config between `tta.py` and `tta_gsd.py`, including `set_alpha_to_one` and installed `diffusers` behavior.
- [ ] Determine the correct final decode input: raw global `shape_latent`, processed/updated `style_cond`, or another LION representation.
- [ ] Verify gamma/eta global/local semantics with unequal values and gradient norms.
- [ ] Confirm that checkpoint loading reports no missing/unexpected keys in every entry point.
- [ ] Quantify batch-size effects under fixed seeds/common random numbers.
- [ ] Require spectral-off tensor/prediction parity with the selected baseline before enabling new guidance.
- [ ] Test complete/default/partial loss-weight dictionaries; short-circuit direct-key access can hide failures.
- [ ] Validate count-based aggregation and immutable fresh-run schemas; prevent duplicate/config-mixed resume (new runner starts without resume).

## P1 — Spectral correctness

- [ ] Visually verify the GSDTTA outlier-threshold equation and map every symbol to `graph_spectral.py`.
- [ ] Validate the recorded gamma/(N*k) versus gamma/N threshold discrepancy with the same adjacency before correction; see the audit.
- [ ] Log spectral/SCD gradient and update scales with actual batch/band/channel counts; raw mean/sum loss numbers are not comparable fidelity measures.
- [ ] Test whether `knn_cuda` returns Euclidean or squared distances in this installed build.
- [ ] Test and, if necessary, exclude self-neighbors.
- [ ] Validate graph symmetrization, degree thresholding, isolated-node treatment, and numerical jitter on small known graphs.
- [ ] Measure spectral energy concentration of LION local latents by corruption/class.
- [ ] Replace raw eigenvector tracking diagnostics with band-projector/subspace measures near degenerate eigenvalues.
- [ ] Validate or repair `spectral_analyzer.py` before trusting it.

## P1 — Gradient projection mechanism

- [ ] Log per-sample conflict rates; compare with current batch-flattened conflict decisions.
- [ ] Log raw/weighted/projected gradient norms and cosine distributions.
- [ ] Correct and validate symmetric PxP CSV schema, including delta fields and reduction.
- [ ] Compare no projection, spectral-priority, Chamfer-priority control, and symmetric projection with common random numbers.
- [ ] Test whether conflict predicts harmful next-step loss/accuracy behavior rather than assuming it.

## P2 — Research extensions

- [ ] Deferred by user: sum spectral loss plus smaller eta/gamma after baseline/spectral-off parity, including reduction-equivalence controls and explicit SCD effects. Earlier mean advantage is [User report], not [Run].
- [ ] Compare static versus dynamic eigenbasis at matched compute and several update intervals.
- [ ] Validate physical-to-latent point correspondence before interpreting physical-basis results.
- [ ] Evaluate global/local sequential and synchronized diffusion after resolving style decoding.
- [ ] Explore sparse/approximate eigensolvers if the dynamic graph is beneficial but too costly.
- [ ] Extend beyond ModelNet40-C only after the core protocol is locked.

## P2 — Thesis reporting

- [ ] Predefine final primary/secondary metrics and statistical tests.
- [ ] Separate hyperparameter-development corruptions/data from final reporting.
- [ ] Record compute budget and sustainability/feasibility constraints.
- [ ] Prepare an ablation table that distinguishes algorithmic effects from scheduler/reduction changes.
