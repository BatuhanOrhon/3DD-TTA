# Experiment and Reproducibility Protocol

## GSD development calibration extension - 2026-09-27

[User report/Code] The approved [calibration plan](gsd_calibration_execution_20260927.md)
adds explicit `calibrate` and `development` stages for smooth-v2 only.
Use the [Colab handoff](colab_gsd_calibration.md). Complete execution on these
shuffled subsets retains CSV coverage `partial`; it is not a full-file pilot
or all-15 benchmark. Preserve exact selected original indices, raw per-sample
metrics and calibration provenance in config.json. Calibration candidate
probes share states; separate screen runs remain seed-controlled, not proven
common-draw paired. Numerical model experiments remain Colab-only.

## GSD v1 extension - 2026-09-23

[User report] The current GSD task explicitly authorizes CPU unit/protocol
tests; this overrides the historical default no-new-tests preference below.
[Code] `gsd_latent_spectral_v1` is opt-in and restricted to the operational
ModelNet40-C severity-5 batch-32 raw/eval original-style protocol. Compare
original, spectral-off and spectral-on arms at seeds 0/1/2. The exact
smoke/pilot/all-15 scopes, locked settings, promotion rule and artifact fields
are in [colab_gsd.md](colab_gsd.md). Graph diagnostic fields live in config;
standard CSV schemas retain counts, accuracy, runtime, memory and status.
[Open] No GSD GPU evaluation has been performed. Full CUDA integration,
spectral-off prediction parity and pilot review precede benchmark claims.

## Objective

Produce Colab results that are comparable across commits and methods, diagnostically useful, and sufficient for a thesis claim. The raw run directory is the unit of evidence.

## Batch 1 implementation - 2026-09-12

**[Code]** run_baseline.py wraps the original configure/process path with optional read-only observers and research_artifacts.py count records. It is smoke-only (default Gaussian, two batches); no resume, LION mode change or guidance equation change. See [the runnable Colab handoff](colab_baseline_smoke.md). Base revision: b31fd23193bbcb9a5c189cfb4118be41506f9333; implementation commit reported in the handoff.

Record actual parsed CLI, runtime source/asset hashes, resolved configs/scheduler, load incompatibilities, module/dropout modes, seeds/cuDNN flags, extension identities, per-corruption counts/time/peak allocated memory, raw subprocess output and ZIP. Seed-controlled is not common-draw-paired. No CUDA/model evaluation occurred locally.

The runner preserves repository defaults (batch=40, lambda=0.95); the proposed smoke explicitly selects batch=32. Example configuration values elsewhere in this document are illustrative, not overrides. A completed prefix has execution_status=complete but coverage status=partial; a failed/empty run has no fabricated accuracy. Running/failed summaries and logs must never be promoted to benchmark results.

Timestamps are UTC. Each invocation owns a new directory and seals it on exit; existing directories/ZIPs are not resumed or overwritten. CSV macro mean covers recorded nonempty corruption rows; only all-15 full coverage can be the benchmark macro mean. Timings exclude checkpoint hashing/model loading.

User preference: do not add unit-test files by default. Use proportionate syntax/structural/protocol checks and controlled, archived Colab experiments. The next acceptance gate is the seven-file smoke ZIP, not full accuracy.

## Evaluation levels

### Level 0 — smoke test

- 1 corruption, a few batches.
- Purpose: syntax, shape, memory, and output-schema validation.
- Never report as accuracy evidence.

### Level 1 — pilot

- Small, explicitly selected subset; current grid scripts often use the first 25 samples.
- Purpose: reject broken/clearly poor configurations and estimate resource use.
- Must be labelled `pilot`; selection bias is expected.

### Level 2 — corruption validation

- Complete examples for selected corruption(s), at least three fixed seeds.
- Purpose: mechanism and hyperparameter validation.

### Level 3 — benchmark evaluation

- All 15 ModelNet40-C corruptions at severity 5, all examples, locked configuration, at least three seeds when computationally feasible.
- Report per-corruption accuracy, macro mean, mean ± standard deviation across seeds, runtime, and peak memory.
- No hyperparameter selection on these final outcomes.

## Baseline ladder

Run in this order with identical classifier/data/preprocessing:

1. `source_only`: corrupted input to Point-MAE without LION.
2. `lion_recon`: LION encode/reconstruct without guidance.
3. `3dd_original`: original SCD-guided path.
4. `gsd_static`: static spectral variant with matched scheduler/steps and SCD configuration.
5. Later variants only after steps 1–4 are archived and internally consistent.

This ladder localizes whether divergence begins in data/classifier, LION, guidance, or the added method.

## Randomness

- Set and record Python, NumPy, PyTorch CPU, and all CUDA seeds.
- Record deterministic/cuDNN flags.
- Use fixed seed lists, initially `0, 1, 2`.
- For paired method comparisons, use common random numbers: same data order, interpolation choices, and diffusion-noise seeds.
- Keep batch size fixed. If batch size changes, treat it as a new experimental factor.

## Required configuration

`config.json` must include at least:

```json
{
  "run_id": "20260912-153000_3dd-original_seed0",
  "stage": "benchmark",
  "git_commit": "full commit hash",
  "git_dirty": false,
  "dataset": "modelnet40_c",
  "severity": 5,
  "corruptions": ["all 15 names in evaluated order"],
  "method": "3dd_original",
  "seed": 0,
  "batch_size": 1,
  "classifier": "pointmae",
  "classifier_checkpoint_sha256": "...",
  "lion_checkpoint_sha256": "...",
  "dataset_hash_manifest": "...",
  "num_input_points": 2048,
  "num_classifier_points": 1024,
  "scale_factor": 3.3885,
  "ddim_total_steps": 100,
  "normal_reverse_steps": 5,
  "background_reverse_steps": 35,
  "gamma": 0.01,
  "eta": 0.01,
  "lambda_cd": 0.96,
  "scheduler_config": {},
  "spectral": {},
  "projection": {},
  "cli_args": {}
}
```

Empty method-specific dictionaries are preferable to silently omitted fields. Record actual runtime values after argument parsing.

## Required run directory

```text
result/
  modelnet40_c/
    <method>/
      <YYYYMMDD-HHMMSS>_<short-run-name>/
        command.txt
        config.json
        environment.txt
        stdout.log
        summary.csv
        per_corruption.csv
        notes.md
```

Use lowercase ASCII method/run names with hyphens or underscores. Never overwrite a run directory. A rerun receives a new timestamp/run ID.

### `command.txt`

Exact Colab shell/Python command(s), including working directory and relevant environment variables. Remove secrets/tokens.

### `environment.txt`

Include:

- Python version;
- `pip freeze` or `conda env export`;
- `torch`, CUDA, cuDNN, `diffusers`, NumPy, `knn_cuda`, FPS/Chamfer extension versions;
- GPU name and `nvidia-smi` output;
- Git branch, full commit, status, and diff/stat if dirty;
- checkpoint/data paths and SHA-256 hashes;
- Colab runtime type and date.

### `stdout.log`

Unedited standard output/error from start to completion, including warnings and tracebacks. Do not paste only the final accuracy.

### `per_corruption.csv`

Required columns:

```text
run_id,dataset,severity,method,seed,corruption,n_examples,n_correct,accuracy,runtime_seconds,peak_gpu_memory_mb,status
```

One row per corruption per seed. Accuracy must use a declared unit consistently; prefer a fraction in `[0,1]` and convert to percent only in presentation tables.

### `summary.csv`

Required columns:

```text
run_id,dataset,severity,method,seed,n_corruptions,macro_accuracy,total_examples,total_correct,micro_accuracy,total_runtime_seconds,status
```

If multiple seeds are aggregated, add a distinct aggregate row or separate file with `accuracy_mean`, `accuracy_std`, and seed list. Do not mix a macro corruption mean with micro example accuracy without labels.

### `notes.md`

Record purpose, hypothesis, deviations, interruptions/resumes, known anomalies, and whether configuration was chosen using test data.

## Naming examples

- `result/modelnet40_c/3dd_original/20260912-153000_paper-config_seed0/`
- `result/modelnet40_c/gsd_static/20260913-091500_low16-mid2-seed0/`
- `result/modelnet40_c/pxp_symmetric/20260914-180000_delta05-pilot-seed1/`

## Result ingestion workflow

After a Colab run, the user should provide the complete run directory or a ZIP whose root is that directory. The agent will:

1. inspect archive paths before extraction;
2. place the artifact under the matching `result/` hierarchy without altering raw files;
3. validate required files, CSV headers/counts, config/log consistency, and completion status;
4. compute derived summaries in separate files if needed;
5. append a dated entry to `knowledge/findings_log.md`;
6. update open questions and method notes only to the strength supported by the run.

If fields are missing, archive the run but mark it **incomplete** and request the missing material. Never invent metadata.

## Statistical and methodological rules

- Prefer paired comparisons per corruption and seed.
- Report absolute percentage-point change, not only relative percentage.
- Inspect whether mean gains are concentrated in one corruption.
- For multiple hyperparameter comparisons, disclose search scope and guard against winner's curse.
- Reserve a validation protocol or nested selection strategy; do not tune repeatedly on the final benchmark and call it unbiased.
- Record failures, OOMs, and interrupted runs because they inform feasibility.
- Measure conflict diagnostics for PxP runs and eigensolver cost/stability for dynamic spectral runs.

## Minimum PxP diagnostics

Per step or aggregated by corruption:

- fraction of negative-cosine conflicts;
- per-sample and batch cosine mean/std/quantiles;
- spectral and SCD gradient norms before weighting;
- weighted norms;
- projected norms and post-projection cosine;
- loss changes after the proposed update;
- NaN/zero-norm counts.

## Minimum spectral diagnostics

- eigenvalue ranges and eigengaps around band boundaries;
- energy fraction by low/mid/high band;
- graph degree/outlier/isolation statistics;
- eigendecomposition runtime and peak memory;
- for dynamic mode, projector/subspace drift between updates;
- checks for self-neighbors and distance-unit assumptions.

## Promotion rule

A configuration moves from pilot to full benchmark only when it:

- completes without numerical/schema errors;
- has a predeclared hypothesis;
- beats or meaningfully diagnoses a matched baseline on validation corruptions across fixed seeds;
- has acceptable runtime/memory;
- does not rely on an untracked protocol change.


## 2026-10-11 - Proposed trajectory observation and stopping protocol

[User report/Code/Inference/Open] See
[trajectory selection design](gsd_trajectory_selection_20261011.md).
For this proposed experiment, preserve original terminal inference and archive
per-original-index/per-seed/per-checkpoint logits, timestep and loop index,
clean-latent versus terminal designation, style/update timing, shared-input/RNG
hashes, and component/end-to-end times. Fix checkpoint schedule before running.
Do not mix pre-update clean latent with post-update style. Extra probes must
not mutate classifier BN, main RNG or the guidance trajectory.

Selection/averaging can be computed from the same full trace. Early stopping
must use observed prefixes only; offline saved-step simulations are not measured
speedups. Retain the terminal fallback and evaluate live batch compaction/cost.
Thresholds and candidate selection use explicitly labelled development data;
target labels never enter inference. Treat timesteps as repeated measurements.
Report seed SD, index-cluster uncertainty, corrected/broken counts and all15
heterogeneity. A guidance-zero accuracy arm is not a speed ablation if graph or
loss computation still executes. No new numerical promotion threshold is set.


## Later 2026-10-11 - Separate benchmark and online runtime protocols

[User report/Inference/Open] OnlineB1 early exit directly skips subsequent
sample computation; B70 active-batch limitations do not apply to a single
sample. Report B70 all15 accuracy/throughput and matched onlineB1 accuracy plus
mean/p50/p95 latency separately. Include additional probe and scheduling cost;
B70 baseline accuracy is not a measured B1 result. A smaller number of actual
B1 prior calls is structural evidence of saved work; wall-time speedup needs
measurement. Keep stopping causal, lock thresholds before evaluation, and
explicitly disclose the existing corruption-name-dependent10/30 horizon.
See [session handoff](gsd_trajectory_session_handoff_20261011.md).


## Later 2026-10-11 — Implemented Legacy observation protocol

[Code/CPU/Open] [Implementation and runnable cells](gsd_trajectory_implementation_20261011.md).
Predeclare checkpoint counts3/5/8 for10 and8/15/23 for30; preserve the Legacy
corruption-name horizon. Label-free seed20261011 selects280 original IDs; sort
and reuse them across all15 and seeds0/1/2. Smoke uses140 IDs on Gaussian and
Background in off/audit/observe modes and requires exact output/state/RNG parity.
Stop for smoke review before the separate12,600-example development pilot.

Artifact completion and subset coverage remain separate. Original terminal is
stored once, probes add no entropy backward, model state is hashed before/after,
and all candidate files are hashed. Pool only matched protocol/source/assets/
native/external/environment identities and original IDs. Use three simple
label-free policies plus separate fixed-probe and label-using oracle diagnostics.
One thousand fixed-seed original-index bootstrap replicates retain seed pairing;
timesteps are not independent samples. Instrumented time includes observer cost,
not evidence for early-stop speedup. No numerical promotion/entropy threshold.


## Later 2026-10-11 — Seed0 first, replication conditional

[User decision/Code/Open] For the current Legacy trajectory study only, replace
the initial seeds0/1/2 pilot with seed0 all15/280 (4,200 examples). Smoke already
uses seed0. Preserve IDs, checkpoints and other settings. Review exploratory
seed0 policy benefit, corrected/broken balance and cost before locking a policy
for separate seeds1/2 replication. Single-seed intervals do not establish seed
stability; no numerical significance/promotion threshold is prescribed.
[Updated delivery](gsd_trajectory_implementation_20261011.md).


## 2026-10-11 - Approved seed0 all15 trajectory screen

The next user-operated Colab run is a full-data development screen: all15 ModelNet40-C severity5 corruptions, original IDs0..2467, seed0 (37,020 inputs). Keep accepted Legacy B70/M800, beta2, spectral1.17, chamfer1, gamma/eta.01, lambda.95, legacy isolate policy, normal/background horizons10/30, and observe checkpoints3/5/8 and8/15/23. No index manifest, batch limit, early stopping, or horizon change. Analyze terminal, minimum entropy and equal probability mean only after full coverage/model-state/provenance validation; report paired and per-corruption results. A one-seed result is exploratory; review before deciding on seeds1/2. Entropy is not a correctness or temporal-stability guarantee; no accuracy threshold or speedup is presumed. Use the appended notebook preflight, run, package/Drive-backup, and download code cells in order. Reserve2h; runtime reset can lose non-checkpointed partial artifacts.

## 2026-10-11 - Opt-in sampling-consistency experiment

[User decision/Code/CPU/Open] The approved two-view extension uses a common10-step horizon and macro14 only, Legacy B70/M800 seed0, checkpoints3/5/8 plus terminal. Per candidate: original A FPS and deterministic alternate-start B, same whole-cloud normalization and1024-point input. Terminal A+B plus six more votes provide the eight-forward terminal control. Acquisition14 forwards must not be reported as method-eight latency. Freeze classifier weights/BN/token algorithm; no entropy gradient, fitted threshold, corruption routing or early stop. Compare four-A mean, eight-view mean, two-view H and H+JS selectors, terminalA and terminal8. Source snapshot and ID manifests are hashed; full source/coverage/state/view checks precede CPU scoring. Gaussian140 smoke is followed by a separately launched macro14x280 pilot after ZIP review; source and user notebooks are preserved. Exact references, test requirements and falsifiers: [sampling consistency](gsd_sampling_consistency_20261011.md). No GPU result yet.
