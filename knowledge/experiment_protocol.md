# Experiment and Reproducibility Protocol

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
