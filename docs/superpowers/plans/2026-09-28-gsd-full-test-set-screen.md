# Full ModelNet40-C Test-Set GSD Candidate Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a resumable Colab workflow and strict CPU analyzer for comparing two locked smooth-GSD candidates with SCD-only on the same full ModelNet40-C test examples as the existing `all_corruptions` runs, across all 15 corruptions and seeds0/1/2.

**Architecture:** Add a distinct `full_dataset_development` protocol stage while preserving existing GSD stages. Keep launch orchestration, immutable-archive validation/analysis, and host execution in separate modules; pin all arms to calibration113047. The same corrupted inputs feed adaptation, labels are used only after prediction for metrics, and test-set-based candidate selection is descriptive rather than independent confirmation.

**Tech Stack:** Python3.8-compatible standard library, existing PyTorch/Colab runner, unittest suite, existing `RunBundle` seven-file archive contract.

**Spec:** [2026-09-28-gsd-full-test-set-screen-design.md](../specs/2026-09-28-gsd-full-test-set-screen-design.md)

## Global Constraints

- Run ModelNet40-C severity5; all 15 corruption files and each file's full recorded length.
- Keep batch32, seeds0/1/2, raw/eval LION, EMA off, lambda .95, gamma=eta=.01, and the current model/data/graph/scheduler/preprocessing/decoder unchanged.
- Pin calibration run ID `20260928-113047_gsd-cal-diagnose-reference-seed0-n64` and config SHA256 `550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6`.
- SCD-only uses weight0/no rho; beta .5/rho .001 alpha is `8.140161356429882`; beta2/rho .01 alpha is `113.10823980075075`.
- Keep the seven raw files per run immutable; put derived analysis outside run bundles.
- Do not install/reinstall packages, edit dependency pins, or run any GPU/model evaluation locally.
- Use the same all-corruptions test examples; adaptation never receives labels, and labels are used only for post-prediction metrics. Do not describe the selected candidate's score as independent confirmation.

## Review Focus

1. A noncanonical, partial, or reordered corruption list must fail before model work; test it in Task1.
2. A guided run with stale calibration ID/hash, alpha/rho, data, source, or checkpoint identity must fail preflight; test it in Tasks1 and3.
3. Incomplete/corrupt/duplicate archives must never be skipped during resume or included in analysis; test in Tasks2 and3.
4. Each corruption's recorded count must equal its loaded full-file count, including a smaller final batch; test in Tasks1 and2.
5. Macro accuracy must average corruption accuracies equally, and seed deltas must compare like arms without claiming common draws; test in Task2.

---

### Task 1: Add the full-dataset development protocol stage

**Files:**
- Modify: `gsd_protocol.py`
- Modify: `gsd_calibration.py`
- Modify: `run_baseline.py`
- Test: `tests/test_gsd_protocol.py`
- Test: `tests/test_gsd_calibration.py`
- Test: `tests/test_gsd_worker.py`

**Interfaces:**
- CLI stage: `--gsd-stage full_dataset_development`.
- Reuse `calibration_provenance(args)`; for this stage require source/settings/coefficient identity but no subset count or overlap guard.
- Reuse `process_batches(...)`; enable aggregate GSD diagnostics but no sample-by-step `gsd_sample_diagnostics` collection.

- [x] Write tests that the new stage accepts exactly the 15 canonical corruption names, full-file mode, severity5, batch32, raw/eval controls, seed0/1/2, and a valid pinned calibration for all three arms.
- [x] Write rejection tests for missing/extra/reordered corruption names, `max_batches != 0`, changed host settings, missing calibration, mismatched alpha/rho/config hash, and a requested development subset.
- [x] Write worker tests using small synthetic complete datasets to check every row count against loaded file size, full-suite status `complete`, failure on an empty/mismatched input, labels used only for metrics after prediction, and bounded aggregate diagnostics without sample-step row storage.
- [x] Run the focused tests and verify they fail on the unmodified code.
- [x] Add the stage to CLI validation without changing existing stage conditions. Preserve target rho/alpha checks; skip subset-size/prefix selection only for the new full-file stage.
- [x] Update `build_config` and the worker path to record `evaluation_scope`, calibration provenance, full per-corruption inventory, reuse of the canonical test inputs, and the accuracy-selection caveat. Do not force `status=partial` for the new stage.
- [x] Keep reverse-step behavior unchanged: background35; other corruptions5.
- [x] Run the focused tests and confirm they pass.

### Task 2: Implement the full-suite archive analyzer

**Files:**
- Create: `scripts/analyze_gsd_full_dataset_screen.py`
- Test: `tests/test_analyze_gsd_full_dataset_screen.py`

**Interfaces:**
- `read_full_run(path: Path) -> dict`: validate and return one complete raw seven-file bundle.
- `analyze_runs(paths: list[Path], calibration: dict) -> dict`: require exactly one SCD-only, beta .5/rho .001, and beta2/rho .01 run for each seed0/1/2.
- CLI: `--calibration PATH --screen-run ZIP [ZIP ...] --output PATH`; do not choose inputs by glob.

- [x] Create synthetic seven-file ZIP fixtures for all nine expected conditions, with 15 corruptions and small known full-file counts.
- [x] Assert correct equal-weight macro, per-corruption pp deltas, three-seed mean/sample SD, runtime/memory aggregation, and sign of each seed delta.
- [x] Assert rejection of unsafe members, CRC/schema errors, truncated counts, missing/duplicate arm-seed pairs, config/log/CSV mismatch, mixed environment/source/assets, and wrong calibration hash/coefficients.
- [x] Run the analyzer tests and verify the absent implementation makes them fail.
- [x] Implement standard-library ZIP/CSV/JSON checks plus calibration validation via `load_calibration`; require all 15 names, complete coverage, and `n_examples == config.dataset_inventory[name].total_examples`.
- [x] Report seed-wise per-corruption accuracy, equal-weight 15-corruption macro, candidate-minus-SCD-only pp deltas, their across-seed mean/sample SD, and runtime/peak memory. Mark outputs as full-test-set development evidence.
- [x] Write only the derived output path requested by the CLI; never alter raw ZIPs.
- [x] Run analyzer tests and confirm they pass.

### Task 3: Implement the sequential, resumable Colab launcher

**Files:**
- Create: `scripts/run_gsd_full_dataset_screen.py`
- Test: `tests/test_gsd_full_dataset_screen_launcher.py`

**Interfaces:**
- `build_screen_commands(calibration_path: Path, result_root: str) -> list[list[str]]`: return the fixed nine arm/seed commands.
- CLI: `--calibration PATH --result-root PATH [--resume]`.
- Reuse Task2 `read_full_run` for validating archives before resume skips.

- [x] Assert the generated matrix is exactly 3 arms x seeds0/1/2, all 15 corruptions, `max-batches=0`, the fixed reference, and the two exact calibrated alphas.
- [x] Assert malformed/missing calibration or changed alpha aborts before any child process starts.
- [x] Assert a complete valid matching archive is skipped only with `--resume`; invalid, failed, or mismatched bundles are preserved and never silently skipped/overwritten.
- [x] Assert command failure stops the sequence and reports completed archives and the failed command.
- [x] Run the launcher tests and verify they fail before implementation.
- [x] Implement ordered subprocess execution with live stdout, exact command/run-path reporting, immutable unique run names, and no package install/Git pull.
- [x] Implement resume by discovering and validating the unique archive for each logical arm/seed; allow a retry with a distinct attempt identity where no valid completion exists.
- [x] Run launcher tests and confirm they pass.

### Task 4: Document handoff, execute CPU verification, and review

**Files:**
- Modify: `knowledge/colab_gsd_calibration.md`
- Modify: `knowledge/open_questions.md`
- Modify: `result/README.md`
- Review: all files from Tasks1-3

- [x] Add the exact Colab command using the existing `3dd_tta_env` and the raw calibration config; document first-run and `--resume` behavior.
- [x] Document all-nine-ZIP ingestion, full-test development status, and required compact-summary fields.
- [x] Run focused tests for all new stage/protocol/worker/analyzer/launcher paths.
- [x] Run `python -m unittest discover -s tests -q`; record exact pass count.
- [x] Run `git diff --check` and inspect the final diff for accidental dependency, environment, raw ZIP, unrelated knowledge, or user-local file changes.
- [x] Do not invoke the Colab launcher locally. Present the exact command and expected new artifact naming scheme to the user for Colab execution.
