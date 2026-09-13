# Session handoff — 2026-09-13

Read this note first after `knowledge/README.md`. It is the current state snapshot for continuing the ModelNet40-C reproduction audit before any GSD work.

## Repository state

- Working repository: `D:/Akademik/Okul/Thesis/code/3DD-TTA`
- Active branch: `baseline-repro-clean`
- HEAD: `9ce5553de9277f9b9d7e26fc729e70b912a7c2d7`
- The working tree is intentionally dirty. Do **not** reset, checkout, clean, stash, or delete existing artifacts/PDFs without user direction.
- EMA implementation was committed and pushed as `c91c1c3` on `baseline-repro-clean`: `models/lion.py`, `main_3dd_tta.py`, `run_baseline.py`.
- Existing dirty knowledge files and untracked PDFs/results are part of the thesis record. Preserve them.

## What is established

### Evidence-backed ModelNet40-C baseline state

- **[Paper]** WACV Table 2 reports 3DD-TTA mean 65.7%. Its 15 displayed cells average 65.6933%. README claims 66.1%, but its displayed cells average 65.44%; do not use 66.1 as an internally validated target.
- **[Run]** Source-only severity-5 all-15 mean is 53.6899%, versus paper source 57.6%. The mismatch exists before LION/TTA. Clean `data_original.npy` source-only is 90.64% (2237/2468).
- **[Run]** Full severity-5 seed-0 macro: legacy 63.0578%, LION eval 63.8817%, +0.8239 pp. Gaussian eval improves over legacy across seeds 0/1/2 by mean +1.2966 pp; Background does not generalize that effect (mean -0.1080 pp). Do not select an inference mode globally yet.
- **[Code/Paper]** Background uses 35 reverse steps in original upstream code and is discussed in paper Section 4.4. Do not infer five Table-2 steps merely from the paper’s lower Background accuracy.
- **[Code/Paper]** Released SCD is unnormalized sum while Eq.11 divides directed sums by point count. This is an untested isolated hypothesis; it is distinct from deferred spectral-loss mean/sum work.
- **[Code/Paper]** `style_cond` is updated in `tta.py`, but final decoding receives initial `shape_latent`. Historical fork tried the alternative; no current controlled result exists.

See `code_audit_20260913.md` for full evidence, comparison table and tests not justified by the audit.

## EMA finding and code change

**[Run]** Diagnostic files:

- `result/modelnet40_c/diagnostics/checkpoint_ema_inventory.json`
- `result/modelnet40_c/diagnostics/checkpoint_sha256.txt`

Checkpoint SHA256: `807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`.

It is a full training checkpoint. `dae_optimizer` has one 462-parameter group and 462/462 `ema` tensors. `vae_optimizer` has zero EMA entries. Config reports EMA enabled, decay .9999.

Local implementation adds an opt-in `--lion-ema-mode`:

- `models/lion.py`: loads raw prior/VAE as before; with `use_ema=True`, enumerates prior parameters in the checkpoint optimizer group order, checks count and every EMA tensor shape, then copies EMA into **prior only**.
- `main_3dd_tta.py`: forwards `args.lion_ema_mode`.
- `run_baseline.py`: parses `--lion-ema-mode`, so the flag is saved in `config.json` CLI args.
- Default remains raw prior. VAE EMA is never attempted.

**[Code]** `python -m py_compile models/lion.py main_3dd_tta.py run_baseline.py`, `git diff --check`, and `python run_baseline.py --help` passed locally. **[Run]** Three-seed all-15 raw/EMA screen is complete: +.1035 ± .1639 pp macro, with seed 2 negative and corruption-dependent signs. EMA remains an ablation, not selected baseline; exact evidence is in `ema_inventory_20260913.md`.

The mapping mirrors original LION training construction (`dae.parameters()`); count and shape checks are safeguards, but this is still an experimental inference-path change that needs output validation.

## Immediate next action

1. On commit `b999a1e`, run all 15 ModelNet40-C corruptions in legacy+raw mode at seeds 1 and 2, severity 5/batch 32.
2. Compare these with current matching eval+raw seeds 1/2 to isolate dropout/eval behavior. Do not include EMA in this comparison.
3. Retain the earlier all-15 legacy/eval seed-0 result as historical/cross-commit support, but report seed 1/2 as current matched-commit evidence.
4. GSD remains parked until the dropout baseline gate is resolved.

## Research constraints

- User runs all GPU experiments in Colab with `!conda run --no-capture-output -n 3dd_tta_env python ...`.
- Ask user to supply full run ZIPs, not screenshots; raw artifacts stay in `result/` and are not overwritten.
- No unit-test files: this is experimental research. Use proportionate structural checks and Colab outputs.
- GSD/PxP, ScanObjectNN, and ShapeNet are parked until the baseline gate is resolved.
- Preserve evidence labels: [Paper], [Code], [Run], [User report], [Inference], [Open]. Do not promote a hypothesis to a result.
- Do not claim that local results exceed paper values unless dataset, protocol, source model, severity, and repeated-seed evaluation are demonstrably matched.

## Required next-session reading

1. `knowledge/README.md`
2. this file
3. `knowledge/code_audit_20260913.md`
4. `knowledge/ema_inventory_20260913.md`
5. `knowledge/reproduction_gap.md`, `knowledge/findings_log.md`, `knowledge/open_questions.md`, and `knowledge/experiment_protocol.md`
6. `git status --short`, `git diff --check`, and `git diff -- models/lion.py main_3dd_tta.py run_baseline.py`
