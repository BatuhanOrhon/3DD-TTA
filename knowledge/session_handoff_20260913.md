# Session handoff — 2026-09-13

Read this note first after `knowledge/README.md`. It is the current state snapshot for continuing the ModelNet40-C reproduction audit before any GSD work.

## Repository state

- Working repository: `D:/Akademik/Okul/Thesis/code/3DD-TTA`
- Active branch: `baseline-repro-clean`
- HEAD: `9ce5553de9277f9b9d7e26fc729e70b912a7c2d7`
- The working tree is intentionally dirty. Do **not** reset, checkout, clean, stash, or delete existing artifacts/PDFs without user direction.
- EMA implementation is local and **not committed/pushed** yet: `models/lion.py`, `main_3dd_tta.py`, `run_baseline.py`.
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

**[Code]** `python -m py_compile models/lion.py main_3dd_tta.py run_baseline.py`, `git diff --check`, and `python run_baseline.py --help` passed locally. No Colab EMA inference has run.

The mapping mirrors original LION training construction (`dae.parameters()`); count and shape checks are safeguards, but this is still an experimental inference-path change that needs output validation.

## Immediate next action

1. Review the current diff. Commit locally; request user direction before pushing to remote, because Colab cannot fetch unpushed code.
2. In Colab, on the exact committed revision, run two full severity-5, seed-0, batch-32 ModelNet40-C pilots with `--corruptions gaussian impulse`, gamma=.01, eta=.01, lambda=.95:
   - `--lion-eval-mode` only: `eval+raw`.
   - `--lion-eval-mode --lion-ema-mode`: `eval+EMA`.
3. Confirm EMA stdout contains `INFO loaded prior EMA parameters: 462` and archive both complete runner ZIPs under `result/modelnet40_c/3dd_original/`. Copy the two folders to `/content/drive/MyDrive/thesis/result/modelnet40_c/3dd_original/` after completion.
4. Ingest and validate the ZIPs: seven expected files, complete Gaussian and Impulse rows (2,468 examples each), same asset hashes/revision/args except `lion_ema_mode`, no traceback. Report per-corruption and macro deltas. Do not interpret one seed as a final gain.
5. If the EMA pilot is promising, repeat both conditions seeds 1/2. If null/negative, record it and proceed to source-only severity-4 control. Do not mix EMA with legacy/train LION mode.

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
