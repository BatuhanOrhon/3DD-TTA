# GSD v1 Colab handoff

[Code] Branch: `gsd-development`, based on `baseline-repro-clean` at `79cc027`.
Method ID: `gsd_latent_spectral_v1`; name: **GSD-inspired latent spectral guidance**.
Design and audit: [integration record](gsd_integration_20260922.md).

## Transfer and preflight

[Code] Use a checkout containing the committed GSD files on `gsd-development`.
Publishing the local branch, if needed, is `git push -u origin gsd-development`.
In the existing Colab repository, fetch and switch to that branch without
resetting or cleaning local data/results. Confirm `git rev-parse HEAD` and
`git status --short`; retain the commit in the run artifacts. Do not change the
installed `3dd_tta_env`, checkpoints, dataset, CUDA extensions or dependencies.

```bash
git fetch origin gsd-development
git switch gsd-development
git pull --ff-only origin gsd-development
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage pilot
```

[Code] The last command only prints the nine planned commands. The locked
arms are original eval/raw baseline, GSD weight zero, and GSD weight one,
each at seeds 0/1/2. The original arm uses unchanged `3dd_original`; its
legacy generic notes/stage metadata is not rewritten. Its explicit command,
coverage, eval flag and runtime inventories define this matched control.

## Smoke and pilot

[Code] Run these in Colab from the repository root. Notebook shell cells
may prefix each command with `!`.

```bash
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage smoke --execute
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage smoke --smoke-corruption background --execute
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage pilot --execute
```

[Code] Equivalent combined script: `bash scripts/colab_gsd_pilot.sh`.
Each smoke uses seed 0 and two batches, including Background's 35 reverse
steps. Pilot evaluates complete Gaussian/Impulse files, batch 32, severity 5,
seeds 0/1/2. Existing raw LION eval, EMA off, frozen Point-MAE,
original-style decoder, gamma=eta=.01, lambda=.96, 100 DDIM schedule and
5/35 reverse steps are fixed. Graph settings are k=10, delta=.1,
graph gamma=.6, requested low modes=100, spectral weight=1.

[Inference] These are initial fixed research settings, not optimized latent
hyperparameters. Gaussian/Impulse is an exploratory test-set pilot; it is
not an independent held-out validation set. Record all null/negative results.

## Lambda reference update - 2026-09-23

[Run] The separate all-15 `scd_lambda96_control` confirmation is complete at
`result/modelnet40_c/scd_lambda96_control/`. Its original-style mean is
63.9339% versus 63.7484% for the matched lambda=.95 original-style control,
with positive paired deltas at seeds 0/1/2 and mean delta +0.1855 pp.

[Inference] The paper-conformant reference for a newly declared GSD pilot is
now `.96`, unless the experiment explicitly targets the historical `.95`
operational control. The existing `.95` GSD pilot remains valid evidence for
`.95`; it must not be relabeled. A `.96` GSD pilot requires a fresh matched
weight-zero and weight-one comparison, with all other settings unchanged.

## All-15 confirmation

[Open] First inspect complete pilot ZIPs: no failure/nonfinite errors,
matching checkpoint/data hashes and scheduler configs, raw/eval inventories,
and acceptable graph cost. Original versus weight-zero counts should agree;
investigate any difference before interpreting the active arm. Local tensor
parity alone does not establish installed CUDA parity.

[Inference] Promote weight one only if it improves the pilot macro at all
three seeds, mean delta is positive, and graph/gradient diagnostics show
nontrivial finite guidance at acceptable runtime/memory. This is a screening
rule, not a significance test. If not met, retain the negative result and
revise a separately declared pilot; do not retune the all-15 configuration.

```bash
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage all15 --execute
# Equivalent: bash scripts/colab_gsd_all15.sh
```

[Code] All-15 prepares the same three arms and seeds, complete canonical
15-corruption order. The GSD benchmark CLI rejects changes to its fixed
graph/weight settings. Launchers stop on a failed process and never resume or
overwrite raw run directories. Use `--seeds 1` and/or `--arms on` only for an
explicitly documented fresh rerun; the complete comparison still needs all
three arms/seeds.

## Artifacts and interpretation

[Code] Every invocation creates an immutable seven-file directory and sibling
ZIP under `result/modelnet40_c/<method>/<UTC-timestamp>_<name>/`:
`command.txt`, `config.json`, `environment.txt`, `stdout.log`, `summary.csv`,
`per_corruption.csv`, `notes.md`. GSD config records source/asset hashes,
resolved scheduler, mode inventories, graph contract and per-corruption
diagnostic aggregates (count/missing/min/max/sum/mean). Graph records are per
example; step records are per batch-step, so step means are not sample-weighted.
Actual batch size is also aggregated. CSV accuracies remain fractions.

[Code] Diagnostics include degree/outlier/isolate counts, selected ranks,
zero modes, boundary gap, selected/complement reference energy, graph and
eigendecomposition timing, separate SCD/spectral local/style gradient norms,
weighted spectral norms and final update norms. Runtime includes diagnostics
and the extra gradient pass; corruption-level peak GPU memory includes models.

[Run] Four smoke ZIPs and six complete pilot ZIPs are now present locally.
The pilot covers Gaussian/Impulse with GSD weight 0/1 at seeds 0/1/2. All
pilot archives pass the seven-file and CRC checks. The paired three-seed mean
GSD-on minus GSD-off delta is -0.0135 percentage points, so the current
promotion screen is not met. No all-15 GSD result exists yet.
Do not add raw result ZIPs to Git.

## Drive export after all-15

[Code] The runner writes each all-15 ZIP locally before any Drive copy:
`result/modelnet40_c/<method>/<run_id>.zip`. In a separate Colab cell, the
export helper creates the matching run-id directory under
`/content/drive/MyDrive/thesis/result/modelnet40_c/<method>/<run_id>/` and
copies only `<run_id>.zip` into it. Existing same-size files are skipped;
different-size files cause a hard failure, so an archive is never silently
overwritten.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
conda run --no-capture-output -n 3dd_tta_env python scripts/export_gsd_results.py \
  --source-root ./result \
  --drive-root /content/drive/MyDrive/thesis/result \
  --stage all15
```

The same cell accepts `--stage pilot` or `--stage smoke` for those artifact
sets. Raw ZIPs remain outside Git.

## Spectral-band pilot variants

[Code] `gsd_modes` is the requested number of lowest Laplacian eigenmodes; it
does not change the graph, scheduler, preprocessing, SCD, or update rates.
The pilot permits exploratory values such as 240 and 400. Run each value as a
separate artifact name and compare it with its own weight-zero control:

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py \
  --stage pilot --gsd-modes 240 --execute
```

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py \
  --stage pilot --gsd-modes 400 --execute
```

[Code] The provisional all-15 launcher currently invokes `--stage all14`,
which evaluates the canonical corruption list without Background. It records
`gsd_stage=benchmark_no_background`; the canonical 15-corruption command
remains available explicitly as `--stage all15` after reviewing runtime and
pilot evidence.
