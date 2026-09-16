# Session handoff - 2026-09-15

Read this note immediately after `knowledge/README.md`. It supersedes the
2026-09-13 handoff as the operational starting point.

## Repository and preservation rules

- Repository: `D:/Akademik/Okul/Thesis/code/3DD-TTA`
- Branch: `baseline-repro-clean`
- Recorded handoff commit: `43bce179683dc6049a133265e3f353ee85d4c279`
- The workspace intentionally contains untracked PDFs, result bundles, caches,
  and historical materials. Never reset, clean, checkout, stash, delete, or
  overwrite them without explicit user direction.
- User runs GPU work in Colab, always through `!conda run --no-capture-output
  -n 3dd_tta_env python ...`. Do not run GPU benchmarks locally.
- No unit-test files by default. Use focused code/protocol checks and archived
  Colab experiments. Do not resume GSD/PxP, ScanObjectNN, or ShapeNet work yet.

## Locked operational baseline

Use `3dd_original` with raw LION weights and `--lion-eval-mode`; do **not** pass
`--lion-ema-mode`.

**[Run]** Same-commit ModelNet40-C severity-5 all-15 screens at seeds 1/2 show
eval/raw 63.8547% versus legacy/raw 63.0970%, a +.7577 +/- .1203 pp difference
and higher mean accuracy for 13/15 corruptions. The four validated ZIPs and
full table are in `dropout_eval_mode_20260913.md`. Legacy/raw remains the
explicit comparator.

This is a seed-controlled rather than common-random-number paired comparison.
It supports an operational default, not a claim that one dropout layer alone
causes the improvement.

For the complete terminology and all seed-0/1/2 macro results, read
`inference_mode_matrix_20260915.md`.

## EMA decision

**[Run]** The full 15-corruption, seed-0/1/2 `eval+raw` versus `eval+EMA`
screen has an EMA-minus-raw macro change of +.1035 +/- .1639 pp, with a negative
seed and corruption-dependent signs. Keep the opt-in EMA implementation as an
ablation only; do not use it in the baseline. See `ema_inventory_20260913.md`.

## Reproduction state

- **[Paper]** WACV Table 2 reports 65.7% ModelNet40-C Point-MAE mean; the
  displayed paper cells average 65.6933%.
- **[Code]** README claims 66.1%, but its displayed cells average 65.44%; keep
  the two references separate.
- **[Run]** Local severity-5 source-only all-15 macro is 53.6899%, below the
  paper source value 57.6%. The clean `data_original.npy` control is 90.64%.
  Thus the unresolved gap predates LION/TTA.
- **[Run]** The selected eval/raw baseline is still below the paper value and
  does not establish paper-level parity.

## Next approved research action

Perform a labelled **source-only ModelNet40-C severity 1--5 probe** before any
new GSD method work. The aim is descriptive: determine whether the released
corruption severity changes explain part of the source-only/paper gap. Preserve
the frozen Point-MAE, direct data loading, FPS(1024), batch 32, all 15
corruptions and separate immutable run directories. Do not select a lower
severity post hoc as the benchmark protocol; record all five levels and compare
only after inspecting the full artifacts.

Before issuing the Colab command, inspect `run_baseline.py` and
`knowledge/experiment_protocol.md` so the runner actually accepts/separates the
desired severity runs. Request the user to provide the complete ZIP directories
under `result/modelnet40_c/source_only/`, each containing:

`command.txt`, `config.json`, `environment.txt`, `stdout.log`, `summary.csv`,
`per_corruption.csv`, and `notes.md`.

## Required reading and recording

1. `knowledge/README.md`
2. this handoff
3. `knowledge/dropout_eval_mode_20260913.md`
4. `knowledge/ema_inventory_20260913.md`
5. `knowledge/reproduction_gap.md`, `knowledge/experiment_protocol.md`,
   `knowledge/findings_log.md`, and `knowledge/open_questions.md`
6. `git status --short`, current branch/HEAD, and relevant source before edits.

Use evidence labels `[Paper]`, `[Code]`, `[Run]`, `[User report]`,
`[Inference]`, and `[Open]`. After a material result, append it to
`findings_log.md`, update `open_questions.md` and this handoff as needed, then
commit/push documentation without adding raw artifacts.

## Subsequent main/LION audit - 2026-09-15

Read `code_audit_20260915.md` while waiting for the five source-only severity
ZIPs. Audit base is `262f3a668b3f5a7bc44c6282c4a8a2723ac6f00a`; no new GPU
experiment or model change. Remote main refs match the local audited references.

New source-path priority: installed FPS/index diagnostics. Historical severity-5
Density/Cutout/LiDAR files contain 649/724/768 points, while source-only requests
FPS(1024); the bundled kernel also filters near-origin candidates. Inspect its
actual runtime behavior before changing sampling. The paper's 57.6 source mean
and Table-2 column order were visually verified; stale paper-note headings fixed.

After source-data gate review, the first small TTA candidate is a shared-trajectory
original/updated final-style decode comparison; SCD scale and lambda are separate
later factors. Extra classifier calls must preserve NumPy RNG because unused
Point-MAE masks still consume it. Details, falsifiers and all source references
are in the audit. GPU work remains Colab-only; no new accuracy benefit is claimed.

The earlier authorized source-only `notes.md` fix is still a working change in
`run_baseline.py`, excluded from the knowledge-only commit. Pulling documentation
alone does not transfer that code edit to Colab. Source-only severity selection
and immutable ZIP generation already exist at the audit base commit.

## Source-only severity probe result - 2026-09-16

Five immutable all-15 ModelNet40-C source-only runs were received and validated:

| severity | mean / micro accuracy | correct / total | archive SHA256 (prefix) |
|---:|---:|---:|---|
| 1 | 75.8806% | 28091 / 37020 | `a99c8998` |
| 2 | 73.2739% | 27126 / 37020 | `a5272e15` |
| 3 | 68.5062% | 25361 / 37020 | `a716844b` |
| 4 | 62.0205% | 22960 / 37020 | `49cef14e` |
| 5 | 53.6899% | 19876 / 37020 | `01fdcfda` |

Evidence: `[Run]` ZIPs under `result/modelnet40_c/source_only/`; each archive
contains exactly the seven required files, 15 complete corruption rows, the
same classifier and label hashes, and direct `data_<corruption>_<severity>.npy`
paths. Macro and micro means coincide because every corruption has 2468 examples.

Interpretation: severity is a strong descriptive factor (`-22.1907 pp` from
severity 1 to 5), but it does not explain the paper-vs-reproduction gap by
itself: severity 5 is `53.6899%` (`-3.9101 pp` vs the paper's `57.6%`), while
severities 1--4 are above that paper source value. Thirteen of fifteen corruption
curves are monotone; Occlusion and LiDAR are non-monotone at low severity, so
the probe is descriptive rather than a claim of a universal severity law.

Metadata caveat: these runs used commit `262f3a6` and the pre-fix runner SHA;
their generic `notes.md` text is stale, but the configs, logs, summaries and
per-corruption rows validate the intended source-only protocol. Do not replace
the raw archives. Keep severity 5 as the benchmark condition and do not present
a lower-severity result as the paper benchmark.

Next P0 checks before new TTA methods: (1) capture actual Colab FPS indices and
duplicate counts, (2) reconcile checkpoint/archive and corruption-file provenance,
and (3) run a preprocessing-identity control. Only after this source-data gate
should the shared-trajectory decode and separate SCD-scale/lambda hypotheses be
tested.
