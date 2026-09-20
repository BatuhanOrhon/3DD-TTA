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

## FPS diagnostic v2 update - 2026-09-19

The actual Colab FPS diagnostic gate is now complete. The valid archive is
`result/modelnet40_c/source_only/20260919-133400_source-only-fpsdiag-v2-s5-seed0.zip`
and records commit `0003743`, schema
`legacy_fps_v2_finite_coordinate_unique`, and unchanged source-only accuracies.
LiDAR contains no NaN/Inf points but averages 396.227 finite-coordinate-unique
rows out of 768 input rows, with 371.773 exact coordinate duplicates per
example. The generator code uses `np.random.choice(..., 768)` without
`replace=False` at `datasets_mate/create_corrupted_dataset.py:655`, a strong
candidate explanation for the archived input structure.

This is source-data evidence, not an inference FPS bug or a resampling
contribution. Next P0 is provenance reconciliation for the historical
`data_lidar_5.npy` generation; do not change inference sampling until that is
resolved. Alternate policies and new TTA methods remain parked.

### Upstream generator cross-check - 2026-09-19

The provenance audit found that the canonical ModelNet40-C generator also uses
`np.random.choice(new_pc.shape[0], 768)` without `replace=False` in its LiDAR
path. The fork mirrors this line. Therefore the observed LiDAR duplicate
coordinates are upstream-consistent benchmark construction, not a justified
local FPS or preprocessing fix. Byte identity with the downloaded Zenodo
archive remains unproven, but no LiDAR regeneration or inference resampling
change is authorized. Resume the broader source-only asset/checkpoint gap
investigation.

## Source-only provenance manifest audit - 2026-09-20

The complete severity-5 source-only archives share identical 15-file data
manifests, Point-MAE checkpoint/config hashes, and label hash across their
different recorded run commits. This closes the internal cross-run consistency
check but not external byte identity: the workspace contains only
`data/readme.md` and `pointnet_ckpts/readme.md`, not the Colab `.npy` files or
checkpoint.

Next action remains provenance reconciliation: obtain or compare the canonical
ModelNet40-C archive and Point-MAE checkpoint hashes. Do not regenerate LiDAR,
change FPS, or start a new unmatched TTA method before that gate is addressed.

### Canonical archive metadata update - 2026-09-20

**[User report]** The Colab Zenodo API lookup identifies the canonical file as
`modelnet40_c.zip`, `1,970,686,633` bytes, MD5
`c4a7fffaa52c80b33f7b3a0ac7782d3b`. This closes archive identification, not
content parity: the next step is to extract the archive outside the repo and
compare all 15 corruption-file hashes against the Colab manifest. The
Point-MAE checkpoint remains a separate `[Open]` provenance question.

The supplied completed report
`result/modelnet40_c/provenance_report/provenance_report.json` supersedes that
interim deferral. It confirms the Zenodo archive MD5/size and all 15 archive
member hashes match the source-only manifest; all 15 current Colab files match
as well. Dataset provenance is therefore closed as a source-gap explanation.
The Point-MAE checkpoint remains a separate `[Open]` question. The next
approved experiment is the shared-trajectory original-versus-updated
final-style decoder control.

## Continuation update - 2026-09-19 preprocessing and pure VAE controls

The stale earlier next-action text above is superseded by the validated
controls below. Preserve both the historical handoff and this continuation
record; do not rewrite the raw archives.

**[Run] Preprocessing identity control:**
`result/modelnet40_c/preprocessing_identity/20260919-140904_identity-s5-all15-seed0.zip`
is a complete seven-file, all-15 ModelNet40-C severity-5 run at batch 32,
seed 0. It obtains **55.0243%** (**20,370/37,020**) versus the direct
source-only comparator's **53.6899%** (**19,876/37,020**): **+1.3344 percentage
points**, or **+494 correct examples**. It is higher on 9/15 corruptions, equal
on Upsampling, and lower on 5/15. Largest gains are Density Increase
`+8.1848 pp`, Cutout `+6.0373 pp`, Density `+3.6467 pp`, and Occlusion
`+2.8363 pp`; largest declines are LiDAR `-4.6596 pp` and Background
`-4.2950 pp`. This is the important source-model finding: the TTA
preprocessing/output chain alone raises source-only accuracy modestly, without
LION, VAE, diffusion, or guidance.

**[Run] Pure VAE control:**
`result/modelnet40_c/pure_vae_encode_decode/20260919-144513_pure-vae-s5-all15-seed0.zip`
is also complete and obtains **54.7947%** (**20,285/37,020**), **+1.1048 pp**
versus source-only but **-0.2296 pp** versus preprocessing identity. Under this
one-seed stochastic control, most of the modest positive source delta is
localized to preprocessing; VAE reconstruction adds no net aggregate gain.
This does not establish a causal diffusion/guidance estimate because the
available 3DD-TTA context run uses a different commit and random draw.

The completed implementation and research records are on
`baseline-repro-clean` through `23e7a443d1f1108f7b503dc9ab79973b70bd47cf`.
This continuation adds two seed-stability diagnostics. The preprocessing-only
screen is now complete: seeds 0/1/2 are 55.0243%, 55.0675%, and 54.9487%,
mean 55.0135% with 0.0602 pp sample SD, and mean +1.3236 pp over source-only.
The two new archives are complete and hash-validated; their filenames both say
seed1, but config/command establish the earlier timestamp as seed 1 and the
later timestamp as seed 2. The pure-VAE screen is also complete: seeds 0/1/2
are 54.7947%, 54.9379%, and 54.8082%, mean 54.8469% with 0.0790 pp sample SD,
and mean +1.1570 pp over source-only. Its filenames likewise both say seed1;
config/command establish the earlier timestamp as seed 1 and the later as seed
2. At every seed, pure VAE is below the matched preprocessing identity result
by 0.2296/0.1297/0.1405 pp.

The two seed-stability diagnostics are:
`--method pure_vae_seed_stability` and
`--method preprocessing_identity_seed_stability`. Both accept only seed 1 or 2
and preserve their seed-0 method contracts. Both seed screens are now complete.
The paired pure-VAE minus preprocessing-identity result at each seed isolates
the VAE contribution and is negative at all three seeds. A same-commit
common-draw pure-VAE versus eval/raw 3DD-TTA control would only strengthen a
causal diffusion/guidance claim; it is deferred because it is not expected to
change the accuracy ordering. Report that causal attribution as [Open] unless
the stronger thesis claim later becomes necessary.
Keep GSD/PxP, EMA, alternate FPS policies, and other datasets parked.

## Shared-trajectory decoder control implementation - 2026-09-20

**[Code]** The opt-in `shared_trajectory_decoder_control` method is now
implemented on `baseline-repro-clean`. It runs one raw/eval LION trajectory
per batch and decodes the same final local latent with original and updated
styles. It is locked to severity 5, Gaussian/Impulse, complete files, batch
32, seeds 0/1/2, raw weights, EMA off, and the existing gamma/eta/lambda and
scheduler settings. The second classifier call restores NumPy and Torch
CPU/CUDA RNG state. The seven-file artifact contract remains intact, with both
decoder branches and paired diagnostics in the two CSVs/config.

**[Open]** No Colab result exists yet. After the code commit is pushed, fetch
`origin/baseline-repro-clean` in Colab and run the supplied seed-0 command,
then repeat seeds 1 and 2. Validate ZIP structure, completion/traceback,
commit, checkpoint/data hashes, branch metrics and per-corruption deltas
before any all-15 confirmation.
