# Colab: fixed-state GSD calibration, then a staged development screen

[Code] Branch `gsd-smooth-spectrum`; the existing smooth-v2 calibration
workflow is extended with the approved full-test-set screen below.
See [the approved execution plan](gsd_calibration_execution_20260927.md) and
[mathematical/statistical rationale](gsd_calibration_review_20260927.md).
These commands use the existing installed Colab environment and downloaded
ModelNet40-C/LION/Point-MAE assets. No dependency reinstall is needed.

## Current next run — full ModelNet40-C test-set candidate screen

**[Approved]** The prior 512-example subset proposal is superseded. The next
screen evaluates every example in the existing ModelNet40-C severity-5 test
files for all 15 corruptions, with seeds 0/1/2 and batch 32. It compares
SCD-only, calibrated beta .5/rho .001, and calibrated beta 2/rho .01, all
bound to reference `20260928-113047_gsd-cal-diagnose-reference-seed0-n64`
(raw config SHA-256
`550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6`).
This uses the same all-corruptions test examples as the existing runs. The
adaptation receives point clouds only; labels are read after predictions for
accuracy metrics. Since the candidates are compared on the full test set,
choosing one from these scores is descriptive development evidence, not an
independent confirmation result.

Run from the repository root after the full-dataset stage and scripts are
present in the Colab checkout. Point `CALIBRATION` to the preserved raw
calibration `config.json` from the reference run:

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
CALIBRATION="./result/modelnet40_c/gsd_latent_spectral_smooth_v2/20260928-113047_gsd-cal-diagnose-reference-seed0-n64/config.json"
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/run_gsd_full_dataset_screen.py \
  --calibration "$CALIBRATION" --result-root ./result
```

If only the reference ZIP remains, copy its `config.json` member to a scratch
path first; do not edit the ZIP:

```bash
ROOT=./result/modelnet40_c/gsd_latent_spectral_smooth_v2
RUN_ID=20260928-113047_gsd-cal-diagnose-reference-seed0-n64
mkdir -p ./tmp/full-gsd-screen-reference
unzip -p "$ROOT/$RUN_ID.zip" "$RUN_ID/config.json" \
  > ./tmp/full-gsd-screen-reference/config.json
CALIBRATION=./tmp/full-gsd-screen-reference/config.json
```

The launcher preflights the reference ID, raw config hash, and two calibrated
coefficients before starting a child run. It then runs the nine arm/seed jobs
sequentially in the existing `3dd_tta_env`; it does not install packages or
pull Git. Use `--resume` after interruption. Resume skips only a complete,
CRC-valid seven-file ZIP whose config matches its arm, seed, full 15-file
coverage and pinned reference. Failed, incomplete, corrupt or mismatched ZIPs
are retained; a retry gets a distinct `-attemptNN` run name.

The output is nine immutable ZIPs under
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/`, named
`<UTC>_gsd-full-screen-<reference>-<arm>-seed<N>.zip`, plus
`full_test_screen_summary_<reference>.json`. The summary reports per-seed
per-corruption accuracy, the unweighted 15-corruption macro, candidate-minus-
SCD-only percentage-point deltas by corruption and seed, across-seed mean and
sample SD, runtime, and peak memory. If any command fails, the launcher stops
and prints completed archives; rerun with `--resume` to continue.

For local ingestion, preserve all nine ZIPs and the summary JSON together.
The strict analyzer interface is:

```bash
ROOT=./result/modelnet40_c/gsd_latent_spectral_smooth_v2
RUN_ID=20260928-113047_gsd-cal-diagnose-reference-seed0-n64
CALIBRATION="$ROOT/$RUN_ID/config.json"
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/analyze_gsd_full_dataset_screen.py \
  --calibration "$CALIBRATION" --screen-run <SCD0.zip> <SCD1.zip> <SCD2.zip> \
  <BETA05_0.zip> <BETA05_1.zip> <BETA05_2.zip> \
  <BETA2_0.zip> <BETA2_1.zip> <BETA2_2.zip> \
  --output "$ROOT/full_test_screen_summary_<reference>.json"
```

No new calibration is run. The existing reference's Gaussian/Impulse data,
checkpoint/config identities and unchanged inference-source identities are
checked; this screen's current protocol/worker source hashes are recorded and
must agree across its nine runs.

## Current restart handoff — 2026-09-28

**[Run] Completed and ingested:** reference113047 and all five development
ZIPs are validated. The command below is historical; do not rerun it.
Read [results and the nine-run larger-development proposal](gsd_interaction_results_20260928.md).
Keep the raw 113047 config and current installed environment for that next
comparison; it uses the existing direct development runner, not fresh diagnosis.

This takes precedence over the historical reuse-of-old-ZIPs instructions in
section 4. The user reported a coefficient mismatch between a fresh reference
and the old beta .5/rho .001 bundle, then requested regenerated comparisons.
Runtime commit `c1c7467` supports:

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
git switch gsd-smooth-spectrum
git pull --ff-only origin gsd-smooth-spectrum
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/run_gsd_interaction_from_scratch.py --rebuild-prerequisites
```

If this is already running, let that invocation finish; this is not a resume
command. It generates one diagnostic reference (64/corruption), SCD-only,
beta .5/rho .001, beta2/rho .001, beta2/rho .01, and beta .5/rho .01
(128/corruption for all five development runs). All coefficients come from
that one reference. Old ZIPs are not inputs. Beta8/hard are still diagnostic
probes, but are not rerun as guided accuracy arms. No alpha from the old report
is hard-coded in this path.

Provide all six seven-file ZIPs and
`beta_rho_interaction_summary_<CALIBRATION_RUN_ID>.json`. Validate fresh
diagnostics, reference identity, environment and matched controls before
interpreting the new result. See the [restart review](gsd_interaction_review_20260928.md)
and [next-agent handoff](gsd_interaction_handoff_20260928.md). The prior
Diffusers import issue was resolved in the existing Colab environment by the
user's .11.1 Diffusers/.11.1 Hub restoration; do not recreate the environment
or change `env.yaml`/`requirements.txt` from `dev`.

## 1. Diagnostic run: first required Colab test

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
git switch gsd-smooth-spectrum
git pull --ff-only origin gsd-smooth-spectrum
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_calibration.py \
  --phase diagnose --execute
```

Default: severity 5 Gaussian+Impulse, seed 0, 64 shuffled indices each,
batch 32, raw/eval LION, SCD weight 1, lambda .95, gamma=eta=.01.
The fixed split seed is 20260927, independent of model RNG. All five SCD-only
reverse steps record per-sample state/update metrics; steps 0/2/4 additionally
probe hard and smooth beta .5/2/8 on the same states. No probe gradient is
applied to the trajectory. Extra backward passes cost time and may affect peak
memory; sharing one eigensystem avoids four stored dense operators.

The runner prints `Run directory:` and `ZIP to provide:`. It produces the
usual seven files, with `development_split`, `gsd_sample_diagnostics` and
`gsd_diagnostics` inside `config.json`. Coverage is deliberately `partial`,
even after successful execution of every selected sample. Execution must be
`complete`. Its classifier score is an exploratory SCD reference, not a
smooth-guidance accuracy measurement.

## 2. Read the label-free report

Set CALIBRATION to the exact directory printed by the diagnostic run. Do not
silently select the most recent unrelated run. Avoid printing the full report
JSON into notebook output; use the compact analyzer below. It reads either
the raw calibration run or an existing `report.json`, then emits only
beta-level statistics and candidate weights. It does not use labels or accuracy.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
CALIBRATION="/content/3DD-TTA/result/modelnet40_c/gsd_latent_spectral_smooth_v2/<RUN_ID>"
conda run --no-capture-output -n 3dd_tta_env python scripts/analyze_gsd_calibration.py \
  --calibration "$CALIBRATION" \
  --output /content/3DD-TTA/result/modelnet40_c/gsd_latent_spectral_smooth_v2/calibration_summary.json
```

For each candidate, `R=median(local spectral norm / local SCD norm)` across
both corruptions, samples and the three probes. Alpha is `rho/R`, for rho
0.0001/0.001/0.01. A missing denominator or a near-zero median suppresses the
weight recommendation. Inspect SCD state/DDIM ratios and tails before running
the screen. No numerical threshold in this report proves instability. If a
`--phase report` command has already produced a large `report.json`, pass that
path as `--calibration`; the analyzer reads the file locally and reduces it.

For the existing report in the result root (the compact summary already
generated as `calibration_summary.json`):

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
conda run --no-capture-output -n 3dd_tta_env python scripts/analyze_gsd_calibration.py \
  --calibration ./result/modelnet40_c/gsd_latent_spectral_smooth_v2/report.json \
  --output ./result/modelnet40_c/gsd_latent_spectral_smooth_v2/calibration_summary.json
```

## 3. Actual guidance screen after diagnostic review

This launches four conditions on the same shuffled 128-index pool per
corruption: SCD-only, and beta2 with three fixed calibrated weights.
The initial 64 diagnostic indices are contained in this pool. Each condition
loads the models separately; graph/forward sharing applies to diagnostic
probes only. Omit `--execute` to inspect commands without loading models.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
CALIBRATION="/content/3DD-TTA/result/modelnet40_c/gsd_latent_spectral_smooth_v2/<RUN_ID>"
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_calibration.py \
  --phase screen-weight --calibration "$CALIBRATION" --execute
```

Summarize the four resulting ZIPs (baseline and beta2 at three rho values)
without printing their contents:

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
ROOT=./result/modelnet40_c/gsd_latent_spectral_smooth_v2
WEIGHT_RUNS=("$ROOT"/*gsd-cal-screen-weight-*.zip)
conda run --no-capture-output -n 3dd_tta_env \
  python scripts/analyze_gsd_calibration.py \
  --calibration "$ROOT/report.json" --weight-screen \
  --screen-run "${WEIGHT_RUNS[@]}" \
  --output "$ROOT/screen_weight_summary.json"
```

Review the compact ranking, then carry its highest observed rho into the beta
comparison. This one-seed subset is exploratory, not confirmation.

The reference config's SHA-256/run ID and target rho are recorded. Data,
checkpoint/config, labels and runtime source hashes must match calibration;
changed inputs are rejected before model loading. Graph/host settings must
match too. Recalibrate after a code change; do not edit archived configs to
bypass this guard. The local tests establish fixture parity, not native CUDA
parity or common random draws between separate screening invocations.

Once those results are reviewed, select rho explicitly and use:

```bash
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_calibration.py \
  --phase screen-beta --calibration "$CALIBRATION" --rho <SELECTED_RHO> --execute
```

This adds beta .5, beta8 and matched-rho hard, reusing the preceding beta2 and
baseline results. Default seed/count remain 0/128; do not change them between
phases of the first screen. Later `--seed 1`/`--seed 2` and `--count 256` or
`512` support explicitly declared expansion. Rerun corresponding controls
whenever seed/count changes. Interaction checks and final held-out evaluation
are a subsequent declared experiment, not automatically selected by this script.

To reduce the five screen bundles to a compact accuracy ranking, provide their
run directories in any order:

```bash
conda run --no-capture-output -n 3dd_tta_env python scripts/analyze_gsd_calibration.py \
  --calibration ./result/modelnet40_c/gsd_latent_spectral_smooth_v2/report.json \
  --screen-run <SCD_RUN_DIR> <BETA2_RUN_DIR> <BETA05_RUN_DIR> <BETA8_RUN_DIR> <HARD_RUN_DIR> \
  --rho 0.001 \
  --output ./result/modelnet40_c/gsd_latent_spectral_smooth_v2/beta_screen_summary.json
```

The top value is the observed development-set winner at this rho, not a final
optimum. The analyzer checks common indices, seed, count, and calibration run
ID, then gives each candidate and its per-corruption delta versus SCD-only.
The analyzer also checks each run's spectral coefficient against the original
label-free calibration report before ranking.

Index correspondence across corruption files is not verified object identity.
Reserve every touched index and establish object correspondence before any
held-out claim. A 128-example subset cannot reliably establish a 1 pp gain.
SCD weight .5, dynamic weights and source-data calibration are not added here.

The registered seed-0 beta screen has since completed. Its compact result is
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/beta_screen_summary.json`:
beta2/rho .001 ranks first (+1.1719 pp macro), with beta .5 at +.7813 pp and
beta8/hard tied with SCD-only. The [2026-09-28 review](gsd_calibration_audit_20260928.md)
corrects the earlier immediate-freeze recommendation: finish the planned
interaction check and larger/repeated-seed development comparison first,
keeping beta .5 and 2. Separately prepare object-disjoint confirmation; the
current launcher cannot produce it because development deliberately includes
the diagnostic pool. Existing nested pools measure development performance
and repeated-seed stability. Counts, run IDs and limits are in the findings log.

## 4. Registered one-arm beta-rho interaction extension

Historical equivalence-based reuse path: the current user-authorized restart
above replaces this command for the ongoing experiment. Keep this description
as provenance for why the old/new coefficient mismatch stopped execution.

This is the only new guidance condition in this handoff: smooth beta .5,
rho .01, seed 0 and 128 examples per corruption. It completes the already
partially observed beta `.5/2` by rho `.001/.01` development table. It is not
a holdout, and no result exists yet. The original diagnostic `config.json` is
missing from the current Colab environment. Use the dedicated orchestrator
below: it creates a fresh diagnostic reference, derives alpha from that run,
and checks that the existing baseline/three interaction bundles have identical
split, source/data/checkpoint manifests and calibrated coefficients before it
starts the new guidance condition. It stops before the interaction run if any
check differs; in that case, do not combine results from the two calibration
generations.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
git switch gsd-smooth-spectrum
git pull --ff-only origin gsd-smooth-spectrum
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/run_gsd_interaction_from_scratch.py
```

The script prints both generated seven-file ZIP paths and writes the compact
summary at `result/modelnet40_c/gsd_latent_spectral_smooth_v2/beta_rho_interaction_summary.json`.
Keep/share the fresh diagnostic ZIP as well as the new interaction ZIP; the
diagnostic is now the provenance source for the new coefficient. Each ZIP
contains `command.txt`, `config.json`, `environment.txt`, `stdout.log`,
`summary.csv`, `per_corruption.csv` and `notes.md`. The script uses existing
Colab dependencies, leaves `eval_gsd_calibration.py` unchanged, and never
launches beta8 or hard reruns.

## 5. Copy immutable ZIPs to Google Drive

Mount Drive in a Python cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then export the calibration/screen archives, including failed runs for diagnosis:

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
conda run --no-capture-output -n 3dd_tta_env python scripts/export_gsd_results.py \
  --source-root ./result \
  --drive-root /content/drive/MyDrive/3DD-TTA-results \
  --stage all --name-contains gsd-cal-
```

The exporter preserves dataset/method/run paths and refuses to overwrite a
different archive with the same name. Provide the complete diagnostic ZIP
first, including config and stdout, rather than only the final accuracy.
Remove any credentials from shared material. Raw artifacts are never rewritten
by report generation.
