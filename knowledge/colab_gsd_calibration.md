# Colab: fixed-state GSD calibration, then a staged development screen

[Code] Branch `gsd-smooth-spectrum`, implementation based on `d75a32d`.
See [the approved execution plan](gsd_calibration_execution_20260927.md) and
[mathematical/statistical rationale](gsd_calibration_review_20260927.md).
These commands use the existing installed Colab environment and downloaded
ModelNet40-C/LION/Point-MAE assets. No dependency reinstall is needed.

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

This is the only new guidance condition in this handoff: smooth beta .5,
rho .01, seed 0 and 128 examples per corruption. It completes the already
partially observed beta `.5/2` by rho `.001/.01` development table. It is not
a holdout, and no result exists yet. Keep the existing calibration reference:
the runner checks its source/input hashes, so do not replace it with a report
JSON or a different run.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
git switch gsd-smooth-spectrum
git pull --ff-only origin gsd-smooth-spectrum
ROOT=./result/modelnet40_c/gsd_latent_spectral_smooth_v2
CALIBRATION="$ROOT/20260927-201522_gsd-cal-diagnose-reference-seed0-n64/config.json"
test -f "$CALIBRATION"
conda run --no-capture-output -n 3dd_tta_env python run_baseline.py \
  --method gsd_latent_spectral_smooth_v2 --batch_size 32 --seed 0 --severity 5 \
  --lambdaa .95 --gamma .01 --eta .01 --max-batches 0 --lion-eval-mode \
  --result-root ./result \
  --run-name gsd-cal-interaction-smooth-beta0p5-rho0p01-seed0-n128 \
  --corruptions gaussian impulse --gsd-stage development \
  --gsd-weight 81.39104941932808 --gsd-scd-weight 1 \
  --gsd-profile smooth --gsd-beta .5 --gsd-development-count 128 \
  --gsd-split-seed 20260927 --gsd-target-rho .01 \
  --gsd-calibration-reference "$CALIBRATION"
```

The direct command intentionally leaves `eval_gsd_calibration.py` unchanged:
that file is in the calibration runtime source manifest, and changing it would
correctly block use of this immutable reference. The runner prints the exact
directory and seven-file ZIP path. Set `NEW_RUN_ZIP` to that printed ZIP, then
produce the compact five-arm interaction record without opening archive rows:

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
ROOT=./result/modelnet40_c/gsd_latent_spectral_smooth_v2
NEW_RUN_ZIP="<ZIP path printed by the new run>"
conda run --no-capture-output -n 3dd_tta_env python scripts/analyze_gsd_calibration.py \
  --calibration "$ROOT/report.json" --interaction-screen \
  --screen-run \
    "$ROOT/20260927-203752_gsd-cal-screen-weight-baseline-seed0-n128.zip" \
    "$ROOT/20260927-205045_gsd-cal-screen-beta-smooth-beta0p5-rho0p001-seed0-n128.zip" \
    "$ROOT/20260927-203909_gsd-cal-screen-weight-smooth-beta2p0-rho0p001-seed0-n128.zip" \
    "$ROOT/20260927-203957_gsd-cal-screen-weight-smooth-beta2p0-rho0p01-seed0-n128.zip" \
    "$NEW_RUN_ZIP" \
  --output "$ROOT/beta_rho_interaction_summary.json"
```

Share the single ZIP printed by this run as an attachment, preserving its
original filename and all seven members: `command.txt`, `config.json`,
`environment.txt`, `stdout.log`, `summary.csv`, `per_corruption.csv` and
`notes.md`. Do not send only the compact summary or screenshots.

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
