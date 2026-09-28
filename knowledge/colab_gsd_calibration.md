# Colab: fixed-state GSD calibration, then a staged development screen

[Code] Branch `gsd-smooth-spectrum`, implementation based on `d75a32d`.
See [the approved execution plan](gsd_calibration_execution_20260927.md) and
[mathematical/statistical rationale](gsd_calibration_review_20260927.md).
These commands use the existing installed Colab environment and downloaded
ModelNet40-C/LION/Point-MAE assets. No dependency reinstall is needed.

## 0. Check the LION import dependencies after a `torch.xpu` error

The supported Colab combination is PyTorch `2.1.2+cu121`, torchvision
`0.16.2+cu121`, torchaudio `2.1.2+cu121`, Diffusers `0.11.1` and Hugging
Face Hub `0.11.1`. The earlier `torch==2.0.1+cu121` environment pin cannot be
resolved from the CUDA 12.1 wheel index. Existing successful run artifacts
record the 2.1.2/0.16.2/2.1.2 tuple. A newer Diffusers release accesses
`torch.xpu` during import; the PyTorch build in the reported failed run lacked
that API.
Do not add a fake `torch.xpu` attribute. First inspect installed versions
without importing Diffusers:

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
conda run --no-capture-output -n 3dd_tta_env python -c "import importlib.metadata as m, torch; print('torch:', torch.__version__); print('torchvision:', m.version('torchvision')); print('torchaudio:', m.version('torchaudio')); print('diffusers:', m.version('diffusers')); print('huggingface-hub:', m.version('huggingface-hub'))"
```

If only Diffusers or Hub differs, restore only these two packages in the
existing environment, then verify the LION scheduler import. This does not
run a model or change repository source hashes:

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
conda run --no-capture-output -n 3dd_tta_env python -m pip install --no-deps \
  diffusers==0.11.1 huggingface-hub==0.11.1
conda run --no-capture-output -n 3dd_tta_env python -c "import importlib.metadata as m, torch; from diffusers import DDPMScheduler; print('torch:', torch.__version__); print('diffusers:', m.version('diffusers')); print('huggingface-hub:', m.version('huggingface-hub')); print('DDPMScheduler import: OK')"
```

Proceed to calibration only after that import check passes. Keep the failed
run ZIP: its `environment.txt` records the installed versions.

For a new environment, pull the corrected `env.yaml` first, then create or
update the environment from that file. It now aligns all three PyTorch-family
versions with the CUDA 12.1 wheels. The CUDA wheel pages are supplied as
`--find-links`, leaving PyPI as pip's normal index for the rest of the
dependencies. `argparse` is omitted because Python 3.8 provides it in the
standard library:

```bash
conda env create -f env.yaml
```

If the failed create left `3dd_tta_env` present, update it instead:

```bash
conda env update -n 3dd_tta_env -f env.yaml
```

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
