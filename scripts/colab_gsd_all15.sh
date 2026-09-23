#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Provisional 14-corruption confirmation: Background is excluded because its
# 35 reverse steps dominate runtime. The canonical all-15 stage remains
# available as `--stage all15` after this gate is reviewed.
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage all14 --execute
