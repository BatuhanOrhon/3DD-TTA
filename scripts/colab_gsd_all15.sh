#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Execute after reviewing complete pilot artifacts and locking the configuration.
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage all15 --execute
