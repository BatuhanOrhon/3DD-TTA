#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Use the existing environment; no installation, checkpoint or dataset mutation.
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage smoke --execute
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage smoke --smoke-corruption background --execute
conda run --no-capture-output -n 3dd_tta_env python eval_gsd_tta.py --stage pilot --execute
