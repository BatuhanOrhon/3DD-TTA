# Repository instructions

@/Users/batuhan/.codex/RTK.md

For thesis work, read `.agents/skills/iterative-thesis-researcher/SKILL.md` and
start from `knowledge/README.md`; reconcile dated notes with the actual checkout.

For Colab work, read `.codex/skills/colab-research-operator/SKILL.md`.
Use Colab CLI. Obtain explicit user approval only for GPU runs reasonably
expected to exceed 60 minutes; estimate the whole planned GPU phase, not each
CLI call. Within the requested task, shorter runs, CPU work, setup, allocation,
retries and artifact retrieval need no additional research-run approval.
Update the knowledge records as the skill specifies.

For each new Colab runtime, prepare the environment with the ten code cells in
`scripts/colab_environment_setup.ipynb`, extracted from the original preparation
notebook. Run cell1 separately; reconnect to the same VM after Conda restarts
the kernel, then run cells2–10 in order. Preserve the project working directory
between CLI calls and verify each step before continuing. Do not run research
cells from the full notebook as part of setup.
Read `knowledge/colab_cli_setup_20261017.md` for the latest validation status;
the r2 A100 setup passed native GPU, dataset and checkpoint checks after the
recorded repairs. The ten-cell split notebook has not been replayed on a fresh
VM. The user defers only the known cell2 fastrlock pip failure. Cell4 prepares
pinned dependencies, environment-local CUDA12.1/GCC11.4 and patched native
sources. Cells5–8 build EMD, Chamfer, PointNet++ and PVCNN/smoke in separate
restartable phases with completion markers. Cell9 uses modelnet40_c.zip for
clean and corrupted data; no separate original ZIP is required. Cell10 runs
scripts/verify_colab_environment.py. CUDA/native,
dataset and checkpoint failures stop setup. ScanObjectNN preparation stays
in its dedicated research section; do not bypass an invalid Drive receipt.
