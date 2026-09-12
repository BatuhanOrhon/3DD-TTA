# Batch 1: Colab baseline smoke handoff

## Purpose and status

2026-09-12, **[Code]**. Local implementation on baseline-repro-clean, based on documentation commit b31fd23193bbcb9a5c189cfb4118be41506f9333. The implementation commit will be reported in the handoff. No Colab artifact or accuracy evidence exists yet.

This checks execution and the seven-file output contract, not whether the accuracy gap is fixed. Do not start source-only, dropout A/B or GSD until this ZIP is reviewed.

## Transfer and environment

The agent does not push. First transfer the current five Python files to the same repository in Colab, or synchronize the branch through Git yourself:

- run_baseline.py
- research_artifacts.py
- main_3dd_tta.py
- tta.py
- utilities_3dd_tta.py

Keep your previously working Colab environment/checkpoints/data. Do not upgrade dependencies while validating this batch. Record any necessary environment repair as a deviation. Run from the repository root. Replace the /content path and asset paths below if your setup differs.

## Run one smoke

```python
%cd /content/3DD-TTA
!python -u run_baseline.py --corruptions gaussian --max-batches 2 --batch_size 32 --seed 0 --gamma 0.01 --eta 0.01 --lambdaa 0.95 --run-name baseline-smoke_seed0 --pointmae_ckpt ./pointnet_ckpts/modelnet_jt.pth --diff_ckpt ./lion_ckpts/epoch_10999_iters_2100999.pt --dataset_root ./data/modelnet40_c --label_path ./data/modelnet40_c/label.npy
```

The chosen smoke batch is 32 (README example); runner default remains original CLI 40. Keep the declared batch fixed in later comparisons; if OOM requires a smaller batch, label that deviation. Repository lambda=0.95 is retained, not silently replaced with paper lambda=0.96. Only two file-order batches of Gaussian severity 5 are evaluated.

No LION eval call is introduced. Original scheduler defaults, 5 normal / 35 background steps, summed latent-first-three-channel SCD, gamma-local/eta-style mapping and original-shape-latent final decoding remain.

## Download and supply

The runner prints the directory and ZIP paths:

```text
result/modelnet40_c/3dd_original/<UTC-timestamp>_baseline-smoke_seed0/
result/modelnet40_c/3dd_original/<UTC-timestamp>_baseline-smoke_seed0.zip
```

Download the exact printed ZIP through the Colab file panel. Or, immediately after your run:

```python
from pathlib import Path
from google.colab import files
archives = sorted(Path("result/modelnet40_c/3dd_original").glob("*_baseline-smoke_seed0.zip"),
                  key=lambda path: path.stat().st_mtime)
if not archives:
    raise RuntimeError("No smoke ZIP found; inspect the run log.")
files.download(str(archives[-1]))
```

Send the complete ZIP, not a screenshot or final percentage. Remove credentials before sharing; do not silently alter raw numerical evidence. Confirm the Colab runtime/GPU type and any setup changes. Send failed-run ZIPs too. A runtime forcibly destroyed by Colab may prevent ZIP finalization; in that case supply the surviving directory/log/notebook rather than assuming completion.

## What the agent checks next

- Seven required files, same run ID, exact command/config agreement.
- Asset hashes, actual library/extension identities, scheduler and model modes.
- Correct/full counts, observed batch sizes and consistent CSV units (fractions).
- execution_status=complete with status=partial is expected for a successful prefix smoke: partial refers to dataset coverage.
- No missing/unexpected classifier keys silently ignored in interpretation; LION strict-load reports inspected.
- No hidden environment or adaptation change.

Raw/failed artifacts will be preserved under result/; derived analyses stay separate. Only after reviewing this evidence do we implement Batch 2 source-only.

## Local verification limits and user preference

Syntax/CLI, count/schema/collision/failure-state checks and observer-stripped AST comparisons were performed without CUDA/model execution. These do not prove GPU tensor/prediction parity or accuracy.

User requested no unit-test files for this research workflow. Use proportionate syntax, structural/protocol checks and archived controlled Colab experiments; do not create a unit-test suite by default.
