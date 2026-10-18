# Point-MAE classifier-boundary parity preparation — 2026-10-18

[Code] Prepared `scripts/verify_multibackbone_pointmae_parity.py`; no GPU phase
has been executed. This is the prerequisite call-parity smoke before new
backbone inference, not source-only all15 accuracy or paper reproduction.
The complete conditional GPU campaign remains subject to its single user
approval; this preparation does not grant that approval.

The actual invocation must execute from the pinned
`<asset-root>/source_snapshot` tree. The worker verifies the externally pinned
asset-manifest SHA-256 and every listed file before importing model/native
code. Dataset file hashes, shapes, counts, class bounds and finite XYZ use
the source runner's existing validators. ScanObjectNN-C requires the
581-example `main_split`; `main_split_nobg` is rejected. ModelNet40-C requires
2,468 objects. External existing checkpoints are pinned separately:

| Dataset | Checkpoint | SHA-256 |
|---|---|---|
| ModelNet40-C | `modelnet_jt.pth` | `507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75` |
| ScanObjectNN-C | `scanobject_jt.pth` | `53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac` |

After validation, the original builder constructs the Point-MAE config for
40/15 classes, loads the `base_model` payload with `weights_only=True` and
`strict=True`, removes only a leading `module.` prefix, and rejects key
collisions. No LION module or checkpoint is loaded.

For Gaussian, Density and Background at severity5 on each dataset, native
`utils_mate.misc.fps(..., 1024)` runs once on the first32 original file IDs.
Both direct `classification_only(..., only_unmasked=False)` and
`FrozenClassifier` receive the exact prepared tensor. Raw pinned XYZ -> FPS
is used solely to check this call boundary; this smoke does not assert parity
with the established reconstruction/preprocessing pipeline or paper accuracy.
Python, NumPy, CPU torch and all CUDA RNG states are restored between calls.
Logits, predictions and post-call RNG states must match exactly. Input values
and the complete model state, including BN buffers, must remain unchanged.

Each dataset gets a fresh standard seven-file RunBundle plus indexed direct
and adapted logits/predictions NPZs, copied executed sources, comparison
diagnostics and native/checkpoint/data identities. All three result rows
remain `partial` with32 examples; successful boundary verification is recorded
as `parity_status="passed"`. Mismatches preserve their logits before failing.
The launcher streams worker output to both stdout and the bundle log, seals
the bundle and writes a non-overwriting ZIP even on a worker failure. A failed
first dataset stops the second dataset. Drive retrieval/hash verification
belongs to the operator's approved phase and has not occurred.

Command preview, with concrete manifest digest and destinations supplied by
the asset operator before approval:

```sh
python /content/multibackbone_assets/source_snapshot/scripts/verify_multibackbone_pointmae_parity.py \
  --asset-root /content/multibackbone_assets \
  --asset-manifest-sha256 <PINNED_MANIFEST_SHA256> \
  --modelnet-root /content/3DD-TTA/data/modelnet40_c \
  --checkpoint-root /content/3DD-TTA/pointnet_ckpts \
  --result-root /content/multibackbone_pointmae_parity \
  --batch-size 32 --seed 0 --preview
```

Remove `--preview` only within the approved GPU phase. The input count is fixed
at32; the CLI deliberately accepts only `--batch-size 32` for this smoke.

[Run] Tests were written before the script. RED showed the missing script;
then three focused preparation tests passed with cached CPU PyTorch2.14.1 and
NumPy. The CLI test uses `python -S` for both `--help` and `--preview`, proving
they need no installed scientific/native packages or existing assets. The
comparison test consumes three CPU RNG families and confirms exact parity,
then detects an intentionally changed logits result. CUDA RNG lists are empty
in that CPU test; actual CUDA RNG/kernel behavior remains unverified.
The third test executes only a deliberately invalid-manifest failure path,
with hardware/environment commands replaced by a CPU test response. It checks
that validation stops before native/model runtime fields appear, preserves the
failure traceback, and seals exactly one failed ZIP. This reproduced the
duplicate-ZIP regression when the shared sealing helper gained ZIP ownership;
the launcher now calls that helper once. The worker runs unbuffered and closes
its stdout stream; the failure-path test also passed with ResourceWarning
promoted to an error. Native extension identities use the shared source
snapshot helper's relevant PointNet2/KNN/Chamfer filter.

Validation command:
`uv run --offline --no-project --with torch --with numpy python -W error::ResourceWarning -m unittest discover -s tests -p test_multibackbone_pointmae_parity.py -v`.
Both new Python files passed `ast.parse(..., feature_version=(3, 8))`.
Actual Python3.8/PyTorch2.1.2+cu121, checkpoints, native FPS and GPU execution
must still be verified remotely; no inference claim follows from these CPU
preparation tests.
