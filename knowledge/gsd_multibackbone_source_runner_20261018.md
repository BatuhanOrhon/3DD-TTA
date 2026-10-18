# Multibackbone source-only runner — 2026-10-18

[Code] Batch3 preparation adds the opt-in
`scripts/eval_multibackbone_source.py` and focused CPU tests only. Existing
Point-MAE runners, GSD math, result columns and skills are untouched. The CLI
supports DGCNN, CurveNet, PointNeXt and an explicitly supplied Point-MAE
control. No LION import/load or GPU execution was performed for this batch.

The default scope is seed0, batch32, canonical all15/severity5, full file order;
`--max-batches 1 --corruptions gaussian` is a prefix smoke. `--preview` resolves
the contract without importing torch/numpy, touching assets, or creating a run.
An actual run requires the exact immutable `--asset-manifest-sha256` and fresh
RunBundle directory. The manifest declares `files` relative to asset-root and
`datasets[dataset] = {split, n_examples, files}`; each dataset basename entry
declares SHA256, byte count and full shape. ScanObjectNN requires 581 objects
and `main_split`; ModelNet40 requires2468. Dataset arrays must contain finite
floating BNC XYZ and valid integer labels. All selected full files are verified
before any inference. Traversal and symlink assets/data are rejected.

Asset source and checkpoint bytes are checked against the manifest before
loading. Model checkpoints use `torch.load(weights_only=True)` and strict state
loading; only a leading DataParallel `module.` prefix is stripped. The DGCNN
source's reversed `dataset_name` head-key convention is recorded explicitly.
The four non-PointNeXt constructor contracts match the existing CPU strict-load
probe. PointNeXt uses official OpenPoints BaseCls and explicit width64/input3
for ModelNet40, width32/input4 for ScanObjectNN.

Predeclared transfer preprocessing is raw -> native FPS1024 for SVWA models;
ScanObjectNN 3DeformRS CurveNet uses centroid/L-infinity normalization before
FPS. PointNeXt ModelNet40 uses raw FPS XYZ features; ScanObjectNN uses FPS,
unscaled height `y-min(y)`, centroid/unit-sphere XYZ and XYZ+height features.
These choices are benchmark transfers, not verified exact GSDTTA preprocessing.
Point-MAE keeps raw FPS without additional normalization.

The worker streams Python/native stdout+stderr to the CLI and `stdout.log`.
Seven standard bundle files are preserved, with additional per-corruption
NPZ logits/predictions/labels/original ordinals, before/after classifier-state
hashes, exact manifest/resolved configuration, imported source/native hashes,
and executed repository source bytes when run from a checkout. For immutable
staging, execute from `asset-root/source_snapshot/scripts/...` and include all
snapshot files plus imported asset sources/native binaries in `manifest.files`.
No ancestor git checkout is inferred when the executed snapshot has no `.git`.
Successful and failed invocations are finalized with a SHA256 manifest and a
fresh sibling ZIP opened in exclusive creation mode. Root workflow performs
the independent Drive/local ZIP verification and knowledge reconciliation.
The manifest's optional `source_snapshot_archive` is verified against both its
declared file identity and `source_snapshot_sha256`, then copied exclusively
to each result bundle's `source_snapshot.zip`. The second source-inventory
pass verifies the existing copy without overwriting it. This archive preserves
the exact repository/backbone/native source bytes for standalone recovery;
weights and evaluation data remain separately archived assets.

[Run] RED evidence: the first targeted unittest invocation failed because the
new runner module did not exist. After implementation12 CPU tests pass with
cached torch/numpy, including deliberately failed manifest execution and exact
failure-bundle/ZIP hash checks. Command:

```sh
uv run --offline --no-project --with torch --with numpy python -W error::ResourceWarning -m unittest discover -s tests -p test_eval_multibackbone_source.py -v
```

System Python without torch/numpy runs the seven standard-library contract tests
and skips the CPU tensor class. Both new Python files parse with Python3.8
grammar; this does not establish native Python3.8/PyTorch2.1.2 execution.

[Open] Real CUDA Point-MAE parity, six checkpoint forward smokes, timing and
all15 source results remain gated by the root agent's comprehensive GPU
approval. This batch made no commit/push and did not touch or close r7.


[Run] Final integration addendum: the combined26-test run (boundary10,
source13, parity3) passes. The extra source regression covers a nonexistent
relative synthetic torch module filename while preserving strict validation
for missing/undeclared absolute asset modules. Actual r7 Python3.8.20 and
torch2.1.2+cu121 strict-load all six models with the runner's exact constructors,
without CUDA initialization or forward execution. Only the final Module.cuda
transfer is replaced in the CPU preflight and Tensor.cuda is forbidden. This
supersedes the earlier lack of native Python3.8 CPU execution evidence, not
the still-open real GPU-forward/parity gate. Exact manifest/source/archive
identities and failed label-metadata attempt are in the main study note.
