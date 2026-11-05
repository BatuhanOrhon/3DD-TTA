# Frozen shared-probe classifier readouts — 2026-11-05 protocol

[Code/CPU/Open] Standalone posthoc operator prepared on 2026-11-06
Europe/Istanbul, after the supplemental scientific overlay was sealed.
`result/preparation/gsd_comprehensive_20261105/operator_v1/classify_readouts.py`
is archived separately with the parent operator package. It does not modify the
frozen probe, generation source, assets or production classifier/preprocessing.
Tests live in `tests/test_shared_probe_readouts.py`. Parent owns GPU execution,
whole-campaign geometry freeze, archive retrieval and later label/clean joins.
No classifier/model-weight inference, GPU or Colab run occurred locally.

## Invocation and acceptance boundary

```sh
python /absolute/operator_v1/classify_readouts.py \
  --capture-root /absolute/frozen_shared_capture \
  --capture-config-sha256 CAPTURE_CONFIG_SHA256 \
  --mechanism-root /absolute/completed_mechanism_probe \
  --mechanism-report-sha256 MECHANISM_COMPLETE_JSON_SHA256 \
  --asset-root /absolute/verified_assets \
  --asset-manifest-sha256 ASSET_MANIFEST_SHA256 \
  --output /absolute/fresh_readout_directory
```

[Code] Schedule this only **after all geometry generation is frozen**. The
operator itself requires a finalized `complete.json` mechanism receipt with
unchanged model/restored RNG and the exact pinned capture config. It accepts
complete-with-unavailable generation, preserving those coverage gaps. Capture
must be the original four-ID (459/545/1229/1435), four-corruption
(Gaussian/Background/Shear/LiDAR), B2/seed0 zero-weight shared-state control with
unchanged classifier state. Pin checks precede any scientific imports/inference.
All manifest Python sources are verified before import; scientific helpers
must originate in `ASSETS/source_snapshot`, whose complete manifest is verified
with the existing source helper. The standalone operator's own bytes/SHA are
recorded separately. Fresh-worker execution is required by the existing native
bootstrap; no JIT build or native repair is performed.

[Code] Validates every mechanism artifact against its final inventory, the
seven raw capture snapshots per B2 case against `shared_states.json`, the raw
shared manifest against `diagnostics.json`, the exact captured frame tensor
hashes/shapes/dtypes and the mechanism's raw-input identity receipts. Before
readout it verifies branch state/cloud payloads, original IDs, physical XYZ
shape/dtype, captured loop and actual next scheduler timestep. Unexpected or
duplicate branch metadata and tampered/missing certified payloads are integrity
errors. A failed/missing generation branch is explicitly unavailable, never
replaced with zeros or classified as if it completed.

## Frozen readout semantics

[Code] Enumerates all **224 B2 cloud keys**: four corruptions × two B2 batches ×
first/last captured checkpoint × seven branches × fractions.5/1. Each key
retains original IDs, corruption, checkpoint, next timestep, arm/fraction,
frame/state/cloud identities and exact cloud tensor hash. `coverage.json`
accounts for every key, including all failed/unavailable cases.

[Code] Loads only the frozen Point-MAE classifier through existing
`source.load_classifier`/`shared.initialize_heads`, requiring its full state
hash to equal the captured classifier. The existing no-build bootstrap imports
native-consumer Python modules but does not instantiate LION or load a LION
checkpoint. It verifies the same recorded native audit/bindings/extensions.
Existing `shared.decoded_to_geometry`, `source.preprocess`, native PointNet++
FPS/gather and `shared.classify_geometry` implement the actual preprocessing.
For ModelNet the canonical helper rotates back and independently normalizes
geometry per cloud; it does not unnormalize using the captured scale/center.
The exact frame is still supplied to the same helper and bound in metadata.
There is no bespoke frame/normalization/FPS replacement.

[Code] Classifier modules stay all-eval, parameters frozen, forward global
no-grad and input gradients disabled. A temporary pre-hook verifies and records
these actual flags; it is removed on success/failure. Each independent readout
uses fixed seed0 inside existing RNG/buffer isolation. Raw input geometry/RNG/
buffers and full classifier state are checked before/after; native file and
loaded scientific-source byte identities are likewise compared before/after.
No classifier gradient or extra PM vote is computed. Custom native kernels may
still vary despite fixed inputs/RNG; matching totals do not certify trajectories.

[Code] Each available branch writes an exclusive NPZ before its receipt:
original IDs, corruption, loop, next timestep, branch/fraction, **raw B2×40
logits**, predicted classes, exact classifier geometry, FPS indices and sampled
classifier points. There is no `label`, `y`, true margin or clean array field.
Each JSON receipt includes raw payload identities, actual classifier flags and
RNG/input/buffer guards. Final `complete.json` inventories every written file
and preserves technical failures. No score selects an arm, fraction or direction.

[Inference/Open] Parent CPU analysis may later join known labels to compute
correct/n and true-class margins for these **one-next-prediction** readouts,
clustering by four original objects and retaining corruption/checkpoint/arm
pairing. These are previously exposed mechanism objects, not a heldout cohort,
full trajectory, probability-mean benchmark or population accuracy estimate.
A classifier result cannot make a failed FD/direction branch available. No
accuracy benefit, deployment speed or universal causal explanation is claimed
by this implementation.

## CPU environment and verification receipt

[CPU] The earlier shared `/tmp/tta-lion-reencode-venv` no longer had its
`pyvenv.cfg` or NumPy/Torch package initializers; it was not repaired. Initial
readout tests therefore failed at imports. A fresh CPU-only environment was
installed once, then moved/regenerated at the parent's persistent location,
avoiding a second NumPy/Torch installation:

```text
/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-guidance-math-audit-20261103/result/preparation/gsd_comprehensive_20261105/cpu_venv/bin/python
```

The installation/relocation commands were:

```sh
python3.12 -m venv /tmp/gsd-shared-readouts-test-20261106
/tmp/gsd-shared-readouts-test-20261106/bin/python -m pip install numpy==1.26.4 torch==2.5.1
mv /tmp/gsd-shared-readouts-test-20261106 result/preparation/gsd_comprehensive_20261105/cpu_venv
python3.12 -m venv --upgrade result/preparation/gsd_comprehensive_20261105/cpu_venv
result/preparation/gsd_comprehensive_20261105/cpu_venv/bin/python -m pip install numpy==1.26.4 scipy==1.14.1 h5py==3.16.0 PyYAML==6.0.3
```

An initial unpinned SciPy dependency install briefly upgraded NumPy to2.5.4;
the explicit final pins above repaired that before acceptance. Final imported
versions: **Python3.12.10 / NumPy1.26.4 / Torch2.5.1 / SciPy1.14.1 /
h5py3.16.0 / PyYAML6.0.3** on macOS arm64. A Torch-to-NumPy conversion assertion
passed, `torch.cuda.is_initialized()` was false, and `pip check` reported no
broken requirements. Parent may reuse this environment for H5 CPU analysis.
The environment is a local runtime/cache artifact, not scientific source for
Git or the separately archived GPU operator. Actual readout CLI requires pinned
Linux/Torch2.1.2+cu121/CUDA12.1 and rejects this local environment before loading
weights. These differing toy/GPU backends are not a numerical parity claim.

[CPU] Three focused native-free tests pass:

```sh
result/preparation/gsd_comprehensive_20261105/cpu_venv/bin/python -m unittest discover -s tests -p test_shared_probe_readouts.py -v
```

They check all224 keyed coverage entries with explicit unavailable branches,
original IDs/frame forwarding, raw snapshot/cloud tamper rejection, immutable
unlabelled outputs, invalid keys/config/path rejection and the actual existing
isolated classifier helper with a tiny toy model. Classifier no-grad/eval/frozen
flags, FPS1024 request, RNG/cloud preservation and hook cleanup are checked.
Python3.8 AST, current-interpreter compile, dependency check, CLI help and
whitespace checks pass. The initial fixture used macOS's symlinked `/tmp` and
was corrected to its resolved real path, preserving strict input-symlink
rejection in the operator. No full suite or real model inference was run.

[Code/Review] The independent H4/math sibling reviewed the operator and tests,
confirmed canonical ModelNet frame/preprocessing semantics, fixed224-key
coverage and no LION checkpoint/label/clean loading, and independently reran
all three focused tests successfully in the persistent CPU environment. It
reported no blocker. Parent must still execute/validate the actual GPU readout
and archive its raw logits before any external-label score analysis.

## Canonical source promotion after completed readouts — 2026-11-06

[Code/CPU] After the actual frozen readout/CPU analysis, the exact executed
24,768-byte operator (SHA2a16c543ec4bc259060276d766e921c68008813ad01f67388c838c8d55009b91)
was copied unchanged to `scripts/classify_shared_probe_readouts.py`.
`tests/test_shared_probe_readouts.py` now imports that canonical tracked source,
so integration does not depend on an ignored operator directory. All three
focused tests pass. External runtime provenance remains the original operator
path/bytes; active ASSETS and sealed source8095 are unchanged.
