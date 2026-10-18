# Prepared multibackbone GPU campaign — 2026-10-18

[Open/User gate] Prepared scope only: not approved and no GPU inference launched.
The existing task authorizes CPU preparation. Whole conditional GPU campaign
exceeds60min and requires one comprehensive explicit approval before its first
GPU smoke. This gate comes from the user's task, AGENTS.md and the Colab
operator skill's “Approval for GPU runs longer than one hour” section.

## Fixed assets and runtime

- Branch/base: `gsd-multibackbone-study`,
  `2684f11922d77696a26c2b954dcd0c43de5f5a6c` plus explicit study overlay.
- Source snapshot ZIP SHA-256:
  `683924c4e8caad6459dfdf48caccfd545185501d3ea04257ae7f114028e09dd5`.
- Asset manifest SHA-256:
  `e6421333fdaab6fdd30167ea905f3d829d40de77f78fe91b696c035caa02a07a`.
- Preparation archive249654807 bytes/SHA-256:
  `7cb6db90868a4151a9096324ebdd95ae4d3816433281a58f778403ed1c55df2b`.
- Revision2 delta archive39060 bytes/SHA-256:
  `032b10d264a2ebbe8bef1a4e715802d39f9f70fd3ab2ec3038620e9998eb8f82`.
  Restore base preparation plus this manifest delta; the first CPU preflight
  rejected the inferred ModelNet label shape. Revision2 uses actual headers
  (`2468x1` labels), identical dataset bytes, unchanged model code and source ZIP.
- Exact six checkpoint identities and source revisions are in
  [the provenance JSON](gsd_multibackbone_provenance_20261018.json); published
  alternatives, never asserted to be exact GSDTTA weights.
- Existing session `tta-scanobjectnn-nobg-a100-r7-20261018`, endpoint
  `gpu-a100-hm-kkb-euw4a2-1d8qxj97k4u94`, A100-SXM4-80GB/High-RAM.
  Reuse Python3.8.20, torch2.1.2+cu121, CUDA12.1, GCC11.4 and existing native
  builds; do not rerun setup. Actual original remote checkout is dirtydev@a458cd4.
- Isolated execution root:
  `/content/gsd_multibackbone_study_20261018/prepared_v2/assets/source_snapshot`.
  The original runtime repository is preserved.
- Dataset protocol: severity5, canonical15, no shuffle, original file ordinals,
  ModelNet40-C2468 and ScanObjectNN-C `main_split`581; never nobg. New Scan
  archive is a published1024-point base with exact main_split membership;
  GSDTTA corruption-byte/seed identity remains unknown.
- Seed0 only; source batch32; classifier-specific1024-point preprocessing is
  locked in the source runner note. No test-label tuning, checkpoint selection
  by accuracy, or parameter sweep.

## Ordered execution and failure gates

1. Exact Point-MAE parity: two existing checkpoints, first32 original IDs each
   of Gaussian, Density and Background on each dataset. Same sampled tensor and
   restored Python/NumPy/Torch/CUDA RNG for direct versus wrapped calls. Require
   identical logits, predictions, RNG after-state, input and model-state hashes.
2. Six source smokes: each backbone/dataset, Gaussian-s5 first32 IDs, one batch.
   Require strict checkpoint load, finite logits, class count, frozen state,
   manifests and sealed/verified artifacts. Prefix accuracy is not benchmark.
3. Six source all15 evaluations: 3×15×(2468+581)=137205 predictions; report each
   corruption, macro and paper delta with checkpoint/preprocessing/split caveats.
   Stop before GSD until discrepancies are documented; do not search for
   preprocessing that maximizes corrupted-test accuracy.
4. Conditional implementation after source diagnosis: minimal frozen classifier
   wiring into the existing GSD-inspired latent spectral guidance path. Preserve
   Point-MAE parity and legacy spectral/diffusion equations. Capture the new
   implementation snapshot/hash before any guidance run; it does not yet exist
   because source-only acceptance is its prerequisite.
5. Conditional guidance smokes then matched all15 for validated backbones on both
   datasets, same fixed checkpoint/source classifier preprocessing and original
   IDs, seed0. Initial locked legacy settings: batch70, M_max800, beta2,
   spectral1.17, SCD1, eta/gamma.01, lambda.95, normal10/Background30 reverse
   steps, raw LION checkpoint, eval modes, final DDIM alphaFalse and updated
   decoder. ScanObjectNN framing/normalization must be explicit in its later
   implementation and smoke; do not silently reuse the old unsupported path.
   Use the same checkpoint's measured source row as primary comparator.

Stop on hash/split/count/strict-load mismatch, parity failure, nonfinite
outputs, classifier-state mutation, OOM or unexpected runtime loss. Preserve
failed artifacts before diagnosis. Short safe retries stay within the approved
scope/budget; material protocol changes require a revised scope. Never close
r7 automatically. Other assignments remain untouched.

## Concrete initial commands

Run under `conda run --no-capture-output -n 3dd_tta_env python -u`, with
`PYTHONDONTWRITEBYTECODE=1`; cwd is the isolated source snapshot above.
`A=/content/gsd_multibackbone_study_20261018/prepared_v2/assets`,
`H=e6421333fdaab6fdd30167ea905f3d829d40de77f78fe91b696c035caa02a07a`.
`R` is a fresh timestamped phase root, never a prior run directory.

```sh
python -u scripts/verify_multibackbone_pointmae_parity.py \
  --asset-root "$A" --asset-manifest-sha256 "$H" --result-root "$R" \
  --batch-size 32 --seed 0
python -u scripts/eval_multibackbone_source.py \
  --backbone dgcnn --dataset modelnet40_c \
  --dataset-root /content/3DD-TTA/data/modelnet40_c \
  --label-path /content/3DD-TTA/data/modelnet40_c/label.npy \
  --asset-root "$A" --asset-manifest-sha256 "$H" --result-root "$R" \
  --seed 0 --batch-size 32 --corruptions gaussian --max-batches 1
```

Repeat the second command for `dgcnn`, `curvenet`, `pointnext` and both datasets;
Scan data/labels root is `$A/datasets/scanobjectnn_c`. Full coverage replaces
`--corruptions gaussian --max-batches 1` with `--corruptions all --max-batches 0`.
Exact source script SHA-256:
`9b7802a80821c5cdf0f29c7eb37788311433e05b3ffb788eaa32b23e74e7362f`;
parity script SHA-256:
`761b8d78ec559a7976ab6039f894b5613190a01a3cbb0ebd910929c50c359389`.
Guidance commands/snapshot will be recorded only after the source gate; no
approval of a paper-reproduction claim is sought.

## Duration, outputs and recovery

[Run/Inference] One historical legacy all15 ModelNet seed0 archive reports
4138.711 seconds (~69min) for37020 predictions:
`result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/20261010T114156946419Z_handoff_20261010T141536060023Z.zip`,
inner `legacy_all15_seed0.zip`. Linear extrapolation to three backbones and
both datasets gives about4.3h of guidance before source/smokes/overhead. This
is an estimate: classifier, batch handling, graph and hardware differences
prevent treating it as a calibrated r7 timing. Allow roughly4–8 GPU hours for
the entire conditional campaign; request an8-hour GPU-work cap and stop/review
if projected remaining work exceeds the cap. Re-estimate from smokes without
silently increasing scope.

Latest observed aggregate CLI rate is6.93 compute units/hour, balance64.48
at preparation. Eight hours at that rate is55.44 units; this is aggregate
billing, includes all active assignments, and is not a guaranteed per-job price.
Fiat cost is unknown. The live assignment may consume units while awaiting
approval; do not close it without explicit task-complete plus close permission.

Every material phase uses fresh seven-file inference bundles plus per-object
logits/predictions, config/checkpoint/data/source hashes, exact source ZIP,
environment, frozen-state checks and a per-file manifest. Archive each run,
copy to `MyDrive/thesis/result/gsd_multibackbone_study/<phase-UTC>/`, verify
remote/Drive/local SHA-256 and CRC, then update findings/questions/study and
commit/push the accepted batch. Local raw artifacts stay under
`result/preparation/multibackbone_20261018/` or canonical dataset result paths.
On disconnect, inspect the existing process/markers/artifacts first; never
start a duplicate job or replay a completed native build. Preserve the session.
