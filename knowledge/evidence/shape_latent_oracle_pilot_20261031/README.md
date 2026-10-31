# Clean-z oracle evidence — 2026-10-31

[Run/Code] [Accepted result](../../shape_latent_oracle_results_20261031.md),
[pre-launch knowledge and history](../../shape_latent_oracle_pilot_20261031.md),
[execution plan/rulings](../../shape_latent_oracle_execution_plan_20261031.md).
64 exposed original objects ×15 fixed corruptions ×seed0, frozen Point-MAE.
Clean data is oracle information; no deployed-TTA or full benchmark claim.

Four arms: baseline; clean-z/exact old h; clean-z/conditional h from corrupt X;
clean pair. Global diffusion is OFF, common posterior/diffusion draws and
post-preparation RNG, fixed baseline h XYZ Chamfer anchor. Independent input
normalization retained; D changes the whole local observation and remains
constrained by the corrupt anchor, so is not a formal upper bound.

- `asset_identity_v1.json`, `registration.json`, commands/IDs and phase receipts
  bind exact source, scope and archives. `artifact_manifest.json` lists raw
  archive SHA/bytes/locations and five exact scientific source/test files.
- Actual setup and CPU deployment receipts are downloaded proofs, not plans.
  Fresh native bytes were preserved before restoring four pinned bindings.
- Independent local `pilot_acceptance.json` / `smoke_acceptance.json` equal
  remote acceptance exactly. Physical posterior equations, branch/input/anchor
  hashes, cache RNG, source/frozen-state guards and PM logits are checked.
- `summary.json`, outcomes and CSV preserve all arm results and conditional
  original-object bootstrap intervals; no method/parameter selection.
- `diagnostics_summary.json` gives descriptive latent changes and instrumented
  timing/memory. A precomputes all oracle posteriors; other arms replay, so
  these timings cannot establish deployment speedup. Clean-h pointwise L2
  includes ordering/resampling changes, not just geometric error.
- All960 clean-z samples lie outside the previous5% ball; this is an encoder/
  preprocessing sample difference, not an optimal correction norm.
- CPU/review/integration receipts preserve RED→GREEN noise-alias/common-anchor/
  metadata/physical-input checks. Operator cluster-bootstrap selfcheck changes
  all15 conditions of one object and resamples objects jointly.

Use NumPy1 from the repository root; safely extract the SHA-verified pilot ZIP
to `PILOT`, with fresh output directories to avoid rewriting raw extraction:

```bash
PYTHONPATH=. python knowledge/evidence/shape_latent_oracle_pilot_20261031/analyze_phase_local.py PILOT PILOT/paired DERIVED
python knowledge/evidence/shape_latent_oracle_pilot_20261031/summarize_results.py DERIVED SUMMARY
python knowledge/evidence/shape_latent_oracle_pilot_20261031/collect_diagnostics.py PILOT SUMMARY
```

Large raw/caches/weights stay outside Git, preserved locally and on Drive.
Exact operator/source capsules are in the validated local derived ZIP; its
extra Drive copy is pending after runtime loss and browser permission denial.
OAuth URLs and credential/session-state files are excluded. Latest integration
CPU verification is distinct from the338-file source actually executed on GPU.
