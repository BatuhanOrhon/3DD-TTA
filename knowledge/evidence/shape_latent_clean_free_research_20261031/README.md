# Clean-data-free shape latent research evidence — 2026-10-31

[Report and unimplemented candidates](../../shape_latent_clean_free_research_20261031.md).

- `analyze_posteriors.py`: NumPy-only descriptive analysis; synthetic self-check,
  all physical tensor hashes, posterior equations, finite/shape/coverage checks.
  It refuses to overwrite a derived output directory.
- `posterior_decomposition.json`: actual result, 30 proof identities and15
  per-corruption summaries. All15 diagonal-Gaussian W2 squared mean-shift share
  97.8677479611%; no new accuracy or model inference. Realized sample difference
  has a cross term; this is not explained accuracy or calibrated corruption error.
- `primary_source_receipts.json`: seven fixed-version primary HTML identities,
  fetched2026-10-31. Full HTML/text retained outside Git under
  `result/preparation/shape_latent_clean_free_research_20261031/primary_sources/`.
- `search_ledger.md`: scope and sections inspected, including negative fit.
- `validation.json`: actual CPU and document/evidence checks.

Run from repository root with NumPy1 and a fresh output path:

```bash
python knowledge/evidence/shape_latent_clean_free_research_20261031/analyze_posteriors.py PILOT_EXTRACTED FRESH_DERIVED
```

Input is the accepted oracle pilot's unmodified extraction; raw ZIP SHA256
`52bffe3f220bc76c3e9cdd486f033f954148b240515d3f33ad659795aa9ce033`.
64 original objects ×15 corruptions, seed0; events are clustered by original
object, not960 independent objects. Oracle inputs only support diagnosis.
No method parameters were fitted to labels or clean targets in this task.

No GPU/runtime/Drive action, production algorithm, checkpoint change,
new dependency or extra full benchmark is part of this evidence package.
