# Fixed shape-latent pilot evidence — 2026-10-31

[Run/Code] Compact companions to [accepted results](../../shape_latent_tta_results_20261031.md)
and [execution history](../../shape_latent_tta_study_20261031.md). Pilot is
64 exposed original objects ×15 fixed corruptions ×seed0, frozen Point-MAE;
not960independent objects or confirmation. All five arms retained, no gain
versus reference. Technical smoke32 overlaps pilot64.

- `registration.json` records scientific source/phase cap; versioned
  `operator_registration_v3.json` supersedes only the initial administrative
  scope guard. Actual commands/IDs and completed phase status are retained.
- `setup_verification.json` and `cpu_deployment_receipt.json` are downloaded
  actual receipts, not plans. Only the known cell2 fastrlock build failure
  was deferred. Fresh native checks passed; deployed pinned native bindings
  are separately identified in the receipts and raw deployment archive.
- `pilot_acceptance.json` / `smoke_acceptance.json` are independent local
  recounts exactly equal to the remote acceptance JSON. Outcomes are binary
  arm×corruption×original-object matrices, derived from raw trajectory logits.
- `pilot_summary.json` and `pilot_per_corruption.csv` give paired contrasts,
  object-bootstrap intervals, drift/gradient/correction metrics and timing/
  memory diagnostics. Same global noise is verified across30 paired batches.
- `administrative_stop_receipt.json` preserves the first controller failure;
  no model failure or numerical baseline rerun is hidden. The raw failed ZIP
  and explicit successful-baseline reuse provenance remain available.
- `cpu_acceptance.json`, `review_acceptance.json` and
  `guidance_mutation_proof.json` record test-first/review/mutation evidence.
- `artifact_manifest.json` gives SHA/bytes/locations for raw source, failed,
  complete and deployment archives; large raw ZIPs/caches/weights stay outside
  Git. `evidence_manifest.json` hashes these compact files, excluding itself.

To recount without touching extracted raw files, use NumPy1 and the repository
source (no GPU or model forward). Extract the verified pilot ZIP safely to
`PILOT`; choose fresh derived directories:

```bash
PYTHONPATH=. python knowledge/evidence/shape_latent_tta_study_20261031/analyze_phase_local.py PILOT PILOT/paired DERIVED
python knowledge/evidence/shape_latent_tta_study_20261031/summarize_pilot.py PILOT DERIVED SUMMARY
```

The same raw analyzer accepts complete technical smoke scope, preserving the
worker's valid `partial` status for that explicitly registered two-corruption
request. Scientific GPU source belongs to a82200ad + sealed dirty snapshot,
not the later publication checkout. Integration CPU checks do not retroactively
change which source ran on A100. Runtime remains owned/live until explicit
task-complete+close permission.
