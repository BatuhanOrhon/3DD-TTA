# r4 development acceptance/freeze — TEMPLATE ONLY

No actual development archive or winners were created or inspected in this preparation. No CLI/GPU/runtime/Git/worktree writes occurred.

After genuine completed32-job local analysis accepts all152 raw archives and all30 physical prepared caches plus the calibration cache proof, package these root members in a compact ZIP:

- `accepted_sensitivity_development.json`
- `validation.json`
- `sensitivity_confirmation_freeze.json`
- `post_development_budget_projection.json`
- `raw_archives.json`
- `cache_validation.json`
- `calibration_cache_validation.json`

CSV/other compact derived evidence can also be included. Required members must be at ZIP root. Do not include synthetic fixtures as experimental evidence.

Compute post-development projection from the actual final r4 ledger. Required fields match the smoke projection plus:

```
status: whole_campaign_reestimate_fits_approved_cap
phase_plan_sha256: 1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42
spent_seconds: exact final ledger spent_seconds
cap_seconds: 19800
remaining_basis: registered_maximum_24_jobs
conservative_max_ratio: at least both prior smoke ratio and1.025455024723712
original_future_budget_seconds.confirmation_per_dataset_seed: 300
original_allowance_increase_seconds: 180
remaining_point_seconds: at least7242.365194455003 * conservative_max_ratio +180
margin_multiplier: 1.2
packing_retry_reserve_seconds: 1200
upper_seconds: spent_seconds +1.2 * remaining_point_seconds +1200, no more than19800
```

The registered maximum24-job allowance remains even when the freeze deduplicates to3 conditions. No additional estimation framework is needed.

Bind the actual ZIP only after its genuine local acceptance:

```
python bind_acceptance.py --package ACTUAL_LOCAL_ACCEPTANCE.zip --output FRESH_BOUND_PHASE.py
```

Upload the same ZIP to `/content/agent_b_r4_development_local_acceptance.zip`; submit the bound phase through sole monitor5314. Exact completion marker: `AGENT_B_SENSITIVITY_DEVELOPMENT_ACCEPTANCE_COMPLETE`.

The phase pins current controller/plan/source, verifies inactive completed development and matching remote completion archive, checks all19 registry-ordered scores and winner/tie policy,128 new+24 reused archive identities,30+30 identical locally verified cache records, and actual ROOT/Drive cache bytes. It checks `controller.frozen_conditions` against candidate files in a temporary owned directory BEFORE installation, which also verifies the physical raw Drive archive. It repeats that construction check after installing the same bytes. Acceptance/freeze targets are never overwritten. One fresh immutable Drive copy of the compact acceptance ZIP and an identified receipt are retained.

Confirmation cache namespace preexistence is deliberately left to the unchanged controller. Before launching any confirmation worker, it requires both local and Drive `paired_sensitivity_confirmation_v2_seed0`, seed1 and seed2 namespaces to be absent. The acceptance phase does not delete or bypass these namespaces.

Three clearly synthetic CPU tests pass: complete registry/raw/cache/identity fixture; underestimated future/stale spent/reduced original allowance rejection; missing physical cache/freeze identity mismatch rejection. Python3.8 AST and whitespace pass. Real completed development ingestion, remote execution and genuine archive binding remain pending.
