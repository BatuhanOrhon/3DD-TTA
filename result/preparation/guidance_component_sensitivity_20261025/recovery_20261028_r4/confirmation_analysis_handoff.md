# Agent B local confirmation analysis — 2026-10-28

[Code/CPU/Open] Preparation only. No new confirmation predictions or labels were inspected. No CLI, runtime, GPU, original worktree or Git mutation by this child.

Files:
- `analyze_sensitivity_confirmation_local.py`: prospective frozen140-ID,3-seed local analysis, explicit migrated plan hash; original calibration is independently verified under its original plan/session.
- `analyze_sensitivity_local.py`: original local helper plus minimal seed and recovery support. Full updated file and combined `shared_confirmation_support.patch` are supplied. Use the combined patch, not the earlier seed-only patch.
- `test_confirmation_analysis.py`:9CPU fixtures/integration checks.
- `source_identities.json`: exact staged byte identities.

Shared smoke/development CLI adds optional `--plan PATH --plan-sha256 SHA`; keep `--prep` pointing to original sensitivity preparation directory. It validates current smoke/dev phase against the migrated plan. `unpack_calibration` separately validates original calibration archive under original plan4f633535..., checks immutable archive/report/independent-acceptance pins from `plan.recovery`, and restores the current plan hash even on failure. Existing coefficient/parity checks and30physical old-cache acceptance gates remain. Old default CLI remains unchanged. Raw inspection now binds actual registered seed0/1/2 and the seed's cache namespace; cross-condition preparation comparisons include seed in the key.

Confirmation CLI:

```
python analyze_sensitivity_confirmation_local.py \
 --prep NEW_PREP --old-prep OLD_PREP \
 --plan MIGRATED_BUNDLE/campaign_plan.json --plan-sha256 REVIEWED_SHA \
 --phase-archive CONFIRMATION_DRIVE_ROUNDTRIP.zip --receipt CONFIRMATION_RECEIPT.json \
 --calibration-archive NEW_PREP/calibrate_evidence_drive_roundtrip.zip \
 --calibration-receipt NEW_PREP/calibrate_receipt.json \
 --development-archive DEVELOPMENT_DRIVE_ROUNDTRIP.zip --development-receipt DEVELOPMENT_RECEIPT.json \
 --development-acceptance DEVELOPMENT_ANALYSIS/accepted_sensitivity_development.json \
 --freeze DEVELOPMENT_ANALYSIS/sensitivity_confirmation_freeze.json \
 --modelnet-labels NEW_PREP/local_dataset_labels/modelnet40_c/label.npy \
 --cache-root CONFIRMATION_CACHE_EXTRACTION_ROOT \
 --output FRESH_ANALYSIS_OUTPUT
```

The ModelNet label file is a required additional physical input: remote `/content/3DD-TTA/data/modelnet40_c/label.npy`,2596bytes SHA256`97b6e660820103074ab192f10b6d9c33aa0446f1547ddb6ad2ca35a029a63b31`. Parent will download it aftercell9. Scan label file is already sealed in assets. New140 IDs cannot borrow labels from old70-ID raw shards. Both full label files are rehashed and selected labels joined to every new raw result.

The frozen development acceptance must bind its own adjacent `validation.json`, its physically downloaded development phase and exact selected rankings. Archived confirmation freeze/acceptance must match these bytes. Conditions are reconstructed without reselection, with exact matched-C deduplication. Current phase headers, approval, cumulative ledger/spent floor and pinned controller/calibration sources verify. Actual raw conditions/jobs/commands/IDs/classes/native/checkpoints/states/logits/CSV/trajectory purity are revalidated using existing strict helper. Original5B32/140 terminal is distinct from GSD2B70/probability mean. LION/classifier contracts must also equal cumulative campaign ledger.

Bootstrap uses one2000x140 matrix per dataset, jointly reused across all3seeds/heads/corruptions/methods/contrasts:280original-object clusters. All15/macro14, seed ranges, eight-case weighting, corrected/harmed events, per-corruption signs and diagnostic readouts are saved. Original contrast is labelled a different-host whole-policy comparison; S-only vsC replaces objective. There is no equivalence/universal-superiority or population-over-seeds inference.

All180physical confirmation NPZs are mandatory for accepted status, including exact receipt bytes, latent/noise/input/RNG metadata binding and no mixed-seed pairing. Missing physical files yield `validated_pending_physical_cache` and no accepted claim. Canonical raw ZIP copies preserve original bytes with collision guards. Shared reconstruction is counted once; classifier costs sum; peak memory uses maximum. Replay/instrumented comparisons do not establish deployment latency.

Verification:

```
PYTHONDONTWRITEBYTECODE=1 CACHED_PYTHON test_confirmation_analysis.py --prep NEW_PREP --migration-plan MIGRATED_BUNDLE/campaign_plan.json -v
```

9tests pass0.179s. Checks cover actual migratedplan14033563...+original calibration archive ingestion, retained old provenance and failure restoration; actual140-ID original five-B32 seed2/remainder with wrongseed rejection; frozen3-condition dedup/tampering; joint bootstrap seed dependence/cancellation; balanced datasets/macro14; sealed labels/tampering; missing180cache receipts; shared cost counting. Existing5local-analyzer fixtures pass0.039s on patched helper and original self-check passes. Python3.8 AST and whitespace checks pass.

Actual smoke/development/confirmation stage ingestion remains untested because those artifacts do not exist. The full confirmation acceptance path and physical180cache RNG checks must be exercised when genuine outputs arrive. This is the material remaining integration risk. Parent should append this preparation/check record to the existing sensitivity knowledge note; this staged handoff is the child's knowledge update draft.


## Independent review corrections — 2026-10-28

Use `/private/tmp/agent_b_numpy1_analysis_env/bin/python` for actual confirmation
analysis (Python3.10.18/NumPy1.21.6). Remote caches use NumPy1 pickle module
paths; NumPy2.5.4 produced0/30 matching hashes on genuine old caches. No hash
bytes are normalized or metadata checks weakened. Compatible environment
reproduces30/30 exactly; helper rejects unsupported NumPy2 explicitly.
`analysis_environment.json` records recreation commands.
Classifier ledger comparison now uses producer's`resolved_model_config` field.
Both defects were reproduced with failing checks before correction. Root11
CPUtests pass under compatible environment(.292s); NumPy2 run(.168s) verifies
unsupported-version error behavior. Actual new-stage ingestion remains pending.
