# Remaining registered slots — 2026-10-28

[Code/CPU/Open] Review-ready local preparation only. No runtime action, GPU work,
repository mutation, or new scientific result by this child. Root must review
these sources before deploying or binding a genuine preservation receipt.

Scope: only registered slots21 (tuned C/Scan/seed2),22 (original/ModelNet/seed2),
23 (original/Scan/seed2). Slot20 stays failed and is never retried. The original
plan/controller/failed receipt/archive/ledger history are preserved. No scientific
parameters, selected policies, IDs, seeds, native code or source assets change.

Files:
- `remaining_slots_controller.py`: fail-closed gates, fresh CPU source/native/
  data/checkpoint/12-head strict-load proof, original24-command reconstruction,
  cumulative watchdog/budget, strict successful worker validation, exact known
  finite-XYZ failure recording, separate immutable outcome archive.
- `run_remaining_slots.py`: explicit reviewed-plan SHA argument and Conda
  `3dd_tta_env` wrapper. Success marker:
  `AGENT_B_REVIEWED_REMAINING_SLOTS_RECORDED`.
- `prepare_reviewed_recovery.py`: local plan binder; requires genuine downloaded
  preservation receipt before writing a new exclusive JSON plan.
- `test_remaining_slots.py`:7 bounded CPU fixtures, including the actual frozen
  grid/failed receipt/ledger pins and original finite-XYZ log.

After genuine preservation receipt is downloaded, root can prepare the plan:

```
PYTHONDONTWRITEBYTECODE=1 /private/tmp/agent_b_numpy1_analysis_env/bin/python \
 /private/tmp/agent_b_r4_remaining_slots/prepare_reviewed_recovery.py \
 --preservation-receipt GENUINE_LOCAL_PRESERVATION_RECEIPT.json \
 --output FRESH_REVIEWED_RECOVERY_PLAN.json
```

Review the exact emitted plan/hash and source identities. Deploy the two source
files plus plan under
`/content/studies/guidance-components/reviewed_remaining_slots/`, with plan named
`reviewed_recovery_plan.json`. Wrapper requires `--recovery-plan-sha256 REVIEWED_SHA`.
The binder does not launch or deploy anything.

The preservation receipt is required at
`ROOT/sensitivity_confirmation_failure_preservation_receipt.json` and must bind
the actual failed receipt5ab888..., post-seal ledger60c8..., raw archivef586...,
source log, physical180-cache proof and verified preserved scope. It does not
assert acceptance of failed slot20 or validation of its final model state.

Budget point:291.4872319210008+300+300=891.4872319210008s. Initial upper:
14114.660198396998+1.2*891.4872319210008+1200=16384.444876702197s.
Every subsequent worker rechecks actual cumulative elapsed time and remaining
point estimate; watchdog retains1200s reserve. Pre-GPU CPU preparation is outside
the existing active research-phase timing, as in the registered campaign.

Only a worker's exact `ValueError: Classifier input contains nonfinite XYZ
coordinates` traceback at pinned classifier_backbones.py:48 can become
`failed_numerical` and permit the next unrun slot. Generic worker error text alone
does not qualify. Unknown failures, identity/native/state/RNG mismatches or cap
failures stop and seal a failed-preserved outcome. Partial raw predictions are
never accuracy-accepted; original failed configs remain `failed` in raw bytes.

Remaining evidence archive contains `reviewed_recovery_plan.json` (source pins
under `sources`), both recovery sources, original failed receipt/post-seal ledger,
preservation receipt, current scientific plan/approval/freeze/accepted development,
both pinned phase sources, original24 commands, selected3 commands, fresh CPU
preflight, per-slot logs/validation/raw files, state and cumulative ledger snapshot.

State: `stage=confirmation_remaining`; if all three slots are recorded as either
`complete` or `failed_numerical`, `status=remaining_slots_recorded`. It never claims
whole confirmation complete. Includes source/session/plan headers,
`original_confirmation_archive`, `original_receipt_identity`,
`reviewed_recovery_plan_sha256`, `registered_jobs=[21,22,23]`,
`excluded_failed_slots=[20]`, `jobs`, `cache_records`, `elapsed_seconds`, and
`full_benchmark=false`. Each job retains its original job dictionary with only
result-root command/path changes, plus `registered_job_index`. Complete records
retain pinned `c.validate_job` fields. Numerical records include partial four-head
ZIP identities, frozen model contracts, exact failure/log identity, and
`accuracy_accepted=false`.

The appended cumulative attempt uses stage `confirmation_remaining` and status
`complete` only for the remaining phase when all three outcomes are recorded.
The entire previous attempts prefix remains exactly equal, including the failed
original confirmation attempt. Final `ROOT/sensitivity_confirmation_remaining_receipt.json`
is exclusive, preserves state plus archive identity and final post-seal spend;
same receipt is written under a new UTC Drive directory. The scientific analysis
must remain failure-aware even if all remaining jobs complete.

Verification:6 core fixtures observed RED before implementation, then GREEN;
the actual-local-evidence seventh fixture passes. Latest7 tests pass0.233s;
all four Python files parse with Python3.8 AST. No full repository test suite is
applicable to isolated files under `/private/tmp`; no source package was changed.

[Code/CPU] Independent review identified the required native import hook before
Point-MAE's eager knn_cuda import. Fixed by verifying/loading the pinned bootstrap
asset and requiring no-build mode, exact audit bindings and CUDA still uninitialized
before any classifier load. The complete native report is archived in fresh CPU
preflight. The focused regression was observed RED, then all8 fixtures passed0.225s;
Python3.8 AST still passes. Current controller25282bytes SHA256
`eb44e15a22e1ccbb6e554f1423df3d1bf41201773f77e495ae308709eac0b09d`.
Wrapper and plan preparer are unchanged. This supersedes the earlier7-test status.

Unverified: actual runtime CPU/native imports and strict loads, preservation
receipt binding and GPU/Drive integration. Root source review and the genuine
preflight gates are required before launch. Knowledge impact: archive this draft
and append the bounded failure/continuation decision to the existing sensitivity
record; this child was explicitly forbidden from editing repository records.
