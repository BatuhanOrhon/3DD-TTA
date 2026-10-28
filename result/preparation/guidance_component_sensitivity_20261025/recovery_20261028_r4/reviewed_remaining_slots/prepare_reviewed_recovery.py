"""Prepare the reviewable source/preservation binding locally; never launch work."""
import argparse
import json
from pathlib import Path

import remaining_slots_controller as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preservation-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    proof = json.loads(args.preservation_receipt.read_text())
    runner.require(proof['status'] == 'verified_failure_preserved' and proof['phase_plan_sha256'] == runner.PLAN_SHA
                   and proof['original_receipt_identity'] == runner.RECEIPT_ID and proof['failed_ledger_identity'] == runner.LEDGER_ID
                   and proof['failed_job_index'] == 20 and proof['numeric_failure_signature'] == runner.NUMERIC_SIGNATURE
                   and proof['physical_cache_count'] == 180 and proof['physical_cache_complete'] is True
                   and proof['integrity_verification']['status'] == 'verified_preserved_scope'
                   and proof['local_and_drive_original_archive_verified'] is True and proof['mismatch_detected'] is False,
                   'Require genuine reviewed failure-preservation evidence first')
    here = Path(__file__).resolve().parent
    plan = dict(status='prepared_for_explicit_source_review_not_launched', phase_plan_sha256=runner.PLAN_SHA,
                session=proof['session'], manifest_sha256=proof['manifest_sha256'], source_snapshot_sha256=proof['source_snapshot_sha256'],
                registered_jobs=[21,22,23], excluded_failed_slots=[20], original_receipt_identity=runner.RECEIPT_ID,
                failed_ledger_identity=runner.LEDGER_ID, original_archive_identity=runner.ARCHIVE_ID,
                original_confirmation_archive=proof['original_confirmation_archive'],
                failure_preservation_receipt=runner.identity(args.preservation_receipt),
                sources={name: runner.identity(here / name) for name in ('remaining_slots_controller.py','run_remaining_slots.py')},
                point_seconds=list(runner.POINT_SECONDS), prior_spent_seconds=runner.SPENT_FLOOR,
                margin_multiplier=1.2, packing_retry_reserve_seconds=1200, cap_seconds=19800,
                projected_upper_seconds=runner.projected_budget(runner.SPENT_FLOOR, sum(runner.POINT_SECONDS)),
                failure_policy='No retry/replacement of20. Only exact finiteXYZ worker failure is recorded; all other errors stop. Preserve every original attempt and no whole-confirmation-complete claim.',
                deployment_directory='/content/studies/guidance-components/reviewed_remaining_slots')
    runner.write_json(args.output, plan)
    print(json.dumps(dict(reviewed_recovery_plan=runner.identity(args.output), sources=plan['sources']), sort_keys=True))


if __name__ == '__main__':
    main()
