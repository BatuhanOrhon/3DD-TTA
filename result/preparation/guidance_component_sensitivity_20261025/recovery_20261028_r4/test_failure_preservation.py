import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import failed_preservation_phase as failed
import join_failed_archive as join
import preserve_failed_cache_phase as cache
import split_stage_archive as split

R4=Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025/recovery_20261028_r4')


def fixture(root):
    receipt=json.loads((R4/'confirmation_failed_receipt.json').read_text())
    ledger_raw=(R4/'confirmation_failed_ledger.json').read_bytes()
    plan_raw=(R4/'controls/bundle/campaign_plan.json').read_bytes()
    payload=b'failed-attempt-artifact-only'*101
    source=root/'failed.zip';source.write_bytes(payload);parts=root/'parts';parts.mkdir()
    archive=dict(split.identity(source),local_path=receipt['archive']['local_path'],drive_path=receipt['archive']['drive_path'])
    receipt['archive']=archive;receipt_raw=json.dumps(receipt).encode()
    with contextlib.redirect_stdout(io.StringIO()):rows=split.split_archive(source,parts,archive,113)
    transport=failed.ROOT/'sensitivity_confirmation_failed_transport_fixture'
    manifest=dict(status='failed_attempt_transport_ready',experimental_result_status='failed',schema_version=1,session=failed.SESSION,stage='confirmation',phase_plan_sha256=failed.PLAN_SHA,original_archive=archive,original_failed_receipt=failed.bytes_identity(receipt_raw),original_failed_ledger=failed.bytes_identity(ledger_raw),transport_root=str(transport),chunk_bytes=113,part_count=len(rows),parts=[dict(row,remote_path=str(transport/row['name'])) for row in rows])
    return payload,parts,manifest,receipt_raw,ledger_raw,plan_raw


class FailurePreservationTests(unittest.TestCase):
    def test_failed_only_toy_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();payload,parts,manifest,receipt,ledger,plan=fixture(root)
            result=join.join_failed_archive(manifest,receipt,ledger,plan,parts,root/'preserved_failed.zip')
            self.assertEqual(result['status'],'failed_attempt_archive_bytes_preserved')
            self.assertEqual((root/'preserved_failed.zip').read_bytes(),payload)
            changed=json.loads(receipt);changed['state']['status']='complete'
            with self.assertRaises(ValueError):failed.validate_failed_state(changed,json.loads(ledger))

    def test_failed_manifest_never_accepts_ready_complete_label_or_foreign_path(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();_,parts,manifest,receipt,ledger,plan=fixture(root)
            for change in (dict(status='ready'),dict(experimental_result_status='accepted')):
                bad=copy.deepcopy(manifest);bad.update(change)
                with self.assertRaises(ValueError):join.join_failed_archive(bad,receipt,ledger,plan,parts,root/'bad.zip')
            bad=copy.deepcopy(manifest);bad['parts'][0]['name']='../escape'
            with self.assertRaises(ValueError):join.join_failed_archive(bad,receipt,ledger,plan,parts,root/'bad.zip')

    def test_real_failed_metadata_has180_consistent_cache_records(self):
        receipt=json.loads((R4/'confirmation_failed_receipt.json').read_text())
        plan=json.loads((R4/'controls/bundle/campaign_plan.json').read_text())
        prep=R4.parent
        ids={dataset:json.loads((prep/'assets_v2/study'/Path(row['path']).name).read_text())['indices'] for dataset,row in plan['id_manifests']['confirmation'].items()}
        records=cache.completed_cache_records(receipt['state'],ids)
        self.assertEqual(len(records),180)
        bad=copy.deepcopy(receipt['state']);bad['jobs'][0]['cache_records'][0]['sha256']='f'*64
        with self.assertRaises(ValueError):cache.completed_cache_records(bad,ids)

if __name__=='__main__':unittest.main()
