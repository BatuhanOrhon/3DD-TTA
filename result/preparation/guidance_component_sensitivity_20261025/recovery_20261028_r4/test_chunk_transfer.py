import copy
import contextlib
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import split_stage_archive as split
import join_stage_archive as join


def fixture(root):
    payload=bytes(range(251))*17
    source=root/'source.zip';source.write_bytes(payload)
    parts=root/'parts';parts.mkdir()
    expected=split.identity(source)
    with contextlib.redirect_stdout(io.StringIO()):
        records=split.split_archive(source,parts,expected,257)
    stage='development'; remote=split.ROOT/'sensitivity_development_20261028T000000Z.zip'
    archived=dict(expected,local_path=str(remote),drive_path=str(split.DRIVE/'phase-evidence'/remote.stem/'sensitivity_development_evidence.zip'))
    phase=dict(state=dict(status='complete',stage=stage,session=split.SESSION,phase_plan_sha256=split.PLAN_SHA,manifest_sha256=split.MANIFEST_SHA,source_snapshot_sha256=split.SOURCE_SHA),archive=archived)
    receipt=json.dumps(phase).encode()
    plan=(Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025/recovery_20261028_r4/controls/bundle/campaign_plan.json')).read_bytes()
    transport=split.ROOT/'sensitivity_development_transport_fixture'
    manifest=dict(status='ready',schema_version=1,stage=stage,session=split.SESSION,phase_plan_sha256=split.PLAN_SHA,manifest_sha256=split.MANIFEST_SHA,source_snapshot_sha256=split.SOURCE_SHA,original_archive=archived,original_phase_receipt=split.bytes_identity(receipt),chunk_bytes=257,transport_root=str(transport),parts=[dict(row,remote_path=str(transport/row['name'])) for row in records],part_count=len(records))
    return payload,parts,manifest,receipt,plan


class ChunkTransferTests(unittest.TestCase):
    def test_exact_roundtrip_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();payload,parts,manifest,receipt,plan=fixture(root)
            output=root/'joined.zip'
            result=join.join_archive(manifest,receipt,plan,'development',parts,output)
            self.assertEqual(output.read_bytes(),payload)
            self.assertEqual(result['sha256'],hashlib.sha256(payload).hexdigest())
            with self.assertRaises(FileExistsError):join.join_archive(manifest,receipt,plan,'development',parts,output)

    def test_corrupt_or_missing_part_fails_with_partial_preserved(self):
        for missing in (False,True):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name).resolve();payload,parts,manifest,receipt,plan=fixture(root)
                target=parts/manifest['parts'][1]['name']
                if missing:target.unlink()
                else:target.write_bytes(b'bad')
                with self.assertRaises((ValueError,FileNotFoundError)):join.join_archive(manifest,receipt,plan,'development',parts,root/'partial.zip')
                self.assertTrue((root/'partial.zip').exists())

    def test_out_of_order_or_foreign_paths_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();payload,parts,manifest,receipt,plan=fixture(root)
            bad=copy.deepcopy(manifest);bad['parts'][0],bad['parts'][1]=bad['parts'][1],bad['parts'][0]
            with self.assertRaises(ValueError):join.join_archive(bad,receipt,plan,'development',parts,root/'bad.zip')
            for changes in [dict(name='../escape'),dict(remote_path=str(split.ROOT/'foreign.bin'))]:
                bad=copy.deepcopy(manifest);bad['parts'][0].update(changes)
                with self.assertRaises(ValueError):join.join_archive(bad,receipt,plan,'development',parts,root/'bad.zip')
            bad=copy.deepcopy(manifest);bad['original_archive']['drive_path']='/content/drive/MyDrive/other.zip'
            with self.assertRaises(ValueError):join.join_archive(bad,receipt,plan,'development',parts,root/'bad.zip')

if __name__=='__main__':unittest.main()
