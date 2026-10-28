import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import prepare_recovery as recovery

PREP = Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025')
ENDPOINT = 'gpu-a100-hm-euw4a0-3s9zrjduwd03k'

class RecoveryTests(unittest.TestCase):
    def test_migration_changes_ownership_only_and_retains_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'generated'
            meta = recovery.prepare(PREP, out, ENDPOINT)
            old = json.loads((PREP / 'campaign_plan.json').read_text())
            plan = json.loads((out / 'bundle/campaign_plan.json').read_text())
            comparable = copy.deepcopy(plan)
            for name in ('session', 'endpoint', 'phase_sources', 'recovery'):
                comparable.pop(name)
                old.pop(name, None)
            self.assertEqual(comparable, old)
            ledger = json.loads((out / 'bundle/campaign_ledger.json').read_text())
            self.assertEqual(ledger['spent_seconds'], 155.86660684299568)
            self.assertEqual(ledger['cap_seconds'], 19800)
            self.assertEqual(ledger['attempts'][0]['stage'], 'calibrate')
            self.assertEqual(ledger['attempts'][0]['status'], 'complete')
            self.assertEqual(ledger['session'], recovery.SESSION)
            controller = out / 'bundle/sensitivity_campaign_controller.py'
            spec = importlib.util.spec_from_file_location('r4_controller_test', controller)
            c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
            self.assertNotEqual(c.LEDGER_NAME, 'sensitivity_campaign_ledger.json')
            c.ledger_valid(ledger, meta['plan_sha256'])
            self.assertLess(c.campaign_budget_verified(plan, 'smoke', ledger['spent_seconds']), 19800)
            report = PREP / 'calibrate_receipt.json'
            raw = json.loads(report.read_text())['state']['calibration']
            physical = out / 'raw_calibration_report.json'
            import zipfile
            with zipfile.ZipFile(PREP / 'calibrate_evidence_drive_roundtrip.zip') as archive:
                physical.write_bytes(archive.read('calibration/calibration_report.json'))
            self.assertEqual(c.calibration_verified(physical, plan), raw)
            physical.write_text(json.dumps(dict(raw, coefficients=dict(raw['coefficients'], power=99.))))
            with self.assertRaises(ValueError):
                c.calibration_verified(physical, plan)
            with self.assertRaises(ValueError):
                c.ledger_valid(dict(ledger, spent_seconds=0., recovery={}), meta['plan_sha256'])

    def test_foreign_endpoint_or_existing_destination_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'generated'
            with self.assertRaises(ValueError):
                recovery.prepare(PREP, out, 'cpu-foreign')
            out.mkdir()
            with self.assertRaises(FileExistsError):
                recovery.prepare(PREP, out, ENDPOINT)

if __name__ == '__main__':
    unittest.main()
