"""Bounded CPU fixtures; no runtime or research execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

HERE = Path(__file__).parent


class RemainingTests(unittest.TestCase):
    def module(self):
        path = HERE / 'remaining_slots_controller.py'
        self.assertTrue(path.is_file(), 'Remaining-slots implementation is missing')
        spec = importlib.util.spec_from_file_location('remaining_test_subject', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def jobs(self):
        rows = []
        for index in range(24):
            root = '/old/slot%d' % index
            rows.append(dict(condition_id='slot%d' % index, dataset='dataset', seed=2,
                             result_root=root, command=['python', '--result-root', root, '--eta', '.03']))
        failed = dict(status='failed', jobs=[dict(row, status='complete' if i < 20 else 'running')
                                            for i, row in enumerate(rows[:21])])
        return rows, failed

    def test_only_unrun21through23_and_only_output_root_changes(self):
        m = self.module(); jobs, failed = self.jobs(); before = copy.deepcopy(jobs)
        selected = m.select_remaining(jobs, copy.deepcopy(jobs), failed, Path('/new'))
        self.assertEqual([row['registered_job_index'] for row in selected], [21, 22, 23])
        for row in selected:
            original = jobs[row['registered_job_index']]
            self.assertEqual(m.without_result_root(row['command']), m.without_result_root(original['command']))
        self.assertEqual(jobs, before)

    def test_changed_archived_command_or_replacing20_rejected(self):
        m = self.module(); jobs, failed = self.jobs(); archived = copy.deepcopy(jobs)
        archived[21]['command'][-1] = '.01'
        with self.assertRaises(ValueError): m.select_remaining(jobs, archived, failed, Path('/new'))
        failed['jobs'][20]['status'] = 'complete'
        with self.assertRaises(ValueError): m.select_remaining(jobs, jobs, failed, Path('/new'))

    def test_budget_includes_prior_spend_margin_and_reserve(self):
        m = self.module()
        upper = m.projected_budget(14114.660198396998, 891.4872319210008)
        self.assertAlmostEqual(upper, 16384.444876702197)
        for spent in (19100., float('nan'), -1.):
            with self.assertRaises(ValueError): m.projected_budget(spent, 900.)

    def test_failed_ledger_history_is_retained_and_remaining_never_repeated(self):
        m = self.module()
        ledger = dict(active=None, spent_seconds=m.SPENT_FLOOR, cap_seconds=19800,
                      attempts=[dict(stage='confirmation', status='failed', output='/old')])
        original = copy.deepcopy(ledger)
        m.validate_ledger_history(ledger, '/old')
        self.assertEqual(ledger, original)
        ledger['attempts'].append(dict(stage='confirmation_remaining', status='failed'))
        with self.assertRaises(ValueError): m.validate_ledger_history(ledger, '/old')

    def test_only_exact_finite_xyz_worker_failure_is_recordable(self):
        m = self.module()
        text = ('Traceback (most recent call last):\n'
                '  File "/sealed/classifier_backbones.py", line 48, in forward\n'
                '    raise ValueError("Classifier input contains nonfinite XYZ coordinates")\n'
                'ValueError: Classifier input contains nonfinite XYZ coordinates\n')
        error = ValueError('Component worker failed; preserve this attempt')
        self.assertTrue(m.known_numerical_failure(text, error))
        for changed in (text.replace('nonfinite XYZ coordinates', 'nonfinite logits'),
                        text + 'RuntimeError: Source identity mismatch\n', text.replace('line 48', 'line 75')):
            self.assertFalse(m.known_numerical_failure(changed, error))
        self.assertFalse(m.known_numerical_failure(text, TimeoutError('watchdog')))

    def test_global_state_contract_change_stops(self):
        m = self.module()
        ledger = dict(lion_state={'vae': 'v'}, classifier_contracts={'dataset/head': {'state': 'h'}})
        result = dict(lion_state={'vae': 'v'}, classifier_contracts={'head': {'state': 'h'}})
        m.validate_global_contracts(result, 'dataset', ledger)
        result['classifier_contracts']['head']['state'] = 'changed'
        with self.assertRaises(ValueError): m.validate_global_contracts(result, 'dataset', ledger)

    def test_actual_failed_receipt_matches_frozen_grid_and_known_guard(self):
        m = self.module()
        prep = Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025')
        sys.path.insert(0, str(prep))
        import analyze_sensitivity_confirmation_local as analysis
        import analyze_sensitivity_local as shared
        recovery = prep / 'recovery_20261028_r4'; plan_path = recovery / 'controls/bundle/campaign_plan.json'
        context = analysis.load_context(prep, plan_path, shared.identity(plan_path)['sha256'])
        failed = shared.read_json(recovery / 'confirmation_failed_receipt.json')
        self.assertEqual(m.identity(recovery / 'confirmation_failed_receipt.json'), m.RECEIPT_ID)
        self.assertEqual(m.identity(recovery / 'confirmation_failed_ledger.json'), m.LEDGER_ID)
        with zipfile.ZipFile(prep / 'calibrate_evidence_drive_roundtrip.zip') as archive:
            calibration = json.loads(archive.read('calibration/calibration_report.json'))
        development = recovery / 'local_development_analysis_v1'
        freeze = shared.read_json(development / 'sensitivity_confirmation_freeze.json')
        accepted_path = development / 'accepted_sensitivity_development.json'
        conditions = analysis.frozen_conditions(freeze, shared.read_json(accepted_path),
                     shared.read_json(development / 'validation.json')['balanced_scores'], context[5], context[1], calibration,
                     shared.identity(accepted_path), shared.identity(plan_path)['sha256'])
        controller = context[1]; original = controller.frozen_conditions
        old_output = failed['archive']['local_path'][:-4]
        try:
            controller.frozen_conditions = lambda *args: conditions
            jobs = controller.jobs_for(context[0], context[5], 'confirmation', old_output, calibration, shared.identity(plan_path)['sha256'])
        finally:
            controller.frozen_conditions = original
        selected = m.select_remaining(jobs, copy.deepcopy(jobs), failed['state'], Path('/new/reviewed_output'))
        self.assertEqual([(j['registered_job_index'],j['dataset'],j['trial']['id']) for j in selected],
                         [(21,'scanobjectnn_c','c-step-high'),(22,'modelnet40_c','original3dd'),(23,'scanobjectnn_c','original3dd')])
        log = (recovery / 'confirmation_failed_job020.log').read_text()
        self.assertTrue(m.known_numerical_failure(log, ValueError('Component worker failed; preserve this attempt')))

    def test_native_bootstrap_requires_pinned_no_build_bindings_before_loads(self):
        m = self.module()
        self.assertTrue(callable(getattr(m, 'bootstrap_native_cpu', None)), 'Native import guard is missing')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); hook = root / 'native_bootstrap.py'
            entry = dict(path='/fixture/native.so', bytes=4, sha256='a' * 64)
            audit = dict(backends=dict(fixture=[entry]))
            torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
            def verify(path, expected):
                self.assertEqual(m.identity(path), expected)
            c = SimpleNamespace(ASSETS=root, verify=verify)
            for report, valid in [(dict(no_build=True, bindings=dict(fixture=entry)), True),
                                  (dict(no_build=False, bindings=dict(fixture=entry)), False),
                                  (dict(no_build=True, bindings={}), False)]:
                hook.write_text('def bootstrap(root):\n    return ' + repr(report) + '\n')
                manifest = dict(guidance=dict(native_bootstrap=hook.name), files={hook.name:m.identity(hook)})
                if valid:
                    self.assertEqual(m.bootstrap_native_cpu(c, manifest, audit, torch), report)
                else:
                    with self.assertRaises(ValueError):m.bootstrap_native_cpu(c, manifest, audit, torch)


if __name__ == '__main__':
    unittest.main()
