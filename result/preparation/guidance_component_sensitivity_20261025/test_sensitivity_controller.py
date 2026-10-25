"""CPU integration contracts against the actual declared sensitivity registry."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import sensitivity_campaign_controller as controller

REGISTRY = None


class ControllerContracts(unittest.TestCase):
    def setUp(self):
        self.registry = controller.registry_verified(copy.deepcopy(REGISTRY))
        self.calibration = dict(status='verified_label_free_gpu', coefficients=dict(legacy_l2=6000., huber=12000., power=.3))
        self.plan = dict(asset_root=str(controller.ASSETS), manifest_sha256='a' * 64, original_trial_id='original3dd',
                         smoke_trial_ids=['c-base', 'cs-base', 's-base', 'cs-l2-r010', 'cs-huber-r010', 'cs-power-r010', 'original3dd'],
                         id_manifests={stage: {dataset: dict(path=str(controller.ASSETS / 'study' / (dataset + '_' + stage + '.json')))
                                               for dataset in controller.DATASETS} for stage in ('smoke', 'development', 'confirmation')})

    def test_declared_family_counts_and_no_reused_accuracy_jobs(self):
        jobs = controller.jobs_for(self.plan, self.registry, 'development', controller.ROOT / 'test-output', self.calibration, 'a' * 64)
        self.assertEqual(len(jobs), 32)
        self.assertEqual(sum(job['trial']['method'] == 'gsd' for job in jobs), 30)
        self.assertEqual(sum(job['trial']['method'] == 'original3dd' for job in jobs), 2)
        self.assertFalse(any(job['trial']['reuse'] for job in jobs))
        self.assertEqual(len({(job['condition_id'], job['dataset'], job['seed']) for job in jobs}), 32)
        for family in ('chamfer', 'combined', 'spectral'):
            self.assertEqual(sum(job['trial']['family'] == family for job in jobs), 10)

    def test_smoke_cache_and_protocols_are_separate_from_all15_cache(self):
        jobs = controller.jobs_for(self.plan, self.registry, 'smoke', controller.ROOT / 'test-smoke', self.calibration, 'a' * 64)
        self.assertEqual(len(jobs), 14)
        for job in jobs:
            command = job['command']
            self.assertEqual(command[command.index('--corruptions') + 1:command.index('--corruptions') + 3], ['gaussian', 'background'])
            if job['trial']['method'] == 'gsd':
                self.assertEqual(command[command.index('--paired-state-root') + 1], str(controller.ROOT / 'paired_sensitivity_smoke_v2_seed0'))
                self.assertNotEqual(command[command.index('--paired-state-root') + 1], str(controller.ROOT / 'paired_state_v1_seed0'))
            else:
                self.assertEqual(command[command.index('--batch-size') + 1], '32')
                self.assertNotIn('--trajectory-observe', command)
                self.assertNotIn('--component-arm', command)
                self.assertEqual(controller.expected_protocol(job['trial'], self.calibration), dict(eta=.01, gamma=.01, p=.95, total=100, scd_normalize=False))

    def test_calibrated_weights_and_baseline_default_flags(self):
        self.assertEqual(controller.spectral_weight(self.registry['cs-l2-r010'], self.calibration), 600.)
        self.assertEqual(controller.spectral_weight(self.registry['s-huber-r010'], self.calibration), 1200.)
        self.assertEqual(controller.spectral_weight(self.registry['cs-power-r010'], self.calibration), .03)
        for name in ('c-base', 'cs-base', 's-base'):
            command = controller.command_for(self.plan, self.registry[name], 'scanobjectnn_c', 'smoke', 0, controller.ROOT / 'test', self.calibration)
            for flag in ('--eta', '--gamma', '--p', '--spectral-loss-kind', '--spectral-weight'):
                self.assertNotIn(flag, command)
        invalid = copy.deepcopy(REGISTRY); invalid['trials'].pop()
        with self.assertRaises(ValueError):
            controller.registry_verified(invalid)
        for value in (True, -1, 3):
            with self.assertRaises(ValueError):
                controller.command_for(self.plan, self.registry['cs-base'], 'scanobjectnn_c', 'confirmation', value, controller.ROOT / 'test', self.calibration)

    def test_remaining_campaign_budget_keeps_future_stages_and_reserve(self):
        plan = dict(stage_expected_seconds=dict(calibrate=300., smoke=840., development=6714., confirmation=7224.), packing_retry_reserve_seconds=1200)
        self.assertLess(controller.campaign_budget_verified(plan, 'calibrate', 0.), controller.CAP)
        self.assertEqual(controller.campaign_budget_verified(plan, 'confirmation', 9000.), 9000. + 1.2 * 7224. + 1200.)
        for stage, spent in [('development', 5000.), ('smoke', 2000.), ('confirmation', 11000.)]:
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                controller.campaign_budget_verified(plan, stage, spent)
        bad = dict(plan, packing_retry_reserve_seconds=0)
        with self.assertRaises(ValueError):
            controller.campaign_budget_verified(bad, 'calibrate', 0.)

    def test_frozen_condition_construction_and_physical_acceptance(self):
        with tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent)) as temporary:
            root = Path(temporary); drive = root / 'drive'; drive.mkdir()
            archive = drive / 'accepted.zip'; archive.write_bytes(b'CPU acceptance fixture')
            accepted_path, freeze_path = root / 'accepted.json', root / 'freeze.json'
            plan = dict(self.plan, accepted_development_path=str(accepted_path), confirmation_freeze_path=str(freeze_path))
            for spectral, chamfer, fourth in [('cs-huber-r010', 'c-base', False), ('cs-l2-r010-step-low', 'c-p80', True)]:
                accepted = dict(status='accepted', phase_plan_sha256='a' * 64, selection_policy='probability_mean',
                                selected_spectral_trial_id=spectral, selected_chamfer_trial_id=chamfer,
                                archive=dict(controller.identity(archive), drive_path=str(archive)))
                accepted_path.write_text(json.dumps(accepted))
                conditions = [dict(id='spectral-candidate', role='spectral_candidate', trial_id=spectral),
                              dict(id='matched-chamfer', role='matched_chamfer', trial_id='matched-chamfer')]
                if fourth:
                    conditions.append(dict(id='tuned-chamfer', role='tuned_chamfer', trial_id=chamfer))
                conditions.append(dict(id='original3dd', role='original3dd', trial_id='original3dd'))
                freeze_path.write_text(json.dumps(dict(status='frozen', phase_plan_sha256='a' * 64,
                    development_acceptance=controller.identity(accepted_path), selected_spectral_trial_id=spectral,
                    selected_chamfer_trial_id=chamfer, conditions=conditions)))
                with mock.patch.object(controller, 'owned', side_effect=lambda path: Path(path)), mock.patch.object(controller, 'DRIVE', drive):
                    jobs = controller.jobs_for(plan, self.registry, 'confirmation', controller.ROOT / 'test-output', self.calibration, 'a' * 64)
                    self.assertEqual(len(jobs), 24 if fourth else 18)
                    self.assertEqual({job['seed'] for job in jobs}, {0, 1, 2})
                    matched = next(job['trial'] for job in jobs if job['role'] == 'matched_chamfer')
                    self.assertEqual((matched['eta'], matched['gamma'], matched['p']), tuple(self.registry[spectral][key] for key in ('eta', 'gamma', 'p')))
                    self.assertEqual(matched['chamfer_weight'], 1.)
                    self.assertEqual(matched['spectral_weight'], 0.)
                    self.assertIsNone(matched['spectral_loss_kind'])
                    bad = json.loads(freeze_path.read_text()); bad['conditions'][0]['trial_id'] = 'c-base'; freeze_path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        controller.frozen_conditions(plan, self.registry, self.calibration, 'a' * 64)
            archive.write_bytes(b'mutated acceptance fixture')
            with mock.patch.object(controller, 'DRIVE', drive), self.assertRaises(ValueError):
                controller.accepted_stage(accepted_path, 'a' * 64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    REGISTRY = json.loads(args.registry.read_text())
    unittest.main(argv=[__file__, *remaining])
