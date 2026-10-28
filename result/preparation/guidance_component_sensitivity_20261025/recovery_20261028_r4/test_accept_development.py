import copy
import json
from pathlib import Path
import unittest

import accept_development_template as phase

PREP=Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025')


def fixture():
    registry={row['id']:row for row in json.loads((PREP/'trial_registry.json').read_text())['trials']}
    plan=json.loads((PREP/'recovery_20261028_r4/controls/bundle/campaign_plan.json').read_text())
    scores=[dict(trial_id=name,family=row['family'],policy='terminal' if name=='original3dd' else 'probability_mean',n_cases=8,accuracy_pct=float(number)) for number,(name,row) in enumerate(registry.items())]
    validation=dict(status='accepted',stage='development',phase_plan_sha256=phase.PLAN_SHA,manifest_sha256=phase.MANIFEST_SHA,source_snapshot_sha256=phase.SOURCE_SHA,physical_cache_complete=True,calibration_physical_cache_complete=True,physically_verified_cache_files=30,cache_files=30,jobs=32,raw_archives=152,registered_trials=19,selection_policy='probability_mean',balanced_scores=scores,phase_archive=dict(bytes=100,sha256='a'*64,drive_path='/owned/raw.zip'))
    spectral=max((row for row in scores if row['family'] in ('combined','spectral')),key=lambda row:row['accuracy_pct'])['trial_id']
    chamfer=max((row for row in scores if row['family']=='chamfer'),key=lambda row:row['accuracy_pct'])['trial_id']
    raw_validation=json.dumps(validation).encode()
    accepted=dict(status='accepted',stage='development',phase_plan_sha256=phase.PLAN_SHA,manifest_sha256=phase.MANIFEST_SHA,source_snapshot_sha256=phase.SOURCE_SHA,physical_cache_complete=True,local_validation=phase.bytes_identity(raw_validation),archive=validation['phase_archive'],selection_policy='probability_mean',selected_spectral_trial_id=spectral,selected_chamfer_trial_id=chamfer)
    raw_accepted=json.dumps(accepted).encode()
    freeze=dict(status='frozen',phase_plan_sha256=phase.PLAN_SHA,development_acceptance=phase.bytes_identity(raw_accepted),selected_spectral_trial_id=spectral,selected_chamfer_trial_id=chamfer)
    ledger=dict(spent_seconds=7000.,cap_seconds=19800)
    prior=dict(conservative_max_ratio=1.025455024723712,original_future_budget_seconds=dict(confirmation_per_dataset_seed=300))
    remaining=plan['stage_expected_seconds']['confirmation']*prior['conservative_max_ratio']+180
    projection=dict(status='whole_campaign_reestimate_fits_approved_cap',phase_plan_sha256=phase.PLAN_SHA,spent_seconds=7000.,cap_seconds=19800,remaining_basis='registered_maximum_24_jobs',conservative_max_ratio=prior['conservative_max_ratio'],original_future_budget_seconds=dict(confirmation_per_dataset_seed=300),original_allowance_increase_seconds=180,remaining_point_seconds=remaining,margin_multiplier=1.2,packing_retry_reserve_seconds=1200,upper_seconds=7000+1.2*remaining+1200)
    raw=[]
    for name,row in registry.items():
        for dataset in ('modelnet40_c','scanobjectnn_c'):
            for head in ('dgcnn','curvenet','pointnext','pointmae'):
                raw.append(dict(trial_id=name,dataset=dataset,backbone=head,reused=row['reuse'],source_manifest_sha256=phase.OLD_MANIFEST if row['reuse'] else phase.MANIFEST_SHA))
    caches=[dict(remote_path='/cache/%d.npz'%n,local_verified=True,local_paths=['fixture'],bytes=123,sha256='a'*64) for n in range(30)]
    content={'accepted_sensitivity_development.json':raw_accepted,'validation.json':raw_validation,'sensitivity_confirmation_freeze.json':json.dumps(freeze).encode(),'post_development_budget_projection.json':json.dumps(projection).encode(),'raw_archives.json':json.dumps(raw).encode(),'cache_validation.json':json.dumps(caches).encode(),'calibration_cache_validation.json':json.dumps(caches).encode()}
    return content,registry,plan,ledger,prior


class AcceptanceTests(unittest.TestCase):
    def test_synthetic_complete_package_passes(self):
        content,*args=fixture()
        accepted,validation,freeze,projection=phase.validate_acceptance(content,*args)
        self.assertEqual(validation['raw_archives'],152)
        self.assertEqual(freeze['development_acceptance'],phase.bytes_identity(content['accepted_sensitivity_development.json']))
        self.assertLess(projection['upper_seconds'],19800)

    def test_understated_future_or_stale_spent_rejected(self):
        for change in [dict(remaining_point_seconds=1),dict(spent_seconds=1),dict(original_allowance_increase_seconds=0),dict(conservative_max_ratio=1.)]:
            content,*args=fixture();projection=json.loads(content['post_development_budget_projection.json']);projection.update(change);content['post_development_budget_projection.json']=json.dumps(projection).encode()
            with self.assertRaises(ValueError):phase.validate_acceptance(content,*args)

    def test_missing_cache_or_wrong_freeze_identity_rejected(self):
        content,*args=fixture();caches=json.loads(content['cache_validation.json']);caches.pop();content['cache_validation.json']=json.dumps(caches).encode()
        with self.assertRaises(ValueError):phase.validate_acceptance(content,*args)
        content,*args=fixture();freeze=json.loads(content['sensitivity_confirmation_freeze.json']);freeze['development_acceptance']['sha256']='b'*64;content['sensitivity_confirmation_freeze.json']=json.dumps(freeze).encode()
        with self.assertRaises(ValueError):phase.validate_acceptance(content,*args)

if __name__=='__main__':unittest.main()
