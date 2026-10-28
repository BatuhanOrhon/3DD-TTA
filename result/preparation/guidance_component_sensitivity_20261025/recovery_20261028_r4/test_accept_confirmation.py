import copy
import json
from pathlib import Path
import unittest

import accept_confirmation_template as phase

PREP=Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025')


def fixture():
    base=PREP/'recovery_20261028_r4';dev=base/'local_development_analysis_v1'
    plan=json.loads((base/'controls/bundle/campaign_plan.json').read_text())
    freeze=(dev/'sensitivity_confirmation_freeze.json').read_bytes();development=(dev/'accepted_sensitivity_development.json').read_bytes()
    calibration=(PREP/'calibration_local_acceptance.json').read_bytes()
    ids={dataset:json.loads((PREP/'assets_v2/study'/Path(row['path']).name).read_text())['indices'] for dataset,row in plan['id_manifests']['confirmation'].items()}
    expected=phase.expected_cache_paths(ids)
    caches=[dict(remote_path=name,bytes=123,sha256='a'*64,local_verified=True,local_paths=['synthetic_fixture']) for name in sorted(expected)]
    raw_archive=dict(bytes=100,sha256='b'*64,local_path=str(phase.ROOT/'sensitivity_confirmation_fixture.zip'),drive_path=str(phase.DRIVE/'phase-evidence/sensitivity_confirmation_fixture/sensitivity_confirmation_evidence.zip'))
    jobs=[];raw=[]
    for seed in range(3):
        for condition in phase.CONDITIONS:
            for dataset in phase.DATASETS:
                archives=[]
                for head in phase.HEADS:
                    row=dict(backbone=head,bytes=100,sha256='c'*64,local_path=str(phase.ROOT/'sensitivity_confirmation_fixture'/('seed%d'%seed)/condition/dataset/(head+'.zip')),drive_path=str(phase.DRIVE/dataset/'synthetic_fixture'/('seed%d'%seed)/condition/head/'final.zip'))
                    archives.append(row)
                    raw.append(dict(row,condition_id=condition,seed=seed,source_manifest_sha256=phase.MANIFEST_SHA))
                jobs.append(dict(status='complete',seed=seed,condition_id=condition,dataset=dataset,raw_archives=archives))
    receipt=dict(state=dict(status='complete',stage='confirmation',session=phase.SESSION,phase_plan_sha256=phase.PLAN_SHA,manifest_sha256=phase.MANIFEST_SHA,source_snapshot_sha256=phase.SOURCE_SHA,jobs=jobs),archive=raw_archive)
    cache_receipt=dict(status='verified',session=phase.SESSION,phase_plan_sha256=phase.PLAN_SHA,manifest_sha256=phase.MANIFEST_SHA,source_snapshot_sha256=phase.SOURCE_SHA,cache_count=180,per_seed_cache_count={'0':60,'1':60,'2':60},confirmation_phase_archive=raw_archive,archive=dict(bytes=200,sha256='d'*64,local_path=str(phase.ROOT/'synthetic_cache.zip'),drive_path=str(phase.DRIVE/'phase-evidence/synthetic_cache/confirmation_cache_evidence.zip')))
    contrasts=[]
    for comparator in phase.CONDITIONS[1:]:
        contrasts.append(dict(scope='all15',seed=None,aggregation='three_seed_mean',comparator=comparator,comparator_policy='terminal' if comparator=='original3dd' else 'probability_mean',candidate='spectral-candidate',candidate_policy='probability_mean',n_cases=8,n_seeds=3,n_original_object_clusters=280,n_prediction_events=50400,delta_pp=0.,corrected_events=0,harmed_events=0,positive_cases=0,negative_cases=0,tied_cases=8,per_seed_min_pp=0.,per_seed_max_pp=0.,conditional_object_bootstrap_low_pp=-1.,conditional_object_bootstrap_high_pp=1.,comparison='whole_policy_different_host' if comparator=='original3dd' else 'same_host_same_readout'))
    validation=dict(status='accepted',stage='confirmation',session=phase.SESSION,phase_plan_sha256=phase.PLAN_SHA,manifest_sha256=phase.MANIFEST_SHA,source_snapshot_sha256=phase.SOURCE_SHA,archive=raw_archive,frozen_selection=phase.bytes_identity(freeze),development_acceptance=phase.bytes_identity(development),calibration_acceptance=phase.bytes_identity(calibration),conditions=list(phase.CONDITIONS),jobs=24,raw_archives=96,physical_cache_files=180,physically_verified_cache_files=180,physical_cache_complete=True,n_original_object_clusters=280,full_benchmark=False,prior_benchmark_exposure=True,campaign_spent_seconds=15000.,selection_policy='probability_mean',primary_contrasts=contrasts,spectral_family='combined',limits='Conditional object-cluster intervals across3fixedseeds; no equivalence/universal superiority. Synthetic test fixture only.')
    content={'validation.json':json.dumps(validation).encode(),'cache_validation.json':json.dumps(caches).encode(),'raw_archives.json':json.dumps(raw).encode(),'sensitivity_confirmation_freeze.json':freeze,'accepted_sensitivity_development.json':development,'calibration_validation.json':calibration,'sensitivity_confirmation_receipt.json':json.dumps(receipt).encode(),'sensitivity_confirmation_cache_evidence_receipt.json':json.dumps(cache_receipt).encode(),'analysis_environment.json':json.dumps(dict(numpy='1.21.6',root_verified=True)).encode()}
    accepted=dict(status='accepted',stage='confirmation',phase_plan_sha256=phase.PLAN_SHA,archive=raw_archive,local_validation=phase.bytes_identity(content['validation.json']),physical_cache_complete=True,n_original_object_clusters=280,frozen_selection=phase.bytes_identity(freeze))
    content['accepted_sensitivity_confirmation.json']=json.dumps(accepted).encode()
    ledger=dict(spent_seconds=15001.,cap_seconds=19800)
    return content,plan,ids,ledger


def mutate_validation(content,change):
    validation=json.loads(content['validation.json']);change(validation);content['validation.json']=json.dumps(validation).encode()
    accepted=json.loads(content['accepted_sensitivity_confirmation.json']);accepted['local_validation']=phase.bytes_identity(content['validation.json']);content['accepted_sensitivity_confirmation.json']=json.dumps(accepted).encode()


class FinalAcceptanceTests(unittest.TestCase):
    def test_synthetic_four_condition_package_passes(self):
        content,*args=fixture();accepted,validation,receipt,cache=phase.validate_final(content,*args)
        self.assertEqual(validation['jobs'],24)
        self.assertEqual(len(validation['primary_contrasts']),3)
        self.assertEqual(accepted['n_original_object_clusters'],280)

    def test_840_clusters_or_missing_cache_rejected(self):
        content,*args=fixture();mutate_validation(content,lambda v:v['primary_contrasts'][0].update(n_original_object_clusters=840))
        with self.assertRaises(ValueError):phase.validate_final(content,*args)
        content,*args=fixture();caches=json.loads(content['cache_validation.json']);caches.pop();content['cache_validation.json']=json.dumps(caches).encode()
        with self.assertRaises(ValueError):phase.validate_final(content,*args)

    def test_changed_freeze_cache_receipt_or_excess_spend_rejected(self):
        content,*args=fixture();content['sensitivity_confirmation_freeze.json']+=b' '
        with self.assertRaises(ValueError):phase.validate_final(content,*args)
        content,*args=fixture();receipt=json.loads(content['sensitivity_confirmation_cache_evidence_receipt.json']);receipt['confirmation_phase_archive']['sha256']='f'*64;content['sensitivity_confirmation_cache_evidence_receipt.json']=json.dumps(receipt).encode()
        with self.assertRaises(ValueError):phase.validate_final(content,*args)
        content,plan,ids,ledger=fixture();ledger['spent_seconds']=19801
        with self.assertRaises(ValueError):phase.validate_final(content,plan,ids,ledger)

if __name__=='__main__':unittest.main()
