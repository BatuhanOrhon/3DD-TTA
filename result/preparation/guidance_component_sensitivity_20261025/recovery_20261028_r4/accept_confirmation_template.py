"""CPU-only final acceptance of genuine frozen r4 confirmation artifacts."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import zipfile

PACKAGE_IDENTITY = None  # Bind only after genuine completed NumPy1 local analysis.
ROOT=Path('/content/studies/guidance-components')
DRIVE=Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
SESSION='tta-guidance-components-a100-20261028-r4'
PLAN_SHA='1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
MANIFEST_SHA='599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef'
SOURCE_SHA='73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8'
DEVELOPMENT_ID=dict(bytes=1053,sha256='f94afc837e26a3ec33c8503f04cc9cfb7dafba44fd40cc19cff845eb7bad54ed')
FREEZE_ID=dict(bytes=798,sha256='ea245f8a75ed729463e34757328875f6d7c789983ef10bac3c0c133b0750c360')
CONDITIONS=('spectral-candidate','matched-chamfer','tuned-chamfer','original3dd')
DATASETS=('modelnet40_c','scanobjectnn_c')
HEADS=('dgcnn','curvenet','pointnext','pointmae')
CORRUPTIONS=('uniform','gaussian','background','impulse','upsampling','distortion_rbf','distortion_rbf_inv',
             'density','density_inc','shear','rotation','cutout','distortion','occlusion','lidar')
FILES=('accepted_sensitivity_confirmation.json','validation.json','cache_validation.json','raw_archives.json',
       'sensitivity_confirmation_freeze.json','accepted_sensitivity_development.json','calibration_validation.json',
       'sensitivity_confirmation_receipt.json','sensitivity_confirmation_cache_evidence_receipt.json','analysis_environment.json')


def require(condition,message):
    if not condition:raise ValueError(message)


def bytes_identity(raw):
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def expected_cache_paths(ids):
    result=set()
    for dataset in DATASETS:
        selected=ids[dataset]
        require(len(selected)==140 and selected==sorted(set(selected)),'Expected140 registered IDs/dataset')
        for seed in (0,1,2):
            for corruption in CORRUPTIONS:
                for batch in (0,1):
                    name=hashlib.sha256(struct.pack('<70q',*selected[70*batch:70*(batch+1)])).hexdigest()+'.npz'
                    result.add(str(ROOT/('paired_sensitivity_confirmation_v2_seed%d'%seed)/dataset/corruption/name))
    require(len(result)==180,'Expected180 unique confirmation caches')
    return result


def validate_final(content,plan,ids,ledger):
    accepted,validation,caches,raw,freeze,development,calibration,receipt,cache_receipt,environment=[json.loads(content[name]) for name in FILES]
    require(accepted['status']==validation['status']=='accepted' and accepted['stage']==validation['stage']=='confirmation'
            and accepted['phase_plan_sha256']==validation['phase_plan_sha256']==PLAN_SHA
            and accepted['local_validation']==bytes_identity(content['validation.json'])
            and accepted['physical_cache_complete'] is validation['physical_cache_complete'] is True,'Final local acceptance/validation incomplete')
    require(validation['session']==SESSION and validation['manifest_sha256']==MANIFEST_SHA and validation['source_snapshot_sha256']==SOURCE_SHA
            and validation['jobs']==24 and validation['raw_archives']==96 and validation['conditions']==list(CONDITIONS)
            and validation['physical_cache_files']==validation['physically_verified_cache_files']==180
            and accepted['n_original_object_clusters']==validation['n_original_object_clusters']==280
            and validation['selection_policy']=='probability_mean' and validation['spectral_family']=='combined'
            and validation['full_benchmark'] is False and validation['prior_benchmark_exposure'] is True,'Final source/coverage/conditional sampling contract differs')
    require(type(ledger['spent_seconds']) in (int,float) and math.isfinite(ledger['spent_seconds'])
            and 155.86660684299568<=validation['campaign_spent_seconds']<=ledger['spent_seconds']<=19800
            and ledger['cap_seconds']==19800,'Cumulative campaign spend changed or exceeded approval')
    require(bytes_identity(content['accepted_sensitivity_development.json'])==DEVELOPMENT_ID
            and bytes_identity(content['sensitivity_confirmation_freeze.json'])==FREEZE_ID
            and accepted['frozen_selection']==validation['frozen_selection']==FREEZE_ID
            and validation['development_acceptance']==DEVELOPMENT_ID and freeze['development_acceptance']==DEVELOPMENT_ID
            and freeze['status']=='frozen' and freeze['phase_plan_sha256']==PLAN_SHA
            and [row['id'] for row in freeze['conditions']]==list(CONDITIONS),'Frozen four-condition development binding changed')
    require(validation['calibration_acceptance']==bytes_identity(content['calibration_validation.json'])==plan['recovery']['independent_acceptance']
            and calibration['status']=='accepted_local_label_free_calibration'
            and str(environment['numpy']).startswith('1.') and environment['root_verified'] is True,'Original calibration/NumPy1 analysis proof missing')
    state=receipt['state'];archive=receipt['archive']
    require(state['status']=='complete' and state['stage']=='confirmation' and state['session']==SESSION
            and state['phase_plan_sha256']==PLAN_SHA and state['manifest_sha256']==MANIFEST_SHA and state['source_snapshot_sha256']==SOURCE_SHA
            and accepted['archive']==validation['archive']==archive,'Genuine completed raw receipt differs from local acceptance')
    jobs=state['jobs'];expected_jobs={(name,dataset,seed) for name in CONDITIONS for dataset in DATASETS for seed in (0,1,2)}
    require(len(jobs)==24 and {(job['condition_id'],job['dataset'],job['seed']) for job in jobs}==expected_jobs
            and all(job['status']=='complete' and len(job['raw_archives'])==4 for job in jobs),'Frozen final job grid incomplete')
    expected_raw={}
    for job in jobs:
        require({row['backbone'] for row in job['raw_archives']}==set(HEADS),'Four final frozen heads required')
        for row in job['raw_archives']:
            require(row['local_path'] not in expected_raw,'Duplicate raw archive identity')
            expected_raw[row['local_path']]=(row,job['condition_id'],job['seed'])
    require(len(raw)==96 and len({row['local_path'] for row in raw})==96
            and {row['local_path'] for row in raw}==set(expected_raw),'Expected96 final raw ZIPs')
    for row in raw:
        original,condition,seed=expected_raw[row['local_path']]
        require(all(row[key]==value for key,value in original.items()) and row['condition_id']==condition and row['seed']==seed
                and row['source_manifest_sha256']==MANIFEST_SHA,'Final raw ZIP/source/condition relabeled')
    expected_caches=expected_cache_paths(ids)
    require(len(caches)==180 and {row['remote_path'] for row in caches}==expected_caches
            and all(row['local_verified'] is True and row['local_paths'] for row in caches),'All180 original cache bytes must be physically verified')
    require(cache_receipt['status']=='verified' and cache_receipt['session']==SESSION and cache_receipt['phase_plan_sha256']==PLAN_SHA
            and cache_receipt['manifest_sha256']==MANIFEST_SHA and cache_receipt['source_snapshot_sha256']==SOURCE_SHA
            and cache_receipt['cache_count']==180 and cache_receipt['per_seed_cache_count']=={'0':60,'1':60,'2':60}
            and cache_receipt['confirmation_phase_archive']==archive,'Original cache-evidence receipt differs from final phase')
    contrasts=validation['primary_contrasts']
    require(len(contrasts)==3 and {row['comparator'] for row in contrasts}==set(CONDITIONS[1:]),'Missing frozen primary contrasts')
    for row in contrasts:
        original=row['comparator']=='original3dd'
        require(row['scope']=='all15' and row['seed'] is None and row['aggregation']=='three_seed_mean'
                and row['candidate']=='spectral-candidate' and row['candidate_policy']=='probability_mean'
                and row['comparator_policy']==('terminal' if original else 'probability_mean')
                and row['comparison']==('whole_policy_different_host' if original else 'same_host_same_readout')
                and row['n_cases']==8 and row['n_seeds']==3 and row['n_original_object_clusters']==280 and row['n_prediction_events']==50400,
                'Conditional paired contrast sampling/readout differs;840 independent clusters are invalid')
        for key in ('delta_pp','per_seed_min_pp','per_seed_max_pp','conditional_object_bootstrap_low_pp','conditional_object_bootstrap_high_pp'):
            require(type(row[key]) in (int,float) and math.isfinite(row[key]) and -100<=row[key]<=100,'Invalid percentage-point estimate/conditional interval')
        require(row['conditional_object_bootstrap_low_pp']<=row['conditional_object_bootstrap_high_pp']
                and row['per_seed_min_pp']-1e-9<=row['delta_pp']<=row['per_seed_max_pp']+1e-9,'Malformed conditional interval/seed range')
        for key in ('corrected_events','harmed_events','positive_cases','negative_cases','tied_cases'):
            require(type(row[key]) is int and row[key]>=0,'Invalid paired event/case count')
        require(row['corrected_events']+row['harmed_events']<=50400 and row['positive_cases']+row['negative_cases']+row['tied_cases']==8
                and abs(row['delta_pp']-100.*(row['corrected_events']-row['harmed_events'])/50400.)<=1e-8,'Paired counts and balanced effect disagree')
    require('Conditional object-cluster intervals across3fixedseeds' in validation['limits']
            and 'no equivalence/universal superiority' in validation['limits'],'Conditional interpretation limitations omitted')
    return accepted,validation,receipt,cache_receipt


def main():
    package=Path('/content/agent_b_r4_confirmation_local_acceptance.zip')
    require(isinstance(PACKAGE_IDENTITY,dict) and set(PACKAGE_IDENTITY)=={'bytes','sha256'},'Bind actual genuine final local package identity before execution')
    require(bytes_identity(package.read_bytes())==PACKAGE_IDENTITY,'Final compact binder ZIP changed')
    controller=ROOT/'sensitivity_phases/sensitivity_campaign_controller.py'
    require(bytes_identity(controller.read_bytes())['sha256']=='0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc','Pinned current controller changed')
    spec=importlib.util.spec_from_file_location('final_confirmation_acceptance',controller)
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    plan,registry,selections=c.plan_verified(ROOT/'sensitivity_campaign_plan.json',PLAN_SHA)
    ledger=c.load_ledger(PLAN_SHA)
    attempts=[row for row in ledger['attempts'] if row['stage']=='confirmation']
    require(ledger['active'] is None and len(attempts)==1 and attempts[0]['status']=='complete' and ledger['spent_seconds']<=19800,'Final confirmation ledger incomplete/active/over budget')
    helper=ROOT/'phases/deploy_assets_template.py'
    require(bytes_identity(helper.read_bytes())['sha256']=='89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0','Pinned immutable evidence helper changed')
    sys.path.insert(0,str(helper.parent))
    from deploy_assets_template import checked_members,copy_checked,identity
    with zipfile.ZipFile(package) as archive:
        checked_members(archive);content={name:archive.read(name) for name in FILES}
    ids={dataset:selections[('confirmation',dataset)]['indices'] for dataset in DATASETS}
    accepted,validation,raw_receipt,cache_receipt=validate_final(content,plan,ids,ledger)
    for name in ('sensitivity_confirmation_receipt.json','sensitivity_confirmation_cache_evidence_receipt.json',
                 'sensitivity_confirmation_freeze.json','accepted_sensitivity_development.json'):
        require(content[name]==c.owned(ROOT/name).read_bytes(),'Local final binder differs from genuine remote '+name)
    remote_validation=json.loads((ROOT/'sensitivity_confirmation_validation.json').read_text())
    require(remote_validation['status']=='validated' and remote_validation['stage']=='confirmation' and remote_validation['phase_plan_sha256']==PLAN_SHA
            and remote_validation['output']==attempts[0]['output'] and remote_validation['evidence']==raw_receipt['archive'],'Remote validated phase differs')
    calibration=c.calibration_verified(plan['calibration']['report_path'],plan)
    conditions=c.frozen_conditions(plan,registry,calibration,PLAN_SHA)
    require([row['id'] for row in conditions]==list(CONDITIONS),'Current frozen protocol construction changed')
    for original in (raw_receipt['archive'],cache_receipt['archive']):
        c.verify(c.owned(original['drive_path'],c.DRIVE),original)
    with zipfile.ZipFile(raw_receipt['archive']['drive_path']) as archive:
        require(archive.read('sensitivity_confirmation_freeze.json')==content['sensitivity_confirmation_freeze.json']
                and archive.read('accepted_sensitivity_development.json')==content['accepted_sensitivity_development.json']
                and json.loads(archive.read('state.json'))==raw_receipt['state'],'Original raw Drive archive frozen bindings changed')
    with zipfile.ZipFile(cache_receipt['archive']['drive_path']) as archive:
        checked_members(archive);cache_records=json.loads(archive.read('cache_records.json'))
    actual={row['local_path']:{key:row[key] for key in ('bytes','sha256')} for row in cache_records}
    local={row['remote_path']:{key:row[key] for key in ('bytes','sha256')} for row in json.loads(content['cache_validation.json'])}
    require(len(cache_records)==len(actual)==180 and actual==local,'Original immutable cache archive differs from180 local physical checks')
    targets={'accepted_sensitivity_confirmation.json':'accepted_sensitivity_confirmation.json','validation.json':'accepted_sensitivity_confirmation_validation.json',
             'cache_validation.json':'accepted_sensitivity_confirmation_cache_validation.json'}
    receipt_path=ROOT/'sensitivity_confirmation_local_acceptance_receipt.json'
    require(all(not c.owned(ROOT/name).exists() for name in targets.values()) and not receipt_path.exists(),'Final acceptance/validation/receipt already exists; inspect before retry')
    drive=c.DRIVE/'phase-evidence'/('sensitivity_confirmation_local_acceptance_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    drive.mkdir(parents=True,exist_ok=False);copy_checked(package,drive/package.name,PACKAGE_IDENTITY)
    for source,target in targets.items():
        with (ROOT/target).open('xb') as stream:stream.write(content[source])
    result=dict(status='accepted',stage='confirmation',session=SESSION,phase_plan_sha256=PLAN_SHA,
                manifest_sha256=MANIFEST_SHA,source_snapshot_sha256=SOURCE_SHA,
                archive=dict(PACKAGE_IDENTITY,drive_path=str(drive/package.name)),acceptance=bytes_identity(content['accepted_sensitivity_confirmation.json']),
                local_validation=bytes_identity(content['validation.json']),local_cache_validation=bytes_identity(content['cache_validation.json']),
                frozen_selection=FREEZE_ID,development_acceptance=DEVELOPMENT_ID,confirmation_phase_archive=raw_receipt['archive'],
                cache_archive=cache_receipt['archive'],n_original_object_clusters=280,conditions=list(CONDITIONS),campaign_spent_seconds=ledger['spent_seconds'])
    with receipt_path.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    (drive/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AGENT_B_SENSITIVITY_CONFIRMATION_LOCAL_ACCEPTANCE',json.dumps(result),flush=True)
    print('AGENT_B_SENSITIVITY_CONFIRMATION_ACCEPTANCE_COMPLETE',flush=True)


if __name__=='__main__':main()
