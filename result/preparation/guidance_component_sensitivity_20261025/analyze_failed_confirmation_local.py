"""Separate local adapter for the immutable failed confirmation and remaining slots.

Failed tuned-C observations remain unavailable; no estimator is changed.
"""
import argparse
from contextlib import closing
import copy
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

import analyze_sensitivity_confirmation_local as complete

shared=complete.shared
require=shared.require
FAILURE_SIGNATURE='Classifier input contains nonfinite XYZ coordinates'
ORIGINAL_ARCHIVE=dict(bytes=1159246521,sha256='f586870206961f88d38daa7ed9da632fddfa51356075e114c5f3b98f020ebe5e')
ORIGINAL_RECEIPT=dict(bytes=3676445,sha256='5ab88805d14edbc9713275816aa11bef1d5e80784ea385b477199f7d6b779754')
ORIGINAL_LEDGER=dict(bytes=12181,sha256='60c8ba55d787c8e1e9ea4fa34f27eeb3e9aa8522300252c6c74863e1cad2736b')
FAILED_SPENT=14114.660198396998
SEMANTIC_KEYS=('condition_id','role','trial','dataset','seed')


def known_failure(log):
    errors=re.findall(r'^[A-Za-z_][\w.]*(?:Error|Exception):[^\r\n]*$',log,re.MULTILINE)
    return (log.count('Traceback (most recent call last):')==1
            and errors==['ValueError: '+FAILURE_SIGNATURE]
            and 'classifier_backbones.py", line 48, in forward' in log
            and 'raise ValueError("Classifier input contains nonfinite XYZ coordinates")' in log)


def phase_header(state,plan,plan_sha,stage,status):
    require(state.get('status')==status and state.get('stage')==stage
            and state.get('session')==plan['session'] and state.get('phase_plan_sha256')==plan_sha
            and state.get('manifest_sha256')==plan['manifest_sha256']
            and state.get('source_snapshot_sha256')==plan['source_snapshot_sha256']
            and state.get('full_benchmark') is False,'Failed-aware phase status/source/ownership mismatch')


def compose_slots(expected,original,remaining,failures):
    """Require every frozen slot; only20/21 may have the independently proved failure."""
    require(len(expected)==24 and len(original)==21 and len(remaining)==3,'Require original20complete+failed20 and exactly3 remaining slots')
    require(set(failures) in ({20},{20,21}) and all(row['failure_type']==FAILURE_SIGNATURE and len(row['partial_raw_archives'])==4 for row in failures.values()),
            'Only the specific recorded finiteXYZ failures at slots20/21 are allowed')
    indexed={}
    for number,job in enumerate(original):
        require(all(job.get(k)==expected[number][k] for k in SEMANTIC_KEYS),'Original registered slot/config changed')
        require(job.get('status')==('complete' if number<20 else 'running'),'Original incomplete slot cannot be called complete')
        indexed[number]=dict(registered_job_index=number,status='complete' if number<20 else 'failed_numerical',
                            condition_id=job['condition_id'],dataset=job['dataset'],seed=job['seed'],source_phase='original_failed')
    seen=set()
    for job in remaining:
        number=job.get('registered_job_index')
        require(type(number) is int and number in (21,22,23) and number not in seen,'Missing/duplicate/unregistered remaining slot')
        seen.add(number)
        require(all(job.get(k)==expected[number][k] for k in SEMANTIC_KEYS),'Remaining registered slot/config changed')
        require(job.get('status')=='complete' or (number==21 and job.get('status')=='failed_numerical' and number in failures),
                'Incomplete/non-numerical or primary-method failure must stop analysis')
        require((number in failures)==(job['status']=='failed_numerical'),'Failure proof and recorded slot status disagree')
        indexed[number]=dict(registered_job_index=number,status=job['status'],condition_id=job['condition_id'],dataset=job['dataset'],seed=job['seed'],source_phase='remaining')
    require(seen=={21,22,23} and set(indexed)==set(range(24)),'Unrun registered slots remain; do not silently drop them')
    return [indexed[i] for i in range(24)]


def endpoint_conditions(slots,frozen_names):
    require(len(frozen_names)==len(set(frozen_names))==4 and frozen_names==['spectral-candidate','matched-chamfer','tuned-chamfer','original3dd'],
            'Preserve the frozen four-condition order')
    result=[]
    for name in frozen_names:
        group=[r for r in slots if r['condition_id']==name]
        require(len(group)==6 and {(r['dataset'],r['seed']) for r in group}=={(d,s) for d in complete.DATASETS for s in complete.SEEDS},
                'Frozen method grid missing/duplicated')
        if all(r['status']=='complete' for r in group):result.append(name)
    require('spectral-candidate' in result and 'matched-chamfer' in result,'Primary matched comparison is incomplete')
    require('tuned-chamfer' not in result,'Known failed tuned-C cannot become a complete endpoint')
    return result


def unavailable_endpoints(slots,frozen_names):
    available=endpoint_conditions(slots,frozen_names)
    result=[]
    for name in frozen_names:
        if name in available:continue
        group=[r for r in slots if r['condition_id']==name]
        result.append(dict(condition_id=name,status='unavailable_due_numerical_failure',required_jobs=6,
            completed_jobs=sum(r['status']=='complete' for r in group),
            failed_registered_jobs=[r['registered_job_index'] for r in group if r['status']=='failed_numerical'],
            unrun_registered_jobs=[r['registered_job_index'] for r in group if r['status']=='unrun'],
            three_seed_balanced_accuracy_pct=None,conditional_object_bootstrap_interval_pp=None,
            policy='probability_mean',reason='Frozen higher-rate tunedC failed; no two-seed mean, dropped dataset, zero substitution or retuned fallback'))
    return result


def extract_evidence(archive_path,reference,output):
    shared.verify(archive_path,reference)
    require(str(reference['drive_path']).startswith(shared.DRIVE),'Evidence archive outside owned Drive')
    require(not output.exists(),'Fresh evidence extraction required')
    with shared.checked_zip(archive_path) as archive:
        names={n for n in archive.namelist() if not n.endswith('/')}
        declared=json.loads(archive.read('evidence_manifest.json'))
        require(set(declared)==names-{'evidence_manifest.json'},'Evidence manifest coverage mismatch')
        for name,expected in declared.items():
            data=archive.read(name)
            require(dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==expected,'Evidence member hash/size changed')
        archive.extractall(output)
    return shared.read_json(output/'state.json')


def common_phase_files(directory,state,plan_path,plan_sha,freeze_path,development_path):
    plan=shared.read_json(plan_path)
    require((directory/'campaign_plan.json').read_bytes()==plan_path.read_bytes(),'Phase campaign plan changed')
    for name,expected in plan['phase_sources'].items():shared.verify(directory/name,expected)
    approval=shared.read_json(directory/'gpu_approval.json')
    require(approval.get('status')=='approved' and approval.get('phase_plan_sha256')==plan_sha
            and approval.get('session')==plan['session'] and approval.get('cap_seconds')==19800
            and bool(approval.get('user_reply','').strip()),'Current approved scope missing')
    for path in (freeze_path,development_path):
        require((directory/path.name).read_bytes()==path.read_bytes(),'Frozen selection/development acceptance changed')


def registered_jobs(plan,conditions,calibration,controller,remote):
    jobs=[]
    for seed in complete.SEEDS:
        for condition in conditions:
            for dataset in complete.DATASETS:
                root=Path(str(remote))/('seed%d'%seed)/condition['id']/dataset
                trial=condition['trial']
                jobs.append(dict(condition_id=condition['id'],role=condition['role'],trial=trial,dataset=dataset,seed=seed,
                    result_root=str(root),command=controller.command_for(plan,trial,dataset,'confirmation',seed,root,calibration)))
    return jobs


def failed_source_contract(config,bundle,context):
    """Recover one failure-truncated inventory entry from the exact sealed bytes."""
    plan,manifest=context[0],context[4]
    recorded=config['executed_source_files']
    required=['scripts/eval_multibackbone_guidance.py','scripts/eval_multibackbone_source.py',
              'classifier_backbones.py','spectral_guidance_losses.py','tta_gsd_v2.py','shared_trajectory.py']
    for name,pin in recorded.items():
        require(name.startswith('assets/') and manifest['files'].get(name[len('assets/'):])==pin,
                'Failed recorded source identity mismatch: '+name)
    missing=['assets/source_snapshot/'+name for name in required if 'assets/source_snapshot/'+name not in recorded]
    guarded=copy.deepcopy(config);guarded['status']='partial'
    proof=dict(source_inventory_recovered_from_snapshot=False,missing_original_inventory_keys=missing)
    if not missing:return guarded,proof
    key='assets/source_snapshot/shared_trajectory.py'
    require(missing==[key], 'Only the sole known failure-truncated shared_trajectory inventory may be recovered')
    failure=config.get('failure','');root=plan['asset_root']+'/source_snapshot/'
    require(config.get('status')=='failed' and known_failure(failure)
            and 'File "'+root+'shared_trajectory.py", line 73, in clean' in failure
            and 'File "'+root+'scripts/eval_multibackbone_guidance.py", line 525, in worker' in failure,
            'Missing the pinned worker/shared-trajectory numerical-failure trace')
    raw=bundle.read('source_snapshot.zip')
    whole=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    require(whole==manifest['files']['source_snapshot.zip'] and whole['sha256']==plan['source_snapshot_sha256']
            =='73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8', 'Archived source snapshot changed')
    member='source_snapshot/shared_trajectory.py'
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(len(archive.namelist())==len(set(archive.namelist())) and archive.testzip() is None, 'Archived source ZIP duplicate/CRC failure')
        data=archive.read(member)
    pin=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    require(pin==manifest['files'][member] and pin==dict(bytes=6524,sha256='dce0e8b2ef3ac4d555f3572ae0daa41b7c1afac3ad3f1cfdf82bdbc095f2fcda'),
            'Archived shared-trajectory member identity changed')
    guarded['executed_source_files'][key]=pin
    proof.update(source_inventory_recovered_from_snapshot=True,archived_source_snapshot=whole,
                 recovered_member=dict(path=member,**pin),known_failure_trace_verified=True,
                 scope='Failed provenance only: original executed-source inventory lacks this key; archived sealed bytes and known pinned trace match. No raw mutation, failed-prefix accuracy or complete-result acceptance.')
    return guarded,proof


def partial_failure(path,job,context,calibration,expected_log,label_refs):
    """Read failed metadata/source contracts only; never score partial predictions."""
    require(known_failure(expected_log),'Missing exact sole recorded numerical failure')
    plan,controller,_,assets,manifest,_,selections,*_=context
    with shared.checked_zip(path):
        pass
    with closing(context[7](path)) as archive:
        config=archive.json('config.json')
        require(config.get('status')=='failed' and known_failure(config.get('failure',''))
                and not any(k.endswith('_audit_error') for k in config),'Partial artifact does not record known failure or failed state audit')
        require(path.name==config['run_id']+'.zip' and config['backbone'] in complete.HEADS,'Failed raw run/head mismatch')
        selected=selections[('confirmation',job['dataset'])]
        # A metadata copy selects the existing strict contract checks; raw status and bytes stay failed.
        contract,source_inventory=failed_source_contract(config,archive,context)
        controller.verify_common_config(contract,job,dict(plan,asset_root=str(assets)),manifest,selected,calibration)
        selection_bytes=archive.read('selection_manifest.json')
        require(json.loads(selection_bytes)==selected and config['selection_manifest_identity']==dict(bytes=len(selection_bytes),sha256=hashlib.sha256(selection_bytes).hexdigest()),
                'Partial selected IDs/source changed')
        prefix=[];previous_complete=True
        ids=np.asarray(selected['indices'],dtype=np.int64)
        for corruption in context[-1]:
            names=sorted(n for n in archive.names if n.startswith('batches/'+corruption+'/') and n.endswith('/predictions.npz'))
            require(len(names)<=2 and (previous_complete or not names), 'Non-prefix partial batch coverage')
            for number,name in enumerate(names):
                base='batches/'+corruption+'/batch%04d/'%number
                require(name==base+'predictions.npz','Partial batch order changed')
                expected_ids=ids[number*70:(number+1)*70]
                with np.load(io.BytesIO(archive.read(name)),allow_pickle=False) as data:
                    require(np.array_equal(data['index'],expected_ids) and np.array_equal(data['label'],label_refs[job['dataset']]['label'][number*70:(number+1)*70]),
                            'Partial prefix IDs/labels changed')
                diagnostic=json.loads(archive.read(base+'diagnostics.json'))
                require(type(diagnostic['seed']) is int and diagnostic['seed']==job['seed'] and diagnostic['rng_unchanged'] is True
                        and diagnostic['geometry_shape']==[70,2048,3],'Partial prefix state/RNG/geometry guard failed')
                prefix.append(dict(corruption=corruption,batch=number,n_ids=70,accuracy_used=False,
                                   initial_hashes=diagnostic['trajectory']['initial_hashes'],paired_state=diagnostic['paired_state']))
            previous_complete=previous_complete and len(names)==2
        require({n for n in archive.names if n.startswith('batches/') and n.endswith('/predictions.npz')}
                =={'batches/'+r['corruption']+'/batch%04d/predictions.npz'%r['batch'] for r in prefix}, 'Undeclared partial batch predictions')
        if job['dataset']=='modelnet40_c':
            require([(r['corruption'],r['batch']) for r in prefix]==[('uniform',0),('uniform',1),('gaussian',0),('gaussian',1),('background',0)],
                    'Original failed slot20 prefix changed')
        summary=list(csv.DictReader(io.StringIO(archive.read('summary.csv').decode())))
        require(len(summary)==1 and summary[0]['status']=='failed','Partial artifact cannot be reported complete')
    return dict(local_path=str(path),**shared.identity(path),backbone=config['backbone'],run_id=config['run_id'],
                status='failed_numerical_provenance_only',failure_type=FAILURE_SIGNATURE,
                manifest_sha256=config['asset_manifest_sha256'],source_snapshot_sha256=plan['source_snapshot_sha256'],
                checkpoint_sha256=config['checkpoint_load']['checkpoint_sha256'],native_bindings=config['native_bootstrap']['bindings'],
                classifier_state_sha256=config['classifier_state_before_sha256'],lion_state=config['lion_state_before'],
                completed_prefix=prefix,source_inventory=source_inventory,accuracy_used=False)


def preparation(args,output):
    prep=args.prep.resolve();plan_path=args.plan.resolve()
    context=complete.load_context(prep,plan_path,args.plan_sha256)
    plan,controller,_,assets,manifest,registry,selections,*_=context
    shared.PLAN_SHA=args.plan_sha256
    recovery=plan['recovery']
    shared.verify(args.calibration_archive,recovery['calibration_archive'])
    accepted_path=prep/'calibration_local_acceptance.json';shared.verify(accepted_path,recovery['independent_acceptance'])
    validator=shared.load_module('partial_original_calibration_validator',prep/'validate_calibration_local.py')
    accepted=validator.validate(prep,args.calibration_receipt,args.calibration_archive,output/'calibration_validation.json',args.old_prep)
    require(accepted==shared.read_json(accepted_path),'Original independent calibration changed')
    with shared.checked_zip(args.calibration_archive) as archive:raw=archive.read('calibration/calibration_report.json')
    require(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==recovery['calibration_report'],'Original calibration report changed')
    calibration=json.loads(raw)
    development=shared.read_json(args.development_acceptance)
    shared.verify(args.development_acceptance.parent/'validation.json',development['local_validation'])
    validation=shared.read_json(args.development_acceptance.parent/'validation.json')
    require(validation['status']=='accepted' and validation['physical_cache_complete'] is True,'Development acceptance incomplete')
    receipt=shared.unpack_phase(args.development_archive,args.development_receipt,'development',plan,output/'development_phase')
    require(development['archive']==receipt['archive'],'Frozen acceptance belongs to another development archive')
    freeze=shared.read_json(args.freeze)
    conditions=complete.frozen_conditions(freeze,development,validation['balanced_scores'],registry,controller,calibration,
                                          shared.identity(args.development_acceptance),args.plan_sha256)
    require([r['id'] for r in conditions]==['spectral-candidate','matched-chamfer','tuned-chamfer','original3dd'],
            'Failure adapter requires the actual frozen four-condition design')
    return context,calibration,conditions


def failure_archives(directory,remote,job,context,calibration,log_text,label_refs,records=None):
    result=[];seen=set()
    if records is None:
        root=directory.joinpath(*PurePosixPath(job['result_root']).relative_to(remote).parts)
        method='gsd_inspired_latent_spectral_guidance'
        paths=sorted((root/job['dataset']/method).glob('*.zip'))
        require(len(paths)==4,'Require all four immutable failed worker ZIPs')
        records=[dict(local_path=str(remote/str(p.relative_to(directory))),drive_path=None,**shared.identity(p)) for p in paths]
    require(len(records)==4,'Four failed head archives required')
    for record in records:
        relative=PurePosixPath(record['local_path']).relative_to(remote)
        require('..' not in relative.parts,'Unsafe failed raw archive path')
        path=directory.joinpath(*relative.parts);shared.verify(path,record)
        row=partial_failure(path,job,context,calibration,log_text,label_refs)
        require(row['backbone'] not in seen,'Duplicate failed head');seen.add(row['backbone'])
        if record.get('backbone') is not None:require(row['backbone']==record['backbone'],'Failed head receipt mismatch')
        if record.get('run_id') is not None:require(row['run_id']==record['run_id'],'Failed run receipt mismatch')
        if record.get('drive_path') is not None:require(record['drive_path'].startswith(shared.DRIVE),'Failed raw outside owned Drive')
        row['remote_path']=record['local_path'];row['drive_path']=record.get('drive_path');result.append(row)
    require(seen==set(complete.HEADS),'Missing failed head')
    return result


def read_phases(args,context,calibration,conditions,label_refs,output):
    plan,controller=context[:2]
    shared.verify(args.original_phase_archive,ORIGINAL_ARCHIVE)
    shared.verify(args.original_receipt,ORIGINAL_RECEIPT)
    shared.verify(args.original_ledger,ORIGINAL_LEDGER)
    original_receipt=shared.read_json(args.original_receipt)
    require({k:original_receipt['archive'][k] for k in ('bytes','sha256')}==ORIGINAL_ARCHIVE,'Original failure archive pin changed')
    original_dir=output/'original_failed_phase'
    original=extract_evidence(args.original_phase_archive,original_receipt['archive'],original_dir)
    require(original==original_receipt['state'],'Original receipt/physical state differ')
    phase_header(original,plan,args.plan_sha256,'confirmation','failed')
    common_phase_files(original_dir,original,plan_path=args.plan,plan_sha=args.plan_sha256,freeze_path=args.freeze,development_path=args.development_acceptance)
    original_ledger=shared.read_json(args.original_ledger);controller.ledger_valid(original_ledger,args.plan_sha256)
    require(original_ledger['spent_seconds']==FAILED_SPENT and original_ledger['attempts'][-1]['stage']=='confirmation'
            and original_ledger['attempts'][-1]['status']=='failed','Original cumulative failed attempt changed')
    snapshot=shared.read_json(original_dir/'ledger_snapshot.json');controller.ledger_valid(snapshot,args.plan_sha256)
    require(snapshot['attempts']==original_ledger['attempts'] and snapshot['spent_seconds']<=original_ledger['spent_seconds'],
            'Original pre/post-seal ledger conflict')
    original_remote=PurePosixPath(original_receipt['archive']['local_path']).with_suffix('')
    require(original_remote.parent==shared.REMOTE and original_remote.name.startswith('sensitivity_confirmation_'),'Foreign original confirmation output')
    expected=registered_jobs(plan,conditions,calibration,controller,original_remote)
    require(shared.read_json(original_dir/'commands.json')==expected,'Original registered commands changed')
    require(len(original['jobs'])==21,'Unexpected failed attempt coverage')
    for i,job in enumerate(original['jobs']):
        require(all(job[k]==v for k,v in expected[i].items()),'Original job command/config drift')
    log_text=(original_dir/'job_020.log').read_text()
    require(known_failure(log_text),'Failed worker log is not the sole recorded numerical failure')
    failures={20:dict(failure_type=FAILURE_SIGNATURE,partial_raw_archives=failure_archives(original_dir,original_remote,original['jobs'][20],context,calibration,log_text,label_refs))}
    review_path=args.reviewed_recovery_plan
    require(shared.identity(review_path)['sha256']==args.reviewed_recovery_plan_sha256,'Reviewed remaining-slots plan changed')
    reviewed=shared.read_json(review_path)
    require(reviewed['original_receipt_identity']==ORIGINAL_RECEIPT and reviewed['failed_ledger_identity']==ORIGINAL_LEDGER,
            'Reviewed recovery is not bound to the actual failed attempt')
    remaining_receipt=shared.read_json(args.remaining_receipt)
    remaining_dir=output/'remaining_phase'
    remaining=extract_evidence(args.remaining_phase_archive,remaining_receipt['archive'],remaining_dir)
    require(remaining==remaining_receipt['state'],'Remaining receipt/physical state differ')
    phase_header(remaining,plan,args.plan_sha256,'confirmation_remaining','remaining_slots_recorded')
    require(remaining['registered_jobs']==[21,22,23] and remaining['excluded_failed_slots']==[20]
            and remaining['original_confirmation_archive']==original_receipt['archive']
            and remaining['original_receipt_identity']==ORIGINAL_RECEIPT
            and remaining['reviewed_recovery_plan_sha256']==args.reviewed_recovery_plan_sha256,
            'Remaining stage changed excluded failure or registered scope')
    require((remaining_dir/'reviewed_recovery_plan.json').read_bytes()==review_path.read_bytes()
            and (remaining_dir/'original_failed_receipt.json').read_bytes()==args.original_receipt.read_bytes()
            and (remaining_dir/'original_failed_ledger.json').read_bytes()==args.original_ledger.read_bytes(),
            'Archived reviewed plan/original failure evidence changed')
    require(set(reviewed['sources'])=={'remaining_slots_controller.py','run_remaining_slots.py'},'Missing reviewed continuation source pins')
    for name,pin in reviewed['sources'].items():shared.verify(remaining_dir/name,pin)
    common_phase_files(remaining_dir,remaining,args.plan,args.plan_sha256,args.freeze,args.development_acceptance)
    remaining_remote=PurePosixPath(remaining_receipt['archive']['local_path']).with_suffix('')
    require(remaining_remote.parent==shared.REMOTE and remaining_remote.name.startswith('sensitivity_confirmation_remaining_'),'Foreign remaining output')
    wanted=registered_jobs(plan,conditions,calibration,controller,remaining_remote)
    commands=shared.read_json(remaining_dir/'commands.json')
    require(len(commands)==3 and [r['registered_job_index'] for r in commands]==[21,22,23], 'Remaining commands are not the original three unrun slots')
    require(shared.read_json(remaining_dir/'original_commands.json')==expected,'Continuation relabelled original commands')
    require(len(remaining['jobs'])==3,'Remaining slot state incomplete')
    for command,job,index in zip(commands,remaining['jobs'],(21,22,23)):
        require(command['registered_job_index']==job['registered_job_index']==index
                and all(command[k]==job[k]==v for k,v in wanted[index].items()),'Remaining job command differs from registered frozen configuration')
        require(shared.read_json(remaining_dir/('job_%03d_validation.json'%index))==job,'Remaining slot validation changed')
        if job['status']=='failed_numerical':
            require(index==21 and job['accuracy_accepted'] is False and job['numerical_failure']=='ValueError: '+FAILURE_SIGNATURE,
                    'Unreviewed numerical failure type/slot')
            log=remaining_dir/('job_%03d.log'%index);shared.verify(log,job['log_identity'])
            failures[index]=dict(failure_type=FAILURE_SIGNATURE,
                partial_raw_archives=failure_archives(remaining_dir,remaining_remote,job,context,calibration,log.read_text(),label_refs,job['partial_raw_archives']))
    slots=compose_slots(expected,original['jobs'],remaining['jobs'],failures)
    ledger=shared.read_json(remaining_dir/'ledger_snapshot.json');controller.ledger_valid(ledger,args.plan_sha256)
    require(ledger['spent_seconds']>=original_ledger['spent_seconds'] and ledger['attempts'][:-1]==original_ledger['attempts']
            and ledger['attempts'][-1]['stage']=='confirmation_remaining' and ledger['attempts'][-1]['status']=='complete',
            'Continuation did not preserve failed history/cumulative budget')
    job_sources={i:(job,original_dir,original_remote) for i,job in enumerate(original['jobs'][:20])}
    for job in remaining['jobs']:
        if job['status']=='complete':job_sources[job['registered_job_index']]=(job,remaining_dir,remaining_remote)
    return slots,job_sources,failures,ledger,original_receipt,remaining_receipt,original_dir,remaining_dir


def raw_outcomes(path,original,corruptions,policies,Bundle):
    """Use the existing sealed Bundle reader for both flat and run-prefixed ZIPs."""
    result={p:[] for p in (('terminal',) if original else complete.POLICIES)}
    with closing(Bundle(path)) as bundle:
        for corruption in corruptions:
            names=sorted(n for n in bundle.names if n.startswith('batches/'+corruption+'/') and n.endswith('/predictions.npz'))
            selected={p:[] for p in result}
            for name in names:
                with np.load(io.BytesIO(bundle.read(name)),allow_pickle=False) as data:
                    predictions={'terminal':data['prediction']} if original else policies(data['trajectory_logits'],terminal=3)
                    for policy in result:selected[policy].append(predictions[policy]==data['label'])
            for policy in result:result[policy].append(np.concatenate(selected[policy]))
    return {p:np.asarray(rows,dtype=bool) for p,rows in result.items()}


def inspect_successes(job_sources,context,calibration,alignment,prep):
    paired,heads,lion={}, {}, []
    outcomes,raw_rows,case_rows,corruption_rows,timing_rows,caches={},[],[],[],[],{}
    for index,(job,directory,remote) in sorted(job_sources.items()):
        require(job['status']=='complete' and len(job['raw_archives'])==4,'Complete-only raw inspector received an incomplete slot')
        if index<20:require(shared.read_json(directory/('job_%03d_validation.json'%index))==job,'Original per-job validation differs')
        geometry=None;seen=set()
        for raw in job['raw_archives']:
            relative=PurePosixPath(raw['local_path']).relative_to(remote);require('..' not in relative.parts,'Unsafe raw path')
            path=directory.joinpath(*relative.parts);shared.verify(path,raw)
            require(raw['drive_path'].startswith(shared.DRIVE),'Complete raw outside owned Drive')
            config,scores,rows,traces,costs=shared.inspect_raw(path,job,'confirmation',context,calibration,False,alignment,paired,heads,lion)
            head=config['backbone'];require(head==raw['backbone'] and head not in seen,'Duplicate/foreign complete head');seen.add(head)
            require(path.name==config['run_id']+'.zip' and traces==job['traces'],'Complete raw trace/run identity differs')
            require(geometry is None or geometry==traces,'Complete heads observed different geometry');geometry=traces
            values=raw_outcomes(path,job['trial']['method']=='original3dd',context[-1],context[-2],context[7])
            for policy,correct in values.items():
                require(correct.shape==(15,140),'Incomplete selected outcomes in complete slot')
                key=job['dataset'],head,job['condition_id'],policy,job['seed']
                require(key not in outcomes,'Duplicate completed outcomes');outcomes[key]=correct
            common=dict(condition_id=job['condition_id'],seed=job['seed'],registered_job_index=index,
                        endpoint_eligible=job['condition_id']!='tuned-chamfer')
            for row in scores:case_rows.append(dict(row,**common))
            for row in rows:corruption_rows.append(dict(row,**common))
            for row in costs:timing_rows.append(dict(row,**common))
            raw_rows.append(dict(raw,**common,run_id=config['run_id'],source_manifest_sha256=config['asset_manifest_sha256'],
                                 canonical_path=shared.copy_canonical(path,prep.parents[2],config)))
        require(seen==set(complete.HEADS),'Incomplete complete-method frozen heads')
        for row in job['cache_records']:
            key=row['local_path'];require(key not in caches or caches[key]==row,'Cache changed across successful slots');caches[key]=row
    require(len(raw_rows)==4*len(job_sources),'Successful raw head count mismatch')
    return outcomes,raw_rows,case_rows,corruption_rows,timing_rows,list(caches.values()),paired,heads,lion


def read_cache_proof(args,context,original_receipt,original_dir,caches,paired,output):
    plan=context[0];receipt=shared.read_json(args.cache_receipt)
    require(receipt['status']=='verified_failure_preserved' and receipt['experimental_result_status']=='failed'
            and receipt['session']==plan['session'] and receipt['phase_plan_sha256']==args.plan_sha256
            and receipt['manifest_sha256']==plan['manifest_sha256'] and receipt['source_snapshot_sha256']==plan['source_snapshot_sha256']
            and receipt['original_confirmation_archive']==original_receipt['archive']
            and receipt['original_receipt_identity']==ORIGINAL_RECEIPT and receipt['failed_ledger_identity']==ORIGINAL_LEDGER
            and receipt['failed_job_index']==20 and receipt['numeric_failure_signature']=='ValueError: '+FAILURE_SIGNATURE
            and receipt['physical_cache_count']==180 and receipt['physical_cache_complete'] is True
            and receipt['local_and_drive_original_archive_verified'] is True and receipt['mismatch_detected'] is False,
            'Failure-aware physical cache preservation proof missing/changed')
    directory=output/'cache_evidence'
    state=extract_evidence(args.cache_archive,receipt['archive'],directory)
    require(state['status']=='failed_attempt_cache_bytes_preserved' and state['experimental_result_status']=='failed'
            and state['session']==plan['session'] and state['phase_plan_sha256']==args.plan_sha256
            and state['manifest_sha256']==plan['manifest_sha256'] and state['source_snapshot_sha256']==plan['source_snapshot_sha256']
            and state['original_confirmation_archive']==original_receipt['archive']
            and state['original_receipt_identity']==ORIGINAL_RECEIPT and state['failed_ledger_identity']==ORIGINAL_LEDGER
            and state['failed_job_index']==20 and state['completed_jobs']==20 and state['unrun_job_indices']==[21,22,23]
            and state['physical_cache_count']==180 and state['per_seed_cache_count']=={'0':60,'1':60,'2':60},
            'Cache preservation state relabelled as completed research')
    integrity=state['integrity_verification']
    require(integrity==receipt['integrity_verification'] and integrity['status']=='verified_preserved_scope'
            and integrity['CPU_only'] is True and integrity['cuda_initialized'] is False
            and integrity['incomplete_job20_final_state_validated'] is False
            and integrity['cache_input_tensor_post_RNG_hashes_verified']==180,
            'Cache proof must preserve failed experimental status/CPU scope')
    require((directory/'failed_phase_receipt.json').read_bytes()==args.original_receipt.read_bytes()
            and (directory/'failed_campaign_ledger.json').read_bytes()==args.original_ledger.read_bytes()
            and (directory/'job_020.log').read_bytes()==(original_dir/'job_020.log').read_bytes(),
            'Cache proof changed original failure evidence')
    for supplied,name in ((args.plan,'campaign_plan.json'),(args.freeze,args.freeze.name),(args.development_acceptance,args.development_acceptance.name)):
        require((directory/name).read_bytes()==supplied.read_bytes(),'Cache proof changed original plan/freeze')
    records=shared.read_json(directory/'cache_records.json')
    require(len(records)==180 and len({r['local_path'] for r in records})==180,'Cache proof does not contain180 distinct NPZs')
    proof_by_path={r['local_path']:r for r in records}
    require({r['local_path']:r for r in caches}==proof_by_path,'Cache proof differs from consistent union of completed worker traces')
    metadata=shared.read_json(directory/'cache_metadata_validation.json')
    require(len(metadata)==180 and len({r['path'] for r in metadata})==180 and {r['path'] for r in metadata}==set(proof_by_path)
            and all(r['status']=='tensor_input_post_rng_verified_CPU' for r in metadata),'Cache CPU metadata proof incomplete')
    physical=complete.validate_physical_caches(records,paired,[directory])
    require(all(r['local_verified'] for r in physical),'Physical180NPZs absent: no partial-confirmation acceptance')
    return receipt,physical


def run(args):
    output=args.output.resolve();require(not output.exists(),'Fresh failure-aware analysis output required')
    output.mkdir(parents=True)
    context,calibration,conditions=preparation(args,output)
    plan,controller=context[:2]
    alignment,label_proofs=complete.label_references(context,args.modelnet_labels)
    slots,sources,failures,preseal,original_receipt,remaining_receipt,original_dir,remaining_dir=read_phases(
        args,context,calibration,conditions,alignment,output)
    ledger=shared.read_json(args.final_ledger);controller.ledger_valid(ledger,args.plan_sha256)
    require(ledger['attempts']==preseal['attempts'] and ledger['classifier_contracts']==preseal['classifier_contracts']
            and ledger['lion_state']==preseal['lion_state'] and ledger['spent_seconds']>=preseal['spent_seconds']
            and ledger['spent_seconds']==remaining_receipt['final_campaign_spent_seconds']<=19800,
            'Post-seal cumulative ledger/receipt conflict')
    outcomes,raw_rows,case_rows,corruption_rows,timing_rows,caches,paired,heads,lion=inspect_successes(sources,context,calibration,alignment,args.prep)
    require(lion==[calibration['lion_state_before']] and ledger['lion_state']==lion[0],'LION changed from accepted calibration/campaign ledger')
    complete.validate_classifier_contracts(heads,ledger)
    for index,failure in failures.items():
        dataset=slots[index]['dataset']
        for row in failure['partial_raw_archives']:
            contract=ledger['classifier_contracts'][dataset+'/'+row['backbone']]
            require(row['lion_state']==lion[0] and row['classifier_state_sha256']==contract['state_sha256']
                    and row['checkpoint_sha256']==contract['checkpoint_sha256'],'Failed raw model contract differs from genuine completed jobs')
            for prefix in row['completed_prefix']:
                key=slots[index]['seed'],dataset,prefix['corruption'],prefix['batch']
                require(key in paired,'Failed prefix has no completed common-preparation producer')
                shared.same_preparation(paired[key],prefix)
    cache_receipt,physical=read_cache_proof(args,context,original_receipt,original_dir,caches,paired,output)
    names=[r['id'] for r in conditions];eligible=endpoint_conditions(slots,names)
    require(eligible==['spectral-candidate','matched-chamfer','original3dd'],'All three primary complete methods required afterremaining slot recording')
    main_raw=[r for r in raw_rows if r['endpoint_eligible']]
    require(len(main_raw)==72 and sum(s['status']=='complete' and s['condition_id'] in eligible for s in slots)==18,
            'Require18 complete-method jobs/72headZIPs, never failed tuned-C fallback')
    main_outcomes={k:v for k,v in outcomes.items() if k[2] in eligible}
    cases,contrasts=complete.paired_tables(main_outcomes,context[-1],['matched-chamfer','original3dd'])
    corruption_deltas=[]
    for comparator in ('matched-chamfer','original3dd'):
        policy='terminal' if comparator=='original3dd' else 'probability_mean'
        for seed in complete.SEEDS:
            for dataset in complete.DATASETS:
                for head in complete.HEADS:
                    left=main_outcomes[(dataset,head,'spectral-candidate','probability_mean',seed)]
                    right=main_outcomes[(dataset,head,comparator,policy,seed)]
                    for i,corruption in enumerate(context[-1]):
                        delta=100.*float((left[i].astype(float)-right[i].astype(float)).mean())
                        corruption_deltas.append(dict(comparator=comparator,seed=seed,dataset=dataset,backbone=head,corruption=corruption,
                            delta_pp=delta,corrected_events=int((left[i]&~right[i]).sum()),harmed_events=int((~left[i]&right[i]).sum()),
                            sign='positive' if delta>0 else 'negative' if delta<0 else 'tied'))
    costs=[dict(r,endpoint_eligible=r['condition_id'] in eligible) for r in complete.cost_table(timing_rows)]
    for filename,rows in [('registered_slot_status.csv',slots),
                          ('case_policy.csv',[r for r in case_rows if r['endpoint_eligible']]),
                          ('corruption_policy.csv',[r for r in corruption_rows if r['endpoint_eligible']]),
                          ('successful_tuned_diagnostics.csv',[r for r in case_rows if not r['endpoint_eligible']]),
                          ('successful_tuned_corruption_diagnostics.csv',[r for r in corruption_rows if not r['endpoint_eligible']]),
                          ('balanced_policy.csv',complete.balanced_table(main_outcomes,eligible,context[-1])),
                          ('paired_cases.csv',cases),('paired_contrasts.csv',contrasts),
                          ('corruption_contrasts.csv',corruption_deltas),('costs.csv',costs)]:
        shared.write_csv(output/filename,rows)
    unavailable=unavailable_endpoints(slots,names)
    shared.write_json(output/'unavailable_endpoints.json',unavailable)
    shared.write_json(output/'raw_archives.json',raw_rows)
    shared.write_json(output/'failed_raw_provenance.json',{str(i):row for i,row in failures.items()})
    shared.write_json(output/'cache_validation.json',physical)
    failed_indices=[s['registered_job_index'] for s in slots if s['status']=='failed_numerical']
    require(len(raw_rows) in (88,92) and len(raw_rows)==4*(24-len(failed_indices)),'Composite complete/failed head counts differ')
    validation=dict(status='accepted_partial_confirmation_with_numerical_failure',stage='confirmation_partial',frozen_stage='confirmation',
        phase_plan_sha256=args.plan_sha256,session=plan['session'],manifest_sha256=plan['manifest_sha256'],source_snapshot_sha256=plan['source_snapshot_sha256'],
        original_failed_phase_archive=original_receipt['archive'],remaining_phase_archive=remaining_receipt['archive'],cache_evidence_archive=cache_receipt['archive'],
        original_receipt_identity=ORIGINAL_RECEIPT,original_failed_ledger_identity=ORIGINAL_LEDGER,final_ledger_identity=shared.identity(args.final_ledger),
        reviewed_recovery_plan=shared.identity(args.reviewed_recovery_plan),frozen_selection=shared.identity(args.freeze),
        development_acceptance=shared.identity(args.development_acceptance),calibration_acceptance=shared.identity(output/'calibration_validation.json'),
        frozen_conditions=names,analysis_complete_conditions=eligible,unavailable_endpoints=unavailable,registered_jobs=24,
        completed_jobs=len(sources),failed_registered_jobs=failed_indices,original_unrun_slots=[21,22,23],final_unrun_slots=[],
        complete_raw_archives=len(raw_rows),partial_failed_raw_archives=4*len(failed_indices),
        source_inventory_recovered_from_snapshot=any(row['source_inventory']['source_inventory_recovered_from_snapshot'] for f in failures.values() for row in f['partial_raw_archives']),
        missing_original_executed_source_inventory_keys=sorted({key for f in failures.values() for row in f['partial_raw_archives'] for key in row['source_inventory']['missing_original_inventory_keys']}),
        failed_source_inventory_recovery_proofs=[dict(registered_job_index=index,backbone=row['backbone'],**row['source_inventory']) for index,f in failures.items() for row in f['partial_raw_archives']],
        primary_complete_jobs=18,primary_complete_raw_archives=72,physical_cache_files=180,physically_verified_cache_files=180,physical_cache_complete=True,
        n_original_object_clusters=280,selection_policy='probability_mean',spectral_family=conditions[0]['trial']['family'],
        label_references=label_proofs,campaign_spent_seconds=ledger['spent_seconds'],full_benchmark=False,prior_benchmark_exposure=True,
        primary_contrasts=[r for r in contrasts if r['scope']=='all15' and r['seed'] is None],
        limits='Failed tuned-C full3seed endpoint/CI is unavailable; no2seed averaging,dataset omission,zero substitution,retuned fallback or rerun of failedslot20. Successful tunedrows are diagnostics only. Primary endpoints useonly3complete frozenmethods withthe unchanged280object common-drawbootstrap across3fixedseeds. Originalcontrastchangeshost/batching/readout. Conditional/multiplecontrast/macro14/readoutintervals are not equivalence or universal-superiority evidence. Shared reconstructioncountedonce; classifiercostssum/peakmax; cachedinstrumentedcomparisons are not deploymentlatency. Campaignspentincludes failedwork andsealing. Any solemissing shared_trajectory executed-inventory entry is explicitlyderivedfrom exactarchivedsealedsourcebytes plusknownfailuretrace forfailedprovenanceonly; originalinventorygapandrawfailedstatusremain. Originalfailedexperiment remainsfailed.')
    shared.write_json(output/'validation.json',validation)
    shared.write_json(output/'accepted_partial_sensitivity_confirmation.json',dict(status=validation['status'],stage='confirmation_partial',
        phase_plan_sha256=args.plan_sha256,original_failed_phase_archive=original_receipt['archive'],remaining_phase_archive=remaining_receipt['archive'],
        local_validation=shared.identity(output/'validation.json'),physical_cache_complete=True,n_original_object_clusters=280,
        failed_registered_jobs=failed_indices,analysis_complete_conditions=eligible,frozen_conditions=names,
        tuned_chamfer_three_seed_endpoint=None,campaign_spent_seconds=ledger['spent_seconds']))
    print(json.dumps(dict(status=validation['status'],output=str(output),complete_raw_archives=len(raw_rows),failed_registered_jobs=failed_indices,
                          primary_complete_raw_archives=72,physical_cache_complete=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prep','old-prep','plan','original-phase-archive','original-receipt','original-ledger','remaining-phase-archive','remaining-receipt',
                 'reviewed-recovery-plan','cache-archive','cache-receipt','final-ledger','calibration-archive','calibration-receipt',
                 'development-archive','development-receipt','development-acceptance','freeze','modelnet-labels','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--reviewed-recovery-plan-sha256',required=True)
    run(parser.parse_args())
