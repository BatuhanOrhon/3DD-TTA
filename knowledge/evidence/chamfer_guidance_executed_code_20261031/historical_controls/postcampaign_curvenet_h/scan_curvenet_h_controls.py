"""Separate postcampaign Scan CurveNet/H controls; nothing executes on import."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import zipfile

PURPOSE='scan_curvenet_h_exploratory_posthoc'
SESSION='tta-chamfer-guidance-variants-a100-20261030-r1'
HEADS=['dgcnn','curvenet','pointnext','pointmae']
STAGES=('calibration','heldout','timing')


def require(value,message):
    if not value:raise ValueError(message)


def identity(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return dict(bytes=Path(path).stat().st_size,sha256=digest.hexdigest())


def read_reference(reference):
    require(identity(reference['path'])=={k:reference[k] for k in ('bytes','sha256')},'Input identity mismatch: '+reference['path'])
    return json.loads(Path(reference['path']).read_text())


def owned(path,root):
    path,root=Path(path),Path(root)
    require(path.is_absolute() and '..' not in path.parts and not path.is_symlink()
            and not any(p.is_symlink() for p in path.parents),'Unsafe write destination')
    require(path!=root and root in path.parents,'Write outside new owned root')
    return path


def fresh_directory(path,root):
    path=owned(path,root);path.mkdir(parents=True,exist_ok=False);return path


def validate_plan(plan):
    require(plan.get('schema_version')==1 and plan.get('purpose')==PURPOSE and plan.get('dataset')=='scanobjectnn_c'
            and plan.get('session')==SESSION and plan.get('cap_seconds')==4200,'Unexpected scope/session/cap')
    for new,old in (('research_root','drive_root'),('research_root','parent_root'),
                    ('research_root','parent_drive_root'),('drive_root','parent_root'),('drive_root','parent_drive_root')):
        a,b=Path(plan[new]),Path(plan[old])
        require(a.is_absolute() and b.is_absolute() and '..' not in a.parts and '..' not in b.parts
                and a!=b and a not in b.parents and b not in a.parents,'New and original roots must be disjoint')
    owned(plan['asset_root'],plan['research_root']);owned(plan['source_root'],plan['research_root'])
    owned(plan['calibration_report'],plan['research_root'])
    require(plan['cache_policy']=='fresh_C_then_exact_H_replay','Unreviewed prepared-state policy')
    require(set(plan['stage_estimates_seconds'])==set(STAGES) and plan['reserve_seconds']==600
            and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in plan['stage_estimates_seconds'].values()),'Whole-phase forecast/reserve required')


def load_helpers(plan):
    modules=[]
    for reference in plan['helper_dependencies'].values():
        require(identity(reference['path'])=={k:reference[k] for k in ('bytes','sha256')},'Unreviewed helper dependency')
    for name in ('house','streamer'):
        reference=plan['helpers'][name]
        require(identity(reference['path'])=={k:reference[k] for k in ('bytes','sha256')},'Unreviewed helper source')
        sys.path.insert(0,str(Path(reference['path']).parent))
        spec=importlib.util.spec_from_file_location('scan_curvenet_h_'+name,reference['path'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    h=modules[0];h.ROOT=Path(plan['research_root']);h.DRIVE=Path(plan['drive_root']);h.SESSION=SESSION
    h.ASSETS=Path(plan['asset_root']);h.PREPARED=h.ASSETS.parent
    h.MANIFEST_SHA=plan['manifest']['sha256'];h.SOURCE_SHA=plan['source_snapshot']['sha256']
    modules[1].h=h
    return h,modules[1].stream


def remaining_budget(plan,ledger,expected_remaining,now=None):
    elapsed=max(ledger['active_seconds'],(time.time() if now is None else now)-ledger['started_epoch'])
    left=plan['cap_seconds']-elapsed
    require(math.isfinite(left) and left>=1.1*expected_remaining+plan['reserve_seconds'],'New4200s whole-phase projection exhausted')
    return left-plan['reserve_seconds']


def build_jobs(plan,stage,coefficient=None,report_sha=None):
    validate_plan(plan);require(stage in STAGES,'Unknown new phase')
    if stage!='calibration':
        require(type(coefficient) in (int,float) and math.isfinite(coefficient) and coefficient>0
                and isinstance(report_sha,str) and re.fullmatch('[0-9a-f]{64}',report_sha),'Frozen real CurveNet H coefficient/report required')
    jobs=[]
    for repeat in (range(3) if stage=='timing' else (0,)):
        arms=['baseline','semantic_h_curvenet'] if repeat%2==0 else ['semantic_h_curvenet','baseline']
        for seed in ((0,1,2) if stage=='heldout' else (0,)):
            for arm in (['calibration'] if stage=='calibration' else arms):
                relative=Path('results')/stage/('repeat%d'%repeat)/('seed%d'%seed)/arm
                result=Path(plan['research_root'])/relative;drive=Path(plan['drive_root'])/relative
                heads=['curvenet'] if stage=='calibration' else HEADS
                selection='heldout' if stage=='heldout' else 'development'
                script='calibrate_scan_curvenet_entropy.py' if stage=='calibration' else 'eval_chamfer_semantic.py' if arm!='baseline' else 'eval_multibackbone_guidance.py'
                command=[plan['python'],'-u',str(Path(plan['source_root'])/'scripts'/script)]
                if stage=='calibration':command+=['run']
                command+=['--dataset','scanobjectnn_c','--dataset-root',plan['data_root'],'--label-path',plan['label_path'],
                    '--asset-root',plan['asset_root'],'--asset-manifest-sha256',plan['manifest']['sha256'],
                    '--result-root',str(result),'--drive-root',str(drive),'--seed',str(seed),'--batch-size','70',
                    '--backbones',*heads,'--method','gsd','--component-arm','chamfer','--trajectory-observe',
                    '--eta','0.01','--gamma','0.01','--p','0.95','--chamfer-weight','1','--spectral-weight','0',
                    '--pointmae-config',plan['pointmae_config'],'--pointmae-checkpoint',plan['pointmae_checkpoint'],
                    '--index-manifest',plan['selections'][selection]['path']]
                if stage=='heldout':command+=['--paired-state-root',str(Path(plan['research_root'])/'paired_states'/'heldout'/('seed%d'%seed))]
                if arm=='semantic_h_curvenet':command+=['--semantic-guidance-head','curvenet','--semantic-kind','marginal_entropy',
                    '--semantic-weight',str(coefficient),'--calibration-report',plan['calibration_report'],'--calibration-report-sha256',report_sha]
                command+=['--corruptions','gaussian','background'] if stage=='timing' else ['--corruptions','all']
                jobs.append(dict(stage=stage,arm=arm,seed=seed,repeat=repeat,heads=heads,result_root=str(result),drive_root=str(drive),command=command))
    return jobs


def extract_assets(plan,h):
    old=read_reference(plan['base_manifest']);new=read_reference(plan['manifest'])
    require(set(old['files'])<=set(new['files']),'Original asset names must remain available')
    changed={name for name in new['files'] if new['files'][name]!=old['files'].get(name)}
    require(changed==set(plan['allowed_asset_changes']) and all(n=='source_snapshot.zip' or n.startswith('source_snapshot/')
            or n in ('source_preparation.json','matched_source_manifest.json') for n in changed),'Unreviewed non-source asset change')
    assets=fresh_directory(plan['asset_root'],plan['research_root'])
    h.copy_checked(plan['manifest']['path'],assets/'manifest.json',plan['manifest'])
    h.copy_checked(plan['source_snapshot']['path'],assets/'source_snapshot.zip',plan['source_snapshot'])
    h.extract_checked(assets/'source_snapshot.zip',assets,overlay=True,declared=new['files'])
    for name,expected in new['files'].items():
        path=assets/name
        if not path.exists():
            require(old['files'].get(name)==expected,'New source payload is incomplete: '+name)
            original=Path(plan['base_asset_root'])/name;h.verify(original,expected)
            path.parent.mkdir(parents=True,exist_ok=True);os.link(original,path)
        h.verify(path,expected)
    for name,expected in old['files'].items():h.verify(Path(plan['base_asset_root'])/name,expected)
    return new


def verify_parent(plan):
    accepted=read_reference(plan['parent_collection'])
    require(accepted.get('status')=='accepted_original_collection','Pinned original local collection acceptance required')
    ledger=read_reference(accepted['ledger'])
    require(all(ledger['stages'].get(stage,{}).get('status')=='complete' for stage in ('confirmation','timing')),
            'Original confirmation/timing are not complete')
    expected={}
    for stage in ('confirmation','timing'):
        receipt=read_reference(accepted[stage+'_receipt'])
        require(receipt['state']['status']=='complete' and len(receipt['state']['jobs'])==30,'Original phase coverage incomplete')
        for job in receipt['state']['jobs']:
            require(len(job['archive_identities'])==4,'Original four-head archive proof missing')
            for record in job['archive_identities']:expected[record['path']]={k:record[k] for k in ('bytes','sha256')}
    require(len(expected)==240 and len(ledger['cache_identities'])==214,'Original240 archives/214 caches required')
    expected.update(ledger['cache_identities'])
    physical=read_reference(accepted['local_raw_manifest'])
    require(physical.get('status')=='all_requested_raw_bytes_verified','Local physical byte verification missing')
    observed={}
    for record in physical['records']:
        current={k:record[k] for k in ('bytes','sha256')}
        require(record['path'] not in observed or observed[record['path']]==current,'Conflicting physical records')
        observed[record['path']]=current
    require(all(observed.get(path)==value for path,value in expected.items()),'Original local physical archive/cache proof incomplete')
    return dict(status='verified_predecessor',original_ledger=accepted['ledger'],verified_physical_records=len(expected))


def check_stage(stage,ledger):
    require(stage not in ledger['stages'],'Stage already attempted; reviewed recovery needed')
    number=STAGES.index(stage)
    require(number==0 or ledger['stages'].get(STAGES[number-1],{}).get('status')=='complete','Prior new-scope stage is incomplete')


def persist_caches(plan,ledger,h):
    for path in sorted((Path(plan['research_root'])/'paired_states').rglob('*.npz')):
        expected=h.identity(path);previous=ledger.setdefault('cache_identities',{}).get(str(path))
        require(previous is None or previous==expected,'New-scope physical cache changed')
        target=owned(Path(plan['drive_root'])/path.relative_to(plan['research_root']),plan['drive_root'])
        if previous is None:
            h.verify(target,expected) if target.exists() else h.copy_checked(path,target,expected)
            ledger['cache_identities'][str(path)]=expected
        require(target.exists(),'Verified new-scope Drive cache missing')


def save_ledger(plan,ledger):
    raw=json.dumps(ledger,indent=2,sort_keys=True,allow_nan=False)+'\n'
    for root in (plan['research_root'],plan['drive_root']):
        path=owned(Path(root)/'postcampaign_ledger.json',root);path.parent.mkdir(parents=True,exist_ok=True)
        pending=path.with_suffix('.pending.json');pending.write_text(raw);pending.replace(path)
    require(identity(Path(plan['research_root'])/'postcampaign_ledger.json')==identity(Path(plan['drive_root'])/'postcampaign_ledger.json'),
            'New ledger copies differ')


def load_ledger(plan,plan_sha):
    local,remote=[Path(root)/'postcampaign_ledger.json' for root in (plan['research_root'],plan['drive_root'])]
    if not local.exists() and not remote.exists():
        return dict(plan_sha256=plan_sha,session=SESSION,cap_seconds=4200,started_epoch=time.time(),active_seconds=0.,active=None,stages={})
    require(local.exists() and remote.exists() and identity(local)==identity(remote),'New ledger copies conflict; inspect before recovery')
    ledger=json.loads(local.read_text())
    require(ledger['plan_sha256']==plan_sha and ledger['cap_seconds']==4200 and ledger['session']==SESSION and ledger['active'] is None,
            'Foreign or unresolved new-scope ledger')
    return ledger


def verify_control_sources(plan,required=Path(__file__).name):
    require(required in plan['control_sources'],'Executing control is not frozen')
    for name,reference in plan['control_sources'].items():
        require(Path(name).name==name and Path(reference['path']).name==name,'Unsafe control source name')
        require(identity(reference['path'])=={k:reference[k] for k in ('bytes','sha256')},'Control source identity mismatch')


def verify_inputs(plan,h,stream):
    validate_plan(plan);verify_control_sources(plan);parent=verify_parent(plan)
    authorization=read_reference(plan['authorization'])
    require(authorization.get('status')=='user_authorized_postcampaign_scope' and authorization.get('purpose')==PURPOSE
            and authorization.get('session')==SESSION and authorization.get('cap_seconds')==4200
            and bool(authorization.get('user_instruction','').strip()),'Retained existing user instruction/scope is required')
    source=read_reference(plan['source_receipt'])
    require(source['status']=='sealed_CPU_only' and source['project_source_commit']==plan['source_commit']
            and source['manifest']=={k:plan['manifest'][k] for k in ('bytes','sha256')}
            and source['source_snapshot']=={k:plan['source_snapshot'][k] for k in ('bytes','sha256')},'Source seal differs')
    # This inherited gate only reads the existing ten-cell/native receipts; never call old configure/ledger writers.
    setup=stream.__globals__['common'].setup_gate()
    return dict(parent=parent,source=source,setup=setup)


def cpu_preflight(plan,h,source):
    sys.path.insert(0,plan['source_root'])
    from scripts import eval_multibackbone_source as reader
    from scripts import eval_multibackbone_guidance as runner
    require(reader.REPO==Path(plan['source_root']),'Noncanonical new source import')
    manifest=reader.read_manifest(Path(plan['asset_root']),plan['manifest']['sha256'])
    require(manifest['source_snapshot_sha256']==plan['source_snapshot']['sha256'],'New source archive identity differs')
    for key in ('lion_checkpoint','lion_config'):
        entry=manifest['guidance'][key];h.verify(entry['path'],entry)
    for name,expected in source['source_files'].items():h.verify(Path(plan['source_root'])/name,expected)
    ids={stage:read_reference(reference) for stage,reference in plan['selections'].items()}
    for stage,count in (('development',70),('heldout',140)):
        values=ids[stage];indices=values['indices']
        require(values['dataset']=='scanobjectnn_c' and values['n_examples']==581 and values['stage']==stage
                and len(indices)==count and indices==sorted(set(indices)),'Fixed Scan selection differs')
    require(not set(ids['development']['indices']) & set(ids['heldout']['indices']),'Development and reused heldout overlap')
    command=build_jobs(plan,'calibration')[0]['command'];options=command[4:]
    config=runner.configurations(['--preview','--chamfer-experiment','semantic','--semantic-guidance-head','curvenet',
                                  '--semantic-kind','marginal_entropy','--semantic-weight','0',*options])[0]
    import torch
    from unittest.mock import patch
    require(not torch.cuda.is_initialized(),'CPU deployment unexpectedly initialized CUDA')
    hook=reader._declared_asset(Path(plan['asset_root']),manifest,manifest['guidance']['native_bootstrap'])
    spec=importlib.util.spec_from_file_location('scan_curvenet_h_native_cpu',hook);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=module.bootstrap(Path(plan['asset_root']));require(native['no_build'],'New native build forbidden')
    with patch.object(torch.nn.Module,'cuda',lambda model,*a,**k:model), \
         patch.object(torch.Tensor,'cuda',side_effect=RuntimeError('GPU tensor forbidden in CPU preflight')), \
         patch.object(torch.cuda,'_lazy_init',side_effect=RuntimeError('CUDA initialization forbidden')), \
         patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Forward forbidden in CPU preflight')):
        model=reader.load_classifier(config,manifest)
    require(not torch.cuda.is_initialized() and all(not p.requires_grad and p.device.type=='cpu' for p in model.parameters())
            and all(b.device.type=='cpu' for b in model.buffers())
            and not any(m.training for m in model.modules()),'New CurveNet is not CPU/frozen/eval')
    return dict(status='complete',cuda_initialized=False,strict_teacher_load=True,forward_executed=False,
                classifier_state_sha256=reader.state_hash(model),checkpoint_load=config['checkpoint_load'],native=native)


def validate_preparation(job,diagnostic):
    expected='cached_replay' if job['stage']=='heldout' and job['arm']=='semantic_h_curvenet' else 'fresh_encode'
    require(diagnostic.get('preparation_mode')==expected,'Actual top-level preparation mode differs from requested fresh/replay policy')


def verify_heldout_pairs(records,load,paired_identity):
    loaded={}
    for record in records:
        for archive in record['archive_identities']:
            value=load(archive['path']);config=value['config'];key=(config['seed'],config['backbone'],value['arm'])
            require(key not in loaded and config['seed']==record['seed'] and value['arm']==record['arm'],
                    'Duplicate/misidentified heldout pair')
            loaded[key]=value
    expected={(seed,head,arm) for seed in range(3) for head in HEADS for arm in ('baseline','semantic_h_curvenet')}
    require(set(loaded)==expected,'Heldout requires all12 seed/head C/H pairs')
    batches=0
    for seed in range(3):
        for head in HEADS:
            baseline,treatment=[loaded[(seed,head,arm)] for arm in ('baseline','semantic_h_curvenet')]
            # Existing accuracy gate retains full initial tensor/RNG/source/native identity;
            # native denoising trajectories themselves need not be bitwise identical.
            paired_identity(baseline,treatment)
            require(set(baseline['cells'])==set(treatment['cells']),'Unpaired heldout corruption scope')
            for corruption,cell in baseline['cells'].items():
                other=treatment['cells'][corruption]
                require(list(cell['indices'])==list(other['indices']) and list(cell['labels'])==list(other['labels']),
                        'Heldout indexed IDs/labels differ')
            batches+=len(baseline['batches'])
    return dict(status='strict_initial_state_and_indexed_labels_paired',paired_seed_heads=12,paired_head_batches=batches)


def validate_job(job,plan,h,analyze_bundle,Bundle):
    paths=sorted(Path(job['result_root']).glob('*/*/*.zip'))
    require(len(paths)==len(job['heads']),'Missing raw head archives')
    records=[];heads=set();ids=read_reference(plan['selections']['heldout' if job['stage']=='heldout' else 'development'])['indices']
    for path in paths:
        with zipfile.ZipFile(path) as archive:h.checked_members(archive)
        report=analyze_bundle(path,allow_partial=True);require(report['selection_complete'],'Truncated declared selection')
        bundle=Bundle(path)
        try:
            config=bundle.json('config.json');heads.add(config['backbone']);protocol=config['guidance_protocol']
            require(config['dataset']=='scanobjectnn_c' and config['seed']==job['seed'] and config['selected_original_ids']==ids
                    and protocol['loss_weights']==dict(chamfer=1.,spectral=0.,invariant=0.) and protocol['skip_inactive_components'],
                    'Job/ID/Chamfer-only identity differs')
            if job['arm']=='baseline':require(not config.get('chamfer_experiment'),'Baseline unexpectedly has an intervention')
            else:require(config['semantic_guidance']['head']=='curvenet' and config['semantic_guidance']['kind']=='marginal_entropy',
                         'Wrong semantic teacher/objective')
            for name in bundle.names:
                if name.startswith('batches/') and name.endswith('/diagnostics.json'):
                    diagnostic=bundle.json(name);validate_preparation(job,diagnostic);metrics=diagnostic['guidance_metrics']
                    require(metrics.get('graph_calls')==metrics.get('spectral_loss_calls')==0,'Spectral/graph execution forbidden')
            target=Path(job['drive_root'])/config['dataset']/config['method']/config['run_id']/'final.zip'
            expected=h.identity(path);h.verify(target,expected);records.append(dict(path=str(path),drive_path=str(target),**expected))
        finally:bundle.close()
    require(heads==set(job['heads']),'Head coverage differs')
    return records


def preserve_controls(plan,plan_path,output):
    shutil.copyfile(plan_path,output/'phase_plan.json')
    for reference in [*plan['control_sources'].values(),*plan['helpers'].values(),*plan['helper_dependencies'].values()]:
        target=output/Path(reference['path']).name
        if target.exists():require(identity(target)=={k:reference[k] for k in ('bytes','sha256')},'Conflicting source basenames')
        else:shutil.copyfile(reference['path'],target)
    for name in ('source_receipt','authorization','parent_collection'):
        shutil.copyfile(plan[name]['path'],output/(name+'.json'))
    parent=read_reference(plan['parent_collection'])
    for name in ('ledger','confirmation_receipt','timing_receipt','local_raw_manifest'):
        read_reference(parent[name]);shutil.copyfile(parent[name]['path'],output/('parent_'+name+'.json'))
    read_reference(plan['base_manifest']);shutil.copyfile(plan['base_manifest']['path'],output/'parent_asset_manifest.json')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--stage',choices=('deploy',*STAGES),required=True);args=parser.parse_args(argv)
    require(identity(args.plan)['sha256']==args.plan_sha256,'Plan SHA differs');plan=json.loads(args.plan.read_text())
    validate_plan(plan);h,stream=load_helpers(plan);inputs=verify_inputs(plan,h,stream)
    stage=args.stage;output=fresh_directory(Path(plan['research_root'])/'phases'/stage,plan['research_root'])
    drive=fresh_directory(Path(plan['drive_root'])/'phase-evidence'/stage,plan['drive_root'])
    state=dict(status='running',stage=stage,purpose=PURPOSE,plan_sha256=args.plan_sha256,jobs=[],exploratory_posthoc=True)
    ledger=None;started=time.monotonic();prior=0.;attempted=[]
    def account():
        if ledger is not None:
            ledger['active_seconds']=prior+time.monotonic()-started;save_ledger(plan,ledger)
    try:
        preserve_controls(plan,args.plan,output)
        if stage=='deploy':
            base=read_reference(plan['base_manifest']);require(len(base['files'])==656,'Expected immutable original656 assets')
            extract_assets(plan,h);proof=cpu_preflight(plan,h,inputs['source'])
            for key,name in (('manifest','asset_manifest.json'),('source_snapshot','source_snapshot.zip')):
                h.copy_checked(plan[key]['path'],output/name,plan[key])
            (output/'cpu_preflight.json').write_text(json.dumps(proof,indent=2)+'\n');state['cpu_preflight']=proof
        else:
            deployment=json.loads((Path(plan['research_root'])/'deploy_receipt.json').read_text())
            require(deployment['state']['status']=='complete' and deployment['state']['plan_sha256']==args.plan_sha256,'Accepted new CPU deployment required')
            for key in ('local_path','drive_path'):h.verify(deployment['archive'][key],deployment['archive'])
            for name,expected in inputs['source']['source_files'].items():h.verify(Path(plan['source_root'])/name,expected)
            ledger=load_ledger(plan,args.plan_sha256);check_stage(stage,ledger);prior=ledger['active_seconds']
            if stage!='calibration':
                previous=STAGES[STAGES.index(stage)-1];accepted=json.loads((Path(plan['research_root'])/(previous+'_receipt.json')).read_text())
                require(accepted['state']['status']=='complete','Previous stage receipt failed')
                for key in ('local_path','drive_path'):h.verify(accepted['archive'][key],accepted['archive'])
            sys.path.insert(0,plan['source_root'])
            from scripts.analyze_scan_trajectory import analyze_bundle,Bundle
            from scan_curvenet_entropy_report import KEY,validate_report
            coefficient=report_sha=None
            if stage!='calibration':
                h.verify(plan['calibration_report'],ledger['calibration_report']);report=validate_report(json.loads(Path(plan['calibration_report']).read_text()))
                coefficient=report['coefficients'][KEY];report_sha=ledger['calibration_report']['sha256']
            jobs=build_jobs(plan,stage,coefficient,report_sha)
            (output/'executed_commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
            require(all(not Path(j['result_root']).exists() and not Path(j['drive_root']).exists() for j in jobs),'Fresh job destinations required')
            if stage=='heldout':require(not (Path(plan['research_root'])/'paired_states').exists(),'Fresh new-scope C caches required')
            for number,job in enumerate(jobs):
                persist_caches(plan,ledger,h)
                remaining=plan['stage_estimates_seconds'][stage]*(len(jobs)-number)/len(jobs)+sum(plan['stage_estimates_seconds'][s] for s in STAGES[STAGES.index(stage)+1:])
                timeout=remaining_budget(plan,ledger,remaining);ledger['active']=dict(stage=stage,job=number,command=job['command']);account()
                def began(pid):ledger['active']['pid']=pid;account()
                attempted.append(job);before=time.monotonic();stream(job['command'],output/('job%02d.log'%number),plan['source_root'],timeout,began)
                records=validate_job(job,plan,h,analyze_bundle,Bundle)
                state['jobs'].append(dict(job,elapsed_seconds=time.monotonic()-before,archive_identities=records));persist_caches(plan,ledger,h);account()
                (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
            if stage=='heldout':
                from scripts.analyze_chamfer_variants import _load,_paired_identity
                state['accuracy_pairing']=verify_heldout_pairs(state['jobs'],_load,_paired_identity)
            if stage=='calibration':
                command=[plan['python'],'-u',str(Path(plan['source_root'])/'scripts/calibrate_scan_curvenet_entropy.py'),'summarize',
                         state['jobs'][0]['archive_identities'][0]['path'],'--output',plan['calibration_report']]
                (output/'summarize_command.json').write_text(json.dumps(command,indent=2)+'\n')
                stream(command,output/'summarize.log',plan['source_root'],remaining_budget(plan,ledger,sum(plan['stage_estimates_seconds'][s] for s in STAGES[1:])))
                validate_report(json.loads(Path(plan['calibration_report']).read_text()));ledger['calibration_report']=identity(plan['calibration_report'])
                h.copy_checked(plan['calibration_report'],drive/'calibration.json',ledger['calibration_report']);shutil.copyfile(plan['calibration_report'],output/'calibration.json')
        state['status']='complete'
    except BaseException:
        state.update(status='failed',failure=traceback.format_exc())
        for job in attempted:
            if Path(job['result_root']).exists() and not any(j['result_root']==job['result_root'] for j in state['jobs']):
                shutil.copytree(job['result_root'],output/'partial_results'/job['arm']/('seed%d'%job['seed']))
        raise
    finally:
        state['elapsed_seconds']=time.monotonic()-started;(output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        if ledger is not None:ledger['active']=None;ledger['stages'][stage]=dict(status=state['status'],output=str(output));account()
        try:
            archive=h.seal(output,drive/'evidence.zip');account()
            if ledger is not None and max(ledger['active_seconds'],time.time()-ledger['started_epoch'])>4200:
                state['status']='failed_cap';ledger['stages'][stage]['status']='failed_cap';account()
            receipt=dict(state=state,archive=archive)
            (drive/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            with (Path(plan['research_root'])/(stage+'_receipt.json')).open('x') as file:file.write(json.dumps(receipt,indent=2)+'\n')
        except BaseException:
            if ledger is not None:ledger['stages'][stage]['status']='failed_delivery';account()
            raise
        require(state['status']=='complete','New stage failed; evidence preserved')
        print('SCAN_CURVENET_H_STAGE_COMPLETE '+stage,flush=True)


if __name__=='__main__':main()
