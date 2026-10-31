"""One metadata-only continuation of the original approval/clock; no auto retry."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import gpu_chamfer_phase as phase
common, h = phase.common, phase.h
SOURCE = 'scripts/eval_multibackbone_source.py'
ESTIMATES = dict(smoke_initial=480,calibration=900,smoke_active=360,development=1780,confirmation=9000,timing=1260)


def read_pinned(path, sha):
    h.require(h.identity(path)['sha256']==sha,'Pinned recovery input SHA mismatch: '+str(path))
    return json.loads(Path(path).read_text())


def read_reference(reference):
    h.verify(reference['path'],reference)
    return json.loads(Path(reference['path']).read_text())


def verify_exact_repair(old, new, old_root, new_root):
    h.require(set(old['source_files'])==set(new['source_files']),'Runtime source inventory changed')
    h.require(old['inherited_bulk_files']==new['inherited_bulk_files'],'Bulk/checkpoint identities changed')
    changed=[name for name in old['source_files'] if old['source_files'][name]!=new['source_files'][name]]
    h.require(changed==[SOURCE],'Only the producer metadata source may change')
    for name in old['source_files']:
        h.verify(Path(old_root)/name,old['source_files'][name]); h.verify(Path(new_root)/name,new['source_files'][name])
    before=(Path(old_root)/SOURCE).read_bytes(); after=(Path(new_root)/SOURCE).read_bytes()
    needle=b'        elif REPO in path.parents:'
    h.require(before.count(needle)==1 and after==before.replace(needle,b'        if REPO in path.parents:',1),
              'Recovery is not the exact reviewed elif-to-if byte patch')


def runtime_plan(original, receipt, receipt_reference, asset_root):
    plan=copy.deepcopy(original); old=original['asset_root']; new=str(asset_root)
    plan.update(asset_root=new,source_root=new+'/source_snapshot',source_receipt=receipt_reference,
                asset_manifest=dict(receipt['manifest'],path=new+'/manifest.json'),
                source_snapshot=receipt['source_snapshot'],source_commit=receipt['project_source_commit'])
    for datasets in plan['selections'].values():
        for reference in datasets.values():
            h.require(reference['path'].startswith(old+'/study/'),'Unexpected original selection path')
            reference['path']=new+reference['path'][len(old):]
    return plan  # Bulk data, outputs, calibration path, caches, seeds and coefficients stay fixed.


def verify_parity_acceptance(path, sha, original_plan_sha):
    acceptance=read_pinned(path,sha)
    h.require(acceptance.get('status')=='accepted' and acceptance.get('original_plan_sha256')==original_plan_sha
              and isinstance(acceptance.get('decision'),str) and bool(acceptance['decision'].strip()),
              'An explicit reviewed initial parity acceptance is required; pending is not acceptance')
    failed=read_reference(acceptance['failed_stage_receipt']); diagnostic=read_reference(acceptance['diagnostic_receipt'])
    h.require('zero_effect_receipt' in acceptance,'Separate native exact-zero proof is required')
    zero=read_reference(acceptance['zero_effect_receipt']); verify_zero_rows(zero['state'])
    h.require(failed['state']['status']=='failed' and failed['state']['plan_sha256']==original_plan_sha
              and len(failed['state']['jobs'])==8,'Original failed8-worker stage proof is missing')
    h.require(diagnostic.get('state',diagnostic).get('status')=='complete','Completed numeric diagnostic proof required')
    expected={p for job in failed['state']['jobs'] for p in job['archives']}
    records=acceptance['initial_archives']
    h.require(len(expected)==len(records)==32 and {r['path'] for r in records}==expected,'Original32-archive coverage changed')
    for reference in records: h.verify(reference['path'],reference)
    for receipt in (failed,diagnostic,zero):
        for key in ('local_path','drive_path'): h.verify(receipt['archive'][key],receipt['archive'])
    return acceptance


def verify_zero_rows(state):
    rows=state.get('zero_check_rows',[])
    expected={(kind,corruption,count) for kind in ('kl','marginal_entropy')
              for corruption,counts in (('gaussian',(5,8)),('background',(15,23))) for count in counts}
    h.require(state.get('status')=='complete' and len(rows)==8
              and {(r['kind'],r['corruption'],r['loop_count']) for r in rows}==expected,'Exact8 native zero checkpoints required')
    for row in rows:
        h.require(row.get('batch_count')==2 and row.get('rng_unchanged') is True
                  and row.get('leaf_grad_fields_unchanged') is True,'Native zero RNG/leaf-grad/batch guard failed')
        for component in ('local','style'):
            value=row[component]
            h.require(value.get('finite') is True and type(value.get('nonzero_count')) is int
                      and value['nonzero_count']==0 and value.get('max_abs')==0 and value.get('norm')==0,
                      'Native zero semantic contribution is not finite and exactly zero: '+component)


def residual_reserve(accounting, ledger, now=None):
    start,end=accounting['recovery_started_epoch'],accounting['accounted_until_epoch']
    consumed=end-start
    h.require(accounting.get('status')=='verified_contingency_accounting' and accounting['initial_reserve_seconds']==1800
              and all(type(v) in (int,float) and math.isfinite(v) for v in (start,end))
              and ledger['started_epoch']<=start<=end<=(time.time() if now is None else now)
              and accounting.get('timing_basis') and accounting['consumed_reserve_seconds']==consumed
              and accounting['remaining_reserve_seconds']==max(0.,1800-consumed), 'Invalid or replenished contingency accounting')
    return accounting['remaining_reserve_seconds']


def archive_references(job, paths, Bundle):
    records=[]
    for path in paths:
        bundle=Bundle(path)
        try: config=bundle.json('config.json')
        finally: bundle.close()
        remote=Path(job['drive_root'])/config['dataset']/config['method']/config['run_id']/'final.zip'
        records.append(dict(path=path,drive_path=str(remote),**h.identity(path)))
    return records  # validate_job already verified identical Drive bytes immediately beforehand.


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binding','parity-acceptance','cpu-repair-receipt','reserve-accounting'):
        parser.add_argument('--'+name,type=Path,required=True); parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--stage',choices=phase.STAGES[1:],required=True); args=parser.parse_args(argv)
    binding=read_pinned(args.binding,args.binding_sha256); original=binding['original']
    h.require(binding['status']=='prepared_metadata_recovery_not_user_approval' and binding['estimates_seconds']==ESTIMATES,'Recovery scope/compute estimates changed')
    h.verify(__file__,binding['recovery_controller'])
    h.verify(original['controller']['path'],original['controller'])
    approved=read_reference(original['plan']); commands=read_reference(original['commands']); approval=read_reference(original['approval'])
    h.require(approval['plan_sha256']==original['plan']['sha256'] and approval['commands_sha256']==original['commands']['sha256'],
              'Original user approval does not cover the original frozen commands')
    phase.verify_phase_sources(approved['phase_sources'],common.PHASES,required='gpu_chamfer_phase.py')
    old=read_reference(original['source_receipt']); revised=read_reference(binding['new_source_receipt']); scope=read_reference(binding['repair_scope'])
    common.configure(old); common.setup_gate()
    plan=runtime_plan(approved,revised,binding['new_source_receipt'],binding['new_asset_root'])
    h.require(plan==binding['runtime_plan'],'Recorded runtime binding differs from exact mechanical rewrite')
    h.require(len(old['source_files'])==296 and old['project_source_commit']==scope['parent_source_commit']
              and revised['project_source_commit']==scope['source_commit'],'Unexpected source revision/inventory')
    h.require(scope['old_source_identity']==old['source_files'][SOURCE] and scope['new_source_identity']==revised['source_files'][SOURCE]
              and scope['manifest']==revised['manifest'] and scope['source_snapshot']==revised['source_snapshot'],'Repair seal identities differ')
    verify_exact_repair(old,revised,approved['source_root'],plan['source_root'])
    h.verify(plan['asset_manifest']['path'],plan['asset_manifest']); h.verify(Path(plan['asset_root'])/'source_snapshot.zip',revised['source_snapshot'])
    old_manifest=read_reference(approved['asset_manifest']); new_manifest=read_reference(plan['asset_manifest'])
    h.require(set(old_manifest['files'])==set(new_manifest['files'])
              and {n for n in old_manifest['files'] if old_manifest['files'][n]!=new_manifest['files'][n]}
              =={'source_snapshot.zip','source_snapshot/'+SOURCE}, 'Nonmetadata asset identity changed')
    for stage in approved['selections']:
        for dataset, reference in approved['selections'][stage].items():
            h.verify(reference['path'],reference); h.verify(plan['selections'][stage][dataset]['path'],reference)
    repair=read_pinned(args.cpu_repair_receipt,args.cpu_repair_receipt_sha256); proof=repair['state']
    h.require(args.cpu_repair_receipt_sha256==binding['cpu_repair_receipt']['sha256'],'CPU repair receipt differs from reviewed binding')
    h.require(proof['status']=='complete' and proof['session']==common.SESSION and proof['cuda_initialized'] is False
              and proof['manifest_sha256']==revised['manifest']['sha256'] and proof['source_snapshot_sha256']==revised['source_snapshot']['sha256'],
              'Matching completed CPU-only metadata deployment proof required')
    for key in ('local_path','drive_path'): h.verify(repair['archive'][key],repair['archive'])
    cpu=proof['original_cpu_proof']
    h.require(cpu['state']['strict_loads']==8 and cpu['state']['cuda_initialized'] is False
              and cpu['state']['manifest_sha256']==approved['asset_manifest']['sha256'],'Original8-head CPU proof differs')
    for key in ('local_path','drive_path'): h.verify(cpu['archive'][key],cpu['archive'])
    acceptance=verify_parity_acceptance(args.parity_acceptance,args.parity_acceptance_sha256,original['plan']['sha256'])
    h.require(acceptance['failed_stage_receipt']['path']==binding['original_failed_stage_receipt_path'],'Different failed initial stage')
    sys.path.insert(0,plan['source_root'])
    from scripts import run_chamfer_variants_campaign as planner
    from scripts.analyze_scan_trajectory import analyze_bundle, Bundle
    h.require(Path(planner.__file__).resolve()==Path(plan['source_root'])/'scripts/run_chamfer_variants_campaign.py','Unsealed planner import')
    planner.verify_approval(original['approval']['path'],original['approval']['sha256'],original['plan']['sha256'],approved)
    h.require(commands['plan']==approved and commands['status']=='prepared_not_executed','Original command ledger differs')
    h.ASSETS=Path(plan['asset_root']); h.PREPARED=h.ASSETS.parent
    h.MANIFEST_SHA=revised['manifest']['sha256']; h.SOURCE_SHA=revised['source_snapshot']['sha256']
    ledger=phase.load_ledger(original['plan']['sha256'])
    start=datetime.fromisoformat(binding['research_started_utc'].replace('Z','+00:00')).timestamp()
    deadline=datetime.fromisoformat(binding['hard_deadline_utc'].replace('Z','+00:00')).timestamp()
    h.require(abs(ledger['started_epoch']-start)<1 and deadline-start==phase.CAP,'Original5h clock changed')
    accounting=read_pinned(args.reserve_accounting,args.reserve_accounting_sha256)
    h.require(accounting['original_plan_sha256']==original['plan']['sha256']
              and accounting['failed_stage_receipt']==acceptance['failed_stage_receipt'],'Contingency accounting scope differs')
    failed=read_reference(acceptance['failed_stage_receipt'])
    earliest_failure=ledger['started_epoch']+failed['state']['elapsed_active_seconds']
    h.require(math.isclose(accounting['recovery_started_epoch'],earliest_failure,rel_tol=0,abs_tol=1e-6),
              'Reserve consumption must begin at the documented derived earliest failure bound')
    reserve=residual_reserve(accounting,ledger)
    frozen=dict(sha256=args.reserve_accounting_sha256,remaining_reserve_seconds=reserve)
    h.require('recovery_reserve_accounting' not in ledger or ledger['recovery_reserve_accounting']==frozen,'Contingency receipt changed/replenished')
    ledger['recovery_reserve_accounting']=frozen; phase.RESERVE=reserve
    h.require(ledger['stages']['smoke_initial']['status']=='failed' and args.stage not in ledger['stages'],'Failed stage was overwritten or recovery stage already attempted')
    previous=phase.STAGES[phase.STAGES.index(args.stage)-1]
    if args.stage!='calibration':
        receipt=json.loads((common.ROOT/('gpu_'+previous+'_receipt.json')).read_text())
        h.require(ledger['stages'][previous]['status']=='complete' and receipt['state']['status']=='complete'
                  and receipt['state']['recovery_binding_sha256']==args.binding_sha256,'Previous recovery stage is not accepted')
        for key in ('local_path','drive_path'): h.verify(receipt['archive'][key],receipt['archive'])
    ledger.setdefault('recovery_acceptances',{})[args.binding_sha256]=dict(parity_sha256=args.parity_acceptance_sha256,
        cpu_repair_sha256=args.cpu_repair_receipt_sha256,manifest_sha256=h.MANIFEST_SHA,deadline_utc=binding['hard_deadline_utc'])
    base='smoke' if args.stage=='smoke_active' else args.stage
    h.require(commands['stages'][base]==planner.build_jobs(approved,base),'Frozen original argv changed')
    coefficients=None
    if args.stage!='calibration':
        h.verify(plan['calibration_report'],ledger['calibration_report']); coefficients=planner.verify_calibration(Path(plan['calibration_report']))['coefficients']
    jobs=phase.select_jobs(planner.build_jobs(plan,base,coefficients,h.identity(plan['calibration_report'])['sha256'] if coefficients else None),args.stage)
    def budget(remaining):
        left=phase.check_budget(ledger,remaining)
        h.require(deadline-time.time()>=1.1*remaining+phase.RESERVE,'Original absolute deadline projection exceeded')
        return min(left,deadline-time.time()-phase.RESERVE)
    budget(sum(ESTIMATES[s] for s in phase.STAGES[phase.STAGES.index(args.stage):]))
    output=common.ROOT/('gpu_'+args.stage+'_metadata_recovery_v3'); output.mkdir(exist_ok=False)
    drive=common.DRIVE/'phase-evidence'/output.name; drive.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',stage=args.stage,plan_sha256=original['plan']['sha256'],recovery_binding_sha256=args.binding_sha256,jobs=[])
    started=time.monotonic(); prior=ledger['active_seconds']; attempted=[]
    def account():
        ledger['active_seconds']=prior+time.monotonic()-started; phase.save_ledger(ledger)
    try:
        h.preserve_phase_sources(output)
        for path in (__file__,args.binding,args.parity_acceptance,args.cpu_repair_receipt,args.reserve_accounting,binding['repair_scope']['path'],binding['new_source_receipt']['path']):
            shutil.copyfile(path,output/Path(path).name)
        for name in ('plan','commands','approval'): shutil.copyfile(original[name]['path'],output/('original_'+name+'.json'))
        shutil.copyfile(original['source_receipt']['path'],output/'original_source_receipt.json')
        (output/'runtime_plan.json').write_text(json.dumps(plan,indent=2)+'\n'); (output/'executed_commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
        h.require(all(j['executable'] and not Path(j['result_root']).exists() and not Path(j['drive_root']).exists() for j in jobs),'Fresh resolved output paths required')
        if args.stage in ('development','confirmation'): h.require(not (common.ROOT/'paired_states'/base).exists(),'Fresh stage caches required')
        for number,job in enumerate(jobs):
            remaining=ESTIMATES[args.stage]*(len(jobs)-number)/len(jobs)+sum(ESTIMATES[s] for s in phase.STAGES[phase.STAGES.index(args.stage)+1:])
            timeout=budget(remaining); ledger['active']=dict(stage=args.stage,job=number,command=job['command'],recovery=args.binding_sha256); account()
            def began(pid): ledger['active']['pid']=pid; account()
            attempted.append(job); before=time.monotonic()
            phase.stream(job['command'],output/('job%03d.log'%number),plan['source_root'],timeout,began)
            record=dict(job,elapsed_active_seconds=time.monotonic()-before,archives=phase.validate_job(job,analyze_bundle,Bundle))
            record['archive_identities']=archive_references(job,record['archives'],Bundle); state['jobs'].append(record)
            phase.preserve_caches(ledger); account(); (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        if args.stage=='calibration':
            phase.native_calibration_gradients(state['jobs'],Bundle)
            command=[plan['python'],'-u',str(Path(plan['source_root'])/'scripts/calibrate_chamfer_variants.py'),'summarize',*[r['archives'][0] for r in state['jobs']],'--output',plan['calibration_report']]
            (output/'summarize_command.json').write_text(json.dumps(command,indent=2)+'\n')
            phase.stream(command,output/'summarize.log',plan['source_root'],budget(sum(ESTIMATES[s] for s in phase.STAGES[2:])))
            planner.verify_calibration(Path(plan['calibration_report'])); ledger['calibration_report']=h.identity(plan['calibration_report'])
            h.copy_checked(plan['calibration_report'],drive/'calibration.json',ledger['calibration_report']); shutil.copyfile(plan['calibration_report'],output/'calibration.json')
        state['status']='complete'
    except BaseException:
        state.update(status='failed',failure=traceback.format_exc())
        for job in attempted:
            if Path(job['result_root']).exists() and not any(r['result_root']==job['result_root'] for r in state['jobs']):
                shutil.copytree(job['result_root'],output/'partial_results'/job['dataset']/job['arm'])
        raise
    finally:
        state['elapsed_active_seconds']=time.monotonic()-started; (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        ledger['stages'][args.stage]=dict(status=state['status'],output=str(output),recovery_binding_sha256=args.binding_sha256)
        ledger['active']=None; account()
        try:
            archive=h.seal(output,drive/'evidence.zip'); receipt=dict(state=state,archive=archive)
            (drive/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            (common.ROOT/('gpu_'+args.stage+'_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        except BaseException:
            ledger['stages'][args.stage]['status']='failed_delivery'; raise
        finally: account()
        h.require(time.time()<=deadline and phase.elapsed_seconds(ledger)<=phase.CAP,'Original continuous5h deadline exceeded')
        print('CHAMFER_METADATA_RECOVERY_STAGE',args.stage,state['status'],json.dumps(archive),flush=True)


if __name__=='__main__': main()
