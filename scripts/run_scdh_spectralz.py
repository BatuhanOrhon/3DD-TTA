"""Bounded registered campaign; explicit whole-phase approval, immutable evidence."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
import zipfile
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import scdh_spectralz_study as study
from scripts import analyze_scdh_spectralz as analysis
from scripts import eval_scdh_spectralz as worker
from scripts.prepare_scdh_spectralz import write_new
from scripts.prepare_chamfer_variants_assets import identity

spec=importlib.util.spec_from_file_location('_scdh_operations',REPO/'result/preparation/chamfer_depth_tuning_20261030/operations.py')
operations=importlib.util.module_from_spec(spec);spec.loader.exec_module(operations)
SCOPE='technical+calibration+A+B+conditional-C+timing'


def budget(root,plan,approval,*,now=None):
    now=time.time() if now is None else now
    cap=plan.get('cumulative_wall_cap_seconds')
    if (approval.get('approved') is not True or approval.get('scope')!=SCOPE
            or approval.get('plan_sha256')!=study.canonical_sha(plan) or not approval.get('user_message')
            or type(cap) is not int or not 0<cap<=14400):
        raise ValueError('Explicit whole scientific GPU approval and <=4h cumulative cap required')
    path=root/'started.json'
    binding=dict(plan_sha256=study.canonical_sha(plan),approval_sha256=study.canonical_sha(approval),cap_seconds=cap)
    if path.exists():
        record=json.loads(path.read_text())
        if any(record.get(k)!=v for k,v in binding.items()):raise ValueError('Immutable cumulative approval binding changed')
    else:
        record=dict(binding,started_unix_seconds=now);write_new(path,record)
    deadline=record['started_unix_seconds']+cap
    if not record['started_unix_seconds']<=now<deadline:raise ValueError('Cumulative whole-phase budget exhausted or clock moved backwards')
    return deadline


def archive_slot(root,destination):
    files=sorted(root.rglob('*'))
    if any(p.is_symlink() for p in files):raise ValueError('Symlink in campaign evidence')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file() and not path.name.endswith('.spectral_basis.npz'):
                archive.write(path,path.relative_to(root).as_posix())
    analysis.safe_archive(destination)
    return destination


def audit_only_source(source,name):
    import ast
    tree=ast.parse(source)
    class RemoveAudit(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if (name=='tta.py' and node.name=='audit_clean') or (name=='scdh_spectralz.py' and node.name=='audit'):
                return ast.Pass()
            return self.generic_visit(node)
    return ast.dump(RemoveAudit().visit(tree),include_attributes=False)


def validate_plan(plan):
    root=Path(plan['root']);assets=Path(plan['asset_root'])
    manifest=worker.shared.source.read_manifest(assets,plan['asset_manifest_sha256'])
    if (plan.get('schema_version')!=1 or plan.get('scope')!=SCOPE or assets/'source_snapshot'!=REPO
            or plan.get('source_snapshot_sha256')!=manifest['source_snapshot_sha256']
            or plan.get('max_development_jobs')!=30 or plan.get('max_confirmation_jobs')!=12
            or plan.get('cumulative_wall_cap_seconds')!=14400 or plan.get('archive_reserve_seconds')!=180):
        raise ValueError('Unregistered plan/source/budget')
    for name in (*analysis.CRITICAL,'scripts/run_scdh_spectralz.py','scripts/scdh_spectralz_study.py','scripts/analyze_scdh_spectralz.py'):
        worker.shared.source.verify_identity(REPO/name,manifest['files']['source_snapshot/'+name])
    for stage in worker.STAGES:
        worker.shared.source.verify_identity(assets/'study'/(stage+'_ids.json'),manifest['files']['study/'+stage+'_ids.json'])
    reuse=plan.get('technical_source_reuse')
    if reuse:
        original=worker.shared.source.read_manifest(Path(reuse['old_asset_root']),reuse['old_asset'])
        if (reuse['new_asset']!=plan['asset_manifest_sha256'] or reuse['new_snapshot']!=manifest['source_snapshot_sha256']
                or reuse['old_snapshot']!=original['source_snapshot_sha256'] or original['datasets']!=manifest['datasets']):
            raise ValueError('Unregistered native-control reuse manifest')
        allowed={'source_snapshot.zip'}|{'source_snapshot/'+n for n in ('graph_spectral_v2_offdiag.py',
            'scripts/run_scdh_spectralz.py','scripts/analyze_scdh_spectralz.py','scripts/deploy_scdh_spectralz.py','scripts/prepare_scdh_spectralz.py')}
        revisions=reuse.get('audit_only_revisions',{})
        if set(revisions)-{'tta.py','scdh_spectralz.py'}:raise ValueError('Unknown audit-only revision')
        for name,revision in revisions.items():
            key='source_snapshot/'+name
            if (revision!={'old':original['files'][key],'new':manifest['files'][key]}
                    or audit_only_source((Path(reuse['old_asset_root'])/key).read_text(),name)
                    !=audit_only_source((assets/key).read_text(),name)):
                raise ValueError('Source reuse changed active non-audit code: '+name)
            allowed.add(key)
        changed={n for n in set(original['files'])|set(manifest['files']) if original['files'].get(n)!=manifest['files'].get(n)}
        if (changed-allowed or reuse['old_graph']!=original['files']['source_snapshot/graph_spectral_v2_offdiag.py']
                or reuse['new_graph']!=manifest['files']['source_snapshot/graph_spectral_v2_offdiag.py']):
            raise ValueError('Source reuse changed data/native/model/forward files outside graph-only repair')
    receipt_path=Path(plan['cpu_preflight_receipt'])
    if identity(receipt_path)!=plan['cpu_preflight_receipt_identity']:raise ValueError('CPU receipt bytes changed')
    receipt=json.loads(receipt_path.read_text())
    if (receipt.get('status')!='complete' or receipt.get('cuda_initialized') is not False
            or receipt.get('forward_executed') is not False or receipt.get('manifest_sha256')!=plan['asset_manifest_sha256']
            or receipt.get('source_snapshot_sha256')!=plan['source_snapshot_sha256']
            or receipt.get('checkpoint_load',{}).get('strict') is not True):
        raise ValueError('Missing strict CPU no-forward deployment receipt')
    archive=receipt['archive']
    for key in ('local_path','drive_path'):
        worker.shared.source.verify_identity(Path(archive[key]),archive)
    if not os.path.ismount('/content/drive') or not Path(plan['drive_root']).is_dir():raise ValueError('Actual Drive mount required')
    return root,manifest


def persist(source,target):
    source,target=Path(source),Path(target)
    expected=identity(source);target.parent.mkdir(parents=True,exist_ok=True)
    intent=target.with_name(target.name+'.copy_intent.json');done=target.with_name(target.name+'.verified.json')
    owned=intent.exists()
    for record in (intent,done):
        if record.exists() and json.loads(record.read_text())!=expected:raise ValueError('Immutable publication identity changed: '+str(target))
    if not owned and target.exists() and identity(target)!=expected:raise ValueError('Unknown publication destination; preserve it')
    if not owned:write_new(intent,expected)
    if target.exists() and identity(target)!=expected:
        if not owned or done.exists():raise ValueError('Previously published or unknown evidence changed: '+str(target))
        target.rename(target.with_name(target.name+'.incomplete-'+str(time.time_ns())))
    operations.copy_verified(source,target)
    if identity(target)!=expected:raise ValueError('Publication readback failed')
    if not done.exists():write_new(done,expected)


def publication_root(plan):
    return Path(plan['drive_root'])/'campaign'/study.canonical_sha(plan)


def publish_slot(plan,job):
    root=Path(plan['root']);slot=root/'slots'/job['slot'];archives=root/'archives';archives.mkdir(exist_ok=True)
    archive=archives/(job['slot']+'.zip');proof=slot/'archive_receipt.json'
    if not proof.exists():
        archive_slot(slot,archive)
        write_new(proof,dict(archive_identity=identity(archive)))
    if identity(archive)!=json.loads(proof.read_text())['archive_identity']:raise ValueError('Immutable slot archive changed')
    persist(archive,publication_root(plan)/archive.name)
    for source in (slot/'receipt.json',slot/'score.json',proof):
        if source.exists():persist(source,publication_root(plan)/source.relative_to(root))
    paired=root/job.get('paired_group',job['stage'])/'paired'/('seed%d'%job['seed'])
    for source in paired.rglob('*.npz'):
        if not source.name.endswith('.spectral_basis.npz'):persist(source,publication_root(plan)/source.relative_to(root))


def checked_score(root,job):
    slot=root/'slots'/job['slot'];receipt=json.loads((slot/'receipt.json').read_text())
    if (receipt.get('status') not in ('accepted','unavailable') or receipt.get('job')!=job
            or identity(slot/'score.json')!=receipt.get('score_identity')):raise ValueError('Invalid immutable slot receipt')
    score=json.loads((slot/'score.json').read_text())
    if score.get('status')!=receipt['status'] or any(score.get(k)!=v for k,v in job.items()):raise ValueError('Slot score/job mismatch')
    if receipt['status']=='accepted' and identity(slot/'accepted.json')!=receipt.get('report_identity'):raise ValueError('Accepted analyzer bytes changed')
    return score


def phase_record(root,name,*,pending=False):
    path=root/(name+('_pending.json' if pending else '_accepted.json'))
    record=json.loads(path.read_text());plan=json.loads((root/'plan.json').read_text());approval=json.loads((root/'approval.json').read_text())
    if (record.get('status')!='accepted' or record.get('phase')!=name or record.get('plan_sha256')!=study.canonical_sha(plan)
            or record.get('approval_sha256')!=study.canonical_sha(approval) or record.get('started_identity')!=identity(root/'started.json')):
        raise ValueError('Invalid completed phase identity')
    names=('technical','calibration','A','B','C','timing');index=names.index(name)
    previous=identity(root/(names[index-1]+'_accepted.json')) if index else None
    if record.get('predecessor_identity')!=previous:raise ValueError('Phase predecessor changed')
    jobs=phase_jobs(name,root)
    if record.get('jobs')!=jobs:raise ValueError('Completed phase scope changed')
    scores=[checked_score(root,j) for j in jobs]
    if record.get('reports')!=scores or record.get('reports_sha256')!=study.canonical_sha(scores):raise ValueError('Completed scores changed')
    if record.get('evidence')!=evidence(root,jobs):raise ValueError('Incomplete or changed completed-phase evidence')
    for relative,expected in record.get('evidence',{}).items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or identity(root/relative)!=expected:raise ValueError('Completed phase evidence changed')
    for name,expected in record.get('frozen_outputs',{}).items():
        if '/' in name or identity(root/name)!=expected:raise ValueError('Frozen phase output changed')
    if not pending:
        persist(path,publication_root(plan)/path.name)
    return record


def frozen_file(root,name,producer):
    record=phase_record(root,producer);path=root/name
    if record.get('frozen_outputs',{}).get(name)!=identity(path):raise ValueError('Missing original freeze identity')
    rows=record['reports'];value=json.loads(path.read_text())
    if producer=='calibration':
        row=rows[0]
        expected=study.freeze_calibration(row['calibration_observations'],dict(row['calibration_binding'],
            accepted_report_identity=identity(root/'slots'/row['slot']/'accepted.json')))
    elif producer=='A':expected=dict(top2=[selection(r) for r in study.rank_candidates(rows)[:2]],scores_sha256=study.canonical_sha(rows))
    else:
        earlier=phase_record(root,'A')['reports'];ranked=study.rank_candidates(earlier+rows)
        baseline=next(r for r in earlier if r['arm']=='scd');positive=bool(ranked) and ranked[0]['correct']>baseline['correct']
        expected=dict(status='frozen' if positive else 'null',winner=selection(ranked[0]) if positive else None,
            baseline_correct=baseline['correct'],scores_sha256=study.canonical_sha(earlier+rows))
    if value!=expected:raise ValueError('Frozen values differ from independently accepted score inputs')
    return path,value,record['frozen_outputs'][name]


def freeze(root,name,value):
    path=root/name
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Previously frozen output changed')
    else:write_new(path,value)


def complete_phase(root,name,plan):
    record=phase_record(root,name,pending=True)
    for j in record['jobs']:publish_slot(plan,j)
    for relative in record['evidence']:
        persist(root/relative,publication_root(plan)/relative)
    for filename in record['frozen_outputs']:
        persist(root/filename,publication_root(plan)/filename)
    pending=root/(name+'_pending.json');endpoint=root/(name+'_accepted.json')
    persist(pending,publication_root(plan)/endpoint.name)
    pending.rename(endpoint)


def command(plan,job):
    stage=job['stage'];slot=Path(plan['root'])/'slots'/job['slot']
    paired_group=job.get('paired_group',stage)
    flags=['--stage',stage,'--arm',job['arm'],'--alpha-h',str(job['alpha_h']),'--seed',str(job['seed']),
           '--dataset-root',plan['dataset_root'],'--label-path',plan['label_path'],
           '--asset-root',plan['asset_root'],'--asset-manifest-sha256',plan['asset_manifest_sha256'],
           '--pointmae-config',plan['pointmae_config'],'--pointmae-checkpoint',plan['pointmae_checkpoint'],
           '--result-root',str(slot/'raw'),'--drive-root',plan['drive_root'],
           '--index-manifest',str(Path(plan['asset_root'])/'study'/(stage+'_ids.json')),
           '--paired-state-root',str(Path(plan['root'])/paired_group/'paired'/('seed%d'%job['seed']))]
    if job['arm']=='spectral':
        flags+=['--modes',str(job['modes']),'--rho',str(job['rho'])]
        if stage!='technical':
            path,_,pinned=frozen_file(Path(plan['root']),'frozen_calibration.json','calibration')
            flags+=['--calibration-report',str(path),'--calibration-report-sha256',pinned['sha256']]
    return [plan['python'],'-u',str(REPO/'scripts/eval_scdh_spectralz.py'),*flags]


def compact_report(report):
    keys=('status','correct','object_ids','labels','corruptions','outcomes','calibration_binding',
          'calibration_observations','technical_logits','cost','scdh_spectralz')
    result={k:report[k] for k in keys if k in report}
    guard_keys=('corruption','batch','ids','initial_hashes','final_hashes','initial_rng_sha256','cache','scheduler')
    result['batch_guards']=[{k:r[k] for k in guard_keys} for r in report['batch_guards']]
    result['spectral_details']=[dict(corruption=r['corruption'],batch=r['batch'],ids=r['ids'],
        metrics={'graph':{k:r['metrics']['graph'][k] for k in ('basis_sha256','eigenvalues_sha256','cache_file_sha256')}})
        for r in report.get('spectral_details',[])]
    return result


def execute_job(plan,job,deadline,manifest):
    root=Path(plan['root']);slot=root/'slots'/job['slot']
    if slot.exists():
        score=checked_score(root,job);publish_slot(plan,job)
        return score
    remaining=deadline-time.time()-plan['archive_reserve_seconds']
    if remaining<plan.get('minimum_job_reserve_seconds',600):raise ValueError('Insufficient cumulative time for job plus archive reserve')
    slot.mkdir(parents=True);cmd=command(plan,job)
    configs=worker.configurations(cmd[3:]);cfg=configs[0]
    write_new(slot/'job.json',job);write_new(slot/'command.json',cmd)
    report=None
    try:
        code=operations.run_worker(cmd,REPO,slot/'console.log',min(remaining,plan.get('job_timeout_seconds',900)),job['slot'])
        if code:
            failures=list((slot/'raw').rglob('config.json'))
            failed=json.loads(failures[0].read_text()) if len(failures)==1 else {}
            if numerical_failure(plan,job,failed,manifest):
                report=dict(job,status='unavailable',failure_category='numerical_candidate_unavailable',
                    failure=failed.get('failure','Worker exit '+str(code)),source_snapshot_sha256=plan['source_snapshot_sha256'],
                    asset_manifest_sha256=plan['asset_manifest_sha256'])
                write_new(slot/'failure.json',report);write_new(slot/'score.json',report)
                write_new(slot/'receipt.json',dict(status='unavailable',job=job,score_identity=identity(slot/'score.json')))
                return report
            raise RuntimeError('Worker failed: '+str(code))
        bundles=list((slot/'raw').rglob('bundle_manifest.json'))
        if len(bundles)!=1:raise ValueError('Expected exactly one sealed result')
        raw=bundles[0].parent
        actual=json.loads((raw/'config.json').read_text())
        expected=dict(cfg,source_snapshot_sha256=plan['source_snapshot_sha256'],asset_manifest_sha256=plan['asset_manifest_sha256'])
        for key in ('checkpoint_load','native_extensions'):
            expected[key]=json.loads(Path(plan['cpu_preflight_receipt']).read_text())[key]
        required={}
        for name in analysis.CRITICAL:
            key='assets/source_snapshot/'+name
            required[key]=manifest['files']['source_snapshot/'+name]
        binding=dict(slot=job['slot'],expected_config={k:expected[k] for k in analysis.BINDING_KEYS if k in expected},required_source_files=required)
        write_new(slot/'binding.json',binding)
        analyze=[plan['analysis_python'],'-u',str(REPO/'scripts/analyze_scdh_spectralz.py'),
            '--bundle',str(raw),'--binding',str(slot/'binding.json'),
            '--cache-root',str(root/job.get('paired_group',job['stage'])),'--output',str(slot/'accepted.json')]
        with (slot/'analysis.log').open('x') as log:
            subprocess.run(analyze,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(180,max(1,deadline-time.time()-60)))
        report=json.loads((slot/'accepted.json').read_text())
        if report['status']!='accepted':raise ValueError('Independent raw acceptance failed')
        report=compact_report(report);report.update(job)
        # Full eigenvalues/norm vectors stay in the immutable analyzer and raw bundles.
        write_new(slot/'score.json',report)
        write_new(slot/'receipt.json',dict(status='accepted',job=job,report_identity=identity(slot/'accepted.json'),score_identity=identity(slot/'score.json')))
    except BaseException:
        write_new(slot/'failure.json',dict(status='unavailable',job=job,traceback=traceback.format_exc()))
        raise
    finally:
        publish_slot(plan,job)
    return report


def numerical_failure(plan,job,cfg,manifest):
    if not (job['stage']=='development' and job['arm']=='spectral' and cfg.get('status')=='failed'
            and cfg.get('scdh_failure_category')=='numerical_candidate_unavailable'):return False
    if (any(cfg.get(k)!=plan[k] for k in ('source_snapshot_sha256','asset_manifest_sha256'))
            or cfg.get('executed_source_sha256')!=study.canonical_sha(cfg.get('executed_source_files',{}))
            or cfg.get('checkpoint_load',{}).get('strict') is not True
            or not analysis.accepted._is_sha(cfg.get('classifier_state_before_sha256'))
            or cfg['classifier_state_before_sha256']!=cfg.get('classifier_state_after_sha256')
            or set(cfg.get('lion_state_before',{}))!={'vae','priors'}
            or not all(analysis.accepted._is_sha(v) for v in cfg['lion_state_before'].values())):
        raise ValueError('Numerical outcome lacks prior source/frozen-state gates')
    for name in analysis.CRITICAL:
        if cfg.get('executed_source_files',{}).get('assets/source_snapshot/'+name)!=manifest['files']['source_snapshot/'+name]:
            raise ValueError('Failed candidate critical source mismatch')
    return True


def phase_jobs(phase,root):
    if phase=='technical':
        return [study.job('scd',stage=phase,suffix='-repeat1'),study.job('scd',stage=phase,suffix='-repeat2'),
            study.job('off',stage=phase),
            study.job('spectral',modes=800,rho=1.,stage=phase)]
    if phase=='calibration':return [study.job('calibration',stage=phase)]
    if phase=='A':
        _,report,_=frozen_file(root,'frozen_calibration.json','calibration')
        return study.phase_a([int(m) for m,v in report['coefficients'].items() if v['status']=='available'])
    if phase=='B':return study.phase_b(frozen_file(root,'frozen_top2.json','A')[1]['top2'])
    _,winner,_=frozen_file(root,'frozen_winner.json','B')
    if winner['status']=='null':return []
    if phase=='C':return study.confirmation(winner['winner'])
    if phase=='timing':
        w=winner['winner'];jobs=[]
        for repeat in ('warmup','repeat1','repeat2','repeat3'):
            for arm in ('scd','spectral'):
                j=study.job(arm,.03 if arm=='scd' else w['alpha_h'],
                    w['modes'] if arm=='spectral' else None,w['rho'] if arm=='spectral' else None,stage='timing',suffix='-'+repeat)
                j['paired_group']='timing-'+repeat+'-'+arm;jobs.append(j)
        return jobs
    raise ValueError('Unknown phase')


def technical_summary(reports):
    baseline=[r for r in reports if r['arm']=='scd']
    off=next(r for r in reports if r['arm']=='off')
    zero=next(r for r in reports if r['arm']=='spectral')
    comparisons={}
    for label,left,right in (('native_C_repeat',*baseline),('zero_spectral_vs_style_off',zero,off)):
        comparisons[label]={}
        for corruption in left['corruptions']:
            a=np.asarray(left['technical_logits'][corruption]);b=np.asarray(right['technical_logits'][corruption])
            comparisons[label][corruption]=dict(max_abs_logit_difference=float(np.abs(a-b).max()),
                candidate_prediction_disagreements=int((a.argmax(-1)!=b.argmax(-1)).sum()),
                PM_prediction_disagreements=int(np.sum(analysis.accepted.policies(a,3)['probability_mean']
                    !=analysis.accepted.policies(b,3)['probability_mean'])))
    times={}
    for report in reports:
        for row in report['cost']:
            value=row['reconstruction_seconds']+row['head_seconds']
            if not math.isfinite(value) or value<=0:raise ValueError('Missing finite positive native timing')
            times[row['corruption']]=max(times.get(row['corruption'],0),value)
    normal,bg=times['gaussian'],times['background']
    forecast=(30*2*(14*normal+bg)+12*4*(14*normal+bg)+8*(normal+bg)+300)*1.3+900
    return dict(comparisons=comparisons,remaining_worst_case_forecast_seconds=forecast,
        forecast_policy='max observed native C/off/routed/FD-inclusive32 timings;30dev64+12confirm128+8timing/warmup64;1.3factor+900s persistence',
        limitation='Recorded native repeats; no bitwise prediction parity assumed or tolerance silently relaxed')


def selection(row):
    return {k:row[k] for k in ('slot','stage','arm','alpha_h','modes','rho','seed','correct','status')}


def evidence(root,jobs):
    names={'plan.json','approval.json','started.json'}
    for j in jobs:
        prefix='slots/'+j['slot']+'/'
        names.update(prefix+n for n in ('receipt.json','score.json','archive_receipt.json'))
        receipt=json.loads((root/(prefix+'receipt.json')).read_text())
        names.add(prefix+('accepted.json' if receipt['status']=='accepted' else 'failure.json'))
        names.add('archives/'+j['slot']+'.zip')
    return {name:identity(root/name) for name in sorted(names)}


def run(plan,approval,phase):
    root,manifest=validate_plan(plan);root.mkdir(parents=True,exist_ok=True)
    deadline=budget(root,plan,approval)
    for name,value in (('plan.json',plan),('approval.json',approval)):
        if (root/name).exists():
            if json.loads((root/name).read_text())!=value:raise ValueError('Immutable '+name+' changed')
        else:write_new(root/name,value)
    for name in ('plan.json','approval.json','started.json'):persist(root/name,publication_root(plan)/name)
    phases=('technical','calibration','A','B','C','timing')
    selected=phases if phase=='all' else (phase,)
    for current in selected:
        index=phases.index(current)
        if index:phase_record(root,phases[index-1])
        endpoint=root/(current+'_accepted.json');pending=root/(current+'_pending.json')
        if endpoint.exists():phase_record(root,current);continue
        if pending.exists():complete_phase(root,current,plan);continue
        jobs=phase_jobs(current,root)
        reports=[execute_job(plan,j,deadline,manifest) for j in jobs]
        available=[r for r in reports if r['status']=='accepted']
        if current in ('technical','A','B'):
            analysis.compare_physical(available,technical_reuse=plan.get('technical_source_reuse') if current=='technical' else None)
        outputs=[]
        if current=='technical' and all(r.get('technical_logits') for r in reports):
            summary=technical_summary(reports);freeze(root,'technical_summary.json',summary);outputs.append('technical_summary.json')
            persist(root/'technical_summary.json',publication_root(plan)/'technical_summary.json')
            if summary['remaining_worst_case_forecast_seconds']>deadline-time.time()-180:
                raise ValueError('Measured remaining whole-phase forecast exceeds approved cumulative budget; reassess before calibration')
        if current=='calibration':
            report=reports[0];records=report['calibration_observations']
            if len(records)!=60 or any(len(r['ratios'])!=32 for r in records):raise ValueError('Incomplete32x4x3x5 calibration')
            freeze(root,'frozen_calibration.json',study.freeze_calibration(records,dict(report['calibration_binding'],
                accepted_report_identity=identity(root/'slots'/report['slot']/'accepted.json'))));outputs.append('frozen_calibration.json')
        if current=='A':
            freeze(root,'frozen_top2.json',dict(top2=[selection(r) for r in study.rank_candidates(reports)[:2]],scores_sha256=study.canonical_sha(reports)))
            outputs.append('frozen_top2.json')
        if current=='B':
            previous=phase_record(root,'A')['reports']
            analysis.compare_physical([r for r in previous+reports if r['status']=='accepted'])
            ranked=study.rank_candidates(previous+reports);baseline=next(r for r in previous if r['arm']=='scd')
            positive=bool(ranked) and ranked[0]['correct']>baseline['correct']
            freeze(root,'frozen_winner.json',dict(status='frozen' if positive else 'null',winner=selection(ranked[0]) if positive else None,
                baseline_correct=baseline['correct'],scores_sha256=study.canonical_sha(previous+reports)));outputs.append('frozen_winner.json')
        if current=='C' and reports:
            winner=frozen_file(root,'frozen_winner.json','B')[1]['winner'];a=winner['alpha_h'];comparisons={}
            for seed in (0,1,2):analysis.compare_physical([r for r in reports if r['seed']==seed])
            spectral=np.asarray([next(r['outcomes']['probability_mean'] for r in reports if r['seed']==s and r['arm']=='spectral') for s in (0,1,2)])
            for label,arm,alpha in (('C_base','scd',.03),('C_match','scd',a),('Z0_match','off',a)):
                control=np.asarray([next(r['outcomes']['probability_mean'] for r in reports if r['seed']==s and r['arm']==arm and r['alpha_h']==alpha) for s in (0,1,2)])
                comparisons[label]=analysis.paired_metrics(spectral,control)
            freeze(root,'confirmation_comparisons.json',comparisons);outputs.append('confirmation_comparisons.json')
        record=dict(status='accepted',phase=current,plan_sha256=study.canonical_sha(plan),approval_sha256=study.canonical_sha(approval),
            started_identity=identity(root/'started.json'),predecessor_identity=identity(root/(phases[index-1]+'_accepted.json')) if index else None,
            jobs=jobs,reports=reports,reports_sha256=study.canonical_sha(reports),evidence=evidence(root,jobs),
            frozen_outputs={name:identity(root/name) for name in outputs},remaining_seconds=deadline-time.time())
        write_new(pending,record);complete_phase(root,current,plan)
        if time.time()>deadline:raise ValueError('Cumulative budget exceeded during persistence')
    print('SCDH_SPECTRALZ_PHASE_COMPLETE',phase,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--phase',choices=('all','technical','calibration','A','B','C','timing'),required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());approval=json.loads(args.approval.read_text())
    try:
        run(plan,approval,args.phase)
    except BaseException:
        root=Path(plan['root'])
        if (root/'plan.json').is_file() and json.loads((root/'plan.json').read_text())==plan:
            folder=root/'operator_failures'/str(time.time_ns());folder.mkdir(parents=True,exist_ok=False)
            write_new(folder/'failure.json',dict(status='unavailable',phase=args.phase,plan_sha256=study.canonical_sha(plan),traceback=traceback.format_exc()))
            for source in root.glob('*.json'):shutil.copyfile(source,folder/source.name)
            archive=archive_slot(folder,folder.with_suffix('.zip'))
            try:persist(archive,publication_root(plan)/'operator_failures'/archive.name)
            except Exception as error:print('FAILURE_ARCHIVE_LOCAL_ONLY',archive,repr(error),flush=True)
        raise

if __name__=='__main__':
    main()
