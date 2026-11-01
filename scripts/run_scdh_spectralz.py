"""Bounded registered campaign; explicit whole-phase approval, immutable evidence."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
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
            path=Path(plan['root'])/'frozen_calibration.json'
            flags+=['--calibration-report',str(path),'--calibration-report-sha256',identity(path)['sha256']]
    return [plan['python'],'-u',str(REPO/'scripts/eval_scdh_spectralz.py'),*flags]


def execute_job(plan,job,deadline,manifest):
    root=Path(plan['root']);slot=root/'slots'/job['slot']
    if slot.exists():
        proof=json.loads((slot/'receipt.json').read_text())
        if (proof['status']!='accepted' or proof['job']!=job or identity(slot/'accepted.json')!=proof['report_identity']
                or identity(slot/'score.json')!=proof['score_identity']):
            raise ValueError('Prior failed/unknown slot cannot be rerun: '+job['slot'])
        operations.copy_verified(root/'archives'/(job['slot']+'.zip'),Path(plan['drive_root'])/'campaign'/study.canonical_sha(plan)/(job['slot']+'.zip'))
        return json.loads((slot/'score.json').read_text())
    remaining=deadline-time.time()-plan['archive_reserve_seconds']
    if remaining<plan.get('minimum_job_reserve_seconds',600):raise ValueError('Insufficient cumulative time for job plus archive reserve')
    slot.mkdir(parents=True);cmd=command(plan,job)
    configs=worker.configurations(cmd[3:]);cfg=configs[0]
    write_new(slot/'job.json',job);write_new(slot/'command.json',cmd)
    report=None
    try:
        code=operations.run_worker(cmd,REPO,slot/'console.log',min(remaining,plan.get('job_timeout_seconds',900)),job['slot'])
        if code:raise RuntimeError('Worker failed: '+str(code))
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
        report.update(job)
        # Keep the accepted analyzer bytes immutable; score binding is saved separately.
        write_new(slot/'score.json',report)
        write_new(slot/'receipt.json',dict(status='accepted',job=job,report_identity=identity(slot/'accepted.json'),score_identity=identity(slot/'score.json')))
    except BaseException:
        write_new(slot/'failure.json',dict(status='unavailable',job=job,traceback=traceback.format_exc()))
        raise
    finally:
        archives=root/'archives';archives.mkdir(exist_ok=True)
        archive=archive_slot(slot,archives/(job['slot']+'.zip'))
        operations.copy_verified(archive,Path(plan['drive_root'])/'campaign'/study.canonical_sha(plan)/archive.name)
        paired=root/job.get('paired_group',job['stage'])/'paired'/('seed%d'%job['seed'])
        for path in paired.rglob('*.npz'):
            if not path.name.endswith('.spectral_basis.npz'):
                operations.copy_verified(path,Path(plan['drive_root'])/'campaign'/study.canonical_sha(plan)/path.relative_to(root))
    return report


def phase_jobs(phase,root):
    if phase=='technical':
        return [study.job('scd',stage=phase,suffix='-repeat1'),study.job('scd',stage=phase,suffix='-repeat2'),
            study.job('off',stage=phase),
            study.job('spectral',modes=800,rho=1.,stage=phase)]
    if phase=='calibration':return [study.job('calibration',stage=phase)]
    if phase=='A':
        report=json.loads((root/'frozen_calibration.json').read_text())
        return study.phase_a([int(m) for m,v in report['coefficients'].items() if v['status']=='available'])
    if phase=='B':return study.phase_b(json.loads((root/'frozen_top2.json').read_text())['top2'])
    winner=json.loads((root/'frozen_winner.json').read_text())
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


def run(plan,approval,phase):
    root,manifest=validate_plan(plan);root.mkdir(parents=True,exist_ok=True)
    deadline=budget(root,plan,approval)
    for name,value in (('plan.json',plan),('approval.json',approval)):
        if (root/name).exists():
            if json.loads((root/name).read_text())!=value:raise ValueError('Immutable '+name+' changed')
        else:write_new(root/name,value)
    phases=('technical','calibration','A','B','C','timing')
    selected=phases if phase=='all' else (phase,)
    for current in selected:
        index=phases.index(current)
        if index and not (root/(phases[index-1]+'_accepted.json')).exists():raise ValueError('Previous phase has not passed independent acceptance')
        endpoint=root/(current+'_accepted.json')
        if endpoint.exists():continue
        reports=[execute_job(plan,j,deadline,manifest) for j in phase_jobs(current,root)]
        if current in ('technical','A','B'):analysis.compare_physical(reports)
        if current=='technical' and all(r.get('technical_logits') for r in reports):
            summary=technical_summary(reports);write_new(root/'technical_summary.json',summary)
            operations.copy_verified(root/'technical_summary.json',Path(plan['drive_root'])/'campaign'/study.canonical_sha(plan)/'technical_summary.json')
            if summary['remaining_worst_case_forecast_seconds']>deadline-time.time()-180:
                raise ValueError('Measured remaining whole-phase forecast exceeds approved cumulative budget; reassess before calibration')
        if current=='calibration':
            report=reports[0];records=report['calibration_observations']
            if len(records)!=60 or any(len(r['ratios'])!=32 for r in records):raise ValueError('Incomplete32x4x3x5 calibration')
            write_new(root/'frozen_calibration.json',study.freeze_calibration(records,dict(report['calibration_binding'],
                accepted_report_identity=identity(root/'slots'/report['slot']/'accepted.json'))))
        if current=='A':write_new(root/'frozen_top2.json',dict(top2=study.rank_candidates(reports)[:2],scores_sha256=study.canonical_sha(reports)))
        if current=='B':
            previous=json.loads((root/'A_accepted.json').read_text())['reports']
            analysis.compare_physical(previous+reports)
            ranked=study.rank_candidates(previous+reports)
            baseline=next(r for r in previous if r['arm']=='scd')
            positive=bool(ranked) and ranked[0]['correct']>baseline['correct']
            write_new(root/'frozen_winner.json',dict(status='frozen' if positive else 'null',winner=ranked[0] if positive else None,
                baseline_correct=baseline['correct'],scores_sha256=study.canonical_sha(previous+reports)))
        if current=='C' and reports:
            winner=json.loads((root/'frozen_winner.json').read_text())['winner'];a=winner['alpha_h']
            comparisons={}
            for seed in (0,1,2):analysis.compare_physical([r for r in reports if r['seed']==seed])
            spectral=np.asarray([next(r['outcomes']['probability_mean'] for r in reports if r['seed']==s and r['arm']=='spectral') for s in (0,1,2)])
            for label,arm,alpha in (('C_base','scd',.03),('C_match','scd',a),('Z0_match','off',a)):
                control=np.asarray([next(r['outcomes']['probability_mean'] for r in reports if r['seed']==s and r['arm']==arm and r['alpha_h']==alpha) for s in (0,1,2)])
                comparisons[label]=analysis.paired_metrics(spectral,control)
            write_new(root/'confirmation_comparisons.json',comparisons)
        write_new(endpoint,dict(status='accepted',phase=current,reports=reports,remaining_seconds=deadline-time.time()))
        for path in root.glob('*.json'):
            if path.name not in ('plan.json','approval.json','started.json'):
                operations.copy_verified(path,Path(plan['drive_root'])/'campaign'/study.canonical_sha(plan)/path.name)
        if time.time()>deadline:raise ValueError('Cumulative budget exceeded during persistence')
    print('SCDH_SPECTRALZ_PHASE_COMPLETE',phase,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--phase',choices=('all','technical','calibration','A','B','C','timing'),required=True)
    args=parser.parse_args();run(json.loads(args.plan.read_text()),json.loads(args.approval.read_text()),args.phase)

if __name__=='__main__':
    main()
