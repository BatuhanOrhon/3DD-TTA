"""Independent NumPy recount and original-object clustered comparisons."""
import numpy as np


def paired_metrics(winner,baseline):
    winner,baseline=np.asarray(winner),np.asarray(baseline)
    if (winner.shape!=baseline.shape or winner.ndim!=3 or winner.shape[:2]!=(3,15)
            or winner.dtype!=np.bool_ or baseline.dtype!=np.bool_ or not winner.shape[2]):
        raise ValueError('Require matched3seed x15corruption xoriginal-object boolean outcomes')
    difference=winner.astype(float)-baseline
    object_delta=difference.mean(axis=(0,1))
    rng=np.random.RandomState(20261101)
    bootstrap=np.empty(10000)
    for i in range(10000):bootstrap[i]=object_delta[rng.randint(len(object_delta),size=len(object_delta))].mean()*100
    return dict(delta_pp=float(difference.mean()*100),winner_accuracy_pct=float(winner.mean()*100),
        baseline_accuracy_pct=float(baseline.mean()*100),seed_delta_pp=(difference.mean(axis=(1,2))*100).tolist(),
        per_corruption_delta_pp=(difference.mean(axis=(0,2))*100).tolist(),
        corrected=int((winner&~baseline).sum()),harmed=int((~winner&baseline).sum()),
        object_cluster_ci95_pp=np.percentile(bootstrap,[2.5,97.5]).tolist(),
        bootstrap_repetitions=10000,bootstrap_seed=20261101,
        limitation='conditional on frozen development selection; seeds and all corruptions remain together within each resampled original object')

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile
import sys
import shlex

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts import analyze_chamfer_depth_transfer as accepted
from scripts import eval_scdh_spectralz as study
from scripts.analyze_scan_trajectory import Bundle,require

CRITICAL=accepted.CRITICAL_SOURCES+('scdh_spectralz.py','spectral_guidance_losses.py',
    'graph_spectral_v2_offdiag.py','scripts/eval_scdh_spectralz.py','scripts/scdh_spectralz_study.py')
BINDING_KEYS=('dataset','backbone','seed','selected_original_ids','source_snapshot_sha256',
              'asset_manifest_sha256','checkpoint_load','lion_state_before','native_extensions',
              'scdh_spectralz','scdh_technical_gradient_only','scdh_calibration','selection_manifest_identity')


def safe_archive(path):
    require(not Path(path).is_symlink(),'Symlink raw archive')
    if Path(path).is_dir():
        require(not Path(path).is_symlink() and not any(p.is_symlink() for p in Path(path).rglob('*')),'Symlink in raw bundle')
        return
    with zipfile.ZipFile(path) as archive:
        members=archive.infolist()
        require(len(members)<=100000 and sum(m.file_size for m in members)<=16*1024**3,'Excessive archive size')
        for m in members:
            p=PurePosixPath(m.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in m.filename
                    and ':' not in m.filename and m.filename.rstrip('/')==p.as_posix()
                    and stat.S_IFMT(m.external_attr>>16) in (0,stat.S_IFREG,stat.S_IFDIR)
                    and m.file_size<=4*1024**3,'Unsafe archive member')
        require(archive.testzip() is None,'Archive CRC mismatch')


def analyze_bundle(path,binding,cache_root):
    require(np.__version__.split('.')[0]=='1','Raw independent acceptance requires NumPy1')
    safe_archive(path)
    bundle=Bundle(path)
    try:
        cfg=bundle.json('config.json')
        calibration=bundle.read('spectral_calibration.json') if cfg.get('scdh_calibration') else None
        report=accepted.analyze_bundle(path,binding,cache_root,
            config_validator=lambda c:study.validate_config(c,calibration_raw=calibration),
            scoped_partial=cfg['scdh_spectralz']['stage'] in ('technical','calibration','timing'))
        required=binding['required_source_files']
        for name in CRITICAL:
            require(any(prefix+name in required for prefix in ('repo/','assets/source_snapshot/')),'Unbound critical study source: '+name)
        for name in ('command.txt','config.json','environment.txt','stdout.log','summary.csv','per_corruption.csv','notes.md'):
            require(name in bundle.names,'Missing standard artifact '+name)
        command=shlex.split(bundle.read('command.txt').decode())
        route=cfg['scdh_spectralz']
        require(len(command)>2 and Path(command[1]).name=='eval_scdh_spectralz.py','Wrong executed command entrypoint')
        for flag,value in (('--stage',route['stage']),('--arm',route['arm']),('--seed',str(cfg['seed']))):
            require(command.count(flag)==1 and command[command.index(flag)+1]==value,'Command/config mismatch '+flag)
        require(command.count('--alpha-h')==1 and float(command[command.index('--alpha-h')+1])==route['alpha_h_local'],'Command local step mismatch')
        require(cfg.get('native_extensions') and cfg.get('lion_state_before')==cfg.get('lion_state_after'),'Missing native/frozen state proof')
        if calibration:
            proof=json.loads(calibration)['binding']
            for key in ('asset_manifest_sha256','source_snapshot_sha256','lion_state_before','native_extensions','executed_source_files'):
                require(proof.get(key)==cfg.get(key),'Calibration runtime binding mismatch: '+key)
        rows=list(csv.DictReader(io.StringIO(bundle.read('per_corruption.csv').decode())))
        require(len(rows)==len(cfg['corruptions']) and [r['corruption'] for r in rows]==cfg['corruptions'],'CSV corruption coverage mismatch')
        for row,correct in zip(rows,report['outcomes']['probability_mean']):
            require(row['status']=='complete' and int(row['n_examples'])==len(correct) and int(row['n_correct'])==sum(correct)
                    and abs(float(row['accuracy'])-np.mean(correct))<=1e-10,'CSV/PM count or fraction mismatch')
            require(all(str(row.get(k))==str(cfg.get(k)) for k in ('run_id','dataset','method','severity','seed')),'CSV run identity mismatch')
        summaries=list(csv.DictReader(io.StringIO(bundle.read('summary.csv').decode())))
        require(len(summaries)==1 and summaries[0]['status']==cfg['status']
                and int(summaries[0]['total_correct'])==sum(map(sum,report['outcomes']['probability_mean']))
                and int(summaries[0]['total_examples'])==len(cfg['corruptions'])*len(cfg['selected_original_ids']),'Summary/PM count mismatch')
        summary=summaries[0]
        require(all(str(summary.get(k))==str(cfg.get(k)) for k in ('run_id','dataset','method','severity','seed'))
                and int(summary['n_corruptions'])==len(rows)
                and np.isclose(float(summary['micro_accuracy']),int(summary['total_correct'])/int(summary['total_examples']),rtol=0,atol=1e-10)
                and np.isclose(float(summary['macro_accuracy']),np.mean([float(r['accuracy']) for r in rows]),rtol=0,atol=1e-10),
                'Summary identity, coverage or fraction units mismatch')
        details,observations,cost=[],[],[]
        technical_logits={}
        arm=cfg['scdh_spectralz']['arm']; route=cfg['scdh_spectralz']
        for guard in report['batch_guards']:
            corruption,batch=guard['corruption'],guard['batch'];n=len(guard['ids'])
            d=bundle.json('batches/%s/batch%04d/diagnostics.json'%(corruption,batch))
            metrics=d.get('guidance_metrics',{}).get('scdh_spectralz',{})
            cost.append(dict(corruption=corruption,batch=batch,n_objects=n,
                reconstruction_seconds=d.get('shared_reconstruction_seconds'),head_seconds=d.get('head_seconds'),
                peak_gpu_memory_mb=d.get('whole_process_peak_gpu_memory_mb')))
            if route['stage']=='technical':
                with np.load(io.BytesIO(bundle.read('batches/%s/batch%04d/predictions.npz'%(corruption,batch))),allow_pickle=False) as values:
                    technical_logits.setdefault(corruption,[]).extend(values['trajectory_logits'].tolist())
            if route['stage']=='timing':
                require(d.get('preparation_mode')=='fresh_encode','Timing used an encoder cache replay')
            if arm not in ('spectral','calibration'):continue
            graph=metrics.get('graph',{})
            if route['stage']=='timing':require(graph.get('cache_mode')=='fresh_eigh','Timing used an eigensystem cache replay')
            require(graph.get('profile')=='offdiag_static_raw_lambda_truncated_l2' and graph.get('beta')==2.
                    and graph.get('reduction')=='batch sum / (3*N)' and graph.get('pruning') is not None,
                    'Missing/wrong spectral graph/profile diagnostics')
            eigenvalues=np.asarray(graph.get('eigenvalues'))
            require(eigenvalues.shape==(n,2048) and np.isfinite(eigenvalues).all()
                    and (np.diff(eigenvalues,axis=1)>=0).all(),'Invalid raw ordered eigensystem')
            require(all(accepted._is_sha(graph.get(k)) for k in ('basis_sha256','weights_sha256','cache_file_sha256','eigenvalues_sha256')),
                    'Missing physical spectral identities')
            cache=Path(cache_root)/guard['cache']['relative_path']
            with np.load(cache,allow_pickle=False) as values:
                reference=values['local_latent'].reshape(n,2048,4)[:,:,:3]
            reference_hash=hashlib.sha256(str(reference.dtype).encode()+str(reference.shape).encode()+np.ascontiguousarray(reference).tobytes()).hexdigest()
            require(reference_hash==graph['binding']['reference_sha256'],
                    'Graph anchor differs from actual physical h0XYZ')
            require(metrics.get('global_diffusion') is False and metrics.get('original_z_decode') is True
                    and metrics.get('final_style_update_consumed_by_decode') is False,'Wrong diffusion/decoder routing')
            if arm=='calibration':
                records=metrics.get('calibration',[])
                steps=[9,18,27] if corruption=='background' else [2,4,6]
                require(len(records)==15 and {(r['loop_count'],r['M_max']) for r in records}=={(s,m) for s in steps for m in study.MODES},
                        'Incomplete/nonregistered label-free calibration states')
                for r in records:
                    c,s=np.asarray(r['style_scd_gradient_norm'],dtype=float),np.asarray(r['style_spectral_gradient_norm'],dtype=float)
                    require(c.shape==s.shape==(n,) and np.isfinite(c).all() and (c>=0).all()
                            and (s[np.isfinite(s)]>=0).all(),'Invalid calibration gradient norms')
                    require(r.get('excluded_nonfinite')==(~np.isfinite(s)).tolist(),'Missing explicit invalid-gradient calibration exclusions')
                    computed=[.003*float(a)/float(b) if np.isfinite(a) and np.isfinite(b) and b>0 else None for a,b in zip(c,s)]
                    require(len(r['ratios'])==n and all((a is None and b is None) or
                        (a is not None and b is not None and abs(a-b)<=max(1e-12,abs(a)*1e-10)) for a,b in zip(computed,r['ratios'])),
                        'Calibration ratio differs from raw norms')
                    observations.append(dict(r,corruption=corruption,original_ids=guard['ids']))
            else:
                records=metrics.get('steps',[])
                horizon=35 if corruption=='background' else 8
                require(len(records)==horizon and [r['loop_count'] for r in records]==list(range(1,horizon+1)), 'Missing routed steps')
                require(graph.get('M_max')==route['M_max'],'M does not reach actual loss cutoff')
                weights=np.exp(-2*eigenvalues);weights[:,route['M_max']:]=0
                require(np.allclose(weights.sum(1),graph['effective_spectral_mass'],rtol=1e-4,atol=1e-5),'Wrong truncated raw spectral filter mass')
                for r in records:
                    for k in ('local_scd_gradient_norm','style_scd_gradient_norm','style_spectral_gradient_norm',
                              'applied_local_spectral_norm','applied_style_scd_norm','applied_style_spectral_norm','applied_local_scd_norm'):
                        value=np.asarray(r[k]);require(value.shape==(n,) and np.isfinite(value).all() and (value>=0).all(),'Nonfinite/invalid route norm '+k)
                    require(r['applied_local_spectral_norm']==[0.]*n and r['applied_style_scd_norm']==[0.]*n
                            and r['alpha_h_local']==route['alpha_h_local'] and r['kappa_z_spectral_effective']==route['kappa_z_spectral_effective'],
                            'Cross-route update leaked or effective step changed')
                    require(np.allclose(r['applied_style_spectral_norm'],np.asarray(r['style_spectral_gradient_norm'])*route['kappa_z_spectral_effective'],rtol=1e-12,atol=0),
                            'Applied style norm differs from effective kappa')
                if route['stage']=='technical':
                    fd=metrics.get('native_finite_difference',{})
                    physical=bundle.read('native_fd/'+corruption+'.npz')
                    require(hashlib.sha256(physical).hexdigest()==fd.get('sha256') and fd.get('passed') is True,'Missing physical native FD gate')
                    verify_native_fd(physical,fd,graph,reference[:2])
                if route['stage']!='technical':
                    verify_calibrated_application(records)
            details.append(dict(corruption=corruption,batch=batch,ids=guard['ids'],metrics=metrics))
        report.update(correct=sum(map(sum,report['outcomes']['probability_mean'])),
            scdh_spectralz=route,slot=binding['slot'],spectral_details=details,calibration_observations=observations,
            calibration_binding={k:cfg[k] for k in ('asset_manifest_sha256','source_snapshot_sha256','lion_state_before','native_extensions','executed_source_files')},
            cost=cost,technical_logits=technical_logits)
        return report
    finally:bundle.close()


def verify_calibrated_application(records):
    values=[v for row in records for v in row['applied_style_spectral_norm']]
    require(values and np.isfinite(values).all() and min(values)>=0 and max(values)>0,
            'Missing nonzero finite calibrated spectral application witness')


def verify_native_fd(physical,fd,graph,reference):
    shapes=dict(basis=(2,2048,2048),eigenvalues=(2,2048),reference=(2,2048,3),
                clean=(2,2048,3),repeat_clean=(2,2048,3),local=(2,8192,1,1),local_gradient=(2,8192,1,1))
    with np.load(io.BytesIO(physical),allow_pickle=False) as archive:
        required=set(shapes)|{'style','style_gradient','direction_0','direction_1','metadata'}
        require(required<=set(archive.files),'Missing physical FD vectors')
        values={k:archive[k].copy() for k in required if k!='metadata'}
        require(json.loads(str(archive['metadata'].item()))=={k:v for k,v in fd.items() if k!='sha256'},'FD physical metadata differs')
    require(all(v.dtype.kind=='f' and np.isfinite(v).all() for v in values.values()),'Nonfinite physical FD vectors')
    require(all(values[k].shape==shape for k,shape in shapes.items()),'Invalid physical FD shapes')
    style,gradient=values['style'],values['style_gradient']
    require(style.ndim==4 and style.shape[0]==2 and style.shape[2:]==(1,1) and gradient.shape==style.shape,'Invalid physical style/gradient shape')
    require((np.linalg.norm(style.reshape(2,-1),axis=1)>0).all()
            and (np.linalg.norm(gradient.reshape(2,-1),axis=1)>0).all(),'Require nonzero finite physical style and gradient')
    require(np.array_equal(values['reference'],reference) and np.array_equal(values['eigenvalues'],np.asarray(graph['eigenvalues'])[:2]),'FD reference/eigenvalues differ from physical graph')
    require(np.allclose((values['basis'].astype(float)**2).sum(axis=1),1.,rtol=1e-3,atol=1e-3),'Invalid physical FD basis norms')
    for key,value in values.items():
        digest=hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
        require(fd.get('physical_hashes',{}).get(key)==digest,'FD physical vector hash mismatch: '+key)
    require(fd.get('direction_policy')=='normalized_style_gradient','Unregistered FD direction policy')
    require(fd.get('relative_tolerance')==.1 and fd.get('absolute_tolerance')==1e-5 and fd.get('passed') is True
            and accepted._is_sha(fd.get('rng_before_sha256')) and fd['rng_before_sha256']==fd.get('rng_after_sha256')
            and set(fd.get('inputs_before',{}))=={'local','style'} and all(accepted._is_sha(v) for v in fd['inputs_before'].values())
            and fd['inputs_before']==fd.get('inputs_after'),'FD changed inputs/RNG or tolerance')
    checks=fd.get('checks',[])
    require(len(checks)==4 and {(r['example'],r['scale']) for r in checks}=={(i,s) for i in (0,1) for s in (.001,.01)},'Incomplete physical FD scales')
    for check in checks:
        i=check['example'];direction=values['direction_%d'%i]
        require(direction.shape==style[i].shape and np.isclose(np.linalg.norm(direction),1.,rtol=1e-4,atol=1e-6),'Invalid FD perturbation direction')
        require(np.allclose(direction,gradient[i]/np.linalg.norm(gradient[i]),rtol=1e-4,atol=1e-6),'FD direction differs from normalized gradient')
        epsilon=float(np.linalg.norm(style[i]))*check['scale']
        require(np.isfinite(epsilon) and epsilon>0 and np.isclose(epsilon,check['epsilon'],rtol=1e-4,atol=0),'Wrong FD perturbation scale')
        derivative=float((gradient[i].astype(float)*direction).sum())
        estimate=(check['loss_plus']-check['loss_minus'])/(2*check['epsilon'])
        require(np.isfinite(estimate) and np.isclose(derivative,check['autograd'],rtol=1e-4,atol=1e-7)
                and np.isclose(estimate,check['finite_difference'],rtol=1e-4,atol=1e-7),'Physical FD derivative mismatch')
        absolute=abs(estimate-derivative);denominator=max(abs(estimate),abs(derivative))
        relative=absolute/denominator if denominator else 0.
        require(check.get('passed') is True and (relative<=.1 or absolute<=1e-5),'Independent finite-difference tolerance failed')


def reuse_binding(report,proof):
    binding=report['calibration_binding']
    if proof is None:return binding
    route=report.get('scdh_spectralz',{})
    require(route.get('stage')=='technical','Source reuse is only for native technical controls')
    pair=(binding.get('asset_manifest_sha256'),binding.get('source_snapshot_sha256'))
    old=pair==(proof['old_asset'],proof['old_snapshot'])
    require(old or pair==(proof['new_asset'],proof['new_snapshot']),'Unregistered source reuse identity')
    require(not old or route.get('arm') in ('scd','off'),'Changed graph source cannot reuse an old spectral run')
    files=dict(binding['executed_source_files'])
    revisions={'graph_spectral_v2_offdiag.py':dict(old=proof['old_graph'],new=proof['new_graph'])}
    revisions.update(proof.get('audit_only_revisions',{}))
    require(set(revisions)<= {'graph_spectral_v2_offdiag.py','tta.py','scdh_spectralz.py'},'Unknown source reuse revision')
    for name,revision in revisions.items():
        expected=revision['old' if old else 'new']
        keys=[k for k in files if k in ('repo/'+name,'assets/source_snapshot/'+name)]
        require(keys and all(files[k]==expected for k in keys),'Unregistered revision in source reuse: '+name)
        for key in keys:files.pop(key)
    return dict({k:v for k,v in binding.items() if k not in ('asset_manifest_sha256','source_snapshot_sha256','executed_source_files')},executed_source_files=files)


def compare_physical(reports,*,technical_reuse=None):
    require(len(reports)>=2,'Need paired accepted reports')
    first=reports[0]
    for report in reports[1:]:
        require(report['object_ids']==first['object_ids'] and report['labels']==first['labels'] and report['seed']==first['seed']
                and report['corruptions']==first['corruptions'] and reuse_binding(report,technical_reuse)==reuse_binding(first,technical_reuse),'Unmatched cohort/seed/source/state')
        require(len(report['batch_guards'])==len(first['batch_guards']),'Incomplete paired guards')
        for a,b in zip(first['batch_guards'],report['batch_guards']):
            for key in ('corruption','batch','ids','initial_hashes','initial_rng_sha256','cache','scheduler'):
                require(a[key]==b[key],'Physical paired guard mismatch: '+key)
    graphs={}
    for report in reports:
        for item in report.get('spectral_details',[]):
            graph=item['metrics']['graph'];key=(item['corruption'],item['batch'])
            identity={k:graph[k] for k in ('basis_sha256','eigenvalues_sha256','cache_file_sha256')}
            require(key not in graphs or graphs[key]==identity,'Matched M/rho arms used a different physical eigensystem')
            graphs[key]=identity
    return True


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=analyze_bundle(args.bundle,json.loads(args.binding.read_text()),args.cache_root)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print('SCDH_SPECTRALZ_RAW_ACCEPTED',args.output,flush=True)

if __name__=='__main__':main()
