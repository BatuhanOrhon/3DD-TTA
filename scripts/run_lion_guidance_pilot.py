#!/usr/bin/env python3
"""Sequential, paired A/B/C screening on source-pinned Colab assets only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('asset-root','manifest-sha256','selection','output','cache-root'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--phase',choices=('technical','a','b','c'),required=True)
    parser.add_argument('--tau',type=float,default=.06)
    parser.add_argument('--corruptions',nargs='+',choices=('gaussian','background','density','occlusion'))
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    config=vars(args).copy()
    write(output/'command.json',sys.argv)
    (output/'command.txt').write_text(' '.join(sys.argv)+'\n')
    write(output/'config.json',config)
    rows=[]
    try:
        import numpy as np
        import torch
        from scripts import eval_multibackbone_source as source
        from scripts import eval_multibackbone_guidance as shared
        from scripts.diagnose_decoded_spectral import configuration,apply_backend_policy,verify_setup
        from legacy_trajectory import isolated_probe,_rng_state,_restore_rng,tensor_hash
        from shared_trajectory import SharedTrajectoryObserver
        from semantic_trajectory import gradient_probe
        from lion_guidance_pilot import sample,repair_candidates,proximal_chamfer,bridge_clean_delta,feature_matches
        from pointnet2_ops import pointnet2_utils
        asset=Path(args.asset_root)
        manifest=source.read_manifest(asset,args.manifest_sha256)
        selection=json.loads(Path(args.selection).read_text())
        assert selection['seed']==0 and selection['dataset']=='modelnet40_c'
        calibration=selection['calibration'];pilot=selection['pilot']
        assert len(calibration)==4 and len(pilot)==16 and len(set(calibration+pilot))==20
        assert all(type(i) is int and 0<=i<2468 for i in calibration+pilot)
        ids=calibration if args.phase=='technical' else pilot
        corruptions=args.corruptions or ['gaussian','background','density','occlusion']
        assert len(corruptions)==len(set(corruptions))
        base=configuration(['--asset-root',str(asset),'--asset-manifest-sha256',args.manifest_sha256,
            '--result-root',str(output),'--paired-state-root',args.cache_root,'--arm','control','--count','2',
            '--backend-policy','controlled','--corruptions',*corruptions])
        backend=apply_backend_policy('controlled')
        setup=verify_setup(asset,manifest,base)
        dataset=source.dataset_manifest(manifest,base)
        base['dataset_root']=dataset['root']
        base.update(count=len(ids),selected_original_ids=ids)
        write(output/'resolved_config.json',dict(base=base,phase=args.phase,arms={
            'a':['full','full_repeat','style_frozen','sparse','cheap','hybrid'],
            'b':['full','full_repeat','geometric','feature','shuffled','metric4d'],
            'c':['full','semantic_zero','semantic_zero_repeat','causal_feature','fixed_feature','causal_all']},
            proximal_tau=args.tau,exact_refresh_zero_based_modulo=3,feature_beta=.1,feature_neighbors=8,
            feature_fraction=.01,decoded_geometry_relative_rms_limit=.001,
            latent_repeat_rms_limit=1e-5,logit_repeat_rms_limit=1e-4,
            decoder_style='original_encoded',probability_mean='equal_three_clean_quarters_and_terminal'))
        labels=source.load_array(Path(dataset['root'])/'label.npy',dataset['files']['label.npy'],base,labels=True)
        arrays={c:source.load_array(Path(dataset['root'])/('data_'+c+'_5.npy'),dataset['files']['data_'+c+'_5.npy'],base) for c in corruptions}
        lion,graph,original,ops,native,resolved=shared.load_guidance_runtime(dict(base,method=shared.ORIGINAL_METHOD),manifest)
        classifier=source.load_classifier(base,manifest)
        models=[lion.vae,lion.priors,classifier]
        initial_states=[source.state_hash(m) for m in models]
        write(output/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),backend=backend,native=native,
            checkpoint_load=base['checkpoint_load'],source_commit=manifest.get('source_commit'),
            source_overlay_receipt=manifest.get('source_overlay_receipt'),setup_identity=setup['identity'],
            dataset=dataset,selection=selection,lion_config=resolved,initial_state_hashes=initial_states,
            timing='B2 synchronized sampler+4decode+4classifier; no encode/cache/hash/archive cost; not end-to-end deployment'))
        (output/'environment.txt').write_text('See environment.json: exact pinned source/native/assets and runtime.\n')
        tau_scores={str(t):[] for t in (.01,.03,.06,.1,.3)}
        parity=[];pilot_native=[]
        for corruption in corruptions:
            steps=35 if corruption=='background' else 8
            for start in range(0,len(ids),2):
                batch_ids=ids[start:start+2]
                case=output/(corruption+'_'+str(start//2));case.mkdir()
                shared.seed_inference(0)
                raw=torch.from_numpy(np.array(arrays[corruption][batch_ids],dtype=np.float32,copy=True))
                normalized,center,scale=ops.normalize(raw)
                prepared=ops.rotate_pointcloud(torch.from_numpy(ops.upsample_all(normalized.numpy(),2048)).float().cuda()*3.3885)
                before_prepare=_rng_state()
                path=Path(args.cache_root)/(corruption+'_'+hashlib.sha256(np.array(batch_ids,dtype=np.int64).tobytes()).hexdigest()+'.npz')
                state,recorder=shared.paired_preparation(path,prepared)
                if state is None:
                    with torch.no_grad():
                        latents=lion.vae.encode(prepared)
                        z=latents[2][0][0].unsqueeze(2).unsqueeze(3)
                        h=latents[2][1][0].unsqueeze(2).unsqueeze(3)
                        lion.vae.global2style(z)
                        state=dict(shape_latent=z,local_latent=h,noise=torch.randn_like(h))
                    recorder(state)
                    state['rng_after_prepare']=_rng_state()
                write(case/'paired_state.json',dict(path=str(path),identity=source.file_identity(path),ids=batch_ids,
                    input_sha256=tensor_hash(prepared),state_hashes={k:tensor_hash(state[k]) for k in ('shape_latent','local_latent','noise')}))
                np.savez_compressed(case/'frame.npz',input=prepared.cpu().numpy(),raw=raw.numpy(),center=center.cpu().numpy(),scale=scale.cpu().numpy())
                def fps(points,n):
                    indices=pointnet2_utils.furthest_point_sample(points.contiguous(),n)
                    return pointnet2_utils.gather_operation(points.transpose(1,2).contiguous(),indices).transpose(1,2).contiguous()
                def decode(h):
                    return lion.vae.decoder(None,beta=None,context=h.reshape(len(h),-1),style=state['shape_latent'].reshape(len(h),-1))
                def classify(decoded):
                    geometry=shared.decoded_to_geometry(decoded,'modelnet40_c',scale,center,ops)
                    points,features=source.preprocess(geometry,base,fps)
                    return classifier(points,features=features)
                def original_run():
                    class Capture(SharedTrajectoryObserver):
                        def clean(self,h,z,i,t):
                            if i+1 in self.steps: snapshots.append(h.detach().clone())
                            return super().clean(h,z,i,t)
                        def final_state(self,h,z,t):
                            snapshots.append(h.detach().clone())
                            return super().final_state(h,z,t)
                    snapshots=[]
                    observer=Capture(lion.vae,{'pointmae':classifier},lambda x:{'pointmae':classify(x)},steps,40,
                        checkpoint_steps=[2,4,6] if steps==8 else [9,18,27])
                    result=original(prepared,lion,steps,.03,.003,.95,prepared_state=state,trajectory_observer=observer)
                    with isolated_probe(models): observer.terminal({'pointmae':classify(result)})
                    return snapshots,torch.as_tensor(observer.arrays('pointmae')['trajectory_logits'],device='cuda')
                arms={'technical':['original','original_repeat','full','full_repeat','semantic_zero','semantic_zero_repeat'],
                      'a':['full','full_repeat','style_frozen','sparse','cheap','hybrid'],
                      'b':['full','full_repeat','geometric','feature','shuffled','metric4d'],
                      'c':['full','semantic_zero','semantic_zero_repeat','causal_feature','fixed_feature','causal_all']}[args.phase]
                reference_snapshots,reference_logits=None,None
                results={}
                for arm in arms:
                    saved_logits=[];decoded_clouds=[];repair_witness=[]
                    def callback(h,z,loop):
                        if loop==steps:
                            with gradient_probe(models):
                                decoded=decode(h.detach().clone().requires_grad_(True))
                            with isolated_probe(models): prediction=classify(decoded)
                        else:
                            with isolated_probe(models):
                                decoded=decode(h.detach().clone())
                                prediction=classify(decoded)
                        saved_logits.append(prediction.detach())
                        decoded_clouds.append(decoded.detach())
                    def diagnostic(values):
                        if values['loop']!=1:return
                        clean=values['clean'].reshape(2,2048,4);ref=values['reference']
                        actual=values['delta'].reshape(2,2048,4)[...,:3].norm(dim=(1,2)).clamp_min(1e-12)
                        from diffusers import DDIMScheduler
                        scheduler=DDIMScheduler();alpha=scheduler.alphas_cumprod[max(0,values['timestep']-10)]
                        for tau in tau_scores:
                            corrected=proximal_chamfer(clean[...,:3],ref[...,:3],values['idx1'],values['idx2'],values['keep1'],values['keep2'],float(tau))
                            norm=(corrected-clean[...,:3]).norm(dim=(1,2))*alpha.sqrt().to(actual.device)
                            tau_scores[tau].extend((norm.clamp_min(1e-12).log()-actual.log()).abs().cpu().tolist())
                        np.savez_compressed(case/'calibration_first_step.npz',clean=clean.cpu().numpy(),reference=ref.cpu().numpy(),
                            exact_delta=values['delta'].cpu().numpy(),idx1=values['idx1'].cpu().numpy(),idx2=values['idx2'].cpu().numpy(),
                            keep1=values['keep1'].cpu().numpy(),keep2=values['keep2'].cpu().numpy())
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                    _restore_rng(state['rng_after_prepare'])
                    if arm in ('original','original_repeat'):
                        snapshots,logits=original_run();details=[]
                    elif arm.startswith(('semantic','causal','fixed')):
                        if reference_snapshots is None: raise RuntimeError('C requires completed same-case full trajectory')
                        fixed=None
                        with gradient_probe(models):
                            if arm=='fixed_feature':
                                points,features=source.preprocess(raw.cuda(),base,fps)
                                fixed=classifier(points,features=features).detach().softmax(-1)
                            logits,details=repair_candidates([h.reshape(2,2048,4) for h in reference_snapshots],decode,classify,
                                fraction=0. if arm.startswith('semantic_zero') else .01,fixed_teacher=fixed,feature_only=arm!='causal_all',
                                diagnostic=lambda values:repair_witness.append({k:v.detach() if hasattr(v,'detach') else v for k,v in values.items()}))
                        snapshots=[d['chosen_latent'].reshape(2,8192,1,1) for d in repair_witness]
                        decoded_clouds=[d['chosen_decoded'] for d in repair_witness]
                    else:
                        snapshots,details=sample(lion,state,steps,'full' if arm=='full_repeat' else arm,tau=args.tau,
                            callback=callback,diagnostic=diagnostic if args.phase=='technical' and arm=='full' else None)
                        logits=torch.stack(saved_logits,1)
                    torch.cuda.synchronize();seconds=time.perf_counter()-tick
                    peak=torch.cuda.max_memory_allocated()/1024**2
                    assert logits.shape==(2,4,40) and torch.isfinite(logits).all()
                    pm=logits.softmax(-1).mean(1);pred=pm.argmax(-1).cpu().numpy()
                    if arm=='full':reference_snapshots,reference_logits=snapshots,logits
                    results[arm]=dict(latents=torch.stack(snapshots,1).cpu().numpy(),logits=logits.cpu().numpy())
                    if repair_witness:
                        np.savez_compressed(case/(arm+'_repair.npz'),**{key:torch.stack([d[key] for d in repair_witness],1).cpu().numpy()
                            for key in repair_witness[0] if key!='position'})
                    np.savez_compressed(case/(arm+'.npz'),**results[arm],index=np.array(batch_ids),label=np.asarray(labels[batch_ids]).reshape(-1),
                        probability_mean=pm.cpu().numpy(),prediction=pred,
                        decoded=np.stack([x.cpu().numpy() for x in decoded_clouds],1) if decoded_clouds else np.empty(0))
                    write(case/(arm+'.json'),dict(seconds=seconds,peak_mb=peak,details=details,
                        extra_cost_only=arm.startswith(('semantic','causal','fixed'))))
                    for position,identity in enumerate(batch_ids):
                        rows.append(dict(phase=args.phase,arm=arm,corruption=corruption,object_id=identity,
                            correct=int(pred[position]==int(labels[identity])),seconds_per_batch=seconds,peak_mb=peak))
                    print('LION_PILOT_PROGRESS',args.phase,corruption,batch_ids,arm,round(seconds,3),flush=True)
                if args.phase=='technical':
                    def differences(a,b):
                        return {key:float(np.sqrt(np.mean((results[a][key].astype(np.float64)-results[b][key])**2))) for key in ('latents','logits')}
                    repeat=differences('original','original_repeat');cross=differences('original','full');newrepeat=differences('full','full_repeat')
                    passed=all(v['latents']<=1e-5 and v['logits']<=1e-4 for v in (repeat,cross,newrepeat))
                    semantic_repeat=differences('semantic_zero','semantic_zero_repeat')
                    zero=[]
                    for name in ('semantic_zero','semantic_zero_repeat'):
                        with np.load(case/(name+'_repair.npz')) as d:
                            zero.append({key:float(np.sqrt(np.mean((d[a].astype(np.float64)-d[b])**2)))
                                for key,a,b in (('decoded','before','after'),('latent','latent','attempted_latent'),('logits','before_logits','after_logits'))})
                    semantic_passed=semantic_repeat['logits']<=1e-4 and all(d['latent']==0 and d['decoded']<=1e-5 and d['logits']<=1e-4 for d in zero)
                    parity.append(dict(ids=batch_ids,corruption=corruption,original_repeat=repeat,cross=cross,experimental_repeat=newrepeat,
                        semantic_repeat=semantic_repeat,semantic_within=zero,semantic_passed=semantic_passed,passed=passed))
                    write(case/'parity.json',parity[-1])
                else:
                    control='semantic_zero' if args.phase=='c' else 'full'
                    repeat={key:float(np.sqrt(np.mean((results[control][key].astype(np.float64)-results[control+'_repeat'][key])**2)))
                        for key in ('latents','logits')}
                    passed=repeat['latents']<=1e-5 and repeat['logits']<=1e-4
                    within=[]
                    if args.phase=='c':
                        for name in ('semantic_zero','semantic_zero_repeat'):
                            with np.load(case/(name+'_repair.npz')) as d:
                                within.append({key:float(np.sqrt(np.mean((d[a].astype(np.float64)-d[b])**2)))
                                    for key,a,b in (('decoded','before','after'),('latent','latent','attempted_latent'),('logits','before_logits','after_logits'))})
                        passed=passed and all(d['latent']==0 and d['decoded']<=1e-5 and d['logits']<=1e-4 for d in within)
                    pilot_native.append(dict(corruption=corruption,ids=batch_ids,control=control,repeat=repeat,within=within,passed=passed))
                    write(case/'pilot_native.json',pilot_native[-1])
                with (output/'progress.json').open('w') as f: json.dump(dict(completed_rows=len(rows),elapsed=time.monotonic()-started),f)
        assert initial_states==[source.state_hash(m) for m in models], 'Frozen model state changed'
        if args.phase=='technical':
            chosen=min(tau_scores,key=lambda t:float(np.median(tau_scores[t])))
            write(output/'technical.json',dict(passed=all(p['passed'] for p in parity),semantic_passed=all(p['semantic_passed'] for p in parity),parity=parity,
                tau_scores=tau_scores,selected_tau=float(chosen),selection='minimum median absolute log update-norm ratio; no labels',
                limits='Strict same-run numerical controls, not proof of population accuracy or determinism'))
        with (output/'per_object.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summary=[]
        for arm in sorted({r['arm'] for r in rows}):
            selected=[r for r in rows if r['arm']==arm]
            summary.append(dict(arm=arm,n=len(selected),n_correct=sum(r['correct'] for r in selected),
                accuracy_percent=100*sum(r['correct'] for r in selected)/len(selected),
                seconds=sum(r['seconds_per_batch'] for r in selected)/2,
                extra_cost_only=arm.startswith(('semantic','causal','fixed'))))
        from research_artifacts import corruption_row,summarize,PER_COLUMNS,SUMMARY_COLUMNS
        per_rows=[];summary_rows=[]
        for arm in sorted({r['arm'] for r in rows}):
            own=[]
            for corruption in corruptions:
                selected=[r for r in rows if r['arm']==arm and r['corruption']==corruption]
                own.append(corruption_row(output.name,0,corruption,len(selected),sum(r['correct'] for r in selected),
                    sum(r['seconds_per_batch'] for r in selected)/2,max(r['peak_mb'] for r in selected),'partial',method=arm))
            per_rows.extend(own);summary_rows.append(summarize(own,'partial'))
        for filename,columns,values in (('per_corruption.csv',PER_COLUMNS,per_rows),('summary.csv',SUMMARY_COLUMNS,summary_rows)):
            with (output/filename).open('x',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(values)
        write(output/'summary.json',dict(status='complete',results=summary,elapsed=time.monotonic()-started,
            pilot_native=pilot_native,pilot_native_passed=all(r['passed'] for r in pilot_native) if pilot_native else None,
            scope='selected exposed development objects/'+','.join(corruptions)+'/seed0/B2; C is conditional on frozen shared readouts; not heldout or deployment benchmark'))
        (output/'notes.md').write_text('Exploratory paired pilot. See config/environment and per-object artifacts; technical gates govern interpretation.\n')
        print('LION_GUIDANCE_PHASE_COMPLETE',args.phase,flush=True)
    except BaseException:
        write(output/'failure.json',dict(traceback=traceback.format_exc(),elapsed=time.monotonic()-started,completed_rows=rows))
        raise


if __name__=='__main__':main()
