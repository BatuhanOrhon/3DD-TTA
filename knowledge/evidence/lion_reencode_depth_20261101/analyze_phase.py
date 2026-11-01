"""Independent NumPy1 recount and immutable physical-reference acceptance."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from scripts.analyze_scan_trajectory import Bundle, require
from scripts.eval_lion_reencode_depth import validate_config


def identity(path):
    data=Path(path).read_bytes()
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def tensor_hash(value):
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def probabilities(logits):
    logits=np.asarray(logits,dtype=np.float64)
    require(logits.ndim==3 and logits.shape[1:]==(4,40) and np.isfinite(logits).all(),'Invalid four-candidate logits')
    exp=np.exp(logits-logits.max(-1,keepdims=True))
    return (exp/exp.sum(-1,keepdims=True)).mean(1)


def validate_noisy_state(proof,alpha,actual_hash):
    alpha=np.float32(alpha)
    noisy=np.sqrt(alpha)*proof['selected_h'].reshape(-1,8192,1,1)+np.sqrt(np.float32(1)-alpha)*proof['diffusion_noise']
    require(np.isfinite(noisy).all() and tensor_hash(noisy)==actual_hash,'Actual noisy state differs from depth-matched alpha/h/noise')


def repeatability(actual,reference):
    left,right=probabilities(actual),probabilities(reference)
    return dict(bitwise_equal=bool(np.array_equal(actual,reference)),
        max_logit_abs=float(np.max(np.abs(actual-reference))),
        candidate_disagreements=int((actual.argmax(-1)!=reference.argmax(-1)).sum()),
        pm_disagreements=int((left.argmax(-1)!=right.argmax(-1)).sum()),
        terminal_disagreements=int((actual[:,-1].argmax(-1)!=reference[:,-1].argmax(-1)).sum()))


def analyze(phase,reference,output=None):
    phase,reference=Path(phase),Path(reference)
    output=Path(output) if output else phase/'analysis';output.mkdir(parents=True,exist_ok=True)
    paths=list((phase/'raw').glob('*/*/*/config.json'));require(len(paths)==1,'Need one replay bundle')
    bundle=Bundle(paths[0].parent)
    try:
        cfg=bundle.json('config.json');validate_config(cfg)
        policy=json.loads((phase/'repeatability_policy.json').read_text())
        require(identity(phase/'prior_repeat_diagnostic.json')==policy['diagnostic_identity'],'Wrong repeatability diagnostic')
        diagnostic=json.loads((phase/'prior_repeat_diagnostic.json').read_text())
        require(diagnostic['rng_unchanged'] and diagnostic['model_unchanged'] and diagnostic['initial_state_matches_archived'] and any(not p['exact'] for p in diagnostic['repeated_noise_prediction']),'Native nonrepeatability was not demonstrated')
        require(cfg['status'] in ('complete','partial') and not cfg.get('failure'),'Failed replay')
        require(cfg['lion_state_before']==cfg['lion_state_after'] and cfg['classifier_state_before_sha256']==cfg['classifier_state_after_sha256'],'Model changed')
        require(identity(reference/'evidence_manifest.json')==cfg['reencode_replay']['manifest_identity'],'Wrong reference manifest')
        rm=json.loads((reference/'evidence_manifest.json').read_text())
        for name,value in cfg['executed_source_files'].items():
            if name.startswith('repo/'):
                raw=bundle.read('source_snapshot/'+name[5:])
                require(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==value,'Missing executed source')
        require(identity(phase/'label.npy')=={k:cfg['dataset_provenance']['files']['label.npy'][k] for k in ('bytes','sha256')},'Labels changed')
        labels=np.load(phase/'label.npy',allow_pickle=False).reshape(-1)
        names=('first_terminal','first_pm','cycle','n3_pm','second_terminal','second_pm','combined_pm')
        outcomes={name:[] for name in names};rows=[];batches=0;repeat_reports=[]
        steps=cfg['reencode']['second_reverse_steps']
        for corruption in cfg['corruptions']:
            seen=[];correct={name:[] for name in names}
            shards=sorted((paths[0].parent/'batches'/corruption).glob('batch*'))
            require(len(shards)==(len(cfg['selected_original_ids'])+31)//32,'Missing batch')
            for shard in shards:
                diag=json.loads((shard/'diagnostics.json').read_text())
                with np.load(shard/'predictions.npz',allow_pickle=False) as saved:a={k:saved[k].copy() for k in saved.files}
                ids=a['index'];seen.extend(ids.tolist());count=len(ids)
                require(np.array_equal(a['label'],labels[ids]),'Wrong labels')
                refs=diag['trajectory']['reference_files']
                require(refs==diag['guidance_metrics']['replay']['reference_files'],'Conflicting reference binding')
                for name,value in refs.items():require(rm[name]==value==identity(reference/name),'Changed physical reference')
                predpath=next(reference/k for k in refs if k.endswith('/predictions.npz'))
                proofpath=next(reference/k for k in refs if k.endswith('.reencode.npz'))
                oldcache=next(reference/k for k in refs if k.endswith('.npz') and '/paired/' in '/'+k and not k.endswith('.reencode.npz'))
                with np.load(predpath,allow_pickle=False) as saved:old={k:saved[k].copy() for k in saved.files}
                with np.load(proofpath,allow_pickle=False) as saved:proof={k:saved[k].copy() for k in saved.files if k!='metadata'}
                require(np.array_equal(ids,old['index']) and np.array_equal(a['first_trajectory_logits'],old['first_trajectory_logits']) and np.array_equal(a['cycle_logits'],old['cycle_logits']) and np.array_equal(a['reference_n3_trajectory_logits'],old['trajectory_logits']),'Archived predictions changed')
                cached=phase/'paired/modelnet40_c'/corruption/oldcache.name
                require(identity(cached)==identity(oldcache)=={k:diag['paired_state'][k] for k in ('bytes','sha256')},'Original cache bytes changed')
                trace=diag['trajectory']['second']
                require(trace['complete'] and trace['probe_rng_unchanged'] and trace['probe_buffers_unchanged'],'Incomplete or impure trace')
                require(all(p['rng_unchanged'] and p['buffers_unchanged'] and p['live_inputs_unchanged'] for p in trace['probes']),'Probe changed trajectory')
                initial=trace['initial_hashes']
                for key,proofkey,width in [('shape_latent','selected_z',128),('local_latent','selected_h',8192)]:
                    require(initial[key]==tensor_hash(proof[proofkey].reshape(count,width,1,1)),'Wrong replay posterior')
                require(initial['noise']==tensor_hash(proof['diffusion_noise']) and initial['input_points']==tensor_hash(proof['bridge_points']),'Wrong Gaussian/frame')
                require(trace['final_hashes']['style']==initial['shape_latent'],'Decoder lost own fixed z')
                anchor=proof['selected_h'].reshape(count,2048,4)[:,:,:3]
                require(tensor_hash(anchor)==diag['guidance_metrics']['replay']['second_chamfer_reference_sha256'],'Wrong new-h SCD anchor')
                olddiag=json.loads((next(reference/k for k in refs if k.endswith('/diagnostics.json'))).read_text())
                require(trace['initial_rng_sha256']==olddiag['trajectory']['second']['initial_rng_sha256'],'Wrong replay RNG')
                scheduler=diag['scheduler'];times=list(range((steps-1)*10,-1,-10))
                require(scheduler['active_timesteps']==times and scheduler['reverse_steps']==steps,'Wrong scheduler depth')
                beta=np.linspace(.0001,.02,1000,dtype=np.float64);alpha=float(np.prod(1-beta[:times[0]+1]))
                require(abs(scheduler['initial_alpha_cumprod']-alpha)<1e-6,'Wrong initial alpha')
                validate_noisy_state(proof,scheduler['initial_alpha_cumprod'],initial['noisy_local'])
                require(trace['candidate_loop_counts']==cfg['trajectory_selection']['normal_loop_counts'],'Wrong PM schedule')
                first,second,n3=probabilities(a['first_trajectory_logits']),probabilities(a['trajectory_logits']),probabilities(a['reference_n3_trajectory_logits'])
                require(np.allclose(a['first_probabilities'],first,rtol=0,atol=1e-12) and np.allclose(a['second_probabilities'],second,rtol=0,atol=1e-12),'Wrong stored PM')
                require(a['cycle_logits'].shape==(count,40) and np.isfinite(a['cycle_logits']).all(),'Bad cycle logits')
                predictions=dict(first_terminal=a['first_trajectory_logits'][:,-1].argmax(-1),first_pm=first.argmax(-1),cycle=a['cycle_logits'].argmax(-1),n3_pm=n3.argmax(-1),second_terminal=a['trajectory_logits'][:,-1].argmax(-1),second_pm=second.argmax(-1),combined_pm=((first+second)*.5).argmax(-1))
                require(np.array_equal(a['prediction'],predictions['second_pm']) and np.array_equal(a['terminal_prediction'],predictions['second_terminal']) and np.array_equal(a['combined_prediction'],predictions['combined_pm']),'Wrong primary/terminal/pooled payload')
                if steps==3:
                    repeat_reports.append(dict(corruption=corruption,ids=ids.tolist(),**repeatability(a['trajectory_logits'],old['trajectory_logits'])))
                for name in names:correct[name].extend((predictions[name]==a['label']).tolist())
                batches+=1
            require(seen==cfg['selected_original_ids'],'Missing/duplicate/reordered IDs')
            row=dict(corruption=corruption,n=len(seen))
            for name in names:outcomes[name].append(correct[name]);row[name]=float(np.mean(correct[name])*100)
            rows.append(row)
        matrices={name:np.asarray(value,dtype=bool) for name,value in outcomes.items()}
        rng=np.random.RandomState(20261101);draw=rng.randint(0,len(seen),(10000,len(seen)));comparisons={}
        for left,right in [('second_pm','first_pm'),('combined_pm','first_pm'),('second_pm','n3_pm'),('second_terminal','first_terminal')]:
            delta=matrices[left].astype(float)-matrices[right]
            comparisons[left+'-'+right]=dict(delta_pp=float(delta.mean()*100),conditional95_pp=np.percentile(delta.mean(0)[draw].mean(1)*100,[2.5,97.5]).tolist(),corrected=int((matrices[left]&~matrices[right]).sum()),harmed=int((~matrices[left]&matrices[right]).sum()))
        report=dict(status='passed',second_reverse_steps=steps,n_original_objects=len(seen),n_corruptions=len(rows),seed=0,physical_batches=batches,accuracy_percent={name:float(value.mean()*100) for name,value in matrices.items()},comparisons=comparisons,per_corruption=rows,n3_repeatability=repeat_reports,repeatability_policy=policy,raw_config=identity(paths[0]),executed_source_sha256=cfg['executed_source_sha256'],limitation='exposed single-seed development scan5/8 plus fresh N3 replay control; object-cluster intervals omit seed/native repeat variability and multiplicity correction; first/cycle/old N3 are archived references; physical inputs exact, numerical outputs not promised bitwise')
        (output/'acceptance.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');np.savez_compressed(output/'outcomes.npz',**matrices)
        return report
    finally:bundle.close()


if __name__=='__main__':
    if sys.argv[1:]==['--self-check']:
        x=np.zeros((2,4,40));assert np.allclose(probabilities(x),.025)
        try:probabilities(x[:,:3])
        except ValueError:pass
        else:raise AssertionError('Accepted wrong candidate count')
        print('REENCODE_DEPTH_ANALYZER_SELF_CHECK_PASSED')
    else:print(json.dumps(analyze(*sys.argv[1:]),indent=2))
