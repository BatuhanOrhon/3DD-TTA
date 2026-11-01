"""Raw-shard and physical-posterior recount; NumPy CPU, no model inference."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from scripts.analyze_scan_trajectory import Bundle, require
from scripts.eval_lion_reencode import validate_config


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def tensor_hash(array):
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def probabilities(logits):
    logits=np.asarray(logits,dtype=np.float64)
    require(logits.ndim==3 and logits.shape[1:]==(4,40) and np.isfinite(logits).all(), 'Bad four-candidate logits')
    exp=np.exp(logits-logits.max(-1,keepdims=True))
    return (exp/exp.sum(-1,keepdims=True)).mean(1)


def cycle_prediction(logits, count):
    require(logits.shape==(count,40) and np.isfinite(logits).all(), 'Invalid cycle classifier logits')
    return logits.argmax(-1)


def physical(phase, corruption, diag, cfg):
    original=diag['paired_state']
    path=phase/'paired'/'modelnet40_c'/corruption/Path(original['path']).name
    require(identity(path)=={k:original[k] for k in ('bytes','sha256')}, 'Changed physical original cache')
    with np.load(path,allow_pickle=False) as saved:
        cache={k:saved[k].copy() for k in saved.files if k!='metadata'}
        meta=json.loads(saved['metadata'].item())
    for key in ('shape_latent','local_latent','noise'):
        require(tensor_hash(cache[key])==meta['tensor_hashes'][key], 'Original cache tensor changed')
    # NumPy1 is needed to reproduce the runtime pickle representation.
    if np.__version__.split('.')[0]=='1':
        py,nr=meta['python_rng'],meta['numpy_rng']
        state=((py[0],tuple(py[1]),py[2]),(nr[0],cache['numpy_rng_keys'],*nr[1:]),cache['torch_rng_cpu'],
               [cache['torch_rng_cuda_'+str(i)] for i in range(meta['cuda_devices'])])
        require(hashlib.sha256(pickle.dumps(state,protocol=4)).hexdigest()==meta['rng_after_sha256'], 'Serialized RNG differs')
    proof_path=path.with_suffix('.reencode.npz')
    expected=diag['guidance_metrics']['reencode']['proof']
    require(identity(proof_path)=={k:expected[k] for k in ('bytes','sha256')}, 'Bridge physical bytes differ')
    with np.load(proof_path,allow_pickle=False) as saved:
        proof={k:saved[k].copy() for k in saved.files if k!='metadata'}
        pm=json.loads(saved['metadata'].item())
    require(pm['binding']==dict(executed_source_sha256=cfg['executed_source_sha256'],
        asset_manifest_sha256=cfg['asset_manifest_sha256'],input_sha256=meta['input_sha256']), 'Wrong proof source/input binding')
    require(set(proof)==set(pm['tensor_hashes']), 'Wrong proof hash coverage')
    for key,value in proof.items():
        require(value.dtype==np.float32 and np.isfinite(value).all() and tensor_hash(value)==pm['tensor_hashes'][key], 'Changed proof tensor')
        if 'sigma' in key:require((value>0).all(), 'Nonpositive posterior scale')
    count=len(proof['selected_z'])
    require(proof['bridge_points'].shape==proof['cycle_points'].shape==(count,2048,3), 'Wrong LION bridge frame')
    equations=[(cache['shape_latent'].reshape(count,128),proof['mu']+proof['sigma']*proof['eps_z']),
        (cache['local_latent'].reshape(count,8192),proof['h_mu']+proof['h_sigma']*proof['eps_h']),
        (proof['selected_z'],proof['new_z_mu']+proof['new_z_sigma']*proof['eps_z']),
        (proof['selected_h'],proof['new_h_mu']+proof['new_h_sigma']*proof['eps_h'])]
    require(all(np.allclose(a,b,rtol=5e-6,atol=1e-6) for a,b in equations), 'Posterior coupling equation differs')
    require(np.array_equal(proof['diffusion_noise'],cache['noise']), 'Local Gaussian changed')
    trajectories=diag['trajectory']
    for name in ('first','second'):
        trace=trajectories[name]
        require(trace['complete'] and trace['probe_rng_unchanged'] and trace['probe_buffers_unchanged'], 'Probe purity/completion failed')
        require(all(p['rng_unchanged'] and p['buffers_unchanged'] and p['live_inputs_unchanged'] for p in trace['probes']), 'Impure probe')
    a,b=trajectories['first'],trajectories['second']
    for name,width in [('shape_latent',128),('local_latent',8192)]:
        key='selected_z' if name=='shape_latent' else 'selected_h'
        require(a['initial_hashes'][name]==tensor_hash(cache[name]) and
                b['initial_hashes'][name]==tensor_hash(proof[key].reshape(count,width,1,1)), 'Wrong pass initial state')
    require(b['initial_hashes']['input_points']==tensor_hash(proof['bridge_points']), 'Second pass normalized/FPSed bridge')
    require(a['final_hashes']['style']==a['initial_hashes']['shape_latent'] and
            b['final_hashes']['style']==b['initial_hashes']['shape_latent'], 'Wrong own-z decoder style')
    metrics=diag['guidance_metrics']['reencode']
    anchor=proof['selected_h'].reshape(count,2048,4)[:,:,:3]
    require(tensor_hash(anchor)==metrics['second_chamfer_reference_sha256'], 'Second SCD target differs')
    require(metrics['second_scheduler']['active_timesteps']==[20,10,0] and metrics['second_reverse_steps']==3, 'Wrong N3 scheduler/noise depth')
    require(a['candidate_names']==cfg['first_trajectory_selection'][('background' if corruption=='background' else 'normal')+'_candidate_names'], 'First schedule changed')
    return 1


def analyze(phase,output=None):
    phase=Path(phase); output=Path(output) if output else phase/'analysis'
    output.mkdir(parents=True,exist_ok=True)
    paths=list((phase/'raw').glob('*/*/*/config.json'))
    require(len(paths)==1, 'Need one sealed Point-MAE two-pass bundle')
    bundle=Bundle(paths[0].parent)
    try:
        cfg=bundle.json('config.json');validate_config(cfg)
        require(cfg['status'] in (('complete','partial') if cfg['technical_smoke'] else ('complete',)) and not cfg.get('failure'), 'Incomplete or failed inference')
        require(cfg['lion_state_before']==cfg['lion_state_after'] and
            cfg['classifier_state_before_sha256']==cfg['classifier_state_after_sha256'], 'Model state changed')
        for name,value in cfg['executed_source_files'].items():
            if name.startswith('repo/'):
                raw=bundle.read('source_snapshot/'+name[5:])
                require(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==value, 'Executed source bytes absent')
        labels=np.load(phase/'label.npy',allow_pickle=False).reshape(-1)
        expected=cfg['dataset_provenance']['files']['label.npy']
        require(identity(phase/'label.npy')=={k:expected[k] for k in ('bytes','sha256')}, 'External label identity differs')
        outcomes={name:[] for name in ('first_terminal','first_pm','cycle','second_terminal','second_pm','combined_pm')}
        rows=[];proofs=0
        for corruption in cfg['corruptions']:
            batches=sorted((paths[0].parent/'batches'/corruption).glob('batch*'))
            require(len(batches)==(len(cfg['selected_original_ids'])+31)//32, 'Missing batch')
            seen=[];correct={name:[] for name in outcomes}
            for batch in batches:
                diag=json.loads((batch/'diagnostics.json').read_text())
                with np.load(batch/'predictions.npz',allow_pickle=False) as saved:
                    a={k:saved[k].copy() for k in saved.files}
                ids=a['index']; seen.extend(ids.tolist())
                require(np.array_equal(a['label'],labels[ids]), 'Wrong labels/IDs')
                first,second=probabilities(a['first_trajectory_logits']),probabilities(a['trajectory_logits'])
                require(np.allclose(first,a['first_probabilities'],rtol=0,atol=1e-12) and
                    np.allclose(second,a['second_probabilities'],rtol=0,atol=1e-12), 'Saved PM probabilities differ')
                pred=dict(first_terminal=a['first_trajectory_logits'][:,-1].argmax(-1),first_pm=first.argmax(-1),
                    cycle=cycle_prediction(a['cycle_logits'],len(ids)),second_terminal=a['trajectory_logits'][:,-1].argmax(-1),
                    second_pm=second.argmax(-1),combined_pm=((first+second)*.5).argmax(-1))
                require(np.array_equal(a['prediction'],pred['second_pm']) and
                    np.array_equal(a['terminal_prediction'],pred['second_terminal']), 'Primary/terminal payload differs')
                require(np.array_equal(a['combined_prediction'],pred['combined_pm']), 'Combined weighting differs')
                proofs+=physical(phase,corruption,diag,cfg)
                for name in outcomes:correct[name].extend((pred[name]==a['label']).tolist())
            require(seen==cfg['selected_original_ids'], 'Incomplete, duplicate or reordered original IDs')
            row=dict(corruption=corruption,n=len(seen))
            for name in outcomes:
                outcomes[name].append(correct[name]);row[name]=float(np.mean(correct[name])*100)
            rows.append(row)
        matrices={k:np.array(v,dtype=bool) for k,v in outcomes.items()}
        summary={k:float(v.mean()*100) for k,v in matrices.items()}
        rng=np.random.RandomState(20261101)
        draw=rng.randint(0,len(cfg['selected_original_ids']),(10000,len(cfg['selected_original_ids'])))
        comparisons={}
        for left,right in [('second_pm','first_pm'),('combined_pm','first_pm'),('cycle','first_terminal'),('second_terminal','first_terminal')]:
            delta=matrices[left].astype(float)-matrices[right]
            boot=delta.mean(0)[draw].mean(1)*100
            comparisons[left+'-'+right]=dict(delta_pp=float(delta.mean()*100),conditional95_pp=np.percentile(boot,[2.5,97.5]).tolist(),
                corrected=int((matrices[left]&~matrices[right]).sum()),harmed=int((~matrices[left]&matrices[right]).sum()))
        report=dict(status='passed',accuracy_percent=summary,comparisons=comparisons,per_corruption=rows,
            n_original_objects=len(cfg['selected_original_ids']),n_corruptions=len(rows),seed=cfg['seed'],physical_batches=proofs,
            limitation='single seed/exposed objects; object-cluster CI omits native numerical repeat variation',
            raw_config=identity(paths[0]),executed_source_sha256=cfg['executed_source_sha256'])
        (output/'acceptance.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        np.savez_compressed(output/'outcomes.npz',**matrices)
        return report
    finally:bundle.close()


def self_check():
    logits=np.array([[[5.,0.]*20]*4],dtype=np.float64)
    p=probabilities(logits)
    assert p.shape==(1,40) and np.allclose(p.sum(-1),1)
    assert np.array_equal(cycle_prediction(np.ones((2,40)),2),[0,0])
    for values in (np.full((2,40),np.nan),np.ones((2,15))):
        try:cycle_prediction(values,2)
        except ValueError:pass
        else:raise AssertionError('Accepted invalid cycle logits')
    print('REENCODE_ANALYZER_SELF_CHECK_PASSED')


if __name__=='__main__':
    if sys.argv[1:]==['--self-check']:self_check()
    else:print(json.dumps(analyze(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None),indent=2))
