"""Independent NumPy1 validation of raw outcomes and physical latent transitions."""
import io
import json
from pathlib import Path
import sys

import numpy as np
from legacy_acceptance import Bundle, policies, require, identity, array_hash, verify_cache, verify_rng_style
from scripts.eval_shape_conditional import ARMS, validate


def finished(config):
    validate(config)
    require(config.get('status') in ('partial','complete') and
            config.get('completed_batches')=={c:1 for c in config['corruptions']}, 'Incomplete requested scope')


def proof(path, expected=None):
    if expected is not None:
        require(identity(path)=={k:expected[k] for k in ('bytes','sha256')},'Proof byte identity changed')
    with np.load(path,allow_pickle=False) as saved:
        meta=json.loads(saved['metadata'].item())
        values={k:saved[k].copy() for k in saved.files if k!='metadata'}
    require(meta['schema_version']==1 and set(values)==set(meta['tensor_hashes']), 'Proof schema/keys')
    require(all(v.dtype==np.float32 and np.isfinite(v).all() and array_hash(v)==meta['tensor_hashes'][k]
                for k,v in values.items()),'Proof tensor identity')
    return values,meta


def analyze(phase, cache_root, output=None):
    phase,cache_root=Path(phase),Path(cache_root);out=Path(output) if output else phase;out.mkdir(exist_ok=True,parents=True)
    paths=list((phase/'raw').glob('*/*/*/*/config.json'));require(len(paths)==7,'Seven arms required')
    results={};outcomes={};labels_seen={};paired={};identities=None;noise_refs={};latents={}
    inputs=json.loads((phase/'imported/references/manifest.json').read_text())
    alpha=np.cumprod(1-np.linspace(.0001,.02,1000,dtype=np.float32))
    for path in sorted(paths):
        bundle=Bundle(path.parent)
        try:
            cfg=bundle.json('config.json');finished(cfg);arm=cfg['conditional']['arm'];require(arm not in results,'Duplicate arm')
            require(cfg['lion_state_before']==cfg['lion_state_after'] and
                    cfg['classifier_state_before_sha256']==cfg['classifier_state_after_sha256'],'Frozen state changed')
            current={k:cfg[k] for k in ['asset_manifest_sha256','source_snapshot_sha256','executed_source_sha256',
                                       'lion_state_before','classifier_state_before_sha256']}
            if identities is None:identities=current
            require(current==identities,'Cross-arm model/source identity')
            for name,expected in cfg['executed_source_files'].items():
                if name.startswith('repo/'):
                    import hashlib
                    data=bundle.read('source_snapshot/'+name[5:])
                    require(expected==dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),'Executed source absent')
            rows=[];diagnostics=[]
            for c in cfg['corruptions']:
                prefix='batches/'+c+'/batch0000/'
                with np.load(io.BytesIO(bundle.read(prefix+'predictions.npz')),allow_pickle=False) as data:
                    require(data['index'].tolist()==cfg['selected_original_ids'],'Wrong original IDs')
                    logits=data['trajectory_logits'];label=data['label'].copy()
                    require(logits.shape==(32,4,40) and np.isfinite(logits).all(),'Wrong logits')
                    counts=[9,18,27,35] if c=='background' else [2,4,6,8]
                    require(data['candidate_loop_counts'].tolist()==counts,'Wrong PM checkpoints')
                    pred=policies(logits,3)['probability_mean'];require(np.array_equal(pred,data['probability_mean']),'PM recount')
                    rows.append(pred==label)
                    labels_seen.setdefault(c,label);require(np.array_equal(label,labels_seen[c]),'Unpaired labels')
                d=bundle.json(prefix+'diagnostics.json');trace=d['trajectory'];initial=trace['initial_hashes']
                require(trace['complete'] and trace['probe_rng_unchanged'] and trace['probe_buffers_unchanged'],'Impure probe')
                metric=d['guidance_metrics']['conditional'];require(metric['arm']==arm,'Wrong arm')
                selected=cache_root/'selected'/arm/'modelnet40_c'/c/Path(d['paired_state']['path']).name
                selected_meta=verify_cache(selected,d['paired_state'],initial,32)
                entry=inputs['cases'][c];base_path=phase/'imported'/entry['common']['path']
                with np.load(base_path,allow_pickle=False) as saved:
                    base={k:saved[k].copy() for k in ('shape_latent','local_latent','noise')};bm=json.loads(saved['metadata'].item())
                verify_cache(base_path,entry['common'],dict(input_points=bm['input_sha256'],**bm['tensor_hashes']),32)
                verify_rng_style(d,initial,selected_meta,bm['rng_after_sha256'])
                require(initial['local_latent']==array_hash(base['local_latent']) and initial['noise']==array_hash(base['noise']),'Changed local input/noise')
                common=(initial['input_points'],initial['local_latent'],initial['noise'],trace['initial_rng_sha256'],
                        d['reconstruction_rng_before_sha256'],d['reconstruction_rng_after_sha256'])
                paired.setdefault(c,common);require(paired[c]==common,'Cross-arm RNG/input differs')
                ref_path=cache_root/'firstpass'/c/Path(metric['firstpass']['path']).name
                ref,rm=proof(ref_path,metric['firstpass'])
                require(rm['binding']==dict(executed_source_sha256=cfg['executed_source_sha256'],common=entry['common'],input_sha256=initial['input_points']),'Firstpass binding')
                require(np.array_equal(ref['encoder_z'],base['shape_latent']),'Encoder start mismatch')
                if arm=='firstpass':
                    require(array_hash(ref['h_star'])==trace['final_hashes']['latent'],'Firstpass h* mismatch')
                    z=ref['encoder_z']
                else:
                    gp=cache_root/'global'/arm/c/Path(metric['global_proof']['path']).name;p,pm=proof(gp,metric['global_proof'])
                    require(pm['binding']==dict(rm['binding'],arm=arm,firstpass=identity(ref_path)),'Global source/ref binding')
                    start=ref['encoder_z'] if arm.startswith('encoder_') else ref['scd_z']
                    require(np.array_equal(p['start_z'],start) and np.array_equal(p['h_star'],ref['h_star']),'Wrong initial z/h*')
                    for key in ['train_noise','heldout_noise']:
                        noise_refs.setdefault((c,key),p[key]);require(np.array_equal(noise_refs[c,key],p[key]),'Unpaired private energy noise')
                    if arm.endswith('none'):require(np.array_equal(p['final_z'],start),'No-prior z changed')
                    else:
                        noise_refs.setdefault((c,'global_noise'),p['global_noise']);require(np.array_equal(noise_refs[c,'global_noise'],p['global_noise']),'Global noise differs')
                        require(np.allclose(p['noisy_before_0'],alpha[20]**.5*start+(1-alpha[20])**.5*p['global_noise'],rtol=2e-5,atol=2e-6),'Forward noise equation')
                        for i,t in enumerate((20,10,0)):
                            update=p['guidance_update_%d'%i];before=p['noisy_before_%d'%i];clean=p['prior_clean_%d'%i]
                            eps=(before-alpha[t]**.5*clean)/(1-alpha[t])**.5;pa=alpha[t-10] if t>=10 else 1.
                            require(np.allclose(p['ddim_previous_%d'%i],pa**.5*clean+(1-pa)**.5*eps,rtol=3e-5,atol=5e-6),'DDIM equation')
                            require(np.array_equal(p['noisy_after_%d'%i],p['ddim_previous_%d'%i]-update),'Gradient correction equation')
                            if arm.endswith('guided3'):
                                grad=p['grad_noisy_%d'%i];norm=np.linalg.norm(grad.reshape(32,-1),axis=1).reshape(32,1,1,1)
                                scale=.02*np.linalg.norm(base['shape_latent'].reshape(32,-1),axis=1).reshape(32,1,1,1)
                                require(np.allclose(update,scale*grad/np.maximum(norm,1e-12),rtol=2e-5,atol=1e-7),'Gradient normalization equation')
                            else:require(np.count_nonzero(update)==0,'Prior control guided')
                            if i<2:require(np.array_equal(p['noisy_after_%d'%i],p['noisy_before_%d'%(i+1)]),'Chain discontinuity')
                        require(np.array_equal(p['final_z'],p['noisy_after_2']),'Final global state')
                    z=p['final_z'];latents[arm+'__'+c+'__start']=start;latents[arm+'__'+c+'__final']=z
                    diagnostics.append(dict(corruption=c,heldout_start=float(p['start_energy'].mean()),heldout_final=float(p['final_energy'].mean())))
                require(array_hash(z)==initial['shape_latent']==metric['selected_z_sha256'],'Selected decode z mismatch')
                latents['encoder__'+c]=ref['encoder_z'];latents['scd__'+c]=ref['scd_z']
                oracle=phase/'imported/oracle_targets'/entry['oracle']['filename']
                require(identity(oracle)=={k:entry['oracle'][k] for k in ('bytes','sha256')},'Posthoc clean target identity')
                with np.load(oracle,allow_pickle=False) as saved:clean_z=saved['clean_z'][:,:,None,None]
                latents['clean__'+c]=clean_z
            outcome=np.asarray(rows,dtype=bool);outcomes[arm]=outcome
            results[arm]=dict(correct=int(outcome.sum()),total=int(outcome.size),accuracy_percent=100*float(outcome.mean()),
                              per_corruption_percent=dict(zip(cfg['corruptions'],(100*outcome.mean(1)).tolist())),heldout_energy=diagnostics)
        finally:bundle.close()
    require(set(results)==set(ARMS),'Missing arm')
    contrasts={};rng=np.random.RandomState(81031);draws=rng.randint(0,32,size=(10000,32))
    for left,right in [('encoder_guided3','encoder_none'),('scd_guided3','scd_none'),
                       ('encoder_guided3','encoder_prior3'),('scd_guided3','scd_prior3'),
                       ('scd_guided3','encoder_guided3'),('scd_none','encoder_none'),('encoder_none','firstpass')]:
        delta=outcomes[left].astype(float)-outcomes[right];samples=delta[:,draws].mean(axis=(0,2))*100
        contrasts[left+'-'+right]=dict(delta_pp=100*float(delta.mean()),corrected=int((delta>0).sum()),harmed=int((delta<0).sum()),
                                       conditional_object_cluster95=np.percentile(samples,[2.5,97.5]).tolist())
    report=dict(status='passed',arms=results,contrasts=contrasts,identity=identities,corruptions=cfg['corruptions'],
                limits='Exposed32objects/seed0; conditional bootstrap excludes seed/native repeat variance; replay cost excludes shared firstpass; uncalibrated surrogate.')
    (out/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out/'outcomes.npz',**outcomes);np.savez_compressed(out/'paired_latents.npz',**latents)
    return report


if __name__=='__main__':print(json.dumps(analyze(*sys.argv[1:]),indent=2))
