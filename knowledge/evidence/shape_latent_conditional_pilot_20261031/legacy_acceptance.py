"""Independent raw/cache acceptance for the fixed shape pilot; NumPy1 CPU."""
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np

from scripts.analyze_scan_trajectory import Bundle, policies, require
from scripts.eval_shape_proximal import ARMS, validate_config


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def verify_cache(path, expected, initial, count):
    require(np.__version__.split('.')[0] == '1', 'Physical RNG proof requires NumPy1')
    require(identity(path) == {k: expected[k] for k in ('bytes', 'sha256')}, 'Cache bytes changed')
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata'].item()))
        require(metadata['input_sha256'] == initial['input_points'] and metadata['cuda_devices'] == 1
                and metadata['schema_version'] == 1, 'Cache input/schema changed')
        for key, width in [('shape_latent', 128), ('local_latent', 8192), ('noise', 8192)]:
            v = cache[key]
            digest = hashlib.sha256(str(v.dtype).encode()+str(v.shape).encode()+v.tobytes()).hexdigest()
            require(v.shape == (count, width, 1, 1) and v.dtype == np.float32 and np.isfinite(v).all()
                    and digest == metadata['tensor_hashes'][key] == initial[key], 'Cached tensor differs')
        py, nr = metadata['python_rng'], metadata['numpy_rng']
        state = ((py[0], tuple(py[1]), py[2]), (nr[0], cache['numpy_rng_keys'].copy(), *nr[1:]),
                 cache['torch_rng_cpu'].copy(), [cache['torch_rng_cuda_0'].copy()])
        require(hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest() == metadata['rng_after_sha256'],
                'Serialized paired RNG differs')
    return metadata


def array_hash(value):
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def verify_global(imported, cache_root, cfg, corruption, diagnostic, initial, count):
    inputs=json.loads((imported/'references/manifest.json').read_text());entry=inputs['cases'][corruption]
    require(inputs['original_ids']==cfg['selected_original_ids'] and inputs['lion_state']==cfg['lion_state_before']
            and inputs['classifier_state']==cfg['classifier_state_before_sha256']
            and inputs['dataset_provenance']==cfg['dataset_provenance'], 'Imported dataset/model/ID lineage differs')
    common=imported/entry['common']['path'];legacy=imported/entry['proof']['path']
    require(identity(common)=={k:entry['common'][k] for k in ('bytes','sha256')},'Imported common bytes differ')
    require(identity(legacy)=={k:entry['proof'][k] for k in ('bytes','sha256')},'Imported proof bytes differ')
    with np.load(common,allow_pickle=False) as saved:
        base={k:saved[k].copy() for k in ('shape_latent','local_latent','noise')};meta=json.loads(saved['metadata'].item())
    verify_cache(common,entry['common'],dict(input_points=meta['input_sha256'],**meta['tensor_hashes']),count)
    with np.load(legacy,allow_pickle=False) as saved:
        reference={k:saved[k].copy() for k in saved.files if k!='metadata'};pm=json.loads(saved['metadata'].item())
    require(pm['binding']==entry['proof_binding'] and pm['binding']['common_state']==identity(common),'Imported proof binding differs')
    require(all(array_hash(v)==pm['tensor_hashes'][k] for k,v in reference.items()),'Imported tensor hash differs')
    path=cache_root/'global'/diagnostic['arm']/'modelnet40_c'/corruption/Path(diagnostic['global_proof']['path']).name
    require(identity(path)=={k:diagnostic['global_proof'][k] for k in ('bytes','sha256')},'Global proposal bytes differ')
    with np.load(path,allow_pickle=False) as saved:
        p={k:saved[k].copy() for k in saved.files if k!='metadata'};gm=json.loads(saved['metadata'].item())
    arm=diagnostic['arm'];oracle_entry=entry['oracle'] if arm=='oracle3' else None
    expected_binding=dict(executed_source_sha256=cfg['executed_source_sha256'],input_manifest_sha256=identity(imported/'references/manifest.json')['sha256'],arm=arm,imported_common=entry['common'],imported_proof=entry['proof'],oracle=oracle_entry)
    require(gm['schema_version']==1 and gm['binding']==expected_binding,'Global proposal current-source/target binding differs')
    require(all(v.shape==(count,128,1,1) and v.dtype==np.float32 and np.isfinite(v).all() and array_hash(v)==gm['tensor_hashes'][k] for k,v in p.items()),'Global proposal tensor differs')
    z=p['final_z'];require(array_hash(z)==initial['shape_latent']==diagnostic['selected_z_sha256'],'Selected global z differs')
    if arm=='original':require(np.array_equal(z,base['shape_latent']),'Original z changed')
    elif arm=='mean':require(np.array_equal(z,reference['mean_z'][:,:,None,None]),'Mean target changed')
    else:
        require(len(p)==14 and diagnostic['global_forward_calls']==3 and diagnostic['global_backward_calls']==0,'Wrong global call/state count')
        alpha=np.cumprod(1-np.linspace(.0001,.02,1000,dtype=np.float32))
        # Torch and NumPy cumulative products differ slightly; validate recorded DDIM algebra with tolerance.
        target=reference['mean_z'][:,:,None,None]
        if arm=='oracle3':
            oracle=imported/'oracle_targets'/entry['oracle']['filename'];require(identity(oracle)=={k:entry['oracle'][k] for k in ('bytes','sha256')},'Oracle bytes differ')
            with np.load(oracle,allow_pickle=False) as q:target=q['clean_z'][:,:,None,None]
        start=alpha[20]**.5*base['shape_latent']+(1-alpha[20])**.5*p['global_noise']
        require(np.allclose(start,p['noisy_before_0'],rtol=2e-5,atol=2e-6),'Global forward-noise equation differs')
        for i,t in enumerate((20,10,0)):
            prior,guided=p['prior_clean_%d'%i],p['guided_clean_%d'%i]
            expected=prior if arm=='prior3' else (prior+target)/2
            require(np.allclose(guided,expected,rtol=5e-6,atol=1e-6),'Clean proximal half-target equation differs')
            inferred_noise=(p['noisy_before_%d'%i]-alpha[t]**.5*prior)/(1-alpha[t])**.5
            previous_alpha=alpha[t-10] if t>=10 else 1.
            expected_previous=previous_alpha**.5*guided+(1-previous_alpha)**.5*inferred_noise
            require(np.allclose(p['noisy_after_%d'%i],expected_previous,rtol=3e-5,atol=5e-6),'Guided DDIM transition differs')
            if i<2:require(np.array_equal(p['noisy_after_%d'%i],p['noisy_before_%d'%(i+1)]),'Global chain continuity differs')
        require(np.array_equal(z,p['noisy_after_2']),'Final global state differs')
    for value,expected in [(base['shape_latent'].reshape(count,128),reference['mu']+reference['sigma']*reference['eps_z']),
                           (base['local_latent'].reshape(count,8192),reference['h_mu']+reference['h_sigma']*reference['eps_h']),
                           (reference['mean_z'],reference['new_mu']+reference['sigma']*reference['eps_z'])]:
        require(np.allclose(value,expected,rtol=5e-6,atol=1e-6),'Imported Gaussian/mean proxy equation differs')
    require(array_hash(base['local_latent'])==initial['local_latent'] and array_hash(base['noise'])==initial['noise'],'Imported h0/noise changed')
    anchor=base['local_latent'].reshape(count,2048,4)[:,:,:3]
    require(array_hash(anchor)==diagnostic['chamfer_reference_sha256'],'Original h XYZ anchor differs')
    return p.get('global_noise'),meta['rng_after_sha256']


def verify_rng_style(diagnostic, initial, selected_metadata, imported_rng):
    trace=diagnostic['trajectory']
    require(selected_metadata['rng_after_sha256']==trace['initial_rng_sha256']==imported_rng,
            'Selected/trace RNG differs from imported boundary')
    require(initial['style']==initial['shape_latent'] and trace['final_hashes']['style']==initial['shape_latent']
            and all(p['style_sha256']==initial['shape_latent'] for p in trace['probes']),
            'Clean/final decoder style differs from own initial z')


def validate_finished_scope(config):
    validate_config(config)
    expected_status = ('partial', 'complete')
    batches = (config['evaluation_n_examples']+31)//32
    require(config.get('status') in expected_status and config.get('max_batches') == 0
            and config.get('completed_batches') == {name:batches for name in config['corruptions']},
            'Requested scope is failed or incomplete')


def analyze(phase, cache_root, output=None):
    phase, cache_root = Path(phase), Path(cache_root)
    report_dir = Path(output) if output is not None else phase
    report_dir.mkdir(parents=True,exist_ok=True)
    configs = list((phase/'raw').glob('*/*/*/*/config.json'))
    require(len(configs) == 5, 'Need exactly five complete global arm bundles')
    results, reference, outcomes, labels_all, identities = {}, {}, {}, {}, {}
    global_noises={}
    for path in sorted(configs):
        bundle = Bundle(path.parent)
        try:
            cfg = bundle.json('config.json')
            validate_config(cfg)
            arm = cfg['shape_proximal']['arm']
            require(arm not in results, 'Duplicate arm')
            validate_finished_scope(cfg)
            require(cfg['classifier_state_before_sha256'] == cfg['classifier_state_after_sha256']
                    and cfg['lion_state_before'] == cfg['lion_state_after'], 'Model state changed')
            common = {k: cfg[k] for k in ['asset_manifest_sha256', 'source_snapshot_sha256',
                      'classifier_state_before_sha256', 'lion_state_before', 'executed_source_sha256']}
            if identities: require(common == identities, 'Cross-arm source/assets/states differ')
            else: identities = common
            for name, value in cfg['executed_source_files'].items():
                if name.startswith('repo/'):
                    data = bundle.read('source_snapshot/'+name[5:])
                    require(value == dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()), 'Executed source bytes absent')
            ids, rows, seconds = cfg['selected_original_ids'], [], 0.
            for corruption in cfg['corruptions']:
                correct, labels = [], []
                horizon = 35 if corruption == 'background' else 8
                checkpoints = [9, 18, 27] if horizon == 35 else [2, 4, 6]
                for batch, start in enumerate(range(0, len(ids), 32)):
                    prefix = 'batches/%s/batch%04d/' % (corruption, batch)
                    with np.load(io.BytesIO(bundle.read(prefix+'predictions.npz')), allow_pickle=False) as data:
                        require(data['index'].tolist() == ids[start:start+32], 'ID/order mismatch')
                        logits, label = data['trajectory_logits'], data['label'].copy()
                        require(logits.shape == (len(label), 4, 40) and np.isfinite(logits).all(), 'Invalid logits')
                        require(data['candidate_names'].tolist() == ['clean_%02d'%s for s in checkpoints]+['terminal']
                                and data['candidate_loop_counts'].tolist() == checkpoints+[horizon], 'Wrong candidates')
                        computed = policies(logits, 3)
                        require(np.array_equal(computed['probability_mean'], data['probability_mean']), 'PM recomputation differs')
                        correct.extend((computed['probability_mean'] == label).tolist())
                        labels.extend(label.tolist())
                    diagnostic = bundle.json(prefix+'diagnostics.json')
                    trace = diagnostic['trajectory']
                    require(trace['complete'] and trace['probe_rng_unchanged'] and trace['probe_buffers_unchanged'], 'Impure probe')
                    require(diagnostic['scheduler']['active_timesteps'] == list(range((horizon-1)*10, -1, -10))
                            and diagnostic['scheduler']['final_alpha_cumprod'] == 1., 'Wrong DDIM schedule')
                    initial = trace['initial_hashes']
                    oracle = diagnostic['guidance_metrics']['shape_proximal']
                    require(oracle['arm'] == arm and oracle['global_forward_calls'] == (3 if arm.endswith('3') else 0) and oracle['clean_information_used']==(arm=='oracle3')
                            and oracle['input_sha256'] == initial['input_points'] and oracle['executed_source_sha256']==cfg['executed_source_sha256'], 'Wrong oracle arm/IDs/prior policy')
                    paired = dict(input_points=initial['input_points'], noise=initial['noise'],
                                  actual_rng_before=diagnostic['reconstruction_rng_before_sha256'],actual_rng_after=diagnostic['reconstruction_rng_after_sha256'],
                                  common_state=oracle['imported_common'], posterior_proof=oracle['imported_proof'],
                                  chamfer_reference_sha256=oracle['chamfer_reference_sha256'])
                    key = (corruption, batch)
                    if key in reference: require(paired == reference[key], 'Cross-arm common proof/anchor/noise differs')
                    else: reference[key] = paired
                    selected = cache_root/'selected'/arm/'modelnet40_c'/corruption/Path(diagnostic['paired_state']['path']).name
                    selected_meta=verify_cache(selected, diagnostic['paired_state'], initial, len(label))
                    actual_noise,imported_rng=verify_global(phase/'imported',cache_root,cfg,corruption,oracle,initial,len(label))
                    verify_rng_style(diagnostic,initial,selected_meta,imported_rng)
                    if actual_noise is not None:
                        if key in global_noises:require(np.array_equal(global_noises[key],actual_noise),'Global private Gaussian differs across three chains')
                        else:global_noises[key]=actual_noise.copy()
                    seconds += diagnostic['shared_reconstruction_seconds']+diagnostic['head_seconds']
                require(len(correct) == len(ids), 'Incomplete corruption')
                labels_all.setdefault(corruption, labels)
                require(labels_all[corruption] == labels, 'Cross-arm labels differ')
                rows.append(correct)
            outcomes[arm] = np.asarray(rows, dtype=bool)
            results[arm] = dict(macro_percent=100*float(outcomes[arm].mean()),
                               per_corruption_percent=dict(zip(cfg['corruptions'],100*outcomes[arm].mean(1))),
                               reconstruction_and_terminal_seconds=seconds, run_id=cfg['run_id'])
        finally:
            bundle.close()
    require(set(results) == set(ARMS), 'Missing registered arm')
    comparisons = {}
    for left,right in [('mean','original'),('prior3','original'),('proxy3','original'),('proxy3','prior3'),('oracle3','original'),('oracle3','proxy3')]:
        delta = outcomes[left].astype(float)-outcomes[right]
        comparisons[left+'-'+right] = dict(delta_pp=100*float(delta.mean()),
                                           corrected=int((delta>0).sum()), harmed=int((delta<0).sum()))
    result = dict(status='passed', arms=results, paired_comparisons=comparisons,
                  identity=identities, physical_cache_files=len(reference),
                  limits='32exposed objects/fourfixed corruptions/seed0; oracle3 diagnostic only; imported shared states; replay excludes firstpass/proxy preparation; no unseen confirmation.')
    (report_dir/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(report_dir/'outcomes.npz', **outcomes)
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(*sys.argv[1:]), indent=2))
