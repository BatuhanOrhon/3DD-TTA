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
from scripts.eval_shape_refresh import ARMS, validate_config


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


def array_hash(value):
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def verify_refresh(common, proof, diagnostic, initial, count, manifest_sha, executed_source_sha):
    require(identity(common) == {k:diagnostic['common_state'][k] for k in ('bytes','sha256')}, 'Common state bytes changed')
    require(identity(proof) == {k:diagnostic['posterior_proof'][k] for k in ('bytes','sha256')}, 'Refresh proof bytes changed')
    with np.load(common,allow_pickle=False) as saved:
        base={k:saved[k].copy() for k in ('shape_latent','local_latent','noise')}
        meta=json.loads(saved['metadata'].item())
    verify_cache(common,diagnostic['common_state'],dict(input_points=initial['input_points'],**meta['tensor_hashes']),count)
    with np.load(proof,allow_pickle=False) as saved:
        p={k:saved[k].copy() for k in saved.files if k != 'metadata'};pm=json.loads(saved['metadata'].item())
    require(pm['schema_version']==1 and pm['binding']==dict(input_sha256=initial['input_points'],common_state=identity(common),asset_manifest_sha256=manifest_sha,executed_source_sha256=executed_source_sha), 'Invalid refresh binding')
    expected={'mu','sigma','eps_z','eps_h','h_mu','h_sigma','prepared_input','decoded_points','new_mu','new_sigma','mean_z','posterior_z','ae_points','ae_mu','ae_sigma'}
    require(set(p)==expected==set(pm['tensor_hashes']), 'Wrong physical refresh proof keys')
    for k,v in p.items():
        shape=(count,2048,3) if k in ('prepared_input','decoded_points','ae_points') else (count,8192 if k in ('eps_h','h_mu','h_sigma') else 128)
        require(v.shape==shape and v.dtype==np.float32 and np.isfinite(v).all() and array_hash(v)==pm['tensor_hashes'][k], 'Refresh tensor/hash/shape differs')
        if 'sigma' in k:require((v>0).all(), 'Nonpositive posterior sigma')
    require(array_hash(p['prepared_input'])==initial['input_points'], 'Prepared input differs')
    for actual,expected in [(base['shape_latent'].reshape(count,128),p['mu']+p['sigma']*p['eps_z']),
                            (base['local_latent'].reshape(count,8192),p['h_mu']+p['h_sigma']*p['eps_h']),
                            (p['mean_z'],p['new_mu']+p['sigma']*p['eps_z']),
                            (p['posterior_z'],p['new_mu']+p['new_sigma']*p['eps_z'])]:
        require(np.allclose(actual,expected,rtol=5e-6,atol=1e-6), 'Common Gaussian refresh equation failed')
    arm=diagnostic['arm'];z=base['shape_latent'] if arm=='baseline' else p['mean_z' if arm=='mean_refresh' else 'posterior_z'][:,:,None,None]
    require(array_hash(z)==initial['shape_latent']==diagnostic['proposal_sha256'], 'Selected refresh z differs')
    require(array_hash(base['local_latent'])==initial['local_latent'] and array_hash(base['noise'])==initial['noise'], 'Initial h/noise changed')
    anchor=base['local_latent'].reshape(count,2048,4)[:,:,:3]
    require(array_hash(anchor)==diagnostic['chamfer_reference_sha256'], 'Original h XYZ anchor changed')


def validate_finished_scope(config):
    validate_config(config)
    expected_status = ('complete', 'partial') if config['technical_smoke'] else ('complete',)
    batches = (config['evaluation_n_examples']+31)//32
    require(config.get('status') in expected_status and config.get('max_batches') == 0
            and config.get('completed_batches') == {name:batches for name in config['corruptions']},
            'Requested scope is failed or incomplete')


def analyze(phase, cache_root, output=None):
    phase, cache_root = Path(phase), Path(cache_root)
    report_dir = Path(output) if output is not None else phase
    report_dir.mkdir(parents=True,exist_ok=True)
    configs = list((phase/'raw').glob('*/*/*/*/config.json'))
    require(len(configs) == 3, 'Need exactly three complete refresh arm bundles')
    results, reference, outcomes, labels_all, identities = {}, {}, {}, {}, {}
    for path in sorted(configs):
        bundle = Bundle(path.parent)
        try:
            cfg = bundle.json('config.json')
            validate_config(cfg)
            arm = cfg['shape_refresh']['arm']
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
                    oracle = diagnostic['guidance_metrics']['shape_refresh']
                    require(oracle['arm'] == arm and oracle['global_prior_calls'] == 0
                            and oracle['input_sha256'] == initial['input_points'] and oracle['executed_source_sha256']==cfg['executed_source_sha256'], 'Wrong oracle arm/IDs/prior policy')
                    paired = dict(input_points=initial['input_points'], noise=initial['noise'],
                                  common_state=oracle['common_state'], posterior_proof=oracle['posterior_proof'],
                                  chamfer_reference_sha256=oracle['chamfer_reference_sha256'])
                    key = (corruption, batch)
                    if key in reference: require(paired == reference[key], 'Cross-arm common proof/anchor/noise differs')
                    else: reference[key] = paired
                    selected = cache_root/'selected'/arm/'modelnet40_c'/corruption/Path(diagnostic['paired_state']['path']).name
                    verify_cache(selected, diagnostic['paired_state'], initial, len(label))
                    common = cache_root/'common'/'modelnet40_c'/corruption/Path(oracle['common_state']['path']).name
                    proof = common.with_suffix('.refresh.npz')
                    verify_refresh(common, proof, oracle, initial, len(label), cfg['asset_manifest_sha256'], cfg['executed_source_sha256'])
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
    baseline_cfg=next(p for p in configs if json.loads(p.read_text())['shape_refresh']['arm']=='baseline')
    completed=json.loads((cache_root/'common'/'baseline_complete.json').read_text())
    require(completed['config_identity']==identity(baseline_cfg), 'Baseline completion receipt/config differs')
    require(set(results) == set(ARMS), 'Missing registered arm')
    comparisons = {}
    for left,right in [('mean_refresh','baseline'),('posterior_refresh','baseline'),('posterior_refresh','mean_refresh')]:
        delta = outcomes[left].astype(float)-outcomes[right]
        comparisons[left+'-'+right] = dict(delta_pp=100*float(delta.mean()),
                                           corrected=int((delta>0).sum()), harmed=int((delta<0).sum()))
    result = dict(status='passed', arms=results, paired_comparisons=comparisons,
                  identity=identities, physical_cache_files=len(reference),
                  limits='64 exposed IDs/15fixed corruptions/seed0; no unseen confirmation. Replay timing excludes baseline first pass; no clean input used in proposal.')
    (report_dir/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(report_dir/'outcomes.npz', **outcomes)
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(*sys.argv[1:]), indent=2))
