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
from scripts.eval_shape_latent import ARMS, validate_config


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
    require(len(configs) == 5, 'Need exactly five complete arm bundles')
    results, reference, outcomes, labels_all, identities = {}, {}, {}, {}, {}
    for path in sorted(configs):
        bundle = Bundle(path.parent)
        try:
            cfg = bundle.json('config.json')
            validate_config(cfg)
            arm = cfg['shape_latent']['arm']
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
                    paired = {k: initial[k] for k in ['input_points','shape_latent','local_latent','noise','noisy_local']}
                    paired['cache'] = {k: diagnostic['paired_state'][k] for k in ['bytes','sha256']}
                    key = (corruption, batch)
                    if key in reference: require(paired == reference[key], 'Cross-arm initial state/RNG cache mismatch')
                    else: reference[key] = paired
                    cache = cache_root/'modelnet40_c'/corruption/Path(diagnostic['paired_state']['path']).name
                    verify_cache(cache, diagnostic['paired_state'], initial, len(label))
                    shape = diagnostic['guidance_metrics']['shape_latent']
                    require(shape['arm'] == arm and shape['global_forward_calls'] == cfg['shape_latent']['global_forward_calls']
                            and shape['global_backward_calls'] == (3 if arm == 'guided3' else 0), 'Global call mismatch')
                    if arm != 'baseline':
                        require(shape['global_rng_unchanged'] and shape['local_style_projections'] == horizon, 'Global RNG/local update mismatch')
                    if arm in ('anchor','proposal','guided3'):
                        require(np.all(np.asarray(shape['final_displacement_norms']) <= .05*np.asarray(shape['anchor_norms'])+1e-5), 'Anchor bound exceeded')
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
    for left,right in [('updated','baseline'),('anchor','updated'),('proposal','anchor'),('guided3','anchor'),('guided3','proposal')]:
        delta = outcomes[left].astype(float)-outcomes[right]
        comparisons[left+'-'+right] = dict(delta_pp=100*float(delta.mean()),
                                           corrected=int((delta>0).sum()), harmed=int((delta<0).sum()))
    result = dict(status='passed', arms=results, paired_comparisons=comparisons,
                  identity=identities, physical_cache_files=len(reference),
                  limits='Fixed exposed IDs/seed0 exploratory pilot. Guided3 vs proposal changes both chain length and guidance.')
    (report_dir/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(report_dir/'outcomes.npz', **outcomes)
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None), indent=2))
