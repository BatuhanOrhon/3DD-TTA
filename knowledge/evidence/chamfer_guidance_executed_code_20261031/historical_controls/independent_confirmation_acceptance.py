"""Local-only validation of physical confirmation evidence; partial mode has no accuracy output."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import pickletools
import struct
import sys
import zipfile

import numpy as np

PREP = Path(__file__).resolve().parent
sys.path.insert(0, str(PREP.parents[2]))
from scripts.analyze_chamfer_variants import _load, _paired_identity, analyze_campaign
from scripts.analyze_scan_trajectory import Bundle
from research_artifacts import CORRUPTIONS

DATASETS = ('modelnet40_c', 'scanobjectnn_c')
ARMS = ('baseline', 'sparse', 'semantic_kl', 'semantic_h', 'weighted')
HEADS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def array_hash(value):
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def postprepare_rng_hash(archive, metadata):
    py = metadata['python_rng']
    state = ((py[0], tuple(py[1]), py[2]),
             (metadata['numpy_rng'][0], archive['numpy_rng_keys'].copy(), *metadata['numpy_rng'][1:]),
             archive['torch_rng_cpu'].copy(),
             [archive['torch_rng_cuda_%d' % i].copy() for i in range(metadata['cuda_devices'])])
    blob = pickle.dumps(state, protocol=4)
    if hashlib.sha256(blob).hexdigest() != metadata['rng_after_sha256']:
        frames = [(arg, pos) for op, arg, pos in pickletools.genops(blob) if op.name == 'FRAME']
        require(len(frames) == 1 and frames[0][1] == 2 and len(blob) == 11 + frames[0][0], 'Unexpected RNG pickle framing')
        # NumPy2 renamed this pickle global; reproduce the verified NumPy1 producer bytes.
        payload = blob[11:].replace(b'\x8c\x16numpy._core.multiarray', b'\x8c\x15numpy.core.multiarray')
        blob = blob[:3] + struct.pack('<Q', len(payload)) + payload
    return hashlib.sha256(blob).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def accept(manifest_path, output, partial=False, cache_manifest_path=None):
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest['records']
    raw = [r for r in records if '/results/confirmation/' in r['path'] and r['path'].endswith('.zip')]
    cache_records = json.loads(Path(cache_manifest_path).read_text())['records'] if cache_manifest_path else records
    physical = [r for r in cache_records if '/paired_states/' in r['path'] and r['path'].endswith('.npz')]
    require(raw and physical, 'Confirmation raw and physical caches are required')
    require(len({r['path'] for r in raw}) == len(raw) and len({r['path'] for r in physical}) == len(physical), 'Duplicate evidence path')
    if not partial:
        require(len(raw) == 120 and len(physical) == 214, 'Full confirmation requires120raw/214physical caches')
    calibration_bytes = (PREP / 'calibration.json').read_bytes()
    calibration = json.loads(calibration_bytes)
    caches, cache_proofs = {}, []
    for record in physical:
        require(identity(record['local_path']) == {k: record[k] for k in ('bytes', 'sha256')}, 'Cache SHA/size mismatch')
        path = Path(record['path'])
        stage = path.parts[path.parts.index('paired_states') + 1]
        dataset, corruption = path.parts[-3:-1]
        seed = int(path.parts[path.parts.index('paired_states') + 2].removeprefix('seed'))
        require(stage in ('smoke', 'development', 'confirmation') and dataset in DATASETS, 'Foreign cache scope')
        ids_stage = {'smoke': 'smoke', 'development': 'development', 'confirmation': 'heldout'}[stage]
        ids = json.loads((PREP / (dataset + '_' + ids_stage + '_ids.json')).read_text())['indices']
        chunks = [ids[start:start + 70] for start in range(0, len(ids), 70)]
        matching = [chunk for chunk in chunks if hashlib.sha256(np.asarray(chunk, dtype=np.int64).tobytes()).hexdigest() == path.stem]
        require(len(matching) == 1, 'Cache filename differs from exact batch IDs')
        with np.load(record['local_path'], allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata'].item()))
            count = len(matching[0])
            require(metadata['schema_version'] == 1 and metadata['cuda_devices'] == 1, 'Cache schema/CUDA device mismatch')
            require(archive['local_latent'].shape == archive['noise'].shape == (count, 8192, 1, 1), 'Cache local/noise shape mismatch')
            require(archive['shape_latent'].shape[0] == count and archive['shape_latent'].shape[2:] == (1, 1), 'Cache shape latent mismatch')
            for key in ('shape_latent', 'local_latent', 'noise'):
                require(archive[key].dtype == np.float32 and np.isfinite(archive[key]).all()
                        and array_hash(archive[key]) == metadata['tensor_hashes'][key], 'Cache tensor hash/finite check failed')
            require(archive['numpy_rng_keys'].dtype == np.uint32 and archive['torch_rng_cpu'].dtype == archive['torch_rng_cuda_0'].dtype == np.uint8, 'Cache RNG schema mismatch')
            require(postprepare_rng_hash(archive, metadata) == metadata['rng_after_sha256'], 'Postprepare RNG SHA mismatch')
        batch = chunks.index(matching[0])
        caches[record['path']] = dict(metadata=metadata, identity={k: record[k] for k in ('bytes', 'sha256')}, ids=matching[0],
                                     stage=stage, dataset=dataset, corruption=corruption, seed=seed, batch=batch)
        cache_proofs.append(dict(path=record['path'], sha256=record['sha256'], stage=stage, dataset=dataset, corruption=corruption, seed=seed, batch=batch))
    if not partial:
        expected_caches = {(stage, dataset, corruption, seed, batch)
            for stage in ('smoke', 'development', 'confirmation') for dataset in DATASETS
            for corruption in (('gaussian', 'background') if stage == 'smoke' else CORRUPTIONS)
            for seed in ((0, 1, 2) if stage == 'confirmation' else (0,))
            for batch in ((0, 1) if stage == 'confirmation' else (0,))}
        require({tuple(row[k] for k in ('stage', 'dataset', 'corruption', 'seed', 'batch')) for row in cache_proofs} == expected_caches,
                'Physical cache stage/dataset/corruption/seed/batch coverage differs')
    print('PHYSICAL_CACHES_VERIFIED', len(caches), flush=True)
    runs, lookup, raw_proofs = [], {}, []
    for number, record in enumerate(raw):
        require(identity(record['local_path']) == {k: record[k] for k in ('bytes', 'sha256')}, 'Raw SHA/size mismatch')
        with zipfile.ZipFile(record['local_path']) as archive:
            require(archive.testzip() is None, 'Raw ZIP CRC mismatch')
        run = _load(record['local_path'])
        config, arm = run['config'], run['arm']
        key = (config['dataset'], arm, config['backbone'], config['seed'])
        require(key not in lookup, 'Duplicate confirmation dataset/arm/head/seed')
        expected_ids = json.loads((PREP / (config['dataset'] + '_heldout_ids.json')).read_text())['indices']
        require(config['selected_original_ids'] == expected_ids and len(expected_ids) == 140
                and config['selection_manifest']['stage'] == 'heldout' and config['batch_size'] == 70
                and len(config['corruptions']) == 15 and config['max_batches'] == 0, 'Confirmation ID/protocol coverage differs')
        if arm == 'sparse':
            require(config['chamfer_experiment']['interval'] == 3, 'Sparse interval changed')
        if arm == 'weighted':
            require(config['chamfer_experiment']['density_k'] == 10, 'Weighted density neighborhood changed')
        bundle = Bundle(record['local_path'])
        try:
            if arm in ('semantic_kl', 'semantic_h', 'weighted'):
                require(bundle.read('calibration_report.json') == calibration_bytes, 'Frozen calibration bytes differ')
            for corruption in config['corruptions']:
                for batch in range(2):
                    diagnostic = bundle.json('batches/%s/batch%04d/diagnostics.json' % (corruption, batch))
                    metrics, trace = diagnostic['guidance_metrics'], diagnostic['trajectory']
                    horizon = 30 if corruption == 'background' else 10
                    backwards = (11 if horizon == 30 else 4) if arm == 'sparse' else horizon
                    require(all(math.isfinite(v) for v in metrics.values()) and metrics['graph_calls'] == metrics['spectral_loss_calls'] == 0
                            and metrics['guidance_backward_calls'] == backwards, 'Forbidden work/nonfinite guidance/wrong backward count')
                    if arm != 'baseline':
                        require(metrics['prior_forward_calls'] == horizon, 'Wrong actual prior forward count')
                    require(diagnostic['geometry_shape'] == [70, 2048, 3] and len(diagnostic['geometry_sha256']) == 64, 'Geometry record mismatch')
                    if arm.startswith('semantic_'):
                        semantic = trace['semantic_guidance']
                        require(metrics['semantic_callback_calls'] == horizon and metrics['semantic_loss_calls'] == 2
                                and semantic['reference']['forward_calls'] == 1 and not semantic['reference_votes'], 'Semantic operation count/vote mismatch')
                        require(trace['operation_counts'] == dict(probe_decode_forward=3, probe_classifier_forward=12, semantic_reference_forward=1), 'Semantic shared probe count mismatch')
                        require(config['semantic_guidance']['weight'] == calibration['coefficients'][arm], 'Semantic fixed weight mismatch')
                        require([row['loop_count'] for row in semantic['records']] == ([15, 23] if horizon == 30 else [5, 8])
                                and all(row['baseline_logits_reused'] and row['decoded_geometry_reused'] for row in semantic['records']), 'Semantic checkpoint/reuse mismatch')
                    elif arm == 'weighted':
                        require(metrics['reference_density_calls'] == 1 and metrics['semantic_loss_calls'] == metrics['semantic_callback_calls'] == 0, 'Weighted operation count mismatch')
                        require(config['guidance_protocol']['eta'] == config['guidance_protocol']['gamma'] == .01 * calibration['coefficients']['weighted_scale'], 'Weighted fixed rates mismatch')
                    cache = caches[diagnostic['paired_state']['path']]
                    require((cache['stage'], cache['dataset'], cache['corruption'], cache['seed'], cache['batch'])
                            == ('confirmation', config['dataset'], corruption, config['seed'], batch), 'Actual cache scope differs')
                    require(cache['identity'] == {k: diagnostic['paired_state'][k] for k in ('bytes', 'sha256')}, 'Actual cache identity differs')
                    require(cache['ids'] == expected_ids[batch * 70:(batch + 1) * 70], 'Actual cache IDs differ')
                    metadata = cache['metadata']
                    require(metadata['input_sha256'] == trace['initial_hashes']['input_points'] and metadata['rng_after_sha256'] == trace['initial_rng_sha256']
                            and all(metadata['tensor_hashes'][name] == trace['initial_hashes'][name] for name in ('shape_latent', 'local_latent', 'noise')), 'Actual cache input/h0/noise/RNG mismatch')
                    require(diagnostic['preparation_mode'] == ('fresh_encode' if arm == 'baseline' else 'cached_replay'), 'Unexpected preparation mode')
            lookup[key] = run
            runs.append(run)
            raw_proofs.append(dict(dataset=key[0], arm=key[1], head=key[2], seed=key[3], sha256=record['sha256']))
        finally:
            bundle.close()
        if (number + 1) % 20 == 0:
            print('RAW_HEADS_VERIFIED', number + 1, flush=True)
    if not partial:
        require(set(lookup) == {(dataset, arm, head, seed) for dataset in DATASETS for arm in ARMS for head in HEADS for seed in (0, 1, 2)}, 'Full confirmation condition coverage differs')
    for key, run in lookup.items():
        baseline = lookup[(key[0], 'baseline', key[2], key[3])]
        if key[1] != 'baseline':
            _paired_identity(baseline, run)
            for corruption in run['cells']:
                require(np.array_equal(run['cells'][corruption]['labels'], baseline['cells'][corruption]['labels']), 'Paired labels differ')
    proof = dict(status='independent_partial_confirmation_structurally_accepted' if partial else 'independent_full_confirmation_accepted',
                 raw_heads=len(raw), raw_batches=len(raw) * 30, caches=len(caches), raw_proofs=raw_proofs, physical_cache_proofs=cache_proofs,
                 manifest_identity=identity(manifest_path), calibration_identity=identity(PREP / 'calibration.json'),
                 cache_manifest_identity=identity(cache_manifest_path or manifest_path),
                 rng_method='Exact protocol4 SHA; normalize only NumPy2 _core pickle global to producing NumPy1 core spelling and repair the single frame length',
                 limitations='Finite geometry follows the verified producing-runtime guard; raw coordinates are not stored. Native execution is not bitwise deterministic. Cached costs do not prove deployment speed.')
    if not partial:
        report = analyze_campaign([row['local_path'] for row in raw], bootstrap_replicates=2000, bootstrap_seed=0)
        analysis_path = Path(output).with_name(Path(output).stem + '_analysis.json')
        write_new(analysis_path, report)
        proof['analysis'] = dict(path=str(analysis_path), **identity(analysis_path))
        proof['per_head'] = {}
        for head in HEADS:
            paths = [run['input'] for run in runs if run['config']['backbone'] == head]
            head_report = analyze_campaign(paths, bootstrap_replicates=2000, bootstrap_seed=0)
            head_path = Path(output).with_name(Path(output).stem + '_' + head + '.json')
            write_new(head_path, head_report)
            proof['per_head'][head] = dict(path=str(head_path), **identity(head_path), comparisons=[
                {key: row[key] for key in ('dataset', 'arm', 'scope', 'baseline_accuracy_pct', 'arm_accuracy_pct', 'delta_pp', 'ci95_delta_pp')}
                for row in head_report['comparisons'] if row['policy'] == 'probability_mean'])
        proof['per_head_per_seed'] = []
        for dataset in DATASETS:
            for head in HEADS:
                for seed in (0, 1, 2):
                    for arm in ARMS:
                        rows = [row for row in report['per_corruption'] if row['dataset'] == dataset and row['backbone'] == head
                                and row['seed'] == seed and row['arm'] == arm and row['policy'] == 'probability_mean']
                        require(len(rows) == 15 and sum(row['n'] for row in rows) == 2100, 'Head/seed all15 denominator differs')
                        proof['per_head_per_seed'].append(dict(dataset=dataset, head=head, seed=seed, arm=arm,
                            correct=sum(row['correct'] for row in rows), n=2100, accuracy_pct=100 * sum(row['correct'] for row in rows) / 2100))
        proof['uncertainty'] = 'Within-dataset original-object bootstrap retains all included seed/corruption values; per-head CIs condition on the three included seeds, without treating seeds or heads as independent objects.'
        proof['per_head_aggregation'] = 'Each head and arm uses6300decisions=140objects*15corruptions*3seeds; each per-seed row uses2100. Per-head CIs are computed anew from paired object clusters, never averaged across seed CIs or across heads.'
    write_new(output, proof)
    print(json.dumps(dict(status=proof['status'], raw_heads=len(raw), caches=len(caches), output=str(output))), flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--cache-manifest', type=Path)
    args = parser.parse_args()
    accept(args.manifest, args.output, args.partial, args.cache_manifest)
