"""Independent CPU-only checks and object-cluster analysis for depth bundles.

Dependencies are the accepted safe sealed Bundle reader, generic temperature-one
policies, NumPy, and this study's registry. Failed endpoints remain unavailable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from itertools import combinations
import json
import pickle
from pathlib import Path
import sys
import zipfile

import numpy as np

PREP = Path(__file__).resolve().parent
sys.path.insert(0, str(PREP.parents[2]))
from research_artifacts import CORRUPTIONS
from scripts.analyze_legacy_trajectory import policies
from scripts.analyze_scan_trajectory import Bundle
from campaign import config_id, digest, identity, is_sha, read, require, write_new

POLICIES = ('terminal', 'minimum_entropy', 'probability_mean')


def subset(expected, actual, context='config'):
    for key, value in expected.items():
        require(key in actual, context + ' missing ' + key)
        if isinstance(value, dict):
            subset(value, actual[key], context + '.' + key)
        else:
            require(value == actual[key], context + ' mismatch ' + key)


def clean_steps(depth):
    return sorted({(depth * q + 3) // 4 for q in (1, 2, 3)})


def finite_nonnegative(value, name):
    require(isinstance(value, (int, float)) and np.isfinite(value) and value >= 0, 'Invalid ' + name)
    return float(value)


def verify_cache(path, expected_identity, initial, count):
    """Verify actual latent/noise bytes and serialized RNG, with no torch import."""
    require(np.__version__.split('.')[0] == '1', 'Physical cache RNG verification requires pinned NumPy1')
    require(identity(path) == {k: expected_identity[k] for k in ('bytes', 'sha256')}, 'Physical paired cache identity mismatch')
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata'].item()))
        require(metadata.get('schema_version') == 1 and metadata.get('cuda_devices') == 1 and
                metadata.get('input_sha256') == initial['input_points'], 'Paired cache input/schema mismatch')
        require(set(cache.files) == {'metadata', 'shape_latent', 'local_latent', 'noise', 'numpy_rng_keys',
                                     'torch_rng_cpu', 'torch_rng_cuda_0'}, 'Unexpected paired cache members')
        for key in ('shape_latent', 'local_latent', 'noise'):
            values = cache[key]
            actual = hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode() + values.tobytes()).hexdigest()
            require(values.dtype == np.float32 and values.ndim == 4 and values.shape[0] == count and
                    values.shape[2:] == (1, 1) and np.isfinite(values).all() and
                    actual == metadata['tensor_hashes'][key] == initial[key], 'Physical cached latent/noise mismatch')
        require(cache['local_latent'].shape == cache['noise'].shape == (count, 8192, 1, 1), 'Local latent/noise shape changed')
        py, nr = metadata['python_rng'], metadata['numpy_rng']
        state = ((py[0], tuple(py[1]), py[2]), (nr[0], cache['numpy_rng_keys'].copy(), *nr[1:]),
                 cache['torch_rng_cpu'].copy(), [cache['torch_rng_cuda_0'].copy()])
        require(hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest() == metadata['rng_after_sha256'] and
                is_sha(metadata['rng_before_sha256']), 'Physical cached RNG mismatch')


def analyze_bundle(path, plan, job, cache_root=None):
    bundle = Bundle(path)
    try:
        cfg = bundle.json('config.json')
        subset(plan['binding']['expected_config'], cfg)
        subset({'normal_reverse_steps': job['config']['normal_reverse_steps']}, cfg)
        subset({k: job['config'][k] for k in ('p', 'gamma', 'eta')}, cfg.get('guidance_protocol', {}), 'guidance_protocol')
        require(cfg.get('method') == '3dd_chamfer_depth_study' and cfg.get('dataset') == 'modelnet40_c' and
                cfg.get('backbone') == 'pointmae' and cfg.get('batch_size') == 32 and cfg.get('severity') == 5 and
                cfg.get('seed') == job['seed'], 'Worker scientific identity mismatch')
        smoke = job.get('technical_smoke', False)
        require(cfg.get('technical_smoke', False) == smoke, 'Smoke/research mismatch')
        require(cfg.get('status') in (('complete', 'partial') if smoke else ('complete',)), 'Unavailable endpoint')
        require(cfg.get('background_reverse_steps') == 35, 'Background must remain35')
        require(cfg.get('guidance_rate_mapping') == {'gamma': 'local_latent', 'eta': 'conditioning_style'}, 'Rate mapping changed')
        subset(dict(total=100, scd_normalize=False), cfg.get('guidance_protocol', {}), 'guidance_protocol')
        subset(dict(temperature=1., style='original encoded shape_latent for every clean and terminal decode',
                    labels_used=False, early_stopping=False), cfg.get('trajectory_selection', {}), 'trajectory_selection')
        selection = plan['selections'][job['split']]['manifest']
        expected_ids = selection['indices']
        if smoke:
            expected_ids = plan['technical_smoke']['indices']
            selection = dict(selection, stage='smoke', indices=expected_ids)
        require(bundle.json('selection_manifest.json') == selection and
                cfg.get('selected_original_ids') == expected_ids and cfg.get('evaluation_n_examples') == len(expected_ids),
                'Original object selection differs from reservation')
        require(cfg.get('selection_manifest_identity') == dict(bytes=len(bundle.read('selection_manifest.json')),
                sha256=hashlib.sha256(bundle.read('selection_manifest.json')).hexdigest()), 'ID file identity mismatch')
        require(cfg.get('classifier_state_before_sha256') and
                cfg['classifier_state_before_sha256'] == cfg.get('classifier_state_after_sha256'), 'Classifier state changed/missing')
        require(cfg.get('lion_state_before') and cfg['lion_state_before'] == cfg.get('lion_state_after'), 'LION state changed/missing')
        subset(plan['binding']['required_source_files'], cfg.get('executed_source_files', {}), 'executed_source_files')
        for name, expected in plan['binding']['required_source_files'].items():
            if name.startswith('repo/'):
                raw = bundle.read('source_snapshot/' + name[len('repo/'):])
            elif name.startswith('assets/source_snapshot/'):
                snapshot_bytes = bundle.read('source_snapshot.zip')
                require(hashlib.sha256(snapshot_bytes).hexdigest() == cfg['source_snapshot_sha256'], 'Preserved source archive SHA mismatch')
                with zipfile.ZipFile(io.BytesIO(snapshot_bytes)) as snapshot:
                    target = name[len('assets/'):]
                    require(snapshot.namelist().count(target) == 1 and snapshot.testzip() is None, 'Source snapshot member absent/duplicate or CRC invalid')
                    raw = snapshot.read(target)
            else:
                continue
            require(expected == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()), 'Saved executed source bytes differ: ' + name)
        corruptions = ['gaussian', 'background'] if smoke else list(CORRUPTIONS)
        require(cfg.get('corruptions') == corruptions, 'Corruption order/scope changed')
        outcomes, cases, per_corruption, batch_guards, costs, cache_proofs = {}, [], [], [], [], []
        all_shards = {n for n in bundle.names if n.startswith('batches/') and n.endswith('/predictions.npz')}
        used = set()
        label_reference = None
        for corruption in corruptions:
            depth = 35 if corruption == 'background' else job['config']['normal_reverse_steps']
            steps = clean_steps(depth)
            names = ['clean_%02d' % s for s in steps] + ['terminal']
            logits, labels = [], []
            for batch, start in enumerate(range(0, len(expected_ids), 32)):
                prefix = 'batches/%s/batch%04d/' % (corruption, batch)
                member = prefix + 'predictions.npz'
                used.add(member)
                with np.load(io.BytesIO(bundle.read(member)), allow_pickle=False) as npz:
                    data = {k: npz[k].copy() for k in npz.files}
                ids = expected_ids[start:start + 32]
                n = len(ids)
                require(data['index'].dtype.kind in 'iu' and data['index'].tolist() == ids, 'Wrong IDs, order, batch or remainder')
                require(data['label'].shape == (n,) and data['label'].dtype.kind in 'iu' and
                        ((data['label'] >= 0) & (data['label'] < 40)).all(), 'Invalid labels')
                require(data['candidate_names'].tolist() == names and data['candidate_loop_counts'].tolist() == steps + [depth],
                        'Unique dynamic candidate contract mismatch')
                require(data['trajectory_logits'].shape == (n, len(names), 40) and np.isfinite(data['trajectory_logits']).all(),
                        'Invalid trajectory logits')
                require(data['logits'].shape == (n, 40) and np.array_equal(data['trajectory_logits'][:, -1], data['logits']) and
                        np.array_equal(data['prediction'], data['logits'].argmax(1)), 'Terminal logits/predictions disagree')
                diagnostic = bundle.json(prefix + 'diagnostics.json')
                trace, scheduler = diagnostic['trajectory'], diagnostic['scheduler']
                require(trace.get('complete') is True and trace.get('horizon') == depth and
                        trace.get('candidate_names') == names and trace.get('candidate_loop_counts') == steps + [depth],
                        'Actual trace horizon/candidates mismatch')
                require(scheduler.get('active_timesteps') == list(range((depth - 1) * 10, -1, -10)) and
                        scheduler.get('reverse_steps') == depth and scheduler.get('initial_timestep') == (depth - 1) * 10 and
                        scheduler.get('config', {}).get('set_alpha_to_one') is True and
                        scheduler.get('final_alpha_cumprod') == 1., 'Actual depth/noise/DDIM schedule mismatch')
                initial = trace.get('initial_hashes', {})
                required_initial = ('input_points', 'shape_latent', 'local_latent', 'noise', 'noisy_local', 'style')
                require(all(is_sha(initial.get(k)) for k in required_initial) and trace.get('final_hashes'), 'Missing paired latent/noise hashes')
                require(trace['final_hashes'].get('style') == initial['shape_latent'],
                        'Terminal decoder style differs from encoded original style')
                require(cache_root is not None, 'Physical paired caches required for acceptance')
                recorded_cache = diagnostic['paired_state']
                cache_path = Path(cache_root) / ('paired-smoke' if smoke else 'paired') / ('seed%d' % job['seed']) / 'modelnet40_c' / corruption / Path(recorded_cache['path']).name
                require(Path(recorded_cache['path']).parent == Path(cfg['paired_state_root']) / 'modelnet40_c' / corruption,
                        'Raw cache path outside registered pairing namespace')
                verify_cache(cache_path, recorded_cache, initial, n)
                cache_proofs.append(dict(relative_path=cache_path.relative_to(cache_root).as_posix(),
                                         **{k: recorded_cache[k] for k in ('bytes', 'sha256')}))
                probes = trace.get('probes', [])
                require([p.get('loop_count') for p in probes] == steps, 'Missing/reordered probes')
                for count, probe in zip(steps, probes):
                    require(probe.get('timestep') == (depth - count) * 10 and
                            all(probe.get(k) is True for k in ('rng_unchanged', 'buffers_unchanged', 'live_inputs_unchanged')) and
                            is_sha(probe.get('rng_before_sha256')) and probe['rng_before_sha256'] == probe.get('rng_after_sha256') and
                            probe.get('buffers_before') == probe.get('buffers_after') and probe.get('buffers_before') and
                            probe.get('style_sha256') == initial['shape_latent'], 'Probe purity/timing/original-style guard failed')
                require(diagnostic.get('rng_unchanged') is True, 'Terminal classifier RNG guard failed/missing')
                require(diagnostic.get('executed_source_sha256') == hashlib.sha256(json.dumps(cfg['executed_source_files'],
                        sort_keys=True, separators=(',', ':')).encode()).hexdigest() and
                        diagnostic.get('source_snapshot_sha256') == cfg['source_snapshot_sha256'] and
                        diagnostic.get('asset_manifest_sha256') == cfg['asset_manifest_sha256'] and
                        diagnostic.get('checkpoint_load') == cfg['checkpoint_load'], 'Batch source/assets/checkpoint identity changed')
                # Only depth-independent preparation is paired; h_t depends intentionally on depth.
                batch_guards.append(dict(seed=job['seed'], corruption=corruption, batch=batch, ids=ids,
                                         paired={k: initial[k] for k in ('input_points', 'shape_latent', 'local_latent', 'noise')},
                                         source_snapshot_sha256=cfg['source_snapshot_sha256']))
                costs.append(dict(corruption=corruption, batch=batch,
                                  reconstruction_seconds=finite_nonnegative(diagnostic['shared_reconstruction_seconds'], 'reconstruction time'),
                                  terminal_head_seconds=finite_nonnegative(diagnostic['head_seconds'], 'terminal head time'),
                                  probe_dispatch_seconds=finite_nonnegative(trace['times'].get('trajectory_probe', 0), 'probe dispatch time'),
                                  peak_gpu_memory_mb=finite_nonnegative(diagnostic['whole_process_peak_gpu_memory_mb'], 'peak memory'),
                                  preparation_mode=diagnostic['preparation_mode']))
                raw_readout = policies(data['trajectory_logits'], terminal=len(names) - 1)
                for key in POLICIES + ('selected_candidate', 'entropy', 'probabilities'):
                    require(key in data and np.allclose(data[key], raw_readout[key], rtol=1e-10, atol=1e-12),
                            'Saved/recomputed readout mismatch: ' + key)
                logits.append(data['trajectory_logits'])
                labels.append(data['label'])
            labels = np.concatenate(labels)
            if label_reference is None:
                label_reference = labels
            require(np.array_equal(labels, label_reference), 'Object labels vary by corruption')
            selected = policies(np.concatenate(logits), terminal=len(names) - 1)
            for policy in POLICIES:
                correct = selected[policy] == labels
                outcomes.setdefault(policy, []).append(correct.tolist())
                terminal = selected['terminal'] == labels
                per_corruption.append(dict(corruption=corruption, policy=policy, n_objects=len(labels),
                    accuracy_pct=100. * float(correct.mean()), corrected=int((correct & ~terminal).sum()),
                    harmed=int((~correct & terminal).sum())))
                for object_id, label, pred, hit in zip(expected_ids, labels, selected[policy], correct):
                    cases.append(dict(corruption=corruption, object_id=object_id, policy=policy,
                                      label=int(label), prediction=int(pred), correct=bool(hit)))
        require(used == all_shards, 'Undeclared/missing batch shard')
        aggregate = []
        for scope in ('all15', 'macro14') if not smoke else ('smoke2',):
            for policy in POLICIES:
                rows = [r for r in per_corruption if r['policy'] == policy and (scope != 'macro14' or r['corruption'] != 'background')]
                aggregate.append(dict(scope=scope, policy=policy, accuracy_pct=float(np.mean([r['accuracy_pct'] for r in rows]))))
        return dict(status='accepted', plan_sha256=digest(plan), config=job['config'], config_id=config_id(job['config']),
                    seed=job['seed'], stage='smoke' if smoke else ('confirmation' if job['split'] == 'heldout' else 'development'),
                    bundle=Path(path).name, bundle_sha256=identity(path)['sha256'],
                    object_ids=expected_ids, corruptions=corruptions, outcomes=outcomes, cases=cases,
                    per_corruption=per_corruption, aggregate=aggregate, batch_guards=batch_guards, costs=costs, cache_proofs=cache_proofs,
                    total_cost=dict(worker_wall_seconds_including_artifact_io=finite_nonnegative(cfg['worker_wall_seconds_including_artifact_io'], 'worker wall'),
                        post_inference_policy_artifact_seconds=sum(finite_nonnegative(v, 'policy/shard/Drive IO')
                            for values in cfg['batch_post_inference_policy_artifact_seconds'].values() for v in values),
                        reconstruction_seconds=sum(r['reconstruction_seconds'] for r in costs),
                        terminal_head_seconds=sum(r['terminal_head_seconds'] for r in costs),
                        probe_dispatch_seconds=sum(r['probe_dispatch_seconds'] for r in costs),
                        peak_gpu_memory_mb=max(r['peak_gpu_memory_mb'] for r in costs)),
                    cost_policy='reconstruction includes preparation and all probe work; terminal head added once; probe dispatch is overlapping diagnostic, not added twice')
    finally:
        bundle.close()


def compare(left, right, repetitions=10000, bootstrap_seed=20261030):
    """Resample original objects jointly, retaining seed/corruption dependence."""
    a, b = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    require(a.shape == b.shape and a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] > 0,
            'Expected paired seeds x corruptions x original objects')
    require(type(repetitions) is int and repetitions > 0, 'Positive bootstrap repetitions required')
    delta = a.astype(float) - b
    object_effect = delta.mean(axis=(0, 1)) * 100.
    rng = np.random.default_rng(bootstrap_seed)
    values = np.empty(repetitions)
    for start in range(0, repetitions, 200):
        count = min(200, repetitions - start)
        indices = rng.integers(0, a.shape[2], size=(count, a.shape[2]))
        values[start:start + count] = object_effect[indices].mean(1)
    return dict(delta_pp=float(object_effect.mean()), corrected_events=int((a & ~b).sum()), harmed_events=int((~a & b).sum()),
                n_original_object_clusters=a.shape[2], seeds=a.shape[0], prediction_events=int(a.size),
                conditional_object_bootstrap_low_pp=float(np.quantile(values, .025)),
                conditional_object_bootstrap_high_pp=float(np.quantile(values, .975)),
                bootstrap_seed=bootstrap_seed, bootstrap_repetitions=repetitions,
                interval_scope='conditional on frozen settings/model/corruptions and three fixed seeds; objects resampled jointly')


def analyze_campaign(plan_path, root, output, repetitions):
    plan = read(plan_path)
    reports, unavailable = [], []
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for slot in sorted((root / 'slots').glob('*')):
        job = read(slot / 'job.json')
        archives = list((slot / 'raw').rglob('*.zip'))
        # Ignore intermediate checkpoints: only the ZIP next to one sealed run root.
        bundles = [Path(str(p.parent) + '.zip') for p in (slot / 'raw').rglob('bundle_manifest.json')]
        if not (slot / 'acceptance.json').exists() or (slot / 'failure.json').exists():
            unavailable.append(dict(slot_id=job['slot_id'], config_id=job['config_id'], seed=job['seed'],
                                    stage='confirmation' if job['split'] == 'heldout' else 'development',
                                    status='unavailable', partial_archives=[str(p) for p in archives],
                                    reason=read(slot / 'failure.json') if (slot / 'failure.json').exists() else 'not accepted/active'))
            continue
        require(len(bundles) == 1 and bundles[0].is_file(), 'Accepted slot lacks exactly one sealed bundle')
        report = analyze_bundle(bundles[0], plan, job, cache_root=root)
        require(report == read(slot / 'acceptance.json'), 'Independent analysis differs from controller acceptance')
        reports.append(report)
    # Paired preparation may be depth-invariant only inside the same split/seed/cache namespace.
    guards = {}
    for report in reports:
        for row in report['batch_guards']:
            key = (report['stage'], row['seed'], row['corruption'], tuple(row['ids']))
            if key in guards:
                require(guards[key] == row['paired'], 'Paired encoded input/latent/noise differs across settings')
            guards[key] = row['paired']
    scores = [dict(status='accepted', config_id=r['config_id'], stage=r['stage'], seed=r['seed'],
                   readout='probability_mean', accuracy_pct=next(a['accuracy_pct'] for a in r['aggregate']
                       if a['policy'] == 'probability_mean' and a['scope'] in ('all15', 'smoke2')),
                   bundle_sha256=r['bundle_sha256'], plan_sha256=digest(plan)) for r in reports if r['stage'] == 'development']
    write_new(output / 'scores.json', scores)
    comparisons = []
    frozen_path = root / 'registered_confirmation.json'
    frozen = read(frozen_path) if frozen_path.exists() else None
    if frozen:
        require(frozen['plan_sha256'] == digest(plan), 'Confirmation freeze from another plan')
    require(frozen is not None or not any(r['stage'] == 'confirmation' for r in reports), 'Confirmation results require preserved preregistered freeze')
    confirmed = {config_id(c): {} for c in frozen['frozen_conditions']} if frozen else {}
    for report in reports:
        if report['stage'] == 'confirmation':
            require(report['config_id'] in confirmed, 'Unfrozen confirmation condition')
            require(report['seed'] not in confirmed[report['config_id']], 'Duplicate configuration seed')
            confirmed[report['config_id']][report['seed']] = report
    for left, right in combinations(sorted(confirmed), 2):
        complete = all(set(confirmed[c]) == {0, 1, 2} for c in (left, right))
        if not complete:
            comparisons.append(dict(left=left, right=right, status='unavailable', reason='complete fixed three-seed endpoint required; no surviving-seed substitution'))
            continue
        a, b = confirmed[left], confirmed[right]
        require(all(a[s]['object_ids'] == b[s]['object_ids'] == plan['selections']['heldout']['manifest']['indices'] for s in (0, 1, 2)), 'Confirmation object mismatch')
        for policy in POLICIES:
            for scope in ('all15', 'macro14'):
                mask = [i for i, c in enumerate(CORRUPTIONS) if scope == 'all15' or c != 'background']
                x = np.asarray([a[s]['outcomes'][policy] for s in (0, 1, 2)])[:, mask]
                y = np.asarray([b[s]['outcomes'][policy] for s in (0, 1, 2)])[:, mask]
                for seed in (None, 0, 1, 2):
                    comparisons.append(dict(left=left, right=right, policy=policy, scope=scope, seed=seed,
                        status='complete', **compare(x if seed is None else x[seed:seed + 1],
                                                   y if seed is None else y[seed:seed + 1], repetitions)))
    # Named whole-policy comparisons to the historical original-code N5 terminal.
    baseline = config_id(__import__('campaign').config(5))
    if frozen and baseline in confirmed:
        for candidate in sorted(confirmed):
            if not all(set(confirmed[c]) == {0, 1, 2} for c in (candidate, baseline)):
                comparisons.append(dict(left=candidate, right=baseline, left_policy='probability_mean',
                    right_policy='terminal', status='unavailable', reason='fixed three-seed endpoint incomplete'))
                continue
            for scope in ('all15', 'macro14'):
                mask = [i for i, name in enumerate(CORRUPTIONS) if scope == 'all15' or name != 'background']
                x = np.asarray([confirmed[candidate][s]['outcomes']['probability_mean'] for s in (0, 1, 2)])[:, mask]
                y = np.asarray([confirmed[baseline][s]['outcomes']['terminal'] for s in (0, 1, 2)])[:, mask]
                comparisons.append(dict(left=candidate, right=baseline, left_policy='probability_mean', right_policy='terminal',
                    scope=scope, seed=None, status='complete', **compare(x, y, repetitions)))
    from campaign import ledger_events
    ledger, ledger_spent = ledger_events(root / 'wall_ledger.jsonl')
    write_new(output / 'analysis.json', dict(status='derived_CPU_analysis', plan_sha256=digest(plan),
        accepted_reports=reports, unavailable_endpoints=unavailable, paired_comparisons=comparisons,
        confirmation_freeze=frozen, cumulative_charged_wall_seconds=ledger_spent, wall_ledger=ledger,
        wall_accounting='actual worker/probe/failure/validation/Drive-copy wall plus60s per-phase bounded final ledger-copy reserve; conservative charge, not GPU billing',
        confirmation_unique_object_clusters=256, historical_benchmark_exposure=True,
        limitations=['conditional intervals, no external unseen claim', 'prediction events are not independent objects',
                     'failures cannot be replaced or converted to a completed successful prefix']))
    for filename, rows in [('per_case.csv', [dict(config_id=r['config_id'], seed=r['seed'], stage=r['stage'], **v) for r in reports for v in r['cases']]),
                           ('per_corruption_seed.csv', [dict(config_id=r['config_id'], seed=r['seed'], stage=r['stage'], **v) for r in reports for v in r['per_corruption']]),
                           ('aggregate_seed.csv', [dict(config_id=r['config_id'], seed=r['seed'], stage=r['stage'], **v) for r in reports for v in r['aggregate']]),
                           ('cost_batches.csv', [dict(config_id=r['config_id'], seed=r['seed'], stage=r['stage'], **v) for r in reports for v in r['costs']])]:
        if rows:
            with (output / filename).open('x', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return dict(accepted=len(reports), unavailable=len(unavailable), comparisons=len(comparisons))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('plan', 'root', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(analyze_campaign(args.plan, args.root, args.output, args.bootstrap_repetitions)))
