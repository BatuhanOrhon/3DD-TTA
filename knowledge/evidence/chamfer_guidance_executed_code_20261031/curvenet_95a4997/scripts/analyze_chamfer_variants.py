"""CPU-only paired accuracy and observed cost analysis of sealed Chamfer arms."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chamfer_calibration_report import validate_calibration_report
from research_artifacts import CORRUPTIONS
from scripts.analyze_scan_trajectory import Bundle, analyze_bundle, policies, require


def _arm(config):
    experiment = config.get('chamfer_experiment')
    if experiment is None:
        require(config.get('component_arm') == 'chamfer', 'Baseline must be registered Chamfer C')
        return 'baseline'
    variant = experiment.get('variant')
    require(variant in ('sparse', 'semantic', 'weighted'), 'Unknown Chamfer experiment')
    if variant == 'semantic':
        kind = config.get('semantic_guidance', {}).get('kind')
        require(kind in ('kl', 'marginal_entropy'), 'Semantic KL/H arm must be explicit')
        if config.get('semantic_guidance', {}).get('head') == 'curvenet':
            require(config['dataset'] == 'scanobjectnn_c' and kind == 'marginal_entropy', 'CurveNet teacher is Scan/H only')
            return 'semantic_h_curvenet'
        return 'semantic_kl' if kind == 'kl' else 'semantic_h'
    return variant


def _identities(values, name):
    require(isinstance(values, dict) and bool(values), 'Missing ' + name + ' identities')
    result = {}
    for key, value in values.items():
        require(isinstance(value, dict) and type(value.get('bytes')) is int and value['bytes'] >= 0
                and re.fullmatch('[0-9a-f]{64}', value.get('sha256', '')),
                'Invalid ' + name + ' identity: ' + key)
        result[key] = {k: value[k] for k in ('bytes', 'sha256')}
    return result


def _calibration(bundle, config, arm):
    reference = config.get('calibration_reference')
    if reference is None:
        return None
    require(arm in ('weighted', 'semantic_kl', 'semantic_h', 'semantic_h_curvenet'), 'Calibration attached to unexpected arm')
    require('calibration_report.json' in bundle.names, 'Sealed calibration report is missing')
    raw = bundle.read('calibration_report.json')
    require(reference.get('identity') == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
            'Calibration report byte/SHA proof mismatch')
    curve = arm == 'semantic_h_curvenet'
    if curve:
        from scan_curvenet_entropy_report import PURPOSE, REDUCTION, RULE, validate_report
        report = validate_report(json.loads(raw))
        purpose, reduction, rule, datasets = PURPOSE, REDUCTION, RULE, {'scanobjectnn_c'}
        teacher = report['teacher']
        require(reference.get('teacher') == teacher
                and config['semantic_guidance'].get('teacher_checkpoint') == teacher['checkpoint']
                and config['semantic_guidance'].get('teacher_preprocessing') == teacher['preprocessing']
                and config['asset_manifest']['files'].get(teacher['checkpoint']) == teacher['checkpoint_identity'],
                'CurveNet calibrated teacher checkpoint/preprocessing mismatch')
    else:
        report = validate_calibration_report(json.loads(raw))
        purpose = 'label_free_chamfer_calibration'
        reduction = 'sum for Chamfer, weighted Chamfer, and per-object semantic losses'
        rule = 'semantic=0.1/median(candidate/C combined update norm); weighted_scale=1/median(weighted/C)'
        datasets = {'modelnet40_c', 'scanobjectnn_c'}
    require(report.get('schema_version') == reference.get('schema_version') == 1
            and report.get('purpose') == reference.get('purpose') == purpose
            and report.get('status') == 'verified_label_free_gpu' and report.get('target_semantic_update_ratio') == .1,
            'Actual frozen-rule label-free calibration proof missing')
    require(report.get('loss_reduction') == reduction,
            'Calibration batch-independent loss reduction is missing')
    proofs = report.get('calibration_bundles', {})
    require(set(proofs) == datasets and reference.get('calibration_bundles') == proofs,
            'Calibration dataset proof mismatch')
    require(reference.get('formula') == report.get('rule') == rule, 'Calibration rule changed')
    for proof in proofs.values():
        require(re.fullmatch('[0-9a-f]{64}', proof.get('asset_manifest_sha256', ''))
                and proof['asset_manifest_sha256'] == config.get('asset_manifest_sha256')
                and re.fullmatch('[0-9a-f]{64}', proof.get('config_identity', {}).get('sha256', ''))
                and proof.get('runtime', {}).get('cuda') and proof['runtime'].get('gpu'),
                'Calibration source/asset/GPU proof missing or mismatched')
        _identities(proof.get('executed_source_files'), 'calibration source')
        _identities(proof.get('native_extensions'), 'calibration native')
    proof = proofs[config['dataset']]
    common = set(proof['executed_source_files']) & set(config.get('executed_source_files', {}))
    calibration_native = _identities(proof['native_extensions'], 'calibration native')
    run_native = _identities(config.get('native_extensions'), 'native')
    require('repo/tta_gsd_v2.py' in common and all(proof['executed_source_files'][key] == config['executed_source_files'][key] for key in common)
            and all(run_native.get(name) == identity for name, identity in calibration_native.items())
            and proof.get('lion_state') == config.get('lion_state_before'), 'Calibration/run source/native/LION mismatch')
    if config['backbone'] == ('curvenet' if curve else 'pointmae'):
        require(proof.get('classifier_state_sha256') == config.get('classifier_state_before_sha256'),
                'Calibration Point-MAE classifier identity mismatch')
    key = 'weighted_scale' if arm == 'weighted' else arm
    scalar = report.get('coefficients', {}).get(key)
    require(reference.get('coefficient_key') == key and reference.get('resolved_scalar') == scalar
            and type(scalar) in (float, int) and math.isfinite(scalar) and scalar > 0,
            'Fixed calibration coefficient mismatch')
    ratio_key = 'weighted' if arm == 'weighted' else arm
    median = report.get('ratios', {}).get(ratio_key, {}).get('median')
    target = 1. if arm == 'weighted' else .1
    require(type(median) in (float, int) and math.isfinite(median) and median > 0
            and math.isclose(scalar, target / median, rel_tol=1e-12), 'Coefficient violates fixed median-ratio rule')
    if arm.startswith('semantic_'):
        require(config['chamfer_experiment'].get('semantic_weight') == scalar
                and config['semantic_guidance'].get('weight') == scalar
                and config['semantic_guidance'].get('loss_reduction') == 'sum', 'Semantic weight/reduction differs from fixed calibration')
    return reference


def _load(path):
    validation = analyze_bundle(path, allow_partial=True)
    require(validation['selection_complete'], 'Truncated declared selection is not accepted')
    bundle = Bundle(path)
    try:
        config = bundle.json('config.json')
        arm = _arm(config)
        calibration = _calibration(bundle, config, arm)
        protocol = config.get('guidance_protocol', {})
        weights = protocol.get('loss_weights', {})
        require(weights.get('chamfer', 0) > 0 and weights.get('spectral') == 0 and weights.get('invariant') == 0
                and protocol.get('skip_inactive_components') is True,
                'Every arm, including baseline C, requires Chamfer-only and skip_inactive_components')
        _identities(config.get('asset_manifest', {}).get('files'), 'asset')
        sources = _identities(config.get('executed_source_files'), 'executed source')
        require('repo/tta_gsd_v2.py' in sources, 'Sampler source identity is required')
        sealed = bundle.json('bundle_manifest.json')
        for name, identity in sources.items():
            if name.startswith('repo/'):
                require(sealed.get('source_snapshot/' + name[5:]) == identity,
                        'Executed source bytes not preserved: ' + name)
        native = _identities(config.get('native_extensions'), 'native extension')
        require(re.fullmatch('[0-9a-f]{64}', config.get('classifier_state_before_sha256', ''))
                and bool(config['lion_state_before']) and
                all(re.fullmatch('[0-9a-f]{64}', value) for value in config['lion_state_before'].values()),
                'Missing frozen classifier/LION state hashes')
        require(config.get('reconstruction_group_id'), 'Missing shared reconstruction group')
        cells, batches = {}, []
        for corruption in config['corruptions']:
            parts = []
            names = sorted(n for n in bundle.names if n.startswith('batches/' + corruption + '/') and n.endswith('/predictions.npz'))
            for name in names:
                with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as values:
                    data = {key: values[key].copy() for key in ('index', 'label', 'trajectory_logits')}
                diagnostic = bundle.json(name.rsplit('/', 1)[0] + '/diagnostics.json')
                metrics = diagnostic.get('guidance_metrics', {})
                require(metrics.get('graph_calls') == 0 and metrics.get('spectral_loss_calls') == 0,
                        'Zero actual graph/spectral calls must be recorded for every batch')
                initial = diagnostic['trajectory']['initial_hashes']
                required = {'input_points', 'shape_latent', 'local_latent', 'style', 'noise', 'noisy_local', 'timesteps', 'alpha_bar'}
                require(required <= set(initial) and all(re.fullmatch('[0-9a-f]{64}', initial[k]) for k in required),
                        'Missing actual input/h0/noise/scheduler initial hashes')
                rng = diagnostic['trajectory'].get('initial_rng_sha256')
                require(isinstance(rng, str) and re.fullmatch('[0-9a-f]{64}', rng), 'Missing actual sampler initial RNG hash')
                mode = diagnostic.get('preparation_mode')
                require(mode in ('fresh_encode', 'cached_replay', 'cache_replay'), 'Unknown preparation cost mode')
                shared, head = diagnostic.get('shared_reconstruction_seconds'), diagnostic.get('head_seconds')
                require(all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0 for v in (shared, head)),
                        'Missing/invalid actual shared reconstruction or terminal head time')
                batches.append(dict(dataset=config['dataset'], arm=arm, seed=config['seed'], backbone=config['backbone'],
                                    corruption=corruption, batch=name.split('/')[-2], group=config['reconstruction_group_id'],
                                    mode='cache_replay' if mode in ('cached_replay', 'cache_replay') else mode,
                                    shared_seconds=shared, head_seconds=head, initial_hashes=initial, initial_rng_sha256=rng,
                                    indices=data['index'].tolist()))
                parts.append(data)
            ids = np.concatenate([part['index'] for part in parts])
            labels = np.concatenate([part['label'] for part in parts])
            predictions = policies(np.concatenate([part['trajectory_logits'] for part in parts]), terminal=3)
            cells[corruption] = dict(indices=ids, labels=labels,
                                     **{policy: predictions[policy] for policy in ('terminal', 'probability_mean')})
        return dict(config=config, arm=arm, cells=cells, batches=batches, sources=sources, native=native, calibration=calibration,
                    status='complete' if validation['status'] == 'complete' else 'complete_subset', input=str(path))
    finally:
        bundle.close()


def cluster_bootstrap(object_values: dict, replicates: int = 2000, seed: int = 0) -> np.ndarray:
    """Resample original objects, retaining all their head/seed/corruption values."""
    require(type(replicates) is int and replicates > 0 and bool(object_values), 'Positive replicates and object clusters required')
    sizes = {len(values) for values in object_values.values()}
    require(len(sizes) == 1 and next(iter(sizes)) > 0, 'Objects must have the same paired condition set')
    means = np.asarray([np.mean(object_values[key]) for key in sorted(object_values)], dtype=float)
    require(np.isfinite(means).all(), 'Nonfinite cluster values')
    rng = np.random.default_rng(seed)
    # Loop avoids a reps x full-benchmark allocation for larger campaigns.
    return np.asarray([means[rng.integers(0, len(means), len(means))].mean() for _ in range(replicates)])


def _paired_identity(baseline, treatment, *, fresh_timing=False):
    a, b = baseline['config'], treatment['config']
    for key in ('asset_manifest', 'lion_state_before', 'classifier_state_before_sha256', 'selected_original_ids'):
        require(a.get(key) == b.get(key), 'Unpaired ' + key)
    for key in ('preprocessing', 'source_classifier_preprocessing', 'decoder_style',
                'scheduler_final_alpha_to_one', 'resolved_lion_config', 'cudnn_policy'):
        require(a.get(key) == b.get(key), 'Unpaired host ' + key)
    protocols = [{k: v for k, v in config['guidance_protocol'].items() if k != 'chamfer_experiment'}
                 for config in (a, b)]
    if treatment['arm'] == 'weighted' and treatment['calibration'] is not None:
        scalar = treatment['calibration']['resolved_scalar']
        for key in ('eta', 'gamma'):
            require(protocols[0].get(key) == .01 and type(protocols[1].get(key)) in (float, int)
                    and math.isclose(protocols[1][key], .01 * scalar, rel_tol=1e-12, abs_tol=0),
                    'Weighted rates differ from the proven fixed calibration')
            protocols[1][key] = protocols[0][key]
    require(protocols[0] == protocols[1], 'Unpaired host guidance protocol')
    require(baseline['native'] == treatment['native'], 'Unpaired native hashes')
    common = set(baseline['sources']) & set(treatment['sources'])
    require(common, 'No common executed sources')
    require(all(baseline['sources'][key] == treatment['sources'][key] for key in common), 'Unpaired executed source hashes')
    require(set(baseline['cells']) == set(treatment['cells']), 'Unpaired corruption scope')
    differences = []
    for first, second in zip(baseline['batches'], treatment['batches']):
        require((first['corruption'], first['batch'], first['indices']) == (second['corruption'], second['batch'], second['indices']),
                'Unpaired batch/IDs')
        if fresh_timing:
            require(first['mode'] == second['mode'] == 'fresh_encode', 'Timing pairing requires fresh encode')
            encoded = {'shape_latent', 'local_latent', 'style', 'noisy_local'}
            before, after = first['initial_hashes'], second['initial_hashes']
            require(set(before) == set(after) and all(before[key] == after[key] for key in set(before) - encoded)
                    and first['initial_rng_sha256'] == second['initial_rng_sha256'],
                    'Unpaired actual input/noise/scheduler/RNG timing controls')
            fields = {key: dict(reference_sha256=before[key], compared_sha256=after[key])
                      for key in sorted(encoded) if before[key] != after[key]}
            if fields:
                differences.append(dict(dataset=b['dataset'], arm=treatment['arm'], backbone=b['backbone'], seed=b['seed'],
                                        corruption=first['corruption'], batch=first['batch'], fields=fields))
        else:
            require(first['initial_hashes'] == second['initial_hashes'] and first['initial_rng_sha256'] == second['initial_rng_sha256'],
                    'Unpaired actual input/h0/noise/RNG initial state')
    require(len(baseline['batches']) == len(treatment['batches']), 'Unpaired batch count')
    return differences


def _costs(runs):
    groups = {}
    identities = {}
    for run in runs:
        for batch in run['batches']:
            key = tuple(batch[k] for k in ('dataset', 'arm', 'seed', 'corruption', 'batch', 'group'))
            if key not in groups:
                groups[key] = {k: batch[k] for k in ('dataset', 'arm', 'seed', 'corruption', 'batch', 'group', 'mode', 'shared_seconds')}
                groups[key].update(terminal_head_seconds=0., heads=[])
                identities[key] = (batch['indices'], batch['initial_hashes'], batch['initial_rng_sha256'])
            row = groups[key]
            require(row['mode'] == batch['mode'] and row['shared_seconds'] == batch['shared_seconds'],
                    'Shared reconstruction cost differs across heads')
            require(identities[key] == (batch['indices'], batch['initial_hashes'], batch['initial_rng_sha256']),
                    'Shared reconstruction initial state differs across heads')
            require(batch['backbone'] not in row['heads'], 'Duplicate head cost')
            row['heads'].append(batch['backbone'])
            row['terminal_head_seconds'] += batch['head_seconds']
    for row in groups.values():
        row['total_seconds'] = row['shared_seconds'] + row['terminal_head_seconds']
    return list(groups.values())


def _cost_comparisons(costs):
    totals = {}
    for row in costs:
        key = (row['dataset'], row['arm'], row['seed'], row['mode'])
        total = totals.setdefault(key, dict(seconds=0., conditions=set()))
        total['seconds'] += row['total_seconds']
        total['conditions'].add((row['corruption'], row['batch'], tuple(sorted(row['heads']))))
    comparisons = []
    for (dataset, arm, seed, mode), value in sorted(totals.items()):
        if arm == 'baseline':
            continue
        baseline = totals.get((dataset, 'baseline', seed, mode))
        if baseline is None or baseline['conditions'] != value['conditions']:
            continue  # Replay and fresh encode never form a timing comparison.
        before, after = baseline['seconds'], value['seconds']
        comparisons.append(dict(dataset=dataset, arm=arm, seed=seed, mode=mode,
                                baseline_seconds=before, arm_seconds=after, delta_seconds=after-before,
                                arm_over_baseline=after/before if before > 0 else None,
                                interpretation='observed replay execution cost' if mode == 'cache_replay'
                                else 'observed fresh-encode execution cost'))
    return comparisons


def analyze_campaign(paths: list, bootstrap_replicates: int = 2000, bootstrap_seed: int = 0) -> dict:
    """Compare independent arms to C on exactly the same declared conditions."""
    runs = [_load(path) for path in paths]
    lookup = {}
    per_corruption = []
    for run in runs:
        config = run['config']
        key = (config['dataset'], run['arm'], config['backbone'], config['seed'])
        require(key not in lookup, 'Duplicate dataset/arm/head/seed bundle')
        lookup[key] = run
        for corruption, cell in run['cells'].items():
            for policy in ('terminal', 'probability_mean'):
                correct = cell[policy] == cell['labels']
                per_corruption.append(dict(dataset=config['dataset'], arm=run['arm'], backbone=config['backbone'], seed=config['seed'],
                                           corruption=corruption, policy=policy, n=len(correct), correct=int(correct.sum()),
                                           accuracy_pct=float(100 * correct.mean())))
    require(any(run['arm'] == 'baseline' for run in runs) and any(run['arm'] != 'baseline' for run in runs),
            'Baseline C and at least one independent arm are required')
    comparisons = []
    for dataset, arm in sorted({key[:2] for key in lookup if key[1] != 'baseline'}):
        baseline_conditions = {(k[2], k[3]) for k in lookup if k[:2] == (dataset, 'baseline')}
        conditions = {(k[2], k[3]) for k in lookup if k[:2] == (dataset, arm)}
        require(conditions == baseline_conditions and bool(conditions), 'Unpaired dataset/head/seed condition set')
        paired = []
        for head, seed in sorted(conditions):
            baseline, treatment = lookup[(dataset, 'baseline', head, seed)], lookup[(dataset, arm, head, seed)]
            _paired_identity(baseline, treatment)
            paired.append((baseline, treatment))
        corruptions = set(paired[0][0]['cells'])
        require(all(set(a['cells']) == corruptions for a, _ in paired), 'Corruption scope differs across head/seed conditions')
        scopes = [('all15', corruptions), ('macro14', corruptions - {'background'})] if corruptions == set(CORRUPTIONS) else [('declared_subset', corruptions)]
        for scope, selected in scopes:
            for policy in ('terminal', 'probability_mean'):
                objects, corrected_ids, harmed_ids, baseline_values, treatment_values = {}, [], [], [], []
                reference_ids = None
                for baseline, treatment in paired:
                    for corruption in sorted(selected):
                        a, b = baseline['cells'][corruption], treatment['cells'][corruption]
                        require(np.array_equal(a['indices'], b['indices']) and np.array_equal(a['labels'], b['labels']),
                                'Paired original IDs/labels differ')
                        if reference_ids is None:
                            reference_ids = a['indices']
                        require(np.array_equal(reference_ids, a['indices']), 'Every head/seed/corruption must use the same object set')
                        first, second = a[policy] == a['labels'], b[policy] == b['labels']
                        baseline_values.extend(first.tolist()); treatment_values.extend(second.tolist())
                        for index, before, after in zip(a['indices'], first, second):
                            objects.setdefault(int(index), []).append(float(after) - float(before))
                            identity = dict(backbone=baseline['config']['backbone'], seed=baseline['config']['seed'],
                                            corruption=corruption, index=int(index))
                            if after and not before:
                                corrected_ids.append(identity)
                            elif before and not after:
                                harmed_ids.append(identity)
                replicates = cluster_bootstrap(objects, bootstrap_replicates, bootstrap_seed)
                comparisons.append(dict(dataset=dataset, arm=arm, policy=policy, scope=scope, objects=len(objects),
                                        paired_observations=len(baseline_values), baseline_accuracy_pct=100 * float(np.mean(baseline_values)),
                                        arm_accuracy_pct=100 * float(np.mean(treatment_values)), delta_pp=100 * float(np.mean(treatment_values) - np.mean(baseline_values)),
                                        ci95_delta_pp=(100 * np.quantile(replicates, [.025, .975])).tolist(),
                                        corrected_ids=corrected_ids, harmed_ids=harmed_ids))
    costs = _costs(runs)
    return dict(status='complete' if all(run['status'] == 'complete' for run in runs) else 'complete_subset',
                primary_policy='probability_mean', primary_scope='all15 when available; otherwise declared_subset',
                bundles=[dict(input=run['input'], status=run['status'], arm=run['arm'],
                              executed_source_files=run['sources'], calibration_reference=run['calibration']) for run in runs], per_corruption=per_corruption,
                comparisons=comparisons, costs=costs, cost_comparisons=_cost_comparisons(costs),
                bootstrap=dict(replicates=bootstrap_replicates, seed=bootstrap_seed,
                               unit='original object within dataset; all paired heads/seeds/corruptions retained'),
                cost_note='Observed wall time: shared reconstruction counted once per group plus terminal heads. '
                          'cache_replay costs include cache I/O and are not fresh-encode deployment speed. '
                          'Fresh-encode timing is separate; no equal-time voting comparison is inferred.')


def analyze_repeated_costs(groups: list) -> dict:
    """Validate three independent fresh-encode timing repeats without pooling accuracy."""
    require(isinstance(groups, list) and len(groups) == 3 and all(isinstance(group, list) and group for group in groups),
            'Timing requires three nonempty repeat groups')
    repeats, reference, used_groups = [], None, set()
    for repeat, paths in enumerate(groups):
        runs = [_load(path) for path in paths]
        differences = []
        lookup = {}
        for run in runs:
            config = run['config']
            key = (config['dataset'], run['arm'], config['backbone'], config['seed'])
            require(key not in lookup, 'Duplicate dataset/arm/head/seed within timing repeat')
            require(config['seed'] == 0 and config['batch_size'] == 70
                    and config.get('selection_manifest', {}).get('stage') == 'development'
                    and len(config.get('selected_original_ids', [])) == 70
                    and set(config['corruptions']) == {'gaussian', 'background'}
                    and all(batch['mode'] == 'fresh_encode' for batch in run['batches']),
                    'Timing requires seed0/B70/development70 Gaussian+background fresh encode')
            lookup[key] = run
        conditions = {(dataset, head, seed) for dataset, arm, head, seed in lookup if arm == 'baseline'}
        arms = {key[1] for key in lookup}
        require('baseline' in arms and len(arms) > 1, 'Timing requires C and independent arms')
        for arm in arms:
            require({(dataset, head, seed) for dataset, candidate, head, seed in lookup if candidate == arm} == conditions,
                    'Unpaired timing dataset/head/seed condition set')
        for (dataset, arm, head, seed), run in lookup.items():
            if arm != 'baseline':
                differences.extend(dict(row, comparison='within_repeat_baseline_to_arm', reference_repeat=repeat)
                                   for row in _paired_identity(lookup[(dataset, 'baseline', head, seed)], run, fresh_timing=True))
        if reference is None:
            reference = lookup
        else:
            require(set(lookup) == set(reference), 'Timing repeat condition set differs')
            for key, run in lookup.items():
                for field in ('guidance_protocol', 'chamfer_experiment', 'semantic_guidance', 'calibration_reference'):
                    require(run['config'].get(field) == reference[key]['config'].get(field),
                            'Timing repeat experiment/calibration changed: ' + field)
            for dataset, head, seed in conditions:
                differences.extend(dict(row, comparison='cross_repeat_baseline', reference_repeat=0)
                                   for row in _paired_identity(reference[(dataset, 'baseline', head, seed)],
                                                               lookup[(dataset, 'baseline', head, seed)], fresh_timing=True))
        owners = {}
        for run in runs:
            owner = (run['config']['dataset'], run['arm'], run['config']['seed'])
            group = run['config']['reconstruction_group_id']
            require(group not in owners or owners[group] == owner, 'Timing reconstruction group reused across arms/datasets')
            owners[group] = owner
        require(not used_groups.intersection(owners), 'Timing reconstruction group reused across repeats')
        used_groups.update(owners)
        costs = _costs(runs)
        comparisons = _cost_comparisons(costs)
        require({(row['dataset'], row['arm'], row['seed']) for row in comparisons}
                == {(dataset, arm, seed) for dataset, arm, head, seed in lookup if arm != 'baseline'},
                'Timing costs lack matched baseline head/condition groups')
        repeats.append(dict(repeat=repeat, bundles=[str(path) for path in paths], costs=costs,
                            cost_comparisons=comparisons, encoded_state_differences=differences))
    summary = {}
    for report in repeats:
        for row in report['cost_comparisons']:
            key = (row['dataset'], row['arm'], row['seed'])
            values = summary.setdefault(key, dict(dataset=key[0], arm=key[1], seed=key[2], mode='fresh_encode',
                                                  baseline_seconds=[], arm_seconds=[], delta_seconds=[], arm_over_baseline=[]))
            for name in ('baseline_seconds', 'arm_seconds', 'delta_seconds', 'arm_over_baseline'):
                values[name].append(row[name])
    for values in summary.values():
        require(len(values['baseline_seconds']) == 3, 'Missing matched timing repeat')
        values['median_baseline_seconds'] = float(np.median(values['baseline_seconds']))
        values['median_arm_seconds'] = float(np.median(values['arm_seconds']))
    return dict(status='complete_repeated_timing', repeats=repeats, summary=list(summary.values()),
                exact_encoded_initial_states=not any(report['encoded_state_differences'] for report in repeats),
                cost_note='Three fresh-encode repeats paired by input/noise/scheduler/RNG, not exact encoded-state pairing; '
                          'observed encoded differences are reported. Shared reconstruction counted once per group plus terminal heads. '
                          'Repeat observations are kept separate; no accuracy or equal-time voting inference.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundles', nargs='*', type=Path)
    parser.add_argument('--timing-manifest', type=Path, help='JSON {schema_version:1, repeats:[[bundle paths], ...]}')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.timing_manifest:
        require(not args.bundles, 'Timing manifest and accuracy bundles are separate inputs')
        manifest = json.loads(args.timing_manifest.read_text())
        require(manifest.get('schema_version') == 1, 'Unknown timing manifest schema')
        report = analyze_repeated_costs(manifest.get('repeats'))
    else:
        require(bool(args.bundles), 'Accuracy bundles or a timing manifest are required')
        report = analyze_campaign(args.bundles, args.bootstrap_replicates, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], comparisons=len(report.get('comparisons', report.get('summary', []))))))


if __name__ == '__main__':
    main()
