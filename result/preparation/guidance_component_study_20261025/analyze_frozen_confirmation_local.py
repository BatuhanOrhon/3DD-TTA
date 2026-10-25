"""Independent frozen70-ID three-seed confirmation; CPU only, no reselection."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

import numpy as np

MANIFEST_SHA = '7012c30ad88222bbc5435d8508f781fc33385ed3976951c45f48da67f625ed75'
SOURCE_SHA = 'f5bca11a543fd5463ef5a09b26cf216b2bd049523d61a97da24e184e5576d0c6'
PLAN_SHA = 'c6ea3e01a03061afd227c82eb30eb128c2a4f1f342dbf937732dd5e61b122d45'
OWN_DRIVE = '/content/drive/MyDrive/thesis/result/guidance_component_study_20261025/'
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
HEADS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')
ARMS = {'combined': (1., 1.17), 'chamfer': (1., 0.)}
SEEDS = (0, 1, 2)
POLICIES = ('terminal', 'minimum_entropy', 'probability_mean')
CONTRASTS = (('primary_same_readout', 'probability_mean'), ('secondary_frozen_readout', 'minimum_entropy'))
METHOD = 'gsd_inspired_latent_spectral_guidance'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(path, expected):
    observed = identity(path)
    require(observed == {key: expected[key] for key in ('bytes', 'sha256')}, 'Archive/file identity mismatch: ' + str(path))
    return observed


def aligned(reference, actual):
    require(np.array_equal(reference['index'], actual['index']) and np.array_equal(reference['label'], actual['label']),
            'Original-ID/label pairing mismatch')


def same_preparation(reference, actual):
    require(reference['initial_hashes'] == actual['initial_hashes'] and reference['paired_state'] == actual['paired_state'],
            'Unmatched prepared input/z0/h0/noise/cache identity')
    # Decoded geometry and final guided states are expected to differ between arms.


def valid_seed(observed, expected):
    require(type(observed) is int and type(expected) is int and expected in SEEDS and observed == expected, 'Unregistered/mismatched seed')


def balanced_table(rows):
    result = []
    for arm in ARMS:
        for policy in POLICIES:
            values, terminal_deltas = [], []
            for seed in SEEDS:
                group = [row for row in rows if row['component_arm'] == arm and row['policy'] == policy and row['seed'] == seed]
                require(len(group) == 8 and {(r['dataset'], r['backbone']) for r in group} ==
                        {(dataset, head) for dataset in DATASETS for head in HEADS}, 'Balanced mean requires eight unique cases per seed')
                accuracy = sum(row['accuracy_pct'] for row in group) / 8
                delta = sum(row['delta_terminal_pp'] for row in group) / 8
                values.append(accuracy); terminal_deltas.append(delta)
                result.append(dict(aggregation='seed', seed=seed, component_arm=arm, policy=policy, n_cases=8,
                                   accuracy_pct=accuracy, delta_own_terminal_pp=delta,
                                   per_seed_min_accuracy_pct=accuracy, per_seed_max_accuracy_pct=accuracy))
            result.append(dict(aggregation='three_seed_mean', seed=None, component_arm=arm, policy=policy, n_cases=8,
                               accuracy_pct=float(np.mean(values)), delta_own_terminal_pp=float(np.mean(terminal_deltas)),
                               per_seed_min_accuracy_pct=min(values), per_seed_max_accuracy_pct=max(values)))
    return result


def paired_tables(correct, corruptions, bootstrap_replicates=2000):
    """A dataset's70 ID draws are shared across all three seeds and all heads."""
    require(len(corruptions) == 15 and len(set(corruptions)) == 15 and 'background' in corruptions, 'Expected declared all15 ordering')
    rng = np.random.default_rng(20261025)
    draws = {dataset: rng.integers(0, 70, size=(bootstrap_replicates, 70)) for dataset in DATASETS}
    cases, balanced = [], []
    for scope in ('all15', 'macro14'):
        mask = np.asarray([True if scope == 'all15' else name != 'background' for name in corruptions])
        for role, comparator_policy in CONTRASTS:
            effects, selected_by_seed, sampled_by_seed = {}, {}, {}
            for seed in SEEDS:
                selected = []
                for dataset in DATASETS:
                    per_object = []
                    for head in HEADS:
                        left = correct[(dataset, head, 'chamfer', 'probability_mean', seed)]
                        right = correct[(dataset, head, 'combined', comparator_policy, seed)]
                        require(left.shape == right.shape == (15, 70) and left.dtype == right.dtype == bool,
                                'Expected paired15×70 outcomes for every seed')
                        left, right = left[mask], right[mask]
                        difference = left.astype(float) - right.astype(float)
                        per_object.append(difference.mean(axis=0))
                        delta = 100. * float(difference.mean())
                        row = dict(scope=scope, contrast=role, seed=seed, dataset=dataset, backbone=head,
                            candidate='chamfer', candidate_policy='probability_mean', comparator='combined', comparator_policy=comparator_policy,
                            n_predictions=int(mask.sum()) * 70, delta_pp=delta, corrected=int((left & ~right).sum()),
                            harmed=int((~left & right).sum()), sign='positive' if delta > 0 else 'negative' if delta < 0 else 'tied')
                        cases.append(row); selected.append(row)
                    effects[(seed, dataset)] = np.mean(per_object, axis=0)
                sampled_by_seed[seed] = 100. * np.mean([effects[(seed, dataset)][draws[dataset]].mean(axis=1) for dataset in DATASETS], axis=0)
                selected_by_seed[seed] = selected
            seed_deltas = [sum(row['delta_pp'] for row in selected_by_seed[seed]) / 8 for seed in SEEDS]
            for seed in (*SEEDS, None):
                seeds = SEEDS if seed is None else (seed,)
                selected = [row for value in seeds for row in selected_by_seed[value]]
                sampled = np.mean([sampled_by_seed[value] for value in seeds], axis=0)
                low, high = np.percentile(sampled, [2.5, 97.5])
                case_effects = [np.mean([row['delta_pp'] for row in selected if row['dataset'] == dataset and row['backbone'] == head])
                                for dataset in DATASETS for head in HEADS]
                balanced.append(dict(scope=scope, contrast=role, aggregation='three_seed_mean' if seed is None else 'seed', seed=seed,
                    candidate='chamfer', candidate_policy='probability_mean', comparator='combined', comparator_policy=comparator_policy,
                    n_cases=8, n_original_object_clusters=140, n_seeds=len(seeds), n_prediction_events=sum(row['n_predictions'] for row in selected),
                    delta_pp=float(np.mean(case_effects)), corrected_events=sum(row['corrected'] for row in selected),
                    harmed_events=sum(row['harmed'] for row in selected), positive_cases=int(sum(value > 0 for value in case_effects)),
                    negative_cases=int(sum(value < 0 for value in case_effects)), tied_cases=int(sum(value == 0 for value in case_effects)),
                    per_seed_min_pp=min(seed_deltas) if seed is None else seed_deltas[seed],
                    per_seed_max_pp=max(seed_deltas) if seed is None else seed_deltas[seed],
                    conditional_object_bootstrap_low_pp=float(low), conditional_object_bootstrap_high_pp=float(high)))
    return cases, balanced


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main(prep):
    prep = Path(prep).resolve()
    repository = prep.parents[2]
    require(repository.name == 'guidance-components', 'Use the isolated Agent B preparation directory')
    extracted = prep / 'frozen_confirmation_extracted'
    receipt = json.loads((prep / 'frozen_confirmation_receipt.json').read_text())
    phase_identity = verify(prep / 'frozen_confirmation_evidence_drive_roundtrip.zip', receipt['archive'])
    remote = PurePosixPath(receipt['archive']['local_path']).with_suffix('')
    require(remote.parent == PurePosixPath('/content/studies/guidance-components') and remote.name.startswith('frozen_confirmation_'), 'Foreign confirmation phase root')
    plan_path = prep / 'frozen_confirmation_plan.json'
    require(identity(plan_path)['sha256'] == PLAN_SHA, 'Frozen confirmation plan identity mismatch')
    plan = json.loads(plan_path.read_text())
    require(plan['seeds'] == list(SEEDS) and plan['arms'] == list(ARMS) and plan['candidate'] == dict(component_arm='chamfer', policy='probability_mean')
            and [(row['component_arm'], row['policy']) for row in plan['comparators']] ==
            [('combined', 'probability_mean'), ('combined', 'minimum_entropy')], 'Frozen candidate/comparator/seed scope changed')
    for name in ('frozen_confirmation_plan.json', 'runtime_frozen_confirmation_plan.json'):
        require((extracted / name).read_bytes() == plan_path.read_bytes(), 'Archived frozen plan bytes mismatch')
    accepted_raw = (extracted / 'accepted_development.json').read_bytes()
    accepted = json.loads(accepted_raw)
    prior_receipt = json.loads((extracted / 'accepted_development_receipt.json').read_text())
    require(dict(bytes=len(accepted_raw), sha256=hashlib.sha256(accepted_raw).hexdigest()) == prior_receipt['acceptance']
            and prior_receipt['frozen_plan'] == accepted['frozen_plan'] == identity(plan_path) and accepted['status'] == 'accepted',
            'Frozen plan/independent development acceptance receipt mismatch')
    state = receipt['state']
    require(state['status'] == 'complete' and state['mode'] == 'frozen_confirmation' and state['seeds'] == list(SEEDS)
            and state['frozen_plan_sha256'] == PLAN_SHA and state['policies_frozen'] is True
            and state['full_benchmark'] is False and state['manifest_sha256'] == MANIFEST_SHA
            and state['source_snapshot_sha256'] == SOURCE_SHA and state['session'] == 'tta-guidance-components-a100-20261025-r1'
            and receipt['archive']['drive_path'].startswith(OWN_DRIVE), 'Confirmation phase/source status mismatch')
    require(state['candidate'] == plan['candidate'] and state['comparators'] == plan['comparators'], 'Frozen readouts were reselected')
    source_manifest = extracted / 'confirmation_phase_source_identities.json'
    require(identity(source_manifest)['sha256'] == 'b95170f32c73967dd97b55057e57e9107d037b0ca8b8bd7a2fdcd39ed2337229', 'Unreviewed confirmation helper manifest')
    require(json.loads(source_manifest.read_text()) == state['helper_sources'], 'Helper state differs from reviewed source manifest')
    for name, expected in state['helper_sources'].items():
        require(Path(name).name == name, 'Unsafe archived helper name')
        verify(extracted / name, expected)
    ledger = json.loads((extracted / 'ledger_snapshot.json').read_text())
    require(ledger['frozen_plan_sha256'] == PLAN_SHA and ledger['cap_seconds'] == 3300 and ledger['active'] is None
            and ledger['session'] == state['session'] and ledger['manifest_sha256'] == MANIFEST_SHA
            and ledger['spent_seconds'] <= 3300 and ledger['attempts'][-1]['status'] == 'complete'
            and ledger['attempts'][-1]['output'] == str(remote), 'Frozen ledger cap/ownership mismatch')
    jobs = state['jobs']
    for job in jobs:
        valid_seed(job['seed'], job['seed'])
    require(len(jobs) == 12 and {(job['dataset'], job['arm'], job['seed']) for job in jobs} ==
            {(dataset, arm, seed) for dataset in DATASETS for arm in ARMS for seed in SEEDS}, 'Missing/duplicate dataset/arm/seed jobs')
    assets = prep / 'assets_v1'
    require(identity(assets / 'manifest.json')['sha256'] == MANIFEST_SHA and identity(assets / 'source_snapshot.zip')['sha256'] == SOURCE_SHA,
            'Local sealed source/assets differ from accepted execution')
    manifest = json.loads((assets / 'manifest.json').read_text())
    native_path = assets / manifest['guidance']['native_audit']
    verify(native_path, manifest['files'][manifest['guidance']['native_audit']])
    native_audit = json.loads(native_path.read_text())
    for name in ('scripts/analyze_scan_trajectory.py', 'scripts/analyze_legacy_trajectory.py'):
        verify(assets / 'source_snapshot' / name, manifest['files']['source_snapshot/' + name])
    sys.path.insert(0, str(assets / 'source_snapshot'))
    from scripts.analyze_scan_trajectory import analyze_bundle, Bundle
    from scripts.analyze_legacy_trajectory import policies
    from research_artifacts import CORRUPTIONS
    require(len(CORRUPTIONS) == 15, 'Expected all15 corruption ordering')
    for dataset in DATASETS:
        name = 'study/' + dataset + '_heldout_ids.json'
        verify(assets / name, manifest['files'][name])
        verify(assets / name, plan['id_manifests'][dataset])
    selected = {dataset: json.loads((assets / 'study' / (dataset + '_heldout_ids.json')).read_text()) for dataset in DATASETS}
    for dataset in DATASETS:
        development = json.loads((assets / 'study' / (dataset + '_development_ids.json')).read_text())
        require(not set(selected[dataset]['indices']) & set(development['indices']), 'Confirmation objects overlap development screen')
    cache_records = {}
    for job in jobs:
        require(job['status'] == 'complete' and len(job['raw_archives']) == 4, 'Incomplete arm job')
        for cached in job['cache']:
            require(cached['drive_path'].startswith(OWN_DRIVE) and len(cached['sha256']) == 64 and cached['bytes'] > 0,
                    'Missing remote-verified cache receipt identity')
            key = cached['local_path']
            namespace = plan['cache_namespace'].format(seed=job['seed'])
            require(PurePosixPath(key).relative_to(PurePosixPath('/content/studies/guidance-components')).parts[0] == namespace
                    and PurePosixPath(cached['drive_path']).relative_to(PurePosixPath(OWN_DRIVE)).parts[0] == namespace,
                    'Confirmation cache crossed seed/development namespaces')
            if key in cache_records:
                require(cache_records[key] == cached, 'Cache receipt identity changed across jobs')
            cache_records[key] = cached
    require(len(cache_records) == 90, 'Require30 separate state/noise files for each of three seeds')
    raw, macro, macro14, corruption_rows, correct, shared, head_states, costs = [], [], [], [], {}, {}, {}, []
    case_payloads = {}
    lion_state = None
    for job in jobs:
        dataset, arm, seed = job['dataset'], job['arm'], job['seed']
        expected_ids = selected[dataset]['indices']
        require(selected[dataset]['stage'] == 'heldout' and len(expected_ids) == 70, 'Expected70 frozen heldout IDs')
        cache_name = hashlib.sha256(np.asarray(expected_ids, dtype=np.int64).tobytes()).hexdigest() + '.npz'
        job_cache_paths = {cached['local_path'] for cached in job['cache']}
        heads_seen, first_timings, terminal_seconds, preparation_modes, peak = set(), {}, 0., {}, 0.
        for reference in job['raw_archives']:
            relative = PurePosixPath(reference['local_path']).relative_to(remote)
            require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe raw archive reference')
            archive = extracted.joinpath(*relative.parts)
            require(archive.suffix == '.zip' and archive.is_file() and not archive.is_symlink()
                    and extracted.resolve() in archive.resolve().parents and reference['drive_path'].startswith(OWN_DRIVE), 'Missing/non-owned raw archive')
            verify(archive, reference)
            report = analyze_bundle(archive, allow_partial=True)
            require(report['status'] == 'partial' and report['selection_complete'] and len(report['per_corruption']) == 45
                    and all(row['n'] == 70 for row in report['per_corruption']), 'Invalid all15×70 trajectory readouts')
            bundle = Bundle(archive)
            try:
                config = bundle.json('config.json')
                head = config['backbone']; heads_seen.add(head)
                require(archive.name == config['run_id'] + '.zip' and head == reference['backbone'], 'Raw run/head identity mismatch')
                valid_seed(config['seed'], seed)
                require(config['dataset'] == dataset and config['component_arm'] == arm and config['status'] == 'partial'
                        and config['component_study'] is True and config['severity'] == 5
                        and config['asset_manifest_sha256'] == MANIFEST_SHA and config['asset_manifest'] == manifest
                        and config['selected_original_ids'] == expected_ids and config['selection_manifest'] == selected[dataset]
                        and set(config['corruptions']) == set(CORRUPTIONS), 'Raw config/source/selection mismatch')
                require(config['selection_manifest']['stage'] == 'heldout' and not config['component_parameters_modified'], 'Tuned/non-heldout arm in frozen confirmation')
                expected_protocol = dict(M_max=800, beta=2., gamma=.01, eta=.01, p=.95, total=100,
                    loss_weights=dict(chamfer=ARMS[arm][0], spectral=ARMS[arm][1], invariant=0.),
                    use_static_style=False, dynamic_graph=False, graph_update_interval=3, skip_inactive_components=True)
                require(config['guidance_protocol'] == expected_protocol and config['decoder_style'] == 'updated'
                        and config['scheduler_final_alpha_to_one'] is False and config['lion_ema'] is False,
                        'Fixed same-host GSD protocol drift')
                require(config['graph'] == dict(k=10, delta=.1, gamma=.6, use_4d_gft=False, isolate_policy='legacy'), 'Graph protocol drift')
                require(config['trajectory_selection']['policies'] == list(POLICIES)
                        and config['trajectory_selection']['temperature'] == 1.
                        and config['trajectory_selection']['labels_used'] is False
                        and config['trajectory_selection']['early_stopping'] is False, 'Readout protocol drift')
                for name in ('tta_gsd_v2.py', 'shared_trajectory.py', 'scripts/eval_multibackbone_guidance.py', 'scripts/eval_multibackbone_source.py'):
                    require(config['executed_source_files'].get('assets/source_snapshot/' + name) == manifest['files']['source_snapshot/' + name],
                            'Executed consumer source mismatch: ' + name)
                require(config['native_bootstrap']['no_build'] is True and
                        set(config['native_bootstrap']['bindings']) == set(native_audit['backends']), 'No-build native binding proof missing')
                for name, entries in native_audit['backends'].items():
                    entry, = entries
                    require(config['native_bootstrap']['bindings'][name] == {key: entry[key] for key in ('path', 'bytes', 'sha256')},
                            'Pinned native binding mismatch: ' + name)
                load = config['checkpoint_load']
                checkpoint = config['pointmae_checkpoint'] if head == 'pointmae' else config['checkpoint']
                require(load['strict'] and not load['missing_keys'] and not load['unexpected_keys']
                        and load['checkpoint_sha256'] == manifest['files'][checkpoint]['sha256'], 'Classifier checkpoint mismatch')
                state_key = (dataset, head)
                current_head = dict(checkpoint_sha256=load['checkpoint_sha256'], state_sha256=config['classifier_state_before_sha256'],
                                    preprocessing=config['source_classifier_preprocessing'], resolved_model_config=config['resolved_model_config'])
                if state_key in head_states:
                    require(head_states[state_key] == current_head, 'Frozen classifier/checkpoint/preprocessing changed across arms')
                head_states[state_key] = current_head
                if lion_state is None:
                    lion_state = config['lion_state_before']
                require(config['lion_state_before'] == lion_state, 'Frozen LION state differs between cases/arms')
                current = {policy: [] for policy in POLICIES}
                for corruption in CORRUPTIONS:
                    prefix = 'batches/' + corruption + '/batch0000/'
                    with np.load(io.BytesIO(bundle.read(prefix + 'predictions.npz')), allow_pickle=False) as values:
                        arrays = {key: values[key].copy() for key in values.files}
                    require(np.array_equal(arrays['index'], expected_ids), 'Wrong original IDs')
                    payload_key = (dataset, corruption)
                    if payload_key in case_payloads:
                        aligned(case_payloads[payload_key], arrays)
                    else:
                        case_payloads[payload_key] = {key: arrays[key] for key in ('index', 'label')}
                    diagnostic = bundle.json(prefix + 'diagnostics.json')
                    peak = max(peak, diagnostic['whole_process_peak_gpu_memory_mb'])
                    paired = diagnostic['paired_state']
                    cached = cache_records.get(paired['path'])
                    expected_cache = PurePosixPath('/content/studies/guidance-components') / plan['cache_namespace'].format(seed=seed) / dataset / corruption / cache_name
                    require(PurePosixPath(paired['path']) == expected_cache and paired['path'] in job_cache_paths,
                            'Raw cache crossed seed/corruption/object-ID namespaces')
                    require(cached is not None and {key: paired[key] for key in ('bytes', 'sha256')} ==
                            {key: cached[key] for key in ('bytes', 'sha256')}, 'Paired cache lacks matching remote-verified receipt')
                    trace = dict(initial_hashes=diagnostic['trajectory']['initial_hashes'], paired_state=paired)
                    group = (dataset, seed, arm, corruption)
                    if group in shared:
                        require(shared[group] == dict(trace=trace, geometry=diagnostic['geometry_sha256'], seconds=diagnostic['shared_reconstruction_seconds']),
                                'Heads observed different supposedly shared reconstruction')
                    else:
                        shared[group] = dict(trace=trace, geometry=diagnostic['geometry_sha256'], seconds=diagnostic['shared_reconstruction_seconds'])
                        first_timings[corruption] = diagnostic['shared_reconstruction_seconds']
                        preparation_modes[corruption] = diagnostic['preparation_mode']
                    require(diagnostic['preparation_mode'] == ('fresh_encode' if arm == 'combined' else 'cached_replay'), 'Heldout cache was not fresh per seed or replayed by its paired arm')
                    horizon = 30 if corruption == 'background' else 10
                    metrics = diagnostic['guidance_metrics']
                    require(metrics['graph_calls'] == int(ARMS[arm][1] > 0)
                            and metrics['chamfer_calls'] == 2 * horizon * int(ARMS[arm][0] > 0)
                            and metrics['guidance_backward_calls'] == horizon * int(any(ARMS[arm])), 'Inactive component skip contract failed')
                    terminal_seconds += diagnostic['head_seconds']
                    predictions = policies(arrays['trajectory_logits'], terminal=3)
                    for policy in POLICIES:
                        outcomes = predictions[policy] == arrays['label']
                        current[policy].append(outcomes)
                        row, = [row for row in report['per_corruption'] if row['corruption'] == corruption and row['policy'] == policy]
                        require(int(outcomes.sum()) == row['correct'], 'Independent policy/count recomputation mismatch')
                for policy in POLICIES:
                    correct[(dataset, head, arm, policy, seed)] = np.stack(current[policy])
                require(abs(config['shared_reconstruction_seconds'] - sum(first_timings.values())) < 1e-6, 'Shared reconstruction timing disagreement')
                macro.extend(dict(row, seed=seed) for row in report['macro'])
                corruption_rows.extend(dict(row, seed=seed) for row in report['per_corruption'])
                for policy in POLICIES:
                    rows = [row for row in report['per_corruption'] if row['policy'] == policy and row['corruption'] != 'background']
                    macro14.append(dict(dataset=dataset, component_arm=arm, backbone=head, policy=policy, seed=seed,
                        corruptions=14, n=980, accuracy_pct=float(np.mean([row['accuracy_pct'] for row in rows])),
                        delta_terminal_pp=float(np.mean([row['delta_terminal_pp'] for row in rows])),
                        corrected=sum(row['corrected'] for row in rows), harmed=sum(row['harmed'] for row in rows)))
                target = repository / 'result' / dataset / METHOD / archive.name
                raw.append(dict(dataset=dataset, backbone=head, component_arm=arm, seed=seed, run_id=config['run_id'],
                                source_path=str(archive), canonical_path=str(target), drive_path=reference['drive_path'], **identity(archive)))
            finally:
                bundle.close()
        require(heads_seen == set(HEADS), 'Missing/duplicate head within job')
        costs.append(dict(dataset=dataset, component_arm=arm, seed=seed, shared_batches=15, n_input_clouds=1050, n_head_input_events=4200,
            shared_reconstruction_seconds=sum(first_timings.values()), terminal_all_heads_seconds=terminal_seconds,
            remote_job_plus_validation_seconds=job['elapsed_seconds'],
            whole_process_peak_gpu_memory_mb=peak,
            fresh_encode_batches=sum(mode == 'fresh_encode' for mode in preparation_modes.values()),
            cached_replay_batches=sum(mode == 'cached_replay' for mode in preparation_modes.values()),
            timing_scope='paired-state confirmation; cache IO included; replay omits native encode; not deployment latency'))
    require(len(raw) == 48 and len(macro) == len(macro14) == 144 and len(corruption_rows) == 2160 and len(correct) == 144, 'Wrong frozen-confirmation event/readout counts')
    for dataset in DATASETS:
        for seed in SEEDS:
            for corruption in CORRUPTIONS:
                reference = shared[(dataset, seed, 'combined', corruption)]['trace']
                for arm in ARMS:
                    same_preparation(reference, shared[(dataset, seed, arm, corruption)]['trace'])
        reference = case_payloads[(dataset, CORRUPTIONS[0])]
        for corruption in CORRUPTIONS:
            aligned(reference, case_payloads[(dataset, corruption)])
    balanced = balanced_table(macro)
    balanced14 = balanced_table(macro14)
    paired_cases, paired_balanced = paired_tables(correct, list(CORRUPTIONS))
    output = prep / 'local_frozen_confirmation_analysis'
    require(not output.exists() and all(not Path(row['canonical_path']).exists() for row in raw), 'Immutable output/canonical archive collision; inspect before retry')
    output.mkdir()
    for row in raw:
        target = Path(row['canonical_path']); target.parent.mkdir(parents=True, exist_ok=True)
        with Path(row['source_path']).open('rb') as incoming, target.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        verify(target, row)
    for name, rows in [('balanced', balanced), ('case_policy', macro), ('per_corruption', corruption_rows),
                       ('macro14_balanced', balanced14), ('macro14_case_policy', macro14),
                       ('paired_case_contrasts', paired_cases), ('paired_balanced_contrasts', paired_balanced), ('cost', costs)]:
        write_csv(output / (name + '.csv'), rows)
    validation = dict(status='complete_frozen_confirmation_subsets', full_benchmark=False, seeds=list(SEEDS), frozen_plan_sha256=PLAN_SHA,
        manifest_sha256=MANIFEST_SHA, source_snapshot_sha256=SOURCE_SHA, phase_archive=phase_identity,
        raw_archives=raw, head_inputs=50400, candidate_predictions=201600, shared_batches=180,
        balanced_policy_rows=24, case_policy_rows=144, per_corruption_rows=2160,
        paired_preparation_verified=True, aligned_original_ids_labels=True, cache_receipt_count=len(cache_records),
        cache_evidence='Raw metadata joined to phase-archived remote-verified Drive cache receipts; local latent-cache bytes not required or rehashed here',
        classifier_contracts=[dict(dataset=dataset, backbone=head, **value) for (dataset, head), value in sorted(head_states.items())],
        frozen_candidate=plan['candidate'], frozen_comparators=plan['comparators'], no_policy_reselection=True,
        balanced=balanced, macro14_balanced=balanced14, paired_balanced=paired_balanced, cost=costs,
        uncertainty=dict(type='conditional dataset-stratified original-object-ID cluster bootstrap', seed=20261025, replicates=2000,
            n_original_object_clusters=140, preserved='same70 original-ID draws across all15 corruptions, four heads, two arms/policies AND all three seeds within each dataset; independent dataset draws',
            limitations='Conditional on these three seeds/object subsets; repeated seeds do not create210 independent objects. No equivalence/noninferiority or broad seed-stability inference'),
        limitations=['Object-disjoint from development but benchmarks previously inspected;70 IDs per dataset, not full2468/581 evaluation',
                    'No tuning, candidate selection, logit averaging or new GPU execution performed',
                    'All15 is primary; macro14 withoutBackground is diagnostic and does not define inference routing',
                    'Paired-cache replay times are not measured deployment latency; shared cost counted once across heads'])
    (output / 'validation.json').write_text(json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(status=validation['status'], output=str(output), raw_archives=len(raw), balanced=balanced, paired_balanced=paired_balanced), indent=2))


def self_check():
    original = dict(index=np.array([3, 9]), label=np.array([0, 1]))
    aligned(original, original)
    for changed in (dict(original, index=np.array([9, 3])), dict(original, label=np.array([1, 0]))):
        try:
            aligned(original, changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Misaligned original IDs/labels accepted')
    same_preparation(dict(initial_hashes={'noise': 'same'}, paired_state={'sha256': 'same'}, geometry='left'),
                     dict(initial_hashes={'noise': 'same'}, paired_state={'sha256': 'same'}, geometry='right'))
    for observed, expected in [(True, 1), (0., 0), ('0', 0), (3, 3), (1, 0)]:
        try:
            valid_seed(observed, expected)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid/mismatched seed accepted')
    for seed in SEEDS:
        valid_seed(seed, seed)
    rows = [dict(dataset=dataset, backbone=head, component_arm=arm, policy=policy, seed=seed, delta_terminal_pp=0.,
                 accuracy_pct=10. if dataset == DATASETS[0] else 90.)
            for dataset in DATASETS for head in HEADS for arm in ARMS for policy in POLICIES for seed in SEEDS]
    require(all(row['accuracy_pct'] == 50. for row in balanced_table(rows)), 'Dataset/head balance is incorrect')
    names = ['background'] + ['corruption%d' % value for value in range(14)]
    outcomes = {(dataset, head, arm, policy, seed): np.zeros((15, 70), dtype=bool)
                for dataset in DATASETS for head in HEADS for arm in ARMS for policy in POLICIES for seed in SEEDS}
    for dataset in DATASETS:
        for head in HEADS:
            for seed in SEEDS:
                outcomes[(dataset, head, 'chamfer', 'probability_mean', seed)][:, :7] = True
    cases, balanced = paired_tables(outcomes, names, bootstrap_replicates=2000)
    selected = [row for row in balanced if row['scope'] == 'all15' and row['contrast'] == 'primary_same_readout']
    mean = next(row for row in selected if row['aggregation'] == 'three_seed_mean')
    require(abs(mean['delta_pp'] - 10.) < 1e-12 and mean['corrected_events'] == 2520 and mean['harmed_events'] == 0
            and mean['positive_cases'] == 8 and len(cases) == 96 and len(balanced) == 16, 'Frozen paired-event aggregation failed')
    for row in selected:
        for key in ('conditional_object_bootstrap_low_pp', 'conditional_object_bootstrap_high_pp'):
            require(abs(row[key] - mean[key]) < 1e-12, 'Repeating the same seeds falsely narrowed object uncertainty')
    # Opposite seed effects on the same objects cancel under shared ID draws.
    for dataset in DATASETS:
        for head in HEADS:
            outcomes[(dataset, head, 'chamfer', 'probability_mean', 1)][:] = False
            outcomes[(dataset, head, 'combined', 'probability_mean', 1)][:, :7] = True
            outcomes[(dataset, head, 'chamfer', 'probability_mean', 2)][:] = False
    _, cancelled = paired_tables(outcomes, names, bootstrap_replicates=200)
    row = next(row for row in cancelled if row['scope'] == 'all15' and row['contrast'] == 'primary_same_readout' and row['seed'] is None)
    require(abs(row['delta_pp']) < 1e-12 and abs(row['conditional_object_bootstrap_low_pp']) < 1e-12
            and abs(row['conditional_object_bootstrap_high_pp']) < 1e-12, 'Seed clustering lost paired cancellation')
    for value in outcomes.values():
        value[:] = False
    for dataset in DATASETS:
        for head in HEADS:
            for seed in SEEDS:
                outcomes[(dataset, head, 'chamfer', 'probability_mean', seed)][0] = True
    _, background = paired_tables(outcomes, names, bootstrap_replicates=200)
    require(all(abs(row['delta_pp']) < 1e-12 for row in background if row['scope'] == 'macro14'), 'Background leaked into macro14')
    json.dumps(dict(cases=cases, balanced=balanced), allow_nan=False)
    print('AGENT_B_FROZEN_CONFIRMATION_ANALYSIS_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--prep', type=Path, default=Path.cwd())
        main(parser.parse_args().prep)
