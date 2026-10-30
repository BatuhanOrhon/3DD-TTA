"""Independent Agent B fixed-screen ingestion; CPU only, no tuning or new runs."""
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
REMOTE = PurePosixPath('/content/studies/guidance-components/development_20261025T133358718403Z')
OWN_DRIVE = '/content/drive/MyDrive/thesis/result/guidance_component_study_20261025/'
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
HEADS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')
ARMS = {'combined': (1., 1.17), 'chamfer': (1., 0.), 'spectral': (0., 1.17), 'unguided': (0., 0.)}
POLICIES = ('terminal', 'minimum_entropy', 'probability_mean')
CONTRASTS = (('combined', 'chamfer'), ('spectral', 'unguided'), ('chamfer', 'unguided'),
             ('combined', 'spectral'), ('combined', 'unguided'))
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


def balanced_table(rows):
    result = []
    for arm in ARMS:
        for policy in POLICIES:
            group = [row for row in rows if row['component_arm'] == arm and row['policy'] == policy]
            require(len(group) == 8 and {(r['dataset'], r['backbone']) for r in group} ==
                    {(dataset, head) for dataset in DATASETS for head in HEADS}, 'Balanced mean requires eight unique cases')
            result.append(dict(component_arm=arm, policy=policy, n_cases=8,
                               accuracy_pct=sum(row['accuracy_pct'] for row in group) / 8))
    return result


def paired_tables(correct, bootstrap_replicates=2000):
    """Resample original IDs jointly across every corruption, head and arm."""
    rng = np.random.default_rng(20261025)
    draws = {dataset: rng.integers(0, 70, size=(bootstrap_replicates, 70)) for dataset in DATASETS}
    cases, balanced = [], []
    for candidate, comparator in CONTRASTS:
        for policy in POLICIES:
            effects, selected = {}, []
            for dataset in DATASETS:
                per_object = []
                for head in HEADS:
                    left = correct[(dataset, head, candidate, policy)]
                    right = correct[(dataset, head, comparator, policy)]
                    require(left.shape == right.shape == (15, 70) and left.dtype == right.dtype == bool,
                            'Expected aligned15-corruption×70-original-ID boolean outcomes')
                    difference = left.astype(float) - right.astype(float)
                    per_object.append(difference.mean(axis=0))
                    delta = 100. * float(difference.mean())
                    row = dict(dataset=dataset, backbone=head, candidate=candidate, comparator=comparator, policy=policy,
                               n_predictions=1050, delta_pp=delta, corrected=int((left & ~right).sum()),
                               harmed=int((~left & right).sum()), sign='positive' if delta > 0 else 'negative' if delta < 0 else 'tied')
                    cases.append(row); selected.append(row)
                effects[dataset] = np.mean(per_object, axis=0)
            sampled = 100. * np.mean([effects[dataset][draws[dataset]].mean(axis=1) for dataset in DATASETS], axis=0)
            low, high = np.percentile(sampled, [2.5, 97.5])
            balanced.append(dict(candidate=candidate, comparator=comparator, policy=policy, n_cases=8, n_predictions=8400,
                delta_pp=sum(row['delta_pp'] for row in selected) / 8,
                corrected=sum(row['corrected'] for row in selected), harmed=sum(row['harmed'] for row in selected),
                positive_cases=sum(row['sign'] == 'positive' for row in selected),
                negative_cases=sum(row['sign'] == 'negative' for row in selected),
                tied_cases=sum(row['sign'] == 'tied' for row in selected),
                conditional_object_bootstrap_low_pp=float(low), conditional_object_bootstrap_high_pp=float(high)))
    return cases, balanced


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main(prep):
    prep = Path(prep).resolve()
    repository = prep.parents[2]
    require(repository.name == 'guidance-components', 'Use the isolated Agent B preparation directory')
    extracted = prep / 'development_extracted'
    receipt = json.loads((prep / 'development_phase_receipt.json').read_text())
    phase_identity = verify(prep / 'development_phase_evidence_drive_roundtrip.zip', receipt['archive'])
    state = receipt['state']
    require(state['status'] == 'complete' and state['mode'] == 'development' and state['single_seed_exploratory']
            and state['full_benchmark'] is False and state['manifest_sha256'] == MANIFEST_SHA
            and state['source_snapshot_sha256'] == SOURCE_SHA and state['session'] == 'tta-guidance-components-a100-20261025-r1'
            and receipt['archive']['drive_path'].startswith(OWN_DRIVE), 'Development phase/source status mismatch')
    jobs = state['jobs']
    require(len(jobs) == 8 and {(job['dataset'], job['arm']) for job in jobs} ==
            {(dataset, arm) for dataset in DATASETS for arm in ARMS}, 'Missing/duplicate dataset/arm jobs')
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
        name = 'study/' + dataset + '_development_ids.json'
        verify(assets / name, manifest['files'][name])
    selected = {dataset: json.loads((assets / 'study' / (dataset + '_development_ids.json')).read_text()) for dataset in DATASETS}
    cache_records = {}
    for job in jobs:
        require(job['status'] == 'complete' and len(job['raw_archives']) == 4, 'Incomplete arm job')
        for cached in job['cache']:
            require(cached['drive_path'].startswith(OWN_DRIVE) and len(cached['sha256']) == 64 and cached['bytes'] > 0,
                    'Missing remote-verified cache receipt identity')
            key = cached['local_path']
            if key in cache_records:
                require(cache_records[key] == cached, 'Cache receipt identity changed across jobs')
            cache_records[key] = cached
    require(len(cache_records) == 30, 'Require one shared cached state/noise file per dataset/corruption')
    raw, macro, corruption_rows, correct, shared, head_states, costs = [], [], [], {}, {}, {}, []
    case_payloads = {}
    lion_state = None
    for job in jobs:
        dataset, arm = job['dataset'], job['arm']
        expected_ids = selected[dataset]['indices']
        require(selected[dataset]['stage'] == 'development' and len(expected_ids) == 70, 'Expected70 registered development IDs')
        heads_seen, first_timings, terminal_seconds, preparation_modes, peak = set(), {}, 0., {}, 0.
        for reference in job['raw_archives']:
            relative = PurePosixPath(reference['local_path']).relative_to(REMOTE)
            archive = extracted.joinpath(*relative.parts)
            require(archive.suffix == '.zip' and archive.is_file() and reference['drive_path'].startswith(OWN_DRIVE), 'Missing/non-owned raw archive')
            verify(archive, reference)
            report = analyze_bundle(archive, allow_partial=True)
            require(report['status'] == 'partial' and report['selection_complete'] and len(report['per_corruption']) == 45
                    and all(row['n'] == 70 for row in report['per_corruption']), 'Invalid all15×70 trajectory readouts')
            bundle = Bundle(archive)
            try:
                config = bundle.json('config.json')
                head = config['backbone']; heads_seen.add(head)
                require(archive.name == config['run_id'] + '.zip' and head == reference['backbone'], 'Raw run/head identity mismatch')
                require(config['dataset'] == dataset and config['component_arm'] == arm and config['status'] == 'partial'
                        and config['component_study'] is True and config['seed'] == 0 and config['severity'] == 5
                        and config['asset_manifest_sha256'] == MANIFEST_SHA and config['asset_manifest'] == manifest
                        and config['selected_original_ids'] == expected_ids and config['selection_manifest'] == selected[dataset]
                        and set(config['corruptions']) == set(CORRUPTIONS), 'Raw config/source/selection mismatch')
                require(config['selection_manifest']['stage'] == 'development' and not config['component_parameters_modified'], 'Tuned/non-development arm in fixed matrix')
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
                    require(cached is not None and {key: paired[key] for key in ('bytes', 'sha256')} ==
                            {key: cached[key] for key in ('bytes', 'sha256')}, 'Paired cache lacks matching remote-verified receipt')
                    trace = dict(initial_hashes=diagnostic['trajectory']['initial_hashes'], paired_state=paired)
                    group = (dataset, arm, corruption)
                    if group in shared:
                        require(shared[group] == dict(trace=trace, geometry=diagnostic['geometry_sha256'], seconds=diagnostic['shared_reconstruction_seconds']),
                                'Heads observed different supposedly shared reconstruction')
                    else:
                        shared[group] = dict(trace=trace, geometry=diagnostic['geometry_sha256'], seconds=diagnostic['shared_reconstruction_seconds'])
                        first_timings[corruption] = diagnostic['shared_reconstruction_seconds']
                        preparation_modes[corruption] = diagnostic['preparation_mode']
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
                    correct[(dataset, head, arm, policy)] = np.stack(current[policy])
                require(abs(config['shared_reconstruction_seconds'] - sum(first_timings.values())) < 1e-6, 'Shared reconstruction timing disagreement')
                macro.extend(report['macro']); corruption_rows.extend(report['per_corruption'])
                target = repository / 'result' / dataset / METHOD / archive.name
                raw.append(dict(dataset=dataset, backbone=head, component_arm=arm, run_id=config['run_id'],
                                source_path=str(archive), canonical_path=str(target), drive_path=reference['drive_path'], **identity(archive)))
            finally:
                bundle.close()
        require(heads_seen == set(HEADS), 'Missing/duplicate head within job')
        costs.append(dict(dataset=dataset, component_arm=arm, shared_batches=15, n_input_clouds=1050, n_head_input_events=4200,
            shared_reconstruction_seconds=sum(first_timings.values()), terminal_all_heads_seconds=terminal_seconds,
            remote_job_plus_validation_seconds=job['elapsed_seconds'],
            whole_process_peak_gpu_memory_mb=peak,
            fresh_encode_batches=sum(mode == 'fresh_encode' for mode in preparation_modes.values()),
            cached_replay_batches=sum(mode == 'cached_replay' for mode in preparation_modes.values()),
            timing_scope='paired-state development; cache IO included; replay omits native encode; not deployment latency'))
    require(len(raw) == 32 and len(macro) == 96 and len(corruption_rows) == 1440 and len(correct) == 96, 'Wrong fixed-screen event/readout counts')
    for dataset in DATASETS:
        for corruption in CORRUPTIONS:
            reference = shared[(dataset, 'combined', corruption)]['trace']
            for arm in ARMS:
                same_preparation(reference, shared[(dataset, arm, corruption)]['trace'])
        reference = case_payloads[(dataset, CORRUPTIONS[0])]
        for corruption in CORRUPTIONS:
            aligned(reference, case_payloads[(dataset, corruption)])
    balanced = balanced_table(macro)
    paired_cases, paired_balanced = paired_tables(correct)
    output = prep / 'local_development_analysis'
    require(not output.exists() and all(not Path(row['canonical_path']).exists() for row in raw), 'Immutable output/canonical archive collision; inspect before retry')
    output.mkdir()
    for row in raw:
        target = Path(row['canonical_path']); target.parent.mkdir(parents=True, exist_ok=True)
        with Path(row['source_path']).open('rb') as incoming, target.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        verify(target, row)
    for name, rows in [('balanced', balanced), ('case_policy', macro), ('per_corruption', corruption_rows),
                       ('paired_case_contrasts', paired_cases), ('paired_balanced_contrasts', paired_balanced), ('cost', costs)]:
        write_csv(output / (name + '.csv'), rows)
    validation = dict(status='complete_fixed_development_subsets', full_benchmark=False, seed=0,
        manifest_sha256=MANIFEST_SHA, source_snapshot_sha256=SOURCE_SHA, phase_archive=phase_identity,
        raw_archives=raw, head_inputs=33600, candidate_predictions=134400, shared_batches=120,
        balanced_policy_rows=12, case_policy_rows=96, per_corruption_rows=1440,
        paired_preparation_verified=True, aligned_original_ids_labels=True, cache_receipt_count=len(cache_records),
        cache_evidence='Raw metadata joined to phase-archived remote-verified Drive cache receipts; local latent-cache bytes not required or rehashed here',
        classifier_contracts=[dict(dataset=dataset, backbone=head, **value) for (dataset, head), value in sorted(head_states.items())],
        balanced=balanced, paired_balanced=paired_balanced, cost=costs,
        uncertainty=dict(type='conditional dataset-stratified original-object-ID cluster bootstrap', seed=20261025, replicates=2000,
            preserved='same resampled original IDs across all15 corruptions, four heads and four arms within each dataset',
            limitations='Conditional seed0 development uncertainty; no seed stability, independent confirmation or multiplicity-adjusted inference'),
        limitations=['Previously benchmark-exposed development subsets;70 IDs per dataset, not full2468/581 evaluation',
                    'No tuning, candidate selection, logit averaging or new GPU execution performed',
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
    rows = [dict(dataset=dataset, backbone=head, component_arm=arm, policy=policy,
                 accuracy_pct=10. if dataset == DATASETS[0] else 90.)
            for dataset in DATASETS for head in HEADS for arm in ARMS for policy in POLICIES]
    require(all(row['accuracy_pct'] == 50. for row in balanced_table(rows)), 'Dataset/head balance is incorrect')
    outcomes = {(dataset, head, arm, policy): np.zeros((15, 70), dtype=bool)
                for dataset in DATASETS for head in HEADS for arm in ARMS for policy in POLICIES}
    for dataset in DATASETS:
        for head in HEADS:
            for policy in POLICIES:
                outcomes[(dataset, head, 'combined', policy)][:, :7] = True
    cases, balanced = paired_tables(outcomes, bootstrap_replicates=200)
    contrast = next(row for row in balanced if row['candidate'] == 'combined' and row['comparator'] == 'chamfer' and row['policy'] == 'terminal')
    require(abs(contrast['delta_pp'] - 10.) < 1e-12 and contrast['corrected'] == 840 and contrast['harmed'] == 0
            and contrast['positive_cases'] == 8 and len(cases) == 120, 'Paired corrected/harmed aggregation failed')
    print('AGENT_B_DEVELOPMENT_ANALYSIS_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--prep', type=Path, default=Path.cwd())
        main(parser.parse_args().prep)
