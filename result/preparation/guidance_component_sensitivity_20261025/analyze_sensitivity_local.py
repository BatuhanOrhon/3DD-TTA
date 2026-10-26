"""Independent CPU evidence validation and frozen sensitivity-screen selection.

No runtime access. Acceptance requires every declared cache file locally verified.
Old baseline archives retain their original run/source identities.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

import numpy as np

PLAN_SHA = '4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1'
REMOTE = PurePosixPath('/content/studies/guidance-components')
DRIVE = '/content/drive/MyDrive/thesis/result/guidance_component_study_20261025/'
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
HEADS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')
POLICIES = ('terminal', 'minimum_entropy', 'probability_mean')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def verify(path, expected):
    actual = identity(path)
    require(actual == {key: expected[key] for key in ('bytes', 'sha256')}, 'File identity mismatch: ' + str(path))
    return actual


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def checked_zip(path):
    archive = zipfile.ZipFile(path)
    try:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)) and len(names) <= 50000, 'Duplicate/excessive ZIP members')
        require(sum(entry.file_size for entry in entries) <= 16 * 1024**3, 'Excessive ZIP expansion')
        for entry in entries:
            name = PurePosixPath(entry.filename)
            require(name.parts and not name.is_absolute() and '..' not in name.parts
                    and name.as_posix() == entry.filename.rstrip('/')
                    and not any(c in entry.filename for c in ('\\', ':', '\x00')),
                    'Unsafe ZIP path')
            require(stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR), 'ZIP symlink/special member')
        require(archive.testzip() is None, 'ZIP CRC mismatch')
        return archive
    except BaseException:
        archive.close()
        raise


def unpack_phase(archive_path, receipt_path, stage, plan, output):
    receipt = read_json(receipt_path)
    verify(archive_path, receipt['archive'])
    state = receipt['state']
    require(state['status'] == 'complete' and state['stage'] == stage and state['full_benchmark'] is False
            and state['session'] == plan['session'] and state['phase_plan_sha256'] == PLAN_SHA
            and state['manifest_sha256'] == plan['manifest_sha256']
            and state['source_snapshot_sha256'] == plan['source_snapshot_sha256']
            and receipt['archive']['drive_path'].startswith(DRIVE), 'Phase status/source/ownership mismatch')
    with checked_zip(archive_path) as archive:
        names = {n for n in archive.namelist() if not n.endswith('/')}
        declared = json.loads(archive.read('evidence_manifest.json'))
        require(set(declared) == names - {'evidence_manifest.json'}, 'Phase manifest coverage mismatch')
        for name, expected in declared.items():
            data = archive.read(name)
            require(dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) == expected, 'Phase member identity mismatch')
        require(json.loads(archive.read('state.json')) == state, 'Receipt and archived phase state differ')
        require(hashlib.sha256(archive.read('campaign_plan.json')).hexdigest() == PLAN_SHA, 'Archived campaign plan changed')
        for name, expected in plan['phase_sources'].items():
            data = archive.read(name)
            require(dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) == expected, 'Archived phase source changed')
        approval = json.loads(archive.read('gpu_approval.json'))
        require(approval['status'] == 'approved' and approval['phase_plan_sha256'] == PLAN_SHA
                and approval['session'] == plan['session'] and approval['cap_seconds'] == 19800
                and approval['user_reply'].strip(), 'Missing exact campaign approval')
        ledger = json.loads(archive.read('ledger_snapshot.json'))
        require(ledger['phase_plan_sha256'] == PLAN_SHA and ledger['cap_seconds'] == 19800
                and ledger['session'] == plan['session'] and ledger['active'] is None
                and math.isfinite(ledger['spent_seconds']) and 0 <= ledger['spent_seconds'] <= 19800,
                'Invalid cumulative campaign ledger')
        archive.extractall(output)
    return receipt


def load_context(prep):
    plan_path = prep / 'campaign_plan.json'
    require(identity(plan_path)['sha256'] == PLAN_SHA, 'Unapproved campaign plan')
    plan = read_json(plan_path)
    for name, expected in plan['phase_sources'].items():
        verify(prep / name, expected)
    controller = load_module('sensitivity_controller_local', prep / 'sensitivity_campaign_controller.py')
    calibration = load_module('sensitivity_calibration_local', prep / 'calibrate_spectral_losses.py')
    assets = prep / 'assets_v2'
    require(identity(assets / 'manifest.json')['sha256'] == plan['manifest_sha256']
            and identity(assets / 'source_snapshot.zip')['sha256'] == plan['source_snapshot_sha256'], 'Local sealed assets changed')
    manifest = read_json(assets / 'manifest.json')
    for name in ('scripts/analyze_scan_trajectory.py', 'scripts/analyze_legacy_trajectory.py', 'research_artifacts.py'):
        verify(assets / 'source_snapshot' / name, manifest['files']['source_snapshot/' + name])
    sys.path.insert(0, str(assets / 'source_snapshot'))
    from scripts.analyze_scan_trajectory import Bundle, analyze_bundle
    from scripts.analyze_legacy_trajectory import policies
    from research_artifacts import CORRUPTIONS
    require(len(CORRUPTIONS) == 15, 'Unexpected corruption ordering')
    verify(prep / 'trial_registry.json', plan['trial_registry'])
    registry = controller.registry_verified(read_json(prep / 'trial_registry.json'))
    selections = {}
    for stage in ('smoke', 'development', 'confirmation'):
        for dataset in DATASETS:
            ref = plan['id_manifests'][stage][dataset]
            path = assets / PurePosixPath(ref['path']).relative_to(PurePosixPath(plan['asset_root']))
            verify(path, ref)
            selected = read_json(path)
            ids = selected['indices']
            require(selected['dataset'] == dataset and len(ids) == (140 if stage == 'confirmation' else 70)
                    and ids == sorted(set(ids)) and all(type(i) is int for i in ids), 'Selection manifest mismatch')
            selections[(stage, dataset)] = selected
    return plan, controller, calibration, assets, manifest, registry, selections, Bundle, analyze_bundle, policies, tuple(CORRUPTIONS)


def calibration_report(directory, receipt, context):
    plan, _, helper, _, _, _, selections, *_ = context
    report = read_json(directory / 'calibration/calibration_report.json')
    require(report == receipt['state']['calibration'] and report['status'] == 'verified_label_free_gpu'
            and report['manifest_sha256'] == plan['manifest_sha256']
            and report['source_snapshot_sha256'] == plan['source_snapshot_sha256']
            and report['labels_used'] is False and report['classifier_forwards'] == 0 and report['seed'] == 0
            and report['n_input_events'] == 2100 and report['completed_batches'] == 30
            and report['lion_state_before'] == report['lion_state_after'] and report['legacy_inline_parity'] is True,
            'Label-free actual-prior/DDIM calibration proof incomplete')
    manifest, assets = context[4], context[3]
    for name, expected in report['executed_source_files'].items():
        relative = PurePosixPath(name).relative_to('assets')
        require(manifest['files'][str(relative)] == expected, 'Calibration executed source is unsealed')
    for name in ('tta_gsd_v2.py', 'spectral_guidance_losses.py', 'scripts/eval_multibackbone_guidance.py', 'graph_spectral_v2.py'):
        require(report['executed_source_files']['assets/source_snapshot/' + name] == manifest['files']['source_snapshot/' + name], 'Calibration source binding missing')
    audit_name = manifest['guidance']['native_audit']
    verify(assets / audit_name, manifest['files'][audit_name])
    audit = read_json(assets / audit_name)
    require(report['native']['no_build'] is True and set(report['native']['bindings']) == set(audit['backends']), 'Calibration native proof changed')
    for name, entries in audit['backends'].items():
        entry, = entries
        require(report['native']['bindings'][name] == {k: entry[k] for k in ('path', 'bytes', 'sha256')}, 'Calibration native binary pin changed')
    norms_path = directory / 'calibration/per_object_norms.jsonl'
    verify(norms_path, report['per_object_norms'])
    rows = [json.loads(line) for line in norms_path.read_text().splitlines()]
    require({(r['dataset'], r['corruption'], r['index']) for r in rows} ==
            {(d, c, i) for d in DATASETS for c in context[-1] for i in selections[('development', d)]['indices']},
            'Calibration ID/corruption coverage mismatch')
    require(all(r['horizon'] == (30 if r['corruption'] == 'background' else 10)
                and r['timestep'] == (290 if r['corruption'] == 'background' else 90) for r in rows), 'Calibration actual first-step timestep drift')
    diagnostics = helper.coefficient_report(rows)
    require(diagnostics == report['coefficient_diagnostics'] and
            {kind: value['coefficient'] for kind, value in diagnostics.items()} == report['coefficients'], 'Recomputed calibration differs')
    parity = report['per_batch_parity']
    require(len(parity) == 30 and {(r['dataset'], r['corruption']) for r in parity} ==
            {(d, c) for d in DATASETS for c in context[-1]}, 'Calibration parity coverage mismatch')
    for row in parity:
        require(row['scalar_exact'] is True and row['direct_coefficient_gradient_exact'] is True
                and row['rng_before_sha256'] == row['rng_after_sha256']
                and row['horizon'] == (30 if row['corruption'] == 'background' else 10), 'Calibration scalar/gradient/RNG guard failed')
        require(set(row['native_gradient_differences']) == {'local', 'style'}, 'Missing gradient blocks')
        for block in row['native_gradient_differences'].values():
            helper.validate_native_parity(block['helper_vs_inline_relative_l2'], block['inline_repeat_relative_l2'])
    records = report['cache_records']
    require(len(records) == 30 and len({r['local_path'] for r in records}) == 30, 'Calibration cache coverage mismatch')
    return report, records


def physical_caches(records, roots):
    result = []
    for record in records:
        remote = PurePosixPath(record['local_path'])
        require(remote.is_relative_to(REMOTE) if hasattr(remote, 'is_relative_to') else str(remote).startswith(str(REMOTE) + '/'), 'Foreign cache path')
        require(record['drive_path'].startswith(DRIVE) and record['bytes'] > 0 and len(record['sha256']) == 64, 'Invalid remote cache receipt')
        relative = remote.relative_to(REMOTE)
        candidates = []
        for root in roots:
            candidates.extend((Path(root) / relative, Path(root).joinpath(*relative.parts[1:])))
        found = [p for p in candidates if p.is_file()]
        for path in found:
            verify(path, record)
            with np.load(path, allow_pickle=False) as values:
                for key in values.files:
                    value = values[key]
                    require(value.dtype.kind != 'O', 'Object array in paired cache')
                    if value.dtype.kind in 'fc':
                        require(np.isfinite(value).all(), 'Nonfinite paired cache tensor')
        result.append(dict(remote_path=str(remote), bytes=record['bytes'], sha256=record['sha256'],
                           local_verified=bool(found), local_paths=[str(p) for p in found]))
    return result


def aligned(reference, data):
    require(np.array_equal(reference['index'], data['index']) and np.array_equal(reference['label'], data['label']), 'Original-ID/label misalignment')


def baseline_labels(prep, context):
    """Use unchanged accepted label shards when a full local dataset is absent."""
    manifest, assets, Bundle = context[4], context[3], context[7]
    path = prep / 'reused_baselines.json'
    verify(path, manifest['files']['study/reused_baselines.json'])
    old = read_json(path)
    require(old['status'] == 'accepted_old_source_unchanged' and len(old['raw_archives']) == 24, 'Missing accepted historical label references')
    result, references = {}, []
    for dataset in DATASETS:
        raw = next(r for r in old['raw_archives'] if r['dataset'] == dataset)
        path = Path(raw['canonical_path']); verify(path, raw)
        with checked_zip(path):
            pass
        bundle = Bundle(path)
        try:
            require(bundle.json('config.json')['asset_manifest_sha256'] == old['original_manifest_sha256'], 'Historical label source relabelled')
            with np.load(io.BytesIO(bundle.read('gaussian_predictions.npz')), allow_pickle=False) as values:
                result[dataset] = {key: values[key].copy() for key in ('index', 'label')}
        finally:
            bundle.close()
        label_path = assets / 'datasets' / dataset / 'label.npy'
        if label_path.exists():
            verify(label_path, manifest['datasets'][dataset]['files']['label.npy'])
            full = np.load(label_path, allow_pickle=False).reshape(-1)
            require(np.array_equal(full[result[dataset]['index']], result[dataset]['label']), 'Historical labels differ from sealed dataset labels')
        references.append(dict(dataset=dataset, historical_archive=str(path), archive_identity=identity(path),
                               full_sealed_label_file_verified=label_path.exists()))
    return result, references


def same_preparation(reference, trace):
    require(reference['initial_hashes'] == trace['initial_hashes'] and reference['paired_state'] == trace['paired_state'], 'Unpaired initial tensors/noise/cache')


def balanced_scores(rows, trial_ids):
    result = []
    for trial in trial_ids:
        selected = [r for r in rows if r['trial_id'] == trial and r['policy'] == ('terminal' if r['family'] == 'original3dd' else 'probability_mean')]
        require(len(selected) == 8 and {(r['dataset'], r['backbone']) for r in selected} ==
                {(d, h) for d in DATASETS for h in HEADS}, 'Balanced scoring requires eight unique dataset/head cases')
        result.append(dict(trial_id=trial, family=selected[0]['family'], policy=selected[0]['policy'],
                           n_cases=8, accuracy_pct=sum(r['accuracy_pct'] for r in selected) / 8))
    return result


def freeze_selection(scores, registry, accepted_identity, controller, calibration):
    # max() retains the first registry item on exact ties; scores must be registry ordered.
    spectral = max((r for r in scores if r['family'] in ('combined', 'spectral')), key=lambda r: r['accuracy_pct'])['trial_id']
    chamfer = max((r for r in scores if r['family'] == 'chamfer'), key=lambda r: r['accuracy_pct'])['trial_id']
    candidate, c = registry[spectral], registry[chamfer]
    matched = copy.deepcopy(registry['c-base'])
    matched.update(id='matched-chamfer', reuse=False, eta=candidate['eta'], gamma=candidate['gamma'], p=candidate['p'])
    conditions = [dict(id='spectral-candidate', role='spectral_candidate', trial_id=spectral),
                  dict(id='matched-chamfer', role='matched_chamfer', trial_id='matched-chamfer')]
    if controller.protocol_key(matched, calibration) != controller.protocol_key(c, calibration):
        conditions.append(dict(id='tuned-chamfer', role='tuned_chamfer', trial_id=chamfer))
    conditions.append(dict(id='original3dd', role='original3dd', trial_id='original3dd'))
    return dict(status='frozen', phase_plan_sha256=PLAN_SHA, development_acceptance=accepted_identity,
                selected_spectral_trial_id=spectral, selected_chamfer_trial_id=chamfer, conditions=conditions)


def copy_canonical(path, repository, config):
    target = repository / 'result' / config['dataset'] / config['method'] / (config['run_id'] + '.zip')
    expected = identity(path)
    if target.exists():
        verify(target, expected)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream, Path(path).open('rb') as source:
            shutil.copyfileobj(source, stream)
        verify(target, expected)
    return str(target)


def inspect_raw(path, job, stage, context, calibration, old, alignment, paired, head_states, lion_states, old_assets=None):
    plan, controller, _, assets, manifest, _, selections, Bundle, analyze_bundle, policies, all_corruptions = context
    dataset, trial = job['dataset'], job['trial']
    original = trial['method'] == 'original3dd'
    corruptions = ('gaussian', 'background') if stage == 'smoke' else all_corruptions
    selected = selections[('development' if old else stage, dataset)]
    expected_ids = np.asarray(selected['indices'], dtype=np.int64)
    with checked_zip(path):
        pass
    bundle = Bundle(path)
    try:
        require({'command.txt', 'config.json', 'environment.txt', 'stdout.log', 'summary.csv', 'per_corruption.csv', 'notes.md'} <= bundle.names, 'Raw required files missing')
        config = bundle.json('config.json')
        head = config['backbone']
        require(head in HEADS and config['dataset'] == dataset and config['seed'] == 0 and type(config['seed']) is int
                and config['selected_original_ids'] == expected_ids.tolist() and config['status'] == 'partial'
                and config['num_classes'] == (40 if dataset == DATASETS[0] else 15)
                and config['n_examples'] == (2468 if dataset == DATASETS[0] else 581)
                and config['severity'] == 5 and config['batch_size'] == (32 if original else 70)
                and set(config['corruptions']) == set(corruptions), 'Raw head/seed/IDs/scope/classes mismatch')
        raw_selection = bundle.read('selection_manifest.json')
        require(json.loads(raw_selection) == config['selection_manifest'] == selected
                and config['selection_manifest_identity'] == dict(bytes=len(raw_selection), sha256=hashlib.sha256(raw_selection).hexdigest()), 'Archived selection differs')
        if old:
            # The baseline's complete manifest and source identities remain historical.
            require(old_assets is not None, 'Historical sealed assets missing')
            old_manifest = read_json(old_assets / 'manifest.json')
            require(config['asset_manifest_sha256'] == controller.OLD_MANIFEST and config['asset_manifest'] == old_manifest
                    and config['component_arm'] == trial['family'] and config['guidance_protocol'] == controller.expected_protocol(trial, calibration)
                    and config['trajectory_observation'] is True and config['component_study'] is True
                    and config['decoder_style'] == 'updated' and config['scheduler_final_alpha_to_one'] is False
                    and config['graph'] == dict(k=10, delta=.1, gamma=.6, use_4d_gft=False, isolate_policy='legacy'), 'Historical baseline contract changed')
            manifest_for_config = old_manifest
            for name in ('tta_gsd_v2.py', 'shared_trajectory.py', 'scripts/eval_multibackbone_guidance.py', 'scripts/eval_multibackbone_source.py'):
                require(config['executed_source_files']['assets/source_snapshot/' + name] == old_manifest['files']['source_snapshot/' + name], 'Historical source relabelled')
        else:
            local_plan = dict(plan, asset_root=str(assets))
            controller.verify_common_config(config, job, local_plan, manifest, selected, calibration)
            manifest_for_config = manifest
        require(config['classifier_state_before_sha256'] == config['classifier_state_after_sha256']
                and config['lion_state_before'] == config['lion_state_after'] and config['initialization_rng_guard']['rng_unchanged'] is True,
                'Frozen model/initialization guard failed')
        checkpoint = config['pointmae_checkpoint'] if head == 'pointmae' else config['checkpoint']
        load = config['checkpoint_load']
        require(load['strict'] is True and not load['missing_keys'] and not load['unexpected_keys']
                and load['checkpoint_sha256'] == manifest_for_config['files'][checkpoint]['sha256'], 'Checkpoint contract changed')
        native = manifest_for_config['guidance']['native_audit']
        audit_assets = old_assets if old else assets
        verify(audit_assets / native, manifest_for_config['files'][native])
        audit = read_json(audit_assets / native)
        require(config['native_bootstrap']['no_build'] is True and set(config['native_bootstrap']['bindings']) == set(audit['backends']), 'Native build/bindings changed')
        for name, entries in audit['backends'].items():
            entry, = entries
            require(config['native_bootstrap']['bindings'][name] == {k: entry[k] for k in ('path', 'bytes', 'sha256')}, 'Native binary pin changed')
        contract = dict(state=config['classifier_state_before_sha256'], checkpoint=load['checkpoint_sha256'],
                        preprocessing=config['source_classifier_preprocessing'], resolved=config['resolved_model_config'])
        key = dataset, head
        require(key not in head_states or head_states[key] == contract, 'Head state/preprocessing differs across trials')
        head_states[key] = contract
        require(not lion_states or lion_states[0] == config['lion_state_before'], 'LION state differs across trials')
        lion_states[:] = [config['lion_state_before']]
        if not original:
            report = analyze_bundle(path, allow_partial=True)
            require(report['status'] == 'partial' and report['selection_complete'], 'Incomplete selected trajectory')
        scores, corruption_rows, trace_rows, costs, consumed = [], [], {}, [], set()
        for corruption in corruptions:
            names = sorted(n for n in bundle.names if n.startswith('batches/' + corruption + '/') and n.endswith('/predictions.npz'))
            batch_size = 32 if original else 70
            require(len(names) == (len(expected_ids) + batch_size - 1) // batch_size, 'Wrong batch count')
            parts, predictions = [], {p: [] for p in (('terminal',) if original else POLICIES)}
            for number, name in enumerate(names):
                prefix = 'batches/' + corruption + '/batch%04d/' % number
                require(name == prefix + 'predictions.npz', 'Unordered batch shards')
                consumed.add(name)
                with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as values:
                    data = {k: values[k].copy() for k in values.files}
                ids = expected_ids[number * batch_size:(number + 1) * batch_size]
                n = len(ids)
                require(np.array_equal(data['index'], ids) and data['label'].shape == data['prediction'].shape == (n,)
                        and data['label'].dtype.kind in 'iu' and data['prediction'].dtype.kind in 'iu'
                        and ((data['label'] >= 0) & (data['label'] < config['num_classes'])).all()
                        and data['logits'].shape == (n, config['num_classes']) and np.isfinite(data['logits']).all()
                        and np.array_equal(data['prediction'], data['logits'].argmax(-1)), 'Raw indexed logits/labels invalid')
                diagnostic = bundle.json(prefix + 'diagnostics.json')
                require(diagnostic['seed'] == 0 and diagnostic['rng_unchanged'] is True and diagnostic['geometry_shape'] == [n, 2048, 3], 'Batch geometry/RNG guard failed')
                batch_key = corruption + '/batch%04d' % number
                trace = dict(geometry_sha256=diagnostic['geometry_sha256'])
                if original:
                    horizon = 35 if corruption == 'background' else 5
                    scheduler = diagnostic['scheduler']
                    scheduler_config = dict(beta_end=.02, beta_start=.0001, beta_schedule='linear', clip_sample=False,
                                            num_train_timesteps=1000, prediction_type='epsilon', set_alpha_to_one=True)
                    require(scheduler == config['scheduler_records'][corruption] and scheduler['config']['set_alpha_to_one'] is True
                            and all(scheduler['config'].get(k) == v for k, v in scheduler_config.items())
                            and scheduler['final_alpha_cumprod'] == 1. and scheduler['timesteps'] == list(range(990, -1, -10))
                            and scheduler['active_timesteps'] == list(range(990, -1, -10))[-horizon:]
                            and 'trajectory_logits' not in data and 'trajectory' not in diagnostic and not diagnostic['guidance_metrics'], 'Original3DD trajectory/scheduler altered')
                    predicted = {'terminal': data['prediction']}
                else:
                    horizon = 30 if corruption == 'background' else 10
                    weights = config['guidance_protocol']['loss_weights']; metrics = diagnostic['guidance_metrics']
                    require(all(math.isfinite(v) and v >= 0 for v in metrics.values())
                            and metrics['graph_calls'] == int(weights['spectral'] > 0)
                            and metrics['chamfer_calls'] == 2 * horizon * int(weights['chamfer'] > 0)
                            and metrics['guidance_backward_calls'] == horizon, 'Component computation counts invalid')
                    cache = diagnostic['paired_state']; remote = PurePosixPath(cache['path'])
                    cache_root = controller.cache_path('development' if old else stage, 0)
                    require(remote.parent.parent.parent == PurePosixPath(cache_root) and remote.parent.parent.name == dataset
                            and remote.parent.name == corruption and remote.name == hashlib.sha256(ids.tobytes()).hexdigest() + '.npz', 'Cache namespace/IDs mismatch')
                    trace.update(initial_hashes=diagnostic['trajectory']['initial_hashes'], paired_state=cache)
                    pair_key = dataset, corruption, number
                    # Smoke uses its own cache; baseline reuse participates only in development pairing.
                    if pair_key in paired:
                        same_preparation(paired[pair_key], trace)
                    else:
                        paired[pair_key] = trace
                    predicted = policies(data['trajectory_logits'], terminal=3)
                trace_rows[batch_key] = trace
                costs.append(dict(dataset=dataset, backbone=head, trial_id=trial['id'], batch=batch_key,
                                  shared_reconstruction_seconds=diagnostic['shared_reconstruction_seconds'],
                                  classifier_forward_seconds=diagnostic['head_seconds'],
                                  whole_process_peak_gpu_memory_mb=diagnostic['whole_process_peak_gpu_memory_mb'],
                                  preparation_mode=diagnostic['preparation_mode']))
                for policy in predictions:
                    predictions[policy].append(predicted[policy])
                parts.append(data)
            merged = {k: np.concatenate([p[k] for p in parts]) for k in ('index', 'label', 'prediction', 'logits')}
            require(np.array_equal(merged['index'], expected_ids), 'Selected coverage incomplete')
            alignment_key = dataset
            if alignment_key in alignment:
                aligned(alignment[alignment_key], merged)
            else:
                alignment[alignment_key] = {k: merged[k] for k in ('index', 'label')}
            with np.load(io.BytesIO(bundle.read(corruption + '_predictions.npz')), allow_pickle=False) as aggregate:
                for key, value in merged.items():
                    require(np.array_equal(aggregate[key], value), 'Aggregate/batch predictions differ')
            for policy, values in predictions.items():
                correct = np.concatenate(values) == merged['label']
                corruption_rows.append(dict(trial_id=trial['id'], family=trial['family'], dataset=dataset, backbone=head,
                                            corruption=corruption, policy=policy, n=len(correct), correct=int(correct.sum()),
                                            accuracy_pct=100. * float(correct.mean())))
        require(consumed == {n for n in bundle.names if n.startswith('batches/') and n.endswith('/predictions.npz')}, 'Undeclared batch/corruption predictions')
        for policy in (('terminal',) if original else POLICIES):
            selected_rows = [r for r in corruption_rows if r['policy'] == policy]
            scores.append(dict(trial_id=trial['id'], family=trial['family'], dataset=dataset, backbone=head, policy=policy,
                               n=sum(r['n'] for r in selected_rows), accuracy_pct=sum(r['accuracy_pct'] for r in selected_rows) / len(selected_rows)))
        saved_rows = list(csv.DictReader(io.StringIO(bundle.read('per_corruption.csv').decode())))
        terminal_rows = [r for r in corruption_rows if r['policy'] == 'terminal']
        require(len(saved_rows) == len(terminal_rows), 'CSV corruption count mismatch')
        for row in terminal_rows:
            saved, = [r for r in saved_rows if r['corruption'] == row['corruption']]
            require(int(saved['n_examples']) == row['n'] and int(saved['n_correct']) == row['correct']
                    and abs(float(saved['accuracy']) * 100. - row['accuracy_pct']) <= 1e-8, 'CSV scores differ from raw logits')
        summary, = list(csv.DictReader(io.StringIO(bundle.read('summary.csv').decode())))
        require(summary['run_id'] == config['run_id'] and summary['status'] == 'partial'
                and int(summary['n_corruptions']) == len(corruptions)
                and int(summary['total_examples']) == len(expected_ids) * len(corruptions)
                and int(summary['total_correct']) == sum(r['correct'] for r in terminal_rows)
                and abs(float(summary['macro_accuracy']) * 100. - next(r['accuracy_pct'] for r in scores if r['policy'] == 'terminal')) <= 1e-8,
                'Summary scores/counts differ')
        return config, scores, corruption_rows, trace_rows, costs
    finally:
        bundle.close()


def write_csv(path, rows):
    require(bool(rows), 'Empty result table')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(args):
    prep = args.prep.resolve(); output = args.output.resolve()
    require(not output.exists(), 'Use a fresh local analysis output directory')
    context = load_context(prep)
    plan, controller, _, assets, manifest, registry, selections, *_ = context
    old_prep = args.old_prep.resolve() if args.old_prep else prep.parent / 'guidance_component_study_20261025'
    old_assets = old_prep / 'assets_v1'
    require(identity(old_assets / 'manifest.json')['sha256'] == controller.OLD_MANIFEST, 'Historical manifest changed')
    output.mkdir(parents=True)
    receipt = unpack_phase(args.phase_archive, args.receipt, args.stage, plan, output / 'phase')
    if args.stage == 'calibrate':
        report, cache_records = calibration_report(output / 'phase', receipt, context)
    else:
        require(args.calibration_archive and args.calibration_receipt, 'Require calibration Drive-roundtrip ZIP and receipt')
        cal_receipt = unpack_phase(args.calibration_archive, args.calibration_receipt, 'calibrate', plan, output / 'calibration_phase')
        report, calibration_caches = calibration_report(output / 'calibration_phase', cal_receipt, context)
        cache_records = []
    validation = dict(status='validated_pending_physical_cache', stage=args.stage, phase_plan_sha256=PLAN_SHA,
                      manifest_sha256=plan['manifest_sha256'], source_snapshot_sha256=plan['source_snapshot_sha256'],
                      phase_archive=receipt['archive'], full_benchmark=False, labels_used_for_calibration=False)
    if args.stage == 'calibrate':
        validation['coefficients'] = report['coefficients']
    else:
        state = receipt['state']; jobs = state['jobs']
        remote_output = PurePosixPath(receipt['archive']['local_path']).with_suffix('')
        expected_jobs = controller.jobs_for(plan, registry, args.stage, str(remote_output), report, PLAN_SHA)
        require(len(jobs) == len(expected_jobs), 'Wrong stage job count')
        alignment, label_references = baseline_labels(prep, context)
        paired, heads, lion, case_rows, corruption_rows, raw_rows, timing_rows = {}, {}, [], [], [], [], []
        shared_heads = {}
        for number, (job, expected) in enumerate(zip(jobs, expected_jobs)):
            require(all(job[k] == value for k, value in expected.items()) and job['status'] == 'complete'
                    and len(job['raw_archives']) == 4, 'Job command/trial/seed/scope changed')
            seen_heads = set()
            require(read_json(output / 'phase' / ('job_%03d_validation.json' % number)) == job, 'Job receipt differs from phase state')
            for raw in job['raw_archives']:
                remote = PurePosixPath(raw['local_path'])
                path = output / 'phase' / remote.relative_to(remote_output)
                verify(path, raw)
                require(raw['drive_path'].startswith(DRIVE), 'Raw archive outside owned Drive')
                config, scores, rows, traces, costs = inspect_raw(path, job, args.stage, context, report, False, alignment, paired, heads, lion)
                head = config['backbone']; require(head == raw['backbone'] and head not in seen_heads, 'Duplicate/misidentified head')
                seen_heads.add(head)
                key = job['dataset'], job['trial']['id']
                require(key not in shared_heads or shared_heads[key] == traces, 'Frozen heads saw different geometry/initial states')
                shared_heads[key] = traces
                require(traces == job['traces'], 'Raw traces differ from recorded remote validation')
                case_rows.extend(scores); corruption_rows.extend(rows); timing_rows.extend(costs)
                raw_rows.append(dict(raw, trial_id=job['trial']['id'], dataset=job['dataset'], backbone=head, run_id=config['run_id'],
                                     source_manifest_sha256=config['asset_manifest_sha256'], reused=False,
                                     canonical_path=copy_canonical(path, prep.parents[2], config)))
            require(seen_heads == set(HEADS), 'Missing frozen head')
            cache_records.extend(job['cache_records'])
        if args.stage == 'development':
            old = read_json(prep / 'reused_baselines.json')
            verify(prep / 'reused_baselines.json', manifest['files']['study/reused_baselines.json'])
            require(old['status'] == 'accepted_old_source_unchanged' and len(old['raw_archives']) == 24
                    and old['original_manifest_sha256'] == controller.OLD_MANIFEST
                    and identity(old_assets / 'source_snapshot.zip')['sha256'] == old['original_source_snapshot_sha256'], 'Historical baseline source evidence differs')
            parity = read_json(prep / 'baseline_equivalence_cpu.json')
            verify(prep / 'baseline_equivalence_cpu.json', plan['baseline_equivalence'])
            require(parity['status'] == 'passed' and parity['default_configurations_exact'] is True and parity['legacy_losses_gradients_exact'] is True, 'Baseline reuse parity missing')
            trial_by_family = {t['family']: t for t in registry.values() if t['reuse']}
            old_cases = set()
            for raw in old['raw_archives']:
                path = Path(raw['canonical_path']); verify(path, raw)
                trial = trial_by_family[raw['component_arm']]
                job = dict(dataset=raw['dataset'], seed=0, trial=trial)
                config, scores, rows, traces, costs = inspect_raw(path, job, 'development', context, report, True, alignment, paired, heads, lion, old_assets)
                key = raw['dataset'], config['backbone'], trial['id']
                require(key not in old_cases and config['run_id'] == raw['run_id'], 'Historical baseline duplicated/relabelled')
                old_cases.add(key)
                shared_key = raw['dataset'], trial['id']
                require(shared_key not in shared_heads or shared_heads[shared_key] == traces, 'Old heads observed different geometry')
                shared_heads[shared_key] = traces
                case_rows.extend(scores); corruption_rows.extend(rows)
                raw_rows.append(dict(trial_id=trial['id'], source_manifest_sha256=controller.OLD_MANIFEST, reused=True, **raw))
            require(len(old_cases) == 24, 'Incomplete historical baseline cases')
        unique_cache = {}
        for cache in cache_records:
            key = cache['local_path']
            require(key not in unique_cache or unique_cache[key] == cache, 'Cache receipt changed across trials')
            unique_cache[key] = cache
        cache_records = list(unique_cache.values())
        require(len(cache_records) == (4 if args.stage == 'smoke' else 30), 'Cache receipt coverage mismatch')
        for trace in paired.values():
            cache = trace['paired_state']; record = unique_cache.get(cache['path'])
            require(record is not None and all(record[k] == cache[k] for k in ('bytes', 'sha256')), 'Initial state lacks matching cache receipt')
        trial_ids = [name for name in registry if args.stage == 'development' or name in plan['smoke_trial_ids']]
        scores = balanced_scores(case_rows, trial_ids)
        write_csv(output / 'case_policy.csv', case_rows)
        write_csv(output / 'corruption_policy.csv', corruption_rows)
        write_csv(output / 'balanced_scores.csv', scores)
        write_json(output / 'raw_archives.json', raw_rows)
        # Count each shared reconstruction once; add the four classifier costs, use maximum peak.
        grouped = {}
        for row in timing_rows:
            key = row['dataset'], row['trial_id'], row['batch']
            entry = grouped.setdefault(key, dict(dataset=key[0], trial_id=key[1], batch=key[2],
                shared_reconstruction_seconds=row['shared_reconstruction_seconds'], classifier_forward_seconds=0.,
                peak_gpu_memory_mb=0., preparation_mode=row['preparation_mode'], heads=0))
            require(entry['shared_reconstruction_seconds'] == row['shared_reconstruction_seconds']
                    and entry['preparation_mode'] == row['preparation_mode'], 'Head timing does not refer to shared reconstruction')
            entry['classifier_forward_seconds'] += row['classifier_forward_seconds']
            entry['peak_gpu_memory_mb'] = max(entry['peak_gpu_memory_mb'], row['whole_process_peak_gpu_memory_mb']); entry['heads'] += 1
        require(all(r['heads'] == 4 for r in grouped.values()), 'Incomplete cost head group')
        write_csv(output / 'costs.csv', list(grouped.values()))
        validation.update(raw_archives=len(raw_rows), jobs=len(jobs), registered_trials=len(scores), selection_policy='probability_mean',
                          balanced_scores=scores, label_references=label_references,
                          cost_interpretation='GSD prepared-state replay/comparison cost; not deployment latency. Original3DD fresh encode-inclusive; original terminal only.')
    physical = physical_caches(cache_records, args.cache_root)
    write_json(output / 'cache_validation.json', physical)
    validation.update(cache_files=len(physical), physically_verified_cache_files=sum(r['local_verified'] for r in physical),
                      physical_cache_complete=all(r['local_verified'] for r in physical))
    if args.stage != 'calibrate':
        calibration_physical = physical_caches(calibration_caches, args.cache_root)
        write_json(output / 'calibration_cache_validation.json', calibration_physical)
        validation['calibration_physical_cache_complete'] = all(r['local_verified'] for r in calibration_physical)
        validation['physical_cache_complete'] = validation['physical_cache_complete'] and validation['calibration_physical_cache_complete']
    if validation['physical_cache_complete']:
        validation['status'] = 'accepted'
    write_json(output / 'validation.json', validation)
    accepted = dict(status=validation['status'], stage=args.stage, phase_plan_sha256=PLAN_SHA,
                    archive=receipt['archive'], manifest_sha256=plan['manifest_sha256'], source_snapshot_sha256=plan['source_snapshot_sha256'],
                    local_validation=identity(output / 'validation.json'), physical_cache_complete=validation['physical_cache_complete'])
    if args.stage == 'development':
        freeze = freeze_selection(validation['balanced_scores'], registry, {}, controller, report)
        accepted.update(selection_policy='probability_mean', selected_spectral_trial_id=freeze['selected_spectral_trial_id'],
                        selected_chamfer_trial_id=freeze['selected_chamfer_trial_id'])
    accepted_path = output / ('accepted_sensitivity_' + args.stage + '.json')
    write_json(accepted_path, accepted)
    if args.stage == 'development':
        freeze['development_acceptance'] = identity(accepted_path)
        if accepted['status'] != 'accepted':
            freeze['status'] = 'pending_physical_cache'
        write_json(output / 'sensitivity_confirmation_freeze.json', freeze)
    print(json.dumps(dict(output=str(output), status=validation['status'], stage=args.stage,
                          physical_cache_complete=validation['physical_cache_complete']), sort_keys=True))


def self_check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in ('../escape', '/absolute', 'a\\b'):
            path = root / 'bad.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr(name, b'bad')
            try:
                checked_zip(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsafe ZIP accepted')
        path = root / 'symlink.zip'
        info = zipfile.ZipInfo('link'); info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(info, 'target')
        try:
            checked_zip(path)
        except ValueError:
            pass
        else:
            raise AssertionError('ZIP symlink accepted')
    a = dict(index=np.array([2, 9]), label=np.array([1, 0]))
    aligned(a, copy.deepcopy(a))
    for bad in (dict(index=np.array([9, 2]), label=a['label']), dict(index=a['index'], label=np.array([0, 1]))):
        try:
            aligned(a, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Mispaired ID/label fixture accepted')
    left = dict(initial_hashes={'noise': 'a'}, paired_state={'sha256': 'b'}, geometry_sha256='left')
    right = dict(left, geometry_sha256='right')
    same_preparation(left, right)
    try:
        same_preparation(left, dict(right, paired_state={'sha256': 'changed'}))
    except ValueError:
        pass
    else:
        raise AssertionError('Changed paired cache accepted')
    rows = [dict(trial_id=t, family='combined', dataset=d, backbone=h, policy='probability_mean', accuracy_pct=10. if d == DATASETS[0] else 90.)
            for t in ('first', 'second') for d in DATASETS for h in HEADS]
    scores = balanced_scores(rows, ('first', 'second'))
    require([r['accuracy_pct'] for r in scores] == [50., 50.] and max(scores, key=lambda r: r['accuracy_pct'])['trial_id'] == 'first', 'Balanced mean/registry tie changed')
    try:
        balanced_scores(rows[:-1], ('first', 'second'))
    except ValueError:
        pass
    else:
        raise AssertionError('Unbalanced head coverage accepted')
    print('AGENT_B_SENSITIVITY_LOCAL_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--prep', type=Path, required=True)
        parser.add_argument('--old-prep', type=Path)
        parser.add_argument('--stage', choices=('calibrate', 'smoke', 'development'), required=True)
        parser.add_argument('--phase-archive', type=Path, required=True)
        parser.add_argument('--receipt', type=Path, required=True)
        parser.add_argument('--calibration-archive', type=Path)
        parser.add_argument('--calibration-receipt', type=Path)
        parser.add_argument('--cache-root', type=Path, action='append', default=[])
        parser.add_argument('--output', type=Path, required=True)
        run(parser.parse_args())
