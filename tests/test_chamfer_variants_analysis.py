"""CPU sealed-bundle acceptance and paired Chamfer experiment analysis."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts.analyze_chamfer_variants import _load, analyze_campaign, cluster_bootstrap
from tests.scan_curvenet_fixtures import valid_curve_report


def seal(path):
    manifest = {}
    for member in path.rglob('*'):
        if member.is_file() and member.name != 'bundle_manifest.json':
            raw = member.read_bytes()
            manifest[member.relative_to(path).as_posix()] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (path / 'bundle_manifest.json').write_text(json.dumps(manifest))


def test_scoped_curvenet_h_proof_is_distinct_from_legacy_semantic_arm(tmp_path):
    baseline=bundle(tmp_path,head='curvenet')
    treatment=bundle(tmp_path,arm='semantic_h_curvenet',head='curvenet')
    report=valid_curve_report();raw=json.dumps(report).encode()
    (treatment/'calibration_report.json').write_bytes(raw)
    reference=dict(identity=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
                   coefficient_key='semantic_h_curvenet',resolved_scalar=.05,purpose=report['purpose'],
                   schema_version=1,formula=report['rule'],calibration_bundles=report['calibration_bundles'],teacher=report['teacher'])
    def configure(config):
        config['calibration_reference']=reference
        config['chamfer_experiment']['semantic_weight']=.05
        config['semantic_guidance'].update(head='curvenet',weight=.05,loss_reduction='sum',
            teacher_checkpoint=report['teacher']['checkpoint'],teacher_preprocessing=report['teacher']['preprocessing'],teacher_input_layout='bcn')
        config['asset_manifest']['files'][report['teacher']['checkpoint']]=report['teacher']['checkpoint_identity']
    change(treatment,'config.json',configure)
    # Both inference arms inherit the same frozen asset manifest.
    change(baseline,'config.json',lambda config:config['asset_manifest']['files'].update({report['teacher']['checkpoint']:report['teacher']['checkpoint_identity']}))
    loaded=_load(treatment)
    assert loaded['arm']=='semantic_h_curvenet'
    assert loaded['calibration']['teacher']==report['teacher']
    assert analyze_campaign([baseline,treatment],bootstrap_replicates=5)['comparisons'][0]['arm']=='semantic_h_curvenet'
    report['native_probes'][0]['zero_local_nonzero']=1
    changed=json.dumps(report).encode();(treatment/'calibration_report.json').write_bytes(changed)
    reference['identity']=dict(bytes=len(changed),sha256=hashlib.sha256(changed).hexdigest())
    change(treatment,'config.json',lambda config:config.update(calibration_reference=reference))
    with pytest.raises(ValueError,match='zero-weight'):_load(treatment)


def change(path, name, mutate):
    target = path / name
    value = json.loads(target.read_text())
    mutate(value)
    target.write_text(json.dumps(value))
    seal(path)


def bundle(root, arm='baseline', head='pointmae', seed=0, corruptions=('gaussian',), ids=(0, 1), predictions=(0, 1)):
    path = root / ('%s-%s-%d' % (arm, head, seed))
    path.mkdir()
    identity = dict(bytes=1, sha256='a' * 64)
    source_identity = dict(bytes=1, sha256=hashlib.sha256(b'x').hexdigest())
    (path / 'source_snapshot').mkdir()
    (path / 'source_snapshot/tta_gsd_v2.py').write_bytes(b'x')
    selection = dict(dataset='scanobjectnn_c', n_examples=581, stage='development', indices=list(ids))
    raw = json.dumps(selection).encode()
    (path / 'selection_manifest.json').write_bytes(raw)
    config = dict(dataset='scanobjectnn_c', n_examples=581, num_classes=15, split='main_split', batch_size=70,
                  method='gsd_inspired_latent_spectral_guidance', severity=5, seed=seed, component_study=True,
                  component_arm='chamfer', backbone=head, trajectory_observation=True, status='partial',
                  corruptions=list(corruptions), selected_original_ids=list(ids), evaluation_n_examples=len(ids),
                  selection_manifest=selection, selection_manifest_identity=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                  classifier_state_before_sha256='b' * 64, classifier_state_after_sha256='b' * 64,
                  lion_state_before={'vae': 'c' * 64}, lion_state_after={'vae': 'c' * 64},
                  guidance_protocol=dict(loss_weights=dict(chamfer=1, spectral=0, invariant=0), skip_inactive_components=True),
                  asset_manifest={'files': {'checkpoint': identity}}, asset_manifest_sha256='a'*64,
                  executed_source_files={'repo/tta_gsd_v2.py': source_identity},
                  native_extensions={'chamfer': dict(path='/native/chamfer.so', **identity)},
                  reconstruction_group_id=arm + '-' + str(seed))
    if arm != 'baseline':
        config['chamfer_experiment'] = {'variant': 'semantic' if arm.startswith('semantic_') else arm}
        if arm.startswith('semantic_'):
            config['semantic_guidance'] = {'kind': 'kl' if arm == 'semantic_kl' else 'marginal_entropy', 'head': 'pointmae', 'weight': .1}
    (path / 'config.json').write_text(json.dumps(config))
    for corruption in corruptions:
        target = path / 'batches' / corruption / 'batch0000'
        target.mkdir(parents=True)
        steps = [8, 15, 23, 30] if corruption == 'background' else [3, 5, 8, 10]
        names = ['clean_%02d' % s for s in steps[:3]] + ['terminal']
        logits = np.zeros((len(ids), 4, 15))
        logits[np.arange(len(ids)), :, np.asarray(predictions)] = 4
        np.savez(target / 'predictions.npz', index=np.asarray(ids), label=np.array([i % 2 for i in ids]),
                 prediction=np.asarray(predictions), logits=logits[:, 3], trajectory_logits=logits,
                 candidate_names=np.array(names), candidate_loop_counts=np.array(steps))
        probes = [dict(loop_count=s, rng_unchanged=True, buffers_unchanged=True, live_inputs_unchanged=True,
                       rng_before_sha256='r', rng_after_sha256='r', buffers_before={}, buffers_after={}) for s in steps[:3]]
        trace = dict(complete=True, candidate_names=names, candidate_loop_counts=steps, probes=probes,
                     initial_hashes={k: 'd' * 64 for k in ('input_points', 'shape_latent', 'local_latent', 'noise',
                                                         'style', 'noisy_local', 'timesteps', 'alpha_bar')},
                     initial_rng_sha256='e' * 64, final_hashes={'local_latent': arm})
        diagnostic = dict(trajectory=trace, rng_unchanged=True, reconstruction_rng_before_sha256='e' * 64,
                          preparation_mode='cached_replay', shared_reconstruction_seconds=2., head_seconds=.5,
                          guidance_metrics=dict(graph_calls=0, spectral_loss_calls=0))
        (target / 'diagnostics.json').write_text(json.dumps(diagnostic))
    seal(path)
    return path


def test_complete_declared_subset_and_same_set_metrics(tmp_path):
    baseline = bundle(tmp_path, predictions=(1, 1))
    sparse = bundle(tmp_path, 'sparse')
    report = analyze_campaign([baseline, sparse], bootstrap_replicates=20)
    assert report['status'] == 'complete_subset'
    paired = report['comparisons'][0]
    assert paired['scope'] == 'declared_subset'
    assert paired['policy'] == 'terminal'
    assert paired['delta_pp'] == 50
    assert paired['corrected_ids'] == [dict(backbone='pointmae', seed=0, corruption='gaussian', index=0)]
    assert paired['harmed_ids'] == []
    assert all(row['policy'] in ('terminal', 'probability_mean') for row in report['per_corruption'])
    assert report['costs'][0]['mode'] == 'cache_replay'
    assert report['costs'][0]['total_seconds'] == 2.5
    assert report['cost_comparisons'][0]['arm_over_baseline'] == 1
    assert 'deployment' in report['cost_note']


@pytest.mark.parametrize('fault', ['running', 'failed', 'missing_ids', 'spectral', 'graph', 'label', 'initial',
                                  'source', 'skip', 'frozen', 'native', 'rng', 'asset'])
def test_rejects_invalid_or_unpaired_evidence(tmp_path, fault):
    baseline = bundle(tmp_path)
    sparse = bundle(tmp_path, 'sparse')
    if fault in ('running', 'failed'):
        change(sparse, 'config.json', lambda x: x.update(status=fault))
    elif fault == 'missing_ids':
        change(sparse, 'config.json', lambda x: x.update(selected_original_ids=[0, 1, 2]))
    elif fault == 'skip':
        change(baseline, 'config.json', lambda x: x['guidance_protocol'].update(skip_inactive_components=False))
    elif fault == 'source':
        change(sparse, 'config.json', lambda x: x['executed_source_files']['repo/tta_gsd_v2.py'].update(sha256='f' * 64))
    elif fault == 'native':
        change(sparse, 'config.json', lambda x: x['native_extensions']['chamfer'].update(sha256='f' * 64))
    elif fault == 'asset':
        change(sparse, 'config.json', lambda x: x.pop('asset_manifest'))
    elif fault == 'frozen':
        change(sparse, 'config.json', lambda x: x.update(classifier_state_after_sha256='f' * 64))
    elif fault == 'label':
        name = sparse / 'batches/gaussian/batch0000/predictions.npz'
        with np.load(name) as values:
            arrays = dict(values)
        arrays['label'] = np.array([1, 1])
        np.savez(name, **arrays)
        seal(sparse)
    else:
        name = 'batches/gaussian/batch0000/diagnostics.json'
        if fault == 'rng':
            change(sparse, name, lambda x: x['trajectory'].update(initial_rng_sha256='f' * 64))
        elif fault == 'initial':
            change(sparse, name, lambda x: x['trajectory']['initial_hashes'].update(noise='f' * 64))
        else:
            key = 'graph_calls' if fault == 'graph' else 'spectral_loss_calls'
            change(sparse, name, lambda x: x['guidance_metrics'].update({key: 1}))
    with pytest.raises(ValueError):
        analyze_campaign([baseline, sparse], bootstrap_replicates=20)


def test_truncated_selected_ids_are_rejected_even_when_sealed(tmp_path):
    baseline = bundle(tmp_path)
    sparse = bundle(tmp_path, 'sparse')
    for path in (baseline, sparse):
        selection = dict(dataset='scanobjectnn_c', n_examples=581, stage='development', indices=[0, 1, 2])
        raw = json.dumps(selection).encode()
        (path / 'selection_manifest.json').write_bytes(raw)
        change(path, 'config.json', lambda x: x.update(selected_original_ids=[0, 1, 2], evaluation_n_examples=3,
               selection_manifest=selection, selection_manifest_identity=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())))
    with pytest.raises(ValueError):
        analyze_campaign([baseline, sparse])


def test_shared_cost_counted_once_and_head_times_added(tmp_path):
    paths = [bundle(tmp_path, arm, head) for arm in ('baseline', 'sparse') for head in ('pointmae', 'dgcnn')]
    report = analyze_campaign(paths, bootstrap_replicates=5)
    assert len(report['costs']) == 2
    assert all(row['total_seconds'] == 3 for row in report['costs'])


def test_independent_semantic_losses_and_weighted_arms_stay_separate(tmp_path):
    arms = ('baseline', 'sparse', 'semantic_kl', 'semantic_h', 'weighted')
    paths = [bundle(tmp_path, arm) for arm in arms]
    report = analyze_campaign(paths, bootstrap_replicates=5)
    assert {row['arm'] for row in report['comparisons']} == set(arms) - {'baseline'}
    assert len(report['comparisons']) == 8


def test_fresh_encode_and_cache_replay_times_are_not_compared(tmp_path):
    baseline = bundle(tmp_path)
    sparse = bundle(tmp_path, 'sparse')
    change(sparse, 'batches/gaussian/batch0000/diagnostics.json', lambda x: x.update(preparation_mode='fresh_encode'))
    report = analyze_campaign([baseline, sparse], bootstrap_replicates=5)
    assert {row['mode'] for row in report['costs']} == {'fresh_encode', 'cache_replay'}
    assert report['cost_comparisons'] == []


def test_missing_head_seed_condition_cannot_be_silently_intersected(tmp_path):
    paths = [bundle(tmp_path), bundle(tmp_path, seed=1), bundle(tmp_path, 'sparse')]
    with pytest.raises(ValueError, match='condition set'):
        analyze_campaign(paths)


def test_missing_copied_source_bytes_are_rejected(tmp_path):
    baseline = bundle(tmp_path)
    sparse = bundle(tmp_path, 'sparse')
    (sparse / 'source_snapshot/tta_gsd_v2.py').unlink()
    seal(sparse)
    with pytest.raises(ValueError, match='source bytes'):
        analyze_campaign([baseline, sparse])


def test_unsafe_and_unsealed_bundles_rejected(tmp_path):
    unsafe = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(unsafe, 'w') as archive:
        archive.writestr('../config.json', '{}')
    for path in (unsafe, tmp_path / 'absent'):
        with pytest.raises(ValueError):
            analyze_campaign([path])


def test_object_cluster_bootstrap_retains_all_conditions():
    # Each object's head/seed/corruption effects cancel; row-resampling would not.
    values = {0: [-1., 1., -1., 1.], 1: [1., -1., 1., -1.]}
    replicates = cluster_bootstrap(values, replicates=100, seed=8)
    np.testing.assert_array_equal(replicates, np.zeros(100))


def test_full_corruption_scope_labels_all15_and_macro14(tmp_path):
    from research_artifacts import CORRUPTIONS
    paths = [bundle(tmp_path, arm, corruptions=CORRUPTIONS) for arm in ('baseline', 'sparse')]
    report = analyze_campaign(paths, bootstrap_replicates=5)
    assert report['status'] == 'complete_subset'  # Two original objects are not a full benchmark.
    assert {row['scope'] for row in report['comparisons']} == {'all15', 'macro14'}


def test_weighted_rates_require_sealed_fixed_calibration_reference(tmp_path):
    baseline = bundle(tmp_path)
    weighted = bundle(tmp_path, 'weighted')
    for path in (baseline, weighted):
        change(path, 'config.json', lambda x: x['guidance_protocol'].update(eta=.01, gamma=.01))
    change(weighted, 'config.json', lambda x: x['guidance_protocol'].update(eta=.005, gamma=.005))
    with pytest.raises(ValueError, match='protocol'):
        analyze_campaign([baseline, weighted], bootstrap_replicates=5)
    config = json.loads((baseline / 'config.json').read_text())
    proof = dict(config_identity=dict(bytes=1, sha256='b'*64), asset_manifest_sha256='a'*64,
                 executed_source_files=config['executed_source_files'], native_extensions=config['native_extensions'],
                 classifier_state_sha256=config['classifier_state_before_sha256'],
                 runtime=dict(cuda='12.1', gpu='A100'), lion_state=config['lion_state_before'])
    report = dict(schema_version=1, purpose='label_free_chamfer_calibration', status='verified_label_free_gpu',
                  loss_reduction='sum for Chamfer, weighted Chamfer, and per-object semantic losses',
                  target_semantic_update_ratio=.1, coefficients=dict(semantic_kl=.2, semantic_h=.3, weighted_scale=.5),
                  ratios={candidate: dict(count=4200, missing=0, min=ratio, median=ratio, p90=ratio, max=ratio)
                          for candidate, ratio in (('semantic_kl', .5), ('semantic_h', 1/3), ('weighted', 2.))},
                  rule='semantic=0.1/median(candidate/C combined update norm); weighted_scale=1/median(weighted/C)',
                  calibration_bundles={'modelnet40_c': proof, 'scanobjectnn_c': proof})
    raw = json.dumps(report).encode()
    (weighted / 'calibration_report.json').write_bytes(raw)
    reference = dict(identity=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                     coefficient_key='weighted_scale', resolved_scalar=.5, purpose=report['purpose'],
                     schema_version=1, formula=report['rule'], calibration_bundles=report['calibration_bundles'])
    change(weighted, 'config.json', lambda x: x.update(calibration_reference=reference))
    accepted = analyze_campaign([baseline, weighted], bootstrap_replicates=5)
    assert accepted['status'] == 'complete_subset'
    # Updating a seal cannot turn a synthetic or numerically altered report into proof.
    for key, value in (('status', 'synthetic_only'), ('coefficients', dict(semantic_kl=.2, semantic_h=.3, weighted_scale=.7))):
        altered = dict(report, **{key: value})
        altered_raw = json.dumps(altered).encode()
        (weighted / 'calibration_report.json').write_bytes(altered_raw)
        altered_reference = dict(reference, identity=dict(bytes=len(altered_raw), sha256=hashlib.sha256(altered_raw).hexdigest()))
        if key == 'coefficients':
            altered_reference['resolved_scalar'] = .7
        change(weighted, 'config.json', lambda x: x.update(calibration_reference=altered_reference))
        with pytest.raises(ValueError):
            analyze_campaign([baseline, weighted], bootstrap_replicates=5)
    (weighted / 'calibration_report.json').write_bytes(raw)
    change(weighted, 'config.json', lambda x: x.update(calibration_reference=reference))
    change(weighted, 'config.json', lambda x: x['guidance_protocol'].update(gamma=.006))
    with pytest.raises(ValueError):
        analyze_campaign([baseline, weighted], bootstrap_replicates=5)


@pytest.mark.parametrize('native_case', ['extra_run_native', 'missing_proof_native', 'tampered_proof_native'])
def test_calibration_native_subset_matches_real_pointmae_to_fourhead_inventory(tmp_path, native_case):
    path = bundle(tmp_path, 'weighted')
    config = json.loads((path / 'config.json').read_text())
    names = ('chamfer', 'chamfer_3D', 'knn', 'pointnet2_ops._ext')
    native = {name: dict(path='/native/' + name + '.so', bytes=1, sha256='a' * 64) for name in names}
    proof = dict(config_identity=dict(bytes=1, sha256='b' * 64), asset_manifest_sha256=config['asset_manifest_sha256'],
                 executed_source_files=config['executed_source_files'], native_extensions=native,
                 classifier_state_sha256=config['classifier_state_before_sha256'],
                 runtime=dict(cuda='12.1', gpu='A100'), lion_state=config['lion_state_before'])
    report = dict(schema_version=1, purpose='label_free_chamfer_calibration', status='verified_label_free_gpu',
                  loss_reduction='sum for Chamfer, weighted Chamfer, and per-object semantic losses',
                  target_semantic_update_ratio=.1, coefficients=dict(semantic_kl=.2, semantic_h=.3, weighted_scale=.5),
                  ratios={candidate: dict(count=4200, missing=0, min=ratio, median=ratio, p90=ratio, max=ratio)
                          for candidate, ratio in (('semantic_kl', .5), ('semantic_h', 1/3), ('weighted', 2.))},
                  rule='semantic=0.1/median(candidate/C combined update norm); weighted_scale=1/median(weighted/C)',
                  calibration_bundles={'modelnet40_c': proof, 'scanobjectnn_c': proof})
    raw = json.dumps(report).encode()
    (path / 'calibration_report.json').write_bytes(raw)
    reference = dict(identity=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                     coefficient_key='weighted_scale', resolved_scalar=.5, purpose=report['purpose'],
                     schema_version=1, formula=report['rule'], calibration_bundles=report['calibration_bundles'])
    run_native = dict(native, pointnet2_batch_cuda=dict(path='/native/pointnet2_batch_cuda.so', bytes=2, sha256='c' * 64))
    if native_case == 'missing_proof_native':
        run_native.pop('knn')
    elif native_case == 'tampered_proof_native':
        run_native['knn'] = dict(native['knn'], sha256='f' * 64)
    change(path, 'config.json', lambda value: value.update(calibration_reference=reference, native_extensions=run_native))
    if native_case != 'extra_run_native':
        with pytest.raises(ValueError, match='Calibration/run source/native/LION mismatch'):
            _load(path)
        return
    assert _load(path)['arm'] == 'weighted'
    # The allowance is only between calibration and a run; paired arms still need complete equality.
    baseline = bundle(tmp_path)
    change(baseline, 'config.json', lambda value: value.update(native_extensions=native))
    change(baseline, 'config.json', lambda value: value['guidance_protocol'].update(eta=.01, gamma=.01))
    change(path, 'config.json', lambda value: value['guidance_protocol'].update(eta=.005, gamma=.005))
    with pytest.raises(ValueError, match='Unpaired native hashes'):
        analyze_campaign([baseline, path], bootstrap_replicates=5)


def timing_groups(root):
    groups = []
    for repeat in range(3):
        directory = root / str(repeat)
        directory.mkdir()
        paths = [bundle(directory, arm, head, corruptions=('gaussian', 'background'),
                        ids=tuple(range(70)), predictions=tuple(index % 2 for index in range(70)))
                 for arm in ('baseline', 'sparse') for head in ('pointmae', 'dgcnn')]
        for path in paths:
            change(path, 'config.json', lambda value: value.update(reconstruction_group_id=value['reconstruction_group_id'] + '-repeat%d' % repeat))
            for corruption in ('gaussian', 'background'):
                change(path, 'batches/' + corruption + '/batch0000/diagnostics.json',
                       lambda value: value.update(preparation_mode='fresh_encode'))
        groups.append(paths)
    return groups


def test_three_timing_repeats_have_separate_costs_without_weakening_accuracy_duplicates(tmp_path):
    from scripts.analyze_chamfer_variants import analyze_repeated_costs
    groups = timing_groups(tmp_path)
    report = analyze_repeated_costs(groups)
    assert len(report['repeats']) == 3
    assert all(len(repeat['costs']) == 4 for repeat in report['repeats'])
    assert report['summary'][0]['baseline_seconds'] == [6., 6., 6.]
    assert report['summary'][0]['arm_seconds'] == [6., 6., 6.]
    with pytest.raises(ValueError, match='Duplicate'):
        analyze_campaign([path for group in groups for path in group], bootstrap_replicates=1)


def test_fresh_timing_reports_encoded_drift_within_and_across_repeats(tmp_path):
    from scripts.analyze_chamfer_variants import analyze_repeated_costs
    groups = timing_groups(tmp_path)
    fields = ('shape_latent', 'local_latent', 'style', 'noisy_local')
    for repeat, group in enumerate(groups):
        for number, path in enumerate(group):
            digest = '%064x' % (1 + repeat * 2 + number // 2)
            for corruption in ('gaussian', 'background'):
                change(path, 'batches/' + corruption + '/batch0000/diagnostics.json',
                       lambda value: value['trajectory']['initial_hashes'].update({key: digest for key in fields}))
    report = analyze_repeated_costs(groups)
    drift = [row for repeat in report['repeats'] for row in repeat['encoded_state_differences']]
    assert {row['comparison'] for row in drift} == {'within_repeat_baseline_to_arm', 'cross_repeat_baseline'}
    assert all(set(row['fields']) == set(fields) for row in drift)
    assert all(value['reference_sha256'] != value['compared_sha256'] for row in drift for value in row['fields'].values())
    assert report['exact_encoded_initial_states'] is False
    assert 'not exact encoded-state pairing' in report['cost_note']


@pytest.mark.parametrize('field', ['input_points', 'noise', 'timesteps', 'alpha_bar', 'initial_rng_sha256'])
@pytest.mark.parametrize('scope', ['within_repeat', 'cross_repeat'])
def test_fresh_timing_still_requires_exact_input_noise_scheduler_rng(tmp_path, field, scope):
    from scripts.analyze_chamfer_variants import analyze_repeated_costs
    groups = timing_groups(tmp_path)
    paths = groups[1][2:] if scope == 'within_repeat' else groups[1]
    for path in paths:
        def tamper(value):
            trace = value['trajectory']
            if field == 'initial_rng_sha256':
                trace[field] = 'f' * 64
            else:
                trace['initial_hashes'][field] = 'f' * 64
        change(path, 'batches/gaussian/batch0000/diagnostics.json', tamper)
    with pytest.raises(ValueError):
        analyze_repeated_costs(groups)


@pytest.mark.parametrize('mode', ['fresh_encode', 'cached_replay'])
def test_accuracy_keeps_exact_encoded_pairing_regardless_preparation_mode(tmp_path, mode):
    paths = [bundle(tmp_path, arm) for arm in ('baseline', 'sparse')]
    for path in paths:
        change(path, 'batches/gaussian/batch0000/diagnostics.json', lambda value: value.update(preparation_mode=mode))
    change(paths[1], 'batches/gaussian/batch0000/diagnostics.json',
           lambda value: value['trajectory']['initial_hashes'].update(local_latent='f' * 64))
    with pytest.raises(ValueError, match='Unpaired actual input/h0/noise/RNG'):
        analyze_campaign(paths, bootstrap_replicates=1)


def test_fresh_timing_preserves_same_group_full_initial_equality_across_heads(tmp_path):
    from scripts.analyze_chamfer_variants import analyze_repeated_costs
    groups = timing_groups(tmp_path)
    change(groups[0][1], 'batches/gaussian/batch0000/diagnostics.json',
           lambda value: value['trajectory']['initial_hashes'].update(local_latent='f' * 64))
    with pytest.raises(ValueError, match='Shared reconstruction initial state'):
        analyze_repeated_costs(groups)


@pytest.mark.parametrize('fault', ['replay', 'repeat_count', 'head', 'group', 'rng', 'source', 'duplicate',
                                  'cross_repeat_source', 'experiment', 'cost_heads'])
def test_timing_repeats_reject_unmatched_or_reused_measurements(tmp_path, fault):
    from scripts.analyze_chamfer_variants import analyze_repeated_costs
    groups = timing_groups(tmp_path)
    path = groups[1][0]
    if fault == 'repeat_count':
        groups.pop()
    elif fault == 'head':
        groups[1].pop()
    elif fault == 'duplicate':
        groups[1].append(path)
    elif fault == 'group':
        change(path, 'config.json', lambda value: value.update(reconstruction_group_id='baseline-0-repeat0'))
    elif fault == 'experiment':
        for path in groups[1][2:]:
            change(path, 'config.json', lambda value: value['chamfer_experiment'].update(interval=7))
    elif fault == 'cost_heads':
        change(path, 'config.json', lambda value: value.update(reconstruction_group_id='independent-head-repeat1'))
    elif fault == 'cross_repeat_source':
        for path in groups[1]:
            (path / 'source_snapshot/tta_gsd_v2.py').write_bytes(b'y')
            change(path, 'config.json', lambda value: value['executed_source_files']['repo/tta_gsd_v2.py'].update(sha256=hashlib.sha256(b'y').hexdigest()))
    elif fault == 'source':
        (path / 'source_snapshot/tta_gsd_v2.py').write_bytes(b'y')
        change(path, 'config.json', lambda value: value['executed_source_files']['repo/tta_gsd_v2.py'].update(sha256=hashlib.sha256(b'y').hexdigest()))
    else:
        name = 'batches/gaussian/batch0000/diagnostics.json'
        if fault == 'rng':
            change(path, name, lambda value: value['trajectory'].update(initial_rng_sha256='f'*64))
        else:
            change(path, name, lambda value: value.update(preparation_mode='cached_replay'))
    with pytest.raises(ValueError):
        analyze_repeated_costs(groups)


def test_timing_manifest_cli_keeps_repeat_measurements_separate(tmp_path):
    from scripts.analyze_chamfer_variants import main
    groups = timing_groups(tmp_path)
    manifest = tmp_path / 'timing.json'
    manifest.write_text(json.dumps(dict(schema_version=1, repeats=[[str(path) for path in group] for group in groups])))
    output = tmp_path / 'report.json'
    main(['--timing-manifest', str(manifest), '--output', str(output)])
    assert json.loads(output.read_text())['status'] == 'complete_repeated_timing'
