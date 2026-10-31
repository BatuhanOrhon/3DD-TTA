"""CPU-only acceptance tests with physically serialized paired caches."""
import copy
import hashlib
import io
import json
import pickle
import random
import zipfile

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    np.__version__.split('.')[0] != '1',
    reason='Physical cache RNG verification requires pinned NumPy1',
)

from scripts import analyze_chamfer_depth_transfer as analysis
from scripts.analyze_legacy_trajectory import policies
from research_artifacts import CORRUPTIONS


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def tensor_hash(value):
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def make_fixture(tmp_path, dataset='scanobjectnn_c', backbone='pointmae', smoke=False):
    from scripts.eval_chamfer_depth import configurations
    ids = list(range(32)) if smoke else [1, 3]
    n = len(ids)
    namespace = 'paired-smoke' if smoke else 'paired'
    total = 581 if dataset == 'scanobjectnn_c' else 2468
    manifest = dict(dataset=dataset, n_examples=total, stage='smoke' if smoke else 'heldout', indices=ids)
    selection_path = tmp_path / 'selection.json'
    selection_path.write_bytes(encoded(manifest))
    cfg = configurations(['--transfer-study', '--dataset', dataset, '--backbones', backbone,
                          '--dataset-root', '/data', '--label-path', '/data/label.npy', '--asset-root', '/assets', '--result-root', '/results', '--pointmae-config', '/assets/config.yaml', '--pointmae-checkpoint', '/assets/model.pth', '--index-manifest', str(selection_path), '--paired-state-root', '/remote/' + namespace + '/seed0', '--preview'] + (['--technical-smoke', '--corruptions', 'gaussian', 'background'] if smoke else []))[0]
    h = 'a' * 64
    cfg.update(status='partial' if smoke else 'complete', classifier_state_before_sha256=h, classifier_state_after_sha256=h,
               lion_state_before=dict(vae=h, priors=h), lion_state_after=dict(vae=h, priors=h),
               checkpoint_load=dict(strict=True, checkpoint_sha256=h, missing_keys=[], unexpected_keys=[]),
               asset_manifest_sha256=h, reconstruction_group_id='fixed-group')
    snapshot = io.BytesIO()
    sources = {}
    with zipfile.ZipFile(snapshot, 'w') as archive:
        for name in analysis.CRITICAL_SOURCES:
            raw = ('sealed source ' + name).encode()
            archive.writestr('source_snapshot/' + name, raw)
            sources['assets/source_snapshot/' + name] = identity(raw)
    cfg['source_snapshot_sha256'] = identity(snapshot.getvalue())['sha256']
    cfg['executed_source_files'] = sources
    cfg['executed_source_sha256'] = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    members = {'selection_manifest.json': selection_path.read_bytes(), 'source_snapshot.zip': snapshot.getvalue()}
    for corruption in cfg['corruptions']:
        depth = 35 if corruption == 'background' else 8
        steps = [9, 18, 27] if depth == 35 else [2, 4, 6]
        names = ['clean_%02d' % s for s in steps] + ['terminal']
        initial = dict(input_points=h, noisy_local=h, style=h)
        arrays = dict(shape_latent=np.zeros((n, 4, 1, 1), np.float32), local_latent=np.zeros((n, 8192, 1, 1), np.float32), noise=np.ones((n, 8192, 1, 1), np.float32))
        hashes = {k: tensor_hash(v) for k, v in arrays.items()}
        initial.update(hashes)
        py = random.Random(9).getstate()
        nr = np.random.RandomState(9).get_state()
        tc, tg = np.array([1, 2], np.uint8), np.array([3, 4], np.uint8)
        state = (py, nr, tc, [tg])
        metadata = dict(schema_version=1, cuda_devices=1, input_sha256=h, tensor_hashes=hashes,
                        python_rng=py, numpy_rng=[nr[0], nr[2], nr[3], nr[4]], rng_before_sha256=h,
                        rng_after_sha256=hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest())
        cache_name = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest() + '.npz'
        cache_path = tmp_path / 'cache' / namespace / 'seed0' / dataset / corruption / cache_name
        cache_path.parent.mkdir(parents=True)
        np.savez(cache_path, metadata=np.asarray(json.dumps(metadata)), numpy_rng_keys=nr[1], torch_rng_cpu=tc, torch_rng_cuda_0=tg, **arrays)
        classes = cfg['num_classes']
        logits = np.zeros((n, 4, classes), np.float64)
        # PM corrects object1; terminal alone gets object3 correct. PM harms object3.
        logits[0, :3, 0] = 4
        logits[0, 3, 1] = 5
        logits[1, :3, 1] = 4
        logits[1, 3, 0] = 5
        data = dict(index=np.array(ids), label=np.zeros(n, np.int64), prediction=logits[:, -1].argmax(1),
                    logits=logits[:, -1], trajectory_logits=logits, candidate_names=np.array(names), candidate_loop_counts=np.array(steps + [depth]), **policies(logits, 3))
        stream = io.BytesIO()
        np.savez(stream, **data)
        prefix = 'batches/%s/batch0000/' % corruption
        members[prefix + 'predictions.npz'] = stream.getvalue()
        probes = [dict(loop_count=s, timestep=(depth - s) * 10, rng_unchanged=True, buffers_unchanged=True,
                       live_inputs_unchanged=True, rng_before_sha256=h, rng_after_sha256=h,
                       buffers_before={'vae': h}, buffers_after={'vae': h}, style_sha256=hashes['shape_latent'],
                       clean_latent_sha256=h, decoded_geometry_sha256=h) for s in steps]
        trace = dict(complete=True, horizon=depth, candidate_names=names, candidate_loop_counts=steps + [depth],
                     initial_hashes=initial, initial_rng_sha256=metadata['rng_after_sha256'], final_hashes=dict(style=hashes['shape_latent'], latent=h, timestep=0), probes=probes)
        diagnostic = dict(trajectory=trace, scheduler=dict(active_timesteps=list(range((depth - 1) * 10, -1, -10)), reverse_steps=depth,
                          initial_timestep=(depth - 1) * 10, config=dict(set_alpha_to_one=True), final_alpha_cumprod=1.),
                          paired_state=dict(path='/remote/%s/seed0/%s/%s/%s' % (namespace, dataset, corruption, cache_name), **identity(cache_path.read_bytes())),
                          rng_unchanged=True, rng_before_sha256=h, rng_after_sha256=h, geometry_sha256=h, geometry_shape=[n, 2048, 3],
                          reconstruction_rng_before_sha256=h, reconstruction_rng_after_sha256=h,
                          executed_source_sha256=cfg['executed_source_sha256'], source_snapshot_sha256=cfg['source_snapshot_sha256'],
                          asset_manifest_sha256=h, checkpoint_load=cfg['checkpoint_load'])
        members[prefix + 'diagnostics.json'] = encoded(diagnostic)
    members['config.json'] = encoded(cfg)
    binding = dict(expected_config={k: copy.deepcopy(cfg[k]) for k in ('dataset', 'backbone', 'seed', 'selected_original_ids', 'source_snapshot_sha256', 'asset_manifest_sha256', 'checkpoint_load')}, required_source_files=sources)
    return members, binding, tmp_path / 'cache'


def seal(tmp_path, members):
    path = tmp_path / 'bundle.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr('bundle_manifest.json', encoded({k: identity(v) for k, v in members.items()}))
    return path


@pytest.mark.parametrize('dataset', ['scanobjectnn_c', 'modelnet40_c'])
@pytest.mark.parametrize('backbone', ['pointmae', 'dgcnn', 'curvenet', 'pointnext'])
def test_real_cache_and_recomputed_probability_mean(tmp_path, dataset, backbone):
    members, binding, cache = make_fixture(tmp_path, dataset, backbone)
    report = analysis.analyze_bundle(seal(tmp_path, members), binding, cache)
    assert report['status'] == 'accepted'
    assert report['outcomes']['probability_mean'][0] == [True, False]
    assert report['outcomes']['terminal'][0] == [False, True]
    assert report['per_corruption'][0]['accuracy_pct'] == 50


@pytest.mark.parametrize('fault', ['horizon', 'readout', 'source', 'checkpoint', 'ids', 'failed', 'nan', 'cache', 'guard', 'source_bytes', 'geometry', 'label', 'batch_checkpoint', 'classes', 'state', 'partial'])
def test_rejects_resealed_but_invalid_evidence(tmp_path, fault):
    members, binding, cache = make_fixture(tmp_path)
    key = 'batches/gaussian/batch0000/'
    if fault in ('horizon', 'guard', 'geometry', 'batch_checkpoint'):
        d = json.loads(members[key + 'diagnostics.json'])
        if fault == 'horizon':
            d['scheduler']['reverse_steps'] = 10
        elif fault == 'geometry':
            d['geometry_shape'] = [2, 1024, 3]
        elif fault == 'batch_checkpoint':
            d['checkpoint_load']['checkpoint_sha256'] = 'b' * 64
        else:
            d['trajectory']['probes'][0]['rng_before_sha256'] = None
        members[key + 'diagnostics.json'] = encoded(d)
    elif fault in ('source', 'checkpoint', 'ids', 'failed', 'classes', 'state', 'partial'):
        cfg = json.loads(members['config.json'])
        if fault == 'source':
            cfg['executed_source_files'][next(iter(binding['required_source_files']))]['sha256'] = 'b' * 64
        elif fault == 'checkpoint':
            cfg['checkpoint_load']['checkpoint_sha256'] = 'b' * 64
        elif fault == 'ids':
            cfg['selected_original_ids'] = [2, 3]
        elif fault == 'classes':
            cfg['num_classes'] = 40
        elif fault == 'state':
            cfg['lion_state_after']['vae'] = 'b' * 64
        else:
            cfg['status'] = 'partial' if fault == 'partial' else 'failed'
        members['config.json'] = encoded(cfg)
    elif fault == 'source_bytes':
        snapshot = io.BytesIO()
        with zipfile.ZipFile(snapshot, 'w') as archive:
            for name in analysis.CRITICAL_SOURCES:
                archive.writestr('source_snapshot/' + name, b'changed source')
        members['source_snapshot.zip'] = snapshot.getvalue()
        cfg = json.loads(members['config.json'])
        cfg['source_snapshot_sha256'] = identity(snapshot.getvalue())['sha256']
        binding['expected_config']['source_snapshot_sha256'] = cfg['source_snapshot_sha256']
        members['config.json'] = encoded(cfg)
    elif fault == 'cache':
        next(cache.rglob('*.npz')).write_bytes(b'changed')
    else:
        with np.load(io.BytesIO(members[key + 'predictions.npz'])) as data:
            values = {k: data[k] for k in data.files}
        values['probability_mean'][0] = 1
        if fault == 'nan':
            values['trajectory_logits'][0, 0, 0] = np.nan
        elif fault == 'label':
            values['label'][0] = 15
        stream = io.BytesIO()
        np.savez(stream, **values)
        members[key + 'predictions.npz'] = stream.getvalue()
    with pytest.raises((ValueError, KeyError)):
        analysis.analyze_bundle(seal(tmp_path, members), binding, cache)


def test_conditional_object_bootstrap_and_incomplete_seed(tmp_path):
    members, binding, cache = make_fixture(tmp_path)
    report = analysis.analyze_bundle(seal(tmp_path, members), binding, cache)
    incomplete = analysis.analyze_reports([report], repetitions=200)
    assert incomplete['comparisons'][0]['status'] == 'unavailable'
    reports = [dict(copy.deepcopy(report), seed=s) for s in (0, 1, 2)]
    result = analysis.analyze_reports(reports, repetitions=200)
    interval = result['comparisons'][0]
    assert interval['delta_pp'] == 0
    assert interval['conditional_object_bootstrap_low_pp'] == -100
    assert interval['conditional_object_bootstrap_high_pp'] == 100
    assert interval['n_original_object_clusters'] == 2
    altered = copy.deepcopy(report)
    altered['backbone'] = 'dgcnn'
    altered['batch_guards'][0]['geometry_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='shared'):
        analysis.analyze_reports([report, altered])


def test_validator_import_preserves_other_campaign():
    import sys
    import types
    previous = sys.modules.get('campaign')
    sentinel = types.ModuleType('campaign')
    sys.modules['campaign'] = sentinel
    try:
        sys.modules.pop('_chamfer_depth_20261030_analysis', None)
        assert callable(analysis._depth_validators().verify_cache)
        assert sys.modules['campaign'] is sentinel
    finally:
        if previous is None:
            sys.modules.pop('campaign', None)
        else:
            sys.modules['campaign'] = previous


def test_explicit_32_object_technical_smoke(tmp_path):
    members, binding, cache = make_fixture(tmp_path, smoke=True)
    report = analysis.analyze_bundle(seal(tmp_path, members), binding, cache)
    assert report['technical_smoke'] is True
    assert report['corruptions'] == ['gaussian', 'background']
    assert len(report['object_ids']) == 32
    assert analysis.analyze_reports([report])['comparisons'][0]['status'] == 'unavailable'


def test_rejects_resealed_trace_rng_not_matching_physical_cache(tmp_path):
    members, binding, cache = make_fixture(tmp_path)
    for name in list(members):
        if name.endswith('/diagnostics.json'):
            diagnostic = json.loads(members[name])
            diagnostic['trajectory']['initial_rng_sha256'] = 'b' * 64
            members[name] = encoded(diagnostic)
    with pytest.raises(ValueError, match='cache.*RNG|RNG.*cache'):
        analysis.analyze_bundle(seal(tmp_path, members), binding, cache)
