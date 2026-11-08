"""CPU acceptance of the separate actual-batch617 screen; never execute inference."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import re
import stat
import statistics
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'prepared/assets/source_snapshot'
REPO = HERE.parents[2]
sys.path.insert(0, str(SOURCE if (SOURCE/'research_artifacts.py').is_file() else REPO))
from scripts.analyze_scan_trajectory import Bundle
from research_artifacts import CORRUPTIONS


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def tensor_identity(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def arrays(bundle, name: str) -> dict:
    with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def validate_predictions(payload: dict):
    values = np.asarray(payload['trajectory_logits'], dtype=np.float64)
    assert values.ndim == 3 and values.shape[1:] == (4, 40) and np.isfinite(values).all()
    probabilities = np.exp(values-values.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    pm, terminal = probabilities.mean(1).argmax(-1), values[:, -1].argmax(-1)
    assert np.array_equal(payload['prediction'], pm) and np.array_equal(payload['probability_mean'], pm)
    assert np.array_equal(payload['terminal_prediction'], terminal) and np.array_equal(payload['logits'], values[:, -1])
    return pm, terminal


REMOTE = Path('/content/studies/gsd-lowband-quarter-fast-20261108-r6')
CHECKPOINT_SHA256 = '507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75'
EVALUATION_SHA256 = '4156e92e3e1206bdd31131e59f93c5b10f38c876e35558fbf6a4bfe41eadcfc2'
BATCHES = (8, 16, 32, 64)
ARMS = ('control', 'half0', 'M100')


def open_bundle(path: Path):
    # Full-rank B64 bases can exceed the old B2 member-size limit; keep finite bounds.
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        assert len(entries) <= 20000 and sum(e.file_size for e in entries) <= 32*1024**3
        assert len(entries) == len({e.filename for e in entries})
        for entry in entries:
            name = PurePosixPath(entry.filename)
            assert name.parts and not name.is_absolute() and '..' not in name.parts
            assert '\\' not in entry.filename and ':' not in entry.filename
            assert stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR)
            assert entry.file_size <= 2*1024**3
    return Bundle(path)


def context(root: Path) -> dict:
    root = Path(root)
    plan = json.loads((root/'operator/plan.json').read_text())
    assert plan['dataset'] == 'modelnet40_c' and plan['corruptions'] == list(CORRUPTIONS)
    evaluation = root/'operator'/plan['evaluation_file']
    assert identity(evaluation)['sha256'] == plan['evaluation_manifest_sha256'] == EVALUATION_SHA256
    ids = json.loads(evaluation.read_text())['indices']
    assert len(ids) == 617 and ids == sorted(set(ids)) and all(type(i) is int and 0 <= i < 2468 for i in ids)
    manifest = json.loads((root/'source/asset_manifest.json').read_text())
    assert identity(root/'source/asset_manifest.json')['sha256'] == plan['asset_manifest_sha256']
    assert identity(root/'source/source_snapshot.zip') == plan['source_snapshot']
    assert plan['source_snapshot']['sha256'] == manifest['source_snapshot_sha256']
    assert plan['checkpoint_sha256'] == CHECKPOINT_SHA256
    selected = json.loads((root/'batch_selection.json').read_text())
    batch = selected['batch_size']
    assert selected['status'] == 'selected' and type(batch) is int and batch in BATCHES
    assert selected['asset_manifest_sha256'] == plan['asset_manifest_sha256']
    assert selected['source_snapshot'] == plan['source_snapshot']
    calibration = root/'operator'/plan['calibration_file']
    assert identity(calibration)['sha256'] == plan['calibration_manifest_sha256']
    cal_ids = json.loads(calibration.read_text())['indices']
    assert len(cal_ids) == 64 and cal_ids == sorted(set(cal_ids)) and all(type(i) is int and 0 <= i < 2468 for i in cal_ids)
    frozen = json.loads((root/'calibration_frozen.json').read_text())
    assert frozen['status'] == 'frozen' and frozen['dataset'] == 'modelnet40_c' and frozen['batch_size'] == batch
    assert frozen['asset_manifest_sha256'] == plan['asset_manifest_sha256']
    assert frozen['checkpoint_sha256'] == CHECKPOINT_SHA256 and frozen['calibration_manifest_sha256'] == plan['calibration_manifest_sha256']
    assert frozen['source_snapshot'] == plan['source_snapshot'] and frozen['ratios_per_mode'] == 384
    assert frozen['batch_selection'] == identity(root/'batch_selection.json')
    expected = {(i, c, t) for i in cal_ids for c in ('gaussian', 'background')
        for t in ([9, 18, 27] if c == 'background' else [2, 4, 6])}
    rows = frozen['rows']
    assert len(rows) == 384 and {(r['id'], r['corruption'], r['checkpoint']) for r in rows} == expected
    for row in rows:
        cn, gn, ratio = (row[k] for k in ('scd_norm', 'gsd_norm', 'ratio'))
        assert all(type(v) in (int, float) and math.isfinite(v) for v in (cn, gn, ratio))
        assert cn > 1e-12 and gn > 1e-12 and ratio > 0 and math.isclose(ratio, cn/gn, rel_tol=1e-12)
    alpha = frozen['alpha']['100']
    assert type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0
    assert alpha == statistics.median(r['ratio'] for r in rows)
    return dict(plan=plan, manifest=manifest, indices=ids, batch_size=batch, alpha=alpha,
        batch_selection_identity=identity(root/'batch_selection.json'), freeze_identity=identity(root/'calibration_frozen.json'))


def validate_cache(raw: bytes, actual: int) -> dict:
    """Recount actual state tensors and the serialized RNG hash without unpickling."""
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata'].item()))
        assert metadata['schema_version'] == 1
        assert re.fullmatch('[0-9a-f]{64}', metadata['input_sha256'])
        assert re.fullmatch('[0-9a-f]{64}', metadata['rng_before_sha256'])
        for key in ('shape_latent', 'local_latent', 'noise'):
            value = saved[key]
            assert value.ndim == 4 and value.shape[0] == actual and value.shape[2:] == (1, 1)
            assert value.dtype in (np.float32, np.float64) and np.isfinite(value).all()
            assert tensor_identity(value) == metadata['tensor_hashes'][key]
        assert saved['local_latent'].shape == saved['noise'].shape == (actual, 8192, 1, 1)
        py = metadata['python_rng']
        py = (py[0], tuple(py[1]), py[2])
        ns = metadata['numpy_rng']
        ns = (ns[0], saved['numpy_rng_keys'], *ns[1:])
        assert type(metadata['cuda_devices']) is int and 0 <= metadata['cuda_devices'] <= 8
        cpu = saved['torch_rng_cpu']
        cuda = [saved['torch_rng_cuda_%d' % i] for i in range(metadata['cuda_devices'])]
        assert all(a.dtype == np.uint8 and a.ndim == 1 and a.size for a in [cpu, *cuda])
        digest = hashlib.sha256(pickle.dumps((py, ns, cpu, cuda), protocol=4)).hexdigest()
        assert digest == metadata['rng_after_sha256'], 'Actual cached RNG bytes changed'
        return dict(metadata=metadata, initial_h0_xyz=saved['local_latent'].reshape(actual, 2048, 4)[:, :, :3].copy())


BASIS_VALIDATOR_VERSION = 'basis-fp32-v2'


def basis_loss_comparison(terms: list, raw_loss: float, producer_dtype: np.dtype) -> dict:
    """New v2 rule: independent FP64 energies, source-dtype scalar left fold."""
    dtype = np.dtype(producer_dtype)
    assert dtype in (np.dtype('float32'), np.dtype('float64')) and terms
    assert math.isfinite(raw_loss) and all(math.isfinite(value) for value in terms)
    legacy = 0.
    replay = dtype.type(0.)
    for value in terms:
        legacy += float(value)
        replay = dtype.type(replay + dtype.type(value))
        assert np.isfinite(replay), 'Nonfinite producer-dtype basis reduction'
    error = abs(float(replay)-raw_loss)
    legacy_error = abs(legacy-raw_loss)
    assert math.isfinite(error) and error <= 1e-6, 'Independent physical basis loss failed (v2 producer-dtype reduction)'
    return dict(schema_version=2, validator_version=BASIS_VALIDATOR_VERSION,
        arithmetic='fp64_per_object_energy_then_producer_dtype_left_fold',
        producer_dtype=dtype.name, per_object_energy_dtype='float64',
        object_order='archived_batch_order', per_object_count=len(terms),
        producer_replayed_sum=float(replay), absolute_error=error, passed=True,
        legacy_fp64_sum=legacy, legacy_absolute_error=legacy_error,
        legacy_gate_passed=legacy_error <= 1e-6, tolerance=1e-6)


def load_run(path: Path, root: Path) -> dict:
    root = Path(root)
    bound = context(root)
    before = identity(path)
    bundle = open_bundle(path)
    try:
        assert {'command.txt', 'config.json', 'environment.txt', 'stdout.log', 'summary.csv', 'per_corruption.csv', 'notes.md'} <= bundle.names
        c = bundle.json('config.json')
        batch = bound['batch_size']
        assert c['status'] == 'partial' and c['diagnostic_complete'] is True
        assert (c['dataset'], c['backbone'], c['n_examples'], c['num_classes'], c['split']) == ('modelnet40_c', 'pointmae', 2468, 40, 'official_modelnet40_test')
        assert (c['seed'], c['severity'], c['batch_size']) == (0, 5, batch)
        assert c['benchmark_shard'] is True and c['corruptions'] == list(CORRUPTIONS)
        ids = c['selected_original_ids']
        assert type(c['count']) is int and 1 <= c['count'] <= 64 and len(ids) == c['count']
        assert ids == sorted(set(ids)) and set(ids) <= set(bound['indices'])
        assert all(type(i) is int for i in ids)
        assert c['completed_batches'] == {corruption: (len(ids)+batch-1)//batch for corruption in CORRUPTIONS}
        selection = root/'operator'/Path(c['index_manifest']).name
        assert bundle.read('selection_manifest.json') == selection.read_bytes()
        assert c['index_manifest_sha256'] == identity(selection)['sha256']
        assert json.loads(selection.read_text()) == c['selection_manifest'] and c['selection_manifest']['indices'] == ids
        assert c['guidance_protocol'] == dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False)
        assert (c['normal_reverse_steps'], c['background_reverse_steps']) == (8, 35)
        assert c['backend_policy'] == 'controlled' and c['runtime']['cudnn_deterministic'] is True
        assert all(c['runtime'][k] is False for k in ('cudnn_benchmark', 'cudnn_allow_tf32', 'cuda_matmul_allow_tf32', 'deterministic_algorithms'))
        assert all(c.get(k) is False for k in ('freeze_decoder_residual', 'update_norm_probes', 'native_decoder_audit', 'capture_h5', 'capture_native_trace', 'capture_shared_states'))
        assert c['skip_dense_filter_witness'] is True
        assert c['lion_state_before'] and c['lion_state_before'] == c['lion_state_after']
        assert c['classifier_state_before_sha256'] and c['classifier_state_before_sha256'] == c['classifier_state_after_sha256']
        assert c['pointmae_checkpoint'] == 'checkpoints/modelnet_jt.pth'
        load = c['checkpoint_load']
        assert load['strict'] is True and load['missing_keys'] == load['unexpected_keys'] == [] and load['checkpoint_sha256'] == CHECKPOINT_SHA256
        arm = {'control': 'control', 'live_control': 'half0', 'physical': 'M100'}[c['arm']]
        assert c['checkpoint_scd_weight'] == (1. if arm == 'control' else .5)
        assert c['weight'] == (.5*bound['alpha'] if arm == 'M100' else 0.)
        assert c['half_mix_alpha'] == (bound['alpha'] if arm == 'M100' else 0. if arm == 'half0' else None)
        assert c['lowband_modes'] == (None if arm == 'control' else 100)
        assert c['physical_graph'] == (None if arm == 'control' else dict(k=10, delta=.1, graph_gamma=.6, modes=100, profile='hard', beta=None))
        manifest = bound['manifest']
        assert c['asset_manifest_sha256'] == bound['plan']['asset_manifest_sha256'] and c['asset_manifest'] == manifest
        assert hashlib.sha256(bundle.read('source_snapshot.zip')).hexdigest() == manifest['source_snapshot_sha256']
        for name, expected in c['executed_source_files'].items():
            member = name[7:] if name.startswith('assets/') else 'source_snapshot/'+name[5:]
            assert name.startswith(('assets/', 'repo/')) and manifest['files'][member] == expected
            if name.startswith('repo/'):
                raw = bundle.read(member)
                assert dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == expected
        native = list(c['native_extensions'].values())+list(c['native_bootstrap']['bindings'].values())
        assert c['executed_source_files'] and native
        for entry in native:
            assert identity(root/'native_payload'/str(entry['path']).lstrip('/')) == {k: entry[k] for k in ('bytes', 'sha256')}
        assert Path(c['paired_state_root']) == REMOTE/('paired_states/B%d' % batch)
        common = {k: c[k] for k in ('executed_source_files', 'native_extensions', 'native_bootstrap', 'asset_manifest_sha256', 'dataset_provenance', 'guidance_protocol', 'runtime', 'lion_state_before', 'classifier_state_before_sha256')}
        cases, counts, costs, projections = {}, {}, [], []
        csvrows = list(csv.DictReader(io.StringIO(bundle.read('per_corruption.csv').decode())))
        assert len(csvrows) == 15 and {r['corruption'] for r in csvrows} == set(CORRUPTIONS)
        for corruption in CORRUPTIONS:
            parts = []
            for ordinal, start in enumerate(range(0, len(ids), batch)):
                selected, actual = ids[start:start+batch], len(ids[start:start+batch])
                prefix = 'batches/%s/batch%04d/' % (corruption, ordinal)
                p = arrays(bundle, prefix+'predictions.npz')
                pm, terminal = validate_predictions(p)
                assert p['index'].dtype.kind in 'iu' and p['index'].tolist() == selected
                assert p['label'].shape == (actual,) and p['label'].dtype.kind in 'iu' and ((p['label'] >= 0) & (p['label'] < 40)).all()
                assert p['trajectory_logits'].shape == (actual, 4, 40)
                steps = [9, 18, 27] if corruption == 'background' else [2, 4, 6]
                assert p['candidate_loop_counts'].tolist() == steps+[35 if corruption == 'background' else 8]
                d = bundle.json(prefix+'diagnostics.json')
                t = d['trajectory']
                assert d['original_ids'] == selected and t['complete'] and t['probe_rng_unchanged'] and t['probe_buffers_unchanged']
                ref = arrays(bundle, prefix+'reference.npz')
                assert set(ref) == {'prepared_x', 'initial_h0_xyz'} and all(v.shape == (actual, 2048, 3) and np.isfinite(v).all() for v in ref.values())
                assert tensor_identity(ref['prepared_x']) == d['prepared_x_sha256']
                relative = Path('paired_states/B%d/modelnet40_c' % batch)/corruption/(hashlib.sha256(np.asarray(selected, dtype=np.int64).tobytes()).hexdigest()+'.npz')
                assert Path(d['paired_state']['path']) == REMOTE/relative
                assert identity(root/relative) == {k: d['paired_state'][k] for k in ('bytes', 'sha256')}
                saved = validate_cache((root/relative).read_bytes(), actual)
                meta = saved['metadata']
                assert meta['input_sha256'] == d['prepared_x_sha256'] and meta['rng_after_sha256'] == t['initial_rng_sha256']
                assert all(meta['tensor_hashes'][k] == t['initial_hashes'][k] for k in ('shape_latent', 'local_latent', 'noise'))
                assert np.array_equal(saved['initial_h0_xyz'], ref['initial_h0_xyz'])
                assert d['scheduler']['final_alpha_cumprod'] == 1 and d['scheduler']['config']['set_alpha_to_one'] is True
                first = arrays(bundle, prefix+'decoded_first_checkpoint.npz')['decoded_xyz']
                assert first.shape == (actual, 2048, 3) and np.isfinite(first).all()
                assert first.dtype == ref['prepared_x'].dtype
                assert tensor_identity(first) == t['probes'][0]['decoded_geometry_sha256'], 'First decoded witness differs from first probe'
                graph = bundle.json(prefix+'graph.json') if prefix+'graph.json' in bundle.names else None
                assert (graph is not None) == (arm != 'control')
                if graph is not None:
                    spectral = t['decoded_spectral']
                    assert spectral['loss_gradient_mode'] == 'full_decoder_gradient' and spectral['decoder_jacobian_in_loss']
                    assert spectral['reference_xyz_sha256'] == graph['reference_xyz_sha256'] == d['prepared_x_sha256']
                    assert [r['loop_count'] for r in spectral['records']] == steps
                    assert all(r['decoded_geometry_reused'] and not r['classifier_gradients'] for r in spectral['records'])
                    assert len(graph['diagnostics']) == len(graph['filter_sha256']) == len(graph['basis_sha256']) == len(graph['active_vertex_indices']) == actual
                    assert all(re.fullmatch('[0-9a-f]{64}', sha) for sha in graph['filter_sha256'])
                    bases = arrays(bundle, prefix+'physical_bases.npz')
                    assert set(bases) == {'basis_%d' % i for i in range(actual)}
                    terms = []
                    for i, row in enumerate(graph['diagnostics']):
                        rank = row['actual_rank']
                        assert type(rank) is int and 0 <= rank <= 2048 and row['requested_rank'] == 100 and row['spectral_profile'] == 'hard'
                        u = bases['basis_%d' % i]
                        assert u.shape == (2048, rank) and u.dtype in (np.float32, np.float64) and u.dtype == ref['prepared_x'].dtype and np.isfinite(u).all()
                        assert tensor_identity(u) == graph['basis_sha256'][i]
                        active = graph['active_vertex_indices'][i]
                        assert active == sorted(set(active)) and all(type(v) is int and 0 <= v < 2048 for v in active)
                        e = first[i].astype(np.float64)-ref['prepared_x'][i].astype(np.float64)
                        terms.append(float(np.square(u.astype(np.float64).T@e).sum()/(3*2048)))
                    assert prefix+'physical_filters.npz' not in bundle.names, 'Dense witness was requested to be omitted'
                    comparison = basis_loss_comparison(terms, spectral['records'][0]['raw_loss'], ref['prepared_x'].dtype)
                    projections.append(dict(case=corruption+'/'+str(ordinal), **comparison))
                paired = dict(scheduler=d['scheduler'], initial=t['initial_hashes'], initial_rng=t['initial_rng_sha256'],
                    paired=d['paired_state']['sha256'], prepared=d['prepared_x_sha256'], input_rng=d['input_rng_before_sha256'],
                    output_rng=d['reconstruction_rng_after_sha256'], reference={k: tensor_identity(v) for k, v in ref.items()})
                cases[corruption+'/'+str(ordinal)] = dict(index=p['index'], label=p['label'], pm=pm, terminal=terminal,
                    logits=p['trajectory_logits'], first=first, identity=paired, physical_metric_identity=graph)
                costs.append(dict(corruption=corruption, batch=ordinal, **d.get('timing', {})))
                parts.append(p)
            correct = sum(int((p['prediction'] == p['label']).sum()) for p in parts)
            row = next(r for r in csvrows if r['corruption'] == corruption)
            assert int(row['n_examples']) == len(ids) and int(row['n_correct']) == correct and float(row['accuracy']) == correct/len(ids)
            aggregate = arrays(bundle, corruption+'_predictions.npz')
            for key in ('index', 'label', 'prediction', 'terminal_prediction', 'logits', 'trajectory_logits'):
                assert np.array_equal(aggregate[key], np.concatenate([p[key] for p in parts]))
            counts[corruption] = dict(n=len(ids), pm_correct=correct,
                terminal_correct=sum(int((p['terminal_prediction'] == p['label']).sum()) for p in parts),
                reconstruction_seconds=float(row['runtime_seconds']), peak_gpu_memory_mb=float(row['peak_gpu_memory_mb']))
        summary = list(csv.DictReader(io.StringIO(bundle.read('summary.csv').decode())))
        assert len(summary) == 1 and int(summary[0]['total_examples']) == len(ids)*15 and int(summary[0]['n_corruptions']) == 15
        assert int(summary[0]['total_correct']) == sum(r['pm_correct'] for r in counts.values())
        assert np.isclose(float(summary[0]['macro_accuracy']), np.mean([r['pm_correct']/r['n'] for r in counts.values()]), rtol=0, atol=1e-12)
        assert all(math.isfinite(v) and v >= 0 for v in [c['wall_seconds'], *[r[k] for r in counts.values() for k in ('reconstruction_seconds', 'peak_gpu_memory_mb')]])
        assert identity(path) == before, 'Accepted raw ZIP changed during validation'
        return dict(basis_validator_version=BASIS_VALIDATOR_VERSION, basis_validator_source=identity(Path(__file__)), config=c, arm=arm, control_identity=common, cases=cases, counts=counts, costs=costs,
            projections=projections, archive=before, batch_size=batch, freeze_identity=bound['freeze_identity'], batch_selection_identity=bound['batch_selection_identity'])
    finally:
        bundle.close()


def receipt_details(result: dict) -> dict:
    cases = []
    pairing, graphs = {}, {}
    for key, case in result['cases'].items():
        graph = case['physical_metric_identity']
        corruption = key.split('/')[0]
        pairing.setdefault(corruption, []).append(case['identity'])
        graphs.setdefault(corruption, [])
        if graph is not None:
            graphs[corruption].append({k: graph[k] for k in ('reference_xyz_sha256', 'filter_sha256', 'basis_sha256', 'active_vertex_indices')})
        cases.append(dict(case=key, ids=case['index'].tolist(), labels=case['label'].tolist(), pm=case['pm'].tolist(),
            terminal=case['terminal'].tolist(), paired_identity=case['identity'], filter_sha256=graph['filter_sha256'] if graph else None,
            rank=[g['actual_rank'] for g in graph['diagnostics']] if graph else None))
    common = {k: result['config'][k] for k in ('executed_source_files', 'native_extensions', 'native_bootstrap', 'asset_manifest_sha256',
        'lion_state_before', 'classifier_state_before_sha256', 'runtime')}
    return dict(basis_validator_version=BASIS_VALIDATOR_VERSION, basis_validator_source=result['basis_validator_source'], basis_projections=result['projections'], counts=result['counts'], cases=cases, costs=result['costs'], control_identity=result['control_identity'],
        common=common, pairing=pairing, graphs=graphs,
        worker_wall_seconds=result['config']['wall_seconds'], batch_size=result['batch_size'], freeze_identity=result['freeze_identity'],
        batch_selection_identity=result['batch_selection_identity'], basis_checks=len(result['projections']), basis_passed=True)


def check_pairing(left: dict, right: dict) -> None:
    assert left['control_identity'] == right['control_identity'], 'Paired source/native/model/backend identity differs'
    assert left['common'] == right['common']
    a = {case['case']: case for case in left['cases']}
    b = {case['case']: case for case in right['cases']}
    assert len(a) == len(left['cases']) and len(b) == len(right['cases']) and set(a) == set(b)
    for key, case in a.items():
        other = b[key]
        assert all(case[k] == other[k] for k in ('ids', 'labels', 'paired_identity')), 'Paired object/label/cache/RNG differs'
        if case['filter_sha256'] is not None and other['filter_sha256'] is not None:
            assert all(case[k] == other[k] for k in ('filter_sha256', 'rank')), 'Matched physical graph differs'
    if any(c['filter_sha256'] is not None for c in a.values()) and any(c['filter_sha256'] is not None for c in b.values()):
        assert left['graphs'] == right['graphs'], 'Matched physical reference/filter/basis/support differs'


def accept_cache(archive: Path, root: Path, shard: dict) -> dict:
    root = Path(root)
    bound = context(root)
    batch = bound['batch_size']
    selection = json.loads((root/'operator'/shard['selection_file']).read_text())
    ids = selection['indices']
    assert len(ids) == shard['count'] and set(ids) <= set(bound['indices'])
    bundle = zipfile.ZipFile(archive)
    try:
        entries = bundle.infolist()
        assert len(entries) <= 2000 and sum(e.file_size for e in entries) <= 2*1024**3
        assert len(entries) == len({e.filename for e in entries}) and bundle.testzip() is None
        for entry in entries:
            p = PurePosixPath(entry.filename)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in entry.filename and ':' not in entry.filename
            assert stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR)
        manifest = json.loads(bundle.read('package_manifest.json'))
        assert set(bundle.namelist()) == set(manifest)|{'package_manifest.json'}
        for name, expected in manifest.items():
            raw = bundle.read(name)
            assert dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == expected
        report_name = 'records/cache_preparation_'+shard['label']+'.json'
        report = json.loads(bundle.read(report_name))
        expected = {(c, tuple(ids[start:start+batch])) for c in CORRUPTIONS for start in range(0, len(ids), batch)}
        assert report['status'] == 'complete' and report['selection']['indices'] == ids
        assert len(report['records']) == len(expected) and {(r['corruption'], tuple(r['ids'])) for r in report['records']} == expected
        assert {r['member'] for r in report['records']}|{report_name} == set(manifest)
        for row in report['records']:
            relative = Path('paired_states/B%d/modelnet40_c' % batch)/row['corruption']/(hashlib.sha256(np.asarray(row['ids'], dtype=np.int64).tobytes()).hexdigest()+'.npz')
            assert row['member'] == relative.as_posix() and Path(row['path']) == REMOTE/relative
            assert manifest[row['member']] == {k: row[k] for k in ('bytes', 'sha256')}
            metadata = validate_cache(bundle.read(row['member']), len(row['ids']))['metadata']
            assert metadata['rng_after_sha256'] == row['rng_after']
        for name, expected_id in manifest.items():
            target = root/name
            assert not target.is_symlink() and all(not p.is_symlink() for p in target.parents)
            if target.exists():
                assert identity(target) == expected_id, 'Preserve mismatched existing local cache'
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(bundle.read(name))
        return dict(report, cache_count=len(expected), selection_ids=ids, batch_size=batch)
    finally:
        bundle.close()
