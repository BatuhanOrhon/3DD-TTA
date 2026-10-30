"""Synthetic CPU guard fixtures; these are never research acceptance evidence."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c
import analyze as a
import operations as op


def binding():
    source_files = {('repo/' + p): c.identity(c.REPO / p)
                    for p in ('tta.py', 'shared_trajectory.py', 'scripts/eval_chamfer_depth.py')}
    return dict(python='/synthetic/python', repo_root='/synthetic/repo', dataset_root='/synthetic/data',
        label_path='/synthetic/data/label.npy', asset_root='/synthetic/assets', pointmae_config='/synthetic/mae.yaml',
        pointmae_checkpoint='/synthetic/mae.pth', required_source_files=source_files,
        expected_config=dict(source_snapshot_sha256='a' * 64, asset_manifest_sha256='b' * 64,
            checkpoint_load=dict(strict=True, checkpoint_sha256='c' * 64),
            dataset_provenance=dict(n_examples=2468, files={'synthetic%d.npy' % i: dict(bytes=1, sha256='d' * 64) for i in range(16)})))


@pytest.fixture
def plan_path(tmp_path):
    selection = tmp_path / 'ids'
    c.reserve(selection)
    bind = tmp_path / 'binding.json'
    c.write_new(bind, binding())
    path = tmp_path / 'plan.json'
    c.register(bind, selection, path, 10800, 9000)
    return path


def scores(plan_path, configs, score=lambda v: -v['normal_reverse_steps']):
    return [dict(config_id=c.config_id(v), status='accepted', stage='development', readout='probability_mean',
                 seed=0, accuracy_pct=float(score(v)), bundle_sha256='e' * 64,
                 plan_sha256=c.identity(plan_path)['sha256']) for v in configs]


def fixture_bundle(tmp_path, plan, job):
    """Small constant latents allow meaningful actual-byte/RNG checks without torch."""
    selection = plan['selections'][job['split']]['manifest']
    smoke = job.get('technical_smoke', False)
    if smoke:
        selection = dict(selection, stage='smoke', indices=plan['technical_smoke']['indices'])
    ids = selection['indices']
    folder = tmp_path / 'raw_bundle'
    folder.mkdir()
    c.write_new(folder / 'selection_manifest.json', selection)
    cfg = copy.deepcopy(plan['binding']['expected_config'])
    cfg.update(method='3dd_chamfer_depth_study', dataset='modelnet40_c', backbone='pointmae', batch_size=32,
               severity=5, seed=job['seed'], technical_smoke=smoke, status='partial' if smoke else 'complete',
               normal_reverse_steps=job['config']['normal_reverse_steps'], background_reverse_steps=35,
               guidance_protocol=dict(gamma=job['config']['gamma'], eta=job['config']['eta'], p=job['config']['p'], total=100, scd_normalize=False),
               guidance_rate_mapping=dict(gamma='local_latent', eta='conditioning_style'),
               trajectory_selection=dict(temperature=1., style='original encoded shape_latent for every clean and terminal decode', labels_used=False, early_stopping=False),
               selected_original_ids=ids, evaluation_n_examples=len(ids), selection_manifest_identity=c.identity(folder / 'selection_manifest.json'),
               classifier_state_before_sha256='f' * 64, classifier_state_after_sha256='f' * 64,
               lion_state_before={'vae': 'f' * 64}, lion_state_after={'vae': 'f' * 64},
               executed_source_files=plan['binding']['required_source_files'],
               worker_wall_seconds_including_artifact_io=1.,
               batch_post_inference_policy_artifact_seconds={'gaussian': [.01], 'background': [.01]},
               corruptions=['gaussian', 'background'] if smoke else list(a.CORRUPTIONS),
               paired_state_root='/synthetic/study/' + ('paired-smoke' if smoke else 'paired') + '/seed%d' % job['seed'])
    for name in cfg['executed_source_files']:
        destination = folder / 'source_snapshot' / name[len('repo/'):]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((c.REPO / name[len('repo/'):]).read_bytes())
    c.write_new(folder / 'config.json', cfg)
    for corruption in cfg['corruptions']:
        depth = 35 if corruption == 'background' else job['config']['normal_reverse_steps']
        steps = a.clean_steps(depth)
        names = ['clean_%02d' % s for s in steps] + ['terminal']
        for batch, start in enumerate(range(0, len(ids), 32)):
            batch_ids = ids[start:start + 32]
            n = len(batch_ids)
            directory = folder / 'batches' / corruption / ('batch%04d' % batch)
            directory.mkdir(parents=True)
            initial = dict(input_points='f' * 64, style='9' * 64, noisy_local='1' * 64)
            arrays = dict(shape_latent=np.zeros((n, 4, 1, 1), dtype=np.float32),
                          local_latent=np.zeros((n, 8192, 1, 1), dtype=np.float32),
                          noise=np.ones((n, 8192, 1, 1), dtype=np.float32))
            for key, values in arrays.items():
                initial[key] = hashlib.sha256(str(values.dtype).encode() + str(values.shape).encode() + values.tobytes()).hexdigest()
            py, nr = random.Random(0).getstate(), np.random.RandomState(0).get_state()
            cpu, cuda = np.arange(8, dtype=np.uint8), np.arange(5, dtype=np.uint8)
            rng = (py, nr, cpu, [cuda])
            metadata = dict(schema_version=1, cuda_devices=1, input_sha256=initial['input_points'],
                rng_before_sha256='2' * 64, rng_after_sha256=hashlib.sha256(pickle.dumps(rng, protocol=4)).hexdigest(),
                python_rng=py, numpy_rng=[nr[0], *nr[2:]], tensor_hashes={k: initial[k] for k in arrays})
            filename = hashlib.sha256(np.asarray(batch_ids, dtype=np.int64).tobytes()).hexdigest() + '.npz'
            cache_path = tmp_path / ('paired-smoke' if smoke else 'paired') / ('seed%d' % job['seed']) / 'modelnet40_c' / corruption / filename
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **arrays, metadata=np.asarray(json.dumps(metadata)),
                                numpy_rng_keys=nr[1], torch_rng_cpu=cpu, torch_rng_cuda_0=cuda)
            logits = np.zeros((n, len(names), 40), dtype=np.float32)
            logits[:, :, 0] = 2.
            np.savez_compressed(directory / 'predictions.npz', index=np.asarray(batch_ids), label=np.zeros(n, dtype=np.int64),
                logits=logits[:, -1], prediction=np.zeros(n, dtype=np.int64), trajectory_logits=logits,
                candidate_names=np.asarray(names), candidate_loop_counts=np.asarray(steps + [depth]), **a.policies(logits, len(names) - 1))
            trace = dict(complete=True, horizon=depth, candidate_names=names, candidate_loop_counts=steps + [depth],
                         initial_hashes=initial, final_hashes={'latent': '3' * 64, 'style': initial['shape_latent']}, times={'trajectory_probe': .02},
                         probes=[dict(loop_count=s, timestep=(depth - s) * 10, rng_unchanged=True,
                             buffers_unchanged=True, live_inputs_unchanged=True, rng_before_sha256='4' * 64,
                             rng_after_sha256='4' * 64, buffers_before={'vae': {}}, buffers_after={'vae': {}},
                             style_sha256=initial['shape_latent']) for s in steps])
            diagnostic = dict(trajectory=trace, scheduler=dict(active_timesteps=list(range((depth - 1) * 10, -1, -10)),
                reverse_steps=depth, initial_timestep=(depth - 1) * 10, config={'set_alpha_to_one': True}, final_alpha_cumprod=1.),
                shared_reconstruction_seconds=.1, head_seconds=.01, whole_process_peak_gpu_memory_mb=12.,
                preparation_mode='fresh_encode', rng_unchanged=True, paired_state=dict(path=str(Path(cfg['paired_state_root']) / 'modelnet40_c' / corruption / filename), **c.identity(cache_path)),
                executed_source_sha256=hashlib.sha256(json.dumps(cfg['executed_source_files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                source_snapshot_sha256=cfg['source_snapshot_sha256'], asset_manifest_sha256=cfg['asset_manifest_sha256'], checkpoint_load=cfg['checkpoint_load'])
            c.write_new(directory / 'diagnostics.json', diagnostic)
    seal_fixture(folder)
    return folder.with_suffix('.zip'), folder


def seal_fixture(folder):
    manifest = {str(p.relative_to(folder)): c.identity(p) for p in folder.rglob('*') if p.is_file() and p.name != 'bundle_manifest.json'}
    (folder / 'bundle_manifest.json').write_bytes(c.encoded(manifest))
    with zipfile.ZipFile(folder.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in folder.rglob('*'):
            if path.is_file():
                archive.write(path, str(Path(folder.name) / path.relative_to(folder)))


def smoke_job(depth=1):
    value = c.config(depth)
    return dict(config=value, config_id=c.config_id(value), seed=0, split='development', technical_smoke=True, slot_id='fixture')


def test_reserved_ids_reproducible_disjoint_and_hashed(tmp_path):
    c.reserve(tmp_path / 'a')
    c.reserve(tmp_path / 'b')
    for name in ('development', 'heldout'):
        assert c.identity(tmp_path / 'a' / ('modelnet40_c_' + name + '_ids.json')) == c.identity(tmp_path / 'b' / ('modelnet40_c_' + name + '_ids.json'))
    dev = c.read(tmp_path / 'a/modelnet40_c_development_ids.json')['indices']
    held = c.read(tmp_path / 'a/modelnet40_c_heldout_ids.json')['indices']
    old = c.read(tmp_path / 'a/reservation.json')['excluded_prior_ids']
    assert (len(dev), len(held), len(old)) == (128, 256, 280)
    assert not set(dev) & set(held) and not (set(dev) | set(held)) & set(old)
    with pytest.raises(ValueError, match='exists'):
        c.reserve(tmp_path / 'a')


def test_adaptive_grid_max32_and_independent_rates(plan_path, tmp_path):
    path = tmp_path / 'scores.json'
    rows = scores(plan_path, c.default_configs())
    path.write_bytes(c.encoded(rows))
    rates = c.stage(plan_path, path, 'rates')
    assert len(rates['jobs']) == 16 and len(rates['reused_config_ids']) == 2
    assert {j['config']['normal_reverse_steps'] for j in rates['jobs']} == {1, 2}
    assert any(j['config']['eta'] != j['config']['gamma'] for j in rates['jobs'])
    rows += scores(plan_path, [j['config'] for j in rates['jobs']])
    path.write_bytes(c.encoded(rows))
    fractions = c.stage(plan_path, path, 'fraction')
    assert len(fractions['jobs']) == 6 and len(fractions['reused_config_ids']) == 2
    assert {j['config']['p'] for j in fractions['jobs']} == {.9, .96, .99}
    rows += scores(plan_path, [j['config'] for j in fractions['jobs']])
    path.write_bytes(c.encoded(rows))
    confirm = c.stage(plan_path, path, 'confirmation')
    assert len(rows) == len({r['config_id'] for r in rows}) == 32
    assert len(confirm['frozen_conditions']) == 3 and len(confirm['jobs']) == 9
    assert confirm['condition_roles'][1]['config_id'] == c.config_id(c.config(1))
    assert {j['seed'] for j in confirm['jobs']} == {0, 1, 2}


def test_confirmation_duplicates_reused(plan_path, tmp_path):
    path = tmp_path / 'scores.json'
    metric = lambda v: 100. if v == c.config(5) else (99. if v == c.config(6) else 0.)
    rows = scores(plan_path, c.default_configs(), metric)
    path.write_bytes(c.encoded(rows))
    rows += scores(plan_path, [j['config'] for j in c.stage(plan_path, path, 'rates')['jobs']], metric)
    path.write_bytes(c.encoded(rows))
    rows += scores(plan_path, [j['config'] for j in c.stage(plan_path, path, 'fraction')['jobs']], metric)
    path.write_bytes(c.encoded(rows))
    result = c.stage(plan_path, path, 'confirmation')
    assert len(result['frozen_conditions']) == 1 and len(result['jobs']) == 3
    assert len(result['condition_roles']) == 3  # aliases preserve all predeclared roles


@pytest.mark.parametrize('mutation', ['failed', 'different_readout', 'different_plan', 'missing'])
def test_no_selection_from_unavailable_or_wrong_endpoint(plan_path, tmp_path, mutation):
    rows = scores(plan_path, c.default_configs())
    if mutation == 'failed':
        rows[0]['status'] = 'unavailable'
    elif mutation == 'different_readout':
        rows[0]['readout'] = 'minimum_entropy'
    elif mutation == 'different_plan':
        rows[0]['plan_sha256'] = '9' * 64
    else:
        rows.pop()
    path = tmp_path / 'scores.json'
    path.write_bytes(c.encoded(rows))
    with pytest.raises(ValueError):
        c.stage(plan_path, path, 'rates')


def test_approval_bound_to_final_plan_source_and_whole_cap(plan_path, tmp_path, monkeypatch):
    plan = c.read(plan_path)
    approval = dict(approved=True, user_statement='synthetic fixture only', approved_at_utc='synthetic',
        plan_sha256=c.identity(plan_path)['sha256'], source_snapshot_sha256='a' * 64, cumulative_wall_cap_seconds=10800)
    assert c.check_approval(plan_path, approval) == plan
    for field in ('plan_sha256', 'source_snapshot_sha256', 'cumulative_wall_cap_seconds'):
        bad = dict(approval, **{field: 'changed'})
        with pytest.raises(ValueError, match='bound'):
            c.check_approval(plan_path, bad)
    c.write_new(tmp_path / 'approval.json', {})
    monkeypatch.setattr(op.subprocess, 'Popen', lambda *x, **y: pytest.fail('Must not launch'))
    with pytest.raises(ValueError, match='approval'):
        c.run_phase(plan_path, tmp_path / 'missing-stage.json', None, tmp_path / 'approval.json', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_ledger_chained_cumulative_and_interruption_stop(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    rows = []
    c.append_event(path, rows, event='phase_started', plan_sha256='a' * 64)
    with pytest.raises(ValueError, match='Unresolved'):
        c.ledger_events(path)
    c.append_event(path, rows, event='job_failed', plan_sha256='a' * 64)
    c.append_event(path, rows, event='phase_finished', elapsed_seconds=12.5, plan_sha256='a' * 64)
    assert c.ledger_events(path)[1] == 12.5
    values = path.read_text().replace('12.5', '2.5')
    path.write_text(values)
    # Tampering with the final row is detected by the next durable event linkage.
    c.append_event(path, rows, event='phase_started', plan_sha256='a' * 64)
    with pytest.raises(ValueError, match='chain'):
        c.ledger_events(path)


def test_smoke_worker_command_and_distinct_cache(plan_path, tmp_path):
    plan = c.read(plan_path)
    smoke = c.stage(plan_path, None, 'smoke')
    assert len(smoke['jobs']) == 3
    argv = c.worker_command(plan, smoke['jobs'][0], '/raw', '/ids', '/paired-smoke')
    assert argv[-4:] == ['--technical-smoke', '--corruptions', 'gaussian', 'background']
    assert argv[argv.index('--paired-state-root') + 1] == '/paired-smoke/seed0'


@pytest.mark.parametrize('depth,expected', [(1, [1]), (2, [1, 2]), (3, [1, 2, 3]), (10, [3, 5, 8]), (35, [9, 18, 27])])
def test_short_depth_unique_candidates(depth, expected):
    assert a.clean_steps(depth) == expected


def test_real_bytes_fixture_dynamic_n1_and_background35(plan_path, tmp_path):
    plan = c.read(plan_path)
    path, folder = fixture_bundle(tmp_path, plan, smoke_job())
    report = a.analyze_bundle(path, plan, smoke_job(), tmp_path)
    assert report['status'] == 'accepted' and report['stage'] == 'smoke'
    assert len(report['costs']) == 2 and all(r['accuracy_pct'] == 100. for r in report['aggregate'])
    assert len(report['cases']) == 32 * 2 * 3


@pytest.mark.parametrize('mutation', ['depth', 'style', 'rng', 'source', 'readout', 'cache', 'missing'])
def test_raw_guard_rejects_contract_drift(plan_path, tmp_path, mutation):
    plan = c.read(plan_path)
    path, folder = fixture_bundle(tmp_path, plan, smoke_job())
    diagnostic_path = folder / 'batches/gaussian/batch0000/diagnostics.json'
    diagnostic = c.read(diagnostic_path)
    if mutation == 'depth':
        diagnostic['scheduler']['active_timesteps'] = [40]
    elif mutation == 'style':
        diagnostic['trajectory']['probes'][0]['style_sha256'] = '0' * 64
    elif mutation == 'rng':
        diagnostic['trajectory']['probes'][0]['rng_after_sha256'] = '0' * 64
    elif mutation == 'source':
        diagnostic['executed_source_sha256'] = '0' * 64
    elif mutation == 'readout':
        target = folder / 'batches/gaussian/batch0000/predictions.npz'
        with np.load(target, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays['probability_mean'][0] = 1
        np.savez_compressed(target, **arrays)
    elif mutation == 'cache':
        next((tmp_path / 'paired-smoke').rglob('*.npz')).write_bytes(b'corrupted physical cache')
    else:
        (folder / 'batches/background/batch0000/predictions.npz').unlink()
    diagnostic_path.write_bytes(c.encoded(diagnostic))
    seal_fixture(folder)
    with pytest.raises((ValueError, KeyError)):
        a.analyze_bundle(path, plan, smoke_job(), tmp_path)


def test_bootstrap_resamples_objects_jointly_across_seeds_and_corruptions():
    left = np.zeros((3, 15, 256), dtype=bool)
    right = left.copy()
    left[:, :, :128] = True
    joint = a.compare(left, right, 700)
    single = a.compare(left[:1], right[:1], 700)
    assert joint['n_original_object_clusters'] == 256 and joint['prediction_events'] == 3 * 15 * 256
    assert joint['delta_pp'] == 50.
    assert joint['conditional_object_bootstrap_low_pp'] == single['conditional_object_bootstrap_low_pp']
    left[1] = False
    right[1, :, :128] = True
    left[2] = False
    assert a.compare(left, right, 700)['delta_pp'] == 0.


def test_unavailable_confirmation_never_uses_surviving_seeds():
    with pytest.raises(ValueError, match='Expected paired'):
        a.compare(np.zeros((2, 15, 256)), np.zeros((2, 15, 256)), 100)


def approved_fixture(plan_path, tmp_path):
    approval = tmp_path / 'approval.json'
    c.write_new(approval, dict(approved=True, user_statement='synthetic CPU fixture only', approved_at_utc='synthetic',
        plan_sha256=c.identity(plan_path)['sha256'], source_snapshot_sha256='a' * 64, cumulative_wall_cap_seconds=10800))
    return approval


@pytest.mark.parametrize('failure_mode', ['numerical', 'timeout'])
def test_failed_slot_stops_preserves_and_never_replaced(plan_path, tmp_path, monkeypatch, failure_mode):
    monkeypatch.setattr(op, 'verify_preflight', lambda plan: None)
    plan = c.read(plan_path)
    plan['drive_root'] = str(tmp_path / 'drive')
    plan_path.write_bytes(c.encoded(plan))
    stage_path = tmp_path / 'smoke.json'
    c.write_new(stage_path, c.stage(plan_path, None, 'smoke'))
    calls = []
    def fail_worker(*args):
        calls.append(args)
        Path(args[2]).write_text('synthetic numerical failure\n')
        partial = Path(args[2]).parent / 'raw/modelnet40_c/run/batches/gaussian/batch0000/predictions.npz'
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b'unsealed timeout-prefix raw bytes')
        if failure_mode == 'timeout':
            raise op.subprocess.TimeoutExpired(args[0], .1)
        return 1
    monkeypatch.setattr(op, 'run_worker', fail_worker)
    root = tmp_path / 'campaign'
    approval = approved_fixture(plan_path, tmp_path)
    with pytest.raises((ValueError, op.subprocess.TimeoutExpired)):
        c.run_phase(plan_path, stage_path, None, approval, root)
    assert len(calls) == 1 and len(list((root / 'slots').iterdir())) == 1
    assert next((root / 'slots').glob('*/failure.json')).exists()
    rows, spent = c.ledger_events(root / 'wall_ledger.jsonl')
    assert spent >= 60. and any(r['event'] == 'job_failed' for r in rows)
    receipts = list((tmp_path / 'drive').rglob('receipt.json'))
    assert len(receipts) == 2 and any('failure.json' in n for receipt in receipts for n in c.read(receipt)['files'])
    raw_copies = list((tmp_path / 'drive').rglob('predictions.npz'))
    assert len(raw_copies) == 1 and raw_copies[0].read_bytes() == b'unsealed timeout-prefix raw bytes'
    assert any('failure raw recursively preserved' in c.read(receipt)['raw_preservation'] for receipt in receipts)
    with pytest.raises(ValueError, match='already attempted'):
        c.run_phase(plan_path, stage_path, None, approval, root)
    assert len(calls) == 1


def test_development_requires_accepted_smoke(plan_path, tmp_path, monkeypatch):
    monkeypatch.setattr(op, 'verify_preflight', lambda plan: None)
    path = tmp_path / 'depth.json'
    c.write_new(path, c.stage(plan_path, None, 'depth'))
    monkeypatch.setattr(op, 'run_worker', lambda *args: pytest.fail('No research without smoke'))
    root = tmp_path / 'root'
    with pytest.raises(ValueError, match='technical smoke required'):
        c.run_phase(plan_path, path, None, approved_fixture(plan_path, tmp_path), root)
    assert not (root / 'wall_ledger.jsonl').exists()


def test_drive_cache_copy_verification_and_immutable_reuse(tmp_path):
    root = tmp_path / 'root'
    cache = root / 'paired/seed0/modelnet40_c/gaussian/cache.npz'
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'synthetic physical cache bytes')
    (root / 'wall_ledger.jsonl').write_text('synthetic ledger\n')
    proof = dict(relative_path=str(cache.relative_to(root)), **c.identity(cache))
    first = Path(op.persist_checkpoint(root, tmp_path / 'drive', 'a' * 64, 1, expected_caches=[proof]))
    assert c.read(first / 'receipt.json')['physical_cache_files'][proof['relative_path']] == c.identity(cache)
    second = Path(op.persist_checkpoint(root, tmp_path / 'drive', 'a' * 64, 2, expected_caches=[proof]))
    assert second != first
    with pytest.raises(ValueError, match='identity changed'):
        op.persist_checkpoint(root, tmp_path / 'drive', 'a' * 64, 3,
                              expected_caches=[dict(proof, sha256='b' * 64)])
    assert c.identity(tmp_path / 'drive/campaign' / ('a' * 64) / proof['relative_path']) == c.identity(cache)


def test_streamed_cpu_worker_and_timeout(tmp_path, capsys):
    log = tmp_path / 'console.log'
    assert op.run_worker([sys.executable, '-u', '-c', "print('gaussian batch0 complete', flush=True)"],
                         tmp_path, log, 5, 'CPU-fixture') == 0
    assert 'gaussian batch0 complete' in log.read_text() and 'gaussian batch0 complete' in capsys.readouterr().out
    with pytest.raises(op.subprocess.TimeoutExpired):
        op.run_worker([sys.executable, '-u', '-c', 'import time; time.sleep(2)'],
                      tmp_path, tmp_path / 'timeout.log', .1, 'CPU-timeout-fixture')


def test_actual_asset_prefix_source_archive_contract(plan_path, tmp_path):
    plan = c.read(plan_path)
    path, folder = fixture_bundle(tmp_path, plan, smoke_job())
    snapshot = folder / 'source_snapshot.zip'
    with zipfile.ZipFile(snapshot, 'w') as archive:
        for source in (folder / 'source_snapshot').rglob('*'):
            if source.is_file():
                archive.write(source, str(source.relative_to(folder)))
    expected = plan['binding']['expected_config']
    expected['source_snapshot_sha256'] = c.identity(snapshot)['sha256']
    required = {'assets/source_snapshot/' + k[len('repo/'):]: v for k, v in plan['binding']['required_source_files'].items()}
    plan['binding']['required_source_files'] = required
    cfg_path = folder / 'config.json'
    cfg = c.read(cfg_path)
    cfg['executed_source_files'] = required
    cfg['source_snapshot_sha256'] = expected['source_snapshot_sha256']
    cfg_path.write_bytes(c.encoded(cfg))
    for diagnostic_path in folder.rglob('diagnostics.json'):
        diagnostic = c.read(diagnostic_path)
        diagnostic['source_snapshot_sha256'] = cfg['source_snapshot_sha256']
        diagnostic['executed_source_sha256'] = hashlib.sha256(json.dumps(required, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        diagnostic_path.write_bytes(c.encoded(diagnostic))
    seal_fixture(folder)
    assert a.analyze_bundle(path, plan, smoke_job(), tmp_path)['status'] == 'accepted'
