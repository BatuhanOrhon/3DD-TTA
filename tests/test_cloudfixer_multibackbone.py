import io
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_cloudfixer_multibackbone as worker
from cloudfixer_store import MountStore, commit_unit, canonical


def payload(unit, labels):
    n = len(unit['ids'])
    out = io.BytesIO()
    np.savez_compressed(out, unit=json.dumps(unit, sort_keys=True), ids=unit['ids'],
        labels=labels[unit['ids']], source_logits=np.zeros((n, 3, 40)),
        frame_logits=np.zeros((n, 3, 40)), cf_logits=np.zeros((n, 3, 40)),
        seconds=np.array([.1, .2, 1.]), peak_mib=10.,
        denoiser_calls=30, input_sha256='a'*64, output_sha256='b'*64,
        rng_before_sha256='c'*64, rng_after=json.dumps(dict(python=[],numpy=[],cpu=[],cuda=[])))
    return out.getvalue()


def test_full_coverage_and_seed_are_independent_of_pending_work():
    units = worker.units('campaign', 'full')
    assert len(units) == 1170
    for corruption in worker.CORRUPTIONS:
        rows = [u for u in units if u['corruption'] == corruption]
        assert [i for u in rows for i in u['ids']] == list(range(2468))
        assert len(rows[-1]['ids']) == 4
    assert worker.seed_for(units[100]) == worker.seed_for(dict(units[100]))


def test_reopen_skips_completed_geometry_and_validates_labels(tmp_path):
    unit = worker.units('campaign', 'smoke')[0]
    labels = np.arange(2468) % 40
    store = MountStore(tmp_path)
    commit_unit(store, unit, payload(unit, labels))
    records = worker.recover(MountStore(tmp_path), 'campaign', [unit], labels)
    assert list(records) == [worker.unit_key(unit)]
    with pytest.raises(ValueError):
        worker.recover(store, 'campaign', [unit], (labels+1) % 40)
    wrong = dict(unit, ids=unit['ids'][::-1])
    with pytest.raises(ValueError):
        worker.recover(store, 'campaign', [wrong], labels)


@pytest.mark.parametrize('key,value', [('cf_logits', np.zeros((4, 3, 39))),
    ('cf_logits', np.full((4, 3, 40), np.nan)), ('seconds', [-1, .2, 1]),
    ('denoiser_calls', 29), ('input_sha256', 'missing')])
def test_invalid_predictions_never_become_recovered_work(key, value):
    unit = worker.units('campaign', 'smoke')[0]
    labels = np.arange(2468) % 40
    with np.load(io.BytesIO(payload(unit, labels)), allow_pickle=False) as p:
        values = {k: p[k] for k in p.files}
    values[key] = value
    out = io.BytesIO(); np.savez_compressed(out, **values)
    with pytest.raises(ValueError):
        worker.validate(out.getvalue(), unit, labels)


def test_reports_do_not_relabel_source_or_allocate_reconstruction_three_times(tmp_path):
    unit = worker.units('campaign', 'smoke')[0]
    labels = np.arange(2468) % 40
    values = worker.validate(payload(unit, labels), unit, labels)
    rows = worker.report_rows([dict(unit=unit, values=values)])
    assert len(rows) == 9
    assert {r['method'] for r in rows} == {'source_raw', 'source_cf_frame', 'cloudfixer_plain'}
    assert all(r['reconstruction_seconds_shared'] == 1. for r in rows if r['method']=='cloudfixer_plain')


def test_missing_drive_and_foreign_destination_are_rejected_before_writes(monkeypatch):
    path = Path('/content/drive/MyDrive/thesis/result/cloudfixer_multibackbone_20261122/test')
    monkeypatch.setattr(worker.os.path, 'ismount', lambda path: False)
    with pytest.raises(ValueError, match='mounted'):
        worker.require_drive(path)
    monkeypatch.setattr(worker.os.path, 'ismount', lambda path: True)
    worker.require_drive(path)
    with pytest.raises(ValueError, match='task'):
        worker.require_drive(Path('/content/drive/MyDrive/thesis/result/peer'))


def test_full_gate_needs_fresh_profile_and_sufficient_shared_balance():
    gate = dict(status='passed', campaign='campaign', profile_payload_sha256='d'*64,
        account_read_epoch=1000., balance_cu=80., all_assignments_cu_per_h=15.61,
        remaining_eta_seconds=3*3600, execution_window_seconds=3*3600, reserve_cu=5.)
    worker.validate_full_gate(gate, 'campaign', 'd'*64, now=1001.)
    for changes in [dict(balance_cu=40), dict(account_read_epoch=0),
                    dict(profile_payload_sha256='e'*64), dict(reserve_cu=0)]:
        with pytest.raises(ValueError):
            worker.validate_full_gate(dict(gate, **changes), 'campaign', 'd'*64, now=1001.)


def test_recovery_rejects_holes_in_native_seed0_stream(tmp_path):
    units = worker.units('campaign', 'full')[:2]
    labels = np.arange(2468) % 40
    store = MountStore(tmp_path)
    commit_unit(store, units[1], payload(units[1], labels))
    with pytest.raises(ValueError, match='prefix'):
        worker.recover(store, 'campaign', units, labels)


def test_resource_receipts_can_be_renewed_without_overwriting_history(tmp_path):
    store=MountStore(tmp_path)
    for epoch in (1000.,1700.):
        gate=dict(status='passed',campaign='campaign',profile_payload_sha256='d'*64,
            account_read_epoch=epoch,balance_cu=80.,all_assignments_cu_per_h=15.61,
            remaining_eta_seconds=10800.,execution_window_seconds=10800.,reserve_cu=5.)
        data=canonical(gate);key='full_gate_'+worker.sha(data)+'.json';store.put(key,data)
        worker.read_full_gate(store,key,'campaign','d'*64,now=epoch+1.)
    assert len(store.keys('full_gate_'))==2


def test_smoke_replay_exercises_two_native_rng_boundaries_per_corruption():
    units=worker.units('campaign','smoke')
    assert len(units)==6
    for c in ('gaussian','background','rotation'):
        assert [u['ids'] for u in units if u['corruption']==c]==[list(range(4)),list(range(4,8))]


def test_segment_gate_does_not_require_funding_entire_campaign():
    gate=dict(status='passed',campaign='c',profile_payload_sha256='d'*64,
        account_read_epoch=1000.,balance_cu=20.,all_assignments_cu_per_h=15.,
        remaining_eta_seconds=4*3600.,execution_window_seconds=1800.,reserve_cu=5.)
    worker.validate_full_gate(gate,'c','d'*64,now=1001.)
    with pytest.raises(ValueError):
        worker.validate_full_gate(dict(gate,execution_window_seconds=7200.),'c','d'*64,now=1001.)


def test_large_checkpoint_copy_is_immutable_and_does_not_use_unit_size_limit(tmp_path):
    from scripts.setup_cloudfixer_multibackbone_colab import publish_checkpoint, identity
    src=tmp_path/'source';src.write_bytes(b'checkpoint')
    with src.open('r+b') as stream:stream.truncate(100_000_001)
    dest=tmp_path/'drive'/'weights.npy';dest.parent.mkdir()
    expected=identity(src)
    publish_checkpoint(src,dest,expected)
    publish_checkpoint(src,dest,expected)
    assert identity(dest)==expected
    src.write_bytes(b'wrong')
    with pytest.raises(ValueError):publish_checkpoint(src,dest,expected)
    assert identity(dest)==expected


def test_analyzer_recounts_saved_predictions_and_detects_csv_tampering(tmp_path):
    from scripts.analyze_cloudfixer_multibackbone import recount, verify_csv
    unit=worker.units('campaign','smoke')[0]
    labels=np.arange(2468)%40
    store=MountStore(tmp_path/'raw');commit_unit(store,unit,payload(unit,labels))
    rows=recount(tmp_path/'raw','campaign',[unit],labels)
    assert len(rows)==9 and all(r['n_correct']==1 and r['n_examples']==4 for r in rows)
    records=list(worker.recover(store,'campaign',[unit],labels).values())
    worker.write_reports(tmp_path,records,'campaign')
    verify_csv(tmp_path/'per_corruption.csv',rows)
    csv=tmp_path/'per_corruption.csv';csv.write_text(csv.read_text().replace(',1,',',2,'))
    with pytest.raises(ValueError):verify_csv(csv,rows)


def test_summary_is_recounted_and_cannot_hide_corrupt_headline(tmp_path):
    from scripts.analyze_cloudfixer_multibackbone import summarize,verify_summary
    unit=worker.units('campaign','smoke')[0];labels=np.arange(2468)%40
    record=dict(unit=unit,values=worker.validate(payload(unit,labels),unit,labels))
    worker.write_reports(tmp_path,[record],'campaign')
    rows=worker.report_rows([record]);expected=summarize(rows)
    verify_summary(tmp_path/'summary.csv',expected)
    p=tmp_path/'summary.csv';p.write_text(p.read_text().replace('0.25','0.5'))
    with pytest.raises(ValueError):verify_summary(p,expected)


def test_completed_corruption_cursor_requires_intact_archive(tmp_path):
    store=MountStore(tmp_path);data=b'archive'
    store.put('background.zip',data)
    cursor=dict(campaign='c',corruption='background',objects=2468,heads=list(worker.HEADS),
        archive='background.zip',archive_sha256=worker.sha(data))
    store.put('completed_background.json',canonical(cursor))
    assert set(worker.completed_corruptions(store,'c'))=={'background'}
    (tmp_path/'background.zip').write_bytes(b'corrupt')
    with pytest.raises(ValueError):worker.completed_corruptions(store,'c')


def test_native_backups_are_reusable_and_untrusted_restore_paths_rejected(tmp_path):
    from scripts.setup_cloudfixer_multibackbone_colab import native_snapshot,restore_native
    src=tmp_path/'test.so';src.write_bytes(b'ELF synthetic')
    store=MountStore(tmp_path/'drive')
    native_snapshot(store,{'test':str(src)})
    native_snapshot(store,{'test':str(src)})
    with pytest.raises(ValueError,match='destination'):restore_native(store)


def test_funding_window_expires_from_account_reading():
    gate=dict(status='passed',campaign='c',profile_payload_sha256='d'*64,
        account_read_epoch=1000.,balance_cu=80.,all_assignments_cu_per_h=15.,
        remaining_eta_seconds=14400.,execution_window_seconds=600.,reserve_cu=5.)
    worker.validate_full_gate(gate,'c','d'*64,now=1001.)
    with pytest.raises(ValueError):worker.validate_full_gate(gate,'c','d'*64,now=1550.)


def test_multiwindow_shards_union_without_double_count_or_mixed_campaign():
    from scripts.analyze_cloudfixer_multibackbone import combine_shards,summarize
    rows=[dict(method=m,backbone=h,corruption='rotation',n_examples=2468,n_correct=1234,accuracy=.5)
          for m in worker.METHODS for h in worker.HEADS]
    other=[dict(r,corruption='background',n_correct=617,accuracy=.25) for r in rows]
    first=dict(campaign='c',rows=rows);second=dict(campaign='c',rows=other)
    result=summarize(combine_shards([first,second]))
    assert all(r['total_examples']==4936 and r['total_correct']==1851 and r['macro_accuracy']==.375 for r in result)
    with pytest.raises(ValueError,match='Overlapping'):combine_shards([first,first])
    with pytest.raises(ValueError,match='Mixed'):combine_shards([first,dict(second,campaign='different')])


@pytest.mark.parametrize('batch_size',[32,64])
def test_analyzer_cli_combines_two_complete_corruption_shards(tmp_path,monkeypatch,batch_size):
    from scripts import analyze_cloudfixer_multibackbone as analyzer
    labels=np.arange(2468)%40
    label_path=tmp_path/'label.npy';np.save(label_path,labels)
    reference=dict(raw_archives=[dict(dataset='modelnet40_c',arm='proximal',backbone=h,
        corruption=c,n_examples=2468,n_correct=1200,sha256='f'*64)
        for h in worker.HEADS for c in ('background','rotation')])
    ref_path=tmp_path/'reference.json';ref_path.write_bytes(canonical(reference))
    spec=dict(dataset_files={'label.npy':worker.sha(label_path.read_bytes())},scientific_protocol=dict(batch_size=batch_size),
        baseline_reference=dict(bytes=ref_path.stat().st_size,sha256=worker.sha(ref_path.read_bytes())),baseline_results=[])
    campaign=worker.sha(canonical(spec));folders=[]
    after=dict(python=[],numpy=[],cpu=[],cuda=[])
    for corr in ('background','rotation'):
        directory=tmp_path/corr;store=MountStore(directory/'raw');records=[];previous='c'*64
        wanted=[u for u in worker.units(campaign,'full',batch_size) if u['corruption']==corr]
        for unit in wanted:
            with np.load(io.BytesIO(payload(unit,labels)),allow_pickle=False) as p:
                values={k:p[k].copy() for k in p.files}
            values['rng_before_sha256']=previous;previous=worker.sha(canonical(after))
            stream=io.BytesIO();np.savez_compressed(stream,**values);data=stream.getvalue()
            commit_unit(store,unit,data);records.append(dict(unit=unit,values=worker.validate(data,unit,labels)))
        worker.write_reports(directory,records,campaign)
        (directory/'config.json').write_bytes(canonical(dict(manifest=spec,campaign=campaign,phase='full',
            artifact_scope='single_corruption',corruption=corr)))
        folders.append(str(directory))
    output=tmp_path/'analysis'
    monkeypatch.setattr(analyzer.sys,'argv',['analyze','--run',*folders,'--labels',str(label_path),
        '--baseline-reference',str(ref_path),'--output',str(output)])
    analyzer.main()
    report=json.loads((output/'recount.json').read_bytes())
    assert report['blocks']==2*((2468+batch_size-1)//batch_size) and not report['full_campaign'] and len(report['comparisons'])==6
    assert all(r['total_examples']==4936 for r in report['summary'])


@pytest.mark.parametrize('accelerator,name,capability,architecture',[
    ('A100','NVIDIA A100-SXM4-40GB',(8,0),'8.0'),('L4','NVIDIA L4',(8,9),'8.9')])
def test_gpu_registration_keeps_architecture_and_model_matched(accelerator,name,capability,architecture):
    from scripts.setup_cloudfixer_multibackbone_colab import gpu_architecture
    assert gpu_architecture(accelerator,name,capability)==architecture
    with pytest.raises(ValueError):gpu_architecture(accelerator,'NVIDIA T4',(7,5))
    with pytest.raises(ValueError):gpu_architecture(accelerator,name,(7,5))


def test_l4_head_native_cannot_fall_back_to_a100_binary(tmp_path):
    from scripts.setup_cloudfixer_multibackbone_colab import preload_head_native
    assert preload_head_native(tmp_path,'A100') is None
    with pytest.raises(ValueError,match='L4'):preload_head_native(tmp_path,'L4')


@pytest.mark.parametrize('filename,digest',[
    ('../outside.so','a'*64),('/content/outside.so','a'*64),
    ('unrelated.so','a'*64),('pointnet2_batch_cuda.cpython-38-x86_64-linux-gnu.so','bad')])
def test_l4_cached_native_receipt_rejects_unsafe_name_or_digest(filename,digest):
    from scripts.setup_cloudfixer_multibackbone_colab import head_native_identity
    with pytest.raises(ValueError):head_native_identity(dict(accelerator='L4',filename=filename,sha256=digest,bytes=100))


def test_drive_shortcut_root_is_accepted_but_child_escape_is_not(tmp_path,monkeypatch):
    from scripts import setup_cloudfixer_multibackbone_colab as setup
    drive=tmp_path/'drive';(drive/'MyDrive').mkdir(parents=True)
    target=drive/'.shortcut-targets-by-id'/'owned'/'thesis';target.mkdir(parents=True)
    shortcut=drive/'MyDrive/thesis';shortcut.symlink_to(target,target_is_directory=True)
    task=shortcut/'result/cloudfixer_multibackbone_20261122';task.mkdir(parents=True)
    monkeypatch.setattr(setup,'DRIVE_MOUNT',drive)
    monkeypatch.setattr(setup,'TASK_DRIVE_ROOT',task)
    monkeypatch.setattr(setup.os.path,'ismount',lambda p:True)
    assert setup.require_task_drive(task/'l4_trial')==target/'result/cloudfixer_multibackbone_20261122/l4_trial'
    foreign=drive/'MyDrive/foreign';foreign.mkdir()
    (task/'escape').symlink_to(foreign,target_is_directory=True)
    with pytest.raises(ValueError):setup.require_task_drive(task/'escape')
    with pytest.raises(ValueError):setup.require_task_drive(task.parent/'cloudfixer_multibackbone_20261122_other')


def test_task_drive_root_cannot_resolve_outside_actual_mount(tmp_path,monkeypatch):
    from scripts import setup_cloudfixer_multibackbone_colab as setup
    drive=tmp_path/'drive';drive.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    monkeypatch.setattr(setup,'DRIVE_MOUNT',drive)
    monkeypatch.setattr(setup,'TASK_DRIVE_ROOT',outside)
    monkeypatch.setattr(setup.os.path,'ismount',lambda p:True)
    with pytest.raises(ValueError):setup.require_task_drive(outside/'l4_trial')


def test_backed_up_setup_source_does_not_become_budget_receipt(tmp_path):
    from scripts.setup_cloudfixer_multibackbone_colab import backup_package
    from cloudfixer_store import record_setup,spent
    package=tmp_path/'package';package.mkdir()
    name='setup_cloudfixer_multibackbone_colab.py';data=b'#!/usr/bin/python\nprint("source")\n'
    (package/name).write_bytes(data)
    receipt={name:dict(bytes=len(data),sha256=worker.sha(data))}
    (package/'package_receipt.json').write_bytes(canonical(receipt))
    store=MountStore(tmp_path/'drive')
    backup_package(store,package,receipt)
    record_setup(store,dict(attempts=[dict(session='owned',started_utc='2026-11-22T00:00:00Z',seconds=2.)]),cap=100.)
    assert spent(store)==2. and store.get('package_'+name)==data


def test_registered_batch64_covers_all_ids_and_profile_without_old_cache_reuse():
    old=worker.units('campaign','full')
    new=worker.units('campaign','full',batch_size=64)
    assert len(new)==585
    for corruption in worker.CORRUPTIONS:
        group=[u for u in new if u['corruption']==corruption]
        assert [i for u in group for i in u['ids']]==list(range(2468))
        assert len(group[-1]['ids'])==36
    assert {worker.unit_key(u) for u in new}.isdisjoint(worker.unit_key(u) for u in old)
    assert len(worker.units('campaign','profile',batch_size=64)[0]['ids'])==64
    assert len(worker.units('campaign','smoke',batch_size=64))==6
    for bad in (0,1,48,128,True):
        with pytest.raises(ValueError):worker.units('campaign','full',batch_size=bad)


@pytest.mark.parametrize('reference', [b'{}', b'{}'+b' '*(94668-2)], ids=['wrong-size','same-size-wrong-sha'])
def test_packager_rejects_changed_accepted_lion_reference(monkeypatch,reference):
    from scripts import prepare_cloudfixer_multibackbone as preparation
    monkeypatch.setattr(preparation.subprocess,'check_output',lambda *args,**kwargs:reference)
    with pytest.raises(ValueError,match='Accepted LION reference'):
        preparation.accepted_reference()


BASELINE_PROFILE = 'cloudfixer_xyz_proximal_pm_n8_k80_late_v1'
FROZEN = {'prior': 'a'*64, 'heads': {name: 'b'*64 for name in worker.HEADS}}


def synthetic_rng_state(seed=0):
    import random
    import torch

    numpy = np.random.RandomState(seed).get_state()
    return dict(python=random.Random(seed).getstate(), numpy=[numpy[0], numpy[1].tolist(), *numpy[2:]],
                cpu=torch.Generator().manual_seed(seed).get_state().tolist(),
                cuda=torch.Generator().manual_seed(seed).get_state().tolist())


def seed0_rng_digest():
    return worker.sha(canonical(synthetic_rng_state()))


def baseline_record(unit, labels, *, same_seed=0):
    import random
    from time import perf_counter
    import torch
    from test_cloudfixer_diffusion import ConstantEpsilon

    model = ConstantEpsilon()
    x = torch.arange(len(unit['ids'])*24, dtype=torch.float32).reshape(-1, 8, 3) / 100
    mask = torch.ones_like(x[..., :1])
    logits_seen = []
    clouds_seen = []
    rng_seen = []

    def classify(cloud):
        clouds_seen.append(cloud.clone())
        rng_seen.append((random.random(), np.random.rand(), torch.rand(()).item()))
        logits = torch.full((len(x), 3, 40), -1000.)
        logits[..., :2] = torch.tensor([2., 0.] if len(logits_seen) < 2 else [-100., 100.])
        logits_seen.append(logits)
        return logits, [0., 0., 0.]

    def timed(fn):
        start = perf_counter()
        value = fn()
        return value, perf_counter() - start

    x_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    values = worker.proximal_readout(model, classify, x, mask, unit, geometry=lambda c: 2*c, timed=timed)
    assert torch.equal(torch.get_rng_state(), x_before)
    np.testing.assert_equal(np.random.get_state(), numpy_before)
    assert random.getstate() == python_before
    assert len({r[:2] for r in rng_seen}) == 1
    assert len({r[2] for r in rng_seen}) == 3

    frozen_rng = synthetic_rng_state(same_seed if same_seed is not None else 0)
    if same_seed is None:
        frozen_rng.update(cpu=[0], cuda=[0])
    values.update(unit=canonical(unit).decode(), ids=unit['ids'], labels=labels[unit['ids']],
        source_logits=np.zeros((len(x), 3, 40), dtype=np.float32), source_head_seconds=[.1, .2, .3],
        source_seconds=.7, peak_mib=10., frozen_state=canonical(FROZEN).decode(),
        rng_before_sha256=worker.sha(canonical(frozen_rng)), rng_after=canonical(frozen_rng).decode())
    return model, clouds_seen, logits_seen, values


def encode_payload(values):
    out = io.BytesIO()
    np.savez_compressed(out, **values)
    return out.getvalue()


def test_xyz_baseline_runs_one_n8_trajectory_and_reports_probability_mean(tmp_path):
    from scripts.analyze_cloudfixer_multibackbone import recount, verify_csv
    unit = worker.units('baseline', 'smoke', 64)[0]
    labels = np.zeros(2468, dtype=int)
    model, clouds, logits, values = baseline_record(unit, labels)

    assert model.calls == [80, 70, 60, 50, 40, 30, 20, 10, 0]
    assert values['denoiser_calls'] == 9
    np.testing.assert_array_equal(values['candidate_indices'], [20, 10, 0])
    assert len(clouds) == 3
    assert not np.array_equal(clouds[1].numpy(), clouds[2].numpy())
    np.testing.assert_array_equal(values['terminal_logits'], logits[2].numpy())
    assert np.all(np.argmax(values['candidate_logits'].mean(axis=1), axis=-1) == 1)
    assert values['pm_probabilities'].dtype == np.float64
    assert np.all(np.argmax(values['pm_probabilities'], axis=-1) == 0)

    data = encode_payload(values)
    values = worker.validate(data, unit, labels, profile=BASELINE_PROFILE,
                             frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())
    record = dict(unit=unit, values=values)
    commit_unit(MountStore(tmp_path/'raw'), unit, data)
    worker.write_reports(tmp_path, [record], 'baseline', profile=BASELINE_PROFILE)
    rows = recount(tmp_path/'raw', 'baseline', [unit], labels, profile=BASELINE_PROFILE,
                   frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())
    verify_csv(tmp_path/'per_corruption.csv', rows)
    assert {r['n_correct'] for r in rows if r['method']=='proximal_n8_pm'} == {4}
    assert {r['n_correct'] for r in rows if r['method']=='proximal_n8_terminal'} == {0}


@pytest.mark.parametrize('field,replacement', [
    ('policy', '{}'), ('candidate_indices', [10, 20, 0]),
    ('candidate_logits', None), ('pm_probabilities', float('nan')),
    ('terminal_logits', None), ('geometry_seed', 1),
    ('trajectory_sha256', '0'*64), ('rng_before_sha256', '0'*64),
    ('frozen_state', '{}'), ('candidate_probabilities', None),
    ('head_order', ['curvenet', 'dgcnn', 'pointnext']),
    ('source_head_seconds', [99999., 99999., 99999.]),
    ('candidate_head_seconds', [[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
])
def test_xyz_baseline_rejects_malformed_trajectory(field, replacement):
    unit = worker.units('baseline', 'smoke', 64)[0]
    labels = np.zeros(2468, dtype=int)
    _, _, _, values = baseline_record(unit, labels)
    if field == 'terminal_logits':
        values[field] = values['candidate_logits'][:, 1]
    elif replacement is None:
        del values[field]
    elif field == 'pm_probabilities':
        values[field][0, 0, 0] = replacement
    else:
        values[field] = replacement
    with pytest.raises(ValueError):
        worker.validate(encode_payload(values), unit, labels, profile=BASELINE_PROFILE,
                        frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())


def test_xyz_baseline_cache_cannot_read_legacy_payload_or_mixed_profile(tmp_path):
    unit = worker.units('campaign', 'smoke', 64)[0]
    labels = np.zeros(2468, dtype=int)
    store = MountStore(tmp_path)
    commit_unit(store, unit, payload(unit, labels))
    assert list(worker.recover(store, 'campaign', [unit], labels)) == [worker.unit_key(unit)]
    with pytest.raises(ValueError):
        worker.recover(store, 'campaign', [unit], labels, profile=BASELINE_PROFILE,
                       frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())
    assert worker.units('baseline', 'full', 64)[-1]['ids'] == list(range(2432, 2468))
    assert len(worker.units('baseline', 'full', 64)) == 585
    assert worker.unit_key(unit) != worker.unit_key(dict(unit, campaign='baseline'))


def test_xyz_baseline_unit_seed_is_addressed_and_classifier_rng_isolation_is_replayable():
    import random
    import torch

    unit = worker.units('baseline', 'smoke', 64)[0]
    labels = np.zeros(2468, dtype=int)
    _, _, _, first = baseline_record(unit, labels)
    random.random(); np.random.rand(); torch.rand(8)
    _, _, _, replay = baseline_record(unit, labels)
    for field in ('geometry_seed','candidate_logits','pm_probabilities','trajectory_sha256'):
        np.testing.assert_array_equal(first[field], replay[field])
    assert first['geometry_seed'] != worker.seed_for(dict(unit, ids=[4,5,6,7]))


def test_xyz_baseline_resume_will_not_skip_a_sha_valid_incomplete_zip(tmp_path):
    store = MountStore(tmp_path)
    import zipfile
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('config.json', '{}')
    data = stream.getvalue()
    store.put('background.zip', data)
    cursor = dict(campaign='baseline', corruption='background', objects=2468,
        heads=list(worker.HEADS), archive='background.zip', archive_sha256=worker.sha(data), profile=BASELINE_PROFILE)
    store.put('completed_background.json',canonical(cursor))
    with pytest.raises(ValueError):
        worker.completed_corruptions(store, 'baseline', profile=BASELINE_PROFILE,
                                     labels=np.zeros(2468, dtype=int), frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())


def baseline_manifest(monkeypatch, label_sha='f'*64):
    old = dict(schema=1, source_commit='a'*40, dataset_files={'label.npy': label_sha},
               head_state_hashes=FROZEN['heads'])
    assets = {k: old.get(k) for k in ('head_configs', 'head_state_hashes', 'dataset_files',
                                     'files', 'diffusion_sha256', 'author_commit', 'official_files')}
    monkeypatch.setattr(worker, 'ORIGINAL_CF_ASSET_SHA256', worker.sha(canonical(assets)))
    spec = dict(old, task='cloudfixer_baseline_heads_20261123', profile=BASELINE_PROFILE,
                overlay_files={path: dict(bytes=10, sha256='d'*64) for path in worker.BASELINE_OVERLAY_FILES},
                policy={'steps':8,'start':80,'tau':.1,'retained':.95,'probe_indices':[20,10,0]},
                scientific_protocol=worker.baseline_protocol(), baseline_reference=worker.ORIGINAL_CF_REFERENCE,
                baseline_archive_index=worker.ORIGINAL_CF_ARCHIVE_INDEX, baseline_source=worker.ORIGINAL_CF_SOURCE)
    return old, spec


def test_xyz_baseline_profile_manifest_pins_policy_and_accepted_assets(monkeypatch):
    old, spec = baseline_manifest(monkeypatch)
    assert worker.manifest_profile(old) == worker.ORIGINAL_PROFILE
    assert worker.manifest_profile(spec) == BASELINE_PROFILE
    assert worker.sha(canonical(spec)) != worker.sha(canonical(old))
    for override in [dict(policy=dict(spec['policy'], probe_indices=[0])), dict(profile='unknown'),
                     dict(head_state_hashes={}), dict(dataset_files={}),
                     dict(scientific_protocol=dict(spec['scientific_protocol'],batch_size=32))]:
        with pytest.raises(ValueError):
            worker.manifest_profile(dict(spec, **override))


@pytest.mark.parametrize('batch_size,blocks,final_size', [(64, 39, 36), (96, 26, 68), (128, 20, 36)])
def test_xyz_baseline_batch_registration_covers_all_original_ids(monkeypatch, batch_size, blocks, final_size):
    _, old = baseline_manifest(monkeypatch)
    spec = dict(old, scientific_protocol=dict(old['scientific_protocol'], batch_size=batch_size))
    assert worker.manifest_profile(spec) == BASELINE_PROFILE
    assert worker.baseline_protocol(batch_size) == spec['scientific_protocol']
    old_protocol_sha = 'f202ca9234ede7feecbaa3a5939d98f511776deb6acd869b9df9e7e26eddfcc4'
    assert worker.sha(canonical(old['scientific_protocol'])) == old_protocol_sha
    campaign = worker.sha(canonical(spec))
    old_campaign = worker.sha(canonical(old))
    wanted = worker.units(campaign, 'full', batch_size, profile=BASELINE_PROFILE)
    assert len(wanted) == 15 * blocks
    for corruption in worker.CORRUPTIONS:
        group = [u for u in wanted if u['corruption'] == corruption]
        assert [i for u in group for i in u['ids']] == list(range(2468))
        assert [len(u['ids']) for u in group] == [batch_size] * (blocks - 1) + [final_size]
    assert len(worker.units(campaign, 'profile', batch_size, profile=BASELINE_PROFILE)[0]['ids']) == batch_size
    assert len(worker.units(campaign, 'smoke', batch_size, profile=BASELINE_PROFILE)) == 6
    if batch_size != 64:
        assert campaign != old_campaign
        assert {worker.unit_key(u) for u in wanted}.isdisjoint(
            worker.unit_key(u) for u in worker.units(old_campaign, 'full', 64, profile=BASELINE_PROFILE))


@pytest.mark.parametrize('profile,batch_size', [
    (worker.ORIGINAL_PROFILE, 96), (worker.ORIGINAL_PROFILE, 128),
    (BASELINE_PROFILE, 32), (BASELINE_PROFILE, 48), (BASELINE_PROFILE, True), (BASELINE_PROFILE, 96.0),
])
def test_xyz_batch_registration_rejects_out_of_scope_manifest_and_package(tmp_path, monkeypatch, profile, batch_size):
    from scripts.prepare_cloudfixer_multibackbone import package

    old, baseline = baseline_manifest(monkeypatch)
    spec = baseline if profile == BASELINE_PROFILE else old
    protocol = dict(spec.get('scientific_protocol', {}), batch_size=batch_size)
    with pytest.raises(ValueError):
        worker.manifest_profile(dict(spec, scientific_protocol=protocol))
    output = tmp_path/'out'
    with pytest.raises(ValueError):
        package(tmp_path/'missing-assets', tmp_path/'missing-official', output, batch_size=batch_size, profile=profile)
    assert not output.exists()


def test_xyz_baseline_requires_its_own_drive_root(tmp_path, monkeypatch):
    from scripts import setup_cloudfixer_multibackbone_colab as setup
    old = tmp_path/'MyDrive/thesis/result/cloudfixer_multibackbone_20261122'
    new = old.with_name('cloudfixer_baseline_heads_20261123')
    monkeypatch.setattr(setup,'DRIVE_MOUNT',tmp_path)
    monkeypatch.setattr(setup,'TASK_DRIVE_ROOT',old)
    monkeypatch.setattr(setup.os.path,'ismount',lambda p:True)
    assert setup.require_task_drive(new/'owned', profile=BASELINE_PROFILE) == new/'owned'
    surface = old.with_name('surface_four_models_20261128')
    assert setup.require_task_drive(surface/'setup', profile=BASELINE_PROFILE, baseline_task=surface.name) == surface/'setup'
    with pytest.raises(ValueError):
        setup.require_task_drive(new/'setup', profile=BASELINE_PROFILE, baseline_task=surface.name)
    for invalid in (old, new.with_name(new.name+'_peer')):
        with pytest.raises(ValueError):
            setup.require_task_drive(invalid, profile=BASELINE_PROFILE)
    with pytest.raises(ValueError):
        setup.require_task_drive(new)


def test_xyz_baseline_pairs_cached_cloudfixer_predictions_by_original_id():
    from scripts.analyze_cloudfixer_multibackbone import paired_rows
    labels = np.array([0, 1, 2, 3])
    pm = np.array([0, 1, 9, 3])
    terminal = np.array([0, 9, 2, 9])
    cf = np.array([9, 1, 2, 9])
    scope = {}
    for name in worker.HEADS:
        scope[('gaussian',name)] = {
            i: (int(labels[i]), int(pm[i]), int(terminal[i])) for i in range(4)}
    reordered = dict((('gaussian',name), {i:(int(labels[i]),int(cf[i])) for i in (3, 2, 1, 0)})
                     for name in worker.HEADS)
    pairs = paired_rows(scope, reordered)
    pm_vs_cf = [r for r in pairs if r['comparison']=='proximal_n8_pm_vs_cloudfixer_plain']
    assert len(pairs) == 9
    assert {(r['reference_only'], r['ours_only'], r['net_correct']) for r in pm_vs_cf} == {(1, 2, 1)}
    reordered[('gaussian','dgcnn')][1] = (9, 1)
    with pytest.raises(ValueError, match='label'):
        paired_rows(scope, reordered)



def test_xyz_baseline_resume_rejects_discontinuous_full_rng_state(tmp_path):
    labels = np.zeros(2468, dtype=int)
    wanted = [u for u in worker.units('baseline', 'smoke', 64) if u['corruption']=='background']
    store = MountStore(tmp_path)
    for i, unit in enumerate(wanted):
        _, _, _, values = baseline_record(unit, labels, same_seed=i)
        commit_unit(store, unit, encode_payload(values))
    with pytest.raises(ValueError, match='registered seed 0'):
        worker.recover(store, 'baseline', wanted, labels, profile=BASELINE_PROFILE,
                       frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())


def baseline_archive(tmp_path, monkeypatch, *, rng_seed=0, source_commit='a'*40,
                     batch_size=64, cache_batch_size=None):
    import zipfile

    labels = np.zeros(2468, dtype=int)
    labels_path = tmp_path/'label.npy'
    np.save(labels_path, labels)
    label_sha = worker.sha(labels_path.read_bytes())
    _, spec = baseline_manifest(monkeypatch, label_sha)
    spec['source_commit'] = source_commit
    spec['scientific_protocol']['batch_size'] = batch_size
    campaign = worker.sha(canonical(spec))
    size = batch_size if cache_batch_size is None else cache_batch_size
    wanted = [dict(campaign=campaign, phase='full', corruption='background', seed=0,
                   ids=list(range(i, min(i+size, 2468))), batch_size=size)
              for i in range(0, 2468, size)]
    _, _, _, template = baseline_record(wanted[0], labels, same_seed=rng_seed)
    directory = tmp_path/'scope'
    raw = MountStore(directory/'raw')
    records = []
    for unit in wanted:
        n = len(unit['ids'])
        fields = ('source_logits', 'terminal_logits', 'candidate_logits', 'candidate_probabilities', 'pm_probabilities')
        values = dict(template, unit=canonical(unit).decode(), ids=unit['ids'], labels=labels[unit['ids']],
                      geometry_seed=worker.seed_for(unit))
        for field in fields:
            values[field] = np.broadcast_to(template[field][0], (n, *template[field].shape[1:]))
        values['trajectory_sha256'] = worker.trajectory_sha(unit, values)
        data = encode_payload(values)
        with np.load(io.BytesIO(data), allow_pickle=False) as record:
            records.append(dict(unit=unit, values={k: record[k] for k in record.files}))
        commit_unit(raw, unit, data)
    worker.write_reports(directory, records, campaign, profile=BASELINE_PROFILE)
    (directory/'config.json').write_bytes(canonical(dict(manifest=spec, campaign=campaign,
        phase='full', artifact_scope='single_corruption', corruption='background', expected_objects=2468)))
    (directory/'environment.txt').write_bytes(canonical(dict(profile=BASELINE_PROFILE, policy=spec['policy'],
        frozen_prior_hash=FROZEN['prior'], frozen_head_hashes=FROZEN['heads'],
        cublas_workspace_config=':4096:8', seed0_rng_sha256=seed0_rng_digest())))
    for name in ('notes.md', 'command.txt'):
        (directory/name).write_text('synthetic CPU witness')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        for p in directory.rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(directory))
    data = stream.getvalue()
    store = MountStore(tmp_path/'drive')
    store.put('background.zip', data)
    receipt = dict(campaign=campaign, corruption='background', objects=2468, heads=list(worker.HEADS),
        archive='background.zip', archive_sha256=worker.sha(data), profile=BASELINE_PROFILE, head_receipts={})
    rows = worker.report_rows(records, profile=BASELINE_PROFILE)
    for head in worker.HEADS:
        name = 'head_'+head+'.json'
        receipt['head_receipts'][head] = name
        store.put(name, canonical(dict(campaign=campaign, profile=BASELINE_PROFILE, backbone=head,
            corruption='background', status='complete', archive=receipt['archive'], archive_sha256=receipt['archive_sha256'],
            rows=[r for r in rows if r['backbone']==head])))
    store.put('completed_background.json', canonical(receipt))
    return directory, labels, labels_path, label_sha, spec, campaign, store


@pytest.mark.parametrize('batch_size,blocks', [(64, 39), (96, 26), (128, 20)])
def test_xyz_baseline_registered_archive_matches_cpu_csv_and_head_receipts(tmp_path, monkeypatch, batch_size, blocks):
    from scripts.analyze_cloudfixer_multibackbone import read_shard

    directory, labels, _, label_sha, _, campaign, store = baseline_archive(
        tmp_path, monkeypatch, batch_size=batch_size)
    shard = read_shard(directory, labels, label_sha, expected_campaign=campaign)
    assert shard['blocks'] == blocks
    assert all(r['n_examples'] == 2468 for r in shard['rows'])
    assert all(r['n_correct'] == 2468 for r in shard['rows'] if r['method']=='proximal_n8_pm')
    completed = worker.completed_corruptions(store, campaign, profile=BASELINE_PROFILE, labels=labels,
                                             frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())
    assert set(completed) == {'background'}
    (store.folder/'head_pointnext.json').write_text('{}')
    with pytest.raises(ValueError, match='head receipt'):
        worker.completed_corruptions(store, campaign, profile=BASELINE_PROFILE, labels=labels,
                                     frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())



@pytest.mark.parametrize('batch_size', [96, 128])
def test_xyz_baseline_completed_archive_rejects_wrong_batch_cache(tmp_path, monkeypatch, batch_size):
    from scripts.analyze_cloudfixer_multibackbone import read_shard

    directory, labels, _, label_sha, _, campaign, store = baseline_archive(
        tmp_path, monkeypatch, batch_size=batch_size, cache_batch_size=64)
    with pytest.raises(ValueError, match='Unexpected recovered work'):
        read_shard(directory, labels, label_sha, expected_campaign=campaign)
    with pytest.raises(ValueError, match='Unexpected recovered work'):
        worker.completed_corruptions(store, campaign, profile=BASELINE_PROFILE, labels=labels,
                                     frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())


@pytest.mark.parametrize('rng_seed', [1, None], ids=['different-seed', 'one-byte-torch-states'])
def test_xyz_baseline_rejects_whole_corruption_at_a_consistent_wrong_rng(tmp_path, monkeypatch, rng_seed):
    from scripts.analyze_cloudfixer_multibackbone import read_shard

    directory, labels, _, label_sha, _, campaign, store = baseline_archive(tmp_path, monkeypatch, rng_seed=rng_seed)
    with pytest.raises(ValueError, match='registered seed 0'):
        read_shard(directory, labels, label_sha, expected_campaign=campaign)
    with pytest.raises(ValueError, match='registered seed 0'):
        worker.completed_corruptions(store, campaign, profile=BASELINE_PROFILE, labels=labels,
                                     frozen=FROZEN, seed0_rng_sha256=seed0_rng_digest())


@pytest.mark.parametrize('batch_size', [64, 96])
def test_xyz_baseline_cli_rejects_self_consistent_different_campaign(tmp_path, monkeypatch, batch_size):
    from scripts import analyze_cloudfixer_multibackbone as analyzer
    from scripts.prepare_cloudfixer_multibackbone import original_cloudfixer_reference

    directory, _, labels_path, _, actual, actual_campaign, _ = baseline_archive(
        tmp_path, monkeypatch, source_commit='e'*40, batch_size=batch_size)
    expected = worker.sha(canonical(dict(actual, source_commit='a'*40)))
    assert actual_campaign != expected
    reference_path = tmp_path/'accepted_reference.json'
    reference_path.write_bytes(original_cloudfixer_reference()[0])
    output = tmp_path/'analysis'
    argv = ['analyze', '--run', str(directory), '--labels', str(labels_path),
            '--baseline-reference', str(reference_path), '--output', str(output)]
    monkeypatch.setattr(analyzer.sys, 'argv', argv)
    with pytest.raises(ValueError, match='requires an independently expected campaign'):
        analyzer.main()
    monkeypatch.setattr(analyzer.sys, 'argv', [*argv, '--expected-campaign', expected])
    with pytest.raises(ValueError, match='differs from expected campaign'):
        analyzer.main()
    assert not output.exists()
    monkeypatch.setattr(analyzer.sys, 'argv', [*argv, '--expected-campaign', actual_campaign])
    analyzer.main()
    report = json.loads((output/'recount.json').read_text())
    assert report['campaign'] == actual_campaign
    assert f'original-ID blocks of up to {batch_size} objects' in report['limitation']
    assert 'original CloudFixer used batch size 64' in report['limitation']
    assert report['seed0_rng_sha256'] == seed0_rng_digest()
