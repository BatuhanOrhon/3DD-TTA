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
