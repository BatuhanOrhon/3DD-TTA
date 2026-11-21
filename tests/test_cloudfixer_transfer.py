import io
import json

import numpy as np
import pytest

from tests.test_cloudfixer_store import Blobs


def implementation():
    from scripts import run_cloudfixer_transfer
    return run_cloudfixer_transfer


def payload(unit):
    logits = np.zeros((len(unit['ids']), 4, 40), dtype=np.float32)
    logits[:, :, 2] = 3
    stream = io.BytesIO()
    np.savez_compressed(stream, unit=json.dumps(unit, sort_keys=True), ids=unit['ids'],
        labels=np.full(len(unit['ids']), 2), logits=logits, seconds=1., peak_mb=5.,
        denoiser_calls=unit['config']['steps']+1,input_shape=[len(unit['ids']),20,3],
        sampler=json.dumps(dict(valid_reference_counts=[20]*len(unit['ids']),
            retained_reference_counts=[19]*len(unit['ids']),retained_forward_count=19)))
    return stream.getvalue()


def fixture_units():
    config = dict(method='proximal', steps=4, start=40, tau=.1, retained=.95)
    return [dict(campaign='a'*64, phase='smoke', config_id='fixture', config=config,
                 corruption='gaussian', seed=0, ids=ids) for ids in ([1, 3], [5])]


def test_registration_fixes_disjoint_objects_and_search_is_not_locked_to_eight():
    m = implementation()
    reg = m.registration()
    groups = [set(reg[k]) for k in ('smoke_ids', 'development_ids', 'confirmation_ids')]
    assert list(map(len, groups)) == [8, 128, 256]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    configs = m.conditions('development')
    ours = [c for c in configs if c['method'] == 'proximal']
    assert len(ours) == 48 and {c['steps'] for c in ours} == {4, 8, 16, 32}
    assert sum(c['tau'] == 0 for c in ours) == 12


def test_restart_skips_committed_batch_and_processes_exact_tail():
    m = implementation()
    store = Blobs()
    calls = []
    def fail_second(unit):
        calls.append(unit['ids'])
        if unit['ids'] == [5]:
            raise RuntimeError('VM lost mid-batch')
        return payload(unit)
    with pytest.raises(RuntimeError):
        m.execute_pending(store, 'a'*64, fixture_units(), fail_second, cap=2000)
    assert calls == [[1, 3], [5]]
    calls.clear()
    def restored(unit):
        calls.append(unit['ids'])
        return payload(unit)
    records = m.execute_pending(store, 'a'*64, fixture_units(), restored, cap=2000)
    assert calls == [[5]] and len(records) == 2
    rows = m.summarize(records, fixture_units())
    assert rows[0]['n_examples'] == 3 and rows[0]['n_correct'] == 3
    assert rows[0]['status'] == 'complete'


def test_wrong_ids_or_wrong_logits_shape_cannot_be_committed():
    m = implementation()
    unit = fixture_units()[0]
    altered = dict(unit, ids=[9, 11])
    with pytest.raises(ValueError):
        m.validate_payload(payload(altered), unit)
    stream = io.BytesIO()
    np.savez_compressed(stream, unit=json.dumps(unit, sort_keys=True), ids=unit['ids'], labels=[0, 0],
        logits=np.zeros((2, 4, 39)), seconds=1., peak_mb=5., denoiser_calls=5)
    with pytest.raises(ValueError):
        m.validate_payload(stream.getvalue(), unit)


def test_batch_rng_does_not_depend_on_trial_or_resume_order():
    m = implementation()
    a, b = fixture_units()
    assert m.seed_for(a) == m.seed_for(dict(a, config_id='other', config=dict(a['config'], tau=0)))
    assert m.seed_for(a) != m.seed_for(b)


def test_partial_grid_never_announces_a_winner():
    m = implementation()
    unit, pending = fixture_units()
    rows = m.summarize([dict(unit=unit, data=payload(unit))], [unit, pending])
    assert rows[0]['status'] == 'partial'


def test_separate_phases_share_one_persisted_budget():
    m = implementation()
    from cloudfixer_store import spent
    budget, cloud = Blobs(), Blobs()
    m.execute_pending(cloud, 'a'*64, fixture_units(), payload, cap=2000, budget_store=budget)
    assert spent(budget) > 0 and spent(cloud) == 0


def test_complete_selection_can_prefer_tau_zero_and_partial_selection_is_absent(tmp_path):
    m = implementation()
    unit = fixture_units()[0]
    row = m.summarize([dict(unit=unit, data=payload(unit))], [unit])[0]
    row['config'] = dict(row['config'], tau=0)
    m.write_reports(tmp_path, [row], 'campaign', 2, 1)
    assert not (tmp_path/'selection.json').exists()
    m.write_reports(tmp_path, [row], 'campaign', 1, 1)
    assert json.loads((tmp_path/'selection.json').read_text())['best_config']['tau'] == 0


def test_capacity_profile_uses_real_B32_without_expanding_selection_scope():
    m = implementation()
    reg = m.registration()
    requested = m.units('a'*64, reg, 'profile')
    assert len(requested) == 60
    assert all(u['ids'] == reg['development_ids'][:32] for u in requested)
    assert {u['config']['method'] for u in requested} == {'source','cloudfixer','proximal'}
    assert {u['config']['tau'] for u in requested if u['config']['method']=='proximal'} == {0,.1}


def test_scientific_launch_requires_matching_passed_gate():
    m = implementation()
    store = Blobs()
    with pytest.raises(ValueError):
        m.require_gate(store, 'smoke_gate.json', 'a'*64)
    store.put('smoke_gate.json', json.dumps(dict(status='passed', task='b'*64)).encode())
    with pytest.raises(ValueError):
        m.require_gate(store, 'smoke_gate.json', 'a'*64)


def test_forecast_counts_parameter_grid_and_actual_cloud_overhead():
    m = implementation()
    records=[]
    for config in m.conditions('profile'):
        unit=dict(campaign='a'*64,phase='profile',config=config,config_id='x',
                  corruption='gaussian',seed=0,ids=[1,3])
        # The forecast consumes measured seconds; two trials have the same profiled cost.
        records.append(dict(unit=unit, seconds=2.))
    forecast=m.project_grid(records, dict(m.registration(),corruptions=['gaussian']), cloud_overhead=1.)
    assert forecast > 50*4*3  #48 grid+2 controls; four B32 batches; measured pipeline+cloud.


def test_smoke_restart_with_multiple_commits_and_completed_rerun_never_replays_probe():
    m=implementation();store=Blobs();wanted=fixture_units()
    m.execute_pending(store,'a'*64,wanted,payload,cap=2000)
    def no_replay(unit):
        pytest.fail('An already committed smoke unit was replayed')
    records=m.execute_smoke(store,lambda:store,'a'*64,wanted,no_replay,cap=2000)
    assert len(records)==2


def test_resume_projection_counts_only_missing_units_not_original_full_grid():
    m=implementation();reg=dict(m.registration(),corruptions=['gaussian'])
    wanted=m.units('a'*64,reg,'development')
    prof=[dict(unit=dict(config=c,corruption='gaussian'),seconds=2.) for c in m.conditions('profile')]
    full=m.project_grid(prof,reg,cloud_overhead=1.)
    remaining=m.project_grid(prof,reg,cloud_overhead=1.,pending=wanted[len(wanted)//2:])
    assert remaining==pytest.approx(full/2)
    assert remaining+full/2 < full*1.1  #spending plus remainder fits; full projection would not.


def test_completed_profile_reuses_immutable_gate_without_new_measurements():
    m=implementation();store=Blobs()
    gate=dict(status='passed',task='a'*64,cloud_overhead_seconds=2.,projected_grid_seconds=123.)
    store.put('profile_gate.json',json.dumps(gate).encode())
    assert m.profile_gate(store,store,'a'*64,[],m.registration())==gate


def test_invalid_retained_reference_counts_cannot_enter_accepted_results():
    m=implementation();unit=fixture_units()[0]
    with np.load(io.BytesIO(payload(unit))) as archive:
        values={k:archive[k] for k in archive.files}
    values['sampler']=json.dumps(dict(valid_reference_counts=[20,20],
        retained_reference_counts=[99,19],retained_forward_count=19))
    stream=io.BytesIO();np.savez_compressed(stream,**values)
    with pytest.raises(ValueError):m.validate_payload(stream.getvalue(),unit)


def test_mounted_task_folder_accepts_drive_shortcut_but_rejects_escape(tmp_path,monkeypatch):
    from pathlib import Path
    import cloudfixer_store
    from scripts.prepare_cloudfixer_transfer import manifest
    m=implementation()
    target=tmp_path/'shared_result';target.mkdir()
    alias=tmp_path/'drive_shortcut';alias.symlink_to(target,target_is_directory=True)
    spec=tmp_path/'manifest.json';spec.write_text(json.dumps(manifest({})))
    expected=Path('/content/drive/MyDrive/thesis/result')
    original=Path.resolve
    monkeypatch.setattr(Path,'resolve',lambda p,*a,**kw: original(alias if p==expected else p,*a,**kw))
    monkeypatch.setattr(Path,'is_mount',lambda p:p==Path('/content/drive'))
    class ReachedStorage(Exception):pass
    def store(folder):
        assert folder==target/'task'
        raise ReachedStorage
    monkeypatch.setattr(cloudfixer_store,'MountStore',store)
    common=['--manifest',str(spec),'--phase','smoke']
    with pytest.raises(ReachedStorage):
        m.main(common+['--output',str(tmp_path/'valid'),'--drive-folder',str(alias/'task')])
    with pytest.raises(ValueError,match='Expected mounted task folder'):
        m.main(common+['--output',str(tmp_path/'invalid'),'--drive-folder',str(tmp_path/'elsewhere')])


def test_native_reference_has_three_distinct_repeats_with_common_random_draws():
    m=implementation();requested=m.units('a'*64,m.registration(),'development')
    refs=[u for u in requested if u['config']['method']=='cloudfixer']
    assert len(requested)==3120 and len(refs)==180
    first=[u for u in refs if u['corruption']=='uniform' and u['ids']==refs[0]['ids']]
    assert {u['config']['operational_repeat'] for u in first}=={0,1,2}
    assert len({m.unit_key(u) for u in first})==3
    assert len({m.seed_for(u) for u in first})==1


def test_repeat_policy_reports_native_variation_but_rejects_input_or_new_method_drift():
    import torch
    m=implementation();first=torch.tensor([[[2.,0.]]]);other=first.flip(-1)
    proof=m.check_repeat(first,other,'cloudfixer','same','same')
    assert proof['argmax_changes']==1 and proof['max_abs']==2. and proof['mode']=='original_native_variation'
    for method in ('source','proximal'):
        with pytest.raises(ValueError):m.check_repeat(first,other,method,'same','same')
    with pytest.raises(ValueError):m.check_repeat(first,other,'cloudfixer','same','changed')
    with pytest.raises(FloatingPointError):m.check_repeat(first,first*float('nan'),'cloudfixer','same','same')
    assert m.check_repeat(first,first,'proximal','same','same')['max_abs']==0.
