"""Real commit/recovery logic with only the external blob transport replaced."""
import json

import pytest


class Blobs:
    def __init__(self):
        self.values = {}
        self.fail_receipt_once = False

    def put(self, key, data):
        if self.fail_receipt_once and key.startswith('unit_'):
            self.fail_receipt_once = False
            raise ConnectionError('interrupted before receipt acknowledgement')
        if key in self.values and self.values[key] != data:
            raise ValueError('immutable key conflict')
        self.values[key] = data

    def get(self, key):
        return self.values[key]

    def keys(self, prefix):
        return sorted(k for k in self.values if k.startswith(prefix))


def implementation():
    import cloudfixer_store
    return cloudfixer_store


def test_payload_without_receipt_is_not_completed_and_retry_adopts_once():
    m = implementation()
    store = Blobs()
    unit = dict(campaign='a'*64, phase='development', config='x', corruption='gaussian', seed=0, ids=[1, 3])
    store.fail_receipt_once = True
    with pytest.raises(ConnectionError):
        m.commit_unit(store, unit, b'closed predictions')
    assert m.load_units(store, 'a'*64) == {}
    m.commit_unit(store, unit, b'closed predictions')
    m.commit_unit(store, unit, b'closed predictions')
    accepted = m.load_units(store, 'a'*64)
    assert len(accepted) == 1
    assert next(iter(accepted.values()))['data'] == b'closed predictions'


def test_committed_payload_tampering_stops_recovery():
    m = implementation()
    store = Blobs()
    unit = dict(campaign='a'*64, ids=[1])
    m.commit_unit(store, unit, b'original')
    store.values[next(k for k in store.values if k.startswith('data_'))] = b'changed'
    with pytest.raises(ValueError):
        m.load_units(store, 'a'*64)


def test_wrong_campaign_cannot_be_silently_reused():
    m = implementation()
    store = Blobs()
    m.commit_unit(store, dict(campaign='a'*64, ids=[1]), b'original')
    with pytest.raises(ValueError):
        m.load_units(store, 'b'*64)


def test_budget_counts_unsettled_reservations_and_survives_a_new_process():
    import time
    m = implementation()
    store = Blobs()
    budget=m.Budget(store,cap=1000,started=time.monotonic()-40)
    with budget.operation(300):pass
    assert 70<=m.spent(store)<71
    store.put('reserve_'+'b'*32+'_000000.json',json.dumps(dict(attempt='b'*32,elapsed_before=0,seconds=300)).encode())
    assert 370<=m.spent(store)<371
    with pytest.raises(TimeoutError):
        exhausted=m.Budget(store,cap=370)
        with exhausted.operation(1):pass


def test_acknowledgement_loss_after_receipt_creation_still_resumes():
    m = implementation()
    store = Blobs()
    unit = dict(campaign='a'*64, ids=[1])
    m.commit_unit(store, unit, b'original')
    # The operator saw no acknowledgement; the cloud already has both objects.
    assert len(m.load_units(store, 'a'*64)) == 1
    with pytest.raises(ValueError):
        m.commit_unit(store, unit, b'different prediction')


def test_canonical_identity_rejects_nonfinite_configuration():
    m = implementation()
    with pytest.raises(ValueError):
        m.canonical(dict(tau=float('nan')))


def test_mount_store_commits_and_recovers_after_reopening(tmp_path):
    m=implementation();store=m.MountStore(tmp_path)
    unit=dict(campaign='a'*64,ids=[1])
    m.commit_unit(store,unit,b'predictions')
    accepted=m.load_units(m.MountStore(tmp_path),'a'*64)
    assert next(iter(accepted.values()))['data']==b'predictions'
    with pytest.raises(ValueError,match='conflict'):
        store.put(m.unit_key(unit),b'changed')


def test_mount_failure_before_atomic_publish_leaves_no_completed_unit(tmp_path,monkeypatch):
    m=implementation();store=m.MountStore(tmp_path)
    def interrupted(source,destination):raise ConnectionError('mount interrupted')
    monkeypatch.setattr(m.os,'replace',interrupted)
    with pytest.raises(ConnectionError):
        m.commit_unit(store,dict(campaign='a'*64,ids=[1]),b'predictions')
    assert m.load_units(m.MountStore(tmp_path),'a'*64)=={}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('key',['../outside','bad/name','.hidden','a..b'])
def test_mount_keys_cannot_escape_destination(tmp_path,key):
    with pytest.raises(ValueError):implementation().MountStore(tmp_path).put(key,b'data')


def test_mount_recovery_refuses_symlink_and_corrupt_payload(tmp_path):
    m=implementation();store=m.MountStore(tmp_path)
    unit=dict(campaign='a'*64,ids=[1]);m.commit_unit(store,unit,b'original')
    payload=tmp_path/next(k for k in store.keys('data_'))
    payload.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='corrupt'):
        m.load_units(m.MountStore(tmp_path),'a'*64)
    payload.unlink();payload.symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError,match='symlink'):
        store.get(payload.name)


def test_process_deadline_actually_stops_a_stalled_operation():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c',
        'from cloudfixer_store import hard_deadline; import time\nwith hard_deadline(.1): time.sleep(5)'],
        capture_output=True,text=True,timeout=3)
    assert result.returncode == -9,result.stderr


def test_cumulative_attempt_budget_covers_setup_and_lost_long_attempts():
    import time
    m=implementation();store=Blobs()
    setup=dict(attempts=[dict(session='owned1',started_utc='2026-11-14T12:00:00Z',seconds=600)])
    m.record_setup(store,setup,cap=2000)
    m.record_setup(store,setup,cap=2000)
    budget=m.Budget(store,cap=2000,started=time.monotonic()-350)
    with budget.operation(100):pass
    assert 950 <= m.spent(store) <= 985
    next_budget=m.Budget(store,cap=2000)
    assert next_budget.remaining() < 1050


def test_unfinished_attempt_counts_elapsed_before_reservation_not_just_last_300_seconds():
    m=implementation();store=Blobs()
    store.put('reserve_'+'a'*32+'_000001.json',json.dumps(dict(attempt='a'*32,seconds=300,elapsed_before=350)).encode())
    assert m.spent(store) == 650
