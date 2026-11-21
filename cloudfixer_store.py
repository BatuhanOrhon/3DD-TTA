"""Immutable batch commits on the user-authorized Google Drive mount."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import signal
import threading
import time
import uuid
from contextlib import contextmanager


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unit_key(unit: dict) -> str:
    return 'unit_' + sha(canonical(unit)) + '.json'


def commit_unit(store, unit: dict, data: bytes) -> str:
    key = unit_key(unit)
    digest = sha(data)
    payload = 'data_' + digest + '.npz'
    store.put(payload, data)
    # Publish the receipt only after the payload is closed and read back.
    store.put(key, canonical(dict(schema=1, unit=unit, payload=payload, sha256=digest, bytes=len(data))))
    return key


def load_units(store, campaign: str) -> dict:
    result = {}
    for key in store.keys('unit_'):
        receipt = json.loads(store.get(key))
        unit = receipt.get('unit', {})
        if receipt.get('schema') != 1 or unit.get('campaign') != campaign or unit_key(unit) != key:
            raise ValueError('Committed unit has another identity: ' + key)
        if receipt.get('payload') != 'data_' + receipt.get('sha256', '') + '.npz':
            raise ValueError('Invalid committed payload key')
        data = store.get(receipt['payload'])
        if sha(data) != receipt['sha256'] or len(data) != receipt['bytes']:
            raise ValueError('Committed payload is missing or corrupt: ' + key)
        result[key] = dict(receipt, data=data)
    return result


def spent(store) -> float:
    latest={}
    for key in store.keys('reserve_'):
        match=re.fullmatch(r'reserve_([a-f0-9]{32})_(\d{6})\.json',key)
        if not match:raise ValueError('Invalid budget reservation identity')
        attempt,sequence=match.groups()
        if attempt not in latest or sequence>latest[attempt][0]:latest[attempt]=(sequence,key)
    total=0.
    settled=set(store.keys('settled_'))
    for attempt,(_,key) in latest.items():
        reservation=json.loads(store.get(key))
        settled_key=key.replace('reserve_','settled_',1)
        if reservation.get('attempt')!=attempt:raise ValueError('Reservation attempt mismatch')
        if settled_key in settled:
            value=json.loads(store.get(settled_key))
            if value.get('reservation')!=key:raise ValueError('Budget settlement identity mismatch')
            seconds=value['elapsed_total']+value['tail_bound']
        else:
            seconds=reservation['elapsed_before']+reservation['seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid persisted duration')
        total+=seconds
    for key in store.keys('setup_'):
        seconds=json.loads(store.get(key))['seconds']
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
            raise ValueError('Invalid persisted setup duration')
        total+=seconds
    return total


@contextmanager
def hard_deadline(seconds: float):
    """Stop this owned worker even when Python is waiting in a native/HTTP call."""
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive operation deadline required')
    watchdog=threading.Timer(seconds,lambda:os.kill(os.getpid(),signal.SIGKILL))
    watchdog.daemon=True;watchdog.start()
    try:
        yield
    finally:
        watchdog.cancel()


def record_setup(store, receipt: dict, *, cap: float) -> None:
    for attempt in receipt['attempts']:
        seconds=attempt['seconds']
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:
            raise ValueError('Finite owned-VM setup duration required')
        key='setup_'+sha(canonical({k:attempt[k] for k in ('session','started_utc')}))+'.json'
        if key not in store.keys('setup_') and spent(store)+seconds>cap:
            raise TimeoutError('Assigned-VM setup exhausted the whole-phase budget')
        store.put(key,canonical(attempt))


class Budget:
    """Cumulative attempt time includes restore/API/report work between batch leases."""
    def __init__(self,store,*,cap:float,started=None):
        if not math.isfinite(cap) or cap<=0:raise ValueError('Positive finite campaign cap required')
        self.started=time.monotonic() if started is None else started
        self.store,self.cap,self.base=store,cap,spent(store)
        self.attempt,self.sequence=uuid.uuid4().hex,0

    def remaining(self) -> float:
        return self.cap-self.base-(time.monotonic()-self.started)

    @contextmanager
    def operation(self,seconds:float):
        limit=min(seconds,self.remaining())
        if limit<1:raise TimeoutError('Persistent campaign budget exhausted')
        key=f'reserve_{self.attempt}_{self.sequence:06d}.json'
        self.sequence+=1
        with hard_deadline(limit):
            self.store.put(key,canonical(dict(attempt=self.attempt,seconds=limit,
                elapsed_before=time.monotonic()-self.started)))
            try:
                yield
            finally:
                # Only the last settlement tail needs a conservative margin per attempt.
                value=dict(reservation=key,elapsed_total=time.monotonic()-self.started,tail_bound=30.)
                with hard_deadline(30):
                    self.store.put(key.replace('reserve_','settled_',1),canonical(value))


class MountStore:
    """Single-writer, atomic local publication on DriveFS; readback may be cached."""
    def __init__(self, folder):
        from pathlib import Path
        self.folder=Path(folder)
        if self.folder.is_symlink():raise ValueError('Destination folder cannot be a symlink')
        self.folder.mkdir(parents=True,exist_ok=True)

    def _path(self,key):
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,179}',key) or '..' in key:
            raise ValueError('Unsafe immutable object key')
        path=self.folder/key
        if path.is_symlink():raise ValueError('Mounted object cannot be a symlink')
        return path

    def keys(self,prefix):
        return sorted(p.name for p in self.folder.iterdir() if p.name.startswith(prefix)
                      and self._path(p.name).is_file())

    def get(self,key):
        path=self._path(key)
        if not path.is_file() or not 0<=path.stat().st_size<=100_000_000:
            raise ValueError('Missing or invalid mounted object: '+key)
        return path.read_bytes()

    def put(self,key,data):
        path=self._path(key)
        if len(data)>100_000_000:raise ValueError('Mounted object exceeds size limit')
        if path.exists():
            if self.get(key)!=data:raise ValueError('Immutable mounted object conflict: '+key)
            return
        temporary=self.folder/('.pending_'+uuid.uuid4().hex)
        try:
            with temporary.open('xb') as handle:
                handle.write(data);handle.flush();os.fsync(handle.fileno())
            os.replace(temporary,path)
            if self.get(key)!=data:raise ValueError('Mounted object readback failed: '+key)
        finally:
            if temporary.exists():temporary.unlink()
