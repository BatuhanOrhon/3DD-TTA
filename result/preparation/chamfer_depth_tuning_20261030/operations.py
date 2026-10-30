"""Small process/progress and verified immutable Drive persistence boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time


def file_identity(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return dict(bytes=Path(path).stat().st_size, sha256=hasher.hexdigest())


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    expected = file_identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    if file_identity(destination) != expected:
        raise ValueError('Immutable Drive copy SHA/size mismatch: ' + str(destination))
    return expected


def verify_preflight(plan):
    """Require the real CPU/setup and Drive roundtrip receipt before research."""
    receipt_path = Path(plan['cpu_preflight_receipt_path'])
    receipt = json.loads(receipt_path.read_text())
    report, expected = receipt['report'], plan['binding']['expected_config']
    if (receipt.get('status') != 'complete' or report.get('status') != 'complete'
            or report.get('cuda_initialized') is not False or report.get('forward_executed') is not False
            or report.get('manifest_sha256') != expected['asset_manifest_sha256']
            or report.get('source_snapshot_sha256') != expected['source_snapshot_sha256']
            or report.get('dataset_provenance') != expected['dataset_provenance']
            or not report.get('checkpoint_load', {}).get('strict')
            or report['checkpoint_load'].get('checkpoint_sha256') != expected['checkpoint_load']['checkpoint_sha256']):
        raise ValueError('CPU source/assets/checkpoint/data preflight receipt invalid')
    archive = receipt['archive']
    pinned = {k: archive[k] for k in ('bytes', 'sha256')}
    if not os.path.ismount('/content/drive') or not Path(plan['drive_root']).is_dir():
        raise ValueError('Actual Google Drive mount missing; no local directory fallback')
    for key in ('local_path', 'drive_path'):
        if file_identity(archive[key]) != pinned:
            raise ValueError('CPU preflight archive Drive/local roundtrip mismatch')


def run_worker(command, cwd, console, timeout, slot_id):
    """Stream meaningful worker output; bounded waits retain cumulative timeout."""
    worker = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, start_new_session=True)
    messages = queue.Queue()
    def read_lines():
        try:
            for line in worker.stdout:
                messages.put(line)
        finally:
            messages.put(None)
    threading.Thread(target=read_lines, daemon=True).start()
    started, last = time.monotonic(), 'worker started; no batch completed yet'
    try:
        with Path(console).open('x') as log:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    line = messages.get(timeout=min(20., remaining))
                except queue.Empty:
                    print('DEPTH_CAMPAIGN_PROGRESS slot=%s elapsed=%.1fs last=%s' %
                          (slot_id, time.monotonic() - started, last), flush=True)
                    continue
                if line is None:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    return worker.wait(timeout=remaining)
                log.write(line)
                log.flush()
                last = line.strip()[-400:] or last
                print(line, end='', flush=True)
    except BaseException:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
        raise


def persist_checkpoint(root, drive, plan_sha, sequence, slot=None, expected_caches=(), preserve_raw=False):
    """Copy cache bytes once; publish an immutable complete metadata snapshot.

    Snapshot receipts cover every file and cache identity. Worker-owned raw ZIPs
    retain their existing Drive protocol. An incomplete copy has no final receipt.
    """
    root, drive = Path(root), Path(drive) / 'campaign' / plan_sha
    local_index = root / 'published_cache_index.json'
    index = json.loads(local_index.read_text()) if local_index.exists() else {}
    for proof in expected_caches:
        relative = proof['relative_path']
        expected = {k: proof[k] for k in ('bytes', 'sha256')}
        if relative in index and index[relative] != expected:
            raise ValueError('Previously published immutable cache identity changed: ' + relative)
    for namespace in ('paired', 'paired-smoke'):
        for source in sorted((root / namespace).rglob('*.npz')):
            relative = source.relative_to(root).as_posix()
            if relative not in index:
                index[relative] = copy_verified(source, drive / relative)
    for proof in expected_caches:
        if index.get(proof['relative_path']) != {k: proof[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Paired raw/Drive physical cache identity mismatch')
    pending = local_index.with_suffix('.pending.json')
    pending.write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    pending.replace(local_index)
    snapshot = drive / 'snapshots' / ('%06d' % sequence)
    snapshot.mkdir(parents=True, exist_ok=False)
    files = [root / n for n in ('plan.json', 'approval.json', 'wall_ledger.jsonl', 'published_cache_index.json')]
    files += sorted(root.glob('registered_*.json'))
    if slot is not None:
        files += [p for p in Path(slot).iterdir() if p.is_file()]
        if preserve_raw:
            files += [p for p in (Path(slot) / 'raw').rglob('*') if p.is_file()]
    manifest = {}
    for source in files:
        if source.is_file():
            relative = source.relative_to(root).as_posix()
            manifest[relative] = copy_verified(source, snapshot / relative)
    receipt = dict(status='complete', plan_sha256=plan_sha, files=manifest, physical_cache_files=index,
                   raw_preservation='failure raw recursively preserved; may lack final ZIP/manifest and is scientifically unavailable'
                       if preserve_raw else 'normal raw ZIPs preserved separately by original save_drive_checkpoint')
    with (snapshot / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    if json.loads((snapshot / 'receipt.json').read_text()) != receipt:
        raise ValueError('Drive snapshot receipt readback mismatch')
    print('DEPTH_CAMPAIGN_DRIVE_VERIFIED', snapshot, 'cache_files', len(index), flush=True)
    return str(snapshot)
