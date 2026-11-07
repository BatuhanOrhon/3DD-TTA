"""Owned full-data jobs, immutable evidence and acknowledged transfer cleanup."""
import argparse
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile

ROOT = Path('/content/studies/gsd-lowband-full-20261107')
ASSETS = ROOT/'prepared/assets'
SOURCE = ASSETS/'source_snapshot'
HERE = Path(__file__).resolve().parent
DRIVE = Path('/content/drive/MyDrive/thesis/result/gsd_lowband_full_20261107')
PYTHON = '/usr/local/envs/3dd_tta_env/bin/python'
MAX_PART_BYTES = 32*1024**2


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
            size += len(block)
    return dict(bytes=size, sha256=digest.hexdigest())


def save(path, value, atomic=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = path.with_name(path.name+'.pending.'+uuid.uuid4().hex) if atomic else path
    with destination.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if atomic:
        os.replace(str(destination), str(path))


def safe_label(label):
    assert isinstance(label, str) and re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]*', label), 'Unsafe label'
    return label


def owned(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    assert path != parent and parent in path.parents, 'Path is outside the owned boundary'
    return path


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_process(label, command, timeout=7200, deadline=None):
    safe_label(label)
    logs = ROOT/'phase_logs'
    logs.mkdir(exist_ok=True)
    log = logs/(label+'.log')
    started, heartbeat = time.monotonic(), time.monotonic()
    lines = queue.Queue()
    result = dict(label=label, command=command, started_epoch=time.time(), timed_out=False)
    process = None
    thread = None
    try:
        with log.open('x') as stream:
            print('FULL_JOB_START', label, flush=True)
            process = subprocess.Popen(command, cwd=str(SOURCE), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            def read_output():
                for line in process.stdout:
                    lines.put(line)
            thread = threading.Thread(target=read_output, daemon=True)
            thread.start()
            while process.poll() is None or thread.is_alive() or not lines.empty():
                try:
                    line = lines.get(timeout=.25)
                    stream.write(line)
                    stream.flush()
                    print(line, end='', flush=True)
                except queue.Empty:
                    pass
                if process.poll() is not None and thread.is_alive():
                    stop_group(process)
                if time.monotonic()-heartbeat >= 30:
                    print('FULL_HEARTBEAT', label, 'seconds', time.monotonic()-started, flush=True)
                    heartbeat = time.monotonic()
                if time.monotonic()-started > timeout or (deadline is not None and time.time() >= deadline):
                    result['timed_out'] = True
                    stop_group(process)
                    break
            result['returncode'] = process.wait()
            if result['timed_out']:
                result['returncode'] = result['returncode'] or 124
    except BaseException:
        result.update(returncode=-1, failure=traceback.format_exc())
        raise
    finally:
        if process is not None:
            stop_group(process)
        if thread is not None:
            thread.join(timeout=2)
        result['wall_seconds'] = time.monotonic()-started
        if log.exists():
            result['log'] = identity(log)
        save(logs/(label+'.process.json'), result)
    print('FULL_JOB_END', label, result['returncode'], result['wall_seconds'], flush=True)
    return result


def split_archive(archive, label, kind, drive, part_bytes=MAX_PART_BYTES):
    safe_label(label)
    assert kind in ('inference', 'cache') and 0 < part_bytes <= MAX_PART_BYTES
    expected = identity(archive)
    assert identity(drive) == expected, 'Drive archive identity differs'
    directory = ROOT/'transfers'/label
    directory.mkdir(parents=True, exist_ok=False)
    parts = []
    digest = hashlib.sha256()
    with Path(archive).open('rb') as source:
        while True:
            block = source.read(part_bytes)
            if not block:
                break
            digest.update(block)
            part = directory/('part%05d.bin' % len(parts))
            with part.open('xb') as target:
                target.write(block)
            parts.append(dict(path=str(part), **identity(part)))
    assert sum(part['bytes'] for part in parts) == expected['bytes']
    assert digest.hexdigest() == expected['sha256'], 'Archive changed during splitting'
    manifest = dict(label=label, kind=kind, archive=dict(path=str(archive), drive=str(drive), **expected), parts=parts)
    destination = directory/'transfer_manifest.json'
    save(destination, manifest)
    return destination


def validate_ack(manifest, ack):
    assert isinstance(ack, dict), 'Ack must be a complete JSON object'
    assert ack.get('status') == 'local_and_drive_verified' and ack.get('label') == manifest['label'], 'Unaccepted local ack'
    assert type(ack.get('bytes')) is int and ack['bytes'] == manifest['archive']['bytes'], 'Ack size differs'
    assert ack.get('sha256') == manifest['archive']['sha256'], 'Ack SHA differs'
    assert isinstance(ack.get('local_archive'), str) and ack['local_archive'], 'Missing local archive path'


def read_ack(path, manifest):
    try:
        ack = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    validate_ack(manifest, ack)
    return ack


def verify_zip(path, sealed=False):
    """Reading each member to EOF checks CRC as well as the sealed SHA/size."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP members'
        for name in names:
            relative = PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name
        files = [name for name in names if not name.endswith('/')]
        observed = {}
        for name in files:
            digest, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(8*1024**2), b''):
                    digest.update(block)
                    size += len(block)
            observed[name] = dict(bytes=size, sha256=digest.hexdigest())
        if sealed:
            roots = {PurePosixPath(name).parts[0] for name in files}
            assert len(roots) == 1
            prefix = roots.pop()+'/'
            manifest = json.loads(archive.read(prefix+'bundle_manifest.json'))
            actual = {name[len(prefix):]: value for name, value in observed.items()
                      if name != prefix+'bundle_manifest.json'}
            assert actual == manifest, 'Sealed ZIP member SHA/size mismatch'
            return prefix, set(actual)
    return None


def validate_bundle(path, shard, arm, plan, common=None):
    import numpy as np
    prefix, names = verify_zip(path, sealed=True)
    assert {'command.txt', 'config.json', 'environment.txt', 'stdout.log', 'summary.csv',
            'per_corruption.csv', 'notes.md'} <= names, 'Missing standard artifacts'
    with zipfile.ZipFile(path) as archive:
        def read_json(name):
            return json.loads(archive.read(prefix+name))
        config = read_json('config.json')
        assert config['status'] == 'partial' and config['diagnostic_complete'] is True
        assert config['benchmark_shard'] is True and config['count'] == shard['count']
        assert config['dataset'] == 'modelnet40_c' and config['selected_original_ids'] == shard['indices']
        assert config['corruptions'] == plan['corruptions']
        assert config['completed_batches'] == {name: shard['count']//2 for name in plan['corruptions']}
        assert (config['seed'], config['severity'], config['batch_size']) == (0, 5, 2)
        assert config['guidance_protocol'] == dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False)
        assert (config['normal_reverse_steps'], config['background_reverse_steps']) == (8, 35)
        assert config['backend_policy'] == 'controlled'
        runtime = config['runtime']
        assert runtime['cudnn_deterministic'] is True
        assert all(runtime[name] is False for name in ('cudnn_benchmark', 'cudnn_allow_tf32', 'cuda_matmul_allow_tf32'))
        assert all(config.get(name) is False for name in ('freeze_decoder_residual', 'update_norm_probes',
            'capture_h5', 'capture_native_trace', 'capture_shared_states', 'native_decoder_audit'))
        assert config['asset_manifest_sha256'] == plan['asset_manifest_sha256']
        assert Path(config['asset_root']).resolve() == ASSETS.resolve()
        assert Path(config['paired_state_root']).resolve() == (ROOT/'paired_states').resolve()
        expected_arm = {'control': 'control', 'half0': 'live_control', 'M100': 'physical'}[arm]
        assert config['arm'] == expected_arm
        assert config['weight'] == (2216.8064620051765 if arm == 'M100' else 0.)
        assert config['checkpoint_scd_weight'] == (1. if arm == 'control' else .5)
        assert config['lowband_modes'] == (None if arm == 'control' else 100)
        assert config['half_mix_alpha'] == (4433.612924010353 if arm == 'M100' else 0. if arm == 'half0' else None)
        assert config['physical_graph'] == (None if arm == 'control' else
            dict(k=10, delta=.1, graph_gamma=.6, modes=100, profile='hard', beta=None))
        assert config['lion_state_before'] == config['lion_state_after'] and config['lion_state_before']
        assert config['classifier_state_before_sha256'] == config['classifier_state_after_sha256']
        keys = ('executed_source_files', 'native_extensions', 'native_bootstrap', 'asset_manifest_sha256',
                'lion_state_before', 'classifier_state_before_sha256', 'runtime')
        observed = {key: config[key] for key in keys}
        assert all(observed.values()), 'Missing source/native/model identity'
        if common is not None:
            assert common == observed, 'Frozen source/native/model/backend identity changed'
        asset_manifest = json.loads((ASSETS/'manifest.json').read_text())
        assert identity(ASSETS/'manifest.json')['sha256'] == plan['asset_manifest_sha256']
        for name, expected in config['executed_source_files'].items():
            if name.startswith('assets/'):
                source = owned(ASSETS/name[7:], ASSETS)
                assert asset_manifest['files'][name[7:]] == expected
            else:
                assert name.startswith('repo/')
                source = owned(SOURCE/name[5:], SOURCE)
            assert identity(source) == expected, 'Executed source bytes changed'
        for expected in list(config['native_extensions'].values()) + list(config['native_bootstrap'].get('bindings', {}).values()):
            if expected.get('path'):
                assert identity(expected['path']) == {key: expected[key] for key in ('bytes', 'sha256')}
        pairing, graphs = {}, {}
        for corruption in plan['corruptions']:
            with np.load(io.BytesIO(archive.read(prefix+corruption+'_predictions.npz')), allow_pickle=False) as prediction:
                assert prediction['index'].dtype.kind in 'iu' and prediction['index'].tolist() == shard['indices']
                for name in ('label', 'prediction', 'terminal_prediction'):
                    values = prediction[name]
                    assert values.shape == (shard['count'],) and values.dtype.kind in 'iu'
                    assert bool(((values >= 0) & (values < 40)).all())
                assert prediction['logits'].shape == (shard['count'], 40)
                assert prediction['trajectory_logits'].shape == (shard['count'], 4, 40)
                assert np.isfinite(prediction['logits']).all() and np.isfinite(prediction['trajectory_logits']).all()
            pairing[corruption], graphs[corruption] = [], []
            for batch in range(shard['count']//2):
                base = 'batches/%s/batch%04d/' % (corruption, batch)
                data = read_json(base+'diagnostics.json')
                trajectory = data['trajectory']
                assert data['original_ids'] == shard['indices'][batch*2:batch*2+2]
                assert trajectory['complete'] is True and trajectory['probe_rng_unchanged'] is True
                assert trajectory['probe_buffers_unchanged'] is True
                pairing[corruption].append(dict(original_ids=data['original_ids'], prepared=data['prepared_x_sha256'],
                    paired=data['paired_state']['sha256'], initial=trajectory['initial_hashes'], rng=trajectory['initial_rng_sha256']))
                if arm != 'control':
                    assert base+'physical_bases.npz' in names, 'Missing retained graph bases'
                    graph = read_json(base+'graph.json')
                    assert graph['reference_xyz_sha256'] == data['prepared_x_sha256'] and len(graph['basis_sha256']) == 2
                    graphs[corruption].append({key: graph[key] for key in
                        ('reference_xyz_sha256', 'filter_sha256', 'basis_sha256', 'active_vertex_indices')})
        return dict(count=config['count'], common=observed, pairing=pairing, graphs=graphs)


def validate_caches(path, shard, plan):
    verify_zip(path)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('package_manifest.json'))
        assert set(archive.namelist()) == set(manifest) | {'package_manifest.json'}
        for name, expected in manifest.items():
            raw = archive.read(name)
            assert dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == expected
        report = json.loads(archive.read('records/cache_preparation_'+shard['label']+'.json'))
    assert report['status'] == 'complete' and report['label'] == shard['label']
    assert report['selection']['indices'] == shard['indices']
    expected_cases = {(corruption, tuple(shard['indices'][start:start+2]))
                      for corruption in plan['corruptions'] for start in range(0, shard['count'], 2)}
    records = report['records']
    assert len(records) == len(expected_cases)
    assert {(row['corruption'], tuple(row['ids'])) for row in records} == expected_cases
    for row in records:
        expected = {key: row[key] for key in ('bytes', 'sha256')}
        assert manifest[row['member']] == expected
        cache = owned(ROOT/row['member'], ROOT/'paired_states')
        assert cache == Path(row['path']).resolve() and identity(cache) == expected
    return report


def inference_command(shard, arm, plan):
    label = shard['label']+'_'+arm
    selection = Path(shard['selection_file'])
    flags = ['--arm', 'control'] if arm == 'control' else ['--arm',
        'physical' if arm == 'M100' else 'live_control', '--lowband-modes', '100', '--checkpoint-scd-weight', '.5']
    if arm == 'M100':
        flags += ['--weight', '2216.8064620051765']
    return [PYTHON, '-u', str(SOURCE/'scripts/diagnose_decoded_spectral.py'), '--benchmark-shard',
        '--asset-root', str(ASSETS), '--asset-manifest-sha256', plan['asset_manifest_sha256'],
        '--result-root', str(ROOT/'raw_runs'/label), '--paired-state-root', str(ROOT/'paired_states'),
        '--drive-root', str(DRIVE/'runs'/label), '--backend-policy', 'controlled',
        '--index-manifest', str(selection), '--index-manifest-sha256', identity(selection)['sha256'],
        '--count', str(shard['count']), '--corruptions', *plan['corruptions'], *flags]


def cleanup_drive_batches(directory, expected):
    directory = owned(directory, DRIVE/'runs')
    final_receipt = json.loads((directory/'final.receipt.json').read_text())
    assert final_receipt['status'] == 'complete'
    assert {key: final_receipt['archive'][key] for key in ('bytes', 'sha256')} == expected
    destination = directory/'interim_receipts.zip'
    if not destination.exists():
        records = {str(path.relative_to(directory)): path.read_bytes()
                   for path in sorted((directory/'batches').rglob('receipt.json'))}
        sealed = {name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                  for name, raw in records.items()}
        with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
            for name, raw in records.items():
                archive.writestr(name, raw)
            archive.writestr('receipts_manifest.json', json.dumps(sealed, sort_keys=True))
    verify_zip(destination)
    with zipfile.ZipFile(destination) as archive:
        sealed = json.loads(archive.read('receipts_manifest.json'))
        assert set(archive.namelist()) == set(sealed) | {'receipts_manifest.json'}
        for name, expected_member in sealed.items():
            raw = archive.read(name)
            assert dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == expected_member
    removed = []
    batches = owned(directory/'batches', directory)
    if batches.exists():
        assert set(str(path.relative_to(directory)) for path in batches.rglob('receipt.json')) <= set(sealed)
        for path in batches.rglob('receipt.json'):
            assert identity(path) == sealed[str(path.relative_to(directory))], 'Interim receipt changed'
        shutil.rmtree(str(batches))
        removed.append(str(batches))
    cleanup = directory/'cleanup.receipt.json'
    if not cleanup.exists():
        save(cleanup, dict(status='cleaned_successful_duplicates', final_archive=expected,
            interim_receipts=dict(path=str(destination), **identity(destination)), removed=removed))
    return str(cleanup)


def prune_transfer(manifest_path, receipt, ack=None):
    manifest_path = owned(manifest_path, ROOT/'transfers')
    manifest = json.loads(manifest_path.read_text())
    label = safe_label(manifest['label'])
    assert manifest_path == (ROOT/'transfers'/label/'transfer_manifest.json').resolve()
    assert receipt['status'] == 'accepted' and receipt['label'] == label and receipt['kind'] == manifest['kind']
    assert receipt['archive'] == manifest['archive'], 'Receipt differs from transfer'
    if ack is None:
        ack = json.loads((ROOT/'acks'/(label+'.json')).read_text())
    validate_ack(manifest, ack)
    archive = Path(manifest['archive']['path'])
    drive = owned(manifest['archive']['drive'], DRIVE)
    expected = {key: manifest['archive'][key] for key in ('bytes', 'sha256')}
    assert identity(drive) == expected, 'Drive archive changed before cleanup'
    if manifest['kind'] == 'inference':
        bundle = owned(receipt['bundle'], ROOT/'raw_runs'/label)
        assert archive.resolve() == Path(str(bundle)+'.zip')
        assert drive == (DRIVE/'runs'/label/'modelnet40_c'/'decoded_physical_spectral_diagnostic'/bundle.name/'final.zip').resolve()
        if bundle.exists():
            config = json.loads((bundle/'config.json').read_text())
            assert config['status'] == 'partial' and config['diagnostic_complete'] is True
    else:
        assert label.endswith('_cache')
        bundle = None
        assert archive.resolve() == (ROOT/'cache_archives'/('cache_'+label[:-6]+'.zip')).resolve()
        assert drive == (DRIVE/'cache_archives'/archive.name).resolve()
    for part in manifest['parts']:
        owned(part['path'], ROOT/'transfers'/label)
    cleanup_path = ROOT/'cleanup_receipts'/(label+'.json')
    if cleanup_path.exists():
        previous = json.loads(cleanup_path.read_text())
        assert previous['archive'] == manifest['archive'] and previous['ack'] == ack
        return previous
    removed = []
    if archive.exists():
        assert identity(archive) == expected
    drive_cleanup = cleanup_drive_batches(drive.parent, expected) if bundle is not None else None
    if bundle is not None and bundle.exists():
        shutil.rmtree(str(bundle))
        removed.append(str(bundle))
    for path in [archive, *[Path(part['path']) for part in manifest['parts']]]:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    cleanup = dict(label=label, archive=manifest['archive'], ack=ack, removed=removed,
                   drive_cleanup_receipt=drive_cleanup, status='cleaned')
    save(cleanup_path, cleanup)
    print('FULL_CLEANUP', json.dumps(cleanup, sort_keys=True), flush=True)
    return cleanup


def load_plan():
    sys.path.insert(0, str(SOURCE))
    from research_artifacts import CORRUPTIONS
    plan = json.loads((HERE/'plan.json').read_text())
    assert plan['corruptions'] == list(CORRUPTIONS) and len(CORRUPTIONS) == 15
    assert plan['frozen_alpha100'] == 4433.612924010353
    assert plan['calibration_sha256'] == 'fb121b127d0ac78ed8ac1ee8dc500d52c5f2a631c4dabaac2a932d6595eca9a1'
    assert identity(plan['calibration_path'])['sha256'] == plan['calibration_sha256']
    assert identity(ASSETS/'manifest.json')['sha256'] == plan['asset_manifest_sha256']
    assert len(plan['shards']) == 39
    all_ids = []
    for number, shard in enumerate(plan['shards']):
        assert shard['number'] == number and shard['label'] == 'shard%03d' % number
        assert shard['count'] == (36 if number == 38 else 64)
        assert len(shard['arms']) == 3 and set(shard['arms']) == {'control', 'half0', 'M100'}
        selection_file = owned(HERE/shard['selection_file'], HERE)
        assert selection_file.stem == shard['label']
        selection = json.loads(selection_file.read_text())
        assert type(selection['schema_version']) is int and selection['schema_version'] == 1
        assert selection['dataset'] == 'modelnet40_c'
        ids = selection['indices']
        assert len(ids) == shard['count'] and all(type(index) is int for index in ids)
        assert ids == list(range(number*64, min((number+1)*64, 2468)))
        shard.update(selection_file=str(selection_file), indices=ids)
        all_ids.extend(ids)
    assert all_ids == list(range(2468))
    return plan


def announce(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    print('FULL_TRANSFER_READY '+json.dumps(dict(label=manifest['label'], kind=manifest['kind'],
        manifest=str(manifest_path), archive_identity={key: manifest['archive'][key]
        for key in ('bytes', 'sha256')}), sort_keys=True), flush=True)


def science_start():
    current = time.time()
    path = ROOT/'science_started.json'
    if not path.exists():
        return current
    started = json.loads(path.read_text())['started_epoch']
    assert type(started) in (int, float) and math.isfinite(started) and 0 < started <= current, 'Invalid science start epoch'
    return started


def run_campaign(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('full',), default='full')
    parser.add_argument('--budget-hours', type=float, default=48.)
    parser.add_argument('--ack-timeout', type=float, default=3600.)
    parser.add_argument('--resume', action='store_true', help='Verify accepted receipts; never rerun a failed/partial job')
    args = parser.parse_args(argv)
    assert 40 <= args.budget_hours <= 168 and 0 < args.ack_timeout <= 7200
    plan = load_plan()
    binding = dict(plan=identity(HERE/'plan.json'), campaign=identity(__file__),
                   prepare_caches=identity(HERE/'prepare_caches.py'))
    journal_path = ROOT/'campaign_journal.json'
    if journal_path.exists():
        assert args.resume, 'Existing campaign requires explicit --resume and verified receipts'
        journal = json.loads(journal_path.read_text())
        assert journal['binding'] == binding and journal['budget_seconds'] == args.budget_hours*3600
        if (ROOT/'science_started.json').exists():
            assert journal['started_epoch'] == science_start(), 'Science clock changed; do not reset the deadline'
    else:
        assert not args.resume, 'No prior journal to resume'
        journal = dict(status='running', started_epoch=science_start(), budget_seconds=args.budget_hours*3600,
                       binding=binding, jobs={}, expected_inference_jobs=117, expected_cache_jobs=39)
    deadline = journal['started_epoch']+journal['budget_seconds']
    (ROOT/'acks').mkdir(exist_ok=True)
    receipts, manifests = {}, {}
    common_path = ROOT/'common_identity.json'
    common = json.loads(common_path.read_text()) if common_path.exists() else None

    def checkpoint():
        journal['elapsed_seconds'] = time.time()-journal['started_epoch']
        journal['updated_epoch'] = time.time()
        save(journal_path, journal, atomic=True)
        save(DRIVE/'campaign_journal.json', journal, atomic=True)

    def reap(wait_below=None):
        started, heartbeat = time.monotonic(), time.monotonic()
        while True:
            pending = []
            for label, manifest_path in manifests.items():
                if journal['jobs'][label].get('cleanup_receipt'):
                    continue
                manifest = json.loads(Path(manifest_path).read_text())
                ack = read_ack(ROOT/'acks'/(label+'.json'), manifest)
                if ack is not None:
                    prune_transfer(manifest_path, receipts[label], ack=ack)
                    journal['jobs'][label].update(status='acked_and_cleaned',
                        cleanup_receipt=str(ROOT/'cleanup_receipts'/(label+'.json')))
                    checkpoint()
                else:
                    pending.append(label)
            if wait_below is None or len(pending) < wait_below:
                return
            assert time.monotonic()-started <= args.ack_timeout and time.time() < deadline, 'Local archive ack deadline exceeded; preserve all pending data'
            if time.monotonic()-heartbeat >= 30:
                print('FULL_ACK_HEARTBEAT', len(pending), pending, flush=True)
                heartbeat = time.monotonic()
            time.sleep(1)

    def transfer(receipt):
        label = receipt['label']
        manifests[label] = ROOT/'transfers'/label/'transfer_manifest.json'
        if not manifests[label].exists():
            created = split_archive(receipt['archive']['path'], label, receipt['kind'], receipt['archive']['drive'])
            assert created == manifests[label]
        else:
            manifest = json.loads(manifests[label].read_text())
            assert manifest['archive'] == receipt['archive'] and manifest['label'] == label
        announce(manifests[label])
        journal['jobs'].setdefault(label, {}).pop('cleanup_receipt', None)
        journal['jobs'].setdefault(label, {}).update(status='accepted_pending_ack', receipt=str(ROOT/'job_receipts'/(label+'.json')))
        checkpoint()
        reap()

    def execute(shard, arm=None):
        nonlocal common
        label = shard['label']+'_'+(arm or 'cache')
        receipt_path = ROOT/'job_receipts'/(label+'.json')
        if receipt_path.exists():
            assert args.resume, 'Existing immutable job receipt; explicit resume required'
            receipt = json.loads(receipt_path.read_text())
            assert identity(receipt_path) == identity(DRIVE/'job_receipts'/(label+'.json')), 'Receipt backup differs'
            assert receipt['status'] == 'accepted' and receipt['binding'] == binding and receipt['label'] == label
            assert receipt['selection_sha256'] == identity(shard['selection_file'])['sha256']
            if arm:
                assert common == receipt['validation']['common']
        else:
            assert label not in journal['jobs'], 'Partial/failed attempt is immutable; do not rerun'
            assert time.time() < deadline, 'Whole campaign budget exhausted'
            reap(wait_below=3)
            journal['jobs'][label] = dict(status='running', started_epoch=time.time())
            checkpoint()
            command = inference_command(shard, arm, plan) if arm else [PYTHON, '-u', str(HERE/'prepare_caches.py'), '--selection', shard['selection_file']]
            result = run_process(label, command, timeout=7200, deadline=deadline)
            assert result['returncode'] == 0 and not result['timed_out'], 'Worker failed; retain immutable attempt'
            receipt = dict(status='accepted', label=label, kind='inference' if arm else 'cache',
                           binding=binding, selection_sha256=identity(shard['selection_file'])['sha256'], process=result)
            if arm:
                configs = list((ROOT/'raw_runs'/label).rglob('config.json'))
                assert len(configs) == 1, 'Require exactly one sealed job bundle'
                bundle = configs[0].parent
                archive = Path(str(bundle)+'.zip')
                validation = validate_bundle(archive, shard, arm, plan, common)
                assert validation['common']['lion_state_before'] == receipts[shard['label']+'_cache']['cache_report']['lion_state']
                assert validation['common']['native_bootstrap'] == receipts[shard['label']+'_cache']['cache_report']['native']
                if common is None:
                    common = validation['common']
                    save(common_path, common)
                    save(DRIVE/'common_identity.json', common)
                for previous in shard['arms']:
                    other = receipts.get(shard['label']+'_'+previous)
                    if other:
                        assert other['validation']['pairing'] == validation['pairing'], 'Prepared cache/state/noise/RNG pairing changed'
                        if arm != 'control' and previous != 'control':
                            assert other['validation']['graphs'] == validation['graphs'], 'Retained graph bases changed'
                receipt.update(bundle=str(bundle), validation=validation)
                drive = DRIVE/'runs'/label/'modelnet40_c'/'decoded_physical_spectral_diagnostic'/bundle.name/'final.zip'
            else:
                archive = ROOT/'cache_archives'/('cache_'+shard['label']+'.zip')
                receipt['cache_report'] = validate_caches(archive, shard, plan)
                drive = DRIVE/'cache_archives'/archive.name
            expected = identity(archive)
            assert identity(drive) == expected, 'Full Drive ZIP differs from accepted archive'
            receipt['archive'] = dict(path=str(archive), drive=str(drive), **expected)
            save(receipt_path, receipt)
            save(DRIVE/'job_receipts'/(label+'.json'), receipt)
        receipts[label] = receipt
        transfer(receipt)
        return receipt

    heartbeat_stop = threading.Event()
    def phase_heartbeat():
        while not heartbeat_stop.wait(30):
            print('FULL_PHASE_HEARTBEAT', 'elapsed_seconds', time.time()-journal['started_epoch'],
                  'accepted_transfers', len(receipts), flush=True)
    heartbeat_thread = threading.Thread(target=phase_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        journal['status'] = 'running'
        checkpoint()
        for shard in plan['shards']:
            cache = execute(shard)
            for arm in shard['arms']:
                for row in cache['cache_report']['records']:
                    assert identity(row['path']) == {key: row[key] for key in ('bytes', 'sha256')}, 'Prepared cache changed'
                execute(shard, arm)
                reap()
        reap(wait_below=1)
        assert len(receipts) == 156 and all(job['status'] == 'acked_and_cleaned' for job in journal['jobs'].values())
        assert sum(receipt['kind'] == 'inference' for receipt in receipts.values()) == 117
        journal['status'] = 'complete'
        checkpoint()
        complete_path = ROOT/'campaign_complete.json'
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            assert complete['status'] == 'complete' and complete['binding'] == binding
            assert (complete['inference_jobs'], complete['cache_jobs']) == (117, 39)
        else:
            save(complete_path, dict(status='complete', inference_jobs=117, cache_jobs=39,
                 binding=binding, journal_identity=identity(journal_path)))
        print('FULL_CAMPAIGN_COMPLETE', flush=True)
    except BaseException:
        journal.update(status='failed', failure=traceback.format_exc())
        checkpoint()
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)


def main(argv=None):
    with (ROOT/'campaign.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Campaign lock is held by another live controller') from None
        return run_campaign(argv)


if __name__ == '__main__':
    main()
