"""CPU assertions for immutable transfers, strict cleanup and process ownership."""
import builtins
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

spec = importlib.util.spec_from_file_location('full_campaign', Path(__file__).with_name('campaign.py'))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def rejects(action):
    try:
        action()
    except (AssertionError, ValueError, FileNotFoundError, FileExistsError):
        return
    raise AssertionError('Invalid input was accepted')


with tempfile.TemporaryDirectory(prefix='full-campaign-selfcheck-') as temporary:
    root = Path(temporary)
    campaign.ROOT = root/'owned'
    campaign.ROOT.mkdir()
    campaign.SOURCE = campaign.ROOT
    campaign.DRIVE = root/'drive'
    campaign.DRIVE.mkdir()
    (campaign.ROOT/'acks').mkdir()
    before = time.time()
    assert before <= campaign.science_start() <= time.time()
    science_started = before-300
    (campaign.ROOT/'science_started.json').write_text(json.dumps(dict(started_epoch=science_started)))
    assert campaign.science_start() == science_started
    for bad in (time.time()+3600, 0, True, 'yesterday'):
        (campaign.ROOT/'science_started.json').write_text(json.dumps(dict(started_epoch=bad)))
        rejects(campaign.science_start)
    (campaign.ROOT/'science_started.json').unlink()
    label = 'shard000_cache'
    archive = campaign.ROOT/'cache_archives/cache_shard000.zip'
    archive.parent.mkdir()
    archive.write_bytes(bytes(range(256))*11)
    drive = campaign.DRIVE/'cache_archives'/archive.name
    drive.parent.mkdir()
    drive.write_bytes(archive.read_bytes())
    manifest_path = campaign.split_archive(archive, label, 'cache', drive, part_bytes=513)
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['parts']) == 6
    assert b''.join(Path(p['path']).read_bytes() for p in manifest['parts']) == archive.read_bytes()
    for part in manifest['parts']:
        assert part['bytes'] <= 513 and campaign.identity(part['path']) == {
            'bytes': part['bytes'], 'sha256': part['sha256']}
    rejects(lambda: campaign.split_archive(archive, label, 'cache', drive, part_bytes=513))
    ack = dict(status='local_and_drive_verified', label=label, local_archive='/mac/cache.zip',
               bytes=manifest['archive']['bytes'], sha256=manifest['archive']['sha256'])
    campaign.validate_ack(manifest, ack)
    partial_ack = campaign.ROOT/'partial_ack.json'
    for raw in ('', '{', '{"status":'):
        partial_ack.write_text(raw)
        assert campaign.read_ack(partial_ack, manifest) is None
    partial_ack.write_text(json.dumps(dict(ack, sha256='wrong')))
    rejects(lambda: campaign.read_ack(partial_ack, manifest))
    partial_ack.write_text(json.dumps(ack))
    assert campaign.read_ack(partial_ack, manifest) == ack
    for changed in (dict(status='copied'), dict(label='other'), dict(bytes=1), dict(sha256='b'*64),
                    dict(local_archive=''), dict(bytes=True)):
        rejects(lambda changed=changed: campaign.validate_ack(manifest, dict(ack, **changed)))
    receipt = dict(status='accepted', label=label, kind='cache', archive=manifest['archive'])
    rejects(lambda: campaign.prune_transfer(manifest_path, receipt))
    assert archive.exists()
    ack_path = campaign.ROOT/'acks'/(label+'.json')
    ack_path.write_text(json.dumps(ack))
    drive.write_bytes(b'bad')
    rejects(lambda: campaign.prune_transfer(manifest_path, receipt))
    assert archive.exists()
    drive.write_bytes(archive.read_bytes())
    rejects(lambda: campaign.prune_transfer(manifest_path, dict(receipt, status='failed')))
    foreign = root/'unowned.zip'
    foreign.write_bytes(archive.read_bytes())
    changed = dict(manifest, archive=dict(manifest['archive'], path=str(foreign)))
    manifest_path.write_text(json.dumps(changed))
    rejects(lambda: campaign.prune_transfer(manifest_path, dict(receipt, archive=changed['archive'])))
    assert foreign.exists()
    manifest_path.write_text(json.dumps(manifest))
    campaign.prune_transfer(manifest_path, receipt)
    assert not archive.exists() and drive.exists() and foreign.exists()
    assert not any(Path(p['path']).exists() for p in manifest['parts'])
    campaign.prune_transfer(manifest_path, receipt)

    code = '''import json,os,subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid]))
try:
 print('CHILD_READY',flush=True)
 time.sleep(60)
finally:
 child.terminate();child.wait()
'''
    for failure in ('output', 'timeout'):
        pidfile = root/(failure+'.pids.json')
        def failed_output(*args, **kwargs):
            if args and str(args[0]).startswith('CHILD_READY'):
                raise RuntimeError('Synthetic output failure')
            builtins.print(*args, **kwargs)
        campaign.print = failed_output if failure == 'output' else builtins.print
        command = [sys.executable, '-u', '-c', code, str(pidfile)]
        try:
            result = campaign.run_process(failure, command, timeout=.4 if failure == 'timeout' else 10)
        except RuntimeError as error:
            assert failure == 'output' and str(error) == 'Synthetic output failure'
        else:
            assert failure == 'timeout' and result['timed_out'] and result['returncode'] != 0
        for pid in json.loads(pidfile.read_text()):
            status = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
            assert status.returncode != 0 or status.stdout.strip().startswith('Z'), (pid, status.stdout)
    campaign.print = builtins.print

    orphan_pidfile = root/'orphan.pids.json'
    orphan_code = '''import json,os,subprocess,sys
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid]))
raise SystemExit(7)
'''
    tick = time.monotonic()
    result = campaign.run_process('orphan', [sys.executable, '-u', '-c', orphan_code, str(orphan_pidfile)], timeout=10)
    assert result['returncode'] == 7 and not result['timed_out'] and time.monotonic()-tick < 5
    for pid in json.loads(orphan_pidfile.read_text()):
        status = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
        assert status.returncode != 0 or status.stdout.strip().startswith('Z'), (pid, status.stdout)

    import io
    import numpy as np
    campaign.ASSETS = campaign.ROOT/'prepared/assets'
    campaign.SOURCE = campaign.ASSETS/'source_snapshot'
    campaign.SOURCE.mkdir(parents=True)
    source_file = campaign.SOURCE/'fixture.py'
    source_file.write_text('print("frozen")\n')
    campaign.save(campaign.ASSETS/'manifest.json', dict(files={'source_snapshot/fixture.py': campaign.identity(source_file)}))
    plan = dict(asset_manifest_sha256=campaign.identity(campaign.ASSETS/'manifest.json')['sha256'],
                corruptions=['gaussian'], frozen_alpha100=4433.612924010353)
    shard = dict(label='shard000', count=2, indices=[0, 1])
    config = dict(status='partial', diagnostic_complete=True, benchmark_shard=True, count=2,
        dataset='modelnet40_c', selected_original_ids=[0, 1], corruptions=['gaussian'], completed_batches={'gaussian': 1},
        asset_manifest_sha256=plan['asset_manifest_sha256'], seed=0, severity=5, batch_size=2,
        guidance_protocol=dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False),
        normal_reverse_steps=8, background_reverse_steps=35, arm='control', weight=0.,
        checkpoint_scd_weight=1., lowband_modes=None, half_mix_alpha=None, physical_graph=None,
        backend_policy='controlled', runtime=dict(cudnn_benchmark=False, cudnn_deterministic=True,
            cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False),
        lion_state_before={'vae': 'frozen', 'priors': 'frozen'}, lion_state_after={'vae': 'frozen', 'priors': 'frozen'},
        classifier_state_before_sha256='frozen', classifier_state_after_sha256='frozen',
        executed_source_files={'assets/source_snapshot/fixture.py': campaign.identity(source_file)},
        native_extensions={'fixture': campaign.identity(source_file)}, native_bootstrap={'fixture': 'frozen'},
        paired_state_root=str(campaign.ROOT/'paired_states'), asset_root=str(campaign.ASSETS),
        freeze_decoder_residual=False, update_norm_probes=False, capture_h5=False, capture_native_trace=False,
        capture_shared_states=False, native_decoder_audit=False)
    def bundle_fixture(name, changes=None, altered_member=False):
        contents = {name: b'fixture\n' for name in ('command.txt', 'environment.txt', 'stdout.log',
                    'summary.csv', 'per_corruption.csv', 'notes.md')}
        contents['config.json'] = json.dumps(dict(config, **(changes or {}))).encode()
        data = io.BytesIO()
        np.savez_compressed(data, index=np.array([0, 1]), label=np.array([0, 1]),
            prediction=np.array([0, 1]), terminal_prediction=np.array([0, 1]),
            logits=np.zeros((2, 40)), trajectory_logits=np.zeros((2, 4, 40)))
        contents['gaussian_predictions.npz'] = data.getvalue()
        trajectory = dict(complete=True, probe_rng_unchanged=True, probe_buffers_unchanged=True,
                          initial_hashes={'initial': 'same'}, initial_rng_sha256='same')
        contents['batches/gaussian/batch0000/diagnostics.json'] = json.dumps(dict(original_ids=[0, 1],
            prepared_x_sha256='same', paired_state={'sha256': 'same'}, trajectory=trajectory)).encode()
        manifest = {name: dict(bytes=len(value), sha256=__import__('hashlib').sha256(value).hexdigest())
                    for name, value in contents.items()}
        if altered_member:
            contents['command.txt'] = b'tampered\n'
        path = campaign.ROOT/(name+'.zip')
        with zipfile.ZipFile(path, 'x') as archive:
            for member, value in contents.items():
                archive.writestr('bundle/'+member, value)
            archive.writestr('bundle/bundle_manifest.json', json.dumps(manifest))
        return path
    good = campaign.validate_bundle(bundle_fixture('good'), shard, 'control', plan)
    assert good['count'] == 2 and good['pairing']['gaussian'][0]['original_ids'] == [0, 1]
    for name, changes in (
        ('incomplete', dict(diagnostic_complete=False)), ('wrongids', dict(selected_original_ids=[0, 2])),
        ('changedmodel', dict(lion_state_after={})), ('wrongweight', dict(weight=1.)),
        ('missingbatch', dict(completed_batches={'gaussian': 0})), ('probes', dict(update_norm_probes=True))):
        rejects(lambda name=name, changes=changes: campaign.validate_bundle(bundle_fixture(name, changes), shard, 'control', plan))
    rejects(lambda: campaign.validate_bundle(bundle_fixture('tampered', altered_member=True), shard, 'control', plan))
    rejects(lambda: campaign.validate_bundle(bundle_fixture('changednative', dict(native_extensions={'different': {}})),
        shard, 'control', plan, common=good['common']))

    label = 'shard001_control'
    bundle = campaign.ROOT/'raw_runs'/label/'modelnet40_c/decoded_physical_spectral_diagnostic/bundle'
    bundle.mkdir(parents=True)
    (bundle/'config.json').write_text(json.dumps(config))
    archive = Path(str(bundle)+'.zip')
    archive.write_bytes(b'accepted sealed ZIP fixture')
    directory = campaign.DRIVE/'runs'/label/'modelnet40_c/decoded_physical_spectral_diagnostic/bundle'
    directory.mkdir(parents=True)
    drive = directory/'final.zip'
    drive.write_bytes(archive.read_bytes())
    (directory/'final.receipt.json').write_text(json.dumps(dict(status='complete', archive=dict(path=str(drive), **campaign.identity(drive)))))
    interim = directory/'batches/gaussian/batch0000'
    interim.mkdir(parents=True)
    (interim/'receipt.json').write_text('{"status":"complete"}\n')
    (interim/'large.npz').write_bytes(b'duplicate data')
    manifest_path = campaign.split_archive(archive, label, 'inference', drive, part_bytes=513)
    manifest = json.loads(manifest_path.read_text())
    ack = dict(status='local_and_drive_verified', label=label, local_archive='/mac/inference.zip',
               bytes=manifest['archive']['bytes'], sha256=manifest['archive']['sha256'])
    (campaign.ROOT/'acks'/(label+'.json')).write_text(json.dumps(ack))
    receipt = dict(status='accepted', label=label, kind='inference', archive=manifest['archive'], bundle=str(bundle))
    campaign.prune_transfer(manifest_path, receipt)
    assert drive.exists() and (directory/'final.receipt.json').exists()
    assert (directory/'cleanup.receipt.json').exists() and not (directory/'batches').exists()
    with zipfile.ZipFile(directory/'interim_receipts.zip') as archive:
        assert archive.read('batches/gaussian/batch0000/receipt.json') == b'{"status":"complete"}\n'
        assert archive.testzip() is None
    assert not bundle.exists()

    selected = campaign.ROOT/'selection.json'
    selected.write_text(json.dumps(dict(schema_version=1, dataset='modelnet40_c', indices=[0, 1])))
    shard['selection_file'] = str(selected)
    arguments = campaign.inference_command(shard, 'M100', plan)
    assert '--benchmark-shard' in arguments and arguments[arguments.index('--count')+1] == '2'
    assert arguments[arguments.index('--weight')+1] == '2216.8064620051765'
    assert arguments[arguments.index('--checkpoint-scd-weight')+1] == '.5'
    assert arguments[arguments.index('--index-manifest-sha256')+1] == campaign.identity(selected)['sha256']
    assert '--update-norm-probes' not in arguments

    cache_file = campaign.ROOT/'paired_states/modelnet40_c/gaussian/fixture.npz'
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b'actual synthetic cached tensor bytes')
    cache_row = dict(corruption='gaussian', ids=[0, 1], member=str(cache_file.relative_to(campaign.ROOT)),
                     path=str(cache_file), **campaign.identity(cache_file))
    report = dict(status='complete', label='shard000', selection=json.loads(selected.read_text()),
                  records=[cache_row], lion_state={'vae': 'frozen'}, native={'binding': 'frozen'})
    cache_zip = campaign.ROOT/'cache.zip'
    raw = json.dumps(report).encode()
    entries = {cache_row['member']: campaign.identity(cache_file), 'records/cache_preparation_shard000.json':
               dict(bytes=len(raw), sha256=__import__('hashlib').sha256(raw).hexdigest())}
    with zipfile.ZipFile(cache_zip, 'x') as archive:
        archive.write(cache_file, cache_row['member'])
        archive.writestr('records/cache_preparation_shard000.json', raw)
        archive.writestr('package_manifest.json', json.dumps(entries))
    cached = campaign.validate_caches(cache_zip, shard, plan)
    assert cached['records'] == [cache_row]
    cache_file.write_bytes(b'tampered')
    rejects(lambda: campaign.validate_caches(cache_zip, shard, plan))

    # Exercise all 156 ledger/transfer/cleanup steps using a fake external worker.
    campaign.ROOT = root/'orchestration'
    campaign.ROOT.mkdir()
    campaign.DRIVE = root/'orchestration-drive'
    campaign.HERE = campaign.ROOT/'operator'
    campaign.HERE.mkdir()
    campaign.SOURCE = campaign.ROOT
    science_started = time.time()-120
    (campaign.ROOT/'science_started.json').write_text(json.dumps(dict(started_epoch=science_started)))
    (campaign.HERE/'plan.json').write_text('{}')
    (campaign.HERE/'prepare_caches.py').write_text('# fake external cache worker\n')
    fake_plan = dict(plan, shards=[])
    for number in range(39):
        label = 'shard%03d' % number
        selection = campaign.HERE/(label+'.json')
        indices = list(range(number*64, min((number+1)*64, 2468)))
        selection.write_text(json.dumps(dict(schema_version=1, dataset='modelnet40_c', indices=indices)))
        fake_plan['shards'].append(dict(number=number, label=label, count=len(indices), indices=indices,
            selection_file=str(selection), arms=['control', 'half0', 'M100']))
    repo = Path(__file__).resolve().parents[4]
    (campaign.SOURCE/'research_artifacts.py').write_bytes((repo/'research_artifacts.py').read_bytes())
    calibration = campaign.ROOT/'calibration_frozen.json'
    calibration.write_bytes((repo/'result/preparation/gsd_lowband_halfmix_20261107/calibration_frozen.json').read_bytes())
    fake_plan.update(calibration_path=str(calibration), calibration_sha256=campaign.identity(calibration)['sha256'])
    sys.path.insert(0, str(repo))
    from research_artifacts import CORRUPTIONS
    fake_plan['corruptions'] = list(CORRUPTIONS)
    (campaign.HERE/'plan.json').write_text(json.dumps(fake_plan))
    assert len(campaign.load_plan()['shards']) == 39
    selected = Path(fake_plan['shards'][0]['selection_file'])
    original = selected.read_text()
    selection = json.loads(original)
    selection['indices'][0] = 1
    selected.write_text(json.dumps(selection))
    rejects(campaign.load_plan)
    selected.write_text(original)
    campaign.load_plan = lambda: fake_plan
    fake_common = good['common']
    fake_report = dict(records=[], lion_state=fake_common['lion_state_before'], native=fake_common['native_bootstrap'])
    campaign.validate_caches = lambda *args: fake_report
    campaign.validate_bundle = lambda *args: dict(common=fake_common, pairing={'same': 'prepared'}, graphs={'same': 'bases'})
    commands = []
    def fake_worker(label, command, **kwargs):
        commands.append((label, command))
        if label.endswith('_cache'):
            archive = campaign.ROOT/'cache_archives'/('cache_'+label[:-6]+'.zip')
            drive = campaign.DRIVE/'cache_archives'/archive.name
        else:
            bundle = campaign.ROOT/'raw_runs'/label/'modelnet40_c/decoded_physical_spectral_diagnostic/bundle'
            bundle.mkdir(parents=True)
            (bundle/'config.json').write_text(json.dumps(dict(status='partial', diagnostic_complete=True)))
            archive = Path(str(bundle)+'.zip')
            drive = campaign.DRIVE/'runs'/label/'modelnet40_c/decoded_physical_spectral_diagnostic/bundle/final.zip'
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b'fake immutable accepted worker ZIP')
        drive.parent.mkdir(parents=True, exist_ok=True)
        drive.write_bytes(archive.read_bytes())
        if not label.endswith('_cache'):
            (drive.parent/'final.receipt.json').write_text(json.dumps(dict(status='complete', archive=campaign.identity(drive))))
        return dict(returncode=0, timed_out=False)
    campaign.run_process = fake_worker
    real_announce = campaign.announce
    def fake_daemon(manifest_path):
        real_announce(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        path = campaign.ROOT/'acks'/(manifest['label']+'.json')
        if not path.exists():
            path.write_text(json.dumps(dict(status='local_and_drive_verified', label=manifest['label'],
                local_archive='/mac/'+manifest['label']+'.zip', bytes=manifest['archive']['bytes'], sha256=manifest['archive']['sha256'])))
    partial_seen = threading.Event()
    settled_written = threading.Event()
    read_ack = campaign.read_ack
    def observe_partial(path, manifest):
        result = read_ack(path, manifest)
        if manifest['label'] == 'shard000_cache' and result is None:
            partial_seen.set()
        return result
    campaign.read_ack = observe_partial
    def settling_daemon(manifest_path):
        manifest = json.loads(manifest_path.read_text())
        if manifest['label'] != 'shard000_cache':
            fake_daemon(manifest_path)
            return
        real_announce(manifest_path)
        path = campaign.ROOT/'acks'/(manifest['label']+'.json')
        path.write_text('{"status":')
        def finish_upload():
            assert partial_seen.wait(5), 'Reader did not defer partial ack'
            fake_daemon(manifest_path) if not path.exists() else path.write_text(json.dumps(dict(
                status='local_and_drive_verified', label=manifest['label'], local_archive='/mac/settled.zip',
                bytes=manifest['archive']['bytes'], sha256=manifest['archive']['sha256'])))
            settled_written.set()
        threading.Thread(target=finish_upload, daemon=True).start()
    campaign.announce = settling_daemon
    output = io.StringIO()
    campaign.print = lambda *args, **kwargs: builtins.print(*args, **dict(kwargs, file=output))
    campaign.main([])
    assert partial_seen.is_set() and settled_written.wait(5)
    assert len(commands) == 156 and output.getvalue().count('FULL_TRANSFER_READY ') == 156
    initial_commands = list(commands)
    journal = json.loads((campaign.ROOT/'campaign_journal.json').read_text())
    assert journal['status'] == 'complete' and journal['started_epoch'] == science_started
    campaign.announce = fake_daemon
    holder = subprocess.Popen([sys.executable, '-u', '-c',
        'import fcntl,sys,time; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX); print("LOCKED",flush=True); time.sleep(60)',
        str(campaign.ROOT/'campaign.lock')], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'LOCKED'
        try:
            campaign.main(['--resume'])
        except RuntimeError as error:
            assert 'lock' in str(error).lower()
        else:
            raise AssertionError('Concurrent controller bypassed the campaign lock')
    finally:
        holder.terminate()
        holder.wait(timeout=5)
    campaign.main(['--resume'])
    assert len(commands) == 156, 'Resume reran an accepted worker'
    assert json.loads((campaign.ROOT/'campaign_journal.json').read_text())['status'] == 'complete'
    assert output.getvalue().count('FULL_CAMPAIGN_COMPLETE') == 2

    campaign.ROOT = root/'failed-orchestration'
    campaign.ROOT.mkdir()
    campaign.DRIVE = root/'failed-orchestration-drive'
    attempts = []
    def failed_worker(label, command, **kwargs):
        attempts.append(label)
        return dict(returncode=1, timed_out=False)
    campaign.run_process = failed_worker
    rejects(lambda: campaign.main([]))
    assert json.loads((campaign.ROOT/'campaign_journal.json').read_text())['status'] == 'failed'
    rejects(lambda: campaign.main(['--resume']))
    assert len(attempts) == 1, 'Failed attempt was silently rerun'

    campaign.ROOT = root/'unacked-orchestration'
    campaign.ROOT.mkdir()
    campaign.DRIVE = root/'unacked-orchestration-drive'
    campaign.run_process = fake_worker
    campaign.announce = real_announce
    count_before = len(commands)
    rejects(lambda: campaign.main(['--ack-timeout', '.01']))
    assert len(commands)-count_before == 3, 'Pending transfer limit did not apply'
    journal = json.loads((campaign.ROOT/'campaign_journal.json').read_text())
    assert journal['status'] == 'failed' and len(journal['jobs']) == 3
    assert all(Path(json.loads((campaign.ROOT/'job_receipts'/(label+'.json')).read_text())['archive']['path']).exists()
               for label in journal['jobs']), 'Unacknowledged raw archive was pruned'
    print(output.getvalue(), end='')
    print('FULL_CAMPAIGN_CPU_COUNTS '+json.dumps(dict(status='passed', synthetic_shards=len(fake_plan['shards']),
        synthetic_cache_jobs=sum(label.endswith('_cache') for label, _ in initial_commands),
        synthetic_inference_jobs=sum(not label.endswith('_cache') for label, _ in initial_commands),
        resumed_worker_reruns=0, process_failure_modes=3, controller_lock_contentions=1,
        truncated_ack_completed=1, failed_worker_attempts=len(attempts),
        unacknowledged_retained_archives=len(journal['jobs'])), sort_keys=True))

print('FULL_CAMPAIGN_CPU_SELFCHECK_COMPLETE')
