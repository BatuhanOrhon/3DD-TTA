"""Agent B smoke/development controller; initial phase shares one 3300s ledger."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_assets_template import (ROOT, PREPARED, ASSETS, DRIVE, PYTHON, MODELNET, SESSION,
                                    MANIFEST_SHA, SOURCE_SHA, identity, verify, require, utc, seal,
                                    copy_checked, preserve_phase_sources)

CAP = 3300
LEDGER = ROOT / 'initial_gpu_phase_ledger.json'
ARMS = ('combined', 'chamfer', 'spectral', 'unguided')
HEADS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')
DATASETS = ('modelnet40_c', 'scanobjectnn_c')
METHOD = 'gsd_inspired_latent_spectral_guidance'
PAIRED = ROOT / 'paired_state_v1_seed0'


def commands(mode, output, *, seed=0, arms=ARMS, paired_state_root=None):
    require(type(seed) is int and seed >= 0, 'Seed must be a nonnegative integer')
    require(mode in ('smoke', 'development', 'heldout'), 'Unknown registered stage')
    require(arms and len(set(arms)) == len(arms) and set(arms) <= set(ARMS), 'Invalid component arms')
    require(mode != 'heldout' or paired_state_root is not None, 'Heldout requires an explicit separate seed cache')
    jobs = []
    for dataset in DATASETS:
        data = MODELNET if dataset == 'modelnet40_c' else ASSETS / 'datasets/scanobjectnn_c'
        config = 'tta_modelnet.yaml' if dataset == 'modelnet40_c' else 'tta_scanobj.yaml'
        checkpoint = 'modelnet_jt.pth' if dataset == 'modelnet40_c' else 'scanobject_jt.pth'
        for arm in arms:
            result_root = Path(output) / 'runs' / dataset / arm
            command = [PYTHON, '-u', str(ASSETS / 'source_snapshot/scripts/eval_multibackbone_guidance.py'),
                '--dataset', dataset, '--dataset-root', str(data), '--label-path', str(data / 'label.npy'),
                '--asset-root', str(ASSETS), '--asset-manifest-sha256', MANIFEST_SHA,
                '--result-root', str(result_root), '--drive-root', str(DRIVE), '--seed', str(seed),
                '--trajectory-observe', '--component-arm', arm, '--backbones', *HEADS,
                '--pointmae-config', 'source_snapshot/cfgs/' + config,
                '--pointmae-checkpoint', 'checkpoints/' + checkpoint,
                '--index-manifest', str(ASSETS / 'study' / (dataset + '_' + mode + '_ids.json')),
                '--corruptions', *(['gaussian', 'background'] if mode == 'smoke' else ['all'])]
            if mode in ('development', 'heldout'):
                default_cache = PAIRED if seed == 0 else ROOT / ('paired_state_v1_seed%d' % seed)
                command.extend(['--paired-state-root', str(default_cache if paired_state_root is None else paired_state_root)])
            jobs.append(dict(dataset=dataset, arm=arm, result_root=str(result_root), command=command))
    return jobs


def validate_ledger(ledger, *, frozen_plan_sha256=None):
    require(ledger.get('schema_version') == 1 and ledger.get('manifest_sha256') == MANIFEST_SHA
            and ledger.get('session') == SESSION and ledger.get('cap_seconds') == CAP, 'Initial GPU phase ledger identity/cap mismatch')
    spent = ledger.get('spent_seconds')
    require(isinstance(spent, (int, float)) and 0 <= spent <= CAP, 'Initial GPU phase budget exhausted/invalid')
    require(ledger.get('active') is None, 'Prior phase/process state unresolved; inspect it before retry')
    if frozen_plan_sha256 is not None:
        require(ledger.get('frozen_plan_sha256') == frozen_plan_sha256, 'Frozen confirmation ledger/plan mismatch')


def load_ledger(*, ledger_path=None, drive_name='initial_gpu_phase_ledger.json', frozen_plan_sha256=None):
    ledger_path = LEDGER if ledger_path is None else Path(ledger_path)
    require(Path(drive_name).name == drive_name and drive_name not in ('', '.', '..'), 'Unsafe ledger namespace')
    remote = DRIVE / drive_name
    if remote.exists():
        ledger = json.loads(remote.read_text())
        validate_ledger(ledger, frozen_plan_sha256=frozen_plan_sha256)
        if ledger_path.exists():
            verify(ledger_path, identity(remote))
        else:
            copy_checked(remote, ledger_path, identity(remote))
        return ledger
    require(not ledger_path.exists(), 'Local/Drive ledger disagreement; reconcile before launch')
    ledger = dict(schema_version=1, session=SESSION, manifest_sha256=MANIFEST_SHA,
                  cap_seconds=CAP, spent_seconds=0., active=None, attempts=[])
    if frozen_plan_sha256 is not None:
        ledger['frozen_plan_sha256'] = frozen_plan_sha256
    return ledger


def raise_on_terminate(signum, frame):
    raise KeyboardInterrupt('Controller terminated; clean up its owned worker group')


def save_ledger(ledger, *, ledger_path=None, drive_name='initial_gpu_phase_ledger.json'):
    ledger_path = LEDGER if ledger_path is None else Path(ledger_path)
    require(Path(drive_name).name == drive_name and drive_name not in ('', '.', '..'), 'Unsafe ledger namespace')
    raw = json.dumps(ledger, indent=2, sort_keys=True) + '\n'
    for path in (ledger_path, DRIVE / drive_name):
        pending = path.with_suffix('.pending.json')
        pending.write_text(raw)
        pending.replace(path)
    verify(DRIVE / drive_name, identity(ledger_path))


def stream_job(command, log_path, timeout, on_start):
    """Bound the owned subprocess group and retain the primary output stream."""
    child = subprocess.Popen(command, cwd=str(ASSETS / 'source_snapshot'),
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1, start_new_session=True)
    watchdog = None
    watchdog_errors = []
    try:
        deadline, done = time.monotonic() + timeout, False
        expired = threading.Event()
        def enforce_deadline():
            expired.set()
            try:
                child.poll()
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException as error:
                watchdog_errors.append(repr(error))
        watchdog = threading.Timer(timeout, enforce_deadline)
        watchdog.daemon = True
        watchdog.start()
        on_start(child.pid)
        lines = queue.Queue()
        def read():
            try:
                for line in child.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        threading.Thread(target=read, daemon=True).start()
        with Path(log_path).open('x') as log:
            while not done:
                if watchdog_errors:
                    raise RuntimeError('GPU watchdog failed: ' + '; '.join(watchdog_errors))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Shared initial GPU phase cap reached')
                try:
                    line = lines.get(timeout=min(60., remaining))
                except queue.Empty:
                    print('AGENT_B_PHASE_PROGRESS', utc(), 'owned worker PID', child.pid, 'running; no new output', flush=True)
                    continue
                if line is None:
                    done = True
                else:
                    print(line, end='', flush=True); log.write(line); log.flush()
            returncode = child.wait(timeout=max(.1, deadline - time.monotonic()))
            if watchdog_errors:
                raise RuntimeError('GPU watchdog failed: ' + '; '.join(watchdog_errors))
            if expired.is_set():
                raise TimeoutError('Shared initial GPU phase cap reached')
            require(returncode == 0, 'Component worker failed; preserve this attempt')
    except BaseException:
        child.poll()  # Reap exited leaders before signaling their remaining process group.
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
        # A terminated controller may have a still-live worker in its owned group.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise
    finally:
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join(timeout=1)
        if child.stdout is not None:
            child.stdout.close()


def persist_cache(paired_state_root=None):
    cache = PAIRED if paired_state_root is None else Path(paired_state_root)
    drive_name = 'paired_state_v1_seed0' if paired_state_root is None else cache.name
    records = []
    if cache.exists():
        for path in sorted(cache.rglob('*.npz')):
            expected = identity(path)
            target = DRIVE / drive_name / path.relative_to(cache)
            if target.exists():
                verify(target, expected)
            else:
                copy_checked(path, target, expected)
            records.append(dict(local_path=str(path), drive_path=str(target), **expected))
    return records


def validate_job(job, mode, manifest, *, seed=0, paired_state_root=None):
    require(type(seed) is int and seed >= 0, 'Seed must be a nonnegative integer')
    require(mode != 'heldout' or paired_state_root is not None, 'Heldout validation requires its explicit seed cache')
    import numpy as np
    from scripts.analyze_scan_trajectory import analyze_bundle
    from research_artifacts import CORRUPTIONS
    expected_corruptions = ['gaussian', 'background'] if mode == 'smoke' else list(CORRUPTIONS)
    configs = list((Path(job['result_root']) / job['dataset'] / METHOD).glob('*/config.json'))
    require(len(configs) == 4, 'Require exactly four final head bundles per arm/dataset job')
    reports, raw_refs, traces, timings = [], [], {}, {}
    found_heads = set()
    ids_path = ASSETS / 'study' / (job['dataset'] + '_' + mode + '_ids.json')
    ids = json.loads(ids_path.read_text())['indices']
    require(len(ids) == 70, 'Initial screen requires exactly70 declared IDs')
    data_root = MODELNET if job['dataset'] == 'modelnet40_c' else ASSETS / 'datasets/scanobjectnn_c'
    label_file = data_root / 'label.npy'
    verify(label_file, manifest['datasets'][job['dataset']]['files']['label.npy'])
    labels = np.load(label_file, allow_pickle=False).reshape(-1)[ids]
    for config_path in configs:
        config = json.loads(config_path.read_text())
        head = config['backbone']; found_heads.add(head)
        require(config['dataset'] == job['dataset'] and config['component_arm'] == job['arm']
                and config.get('component_study') is True and config.get('status') == 'partial'
                and config['asset_manifest_sha256'] == MANIFEST_SHA and config['selected_original_ids'] == ids
                and type(config['seed']) is int and config['seed'] == seed and config['severity'] == 5
                and config['selection_manifest']['stage'] == mode
                and set(config['corruptions']) == set(expected_corruptions), 'Arm/dataset/selection/source scope mismatch')
        require('guidance-components-' + job['arm'] in config['run_id'], 'Component run ownership missing')
        protocol = config['guidance_protocol']
        expected_weights = dict(chamfer=1. if job['arm'] in ('combined', 'chamfer') else 0.,
                                spectral=1.17 if job['arm'] in ('combined', 'spectral') else 0., invariant=0.)
        require(protocol == dict(M_max=800, beta=2., gamma=.01, eta=.01, p=.95, total=100,
                loss_weights=expected_weights, use_static_style=False, dynamic_graph=False,
                graph_update_interval=3, skip_inactive_components=True), 'Locked component protocol changed')
        require(protocol['skip_inactive_components'] and not protocol['dynamic_graph'] and protocol['loss_weights']['invariant'] == 0,
                'Component switches changed from fixed screen')
        for relative in ('tta_gsd_v2.py', 'shared_trajectory.py', 'scripts/eval_multibackbone_guidance.py', 'scripts/eval_multibackbone_source.py'):
            require(config['executed_source_files'].get('assets/source_snapshot/' + relative) == manifest['files']['source_snapshot/' + relative], 'Executed source mismatch: ' + relative)
        archive = Path(str(config_path.parent) + '.zip')
        report = analyze_bundle(archive, allow_partial=True)
        require(report['status'] == 'partial' and report['selection_complete']
                and len(report['per_corruption']) == len(expected_corruptions) * 3
                and all(row['n'] == 70 for row in report['per_corruption']), 'Incomplete/invalid declared subset')
        drive_run = DRIVE / config['dataset'] / config['method'] / config['run_id']
        final_receipt = json.loads((drive_run / 'final.receipt.json').read_text())
        verify(archive, final_receipt['archive']); verify(drive_run / 'final.zip', final_receipt['archive'])
        require(final_receipt['status'] == 'complete' and final_receipt['result_status'] == 'partial', 'Final Drive receipt missing')
        raw_refs.append(dict(backbone=head, local_path=str(archive), drive_path=str(drive_run / 'final.zip'),
                             receipt_path=str(drive_run / 'final.receipt.json'), **identity(archive)))
        for corruption in expected_corruptions:
            directory = config_path.parent / 'batches' / corruption / 'batch0000'
            with np.load(directory / 'predictions.npz', allow_pickle=False) as data:
                require(np.array_equal(data['index'], ids) and np.array_equal(data['label'], labels), 'Indexed labels/IDs mismatch')
            diagnostic = json.loads((directory / 'diagnostics.json').read_text())
            weights = protocol['loss_weights']; metrics = diagnostic['guidance_metrics']
            horizon = 30 if corruption == 'background' else 10
            require(metrics['graph_calls'] == int(weights['spectral'] > 0)
                    and metrics['chamfer_calls'] == 2 * horizon * int(weights['chamfer'] > 0)
                    and metrics['guidance_backward_calls'] == horizon * int(weights['spectral'] > 0 or weights['chamfer'] > 0), 'Inactive computation was not skipped')
            require(all(np.isfinite(value) and value >= 0 for value in metrics.values()), 'Nonfinite/negative component metrics')
            require((weights['spectral'] > 0 or metrics['mean_raw_spectral'] == 0)
                    and (weights['chamfer'] > 0 or metrics['mean_raw_chamfer'] == 0), 'Inactive loss is nonzero')
            if job['arm'] == 'unguided':
                require(metrics['mean_local_update_norm'] == 0 and metrics['mean_style_update_norm'] == 0, 'Unguided update was nonzero')
            if mode == 'smoke':
                require(diagnostic['preparation_mode'] == 'fresh_encode' and 'paired_state' not in diagnostic, 'Smoke timing used cached encode')
            else:
                cached = diagnostic['paired_state']
                verify(cached['path'], cached)
                require(diagnostic['preparation_mode'] in ('fresh_encode', 'cached_replay'), 'Missing paired preparation mode')
            trace = dict(initial_hashes=diagnostic['trajectory']['initial_hashes'], geometry_sha256=diagnostic['geometry_sha256'],
                         paired_state=diagnostic.get('paired_state'))
            if corruption in traces:
                require(traces[corruption] == trace, 'Frozen heads observed different shared trajectories')
            else:
                traces[corruption] = trace
                timings[corruption] = dict(shared_reconstruction_seconds=diagnostic['shared_reconstruction_seconds'],
                    whole_process_peak_gpu_memory_mb=diagnostic['whole_process_peak_gpu_memory_mb'],
                    preparation_mode=diagnostic['preparation_mode'], cache_io_included=diagnostic['paired_cache_io_in_reconstruction_time'])
            receipt_path = drive_run / 'batches' / corruption / 'batch0000' / 'receipt.json'
            receipt = json.loads(receipt_path.read_text())
            require(receipt['status'] == 'complete' and set(receipt['files']) == {'predictions.npz', 'diagnostics.json'}, 'Batch Drive receipt invalid')
            for name, expected in receipt['files'].items():
                verify(directory / name, expected); verify(receipt_path.parent / name, expected)
            corruption_receipt = json.loads((drive_run / ('corruption-' + corruption + '.receipt.json')).read_text())
            verify(corruption_receipt['archive']['path'], corruption_receipt['archive'])
            require(corruption_receipt['status'] == 'complete', 'Corruption Drive receipt invalid')
        reports.append(report)
    require(found_heads == set(HEADS), 'Missing/duplicate frozen heads')
    return dict(dataset=job['dataset'], arm=job['arm'], reports=reports, raw_archives=raw_refs,
                traces=traces, timings=timings, cache=persist_cache(paired_state_root) if mode in ('development', 'heldout') else [],
                timing_scope='fresh-encode inclusive whole reconstruction and job wall' if mode == 'smoke' else 'cached-state comparison cost; no deployment speedup claim')


def validate_pairing(jobs, *, arms=ARMS):
    for dataset in DATASETS:
        group = [job for job in jobs if job['dataset'] == dataset]
        require(len(group) == len(arms) and {job['arm'] for job in group} == set(arms), 'Incomplete component arms')
        reference = next(job for job in group if job['arm'] == 'combined')
        for job in group:
            require(set(job['traces']) == set(reference['traces']), 'Different paired corruption scope')
            for corruption, expected in reference['traces'].items():
                actual = job['traces'][corruption]
                require(actual['initial_hashes'] == expected['initial_hashes'] and actual['paired_state'] == expected['paired_state'],
                        'Unmatched prepared cloud/z0/h0/noise/RNG cache across component arms')


def confirm_preflight():
    require((PREPARED / '.complete').read_text().strip() == MANIFEST_SHA, 'Accepted CPU proof required')
    receipt = json.loads((PREPARED / 'accepted_cpu_preflight_receipt.json').read_text())
    require(receipt['state']['status'] == 'complete' and receipt['state']['strict_loads'] == 8, 'CPU proof is incomplete')
    verify(receipt['archive']['local_path'], receipt['archive']); verify(receipt['archive']['drive_path'], receipt['archive'])
    report = json.loads((PREPARED / 'cpu_preflight.json').read_text())
    verify(PREPARED / 'cpu_preflight.json', receipt['report_identity'])
    require(report['status'] == 'complete' and report['manifest_sha256'] == MANIFEST_SHA
            and not report['cuda_initialized'] and len(report['models']) == 8
            and report['native']['no_build'] and len(report['native']['bindings']) == 4, 'CPU proof identity/CUDA mismatch')


def main(mode):
    confirm_preflight()
    ledger = load_ledger()
    validate_ledger(ledger)
    if mode == 'development':
        acceptance = json.loads((ROOT / 'accepted_smoke.json').read_text())
        require(acceptance.get('status') == 'accepted' and acceptance.get('manifest_sha256') == MANIFEST_SHA, 'Independent smoke acceptance required')
        validation = ROOT / 'smoke_validation.json'
        verify(validation, acceptance['validation'])
        smoke = json.loads(validation.read_text())
        require(smoke['status'] == 'validated' and smoke['mode'] == 'smoke' and smoke['manifest_sha256'] == MANIFEST_SHA,
                'Smoke validation identity/status mismatch')
        verify(smoke['evidence']['drive_path'], smoke['evidence'])
        projected = ledger['spent_seconds'] + smoke['projected_development_seconds']
        require(projected <= 3600, 'Projected whole GPU phase exceeds60min; obtain comprehensive approval before continuing')
        require(projected <= CAP, 'Projected phase cannot finish within the initial3300s cap; review scope/budget')
    sys.path.insert(0, str(ASSETS / 'source_snapshot'))
    from scripts import eval_multibackbone_source as source
    require(source.REPO == ASSETS / 'source_snapshot', 'Unsealed source imported')
    manifest = source.read_manifest(ASSETS, MANIFEST_SHA)
    stamp = mode + '_' + utc()
    output = ROOT / stamp; output.mkdir()
    drive_output = DRIVE / 'phase-evidence' / stamp; drive_output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', session=SESSION, mode=mode, manifest_sha256=MANIFEST_SHA, source_snapshot_sha256=SOURCE_SHA,
                 jobs=[], single_seed_exploratory=True, full_benchmark=False)
    started, previously_spent = time.monotonic(), ledger['spent_seconds']
    ledger['active'] = dict(mode=mode, output=str(output), phase_pid=os.getpid())
    ledger['attempts'].append(dict(mode=mode, output=str(output), status='running', utc=utc()))
    save_ledger(ledger)
    try:
        preserve_phase_sources(output)
        plan = commands(mode, output)
        (output / 'commands.json').write_text(json.dumps(plan, indent=2) + '\n')
        for number, job in enumerate(plan):
            remaining = CAP - previously_spent - (time.monotonic() - started) - 60
            require(remaining > 0, 'Shared GPU phase cap exhausted before next job')
            record = dict(job, status='running', utc=utc())
            state['jobs'].append(record)
            def on_start(pid):
                ledger['active'].update(job=number, worker_group_pid=pid, command=job['command'])
                save_ledger(ledger)
            before = time.monotonic()
            stream_job(job['command'], output / ('job_%02d.log' % number), remaining, on_start)
            result = validate_job(job, mode, manifest)
            record.update(result, status='complete', elapsed_seconds=time.monotonic() - before)
            (output / ('job_%02d_validation.json' % number)).write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
            ledger['spent_seconds'] = previously_spent + time.monotonic() - started
            ledger['active'].pop('worker_group_pid', None)
            save_ledger(ledger)
        if mode == 'development':
            validate_pairing(state['jobs'])
        rows = [row for job in state['jobs'] for report in job['reports'] for row in report['macro']]
        require(len(rows) == 96, 'Require96 declared arm/dataset/head/readout rows')
        balanced = []
        for arm in ARMS:
            for policy in ('terminal', 'minimum_entropy', 'probability_mean'):
                cases = [row for row in rows if row['component_arm'] == arm and row['policy'] == policy]
                require(len(cases) == 8, 'Balanced average requires eight dataset/backbone cases')
                balanced.append(dict(arm=arm, policy=policy, n_cases=8, accuracy_pct=sum(row['accuracy_pct'] for row in cases) / 8))
        projection = None
        if mode == 'smoke':
            projection = 0.
            for job in state['jobs']:
                normal = job['timings']['gaussian']['shared_reconstruction_seconds']
                background = job['timings']['background']['shared_reconstruction_seconds']
                # Conservatively scale remaining startup/classifier/archive/verification time by corruption count.
                other = max(0., job['elapsed_seconds'] - normal - background)
                projection += 1.2 * (14 * normal + background + 7.5 * other)
        state.update(status='complete', macro=rows, balanced=balanced, projected_development_seconds=projection)
        for name, data in [('macro', rows), ('balanced', balanced)]:
            with (output / (name + '.csv')).open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    except BaseException:
        state.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        ledger['spent_seconds'] = previously_spent + state['elapsed_seconds']
        ledger['active'] = None
        ledger['attempts'][-1].update(status=state['status'], elapsed_seconds=state['elapsed_seconds'])
        save_ledger(ledger)
        (output / 'state.json').write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
        (output / 'ledger_snapshot.json').write_text(json.dumps(ledger, indent=2) + '\n')
        receipt = dict(state=state, archive=seal(output, drive_output / (mode + '_phase_evidence.zip')))
        ledger['spent_seconds'] = previously_spent + time.monotonic() - started
        save_ledger(ledger)
        (ROOT / (mode + '_phase_receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        (drive_output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('AGENT_B_COMPONENT_PHASE_RECEIPT', json.dumps(receipt['archive']), flush=True)
    require(ledger['spent_seconds'] <= CAP, 'Initial phase cap exceeded during final evidence persistence')
    result = dict(status='validated', mode=mode, manifest_sha256=MANIFEST_SHA, evidence=receipt['archive'],
                  projected_development_seconds=state['projected_development_seconds'], spent_seconds=ledger['spent_seconds'],
                  output=str(output), independent_smoke_acceptance_required=mode == 'smoke')
    (ROOT / (mode + '_validation.json')).write_text(json.dumps(result, indent=2) + '\n')
    print('AGENT_B_COMPONENT_' + mode.upper() + '_VALIDATED', flush=True)


def self_check():
    global ASSETS, LEDGER, DRIVE
    smoke, development = commands('smoke', Path('/tmp/test')), commands('development', Path('/tmp/test'))
    require(len(smoke) == len(development) == 8, 'Expected eight jobs per mode')
    require(all('--paired-state-root' not in job['command'] for job in smoke), 'Smoke must include fresh encode')
    require(all('--paired-state-root' in job['command'] and job['command'][-1] == str(PAIRED) for job in development), 'Development must share exact states')
    valid = dict(schema_version=1, session=SESSION, manifest_sha256=MANIFEST_SHA, cap_seconds=CAP, spent_seconds=0., active=None)
    validate_ledger(valid)
    for changed in [dict(valid, spent_seconds=CAP + 1), dict(valid, active={'unresolved': True}), dict(valid, cap_seconds=3600)]:
        try:
            validate_ledger(changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid/unsafe ledger accepted')
    jobs = [dict(dataset=dataset, arm=arm, traces={'gaussian': dict(initial_hashes={'noise': 'same'}, paired_state={'sha256': 'same'})})
            for dataset in DATASETS for arm in ARMS]
    validate_pairing(jobs)
    jobs[-1]['traces']['gaussian']['initial_hashes'] = {'noise': 'different'}
    try:
        validate_pairing(jobs)
    except ValueError:
        pass
    else:
        raise AssertionError('Unmatched noise accepted')
    original_assets, original_ledger, original_drive = ASSETS, LEDGER, DRIVE
    with tempfile.TemporaryDirectory() as temporary:
        ASSETS = Path(temporary).resolve()
        (ASSETS / 'source_snapshot').mkdir()
        stream_job([sys.executable, '-c', 'print("SYNTHETIC_CHILD_OUTPUT")'], ASSETS / 'success.log', 5, lambda pid: None)
        require((ASSETS / 'success.log').read_text() == 'SYNTHETIC_CHILD_OUTPUT\n', 'Child output lost')
        timed = []
        def delayed_callback(pid):
            timed.append(pid)
            time.sleep(.2)
        try:
            stream_job([sys.executable, '-c', 'import time; time.sleep(10)'], ASSETS / 'timeout.log', .05, delayed_callback)
        except TimeoutError:
            pass
        else:
            raise AssertionError('Owned worker deadline was not enforced')
        try:
            os.killpg(timed[0], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError('Watchdog did not stop the worker during the blocked callback')
        started = []
        def fail_callback(pid):
            started.append(pid)
            raise RuntimeError('synthetic ledger write failed')
        try:
            stream_job([sys.executable, '-c', 'import time; time.sleep(10)'], ASSETS / 'callback.log', 5, fail_callback)
        except RuntimeError as error:
            require(str(error) == 'synthetic ledger write failed', 'Unexpected callback failure')
        else:
            raise AssertionError('Callback failure was swallowed')
        require(len(started) == 1, 'Synthetic worker did not start')
        try:
            os.killpg(started[0], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError('Failed start callback orphaned the owned child group')
        DRIVE, LEDGER = ASSETS / 'drive', ASSETS / 'local/ledger.json'
        DRIVE.mkdir(); LEDGER.parent.mkdir()
        retained = dict(valid, spent_seconds=123., attempts=[{'status': 'failed'}])
        (DRIVE / 'initial_gpu_phase_ledger.json').write_text(json.dumps(retained))
        require(load_ledger() == retained and json.loads(LEDGER.read_text()) == retained, 'Drive retry budget was reset')
        LEDGER.write_text(json.dumps(dict(retained, spent_seconds=0.)))
        try:
            load_ledger()
        except ValueError:
            pass
        else:
            raise AssertionError('Conflicting local/Drive ledgers accepted')
        controller_pid = ASSETS / 'terminated_worker.pid'
        program = ("import sys,signal; sys.path.insert(0," + repr(str(Path(__file__).resolve().parent)) + "); "
                   "import gpu_components_template as m; from pathlib import Path; "
                   "m.ASSETS=Path(" + repr(str(ASSETS)) + "); signal.signal(signal.SIGTERM,m.raise_on_terminate); "
                   "m.stream_job([sys.executable,'-c','import time; time.sleep(10)'],m.ASSETS/'signal.log',5,"
                   "lambda pid: Path(" + repr(str(controller_pid)) + ").write_text(str(pid)))")
        outer = subprocess.Popen([sys.executable, '-c', program], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not controller_pid.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            require(controller_pid.exists(), 'Synthetic controller did not launch its worker')
            worker_pid = int(controller_pid.read_text())
            outer.terminate(); outer.communicate(timeout=5)
            try:
                os.killpg(worker_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError('SIGTERM orphaned the owned worker group')
        finally:
            if outer.poll() is None:
                outer.kill(); outer.communicate(timeout=5)
    ASSETS, LEDGER, DRIVE = original_assets, original_ledger, original_drive
    print('AGENT_B_GPU_TEMPLATE_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, raise_on_terminate)
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('mode', choices=('smoke', 'development'))
        main(parser.parse_args().mode)
