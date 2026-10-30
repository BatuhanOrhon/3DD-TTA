"""Reviewed recovery of only unrun registered slots21–23; slot20 stays failed."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import traceback
import zipfile

ROOT = Path('/content/studies/guidance-components')
PLAN_SHA = '1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
CONTROLLER_SHA = '0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc'
RECEIPT_ID = dict(bytes=3676445, sha256='5ab88805d14edbc9713275816aa11bef1d5e80784ea385b477199f7d6b779754')
LEDGER_ID = dict(bytes=12181, sha256='60c8ba55d787c8e1e9ea4fa34f27eeb3e9aa8522300252c6c74863e1cad2736b')
ARCHIVE_ID = dict(bytes=1159246521, sha256='f586870206961f88d38daa7ed9da632fddfa51356075e114c5f3b98f020ebe5e')
FREEZE_ID = dict(bytes=798, sha256='ea245f8a75ed729463e34757328875f6d7c789983ef10bac3c0c133b0750c360')
DEVELOPMENT_ID = dict(bytes=1053, sha256='f94afc837e26a3ec33c8503f04cc9cfb7dafba44fd40cc19cff845eb7bad54ed')
SPENT_FLOOR = 14114.660198396998
POINT_SECONDS = (291.4872319210008, 300., 300.)
NUMERIC_SIGNATURE = 'ValueError: Classifier input contains nonfinite XYZ coordinates'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def without_result_root(command):
    require(command.count('--result-root') == 1, 'Exactly one result-root is required')
    result = list(command); result[result.index('--result-root') + 1] = '<result-root>'
    return result


def select_remaining(registered, archived, failed, output):
    require(len(registered) == 24 and archived == registered, 'Archived24 commands changed from frozen construction')
    require(failed['status'] == 'failed' and len(failed['jobs']) == 21
            and all(job['status'] == 'complete' for job in failed['jobs'][:20])
            and failed['jobs'][20]['status'] == 'running', 'Expected20 complete and immutable partial slot20')
    for record, expected in zip(failed['jobs'], registered):
        require(all(record[key] == value for key, value in expected.items()), 'Original failed job grid changed')
    result = []
    for index in (21, 22, 23):
        job = copy.deepcopy(registered[index]); old_root = job['result_root']
        root = Path(output) / ('seed%d' % job['seed']) / job['condition_id'] / job['dataset']
        job['result_root'] = str(root)
        position = job['command'].index('--result-root') + 1
        require(job['command'][position] == old_root, 'Original result-root command differs')
        job['command'][position] = str(root)
        require(without_result_root(job['command']) == without_result_root(registered[index]['command']), 'Scientific command changed')
        result.append(dict(job, registered_job_index=index))
    return result


def projected_budget(spent, remaining):
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (spent, remaining)), 'Invalid budget inputs')
    upper = spent + 1.2 * remaining + 1200
    require(upper <= 19800, 'Remaining frozen slots do not fit cumulative approved cap')
    return upper


def validate_ledger_history(ledger, old_output):
    attempts = [row for row in ledger['attempts'] if row['stage'] == 'confirmation']
    require(ledger['active'] is None and ledger['cap_seconds'] == 19800
            and SPENT_FLOOR <= ledger['spent_seconds'] <= 19800
            and len(attempts) == 1 and attempts[0]['status'] == 'failed' and attempts[0]['output'] == old_output
            and not any(row['stage'] == 'confirmation_remaining' for row in ledger['attempts']), 'History changed or remaining slots already attempted')


def known_numerical_failure(log, error):
    errors = re.findall(r'^[A-Za-z_][\w.]*(?:Error|Exception):[^\r\n]*$', log, re.MULTILINE)
    return (type(error) is ValueError and str(error) == 'Component worker failed; preserve this attempt'
            and log.count('Traceback (most recent call last):') == 1 and errors == [NUMERIC_SIGNATURE]
            and 'classifier_backbones.py", line 48, in forward' in log
            and 'raise ValueError("Classifier input contains nonfinite XYZ coordinates")' in log)


def validate_global_contracts(result, dataset, ledger):
    require(result['lion_state'] == ledger['lion_state'], 'LION differs from preserved cumulative contract')
    for head, contract in result['classifier_contracts'].items():
        require(ledger['classifier_contracts'][dataset + '/' + head] == contract, 'Classifier differs from preserved cumulative contract')


def verify_preservation(proof, failed, recovery, c):
    require(proof['status'] == 'verified_failure_preserved' and proof['session'] == c.SESSION
            and proof['phase_plan_sha256'] == PLAN_SHA and proof['manifest_sha256'] == recovery['manifest_sha256']
            and proof['source_snapshot_sha256'] == recovery['source_snapshot_sha256']
            and proof['original_confirmation_archive'] == failed['archive']
            and proof['original_receipt_identity'] == RECEIPT_ID and proof['failed_ledger_identity'] == LEDGER_ID
            and proof['failed_job_index'] == 20 and proof['numeric_failure_signature'] == NUMERIC_SIGNATURE
            and proof['physical_cache_count'] == 180 and proof['physical_cache_complete'] is True
            and proof['local_and_drive_original_archive_verified'] is True and proof['mismatch_detected'] is False
            and proof['integrity_verification']['status'] == 'verified_preserved_scope', 'Failure preservation/integrity gate incomplete')
    for key, root in (('local_path', ROOT), ('drive_path', c.DRIVE)):
        c.verify(c.owned(proof['archive'][key], root), proof['archive'])
    log = c.owned(proof['source_log_identity']['local_path']); c.verify(log, proof['source_log_identity'])
    require(known_numerical_failure(log.read_text(), ValueError('Component worker failed; preserve this attempt')), 'Original error is not exact known finite-XYZ failure')


def bootstrap_native_cpu(c, manifest, audit, torch):
    require(not torch.cuda.is_initialized(), 'CUDA initialized before native CPU bootstrap')
    relative = manifest['guidance']['native_bootstrap']; path = c.ASSETS / relative
    c.verify(path, manifest['files'][relative])
    spec = importlib.util.spec_from_file_location('remaining_native_cpu', path)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    report = native.bootstrap(c.ASSETS)
    bindings = {}
    for name, entries in audit['backends'].items():
        entry, = entries
        bindings[name] = {key: entry[key] for key in ('path', 'bytes', 'sha256')}
    require(report['no_build'] is True and report['bindings'] == bindings and not torch.cuda.is_initialized(),
            'Native bootstrap allowed a build, changed bindings or initialized CUDA')
    return report


def fresh_cpu_preflight(c, manifest, selected_jobs, ledger):
    from unittest.mock import patch
    c.verify(c.ASSETS / 'source_snapshot.zip', dict(bytes=(c.ASSETS / 'source_snapshot.zip').stat().st_size, sha256=manifest['source_snapshot_sha256']))
    for relative, expected in manifest['files'].items():
        c.verify(c.owned(c.ASSETS / relative), expected)
    for key in ('lion_checkpoint', 'lion_config'):
        row = manifest['guidance'][key]; c.verify(row['path'], row)
    for dataset, declaration in manifest['datasets'].items():
        root = Path('/content/3DD-TTA/data/modelnet40_c') if dataset == 'modelnet40_c' else c.ASSETS / 'datasets' / dataset
        for name, expected in declaration['files'].items():
            c.verify(root / name, expected)
    audit = json.loads((c.ASSETS / manifest['guidance']['native_audit']).read_text())
    for entries in audit['backends'].values():
        for row in entries:
            c.verify(row['path'], row)
    sys.path.insert(0, str(c.ASSETS / 'source_snapshot'))
    from scripts import eval_multibackbone_source as source
    from scripts import eval_multibackbone_guidance as guidance
    import torch
    require(source.REPO == c.ASSETS / 'source_snapshot' and not torch.cuda.is_initialized(), 'Unexpected project import/CUDA before CPU preflight')
    native = bootstrap_native_cpu(c, manifest, audit, torch)
    old_cwd = Path.cwd(); rows = []
    try:
        os.chdir(str(source.REPO))
        for job in selected_jobs:
            for config in guidance.configurations(job['command'][3:]):
                with patch.object(torch.nn.Module, 'cuda', lambda model, *args, **kwargs: model), \
                     patch.object(torch.Tensor, 'cuda', side_effect=RuntimeError('CPU preflight forbids GPU tensors')), \
                     patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('CPU preflight forbids CUDA initialization')), \
                     patch.object(torch.nn.Module, '_call_impl', side_effect=RuntimeError('CPU preflight forbids forward')):
                    model = source.load_classifier(config, manifest)
                load = config['checkpoint_load']
                require(load['strict'] is True and not load['missing_keys'] and not load['unexpected_keys'], 'Fresh strict CPU checkpoint load failed')
                contract = dict(state_sha256=source.state_hash(model), checkpoint_sha256=load['checkpoint_sha256'],
                                preprocessing=config['source_classifier_preprocessing'], resolved_model_config=config['resolved_model_config'])
                require(contract == ledger['classifier_contracts'][job['dataset'] + '/' + config['backbone']]
                        and not torch.cuda.is_initialized() and all(p.device.type == 'cpu' for p in model.parameters()), 'Fresh CPU classifier state changed')
                rows.append(dict(dataset=job['dataset'], backbone=config['backbone'], method=job['trial']['method'], contract=contract))
                del model
    finally:
        os.chdir(str(old_cwd))
    require(len(rows) == 12, 'Expected12 fresh CPU strict boundary loads')
    return dict(status='passed_CPU_no_forward_no_CUDA', strict_loads=rows, native=native, native_binary_files_verified=True,
                asset_dataset_checkpoint_files_verified=True, lion_checkpoint_config_bytes_verified=True,
                limits='LION state is pinned by checkpoint/config/source bytes before GPU and checked against preserved global hashes after every worker.')


def failed_worker_metadata(job, plan, manifest, selection, calibration, c, evidence):
    from scripts.analyze_scan_trajectory import Bundle
    method = '3dd_original' if job['trial']['method'] == 'original3dd' else c.GSD_METHOD
    paths = list((Path(job['result_root']) / job['dataset'] / method).glob('*/config.json'))
    require(len(paths) == 4, 'Known numerical worker lacks four preserved head configs')
    raw, contracts, lion, heads = [], {}, None, set()
    for path in paths:
        archive_path = Path(str(path.parent) + '.zip')
        with zipfile.ZipFile(archive_path) as archive:
            evidence.checked_members(archive)
        bundle = Bundle(archive_path)
        try:
            config = bundle.json('config.json'); head = config['backbone']
            require(head in c.HEADS and head not in heads and config['status'] == 'failed'
                    and known_numerical_failure(config['failure'], ValueError('Component worker failed; preserve this attempt'))
                    and not any(key.endswith('_audit_error') for key in config), 'Unrecognized or unaudited numerical failure')
            heads.add(head)
            # Reuse immutable protocol/state/native guards without accepting predictions.
            guarded = dict(config, status='partial'); c.verify_common_config(guarded, job, plan, manifest, selection, calibration)
            require(config['initialization_rng_guard']['rng_unchanged'] is True, 'Failed worker initialization RNG mismatch')
            contracts[head] = dict(state_sha256=config['classifier_state_before_sha256'], checkpoint_sha256=config['checkpoint_load']['checkpoint_sha256'],
                                   preprocessing=config['source_classifier_preprocessing'], resolved_model_config=config['resolved_model_config'])
            require(lion is None or lion == config['lion_state_before'], 'Failed heads have different LION states'); lion = config['lion_state_before']
            for name in bundle.names:
                if name.startswith('batches/') and name.endswith('/diagnostics.json'):
                    diagnostic = bundle.json(name)
                    require(diagnostic['rng_unchanged'] is True and diagnostic['seed'] == job['seed'], 'Failed-prefix RNG/seed guard mismatch')
            drive = c.DRIVE / job['dataset'] / method / config['run_id']
            final = json.loads((drive / 'final.receipt.json').read_text())
            require(final['status'] == 'complete' and final['result_status'] == 'failed', 'Numerical partial Drive receipt missing')
            c.verify(archive_path, final['archive']); c.verify(drive / 'final.zip', final['archive'])
            raw.append(dict(backbone=head, run_id=config['run_id'], local_path=str(archive_path), drive_path=str(drive / 'final.zip'), **identity(archive_path)))
        finally:
            bundle.close()
    require(heads == set(c.HEADS), 'Numerical partial head coverage differs')
    return dict(partial_raw_archives=raw, classifier_contracts=contracts, lion_state=lion,
                numerical_failure=NUMERIC_SIGNATURE, accuracy_accepted=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--recovery-plan-sha256', required=True); args = parser.parse_args()
    here = Path(__file__).resolve().parent; recovery_path = here / 'reviewed_recovery_plan.json'
    require(identity(recovery_path)['sha256'] == args.recovery_plan_sha256, 'Reviewed recovery plan changed')
    recovery = json.loads(recovery_path.read_text())
    require(recovery['phase_plan_sha256'] == PLAN_SHA and recovery['registered_jobs'] == [21, 22, 23]
            and recovery['excluded_failed_slots'] == [20] and recovery['original_receipt_identity'] == RECEIPT_ID
            and recovery['failed_ledger_identity'] == LEDGER_ID and recovery['original_archive_identity'] == ARCHIVE_ID
            and recovery['point_seconds'] == list(POINT_SECONDS), 'Unreviewed recovery scope or provenance')
    for name, expected in recovery['sources'].items():
        require(Path(name).name == name and identity(here / name) == expected, 'Reviewed recovery source changed')
    require(set(recovery['sources']) == {'remaining_slots_controller.py', 'run_remaining_slots.py'}, 'Recovery source coverage differs')
    controller_path = ROOT / 'sensitivity_phases/sensitivity_campaign_controller.py'
    require(identity(controller_path)['sha256'] == CONTROLLER_SHA, 'Pinned original controller changed')
    spec = importlib.util.spec_from_file_location('pinned_remaining_control', controller_path)
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    c.owned(here); plan, registry, selections = c.plan_verified(ROOT / 'sensitivity_campaign_plan.json', PLAN_SHA)
    require(recovery['session'] == c.SESSION and recovery['manifest_sha256'] == plan['manifest_sha256']
            and recovery['source_snapshot_sha256'] == plan['source_snapshot_sha256'], 'Recovery source/session differs from scientific plan')
    c.approval_verified(PLAN_SHA); c.cpu_preflight_verified(plan)
    gpu, evidence = c.load_runtime_tools(); signal.signal(signal.SIGTERM, gpu.raise_on_terminate)
    original_receipt_path = c.owned(ROOT / 'sensitivity_confirmation_receipt.json'); c.verify(original_receipt_path, RECEIPT_ID)
    failed = json.loads(original_receipt_path.read_text()); old_output = failed['archive']['local_path'][:-4]
    require({key: failed['archive'][key] for key in ARCHIVE_ID} == ARCHIVE_ID, 'Original failed archive changed')
    require(recovery['original_confirmation_archive'] == failed['archive'], 'Reviewed failed archive paths differ')
    ledger = c.load_ledger(PLAN_SHA); c.verify(ROOT / c.LEDGER_NAME, LEDGER_ID); validate_ledger_history(ledger, old_output)
    preservation_path = c.owned(ROOT / 'sensitivity_confirmation_failure_preservation_receipt.json')
    c.verify(preservation_path, recovery['failure_preservation_receipt']); proof = json.loads(preservation_path.read_text())
    verify_preservation(proof, failed, recovery, c)
    for key, root in (('local_path', ROOT), ('drive_path', c.DRIVE)):
        c.verify(c.owned(failed['archive'][key], root), failed['archive'])
    with zipfile.ZipFile(failed['archive']['local_path']) as archive:
        evidence.checked_members(archive)
        require(json.loads(archive.read('state.json')) == failed['state'], 'Original failed state differs')
        require(hashlib.sha256(archive.read('campaign_plan.json')).hexdigest() == PLAN_SHA, 'Original plan changed')
        archived_commands = json.loads(archive.read('commands.json'))
    calibration = c.calibration_verified(plan['calibration']['report_path'], plan)
    c.verify(plan['confirmation_freeze_path'], FREEZE_ID); c.verify(plan['accepted_development_path'], DEVELOPMENT_ID)
    registered = c.jobs_for(plan, registry, 'confirmation', old_output, calibration, PLAN_SHA)
    require([(registered[i]['condition_id'], registered[i]['dataset'], registered[i]['seed']) for i in (20,21,22,23)] ==
            [('tuned-chamfer','modelnet40_c',2), ('tuned-chamfer','scanobjectnn_c',2), ('original3dd','modelnet40_c',2), ('original3dd','scanobjectnn_c',2)], 'Frozen remaining slot roles changed')
    projected_budget(ledger['spent_seconds'], sum(POINT_SECONDS))
    receipt_path = ROOT / 'sensitivity_confirmation_remaining_receipt.json'
    require(not receipt_path.exists(), 'Remaining receipt exists; never resume automatically')
    output = c.owned(ROOT / ('sensitivity_confirmation_remaining_' + evidence.utc())); output.mkdir(exist_ok=False)
    drive_output = c.DRIVE / 'phase-evidence' / output.name; drive_output.mkdir(parents=True, exist_ok=False)
    jobs = select_remaining(registered, archived_commands, failed['state'], output)
    for job in jobs:
        expected = c.command_for(plan, job['trial'], job['dataset'], 'confirmation', job['seed'], Path(job['result_root']), calibration)
        require(job['command'] == expected, 'New output command differs from pinned builder')
    for source, name in ((original_receipt_path, 'original_failed_receipt.json'), (ROOT / c.LEDGER_NAME, 'original_failed_ledger.json'),
                         (preservation_path, 'failure_preservation_receipt.json'), (recovery_path, 'reviewed_recovery_plan.json')):
        shutil.copyfile(source, output / name)
    for name in recovery['sources']:
        shutil.copyfile(here / name, output / name)
    for source, name in ((ROOT / 'sensitivity_campaign_plan.json', 'campaign_plan.json'),
                         (ROOT / 'sensitivity_gpu_approval.json', 'gpu_approval.json'),
                         (Path(plan['confirmation_freeze_path']), Path(plan['confirmation_freeze_path']).name),
                         (Path(plan['accepted_development_path']), Path(plan['accepted_development_path']).name)):
        shutil.copyfile(source, output / name)
    for name in plan['phase_sources']:
        shutil.copyfile(controller_path.parent / name, output / name)
    write_json(output / 'original_commands.json', archived_commands); write_json(output / 'commands.json', jobs)
    manifest = json.loads((c.ASSETS / 'manifest.json').read_text())
    require(identity(c.ASSETS / 'manifest.json')['sha256'] == plan['manifest_sha256'], 'Asset manifest changed')
    write_json(output / 'fresh_cpu_preflight.json', fresh_cpu_preflight(c, manifest, jobs, ledger))
    with zipfile.ZipFile(proof['archive']['local_path']) as archive:
        evidence.checked_members(archive); cache_records = json.loads(archive.read('cache_records.json'))
    require(len(cache_records) == len({row['local_path'] for row in cache_records}) == 180, 'Preserved180-cache records incomplete')
    for row in cache_records:
        c.verify(c.owned(row['local_path']), row); c.verify(c.owned(row['drive_path'], c.DRIVE), row)
    state = dict(status='running', stage='confirmation_remaining', session=c.SESSION, phase_plan_sha256=PLAN_SHA,
                 manifest_sha256=plan['manifest_sha256'], source_snapshot_sha256=plan['source_snapshot_sha256'],
                 original_confirmation_archive=failed['archive'], original_receipt_identity=RECEIPT_ID,
                 reviewed_recovery_plan_sha256=args.recovery_plan_sha256, registered_jobs=[21,22,23], excluded_failed_slots=[20],
                 jobs=[], cache_records=[], full_benchmark=False)
    started = time.monotonic(); previous_spent = ledger['spent_seconds']; history = copy.deepcopy(ledger['attempts'])
    ledger['attempts'].append(dict(stage='confirmation_remaining', output=str(output), status='running', utc=evidence.utc()))
    ledger['active'] = dict(stage='confirmation_remaining', output=str(output), controller_pid=os.getpid())
    try:
        c.save_ledger(ledger)
        for position, job in enumerate(jobs):
            projected_budget(previous_spent + time.monotonic() - started, sum(POINT_SECONDS[position:]))
            index = job['registered_job_index']; log = output / ('job_%03d.log' % index)
            record = dict(job, status='running'); state['jobs'].append(record); before = time.monotonic()
            def on_start(pid):
                ledger['active'].update(worker_group_pid=pid, registered_job_index=index, command=job['command']); c.save_ledger(ledger)
            try:
                gpu.stream_job(job['command'], log, c.remaining_seconds(previous_spent, time.monotonic() - started, 1200), on_start)
            except Exception as error:
                if not log.exists() or not known_numerical_failure(log.read_text(), error):
                    raise
                result = failed_worker_metadata(job, plan, manifest, selections[('confirmation', job['dataset'])], calibration, c, evidence)
                validate_global_contracts(result, job['dataset'], ledger)
                record.update(result, status='failed_numerical', error=repr(error), log_identity=identity(log), elapsed_seconds=time.monotonic()-before)
            else:
                result = c.validate_job(job, 'confirmation', plan, manifest, selections[('confirmation', job['dataset'])], calibration, evidence)
                validate_global_contracts(result, job['dataset'], ledger)
                record.update(result, status='complete', elapsed_seconds=time.monotonic()-before)
                record['cache_records'] = c.persist_caches([record], evidence)
            ledger['active'].pop('worker_group_pid', None)
            write_json(output / ('job_%03d_validation.json' % index), record)
            ledger['spent_seconds'] = previous_spent + time.monotonic() - started; c.save_ledger(ledger)
        c.pairing_verified(failed['state']['jobs'][:20] + [job for job in state['jobs'] if job['status']=='complete'])
        state['cache_records'] = c.persist_caches([job for job in state['jobs'] if job['status']=='complete'], evidence)
        require([job['registered_job_index'] for job in state['jobs']] == [21,22,23]
                and all(job['status'] in ('complete','failed_numerical') for job in state['jobs']), 'Remaining slots not fully recorded')
        state['status'] = 'remaining_slots_recorded'
    except BaseException:
        state.update(status='failed_preserved', failure=traceback.format_exc())
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        require(ledger['attempts'][:-1] == history, 'Original failed ledger history was altered')
        ledger['spent_seconds'] = previous_spent + state['elapsed_seconds']; ledger['active'] = None
        ledger['attempts'][-1].update(status='complete' if state['status']=='remaining_slots_recorded' else 'failed', elapsed_seconds=state['elapsed_seconds'])
        c.save_ledger(ledger); write_json(output / 'state.json', state); write_json(output / 'ledger_snapshot.json', ledger)
        sealed = evidence.seal(output, drive_output / 'confirmation_remaining_evidence.zip')
        ledger['spent_seconds'] = previous_spent + time.monotonic() - started; c.save_ledger(ledger)
        receipt = dict(state=state, archive=sealed, final_campaign_spent_seconds=ledger['spent_seconds'])
        write_json(receipt_path, receipt); write_json(drive_output / 'receipt.json', receipt)
        print('AGENT_B_REMAINING_SLOTS_RECEIPT', json.dumps(sealed), flush=True)
    require(ledger['spent_seconds'] <= 19800 and state['status']=='remaining_slots_recorded', 'Remaining phase failed or cumulative cap exceeded')
    print('AGENT_B_REMAINING_PHASE_RECORDED', flush=True)


if __name__ == '__main__':
    main()
