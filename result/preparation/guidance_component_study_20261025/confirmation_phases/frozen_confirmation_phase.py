"""Execute only the frozen three-seed C/C+S heldout plan on Agent B's VM."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
import traceback
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_components_template as gpu
from deploy_assets_template import (ROOT, ASSETS, DRIVE, SESSION, MANIFEST_SHA, SOURCE_SHA,
                                    identity, verify, require, utc, seal, checked_members, preserve_phase_sources)

HERE = Path(__file__).resolve().parent
PLAN_SHA = 'c6ea3e01a03061afd227c82eb30eb128c2a4f1f342dbf937732dd5e61b122d45'
PLAN_PATH = ROOT / 'frozen_confirmation_plan.json'
LEDGER_NAME = 'frozen_confirmation_ledger.json'
LEDGER_PATH = ROOT / LEDGER_NAME
SOURCE_FILES = ('deploy_assets_template.py', 'preflight_components_cpu.py', 'gpu_components_template.py',
                'frozen_confirmation_phase.py', 'frozen_confirmation_plan.json')


def bytes_identity(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def plan_contract(plan):
    require(plan['status'] == 'frozen_before_any_confirmation_scoring' and plan['session'] == SESSION
            and plan['manifest_sha256'] == MANIFEST_SHA and plan['source_snapshot_sha256'] == SOURCE_SHA,
            'Frozen plan identity/status mismatch')
    require(plan['seeds'] == [0, 1, 2] and all(type(seed) is int for seed in plan['seeds'])
            and plan['arms'] == ['combined', 'chamfer'] and plan['datasets'] == list(gpu.DATASETS)
            and plan['backbones'] == list(gpu.HEADS) and plan['corruptions'] == 'all15'
            and plan['original_objects_per_dataset'] == 70 and plan['id_stage'] == 'heldout' and plan['severity'] == 5,
            'Frozen three-seed/two-arm/heldout scope changed')
    require(plan['compute']['new_ledger'] == LEDGER_NAME and plan['compute']['hard_active_cap_seconds'] == gpu.CAP == 3300
            and plan['compute']['conservative_planned_seconds'] <= gpu.CAP
            and plan['compute']['expected_seconds'] < 3600
            and plan['compute']['setup_receipt_retry_reserve_seconds'] == 300, 'Whole-phase budget/ledger scope changed')
    require(plan['candidate'] == dict(component_arm='chamfer', policy='probability_mean')
            and [(row['component_arm'], row['policy']) for row in plan['comparators']] ==
            [('combined', 'probability_mean'), ('combined', 'minimum_entropy')], 'Frozen policy/candidate changed')
    require(plan['cache_namespace'] == 'paired_confirmation_heldout_v1_seed{seed}' and
            plan['counts'] == dict(candidate_predictions=201600, case_policy_rows=144, jobs=12,
                                  raw_head_archives=48, shared_batches=180, terminal_head_input_events=50400),
            'Frozen cache namespace/event scope changed')


def sources_verified(expected_sha):
    path = HERE / 'confirmation_phase_source_identities.json'
    require(identity(path)['sha256'] == expected_sha, 'Controller/helper source manifest SHA mismatch')
    sources = json.loads(path.read_text())
    require(set(sources) == set(SOURCE_FILES), 'Incomplete/extra frozen phase sources')
    for name, expected in sources.items():
        verify(HERE / name, expected)
    require(sources['frozen_confirmation_plan.json']['sha256'] == PLAN_SHA, 'Phase sources contain a different plan')
    return sources


def accepted_inputs():
    raw_plan = PLAN_PATH.read_bytes()
    require(bytes_identity(raw_plan)['sha256'] == PLAN_SHA, 'Frozen confirmation plan SHA mismatch')
    plan = json.loads(raw_plan); plan_contract(plan)
    require((HERE / 'frozen_confirmation_plan.json').read_bytes() == raw_plan, 'Uploaded/root frozen plan mismatch')
    raw_acceptance = (ROOT / 'accepted_development.json').read_bytes()
    accepted = json.loads(raw_acceptance)
    receipt = json.loads((ROOT / 'accepted_development_receipt.json').read_text())
    require(accepted['status'] == 'accepted' and accepted['manifest_sha256'] == MANIFEST_SHA
            and accepted['source_snapshot_sha256'] == SOURCE_SHA, 'Independent development acceptance missing')
    require(bytes_identity(raw_acceptance) == receipt['acceptance'] and bytes_identity(raw_plan) == accepted['frozen_plan'] == receipt['frozen_plan'],
            'Acceptance/plan receipt identity mismatch')
    require(str(receipt['archive']['drive_path']).startswith(str(DRIVE) + '/'), 'Acceptance archive outside owned Drive root')
    verify(receipt['archive']['drive_path'], receipt['archive'])
    with zipfile.ZipFile(receipt['archive']['drive_path']) as archive:
        checked_members(archive)
        require(archive.read('accepted_development.json') == raw_acceptance and archive.read('frozen_confirmation_plan.json') == raw_plan,
                'Physical Drive acceptance/plan bytes differ')
        validation_bytes = archive.read('local_development_analysis/validation.json')
    require(bytes_identity(validation_bytes) == accepted['local_validation'] == plan['development_validation'],
            'Accepted independent development validation mismatch')
    validation = json.loads(validation_bytes)
    require(validation['status'] == 'complete_fixed_development_subsets' and validation['aligned_original_ids_labels']
            and validation['paired_preparation_verified'] and validation['cache_receipt_count'] == 30,
            'Prior development evidence is incomplete')
    phase = accepted['phase_archive']
    require(phase == plan['development_phase_archive'] and str(phase['drive_path']).startswith(str(DRIVE) + '/'),
            'Prior phase archive identity/ownership mismatch')
    verify(phase['drive_path'], phase)
    costs = {(row['dataset'], row['component_arm']): row['remote_job_plus_validation_seconds']
             for row in validation['cost'] if row['component_arm'] in plan['arms']}
    require(len(costs) == 4 and abs(sum(costs.values()) - plan['compute']['measured_development_selected_job_seconds']) < 1e-6,
            'Registered whole-phase timing basis differs from accepted evidence')
    return plan, accepted, receipt, costs


def cache_for(plan, seed):
    require(type(seed) is int and seed in plan['seeds'], 'Unregistered confirmation seed')
    path = ROOT / plan['cache_namespace'].format(seed=seed)
    require(path != gpu.PAIRED and path.name == 'paired_confirmation_heldout_v1_seed%d' % seed, 'Confirmation/development cache namespace collision')
    return path


def job_plan(plan, output):
    result = []
    for seed in plan['seeds']:
        for job in gpu.commands('heldout', Path(output) / ('seed%d' % seed), seed=seed,
                                arms=tuple(plan['arms']), paired_state_root=cache_for(plan, seed)):
            result.append(dict(job, seed=seed))
    require(len(result) == plan['counts']['jobs'], 'Frozen job count mismatch')
    return result


def heldout_verified(plan, manifest):
    for dataset in plan['datasets']:
        entry = plan['id_manifests'][dataset]
        path = ASSETS / 'study' / entry['path']
        verify(path, entry)
        require({key: entry[key] for key in ('bytes', 'sha256')} == manifest['files']['study/' + entry['path']], 'Heldout manifest is not presealed')
        selection = json.loads(path.read_text())
        ids = selection['indices']
        require(selection['stage'] == 'heldout' and selection['dataset'] == dataset and len(ids) == 70
                and all(type(value) is int for value in ids) and ids == sorted(set(ids)), 'Invalid frozen heldout IDs')
        development = json.loads((ASSETS / 'study' / (dataset + '_development_ids.json')).read_text())
        require(not set(ids) & set(development['indices']), 'Heldout/development original objects overlap')
    for seed in plan['seeds']:
        cache = cache_for(plan, seed)
        require(not cache.exists() and not (DRIVE / cache.name).exists(), 'Confirmation cache already exists; inspect prior attempt before any retry')


def save(ledger):
    gpu.save_ledger(ledger, ledger_path=LEDGER_PATH, drive_name=LEDGER_NAME)


def execute(source_manifest_sha):
    signal.signal(signal.SIGTERM, gpu.raise_on_terminate)
    sources = sources_verified(source_manifest_sha)
    ledger = gpu.load_ledger(ledger_path=LEDGER_PATH, drive_name=LEDGER_NAME, frozen_plan_sha256=PLAN_SHA)
    gpu.validate_ledger(ledger, frozen_plan_sha256=PLAN_SHA)
    require(not ledger['attempts'], 'Frozen phase already attempted; review its immutable results/ledger before proposing further work')
    stamp = 'frozen_confirmation_' + utc()
    output = ROOT / stamp; output.mkdir()
    drive_output = DRIVE / 'phase-evidence' / stamp; drive_output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', mode='frozen_confirmation', session=SESSION, frozen_plan_sha256=PLAN_SHA,
                 manifest_sha256=MANIFEST_SHA, source_snapshot_sha256=SOURCE_SHA, seeds=[0, 1, 2],
                 jobs=[], full_benchmark=False, helper_sources=sources, policies_frozen=True)
    started, previously_spent = time.monotonic(), ledger['spent_seconds']
    ledger['active'] = dict(mode='frozen_confirmation', output=str(output), phase_pid=os.getpid())
    ledger['attempts'].append(dict(output=str(output), status='running', utc=utc(), seeds=[0, 1, 2]))
    try:
        save(ledger)
        preserve_phase_sources(output)
        for name in ('frozen_confirmation_phase.py', 'confirmation_phase_source_identities.json', 'frozen_confirmation_plan.json'):
            shutil.copyfile(HERE / name, output / name)
        for name in ('accepted_development.json', 'accepted_development_receipt.json'):
            if (ROOT / name).exists():
                shutil.copyfile(ROOT / name, output / name)
        if PLAN_PATH.exists():
            shutil.copyfile(PLAN_PATH, output / 'runtime_frozen_confirmation_plan.json')
        plan, accepted, receipt, costs = accepted_inputs()
        gpu.confirm_preflight()
        sys.path.insert(0, str(ASSETS / 'source_snapshot'))
        from scripts import eval_multibackbone_source as source
        require(source.REPO == ASSETS / 'source_snapshot', 'Unsealed project source imported')
        manifest = source.read_manifest(ASSETS, MANIFEST_SHA)
        require(manifest['source_snapshot_sha256'] == SOURCE_SHA, 'Project source snapshot changed')
        heldout_verified(plan, manifest)
        jobs = job_plan(plan, output)
        (output / 'commands.json').write_text(json.dumps(jobs, indent=2) + '\n')
        reserve = plan['compute']['setup_receipt_retry_reserve_seconds']
        completed_reference_seconds, observed_job_seconds = 0., 0.
        for number, job in enumerate(jobs):
            elapsed = time.monotonic() - started
            scale = max(1., observed_job_seconds / completed_reference_seconds) if completed_reference_seconds else 1.
            projected = previously_spent + elapsed + scale * sum(costs[(r['dataset'], r['arm'])] for r in jobs[number:]) + reserve
            state['last_projected_whole_seconds'] = projected
            require(projected < 3600, 'Updated whole three-seed phase estimate exceeds60min; stop before further GPU and obtain comprehensive approval')
            require(projected <= gpu.CAP, 'Updated phase cannot finish within its frozen3300s cap')
            remaining = gpu.CAP - previously_spent - elapsed - reserve
            require(remaining > 0, 'Frozen phase cap/packing reserve exhausted')
            record = dict(job, status='running', utc=utc(), projected_whole_seconds_before_job=projected); state['jobs'].append(record)
            def on_start(pid):
                ledger['active'].update(job=number, seed=job['seed'], worker_group_pid=pid, command=job['command'])
                save(ledger)
            before = time.monotonic()
            gpu.stream_job(job['command'], output / ('job_%02d.log' % number), remaining, on_start)
            result = gpu.validate_job(job, 'heldout', manifest, seed=job['seed'], paired_state_root=cache_for(plan, job['seed']))
            record.update(result, status='complete', elapsed_seconds=time.monotonic() - before)
            completed_reference_seconds += costs[(job['dataset'], job['arm'])]
            observed_job_seconds += record['elapsed_seconds']
            (output / ('job_%02d_validation.json' % number)).write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
            ledger['spent_seconds'] = previously_spent + time.monotonic() - started
            ledger['active'].pop('worker_group_pid', None)
            save(ledger)
        for seed in plan['seeds']:
            group = [job for job in state['jobs'] if job['seed'] == seed]
            gpu.validate_pairing(group, arms=tuple(plan['arms']))
        rows = [dict(row, seed=job['seed']) for job in state['jobs'] for report in job['reports'] for row in report['macro']]
        require(len(rows) == 144 and sum(len(job['raw_archives']) for job in state['jobs']) == 48
                and sum(row['n'] for row in rows if row['policy'] == 'terminal') == 50400, 'Frozen confirmation coverage/counts mismatch')
        balanced = []
        for seed in plan['seeds']:
            for arm in plan['arms']:
                for policy in plan['policies_recorded']:
                    cases = [row for row in rows if row['seed'] == seed and row['component_arm'] == arm and row['policy'] == policy]
                    require(len(cases) == 8 and {(r['dataset'], r['backbone']) for r in cases} ==
                            {(dataset, head) for dataset in plan['datasets'] for head in plan['backbones']}, 'Missing balanced confirmation cases')
                    balanced.append(dict(seed=seed, arm=arm, policy=policy, n_cases=8, accuracy_pct=sum(row['accuracy_pct'] for row in cases) / 8))
        state.update(status='complete', macro=rows, balanced=balanced, candidate=plan['candidate'], comparators=plan['comparators'],
                     interpretation='Frozen object-disjoint70-ID confirmation on previously benchmark-exposed datasets; no policy reselection or broad seed-stability claim')
        for name, rows_to_write in [('macro', rows), ('balanced', balanced)]:
            with (output / (name + '.csv')).open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows_to_write[0])); writer.writeheader(); writer.writerows(rows_to_write)
    except BaseException:
        state.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        ledger['spent_seconds'] = previously_spent + state['elapsed_seconds']
        ledger['active'] = None
        ledger['attempts'][-1].update(status=state['status'], elapsed_seconds=state['elapsed_seconds'])
        save(ledger)
        (output / 'state.json').write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
        (output / 'ledger_snapshot.json').write_text(json.dumps(ledger, indent=2) + '\n')
        result = dict(state=state, archive=seal(output, drive_output / 'frozen_confirmation_evidence.zip'))
        ledger['spent_seconds'] = previously_spent + time.monotonic() - started
        save(ledger)
        (ROOT / 'frozen_confirmation_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        (drive_output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print('AGENT_B_FROZEN_CONFIRMATION_RECEIPT', json.dumps(result['archive']), flush=True)
    require(ledger['spent_seconds'] <= gpu.CAP, 'Frozen phase exceeded cap while persisting final evidence')
    validation = dict(status='validated', mode='frozen_confirmation', frozen_plan_sha256=PLAN_SHA,
                      manifest_sha256=MANIFEST_SHA, seeds=plan['seeds'], evidence=result['archive'], output=str(output),
                      spent_seconds=ledger['spent_seconds'], independent_local_validation_pending=True)
    (ROOT / 'frozen_confirmation_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print('AGENT_B_FROZEN_CONFIRMATION_VALIDATED', flush=True)


def self_check():
    plan = json.loads((HERE / 'frozen_confirmation_plan.json').read_text()); plan_contract(plan)
    require(identity(HERE / 'frozen_confirmation_plan.json')['sha256'] == PLAN_SHA, 'Self-check plan SHA changed')
    jobs = job_plan(plan, Path('/tmp/confirmation-test'))
    require(len(jobs) == 12 and {job['seed'] for job in jobs} == {0, 1, 2}, 'Frozen job grid/seed mismatch')
    require(len({cache_for(plan, seed) for seed in plan['seeds']}) == 3 and all(cache_for(plan, seed) != gpu.PAIRED for seed in plan['seeds']), 'Confirmation cache aliases development/another seed')
    for job in jobs:
        command = job['command']
        require(command[command.index('--seed') + 1] == str(job['seed']) and command[-1] == str(cache_for(plan, job['seed']))
                and command[command.index('--index-manifest') + 1].endswith(job['dataset'] + '_heldout_ids.json'), 'Seed/stage/cache command mismatch')
    for seed in (-1, True, .5, '1'):
        for call in (lambda: gpu.commands('heldout', '/tmp/test', seed=seed, paired_state_root='/tmp/explicit'),
                     lambda: gpu.validate_job({}, 'heldout', {}, seed=seed, paired_state_root='/tmp/explicit')):
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid seed accepted')
    original_drive = gpu.DRIVE
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve(); gpu.DRIVE = root / 'drive'; gpu.DRIVE.mkdir()
        initial = gpu.DRIVE / 'initial_gpu_phase_ledger.json'; initial.write_text('accepted initial ledger remains unchanged\n')
        before = identity(initial)
        local = root / LEDGER_NAME
        ledger = gpu.load_ledger(ledger_path=local, drive_name=LEDGER_NAME, frozen_plan_sha256=PLAN_SHA)
        require(ledger['spent_seconds'] == 0 and ledger['frozen_plan_sha256'] == PLAN_SHA, 'New frozen ledger missing plan binding')
        ledger['spent_seconds'] = 123.
        gpu.save_ledger(ledger, ledger_path=local, drive_name=LEDGER_NAME)
        local.unlink()
        restored = gpu.load_ledger(ledger_path=local, drive_name=LEDGER_NAME, frozen_plan_sha256=PLAN_SHA)
        require(restored['spent_seconds'] == 123. and identity(initial) == before, 'Retry reset frozen budget or touched initial phase')
        wrong = dict(restored, frozen_plan_sha256='0' * 64)
        gpu.save_ledger(wrong, ledger_path=local, drive_name=LEDGER_NAME)
        try:
            gpu.load_ledger(ledger_path=local, drive_name=LEDGER_NAME, frozen_plan_sha256=PLAN_SHA)
        except ValueError:
            pass
        else:
            raise AssertionError('Foreign frozen ledger accepted')
    gpu.DRIVE = original_drive
    print('AGENT_B_FROZEN_CONFIRMATION_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, gpu.raise_on_terminate)
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--source-manifest-sha256', required=True)
        execute(parser.parse_args().source_manifest_sha256)
