"""Three bounded frozen-source repeat diagnostics inside the original5h ledger."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

REMOTE = Path('/content/studies/chamfer-guidance-variants/remote')
sys.path.insert(0, str(REMOTE))
import chamfer_phase_common as common
import gpu_chamfer_phase as phase
h = common.house


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'commands', 'approval', 'failure-receipt', 'worker', 'phase', 'launcher'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--failed-archive-sha256', required=True)
    args = parser.parse_args()
    paths = dict(plan=REMOTE/'campaign_plan.json', commands=REMOTE/'campaign_commands.json',
                 approval=REMOTE/'gpu_approval.json', failure_receipt=common.ROOT/'gpu_smoke_initial_receipt.json',
                 worker=REMOTE/'diagnose_native_repeat_worker.py', phase=Path(__file__).resolve(),
                 launcher=REMOTE/'launch_native_repeat_diagnostic.py')
    for name, path in paths.items():
        h.require(h.identity(path)['sha256'] == getattr(args, name + '_sha256'), 'Diagnostic pinned identity mismatch: ' + name)
    plan = json.loads(paths['plan'].read_text())
    prepared = json.loads(paths['commands'].read_text())
    failed = json.loads(paths['failure_receipt'].read_text())
    phase.verify_phase_sources(plan['phase_sources'], REMOTE)
    common.configure(); common.setup_gate()
    sys.path.insert(0, plan['source_root'])
    from scripts import run_chamfer_variants_campaign as planner
    from scripts.analyze_scan_trajectory import analyze_bundle, Bundle
    planner.verify_approval(paths['approval'], args.approval_sha256, args.plan_sha256, plan)
    h.require(prepared['plan'] == plan and prepared['status'] == 'prepared_not_executed', 'Frozen command plan mismatch')
    h.require(failed['state']['status'] == 'failed' and failed['state']['stage'] == 'smoke_initial'
              and failed['state']['plan_sha256'] == args.plan_sha256 and len(failed['state']['jobs']) == 8,
              'Expected failed eight-job technical smoke receipt')
    for key in ('local_path', 'drive_path'):
        h.verify(failed['archive'][key], failed['archive'])
    h.require(failed['archive']['sha256'] == args.failed_archive_sha256, 'Original failed evidence archive changed')
    with zipfile.ZipFile(failed['archive']['local_path']) as archive:
        h.require(json.loads(archive.read('state.json')) == failed['state'], 'Failed receipt/state archive mismatch')
    h.verify(common.DRIVE/'phase-evidence/gpu_smoke_initial/receipt.json', h.identity(paths['failure_receipt']))
    h.verify(Path(plan['asset_root'])/'manifest.json', plan['asset_manifest'])
    manifest = json.loads((Path(plan['asset_root'])/'manifest.json').read_text())
    for name, identity in manifest['files'].items():
        if name.startswith('source_snapshot/'):
            h.verify(Path(plan['asset_root'])/name, identity)
    h.verify(plan['source_receipt']['path'], plan['source_receipt'])
    ledger = phase.load_ledger(args.plan_sha256)
    h.require(ledger['stages']['smoke_initial']['status'] == 'failed', 'Failed-stage ledger changed')
    h.require('native_repeat_diagnostic' not in ledger.get('diagnostics', {}), 'No automatic diagnostic retry')
    phase.check_budget(ledger, 600)
    base_jobs = {job['arm']: job for job in failed['state']['jobs'] if job['dataset'] == 'modelnet40_c'}
    frozen = phase.select_jobs(planner.build_jobs(plan, 'smoke'), 'smoke_initial')
    for arm in ('baseline', 'sham_kl'):
        original = next(job for job in frozen if job['dataset'] == 'modelnet40_c' and job['arm'] == arm)
        h.require(base_jobs[arm]['command'] == original['command'], 'Diagnostic source argv differ from frozen smoke')
    output = common.ROOT/'native_repeat_diagnostic'
    drive = common.DRIVE/'phase-evidence/native_repeat_diagnostic'
    output.mkdir(exist_ok=False); drive.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', purpose='C1/C2/KL0 native repeat floor; no parity tolerance changes',
                 plan_sha256=args.plan_sha256, original_started_epoch=ledger['started_epoch'],
                 original_deadline_epoch=ledger['started_epoch'] + phase.CAP, phase_limit_seconds=600, jobs=[])
    previous_active = ledger['active_seconds']
    event = dict(status='running', output=str(output), started_epoch=time.time())
    ledger.setdefault('diagnostics', {})['native_repeat_diagnostic'] = event
    def account():
        ledger['active_seconds'] = previous_active + time.monotonic() - started
        event['elapsed_seconds'] = time.monotonic() - started
        phase.save_ledger(ledger)
    try:
        h.preserve_phase_sources(output)
        for name, path in paths.items():
            if path.is_file() and path.name not in {file.name for file in output.iterdir()}:
                shutil.copyfile(path, output/path.name)
        for number, (case, arm) in enumerate((('C1','baseline'), ('C2','baseline'), ('KL0','sham_kl'))):
            original = base_jobs[arm]
            job = {key: value for key, value in original.items() if key not in ('archives', 'elapsed_active_seconds')}
            job['result_root'] = str(common.ROOT/'results/native_repeat_diagnostic'/case)
            job['drive_root'] = str(common.DRIVE/'results/native_repeat_diagnostic'/case)
            command = list(original['command'])
            for flag, key in (('--result-root','result_root'), ('--drive-root','drive_root')):
                command[command.index(flag)+1] = job[key]
            job['command'] = command
            h.require(not Path(job['result_root']).exists() and not Path(job['drive_root']).exists(), 'Fresh diagnostic outputs required')
            cache = Path(command[command.index('--paired-state-root')+1])
            cache_paths = sorted((cache/'modelnet40_c').rglob('*.npz'))
            h.require(len(cache_paths) == 2, 'Expected immutable Gaussian/background paired cache')
            for path in cache_paths:
                h.verify(path, ledger['cache_identities'][str(path)])
            arguments = command[3:]
            if arm == 'sham_kl':
                arguments = ['--chamfer-experiment', 'semantic', *arguments]
            spec = dict(case=case, source_root=plan['source_root'], original_command=command,
                        effective_runner_arguments=arguments, sidecar_root=str(output/'tensors'/case))
            spec_path = output/('case_' + case + '.json')
            spec_path.write_text(json.dumps(spec, indent=2) + '\n')
            ledger['active'] = dict(stage='native_repeat_diagnostic', job=case, command=command)
            account()
            old = os.environ.get('CHAMFER_NATIVE_DIAGNOSTIC_SPEC')
            os.environ['CHAMFER_NATIVE_DIAGNOSTIC_SPEC'] = str(spec_path)
            try:
                timeout = min(600 - (time.monotonic()-started) - 60, phase.check_budget(ledger, 60))
                h.require(timeout > 0, 'Diagnostic600s budget exhausted')
                def began(pid):
                    ledger['active']['pid'] = pid; account()
                phase.stream([plan['python'], '-u', str(paths['worker'])], output/('job%02d_%s.log' % (number,case)),
                             plan['source_root'], timeout, began)
            finally:
                if old is None: os.environ.pop('CHAMFER_NATIVE_DIAGNOSTIC_SPEC', None)
                else: os.environ['CHAMFER_NATIVE_DIAGNOSTIC_SPEC'] = old
            archives = phase.validate_job(job, analyze_bundle, Bundle)
            for path in cache_paths:
                h.verify(path, ledger['cache_identities'][str(path)])
            state['jobs'].append(dict(case=case, original_arm=arm, command=command, archives=archives,
                                      sidecars={str(path.relative_to(output)):h.identity(path) for path in (output/'tensors'/case).rglob('*') if path.is_file()}))
            account()
        state['status'] = 'complete'
    except BaseException:
        state.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        (output/'state.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        ledger['active'] = None; event['status'] = state['status']; account()
        archive = h.seal(output, drive/'evidence.zip')
        receipt = dict(state=state, archive=archive)
        with (drive/'receipt.json').open('x') as stream: stream.write(json.dumps(receipt,indent=2)+'\n')
        with (common.ROOT/'native_repeat_diagnostic_receipt.json').open('x') as stream: stream.write(json.dumps(receipt,indent=2)+'\n')
        h.verify(drive/'receipt.json', h.identity(common.ROOT/'native_repeat_diagnostic_receipt.json'))
        account()
        h.require(time.monotonic()-started <= 600 and phase.elapsed_seconds(ledger) <= phase.CAP, 'Diagnostic/original continuous deadline exceeded')
        print('CHAMFER_NATIVE_REPEAT_DIAGNOSTIC_SEALED', state['status'], json.dumps(archive), flush=True)


if __name__ == '__main__': main()
