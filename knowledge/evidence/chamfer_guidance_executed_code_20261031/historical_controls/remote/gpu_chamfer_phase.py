"""One reviewed stage of the exact approved5h Chamfer campaign; no auto retry."""
import argparse
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

import chamfer_phase_common as common
h = common.house
STAGES = ('smoke_initial', 'calibration', 'smoke_active', 'development', 'confirmation', 'timing')
CAP, RESERVE = 18000, 1800


def elapsed_seconds(ledger, now=None):
    return max(ledger['active_seconds'], (time.time() if now is None else now) - ledger['started_epoch'])


def check_budget(ledger, expected_remaining, now=None):
    left = CAP - elapsed_seconds(ledger, now)
    h.require(math.isfinite(left) and left >= 1.1 * expected_remaining + RESERVE,
              'Whole continuous research wall-clock projection exceeds5h cap; stop for review')
    return left - RESERVE


def select_jobs(jobs, stage):
    if stage not in ('smoke_initial', 'smoke_active'):
        return jobs
    initial = {'baseline', 'sparse', 'sham_kl', 'sham_h'}
    return [job for job in jobs if (job['arm'] in initial) == (stage == 'smoke_initial')]


def verify_phase_sources(records, directory, required=Path(__file__).name):
    h.require(required in records, 'Executing GPU phase is not bound by approved plan')
    for name, expected in records.items():
        h.require(Path(name).name == name, 'Unsafe phase source path')
        h.verify(Path(directory) / name, expected)


def save_ledger(ledger):
    raw = json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + '\n'
    for root in (common.ROOT, common.DRIVE):
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'gpu_campaign_ledger.json'
        pending = path.with_suffix('.pending.json'); pending.write_text(raw); pending.replace(path)
    h.verify(common.DRIVE / 'gpu_campaign_ledger.json', h.identity(common.ROOT / 'gpu_campaign_ledger.json'))


def load_ledger(plan_sha):
    local, remote = [root / 'gpu_campaign_ledger.json' for root in (common.ROOT, common.DRIVE)]
    if not local.exists() and not remote.exists():
        return dict(plan_sha256=plan_sha, session=common.SESSION, cap_seconds=CAP,
                    started_epoch=time.time(), active_seconds=0., active=None, stages={})
    h.require(local.exists() and remote.exists(), 'Ledger copies disagree; inspect before recovery')
    h.verify(local, h.identity(remote)); ledger = json.loads(local.read_text())
    h.require(ledger['plan_sha256'] == plan_sha and ledger['session'] == common.SESSION
              and ledger['cap_seconds'] == CAP and ledger['active'] is None, 'Foreign/unresolved campaign ledger')
    return ledger


def stream(command, log_path, cwd, timeout, on_start=lambda pid: None):
    """Own child group only; raw logs survive failure and bounded termination."""
    h.require(timeout > 0, 'Approved budget exhausted before worker launch')
    child = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
    selector = selectors.DefaultSelector(); selector.register(child.stdout, selectors.EVENT_READ)
    started, beat, chunks = time.monotonic(), time.monotonic(), 0
    try:
        on_start(child.pid)
        with Path(log_path).open('xb') as log:
            while selector.get_map():
                if time.monotonic() - started >= timeout:
                    raise TimeoutError('Owned worker reached remaining whole-campaign budget')
                for key, _ in selector.select(.2):
                    raw = os.read(key.fd, 65536)
                    if not raw:
                        selector.unregister(key.fileobj); continue
                    log.write(raw); log.flush(); chunks += 1
                    sys.stdout.write(raw.decode('utf-8', errors='replace')); sys.stdout.flush()
                if time.monotonic() - beat >= 30:
                    print('CHAMFER_WORKER_PROGRESS', h.utc(), 'pid', child.pid, 'elapsed_s', round(time.monotonic()-started),
                          'output_chunks', chunks, 'log', str(log_path), flush=True); beat = time.monotonic()
            h.require(child.wait(timeout=max(.1, timeout-(time.monotonic()-started))) == 0, 'Owned worker failed: ' + str(command))
    finally:
        selector.close(); child.stdout.close()
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()


def validate_job(job, analyze_bundle, Bundle):
    archives = sorted(Path(job['result_root']).glob('*/*/*.zip'))
    h.require(len(archives) == (1 if job['stage'] == 'calibration' else 4), 'Missing raw head archives')
    heads, paths = set(), []
    for archive in archives:
        with zipfile.ZipFile(archive) as raw: h.checked_members(raw)
        report = analyze_bundle(archive, allow_partial=True)
        h.require(report['selection_complete'], 'Truncated declared IDs are not accepted')
        bundle = Bundle(archive)
        try:
            config = bundle.json('config.json'); heads.add(config['backbone'])
            selection=Path(job['command'][job['command'].index('--index-manifest')+1])
            expected_variant='semantic' if job['arm'].startswith(('semantic_','sham_')) or job['arm']=='calibration' else None if job['arm']=='baseline' else job['arm']
            h.require(config['dataset']==job['dataset'] and config['seed']==job['seed']
                      and config.get('selected_original_ids')==json.loads(selection.read_text())['indices']
                      and config.get('chamfer_experiment',{}).get('variant')==expected_variant, 'Requested job/config/ID identity mismatch')
            weights = config['guidance_protocol']['loss_weights']
            h.require(weights == dict(chamfer=1., spectral=0., invariant=0.)
                      and config['guidance_protocol']['skip_inactive_components'], 'Forbidden objective component')
            for name in bundle.names:
                if name.startswith('batches/') and name.endswith('/diagnostics.json'):
                    metrics = bundle.json(name)['guidance_metrics']
                    h.require(metrics.get('graph_calls') == metrics.get('spectral_loss_calls') == 0, 'Forbidden graph/spectral execution')
            remote = Path(job['drive_root']) / config['dataset'] / config['method'] / config['run_id'] / 'final.zip'
            h.verify(remote, h.identity(archive)); paths.append(str(archive))
        finally:
            bundle.close()
    h.require(heads == ({'pointmae'} if job['stage'] == 'calibration' else {'dgcnn','curvenet','pointnext','pointmae'}), 'Head coverage mismatch')
    return paths


def sham_parity(records, Bundle):
    import io
    import numpy as np
    baseline = {record['dataset']: record for record in records if record['arm'] == 'baseline'}
    for record in records:
        if record['arm'] not in ('sham_kl', 'sham_h'):
            continue
        reference = baseline[record['dataset']]
        by_head = {}
        for item in (reference, record):
            for path in item['archives']:
                bundle = Bundle(path)
                try:
                    config = bundle.json('config.json'); head = config['backbone']
                    values = dict(config={key: config.get(key) for key in ('lion_state_before','lion_state_after','classifier_state_before_sha256',
                        'classifier_state_after_sha256','asset_manifest_sha256','native_extensions')}, batches={})
                    for name in sorted(bundle.names):
                        if name.startswith('batches/') and name.endswith('/predictions.npz'):
                            diagnostic = bundle.json(name.rsplit('/',1)[0]+'/diagnostics.json'); trace = diagnostic['trajectory']
                            with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as arrays:
                                values['batches'][name] = {key: arrays[key].tolist() for key in ('index','label','trajectory_logits')}
                            values['batches'][name].update({key: diagnostic[key] for key in ('geometry_sha256','reconstruction_rng_after_sha256')})
                            values['batches'][name].update({key: trace[key] for key in ('initial_hashes','initial_rng_sha256','final_hashes')})
                            values['batches'][name]['probe_states']=[{key:p[key] for key in ('clean_latent_sha256','style_sha256','decoded_geometry_sha256','rng_before_sha256','rng_after_sha256')} for p in trace['probes']]
                    if item is reference:
                        by_head[head] = values
                    else:
                        h.require(by_head[head] == values, 'Semantic0/C geometry/logit/RNG/frozen parity failed')
                finally:
                    bundle.close()


def preserve_caches(ledger):
    for path in sorted((common.ROOT / 'paired_states').rglob('*.npz')):
        target = common.DRIVE / path.relative_to(common.ROOT); expected = h.identity(path)
        previous = ledger.setdefault('cache_identities',{}).get(str(path))
        h.require(previous is None or previous==expected, 'Immutable paired-state cache changed')
        if previous is None:
            h.verify(target, expected) if target.exists() else h.copy_checked(path, target, expected)
            ledger['cache_identities'][str(path)]=expected
        h.require(target.exists(), 'Verified Drive cache disappeared')


def native_calibration_gradients(records, Bundle):
    norms = {(r['dataset'],candidate,component): [] for r in records for candidate in ('semantic_kl','semantic_h','weighted') for component in ('local','style')}
    for record in records:
        bundle = Bundle(record['archives'][0])
        try:
            for name in bundle.names:
                if name.startswith('batches/') and name.endswith('/diagnostics.json'):
                    for row in bundle.json(name)['trajectory']['calibration']['rows']:
                        for component in ('local','style'): norms[(record['dataset'],row['candidate'],component)].append(row[component+'_candidate_grad_norm'])
        finally: bundle.close()
    h.require(all(values and all(math.isfinite(v) and v >= 0 for v in values) and max(values)>1e-12
                  for values in norms.values()), 'Native candidate gradients must be finite/nonzero for local and style')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'commands', 'approval'):
        parser.add_argument('--'+name, type=Path, required=True); parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--stage', choices=STAGES, required=True); args = parser.parse_args(argv)
    for name in ('plan','commands','approval'):
        h.require(h.identity(getattr(args,name))['sha256'] == getattr(args,name+'_sha256'), 'Pinned '+name+' SHA mismatch')
    plan = json.loads(args.plan.read_text()); prepared = json.loads(args.commands.read_text())
    verify_phase_sources(plan['phase_sources'], common.PHASES)
    common.configure(); common.setup_gate()
    sys.path.insert(0, plan['source_root'])
    from scripts import run_chamfer_variants_campaign as planner
    from scripts.analyze_scan_trajectory import analyze_bundle, Bundle
    h.require(Path(planner.__file__).resolve() == Path(plan['source_root'])/'scripts/run_chamfer_variants_campaign.py', 'Unsealed planner import')
    planner.verify_approval(args.approval, args.approval_sha256, args.plan_sha256, plan)
    h.require(prepared['plan'] == plan and prepared['status'] == 'prepared_not_executed', 'Prepared command plan mismatch')
    h.require(plan['research_root'] == str(common.ROOT) and plan['drive_root'] == str(common.DRIVE)
              and plan['asset_manifest']['sha256'] == h.MANIFEST_SHA, 'Foreign root/assets')
    h.verify(Path(plan['asset_root'])/'manifest.json', plan['asset_manifest'])
    manifest=json.loads((Path(plan['asset_root'])/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if name.startswith('source_snapshot/'): h.verify(Path(plan['asset_root'])/name,expected)
    h.verify(plan['source_receipt']['path'],plan['source_receipt'])
    for datasets in plan['selections'].values():
        for reference in datasets.values(): h.verify(reference['path'],reference)
    h.require((h.PREPARED/'.complete').read_text().strip() == h.MANIFEST_SHA, 'Accepted CPU proof missing')
    cpu = json.loads((h.PREPARED/'accepted_cpu_preflight_receipt.json').read_text())
    h.require(cpu['state']['status'] == 'complete' and cpu['state']['strict_loads'] == 8 and cpu['state']['cuda_initialized'] is False, 'CPU proof incomplete')
    h.require(cpu['state']['session']==common.SESSION and cpu['state']['manifest_sha256']==h.MANIFEST_SHA, 'CPU source/session mismatch')
    h.verify(h.PREPARED/'cpu_preflight.json',cpu['report_identity'])
    h.require(json.loads((h.PREPARED/'cpu_preflight.json').read_text())['source_snapshot_sha256']==h.SOURCE_SHA, 'CPU source archive mismatch')
    for key in ('local_path','drive_path'): h.verify(cpu['archive'][key], cpu['archive'])
    h.verify(Path(plan['asset_root'])/'source_snapshot.zip', dict(bytes=(Path(plan['asset_root'])/'source_snapshot.zip').stat().st_size,sha256=h.SOURCE_SHA))
    ledger = load_ledger(args.plan_sha256); previous = STAGES[STAGES.index(args.stage)-1] if args.stage != STAGES[0] else None
    h.require(args.stage not in ledger['stages'] and (previous is None or ledger['stages'].get(previous,{}).get('status')=='complete'), 'Prior stage incomplete or stage already attempted')
    if previous is not None:
        accepted=json.loads((common.ROOT/('gpu_'+previous+'_receipt.json')).read_text())
        h.require(accepted['state']['status']=='complete' and accepted['state']['plan_sha256']==args.plan_sha256, 'Prior accepted receipt missing')
        for key in ('local_path','drive_path'): h.verify(accepted['archive'][key],accepted['archive'])
    base_stage = 'smoke' if args.stage.startswith('smoke_') else args.stage
    h.require(prepared['stages'][base_stage]==planner.build_jobs(plan,base_stage), 'Prepared argv differ from frozen source')
    coefficients = None
    if args.stage not in ('smoke_initial','calibration'):
        h.verify(plan['calibration_report'],ledger['calibration_report'])
        coefficients = planner.verify_calibration(Path(plan['calibration_report']))['coefficients']
    jobs = select_jobs(planner.build_jobs(plan, base_stage, coefficients, h.identity(plan['calibration_report'])['sha256'] if coefficients else None), args.stage)
    frozen = select_jobs(prepared['stages'][base_stage], args.stage)
    h.require([(j['arm'],j['dataset'],j['seed'],j['repeat'],j['result_root']) for j in jobs] == [(j['arm'],j['dataset'],j['seed'],j['repeat'],j['result_root']) for j in frozen], 'Frozen jobs changed')
    output = common.ROOT/('gpu_'+args.stage); output.mkdir(exist_ok=False)
    drive = common.DRIVE/'phase-evidence'/output.name; drive.mkdir(parents=True,exist_ok=False)
    state = dict(status='running',stage=args.stage,plan_sha256=args.plan_sha256,jobs=[]); started = time.monotonic(); prior_active = ledger['active_seconds']
    attempted = []
    estimates = dict(smoke_initial=480,calibration=900,smoke_active=360,development=1780,confirmation=9000,timing=1260)
    def account():
        ledger['active_seconds'] = prior_active + time.monotonic()-started
        if args.stage in ledger['stages']: ledger['stages'][args.stage]['active_seconds']=time.monotonic()-started
        save_ledger(ledger)
    try:
        h.preserve_phase_sources(output)
        for name in ('plan','commands','approval'): shutil.copyfile(getattr(args,name),output/(name+'.json'))
        (output/'executed_commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
        h.require(all(not Path(job['result_root']).exists() and not Path(job['drive_root']).exists() for job in jobs), 'Fresh job roots required')
        if args.stage in ('smoke_initial','development','confirmation'):
            h.require(not (common.ROOT/'paired_states'/base_stage).exists(), 'Fresh stage cache namespace required')
        for number, job in enumerate(jobs):
            remaining = estimates[args.stage]*(len(jobs)-number)/len(jobs)+sum(estimates[s] for s in STAGES[STAGES.index(args.stage)+1:])
            timeout = check_budget(ledger, remaining)
            h.require(job['executable'] and not Path(job['result_root']).exists() and not Path(job['drive_root']).exists(), 'Unresolved/reused job output')
            ledger['active'] = dict(stage=args.stage,job=number,command=job['command']); account()
            def began(pid): ledger['active']['pid']=pid; account()
            attempted.append(job); job_started=time.monotonic()
            stream(job['command'],output/('job%03d.log'%number),plan['source_root'],timeout,began)
            record = dict(job,elapsed_active_seconds=time.monotonic()-job_started,archives=validate_job(job,analyze_bundle,Bundle)); state['jobs'].append(record)
            preserve_caches(ledger); account()
            (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        if args.stage == 'smoke_initial': sham_parity(state['jobs'],Bundle)
        if args.stage == 'calibration':
            native_calibration_gradients(state['jobs'],Bundle)
            command = [plan['python'],'-u',str(Path(plan['source_root'])/'scripts/calibrate_chamfer_variants.py'),'summarize',
                       *[record['archives'][0] for record in state['jobs']],'--output',plan['calibration_report']]
            (output/'summarize_command.json').write_text(json.dumps(command,indent=2)+'\n')
            stream(command,output/'summarize.log',plan['source_root'],check_budget(ledger,sum(estimates[s] for s in STAGES[2:])))
            planner.verify_calibration(Path(plan['calibration_report']))
            ledger['calibration_report']=h.identity(plan['calibration_report'])
            h.copy_checked(plan['calibration_report'],drive/'calibration.json',h.identity(plan['calibration_report']))
            shutil.copyfile(plan['calibration_report'],output/'calibration.json')
        state['status']='complete'
    except BaseException:
        state.update(status='failed',failure=traceback.format_exc())
        for job in attempted:
            path=Path(job['result_root'])
            if path.exists() and not any(r['result_root']==str(path) for r in state['jobs']):
                shutil.copytree(path,output/'partial_results'/job['dataset']/job['arm'])
        raise
    finally:
        state['elapsed_active_seconds']=time.monotonic()-started
        (output/'state.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
        ledger['stages'][args.stage]=dict(status=state['status'],output=str(output)); ledger['active']=None; account()
        try:
            archive=h.seal(output,drive/'evidence.zip')
            receipt=dict(state=state,archive=archive); (drive/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        except BaseException:
            ledger['stages'][args.stage]['status']='failed_delivery'; raise
        finally: account()
        (common.ROOT/('gpu_'+args.stage+'_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        h.require(elapsed_seconds(ledger)<=CAP,'Whole5h cap exceeded while preserving evidence')
        print('CHAMFER_GPU_STAGE_COMPLETE',args.stage,state['status'],json.dumps(archive),flush=True)


if __name__ == '__main__': main()
