"""Run only one fixed registered phase; preserve failures and a shared55min cap."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/content/studies/shape-refresh')
OPERATOR = Path('/content/refresh_operator')
ASSETS = ROOT/'prepared/assets'
REPO = ASSETS/'source_snapshot'
DRIVE = Path('/content/drive/MyDrive/thesis/result/shape_latent_refresh_pilot_20261031')
PYTHON = '/usr/local/envs/3dd_tta_env/bin/python'
ARMS = ('baseline','mean_refresh','posterior_refresh')


def commands(stage, result, cache, manifest_sha):
    assert stage in ('smoke','pilot')
    ids = OPERATOR/('modelnet40_c_'+('smoke' if stage == 'smoke' else 'development')+'_ids.json')
    shared = [PYTHON,'-u',str(REPO/'scripts/eval_shape_refresh.py'),
        '--dataset-root','/content/3DD-TTA/data/modelnet40_c',
        '--label-path','/content/3DD-TTA/data/modelnet40_c/label.npy',
        '--asset-root',str(ASSETS),'--asset-manifest-sha256',manifest_sha,
        '--index-manifest',str(ids),'--refresh-state-root',str(cache/'common'),
        '--pointmae-config',str(REPO/'cfgs/tta_modelnet.yaml'),
        '--pointmae-checkpoint',str(ASSETS/'checkpoints/modelnet_jt.pth')]
    return {arm: shared+['--refresh-arm',arm,'--paired-state-root',str(cache/'selected'/arm),'--result-root',str(result/arm),
                         '--drive-root',str(DRIVE/'raw'/stage/arm)]+
                  (['--refresh-technical-smoke'] if stage == 'smoke' else []) for arm in ARMS}


def stream(command, log, deadline):
    with log.open('xb') as saved:
        child = subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(child.stdout,selectors.EVENT_READ)
        last = time.monotonic()
        try:
            while selector.get_map():
                if time.monotonic() > deadline:
                    raise TimeoutError('Whole initial research phase reaches its shared55min limit')
                for key,_ in selector.select(timeout=min(5,max(0,deadline-time.monotonic()))):
                    line = os.read(key.fd,65536)
                    if line:
                        print(line.decode('utf-8',errors='replace'),end='',flush=True)
                        saved.write(line);saved.flush()
                    else:
                        selector.unregister(key.fileobj)
                if time.monotonic()-last >= 30:
                    count = len(list(ROOT.glob('phase_*/raw/*/*/*/*/batches/*/*/predictions.npz')))
                    print(datetime.now(timezone.utc).isoformat(),'active child',child.pid,
                          'completed prediction shards',count,flush=True)
                    last = time.monotonic()
            code = child.wait(timeout=max(1,deadline-time.monotonic()))
            if code: raise subprocess.CalledProcessError(code,command)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        finally:
            selector.close()


def main(stage):
    sys.path.insert(0,str(REPO));sys.path.insert(0,str(OPERATOR))
    import deploy_assets_template as tools
    from analyze_phase import analyze, validate_finished_scope
    preflight = json.loads((ROOT/'cpu_preflight_receipt.json').read_text())
    assert preflight['status'] == 'complete'
    asset = json.loads((OPERATOR/'asset_identity_v1.json').read_text())
    tools.verify(ASSETS/'manifest.json',asset['manifest'])
    registered = json.loads((OPERATOR/'registration.json').read_text())
    assert registered['session'] == 'tta-shape-refresh-a100-20261031-r1'
    assert registered['manifest'] == asset['manifest'] and registered['research_cap_seconds'] == 3300
    ledger_path = ROOT/'research_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(spent_seconds=0.,completed=[],research_started_utc=datetime.now(timezone.utc).isoformat())
    if 'research_started_utc' in ledger:
        ledger['spent_seconds'] = max(ledger['spent_seconds'],(datetime.now(timezone.utc)-datetime.fromisoformat(ledger['research_started_utc'])).total_seconds())
    assert stage in ('smoke','pilot') and stage not in ledger['completed']
    assert not ledger.get('running') and not ledger.get('failed'), 'Inspect incomplete/failed phase; never silently retry'
    if stage == 'pilot':
        assert 'smoke' in ledger['completed'] and ledger['smoke_acceptance'] == 'passed'
    assert ledger['spent_seconds'] < 3300-180
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/('phase_'+stage+'_'+stamp);output.mkdir()
    cache = output/'paired'
    jobs = commands(stage,output/'raw',cache,asset['manifest']['sha256'])
    for name in ['run_phase.py','analyze_phase.py','registration.json','asset_identity_v1.json',
                 'modelnet40_c_smoke_ids.json','modelnet40_c_development_ids.json']:
        shutil.copyfile(OPERATOR/name,output/name)
    (output/'commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
    original_spent = ledger['spent_seconds'];start = time.monotonic()
    deadline = start+3300-original_spent-120
    ledger['running'] = dict(stage=stage,path=str(output),pid=os.getpid())
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    status = dict(stage=stage,status='running',completed_arms=[],source_manifest=asset['manifest'])

    try:
        for arm,command in jobs.items():
            if arm in status['completed_arms']: continue
            print(datetime.now(timezone.utc).isoformat(),'START',stage,arm,flush=True)
            stream(command,output/(arm+'.stdout.log'),deadline)
            config_paths = list((output/'raw'/arm).glob('*/*/*/config.json'))
            assert len(config_paths) == 1
            cfg = json.loads(config_paths[0].read_text())
            validate_finished_scope(cfg)
            if arm == 'baseline':
                from scripts.eval_shape_refresh import validate_baseline
                validate_baseline(config_paths[0])
                expected_shards=len(cfg['corruptions'])*((cfg['evaluation_n_examples']+31)//32)
                assert len(list(config_paths[0].parent.glob('batches/*/*/predictions.npz'))) == expected_shards
                receipt=dict(config_path=str(config_paths[0]),config_identity=tools.identity(config_paths[0]))
                with (cache/'common'/'baseline_complete.json').open('x') as receipt_file:
                    json.dump(receipt,receipt_file,indent=2)
            status['completed_arms'].append(arm)
            (output/'phase_status.json').write_text(json.dumps(status,indent=2)+'\n')
        acceptance = analyze(output,cache)
        assert acceptance['status'] == 'passed'
        status['status'] = 'complete'
        ledger['completed'].append(stage)
        if stage == 'smoke': ledger['smoke_acceptance'] = 'passed'
    except BaseException:
        status.update(status='failed',failure=traceback.format_exc())
        ledger['failed'] = dict(stage=stage,path=str(output))
        raise
    finally:
        status['wall_seconds_before_archive'] = time.monotonic()-start
        (output/'phase_status.json').write_text(json.dumps(status,indent=2)+'\n')
        # Successful and failed phase bytes both survive on Drive.
        sealed = tools.seal(output,DRIVE/'phase-evidence'/(output.name+'.zip'))
        ledger['spent_seconds'] = max(original_spent+time.monotonic()-start,
            (datetime.now(timezone.utc)-datetime.fromisoformat(ledger['research_started_utc'])).total_seconds())
        ledger.pop('running',None)
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        receipt = dict(status=status['status'],stage=stage,archive=sealed,
                       ledger=ledger,output=str(output),cache_root=str(cache))
        (ROOT/(stage+'_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        target = DRIVE/'phase-evidence'/(output.name+'.receipt.json')
        target.write_text(json.dumps(receipt,indent=2)+'\n')
        print('REFRESH_PHASE_RECEIPT',json.dumps(receipt),flush=True)
    print('REFRESH_'+stage.upper()+'_COMPLETE',flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
