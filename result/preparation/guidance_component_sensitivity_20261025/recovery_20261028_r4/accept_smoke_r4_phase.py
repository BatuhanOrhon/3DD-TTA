from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,shutil,zipfile,sys
root=Path('/content/studies/guidance-components');package=Path('/content/agent_b_r4_smoke_local_acceptance.zip')
def identity(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert identity(package)=={'bytes': 37355, 'sha256': 'f285c4e5e32c5d5b4e910a0cdbbdc1d1c9c1d85e18272bb03e85d0eb6041c74e'}
path=root/'sensitivity_phases/sensitivity_campaign_controller.py'
assert identity(path)['sha256']=='0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc'
spec=importlib.util.spec_from_file_location('smoke_acceptance_control',path);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sha='1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
plan,_,_=c.plan_verified(root/'sensitivity_campaign_plan.json',sha);ledger=c.load_ledger(sha)
assert ledger['active'] is None
with zipfile.ZipFile(package) as z:
 assert z.testzip() is None
 accepted_raw=z.read('accepted_sensitivity_smoke.json');validation_raw=z.read('validation.json');projection_raw=z.read('post_smoke_budget_projection.json')
 accepted=json.loads(accepted_raw);validation=json.loads(validation_raw);projection=json.loads(projection_raw)
 assert accepted['status']==validation['status']=='accepted' and accepted['phase_plan_sha256']==sha==projection['phase_plan_sha256']
 assert accepted['physical_cache_complete'] and validation['physical_cache_complete'] and validation['calibration_physical_cache_complete']
 assert accepted['local_validation']==dict(bytes=len(validation_raw),sha256=hashlib.sha256(validation_raw).hexdigest())
 assert validation['raw_archives']==56 and validation['jobs']==14 and validation['physically_verified_cache_files']==4
 assert projection['upper_seconds']<=19800 and projection['spent_seconds']==ledger['spent_seconds'] and projection['cap_seconds']==19800
 c.verify(accepted['archive']['drive_path'],accepted['archive']);c.campaign_budget_verified(plan,'development',ledger['spent_seconds'])
 target=root/'accepted_sensitivity_smoke.json';assert not target.exists()
 drive=c.DRIVE/'phase-evidence'/('sensitivity_smoke_local_acceptance_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));drive.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(package,drive/package.name);assert identity(package)==identity(drive/package.name)
 target.write_bytes(accepted_raw);(root/'accepted_sensitivity_smoke_validation.json').write_bytes(validation_raw);(root/'post_smoke_budget_projection.json').write_bytes(projection_raw)
 c.accepted_stage(target,sha)
 receipt=dict(status='accepted',phase_plan_sha256=sha,archive=dict(identity(package),drive_path=str(drive/package.name)),raw_phase=accepted['archive'],upper_campaign_seconds=projection['upper_seconds'])
 (root/'sensitivity_smoke_local_acceptance_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(drive/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('AGENT_B_SENSITIVITY_SMOKE_LOCAL_ACCEPTANCE',json.dumps(receipt),flush=True)
print('AGENT_B_SENSITIVITY_SMOKE_ACCEPTANCE_COMPLETE',flush=True)
