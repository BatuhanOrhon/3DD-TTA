"""CPU-only physical smoke-cache retrieval; exact immutable NPZ bytes."""
from pathlib import Path
import importlib.util,hashlib,json,sys
root=Path('/content/studies/guidance-components');phases=root/'phases'
assert hashlib.sha256((phases/'deploy_assets_template.py').read_bytes()).hexdigest()=='89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'
sys.path.insert(0,str(phases))
from deploy_assets_template import DRIVE,identity,verify,copy_checked,seal,utc
path=root/'sensitivity_phases/sensitivity_campaign_controller.py'
assert identity(path)['sha256']=='0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc'
spec=importlib.util.spec_from_file_location('smoke_cache_control',path);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sha='1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
plan,_,_=c.plan_verified(root/'sensitivity_campaign_plan.json',sha)
ledger=c.load_ledger(sha);assert ledger['active'] is None
receipt=json.loads((root/'sensitivity_smoke_receipt.json').read_text());validation=json.loads((root/'sensitivity_smoke_validation.json').read_text())
assert receipt['state']['status']=='complete' and validation['status']=='validated' and validation['phase_plan_sha256']==sha
assert len(receipt['state']['jobs'])==14 and all(job['status']=='complete' for job in receipt['state']['jobs'])
verify(receipt['archive']['drive_path'],receipt['archive'])
records=receipt['state']['cache_records'];assert len(records)==4 and len({r['local_path'] for r in records})==4
namespace='paired_sensitivity_smoke_v2_seed0';output=root/('sensitivity_smoke_cache_'+utc());output.mkdir(exist_ok=False)
for row in records:
 source=c.owned(row['local_path']);c.owned(row['drive_path'],DRIVE)
 relative=source.relative_to(root/namespace)
 assert source.suffix=='.npz' and relative.parts[0] in c.DATASETS and relative.parts[1] in ('gaussian','background')
 assert Path(row['drive_path']).relative_to(DRIVE/namespace)==relative
 verify(row['drive_path'],row);copy_checked(source,output/namespace/relative,row)
(output/'cache_records.json').write_text(json.dumps(records,indent=2)+'\n')
(output/'smoke_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
archive=seal(output,DRIVE/'phase-evidence'/output.name/'smoke_cache_evidence.zip')
result=dict(status='verified',session=c.SESSION,phase_plan_sha256=sha,cache_count=4,archive=archive)
(root/'sensitivity_smoke_cache_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
(Path(archive['drive_path']).parent/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print('AGENT_B_SENSITIVITY_SMOKE_CACHE_RECEIPT',json.dumps(archive),flush=True)
print('AGENT_B_SENSITIVITY_SMOKE_CACHE_COMPLETE',flush=True)
