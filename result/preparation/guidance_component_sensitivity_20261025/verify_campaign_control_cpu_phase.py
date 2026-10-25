import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/content/studies/guidance-components')
plan=root/'sensitivity_campaign_plan.json'
assert hashlib.sha256(plan.read_bytes()).hexdigest()=='4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1'
p=json.loads(plan.read_text());phases=root/'sensitivity_phases'
for name,row in p['phase_sources'].items():
 b=(phases/name).read_bytes();assert row==dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
assert not (root/'sensitivity_gpu_approval.json').exists(),'Unexpected approval file; inspect provenance'
for name in ['sensitivity_campaign_controller.py','calibrate_spectral_losses.py']:
 command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u',str(phases/name),'--self-check']
 with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')) as process:
  for line in process.stdout:print(line,end='',flush=True)
  assert process.wait()==0
import sys
sys.path.insert(0,str(root/'phases'))
from deploy_assets_template import identity,verify,copy_checked,utc,DRIVE
package=Path('/content/agent_b_sensitivity_readiness_package.zip')
verify(package,{'bytes': 5570596, 'sha256': 'e2af93ccaca4a6415489fb92a19ded2253ca916293fbffbb7e322e64568b7741'})
target=DRIVE/'phase-evidence'/('sensitivity_CPU_readiness_'+utc())/package.name
copy_checked(package,target,{'bytes': 5570596, 'sha256': 'e2af93ccaca4a6415489fb92a19ded2253ca916293fbffbb7e322e64568b7741'})
receipt=dict(status='CPU_ready_GPU_approval_pending',phase_plan_sha256='4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1',archive=dict(identity(package),drive_path=str(target)),research_gpu_launched=False)
(root/'sensitivity_control_readiness_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(target.parent/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('AGENT_B_SENSITIVITY_CONTROL_CPU_READY',flush=True)
