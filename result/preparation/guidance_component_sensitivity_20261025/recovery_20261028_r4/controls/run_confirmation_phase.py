from datetime import datetime,timezone
import hashlib,os,subprocess
from pathlib import Path
root=Path('/content/studies/guidance-components')
plan=root/'sensitivity_campaign_plan.json'
assert hashlib.sha256(plan.read_bytes()).hexdigest()=='1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u',str(root/'sensitivity_phases/sensitivity_campaign_controller.py'),'--phase-plan',str(plan),'--phase-plan-sha256','1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42','--stage','confirmation']
logs=root/'phase_command_logs';logs.mkdir(exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
with (logs/(stamp+'_sensitivity_confirmation.log')).open('x') as log:
 with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')) as process:
  for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
  assert process.wait()==0,'Sensitivity stage failed; preserve all evidence and campaign ledger before further work'
print('AGENT_B_SENSITIVITY_CONFIRMATION_WRAPPER_COMPLETE',flush=True)
