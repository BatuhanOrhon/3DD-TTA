from datetime import datetime,timezone
import hashlib,os,subprocess
from pathlib import Path
phases=Path('/content/studies/guidance-components/confirmation_phases')
manifest=phases/'confirmation_phase_source_identities.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='b95170f32c73967dd97b55057e57e9107d037b0ca8b8bd7a2fdcd39ed2337229'
logs=Path('/content/studies/guidance-components/phase_command_logs');logs.mkdir(parents=True,exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u',str(phases/'frozen_confirmation_phase.py'),'--source-manifest-sha256','b95170f32c73967dd97b55057e57e9107d037b0ca8b8bd7a2fdcd39ed2337229']
with (logs/(stamp+'_frozen_confirmation.log')).open('x') as log:
 with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')) as process:
  for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
  assert process.wait()==0,'Frozen confirmation failed; preserve all evidence and ledger before any further work'
print('AGENT_B_FROZEN_CONFIRMATION_WRAPPER_COMPLETE',flush=True)
