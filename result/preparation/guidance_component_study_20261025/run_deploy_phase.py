from datetime import datetime,timezone
import hashlib,os,subprocess
from pathlib import Path
phases=Path('/content/studies/guidance-components/phases')
expected={'deploy_assets_template.py': '89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0', 'preflight_components_cpu.py': '94e3ae0f460335d302c95f548b85e00743eb4bd4ea8bb061d7a8ebc18a354356', 'gpu_components_template.py': '1b7410a33189481978df630c1de12278a6564f7d3a665ddae6658b48d02491ac'}
for name,sha in expected.items():
    assert hashlib.sha256((phases/name).read_bytes()).hexdigest()==sha, 'Phase source mismatch: '+name
logs=Path('/content/studies/guidance-components/phase_command_logs');logs.mkdir(parents=True,exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u',str(phases/'deploy_assets_template.py'),*[]]
with (logs/(stamp+'_deploy.log')).open('x') as log:
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')) as process:
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        assert process.wait()==0, 'Phase failed; preserve all evidence before retry'
print('AGENT_B_DEPLOY_WRAPPER_COMPLETE',flush=True)
