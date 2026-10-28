from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,zipfile
root=Path('/content/studies/guidance-components');phases=root/'phases';phases.mkdir(parents=True,exist_ok=True)
archive=Path('/content/agent_b_pinned_legacy_phase_sources.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='d92756f3ac46ac715ecc20dc79ebd02a393ed90796357b03a321756810675332'
expected={'deploy_assets_template.py': '89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0', 'preflight_components_cpu.py': '94e3ae0f460335d302c95f548b85e00743eb4bd4ea8bb061d7a8ebc18a354356', 'gpu_components_template.py': '1b7410a33189481978df630c1de12278a6564f7d3a665ddae6658b48d02491ac'}
with zipfile.ZipFile(archive) as z:
 assert set(z.namelist())==set(expected) and len(z.namelist())==3 and z.testzip() is None
 for name,sha in expected.items():
  raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==sha
  target=phases/name
  if target.exists():assert target.read_bytes()==raw
  else:
   with target.open('xb') as f:f.write(raw)
command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u','-c',"import sys;sys.path.insert(0,'/content/studies/guidance-components/phases');import deploy_assets_template as d;d.SESSION='tta-guidance-components-a100-20261028-r4';d.main()"]
logs=root/'phase_command_logs';logs.mkdir(exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
with (logs/(stamp+'_r4_deploy.log')).open('x') as log:
 with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')) as process:
  for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
  assert process.wait()==0,'Legacy setup phase failed; preserve evidence before reviewed recovery'
print('AGENT_B_R4_LEGACY_DEPLOY_COMPLETE',flush=True)
