import hashlib,json,os,subprocess,sys
from pathlib import Path
old=Path('/content/studies/guidance-components/phases')
assert hashlib.sha256((old/'deploy_assets_template.py').read_bytes()).hexdigest()=='89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'
sys.path.insert(0,str(old))
from deploy_assets_template import ROOT,DRIVE,verify,extract_checked,copy_checked,identity,utc
archive=Path('/content/agent_b_confirmation_phase_sources.zip')
verify(archive,{'bytes': 26413, 'sha256': 'ad48469bb93fe2ef01da56ee93e32425b014b519aad5352adbb1895935c0cac3'})
phases=ROOT/'confirmation_phases'
assert not phases.exists()
extract_checked(archive,phases)
manifest=phases/'confirmation_phase_source_identities.json'
assert identity(manifest)['sha256']=='b95170f32c73967dd97b55057e57e9107d037b0ca8b8bd7a2fdcd39ed2337229'
for name,row in json.loads(manifest.read_text()).items():verify(phases/name,row)
assert (phases/'frozen_confirmation_plan.json').read_bytes()==(ROOT/'frozen_confirmation_plan.json').read_bytes()
command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u',str(phases/'frozen_confirmation_phase.py'),'--self-check']
with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')) as process:
 for line in process.stdout:print(line,end='',flush=True)
 assert process.wait()==0
target=DRIVE/'phase-evidence'/('confirmation_sources_'+utc())/archive.name
copy_checked(archive,target,{'bytes': 26413, 'sha256': 'ad48469bb93fe2ef01da56ee93e32425b014b519aad5352adbb1895935c0cac3'})
receipt=dict(status='verified',archive=dict(identity(archive),drive_path=str(target)),source_manifest_sha256=identity(manifest)['sha256'])
(ROOT/'confirmation_sources_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(target.parent/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('AGENT_B_CONFIRMATION_SOURCES_DEPLOYED',flush=True)
