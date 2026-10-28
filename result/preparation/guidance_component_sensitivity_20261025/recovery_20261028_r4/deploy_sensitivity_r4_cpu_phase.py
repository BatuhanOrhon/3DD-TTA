import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/content/studies/guidance-components')
old=root/'phases'
assert hashlib.sha256((old/'deploy_assets_template.py').read_bytes()).hexdigest()=='89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0'
sys.path.insert(0,str(old))
from deploy_assets_template import identity,verify,extract_checked,seal,utc,copy_checked
expected={'bytes': 5500929, 'sha256': '75464d84a1adf84e432ce5711feed242a345f2f1e198dc822104951acce90b3e'}
archive=Path('/content/agent_b_sensitivity_v2_delta.zip');verify(archive,expected)
prepared=root/'prepared_sensitivity_v2';assets=prepared/'assets'
assert not prepared.exists()
base=root/'prepared_v1/assets'
assert identity(base/'manifest.json')['sha256']=='7012c30ad88222bbc5435d8508f781fc33385ed3976951c45f48da67f625ed75'
assert not any(p.is_symlink() for p in base.rglob('*'))
needed=sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
assert shutil.disk_usage(root).free>needed+2*1024**3
prepared.mkdir();shutil.copytree(base,assets)
extract_checked(archive,prepared,overlay=True)
assert identity(assets/'manifest.json')['sha256']=='599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef'
m=json.loads((assets/'manifest.json').read_text())
for name,row in m['files'].items():
 rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
 verify(assets/rel,row)
phases=root/'sensitivity_phases';phases.mkdir(exist_ok=True)
pref=Path('/content/agent_b_preflight_sensitivity_cpu.py')
assert identity(pref)['sha256']=='3134c501f263c1010d67f5f5eeeb921450ef1e0b0e937fc49c43af104acf8c07'
copy_checked(pref,phases/'preflight_sensitivity_cpu.py',identity(pref))
output=root/('sensitivity_deployment_'+utc());output.mkdir()
shutil.copyfile(assets/'manifest.json',output/'manifest.json');shutil.copyfile(pref,output/pref.name)
state=dict(status='complete',session='tta-guidance-components-a100-20261028-r4',manifest_sha256=identity(assets/'manifest.json')['sha256'],source_snapshot_sha256=m['source_snapshot_sha256'],verified_files=len(m['files']),research_gpu_launched=False)
(output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
(target:=Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025/phase-evidence')/output.name).mkdir(parents=True,exist_ok=False)
receipt=dict(state=state,archive=seal(output,target/'sensitivity_deployment_evidence.zip'))
(root/'sensitivity_deployment_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(target/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(prepared/'.deployed').write_text(state['manifest_sha256']+'\n')
print('AGENT_B_SENSITIVITY_ASSET_FILES_VERIFIED',state['verified_files'],flush=True)
command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u',str(phases/'preflight_sensitivity_cpu.py')]
with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')) as process:
 for line in process.stdout:print(line,end='',flush=True)
 assert process.wait()==0,'CPU preflight failed; preserve evidence and inspect'
assert (prepared/'.complete').read_text().strip()==state['manifest_sha256']
print('AGENT_B_SENSITIVITY_CPU_SETUP_COMPLETE',flush=True)
