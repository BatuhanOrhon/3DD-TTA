import hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
root=Path('/content/studies/gsd-lowband-full-20261107');archive=Path('/content/full_operator_v1.zip')
expected={'bytes': 36034, 'sha256': '344cfb5169a00e4adce58e0dfd49b155e3ea221bc84e17da5ca45bb9e4d74042'}
assert archive.stat().st_size==expected['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==expected['sha256']
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 manifest=json.loads(z.read('operator_manifest.json'))
 assert set(z.namelist())==set(manifest)|{'operator_manifest.json'}
 for n,v in manifest.items():
  assert Path(n).name==n and n.endswith(('.py','.json'))
  raw=z.read(n);assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
  target=root/'operator'/n
  if target.exists():assert target.read_bytes()==raw,'Existing operator mismatch: '+n
  else:
   with target.open('xb') as f:f.write(raw)
backup=Path('/content/drive/MyDrive/thesis/result/gsd_lowband_full_20261107/operator/operator_v1.zip')
with archive.open('rb') as a,backup.open('xb') as b:shutil.copyfileobj(a,b)
assert hashlib.sha256(backup.read_bytes()).hexdigest()==expected['sha256']
command=['/usr/local/envs/3dd_tta_env/bin/python','-u',str(root/'operator/campaign.py'),'--stage','full','--budget-hours','48','--ack-timeout','3600']
print('LOWBAND_FULL_MAIN_LAUNCH',json.dumps(dict(command=command,operator=expected)),flush=True)
with subprocess.Popen(command,cwd=str(root),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
 for line in child.stdout:print(line,end='',flush=True)
 assert child.wait()==0,'Full campaign failed; preserve all outputs'
complete=json.loads((root/'campaign_complete.json').read_text());assert complete['status']=='complete' and (complete['inference_jobs'],complete['cache_jobs'])==(117,39)
print('LOWBAND_FULL_MAIN_PHASE_COMPLETE',flush=True)
