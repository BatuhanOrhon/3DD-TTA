"""Independent CPU acceptance and within-observer versus between-observer repeats."""
import hashlib,importlib.util,itertools,json,sys,zipfile,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[2];sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'knowledge/evidence/decoded_physical_accuracy_20261104'))
import analyze_accuracy as a
spec=importlib.util.spec_from_file_location('safety',REPO/'result/preparation/gsd_physical_accuracy_20261104/operator_v1/deploy_assets_template.py');safe=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe)
receipt=json.loads((R/'archive_receipt.json').read_text());archive=R/'zero_repeat_raw.zip';assert a.identity(archive)=={k:receipt[k] for k in ('bytes','sha256')}
with zipfile.ZipFile(archive) as z:
 safe.checked_members(z);manifest=json.loads(z.read('package_manifest.json'));assert set(z.namelist())==set(manifest)|{'package_manifest.json'}
 for n,expected in manifest.items():
  data=z.read(n);assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==expected
D=R/'downloaded';safe.extract_checked(archive,D)
research=json.loads((D/'metadata/research_receipt.json').read_text());assert research['status']=='complete' and len(research['records'])==12 and research['elapsed_seconds']<900
loaded={}
for record in research['records']:
 p=D/'raw_runs'/Path(record['archives'][0]['path']).name;assert a.identity(p)=={k:record['archives'][0][k] for k in ('bytes','sha256')}
 canonical=REPO/'result/modelnet40_c/decoded_physical_spectral_diagnostic'/p.name
 with p.open('rb') as incoming,canonical.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing)
 assert a.identity(canonical)==a.identity(p)
 run=a.load_run(canonical);c=run['corruptions'][0];assert run['ids']==({'gaussian':[44,91],'background':[495,512]}[c]) and len(run['cases'])==1
 bundle=a.open_bundle(p)
 try:
  run['first_probe']=bundle.json('batches/'+c+'/batch0000/diagnostics.json')['trajectory']['probes'][0]
  cfg=bundle.json('config.json')
  for entry in list(cfg['native_extensions'].values())+list(cfg['native_bootstrap']['bindings'].values()):assert a.identity(D/'native'/Path(entry['path']).name)=={k:entry[k] for k in ('bytes','sha256')}
 finally:bundle.close()
 loaded[record['stage']]=run
reports={}
for c in ('gaussian','background'):
 groups={arm:[loaded[c+'-'+arm+'-'+str(i)] for i in range(3)] for arm in ('control','live_control')}
 result={}
 for name,pairs in [('control_repeat',itertools.combinations(groups['control'],2)),('live_repeat',itertools.combinations(groups['live_control'],2)),('control_vs_live',itertools.product(groups['control'],groups['live_control']))]:
  rows=[]
  for left,right in pairs:
   gate=a.repeat_gate(left,right);lc=next(iter(left['cases'].values()));rc=next(iter(right['cases'].values()))
   rows.append(dict(left=left['archive_identity'],right=right['archive_identity'],**gate,first_clean_latent_hash_equal=left['first_probe']['clean_latent_sha256']==right['first_probe']['clean_latent_sha256'],decoder_style_hash_equal=left['first_probe']['style_sha256']==right['first_probe']['style_sha256']))
  result[name]=dict(passed=sum(v['passed'] for v in rows),total=len(rows),worst_logit=max(v['logit_max_absolute'] for v in rows),worst_rms=max(v['logit_per_object_checkpoint_rms_max'] for v in rows),pairs=rows)
 reports[c]=result
report=dict(status='accepted',raw=receipt,source='cdbdfb1995296b08bcc4345161af6c01213f4b2f',groups=reports,limitation='Two deliberately selected B2 cases/three repeats per observer; no accuracy improvement or unique causal kernel claim. Pair comparisons share runs and are not independent trials.')
with (R/'analysis.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({c:{k:{x:v[x] for x in ('passed','total','worst_logit','worst_rms')} for k,v in r.items()} for c,r in reports.items()},indent=2))
