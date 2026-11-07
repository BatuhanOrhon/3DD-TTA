"""Immutable raw acceptance and independent PM recount; analysis only."""
import csv,hashlib,io,json,stat,sys,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
archive,expected,out=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
out.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 names=z.namelist();assert len(names)==len(set(names)) and len(names)<20000
 assert sum(i.file_size for i in z.infolist())<4*1024**3
 for i in z.infolist():
  p=PurePosixPath(i.filename)
  assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename
  assert stat.S_IFMT(i.external_attr>>16) in (0,stat.S_IFREG,stat.S_IFDIR)
 manifest=json.loads(z.read('evidence_manifest.json'))
 assert set(names)==set(manifest)|{'evidence_manifest.json'}
 for name,r in manifest.items():
  data=z.read(name);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],name
 exit_record=json.loads(z.read('exit.json'))
 if exit_record['returncode']:
  report=dict(status='accepted_failed_attempt',exit=exit_record,failure=json.loads(z.read('failure.json')))
 else:
  required=['command.txt','config.json','environment.txt','stdout.log','summary.csv','per_corruption.csv','notes.md','per_object.csv','summary.json']
  assert all(n in names for n in required)
  config=json.loads(z.read('config.json'));summary=json.loads(z.read('summary.json'))
  assert summary['status']=='complete'
  rows=list(csv.DictReader(io.StringIO(z.read('per_object.csv').decode())))
  parsed={};recount={}
  for n in names:
   if n.count('/')!=1 or not n.endswith('.npz') or n.endswith(('_repair.npz','frame.npz','calibration_first_step.npz')):continue
   with np.load(io.BytesIO(z.read(n)),allow_pickle=False) as d:
    if 'logits' not in d.files:continue
    logits=d['logits'].astype(np.float64);assert logits.shape==(2,4,40) and np.isfinite(logits).all()
    exp=np.exp(logits-logits.max(-1,keepdims=True));pm=(exp/exp.sum(-1,keepdims=True)).mean(1)
    pred=pm.argmax(-1);assert np.array_equal(pred,d['prediction'])
    assert np.allclose(pm,d['probability_mean'],rtol=1e-5,atol=1e-7)
    case,arm=n.split('/');arm=arm[:-4];corruption=case.rsplit('_',1)[0]
    for idx,label,prediction in zip(d['index'],d['label'],pred):
     key=(arm,corruption,int(idx));assert key not in recount
     recount[key]=int(label==prediction)
    parsed[(case,arm)]={k:d[k].copy() for k in ('logits','latents','decoded')}
  assert len(recount)==len(rows)
  for r in rows:assert recount[(r['arm'],r['corruption'],int(r['object_id']))]==int(r['correct'])
  results=[];comparisons=[]
  arms=sorted({k[0] for k in recount})
  for arm in arms:
   own={k[1:]:v for k,v in recount.items() if k[0]==arm}
   results.append(dict(arm=arm,n=len(own),correct=sum(own.values()),accuracy_percent=100*np.mean(list(own.values()))))
   if arm=='full' or 'full' not in arms:continue
   for control in (['full','semantic_zero'] if arm.startswith(('causal','fixed')) else ['full']):
    baseline={k[1:]:v for k,v in recount.items() if k[0]==control};assert own.keys()==baseline.keys()
    identities=sorted({k[1] for k in own});corruptions=sorted({k[0] for k in own})
    delta=np.array([[own[(c,i)]-baseline[(c,i)] for c in corruptions] for i in identities],dtype=float)
    rng=np.random.default_rng(20261107)
    bootstrap=delta[rng.integers(len(identities),size=(10000,len(identities)))].mean((1,2))*100
    comparisons.append(dict(arm=arm,control=control,delta_pp=100*delta.mean(),cluster_object_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
      wins=int((delta==1).sum()),losses=int((delta==-1).sum()),per_corruption_pp={c:100*delta[:,j].mean() for j,c in enumerate(corruptions)}))
  technical=json.loads(z.read('technical.json')) if 'technical.json' in names else None
  repair_count=0
  for name in names:
   if not name.endswith('_repair.npz'):continue
   case,arm=name.split('/');arm=arm.removesuffix('_repair.npz')
   detail=json.loads(z.read(case+'/'+arm+'.json'))['details']
   with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as d:
    accepted=np.array([r['accepted'] for r in detail],dtype=bool).T
    assert np.allclose(d['chosen_latent'],np.where(accepted[:,:,None,None],d['attempted_latent'],d['latent']),rtol=0,atol=0)
    assert np.allclose(d['chosen_decoded'],np.where(accepted[:,:,None,None],d['after'],d['before']),rtol=0,atol=0)
    if arm!='causal_all':assert np.array_equal(d['latent'][...,:3],d['attempted_latent'][...,:3])
    ratio=np.sqrt(np.mean((d['after'].astype(np.float64)-d['before'])**2,axis=(2,3)))/np.maximum(np.sqrt(np.mean(d['before'].astype(np.float64)**2,axis=(2,3))),1e-6)
    assert np.allclose(ratio,np.array([r['geometry_ratio'] for r in detail]).T,rtol=1e-5,atol=1e-8)
    chosen=parsed[(case,arm)]['logits'].astype(np.float64)
    for i,r in enumerate(detail):
     if i==0:continue
     if r['teacher_source']=='past_corrected_predictions':
      e=np.exp(chosen[:,:i]-chosen[:,:i].max(-1,keepdims=True));teacher=(e/e.sum(-1,keepdims=True)).mean(1)
      assert np.allclose(teacher,r['teacher'],rtol=1e-5,atol=1e-7)
      assert r['teacher_count']==i
     step=d['attempted_latent'][:,i].astype(np.float64)-d['latent'][:,i]
     cap=r['fraction']*np.maximum(np.sqrt(np.mean(d['latent'][:,i,:,3].astype(np.float64)**2,axis=1)),1e-6)*step.shape[1]**.5
     assert np.all(np.linalg.norm(step,axis=(1,2))<=cap*(1+1e-4)+1e-7)
    repair_count+=1
  if technical is not None:
   selection=json.loads(z.read('selection.json'))
   for row in technical['parity']:
    start=selection['calibration'].index(row['ids'][0]);case=row['corruption']+'_'+str(start//2)
    for key,a,b in [('original_repeat','original','original_repeat'),('cross','original','full'),('experimental_repeat','full','full_repeat'),('semantic_repeat','semantic_zero','semantic_zero_repeat')]:
     for field in ('latents','logits'):
      value=float(np.sqrt(np.mean((parsed[(case,a)][field].astype(np.float64)-parsed[(case,b)][field])**2)))
      assert np.isclose(value,row[key][field],rtol=1e-10,atol=1e-12)
   assert technical['passed']==all(r['passed'] for r in technical['parity'])
   assert technical['semantic_passed']==all(r['semantic_passed'] for r in technical['parity'])
  selection=json.loads(z.read('selection.json'))
  for row in summary.get('pilot_native',[]):
   case=row['corruption']+'_'+str(selection['pilot'].index(row['ids'][0])//2);control=row['control']
   for field in ('latents','logits'):
    value=float(np.sqrt(np.mean((parsed[(case,control)][field].astype(np.float64)-parsed[(case,control+'_repeat')][field])**2)))
    assert np.isclose(value,row['repeat'][field],rtol=1e-10,atol=1e-12)
   passed=row['repeat']['latents']<=1e-5 and row['repeat']['logits']<=1e-4
   if row['within']:
    for arm,record in zip(('semantic_zero','semantic_zero_repeat'),row['within']):
     with np.load(io.BytesIO(z.read(case+'/'+arm+'_repair.npz')),allow_pickle=False) as d:
      for key,a,b in (('decoded','before','after'),('latent','latent','attempted_latent'),('logits','before_logits','after_logits')):
       value=float(np.sqrt(np.mean((d[a].astype(np.float64)-d[b])**2)))
       assert np.isclose(value,record[key],rtol=1e-10,atol=1e-12)
    passed=passed and all(d['latent']==0 and d['decoded']<=1e-5 and d['logits']<=1e-4 for d in row['within'])
   assert row['passed']==passed
  if summary.get('pilot_native'):assert summary['pilot_native_passed']==all(r['passed'] for r in summary['pilot_native'])
  report=dict(status='accepted',phase=config['phase'],results=results,comparisons=comparisons,technical=technical,
   elapsed=summary['elapsed'],pilot_native=summary.get('pilot_native'),pilot_native_passed=summary.get('pilot_native_passed'),repair_witnesses_verified=repair_count,scope=summary['scope'],timing='cold fixed-order B2 pilot; no speed inference')
 report['raw_archive']=str(archive);report['raw_sha256']=expected;report['manifest_files']=len(manifest)
(out/'acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(out/'analysis_source.py').write_text(Path(__file__).read_text())
print(json.dumps(report,indent=2,allow_nan=False))
