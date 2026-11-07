"""Download and validate completed owned job archives, then acknowledge safe duplicate cleanup."""
import argparse,fcntl,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,tempfile,time,uuid,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];sys.path.insert(0,str(REPO))
from scripts.analyze_scan_trajectory import Bundle
spec=importlib.util.spec_from_file_location('full_accept',REPO/'knowledge/evidence/gsd_lowband_halfmix_20261107/analyze.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
spec=importlib.util.spec_from_file_location('archive_safety',REPO/'knowledge/evidence/chamfer_guidance_executed_code_20261031/historical_controls/remote/deploy_assets_template.py')
safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)
REMOTE=Path('/content/studies/gsd-lowband-full-20261107');analysis.REMOTE=REMOTE
p=argparse.ArgumentParser();p.add_argument('--monitor-log',type=Path,required=True);p.add_argument('--local-root',type=Path,required=True);p.add_argument('--session',required=True);args=p.parse_args()
ROOT=args.local_root;ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'accepted').mkdir(exist_ok=True);(ROOT/'failed_transfers').mkdir(exist_ok=True)
lock=(ROOT/'transfer_daemon.lock').open('a+')
fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((ROOT/'operator/plan.json').read_text());shards={s['label']:s for s in plan['shards']};PYTHON=sys.executable

def identity(path):
 p=Path(path);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def atomic_save(path,value):
 path=Path(path);temporary=path.with_name(path.name+'.pending.'+uuid.uuid4().hex)
 with temporary.open('x') as f:
  json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,path)
 directory=os.open(str(path.parent),os.O_RDONLY)
 try:os.fsync(directory)
 finally:os.close(directory)
def cli(*command):
 r=subprocess.run(['colab',*command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
 import re
 output=re.sub(r'(colab-runtime-proxy-token=)[^&\s\x1b]+',r'\1[REDACTED]',r.stdout)
 if r.returncode:raise RuntimeError(output)
 return output
def accept_cache(archive,label,shard):
 with zipfile.ZipFile(archive) as z:
  safety.checked_members(z);names=z.namelist()
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n for n in names)
  manifest=json.loads(z.read('package_manifest.json'));assert set(names)==set(manifest)|{'package_manifest.json'}
  assert len(names)<=1000 and sum(e.file_size for e in z.infolist())<1024**3
  for e in z.infolist():assert (__import__('stat').S_IFMT(e.external_attr>>16) in (0,__import__('stat').S_IFREG,__import__('stat').S_IFDIR))
  for name,expected in manifest.items():
   raw=z.read(name);assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==expected
  report=json.loads(z.read('records/cache_preparation_'+shard['label']+'.json'));assert report['status']=='complete' and len(report['records'])==15*shard['count']//2
  ids=json.loads((ROOT/'operator'/shard['selection_file']).read_text())['indices']
  assert report['selection']['indices']==ids
  expected={(c,tuple(ids[i:i+2])) for c in plan['corruptions'] for i in range(0,len(ids),2)}
  assert {(r['corruption'],tuple(r['ids'])) for r in report['records']}==expected
  assert {r['member'] for r in report['records']}|{'records/cache_preparation_'+shard['label']+'.json'}==set(manifest)
  for row in report['records']:
   assert manifest[row['member']]=={k:row[k] for k in ('bytes','sha256')}
   with analysis.np.load(__import__('io').BytesIO(z.read(row['member'])),allow_pickle=False) as saved:
    metadata=json.loads(str(saved['metadata'].item()))
    assert metadata['rng_after_sha256']==row['rng_after']
    for key in ('shape_latent','local_latent','noise'):assert analysis.ar.tensor_identity(saved[key])==metadata['tensor_hashes'][key]
  for name in manifest:
   target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():
    assert identity(target)==manifest[name],'Preserve mismatched existing cache; no overwrite'
   else:
    temporary=target.with_name(target.name+'.pending.'+uuid.uuid4().hex)
    with temporary.open('xb') as f:f.write(z.read(name));f.flush();os.fsync(f.fileno())
    os.replace(temporary,target)
 return dict(cache_count=len(report['records']),selection_ids=report['selection']['indices'])
def accept_inference(archive,label,shard,arm):
 result=analysis.load_run(archive,ROOT);c=result['config'];ids=json.loads((ROOT/'operator'/shard['selection_file']).read_text())['indices']
 assert c['benchmark_shard'] and c['count']==shard['count'] and c['selected_original_ids']==ids and c['corruptions']==plan['corruptions']
 assert c['completed_batches']=={x:shard['count']//2 for x in plan['corruptions']} and not c['update_norm_probes']
 assert c['checkpoint_scd_weight']==(1. if arm=='control' else .5)
 assert c['weight']==(0.5*plan['frozen_alpha100'] if arm=='M100' else 0.)
 assert c['arm']==dict(control='control',half0='live_control',M100='physical')[arm]
 assert c['lowband_modes']==(None if arm=='control' else 100) and all(x['passed'] for x in result['projections'])
 common=result['control_identity'];common_path=ROOT/'common_identity.json'
 if common_path.exists():assert json.loads(common_path.read_text())==common
 else:
  with common_path.open('x') as f:json.dump(common,f,indent=2)
 cases=[]
 for key,case in result['cases'].items():
  graph=case['physical_metric_identity']
  cases.append(dict(case=key,ids=case['index'].tolist(),labels=case['label'].tolist(),pm=case['pm'].tolist(),terminal=case['terminal'].tolist(),
   paired_identity=case['identity'],filter_sha256=graph['filter_sha256'] if graph else None,rank=[g['actual_rank'] for g in graph['diagnostics']] if graph else None))
 return dict(counts=result['counts'],cases=cases,worker_wall_seconds=c['wall_seconds'],basis_checks=len(result['projections']),basis_passed=True)
def publish_ack(receipt,folder):
 ack=receipt['ack'];remote=str(REMOTE/'acks'/(ack['label']+'.json'))
 local=Path(folder)/('ack_'+uuid.uuid4().hex+'.json');atomic_save(local,ack)
 previous=local.with_name(local.name+'.remote')
 try:
  cli('download','-s',args.session,remote,str(previous))
 except RuntimeError as error:
  if 'File or directory not found' not in str(error) and '404' not in str(error):raise
 else:
  assert json.loads(previous.read_text())==ack,'Existing remote acknowledgement differs; preserve it'
  return
 cli('upload','-s',args.session,str(local),remote)
def prune_local_parts(receipt):
 archive=Path(receipt['local_archive']);assert identity(archive)==receipt['archive_identity']
 parts=archive.parent/'parts';boundary=(ROOT/'transfers'/receipt['label']).resolve()
 assert boundary in parts.resolve().parents and not parts.is_symlink()
 if not parts.exists():return
 expected={Path(x['path']).name:{k:x[k] for k in ('bytes','sha256')} for x in receipt['manifest']['parts']}
 assert {x.name for x in parts.iterdir()}==set(expected)
 for path in parts.iterdir():assert path.is_file() and not path.is_symlink() and identity(path)==expected[path.name]
 atomic_save(archive.parent/'transfer_parts_cleanup.json',dict(archive=receipt['archive_identity'],removed=expected))
 shutil.rmtree(parts)
def check_event(event):
 label=event['label'];kind=event['kind'];assert kind in ('cache','inference') and re.fullmatch(r'shard[0-9]{3}_(cache|control|half0|M100)',label)
 shard_label=label.rsplit('_',1)[0];assert shard_label in shards;shard=shards[shard_label];arm=label.rsplit('_',1)[1]
 assert (kind=='cache' and arm=='cache') or (kind=='inference' and arm in shard['arms'])
 expected=event['archive_identity'];assert type(expected['bytes']) is int and expected['bytes']>0 and re.fullmatch('[0-9a-f]{64}',expected['sha256'])
 return shard,arm
def check_manifest(record,event):
 label=event['label'];check_event(event)
 assert record['label']==label and record['kind']==event['kind'] and {k:record['archive'][k] for k in ('bytes','sha256')}==event['archive_identity']
 remote=Path(record['archive']['path']);assert REMOTE in remote.parents and '..' not in remote.parts
 names=[Path(x['path']).name for x in record['parts']]
 assert len(names)==len(set(names)) and names==['part%05d.bin'%i for i in range(len(names))]
 assert sum(x['bytes'] for x in record['parts'])==record['archive']['bytes']
 for part in record['parts']:
  q=Path(part['path']);assert REMOTE/'transfers'/label in q.parents and '..' not in q.parts
  assert type(part['bytes']) is int and 0<part['bytes']<=32*1024*1024 and re.fullmatch('[0-9a-f]{64}',part['sha256'])
def journal_events():
 events=[]
 with tempfile.TemporaryDirectory(prefix='journal_poll_',dir=str(ROOT)) as temporary:
  directory=Path(temporary);local=directory/'campaign_journal.json'
  cli('download','-s',args.session,str(REMOTE/'campaign_journal.json'),str(local))
  journal=json.loads(local.read_text());assert isinstance(journal['jobs'],dict)
  for label,job in journal['jobs'].items():
   if job['status']!='accepted_pending_ack':continue
   assert re.fullmatch(r'shard[0-9]{3}_(cache|control|half0|M100)',label) and label.rsplit('_',1)[0] in shards
   remote=REMOTE/'transfers'/label/'transfer_manifest.json';local=directory/(label+'.json')
   cli('download','-s',args.session,str(remote),str(local));record=json.loads(local.read_text())
   event=dict(label=label,kind=record['kind'],manifest=str(remote),archive_identity={k:record['archive'][k] for k in ('bytes','sha256')})
   check_manifest(record,event);events.append(event)
 return events,journal.get('status')=='complete'
def process(event):
 label=event['label'];kind=event['kind'];shard,arm=check_event(event)
 accepted=ROOT/'accepted'/(label+'.json')
 if accepted.exists():
  r=json.loads(accepted.read_text());assert r['archive_identity']==event['archive_identity']
  assert identity(r['local_archive'])==event['archive_identity']
  publish_ack(r,accepted.parent)
  if not r.get('ack_uploaded'):
   r.update(status='accepted and cleanup ack uploaded',ack_uploaded=True);atomic_save(accepted,r)
  prune_local_parts(r)
  return
 attempt=ROOT/'transfers'/label/('download_'+time.strftime('%Y%m%dT%H%M%S',time.gmtime()));attempt.mkdir(parents=True,exist_ok=False)
 manifest=attempt/'transfer_manifest.json';cli('download','-s',args.session,event['manifest'],str(manifest));record=json.loads(manifest.read_text())
 check_manifest(record,event)
 command=[PYTHON,str(HERE/'download_bundle.py'),'--session',args.session,str(manifest),str(attempt/'payload')]
 with (attempt/'download.log').open('x') as log:
  r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=7200);assert r.returncode==0
 archive=attempt/'payload'/Path(record['archive']['path']).name;assert identity(archive)==event['archive_identity']
 details=accept_cache(archive,label,shard) if kind=='cache' else accept_inference(archive,label,shard,arm)
 ack=dict(status='local_and_drive_verified',label=label,kind=kind,**event['archive_identity'],local_archive=str(archive.resolve()))
 receipt=dict(status='locally validated; cleanup ack pending',label=label,kind=kind,archive_identity=event['archive_identity'],local_archive=str(archive.resolve()),manifest=record,details=details,ack=ack,ack_uploaded=False)
 assert not accepted.exists();atomic_save(accepted,receipt)
 publish_ack(receipt,attempt)
 receipt.update(status='accepted and cleanup ack uploaded',ack_uploaded=True);atomic_save(accepted,receipt)
 # Transfer pieces duplicate the fully verified immutable ZIP; preserve its bytes and receipts.
 prune_local_parts(receipt)
 print('FULL_LOCAL_ACCEPTED',label,kind,'bytes',ack['bytes'],flush=True)
def process_event(event):
 try:process(event)
 except BaseException:
  import traceback
  path=ROOT/'failed_transfers'/('failure_'+time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'.json')
  path.write_text(json.dumps(dict(event=event,error=traceback.format_exc()),indent=2)+'\n');raise

with args.monitor_log.open() as stream:
 complete=False;next_journal_poll=0.
 while True:
  if not complete and time.monotonic()>=next_journal_poll:
   try:events,journal_complete=journal_events()
   except (RuntimeError,json.JSONDecodeError,subprocess.TimeoutExpired) as error:
    print('FULL_JOURNAL_POLL_RETRY',str(error),flush=True)
   else:
    for event in events:process_event(event)
    complete=journal_complete
   next_journal_poll=time.monotonic()+60
  line=stream.readline()
  if not line:
   if complete:break
   time.sleep(2);continue
  try:outer=json.loads(line)
  except ValueError:continue
  text=outer.get('line','')
  if text.startswith('FULL_TRANSFER_READY '):
   event=json.loads(text.split(' ',1)[1])
   process_event(event)
  if text=='FULL_CAMPAIGN_COMPLETE':complete=True
print('FULL_LOCAL_ALL_TRANSFERS_COMPLETE',flush=True)
