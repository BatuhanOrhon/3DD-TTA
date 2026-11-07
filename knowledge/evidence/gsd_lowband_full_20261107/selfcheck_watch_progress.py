"""Small local receipt/progress fixtures; no real analysis, watcher or remote launch."""
import copy,importlib.util,json,tempfile,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('watch_progress',HERE/'watch_progress.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)

def rejects(action):
 try:action()
 except (AssertionError,ValueError,KeyError,FileNotFoundError,FileExistsError):return
 raise AssertionError('Invalid watcher metadata accepted')

with tempfile.TemporaryDirectory(prefix='full-watch-cpu-') as temporary:
 root=Path(temporary)/'downloaded';(root/'operator').mkdir(parents=True);(root/'accepted').mkdir()
 derived=root.parent/'derived';derived.mkdir();knowledge=root.parent/'progress.md'
 plan=dict(shards=[])
 for ordinal in range(39):
  label='shard%03d'%ordinal;ids=list(range(ordinal*64,min((ordinal+1)*64,2468)))
  selection=label+'.json';(root/'operator'/selection).write_text(json.dumps(dict(indices=ids)))
  plan['shards'].append(dict(label=label,count=len(ids),selection_file=selection,arms=['control','half0','M100']))
 (root/'operator/plan.json').write_text(json.dumps(plan));(root/'common_identity.json').write_text('{"native":"same"}')
 def receipt(ordinal,role,acked=True):
  label='shard%03d_%s'%(ordinal,role);kind='cache' if role=='cache' else 'inference'
  archive=root/(label+'.zip');archive.write_bytes(b'synthetic local archive')
  expected=watch.identity(archive);ack=dict(status='local_and_drive_verified',label=label,kind=kind,local_archive=str(archive),**expected)
  value=dict(label=label,kind=kind,ack_uploaded=acked,archive_identity=expected,local_archive=str(archive),ack=ack,
   status='accepted and cleanup ack uploaded' if acked else 'locally validated; cleanup ack pending')
  path=root/'accepted'/(label+'.json');path.write_text(json.dumps(value));return path
 for role in ('cache','control','half0'):receipt(0,role)
 pending=receipt(0,'M100',False)
 state=watch.receipt_snapshot(root,plan)
 assert state['contiguous_shards']==0 and state['ack_cache_jobs']==1 and state['ack_inference_jobs']==2
 assert state['awaiting_ack']==1
 receipt(0,'M100',True)
 assert watch.receipt_snapshot(root,plan)['contiguous_shards']==1
 def report(count):
  ids=[i for shard in plan['shards'][:count] for i in json.loads((root/'operator'/shard['selection_file']).read_text())['indices']]
  proofs=[]
  for shard in plan['shards'][:count]:
   for role in ('cache','control','half0','M100'):
    label=shard['label']+'_'+role;path=root/'accepted'/(label+'.json');r=json.loads(path.read_text())
    proofs.append(dict(label=label,receipt=dict(path=str(path),**watch.identity(path)),archive=r['archive_identity']))
  return dict(status='partial',external_gate=None,coverage=dict(shards=count,objects=len(ids),indices=ids,predictions_per_arm=15*len(ids),
   inference_jobs=3*count,cache_jobs=count,actual_prepared_caches=15*len(ids)//2),
   per_corruption={arm:{str(c):dict(n=len(ids),pm_correct=count*correct) for c in range(15)} for arm,correct in [('control',32),('half0',30),('M100',31)]},
   comparisons={'M100_vs_control':{'pm':dict(delta_pp=-1.5625,object_cluster_ci95_pp=[-2.,1.])}},
   provenance=dict(plan=watch.identity(root/'operator/plan.json'),common_source_identity=watch.identity(root/'common_identity.json'),raw_zip_sha_verified=False,receipts=proofs))
 first=report(1);legacy=derived/'partial_shard000.json';legacy.write_text(json.dumps(first))
 calls=[]
 def aggregate(path,final=False):calls.append((path,final));return report(2)
 watcher=watch.Watcher(root,knowledge,types.SimpleNamespace(aggregate=aggregate),HERE/'analyze_full.py')
 assert watcher.step()['contiguous_shards']==1 and not calls,'Initial legacy report must be reused without recount'
 canonical=derived/'partial_shards001.json';assert canonical.read_bytes()==legacy.read_bytes()
 first_knowledge=knowledge.read_bytes();watcher.step();assert knowledge.read_bytes()==first_knowledge and not calls
 # Recovery replays immutable output and leaves the progress record unique.
 restarted=watch.Watcher(root,knowledge,types.SimpleNamespace(aggregate=aggregate),HERE/'analyze_full.py')
 restarted.step();assert knowledge.read_bytes()==first_knowledge and not calls
 for role in ('cache','control','half0','M100'):receipt(1,role)
 restarted.step();assert len(calls)==1 and calls[0][1] is False
 assert knowledge.read_text().count('<!-- progress shards=')==2
 assert json.loads((derived/'latest_progress.json').read_text())['contiguous_shards']==2
 sidecar=derived/'partial_shards002.meta.json';good_meta=sidecar.read_text();bad=json.loads(good_meta);bad['analyzer']['sha256']='bad'
 sidecar.write_text(json.dumps(bad));rejects(restarted.step);assert sidecar.read_text()==json.dumps(bad)
 sidecar.write_text(good_meta)
 malformed=root/'accepted/shard001_M100.json';good=malformed.read_text();bad=json.loads(good);bad['ack_uploaded']='true'
 malformed.write_text(json.dumps(bad));rejects(restarted.step);malformed.write_text(good)
 invalid=copy.deepcopy(first);invalid['external_gate']={'advance_to_external':True}
 rejects(lambda:watcher.check_report(invalid,1))
 foreign=root.parent/'foreign.md';foreign.write_text('User notes; not watcher owned\n')
 rejects(lambda:watch.Watcher(root,foreign,types.SimpleNamespace(aggregate=aggregate),HERE/'analyze_full.py'))
print('FULL_LOCAL_WATCHER_CPU_SELFCHECK_COMPLETE')
