"""Audit final cohort/source coverage and additional saved-array measurements."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
root=Path('result/preparation/gsd_hypotheses_20261105');result={};manifests=[];cos_errors=[]
for group in ('anchors','transfer'):
 accepted=json.loads((root/('downloaded_'+group)/'accepted.json').read_text());assert set(accepted)=={str(i) for i in range(18)}
 technical=json.loads((root/(group+'_v3_receipt.json')).read_text())
 packages=[(root/(group+'_v3_technical.zip'),technical)]+[(Path(accepted[str(i)]['package']),json.loads((root/('downloaded_'+group)/('condition%02d'%i)/'receipt.json').read_text())) for i in range(18)]
 seen=set();primary=set();seconds=[];peaks=[];source_hashes=set();permutation=[]
 for p,receipt in packages:
  assert p.stat().st_size==receipt['identity']['bytes']
  with zipfile.ZipFile(p) as z:
   config=json.loads(z.read('science/config.json'));complete=json.loads(z.read('science/complete.json'));status=json.loads(z.read('status.json'))
   assert complete['status']=='complete' and status['science_exit']==0 and status['cpu_gate_exit']==0
   operator='operator/knowledge/evidence/gsd_hypotheses_20261105/pilot.py'
   source_hashes.add(hashlib.sha256(z.read(operator)).hexdigest())
   seconds.append(complete['seconds']);peaks.append(complete['peak_gpu_bytes'])
   for c in config['corruptions']:
    for object_id in config['ids']:
     key=(object_id,c,config['seed'],config['timestep_override']);assert key not in seen,key;seen.add(key)
     if key[-1] is None:primary.add(key)
   rows=json.loads(z.read('science/records.json'))
   permutation += [r['relative_rhs_error'] for r in rows if r['method']=='H2_observation_permutation']
   if group=='transfer':
    states={}
    for row in rows:
     if row['method']!='H4_transfer':continue
     c=row['corruption'];oid=row['object_id'];index=config['ids'].index(oid)
     if c not in states:
      with np.load(io.BytesIO(z.read('science/'+c+'_states.npz')),allow_pickle=False) as a:states[c]=a['y']
     with np.load(io.BytesIO(z.read(f"science/{c}_{oid}_lift_{row['rho']}.npz")),allow_pickle=False) as a:
      delta=a['delta'].astype(float);movement=a['changed'].astype(float)-states[c][index].astype(float)
      cosine=float(np.sum(delta*movement)/max(np.linalg.norm(delta)*np.linalg.norm(movement),1e-30))
      cos_errors.append(abs(cosine-row['finite_lift_cosine']))
  manifests.append(dict(group=group,local=str(p),drive=receipt['drive_package'],identity=receipt['identity'],accepted_package_member_integrity='passed by immutable artifact acceptance/independent analyzer'))
 assert len(seen)==392 and len(primary)==360 and len(source_hashes)==1
 assert {sum(k[1]==c for k in primary) for c in {k[1] for k in primary}}=={24}
 result[group]=dict(packages=len(packages),unique_states=len(seen),primary_states=len(primary),probe_states=len(seen-primary),original_objects=sorted({k[0] for k in seen}),source_pilot_sha256=next(iter(source_hashes)),loop_wall_seconds=sum(seconds),loop_wall_seconds_min_max=[min(seconds),max(seconds)],max_torch_allocated_bytes=max(peaks),timing_limit='diagnostic loop includes asset hashing, state/filter compression and writes; excludes model init; not deployment GPU time')
 if permutation:
  assert len(permutation)==392
  result[group]['observation_order_permutation_max']=max(permutation)
  result[group]['observation_order_permutation_pass_1e_10']=max(permutation)<1e-10
assert len(cos_errors)==784 and max(cos_errors)<1e-6
result['additional_H4_cosine_recount']=dict(cases=len(cos_errors),max_error=max(cos_errors),passed=True)
(root/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n');(root/'raw_acceptance_manifest.json').write_text(json.dumps(manifests,indent=2)+'\n');print(json.dumps(result,indent=2))
