"""Validate supplemental archive identities and recount saved finite-step witnesses."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
root=Path('result/preparation/gsd_hypotheses_20261105')
for archive_name,receipt_name in [('supplement_v2.zip','supplement_receipt.json'),('supplement_v1_failed.zip','failed_supplement_receipt.json')]:
 p=root/archive_name;r=json.loads((root/receipt_name).read_text());i=r.get('identity',r)
 assert p.stat().st_size==i['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==i['sha256']
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
with zipfile.ZipFile(root/'supplement_v2.zip') as z:
 rows=json.loads(z.read('records.json'));assert len(rows)==109
 finite_errors=[];slope_errors=[];scheduler_errors=[]
 for row in rows:
  if row['test']=='decoder_direction':
   name=f"{row['corruption']}_{row['object_id']}_{row['field']}_{row['rho']}.npz"
   with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:
    unit=a['unit'].astype(float);amount=float(a['amount']);movement=a['changed'].astype(float)-a['base'].astype(float)
    finite_errors.append(abs(np.linalg.norm(movement-amount*unit)/amount-row['relative_lift_error']))
    slope_errors.append(abs(np.sum(movement*unit)/amount-row['finite_directional_slope']))
  if row['test']=='DDIM_actual_scheduler':
   with np.load(io.BytesIO(z.read(f"scheduler{row['t']}.npz")),allow_pickle=False) as a:
    error=float(abs(a['actual']-a['expected']).max());scheduler_errors.append(error)
    assert abs(error-row['max_error'])<1e-10 and error<3e-6
 assert max(finite_errors)<3e-6 and max(slope_errors)<3e-6
 result=dict(records=len(rows),archive_integrity='passed',finite_error_recount_max=max(finite_errors),slope_recount_max=max(slope_errors),scheduler_max=max(scheduler_errors),
  permutation_max=max(r['relative_solution_error'] for r in rows if r['test']=='rebuilt_Y_graph_permutation'),
  permutation_all_converged=all(r['converged'] for r in rows if r['test']=='rebuilt_Y_graph_permutation'),
  counterexample=[r for r in rows if r['test']=='coherent_wrong_anchor'][0],fields={})
 for name in ('robust_graph','translation','scale','random'):
  result['fields'][name]={}
  for rho in (.0001,.001,.01):
   subset=[r for r in rows if r['test']=='decoder_direction' and r['field']==name and r['rho']==rho]
   assert len(subset)==8
   result['fields'][name][str(rho)]=dict(relative_error_min_median_max=np.quantile([r['relative_lift_error'] for r in subset],[0,.5,1]).tolist(),central_slope_min_median_max=np.quantile([r['central_difference_slope'] for r in subset],[0,.5,1]).tolist(),finite_slope_min_median_max=np.quantile([r['finite_directional_slope'] for r in subset],[0,.5,1]).tolist())
 result['limits']='Central finite-difference slopes and rebuilt-graph permutation statistics are GPU-recorded; central plus/minus decodes and permuted sparse arrays were not saved, so those two quantities are not independently recounted. Finite lift and scheduler arrays are independently recounted. This is no exact VJP.'
 (root/'supplement_cpu_recount.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
