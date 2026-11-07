"""Posthoc CPU diagnostic: actual decoded lift versus its ideal physical target."""
import io,json,zipfile
from pathlib import Path
import numpy as np
from analyze import cd
from aggregate import interval
root=Path('result/preparation/gsd_hypotheses_20261105')
paths=[root/'transfer_v3_technical.zip']+sorted((root/'downloaded_transfer').glob('condition*/raw.zip'))
assert len(paths)==19;rows=[]
for p in paths:
 with zipfile.ZipFile(p) as z:
  config=json.loads(z.read('science/config.json'));records=json.loads(z.read('science/records.json'));states={};baselines={}
  for row in records:
   if row['method']!='H4_transfer':continue
   c=row['corruption'];oid=row['object_id'];index=config['ids'].index(oid)
   if c not in states:
    with np.load(io.BytesIO(z.read('science/'+c+'_states.npz')),allow_pickle=False) as a:states[c]=dict(a)
   state=states[c];key=(c,oid)
   if key not in baselines:baselines[key]=cd(state['y'][index],state['clean'][index])
   with np.load(io.BytesIO(z.read(f"science/{c}_{oid}_lift_{row['rho']}.npz")),allow_pickle=False) as a:
    actual=cd(a['changed'],state['clean'][index]);ideal=cd(state['y'][index]+a['delta'],state['clean'][index])
   rows.append(dict(object_id=oid,corruption=c,seed=config['seed'],timestep=config['timestep_override'],rho=row['rho'],actual_change=actual-baselines[key],ideal_change=ideal-baselines[key],difference=actual-ideal))
assert len(rows)==784
result=dict(scope='posthoc clean-geometry diagnostic of cached decoded prediction only; no trajectory/accuracy or tuning',summaries={})
for rho in (.001,.01):
 subset=[r for r in rows if r['rho']==rho and r['timestep'] is None];assert len(subset)==360
 contrast=interval(subset)
 actual=interval([dict(r,difference=r['actual_change']) for r in subset]);ideal=interval([dict(r,difference=r['ideal_change']) for r in subset])
 result['summaries'][str(rho)]=dict(actual_minus_ideal=contrast,actual_minus_baseline=actual,ideal_minus_baseline=ideal,actual_decreases=sum(r['actual_change']<0 for r in subset),ideal_decreases=sum(r['ideal_change']<0 for r in subset))
(root/'h4_geometry_posthoc.json').write_text(json.dumps(result,indent=2)+'\n');(root/'h4_geometry_posthoc_rows.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(result,indent=2))
