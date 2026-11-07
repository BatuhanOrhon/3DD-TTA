import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
archive=Path('result/modelnet40_c/lion_guidance_pilot/raw/20261107T111253Z_c.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='b23899310bbc6b58dc63ff384abdb936957236b86caed85d32fdbf48c5a446dd'
rows=[]
with zipfile.ZipFile(archive) as z:
 for name in sorted(n for n in z.namelist() if n.endswith(('/causal_feature.json','/fixed_feature.json','/causal_all.json'))):
  case,filename=name.split('/');arm=filename[:-5];detail=json.loads(z.read(name))['details']
  with np.load(io.BytesIO(z.read(case+'/'+arm+'_repair.npz')),allow_pickle=False) as d:
   for position,record in enumerate(detail):
    if position==0:continue
    teacher=np.asarray(record['teacher'],dtype=float)
    before=np.asarray(record['before_logits'],dtype=float);after=np.asarray(record['attempted_logits'],dtype=float)
    def logsoftmax(x):
     y=x-x.max(-1,keepdims=True);return y-np.log(np.exp(y).sum(-1,keepdims=True))
    delta_ce=-(teacher*(logsoftmax(after)-logsoftmax(before))).sum(-1)
    h=d['latent'][:,position].astype(float);change=d['attempted_latent'][:,position].astype(float)-h
    for i in range(2):
     rows.append(dict(case=case,arm=arm,position=position,batch_position=i,accepted=record['accepted'][i],
      geometry_ratio=record['geometry_ratio'][i],teacher_ce_change=float(delta_ce[i]),
      xyz_delta_max=float(np.max(np.abs(change[i,:,:3]))),feature_delta_rms=float(np.sqrt(np.mean(change[i,:,3]**2))),
      logit_delta_rms=float(np.sqrt(np.mean((before[i]-after[i])**2)))))
aggregate={}
for arm in ('causal_feature','fixed_feature','causal_all'):
 own=[r for r in rows if r['arm']==arm];accepted=[r for r in own if r['accepted']]
 aggregate[arm]=dict(attempts=len(own),accepted=len(accepted),
  geometry_ratio_quantiles=np.quantile([r['geometry_ratio'] for r in own],[0,.5,.95,1]).tolist(),
  xyz_delta_max=max(r['xyz_delta_max'] for r in own),
  median_feature_delta_rms=float(np.median([r['feature_delta_rms'] for r in own])),
  median_logit_delta_rms=float(np.median([r['logit_delta_rms'] for r in own])),
  accepted_ce_decreased=sum(r['teacher_ce_change']<0 for r in accepted),
  median_accepted_ce_change=float(np.median([r['teacher_ce_change'] for r in accepted])) if accepted else None)
report=dict(scope='Archived attempted corrections; CE decrease is consistency with pseudo-teacher, not correctness',aggregate=aggregate,rows=rows)
out=Path('result/modelnet40_c/lion_guidance_pilot/derived/20261107T111253Z_c')
(out/'repair_activity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(out/'repair_analysis.py').write_text(Path(__file__).read_text())
print(json.dumps(aggregate,indent=2))
