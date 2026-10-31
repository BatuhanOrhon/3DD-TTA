"""Fixed paired object summaries; oracle diagnosis, no method/parameter selection."""
import csv,json
from pathlib import Path
import sys
import numpy as np


def summarize(accepted,output):
    accepted,output=Path(accepted),Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((accepted/'acceptance.json').read_text());assert report['status']=='passed'
    names=list(report['arms']['baseline']['per_corruption_percent'])
    with np.load(accepted/'outcomes.npz',allow_pickle=False) as saved:
        outcomes={k:saved[k].copy() for k in saved.files}
    assert set(outcomes)=={'baseline','clean_z_fixed_h','clean_z_reencode_h','clean_pair'}
    assert all(v.shape==(15,64) and v.dtype==bool for v in outcomes.values())
    draws=np.random.default_rng(20261031).integers(0,64,size=(10000,64))
    contrasts={}
    for left,right in [('clean_z_fixed_h','baseline'),('clean_z_reencode_h','baseline'),
                       ('clean_z_reencode_h','clean_z_fixed_h'),('clean_pair','clean_z_reencode_h'),('clean_pair','baseline')]:
        d=outcomes[left].astype(float)-outcomes[right]
        intervals=np.quantile(100*d.mean(0)[draws].mean(1),[.025,.975])
        contrasts[left+'-'+right]=dict(delta_pp=float(100*d.mean()),corrected=int((d>0).sum()),
                                       harmed=int((d<0).sum()),conditional_object_bootstrap_95=intervals.tolist())
    metrics={}
    b=names.index('background')
    for arm,v in outcomes.items():
        metrics[arm]=dict(correct=int(v.sum()),events=int(v.size),all15_percent=float(100*v.mean()),
                          macro14_percent=float(100*np.delete(v,b,axis=0).mean()),background_percent=float(100*v[b].mean()))
    result=dict(metrics=metrics,contrasts=contrasts,
                scope='64exposed objects;15fixed corruptions;seed0;10,000 original-object bootstrap draws;clean oracle not deployed TTA;D changes full canonical local input under A-h anchor;no multiplicity correction')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'per_corruption.csv').open('w',newline='') as file:
        writer=csv.writer(file,lineterminator='\n');writer.writerow(['corruption',*outcomes])
        for i,name in enumerate(names):writer.writerow([name,*[float(100*v[i].mean()) for v in outcomes.values()]])
    return result


if __name__=='__main__':print(json.dumps(summarize(*sys.argv[1:]),indent=2))
