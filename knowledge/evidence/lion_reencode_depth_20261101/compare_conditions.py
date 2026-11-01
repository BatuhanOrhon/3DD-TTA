"""Paired depth comparisons against a fresh replay N3 and archived first PM."""
import csv
import json
from pathlib import Path
import sys

import numpy as np


def paired(left,right):
    assert left.shape==right.shape and left.ndim==2 and left.dtype==right.dtype==bool
    delta=left.astype(float)-right
    rng=np.random.RandomState(20261101)
    draw=rng.randint(0,left.shape[1],(10000,left.shape[1]))
    return dict(delta_pp=float(delta.mean()*100),conditional95_pp=np.percentile(delta.mean(0)[draw].mean(1)*100,[2.5,97.5]).tolist(),corrected=int((left&~right).sum()),harmed=int((~left&right).sum()))


def compare(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    reports={n:json.loads((root/('pilot%d_local_analysis'%n)/'acceptance.json').read_text()) for n in (3,5,8)}
    conditions={}
    for n in (3,5,8):
        with np.load(root/('pilot%d_local_analysis'%n)/'outcomes.npz',allow_pickle=False) as saved:conditions[n]={k:saved[k].copy() for k in saved.files}
        assert reports[n]['status']=='passed' and reports[n]['n_original_objects']==64 and reports[n]['n_corruptions']==15
        for key in ('first_pm','first_terminal','cycle','n3_pm'):assert np.array_equal(conditions[n][key],conditions[3][key])
    corruptions=[r['corruption'] for r in reports[3]['per_corruption']]
    assert all([r['corruption'] for r in reports[n]['per_corruption']]==corruptions for n in (5,8))
    comparisons={}
    for n in (3,5,8):
        for method in ('second_pm','combined_pm'):
            comparisons['N%d_%s-first_pm'%(n,method)]=paired(conditions[n][method],conditions[3]['first_pm'])
        if n!=3:comparisons['N%d_second_pm-fresh_N3'%n]=paired(conditions[n]['second_pm'],conditions[3]['second_pm'])
    comparisons['N8_second_pm-N5_second_pm']=paired(conditions[8]['second_pm'],conditions[5]['second_pm'])
    comparisons['fresh_N3-archived_N3']=paired(conditions[3]['second_pm'],conditions[3]['n3_pm'])
    table={str(n):reports[n]['accuracy_percent'] for n in (3,5,8)}
    report=dict(accuracy_percent=table,comparisons=comparisons,scope='64 exposed objects/all15/s5/seed0/Point-MAE; identical archived first/z1/h1/Gaussian/RNG; additional fresh N3 second-only control',limitation='Exploratory requested5/8 scan; cluster intervals omit stochastic native/seed variation and multiple-comparison correction; no full-pipeline speed claim')
    (output/'depth_comparison.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    fields=['corruption','n','first_pm','archived_n3_pm','fresh_n3_pm','n5_pm','n8_pm','n3_combined_pm','n5_combined_pm','n8_combined_pm']
    with (output/'per_corruption.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for i,c in enumerate(corruptions):
            row=dict(corruption=c,n=64,first_pm=conditions[3]['first_pm'][i].mean()*100,archived_n3_pm=conditions[3]['n3_pm'][i].mean()*100)
            for n in (3,5,8):
                row[('fresh_n3_pm' if n==3 else 'n%d_pm'%n)]=conditions[n]['second_pm'][i].mean()*100
                row['n%d_combined_pm'%n]=conditions[n]['combined_pm'][i].mean()*100
            writer.writerow(row)
    return report


if __name__=='__main__':
    if sys.argv[1:]==['--self-check']:
        a=np.array([[True,False],[False,True]]);b=~a
        assert paired(a,b)==dict(delta_pp=0.,conditional95_pp=[0.,0.],corrected=2,harmed=2)
        print('REENCODE_DEPTH_COMPARISON_SELF_CHECK_PASSED')
    else:print(json.dumps(compare(*sys.argv[1:]),indent=2))
