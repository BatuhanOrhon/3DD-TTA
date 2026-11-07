"""Aggregate the fixed all15/three-seed matrix with original-object clusters."""
import argparse,csv,json
from pathlib import Path
import numpy as np
ROOT=Path('result/preparation/gsd_hypotheses_20261105')

def interval(rows):
    ids=sorted({r['object_id'] for r in rows})
    means=np.array([np.mean([r['difference'] for r in rows if r['object_id']==i]) for i in ids])
    rng=np.random.default_rng(20261105)
    sampled=means[rng.integers(0,len(ids),size=(10000,len(ids)))].mean(1)
    return dict(objects=len(ids),states=len(rows),mean=float(means.mean()),ci95=np.quantile(sampled,[.025,.975]).tolist(),fraction_negative=float(np.mean([r['difference']<0 for r in rows])))

def summarize(group):
    folders=sorted((ROOT/('downloaded_'+group)).glob('condition*/analysis.json'))
    assert len(folders)==18,(group,len(folders))
    tech=ROOT/(group+'_v3_cpu_recount.json')
    if group=='anchors':tech=ROOT/'anchors_early_cpu_recount_v2.json'
    technical=json.loads(tech.read_text())[0]
    analyses=[technical]+[json.loads(p.read_text())[0] for p in folders]
    rows=[r for a in analyses for r in a['rows']]
    for a in analyses:
        for key,value in a['gates'].items():
            if type(value) is bool:assert value,(a['path'],key)
    methods=sorted({r['method'] for r in rows})
    result=dict(group=group,packages=len(analyses),row_count=len(rows),methods={},contrasts={},limits='8 exposed original objects; frozen first-prediction geometry only, no classifier accuracy or end-to-end speed; intervals conditional on this cohort')
    for method in methods:
        primary=[r for r in rows if r['method']==method and r.get('timestep') is None]
        probe=[r for r in rows if r['method']==method and r.get('timestep') is not None]
        item=dict(primary_rows=len(primary),noise_probe_rows=len(probe))
        geometric=[r for r in primary if 'cpu_clean_cd_change' in r]
        if geometric:item.update(clean_cd_decreases=sum(r['cpu_clean_cd_change']<0 for r in geometric),mean_clean_cd_change=float(np.mean([r['cpu_clean_cd_change'] for r in geometric])))
        result['methods'][method]=item
    pairs=[('H1_reconstructed_anchor','H1_raw_anchor'),('H2_robust_surface','H1_raw_anchor')] if group=='anchors' else [('H3_lambda1.0','H3_lambda0.0')]
    keys=lambda r:(r['object_id'],r['corruption'],r['seed'],r.get('timestep'))
    for left,right in pairs:
        a={keys(r):r for r in rows if r['method']==left};b={keys(r):r for r in rows if r['method']==right}
        assert len(a)==392 and a.keys()==b.keys()
        differences=[]
        for key,v in a.items():
            differences.append(dict(object_id=v['object_id'],corruption=v['corruption'],seed=v['seed'],timestep=v.get('timestep'),difference=v['cpu_clean_cd_after']-b[key]['cpu_clean_cd_after']))
        primary=[r for r in differences if r['timestep'] is None];assert len(primary)==360
        name=left+'_minus_'+right
        result['contrasts'][name]=dict(primary=interval(primary),probes=interval([r for r in differences if r['timestep'] is not None]),per_corruption={c:interval([r for r in primary if r['corruption']==c]) for c in sorted({r['corruption'] for r in primary})})
    if group=='transfer':
        lifted=[r for r in rows if r['method']=='H4_transfer']
        result['lift']={}
        for rho in (.001,.01):
            subset=[r for r in lifted if r['rho']==rho and r.get('timestep') is None]
            assert len(subset)==360
            values=np.array([r['cpu_finite_lift_relative_error'] for r in subset])
            result['lift'][str(rho)]=dict(n=len(subset),relative_error_min_median_max=[float(values.min()),float(np.median(values)),float(values.max())],cosine_min_median=[min(r['finite_lift_cosine'] for r in subset),float(np.median([r['finite_lift_cosine'] for r in subset]))],scheduler_max_error=max(r['cpu_scheduler_max_error'] for r in subset),decoder_repeat_rms_max=max(r['decoder_repeat_rms'] for r in subset),prior_repeat_rms_max=max(r['prior_repeat_rms'] for r in subset))
        solvers=[r for r in rows if r['method']=='H3_solver']
        result['solver']=dict(n=len(solvers),all_converged=all(r['converged'] for r in solvers),max_iterations=max(r['iterations'] for r in solvers),max_relative_residual=max(r['relative_residual'] for r in solvers),max_cpu_huber_change=max(r['cpu_huber_change'] for r in rows if 'cpu_huber_change' in r))
    result['verification']=dict(max_geometry_cpu_gpu_error=max(a['gates']['geometry_cpu_gpu_max_error'] for a in analyses),max_direction_error=max(a['gates']['direction_max_error'] for a in analyses),all_packages_cpu_gates_pass=True)
    output=ROOT/(group+'_final_summary.json');output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=sorted({k for r in rows for k,v in r.items() if type(v) in (int,float,str,bool) or v is None})
    with (ROOT/(group+'_all_rows.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('methods','contrasts')},indent=2))
    print(json.dumps({k:v['primary'] for k,v in result['contrasts'].items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=('anchors','transfer'));summarize(p.parse_args().group)
