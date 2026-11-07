"""Independent CPU recount of immutable diagnostic packages; no GPU replay."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def cd(points,clean):
    p=np.asarray(points,dtype=np.float64);c=np.asarray(clean,dtype=np.float64)
    return float(np.mean(cKDTree(c).query(p)[0]**2)+np.mean(cKDTree(p).query(c)[0]**2))


def arrays(archive,name):
    with np.load(io.BytesIO(archive.read(name)),allow_pickle=False) as a:return {k:a[k] for k in a.files}


def analyze(path):
    result={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gates':{},'rows':[]}
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('package_manifest.json'))
        for name,item in manifest.items():
            data=archive.read(name)
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
        config=json.loads(archive.read('science/config.json'))
        rows=json.loads(archive.read('science/records.json'))
        errors=[];direction_errors=[];objective_changes=[];lift_errors=[]
        for corruption in config['corruptions']:
            state=arrays(archive,'science/'+corruption+'_states.npz')
            filt=arrays(archive,'science/'+corruption+'_actual_filters.npz') if config['group']=='anchors' else None
            for index,object_id in enumerate(config['ids']):
                points=state['y'][index].astype(np.float64);clean=state['clean'][index].astype(np.float64)
                baseline=cd(points,clean)
                operator=arrays(archive,'science/'+corruption+'_'+str(object_id)+'_operator.npz')
                node=operator['node'];normal=operator['normals'][node]
                rhs=np.zeros_like(points)
                np.add.at(rhs,node,(operator['weights']*operator['residual'])[:,None]*normal)
                for row in (v for v in rows if v['corruption']==corruption and v['object_id']==object_id):
                    output=dict(row,seed=config['seed'],timestep=config.get('timestep_override'),cpu_baseline_clean_cd=baseline)
                    errors.append(abs(baseline-row['baseline_clean_cd']))
                    method=row['method']
                    name='science/'+corruption+'_'+str(object_id)+'_'+method+'_delta.npz'
                    if name in archive.namelist():
                        delta=arrays(archive,name)['delta'].astype(np.float64)
                        after=cd(points+delta,clean)
                        output.update(cpu_clean_cd_after=after,cpu_clean_cd_change=after-baseline)
                        errors.append(abs(after-row['clean_cd_after']))
                        if method.startswith('H1') or method=='H2_robust_surface':
                            if method=='H2_robust_surface':direction=rhs
                            else:
                                anchor=state['x'][index] if method=='H1_raw_anchor' else state['anchor'][index]
                                direction=-filt['filter'+str(index)].astype(np.float64)@(points-anchor)
                            factor=min(1.,.001*np.linalg.norm(points)/max(np.linalg.norm(direction),1e-12))
                            direction_errors.append(float(np.max(abs(delta-factor*direction))))
                        if method.startswith('H3_lambda'):
                            strength=float(method.removeprefix('H3_lambda'))
                            edge=operator['edges'];ew=operator['edge_weights']
                            projected=(normal*delta[node]).sum(1)
                            quadratic=.5*np.sum(delta**2)+.5*strength*np.sum(ew[:,None]*(delta[edge[:,0]]-delta[edge[:,1]])**2)
                            quadratic+=float(np.sum(operator['weights']*(.5*projected**2-projected*operator['residual'])))
                            sigma=float(operator['spacing']);q=operator['confidence'][node];c=1.345
                            def huber(value):
                                a=abs(value);return np.where(a<=c,.5*a*a,c*a-.5*c*c)
                            robust=.5*np.sum(delta**2)+.5*strength*np.sum(ew[:,None]*(delta[edge[:,0]]-delta[edge[:,1]])**2)
                            robust+=np.sum(q*(huber((projected-operator['residual'])/sigma)-huber(operator['residual']/sigma)))
                            output.update(cpu_quadratic_change=float(quadratic),cpu_huber_change=float(robust))
                            objective_changes.append(float(robust))
                    if method=='H4_transfer':
                        witness=arrays(archive,'science/'+corruption+'_'+str(object_id)+'_lift_'+str(row['rho'])+'.npz')
                        delta=witness['delta'].astype(np.float64)
                        denom=max(np.linalg.norm(delta),1e-12)
                        error=float(np.linalg.norm(witness['changed'].astype(np.float64)-points-delta)/denom)
                        scheduler=float(np.max(abs(witness['actual_scheduler_difference']-witness['expected_scheduler_difference'])))
                        output.update(cpu_finite_lift_relative_error=error,cpu_scheduler_max_error=scheduler)
                        lift_errors.append(abs(error-row['finite_lift_relative_error']))
                    result['rows'].append(output)
        result['gates'].update(archive_crc_and_member_hashes=True,geometry_cpu_gpu_max_error=max(errors,default=0),
                               geometry_tolerance_absolute=3e-6,geometry_gate=max(errors,default=0)<=3e-6,
                               direction_max_error=max(direction_errors,default=0),direction_gate=max(direction_errors,default=0)<=1e-6,
                               max_robust_objective_change=max(objective_changes,default=0),
                               frozen_objective_descent=max(objective_changes,default=0)<=1e-8,
                               lift_recount_max_error=max(lift_errors,default=0))
        result['config']=config
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archives',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args();records=[analyze(Path(p)) for p in args.archives]
    Path(args.output).write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{'path':v['path'],'rows':len(v['rows']),'gates':v['gates']} for v in records],indent=2))
