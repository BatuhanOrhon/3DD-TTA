"""Posthoc CPU geometry/clean-z diagnostics; no clean input reaches inference."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def read(path):
    with np.load(path,allow_pickle=False) as saved:
        values={k:saved[k].copy() for k in saved.files if k!='metadata'}
        meta=json.loads(saved['metadata'].item())
    for key,value in values.items():
        digest=hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
        assert digest==meta['tensor_hashes'][key]
    return values


def geometry_distance(left,right):
    return np.mean(cKDTree(left).query(right)[0]**2)+np.mean(cKDTree(right).query(left)[0]**2)


def analyze(phase,old_root,output):
    phase,old_root=Path(phase),Path(old_root)
    new_configs=list((phase/'raw').glob('*/*/*/config.json'))
    old_configs=list((old_root/'raw/baseline').glob('*/*/*/config.json'))
    assert len(new_configs)==len(old_configs)==1
    new_cfg=json.loads(new_configs[0].read_text());old_cfg=json.loads(old_configs[0].read_text())
    assert new_cfg['lion_state_before']==old_cfg['lion_state_before'], 'Archived clean target uses different LION weights'
    assert new_cfg['asset_manifest']['guidance']['lion_checkpoint']==old_cfg['asset_manifest']['guidance']['lion_checkpoint']
    before,after,cycles,eligible=[],[],[],[]
    for path in sorted((phase/'paired').glob('modelnet40_c/*/*.reencode.npz')):
        current=read(path)
        cycles.extend(geometry_distance(a,b) for a,b in zip(current['bridge_points'],current['cycle_points']))
        reference=old_root/'paired/common/modelnet40_c'/path.parent.name/path.name.replace('.reencode.npz','.oracle.npz')
        if not reference.is_file():
            eligible.append(dict(path=str(path),eligible=False,reason='no matching archived clean oracle batch'))
            continue
        old=read(reference)
        matched=(np.array_equal(current['prepared_input'],old['corrupted_input']) and
                 np.array_equal(current['eps_z'],old['eps_z']) and np.array_equal(current['eps_h'],old['eps_h']))
        eligible.append(dict(path=str(path),eligible=matched,reference=str(reference)))
        if not matched:continue
        with np.load(path.with_name(path.name.replace('.reencode.npz','.npz')),allow_pickle=False) as state:
            original=state['shape_latent'].copy().reshape(-1,128)
        before.extend(np.mean((original.astype(float)-old['clean_z'])**2,axis=1))
        after.extend(np.mean((current['selected_z'].astype(float)-old['clean_z'])**2,axis=1))
    report=dict(cycle_symmetric_squared_chamfer_LION_frame=dict(mean=float(np.mean(cycles)),median=float(np.median(cycles))),
        eligibility=eligible,clean_targets_posthoc_only=True,
        limitation='clean encoder target from archived runtime; numerical/native/normalization effects remain, no h pointwise correspondence or physical corruption-error claim')
    if before:
        before,after=np.asarray(before),np.asarray(after)
        report['clean_z']=dict(n_events=len(before),original_mse=float(before.mean()),reencoded_mse=float(after.mean()),
            aggregate_mse_ratio=float(after.sum()/before.sum()),fraction_improved=float((after<before).mean()))
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    if sys.argv[1:]==['--self-check']:
        x=np.array([[0.,0.,0.],[1.,0.,0.]])
        assert geometry_distance(x,x)==0 and geometry_distance(x,x[::-1])==0
        assert geometry_distance(x,x+np.array([0.,1.,0.]))==2
        print('REENCODE_BRIDGE_DIAGNOSTIC_SELF_CHECK_PASSED')
    else:print(json.dumps(analyze(*sys.argv[1:]),indent=2))
