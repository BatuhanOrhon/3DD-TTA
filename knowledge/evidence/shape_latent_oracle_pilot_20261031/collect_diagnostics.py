"""Descriptive latent displacements and instrumented timing after acceptance."""
import json
from pathlib import Path
import sys
import numpy as np


def collect(phase,output):
    phase,output=Path(phase),Path(output);output.mkdir(parents=True,exist_ok=True)
    reports={}
    for f in sorted((phase/'paired/common/modelnet40_c').glob('*/*.oracle.npz')):
        corruption=f.parent.name
        with np.load(f,allow_pickle=False) as p, np.load(f.with_name(f.name.replace('.oracle.npz','.npz')),allow_pickle=False) as a:
            z=a['shape_latent'].reshape(-1,128).astype(float)
            h=a['local_latent'].reshape(-1,8192).astype(float)
            pairs={'clean_z_relative':(p['clean_z'],z),
                   'reencoded_h_relative':(p['h_reencoded'],h),
                   'clean_h_relative':(p['h_clean'],h)}
            for name,(v,ref) in pairs.items():
                ratio=np.linalg.norm(v.astype(float)-ref,axis=1)/np.maximum(np.linalg.norm(ref,axis=1),1e-12)
                assert np.isfinite(ratio).all()
                reports.setdefault(corruption,{}).setdefault(name,[]).extend(ratio.tolist())
    assert len(reports)==15
    summaries={c:{k:dict(median=float(np.median(v)),maximum=float(np.max(v)),objects=len(v)) for k,v in row.items()} for c,row in reports.items()}
    timing={}
    for f in sorted((phase/'raw').glob('*/*/*/*/config.json')):
        cfg=json.loads(f.read_text());arm=cfg['shape_oracle']['arm']
        diagnostics=[json.loads(p.read_text()) for p in f.parent.glob('batches/*/*/diagnostics.json')]
        assert len(diagnostics)==30
        timing[arm]=dict(worker_wall_seconds=cfg['worker_wall_seconds_including_artifact_io'],
                        reconstruction_and_terminal_seconds=sum(d['shared_reconstruction_seconds']+d['head_seconds'] for d in diagnostics),
                        peak_whole_process_gpu_memory_mib=max(d['whole_process_peak_gpu_memory_mb'] for d in diagnostics),
                        run_id=cfg['run_id'])
    radius={c:dict(objects=len(v['clean_z_relative']),outside_previous_05_ball=int((np.asarray(v['clean_z_relative'])>.05).sum()),minimum=float(np.min(v['clean_z_relative']))) for c,v in reports.items()}
    result=dict(displacement=summaries,timing=timing,oracle_radius=radius,
                limits='Descriptive under paired encoder draws. A creates every oracle posterior/proof; other arms replay common preparation. No matched deployment timing or oracle-density calibration claim. Clean-h pointwise L2 also changes ordering/resampling and is not a geometric error metric.')
    (output/'diagnostics_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(collect(*sys.argv[1:]),indent=2))
