"""Posthoc clean-mean direction diagnosis; clean proof is never a proposal input."""
from pathlib import Path
import json,sys
import numpy as np


def collect(phase,oracle,output):
    phase,oracle,output=Path(phase),Path(oracle),Path(output)
    if output.exists():raise FileExistsError('Keep previous derived analysis intact')
    rows={};matched_inputs=matched_eps=0
    for file in sorted((phase/'paired/common/modelnet40_c').glob('*/*.refresh.npz')):
        old=oracle/'paired/common/modelnet40_c'/file.parent.name/file.name.replace('.refresh.npz','.oracle.npz')
        with np.load(file,allow_pickle=False) as saved,np.load(old,allow_pickle=False) as prior:
            p={k:saved[k].astype(np.float64) for k in saved.files if k!='metadata'}
            target=prior['z_clean_mu'].astype(np.float64)
            matched_inputs+=int(np.all(saved['prepared_input']==prior['corrupted_input'],axis=(1,2)).sum())
            matched_eps+=int(np.all(saved['eps_z']==prior['eps_z'],axis=1).sum())
        desired=target-p['mu'];direction=p['new_mu']-p['mu'];ae=p['ae_mu']-p['mu']
        norm=lambda x:np.linalg.norm(x,axis=1)
        denominator=(desired**2).sum(1);valid=denominator>1e-12
        cosine_valid=valid & (norm(direction)>1e-12)
        base_z=p['mu']+p['sigma']*p['eps_z'];z_norm=norm(base_z)
        rows.setdefault(file.parent.name,[]).append(dict(
            mu_error_ratio=((p['new_mu']-target)**2).sum(1)[valid]/denominator[valid],
            ae_mu_error_ratio=((p['ae_mu']-target)**2).sum(1)[valid]/denominator[valid],
            cosine=(direction*desired).sum(1)[cosine_valid]/(norm(direction)[cosine_valid]*norm(desired)[cosine_valid]),
            mean_z_displacement=norm(p['mean_z']-base_z)/z_norm,
            posterior_z_displacement=norm(p['posterior_z']-base_z)/z_norm,
            sigma_specific_displacement=norm(p['posterior_z']-p['mean_z'])/z_norm,
            mean_vs_ae_direction_difference=norm(direction-ae),
            objects=np.ones(len(target))))
    assert len(rows)==15 and sum(len(batch['objects']) for batches in rows.values() for batch in batches)==960
    def summarize(batches):
        merged={k:np.concatenate([b[k] for b in batches]) for k in batches[0]}
        return {k:dict(events=len(v),median=float(np.median(v)),mean=float(np.mean(v)),
                       minimum=float(np.min(v)),maximum=float(np.max(v))) for k,v in merged.items() if k!='objects'}
    result=dict(per_corruption={k:summarize(v) for k,v in rows.items()},
                overall=summarize([b for v in rows.values() for b in v]),
                exact_old_prepared_input_matches=matched_inputs,exact_old_eps_matches=matched_eps,
                limits='Posthoc direction diagnosis against previous same-ID clean-global means only. Clean data not used to generate proposals or select inference.64exposed objects/15fixed corruptions/seed0; latent closeness not accuracy causation. AE-only is a direction control, no extra accuracy arm.')
    output.mkdir(parents=True);(output/'direction_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(collect(*sys.argv[1:]),indent=2))
