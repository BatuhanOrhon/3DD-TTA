"""Posthoc clean-z distance/alignment; never an inference objective or selector."""
import json
from pathlib import Path
import sys

import numpy as np


def cosine(a,b):
    a,b=a.reshape(len(a),-1),b.reshape(len(b),-1)
    scale=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    valid=scale>1e-12
    values=np.sum(a[valid]*b[valid],axis=1)/scale[valid]
    return dict(median=float(np.median(values)) if len(values) else None,
                mean=float(values.mean()) if len(values) else None,
                positive_fraction=float((values>0).mean()) if len(values) else None,
                defined=int(valid.sum()),undefined=int((~valid).sum()))


def analyze(derived,raw,output):
    derived,raw,output=Path(derived),Path(raw),Path(output)
    acceptance=json.loads((derived/'acceptance.json').read_text())
    corruptions=acceptance['corruptions'];arms=[a for a in acceptance['arms'] if a!='firstpass']
    with np.load(derived/'paired_latents.npz',allow_pickle=False) as saved:
        values={k:saved[k].copy() for k in saved.files}
    clean=np.concatenate([values['clean__'+c] for c in corruptions]).astype(np.float64)
    encoder=np.concatenate([values['encoder__'+c] for c in corruptions]).astype(np.float64)
    scd=np.concatenate([values['scd__'+c] for c in corruptions]).astype(np.float64)
    reference=float(np.square(encoder-clean).mean());result={}
    for arm in arms:
        start=np.concatenate([values[arm+'__'+c+'__start'] for c in corruptions]).astype(np.float64)
        final=np.concatenate([values[arm+'__'+c+'__final'] for c in corruptions]).astype(np.float64)
        original_error=np.square(start-clean).reshape(len(start),-1).mean(1)
        final_error=np.square(final-clean).reshape(len(final),-1).mean(1)
        row=dict(start_mse=float(original_error.mean()),final_mse=float(final_error.mean()),
                 aggregate_ratio_to_encoder=float(final_error.mean()/reference),
                 aggregate_ratio_to_own_start=float(final_error.mean()/original_error.mean()),
                 median_per_event_mse_ratio=float(np.median(final_error/original_error)),
                 median_displacement_norm=float(np.median(np.linalg.norm((final-start).reshape(len(start),-1),axis=1))),
                 movement_cosine_to_clean=cosine(final-start,clean-start))
        if arm.endswith('guided3'):
            paths=list((raw/'paired/global'/arm).glob('*/*.npz'));assert len(paths)==len(corruptions)
            rows=[]
            for i in range(3):
                grad=[];target=[]
                for c in corruptions:
                    path=next((raw/'paired/global'/arm/c).glob('*.npz'))
                    with np.load(path,allow_pickle=False) as p:
                        grad.append(-p['grad_clean_%d'%i].astype(np.float64))
                        target.append(values['clean__'+c].astype(np.float64)-p['prior_clean_%d'%i].astype(np.float64))
                rows.append(dict(step=i,**cosine(np.concatenate(grad),np.concatenate(target))))
            row['negative_clean_gradient_alignment']=rows
        result[arm]=row
    report=dict(status='complete_posthoc_only',encoder_reference_mse=reference,
                scd_start_mse=float(np.square(scd-clean).mean()),
                encoder_to_scd_cosine=cosine(scd-encoder,clean-encoder),arms=result,
                oracle_used_in_inference=False,
                limits='Same paired clean posterior sample, not unique valid latent or accuracy guarantee; no parameter/step selection.')
    output.write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':print(json.dumps(analyze(*sys.argv[1:]),indent=2))
