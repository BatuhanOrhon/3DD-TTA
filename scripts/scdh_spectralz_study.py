"""Predeclared finite study: label-free IDs/scales, deterministic grid and ties."""
import hashlib
import json
import math
import statistics

MODES=(200,400,800,1200,1800)
RHOS=(.1,.3,1.,3.)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def register_ids(excluded, *, salt='scdh-spectralz-20261101-v1'):
    excluded=sorted(set(excluded))
    if any(type(i) is not int or not 0<=i<2468 for i in excluded):
        raise ValueError('Invalid prior subset original IDs')
    available=set(range(2468))-set(excluded)
    if len(available)<192:raise ValueError('Insufficient unexposed IDs for disjoint64+128 registration')
    ranked=sorted(available,key=lambda i:(hashlib.sha256((salt+':'+str(i)).encode()).hexdigest(),i))
    dev,heldout=sorted(ranked[:64]),sorted(ranked[64:192])
    return dict(schema_version=1,salt=salt,excluded_subset_union=excluded,
                development=dev,confirmation=heldout,calibration=dev[:32],technical=dev[:32],timing=dev[:32],
                limitation='Historical full-benchmark exposure remains; heldout is a disjoint stage, not an unseen benchmark')


def freeze_calibration(records,binding):
    coefficients={}
    for modes in MODES:
        ratios=[v for r in records if r['M_max']==modes for v in r['ratios']]
        valid=[v for v in ratios if type(v) in (int,float) and math.isfinite(v) and v>=0]
        b=statistics.median(valid) if valid else None
        available=b is not None and b>0 and math.isfinite(b)
        coefficients[str(modes)]=dict(status='available' if available else 'unavailable',
            b_M=b if available else None,n_observations=len(ratios),n_valid=len(valid),n_excluded=len(ratios)-len(valid))
    equivalence={}
    for modes in MODES:
        rows=[r for r in records if r['M_max']==modes]
        if rows and all(r.get('weights_sha256') for r in rows):
            signature=tuple(r['weights_sha256'] for r in sorted(rows,key=lambda r:(r['corruption'],r['loop_count'],r['original_ids'])))
            equivalence.setdefault(signature,[]).append(modes)
    return dict(schema_version=1,status='frozen',method='pooled median of .003*SCD-z-norm/spectral-z-norm; no epsilon',
                coefficients=coefficients,binding=binding,observations_sha256=canonical_sha(records),
                calibration_weight_equivalence_groups=[v for v in equivalence.values() if len(v)>1],
                equivalence_scope='calibration states only; not proof of equivalent development/confirmation operators')


def job(arm,alpha=.03,modes=None,rho=None,seed=0,stage='development',suffix=''):
    slot='%s-%s-h%s'%(stage,arm,format(alpha,'.2f').replace('.','p'))
    if modes is not None:slot+='-M%d-r%s'%(modes,str(rho).replace('.','p'))
    slot+='-s%d'%seed+suffix
    return dict(slot=slot,stage=stage,arm=arm,alpha_h=alpha,modes=modes,rho=rho,seed=seed)


def phase_a(available=MODES):
    return [job('scd'),job('off')]+[job('spectral',modes=m,rho=r) for m in MODES if m in available for r in RHOS]


def rank_candidates(scores):
    rows=[r for r in scores if r['arm']=='spectral' and r.get('status','accepted')=='accepted']
    if any(type(r.get('correct')) is not int for r in rows):raise ValueError('Rank requires complete integer correct counts')
    return sorted(rows,key=lambda r:(-r['correct'],r['modes'],r['rho'],abs(r['alpha_h']-.03),r['alpha_h'],r['slot']))


def phase_b(top):
    if len(top)>2 or len({(r['modes'],r['rho']) for r in top})!=len(top):raise ValueError('At most two unique frozen A pairs')
    return ([job('spectral',a,r['modes'],r['rho']) for r in top for a in (.01,.05)]
            +[job(arm,a) for a in (.01,.05) for arm in ('scd','off')])


def confirmation(winner):
    a=winner['alpha_h']
    return [j for s in (0,1,2) for j in (
        [job('spectral',a,winner['modes'],winner['rho'],s,'confirmation'),
         job('scd',.03,seed=s,stage='confirmation'),job('off',a,seed=s,stage='confirmation')]
        +([job('scd',a,seed=s,stage='confirmation')] if a!=.03 else []))]
