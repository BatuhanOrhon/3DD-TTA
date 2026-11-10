#!/usr/bin/env python3
"""Validate a closed pilot grid, then reduce the SAME four logits on CPU."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_terminal_logit_pilot import ARMS, CANDIDATES, validate_registration


def softmax(value):
    value=value-value.max(axis=-1,keepdims=True)
    result=np.exp(value)
    return result/result.sum(axis=-1,keepdims=True)


def reductions(logits):
    value=np.asarray(logits,dtype=np.float64)
    if value.ndim!=3 or value.shape[1:]!=(4,40) or len(value)==0 or not np.isfinite(value).all():
        raise ValueError('Expected finite nonempty B x 4 x 40 class logits')
    return softmax(value).mean(1),softmax(value.mean(1))


def paired(reference,candidate):
    """Arrays are original object x corruption; operational repeats stay separate."""
    a,b=np.asarray(reference),np.asarray(candidate)
    if a.shape!=b.shape or a.ndim!=2 or not a.size or not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():
        raise ValueError('Expected matching binary object x corruption grids')
    delta=(b.astype(float)-a).mean(1)*100
    picks=np.random.default_rng(20261108).integers(len(a),size=(10000,len(a)))
    return dict(delta_pp=float(delta.mean()),ci95_pp=np.quantile(delta[picks].mean(1),[.025,.975]).tolist(),
        corrected=int(((a==0)&(b==1)).sum()),harmed=int(((a==1)&(b==0)).sum()),
        net=int(b.sum())-int(a.sum()),objects=len(a),pairs=a.size,degenerate=bool(np.ptp(delta)==0))


def write_csv(path,rows):
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def analyze(root,registration):
    root=Path(root)
    validate_registration(registration)
    complete=json.loads((root/'completion.json').read_text())
    config=json.loads((root/'config.json').read_text())
    if complete['status']!='complete' or config['phase']!='pilot' or config['protocol']!=registration:
        raise ValueError('A complete frozen pilot is required')
    if complete['cases']!=225 or len(list(root.glob('*/*.npz')))!=225:
        raise ValueError('Incomplete or extra inference cases')
    environment=json.loads((root/'environment.json').read_text())
    ids,corruptions=registration['pilot_ids'],registration['corruptions']
    summaries,per_corruption,comparisons,diagnostics,per_object=[],[],[],[],[]
    all_correct,all_predictions={},{}
    float32_changes=0
    for repeat in range(3):
        correct={arm:[] for arm in [*ARMS,'P8-LM']}
        predictions={arm:[] for arm in [*ARMS,'P8-LM']}
        seconds={arm:[] for arm in ARMS}
        memory={arm:[] for arm in ARMS}
        reserved={arm:[] for arm in ARMS}
        for ci,c in enumerate(corruptions):
            case=root/c
            integrity=json.loads((root/'integrity'/c/'integrity.json').read_text())
            if (integrity['status']!='verified' or integrity['before']!=integrity['after']
                    or integrity['before']!=environment['model_hashes']):
                raise ValueError('Missing frozen model proof')
            labels=None
            terminal_pm=None
            for arm,policy in ARMS.items():
                stem='%s_b0_r%d'%(arm,repeat)
                with np.load(case/(stem+'.npz'),allow_pickle=False) as data:
                    values={k:data[k].copy() for k in data.files}
                meta=json.loads((case/(stem+'.json')).read_text())
                expected_candidates=CANDIDATES if policy['pm'] else ['terminal']
                logits=values['logits']
                if (values['index'].tolist()!=ids or values['candidates'].tolist()!=expected_candidates
                        or logits.shape!=(32,len(expected_candidates),40) or not np.isfinite(logits).all()
                        or values['label'].shape!=(32,) or not np.isin(values['label'],range(40)).all()
                        or meta['draw_seed']!=ci*1000 or meta['inference_seed']!=0
                        or meta['operational_repeat']!=repeat or meta['ids']!=ids):
                    raise ValueError('Malformed/unmatched pilot case '+stem+' '+c)
                if labels is not None and not np.array_equal(labels,values['label']):
                    raise ValueError('Unmatched labels')
                labels=values['label']
                gpu_pm=values['probability_mean']
                if (gpu_pm.shape!=(32,40) or not np.isfinite(gpu_pm).all() or (gpu_pm<0).any()
                        or not np.allclose(gpu_pm.sum(-1),1,atol=1e-6,rtol=0)
                        or not np.array_equal(gpu_pm.argmax(-1),values['prediction'])):
                    raise ValueError('Saved GPU PM prediction/probability mismatch')
                expected_gpu=softmax(logits).mean(1)
                if not np.allclose(expected_gpu,gpu_pm,atol=2e-6,rtol=1e-5):
                    raise ValueError('Archived GPU PM differs from its logits')
                if not all(np.isfinite(meta[k]) and meta[k]>0 for k in ('seconds','peak_mib','reserved_mib')):
                    raise ValueError('Invalid timing/memory')
                seconds[arm].append(meta['seconds']);memory[arm].append(meta['peak_mib']);reserved[arm].append(meta['reserved_mib'])
                pm=softmax(logits.astype(np.float64)).mean(1)
                policies={arm:pm}
                if policy['pm']:
                    pm,lm=reductions(logits)
                    policies={arm:pm,'P8-LM':lm}
                    float32_changes+=int((pm.argmax(-1)!=values['prediction']).sum())
                    terminal_pm=logits[:,-1].argmax(-1)
                    centered=logits.astype(np.float64)-logits.astype(np.float64).mean(-1,keepdims=True)
                    probs=softmax(logits.astype(np.float64))
                    sorted_p=np.sort(probs,axis=-1)
                    for j,name in enumerate(CANDIDATES):
                        diagnostics.append(dict(repeat=repeat,corruption=c,candidate=name,
                            centered_logit_norm_mean=float(np.linalg.norm(centered[:,j],axis=-1).mean()),
                            logit_range_mean=float(np.ptp(centered[:,j],axis=-1).mean()),
                            entropy_mean=float(-(probs[:,j]*np.log(probs[:,j].clip(1e-300))).sum(-1).mean()),
                            margin_mean=float((sorted_p[:,j,-1]-sorted_p[:,j,-2]).mean()),
                            lm_pm_disagreements=int((lm.argmax(-1)!=pm.argmax(-1)).sum()),
                            candidate_disagreements=int((np.ptp(logits.argmax(-1),axis=1)>0).sum()),
                            gpu32_cpu64_pm_prediction_changes=int((pm.argmax(-1)!=values['prediction']).sum())))
                for name,probability in policies.items():
                    pred=probability.argmax(-1)
                    predictions[name].append(pred)
                    own=(pred==labels).astype(int)
                    correct[name].append(own)
                    per_corruption.append(dict(run_id=root.name,dataset='modelnet40_c',severity=5,method=name,
                        seed=0,repeat=repeat,corruption=c,n_examples=32,n_correct=int(own.sum()),accuracy=float(own.mean()),
                        runtime_seconds=meta['seconds'] if name!='P8-LM' else '',
                        peak_gpu_memory_mb=meta['peak_mib'] if name!='P8-LM' else '',status='complete_subset'))
                    for i,identity in enumerate(ids):
                        per_object.append(dict(repeat=repeat,corruption=c,object_id=identity,label=int(labels[i]),
                            arm=name,prediction=int(pred[i]),correct=int(own[i])))
            diagnostics.append(dict(repeat=repeat,corruption=c,candidate='P8PMterminal_vs_P8',
                prediction_changes=int((terminal_pm!=predictions['P8'][-1]).sum())))
        for name in correct:
            correct[name]=np.stack(correct[name],axis=1)
            all_correct[name,repeat]=correct[name]
            all_predictions[name,repeat]=np.stack(predictions[name],axis=1)
            summaries.append(dict(run_id=root.name,dataset='modelnet40_c',severity=5,method=name,seed=0,repeat=repeat,
                n_corruptions=15,macro_accuracy=float(correct[name].mean()),total_examples=480,
                total_correct=int(correct[name].sum()),micro_accuracy=float(correct[name].mean()),
                macro14_accuracy=float(correct[name][:,[c!='background' for c in corruptions]].mean()),
                total_runtime_seconds=sum(seconds[name]) if name in seconds else '',
                amortized_ms_input=1000*sum(seconds[name])/480 if name in seconds else '',
                max_allocated_mib=max(memory[name]) if name in memory else '',
                max_reserved_mib=max(reserved[name]) if name in reserved else '',status='complete_subset'))
        for reference,candidate in [('T5','P5'),('T8','P8'),('P8-PM','P8-LM'),('P8','P8-PM')]:
            for group,indices in [('all15',list(range(15))),('macro14',[i for i,c in enumerate(corruptions) if c!='background']),
                                   *[(c,[i]) for i,c in enumerate(corruptions)]]:
                result=paired(correct[reference][:,indices],correct[candidate][:,indices])
                result.update(reference=reference,candidate=candidate,repeat=repeat,group=group)
                if candidate!='P8-LM':
                    a=sum(seconds[reference][i] for i in indices);b=sum(seconds[candidate][i] for i in indices)
                    result.update(reference_seconds=a,candidate_seconds=b,speedup=a/b,time_saved_percent=100*(1-b/a))
                comparisons.append(result)
    stability=[]
    for arm in [*ARMS,'P8-LM']:
        for repeat in (1,2):
            stability.append(dict(arm=arm,repeat=repeat,prediction_changes_vs_repeat0=int(
                (all_predictions[arm,0]!=all_predictions[arm,repeat]).sum())))
    return dict(summary=summaries,per_corruption=per_corruption,comparisons=comparisons,diagnostics=diagnostics,
                per_object=per_object,stability=stability,gpu32_cpu64_pm_changes=float32_changes,
                uncertainty='Conditional object-cluster bootstrap10000 seed20261108. Repeat0 primary; 1/2 separate. '
                'Corruption intervals unadjusted/descriptive. Official test development, B32, seed0, one runtime. '
                'A degenerate CI is not equivalence; native floor prevents causal attribution. No full-data promotion.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--registration',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=analyze(args.root,json.loads(args.registration.read_text()))
    args.output.mkdir(parents=True,exist_ok=False)
    for key in ('summary','per_corruption','per_object'):
        write_csv(args.output/(key+'.csv'),result[key])
    with (args.output/'analysis.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print('TERMINAL_LOGIT_ANALYSIS_VERIFIED',len(result['per_object']))


if __name__=='__main__':main()
