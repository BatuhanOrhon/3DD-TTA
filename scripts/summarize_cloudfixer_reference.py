#!/usr/bin/env python3
"""Derive native CF operational-repeat statistics without combining predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def reference_summary(analysis: dict, config: dict) -> list:
    if analysis['campaign']!=config['campaign']:
        raise ValueError('Analysis/config campaign mismatch')
    reg=config['manifest']['registration']
    expected=set(reg['corruptions']);size=len(reg[config['phase']+'_ids'])
    groups={}
    for row in analysis['rows']:
        if row['config']['method']!='cloudfixer':continue
        groups.setdefault((row['seed'],row['readout']),[]).append(row)
    if not groups:raise ValueError('No original CF reference rows')
    result=[]
    for (seed,readout),rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['config']['operational_repeat'])
        if (reg['native_reference_repeats']!=3 or len(rows)!=3
                or [r['config']['operational_repeat'] for r in rows]!=[0,1,2]):
            raise ValueError('Need three separately identified native repetitions')
        accuracies=[];counts=[]
        for row in rows:
            coverage=row['corruptions']
            if (row['status']!='complete' or set(coverage)!=expected
                    or row['n_examples']!=size*len(expected)
                    or any(c['n_examples']!=size or not 0<=c['n_correct']<=size for c in coverage.values())
                    or row['n_correct']!=sum(c['n_correct'] for c in coverage.values())):
                raise ValueError('Incomplete or unmatched native-reference coverage')
            counts.append(row['n_correct'])
            accuracies.append(100*statistics.mean(c['n_correct']/c['n_examples'] for c in coverage.values()))
        result.append(dict(seed=seed,readout=readout,operational_repeats=[0,1,2],
            n_examples_per_repeat=size*len(expected),correct_by_repeat=counts,
            accuracy_pct_by_repeat=accuracies,mean_accuracy_pct=statistics.mean(accuracies),
            sample_sd_pp=statistics.stdev(accuracies),
            interpretation='Same-seed native operational variation; no logit ensemble; coverage from config and complete rows'))
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw_analysis=args.analysis.read_bytes();raw_config=args.config.read_bytes()
    result=dict(analysis_sha256=hashlib.sha256(raw_analysis).hexdigest(),
        config_sha256=hashlib.sha256(raw_config).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_reference=reference_summary(json.loads(raw_analysis),json.loads(raw_config)))
    with args.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')


if __name__=='__main__':main()
