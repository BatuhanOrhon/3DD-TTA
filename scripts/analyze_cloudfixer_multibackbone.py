#!/usr/bin/env python3
"""Recount downloaded full or corruption-scoped CF results; never run inference."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloudfixer_store import MountStore, canonical, sha
from scripts.run_cloudfixer_multibackbone import HEADS, METHODS, recover, units


def recount(raw, campaign, wanted, labels):
    if not raw.is_dir():raise ValueError('Downloaded raw receipts required')
    records=recover(MountStore(raw),campaign,wanted,labels)
    if len(records)!=len(wanted):raise ValueError('Artifact scope incomplete')
    totals={};previous={}
    for unit in wanted:
        from cloudfixer_store import unit_key
        p=records[unit_key(unit)]['values'];corr=unit['corruption']
        if corr in previous and str(p['rng_before_sha256'])!=previous[corr]:
            raise ValueError('Broken native RNG chain')
        previous[corr]=sha(canonical(json.loads(str(p['rng_after']))))
        for method,field in zip(METHODS,('source_logits','frame_logits','cf_logits')):
            for i,head in enumerate(HEADS):
                key=(method,head,corr);count,correct=totals.get(key,(0,0))
                totals[key]=(count+len(unit['ids']),correct+int(np.count_nonzero(np.argmax(p[field][:,i],axis=1)==p['labels'])))
    return [dict(method=m,backbone=h,corruption=c,n_examples=n,n_correct=k,accuracy=k/n)
            for (m,h,c),(n,k) in sorted(totals.items())]


def verify_csv(path, expected):
    actual=list(csv.DictReader(path.open()))
    def key(row):return tuple(row[k] for k in ('method','backbone','corruption'))
    lookup={key(r):r for r in actual}
    if len(lookup)!=len(actual) or set(lookup)!={key(r) for r in expected}:
        raise ValueError('CSV scope mismatch')
    for row in expected:
        saved=lookup[key(row)]
        if (int(saved['n_examples'])!=row['n_examples'] or int(saved['n_correct'])!=row['n_correct']
                or not np.isclose(float(saved['accuracy']),row['accuracy'],rtol=0,atol=1e-12)):
            raise ValueError('CSV differs from raw predictions')


def summarize(rows):
    result=[]
    for method in METHODS:
        for head in HEADS:
            group=[r for r in rows if r['method']==method and r['backbone']==head]
            n=sum(r['n_examples'] for r in group);correct=sum(r['n_correct'] for r in group)
            result.append(dict(method=method,backbone=head,total_examples=n,total_correct=correct,
                n_corruptions=len(group),micro_accuracy=correct/n,
                macro_accuracy=sum(r['accuracy'] for r in group)/len(group)))
    return result


def verify_summary(path, expected):
    actual=list(csv.DictReader(path.open()))
    lookup={(r['method'],r['backbone']):r for r in actual}
    if len(actual)!=9 or len(lookup)!=9:raise ValueError('Summary scope mismatch')
    for row in expected:
        saved=lookup[(row['method'],row['backbone'])]
        for key in ('total_examples','total_correct','n_corruptions'):
            if int(saved[key])!=row[key]:raise ValueError('Summary count mismatch')
        for key in ('micro_accuracy','macro_accuracy'):
            if not np.isclose(float(saved[key]),row[key],rtol=0,atol=1e-12):raise ValueError('Summary accuracy mismatch')


def combine_shards(shards):
    rows=[];seen=set();campaign=None
    for item in shards:
        if campaign is not None and item['campaign']!=campaign:raise ValueError('Mixed shard campaigns')
        campaign=item['campaign']
        for row in item['rows']:
            key=(row['method'],row['backbone'],row['corruption'])
            if key in seen:raise ValueError('Overlapping shard coverage')
            seen.add(key);rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,nargs='+',required=True,
        help='One full directory or all independently extracted corruption directories')
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--baseline-reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    labels=np.load(a.labels,allow_pickle=False).reshape(-1)
    reference=a.baseline_reference.read_bytes();shards=[];blocks=0
    for directory in a.run:
        config=json.loads((directory/'config.json').read_text());spec=config['manifest']
        campaign=sha(canonical(spec))
        if config['campaign']!=campaign or config['phase']!='full':raise ValueError('Full-phase campaign required')
        if sha(a.labels.read_bytes())!=spec['dataset_files']['label.npy']:raise ValueError('Label identity mismatch')
        if dict(bytes=len(reference),sha256=sha(reference))!=spec['baseline_reference']:
            raise ValueError('Accepted baseline identity mismatch')
        wanted=units(campaign,'full',spec.get('scientific_protocol',{}).get('batch_size',32))
        if config.get('artifact_scope')=='single_corruption':
            wanted=[u for u in wanted if u['corruption']==config['corruption']]
            if not wanted:raise ValueError('Unknown corruption scope')
        rows=recount(directory/'raw',campaign,wanted,labels)
        verify_csv(directory/'per_corruption.csv',rows)
        verify_summary(directory/'summary.csv',summarize(rows))
        shards.append(dict(campaign=campaign,rows=rows));blocks+=len(wanted)
    rows=combine_shards(shards);summary=summarize(rows)
    old=json.loads(reference)
    comparisons=[]
    for row in rows:
        if row['method']!='cloudfixer_plain':continue
        matches=[r for r in old['raw_archives'] if r['dataset']=='modelnet40_c'
            and r['arm']=='proximal' and r['backbone']==row['backbone'] and r['corruption']==row['corruption']]
        if len(matches)!=1 or matches[0]['n_examples']!=row['n_examples']:
            raise ValueError('Matched accepted LION scope missing')
        prior=matches[0]
        comparisons.append(dict(backbone=row['backbone'],corruption=row['corruption'],
            cloudfixer_correct=row['n_correct'],lion_pm_correct=prior['n_correct'],n_examples=row['n_examples'],
            cloudfixer_minus_lion_pp=100*(row['n_correct']-prior['n_correct'])/row['n_examples'],
            reference_archive_sha256=prior['sha256']))
    a.output.mkdir(parents=True,exist_ok=False)
    expected_blocks=len(units(campaign,'full',spec.get('scientific_protocol',{}).get('batch_size',32)))
    report=dict(campaign=campaign,status='validated_artifact_scope',full_campaign=blocks==expected_blocks,
        blocks=blocks,rows=rows,summary=summary,comparisons=comparisons,accepted_lion_reference=spec['baseline_results'],
        limitation='Different priors/preprocessing/RNG: whole pipelines. Scalar reference is insufficient for paired object statistics.')
    (a.output/'recount.json').write_bytes(canonical(report))
    print(json.dumps(dict(blocks=blocks,full_campaign=report['full_campaign'],output=str(a.output)),indent=2))


if __name__=='__main__':main()
