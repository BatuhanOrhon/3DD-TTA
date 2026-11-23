#!/usr/bin/env python3
"""Recount downloaded full or corruption-scoped CF results; never run inference."""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloudfixer_store import MountStore, canonical, sha, unit_key
from scripts.run_cloudfixer_multibackbone import (
    BASELINE_POLICY, BASELINE_PROFILE, CORRUPTIONS, HEADS, ORIGINAL_FIELDS,
    ORIGINAL_CF_CAMPAIGN, ORIGINAL_CF_SOURCE, ORIGINAL_PROFILE,
    candidate_probabilities, frozen_receipt, is_digest, manifest_profile, methods_for, recover, report_rows,
    seed0_digest_from_runtime, units,
)


def recount(raw, campaign, wanted, labels, *, profile=ORIGINAL_PROFILE, frozen=None,
            seed0_rng_sha256=None, predictions=None):
    if not raw.is_dir():raise ValueError('Downloaded raw receipts required')
    records = recover(MountStore(raw), campaign, wanted, labels, profile=profile, frozen=frozen,
                      seed0_rng_sha256=seed0_rng_sha256)
    if len(records) != len(wanted):
        raise ValueError('Artifact scope incomplete')
    methods = methods_for(profile)
    totals = {}
    previous = {}
    for unit in wanted:
        p = records[unit_key(unit)]['values']
        corr = unit['corruption']
        if corr in previous and str(p['rng_before_sha256']) != previous[corr]:
            raise ValueError('Broken RNG chain')
        previous[corr] = sha(canonical(json.loads(str(p['rng_after']))))
        if profile == BASELINE_PROFILE:
            readouts = (p['source_logits'], p['candidate_logits'][:, -1],
                        candidate_probabilities(p['candidate_logits']).mean(axis=1))
        else:
            readouts = tuple(p[field] for field in ORIGINAL_FIELDS)
        classes = [np.argmax(readout, axis=-1) for readout in readouts]
        for mode, method in enumerate(methods):
            for i, head in enumerate(HEADS):
                key = (method, head, corr)
                count, correct = totals.get(key, (0, 0))
                totals[key] = (count+len(unit['ids']), correct+int(np.count_nonzero(classes[mode][:, i]==p['labels'])))
        if predictions is not None:
            for i, head in enumerate(HEADS):
                scope = predictions.setdefault((corr, head), {})
                for j, object_id in enumerate(p['ids']):
                    if object_id in scope:
                        raise ValueError('Overlapping original IDs in raw shards')
                    prediction = ((int(classes[2][j, i]), int(classes[1][j, i])) if profile == BASELINE_PROFILE
                                  else (int(classes[2][j, i]),))
                    scope[int(object_id)] = (int(p['labels'][j]), *prediction)
    if profile == BASELINE_PROFILE:
        timed_rows = report_rows((records[unit_key(unit)] for unit in wanted), profile=profile)
        metadata = {(r['method'], r['backbone'], r['corruption']): r for r in timed_rows}
    else:
        metadata = {}
    return [dict(metadata.get((m,h,c), {}), method=m, backbone=h, corruption=c,
                 n_examples=n, n_correct=k, accuracy=k/n)
            for (m,h,c),(n,k) in sorted(totals.items())]


def verify_csv(path, expected, *, campaign=None):
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
        if campaign is not None:
            verify_metadata(saved, row, campaign,
                ('dataset', 'severity', 'seed', 'status'),
                ('runtime_seconds', 'reconstruction_seconds_shared', 'peak_gpu_memory_mb'))


def summarize(rows, *, profile=ORIGINAL_PROFILE):
    result=[]
    for method in methods_for(profile):
        for head in HEADS:
            group=[r for r in rows if r['method']==method and r['backbone']==head]
            n=sum(r['n_examples'] for r in group);correct=sum(r['n_correct'] for r in group)
            result.append(dict(method=method,backbone=head,total_examples=n,total_correct=correct,
                n_corruptions=len(group),micro_accuracy=correct/n,
                macro_accuracy=sum(r['accuracy'] for r in group)/len(group)))
            if profile == BASELINE_PROFILE:
                result[-1].update(dataset='modelnet40_c', severity=5, seed=0,
                    total_runtime_seconds=sum(r['runtime_seconds'] for r in group),
                    status='complete' if len(group)==15 and n==37020 else 'partial')
    return result


def verify_metadata(saved, expected, campaign, identity_fields, timing_fields):
    if saved.get('run_id') != campaign or any(saved.get(k) != str(expected[k]) for k in identity_fields):
        raise ValueError('CSV campaign or scientific scope differs from raw predictions')
    for key in timing_fields:
        if not np.isclose(float(saved[key]), expected[key], rtol=1e-12, atol=1e-12):
            raise ValueError('CSV timing differs from raw measurements')


def verify_summary(path, expected, *, campaign=None):
    actual=list(csv.DictReader(path.open()))
    lookup={(r['method'],r['backbone']):r for r in actual}
    if len(actual)!=9 or set(lookup)!={(r['method'], r['backbone']) for r in expected}:
        raise ValueError('Summary scope mismatch')
    for row in expected:
        saved=lookup[(row['method'],row['backbone'])]
        for key in ('total_examples','total_correct','n_corruptions'):
            if int(saved[key])!=row[key]:raise ValueError('Summary count mismatch')
        for key in ('micro_accuracy','macro_accuracy'):
            if not np.isclose(float(saved[key]),row[key],rtol=0,atol=1e-12):raise ValueError('Summary accuracy mismatch')
        if campaign is not None:
            verify_metadata(saved, row, campaign, ('dataset','severity','seed','status'), ('total_runtime_seconds',))


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


def paired_rows(baseline, cloudfixer=None):
    rows = []
    for (corruption, head), samples in sorted(baseline.items()):
        reference = cloudfixer.get((corruption, head)) if cloudfixer is not None else None
        if cloudfixer is not None and (reference is None or set(reference) != set(samples)):
            raise ValueError('Original CloudFixer and baseline original-ID coverage differs')
        if reference is not None and any(reference[i][0] != values[0] for i, values in samples.items()):
            raise ValueError('Original CloudFixer and baseline labels differ')
        labels = np.array([values[0] for values in samples.values()])
        pm = np.array([values[1] for values in samples.values()]) == labels
        terminal = np.array([values[2] for values in samples.values()]) == labels
        pairs = [('proximal_n8_pm_vs_proximal_n8_terminal', pm, terminal)]
        if reference is not None:
            cf = np.array([reference[i][1] for i in samples]) == labels
            pairs += [('proximal_n8_pm_vs_cloudfixer_plain', pm, cf),
                      ('proximal_n8_terminal_vs_cloudfixer_plain', terminal, cf)]
        for comparison, ours, ref in pairs:
            ours_only = int(np.count_nonzero(ours & ~ref))
            reference_only = int(np.count_nonzero(ref & ~ours))
            net = ours_only-reference_only
            rows.append(dict(comparison=comparison, corruption=corruption, backbone=head,
                n_examples=len(labels), ours_correct=int(ours.sum()), reference_correct=int(ref.sum()),
                ours_only=ours_only, reference_only=reference_only, net_correct=net,
                delta_percentage_points=100*net/len(labels)))
    return rows


def read_shard(directory, labels, label_sha, *, predictions=None, reference=None, expected_campaign=None):
    config = json.loads((directory/'config.json').read_text())
    spec = config['manifest']
    campaign = sha(canonical(spec))
    profile = manifest_profile(spec)
    baseline = profile == BASELINE_PROFILE
    if config['campaign'] != campaign or config['phase'] != 'full':
        raise ValueError('Full-phase campaign required')
    if baseline and not is_digest(expected_campaign):
        raise ValueError('Baseline analysis requires an independently expected campaign SHA256')
    if expected_campaign is not None and campaign != expected_campaign:
        raise ValueError('Downloaded run differs from expected campaign')
    if label_sha != spec['dataset_files']['label.npy']:
        raise ValueError('Label identity mismatch')
    if reference is not None and dict(bytes=len(reference),sha256=sha(reference)) != spec['baseline_reference']:
        raise ValueError('Accepted baseline identity mismatch')
    batch_size = spec.get('scientific_protocol', {}).get('batch_size', 32)
    wanted = units(campaign, 'full', batch_size, profile=profile)
    if config.get('artifact_scope') == 'single_corruption':
        wanted = [u for u in wanted if u['corruption'] == config['corruption']]
        if not wanted or (baseline and config.get('expected_objects') != 2468):
            raise ValueError('Unknown or incomplete corruption scope')
    elif baseline and config.get('artifact_scope') is not None:
        raise ValueError('Unknown baseline artifact scope')
    frozen, seed0_rng_sha256 = None, None
    if baseline:
        environment = json.loads((directory/'environment.txt').read_text())
        frozen = frozen_receipt(environment, spec)
        seed0_rng_sha256 = seed0_digest_from_runtime(environment)
    rows = recount(directory/'raw', campaign, wanted, labels, profile=profile, frozen=frozen,
                   seed0_rng_sha256=seed0_rng_sha256, predictions=predictions)
    metadata = dict(campaign=campaign) if baseline else {}
    verify_csv(directory/'per_corruption.csv', rows, **metadata)
    verify_summary(directory/'summary.csv', summarize(rows, profile=profile), **metadata)
    return dict(campaign=campaign, rows=rows, blocks=len(wanted), frozen=frozen,
                seed0_rng_sha256=seed0_rng_sha256, spec=spec)


def verify_original_archive(directory, expected):
    path = directory.parent/expected['archive']
    data = path.read_bytes()
    if len(data) != expected['bytes'] or sha(data) != expected['sha256']:
        raise ValueError('Accepted original CloudFixer archive identity mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if archive.testzip() is not None or len(set(names)) != len(names):
            raise ValueError('Accepted original CloudFixer archive failed CRC or has duplicate members')
        for name in names:
            if archive.read(name) != (directory/name).read_bytes():
                raise ValueError('Extracted original CloudFixer member differs from accepted archive: '+name)


def original_predictions(directories, labels, label_sha, spec, reference):
    from scripts.prepare_cloudfixer_multibackbone import original_cloudfixer_reference

    trusted_recount, trusted_index = original_cloudfixer_reference()
    if trusted_recount != reference:
        raise ValueError('Original CloudFixer comparison requires the accepted recount')
    index = {r['corruption']: r for r in json.loads(trusted_index)}
    if len(directories) != len(CORRUPTIONS) or set(index) != set(CORRUPTIONS):
        raise ValueError('All 15 accepted original CloudFixer extraction directories are required')
    predictions = {}
    shards = []
    seen = set()
    for directory in directories:
        config = json.loads((directory/'config.json').read_text())
        corr = config.get('corruption')
        if (corr not in index or corr in seen or config.get('artifact_scope') != 'single_corruption'
                or config.get('campaign') != ORIGINAL_CF_CAMPAIGN):
            raise ValueError('Mixed or incomplete original CloudFixer campaign scope')
        seen.add(corr)
        verify_original_archive(directory, index[corr])
        shard = read_shard(directory, labels, label_sha, predictions=predictions)
        original = shard['spec']
        environment = json.loads((directory/'environment.txt').read_text())
        if (shard['campaign'] != ORIGINAL_CF_CAMPAIGN or original['source_commit'] != ORIGINAL_CF_SOURCE
                or any(original[key] != spec[key] for key in ('dataset_files', 'head_configs', 'head_state_hashes',
                    'diffusion_sha256', 'author_commit'))
                or environment['frozen_head_hashes'] != spec['head_state_hashes']):
            raise ValueError('Original CloudFixer comparison source/head/dataset identity mismatch')
        shards.append(shard)
    actual = sorted(combine_shards(shards), key=lambda r: (r['method'], r['backbone'], r['corruption']))
    accepted = sorted(json.loads(trusted_recount)['rows'], key=lambda r: (r['method'], r['backbone'], r['corruption']))
    if actual != accepted:
        raise ValueError('Downloaded original CloudFixer raw recount differs from accepted counts')
    return predictions


def write_csv(path, rows, campaign):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['run_id', *rows[0]])
            writer.writeheader()
            writer.writerows(dict(run_id=campaign, **row) for row in rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,nargs='+',required=True,
        help='One full directory or all independently extracted corruption directories')
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--baseline-reference',type=Path,required=True)
    p.add_argument('--expected-campaign', help='SHA256 of the canonical prepared manifest, required for the baseline')
    p.add_argument('--cloudfixer-run',type=Path,nargs='+',
        help='All 15 accepted original CloudFixer extraction directories and their sibling source ZIPs')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    labels=np.load(a.labels,allow_pickle=False).reshape(-1)
    label_sha=sha(a.labels.read_bytes())
    reference=a.baseline_reference.read_bytes()
    spec=json.loads((a.run[0]/'config.json').read_text())['manifest']
    profile=manifest_profile(spec)
    baseline=profile==BASELINE_PROFILE
    if a.cloudfixer_run and not baseline:
        raise ValueError('--cloudfixer-run is available for the XYZ proximal baseline profile')
    predictions={} if baseline else None
    shards=[read_shard(directory, labels, label_sha, predictions=predictions, reference=reference,
                       expected_campaign=a.expected_campaign) for directory in a.run]
    rows=combine_shards(shards)
    summary=summarize(rows,profile=profile)
    campaign=shards[0]['campaign']
    blocks=sum(shard['blocks'] for shard in shards)
    batch_size = spec.get('scientific_protocol', {}).get('batch_size', 32)
    expected_blocks=len(units(campaign, 'full', batch_size, profile=profile))
    report=dict(campaign=campaign,status='validated_artifact_scope',full_campaign=blocks==expected_blocks,
        blocks=blocks,rows=rows,summary=summary)
    if baseline:
        if any(shard['frozen'] != shards[0]['frozen'] for shard in shards):
            raise ValueError('Mixed frozen baseline states across corruption shards')
        if any(shard['seed0_rng_sha256'] != shards[0]['seed0_rng_sha256'] for shard in shards):
            raise ValueError('Mixed registered seed 0 RNG receipts across corruption shards')
        originals=original_predictions(a.cloudfixer_run,labels,label_sha,spec,reference) if a.cloudfixer_run else None
        pairs=paired_rows(predictions,originals)
        report.update(profile=profile, policy=BASELINE_POLICY, comparisons=pairs,
            seed0_rng_sha256=shards[0]['seed0_rng_sha256'],
            original_cloudfixer_campaign=ORIGINAL_CF_CAMPAIGN if originals is not None else None,
            limitation='Single-seed descriptive comparisons. N8 PM and N8 terminal share the same geometry and noise. '
                       f'Baseline geometry RNG is seeded by original-ID blocks of up to {batch_size} objects. '
                       'Accepted original CloudFixer used batch size 64 and a different geometry pipeline and RNG stream.')
    else:
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
        report.update(comparisons=comparisons, accepted_lion_reference=spec['baseline_results'],
            limitation='Different priors/preprocessing/RNG: whole pipelines. Scalar reference is insufficient for paired object statistics.')
    a.output.mkdir(parents=True,exist_ok=False)
    if baseline:
        write_csv(a.output/'per_corruption.csv', rows, campaign)
        write_csv(a.output/'summary.csv', summary, campaign)
        write_csv(a.output/'paired_per_corruption.csv', pairs, campaign)
    (a.output/'recount.json').write_bytes(canonical(report))
    print(json.dumps(dict(blocks=blocks,full_campaign=report['full_campaign'],output=str(a.output)),indent=2))


if __name__=='__main__':main()
