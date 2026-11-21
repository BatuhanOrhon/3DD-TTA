#!/usr/bin/env python3
"""Explain registered canonical PM/terminal reversals in a previously recounted B archive."""
import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudfixer_store import canonical, sha, unit_key
from scripts.run_cloudfixer_hypotheses import registration, units
from scripts.run_cloudfixer_transfer import file_hash


CANONICAL = {8: (70, 50, 30, 0), 12: (67, 47, 27, 0)}
CONFIDENCE = .9


def contribution_row(prob, labels, config, seed):
    times = CANONICAL[config['steps']]
    means = prob.mean(axis=1)
    pm = means.argmax(-1)
    terminal = prob[:, -1].argmax(-1)
    terminal_right = terminal == labels
    row = dict(config=config, seed=seed, n=len(labels), terminal_correct=int(terminal_right.sum()),
               terminal_wrong=int((~terminal_right).sum()))
    for name, take, rival in (('harm', terminal_right & (pm != labels), pm),
                              ('repair', ~terminal_right & (pm == labels), terminal)):
        p, y, r = prob[take], labels[take], rival[take]
        i = np.arange(len(y))
        slots = np.arange(len(times))
        margins = p[i[:,None], slots, y[:,None]] - p[i[:,None], slots, r[:,None]]
        pm_margin = means[take][i,y] - means[take][i,r]
        if not np.allclose(margins.mean(axis=1), pm_margin, atol=8*np.finfo(np.float64).eps, rtol=0):
            raise ValueError('Probe contributions do not reproduce the canonical PM margin')
        pm_tie = pm_margin == 0
        terminal_tie = margins[:, -1] == 0
        tied = pm_tie | terminal_tie
        pred = p.argmax(-1)
        confident_wrong = (p.max(-1) >= CONFIDENCE) & (pred != y[:,None])
        row[name] = dict(count=len(y), strict=int((~tied).sum()), tie=int(tied.sum()),
                         pm_tie=int(pm_tie.sum()), terminal_tie=int(terminal_tie.sum()), probes=[
            dict(native=t, mean_true_minus_rival=float(margins[:,j].mean()) if len(y) else None,
                 top1_true=int((pred[:,j] == y).sum()), top1_rival=int((pred[:,j] == r).sum()),
                 confident_wrong=int(confident_wrong[:,j].sum())) for j,t in enumerate(times)])
    return row


def canonical_contributions(records):
    """Aggregate already validated B unit records; retain just the four registered probability vectors."""
    groups = {}
    for record in records:
        unit = record['unit']
        config = unit['config']
        with np.load(io.BytesIO(record['data']), allow_pickle=False) as p:
            times = json.loads(str(p['sampler']))['candidate_indices']
            columns = [times.index(t) for t in CANONICAL[config['steps']]]
            saved = p['probabilities']
            if saved.dtype != np.float64:
                raise ValueError('Expected saved float64 probabilities')
            groups.setdefault(canonical(config), {}).setdefault(unit['seed'], []).append(
                (saved[:,columns], p['labels']))
    result = []
    for key, seeds in groups.items():
        config = json.loads(key)
        pooled = []
        for seed, batches in sorted(seeds.items()):
            p, labels = (np.concatenate(parts) for parts in zip(*batches))
            result.append(contribution_row(p, labels, config, seed))
            pooled.append((p, labels))
        p, labels = (np.concatenate(parts) for parts in zip(*pooled))
        result.append(contribution_row(p, labels, config, 'pooled'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--verified-sha256', required=True, help='Exact raw SHA from the successful independent B recount')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report path')
    args = parser.parse_args()
    if file_hash(args.archive) != args.verified_sha256:
        raise ValueError('Archive differs from the accepted B recount SHA')
    reg = registration()
    with zipfile.ZipFile(args.archive) as z:
        config_name, = [n for n in z.namelist() if PurePosixPath(n).name == 'config.json']
        prefix = config_name[:-len('config.json')]
        cfg = json.loads(z.read(config_name))
        manifest = cfg['manifest']
        campaign = sha(canonical(dict(manifest=manifest, role=reg['role'])))
        if cfg['execution_status'] != 'complete' or cfg['campaign'] != campaign or manifest['registration'] != reg:
            raise ValueError('Expected a completed registered B campaign')
        records = (dict(unit=u, data=z.read(prefix+'pending_'+unit_key(u)+'.npz')) for u in units(campaign,reg))
        rows = canonical_contributions(records)
        paired = {(r['schedule'],r['seeds']): r for r in csv.DictReader(io.StringIO(z.read(prefix+'paired.csv').decode()))
                  if r['method'] == 'canonical_equal_pm' and r['reference'] == 'terminal'}
        for row in rows:
            cfg = row['config']
            ref = paired['N'+str(cfg['steps'])+'_tau'+str(cfg['tau']), str(row['seed'])]
            if (row['n'], row['terminal_correct'], row['harm']['count'], row['repair']['count']) != tuple(
                    int(ref[k]) for k in ('n','reference_correct','harm','repair')):
                raise ValueError('Contribution counts differ from standard paired recount')
    report = dict(campaign=campaign, archive_sha256=args.verified_sha256, confidence_threshold=CONFIDENCE,
        population='One observation per original object, corruption and seed. Pooled rows concatenate seeds.',
        denominators='Harm is eligible among terminal_correct; repair among terminal_wrong. Probe counts and means condition on the named group.',
        contribution='Truth probability minus the same rival probability at every probe. Rival is the PM winner on harms and terminal winner on repairs.',
        interpretation='The average of the four margins equals the PM margin algebraically. Probe sizes and counts are empirical. They describe voting on a fixed trajectory and cannot establish a cause of geometric transport.',
        ties='First-index argmax. Tie means exact float64 equality of truth and fixed rival at PM or terminal; strict requires both preferences to be strict.',
        rows=rows)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
