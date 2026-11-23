"""Recount the fixed surface-confirmed t50 arm and gate candidate-only full evaluation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_cloudfixer_rotation import read_records, csv_write

ARM = 'pose_pca_surface_t50'
CONFIG = json.loads((ROOT / 'cfgs/cloudfixer_surface512.json').read_text())


def summarize(root, derived):
    config, records = read_records(root)
    phase = config['phase']
    paired = phase != 'feedback_surface_full'
    ids, seeds, conditions = (config[k] for k in ('ids', 'seeds', 'conditions'))
    if phase == 'feedback_surface512':
        assert ids == CONFIG['ids'] and seeds == [0, 1, 2] and len(conditions) == 16
    if not paired:
        assert ids == list(range(2468)) and seeds == [2] and len(conditions) == 15
    timing = set(CONFIG['timing_ids']) if phase == 'feedback_surface512' else set(ids)
    groups = {c: [c] for c in conditions}
    if len(conditions) >= 15:
        groups.update(other14=[c for c in conditions if c not in ('clean','rotation')],
                      all15=[c for c in conditions if c != 'clean'])
    table, differences = [], []
    if not paired:
        assert sum('reused_from' in row for row in records.values()) == 512 * 15
    for group, members in groups.items():
        rows = [records[c, s, i] for i in ids for c in members for s in seeds]
        cand = [r['arms'][ARM] for r in rows]
        right = np.array([a['prediction'] == r['label'] for a, r in zip(cand, rows)])
        term = sum(np.argmax(a['logits'][-1]) == r['label'] for a, r in zip(cand, rows))
        accepted = [a['diagnostics']['updates'][0]['reason'] == 'accepted' for a in cand]
        assert all(a['diagnostics']['denoiser_calls'] == 9 and a['diagnostics']['classifier_calls'] == 3 for a in cand)
        measure = [r for r in rows if r['id'] in timing]
        ms = [r['arms'][ARM]['seconds'] * 1000 for r in measure]
        p50, p95 = np.quantile(ms, [.5, .95])
        row = dict(group=group, n=len(rows), candidate=int(right.sum()), pm_pct=100*right.mean(),
                   terminal=int(term), accepted=sum(accepted), timing_n=len(ms), p50_ms=p50, p95_ms=p95,
                   peak_mib=max(a['peak_mib'] for a in cand))
        if paired:
            base = [r['arms']['identity'] for r in rows]
            before = np.array([b['prediction'] == r['label'] for r,b in zip(rows,base)])
            assert all(not b.get('from_noop') or (not apply and b['logits'] == a['logits'])
                       for b, apply, a in zip(base, accepted, cand))
            assert all('seconds' in r['arms']['identity'] for r in measure)
            times = [r['arms']['identity']['seconds'] * 1000 for r in measure]
            b50, b95 = np.quantile(times, [.5,.95])
            per_id = (right.astype(float)-before).reshape(len(ids),-1).mean(1)*100
            differences.append(per_id)
            row.update(baseline=int(before.sum()), baseline_terminal=sum(int(np.argmax(b['logits'][-1]) == r['label']) for b,r in zip(base,rows)),
                       delta_pp=per_id.mean(), repairs=int((~before & right).sum()), harms=int((before & ~right).sum()),
                       baseline_p50_ms=b50, baseline_p95_ms=b95, ratio50=p50/b50, ratio95=p95/b95)
        table.append(row)
    if paired:
        effects = np.stack(differences, axis=1)
        rng = np.random.default_rng(20261123)
        draws = np.concatenate([effects[rng.integers(len(ids),size=(100,len(ids)))].mean(1) for _ in range(100)])
        radius = np.quantile(np.abs(draws-effects.mean(0)).max(axis=1),.95)
        for j,r in enumerate(table):
            r['ci95_low'], r['ci95_high'] = np.quantile(draws[:,j],[.025,.975])
            r['joint95_low'], r['joint95_high'] = r['delta_pp']-radius,r['delta_pp']+radius
    indexed = {r['group']:r for r in table}
    derived.mkdir(parents=True,exist_ok=False)
    csv_write(root/'per_corruption.csv',[r for r in table if r['group'] in conditions])
    csv_write(root/'summary.csv',[r for r in table if r['group'] not in conditions] or table)
    if phase == 'feedback_surface512':
        checks = dict(rotation=indexed['rotation']['delta_pp']>=3, all15=indexed['all15']['delta_pp']>0,
                      protected=all(indexed[c]['delta_pp']>=0 for c in conditions if c!='rotation'),
                      speed=max(indexed['all15']['ratio50'],indexed['all15']['ratio95'])<=1.05)
        checks = {key: bool(value) for key,value in checks.items()}
        (derived/'decision.json').write_text(json.dumps(dict(checks=checks, **{'pass': all(checks.values())},
                          campaign=config['campaign'], run=config['run_id'], measurements=table),indent=2))
    print('CLOUDFIXER_ROTATION_ANALYSIS_DONE', phase, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--derived', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.derived)
