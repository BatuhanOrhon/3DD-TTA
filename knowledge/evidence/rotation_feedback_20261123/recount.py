"""Recount archived feedback rows with paired IDs, equal-seed weighting and B1 timings."""
import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np


def recount(path):
    run = path.stem
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        assert len({m.filename for m in members}) == len(members)
        for m in members:
            name = PurePosixPath(m.filename)
            assert not name.is_absolute() and '..' not in name.parts and name.parts[0] in (run, run+'_derived')
            assert (m.external_attr >> 16) & 0o170000 != 0o120000
        assert archive.testzip() is None
        def read(name):
            return archive.read(run+'/'+name)
        assert json.loads(read('exit_status.json'))['returncode'] == 0
        config = json.loads(read('config.json'))
        assert config['execution_status'] == 'complete' and config['run_id'] == run
        for required in ('command.txt','notes.md','environment.txt','stdout.log','per_corruption.csv','summary.csv'):
            assert read(required).strip()
        ids, seeds, conditions = (config[k] for k in ('ids','seeds','conditions'))
        arms = list(config['params'])
        assert all(len(v) == len(set(v)) for v in (ids,seeds,conditions,arms))
        rows = {}
        for block in range(1, config['blocks']+1):
            key = f'raw/block_{block:04d}'
            receipt = json.loads(read(key+'.json'))
            data = read(key+'.npz')
            assert len(data) == receipt['bytes'] and hashlib.sha256(data).hexdigest() == receipt['sha256']
            unit = receipt['unit']
            assert unit['campaign'] == config['campaign']
            with np.load(io.BytesIO(data),allow_pickle=False) as blob:
                records = json.loads(blob['records'].tobytes())
            assert [r['id'] for r in records] == unit['ids']
            for r in records:
                assert (r['condition'],r['seed']) == (unit['condition'],unit['seed'])
                k = r['id'],r['condition'],r['seed']
                assert k not in rows and set(r['arms']) == set(arms)
                rows[k] = r
        assert set(rows) == set(itertools.product(ids,conditions,seeds))
        paired = list(csv.DictReader(io.StringIO(archive.read(run+'_derived/paired.csv').decode())))
        table = {}
        for cond in conditions:
            group = [rows[i,cond,s] for i in ids for s in seeds]
            gt = np.array([r['label'] for r in group])
            baseline = np.array([r['arms']['identity']['prediction'] for r in group]) == gt
            t = np.array([r['arms']['identity']['seconds'] for r in group])
            table[cond] = {}
            for arm in arms:
                entries = [r['arms'][arm] for r in group]
                logits = np.array([e['logits'] for e in entries])
                assert logits.shape == (len(gt),3,40) and np.isfinite(logits).all()
                prob = np.exp(logits-logits.max(-1,keepdims=True))
                pm = (prob/prob.sum(-1,keepdims=True)).mean(1)
                assert np.allclose(pm, [e['probabilities'] for e in entries], atol=1e-12)
                correct = pm.argmax(-1) == gt
                assert np.array_equal(pm.argmax(-1), [e['prediction'] for e in entries])
                diag=[e['diagnostics'] for e in entries]
                assert all((d['denoiser_calls'],d['classifier_calls'])==(9,3) for d in diag)
                updates=[d.get('updates',[]) for d in diag]
                if arm.startswith('pose_'):
                    expected=[50] if arm.endswith('t50') else list(range(80,0,-10))
                    for series in updates:
                        assert [u['index'] for u in series] == expected
                        assert series[-1]['travel_degrees'] <= 30.0001
                        assert all(u['score_after'] < u['score_before'] for u in series
                                   if 'scd' in arm and u['reason']=='accepted')
                ms=np.quantile([e['seconds'] for e in entries],[.5,.95])*1000
                effect=next((m for m in paired if m['group']==cond and m['arm']==arm and m['reference']=='identity'),None)
                if effect:
                    assert int(effect['candidate_correct'])==int(correct.sum()) and int(effect['baseline_correct'])==int(baseline.sum())
                table[cond][arm]=dict(n=len(gt),pm=int(correct.sum()),terminal=int((logits[:,-1].argmax(-1)==gt).sum()),
                    repairs=int((correct & ~baseline).sum()), harms=int((baseline & ~correct).sum()),
                    p50_ms=float(ms[0]), p95_ms=float(ms[1]), p95_ratio=float(ms[1]/(np.quantile(t,.95)*1000)),
                    any_update=sum(any(u['reason']=='accepted' for u in series) for series in updates),
                    final_angle_deg=float(np.mean([s[-1]['total_degrees'] if s else 0 for s in updates])),
                    simultaneous_ci_pp=([float(effect[k]) for k in ('simultaneous_lower_pp','simultaneous_upper_pp')]
                                        if effect else None))
        result=dict(run=run, archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),blocks=config['blocks'],
                    objects=len(ids),events=len(rows),method_events=len(rows)*len(arms),table=table)
        for key in ('decision','synthetic'):
            name=run+'_derived/'+key+'.json'
            if name in archive.namelist():
                result[key]=json.loads(archive.read(name))
        return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    print(json.dumps(recount(parser.parse_args().archive),indent=2))
