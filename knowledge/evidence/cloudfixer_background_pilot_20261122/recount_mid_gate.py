"""Recount the fixed 3%-trigger K120 PM from a completed all-condition pilot ZIP."""
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


def recount(path):
    with ZipFile(path) as archive:
        run = path.stem
        config = json.loads(archive.read(run + '/config.json'))
        assert config['phase'] == 'background' and config['execution_status'] == 'complete'
        conditions, ids, seeds = (config[k] for k in ('conditions', 'ids', 'seeds'))
        threshold = config['params']['bg_gate']['background_config']['gate_fraction']
        assert (len(conditions), len(ids), seeds, threshold) == (16, 32, [0, 1, 2], .03)
        records = {}
        for name in archive.namelist():
            if not name.startswith(run + '/raw/') or not name.endswith('.npz'):
                continue
            payload = archive.read(name)
            receipt = json.loads(archive.read(name[:-4] + '.json'))
            assert hashlib.sha256(payload).hexdigest() == receipt['sha256']
            with np.load(io.BytesIO(payload), allow_pickle=False) as z:
                rows = json.loads(z['records'].tobytes())
            for row in rows:
                key = row['condition'], row['seed'], row['id']
                assert key not in records
                arms = row['arms']
                gate = arms['bg_gate']
                triggered = gate['diagnostics']['weak_fraction'] > threshold
                reproduced = arms['bg_high' if triggered else 'identity']
                assert gate['logits'] == reproduced['logits'] and gate['prediction'] == reproduced['prediction']
                base, high = arms['identity']['prediction'], arms['bg_mid']['prediction']
                truth = row['label']
                records[key] = (triggered, base == truth, (high if triggered else base) == truth)
    assert len(records) == len(conditions) * len(seeds) * len(ids)
    groups = {name: [name] for name in conditions}
    groups.update(all15=[c for c in conditions if c != 'clean'],
                  macro13=[c for c in conditions if c not in ('clean', 'rotation', 'background')])
    weights = np.random.default_rng(20261122).multinomial(len(ids), np.full(len(ids), 1 / len(ids)),
                                                          size=5000) / len(ids)
    output = {}
    for name, group in groups.items():
        data = np.array([[[records[c, s, i] for i in ids] for s in seeds] for c in group], dtype=int)
        trigger, base, candidate = data.transpose(3, 0, 1, 2)
        delta = candidate - base
        cluster = 100 * delta.mean(axis=(0, 1))
        output[name] = dict(events=base.size, flagged=int(trigger.sum()), correct_baseline=int(base.sum()),
                            correct_mid_gate=int(candidate.sum()), repairs=int((delta > 0).sum()),
                            harms=int((delta < 0).sum()), delta_pp=float(cluster.mean()),
                            nominal_95_pp=np.quantile(weights @ cluster, [.025, .975]).tolist())
    return dict(source_archive=path.name, gate_threshold=threshold, verified_registered_gate_rows=len(records),
                conditions=output)


if __name__ == '__main__':
    print(json.dumps(recount(Path(sys.argv[1])), indent=2))
