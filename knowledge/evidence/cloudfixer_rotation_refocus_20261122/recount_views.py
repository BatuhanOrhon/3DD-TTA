"""Count label-assisted coverage of seven archived t20 poses. No inference or fitting."""
import argparse
import hashlib
import io
import itertools
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def recount(path):
    receipt = json.loads((Path(__file__).resolve().parents[1] / 'cloudfixer_rotation_pilot_20261121' /
                          'archive_receipt.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != receipt['archive_sha256']:
        raise ValueError('Unexpected validation archive')
    results = {}
    seen = set()
    conditions = ('rotation', 'clean')
    root = receipt['run_id']
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not (name.startswith(root + '/raw/block_') and name.endswith('.json')):
                continue
            block = json.loads(archive.read(name))
            unit = block['unit']
            if unit['condition'] not in conditions:
                continue
            data = archive.read(name[:-5] + '.npz')
            assert hashlib.sha256(data).hexdigest() == block['sha256']
            with np.load(io.BytesIO(data), allow_pickle=False) as payload:
                rows = json.loads(payload['records'].tobytes())
            assert [row['id'] for row in rows] == unit['ids']
            for row in rows:
                condition, seed = row['condition'], row['seed']
                key = (condition, seed, row['id'])
                assert key not in seen and (condition, seed) == (unit['condition'], unit['seed'])
                seen.add(key)
                arm = row['arms']['output']
                options = np.asarray(arm['diagnostics']['option_logits'])
                choice = arm['diagnostics']['choice']
                assert options.shape == (7, 40) and np.isfinite(options).all() and 0 <= choice < 7
                assert np.array_equal(options[0], row['arms']['identity']['logits'][0])
                assert np.array_equal(options[choice], arm['logits'][0])
                matches = options.argmax(axis=1) == row['label']
                for group in ('pooled', str(seed)):
                    counts = results.setdefault(condition, {}).setdefault(group, Counter())
                    counts.update(events=1, identity=int(matches[0]), selected=int(matches[choice]),
                                  any_correct=int(matches.any()), all_wrong=int(not matches.any()),
                                  repairs=int(not matches[0] and matches[choice]),
                                  harms=int(matches[0] and not matches[choice]))
    assert seen == set(itertools.product(conditions, receipt['seeds'], receipt['objects']))
    for groups in results.values():
        for c in groups.values():
            assert c['identity'] + c['repairs'] - c['harms'] == c['selected']
            assert c['any_correct'] + c['all_wrong'] == c['events']
    return dict(archive_sha256=digest, readout='t20 top-1, identity and six signed 20-degree axis rotations',
                counts=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('validation_zip', type=Path)
    print(json.dumps(recount(parser.parse_args().validation_zip), indent=2, sort_keys=True))
