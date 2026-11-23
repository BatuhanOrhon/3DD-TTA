"""Check the simple normal rule with the older two-candidate selector and time its CPU work."""
import argparse
import json
import platform
from pathlib import Path
import subprocess
import time

import numpy as np

from replay import FRAME, IDENTITY, archive_rows, normal_patches
from frame_check import select


def measure(archive, dataset, evidence):
    cfg, rows = archive_rows(archive)
    assert cfg['phase'] == 'feedback_all15'
    clouds = {c: np.load(dataset / f'data_{c}_5.npy', mmap_mode='r') for c in cfg['conditions']}
    keys = sorted(rows)
    order = np.random.default_rng(20261123).permutation(len(keys))
    summary = {c: [0, 0, 0, 0] for c in cfg['conditions']}
    costs, conditional = [], []
    for ix in order:
        key = keys[ix]
        r = rows[key]
        base, pca = (r['arms'][k] for k in ('identity', 'pose_pca_t50'))
        update, = pca['diagnostics']['updates']
        start = time.perf_counter()
        apply = False
        if update['reason'] == 'accepted':
            xyz = clouds[key[1]][key[0], :, :3]
            centered = xyz - xyz.mean(0)
            obs = np.dot(centered / np.std(centered), FRAME).astype(np.float32)
            normals, weights = normal_patches(obs)
            apply = bool(select(normals, weights, np.stack([IDENTITY, update['pose']])))
        ms = 1000 * (time.perf_counter() - start)
        costs.append(ms)
        if update['reason'] == 'accepted':
            conditional.append(ms)
        ref = base['prediction'] == r['label']
        right = (pca if apply else base)['prediction'] == r['label']
        values = summary[key[1]]
        for j, event in enumerate((apply, right, right and not ref, ref and not right)):
            values[j] += int(event)
    for c, counts in summary.items():
        row = evidence['all15']['table'][c]['normals']
        assert counts == [row[k] for k in ('accepted', 'correct', 'repairs', 'harms')]
    assert len(conditional) == evidence['all15']['table']['all15']['pca']['accepted']
    def stats(times):
        return dict(calls=len(times), mean_ms=np.mean(times), p50_ms=np.median(times),
                    p95_ms=np.quantile(times, .95))
    cpu = platform.processor()
    if platform.system() == 'Darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    return dict(cpu=cpu, numpy=np.__version__, with_candidate=stats(conditional), mixed=stats(costs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('all15_zip', 'dataset', 'evidence'):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    print(json.dumps(measure(a.all15_zip, a.dataset, json.loads(a.evidence.read_text())), indent=2))
