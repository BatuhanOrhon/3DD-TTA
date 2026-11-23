"""Replay three fixed t50 acceptance rules against the original per-object PM logits."""
import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

EVIDENCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVIDENCE / 'cloudfixer_rotation_refocus_20261122'))
from frame_check import normal_patches, rotation

ARCHIVES = {
    'feedback_all15': 'a2cf85604c94477ec3671958bc0ba534bbac4a2256714ca3a1785a818638e153',
    'feedback_focus': 'f17ea646443b883f68f6744d2149e78a9b120f38c3887eac026f64403ccc501a',
}
CLEAN_SHA = '8aa5000d74a7ee7c3f827d9e80a1b325a12544a01b23aa2c76f990c0109a260c'
X, Y, Z = np.eye(3)
FRAME = rotation(X, -90)
IDENTITY = np.eye(3)
CENTERS = []
for a, b, c in itertools.product((-15, 15), repeat=3):
    corruption = rotation(X, a) @ rotation(Y, b) @ rotation(Z, c)
    undo = FRAME.T @ corruption.T @ FRAME
    assert np.allclose(corruption @ FRAME @ undo, FRAME)
    CENTERS.append(undo)
CENTER_TRACE = 1 + 2 * np.cos(np.deg2rad(7.5))
AXIS_SUPPORT = np.cos(np.deg2rad(10))
POLICIES = ('identity', 'pca', 'normals', 'two_planes', 'generator')


def surface_scores(normals, weights, pose):
    alignment = np.abs(normals @ pose)
    scores = []
    supported = np.ones(3, dtype=bool)
    for half in np.array_split(np.arange(len(normals)), 2):
        w = weights[half]
        if w.sum() <= 0:
            return np.full(2, -np.inf), False
        scores.append(np.average(alignment[half].max(axis=1) ** 2, weights=w))
        supported &= np.average(alignment[half] >= AXIS_SUPPORT, axis=0, weights=w) >= .10
    return np.array(scores), np.count_nonzero(supported) >= 2


def choices(update, points, cache, ident):
    if update['reason'] != 'accepted':
        return (False,) * len(POLICIES)
    pose = np.asarray(update['pose'])
    assert np.allclose(pose.T @ pose, IDENTITY, atol=1e-5) and np.isclose(np.linalg.det(pose), 1)
    generator = np.max(np.einsum('kij,ij->k', CENTERS, pose)) >= CENTER_TRACE
    if ident not in cache:
        cloud = points[ident, :, :3]
        centered = cloud - cloud.mean(axis=0, keepdims=True)
        observed = ((centered / centered.std()) @ FRAME).astype(np.float32)
        normals, weights = normal_patches(observed)
        initial, axes_before = surface_scores(normals, weights, IDENTITY)
        cache[ident] = normals, weights, initial, axes_before
    normals, weights, initial, axes_before = cache[ident]
    final, axes_after = surface_scores(normals, weights, pose)
    gain = np.isfinite(initial).all() and np.all(final > initial + .015)
    return False, True, gain, gain and axes_after and not axes_before, generator


def correctness(arm, label):
    logits = np.asarray(arm['logits'])
    assert logits.shape == (3, 40) and np.isfinite(logits).all()
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    pm = (p / p.sum(axis=1, keepdims=True)).mean(axis=0)
    assert pm.argmax() == arm['prediction'] and np.allclose(pm, arm['probabilities'])
    return pm.argmax() == label


def archive_rows(path):
    with zipfile.ZipFile(path) as archive:
        def read(name):
            return archive.read(path.stem + '/' + name)
        cfg = json.loads(read('config.json'))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ARCHIVES[cfg['phase']]
        assert cfg['execution_status'] == 'complete' and len(cfg['ids']) == 128
        rows = {}
        for i in range(1, cfg['blocks'] + 1):
            key = f'raw/block_{i:04d}'
            data = read(key + '.npz')
            assert hashlib.sha256(data).hexdigest() == json.loads(read(key + '.json'))['sha256']
            with np.load(io.BytesIO(data), allow_pickle=False) as blob:
                for r in json.loads(blob['records'].tobytes()):
                    key = r['id'], r['condition'], r['seed']
                    assert key not in rows
                    rows[key] = r
        assert set(rows) == set(itertools.product(cfg['ids'], cfg['conditions'], cfg['seeds']))
        return cfg, rows


def report(take, base, candidate, draws):
    # Each row is one object. Keep all its conditions and seeds together.
    right = np.where(take, candidate, base)
    repairs = right & ~base
    harms = ~right & base
    delta = (repairs.astype(float) - harms).mean(axis=(1, 2)) * 100
    return dict(n=base.size, accepted=int(take.sum()), correct=int(right.sum()),
                repairs=int(repairs.sum()), harms=int(harms.sum()), delta_pp=round(delta.mean(), 6),
                nominal_ci_pp=np.quantile(delta[draws].mean(axis=1), [.025, .975]).round(4).tolist())


def replay(path, dataset):
    cfg, rows = archive_rows(path)
    ids, conditions, seeds = (cfg[k] for k in ('ids', 'conditions', 'seeds'))
    base = np.zeros((len(ids), len(conditions), len(seeds)), dtype=bool)
    candidate = np.zeros_like(base)
    accepted = np.zeros((len(POLICIES), *base.shape), dtype=bool)
    for j, cond in enumerate(conditions):
        name = 'data_original.npy' if cond == 'clean' else f'data_{cond}_5.npy'
        source = dataset / name
        digest = CLEAN_SHA if cond == 'clean' else cfg['spec']['dataset_files'][name]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        points = np.load(source, mmap_mode='r', allow_pickle=False)
        assert points.shape[0] == 2468 and points.shape[2] >= 3
        cache = {}
        for i, ident in enumerate(ids):
            for k, seed in enumerate(seeds):
                row = rows[ident, cond, seed]
                baseline, pca = (row['arms'][arm] for arm in ('identity', 'pose_pca_t50'))
                update, = pca['diagnostics']['updates']
                assert update['index'] == 50
                if update['reason'] != 'accepted':
                    assert pca['logits'] == baseline['logits']
                accepted[:, i, j, k] = choices(update, points, cache, ident)
                base[i, j, k] = correctness(baseline, row['label'])
                candidate[i, j, k] = correctness(pca, row['label'])
    groups = {c: [i] for i, c in enumerate(conditions)}
    extra = {}
    if cfg['phase'] == 'feedback_all15':
        others = [i for i, c in enumerate(conditions) if c != 'rotation']
        groups.update(other14=others, all15=list(range(15)))
        harmful = (base[:, others] & ~candidate[:, others]).sum(axis=(1, 2))
        ri = conditions.index('rotation')
        fixes = (~base[:, ri] & candidate[:, ri]).any(axis=1)
        extra = dict(objects_harmed_other14=int((harmful > 0).sum()),
                     harms_on_top_five=int(np.sort(harmful)[-5:].sum()),
                     harms_on_top_ten=int(np.sort(harmful)[-10:].sum()),
                     rotation_repairs_with_other_harm=int((fixes & (harmful > 0)).sum()))
    draws = np.random.default_rng(20261121).integers(len(ids), size=(10000, len(ids)))
    table = {group: {p: report(accepted[ix][:, js], base[:, js], candidate[:, js], draws)
                     for ix, p in enumerate(POLICIES)} for group, js in groups.items()}
    return cfg, dict(archive_sha256=ARCHIVES[cfg['phase']], table=table, overlap=extra)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    for arg in ('dataset', 'all15_zip', 'focus_zip'):
        args.add_argument(arg, type=Path)
    a = args.parse_args()
    ca, all15 = replay(a.all15_zip, a.dataset)
    cf, focus = replay(a.focus_zip, a.dataset)
    assert set(ca['ids']).isdisjoint(cf['ids']) and ca['seeds'] == [0] and cf['seeds'] == [0, 1, 2]
    assert all15['table']['all15']['pca']['correct'] == 1410 and focus['table']['clean']['pca']['correct'] == 341
    print(json.dumps(dict(all15=all15, focus=focus), indent=2))
