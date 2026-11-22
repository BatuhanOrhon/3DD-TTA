"""Geometry-only check of the fixed proposal in frame_design.md on 16 development objects."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def rotation(direction, degrees):
    x, y, z = direction / np.linalg.norm(direction)
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.deg2rad(degrees)
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def angle(matrix):
    return np.rad2deg(np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1., 1.)))


def normal_patches(cloud):
    points = np.unique(cloud.astype(np.float64), axis=0)
    rng = np.random.default_rng(20261122)
    centers = rng.permutation(len(points))[:256]
    delta = points[None, :, :] - points[centers, None, :]
    distances = (delta * delta).sum(axis=-1)
    distances[np.arange(len(centers)), centers] = np.inf
    nearest = np.argpartition(distances, 15, axis=1)[:, :16]
    patches = points[nearest]
    patches -= patches.mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(patches.transpose(0, 2, 1) @ patches)
    weights = np.maximum(values[:, 1] - values[:, 0], 0.) / np.maximum(values[:, 2], 1e-12)
    return vectors[:, :, 0], weights


def select(normals, weights, candidates):
    fitted = np.abs(normals @ candidates).max(axis=-1) ** 2
    halves = np.array_split(np.arange(len(normals)), 2)
    if any(weights[h].sum() <= 0. for h in halves):
        return 0
    scores = np.stack([np.average(fitted[:, h], axis=1, weights=weights[h]) for h in halves])
    left, right = candidates[scores.argmax(axis=1)]
    best = np.average(fitted, axis=1, weights=weights).argmax()
    if angle(left.T @ right) <= 10. + 1e-8 and min(scores[:, best] - scores[:, 0]) > .015:
        return int(best)
    return 0


def check(path):
    ids = json.loads((ROOT / 'cfgs/cloudfixer_rotation_pilot.json').read_text())['development_ids']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = ROOT / 'knowledge/evidence/cloudfixer_rotation_pilot_20261121/archive_receipt.json'
    assert digest == json.loads(receipt.read_text())['clean_sha256']
    clouds = np.load(path, allow_pickle=False)
    assert clouds.shape == (2468, 1024, 3) and len(ids) == 16
    axis = np.eye(3)
    directions = [np.asarray(v) for v in itertools.product((-1, 0, 1), repeat=3)
                  if next((a for a in v if a), 0) == 1]
    candidates = np.array([np.eye(3)] + [rotation(v, a) for v in directions for a in (-30,-20,-10,10,20,30)])
    assert len(candidates) == 79 and np.allclose(np.linalg.det(candidates), 1.)
    assert np.allclose(candidates.transpose(0, 2, 1) @ candidates, np.eye(3))
    poses = {'identity': np.eye(3), 'small_x': rotation(axis[0], 5),
             'compound': rotation(axis[0], 15) @ rotation(axis[1], -15) @ rotation(axis[2], 15),
             'large_y': rotation(axis[1], 60),
             'off_grid': rotation(axis[0], -8) @ rotation(axis[1], 21) @ rotation(axis[2], 11)}
    results = {key: [] for key in poses}
    for obj in ids:
        normals, weights = normal_patches(clouds[obj])
        for key, pose in poses.items():
            chosen = select(normals @ pose, weights, candidates)
            residual = angle(pose @ candidates[chosen])
            results[key].append({'id': obj, 'applied': chosen != 0, 'residual_degrees': float(residual)})
    summary = {}
    for key, values in results.items():
        before = angle(poses[key])
        after = np.array([r['residual_degrees'] for r in values])
        summary[key] = dict(before=float(before), mean_after=float(after.mean()),
                            applied=sum(x['applied'] for x in values), better=int((after < before-1e-6).sum()),
                            worse=int((after > before+1e-6).sum()), tied=int((np.abs(after-before)<=1e-6).sum()))
    return dict(clean_sha256=digest, summary=summary, objects=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clean_npy', type=Path)
    print(json.dumps(check(parser.parse_args().clean_npy), indent=2))
