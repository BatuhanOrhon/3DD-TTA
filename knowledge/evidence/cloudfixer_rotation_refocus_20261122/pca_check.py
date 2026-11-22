"""Paired geometry-only whole-cloud PCA check. Use the clean NPY from the frame check."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from frame_check import ROOT, angle, rotation


IDENTITY = np.eye(3)
AXES = np.array([IDENTITY[:, perm] * signs for perm in itertools.permutations(range(3))
                 for signs in itertools.product((-1, 1), repeat=3)
                 if np.linalg.det(IDENTITY[:, perm] * signs) > 0])


def estimate(points):
    centered = points - points.mean(axis=0)
    values, basis = np.linalg.eigh(centered.T @ centered)
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    candidates = basis @ AXES
    # Maximum trace selects the proper diagonalizing rotation nearest identity.
    best = candidates[np.trace(candidates, axis1=1, axis2=2).argmax()]
    gap = np.min(np.diff(values)) / max(values[-1], 1e-12)
    return best, gap


def check(path):
    previous = json.loads(Path(__file__).with_name('frame_check.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == previous['clean_sha256']
    clouds = np.load(path, allow_pickle=False)
    ids = json.loads((ROOT / 'cfgs/cloudfixer_rotation_pilot.json').read_text())['development_ids']
    assert clouds.shape == (2468, 1024, 3) and len(ids) == 16
    x, y, z = IDENTITY
    poses = dict(identity=IDENTITY, small_x=rotation(x, 5),
                 compound=rotation(x, 15) @ rotation(y, -15) @ rotation(z, 15),
                 large_y=rotation(y, 60), off_grid=rotation(x, -8) @ rotation(y, 21) @ rotation(z, 11))
    assert len(AXES) == 24 and np.allclose(np.linalg.det(AXES), 1)
    box = np.array(list(itertools.product((-3, 3), (-2, 2), (-1, 1))))
    assert np.allclose(estimate(box @ poses['compound'] + 7)[0], poses['compound'].T)
    assert estimate(np.zeros((8, 3)))[1] == 0
    rows = {name: [] for name in poses}
    for obj in ids:
        points = np.unique(clouds[obj].astype(np.float64), axis=0)
        halves = np.array_split(np.random.default_rng(20261122).permutation(len(points)), 2)
        for name, pose in poses.items():
            rotated = points @ pose
            raw, gap = estimate(rotated)
            left, right = [estimate(rotated[h])[0] for h in halves]
            applied = bool(5 <= angle(raw) <= 30 and gap >= .05 and angle(left.T @ right) <= 10)
            rows[name].append(dict(id=obj, applied=applied, raw=float(angle(pose @ raw)),
                                   gated=float(angle(pose @ raw) if applied else angle(pose))))
    summary = {}
    for name, entries in rows.items():
        before = angle(poses[name])
        assert np.isclose(before, previous['summary'][name]['before'])
        assert [e['id'] for e in entries] == [e['id'] for e in previous['objects'][name]]
        after = np.array([e['gated'] for e in entries])
        summary[name] = dict(before=float(before), raw=float(np.mean([e['raw'] for e in entries])),
                             gated=float(after.mean()), applied=sum(e['applied'] for e in entries),
                             better=int((after < before - 1e-6).sum()), worse=int((after > before + 1e-6).sum()))
    return dict(clean_sha256=digest, summary=summary, objects=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clean_npy', type=Path)
    print(json.dumps(check(parser.parse_args().clean_npy), indent=2))
