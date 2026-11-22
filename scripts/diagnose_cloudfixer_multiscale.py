"""Audit frozen support weights with known point provenance on reserved clean IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_background import anchor_weights


def diagnosed_clouds(points, seed):
    rng = torch.Generator().manual_seed(seed)
    lo, hi = points.amin(0), points.amax(0)
    extent = hi - lo
    count = len(points) // 10
    uniform = lo + torch.rand((count, 3), generator=rng) * extent
    compact = hi + extent * (.1 + .02 * torch.randn((count, 3), generator=rng))
    n_groups = max(2, count // 8)
    centers = lo + torch.rand((n_groups, 3), generator=rng) * extent
    groups = centers[torch.arange(count) % n_groups] + .008 * extent * torch.randn((count, 3), generator=rng)
    # A narrow, attached, two-sided strip of known inliers along the x axis.
    anchor = points[points[:, 0].argmax()]
    length = torch.linspace(0, .4, count // 2 + 1)[1:].repeat_interleave(2)
    rod = anchor.repeat(len(length), 1)
    rod[:, 0] += extent.max() * length
    rod[:, 1] += .004 * extent.max() * torch.tensor([-1., 1.]).repeat(len(length) // 2)
    return dict(uniform=uniform, compact=compact, small_clusters=groups, thin_strip=rod)


def run(path):
    cfg = json.loads((ROOT/'cfgs/cloudfixer_background_followup.json').read_text())
    base = json.loads((ROOT/'cfgs/cloudfixer_rotation_pilot.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != base['clean_sha256']:
        raise ValueError('Unexpected clean ModelNet40 array')
    data = np.load(path, allow_pickle=False)
    rows = []
    for ident in cfg['geometry_ids']:
        points = torch.unique(torch.from_numpy(data[ident, :, :3].copy()).float(), dim=0)
        for case, extra in diagnosed_clouds(points, ident).items():
            x = torch.cat((points, extra)).unsqueeze(0)
            mask = torch.ones_like(x[..., 0], dtype=torch.bool)
            for name, wide in [('k5', None), ('k5_k16', cfg['wide_k'])]:
                weights = anchor_weights(x, mask, 5, wide)
                rotated = anchor_weights(x[..., [1, 2, 0]], mask, 5, wide)
                error = float((weights - rotated).abs().max())
                if error > 1e-4 or not torch.isfinite(weights).all():
                    raise ValueError('Rigid permutation or finite check failed')
                weak = weights[0] < .5
                rows.append(dict(id=ident, case=case, method=name, inliers=len(points), added=len(extra),
                                 weak_inliers=int(weak[:len(points)].sum()), weak_added=int(weak[len(points):].sum()),
                                 rotation_max_abs=error))
    return dict(clean_sha256=digest, ids=cfg['geometry_ids'], rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clean', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with args.output.open('x') as f:
        json.dump(run(args.clean), f, separators=(',', ':'))
        f.write('\n')
