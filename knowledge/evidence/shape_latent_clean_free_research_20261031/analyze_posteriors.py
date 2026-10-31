"""Read accepted oracle caches; descriptive algebra only, no model or selection."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(array):
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode()
                          + array.tobytes()).hexdigest()


def components(mu_dirty, sigma_dirty, mu_clean, sigma_clean, eps):
    mean = mu_clean - mu_dirty
    scale = sigma_clean - sigma_dirty
    realized_scale = scale * eps
    squared = lambda x: np.square(x).sum(axis=1)
    return dict(mean_energy=squared(mean), scale_energy=squared(scale),
                realized_scale_energy=squared(realized_scale),
                realized_cross=2 * (mean * realized_scale).sum(axis=1),
                realized_total=squared(mean + realized_scale))


def self_check():
    x = np.array([[1., 2.], [3., 4.]])
    result = components(x, x, x + 1, x + 2, np.array([[1., -1.], [0., 0.]]))
    np.testing.assert_array_equal(result['mean_energy'], [2, 2])
    np.testing.assert_array_equal(result['scale_energy'], [8, 8])
    np.testing.assert_array_equal(result['realized_total'], [10, 2])
    np.testing.assert_allclose(result['realized_total'], result['mean_energy']
                              + result['realized_scale_energy'] + result['realized_cross'])


def analyze(phase, output):
    phase, output = Path(phase), Path(output)
    if output.exists():
        raise FileExistsError('Keep previous derived analyses intact: ' + str(output))
    root = phase / 'paired/common/modelnet40_c'
    expected = {'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
                'distortion_rbf', 'distortion_rbf_inv', 'density', 'density_inc',
                'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar'}
    rows, files = {}, []
    for path in sorted(root.glob('*/*.oracle.npz')):
        with np.load(path, allow_pickle=False) as saved, np.load(
                path.with_name(path.name.replace('.oracle.npz', '.npz')),
                allow_pickle=False) as base:
            metadata = json.loads(saved['metadata'].item())
            assert all(digest(saved[k]) == v for k, v in metadata['tensor_hashes'].items())
            assert all(digest(base[k]) == v for k, v in
                       json.loads(base['metadata'].item())['tensor_hashes'].items())
            dirty_mu, dirty_sigma, clean_mu, clean_sigma, eps = [saved[k].astype(np.float64)
                for k in ['z_corrupt_mu', 'z_corrupt_sigma', 'z_clean_mu',
                          'z_clean_sigma', 'eps_z']]
            assert dirty_mu.shape == (32, 128)
            assert all(np.isfinite(x).all() for x in [dirty_mu, dirty_sigma, clean_mu, clean_sigma, eps])
            assert (dirty_sigma > 0).all() and (clean_sigma > 0).all()
            dirty_z = base['shape_latent'].reshape(32, 128).astype(np.float64)
            clean_z = saved['clean_z'].astype(np.float64)
            np.testing.assert_allclose(dirty_z, dirty_mu + dirty_sigma * eps, atol=8e-7, rtol=1e-6)
            np.testing.assert_allclose(clean_z, clean_mu + clean_sigma * eps, atol=8e-7, rtol=1e-6)
            row = components(dirty_mu, dirty_sigma, clean_mu, clean_sigma, eps)
            np.testing.assert_allclose(row['realized_total'], row['mean_energy']
                + row['realized_scale_energy'] + row['realized_cross'], atol=1e-12)
            row['relative_clean_displacement'] = np.linalg.norm(clean_z-dirty_z, axis=1) / np.linalg.norm(dirty_z, axis=1)
            row['dirty_sigma_norm'] = np.linalg.norm(dirty_sigma, axis=1)
            row['dirty_mu_norm'] = np.linalg.norm(dirty_mu, axis=1)
            rows.setdefault(path.parent.name, []).append(row)
            files.append(dict(path=str(path.relative_to(phase)),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert set(rows) == expected and len(files) == 30
    summaries, all_rows = {}, []
    def summarize(row):
        w2 = row['mean_energy'] + row['scale_energy']
        return dict(objects=len(w2),
                    gaussian_w2_mean_energy_share=float(row['mean_energy'].sum() / w2.sum()),
                    total_gaussian_w2_squared=float(w2.sum()),
                    medians={k: float(np.median(v)) for k, v in row.items()},
                    outside_old_05_ball=int((row['relative_clean_displacement'] > .05).sum()))
    for corruption, batches in rows.items():
        row = {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}
        assert all(len(v) == 64 for v in row.values())
        summaries[corruption] = summarize(row)
        all_rows.append(row)
    combined = {key: np.concatenate([r[key] for r in all_rows]) for key in all_rows[0]}
    report = dict(schema_version=1, self_check='passed', posterior_proofs=files,
                  overall=summarize(combined), per_corruption=summaries,
                  definition='Diagonal Gaussian W2^2 = ||mu_clean-mu_dirty||^2 + ||sigma_clean-sigma_dirty||^2. Realized common-eps displacement additionally has a cross term.',
                  limits='64 exposed objects x15 fixed corruptions, seed0; not960 independent objects. Posterior/preprocessing differences, not calibrated corruption error. No new accuracy, inference, method selection or causal attribution.')
    output.mkdir(parents=True)
    (output / 'posterior_decomposition.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    self_check()
    report = analyze(*sys.argv[1:])
    print(json.dumps(dict(overall=report['overall'], corruptions=len(report['per_corruption'])), indent=2))
