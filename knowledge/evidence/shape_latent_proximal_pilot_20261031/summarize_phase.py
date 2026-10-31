"""Posthoc clean-distance and object-cluster analysis after independent acceptance."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ARMS = ('original', 'mean', 'prior3', 'proxy3', 'oracle3')
PAIRS = [('mean', 'original'), ('prior3', 'original'), ('proxy3', 'original'),
         ('proxy3', 'prior3'), ('oracle3', 'original'), ('oracle3', 'proxy3')]


def distance_metrics(actual, base, clean):
    actual, base, clean = [np.asarray(x, dtype=np.float64) for x in (actual, base, clean)]
    assert actual.shape == base.shape == clean.shape and actual.ndim == 2
    assert all(np.isfinite(x).all() for x in (actual, base, clean))
    error = np.mean((actual - clean) ** 2, axis=1)
    baseline_error = np.mean((base - clean) ** 2, axis=1)
    defined = baseline_error > 0
    ratios = error[defined] / baseline_error[defined]
    movement, direction = actual - base, clean - base
    norm = np.linalg.norm(movement, axis=1)
    denominator = norm * np.linalg.norm(direction, axis=1)
    cosine = np.sum(movement * direction, axis=1)[denominator > 0] / denominator[denominator > 0]
    return dict(absolute_mse=float(error.mean()),
                aggregate_mse_ratio=float(error.sum() / baseline_error.sum()) if baseline_error.sum() else None,
                mean_per_event_ratio=float(ratios.mean()) if len(ratios) else None,
                median_per_event_ratio=float(np.median(ratios)) if len(ratios) else None,
                zero_baseline_error_count=int((~defined).sum()),
                median_movement_l2=float(np.median(norm)), mean_movement_l2=float(norm.mean()),
                median_defined_cosine=float(np.median(cosine)) if len(cosine) else None,
                mean_defined_cosine=float(cosine.mean()) if len(cosine) else None,
                undefined_cosine_count=int((denominator == 0).sum()))


def accuracy_contrast(left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.shape == right.shape and left.ndim == 2 and left.dtype == right.dtype == bool
    delta = left.astype(float) - right
    # Resample original objects, carrying all fixed corruption outcomes together.
    draws = np.random.default_rng(20261031).integers(0, left.shape[1], (10000, left.shape[1]))
    bootstrap = delta.mean(axis=0)[draws].mean(axis=1) * 100
    return dict(delta_pp=float(delta.mean() * 100), corrected=int((delta > 0).sum()),
                harmed=int((delta < 0).sum()), cluster_ci95_pp=np.quantile(bootstrap, [.025, .975]).tolist())


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def summarize(phase, accepted, output):
    phase, accepted, output = map(Path, (phase, accepted, output))
    assert not output.exists(), 'Use a fresh derived directory'
    assert phase.resolve() not in output.resolve().parents
    acceptance = json.loads((accepted / 'acceptance.json').read_text())
    assert acceptance['status'] == 'passed'
    inputs = json.loads((phase / 'imported/references/manifest.json').read_text())
    configs = sorted((phase / 'raw').glob('*/*/*/*/config.json'))
    assert len(configs) == 5
    with np.load(accepted / 'outcomes.npz', allow_pickle=False) as saved:
        outcomes = {arm: saved[arm].copy() for arm in ARMS}
    z_values, performance, corruptions = {}, {}, None
    for path in configs:
        cfg = json.loads(path.read_text())
        arm = cfg['shape_proximal']['arm']
        assert arm not in z_values and arm in ARMS
        assert cfg['selected_original_ids'] == inputs['original_ids'] and len(inputs['original_ids']) == 32
        if corruptions is None:
            corruptions = cfg['corruptions']
        assert cfg['corruptions'] == corruptions
        assert corruptions in (['gaussian', 'background'], ['gaussian', 'background', 'rotation', 'lidar'])
        z_values[arm], peak = [], 0.
        for corruption in corruptions:
            diagnostics = list((path.parent / 'batches' / corruption).glob('*/diagnostics.json'))
            assert len(diagnostics) == 1
            diag = json.loads(diagnostics[0].read_text())
            proof = diag['guidance_metrics']['shape_proximal']['global_proof']
            target = phase / 'paired/global' / arm / 'modelnet40_c' / corruption / Path(proof['path']).name
            assert identity(target) == {k: proof[k] for k in ('bytes', 'sha256')}
            with np.load(target, allow_pickle=False) as saved:
                z_values[arm].append(saved['final_z'].reshape(32, 128).copy())
            peak = max(peak, diag['whole_process_peak_gpu_memory_mb'])
        z_values[arm] = np.stack(z_values[arm])
        assert outcomes[arm].shape == (len(corruptions), 32)
        performance[arm] = dict(worker_wall_seconds=cfg['worker_wall_seconds_including_artifact_io'],
                               reconstruction_and_terminal_seconds=acceptance['arms'][arm]['reconstruction_and_terminal_seconds'],
                               peak_gpu_mib=peak, global_calls_per_batch=3 if arm.endswith('3') else 0,
                               correct=int(outcomes[arm].sum()), total=int(outcomes[arm].size),
                               accuracy_percent=float(outcomes[arm].mean() * 100))
    clean_values = []
    for corruption in corruptions:
        expected = inputs['cases'][corruption]['oracle']
        target = phase / 'imported/oracle_targets' / expected['filename']
        assert identity(target) == {k: expected[k] for k in ('bytes', 'sha256')}
        with np.load(target, allow_pickle=False) as saved:
            clean_values.append(saved['clean_z'].reshape(32, 128).copy())
    clean = np.stack(clean_values)
    distances = {}
    for arm in ARMS:
        distances[arm] = {'all': distance_metrics(z_values[arm].reshape(-1, 128), z_values['original'].reshape(-1, 128), clean.reshape(-1, 128))}
        distances[arm]['per_corruption'] = {
            name: distance_metrics(z_values[arm][i], z_values['original'][i], clean[i]) for i, name in enumerate(corruptions)}
    comparisons = {a + '-' + b: accuracy_contrast(outcomes[a], outcomes[b]) for a, b in PAIRS}
    for a, b in PAIRS:
        comparisons[a + '-' + b]['per_corruption'] = {}
        for i, name in enumerate(corruptions):
            delta = outcomes[a][i].astype(float) - outcomes[b][i]
            comparisons[a + '-' + b]['per_corruption'][name] = dict(
                delta_pp=float(delta.mean() * 100), corrected=int((delta > 0).sum()),
                harmed=int((delta < 0).sum()))
    report = dict(status='complete', corruptions=corruptions, original_ids=inputs['original_ids'],
                  arms=performance, paired_comparisons=comparisons, latent_distances=distances,
                  bootstrap=dict(unit='original object; all corruption rows together', draws=10000, seed=20261031),
                  definitions=dict(absolute_mse='mean over events and 128 latent coordinates',
                                   aggregate_mse_ratio='sum arm squared errors / sum original squared errors',
                                   per_event_ratio='arm squared error / original squared error; undefined denominators excluded and counted',
                                   cosine='cos(final_z-z0, clean_z-z0); zero norms excluded and counted'),
                  limits='Exploratory exposed32/fixedcorruptions/seed0; conditional CI; no multiple-comparison correction; clean targets posthoc except oracle3 diagnostic; cross-runtime clean encoder numerical caveat; replay omits baseline firstpass/proxy encoder; no deployment speed claim.',
                  accepted_identity=identity(accepted / 'acceptance.json'))
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (output / 'per_corruption.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['corruption', 'arm', 'correct', 'total', 'accuracy_percent'])
        for i, name in enumerate(corruptions):
            for arm in ARMS:
                writer.writerow([name, arm, int(outcomes[arm][i].sum()), 32, float(outcomes[arm][i].mean() * 100)])
    np.savez_compressed(output / 'paired_latents.npz', clean_z=clean, **z_values)
    return report


if __name__ == '__main__':
    print(json.dumps(summarize(*sys.argv[1:]), indent=2))
