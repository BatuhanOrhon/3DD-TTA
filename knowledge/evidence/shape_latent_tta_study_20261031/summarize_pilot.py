"""Summarize accepted paired outcomes; bootstrap original objects, not corruptions."""
import csv
import json
from pathlib import Path
import sys

import numpy as np


def summarize(phase, accepted, output):
    phase, accepted, output = map(Path, (phase, accepted, output))
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((accepted / 'acceptance.json').read_text())
    assert report['status'] == 'passed'
    names = list(report['arms']['baseline']['per_corruption_percent'])
    with np.load(accepted / 'outcomes.npz', allow_pickle=False) as data:
        outcomes = {arm: data[arm].copy() for arm in data.files}
    assert all(v.shape == (15, 64) and v.dtype == bool for v in outcomes.values())
    draws = np.random.default_rng(20261031).integers(0, 64, size=(10000, 64))
    pairs = [('updated', 'baseline'), ('anchor', 'baseline'), ('proposal', 'baseline'),
             ('guided3', 'baseline'), ('anchor', 'updated'), ('proposal', 'anchor'),
             ('guided3', 'anchor'), ('guided3', 'proposal')]
    comparisons = {}
    for left, right in pairs:
        delta = outcomes[left].astype(float) - outcomes[right]
        per_object = delta.mean(axis=0)
        interval = np.quantile(100 * per_object[draws].mean(axis=1), [.025, .975])
        comparisons[left + '-' + right] = dict(
            delta_pp=float(100 * delta.mean()), corrected=int((delta > 0).sum()),
            harmed=int((delta < 0).sum()), conditional_object_bootstrap_95=interval.tolist())
    metrics, diagnostics, noises, shard_count = {}, {}, {}, 0
    for arm, values in outcomes.items():
        bg = names.index('background')
        metrics[arm] = dict(correct=int(values.sum()), events=int(values.size),
                            all15_percent=float(100 * values.mean()),
                            macro14_percent=float(100 * np.delete(values, bg, axis=0).mean()),
                            background_percent=float(100 * values[bg].mean()))
        config, = (phase / 'raw' / arm).glob('*/*/*/config.json')
        metrics[arm]['worker_wall_seconds'] = json.loads(config.read_text())['worker_wall_seconds_including_artifact_io']
        peaks = []
        diagnostics[arm] = {}
        for corruption in names:
            relative, clean, gradient, correction = [], [], [], []
            files = sorted((phase / 'raw' / arm).glob('**/batches/' + corruption + '/batch*/diagnostics.json'))
            assert len(files) == 2
            for file in files:
                shard_count += 1
                diagnostic = json.loads(file.read_text())
                peaks.append(diagnostic['whole_process_peak_gpu_memory_mb'])
                shape = diagnostic['guidance_metrics']['shape_latent']
                if arm == 'baseline':
                    continue
                norms = np.asarray(shape['anchor_norms'])
                relative.extend(np.asarray(shape['final_displacement_norms']) / norms)
                if arm in ('proposal', 'guided3'):
                    key = (corruption, file.parent.name)
                    if key in noises:
                        assert noises[key] == shape['global_noise_sha256']
                    else:
                        noises[key] = shape['global_noise_sha256']
                    clean.extend(np.asarray(shape['global_steps'][0]['clean_displacement_norms']) / norms)
                if arm == 'guided3':
                    assert [step['timestep'] for step in shape['global_steps']] == [20, 10, 0]
                    for step in shape['global_steps']:
                        gradient.extend(step['anchor_gradient_norms'])
                        correction.extend(np.asarray(step['correction_norms']) / norms)
            if relative:
                row = dict(final_relative_median=float(np.median(relative)),
                           final_relative_max=float(np.max(relative)),
                           near_radius_fraction=float(np.mean(np.asarray(relative) >= .04999)))
                if clean:
                    row['first_clean_relative_median'] = float(np.median(clean))
                if gradient:
                    assert np.isfinite(gradient).all() and min(gradient) > 0
                    assert max(correction) <= .010001
                    row.update(gradient_norm_median=float(np.median(gradient)),
                               correction_relative_median=float(np.median(correction)),
                               correction_relative_max=float(np.max(correction)))
                diagnostics[arm][corruption] = row
        assert len(peaks) == 30 and min(peaks) > 0
        metrics[arm]['peak_whole_process_gpu_memory_mb'] = max(peaks)
    assert shard_count == 150 and len(noises) == 30
    result = dict(metrics=metrics, paired_comparisons=comparisons, displacement=diagnostics,
                  validation=dict(batch_shards=shard_count, matched_global_noise_batches=len(noises)),
                  interval_scope='10,000 paired original-object bootstrap draws; seed0 and fixed 15 corruptions only; exploratory exposed IDs; no multiplicity adjustment')
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'per_corruption.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['corruption', *outcomes])
        for i, corruption in enumerate(names):
            writer.writerow([corruption, *[float(100 * v[i].mean()) for v in outcomes.values()]])
    return result


if __name__ == '__main__':
    print(json.dumps(summarize(*sys.argv[1:]), indent=2))
