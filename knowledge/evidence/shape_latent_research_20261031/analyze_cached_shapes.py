"""CPU-only descriptive global-latent variation; no clean anchor or inference."""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

PREP = Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/chamfer-guidance-variants-20261030/result/preparation/chamfer_guidance_variants_20261030')


def summaries(z):
    # axes: seed, corruption, original object, latent coordinate
    assert z.ndim == 4 and z.shape[0] == 3 and z.shape[1] > 1
    center = z.mean(0)
    other = (center.sum(0, keepdims=True) - center) / (z.shape[1] - 1)
    drift = np.sqrt(np.square(center - other).mean(-1))
    seed_scatter = np.sqrt(np.square(z - center[None]).mean((0, 3)))
    return drift, seed_scatter


def main():
    # Self-check the metric axes and distinguish drift from seed variation.
    toy = np.ones((3, 15, 4, 2))
    assert all(np.array_equal(a, np.zeros((15, 4))) for a in summaries(toy))
    toy[0] += 1; toy[2] -= 1
    d, s = summaries(toy)
    assert np.all(d == 0) and np.all(s > 0)
    manifest_path = PREP / 'original_combined_physical_manifest.json'
    records = json.loads(manifest_path.read_text())['records']
    records = [r for r in records if '/paired_states/confirmation/' in r['path'] and r['path'].endswith('.npz')]
    assert len(records) == 180
    datasets, evidence = {}, []
    for dataset in ('modelnet40_c', 'scanobjectnn_c'):
        ids = json.loads((PREP / (dataset + '_heldout_ids.json')).read_text())['indices']
        assert ids == sorted(set(ids)) and len(ids) == 140
        chunks = [ids[:70], ids[70:]]
        batch_names = {hashlib.sha256(np.asarray(v, dtype=np.int64).tobytes()).hexdigest(): i for i, v in enumerate(chunks)}
        cells = {}
        for record in records:
            path = Path(record['path'])
            if path.parts[-3] != dataset:
                continue
            corruption = path.parts[-2]
            marker = path.parts.index('paired_states')
            seed = int(path.parts[marker + 2].removeprefix('seed'))
            batch = batch_names[path.stem]
            raw = Path(record['local_path']).read_bytes()
            assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
            with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                z = arrays['shape_latent']
                assert z.shape == (70, 128, 1, 1) and z.dtype == np.float32 and np.isfinite(z).all()
                metadata = json.loads(str(arrays['metadata'].item()))
                sha = hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()
                assert sha == metadata['tensor_hashes']['shape_latent']
                key = (seed, corruption, batch)
                assert key not in cells
                cells[key] = z.reshape(70, 128).astype(np.float64)
            evidence.append(dict(path=record['local_path'], sha256=record['sha256']))
        corruptions = sorted({k[1] for k in cells})
        assert len(cells) == 90 and len(corruptions) == 15
        z = np.stack([np.stack([np.concatenate([cells[(seed, c, b)] for b in (0, 1)])
                              for c in corruptions]) for seed in range(3)])
        drift, scatter = summaries(z)
        rows = [dict(corruption=c, median_rms_to_other_corruptions=float(np.median(drift[i])),
                     median_seed_scatter_rms=float(np.median(scatter[i])),
                     median_sample_norm=float(np.median(np.linalg.norm(z[:, i], axis=-1))))
                for i, c in enumerate(corruptions)]
        datasets[dataset] = dict(objects=140, seeds=[0, 1, 2], dimensions=128,
                                 per_corruption=sorted(rows, key=lambda r: -r['median_rms_to_other_corruptions']))
    result = dict(status='passed', physical_caches=len(evidence), metric_self_checks='passed', datasets=datasets,
                  evidence=evidence, limitations='No clean z, posterior means/log-stds or common encoder Gaussian draws; cross-corruption disagreement is not corruption error, and seed scatter mixes posterior/native/preparation variation. No significance, causal, class-quality or inference claim.')
    with Path(sys.argv[1]).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    for name, value in datasets.items():
        print(name, json.dumps(value['per_corruption'][:5]))


if __name__ == '__main__':
    main()
