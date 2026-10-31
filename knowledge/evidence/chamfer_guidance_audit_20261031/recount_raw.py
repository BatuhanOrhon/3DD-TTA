"""Independent CPU recount of immutable archived logits and paired initial states."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PREP = Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/chamfer-guidance-variants-20261030/result/preparation/chamfer_guidance_variants_20261030')


def main():
    manifests = [('original', PREP / 'original_combined_physical_manifest.json', '/results/confirmation/', 120),
                 ('curvenet', PREP / 'postcampaign_curvenet_h/heldout_receipt_local_raw_manifest.json', '/results/heldout/', 24)]
    runs, identities = {}, []
    for phase, manifest_path, selector, expected in manifests:
        records = json.loads(manifest_path.read_text())['records']
        records = [r for r in records if selector in r['path'] and r['path'].endswith('.zip')]
        assert len(records) == expected
        for record in records:
            raw = Path(record['local_path']).read_bytes()
            assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
            identities.append({k: record[k] for k in ('local_path', 'bytes', 'sha256')})
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                names = archive.namelist()
                assert len(names) == len(set(names))
                config_name = next(n for n in names if n.endswith('/config.json') or n == 'config.json')
                prefix = config_name[:-len('config.json')]
                config = json.loads(archive.read(config_name))
                assert config['seed'] == record['seed'] and config['dataset'] == record['dataset']
                assert config['classifier_state_before_sha256'] == config['classifier_state_after_sha256']
                assert config['lion_state_before'] == config['lion_state_after']
                experiment = config.get('chamfer_experiment')
                arm = 'baseline' if experiment is None else experiment['variant']
                if arm == 'semantic':
                    arm = 'semantic_kl' if config['semantic_guidance']['kind'] == 'kl' else 'semantic_h'
                assert arm == record['arm'] or (phase == 'curvenet' and arm == 'semantic_h')
                ids = config['selected_original_ids']
                assert len(ids) == len(set(ids)) == 140 and ids == sorted(ids)
                corruption_data, initials, candidate_counts = [], {}, set()
                for corruption in sorted(config['corruptions']):
                    shards = sorted(n for n in names if n.startswith(prefix + 'batches/' + corruption + '/') and n.endswith('/predictions.npz'))
                    assert len(shards) == 2
                    logits, labels, indices = [], [], []
                    for shard in shards:
                        with np.load(io.BytesIO(archive.read(shard)), allow_pickle=False) as arrays:
                            value = arrays['trajectory_logits'].astype(np.float64)
                            assert value.shape == (70, 4, config['num_classes']) and np.isfinite(value).all()
                            logits.append(value); labels.append(arrays['label']); indices.append(arrays['index'])
                            assert np.array_equal(arrays['prediction'], value[:, 3].argmax(-1))
                        diagnostic = json.loads(archive.read(shard.replace('predictions.npz', 'diagnostics.json')))
                        trajectory = diagnostic['trajectory']
                        metrics = diagnostic['guidance_metrics']
                        assert metrics['graph_calls'] == metrics['spectral_loss_calls'] == 0
                        horizon = 30 if corruption == 'background' else 10
                        if arm != 'baseline':
                            assert metrics['prior_forward_calls'] == horizon
                        expected_backward = (11 if horizon == 30 else 4) if arm == 'sparse' else horizon
                        assert metrics['guidance_backward_calls'] == expected_backward
                        if arm.startswith('semantic'):
                            assert trajectory['semantic_guidance']['reference_votes'] is False
                            assert [r['loop_count'] for r in trajectory['semantic_guidance']['records']] == ([15, 23] if horizon == 30 else [5, 8])
                        initials[(corruption, shard.split('/')[-2])] = (trajectory['initial_hashes'], trajectory['initial_rng_sha256'])
                        candidate_counts.add(tuple(trajectory['candidate_loop_counts']))
                    assert np.array_equal(np.concatenate(indices), ids)
                    value, label = np.concatenate(logits), np.concatenate(labels).reshape(-1)
                    value -= value.max(-1, keepdims=True)
                    probability = np.exp(value)
                    probability /= probability.sum(-1, keepdims=True)
                    prediction = probability.mean(1).argmax(-1)
                    corruption_data.append((label, prediction == label))
                assert len(corruption_data) == 15
                key = (phase, config['dataset'], arm, config['backbone'], config['seed'])
                assert key not in runs
                runs[key] = dict(correct=np.stack([d[1] for d in corruption_data]), labels=np.stack([d[0] for d in corruption_data]),
                                 initial=initials, ids=ids, candidates=sorted(candidate_counts))
    summary = []
    for phase, dataset, arm in sorted({k[:3] for k in runs if k[2] != 'baseline'}):
        paired, by_head = [], {}
        for key, treatment in runs.items():
            if key[:3] != (phase, dataset, arm):
                continue
            baseline = runs[(phase, dataset, 'baseline', *key[3:])]
            assert baseline['initial'] == treatment['initial'] and baseline['ids'] == treatment['ids']
            assert np.array_equal(baseline['labels'], treatment['labels'])
            delta = treatment['correct'].astype(float) - baseline['correct']
            paired.append((baseline['correct'], treatment['correct']))
            by_head.setdefault(key[3], []).append((baseline['correct'], treatment['correct']))
        for head, pairs in [('four_head_mean', paired), *sorted(by_head.items())]:
            before = np.stack([p[0] for p in pairs]); after = np.stack([p[1] for p in pairs])
            values = (after.astype(float) - before).mean(axis=(0, 1))
            rng = np.random.default_rng(0)
            bootstrap = [values[rng.integers(0, 140, 140)].mean() for _ in range(2000)]
            summary.append(dict(phase=phase, dataset=dataset, arm=arm, head=head,
                                baseline_accuracy=100 * before.mean(), accuracy=100 * after.mean(),
                                delta_pp=100 * values.mean(), ci95=(100 * np.quantile(bootstrap, [.025, .975])).tolist(),
                                corrected=int((~before & after).sum()), harmed=int((before & ~after).sum())))
    result = dict(status='passed', raw_archives=len(identities), batches=144 * 30,
                  source_manifests=[str(m[1]) for m in manifests], records=identities, summary=summary,
                  limits='Read-only local recount; no inference or fresh Drive/cache retrieval. Conditional object-cluster CI; fixed heads/seeds.')
    with Path(sys.argv[1]).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records' and k != 'summary'}, indent=2))
    for row in summary:
        if row['head'] == 'four_head_mean' or (row['phase'] == 'curvenet' and row['head'] == 'curvenet'):
            print(json.dumps(row))


if __name__ == '__main__':
    main()
