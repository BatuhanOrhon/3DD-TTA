"""Recount archived logits on CPU. Pass LION and XYZ checkout roots containing result/."""
import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


LION_MANIFEST = 'knowledge/evidence/fulltest_n5_proximal_n8_20261112/complete_results_20261114.json'
CF_INPUTS = {
    'grid': ('cloudfixer_transfer/20261114-171811_development.zip',
             '28baa27f2b80ae32c42fb5b56d569f8bb3dfac4e94fb6c055a2e89d1dabf50e7', 3120),
    'N': ('cloudfixer_exploration/20261115-110019_nstep_run.zip',
          '2a4c544f2f08de453205f556689a47a1aedabf37be4b87e07d8c9348e68d58f7', 420),
    'dense': ('cloudfixer_exploration/20261115-114749_entropy_run.zip',
              '509ea9699cd2fcde8542fc52564c03146a1f2f30db0d4e20d176e1aa479506df', 60),
}


def open_verified(path, sha):
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != sha:
        raise ValueError(f'SHA-256 mismatch for {path}')
    return zipfile.ZipFile(io.BytesIO(contents))


def npz(archive, name):
    return np.load(io.BytesIO(archive.read(name)), allow_pickle=False)


def config(archive):
    name, = [n for n in archive.namelist() if n.count('/') == 1 and n.endswith('/config.json')]
    return json.loads(archive.read(name))


def softmax(logits):
    logits = logits.astype(np.float64)
    assert np.isfinite(logits).all()
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def read_lion(root, manifest):
    cells, labels, modelnet_inputs = defaultdict(dict), {}, {}
    assert len(manifest['raw_archives']) == 240
    for item in manifest['raw_archives']:
        dataset, head, corruption = (item[k] for k in ('dataset', 'backbone', 'corruption'))
        cell = cells[dataset, head].setdefault(corruption, {})
        with open_verified(root / item['archive'], item['sha256']) as archive:
            if not modelnet_inputs and dataset == 'modelnet40_c':
                modelnet_inputs = config(archive)['dataset_provenance']['files']
            summary, = [s for s in archive.namelist() if s.endswith('/' + corruption + '_predictions.npz')]
            with npz(archive, summary) as data:
                ids, y, prediction = data['index'], data['label'], data['prediction']
                terminal = data['logits'].argmax(axis=-1)
                assert np.array_equal(ids, np.arange(item['n_examples']))
                assert np.array_equal(labels.setdefault(dataset, y), y)
                assert (prediction == y).sum() == item['n_correct']
                if item['arm'] == 'terminal':
                    assert np.array_equal(prediction, terminal) and 'n5' not in cell
                    cell['n5'] = terminal == y
                    continue
                assert item['arm'] == 'proximal' and 'pm' not in cell
                assert np.array_equal(terminal, data['terminal_prediction'])
                cell.update(pm=prediction == y, terminal=terminal == y)
            all_ids, candidate_scores = [], []
            for name in sorted(n for n in archive.namelist() if n.endswith('/predictions.npz')):
                with npz(archive, name) as data:
                    indices, logits = data['index'], data['trajectory_logits']
                    assert np.array_equal(data['candidate_loop_counts'], [2, 4, 6, 8])
                    assert np.array_equal(data['label'], y[indices])
                    assert np.array_equal(softmax(logits).mean(1).argmax(-1), prediction[indices])
                    assert np.array_equal(logits[:, -1].argmax(-1), terminal[indices])
                    all_ids.extend(indices)
                    candidate_scores.append(logits.argmax(-1) == data['label'][:, None])
            assert np.array_equal(all_ids, ids)
            cell['candidates'] = np.concatenate(candidate_scores)
    assert len(cells) == 8 and all(len(corr) == 15 for corr in cells.values())
    return cells, labels, modelnet_inputs


def paired(candidate, reference, draws):
    # corruption x original objects, with the same object retained across all corruptions
    delta = candidate.astype(np.float64) - reference
    object_effects = 100 * delta.mean(axis=0)
    interval = np.quantile(object_effects[draws].mean(axis=1), [.025, .975])
    corruption_effects = delta.mean(axis=1) * 100
    return dict(delta_pp=float(object_effects.mean()), object_ci95_pp=interval.tolist(),
                improved_tied_worse=[int((corruption_effects > 0).sum()),
                                     int((corruption_effects == 0).sum()),
                                     int((corruption_effects < 0).sum())],
                corruption_delta_pp=corruption_effects.tolist())


def bootstrap_draws(n):
    return np.random.default_rng(20261118).integers(0, n, (10000, n), dtype=np.int32)


def report_lion(cells):
    result = {}
    for (dataset, head), entries in sorted(cells.items()):
        groups = sorted(entries)
        arrays = {key: np.stack([entries[c][key] for c in groups]) for key in next(iter(entries.values()))}
        n5, terminal, pm = (arrays[key] for key in ('n5', 'terminal', 'pm'))
        draws = bootstrap_draws(pm.shape[1])
        step_counts = arrays['candidates'].sum(axis=(0, 1)).tolist()
        best = int(np.argmax(step_counts))
        result[dataset + '/' + head] = dict(
            n=int(pm.size), n5=int(n5.sum()), n8_terminal=int(terminal.sum()), n8_pm=int(pm.sum()),
            corruption_order=groups, fixed_probe_correct=dict(zip(('2','4','6','8'), step_counts)),
            pm_vs_best_fixed=paired(pm, arrays['candidates'][:, :, best], draws),
            pm_vs_n5=paired(pm, n5, draws), pm_vs_own_terminal=paired(pm, terminal, draws),
            repairs=int((pm & ~terminal).sum()), harms=int((terminal & ~pm).sum()))
    return result


def read_cloudfixer(root, labels, modelnet_inputs, head_sha):
    records, seconds, common_events = defaultdict(dict), defaultdict(float), None
    for role, (relative, sha, expected) in CF_INPUTS.items():
        with open_verified(root / 'result/modelnet40_c' / relative, sha) as archive:
            assert archive.testzip() is None
            if role == 'grid':
                manifest = config(archive)['manifest']
                assert manifest['classifier_sha256'] == head_sha
                assert len(manifest['dataset_files']) == 16
                for filename, digest in manifest['dataset_files'].items():
                    assert modelnet_inputs[filename]['sha256'] == digest
            units = [n for n in archive.namelist() if '/pending_unit_' in n and n.endswith('.npz')]
            assert len(units) == expected
            for name in units:
                with npz(archive, name) as data:
                    unit = json.loads(str(data['unit']))
                    cfg, ids, y = unit['config'], data['ids'], data['labels']
                    assert np.array_equal(ids, unit['ids']) and np.array_equal(y, labels[ids])
                    if cfg['method'] == 'proximal':
                        arm = f"N{cfg['steps']}_K{cfg['start']}_tau{cfg['tau']}"
                    elif cfg['method'] == 'cloudfixer':
                        arm = 'native' + str(cfg['operational_repeat'])
                    else:
                        assert cfg['method'] == 'source'
                        arm = 'source'
                    key = role + '/' + arm
                    p = softmax(data['logits'])
                    reads = {'pm': p.mean(1), 'terminal': p[:, -1]}
                    if role == 'dense':
                        reads.update(late=p[:, -3:].mean(1), single20=p[:, -3])
                    if role == 'N' and cfg['steps'] == 12:
                        reads.update(late=p[:, -2:].mean(1), single27=p[:, -2])
                    keys = [(unit['corruption'], int(i)) for i in ids]
                    for read, prob in reads.items():
                        store = records[key + '/' + read]
                        assert not any(k in store for k in keys)
                        store.update(zip(keys, prob.argmax(-1) == y))
                    seconds[key] += float(data['seconds'])
    for rows in records.values():
        if common_events is None:
            common_events = set(rows)
        assert set(rows) == common_events
    return records, seconds, common_events


def report_cloudfixer(records, seconds, events, lion):
    corruptions = sorted({c for c, _ in events})
    ids = sorted({i for _, i in events})
    assert len(corruptions) == 15 and len(ids) == 128 and len(events) == 1920
    counts = {k: int(sum(v.values())) for k, v in sorted(records.items())}
    settings = [k for k in counts if k.startswith('grid/N') and k.endswith('/pm')]
    wins = sum(counts[k] > counts[k[:-2] + 'terminal'] for k in settings)
    assert len(settings) == 48 and wins == 26
    def matrix(key):
        return np.array([[records[key][c, i] for i in ids] for c in corruptions])
    native = np.mean([matrix(f'grid/native{k}/terminal') for k in range(3)], axis=0)
    choices = {f'lion_{key}': np.stack([lion[c][key][ids] for c in corruptions])
               for key in ('n5', 'terminal', 'pm')}
    for name in ('grid/N32_K160_tau0.03/terminal', 'dense/N8_K80_tau0.1/late'):
        choices[name] = matrix(name)
    draws = bootstrap_draws(len(ids))
    return dict(corruption_order=corruptions, grid_pm_wins=wins, correct=counts,
                full_trace_seconds=dict(sorted(seconds.items())),
                same_id_comparisons_vs_mean_native={k: dict(correct=int(v.sum()), **paired(v, native, draws))
                                                    for k, v in choices.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lion-root', type=Path, required=True)
    parser.add_argument('--xyz-root', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.lion_root / LION_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    cells, labels, files = read_lion(args.lion_root, manifest)
    head_sha = manifest['checkpoint_identities']['checkpoints/modelnet_jt.pth']['sha256']
    cf, seconds, events = read_cloudfixer(args.xyz_root, labels['modelnet40_c'], files, head_sha)
    out = dict(lion_receipt_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               xyz_inputs=CF_INPUTS, verified_common_data_files=16,
               pointmae_checkpoint_sha256=head_sha, bootstrap_seed=20261118, draws=10000,
               lion=report_lion(cells),
               cloudfixer=report_cloudfixer(cf, seconds, events, cells['modelnet40_c','pointmae']))
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
