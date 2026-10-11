"""CPU-only analysis of immutable Legacy trajectory directories or ZIPs.

Labels are used only for evaluation and an explicitly labelled oracle ceiling.
Full-trace selection does not implement early stopping or measure speedup.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np


def probabilities(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty N x candidates x classes logits')
    x = x - x.max(-1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(-1, keepdims=True))
    p = np.exp(logp)
    return p, -(p * logp).sum(-1) / np.log(x.shape[-1])


def policies(logits, terminal: int) -> dict:
    """Label-free policies; ties choose terminal, then earliest clean candidate."""
    p, entropy = probabilities(logits)
    if not 0 <= terminal < p.shape[1]:
        raise ValueError('Terminal candidate missing')
    order = [terminal] + [i for i in range(p.shape[1]) if i != terminal]
    selected = np.asarray(order)[entropy[:, order].argmin(1)]
    return dict(terminal=p[:, terminal].argmax(1),
                minimum_entropy=p[np.arange(len(p)), selected].argmax(1),
                probability_mean=p.mean(1).argmax(1), selected_candidate=selected,
                entropy=entropy, probabilities=p)


class Bundle:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(path) if self.path.is_file() else None
        if self.archive:
            names = self.archive.namelist()
            if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts
                                                   or '\\' in n or ':' in n for n in names):
                raise ValueError('Unsafe/duplicate ZIP members')
            if self.archive.testzip() is not None:
                raise ValueError('ZIP CRC failure')
            self.names = set(names)
        else:
            self.names = {p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()}
        self.config = self.json('config.json')
        if self.config['execution_status'] != 'complete':
            raise ValueError('Incomplete/failed run cannot be analyzed as a completed trace')

    def read(self, name):
        return self.archive.read(name) if self.archive else (self.path / name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def arrays(self, name):
        with np.load(io.BytesIO(self.read(name)), allow_pickle=False) as values:
            return {key: values[key].copy() for key in values.files}

    def close(self):
        if self.archive:
            self.archive.close()


def load_trace(bundle) -> list:
    config = bundle.config
    if config['trajectory']['mode'] != 'observe':
        raise ValueError('Analysis requires an observe trace')
    if not config.get('model_state_unchanged') or config['model_state_before'] != config['model_state_after']:
        raise ValueError('Missing or changed model weights/buffers')
    manifest = config['trajectory_artifact_manifest']
    expected_files = {n for n in bundle.names if n.startswith(('trajectory/', 'predictions/')) and not n.endswith('/')}
    if set(manifest) != expected_files:
        raise ValueError('Artifact manifest coverage mismatch')
    for name, info in manifest.items():
        data = bundle.read(name)
        if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError('Artifact SHA/size mismatch: ' + name)
    output = []
    for corruption in config['corruptions']:
        names = sorted(n for n in bundle.names if n.startswith('trajectory/' + corruption + '/') and n.endswith('.json'))
        ids, labels, logits, metadata = [], [], [], None
        horizon = 30 if corruption == 'background' else 10
        steps = config['trajectory']['checkpoints'][str(horizon)]
        expected_names = ['clean_%02d' % s for s in steps] + ['terminal']
        used_files = set()
        for number, name in enumerate(names):
            record = bundle.json(name)
            if PurePosixPath(name).name != '%04d.json' % number or not record['complete'] or record['corruption'] != corruption or record['seed'] != config['seed']:
                raise ValueError('Batch identity/completion mismatch')
            if [c['name'] for c in record['candidates']] != expected_names:
                raise ValueError('Missing, reordered or unexpected candidate')
            batch = []
            for j, candidate in enumerate(record['candidates']):
                terminal = j == len(steps)
                count = horizon if terminal else steps[j]
                if (candidate['loop_count'] != count or candidate['loop_index'] != count-1
                        or candidate['timestep'] != (horizon-count)*10
                        or candidate['style_timing'] != ('post_update' if terminal else 'pre_update')
                        or candidate['kind'] != ('original_terminal' if terminal else 'predicted_clean')):
                    raise ValueError('Candidate timing/style contract mismatch')
                filename = 'trajectory/' + corruption + '/' + candidate['file']
                if candidate['file'] != '%04d_%s.npz' % (number, expected_names[j]):
                    raise ValueError('Candidate filename mismatch')
                used_files.add(filename)
                arrays = bundle.arrays(filename)
                if not np.array_equal(arrays['indices'], record['indices']) or arrays['logits'].shape != (len(record['indices']), 40):
                    raise ValueError('Candidate IDs/logit shape mismatch')
                _, entropy = probabilities(arrays['logits'][:, None, :])
                if not np.array_equal(arrays['predictions'], arrays['logits'].argmax(1)) or not np.allclose(arrays['entropy'], entropy[:, 0], atol=1e-12, rtol=1e-10):
                    raise ValueError('Saved prediction/entropy mismatch')
                batch.append(arrays['logits'])
            terminal_saved = bundle.arrays('predictions/%s_%04d.npz' % (corruption, number))
            if (not np.array_equal(terminal_saved['logits'], batch[-1])
                    or not np.array_equal(terminal_saved['indices'], record['indices'])
                    or not np.array_equal(terminal_saved['targets'], record['targets'])):
                raise ValueError('Original terminal prediction mismatch')
            ids.extend(record['indices']); labels.extend(record['targets'])
            logits.append(np.stack(batch, axis=1))
            metadata = expected_names
        actual_files = {n for n in bundle.names if n.startswith('trajectory/' + corruption + '/') and n.endswith('.npz')}
        expected_ids = config['selected_indices'].get(corruption)
        if expected_ids is None:
            cli = config['cli']
            n = min(2468, cli['batch_size'] * cli['max_batches']) if cli['max_batches'] else 2468
            expected_ids = list(range(n))
        if ids != expected_ids or len(ids) != len(set(ids)) or actual_files != used_files:
            raise ValueError('Missing/duplicate/extra trajectory examples or candidates')
        labels = np.asarray(labels)
        if labels.shape != (len(ids),) or not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= 40)):
            raise ValueError('Invalid targets')
        output.append(dict(corruption=corruption, seed=config['seed'], indices=np.asarray(ids),
                           targets=labels, logits=np.concatenate(logits), candidates=metadata))
    if sum(len(v['indices']) for v in output) != config['expected_classifications'] or config['completed_examples'] != config['expected_classifications']:
        raise ValueError('Config completion count mismatch')
    return output


def analyze_group(group):
    logits, y = group['logits'], group['targets']
    result = policies(logits, logits.shape[1]-1)
    baseline = result['terminal'] == y
    predictions = {key: result[key] for key in ('terminal', 'minimum_entropy', 'probability_mean')}
    predictions.update({'fixed_probe_%d' % (j+1): logits[:, j].argmax(1) for j in range(logits.shape[1]-1)})
    rows, samples = [], []
    candidate_correct = logits.argmax(2) == y[:, None]
    for policy, pred in predictions.items():
        correct = pred == y
        rows.append(dict(corruption=group['corruption'], seed=group['seed'], policy=policy, n=len(y),
                         n_correct=int(correct.sum()), accuracy=float(correct.mean()),
                         delta_pp=float(100*(correct.mean()-baseline.mean())),
                         corrected=int((correct & ~baseline).sum()), broken=int((~correct & baseline).sum())))
        for i, index in enumerate(group['indices']):
            samples.append(dict(corruption=group['corruption'], seed=group['seed'], index=int(index),
                                policy=policy, target=int(y[i]), prediction=int(pred[i]),
                                correct=int(correct[i]), terminal_correct=int(baseline[i])))
    selected = result['selected_candidate']
    top = logits.argmax(2)
    stable = (top == top[:, -1:]).all(1)
    entropy = result['entropy']
    diag = dict(corruption=group['corruption'], seed=group['seed'], n=len(y),
                label_using_oracle_any_correct=float(candidate_correct.any(1).mean()),
                oracle_only_note='Diagnostic ceiling for choosing an existing candidate; uses labels; not an inference policy or an averaging ceiling.',
                selected_histogram={name: int((selected == j).sum()) for j, name in enumerate(group['candidates'])},
                any_temporal_disagreement=float((~stable).mean()),
                adjacent_class_change_rate=float((top[:, 1:] != top[:, :-1]).mean()),
                stable_wrong_count=int((stable & ~baseline).sum()),
                minimum_entropy_wrong_count=int((result['minimum_entropy'] != y).sum()),
                entropy_correct=float(entropy[candidate_correct].mean()) if candidate_correct.any() else None,
                entropy_wrong=float(entropy[~candidate_correct].mean()) if (~candidate_correct).any() else None)
    return rows, samples, diag


def clustered_deltas(samples, replicates=1000):
    """Stratified original-index bootstrap, retaining all seeds per sampled ID.

    Descriptive development intervals; no timestep independence assumption.
    """
    random_state = np.random.RandomState(20261011)
    results = []
    for policy in sorted({s['policy'] for s in samples} - {'terminal'}):
        buckets = {}
        for row in samples:
            if row['policy'] == policy:
                buckets.setdefault(row['corruption'], {}).setdefault(row['index'], []).append(
                    row['correct'] - row['terminal_correct'])
        boot = np.zeros(replicates)
        for by_index in buckets.values():
            delta = np.asarray([np.mean(values) for _, values in sorted(by_index.items())])
            boot += delta[random_state.randint(len(delta), size=(replicates, len(delta)))].mean(1) / len(buckets)
        results.append(dict(policy=policy, scope='macro of available corruptions', replicates=replicates,
                            delta_pp_interval95=(100*np.percentile(boot, [2.5, 97.5])).tolist()))
    return results


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(paths, output):
    rows, samples, diagnostics, receipts, seen, ids_by_corruption = [], [], [], [], set(), {}
    signature = None
    for path in paths:
        bundle = Bundle(path)
        try:
            config = bundle.config
            current_signature = dict(assets=config['assets'], scheduler=config['scheduler'], graph=config['graph'],
                                     native_extensions=config['native_extensions'],
                                     external_python_sources=config['external_python_sources'],
                                     environment=bundle.json('environment.txt'),
                                     model_state=config['model_state_before'], model_modes=config['model_modes'],
                                     trajectory=config['trajectory'], source={k: v['sha256'] for k, v in config['source_manifest'].items()},
                                     cli={k: v for k, v in config['cli'].items() if k not in {'seed', 'output_dir', 'corruption', 'indices_manifest'}},
                                     corruptions=config['corruptions'])
            if signature is None:
                signature = current_signature
            elif current_signature != signature:
                raise ValueError('Mixed protocol/source/assets/coverage across runs')
            for group in load_trace(bundle):
                key = (group['corruption'], group['seed'])
                if key in seen:
                    raise ValueError('Duplicate corruption/seed; do not pool reruns')
                seen.add(key)
                ids = group['indices'].tolist()
                if group['corruption'] in ids_by_corruption and ids_by_corruption[group['corruption']] != ids:
                    raise ValueError('Original IDs differ across seeds')
                ids_by_corruption[group['corruption']] = ids
                r, s, d = analyze_group(group)
                rows.extend(r); samples.extend(s); diagnostics.append(d)
            receipts.append(dict(path=str(path), config_sha256=hashlib.sha256(bundle.read('config.json')).hexdigest(),
                                 source_manifest=config['source_manifest'], trajectory_runtime=config.get('trajectory_runtime', {})))
        finally:
            bundle.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'per_corruption_seed.csv', rows)
    write_csv(output / 'paired_examples.csv', samples)
    summary = []
    for policy in sorted({r['policy'] for r in rows}):
        for scope in ('available', 'without_background', 'background'):
            relevant = [r for r in rows if r['policy'] == policy and
                        (scope == 'available' or (r['corruption'] == 'background') == (scope == 'background'))]
            per_seed = {str(seed): float(np.mean([r['accuracy'] for r in relevant if r['seed'] == seed]))
                        for seed in sorted({r['seed'] for r in relevant})}
            values = list(per_seed.values())
            summary.append(dict(policy=policy, scope=scope, per_seed=per_seed,
                                mean=float(np.mean(values)) if values else None,
                                sample_seed_sd=float(np.std(values, ddof=1)) if len(values)>1 else None))
    report = dict(schema_version=1, scope='development; available corruptions only, never automatic all15 benchmark',
                  summary=summary, diagnostics=diagnostics, inputs=receipts,
                  index_cluster_intervals=clustered_deltas(samples),
                  limitations=['Low entropy does not guarantee correctness or temporal stability.',
                               'Stable predictions can be confidently wrong; timesteps are not independent examples.',
                               'Full-trace selection and averaging use all probes, do not early-stop, and imply no speedup.',
                               'Same-cost terminal voting, independent evaluation and matched online B1 latency remain open.'])
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def compare(left_path, right_path) -> dict:
    """Strict smoke parity: original logits/IDs; state/noise/RNG when both audited."""
    left, right = Bundle(left_path), Bundle(right_path)
    try:
        for key in ('assets', 'native_extensions', 'scheduler', 'graph', 'selected_indices', 'seed', 'corruptions', 'model_modes'):
            if left.config[key] != right.config[key]:
                raise ValueError('Parity provenance mismatch: ' + key)
        ignored = {'output_dir', 'trajectory_mode', 'trajectory_geometry_samples'}
        for key in set(left.config['cli']) | set(right.config['cli']):
            if key not in ignored and left.config['cli'].get(key) != right.config['cli'].get(key):
                raise ValueError('Parity CLI mismatch: ' + key)
        for key in ('source_manifest', 'external_python_sources'):
            a = {k: v['sha256'] for k, v in left.config[key].items()}
            b = {k: v['sha256'] for k, v in right.config[key].items()}
            if a != b:
                raise ValueError('Parity source mismatch: ' + key)
        if left.json('environment.txt') != right.json('environment.txt'):
            raise ValueError('Parity environment mismatch')
        a_names = {n for n in left.names if n.startswith('predictions/') and n.endswith('.npz')}
        b_names = {n for n in right.names if n.startswith('predictions/') and n.endswith('.npz')}
        if not a_names or a_names != b_names:
            raise ValueError('Parity prediction coverage mismatch')
        for name in sorted(a_names):
            a, b = left.arrays(name), right.arrays(name)
            for field in ('indices', 'targets', 'predictions', 'logits'):
                if not np.array_equal(a[field], b[field]):
                    raise ValueError('Nonidentical terminal %s: %s' % (field, name))
        audited = left.config['trajectory']['mode'] != 'off' and right.config['trajectory']['mode'] != 'off'
        if audited:
            if not left.config.get('model_state_unchanged') or not right.config.get('model_state_unchanged'):
                raise ValueError('Missing model-state preservation evidence')
            if left.config['model_state_before'] != right.config['model_state_before']:
                raise ValueError('Parity model-state mismatch')
            names = {n for n in left.names if n.startswith('trajectory/') and n.endswith('.json')}
            if names != {n for n in right.names if n.startswith('trajectory/') and n.endswith('.json')}:
                raise ValueError('Parity trace coverage mismatch')
            for name in sorted(names):
                a, b = left.json(name), right.json(name)
                for field in ('initial_hashes', 'rng_before', 'rng_after', 'indices', 'targets'):
                    if a[field] != b[field]:
                        raise ValueError('Parity state mismatch: ' + field)
                if a['candidates'][-1]['hashes'] != b['candidates'][-1]['hashes']:
                    raise ValueError('Parity terminal latent/style mismatch')
        return dict(exact_terminal_parity=True, state_noise_rng_parity=audited,
                    left=str(left_path), right=str(right_path), batches=len(a_names))
    finally:
        left.close(); right.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundles', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    if args.compare:
        if len(args.bundles) != 2:
            parser.error('--compare needs exactly two bundles')
        result = compare(*args.bundles)
        if args.output:
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2))
    else:
        if not args.output:
            parser.error('--output (new directory) required')
        analyze(args.bundles, args.output)
        print('Validated CPU analysis written to', args.output)


if __name__ == '__main__':
    main()
