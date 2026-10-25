"""Independently score fixed Scan/ModelNet trajectory candidates; CPU-only, no label tuning."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_artifacts import CORRUPTIONS
from scripts.analyze_legacy_trajectory import policies


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Bundle:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(path) if self.path.is_file() else None
        if self.archive:
            names = self.archive.namelist()
            require(len(names) == len(set(names)), 'Duplicate ZIP members')
            require(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
                        and '\\' not in n and ':' not in n for n in names), 'Unsafe ZIP member')
            require(self.archive.testzip() is None, 'ZIP CRC failure')
            files = {n for n in names if not n.endswith('/')}
            if 'config.json' in files:
                self.prefix = ''
            else:
                roots = {PurePosixPath(n).parts[0] for n in files}
                require(len(roots) == 1, 'Expected one sealed run root')
                self.prefix = next(iter(roots)) + '/'
            self.names = {n[len(self.prefix):] for n in files}
        else:
            self.prefix = ''
            self.names = {p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()}
        require('bundle_manifest.json' in self.names, 'Sealed bundle manifest is required')
        manifest = self.json('bundle_manifest.json')
        require(set(manifest) == self.names - {'bundle_manifest.json'}, 'Sealed manifest coverage mismatch')
        for name, expected in manifest.items():
            raw = self.read(name)
            require(expected.get('bytes') == len(raw) and expected.get('sha256') == hashlib.sha256(raw).hexdigest(),
                    'Sealed member SHA/size mismatch: ' + name)

    def read(self, name):
        return self.archive.read(self.prefix + name) if self.archive else (self.path / name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def close(self):
        if self.archive:
            self.archive.close()


def analyze_bundle(path, allow_partial=False, *, dataset='scanobjectnn_c') -> dict:
    protocols = {
        'scanobjectnn_c': (581, 15, 'main_split'),
        'modelnet40_c': (2468, 40, 'official_modelnet40_test'),
    }
    require(dataset in protocols, 'Unknown trajectory dataset')
    n_examples, num_classes, split = protocols[dataset]
    bundle = Bundle(path)
    try:
        config = bundle.json('config.json')
        require(config.get('trajectory_observation') is True, 'Trajectory observation flag missing')
        require(config.get('dataset') == dataset and config.get('n_examples') == n_examples,
                'Expected trajectory dataset %s with %d examples' % (dataset, n_examples))
        require(config.get('split') == split and config.get('num_classes') == num_classes and
                config.get('batch_size') == 70 and config.get('method') == 'gsd_inspired_latent_spectral_guidance',
                'Expected GSD-inspired %s, %d classes and B70 protocol' % (split, num_classes))
        require(config.get('severity') == 5 and config.get('seed') == 0, 'Expected severity5 seed0')
        require(config.get('status') in ('complete', 'partial'), 'Failed/running bundle is not accepted')
        require(config.get('classifier_state_before_sha256') is not None and
                config['classifier_state_before_sha256'] == config.get('classifier_state_after_sha256'),
                'Classifier state guard missing or failed')
        require(config.get('lion_state_before') is not None and config['lion_state_before'] == config.get('lion_state_after'),
                'LION state guard missing or failed')
        corruptions = config['corruptions']
        require(corruptions and len(set(corruptions)) == len(corruptions) and
                set(corruptions) <= set(CORRUPTIONS), 'Invalid corruption scope')
        if not allow_partial:
            require(config['status'] == 'complete' and set(corruptions) == set(CORRUPTIONS),
                    'Full acceptance requires complete all15')
        rows, manifests = [], {}
        complete = config['status'] == 'complete' and set(corruptions) == set(CORRUPTIONS)
        all_batches = {n for n in bundle.names if n.startswith('batches/') and n.endswith('/predictions.npz')}
        consumed = set()
        for corruption in corruptions:
            prefix = 'batches/' + corruption + '/'
            names = sorted(n for n in all_batches if n.startswith(prefix))
            require(bool(names), 'No trajectory batches: ' + corruption)
            horizon = 30 if corruption == 'background' else 10
            steps = [8, 15, 23] if horizon == 30 else [3, 5, 8]
            expected_names = ['clean_%02d' % s for s in steps] + ['terminal']
            parts = []
            for number, name in enumerate(names):
                expected = prefix + 'batch%04d/predictions.npz' % number
                require(name == expected, 'Missing or unordered batch number')
                raw = bundle.read(name)
                manifests[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                with np.load(io.BytesIO(raw), allow_pickle=False) as values:
                    data = {key: values[key].copy() for key in values.files}
                require({'index', 'label', 'prediction', 'logits', 'trajectory_logits',
                         'candidate_names', 'candidate_loop_counts'} <= set(data), 'Missing trajectory arrays')
                require(data['candidate_names'].tolist() == expected_names and
                        data['candidate_loop_counts'].tolist() == steps + [horizon], 'Candidate ordering mismatch')
                ids, labels, prediction = data['index'], data['label'], data['prediction']
                n = len(ids)
                require(n == min(70, n_examples - number * 70), 'Batch size/remainder does not match fixed B70')
                require(n > 0 and all(a.shape == (n,) and a.dtype.kind in 'iu'
                                     for a in (ids, labels, prediction)), 'Invalid indexed predictions')
                require(((ids >= 0) & (ids < n_examples)).all() and ((labels >= 0) & (labels < num_classes)).all(),
                        'Out-of-range original ID or label')
                require(data['logits'].shape == (n, num_classes) and data['trajectory_logits'].shape == (n, 4, num_classes)
                        and np.isfinite(data['trajectory_logits']).all(), 'Invalid %d-class candidate logits' % num_classes)
                require(np.array_equal(data['trajectory_logits'][:, 3], data['logits']) and
                        np.array_equal(data['logits'].argmax(1), prediction), 'Terminal predictions mismatch')
                diagnostic_name = name.rsplit('/', 1)[0] + '/diagnostics.json'
                diagnostic_raw = bundle.read(diagnostic_name)
                manifests[diagnostic_name] = dict(bytes=len(diagnostic_raw), sha256=hashlib.sha256(diagnostic_raw).hexdigest())
                diagnostic = json.loads(diagnostic_raw)
                trace = diagnostic.get('trajectory', {})
                require(trace.get('complete') is True and trace.get('candidate_names') == expected_names and
                        trace.get('candidate_loop_counts') == steps + [horizon] and
                        bool(trace.get('initial_hashes')) and bool(trace.get('final_hashes')),
                        'Missing trajectory state/candidate guards')
                probes = trace.get('probes', [])
                require(len(probes) == 3 and [p['loop_count'] for p in probes] == steps,
                        'Missing clean probe diagnostics')
                for probe in probes:
                    require(probe.get('rng_unchanged') is True and probe.get('buffers_unchanged') is True
                            and probe.get('live_inputs_unchanged') is True and
                            probe.get('rng_before_sha256') == probe.get('rng_after_sha256') and
                            probe.get('buffers_before') == probe.get('buffers_after'), 'Probe purity guard failed')
                require(diagnostic.get('rng_unchanged') is True, 'Terminal classifier RNG guard missing')
                parts.append(data)
                consumed.add(name)
            ids = np.concatenate([d['index'] for d in parts])
            labels = np.concatenate([d['label'] for d in parts])
            require(np.array_equal(ids, np.arange(len(ids))) and len(ids) <= n_examples,
                    'Original IDs must be an ordered unique prefix of 0..%d' % (n_examples - 1))
            if not allow_partial:
                require(len(ids) == n_examples, 'Full acceptance requires every original ID')
            complete = complete and len(ids) == n_examples
            selected = policies(np.concatenate([d['trajectory_logits'] for d in parts]), terminal=3)
            terminal_correct = selected['terminal'] == labels
            for policy in ('terminal', 'minimum_entropy', 'probability_mean'):
                correct = selected[policy] == labels
                rows.append(dict(backbone=config['backbone'], corruption=corruption, policy=policy,
                                 n=len(ids), correct=int(correct.sum()), accuracy_pct=100. * float(correct.mean()),
                                 delta_terminal_pp=100. * float(correct.mean() - terminal_correct.mean()),
                                 corrected=int((correct & ~terminal_correct).sum()),
                                 harmed=int((~correct & terminal_correct).sum())))
        require(consumed == all_batches, 'Undeclared corruption/batch predictions')
        macro = []
        for policy in ('terminal', 'minimum_entropy', 'probability_mean'):
            selected = [r for r in rows if r['policy'] == policy]
            macro.append(dict(backbone=config['backbone'], policy=policy, corruptions=len(selected),
                              n=sum(r['n'] for r in selected),
                              accuracy_pct=float(np.mean([r['accuracy_pct'] for r in selected])),
                              delta_terminal_pp=float(np.mean([r['delta_terminal_pp'] for r in selected])),
                              corrected=sum(r['corrected'] for r in selected), harmed=sum(r['harmed'] for r in selected)))
        return dict(status='complete' if complete else 'partial', input=str(path), dataset=dataset, backbone=config['backbone'],
                    per_corruption=rows, macro=macro, analyzed_members=manifests,
                    comparison='same-run terminal; no fitted thresholds or early stopping; single seed')
    finally:
        bundle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundles', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--dataset', choices=('scanobjectnn_c', 'modelnet40_c'), default='scanobjectnn_c')
    args = parser.parse_args(argv)
    reports = [analyze_bundle(p, args.allow_partial, dataset=args.dataset) for p in args.bundles]
    require(len({r['backbone'] for r in reports}) == len(reports), 'Duplicate backbone bundles')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='complete' if all(r['status'] == 'complete' for r in reports) else 'partial',
                  bundles=reports, macro=[row for r in reports for row in r['macro']])
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    for name, rows in [('macro', report['macro']), ('per_corruption', [row for r in reports for row in r['per_corruption']])]:
        with (args.output / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(status=report['status'], macro=report['macro']), sort_keys=True))


if __name__ == '__main__':
    main()
