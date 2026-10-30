"""Recount immutable Point-MAE head ZIPs; CPU NumPy only, no inference."""
import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive-root', type=Path, required=True)
args = parser.parse_args()
root = args.archive_root.resolve()
prep = root / 'result/preparation'
out = Path(__file__).resolve().parent
inputs = []
for phase, folder in [('fixed_development', 'local_development_analysis'),
                      ('fixed_confirmation', 'local_frozen_confirmation_analysis')]:
    validation = json.loads((prep / 'guidance_component_study_20261025' / folder / 'validation.json').read_text())
    for row in validation['raw_archives']:
        if row['dataset'] == 'modelnet40_c' and row['backbone'] == 'pointmae' and row['component_arm'] in ('combined', 'chamfer'):
            inputs.append((phase, row))
path = prep / 'guidance_component_sensitivity_20261025/recovery_20261028_r4/local_partial_confirmation_analysis_v1/raw_archives.json'
for row in json.loads(path.read_text()):
    if row['canonical_path'].startswith('result/modelnet40_c/') and row['backbone'] == 'pointmae' and row['condition_id'] in ('spectral-candidate', 'matched-chamfer', 'original3dd'):
        inputs.append(('sensitivity_confirmation', row))
assert len(inputs) == 17
records, pairs, cohorts = [], {}, {}
for phase, row in inputs:
    path = Path(row['canonical_path'])
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert z.testzip() is None
        names = [n for n in z.namelist() if not n.endswith('/')]
        assert len(names) == len(set(names))
        prefix = row['run_id'] + '/'
        manifest = json.loads(z.read(prefix + 'bundle_manifest.json'))
        assert {prefix + n for n in manifest} == set(names) - {prefix + 'bundle_manifest.json'}
        for name, identity in manifest.items():
            payload = z.read(prefix + name)
            assert len(payload) == identity['bytes'] and hashlib.sha256(payload).hexdigest() == identity['sha256']
        config = json.loads(z.read(prefix + 'config.json'))
        assert config['dataset'] == 'modelnet40_c' and config['backbone'] == 'pointmae'
        assert config['num_classes'] == 40 and config['severity'] == 5 and len(config['corruptions']) == 15
        assert config['classifier_state_before_sha256'] == config['classifier_state_after_sha256']
        assert config['lion_state_before'] == config['lion_state_after']
        condition = row.get('condition_id', row.get('component_arm'))
        original = condition == 'original3dd'
        if not original:
            protocol = config['guidance_protocol']
            expected_weight = (937.3767440639812 if phase == 'sensitivity_confirmation' else 1.17) if condition in ('combined', 'spectral-candidate') else 0.
            assert protocol['loss_weights'] == dict(chamfer=1., spectral=expected_weight, invariant=0.)
            assert all(protocol[k] == v for k, v in dict(M_max=800, beta=2., eta=.01, gamma=.01, p=.95, dynamic_graph=False, use_static_style=False).items())
            assert config['graph']['isolate_policy'] == 'legacy'
        seed = int(config['seed'])
        ids = tuple(config['selected_original_ids'])
        assert len(ids) == (140 if phase == 'sensitivity_confirmation' else 70)
        if phase in cohorts:
            assert cohorts[phase] == ids
        cohorts[phase] = ids
        totals = collections.Counter()
        hashes = {}
        labels_by_corruption = {}
        for corruption in config['corruptions']:
            with np.load(io.BytesIO(z.read(prefix + corruption + '_predictions.npz')), allow_pickle=False) as a:
                labels, predicted, logits = a['label'], a['prediction'], a['logits']
                assert tuple(a['index']) == ids
                assert np.isfinite(logits).all() and np.array_equal(logits.argmax(1), predicted)
                labels_by_corruption[corruption] = labels.tolist()
                totals['terminal'] += int((predicted == labels).sum())
            if not original:
                batch_names = sorted(n for n in manifest if n.startswith('batches/' + corruption + '/') and n.endswith('/predictions.npz'))
                assert len(batch_names) == len(ids) // 70
                collected_ids = []
                for name in batch_names:
                    with np.load(io.BytesIO(z.read(prefix + name)), allow_pickle=False) as a:
                        l = a['trajectory_logits'].astype(np.float64)
                        assert l.shape == (70, 4, 40) and np.isfinite(l).all()
                        assert np.array_equal(l[:, 3].argmax(1), a['prediction'])
                        offset = len(collected_ids)
                        assert np.array_equal(a['label'], labels[offset:offset + 70])
                        assert np.array_equal(l[:, 3], logits[offset:offset + 70])
                        collected_ids.extend(a['index'].tolist())
                        probs = np.exp(l - l.max(-1, keepdims=True))
                        probs /= probs.sum(-1, keepdims=True)
                        predicted = probs.mean(1).argmax(1)
                        totals['probability_mean'] += int((predicted == a['label']).sum())
                    diagnostic = json.loads(z.read(prefix + name.replace('predictions.npz', 'diagnostics.json')))
                    assert diagnostic['rng_unchanged']
                    hashes[name] = diagnostic['trajectory']['initial_hashes']
                assert tuple(collected_ids) == ids
        summary = next(csv.DictReader(io.StringIO(z.read(prefix + 'summary.csv').decode())))
        assert totals['terminal'] == int(summary['total_correct'])
        assert int(summary['total_examples']) == len(ids) * 15
        if not original:
            key = (phase, seed)
            paired = dict(ids=ids, labels=labels_by_corruption, initial_hashes=hashes,
                          classifier_state=config['classifier_state_before_sha256'])
            if key in pairs:
                assert paired == pairs[key]
            pairs[key] = paired
            expected = 'a3b8452fe5e28aca6647cbd23420069a8aee42e77d89ae7e18821ccfc034650e' if phase.startswith('fixed') else '71f8f75b07e3317cf55b301801b093b90dd926d077d2db99f0f3cbe291a284a3'
            identities = [v for k, v in config['executed_source_files'].items() if k.endswith('/tta_gsd_v2.py')]
            assert len(identities) == 1 and identities[0]['sha256'] == expected
        records.append(dict(phase=phase, condition=condition, seed=seed, n=len(ids) * 15,
                            correct=dict(totals), accuracy_pct={k: 100 * v / (len(ids) * 15) for k, v in totals.items()},
                            archive_path=str(path), archive_sha256=row['sha256'],
                            protocol=config['guidance_protocol']))
assert all(not (set(cohorts[a]) & set(cohorts[b])) for a, b in
           [('fixed_development', 'fixed_confirmation'), ('fixed_development', 'sensitivity_confirmation'), ('fixed_confirmation', 'sensitivity_confirmation')])
grouped = collections.defaultdict(lambda: collections.Counter())
csv_paths = dict(
    fixed_development=prep / 'guidance_component_study_20261025/local_development_analysis/case_policy.csv',
    fixed_confirmation=prep / 'guidance_component_study_20261025/local_frozen_confirmation_analysis/case_policy.csv',
    sensitivity_confirmation=prep / 'guidance_component_sensitivity_20261025/recovery_20261028_r4/local_partial_confirmation_analysis_v1/case_policy.csv')
csv_rows = {phase: list(csv.DictReader(path.open())) for phase, path in csv_paths.items()}
reproduced_rows = 0
for r in records:
    for policy, correct in r['correct'].items():
        expected = [a for a in csv_rows[r['phase']] if a['dataset'] == 'modelnet40_c'
                    and a['backbone'] == 'pointmae' and a['policy'] == policy
                    and int(a.get('seed', 0)) == r['seed']
                    and a.get('condition_id', a.get('component_arm')) == r['condition']]
        assert len(expected) == 1 and int(expected[0]['n']) == r['n']
        assert abs(float(expected[0]['accuracy_pct']) - r['accuracy_pct'][policy]) < 1e-10
        reproduced_rows += 1
        grouped[(r['phase'], r['condition'], policy)].update(correct=correct, n=r['n'])
means = [dict(phase=k[0], condition=k[1], policy=k[2], correct=v['correct'], n=v['n'], accuracy_pct=100 * v['correct'] / v['n']) for k, v in grouped.items()]
(out / 'verification.json').write_text(json.dumps(dict(status='passed', scope=__doc__, raw_head_archives=17,
    accepted_csv_rows_reproduced=reproduced_rows,
    paired_guided_seed_groups=len(pairs), independent_readout='float64 softmax followed by equal probability mean; labels only score predictions',
    checks='Outer SHA/bytes, all member hashes/CRC, terminal counts, finite logits, original IDs, disjoint cohorts, unchanged model states, paired labels/initial input-latent-noise hashes, pinned TTA source.',
    limitations='No fresh GPU/native execution, no assertion of absence of every algorithm defect; original3DD uses a distinct host.', records=records, means=means), indent=2) + '\n')
for row in means:
    print(row)
