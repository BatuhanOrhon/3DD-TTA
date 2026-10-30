"""Register finite trials and fresh object IDs without reading labels or using GPU."""
import hashlib
import json
from pathlib import Path
import random

PREP = Path(__file__).resolve().parent
BASE = PREP.parent / 'guidance_component_study_20261025'
FAMILIES = ('chamfer', 'combined', 'spectral')


def identity(path):
    blob = Path(path).read_bytes()
    return dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())


def trials():
    result = []
    for family, prefix in zip(FAMILIES, ('c', 'cs', 's')):
        base = dict(id=prefix + '-base', family=family, method='gsd', reuse=True,
                    eta=.01, gamma=.01, p=.95, spectral_loss_kind=None,
                    spectral_weight=0. if family == 'chamfer' else 1.17,
                    chamfer_weight=0. if family == 'spectral' else 1.)
        result.append(base)
        if family == 'chamfer':
            variants = [('step-low', dict(eta=.003, gamma=.003)),
                        ('step-high', dict(eta=.03, gamma=.03)),
                        ('eta-low', dict(eta=.003)), ('p80', dict(p=.80)), ('p99', dict(p=.99))]
        else:
            variants = []
            for suffix, kind, rho in [('l2-r010', 'legacy_l2', .1), ('l2-r100', 'legacy_l2', 1.),
                                      ('huber-r010', 'huber', .1), ('power-r010', 'power', .1),
                                      ('l2-r010-step-low', 'legacy_l2', .1)]:
                parameters = dict(spectral_loss_kind=kind, spectral_weight=dict(loss_kind=kind, rho=rho))
                if suffix.endswith('step-low'):
                    parameters.update(eta=.003, gamma=.003)
                variants.append((suffix, parameters))
        for suffix, parameters in variants:
            result.append(dict(base, id=prefix + '-' + suffix, reuse=False, **parameters))
    result.append(dict(id='original3dd', family='original3dd', method='original3dd', reuse=False,
                       eta=.01, gamma=.01, p=.95, spectral_loss_kind=None, spectral_weight=0., chamfer_weight=1.))
    return result


def reserve(n, used):
    shuffled = list(range(n))
    random.Random(20261025).shuffle(shuffled)
    selected = sorted([value for value in shuffled if value not in used][:140])
    assert len(selected) == 140 and not used.intersection(selected)
    return selected


def self_check():
    rows = trials()
    assert len(rows) == 19 and len({r['id'] for r in rows}) == 19
    assert all(len([r for r in rows if r['family'] == family]) == 6 for family in FAMILIES)
    assert len([r for r in rows if r['method'] == 'gsd' and not r['reuse']]) == 15
    assert {r['id'] for r in rows if r['reuse']} == {'c-base', 'cs-base', 's-base'}
    for n in (581, 2468):
        selected = reserve(n, set(range(140)))
        assert selected == sorted(set(selected)) and max(selected) < n
    print('AGENT_B_SENSITIVITY_REGISTRATION_SELF_CHECK_COMPLETE')


def main():
    assert PREP.parent.parent.parent.name == 'guidance-components'
    assert not (PREP / 'trial_registry.json').exists(), 'Immutable registration already exists'
    registry = dict(schema_version=1, trials=trials(), primary_readout='probability_mean',
                    selection='balanced all15 across eight cases; earlier registered trial breaks exact ties',
                    trial_budget='six configurations per GSD arm including one accepted baseline; no failed-slot replacement')
    (PREP / 'trial_registry.json').write_text(json.dumps(registry, indent=2, sort_keys=True) + '\n')
    selections = {}
    for dataset, n in [('modelnet40_c', 2468), ('scanobjectnn_c', 581)]:
        development = json.loads((BASE / (dataset + '_development_ids.json')).read_text())
        old_heldout = json.loads((BASE / (dataset + '_heldout_ids.json')).read_text())
        used = set(development['indices']) | set(old_heldout['indices'])
        assert len(used) == 140
        for stage in ('development', 'smoke'):
            name = dataset + '_' + stage + '_ids.json'
            (PREP / name).write_bytes((BASE / name).read_bytes())
        name = dataset + '_sensitivity_confirmation_ids.json'
        selection = dict(dataset=dataset, n_examples=n, stage='heldout', indices=reserve(n, used),
                         selection_seed=20261025, selected_without_labels=True, split_unit='original_object_id_across_all_corruptions',
                         prior_benchmark_exposure=True, excludes_previous_development_and_confirmation=True,
                         purpose='newly reserved sensitivity confirmation, not an external unseen test')
        path = PREP / name
        with path.open('x') as stream:
            stream.write(json.dumps(selection, indent=2, sort_keys=True) + '\n')
        selections[dataset] = dict(confirmation=identity(path), excluded_previous_objects=len(used), selected_new_objects=140)
    accepted = json.loads((BASE / 'local_development_analysis/validation.json').read_text())
    reused = [row for row in accepted['raw_archives'] if row['component_arm'] in FAMILIES]
    assert len(reused) == 24
    (PREP / 'reused_baselines.json').write_text(json.dumps(dict(status='accepted_old_source_unchanged',
        original_manifest_sha256=accepted['manifest_sha256'], original_source_snapshot_sha256=accepted['source_snapshot_sha256'],
        raw_archives=reused, eligibility='CPU defaults/trajectory exact parity and real first-step GPU legacy-inline loss/gradient/RNG parity required before reuse'),
        indent=2, sort_keys=True) + '\n')
    (PREP / 'registration_validation.json').write_text(json.dumps(dict(status='registered_CPU_only',
        labels_read=False, original_sets_preserved=True, original_development_validation=identity(BASE / 'local_development_analysis/validation.json'),
        selections=selections, registry=identity(PREP / 'trial_registry.json'), new_development_dataset_jobs=30,
        reused_baseline_dataset_jobs=6, new_original_development_dataset_jobs=2, maximum_confirmation_jobs=24,
        confirmation_unique_object_clusters=280, confirmation_seeds=[0, 1, 2], gpu_approval='required before any campaign GPU work'),
        indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(trial_registry=identity(PREP / 'trial_registry.json'), selections=selections), indent=2))


if __name__ == '__main__':
    import sys
    self_check() if sys.argv[1:] == ['--self-check'] else main()
