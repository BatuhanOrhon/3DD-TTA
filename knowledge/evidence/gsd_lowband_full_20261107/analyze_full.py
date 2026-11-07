"""Aggregate durable paired full-study receipts; partial results never trigger a gate."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('paired_accuracy',
    REPO/'knowledge/evidence/decoded_physical_accuracy_20261104/analyze_accuracy.py')
accuracy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accuracy)
CORRUPTIONS = accuracy.CORRUPTIONS
ARMS = ('control', 'half0', 'M100')
identity = accuracy.identity


def number(value):
    assert type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid count/timing'
    return value


def read_receipt(path, label, kind, final):
    receipt = json.loads(path.read_text())
    assert receipt['label'] == label and receipt['kind'] == kind
    assert receipt['status'] in ('locally validated; cleanup ack pending', 'accepted and cleanup ack uploaded')
    assert type(receipt['ack_uploaded']) is bool and (not final or receipt['ack_uploaded'])
    expected = receipt['archive_identity']
    assert type(expected['bytes']) is int and expected['bytes'] > 0
    assert re.fullmatch('[0-9a-f]{64}', expected['sha256'])
    ack = receipt['ack']
    assert ack['status'] == 'local_and_drive_verified' and ack['label'] == label and ack['kind'] == kind
    assert {key: ack[key] for key in ('bytes', 'sha256')} == expected
    archive = Path(receipt['local_archive'])
    assert archive.resolve() == Path(ack['local_archive']).resolve() and archive.is_file()
    if final:
        assert identity(archive) == expected, 'Final raw ZIP SHA/size differs from durable acceptance'
    return receipt


def check_inference(receipt, indices):
    details = receipt['details']
    arm = receipt['label'].rsplit('_', 1)[1]
    expected_keys = {corruption+'/'+str(batch) for corruption in CORRUPTIONS for batch in range(len(indices)//2)}
    cases = {case['case']: case for case in details['cases']}
    assert len(cases) == len(details['cases']) and set(cases) == expected_keys, 'Duplicate/missing case coverage'
    assert set(details['counts']) == set(CORRUPTIONS)
    assert details['basis_passed'] is True and details['basis_checks'] == (0 if arm == 'control' else len(expected_keys))
    number(details['worker_wall_seconds'])
    labels_by_id = {}
    for corruption in CORRUPTIONS:
        pm_correct, terminal_correct = 0, 0
        for batch in range(len(indices)//2):
            case = cases[corruption+'/'+str(batch)]
            assert case['ids'] == indices[batch*2:batch*2+2]
            assert isinstance(case['paired_identity'], dict) and case['paired_identity']
            for field in ('ids', 'labels', 'pm', 'terminal'):
                assert isinstance(case[field], list) and len(case[field]) == 2
                assert all(type(value) is int for value in case[field])
            for field in ('labels', 'pm', 'terminal'):
                assert all(0 <= value < 40 for value in case[field])
            for index, label in zip(case['ids'], case['labels']):
                if index in labels_by_id:
                    assert labels_by_id[index] == label, 'Object labels changed between corruptions'
                labels_by_id[index] = label
            pm_correct += sum(a == b for a, b in zip(case['pm'], case['labels']))
            terminal_correct += sum(a == b for a, b in zip(case['terminal'], case['labels']))
            if arm == 'control':
                assert case['filter_sha256'] is None and case['rank'] is None
            else:
                assert len(case['filter_sha256']) == len(case['rank']) == 2
                assert all(re.fullmatch('[0-9a-f]{64}', value) for value in case['filter_sha256'])
                assert all(type(value) is int and 0 <= value <= 2048 for value in case['rank'])
        counts = details['counts'][corruption]
        for key, expected in (('n', len(indices)), ('pm_correct', pm_correct), ('terminal_correct', terminal_correct)):
            assert type(counts[key]) is int and counts[key] == expected, 'Receipt prediction counts differ: '+corruption+'/'+key
        number(counts['reconstruction_seconds'])
        number(counts['peak_gpu_memory_mb'])
    return cases


def final_components(path, cases):
    totals = dict(graph_build_seconds=0., decoder_and_hash_seconds=0., spectral_loss_seconds=0.,
                  preprocessing_seconds=0., terminal_geometry_head_io_seconds=0., case_wall_before_drive_copy_seconds=0.)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts and '\\' not in name and ':' not in name for name in names)
        configs = [name for name in names if name == 'config.json' or name.endswith('/config.json')]
        assert len(configs) == 1
        prefix = configs[0][:-len('config.json')]
        for case in cases:
            corruption, batch = case['case'].split('/')
            diagnostic = json.loads(archive.read(prefix+'batches/%s/batch%04d/diagnostics.json' % (corruption, int(batch))))
            assert diagnostic['original_ids'] == case['ids']
            for key in ('graph_build_seconds', 'preprocessing_seconds', 'terminal_geometry_head_io_seconds',
                        'case_wall_before_drive_copy_seconds'):
                totals[key] += number(diagnostic['timing'][key])
            for record in diagnostic['trajectory'].get('decoded_spectral', {}).get('records', []):
                for key in ('decoder_and_hash_seconds', 'spectral_loss_seconds'):
                    totals[key] += number(record[key])
    return totals


def comparisons(outcomes, mask):
    if not bool(mask.any()):
        return None
    return {winner+'_vs_'+baseline: {readout: accuracy.paired_metrics(
            matrices[winner][:, mask], matrices[baseline][:, mask])
            for readout, matrices in outcomes.items()}
        for winner, baseline in (('M100', 'control'), ('M100', 'half0'), ('half0', 'control'))}


def aggregate(root, final=False):
    root = Path(root)
    plan_path = root/'operator/plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['corruptions'] == list(CORRUPTIONS) and plan['frozen_alpha100'] == 4433.612924010353
    assert plan['transfer_gate']['minimum_delta_pp'] == 1.
    assert plan['transfer_gate']['paired_object_cluster_ci95_lower_strictly_above'] == 0.
    prior = plan['prior_M100_ids']
    assert len(prior) == len(set(prior)) == 20 and all(type(index) is int and 0 <= index < 2468 for index in prior)
    source_path = root/'common_identity.json'
    source_identity = json.loads(source_path.read_text())
    assert source_identity and source_identity['asset_manifest_sha256'] == plan['asset_manifest_sha256']
    planned_labels, planned_ids, shards = set(), [], []
    for ordinal, shard in enumerate(plan['shards']):
        assert shard['number'] == ordinal and shard['label'] == 'shard%03d' % ordinal
        assert len(shard['arms']) == 3 and set(shard['arms']) == set(ARMS)
        selection_path = (root/'operator'/shard['selection_file']).resolve()
        assert (root/'operator').resolve() in selection_path.parents
        selection = json.loads(selection_path.read_text())
        indices = selection['indices']
        assert type(selection['schema_version']) is int and selection['schema_version'] == 1 and selection['dataset'] == 'modelnet40_c'
        assert type(shard['count']) is int and 2 <= shard['count'] <= 2468 and shard['count'] % 2 == 0
        assert len(indices) == shard['count'] and indices == sorted(set(indices))
        assert all(type(index) is int and 0 <= index < 2468 for index in indices)
        planned_ids.extend(indices)
        labels = [shard['label']+'_'+arm for arm in ('cache', *ARMS)]
        planned_labels.update(labels)
        shards.append((shard, indices, labels))
    assert len(planned_ids) == len(set(planned_ids)), 'Planned shards overlap'
    found = {}
    for path in sorted((root/'accepted').glob('*.json')):
        if re.fullmatch('ack_[0-9a-f]{32}', path.stem):
            continue
        assert path.stem in planned_labels, 'Unexpected accepted receipt label: '+path.stem
        found[path.stem] = path
    complete = [(shard, indices, labels) for shard, indices, labels in shards if all(label in found for label in labels)]
    incomplete = [shard['label'] for shard, _, labels in shards if not all(label in found for label in labels)]
    all_ids = sorted(index for _, indices, _ in complete for index in indices)
    if final:
        assert len(shards) == len(complete) == 39 and not incomplete
        assert sorted(planned_ids) == all_ids == list(range(2468))
        assert len(found) == 156, 'Final requires 117 inference and 39 cache receipts'
    positions = {index: position for position, index in enumerate(all_ids)}
    outcomes = {readout: {arm: np.zeros((15, len(all_ids)), dtype=np.bool_) for arm in ARMS}
                for readout in ('pm', 'terminal')}
    per_corruption = {arm: {corruption: dict(n=0, pm_correct=0, terminal_correct=0) for corruption in CORRUPTIONS} for arm in ARMS}
    timing = {arm: dict(worker_wall_seconds=0., reconstruction_seconds=0., peak_gpu_memory_mb=0.,
                       graph_build_seconds=0. if final else None, decoder_and_hash_seconds=0. if final else None,
                       spectral_loss_seconds=0. if final else None, preprocessing_seconds=0. if final else None,
                       terminal_geometry_head_io_seconds=0. if final else None,
                       case_wall_before_drive_copy_seconds=0. if final else None) for arm in ARMS}
    ranks, receipt_proofs = {arm: [] for arm in ARMS[1:]}, []
    cache_count = 0
    cache_jobs, inference_jobs = 0, 0
    for shard, indices, labels in complete:
        receipts = {arm: read_receipt(found[shard['label']+'_'+arm], shard['label']+'_'+arm,
                    'cache' if arm == 'cache' else 'inference', final) for arm in ('cache', *ARMS)}
        cache = receipts['cache']['details']
        assert cache['selection_ids'] == indices and type(cache['cache_count']) is int and cache['cache_count'] == 15*len(indices)//2
        cache_count += cache['cache_count']
        cache_jobs += 1
        cases = {arm: check_inference(receipts[arm], indices) for arm in ARMS}
        for key, baseline in cases['control'].items():
            for arm in ARMS[1:]:
                other = cases[arm][key]
                assert all(other[field] == baseline[field] for field in ('ids', 'labels', 'paired_identity')), 'Paired IDs/labels/preparation identity differs'
            assert cases['half0'][key]['filter_sha256'] == cases['M100'][key]['filter_sha256'], 'Half-control/M100 filters differ'
            assert cases['half0'][key]['rank'] == cases['M100'][key]['rank'], 'Matched graph rank differs'
        for arm in ARMS:
            receipt = receipts[arm]
            details = receipt['details']
            inference_jobs += 1
            timing[arm]['worker_wall_seconds'] += number(details['worker_wall_seconds'])
            for corruption, counts in details['counts'].items():
                for key in ('n', 'pm_correct', 'terminal_correct'):
                    per_corruption[arm][corruption][key] += counts[key]
                timing[arm]['reconstruction_seconds'] += counts['reconstruction_seconds']
                timing[arm]['peak_gpu_memory_mb'] = max(timing[arm]['peak_gpu_memory_mb'], counts['peak_gpu_memory_mb'])
            for case in cases[arm].values():
                corruption = case['case'].split('/')[0]
                row = list(CORRUPTIONS).index(corruption)
                for offset, index in enumerate(case['ids']):
                    for readout in ('pm', 'terminal'):
                        outcomes[readout][arm][row, positions[index]] = case[readout][offset] == case['labels'][offset]
                if arm != 'control':
                    ranks[arm].extend(case['rank'])
            if final:
                for key, value in final_components(receipt['local_archive'], details['cases']).items():
                    timing[arm][key] += value
        for label in labels:
            receipt_proofs.append(dict(label=label, receipt=dict(path=str(found[label]), **identity(found[label])),
                archive=receipts[label.rsplit('_', 1)[1]]['archive_identity']))
    predictions_per_arm = 15*len(all_ids)
    for arm in ARMS:
        assert all(row['n'] == len(all_ids) for row in per_corruption[arm].values())
        assert sum(row['n'] for row in per_corruption[arm].values()) == predictions_per_arm
        for readout in ('pm', 'terminal'):
            assert sum(row[readout+'_correct'] for row in per_corruption[arm].values()) == int(outcomes[readout][arm].sum())
    if final:
        assert predictions_per_arm == 37020 and inference_jobs == 117 and cache_jobs == 39 and cache_count == 18510
    all_metrics = comparisons(outcomes, np.ones(len(all_ids), dtype=np.bool_))
    prior_mask = np.asarray([index in prior for index in all_ids], dtype=np.bool_)
    strata = {name: dict(objects=int(mask.sum()), predictions_per_arm=15*int(mask.sum()),
              comparisons=comparisons(outcomes, mask)) for name, mask in
              (('prior_M100_20', prior_mask), ('outside_prior_M100_2448', ~prior_mask))}
    gate = None
    if final:
        assert (strata['prior_M100_20']['objects'], strata['outside_prior_M100_2448']['objects']) == (20, 2448)
        primary = all_metrics['M100_vs_control']['pm']
        gate = dict(advance_to_external=primary['delta_pp'] >= 1. and primary['object_cluster_ci95_pp'][0] > 0.,
                    rule=plan['transfer_gate'], observed_delta_pp=primary['delta_pp'],
                    observed_object_cluster_ci95_pp=primary['object_cluster_ci95_pp'])
    rank_stats = {arm: dict(count=len(values), min=min(values) if values else None,
                  max=max(values) if values else None, median=float(np.median(values)) if values else None,
                  histogram={str(value): values.count(value) for value in sorted(set(values))}) for arm, values in ranks.items()}
    for arm in ARMS:
        if final:
            if arm == 'control':
                timing[arm]['decoder_and_hash_seconds'] = None
            timing[arm]['graph_plus_reconstruction_seconds'] = timing[arm]['graph_build_seconds']+timing[arm]['reconstruction_seconds']
    timing_report = dict(arms=timing, component_pending_until_final=not final, overhead_vs_control={})
    if final:
        baseline = timing['control']['graph_plus_reconstruction_seconds']
        timing_report['overhead_vs_control'] = {arm: dict(
            graph_plus_reconstruction_extra_seconds=timing[arm]['graph_plus_reconstruction_seconds']-baseline,
            graph_plus_reconstruction_ratio=timing[arm]['graph_plus_reconstruction_seconds']/baseline if baseline else None,
            reconstruction_extra_seconds=timing[arm]['reconstruction_seconds']-timing['control']['reconstruction_seconds'],
            scd_weight=.5, gsd_weight=0. if arm == 'half0' else 2216.8064620051765) for arm in ARMS[1:]}
    return dict(status='complete' if final else 'partial', external_gate=gate,
        coverage=dict(shards=len(complete), objects=len(all_ids), indices=all_ids, predictions_per_arm=predictions_per_arm,
                      inference_jobs=inference_jobs, cache_jobs=cache_jobs, actual_prepared_caches=cache_count),
        incomplete_shards=incomplete, unpaired_accepted_labels=sorted(set(found)-{label for _, _, labels in complete for label in labels}),
        comparisons=all_metrics, strata=strata, per_corruption=per_corruption, ranks=rank_stats, timing=timing_report,
        provenance=dict(plan=dict(path=str(plan_path), **identity(plan_path)),
            common_source_identity=dict(path=str(source_path), **identity(source_path)),
            raw_zip_sha_verified=final, receipts=receipt_proofs),
        timing_scope='Reconstruction includes PM probes; preprocessing is recorded separately. Control decoder/hash time is not separately measured, is included in reconstruction, and is reported as null; control spectral-loss time is zero. Terminal geometry/head/IO is one combined measurement covering geometry conversion/archive and final classifier work, not classifier-only time. Full case wall stops before the Drive copy and overlaps preprocessing, graph, reconstruction and terminal geometry/head/IO; do not add those components to case wall. Measured decoder/hash and spectral losses are within reconstruction and must not be added again. Graph + reconstruction excludes preprocessing, terminal geometry/head/IO, sealing and transfer. Common encode/style/noise preparation occurs outside measured replay; graph setup temporaries are excluded from reconstruction peak memory. Worker wall includes initialization/IO through worker completion. Cache-preparation, outer-job and whole-study wall timing remain pending separate journal receipts; these measurements are not deployment throughput.',
        limitation='Historical full-benchmark exposure includes all2468 IDs; the20/2448 strata refer only to recorded prior M100 use. One observed seed0; object-cluster intervals retain all15 corruptions per object and exclude native numerical variability. Prior native repeat gates failed; operational comparisons do not establish native causal attribution. Partial completion never changes the fixed full coverage or stopping rule.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    report = aggregate(args.root, args.final)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FULL_AGGREGATION', report['status'], report['coverage'], flush=True)
