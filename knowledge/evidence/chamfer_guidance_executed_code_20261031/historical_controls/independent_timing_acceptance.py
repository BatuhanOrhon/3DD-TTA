"""Local-only original timing acceptance; no partial accuracy or remote actions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

import numpy as np

PREP = Path(__file__).resolve().parent
sys.path.insert(0, str(PREP.parents[2]))
from scripts.analyze_chamfer_variants import _load, analyze_repeated_costs
from scripts.analyze_scan_trajectory import Bundle

DATASETS = ('modelnet40_c', 'scanobjectnn_c')
ARMS = ('baseline', 'sparse', 'semantic_kl', 'semantic_h', 'weighted')
HEADS = ('dgcnn', 'curvenet', 'pointnext', 'pointmae')


def identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def accept(manifest_path, output):
    records = [row for row in json.loads(Path(manifest_path).read_text())['records']
               if '/results/timing/' in row['path'] and row['path'].endswith('.zip')]
    assert len(records) == len({row['path'] for row in records}) == 120, 'Require all120 unique timing raw heads'
    receipt_path = PREP / 'gpu_timing_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['state']['status'] == 'complete' and len(receipt['state']['jobs']) == 30
    declared = {row['path']: {key: row[key] for key in ('bytes', 'sha256')}
                for job in receipt['state']['jobs'] for row in job['archive_identities']}
    assert declared == {row['path']: {key: row[key] for key in ('bytes', 'sha256')} for row in records}
    pinned_manifest_path = PREP / 'revision_3/assets_v1/manifest.json'
    pinned_manifest = json.loads(pinned_manifest_path.read_text())
    calibration_raw = (PREP / 'calibration.json').read_bytes()
    groups, runs, coverage, proofs = [[], [], []], {}, set(), []
    for number, row in enumerate(records):
        path = row['local_path']
        assert identity(path) == {key: row[key] for key in ('bytes', 'sha256')}
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        run = _load(path)
        config, arm, repeat = run['config'], run['arm'], row['repeat']
        dataset, head = config['dataset'], config['backbone']
        key = (dataset, arm, repeat, head)
        assert key not in coverage and repeat in (0, 1, 2)
        coverage.add(key)
        selected = json.loads((PREP / (dataset + '_development_ids.json')).read_text())['indices']
        assert config['seed'] == 0 and config['batch_size'] == 70 and config['selected_original_ids'] == selected
        assert len(selected) == 70 and set(config['corruptions']) == {'gaussian', 'background'} and config['max_batches'] == 0
        assert config['asset_manifest'] == pinned_manifest and not config.get('paired_state_root')
        for name, expected in run['sources'].items():
            relative = 'source_snapshot/' + name[5:] if name.startswith('repo/') else name[7:] if name.startswith('assets/') else None
            assert relative is not None and pinned_manifest['files'][relative] == expected, 'Executed source differs from original0fd seal'
        if arm == 'sparse':
            assert config['chamfer_experiment']['interval'] == 3
        if arm == 'weighted':
            assert config['chamfer_experiment']['density_k'] == 10
        bundle = Bundle(path)
        try:
            if arm in ('semantic_kl', 'semantic_h', 'weighted'):
                assert bundle.read('calibration_report.json') == calibration_raw
            for corruption in ('gaussian', 'background'):
                diagnostic = bundle.json('batches/' + corruption + '/batch0000/diagnostics.json')
                metrics, trace = diagnostic['guidance_metrics'], diagnostic['trajectory']
                horizon = 30 if corruption == 'background' else 10
                backwards = (11 if horizon == 30 else 4) if arm == 'sparse' else horizon
                assert diagnostic['preparation_mode'] == 'fresh_encode' and not diagnostic.get('paired_state')
                assert metrics['graph_calls'] == metrics['spectral_loss_calls'] == 0
                assert metrics['guidance_backward_calls'] == backwards and metrics['chamfer_calls'] == 2 * backwards
                assert all(math.isfinite(value) for value in metrics.values())
                if arm != 'baseline':
                    assert metrics['prior_forward_calls'] == horizon
                assert diagnostic['geometry_shape'] == [70, 2048, 3]
                if arm.startswith('semantic_'):
                    semantic = trace['semantic_guidance']
                    assert metrics['semantic_callback_calls'] == horizon and metrics['semantic_loss_calls'] == 2
                    assert semantic['reference']['forward_calls'] == 1 and not semantic['reference_votes']
                    assert [record['loop_count'] for record in semantic['records']] == ([15, 23] if horizon == 30 else [5, 8])
                    assert trace['operation_counts'] == dict(probe_decode_forward=3, probe_classifier_forward=12, semantic_reference_forward=1)
                if arm == 'weighted':
                    assert metrics['reference_density_calls'] == 1
        finally:
            bundle.close()
        groups[repeat].append(path)
        runs[key] = run
        proofs.append(dict(dataset=dataset, arm=arm, repeat=repeat, head=head, sha256=row['sha256']))
        if (number + 1) % 20 == 0:
            print('TIMING_RAW_HEADS_VERIFIED', number + 1, flush=True)
    assert coverage == {(dataset, arm, repeat, head) for dataset in DATASETS for arm in ARMS for repeat in range(3) for head in HEADS}
    for (dataset, arm, repeat, head), run in runs.items():
        baseline = runs[(dataset, 'baseline', repeat, head)]
        for corruption in ('gaussian', 'background'):
            assert np.array_equal(run['cells'][corruption]['labels'], baseline['cells'][corruption]['labels'])
    report = analyze_repeated_costs(groups)
    analysis_path = Path(output).with_name(Path(output).stem + '_analysis.json')
    write_new(analysis_path, report)
    costs = {(repeat['repeat'], row['dataset'], row['arm'], row['corruption']): row
             for repeat in report['repeats'] for row in repeat['costs']}
    assert len(costs) == 60 and all(set(row['heads']) == set(HEADS) for row in costs.values())
    summaries = []
    for dataset in DATASETS:
        for arm in ARMS[1:]:
            row = next(row for row in report['summary'] if row['dataset'] == dataset and row['arm'] == arm)
            value = dict(dataset=dataset, arm=arm, paired_arm_over_C=row['arm_over_baseline'],
                         median_paired_arm_over_C=statistics.median(row['arm_over_baseline']),
                         baseline_seconds=row['baseline_seconds'], arm_seconds=row['arm_seconds'], per_corruption={})
            for corruption in ('gaussian', 'background'):
                before = [costs[(repeat, dataset, 'baseline', corruption)]['total_seconds'] for repeat in range(3)]
                after = [costs[(repeat, dataset, arm, corruption)]['total_seconds'] for repeat in range(3)]
                ratios = [b / a for a, b in zip(before, after)]
                value['per_corruption'][corruption] = dict(baseline_seconds=before, arm_seconds=after,
                                                           paired_arm_over_C=ratios, median_paired_arm_over_C=statistics.median(ratios))
            summaries.append(value)
    proof = dict(status='independent_full_original_timing_accepted', raw_heads=120, raw_batches=240, repeats=3,
                 raw_proofs=proofs, manifest=dict(path=str(manifest_path), **identity(manifest_path)),
                 phase_receipt=dict(path=str(receipt_path), **identity(receipt_path)),
                 source_manifest=dict(path=str(pinned_manifest_path), **identity(pinned_manifest_path)),
                 analysis=dict(path=str(analysis_path), **identity(analysis_path)), summaries=summaries,
                 exact_encoded_initial_states=report['exact_encoded_initial_states'],
                 encoded_differences_reported=sum(len(repeat['encoded_state_differences']) for repeat in report['repeats']),
                 controls='Exact input/noise/scheduler/RNG, fixed native/source/checkpoint/state/IDs and coefficients; only explicitly reported encoded-state drift allowed in fresh timing',
                 accounting='Shared reconstruction counted once per actual group plus all4 terminal heads; primary summary is median of3 matched arm/C ratios, not ratio ofmedians',
                 limitations='Three measured timing repeats include probe/hash/runtime overhead; no accuracy, equal-time voting or single-head deployment inference. Geometry finiteness follows the verified runtime guard; raw coordinates are not stored.')
    write_new(output, proof)
    print(json.dumps(dict(status=proof['status'], raw_heads=120, output=str(output))), flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    accept(arguments.manifest, arguments.output)
