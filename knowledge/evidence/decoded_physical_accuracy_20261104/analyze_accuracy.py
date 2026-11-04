"""Independent CPU acceptance/recount of sealed physical-guidance accuracy arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
import stat
import sys
import zipfile

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent/'scripts/analyze_scan_trajectory.py').is_file():
        sys.path.insert(0, str(parent))
        break
from scripts.analyze_scan_trajectory import Bundle, require
from research_artifacts import CORRUPTIONS

WEIGHT = 802.5000841164167
ARMS = ('control', 'live0', 'full', 'h8')
THRESHOLDS = dict(logit_max=.05, logit_per_object_checkpoint_rms=.01, first_decoded_per_object_rms=1e-4)
COHORT_SHA256 = '40a03b0db7bcbaf1d0f7ac8e554c9ad53bf4c1c882bd32f2d2f9db7ecf07e5b4'
COHORT_IDS = [19, 57, 179, 183, 284, 487, 495, 512, 678, 716, 854, 869, 905, 1041, 1075, 1178,
              1187, 1305, 1311, 1360, 1541, 1593, 1728, 1759, 1766, 1811, 1993, 2021, 2266, 2419, 2441, 2442]


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_identity(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def arrays(bundle, name):
    with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def open_bundle(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) < 20000 and sum(m.file_size for m in members) < 3*1024**3
                and all(stat.S_IFMT(m.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR)
                        and m.file_size < 256*1024**2 for m in members), 'Unsafe/oversized sealed ZIP')
    return Bundle(path)


def recount(logits):
    values = np.asarray(logits, dtype=np.float64)
    require(values.ndim == 3 and values.shape[0] > 0 and values.shape[1] == 4
            and values.shape[2] >= 2 and np.isfinite(values).all(), 'Invalid finite trajectory logits')
    probabilities = np.exp(values-values.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    return probabilities.mean(1).argmax(-1), values[:, -1].argmax(-1)


def validate_predictions(payload):
    pm, terminal = recount(payload['trajectory_logits'])
    require(np.array_equal(payload['prediction'], pm) and np.array_equal(payload['probability_mean'], pm), 'Stored PM differs from independent recount')
    require(np.array_equal(payload['terminal_prediction'], terminal)
            and np.array_equal(payload['logits'], payload['trajectory_logits'][:, -1]), 'Stored terminal differs from recount')
    return pm, terminal


def arm_name(config):
    arm, weight, frozen = config['arm'], config['weight'], config['freeze_decoder_residual']
    require(type(frozen) is bool and type(weight) in (int, float) and np.isfinite(weight), 'Invalid arm flags/weight')
    if arm in ('control', 'live_control'):
        require(weight == 0 and frozen is False, 'Zero-weight control flags differ')
        return 'control' if arm == 'control' else 'live0'
    require(arm == 'physical' and weight == WEIGHT, 'Applied arm differs from frozen physical coefficient')
    return 'h8' if frozen else 'full'


def load_run(path):
    archive_identity = identity(path)
    bundle = open_bundle(path)
    try:
        require(all(name in bundle.names for name in ('command.txt', 'config.json', 'environment.txt',
                'stdout.log', 'summary.csv', 'per_corruption.csv', 'notes.md')), 'Required raw bundle files missing')
        config = bundle.json('config.json')
        require(config['diagnostic_complete'] is True and config['status'] == 'partial'
                and config['dataset'] == 'modelnet40_c' and config['severity'] == 5 and config['seed'] == 0
                and config['backbone'] == 'pointmae' and config['num_classes'] == 40
                and config['batch_size'] == 2 and config['count'] in (2, 8), 'Incomplete/wrong accuracy protocol')
        arm = arm_name(config)
        require(config['physical_graph'] == (None if arm == 'control' else
                dict(k=10, delta=.1, graph_gamma=.6, modes=100, profile='smooth', beta=2.)), 'Physical graph protocol differs')
        require(config['lion_state_before'] == config['lion_state_after']
                and config['classifier_state_before_sha256'] == config['classifier_state_after_sha256'], 'Frozen model state changed')
        require(config['backend_policy'] == 'controlled' and not config['native_decoder_audit'], 'Wrong backend/diagnostic cost policy')
        runtime = config['runtime']
        require(runtime['cudnn_benchmark'] is False and runtime['cudnn_deterministic'] is True
                and runtime['cudnn_allow_tf32'] is False and runtime['cuda_matmul_allow_tf32'] is False
                and runtime['deterministic_algorithms'] is False, 'Actual controlled backend flags differ')
        require(config['guidance_protocol'] == dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False)
                and config['normal_reverse_steps'] == 8 and config['background_reverse_steps'] == 35, 'Guidance host differs')
        ids, corruptions = config['selected_original_ids'], config['corruptions']
        require(len(ids) == config['count'] and ids == sorted(set(ids))
                and all(type(i) is int and 0 <= i < 2468 for i in ids), 'Invalid selected IDs')
        require(corruptions and len(set(corruptions)) == len(corruptions) and set(corruptions) <= set(CORRUPTIONS), 'Invalid corruption scope')
        if config.get('index_manifest'):
            selection = bundle.read('selection_manifest.json')
            require(hashlib.sha256(selection).hexdigest() == config['index_manifest_sha256']
                    and json.loads(selection) == config['selection_manifest']
                    and config['selection_manifest']['indices'] == ids, 'Selection manifest differs')
        common = {key: config[key] for key in ('executed_source_files', 'native_extensions', 'native_bootstrap',
                  'asset_manifest_sha256', 'dataset_provenance', 'guidance_protocol', 'runtime',
                  'lion_state_before', 'classifier_state_before_sha256')}
        rows = list(csv.DictReader(io.StringIO(bundle.read('per_corruption.csv').decode())))
        require(len(rows) == len(corruptions) and {r['corruption'] for r in rows} == set(corruptions), 'Corruption CSV coverage differs')
        csv_by_corruption = {r['corruption']: r for r in rows}
        cases, counts, costs = {}, {}, []
        for corruption in corruptions:
            require(config['completed_batches'][corruption] == config['count']//2, 'Incomplete corruption batches')
            corruption_payloads = []
            for batch in range(config['count']//2):
                case = 'batches/%s/batch%04d/' % (corruption, batch)
                payload = arrays(bundle, case+'predictions.npz')
                expected_ids = ids[batch*2:batch*2+2]
                require(payload['trajectory_logits'].shape == (2, 4, 40) and np.array_equal(payload['index'], expected_ids), 'Batch IDs/logit shape differ')
                require(payload['label'].shape == (2,) and np.issubdtype(payload['label'].dtype, np.integer)
                        and ((payload['label'] >= 0) & (payload['label'] < 40)).all(), 'Invalid labels')
                pm, terminal = validate_predictions(payload)
                reference = arrays(bundle, case+'reference.npz')
                require(set(reference) == {'prepared_x', 'initial_h0_xyz'}
                        and all(value.shape == (2, 2048, 3) and np.isfinite(value).all() for value in reference.values()), 'Invalid paired reference')
                first = arrays(bundle, case+'decoded_first_checkpoint.npz')['decoded_xyz']
                require(first.shape == (2, 2048, 3) and np.isfinite(first).all(), 'Invalid first decoded geometry')
                diagnostic = bundle.json(case+'diagnostics.json')
                trajectory = diagnostic['trajectory']
                require(trajectory['complete'] is True and trajectory['probe_rng_unchanged'] is True
                        and trajectory['probe_buffers_unchanged'] is True, 'Incomplete/interfering observer')
                steps = [9, 18, 27, 35] if corruption == 'background' else [2, 4, 6, 8]
                require(payload['candidate_loop_counts'].tolist() == steps, 'Candidate schedule differs')
                require(tensor_identity(reference['prepared_x']) == diagnostic['prepared_x_sha256'], 'Prepared tensor SHA differs')
                paired = dict(scheduler=diagnostic['scheduler'], initial=trajectory['initial_hashes'],
                    initial_rng=trajectory['initial_rng_sha256'], paired=diagnostic['paired_state']['sha256'],
                    prepared=diagnostic['prepared_x_sha256'], input_rng=diagnostic['input_rng_before_sha256'],
                    output_rng=diagnostic['reconstruction_rng_after_sha256'],
                    reference={k: tensor_identity(v) for k, v in reference.items()})
                key = corruption+'/'+str(batch)
                cases[key] = dict(identity=paired, index=payload['index'], label=payload['label'],
                                  logits=payload['trajectory_logits'], first=first, pm=pm, terminal=terminal,
                                  corruption=corruption, batch=batch, physical_metric_identity=None)
                graph_seconds = 0.
                require((case+'graph.json' in bundle.names) == (arm != 'control'), 'Graph diagnostic coverage differs')
                if case+'graph.json' in bundle.names:
                    graph = bundle.json(case+'graph.json')
                    graph_rows = graph['diagnostics']
                    require(isinstance(graph_rows, list) and len(graph_rows) == 2, 'Graph diagnostic shape differs')
                    require(graph['reference_xyz_sha256'] == diagnostic['prepared_x_sha256']
                            and isinstance(graph['filter_sha256'], list) and len(graph['filter_sha256']) == 2
                            and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in graph['filter_sha256']), 'Physical graph tensor identities differ')
                    cases[key]['physical_metric_identity'] = {k: graph[k] for k in ('reference_xyz_sha256', 'filter_sha256')}
                    if case+'physical_filters.npz' in bundle.names:
                        filters = arrays(bundle, case+'physical_filters.npz')['filters']
                        require(filters.shape == (2, 2048, 2048) and filters.dtype == np.float32
                                and np.isfinite(filters).all()
                                and [tensor_identity(f) for f in filters] == graph['filter_sha256'], 'Archived physical filter SHA differs')
                    seconds = [float(r['runtime_seconds']) for r in graph_rows]
                    require(np.isfinite(seconds).all() and min(seconds) >= 0, 'Invalid graph diagnostic timing')
                    graph_seconds = sum(seconds)
                elapsed = float(diagnostic['elapsed_seconds'])
                require(np.isfinite(elapsed) and elapsed >= 0, 'Invalid case timing')
                costs.append(dict(corruption=corruption, batch=batch, case_elapsed_seconds=elapsed, graph_diagnostic_seconds=graph_seconds))
                corruption_payloads.append(payload)
            correct = sum(int((p['prediction'] == p['label']).sum()) for p in corruption_payloads)
            terminal_correct = sum(int((p['terminal_prediction'] == p['label']).sum()) for p in corruption_payloads)
            row = csv_by_corruption[corruption]
            require(all(row[key] == str(config[key]) for key in ('run_id', 'dataset', 'method', 'seed', 'severity')),
                    'Corruption CSV metadata differs from config')
            require(int(row['n_examples']) == config['count'] and int(row['n_correct']) == correct
                    and float(row['accuracy']) == correct/config['count'] and row['status'] == 'partial', 'CSV count/PM recount differs')
            aggregate = arrays(bundle, corruption+'_predictions.npz')
            for key in ('index', 'label', 'prediction', 'terminal_prediction', 'logits', 'trajectory_logits'):
                require(np.array_equal(aggregate[key], np.concatenate([p[key] for p in corruption_payloads])), 'Corruption payload differs: '+key)
            counts[corruption] = dict(n=config['count'], pm_correct=correct, terminal_correct=terminal_correct,
                reconstruction_seconds=float(row['runtime_seconds']), peak_gpu_memory_mb=float(row['peak_gpu_memory_mb']))
            require(np.isfinite([counts[corruption]['reconstruction_seconds'], counts[corruption]['peak_gpu_memory_mb']]).all()
                    and min(counts[corruption]['reconstruction_seconds'], counts[corruption]['peak_gpu_memory_mb']) >= 0, 'Invalid CSV cost')
        summary = list(csv.DictReader(io.StringIO(bundle.read('summary.csv').decode())))
        require(len(summary) == 1 and int(summary[0]['total_examples']) == config['count']*len(corruptions)
                and int(summary[0]['total_correct']) == sum(c['pm_correct'] for c in counts.values())
                and int(summary[0]['n_corruptions']) == len(corruptions)
                and np.isclose(float(summary[0]['macro_accuracy']), np.mean([c['pm_correct']/c['n'] for c in counts.values()]), rtol=0, atol=1e-12), 'Summary recount differs')
        wall = float(config['wall_seconds'])
        require(np.isfinite(wall) and wall >= 0, 'Invalid worker wall time')
        require(identity(path) == archive_identity, 'Raw archive changed during acceptance')
        return dict(path=str(path), archive_identity=archive_identity, arm=arm, ids=ids, corruptions=corruptions,
                    selection_manifest=config.get('selection_manifest'), control_identity=common, cases=cases,
                    counts=counts, costs=costs, worker_wall_seconds=wall)
    finally:
        bundle.close()


def check_pairing(left, right):
    require(left['control_identity'] == right['control_identity'], 'Source/native/backend/model control identity changed')
    require(set(left['cases']) == set(right['cases']), 'Paired case coverage differs')
    for key, a in left['cases'].items():
        b = right['cases'][key]
        require(a['identity'] == b['identity'], 'Paired preparation/RNG identity changed: '+key)
        require(np.array_equal(a['index'], b['index']) and np.array_equal(a['label'], b['label']), 'Paired IDs/labels differ: '+key)


def repeat_gate(control, live):
    check_pairing(control, live)
    rows = []
    for key, a in control['cases'].items():
        b = live['cases'][key]
        delta = a['logits'].astype(np.float64)-b['logits'].astype(np.float64)
        decoded = a['first'].astype(np.float64)-b['first'].astype(np.float64)
        rms = np.sqrt(np.mean(delta**2, axis=-1))
        first_rms = np.sqrt(np.mean(decoded**2, axis=(1, 2)))
        rows.append(dict(case=key, original_ids=a['index'].tolist(), logit_max_absolute=float(np.abs(delta).max()),
            logit_per_object_checkpoint_rms=rms.tolist(), first_decoded_per_object_rms=first_rms.tolist(),
            pm_agreement=(a['pm'] == b['pm']).tolist(), terminal_agreement=(a['terminal'] == b['terminal']).tolist(),
            passed=bool(np.abs(delta).max() <= .05 and (rms <= .01).all() and (first_rms <= 1e-4).all())))
    return dict(passed=all(r['passed'] for r in rows), thresholds=THRESHOLDS, cases=rows,
                logit_max_absolute=max(r['logit_max_absolute'] for r in rows),
                logit_per_object_checkpoint_rms_max=max(max(max(v) for v in r['logit_per_object_checkpoint_rms']) for r in rows))


def physical_metric_gate(runs):
    mismatches = []
    for key, case in runs['live0']['cases'].items():
        for arm in ('full', 'h8'):
            other = runs[arm]['cases'][key]['physical_metric_identity']
            if case['physical_metric_identity'] != other:
                mismatches.append(dict(case=key, arm=arm, live0=case['physical_metric_identity'], other=other))
    return dict(passed=not mismatches, compared_cases=len(runs['live0']['cases']),
                mismatches=mismatches, criterion='Exact physical reference/filter tensor SHA equality across live0/full/H8; control has no graph')


def run_summary(run):
    result = {key: run[key] for key in ('path', 'archive_identity', 'arm', 'ids', 'corruptions', 'counts', 'costs', 'worker_wall_seconds')}
    result['physical_metrics'] = {key: case['physical_metric_identity'] for key, case in run['cases'].items()}
    return result


def technical(paths):
    require(len(paths) == 4, 'Technical acceptance requires four archives')
    loaded = [load_run(p) for p in paths]
    runs = {r['arm']: r for r in loaded}
    require(set(runs) == set(ARMS) and all(r['ids'] == [44, 91] and r['corruptions'] == ['gaussian']
                                         for r in loaded), 'Technical arms/IDs/corruptions differ')
    for arm in ARMS[1:]:
        check_pairing(runs['control'], runs[arm])
    gate = repeat_gate(runs['control'], runs['live0'])
    metric_gate = physical_metric_gate(runs)
    return dict(stage='technical', passed=gate['passed'] and metric_gate['passed'], zero_guidance_control_live0=gate,
                matched_physical_metric_gate=metric_gate,
                positive_arms_finite_and_paired=True, control_identity_sha256=fingerprint(runs['control']['control_identity']),
                runs={arm: run_summary(run) for arm, run in runs.items()},
                limitation='Controlled backend does not guarantee native determinism. Only control/live0 numerical compatibility is gated; applied full/H8 outputs may differ.')


def paired_metrics(winner, baseline):
    """One observed seed; resample objects retaining all15 corruptions together."""
    winner, baseline = np.asarray(winner), np.asarray(baseline)
    require(winner.shape == baseline.shape and winner.ndim == 2 and winner.shape[0] == 15
            and winner.shape[1] > 0 and winner.dtype == baseline.dtype == np.bool_, 'Require paired15corruption xobject boolean outcomes for seed0')
    delta = winner.astype(np.float64)-baseline
    object_delta = delta.mean(0)
    rng = np.random.RandomState(20261101)
    bootstrap = object_delta[rng.randint(len(object_delta), size=(10000, len(object_delta)))].mean(1)*100
    return dict(delta_pp=float(delta.mean()*100), winner_accuracy_pct=float(winner.mean()*100),
                baseline_accuracy_pct=float(baseline.mean()*100), seeds=[0], seed_delta_pp=[float(delta.mean()*100)],
                per_corruption_delta_pp={c: float(value) for c, value in zip(CORRUPTIONS, delta.mean(1)*100)},
                corrected=int((winner & ~baseline).sum()), harmed=int((~winner & baseline).sum()),
                object_cluster_ci95_pp=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                bootstrap_repetitions=10000, bootstrap_seed=20261101,
                limitation='Conditional development-cohort interval at one observed seed0; all15 corruptions stay together within each resampled object. No seed replication or unseen-confirmation claim.')


def main_analysis(mapping_path):
    mapping_path = Path(mapping_path)
    mapping_identity = identity(mapping_path)
    stages = json.loads(mapping_path.read_text())
    require(isinstance(stages, dict) and set(stages) == {'shard%d'%i for i in range(4)}, 'Require four declared shards')
    loaded, gates = {}, {}
    combined = {arm: dict(control_identity=None, cases={}) for arm in ARMS}
    outcomes = {readout: {arm: np.zeros((15, 32), dtype=bool) for arm in ARMS} for readout in ('pm', 'terminal')}
    coverage = {arm: set() for arm in ARMS}
    for shard in range(4):
        name = 'shard%d'%shard
        require(set(stages[name]) == set(ARMS), 'Each shard requires control/live0/full/h8')
        runs = {arm: load_run(stages[name][arm]) for arm in ARMS}
        for arm, run in runs.items():
            require(run['arm'] == arm and run['ids'] == COHORT_IDS[shard*8:shard*8+8]
                    and run['corruptions'] == list(CORRUPTIONS), 'Main arm/shard/corruption scope differs')
            selection = run['selection_manifest']
            require(selection['parent_manifest_sha256'] == COHORT_SHA256 and selection['shard'] == shard,
                    'Shard parent selection identity differs')
            check_pairing(runs['control'], run)
            if combined[arm]['control_identity'] is None:
                combined[arm]['control_identity'] = run['control_identity']
            require(combined[arm]['control_identity'] == run['control_identity'], 'Source/native/backend/model identity changed across shards')
            for key, case in run['cases'].items():
                combined[arm]['cases'][name+'/'+key] = case
                ci = list(CORRUPTIONS).index(case['corruption'])
                for j, object_id in enumerate(case['index']):
                    oi = COHORT_IDS.index(int(object_id))
                    require((ci, oi) not in coverage[arm], 'Duplicate object/corruption pairing')
                    coverage[arm].add((ci, oi))
                    for readout in outcomes:
                        outcomes[readout][arm][ci, oi] = case[readout][j] == case['label'][j]
        gates[name] = repeat_gate(runs['control'], runs['live0'])
        loaded[name] = runs
    require(all(len(seen) == 15*32 for seen in coverage.values()), 'Missing main object/corruption conditions')
    overall_gate = repeat_gate(combined['control'], combined['live0'])
    metric_gate = physical_metric_gate(combined)
    comparisons = {}
    for winner, baseline, role in (('full', 'live0', 'primary'), ('h8', 'live0', 'primary'),
            ('full', 'control', 'operational'), ('h8', 'control', 'operational'),
            ('h8', 'full', 'surrogate_vs_full'), ('live0', 'control', 'forward_control')):
        comparisons[winner+'_vs_'+baseline] = dict(role=role,
            **{readout: paired_metrics(matrix[winner], matrix[baseline]) for readout, matrix in outcomes.items()})
    timing = {}
    for arm in ARMS:
        runs = [loaded['shard%d'%s][arm] for s in range(4)]
        timing[arm] = dict(reconstruction_seconds=sum(c['reconstruction_seconds'] for run in runs for c in run['counts'].values()),
            per_case_elapsed_seconds_sum=sum(c['case_elapsed_seconds'] for run in runs for c in run['costs']),
            graph_diagnostic_seconds_sum=sum(c['graph_diagnostic_seconds'] for run in runs for c in run['costs']),
            worker_wall_seconds_sum=sum(run['worker_wall_seconds'] for run in runs),
            peak_gpu_memory_mb_max=max(c['peak_gpu_memory_mb'] for run in runs for c in run['counts'].values()))
    require(identity(mapping_path) == mapping_identity, 'Stage mapping changed during analysis')
    return dict(stage='main', status='complete' if metric_gate['passed'] else 'complete_with_matched_metric_failure',
        numerical_attribution_gate_passed=overall_gate['passed'] and metric_gate['passed'],
        matched_physical_metric_gate=metric_gate,
        stage_mapping=dict(path=str(mapping_path), **mapping_identity), cohort_manifest_sha256=COHORT_SHA256,
        indices=COHORT_IDS, corruptions=list(CORRUPTIONS), seeds=[0], object_conditions_per_arm=480,
        outcomes={r: {a: m.tolist() for a, m in values.items()} for r, values in outcomes.items()},
        comparisons=comparisons, live0_control_gate=overall_gate, shard_gates=gates, timing=timing,
        runs={s: {a: run_summary(r) for a, r in runs.items()} for s, runs in loaded.items()},
        timing_scope='Reconstruction CSV is synchronized reconstruction including PM probes, excluding graph setup/final head. Per-case elapsed also includes terminal head/geometry/artifact work. Graph diagnostic runtime sums follow per-object graph records and may include asynchronous dispatch. Worker wall includes initialization and IO through the worker loop, excludes final sealing/Drive publication. Common cache preparation is external and must be reported separately. These are development-run costs, not deployment throughput.',
        limitation='Four disjoint8-ID shards, one real seed0, stage-disjoint development with historical benchmark exposure. Live0/control gate failure prevents attributing small separate-process changes to guidance. Positive guidance effect and uncertainty remain conditional; no full-benchmark/default promotion.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('technical', 'main'))
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    if args.mode == 'main':
        require(len(args.paths) == 1, 'Main analysis requires one stage-path mapping JSON')
        result = main_analysis(args.paths[0])
    else:
        result = technical(args.paths)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.mode == 'technical' and not result['passed']:
        raise SystemExit(1)
