#!/usr/bin/env python3
"""CPU localization of native variation in six sealed zero-weight trace runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legacy_trajectory import tensor_hash

HELPER = ROOT/'knowledge/evidence/decoded_physical_accuracy_20261104/analyze_accuracy.py'
spec = importlib.util.spec_from_file_location('native_trace_accepted_accuracy', HELPER)
accepted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accepted)
require = accepted.require
BOUNDARIES = ('prior_input', 'scheduler', 'scd', 'gradients', 'post_update')
IDS = {'gaussian': [44, 91], 'background': [495, 512]}


def tensor_sha(value):
    # Existing Torch helper preserves scalar shape(), unlike ascontiguousarray.
    return tensor_hash(torch.from_numpy(np.asarray(value)))


def byte_identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def verify_executed_sources(bundle, sources):
    """Producer repo/assets aliases bind the same canonical sealed source bytes."""
    for name, expected in sources.items():
        prefix = next((p for p in ('assets/source_snapshot/', 'repo/') if name.startswith(p)), None)
        require(prefix is not None, 'Unexpected executed source prefix')
        relative = name[len(prefix):]
        path = PurePosixPath(relative)
        require(bool(path.parts) and not path.is_absolute() and '..' not in path.parts
                and '.' not in path.parts and path.as_posix() == relative
                and '\\' not in relative and ':' not in relative and '\x00' not in relative,
                'Unsafe executed source relative path')
        member = 'source_snapshot/'+relative
        require(byte_identity(bundle.read(member)) == expected, 'Executed source snapshot differs: '+member)


def field_shapes(boundary, n):
    local, style = (2, 4*n, 1, 1), (2, 128, 1, 1)
    if boundary == 'prior_input':
        return dict(noisy_local=local, conditioning_style=style, original_decoder_style=style,
                    original_h0_xyz=(2, n, 3), prior_t=(2,))
    if boundary == 'scheduler':
        return {key: local for key in ('noise_pred', 'pred_original_sample', 'prev_sample')}
    if boundary == 'gradients':
        return dict(noisy_local_gradient=local, conditioning_style_gradient=style)
    if boundary == 'post_update':
        return dict(noisy_local=local, conditioning_style=style)
    shapes = dict(scd_loss=())
    for direction in (1, 2):
        for field in ('distances', 'nearest_indices', 'sorted_distances'):
            shapes['%s_%d'%(field, direction)] = (2, n)
        shapes['retained_indices_%d'%direction] = (2, int(.95*n))
        for field in ('cutoff_distance', 'next_distance'):
            shapes['%s_%d'%(field, direction)] = (2,)
    return shapes


def load_trace(bundle, prefix, case, reference):
    """Strict trace payload checks after existing full-run acceptance."""
    report = bundle.json(prefix+'native_trace.json')
    timesteps = case['identity']['scheduler']['active_timesteps']
    limit, n = min(len(timesteps), 9), reference['prepared_x'].shape[1]
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1
            and report.get('original_ids') == case['index'].tolist() and report.get('complete') is True
            and type(report.get('loop_limit')) is int and report['loop_limit'] == limit,
            'Invalid native trace header/coverage')
    require(all(type(report.get(key)) is int and report[key] == 0 for key in
                ('extra_model_forward_calls', 'extra_chamfer_forward_calls', 'extra_backward_calls')),
            'Native trace added model/Chamfer/backward calls')
    snapshots = report['snapshots']
    expected = [(loop, boundary) for loop in range(1, limit+1) for boundary in BOUNDARIES]
    require([(v['loop_count'], v['boundary']) for v in snapshots] == expected, 'Missing/extra/reordered native boundary')
    expected_members = {prefix+'native_trace_loop%02d_%s.npz'%key for key in expected}
    require({name for name in bundle.names if name.startswith(prefix+'native_trace_loop')} == expected_members,
            'Native trace NPZ member coverage differs')
    result = []
    for row, (loop, boundary) in zip(snapshots, expected):
        filename = 'native_trace_loop%02d_%s.npz'%(loop, boundary)
        require(row['file'] == filename and type(row['loop_count']) is int
                and type(row['timestep']) is int and row['timestep'] == timesteps[loop-1], 'Trace filename/loop/timestep differs')
        raw = bundle.read(prefix+filename)
        require(byte_identity(raw) == row['identity'], 'Trace NPZ SHA/size mismatch')
        before, after = row['rng_before_sha256'], row['rng_after_sha256']
        require(row.get('rng_unchanged') is True and isinstance(before, str) and re.fullmatch('[0-9a-f]{64}', before)
                and before == after, 'Trace RNG guard failed')
        seconds = row['copy_and_serialization_seconds']
        require(type(seconds) in (int, float) and np.isfinite(seconds) and seconds >= 0, 'Invalid trace timing')
        values = accepted.arrays(bundle, prefix+filename)
        shapes = field_shapes(boundary, n)
        require(set(values) == set(shapes)|{'index'} and set(row['tensors']) == set(shapes), 'Trace tensor key coverage differs')
        require(values['index'].dtype == np.int64 and np.array_equal(values['index'], case['index']), 'Trace original IDs differ')
        for field, shape in shapes.items():
            value, metadata = values[field], row['tensors'][field]
            integer = field.startswith(('nearest_indices_', 'retained_indices_'))
            expected_dtype = np.int32 if field.startswith('nearest_indices_') else np.int64 if integer else np.float32
            require(value.shape == shape and value.dtype == expected_dtype and np.isfinite(value).all(), 'Invalid trace tensor shape/dtype/finite: '+field)
            require(metadata.get('available') is True and metadata.get('finite') is True
                    and metadata.get('shape') == list(shape) and metadata.get('dtype') == str(torch.from_numpy(value).dtype), 'Trace tensor header differs: '+field)
            require(tensor_sha(value) == metadata.get('sha256'), 'Tensor hash mismatch: '+field)
            norm = metadata.get('norm')
            require(type(norm) in (int, float) and np.isfinite(norm) and norm >= 0, 'Invalid recorded tensor norm')
            if value.ndim:
                norms = metadata.get('norm_per_object')
                require(isinstance(norms, list) and len(norms) == 2 and np.isfinite(norms).all()
                        and min(norms) >= 0, 'Invalid recorded per-object norms')
            if integer:
                require(((value >= 0) & (value < n)).all(), 'Trace index outside vertex range')
        if boundary == 'prior_input':
            initial = case['identity']['initial']
            require(tensor_sha(values['original_decoder_style']) == initial['shape_latent']
                    and tensor_sha(values['original_h0_xyz']) == tensor_sha(reference['initial_h0_xyz']), 'Original-z/h0 anchor changed')
            require(np.array_equal(values['prior_t'], np.full(2, timesteps[loop-1]+1, np.float32)), 'Actual prior t differs')
            if loop == 1:
                require(tensor_sha(values['noisy_local']) == initial['noisy_local']
                        and tensor_sha(values['conditioning_style']) == initial['style']
                        and before == case['identity']['initial_rng'], 'First prior input/RNG differs from paired initial state')
        if boundary == 'scd':
            for direction in (1, 2):
                distances, ordered = values['distances_%d'%direction], values['sorted_distances_%d'%direction]
                retained = values['retained_indices_%d'%direction]
                require((distances >= 0).all() and np.array_equal(np.sort(distances, axis=1), ordered)
                        and all(len(np.unique(indices)) == retained.shape[1] for indices in retained)
                        and np.array_equal(np.take_along_axis(distances, retained, 1), ordered[:, :retained.shape[1]])
                        and np.array_equal(values['cutoff_distance_%d'%direction], ordered[:, retained.shape[1]-1])
                        and np.array_equal(values['next_distance_%d'%direction], ordered[:, retained.shape[1]]), 'Retained sort/cutoff witness differs')
        result.append(dict(loop_count=loop, boundary=boundary, timestep=row['timestep'], member=bundle.prefix+prefix+filename,
                           rng_sha256=before, arrays={k: values[k] for k in shapes}))
    return result


def compare_arrays(left, right):
    require(left.shape == right.shape and left.dtype == right.dtype, 'Paired tensor shape/dtype differs')
    delta = left.astype(np.float64)-right.astype(np.float64)
    changed = left != right
    denominator = max(float(np.linalg.norm(left.astype(np.float64))), float(np.linalg.norm(right.astype(np.float64))))
    return dict(hash_equal=tensor_sha(left) == tensor_sha(right), numeric_difference=bool(changed.any()),
        changed_count=int(changed.sum()), discrete_changed_count=int(changed.sum()) if left.dtype.kind in 'iu' else None,
        max_absolute_difference=float(np.abs(delta).max()),
        relative_l2=float(np.linalg.norm(delta))/denominator if denominator else 0.,
        per_object_rms=np.sqrt(np.mean(delta.reshape(2, -1)**2, axis=1)).tolist() if left.ndim and left.shape[0] == 2 else None,
        changed_count_per_object=changed.reshape(2, -1).sum(1).tolist() if left.ndim and left.shape[0] == 2 else None)


def compare_traces(left, right, ids):
    """No new causal threshold: bit identity, nonzero numbers and sizes are separate."""
    require([(v['loop_count'], v['boundary']) for v in left] == [(v['loop_count'], v['boundary']) for v in right], 'Trace order differs')
    lookup = [{(v['loop_count'], v['boundary']): v for v in trace} for trace in (left, right)]
    rows, observations = [], []
    rng_differences = []
    earliest_hash, earliest_numeric = None, None
    for a, b in zip(left, right):
        loop, boundary = a['loop_count'], a['boundary']
        if a['rng_sha256'] != b['rng_sha256']:
            rng_differences.append(dict(loop_count=loop, boundary=boundary,
                                       sha256=[a['rng_sha256'], b['rng_sha256']]))
        for field in a['arrays']:
            row = dict(loop_count=loop, boundary=boundary, timestep=a['timestep'], tensor=field,
                       original_ids=ids, **compare_arrays(a['arrays'][field], b['arrays'][field]))
            rows.append(row)
            if not row['hash_equal'] and earliest_hash is None:
                earliest_hash = row
            if row['numeric_difference'] and earliest_numeric is None:
                earliest_numeric = row
        # Inputs below are exactly archived inputs, not claims about unobserved activations.
        specs = []
        if boundary == 'scheduler':
            specs = [('local_prior_forward', ['noise_pred'], [('prior_input', None)]),
                     ('scheduler_step', ['pred_original_sample', 'prev_sample'], [('prior_input', ['noisy_local']), ('scheduler', ['noise_pred'])])]
        elif boundary == 'scd':
            specs = [('chamfer_forward', ['distances_1', 'distances_2', 'nearest_indices_1', 'nearest_indices_2'], [('scheduler', ['pred_original_sample']), ('prior_input', ['original_h0_xyz'])]),
                     ('sort_and_truncate', [k for k in a['arrays'] if k.startswith(('sorted_', 'retained_', 'cutoff_', 'next_'))], [('scd', ['distances_1', 'distances_2'])]),
                     ('SCD_reduction', ['scd_loss'], [('scd', ['sorted_distances_1', 'sorted_distances_2'])])]
        elif boundary == 'gradients':
            specs = [('joint_backward_or_accumulation', list(a['arrays']), [('prior_input', None), ('scheduler', None), ('scd', None)])]
        elif boundary == 'post_update':
            specs = [('joint_update_arithmetic', list(a['arrays']), [('prior_input', ['conditioning_style']), ('scheduler', ['prev_sample']), ('gradients', None)])]
        for operator, outputs, sources in specs:
            differing = [k for k in outputs if not np.array_equal(a['arrays'][k], b['arrays'][k])]
            if not differing:
                continue
            inputs, equal = [], True
            for source_boundary, fields in sources:
                source_a, source_b = (table[(loop, source_boundary)] for table in lookup)
                for field in source_a['arrays'] if fields is None else fields:
                    hashes = [tensor_sha(v['arrays'][field]) for v in (source_a, source_b)]
                    equal = equal and hashes[0] == hashes[1]
                    inputs.append(dict(boundary=source_boundary, tensor=field, members=[source_a['member'], source_b['member']], sha256=hashes, original_ids=ids))
            # Candidate stochastic/branch inputs must share their actual RNG boundary.
            input_stage = 'prior_input' if operator in ('local_prior_forward', 'scheduler_step') else boundary
            rng_equal = lookup[0][(loop, input_stage)]['rng_sha256'] == lookup[1][(loop, input_stage)]['rng_sha256']
            observations.append(dict(operator=operator, loop_count=loop, boundary=boundary, outputs=differing,
                inputs_bit_exact=bool(equal), rng_boundary_equal=rng_equal, exact_input_candidate=bool(equal and rng_equal),
                status='EXACT_ARCHIVED_INPUT_CANDIDATE' if equal and rng_equal else 'PROPAGATED_INPUT_OR_RNG_DIFFERENCE',
                input_references=inputs, internal_activations_archived=False))
    candidates = [v for v in observations if v['exact_input_candidate']]
    return dict(earliest_hash_difference=earliest_hash, earliest_numeric_difference=earliest_numeric,
        earliest_rng_difference=rng_differences[0] if rng_differences else None,
        all_trace_rng_boundaries_equal=not rng_differences, rng_differences=rng_differences,
        earliest_exact_input_candidate=candidates[0] if candidates else None, operator_observations=observations,
        numeric_meaning='Nonzero numeric difference ignores signed-zero-only hashes; no boundary-specific materiality threshold was preregistered, so sizes are descriptive and do not certify a native cause',
        native_kernel_identified=False, full_rng_values_available=False), rows


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def analyze(paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs, validated = [], []
    try:
        require(len(paths) == 6, 'Exactly six fresh trace ZIPs are required')
        inputs = [dict(path=str(Path(p).absolute()), **accepted.identity(p)) for p in paths]
        require(len({v['sha256'] for v in inputs}) == 6, 'Repeated ZIP identity is not a fresh process replicate')
        write_json(output/'inputs.json', dict(raw=inputs, analyzer=accepted.identity(Path(__file__)), helper=accepted.identity(HELPER)))
        groups = {c: [] for c in IDS}
        global_identity = None
        for path in paths:
            run = accepted.load_run(path)
            require(run['arm'] == 'control' and len(run['corruptions']) == 1, 'Expected one-corruption operational control')
            corruption = run['corruptions'][0]
            require(corruption in IDS and run['ids'] == IDS[corruption] and len(run['cases']) == 1, 'Wrong old Gaussian/Background scope')
            bundle = accepted.open_bundle(path)
            try:
                config = bundle.json('config.json')
                require(config.get('capture_native_trace') is True and not config.get('capture_h5')
                        and not config.get('capture_shared_states'), 'Wrong separate native-trace capture flags')
                common = {k: v for k, v in run['control_identity'].items() if k != 'dataset_provenance'}
                if global_identity is None:
                    global_identity = common
                require(common == global_identity, 'Cross-corruption source/native/model/backend identity differs')
                verify_executed_sources(bundle, config['executed_source_files'])
                key, case = next(iter(run['cases'].items()))
                prefix = 'batches/%s/batch0000/'%corruption
                diagnostic = bundle.json(prefix+'diagnostics.json')
                link = diagnostic['native_trace']
                require(link == dict(file='native_trace.json', **byte_identity(bundle.read(prefix+'native_trace.json'))), 'Trace diagnostic identity differs')
                reference = accepted.arrays(bundle, prefix+'reference.npz')
                trace = load_trace(bundle, prefix, case, reference)
                item = dict(run=run, trace=trace, run_id=config['run_id'])
                groups[corruption].append(item)
                validated.append(dict(path=str(path), archive_identity=run['archive_identity'], run_id=config['run_id'],
                                      corruption=corruption, snapshots=len(trace), original_ids=run['ids']))
                write_json(output/('validated_run_%02d.json'%len(validated)), validated[-1])
            finally:
                bundle.close()
        require(all(len(values) == 3 and len({v['run_id'] for v in values}) == 3 for values in groups.values()), 'Need three distinct process/run IDs per corruption')
        summaries, csv_rows = [], []
        for corruption, group in groups.items():
            group.sort(key=lambda v: v['run_id'])
            for left, right in itertools.combinations(group, 2):
                accepted.check_pairing(left['run'], right['run'])
                gate = accepted.repeat_gate(left['run'], right['run'])
                summary, differences = compare_traces(left['trace'], right['trace'], IDS[corruption])
                name = '%s_%s_vs_%s'%(corruption, left['run_id'], right['run_id'])
                require(re.fullmatch('[A-Za-z0-9_.-]+', name), 'Unsafe run ID')
                archives = [dict(path=v['run']['path'], **v['run']['archive_identity']) for v in (left, right)]
                for observation in summary['operator_observations']:
                    observation['input_archives'] = archives
                write_json(output/(name+'_trace_differences.json'),
                           {str(v['loop_count'])+'/'+v['boundary']+'/'+v['tensor']: v for v in differences})
                summary.update(corruption=corruption, run_ids=[left['run_id'], right['run_id']], archives=archives,
                               old_native_repeat_gate=gate)
                summaries.append(summary)
                for row in differences:
                    csv_rows.append(dict(pair=name, **{key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}))
                write_json(output/(name+'_localization.json'), summary)
        with (output/'tensor_differences.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader(); writer.writerows(csv_rows)
        require(all(accepted.identity(v['path']) == {key: v[key] for key in ('bytes', 'sha256')} for v in inputs), 'Raw input changed during analysis')
        result = dict(status='complete', raw_inputs=inputs, validated_runs=validated, repeat_pairs=summaries,
            old_native_repeat_gates_all_passed=all(v['old_native_repeat_gate']['passed'] for v in summaries),
            native_kernel_identified=False, new_causal_threshold=None,
            limitations=['Only loops1..9 are bracketed; Background later divergence can remain outside capture.',
                'Nonzero numeric size is reported without a new materiality gate; hash-only differences are not causal evidence.',
                'Exact archived inputs bracket an operator candidate, not its internal activation/Jacobian or a unique native kernel.',
                'Trace I/O synchronizes execution; no uninstrumented CUDA noninterference, timing or accuracy claim.'])
        write_json(output/'summary.json', result)
        return result
    except BaseException:
        write_json(output/'failure.json', dict(status='failed', inputs=inputs, validated_runs=validated, traceback=traceback.format_exc()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles', nargs=6, required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.bundles, args.output_dir), allow_nan=False))


if __name__ == '__main__':
    main()
