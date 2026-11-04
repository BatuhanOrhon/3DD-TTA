"""Posthoc CPU clean-oracle analysis of sealed P1 physical geometry; no inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_followup import arrays, open_bundle
from scripts.analyze_scan_trajectory import require

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

IDS = [459, 545, 1229, 1435, 1441, 1557, 1881, 2095]
SELECTION_SHA256 = '9d39e53cb485552e45ba74131a31902d326783ce5d908bfb96ebcb1138f7e3d5'
CLEAN_IDENTITY = dict(path='/content/3DD-TTA/data/modelnet40_c/data_original.npy', bytes=30326912,
                      sha256='8aa5000d74a7ee7c3f827d9e80a1b325a12544a01b23aa2c76f990c0109a260c')
METRIC_ATOL = 1e-6


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def xyz(value):
    result = np.asarray(value, dtype=np.float64)
    require(result.ndim == 2 and result.shape[0] > 0 and result.shape[1] == 3
            and np.isfinite(result).all(), 'Expected finite nonempty N x 3 XYZ')
    return result


def physical_clean(clean, center, scale):
    center, scale = np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    require(center.shape == (1, 3) and scale.shape == (1, 1) and np.isfinite(center).all()
            and np.isfinite(scale).all() and (scale > 0).all(), 'Invalid corrupted normalization frame')
    normalized = (xyz(clean)-center)/scale*3.3885
    result = normalized[:, [2, 1, 0]].copy()
    result[:, 0] *= -1
    return result


def nearest_squared(left, right):
    left, right = xyz(left), xyz(right)
    if cKDTree is not None:
        return cKDTree(right).query(left, k=1)[0]**2
    result = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), 128):
        delta = left[start:start+128, None, :] - right[None, :, :]
        result[start:start+128] = np.square(delta).sum(-1).min(1)
    return result


def squared_chamfer(left, right):
    """Sum the two directional means of squared Euclidean nearest distances."""
    return float(nearest_squared(left, right).mean()+nearest_squared(right, left).mean())


def quadratic_loss(decoded, reference, metric):
    residual = np.asarray(decoded, dtype=np.float64)-np.asarray(reference, dtype=np.float64)
    return float(np.sum(residual*(metric @ residual))/(3*len(residual)))


def validate_metric(metric, first, reference, saved_loss, saved_filter=None):
    require(metric.shape == (len(reference), len(reference)) and np.isfinite(metric).all(), 'Invalid physical filter')
    symmetry_error = float(np.abs(metric-metric.T).max())
    require(symmetry_error <= 1e-7, 'Physical filter is not symmetric')
    loss = quadratic_loss(first, reference, metric)
    error = abs(loss-float(saved_loss))
    require(np.isfinite(loss) and np.isfinite(saved_loss) and error <= METRIC_ATOL,
            'First checkpoint loss mismatch: abs_error=%r' % error)
    filter_error = None
    if saved_filter is not None:
        require(saved_filter.shape == metric.shape and np.isfinite(saved_filter).all(), 'Invalid saved physical filter')
        filter_error = float(np.abs(metric-np.asarray(saved_filter, dtype=np.float64)).max())
        require(filter_error <= METRIC_ATOL, 'Reconstructed filter mismatch: max_abs_error=%r' % filter_error)
    return dict(passed=True, first_loss_cpu_float64=loss, first_loss_saved=float(saved_loss),
                first_loss_absolute_error=error, symmetry_max_absolute_error=symmetry_error,
                gpu_filter_available=saved_filter is not None, operator_comparison_available=saved_filter is not None,
                saved_filter_max_absolute_error=filter_error)


def analyze_direction(decoded, reference, metric, clean):
    """One fixed output-space direction; this does not propagate a sampler step."""
    decoded, reference, clean = xyz(decoded), xyz(reference), xyz(clean)
    require(decoded.shape == reference.shape, 'Decoded/reference vertex shapes differ')
    before = quadratic_loss(decoded, reference, metric)
    clean_before = squared_chamfer(decoded, clean)
    residual = decoded-reference
    gradient = 2*(metric @ residual)/(3*len(decoded))
    gradient_norm, epsilon = float(np.linalg.norm(gradient)), .001*float(np.linalg.norm(decoded))
    require(np.isfinite(gradient).all() and np.isfinite(gradient_norm+epsilon), 'Nonfinite physical direction')
    perm = np.arange(len(decoded)-1, -1, -1)
    permuted_loss = quadratic_loss(decoded[perm], reference, metric)
    joint_loss = quadratic_loss(decoded[perm], reference[perm], metric[np.ix_(perm, perm)])
    permuted_cd = squared_chamfer(decoded[perm], clean)
    require(np.isclose(joint_loss, before, rtol=1e-10, atol=1e-12), 'Joint relabeling is not quadratic-invariant')
    require(np.isclose(permuted_cd, clean_before, rtol=1e-10, atol=1e-12), 'Y-only permutation changed set Chamfer')
    row = dict(spectral_before=before, clean_cd_before=clean_before, gradient_norm=gradient_norm,
               epsilon=epsilon, slot_mse_coordinate=float(np.mean(residual**2)),
               slot_mean_squared_point_distance=float(np.mean(np.sum(residual**2, axis=1))),
               decoded_to_prepared_mean_nn_squared=float(nearest_squared(decoded, reference).mean()),
               prepared_to_decoded_mean_nn_squared=float(nearest_squared(reference, decoded).mean()),
               permutation=dict(kind='fixed vertex reversal', y_only_spectral_loss=permuted_loss,
                   y_only_spectral_delta=permuted_loss-before, joint_spectral_loss=joint_loss,
                   joint_absolute_error=abs(joint_loss-before), joint_invariant=True,
                   clean_cd_absolute_error=abs(permuted_cd-clean_before), clean_cd_invariant=True),
               resolved=False, unresolved_reason=None, spectral_after=None, spectral_delta=None,
               clean_cd_after=None, clean_cd_delta=None, predicted_quadratic_delta=None,
               quadratic_delta_absolute_error=None, finite_step_overshoot=None,
               spectral_delta_resolved=False, clean_cd_delta_resolved=False,
               spectral_decrease=False, clean_cd_increase=False, spectral_decrease_clean_cd_increase=False)
    if gradient_norm == 0 or epsilon == 0:
        row['unresolved_reason'] = 'zero_gradient' if gradient_norm == 0 else 'zero_epsilon'
        return row
    direction = -gradient/gradient_norm
    after_points = decoded+epsilon*direction
    after, clean_after = quadratic_loss(after_points, reference, metric), squared_chamfer(after_points, clean)
    curvature = float(np.sum(direction*(metric @ direction))/(3*len(decoded)))
    predicted = -epsilon*gradient_norm+epsilon**2*curvature
    delta, clean_delta = after-before, clean_after-clean_before
    quadratic_error = abs(delta-predicted)
    require(np.isfinite([after, clean_after, predicted, curvature]).all()
            and quadratic_error <= 1e-12+1e-9*max(abs(before), abs(after), abs(predicted)),
            'Exact finite-step quadratic prediction failed')
    spectral_resolution = 1e-12+1e-9*max(abs(before), abs(after))
    clean_resolution = 1e-12+1e-9*max(clean_before, clean_after)
    decreasing, increasing = delta < -spectral_resolution, clean_delta > clean_resolution
    row.update(resolved=True, spectral_after=after, spectral_delta=delta,
               clean_cd_after=clean_after, clean_cd_delta=clean_delta,
               predicted_quadratic_delta=predicted, quadratic_delta_absolute_error=quadratic_error,
               directional_linear_delta=-epsilon*gradient_norm, curvature_term=epsilon**2*curvature,
               finite_step_overshoot=predicted >= 0, spectral_delta_resolved=abs(delta) > spectral_resolution,
               clean_cd_delta_resolved=abs(clean_delta) > clean_resolution,
               spectral_resolution=spectral_resolution, clean_cd_resolution=clean_resolution,
               spectral_decrease=bool(decreasing), clean_cd_increase=bool(increasing),
               spectral_decrease_clean_cd_increase=bool(decreasing and increasing))
    return row


def load_oracle(path, receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    require(type(receipt.get('schema_version')) is int and receipt['schema_version'] == 1
            and receipt.get('status') == 'CPU_oracle_only', 'Invalid clean oracle receipt')
    require(all(receipt.get('full_clean_asset', {}).get(k) == CLEAN_IDENTITY[k] for k in ('bytes', 'sha256')),
            'Pinned full clean asset identity differs')
    subset_identity = identity(path)
    require(all(receipt.get('subset', {}).get(k) == subset_identity[k] for k in ('bytes', 'sha256')),
            'Clean subset SHA/size differs from receipt')
    require(receipt.get('indices') == IDS and receipt.get('selection_manifest_sha256') == SELECTION_SHA256,
            'Clean oracle selected-ID manifest differs')
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'index', 'clean_xyz'} and saved['index'].dtype == np.int64
                and np.array_equal(saved['index'], IDS), 'Invalid clean oracle index arrays')
        clean = saved['clean_xyz'].copy()
    require(clean.shape == (8, 1024, 3) and np.issubdtype(clean.dtype, np.floating)
            and np.isfinite(clean).all(), 'Invalid pinned clean XYZ subset')
    return clean, receipt


def summarize(path, oracle_path, receipt_path):
    import torch
    import graph_spectral
    from legacy_trajectory import tensor_hash
    inputs = {str(Path(p).resolve()): identity(p) for p in
              (path, oracle_path, receipt_path, __file__, Path(__file__).with_name('analyze_followup.py'), graph_spectral.__file__)}
    clean, receipt = load_oracle(oracle_path, receipt_path)
    bundle, config = open_bundle(path)
    rows, checks, mismatches = [], [], []
    try:
        require(config['dataset'] == 'modelnet40_c' and config['selected_original_ids'] == IDS
                and config['count'] == 8 and config['backend_policy'] == 'controlled'
                and config['severity'] == 5 and config['seed'] == 0
                and set(config['corruptions']) == {'gaussian', 'background', 'shear', 'lidar'}, 'P1 scope differs')
        selection = bundle.read('selection_manifest.json')
        require(hashlib.sha256(selection).hexdigest() == SELECTION_SHA256 == config['index_manifest_sha256']
                and json.loads(selection) == config['selection_manifest'], 'P1 selection identity differs')
        require(config['guidance_protocol'] == dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False)
                and config['normal_reverse_steps'] == 8 and config['background_reverse_steps'] == 35
                and config['physical_graph'] == dict(k=10, delta=.1, graph_gamma=.6, modes=100, profile='smooth', beta=2.)
                and config.get('freeze_decoder_residual') is False, 'P1 production protocol differs')
        setup_clean = config['setup_verification']['report']['assets']['data/modelnet40_c/data_original.npy']
        require(all(setup_clean[k] == CLEAN_IDENTITY[k] for k in ('bytes', 'sha256')), 'P1 setup clean identity differs')
        sources = [v for k, v in config['executed_source_files'].items() if k.endswith('/graph_spectral.py')]
        require(sources and all(v == identity(graph_spectral.__file__) for v in sources), 'Production graph source identity differs')
        for corruption in config['corruptions']:
            for batch in range(4):
                case = 'batches/%s/batch%04d/' % (corruption, batch)
                expected_ids = IDS[batch*2:batch*2+2]
                reference = arrays(bundle, case+'reference.npz')
                terminal = arrays(bundle, case+'terminal_geometry.npz')
                first = arrays(bundle, case+'decoded_first_checkpoint.npz')['decoded_xyz']
                predictions = arrays(bundle, case+'predictions.npz')
                records = bundle.json(case+'probe_records.json')
                diagnostic = bundle.json(case+'diagnostics.json')
                prepared = reference['prepared_x']
                require(prepared.dtype == np.float32 and prepared.shape == first.shape == terminal['terminal_decoded_xyz'].shape == (2, 2048, 3)
                        and np.isfinite(prepared).all() and np.isfinite(first).all()
                        and np.isfinite(terminal['terminal_decoded_xyz']).all(), 'Invalid physical XYZ artifact')
                require(np.array_equal(terminal['index'], expected_ids) and np.array_equal(predictions['index'], expected_ids), 'Batch original IDs differ')
                require(terminal['normalization_center'].shape == (2, 1, 3) and terminal['normalization_scale'].shape == (2, 1, 1), 'Invalid normalization frame shapes')
                require(reference['initial_h0_xyz'].shape == (2, 2048, 3)
                        and terminal['terminal_classifier_geometry'].shape == (2, 2048, 3)
                        and terminal['raw_corrupted_xyz'].ndim == 3 and terminal['raw_corrupted_xyz'].shape[0] == 2
                        and terminal['raw_corrupted_xyz'].shape[2] == 3
                        and all(np.isfinite(value).all() for value in terminal.values())
                        and len(records) == 3 and records[0]['loop_count'] == (9 if corruption == 'background' else 2),
                        'Invalid terminal/raw or first-checkpoint metadata')
                require(tensor_hash(torch.from_numpy(prepared)) == diagnostic['prepared_x_sha256'], 'Prepared X hash differs')
                saved_filters = arrays(bundle, case+'physical_filters.npz')['filters'] if case+'physical_filters.npz' in bundle.names else None
                target = graph_spectral.build_smooth_spectral_target(torch.from_numpy(prepared), graph_spectral.SpectralConfig(), profile='smooth', beta=2.)
                case_checks = []
                try:
                    for index, object_id in enumerate(expected_ids):
                        metric = target.filters[index].numpy().astype(np.float64)
                        check = validate_metric(metric, first[index], prepared[index], records[0]['spectral_loss_per_object'][index],
                                                None if saved_filters is None else saved_filters[index])
                        case_checks.append(dict(corruption=corruption, object_id=object_id, **check))
                except ValueError as error:
                    mismatches.append(dict(case=case, object_ids=expected_ids, error=str(error), comparison_aborted=True))
                    continue
                checks.extend(case_checks)
                for index, object_id in enumerate(expected_ids):
                    metric = target.filters[index].numpy().astype(np.float64)
                    clean_physical = physical_clean(clean[batch*2+index], terminal['normalization_center'][index], terminal['normalization_scale'][index])
                    for checkpoint, decoded, loop_count in (
                            ('first', first[index], records[0]['loop_count']),
                            ('terminal', terminal['terminal_decoded_xyz'][index], 35 if corruption == 'background' else 8)):
                        rows.append(dict(corruption=corruption, object_id=object_id, checkpoint=checkpoint,
                                         gpu_filter_available=saved_filters is not None,
                                         operator_comparison_available=saved_filters is not None,
                                         metric_direction_scope='CPU reconstructed metric; GPU scalar-loss gate only'
                                             if saved_filters is None else 'CPU metric compared with saved GPU filter; no GPU-direction certificate',
                                         loop_count=loop_count, **analyze_direction(decoded, prepared[index], metric, clean_physical)))
        aggregates = {}
        for corruption in config['corruptions']:
            aggregates[corruption] = {}
            for checkpoint in ('first', 'terminal'):
                group = [r for r in rows if r['corruption'] == corruption and r['checkpoint'] == checkpoint]
                aggregates[corruption][checkpoint] = dict(total=len(group), resolved=sum(r['resolved'] for r in group),
                    spectral_decrease=sum(r['spectral_decrease'] for r in group), clean_cd_increase=sum(r['clean_cd_increase'] for r in group),
                    counterexamples=sum(r['spectral_decrease_clean_cd_increase'] for r in group),
                    overshoot=sum(r['finite_step_overshoot'] is True for r in group))
        require(len(rows) == 64 or mismatches, 'Expected 32 conditions x 2 checkpoints')
        require(all(identity(p) == expected for p, expected in inputs.items()), 'Analysis inputs changed during CPU work')
        return dict(stage='P2', status='analysis_mismatch' if mismatches else 'complete', expected_rows=64, total_rows=len(rows),
                    inputs=inputs, oracle_receipt=receipt, metric_checks=checks, mismatches=mismatches,
                    metric_provenance=dict(validated_conditions=len(checks),
                        saved_gpu_filter_conditions=sum(c['gpu_filter_available'] for c in checks),
                        cpu_reconstructed_scalar_gated_conditions=sum(not c['gpu_filter_available'] for c in checks)),
                    production_graph=dict(config=vars(graph_spectral.SpectralConfig()), profile='smooth', beta=2., reference_dtype='float32'),
                    metric_absolute_tolerance=METRIC_ATOL, nearest_neighbor_backend='scipy.cKDTree' if cKDTree is not None else 'numpy_chunk128',
                    rows=rows, aggregates=aggregates,
                    limitation='Posthoc CPU-reconstructed output-space directions; only saved first-B2 filters permit GPU-operator comparison. Other conditions have a first-loss scalar gate only, no GPU-operator/direction certificate. No sampler/accuracy inference; clean never guidance or selection')
    finally:
        bundle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('p1_zip', type=Path)
    parser.add_argument('oracle_npz', type=Path)
    parser.add_argument('oracle_receipt', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = summarize(args.p1_zip, args.oracle_npz, args.oracle_receipt)
    rendered = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(rendered)
    else:
        print(rendered, end='')
    if report['status'] != 'complete':
        raise SystemExit(1)
