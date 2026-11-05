#!/usr/bin/env python3
"""Frozen H5 clean-free features and posthoc CPU set-oracle diagnostics."""
from __future__ import annotations

import argparse
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def helper(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


accepted = helper('h5_accepted_accuracy', 'knowledge/evidence/decoded_physical_accuracy_20261104/analyze_accuracy.py')
oracle = helper('h5_accepted_oracle', 'knowledge/evidence/decoded_physical_followup_20261104/analyze_oracle.py')
require = accepted.require
IDS = oracle.IDS
CORRUPTIONS = ('gaussian', 'background', 'shear', 'lidar')
EDGES = (0., .25, .5, 1., 2., 4., float('inf'))
SCORES = ('frequency_reliability', 'residual_reliability', 'combined_reliability')


def tensor_hash(value):
    # Preserve scalar rank shape(); ascontiguousarray would promote it to(1,).
    value = np.asarray(value)
    return accepted.hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes(order='C')).hexdigest()


def make_bands(q, diagnostic, config):
    q = np.asarray(q, dtype=np.float64)
    require(q.ndim == 1 and np.isfinite(q).all() and (q >= 0).all() and (np.diff(q) >= 0).all(), 'Invalid normalized eigenvalues')
    mean_degree = float(diagnostic['mean_active_degree'])
    require(np.isfinite(mean_degree) and (not len(q) or mean_degree > 0), 'Active spectrum has no positive finite mean degree')
    atol = float(diagnostic['effective_boundary_atol'])/mean_degree if len(q) else 0.
    rtol = float(config['eigenspace_rtol'])
    require(np.isfinite([atol, rtol]).all() and min(atol, rtol) >= 0, 'Invalid repeated-eigenspace tolerance')
    bands = [dict(band_index=i, nominal_lower=EDGES[i], nominal_upper=None if i == 5 else EDGES[i+1],
                  mode_indices=[], repeated_blocks=[], normalized_atol=atol, eigenspace_rtol=rtol) for i in range(6)]
    start = 0
    while start < len(q):
        end = start+1
        tolerance = atol+rtol*abs(q[start])
        while end < len(q) and q[end]-q[start] <= tolerance:
            end += 1
        index = min(5, max(0, int(np.searchsorted(EDGES, q[start], side='right')-1)))
        bands[index]['mode_indices'].extend(range(start, end))
        bands[index]['repeated_blocks'].append([start, end])
        start = end
    for band in bands:
        values = q[band['mode_indices']]
        band.update(empty=not len(values), mode_count=len(values),
                    actual_min=None if not len(values) else float(values.min()),
                    actual_max=None if not len(values) else float(values.max()),
                    mean_lambda=None if not len(values) else float(values.mean()))
    return bands


def project(value, indices, basis, band):
    result = np.zeros_like(value, dtype=np.float64)
    selected = basis[:, band['mode_indices']]
    result[indices] = selected @ (selected.T @ value[indices])
    return result


def clean_free_features(history, reference, indices, basis, bands):
    """No oracle/labels; checkpoint t reads only Y[t],Y[t-1] and fixed X/U."""
    history, reference, basis = (np.asarray(v, dtype=np.float64) for v in (history, reference, basis))
    rows = []
    projected_x = [project(reference, indices, basis, band) for band in bands]
    previous = [None]*len(bands)
    for t, current in enumerate(history):
        for band, px in zip(bands, projected_x):
            b = band['band_index']
            row = dict(checkpoint_index=t, previous_checkpoint_index=None if t == 0 else t-1,
                band_index=b, r=None, v=None, s=None, r_status='EMPTY_BAND_UNAVAILABLE',
                v_status='EMPTY_BAND_UNAVAILABLE', s_status='EMPTY_BAND_UNAVAILABLE',
                scores=dict(frequency_reliability=None, residual_reliability=None, combined_reliability=None))
            if not band['empty']:
                py = project(current, indices, basis, band)
                pr = project(current-reference, indices, basis, band)
                denom = float(np.sum(py**2)+np.sum(px**2))
                threshold = 1e-12*(1+float(np.sum(current**2)+np.sum(reference**2)))
                row.update(r_denominator=denom, r_stabilizer=threshold,
                           projected_residual_norm=float(np.linalg.norm(pr)),
                           projected_residual_resolution=1e-6*(1+float(np.linalg.norm(current))))
                row['scores']['frequency_reliability'] = -band['mean_lambda']
                row['r_status'] = 'TINY_DENOMINATOR_UNAVAILABLE' if denom <= threshold else 'AVAILABLE'
                if denom > threshold:
                    row['r'] = float(np.sum(pr**2)/(denom+threshold))
                    row['scores']['residual_reliability'] = -row['r']
                row['v_status'] = 'NO_PREVIOUS_CHECKPOINT_UNAVAILABLE'
                if t:
                    pp = previous[b]
                    denom_v = float(np.sum(py**2)+np.sum(pp**2))
                    threshold_v = 1e-12*(1+float(np.sum(current**2)+np.sum(history[t-1]**2)))
                    row.update(v_denominator=denom_v, v_stabilizer=threshold_v)
                    row['v_status'] = 'TINY_DENOMINATOR_UNAVAILABLE' if denom_v <= threshold_v else 'AVAILABLE'
                    if denom_v > threshold_v:
                        drift = project(current-history[t-1], indices, basis, band)
                        row['v'] = float(np.sum(drift**2)/(denom_v+threshold_v))
                row['s_status'] = 'AVAILABLE' if row['r'] is not None and row['v'] is not None else 'INPUT_FEATURE_UNAVAILABLE'
                if row['s_status'] == 'AVAILABLE':
                    row['s'] = row['r']/(1+row['v'])
                    row['scores']['combined_reliability'] = -row['s']
                previous[b] = py
            rows.append(row)
    return rows


def band_step(current, reference, indices, basis, band):
    """Determine finite-step eligibility using spectral information alone."""
    current, reference, basis = (np.asarray(v, dtype=np.float64) for v in (current, reference, basis))
    result = dict(oracle_eligible=False, spectral_status='EMPTY_BAND_UNAVAILABLE')
    if band['empty']:
        return result, None
    selected = basis[:, band['mode_indices']]
    coefficients = selected.T @ (current-reference)[indices]
    residual = project(current-reference, indices, basis, band)
    norm = float(np.linalg.norm(residual))
    resolution = 1e-6*(1+float(np.linalg.norm(current)))
    result.update(projected_residual_norm=norm, projected_residual_resolution=resolution)
    if norm <= resolution:
        result['spectral_status'] = 'TINY_PROJECTED_RESIDUAL_UNAVAILABLE'
        return result, None
    gradient = 2*residual/(3*len(current))
    gradient_norm = float(np.linalg.norm(gradient))
    epsilon = .001*float(np.linalg.norm(current))
    require(np.isfinite([gradient_norm, epsilon]).all(), 'Nonfinite band direction')
    if gradient_norm == 0 or epsilon == 0:
        result['spectral_status'] = 'ZERO_DIRECTION_UNAVAILABLE'
        return result, None
    direction = -gradient/gradient_norm
    after = current+epsilon*direction
    before_loss = float(np.sum(coefficients**2)/(3*len(current)))
    after_coefficients = selected.T @ (after-reference)[indices]
    after_loss = float(np.sum(after_coefficients**2)/(3*len(current)))
    curvature = float(np.sum((selected.T @ direction[indices])**2)/(3*len(current)))
    predicted = -epsilon*gradient_norm+epsilon**2*curvature
    observed = after_loss-before_loss
    error = abs(observed-predicted)
    require(np.isfinite([before_loss, after_loss, predicted]).all()
            and error <= 1e-12*(1+abs(before_loss)+abs(after_loss)+abs(predicted)), 'Band quadratic finite-step identity failed')
    px, py = project(reference, indices, basis, band), project(current, indices, basis, band)
    symmetry_error = abs(float(np.sum(reference*py)-np.sum(px*current)))
    require(symmetry_error <= 1e-10*(1+np.linalg.norm(reference)*np.linalg.norm(current)), 'Band projector symmetry failed')
    permutation = np.arange(len(current)-1, -1, -1)
    new_indices = len(current)-1-indices
    order = np.argsort(new_indices)
    joint_coeff = selected[order].T @ (current[permutation]-reference[permutation])[new_indices[order]]
    joint_loss = float(np.sum(joint_coeff**2)/(3*len(current)))
    require(abs(joint_loss-before_loss) <= 1e-12*(1+abs(before_loss)), 'Joint band relabeling failed')
    y_only = selected.T @ (current[permutation]-reference)[indices]
    result.update(gradient_norm=gradient_norm, epsilon=epsilon, spectral_before=before_loss,
        spectral_after=after_loss, spectral_delta=observed, predicted_quadratic_delta=predicted,
        quadratic_absolute_error=error, projector_symmetry_absolute_error=symmetry_error,
        joint_permutation_absolute_error=abs(joint_loss-before_loss),
        y_only_permutation_spectral_delta=float(np.sum(y_only**2)/(3*len(current)))-before_loss)
    if predicted >= 0 or observed >= 0:
        result['spectral_status'] = 'OVERSHOOT_UNAVAILABLE'
        return result, None
    result.update(oracle_eligible=True, spectral_status='AVAILABLE')
    return result, after


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = .5*(start+1+end)
        start = end
    return ranks


def association(rows, score):
    chosen = [r for r in rows if r.get('oracle_benefit') is not None and r['scores'][score] is not None
              and np.isfinite(r['scores'][score]) and np.isfinite(r['clean_cd_delta'])]
    result = dict(n=len(chosen), spearman=None, auroc=None, positive=0, negative=0)
    if not chosen:
        return result
    values = np.array([r['scores'][score] for r in chosen])
    gains = np.array([-r['clean_cd_delta'] for r in chosen])
    labels = np.array([r['oracle_benefit'] for r in chosen], dtype=bool)
    result.update(positive=int(labels.sum()), negative=int((~labels).sum()))
    ranks, gain_ranks = average_ranks(values), average_ranks(gains)
    if len(chosen) >= 2 and np.std(ranks) > 0 and np.std(gain_ranks) > 0:
        result['spearman'] = float(np.corrcoef(ranks, gain_ranks)[0, 1])
    if labels.any() and (~labels).any():
        positive, negative = int(labels.sum()), int((~labels).sum())
        result['auroc'] = float((ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative))
    return result


def score_comparison(rows):
    common = [r for r in rows if r.get('oracle_benefit') is not None
              and np.isfinite(r['clean_cd_delta'])
              and all(r['scores'][score] is not None and np.isfinite(r['scores'][score]) for score in SCORES)]
    metrics = {score: association(common, score) for score in SCORES}
    require(all(result['n'] == len(common) for result in metrics.values()), 'Comparative score mask changed')
    keys = [{key: row.get(key) for key in ('repeat', 'case', 'object_id', 'checkpoint_index', 'band_index')} for row in common]
    return dict(common_eligible_rows=len(common), common_ordered_row_keys_sha256=accepted.fingerprint(keys), scores=metrics,
                mask='Same resolved oracle rows with all three frozen features available; first checkpoint excluded')


def load_capture(bundle, case, basic):
    prefix = 'batches/%s/batch%04d/'%(basic['corruption'], basic['batch'])
    diagnostic = bundle.json(prefix+'diagnostics.json')
    raw_meta = bundle.read(prefix+'h5_capture.json')
    meta_identity = dict(bytes=len(raw_meta), sha256=accepted.hashlib.sha256(raw_meta).hexdigest())
    require({k: diagnostic['h5_capture'][k] for k in meta_identity} == meta_identity, 'H5 metadata identity mismatch')
    meta = json.loads(raw_meta)
    require(meta['schema_version'] == 1 and meta['graph_build_calls'] == 1 and meta['extra_decoder_forward_calls'] == 0
            and meta['rng_unchanged'] is True and meta['rng_before_sha256'] == meta['rng_after_sha256']
            == basic['identity']['output_rng'], 'Invalid H5 capture/RNG protocol')
    require(meta['graph_config'] == dict(k=10, delta=.1, graph_gamma=.6, modes=100,
                eigenspace_rtol=1e-5, eigenspace_atol=1e-7), 'H5 graph configuration differs')
    data = {}
    for filename in ('h5_geometry.npz', 'h5_spectra.npz'):
        raw = bundle.read(prefix+filename)
        require(meta['files'][filename] == dict(bytes=len(raw), sha256=accepted.hashlib.sha256(raw).hexdigest()), 'H5 NPZ identity mismatch')
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            data[filename] = {k: saved[k] for k in saved.files}
    geometry, spectra = data['h5_geometry.npz'], data['h5_spectra.npz']
    counts = [9, 18, 27, 35] if basic['corruption'] == 'background' else [2, 4, 6, 8]
    names = ['clean_%02d'%step for step in counts[:3]]+['terminal']
    require(np.array_equal(geometry['index'], basic['index']) and np.array_equal(spectra['index'], basic['index'])
            and geometry['candidate_loop_counts'].tolist() == meta['candidate_loop_counts'] == counts
            and geometry['candidate_names'].tolist() == meta['candidate_names'] == names, 'H5 candidate/index ordering differs')
    require(geometry['decoded_xyz'].shape == (2, 4, 2048, 3) and geometry['decoded_xyz'].dtype == np.float32
            and geometry['prepared_x'].shape == (2, 2048, 3) and geometry['prepared_x'].dtype == np.float32
            and np.isfinite(geometry['decoded_xyz']).all() and np.isfinite(geometry['prepared_x']).all(), 'Invalid H5 physical XYZ')
    require(geometry['normalization_center'].shape == (2, 1, 3) and geometry['normalization_scale'].shape == (2, 1, 1)
            and np.isfinite(geometry['normalization_center']).all() and np.isfinite(geometry['normalization_scale']).all()
            and (geometry['normalization_scale'] > 0).all(), 'Invalid H5 physical frame')
    require(tensor_hash(geometry['prepared_x']) == meta['reference_xyz_sha256'] == basic['identity']['prepared']
            and [tensor_hash(geometry['decoded_xyz'][:, t]) for t in range(4)] == meta['decoded_tensor_sha256'], 'H5 geometry tensor hash mismatch')
    expected_hashes = [p['decoded_geometry_sha256'] for p in diagnostic['trajectory']['probes']]
    require(meta['decoded_tensor_sha256'][:3] == expected_hashes
            and np.array_equal(geometry['decoded_xyz'][:, 0], basic['first']), 'H5 clean snapshots differ from PM probes')
    terminal = accepted.arrays(bundle, prefix+'terminal_geometry.npz')
    require(np.array_equal(geometry['decoded_xyz'][:, -1], terminal['terminal_decoded_xyz'])
            and np.array_equal(geometry['normalization_center'], terminal['normalization_center'])
            and np.array_equal(geometry['normalization_scale'], terminal['normalization_scale']), 'H5 terminal/frame differs')
    require(len(meta['spectra_tensor_sha256']) == len(meta['graph_diagnostics']) == 2, 'H5 spectrum metadata count differs')
    objects = []
    for i in range(2):
        values = {k: spectra['%s_%d'%(k, i)] for k in ('active_indices', 'basis', 'q', 'rank')}
        require({k: tensor_hash(v) for k, v in values.items()} == meta['spectra_tensor_sha256'][i], 'H5 eigensystem tensor hash mismatch')
        indices, basis, q, rank = (values[k] for k in ('active_indices', 'basis', 'q', 'rank'))
        require(indices.dtype == np.int64 and indices.ndim == 1 and np.array_equal(indices, np.unique(indices))
                and ((indices >= 0) & (indices < 2048)).all() and basis.shape == (len(indices), len(indices))
                and basis.dtype == q.dtype == np.float32 and np.isfinite(basis).all() and np.isfinite(q).all()
                and q.shape == (len(indices),) and rank.shape == () and rank.dtype == np.int64
                and 0 <= int(rank) <= len(indices), 'Invalid archived eigensystem shape/support')
        graph = meta['graph_diagnostics'][i]
        require(graph['active_vertices'] == len(indices) and graph['actual_rank'] == int(rank), 'H5 rank/support diagnostics differ')
        bands = make_bands(q, graph, meta['graph_config'])
        objects.append(dict(indices=indices, basis=basis, q=q, rank=int(rank), bands=bands,
                            tensor_sha256=meta['spectra_tensor_sha256'][i]))
    return dict(geometry=geometry, objects=objects, metadata=meta)


def basis_comparison(reference, other, x, first_y):
    same_support = np.array_equal(reference['indices'], other['indices'])
    result = dict(active_indices_equal=same_support, reference_active_count=len(reference['indices']),
        other_active_count=len(other['indices']), rank_equal=reference['rank'] == other['rank'],
        reference_tensor_sha256=reference['tensor_sha256'], other_tensor_sha256=other['tensor_sha256'],
        normalized_eigenvalue_max_absolute_difference=None, band_projector_witnesses=[])
    if same_support and reference['q'].shape == other['q'].shape:
        result['normalized_eigenvalue_max_absolute_difference'] = float(np.max(np.abs(
            reference['q'].astype(np.float64)-other['q']))) if len(reference['q']) else 0.
    reference_basis, other_basis = reference['basis'].astype(np.float64), other['basis'].astype(np.float64)
    for a, b in zip(reference['bands'], other['bands']):
        witness = dict(band_index=a['band_index'], reference_mode_count=a['mode_count'], other_mode_count=b['mode_count'])
        for name, value in (('prepared_x', x), ('first_repeat_first_y', first_y)):
            pa = project(value, reference['indices'], reference_basis, a)
            pb = project(value, other['indices'], other_basis, b)
            witness[name+'_projector_relative_difference'] = float(np.linalg.norm(pa-pb)/(1+np.linalg.norm(pa)))
        result['band_projector_witnesses'].append(witness)
    return result


def feature_variation(features):
    groups = {}
    for row in features:
        groups.setdefault((row['object_id'], row['corruption'], row['band_index']), []).append(row)
    result = []
    for (object_id, corruption, band), group in groups.items():
        for feature in ('r', 'v', 's'):
            ranges, temporal = [], []
            for t in range(4):
                values = [r[feature] for r in group if r['checkpoint_index'] == t and r[feature] is not None]
                if len(values) == 3:
                    ranges.append(dict(checkpoint_index=t, repeat_range=max(values)-min(values)))
            for repeat in range(3):
                ordered = sorted((r for r in group if r['repeat'] == repeat), key=lambda r: r['checkpoint_index'])
                for a, b in zip(ordered, ordered[1:]):
                    if a[feature] is not None and b[feature] is not None:
                        temporal.append(abs(b[feature]-a[feature]))
            native = float(np.median([r['repeat_range'] for r in ranges])) if ranges else None
            drift = float(np.median(temporal)) if temporal else None
            resolved = native is not None and drift is not None and drift > 1e-12
            result.append(dict(object_id=object_id, corruption=corruption, band_index=band, feature=feature,
                native_ranges_by_checkpoint=ranges, adjacent_temporal_absolute_changes=temporal,
                native_range_median=native, temporal_change_median=drift,
                native_over_temporal_median=None if not resolved else native/drift,
                native_dominates_temporal=None if native is None or drift is None else native > drift))
    return result


def write_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def analyze(bundles, clean_subset, clean_receipt, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    opened = []
    inputs = {}
    try:
        inputs = {str(Path(p).resolve()): accepted.identity(p) for p in
            [*bundles, clean_subset, clean_receipt, __file__, accepted.__file__, oracle.__file__]}
        write_json(output/'inputs.json', inputs)
        require(len(bundles) == 3 and len({Path(p).resolve() for p in bundles}) == 3, 'Require three distinct fresh-process capture ZIPs')
        runs = [accepted.load_run(path) for path in bundles]
        for path, run in zip(bundles, runs):
            require(run['arm'] == 'control' and run['ids'] == IDS and run['corruptions'] == list(CORRUPTIONS), 'Wrong H5 capture scope/order')
            bundle = accepted.open_bundle(path)
            opened.append(bundle)
            require(bundle.json('config.json').get('capture_h5') is True, 'H5 opt-in capture flag missing')
            accepted.check_pairing(runs[0], run)
        run_ids = [bundle.json('config.json')['run_id'] for bundle in opened]
        require(run_ids == sorted(run_ids) and len(set(run_ids)) == 3, 'Captures must be ordered first-to-third fresh process')
        gates = [dict(repeats=[a, b], **accepted.repeat_gate(runs[a], runs[b])) for a, b in itertools.combinations(range(3), 2)]
        write_json(output/'native_repeat_gates.json', gates)
        captures, features, definitions, comparisons = {}, [], [], []
        for case, basic in runs[0]['cases'].items():
            case_repeats = [load_capture(bundle, case, run['cases'][case]) for bundle, run in zip(opened, runs)]
            common = case_repeats[0]
            for other in case_repeats[1:]:
                require(np.array_equal(other['geometry']['prepared_x'], common['geometry']['prepared_x'])
                        and np.array_equal(other['geometry']['normalization_center'], common['geometry']['normalization_center'])
                        and np.array_equal(other['geometry']['normalization_scale'], common['geometry']['normalization_scale']), 'Capture reference/frame changed across repeats')
            for i, object_id in enumerate(basic['index']):
                spectrum = common['objects'][i]
                definitions.append(dict(case=case, object_id=int(object_id), basis_repeat=0,
                    basis_archive=inputs[str(Path(bundles[0]).resolve())], tensor_sha256=spectrum['tensor_sha256'], bands=spectrum['bands']))
                for repeat, capture in enumerate(case_repeats):
                    geometry = capture['geometry']
                    rows = clean_free_features(geometry['decoded_xyz'][i], geometry['prepared_x'][i],
                                               spectrum['indices'], spectrum['basis'], spectrum['bands'])
                    for row in rows:
                        row.update(repeat=repeat, object_id=int(object_id), corruption=basic['corruption'], case=case,
                            loop_count=int(geometry['candidate_loop_counts'][row['checkpoint_index']]), basis_repeat=0)
                    features.extend(rows)
                    if repeat:
                        comparisons.append(dict(case=case, object_id=int(object_id), other_repeat=repeat,
                            **basis_comparison(spectrum, capture['objects'][i], common['geometry']['prepared_x'][i].astype(np.float64),
                                               common['geometry']['decoded_xyz'][i, 0].astype(np.float64))))
            # Keep the first basis; other eigensystems remain recoverable in immutable ZIPs.
            captures[case] = dict(common=common, geometries=[c['geometry'] for c in case_repeats], basic=basic)
        require(len(features) == 3*8*4*4*6, 'H5 feature coverage differs')
        write_json(output/'bands.json', definitions)
        write_json(output/'other_basis_comparisons.json', comparisons)
        write_json(output/'features.json', features)
        feature_identity = accepted.identity(output/'features.json')
        write_json(output/'feature_variation.json', feature_variation(features))
        # Clean is read only after every feature row has been frozen on disk.
        clean, clean_proof = oracle.load_oracle(clean_subset, clean_receipt)
        rows = []
        lookup = {(r['case'], r['object_id'], r['repeat'], r['checkpoint_index'], r['band_index']): r for r in features}
        with (output/'rows.jsonl').open('x') as stream:
            for case, data in captures.items():
                for i, object_id in enumerate(data['basic']['index']):
                    spectrum = data['common']['objects'][i]
                    basis = spectrum['basis'].astype(np.float64)
                    for repeat, geometry in enumerate(data['geometries']):
                        clean_physical = None
                        x = geometry['prepared_x'][i].astype(np.float64)
                        for t, current in enumerate(geometry['decoded_xyz'][i].astype(np.float64)):
                            baseline_cd, permutation_error = None, None
                            for band in spectrum['bands']:
                                row = dict(lookup[(case, int(object_id), repeat, t, band['band_index'])])
                                step, after = band_step(current, x, spectrum['indices'], basis, band)
                                row.update(step, oracle_benefit=None, clean_cd_before=None, clean_cd_after=None,
                                           clean_cd_delta=None, oracle_status='UNAVAILABLE')
                                if step['oracle_eligible']:
                                    if clean_physical is None:
                                        clean_physical = oracle.physical_clean(clean[IDS.index(int(object_id))],
                                            geometry['normalization_center'][i], geometry['normalization_scale'][i])
                                    if baseline_cd is None:
                                        baseline_cd = oracle.squared_chamfer(current, clean_physical)
                                        permutation_error = abs(oracle.squared_chamfer(current[::-1], clean_physical)-baseline_cd)
                                        require(permutation_error <= 1e-12*(1+baseline_cd), 'Clean-set distance changed under Y permutation')
                                    after_cd = oracle.squared_chamfer(after, clean_physical)
                                    require(np.isfinite([baseline_cd, after_cd]).all(), 'Nonfinite clean-set oracle')
                                    delta, tolerance = after_cd-baseline_cd, 1e-12*(1+baseline_cd)
                                    benefit = True if delta < -tolerance else False if delta > tolerance else None
                                    row.update(clean_cd_before=baseline_cd, clean_cd_after=after_cd, clean_cd_delta=delta,
                                        clean_cd_resolution=tolerance, clean_cd_reversal_absolute_error=permutation_error,
                                        oracle_benefit=benefit, oracle_status='BENEFIT' if benefit is True else 'HARM' if benefit is False else 'UNRESOLVED_TIE')
                                rows.append(row)
                                stream.write(json.dumps(row, allow_nan=False)+'\n')
                                stream.flush()
        require(len(rows) == len(features) and accepted.identity(output/'features.json') == feature_identity, 'Frozen feature/row coverage changed')
        groups = {}
        for corruption in CORRUPTIONS:
            groups[corruption] = {}
            for band in range(6):
                group = [r for r in rows if r['corruption'] == corruption and r['band_index'] == band]
                groups[corruption][str(band)] = dict(rows=len(group), oracle_eligible=sum(r['oracle_eligible'] for r in group),
                    benefit=sum(r['oracle_benefit'] is True for r in group), harm=sum(r['oracle_benefit'] is False for r in group),
                    ties=sum(r['oracle_status'] == 'UNRESOLVED_TIE' for r in group),
                    feature_distributions={key: distribution([r[key] for r in group if r[key] is not None]) for key in ('r', 'v', 's')},
                    comparative_scores=score_comparison(group),
                    per_feature_full_coverage_scores={score: association(group, score) for score in SCORES})
        require(all(accepted.identity(p) == expected for p, expected in inputs.items()), 'Input artifacts changed during H5 analysis')
        summary = dict(status='complete' if all(g['passed'] for g in gates) else 'complete_with_repeat_floor_failure',
            input_identities=inputs, feature_identity=feature_identity, rows=len(rows), ids=IDS, corruptions=list(CORRUPTIONS),
            repeats=3, seed=0, basis_policy='First repeat GPU basis for each exact X across all repeats/checkpoints; no unmatched coefficient comparison',
            ordered_run_ids=run_ids,
            feature_policy='r/(1+v) unreliability; reliability scores -mean normalized lambda,-r,-s; no fitted scales or applied q',
            eligibility_policy='Empty/tiny residual/tiny feature denominator unavailable; frozen epsilon; spectral overshoot excluded before clean lookup; clean-CD ties unresolved',
            q_semantics='Archived normalized/clamped Laplacian eigenvalues, never H5 reliability weights',
            spectral_status_counts={s: sum(r['spectral_status'] == s for r in rows) for s in sorted({r['spectral_status'] for r in rows})},
            native_repeat_gates_all_passed=all(g['passed'] for g in gates), clean_receipt=clean_proof, by_corruption_band=groups,
            pooled_descriptive_comparison=score_comparison(rows),
            pooled_per_feature_full_coverage={score: association(rows, score) for score in SCORES},
            limitation='Eight historically exposed original-object clusters. Repeats/checkpoints/bands/corruptions are dependent; no rowwise significance or heldout/accuracy claim. Cluster intervals omitted; no feature selection/tuning or guidance applied.')
        write_json(output/'summary.json', summary)
        return summary
    except Exception:
        write_json(output/'failure.json', dict(status='failed', traceback=traceback.format_exc(), input_identities=inputs))
        raise
    finally:
        for bundle in opened:
            bundle.close()


def distribution(values):
    if not values:
        return dict(available=0, minimum=None, median=None, maximum=None)
    return dict(available=len(values), minimum=float(np.min(values)), median=float(np.median(values)), maximum=float(np.max(values)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles', nargs=3, required=True, type=Path)
    parser.add_argument('--clean-subset', required=True, type=Path)
    parser.add_argument('--clean-receipt', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.bundles, args.clean_subset, args.clean_receipt, args.output_dir), indent=2, allow_nan=False))
