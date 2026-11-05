"""Frozen H4 equivalence/cost probe; no model initialization or applied guidance."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import graph_spectral
import sparse_heat
from graph_spectral import SpectralConfig, _validate_xyz, build_probe_spectral_target

ORDERS = (8, 16, 32)
GATES = dict(relative_loss_tolerance=1e-6, gradient_relative_l2_tolerance=1e-6,
             loss_denominator_floor=1e-12, gradient_l2_denominator_floor=1e-12)


def apply_controlled_backend():
    """Configure the frozen native numeric recipe before any CUDA initialization."""
    if torch.cuda.is_initialized():
        raise RuntimeError('Controlled backend must be configured before CUDA initialization')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    return dict(configured_before_cuda_initialization=True,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def exact_loss_function(probe):
    """Existing probe U/q quadratic, directly on E to avoid Y-X cancellation."""
    weights = [torch.exp(-2*q.to(torch.float64)).to(probe.reference_xyz.dtype)
               for _, _, q, _ in probe.spectra]

    def per_object(residual):
        _validate_xyz(residual, 'residual')
        if (residual.shape != probe.reference_xyz.shape or residual.dtype != probe.reference_xyz.dtype
                or residual.device != probe.reference_xyz.device):
            raise ValueError('Exact residual must match reference shape/device/dtype')
        losses = []
        for e, (indices, u, _, _), w in zip(residual, probe.spectra, weights):
            losses.append((w[:, None]*(u.T @ e[indices]).square()).sum()/(3*residual.shape[1]))
        result = torch.stack(losses)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('Exact heat loss is nonfinite')
        return result
    return per_object


def scalar_polynomial(q, upper, coefficients):
    if q.numel() == 0:
        return q.to(torch.float64)
    z = 2*q.to(torch.float64)/upper-1
    previous, current = torch.ones_like(z), z
    result = coefficients[0]*previous + coefficients[1]*current
    for c in coefficients[2:]:
        following = 2*z*current-previous
        result = result+c*following
        previous, current = current, following
    return result


def check_equivalence(x, residuals, config, output):
    """All checkpoint/object native-dtype gates; retain U, raw L, q and residuals."""
    output = Path(output)
    captured = []
    original_eigh = torch.linalg.eigh

    def capture_eigh(matrix, *args, **kwargs):
        result = original_eigh(matrix, *args, **kwargs)
        captured.append((matrix.detach().clone(), result[0].detach().clone()))
        return result

    # Diagnostic capture only; timed fresh builds below use the original API.
    with patch('torch.linalg.eigh', side_effect=capture_eigh):
        exact = build_probe_spectral_target(x, config)
    sparse = sparse_heat.build_sparse_heat_target(x, config)
    arrays = dict(prepared_x=x.detach().cpu().numpy(), residuals=residuals.detach().cpu().numpy())
    diagnostics, cursor, same_graph = [], 0, True
    for i, (graph, (indices, u, q, _), row) in enumerate(zip(sparse.graphs, exact.spectra, exact.diagnostics)):
        raw, eigenvalues = (captured[cursor] if indices.numel() else
                            (x.new_empty((0, 0)), x.new_empty(0)))
        cursor += int(indices.numel() > 0)
        mean = row['mean_active_degree']
        normalized = graph.laplacian.to_dense()
        support_equal = torch.equal(indices, graph.active_indices)
        graph_equal = support_equal and mean == graph.mean_active_degree and torch.equal(
            normalized, raw/mean if mean else raw)
        same_graph = same_graph and graph_equal
        ud = u.to(torch.float64)
        orth = ud.T @ ud-torch.eye(len(indices), device=x.device, dtype=torch.float64)
        eig_residual = normalized.to(torch.float64) @ ud-ud*q.to(torch.float64)
        rawq = eigenvalues/mean if mean else eigenvalues
        diagnostics.append(dict(object_position=i, same_graph=graph_equal, support_equal=support_equal,
            active_count=len(indices), isolate_count=x.shape[1]-len(indices), laplacian_nnz=graph.laplacian._nnz(),
            mean_active_degree=mean, upper_bound=graph.upper_bound,
            raw_eigenvalue_min=float(eigenvalues.min()) if len(indices) else None,
            raw_negative_eigenvalue_count=int((eigenvalues < 0).sum()),
            q_clamp_max_abs=float((q-rawq).abs().max()) if len(indices) else 0.,
            eigen_orthogonality_max_abs=float(orth.abs().max()) if len(indices) else 0.,
            eigen_orthogonality_frobenius=float(orth.norm()),
            stored_l_vs_u_clamped_q_frobenius=float(eig_residual.norm()),
            stored_l_row_sum_max_abs=float(normalized.sum(1).abs().max()) if len(indices) else 0.))
        for name, value in dict(active_indices=indices, basis=u, q=q, raw_eigenvalues=eigenvalues,
                                raw_laplacian=raw, normalized_laplacian=normalized,
                                sparse_indices=graph.laplacian.indices(), sparse_values=graph.laplacian.values(),
                                mean_degree=x.new_tensor(mean)).items():
            arrays[name+'_'+str(i)] = value.detach().cpu().numpy()
    report = dict(same_graph_passed=bool(same_graph), graph_diagnostics=diagnostics,
                  numeric_gates=GATES, orders=[], equivalence_passed=False, first_acceptable_order=None,
                  bound_semantics='Ideal orthonormal U/q scalar-polynomial bound; excludes native sparse recurrence, U orthogonality, stored-L/eigenvalue-clamp roundoff. These discrepancies remain in the actual native gate.')
    with (output/'native_graph_and_residuals.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report['native_arrays'] = identity(output/'native_graph_and_residuals.npz')
    write_json(output/'equivalence.json', report)
    exact_fn = exact_loss_function(exact)
    for order in ORDERS:
        prepared = sparse.prepare(order=order)
        object_rows, batch_rows = [], []
        for checkpoint in range(residuals.shape[1]):
            e = residuals[:, checkpoint].detach().clone().requires_grad_()
            exact_values = exact_fn(e)
            exact_gradient, = torch.autograd.grad(exact_values.sum(), e)
            # A single-object target preserves the full-batch residual path and 3N reduction.
            losses = []
            for i, graph in enumerate(sparse.graphs):
                single = sparse_heat.SparseHeatTarget(e[i:i+1].shape, e.dtype, e.device, (graph,))
                losses.append(sparse_heat.SparseHeatFilter(single, (prepared.coefficients[i],), order, 2.).loss(e[i:i+1]))
            approximate_values = torch.stack(losses)
            approximate_gradient, = torch.autograd.grad(approximate_values.sum(), e)
            with (output/('native_k%d_checkpoint%d.npz' % (order, checkpoint))).open('xb') as stream:
                np.savez_compressed(stream, exact_gradient=exact_gradient.cpu().numpy(),
                    approximate_gradient=approximate_gradient.cpu().numpy(),
                    exact_loss=exact_values.detach().cpu().numpy(), approximate_loss=approximate_values.detach().cpu().numpy())
            batch_relative = abs(float((approximate_values.sum()-exact_values.sum()).detach()))/max(
                abs(float(exact_values.sum().detach())), GATES['loss_denominator_floor'])
            batch_rows.append(dict(checkpoint_position=checkpoint, relative_loss_error=batch_relative,
                                   passed=batch_relative <= GATES['relative_loss_tolerance']))
            for i, (graph, (_, u, q, _)) in enumerate(zip(sparse.graphs, exact.spectra)):
                absolute = float((approximate_values[i]-exact_values[i]).abs().detach())
                relative = absolute/max(abs(float(exact_values[i].detach())), GATES['loss_denominator_floor'])
                gradient_error = float((approximate_gradient[i]-exact_gradient[i]).norm())
                gradient_relative = gradient_error/max(float(exact_gradient[i].norm()), GATES['gradient_l2_denominator_floor'])
                polynomial = scalar_polynomial(q, graph.upper_bound, prepared.coefficients[i])
                epsilon = float((polynomial-torch.exp(-q.to(torch.float64))).abs().max()) if q.numel() else 0.
                metric_bound = epsilon*(2+epsilon)
                active_e = e[i, graph.active_indices].detach().to(torch.float64)
                mask = torch.ones(e.shape[1], dtype=torch.bool, device=e.device)
                mask[graph.active_indices] = False
                isolate_gradient = float(approximate_gradient[i, mask].abs().max()) if bool(mask.any()) else 0.
                object_rows.append(dict(checkpoint_position=checkpoint, object_position=i,
                    exact_loss=float(exact_values[i].detach()), approximate_loss=float(approximate_values[i].detach()),
                    absolute_loss_error=absolute, relative_loss_error=relative,
                    gradient_absolute_l2_error=gradient_error, gradient_relative_l2_error=gradient_relative,
                    exact_gradient_l2=float(exact_gradient[i].norm()), isolate_gradient_max_abs=isolate_gradient,
                    scalar_polynomial_error_on_actual_q=epsilon,
                    ideal_loss_error_bound=metric_bound*float(active_e.square().sum())/(3*e.shape[1]),
                    ideal_gradient_l2_error_bound=2*metric_bound*float(active_e.norm())/(3*e.shape[1]),
                    passed=bool(relative <= GATES['relative_loss_tolerance'] and
                        gradient_relative <= GATES['gradient_relative_l2_tolerance'] and isolate_gradient == 0.)))
        row = dict(order=order, objects=object_rows, batch_loss_gates=batch_rows,
                   passed=bool(same_graph and all(v['passed'] for v in object_rows+batch_rows)))
        report['orders'].append(row)
        if row['passed'] and report['first_acceptable_order'] is None:
            report['first_acceptable_order'] = order
        report['equivalence_passed'] = report['first_acceptable_order'] is not None
        write_json(output/'equivalence.json', report)
    write_json(output/'equivalence.json', report)
    return report


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _start(device):
    _sync(device)
    if device.type != 'cuda':
        return None
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def _memory(device, baseline):
    if baseline is None:
        return dict(peak_allocated_bytes=None, peak_increase_bytes=None, starting_allocated_bytes=None)
    maximum = torch.cuda.max_memory_allocated(device)
    return dict(peak_allocated_bytes=maximum, peak_increase_bytes=maximum-baseline, starting_allocated_bytes=baseline)


def _validate_sparse(target):
    for graph in target.graphs:
        if (graph.laplacian.layout != torch.sparse_coo or not graph.laplacian.is_coalesced()
                or not bool(torch.isfinite(graph.laplacian.values()).all())):
            raise ValueError('Invalid sparse COO Laplacian')


def benchmark(x, residual, config, original_ids, progress=None):
    """Fresh graph scopes and cached-operator residual VJPs; no model cost."""
    output = []
    for batch_size in (1, 32):
        positions = torch.arange(batch_size, device=x.device) % len(x)
        xb, eb = x[positions], residual[positions]
        result = dict(batch_size=batch_size, replicated_input_positions=positions.cpu().tolist(),
                      replicated_original_ids=[int(original_ids[int(i)]) for i in positions.cpu()], filtering=[])
        for name, builder in (('exact', build_probe_spectral_target), ('sparse', sparse_heat.build_sparse_heat_target)):
            samples = []
            for iteration in range(5):  # two warmups, three measured fresh builds
                baseline = _start(x.device)
                started = time.perf_counter()
                target = builder(xb, config)
                _sync(x.device)
                seconds = time.perf_counter()-started
                validation_seconds, eigh_seconds = 0., None
                if name == 'sparse':
                    started = time.perf_counter()
                    _validate_sparse(target)
                    _sync(x.device)
                    validation_seconds = time.perf_counter()-started
                else:
                    eigh_seconds = sum(d['eigendecomposition_seconds'] for d in target.diagnostics)
                if iteration >= 2:
                    samples.append(dict(seconds=seconds, eigh_seconds=eigh_seconds,
                        non_eigh_setup_seconds=seconds-eigh_seconds if eigh_seconds is not None else None,
                        sparse_validation_seconds=validation_seconds, **_memory(x.device, baseline)))
                del target
            result[name+'_setup'] = dict(warmup=2, measured=3, samples=samples,
                                        median_seconds=statistics.median(v['seconds'] for v in samples))
            if progress is not None:
                progress(batch_size, name+'_setup', result[name+'_setup'])
        exact = build_probe_spectral_target(xb, config)
        sparse = sparse_heat.build_sparse_heat_target(xb, config)
        for arm in ('exact', 8, 16, 32):
            started = time.perf_counter()
            if arm == 'exact':
                exact_fn = exact_loss_function(exact)
                loss_fn = lambda e: exact_fn(e).sum()
            else:
                prepared = sparse.prepare(order=arm)
                loss_fn = prepared.loss
            _sync(x.device)
            coefficient_seconds = time.perf_counter()-started
            samples = []
            for iteration in range(13):  # three warmups, ten measured loss/VJPs
                e = eb.detach().clone().requires_grad_()
                baseline = _start(x.device)
                started = time.perf_counter()
                loss = loss_fn(e)
                _sync(x.device)
                loss_seconds = time.perf_counter()-started
                started = time.perf_counter()
                gradient, = torch.autograd.grad(loss, e)
                _sync(x.device)
                backward_seconds = time.perf_counter()-started
                if iteration >= 3:
                    samples.append(dict(loss_seconds=loss_seconds, backward_seconds=backward_seconds,
                        combined_seconds=loss_seconds+backward_seconds, **_memory(x.device, baseline)))
                del loss, gradient, e
            result['filtering'].append(dict(arm=arm, coefficient_seconds=coefficient_seconds,
                warmup=3, measured=10, samples=samples,
                median_combined_seconds=statistics.median(v['combined_seconds'] for v in samples)))
            if progress is not None:
                progress(batch_size, 'filter_'+str(arm), result['filtering'][-1])
        output.append(result)
        if progress is not None:
            progress(batch_size, 'batch', result)
        del exact, sparse, loss_fn, exact_fn, prepared, xb, eb
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('geometry', 'geometry-sha256', 'asset-manifest', 'asset-manifest-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema_version=1, beta=2., orders=list(ORDERS), numeric_gates=GATES,
        cuda_backend_recipe=dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                                 cudnn_benchmark=False, cudnn_deterministic=True, deterministic_algorithms=False),
        graph_config=asdict(SpectralConfig()), equivalence_checkpoints='all captured H5 checkpoints',
        timing_checkpoint_position=0, setup_warmup=2, setup_measured=3, filtering_warmup=3, filtering_measured=10,
        timing_batch_sizes=[1, 32], first_acceptable_order='first K passing every checkpoint/object gate; no label selection',
        scope='Heat objective and residual VJP only. No decoder, sampler, encoder, PM, sparse-kNN or end-to-end speed claim.',
        bounds='Ideal scalar polynomial-on-q bounds do not absorb native U/clamp/stored-L/recurrence roundoff.',
        prior_evidence='Prior CPU filter reconstruction 1e-6 failure remains unchanged.',
        inputs=vars(args))
    write_json(output/'protocol.json', protocol)
    report = dict(status='running', gpu_executed=False, protocol=identity(output/'protocol.json'),
                  completed_timing_scopes=[])
    try:
        if args.device == 'cuda':
            report['backend'] = apply_controlled_backend()
        for name, path, expected in (('geometry', args.geometry, args.geometry_sha256),
                                     ('asset manifest', args.asset_manifest, args.asset_manifest_sha256)):
            if identity(path)['sha256'] != expected:
                raise ValueError(name+' identity mismatch')
        report['inputs'] = dict(geometry=identity(args.geometry), asset_manifest=identity(args.asset_manifest))
        source_paths = [(name, Path(path).resolve()) for name, path in
            (('graph_spectral.py', graph_spectral.__file__), ('sparse_heat.py', sparse_heat.__file__),
             ('scripts/probe_sparse_heat.py', __file__))]
        report['sources'] = {name: identity(path) for name, path in source_paths}
        manifest = json.loads(Path(args.asset_manifest).read_text())
        report['sealed_source_commit'] = manifest.get('source_commit')
        report['source_overlay_receipt'] = manifest.get('source_overlay_receipt')
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        report['checkout_head'] = git.stdout.strip() if git.returncode == 0 else None
        # Deployed source overlays have no .git; the pinned asset manifest binds their bytes.
        if args.device == 'cuda':
            for name, path in source_paths:
                relative = str(path.relative_to(Path(args.asset_manifest).resolve().parent))
                if manifest.get('files', {}).get(relative) != report['sources'][name]:
                    raise ValueError('Imported source differs from pinned asset manifest: '+name)
        report.setdefault('backend', {}).update(cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        report['torch'] = dict(version=str(torch.__version__), cuda=torch.version.cuda,
                               native_core=identity(torch._C.__file__))
        with np.load(args.geometry, allow_pickle=False) as saved:
            indices, prepared, decoded = (saved[k].copy() for k in ('index', 'prepared_x', 'decoded_xyz'))
            names, counts = saved['candidate_names'].tolist(), saved['candidate_loop_counts'].tolist()
        if (prepared.ndim != 3 or prepared.shape[-1] != 3 or prepared.shape[0] < 1 or prepared.shape[1] < 11
                or prepared.dtype not in (np.float32, np.float64) or decoded.dtype != prepared.dtype
                or decoded.shape != (len(prepared), 4, prepared.shape[1], 3)
                or indices.shape != (len(prepared),) or indices.dtype != np.int64
                or len(set(indices.tolist())) != len(indices) or len(names) != 4 or len(counts) != 4
                or counts != sorted(set(counts)) or not np.isfinite(prepared).all() or not np.isfinite(decoded).all()):
            raise ValueError('Invalid finite H5 geometry/schema')
        if args.device == 'cuda' and (prepared.shape[1] != 2048 or prepared.dtype != np.float32):
            raise ValueError('Native GPU H5 geometry must retain float32 N2048')
        x = torch.from_numpy(prepared).to(args.device)
        residuals = torch.from_numpy(decoded).to(args.device)-x[:, None]
        report.update(gpu_executed=x.is_cuda, original_ids=indices.tolist(), candidate_names=names,
                      candidate_loop_counts=counts, native_dtype=str(x.dtype), device=str(x.device))
        if x.is_cuda:
            report['gpu'] = torch.cuda.get_device_name(x.device)
        report['equivalence'] = check_equivalence(x, residuals, SpectralConfig(), output)
        def save_timing_scope(batch_size, scope, values):
            path = output/('timing_b%d_%s.json' % (batch_size, scope))
            write_json(path, values)
            report['completed_timing_scopes'].append(dict(batch_size=batch_size, scope=scope,
                                                         file=path.name, identity=identity(path)))
            write_json(output/'report.json', report)
        report['timings'] = benchmark(x, residuals[:, 0], SpectralConfig(), indices, save_timing_scope)
        report['status'] = 'complete' if report['equivalence']['equivalence_passed'] else 'equivalence_gate_failed'
        report['cost_claim'] = 'descriptive technical timing only; whole-pipeline/trajectory stage remains conditional'
    except BaseException:
        report.update(status='failed', failure=traceback.format_exc())
        raise
    finally:
        write_json(output/'report.json', report)
        write_json(output/'artifact_manifest.json', {p.name: identity(p) for p in output.iterdir()
                                                    if p.is_file() and p.name != 'artifact_manifest.json'})
    return report


if __name__ == '__main__':
    main()
