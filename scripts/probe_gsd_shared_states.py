#!/usr/bin/env python3
"""Bounded H1/H2/H3/H6/H7 diagnostics on archived, label-free LION states.

No classifier, clean array, trajectory continuation, tuning or production update.
The CLI is CUDA-only; helpers are independently testable on native-free toys.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import eval_multibackbone_source as source

IDS = [459, 545, 1229, 1435]
CORRUPTIONS = ['gaussian', 'background', 'shear', 'lidar']
FRACTIONS = (.5, 1.)
FORWARD_MAX, FORWARD_RMS = 5e-4, 1e-5


def write_json(path: Path, value: dict) -> None:
    """Write one immutable receipt; every stage has a distinct filename."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_arrays(path: Path, values: dict) -> dict:
    arrays = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().copy()
        value = np.asarray(value)
        if value.dtype.kind not in 'biufU' or (value.dtype.kind in 'f' and not np.isfinite(value).all()):
            raise ValueError('Non-numeric/nonfinite artifact: ' + key)
        arrays[key] = value
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return source.file_identity(Path(path))


def forward_gate(reference: torch.Tensor, fresh: torch.Tensor) -> dict:
    if reference.shape != fresh.shape or not torch.isfinite(reference).all() or not torch.isfinite(fresh).all():
        raise ValueError('Forward gate requires equal finite shapes')
    delta = (fresh - reference).detach().flatten(1).double()
    maximum, rms = delta.abs().amax(1), delta.square().mean(1).sqrt()
    return dict(max_absolute_per_object=maximum.cpu().tolist(), rms_per_object=rms.cpu().tolist(),
                max_tolerance=FORWARD_MAX, rms_tolerance=FORWARD_RMS,
                passed=bool(((maximum <= FORWARD_MAX) & (rms <= FORWARD_RMS)).all()))


def xyz(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1, 4)[:, :, :3]


def band_bounds(q: torch.Tensor, low_rank: int, normalized_laplacian: torch.Tensor,
                mean_degree: float, config) -> dict:
    """High100 expands the entire boundary block using production raw-L tolerance.

    Low rank is the actual production expanded rank, never a reconstructed cutoff.
    The tolerance is anchored once at the requested high boundary, without chaining.
    """
    count = len(q)
    if q.ndim != 1 or count != normalized_laplacian.shape[0] or not 0 <= low_rank <= count:
        raise ValueError('Invalid spectrum/support/rank')
    if count == 0:
        return dict(low_rank=0, high_start=0, high_rank=0, high_boundary_tolerance=None, overlap_rank=0)
    if mean_degree <= 0 or not math.isfinite(mean_degree) or not bool(torch.isfinite(q).all()):
        raise ValueError('Invalid degree or ordered spectrum')
    norm = float(normalized_laplacian.abs().sum(1).max())
    atol = max(config.eigenspace_atol / mean_degree,
               2 * torch.finfo(q.dtype).eps * count * norm)
    block_start = 0
    while block_start < count:
        block_end = block_start+1
        anchor = q[block_start]
        block_tolerance = atol + config.eigenspace_rtol * float(anchor.abs())
        while block_end < count and bool(q[block_end]-anchor <= block_tolerance):
            block_end += 1
        # Preserve raw native order; only reversals inside the frozen anchored block are unresolved.
        if not bool(((q[block_start:block_end]-anchor).abs() <= block_tolerance).all()):
            raise ValueError('Normalized eigenvalues leave their anchored eigenspace block')
        block_start = block_end
    start = max(0, count - config.modes)
    boundary = q[start]
    tolerance = atol + config.eigenspace_rtol * float(boundary.abs())
    while start > 0 and bool((q[start-1] - boundary).abs() <= tolerance):
        start -= 1
    return dict(low_rank=low_rank, high_start=start, high_rank=count-start,
                high_boundary_tolerance=tolerance, overlap_rank=max(0, low_rank-start))


def project_band(vector: torch.Tensor, spectrum: tuple, start: int, stop: int) -> torch.Tensor:
    indices, basis, _, _ = spectrum
    if vector.ndim != 2 or vector.shape[-1] != 3 or not 0 <= start <= stop <= basis.shape[1]:
        raise ValueError('Invalid vector or band')
    output = torch.zeros_like(vector)
    selected = basis[:, start:stop]
    output[indices] = selected @ (selected.T @ vector[indices])
    return output


def deterministic_band_direction(reference_xyz: torch.Tensor, spectrum: tuple,
                                 start: int, stop: int) -> torch.Tensor:
    vertices = torch.arange(1, len(reference_xyz)+1, device=reference_xyz.device, dtype=reference_xyz.dtype)
    coordinates = torch.arange(1, 4, device=reference_xyz.device, dtype=reference_xyz.dtype)
    witness = torch.sin(vertices[:, None] * coordinates[None, :])
    direction = project_band(witness, spectrum, start, stop)
    norm = direction.norm()
    if not torch.isfinite(norm) or float(norm) <= 1e-6 * (1 + float(witness.norm())):
        raise ValueError('Deterministic projected direction is tiny; unavailable without amplification')
    return direction / norm


def modal_energy(delta: torch.Tensor, spectrum: tuple) -> dict:
    indices, basis, _, rank = spectrum
    coefficients = basis.T @ delta[indices]
    active = delta[indices].square().sum()
    total = delta.square().sum()
    low = coefficients[:rank].square().sum()
    other = coefficients[rank:].square().sum()
    return dict(total=float(total), active=float(active), low=float(low), other=float(other),
                isolate=float((total-active).clamp_min(0)),
                low_fraction=float(low/total) if float(total) > 0 else None,
                other_fraction=float(other/total) if float(total) > 0 else None,
                parseval_absolute_error=float((coefficients.square().sum()-active).abs()))


def project_scd_xyz(local_gradient: torch.Tensor, style_gradient: torch.Tensor,
                    spectra: tuple) -> tuple:
    """Precondition noisy local XYZ only; preserve feature/style and XYZ norm."""
    if not torch.isfinite(local_gradient).all() or not torch.isfinite(style_gradient).all():
        raise ValueError('Gradients must be finite')
    result = local_gradient.detach().clone()
    original = xyz(local_gradient)
    records = []
    for index, spectrum in enumerate(spectra):
        projected = project_band(original[index], spectrum, 0, spectrum[3])
        norm, projected_norm = float(original[index].norm()), float(projected.norm())
        if projected_norm <= 1e-6 * (1 + norm):
            raise ValueError('P100_C projected XYZ norm is tiny for object %d; unavailable' % index)
        xyz(result)[index] = projected * (norm/projected_norm)
        records.append(dict(original_xyz_norm=norm, projected_xyz_norm=projected_norm,
                            normalized_xyz_norm=float(xyz(result)[index].norm())))
    return result, style_gradient.detach().clone(), records


def joint_norm(local: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    return (local.flatten(1).double().square().sum(1) + style.flatten(1).double().square().sum(1)).sqrt()


def matched_update(gradient: tuple, reference_update: tuple) -> tuple:
    h, style = .03 * gradient[0].detach(), .003 * gradient[1].detach()
    norm, desired = joint_norm(h, style), joint_norm(*reference_update)
    if not torch.isfinite(norm).all() or bool((norm <= 1e-12).any()) or bool((desired <= 1e-12).any()):
        raise ValueError('Joint norm calibration unavailable for zero/nonfinite update')
    ratio = desired / norm
    return (h * ratio.to(h).reshape((-1,) + (1,)*(h.ndim-1)),
            style * ratio.to(style).reshape((-1,) + (1,)*(style.ndim-1)))


def combined_update(scd_update: tuple, spectral_update: tuple) -> tuple:
    combined = tuple(a+b for a, b in zip(scd_update, spectral_update))
    norm, desired = joint_norm(*combined), joint_norm(*scd_update)
    if bool((norm <= 1e-6*(1+desired)).any()) or not torch.isfinite(norm).all():
        raise ValueError('C+S joint direction is tiny; unavailable without amplification')
    ratio = desired/norm
    return tuple(value*ratio.to(value).reshape((-1,)+(1,)*(value.ndim-1)) for value in combined)


def sampler_branch(prev_sample: torch.Tensor, current_style: torch.Tensor,
                   update: tuple, fraction: float) -> tuple:
    if fraction not in FRACTIONS or update[0].shape != prev_sample.shape or update[1].shape != current_style.shape:
        raise ValueError('Invalid fixed fraction or state/update shape')
    if not all(bool(torch.isfinite(value).all()) for value in (prev_sample, current_style, *update)):
        raise ValueError('Nonfinite branch state')
    return prev_sample.detach() - fraction*update[0], current_style.detach() - fraction*update[1]


def next_timestep(schedule: list, loop_count: int, timestep: int) -> int:
    if (type(loop_count) is not int or not 1 <= loop_count < len(schedule)
            or schedule[loop_count-1] != timestep
            or any(a <= b for a, b in zip(schedule, schedule[1:]))):
        raise ValueError('Captured state must be nonterminal on the exact decreasing active schedule')
    return schedule[loop_count]


def embedded_operator(spectrum: tuple, count: int, *, heat: bool) -> torch.Tensor:
    indices, basis, q, _ = spectrum
    weights = torch.exp(-2*q.double()).to(q) if heat else q
    result = basis.new_zeros((count, count))
    result[indices[:, None], indices[None, :]] = (basis * weights[None, :]) @ basis.T
    return (result + result.T) * .5


def relative_pair(left: torch.Tensor, right: torch.Tensor) -> dict:
    a, b = left.double().flatten(), right.double().flatten()
    na, nb = float(a.norm()), float(b.norm())
    return dict(reference_norm=na, current_norm=nb, difference_norm=float((a-b).norm()),
                relative_error=float((a-b).norm())/na if na > 0 else None,
                cosine=float(torch.dot(a, b)/(na*nb)) if na > 0 and nb > 0 else None)


def full_sparse_laplacian(graph, count: int) -> torch.Tensor:
    active = graph.laplacian.to_dense()
    output = active.new_zeros((count, count))
    output[graph.active_indices[:, None], graph.active_indices[None, :]] = active
    return output


def operator_comparison(initial, current, sparse_initial, sparse_current,
                        residual: torch.Tensor) -> list:
    rows = []
    for i, (sx, sy, gx, gy) in enumerate(zip(initial.spectra, current.spectra, sparse_initial.graphs, sparse_current.graphs)):
        count = residual.shape[1]
        if not torch.equal(sx[0], gx.active_indices) or not torch.equal(sy[0], gy.active_indices):
            raise ValueError('Actual eigensystem/sparse support mismatch')
        lx, ly = full_sparse_laplacian(gx, count), full_sparse_laplacian(gy, count)
        ex, ey = lx < 0, ly < 0
        union, intersection = int((ex | ey).sum()), int((ex & ey).sum())
        support_x, support_y = torch.zeros(count, dtype=torch.bool, device=lx.device), torch.zeros(count, dtype=torch.bool, device=lx.device)
        support_x[sx[0]], support_y[sy[0]] = True, True
        union_support = int((support_x | support_y).sum())
        row = dict(edge_overlap_jaccard=intersection/union if union else None,
                   directed_nonzero_edge_counts=[int(ex.sum()), int(ey.sum())],
                   active_counts=[len(sx[0]), len(sy[0])],
                   support_jaccard=int((support_x & support_y).sum())/union_support if union_support else None,
                   expanded_low_ranks=[sx[3], sy[3]],
                   normalized_L=relative_pair(lx, ly),
                   exact_U_L_vs_sparse=[relative_pair(lx, embedded_operator(sx, count, heat=False)),
                                       relative_pair(ly, embedded_operator(sy, count, heat=False))])
        fx, fy = embedded_operator(sx, count, heat=True), embedded_operator(sy, count, heat=True)
        row.update(heat_F=relative_pair(fx, fy),
                   heat_action=relative_pair(fx @ residual[i], fy @ residual[i]),
                   normalized_L_action=relative_pair(lx @ residual[i], ly @ residual[i]),
                   low_boundary_gaps=[float(s[2][s[3]]-s[2][s[3]-1]) if 0 < s[3] < len(s[2]) else None for s in (sx, sy)])
        rows.append(row)
    return rows


def save_target(directory: Path, name: str, target, sparse) -> dict:
    from legacy_trajectory import tensor_hash
    arrays = dict(reference_xyz=target.reference_xyz)
    bands = []
    from graph_spectral import SpectralConfig
    for i, (spectrum, graph) in enumerate(zip(target.spectra, sparse.graphs)):
        indices, basis, q, rank = spectrum
        if not torch.equal(indices, graph.active_indices):
            raise ValueError('Spectral/sparse graph support differs')
        arrays.update({'active_%d' % i: indices, 'basis_%d' % i: basis, 'q_%d' % i: q,
                       'L_indices_%d' % i: graph.laplacian.indices(), 'L_values_%d' % i: graph.laplacian.values()})
        bounds = band_bounds(q, rank, graph.laplacian.to_dense(), graph.mean_active_degree, SpectralConfig())
        bounds.update(mean_active_degree=graph.mean_active_degree, active_count=len(indices),
                      reference_sha256=tensor_hash(target.reference_xyz[i]))
        bands.append(bounds)
    identity = write_arrays(directory/(name+'.npz'), arrays)
    write_json(directory/(name+'.json'), dict(identity=identity, bands=bands, diagnostics=target.diagnostics,
        eigensystem_origin='Actual build_probe_spectral_target on current CUDA device; no CPU reconstruction'))
    return dict(bands=bands, identity=identity)


def timed(operation):
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    return value, time.perf_counter()-started


def decode(vae, local: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    def squeeze(value):
        return value.squeeze(3).squeeze(2) if value.ndim == 4 else value
    return vae.decoder(None, beta=None, context=squeeze(local), style=squeeze(style))


def run_h3(directory: Path, vae, latent, z0, archived_decoded, decoder_rng,
           latent_target, physical_target, latent_bands) -> dict:
    from legacy_trajectory import _restore_rng, isolated_probe
    result = dict(status='running', repeat_gates=[], perturbations=[], errors=[],
                  scales=[.001, .0005], signs=[-1, 1], features='unchanged', style='original z0 fixed',
                  physical_energy='Fixed prepared-X low expanded100/other active/isolate; vertex domain retained')
    def forward(value):
        with isolated_probe([vae]):
            _restore_rng(decoder_rng)
            return decode(vae, value.detach().clone(), z0.detach().clone()).detach()
    repeats = []
    for i in range(3):
        fresh, seconds = timed(lambda: forward(latent))
        repeats.append(fresh)
        write_arrays(directory/('decoder_baseline%d.npz' % i), dict(decoded_xyz=fresh,
                     delta_from_archive=fresh-archived_decoded))
        result['repeat_gates'].append(dict(repeat=i, seconds=seconds, **forward_gate(archived_decoded, fresh)))
    native_floor = []
    for i in range(len(latent)):
        deltas = [repeats[a][i]-repeats[b][i] for a in range(3) for b in range(a)]
        native_floor.append(dict(max_delta_norm=max(float(d.norm()) for d in deltas),
                                 max_delta_rms=max(float(d.square().mean().sqrt()) for d in deltas),
                                 modal=[modal_energy(d, physical_target.spectra[i]) for d in deltas]))
    result['same_state_native_floor'] = native_floor
    result['baseline_fresh_pair_gates'] = [forward_gate(repeats[0], repeat) for repeat in repeats[1:]]
    if not all(gate['passed'] for gate in result['repeat_gates'] + result['baseline_fresh_pair_gates']):
        result['status'] = 'unavailable_forward_gate'
        write_json(directory/'h3.json', result)
        return result
    local_xyz = xyz(latent)
    for band in ('low', 'high'):
        try:
            directions = torch.stack([deterministic_band_direction(local_xyz[i], spectrum,
                0 if band == 'low' else latent_bands[i]['high_start'],
                spectrum[3] if band == 'low' else len(spectrum[2]))
                for i, spectrum in enumerate(latent_target.spectra)])
            for scale in (.001, .0005):
                magnitude = local_xyz.flatten(1).norm(dim=1) * scale
                for sign in (-1, 1):
                    delta = sign * magnitude[:, None, None] * directions
                    changed = latent.detach().clone()
                    xyz(changed).add_(delta)
                    fresh, seconds = timed(lambda: forward(changed))
                    observed = fresh-repeats[0]
                    stem = 'decoder_%s_%s_%s' % (band, str(scale).replace('.', 'p'), 'minus' if sign < 0 else 'plus')
                    write_arrays(directory/(stem+'.npz'), dict(latent_delta_xyz=delta, decoded_xyz=fresh,
                                                               decoded_delta_xyz=observed))
                    rows = []
                    for i in range(len(latent)):
                        energy = modal_energy(observed[i], physical_target.spectra[i])
                        rows.append(dict(output_energy=energy,
                                         identity_xyz_control_energy=modal_energy(delta[i], physical_target.spectra[i]),
                                         delta_norm=float(observed[i].norm()),
                                         modal_energy_above_native_floor={key: energy[key] > max(row[key] for row in native_floor[i]['modal'])
                                                                         for key in ('low', 'other', 'isolate')},
                                         resolved_above_native_floor=float(observed[i].norm()) > native_floor[i]['max_delta_norm']))
                    result['perturbations'].append(dict(band=band, scale=scale, sign=sign, seconds=seconds,
                                                       magnitude_per_object=magnitude.cpu().tolist(), objects=rows))
        except Exception as error:
            result['errors'].append(dict(band=band, error=type(error).__name__+': '+str(error)))
    result['status'] = 'complete' if not result['errors'] else 'complete_with_unavailable_band'
    write_json(directory/'h3.json', result)
    return result


def objective(kind, predicted, decoded, h0, latent_target, physical_target, chamfer):
    if kind == 'C':
        first, second, _, _ = chamfer(xyz(predicted).contiguous(), h0.contiguous())
        retained = int(h0.shape[1]*.95)
        return first.sort(dim=1).values[:, :retained].sum() + second.sort(dim=1).values[:, :retained].sum()
    if kind == 'S_lat':
        return latent_target.loss(xyz(predicted), profile='smooth', beta=2.)
    if kind == 'S_phys':
        return physical_target.loss(decoded, profile='smooth', beta=2.)
    raise ValueError('Unknown frozen objective')


def direction_fd(directory: Path, kind: str, h, style, gradients, evaluate) -> dict:
    """One fixed joint direction per B2 batch, central ±.001 of joint state norm.

    This is a local directional witness, not a global gradient certificate.
    """
    norm = float((gradients[0].double().square().sum()+gradients[1].double().square().sum()).sqrt())
    report = dict(relative_tolerance=.1, absolute_tolerance=1e-5, scale=.001, passed=False)
    if norm <= 1e-5 or not math.isfinite(norm):
        report['status'] = 'unresolved_tiny_derivative'
        return report
    direction = (gradients[0]/norm, gradients[1]/norm)
    epsilon = .001 * float((h.double().square().sum()+style.double().square().sum()).sqrt())
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError('Invalid fixed FD epsilon')
    values, predictions, clouds = [], [], []
    for name, sign in (('repeat0', 0), ('repeat1', 0), ('minus', -1), ('plus', 1)):
        loss, prediction, cloud = evaluate(h + sign*epsilon*direction[0], style + sign*epsilon*direction[1], kind)
        write_arrays(directory/('fd_%s_%s.npz' % (kind, name)), dict(loss=np.asarray(loss),
                     predicted_clean_latent=prediction, decoded_xyz=cloud))
        values.append(loss)
        if sign == 0:
            predictions.append(prediction)
            clouds.append(cloud)
    fd = (values[3]-values[2])/(2*epsilon)
    error = abs(fd-norm)
    tolerance = 1e-5 + .1*norm
    uncertainty = abs(values[1]-values[0])/epsilon
    pred_repeat_gate = forward_gate(predictions[0], predictions[1])
    decoder_repeat_gate = forward_gate(clouds[0], clouds[1])
    passed = (abs(fd) > 1e-5 and error <= tolerance and uncertainty <= tolerance
              and pred_repeat_gate['passed'] and decoder_repeat_gate['passed'])
    report.update(epsilon=epsilon, autograd_directional_derivative=norm, finite_difference=fd,
                  absolute_error=error, allowed_absolute_error=tolerance,
                  repeat_slope_uncertainty=uncertainty, losses=values,
                  predicted_clean_repeat_gate=pred_repeat_gate, decoded_repeat_gate=decoder_repeat_gate,
                  passed=passed, status='passed' if passed else 'failed_or_unresolved')
    return report


def run_directions(directory: Path, lion, scheduler, schedule, loop, timestep, h, style,
                   prior_t, pred_archive, decoded_archive, z0, prior_rng, decoder_rng,
                   h0, latent_target, physical_target, chamfer) -> dict:
    from legacy_trajectory import _restore_rng, _rng_state_hash
    from semantic_trajectory import gradient_probe
    models = [lion.vae, lion.priors]
    report = dict(status='running', objectives={}, branches=[], unavailable={},
        update_scales=dict(local=.03, conditioning_style=.003), fractions=list(FRACTIONS),
        next_timestep=next_timestep(schedule, loop, timestep),
        H1_limit='No negative-step branch; bounded C versus CS comparisons cannot establish universal causal accuracy harm',
        scope='One actual sampler update from scheduler.prev_sample, then next active predicted-clean decode; no full trajectory/PM accuracy')
    def replay(local, condition, *, grad=False, kind=None):
        with gradient_probe(models):
            _restore_rng(prior_rng)
            live_h = local.detach().clone().requires_grad_(True)
            live_s = condition.detach().clone().requires_grad_(True)
            epsilon = lion.priors[1](x=live_h, t=prior_t, condition_input=live_s, clip_feat=None)
            output = scheduler.step(epsilon, timestep, live_h)
            _restore_rng(decoder_rng)
            cloud = decode(lion.vae, output.pred_original_sample, z0.detach().clone())
            loss = objective(kind, output.pred_original_sample, cloud, h0, latent_target, physical_target, chamfer) if kind else None
            gradients = None
            if grad:
                write_arrays(directory/('objective_%s_forward.npz' % kind), dict(loss=loss.detach(),
                    predicted_clean_latent=output.pred_original_sample, decoded_xyz=cloud))
                if not forward_gate(pred_archive, output.pred_original_sample)['passed'] or not forward_gate(decoded_archive, cloud)['passed']:
                    raise ValueError('Objective replay forward gate failed before native backward')
                gradients = torch.autograd.grad(loss, (live_h, live_s), allow_unused=False)
                if not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise ValueError('Nonfinite native gradient')
                gradients = tuple(g.detach().clone() for g in gradients)
            return (None if loss is None else float(loss.detach()), output.pred_original_sample.detach(),
                    cloud.detach(), output.prev_sample.detach(), gradients)
    fresh, seconds = timed(lambda: replay(h, style))
    write_arrays(directory/'prior_live_baseline.npz', dict(predicted_clean_latent=fresh[1], decoded_xyz=fresh[2], prev_sample=fresh[3],
                 predicted_delta_from_archive=fresh[1]-pred_archive, decoded_delta_from_archive=fresh[2]-decoded_archive))
    report.update(live_prior_forward_gate=forward_gate(pred_archive, fresh[1]),
                  live_decoder_forward_gate=forward_gate(decoded_archive, fresh[2]), live_baseline_seconds=seconds)
    if not report['live_prior_forward_gate']['passed'] or not report['live_decoder_forward_gate']['passed']:
        report['status'] = 'unavailable_forward_gate'
        write_json(directory/'directions.json', report)
        return report
    gradients, updates = {}, {}
    for kind in ('C', 'S_lat', 'S_phys'):
        try:
            values, seconds = timed(lambda: replay(h, style, grad=True, kind=kind))
            write_arrays(directory/('gradient_%s.npz' % kind), dict(local_gradient=values[4][0],
                         style_gradient=values[4][1], predicted_clean_latent=values[1], decoded_xyz=values[2]))
            row = dict(loss=values[0], seconds=seconds,
                       local_gradient_norm=values[4][0].flatten(1).norm(dim=1).cpu().tolist(),
                       style_gradient_norm=values[4][1].flatten(1).norm(dim=1).cpu().tolist(),
                       prior_gate=forward_gate(pred_archive, values[1]), decoder_gate=forward_gate(decoded_archive, values[2]))
            report['objectives'][kind] = row
            if not row['prior_gate']['passed'] or not row['decoder_gate']['passed']:
                raise ValueError('Objective replay forward gate failed')
            fd, seconds = timed(lambda: direction_fd(directory, kind, h, style, values[4],
                  lambda a, b, c: replay(a, b, kind=c)[:3]))
            row.update(finite_difference=fd, finite_difference_seconds=seconds)
            if not fd['passed']:
                raise ValueError('Native derivative finite-difference witness failed/unresolved')
            gradients[kind] = values[4]
        except Exception as error:
            report['unavailable'][kind] = type(error).__name__+': '+str(error)
    if 'C' in gradients:
        updates['C'] = (.03*gradients['C'][0], .003*gradients['C'][1])
        try:
            gh, gs, norms = project_scd_xyz(*gradients['C'], latent_target.spectra)
            updates['P100_C'] = (.03*gh, .003*gs)
            write_arrays(directory/'gradient_P100_C.npz', dict(local_gradient=gh, style_gradient=gs))
            report['P100_C'] = dict(norm_witness=norms,
                joint_original=joint_norm(*updates['C']).cpu().tolist(),
                joint_projected=joint_norm(*updates['P100_C']).cpu().tolist(),
                domain='Noisy sampling-state XYZ; projector and denoiser Jacobian do not commute',
                feature_style_unchanged=True)
        except Exception as error:
            report['unavailable']['P100_C'] = type(error).__name__+': '+str(error)
        for kind in ('S_lat', 'S_phys'):
            if kind in gradients:
                try:
                    updates[kind] = matched_update(gradients[kind], updates['C'])
                    combined_name = 'CS_lat' if kind == 'S_lat' else 'CS_phys'
                    updates[combined_name] = combined_update(updates['C'], updates[kind])
                except Exception as error:
                    report['unavailable'][kind if kind not in updates else combined_name] = type(error).__name__+': '+str(error)
    else:
        report['unavailable']['all_norm_matched_branches'] = 'C gradient unavailable'
    updates = dict(zero=(torch.zeros_like(h), torch.zeros_like(style)), **updates)
    # A declared fresh seed produces a saved common RNG independent of branch order.
    from scripts.eval_multibackbone_guidance import seed_inference
    from legacy_trajectory import _rng_state, isolated_probe
    with isolated_probe(models):
        seed_inference(1729)
        common_rng = _rng_state()
    report['next_forward_common_rng'] = dict(seed=1729, sha256=_rng_state_hash(common_rng),
        semantics='Same declared fresh full RNG before next prior; decoder follows that prior without an intermediate reset')
    py, nr, cpu, cuda = common_rng
    write_arrays(directory/'next_common_rng.npz', dict(rng_metadata=np.asarray(json.dumps(dict(sha256=_rng_state_hash(common_rng),
        python_rng=py, numpy_rng=[nr[0], *nr[2:]], cuda_devices=len(cuda)))), numpy_rng_keys=nr[1], torch_rng_cpu=cpu,
        **{'torch_rng_cuda_%d' % i: value for i, value in enumerate(cuda)}))
    def next_forward(local, condition):
        with isolated_probe(models):
            _restore_rng(common_rng)
            t = report['next_timestep']
            t_input = torch.ones_like(prior_t)*(t+1)
            with torch.enable_grad():
                live_h = local.detach().clone().requires_grad_(True)
                live_s = condition.detach().clone().requires_grad_(True)
                eps = lion.priors[1](x=live_h, t=t_input, condition_input=live_s, clip_feat=None)
                output = scheduler.step(eps, t, live_h)
            # Match the existing PM branch after the grad-enabled prior/scheduler.
            predicted = output.pred_original_sample.detach().clone()
            cloud = decode(lion.vae, predicted, z0.detach().clone())
            return predicted, cloud.detach()
    for name, update in updates.items():
        for fraction in FRACTIONS:
            stem = 'branch_%s_%s' % (name, str(fraction).replace('.', 'p'))
            try:
                local, condition = sampler_branch(fresh[3], style, update, fraction)
                write_arrays(directory/(stem+'_state.npz'), dict(noisy_local_next=local, conditioning_style_next=condition,
                                                               update_local=update[0], update_style=update[1]))
                (prediction, cloud), seconds = timed(lambda: next_forward(local, condition))
                write_arrays(directory/(stem+'_prediction.npz'), dict(predicted_clean_latent=prediction, decoded_xyz=cloud))
                report['branches'].append(dict(name=name, fraction=fraction, status='complete', seconds=seconds,
                    joint_update_norm=joint_norm(*update).cpu().tolist(),
                    actual_joint_update_norm=(fraction*joint_norm(*update)).cpu().tolist()))
            except Exception as error:
                report['branches'].append(dict(name=name, fraction=fraction, status='failed_technical',
                    error=type(error).__name__+': '+str(error)))
    report['status'] = 'complete_with_unavailable' if report['unavailable'] or any(r['status'] != 'complete' for r in report['branches']) else 'complete'
    write_json(directory/'directions.json', report)
    return report


def validate_capture_config(config: dict) -> None:
    required = dict(capture_shared_states=True, arm='control', weight=0., seed=0,
                    dataset='modelnet40_c', count=4, batch_size=2, normal_reverse_steps=8,
                    background_reverse_steps=35, selected_original_ids=IDS)
    if any(config.get(k) != value for k, value in required.items()):
        raise ValueError('Capture differs from the frozen shared-state mechanism scope')
    if (config.get('capture_h5') or config.get('capture_native_trace') or
            config.get('corruptions') != CORRUPTIONS or
            config.get('guidance_protocol') != dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False)):
        raise ValueError('Capture protocol/corruption order differs')


def load_case(case: Path, config: dict, corruption: str, batch: int) -> dict:
    """Validate all archived identities before moving any state to the GPU."""
    from legacy_trajectory import tensor_hash
    from scripts.diagnose_decoded_spectral import load_shared_state_rng
    case = Path(case)
    def read(name):
        return source.asset_path(case, name)
    diagnostics = json.loads(read('diagnostics.json').read_text())
    source.verify_identity(read('shared_states.json'), diagnostics['shared_states'])
    record = json.loads(read('shared_states.json').read_text())
    ids = IDS[batch*2:batch*2+2]
    checkpoints = [9, 27] if corruption == 'background' else [2, 6]
    if (record.get('schema_version') != 1 or record.get('complete') is not True
            or record.get('original_ids') != ids or diagnostics.get('original_ids') != ids
            or record.get('checkpoint_steps') != checkpoints or record.get('completed_checkpoints') != checkpoints):
        raise ValueError('Invalid/incomplete capture identity or checkpoints')
    for name, expected in record['source'].items():
        source.verify_identity(source.asset_path(REPO, name), expected)
    ref_entry = record['references']['reference']
    ref_path = read(ref_entry['file'])
    source.verify_identity(ref_path, ref_entry)
    with np.load(ref_path, allow_pickle=False) as archive:
        reference = {k: torch.from_numpy(archive[k].copy()) for k in ('prepared_x', 'initial_h0_xyz')}
    if any(v.shape != (2, 2048, 3) or v.dtype != torch.float32 or not torch.isfinite(v).all() for v in reference.values()):
        raise ValueError('Expected finite float32 B2 reference/h0 XYZ')
    if tensor_hash(reference['prepared_x']) != diagnostics['prepared_x_sha256']:
        raise ValueError('Prepared-X identity differs from original diagnostic')
    paired_entry = record['references']['paired_state']
    paired_path = Path(paired_entry['path'])
    if paired_path.is_symlink() or any(p.is_symlink() for p in paired_path.parents):
        raise ValueError('Paired state symlinks forbidden')
    source.verify_identity(paired_path, paired_entry)
    if any(paired_entry.get(k) != diagnostics['paired_state'].get(k) for k in ('bytes', 'sha256')):
        raise ValueError('Paired reference/diagnostics identity differs')
    with np.load(paired_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        paired = {k: torch.from_numpy(archive[k].copy()) for k in ('local_latent', 'shape_latent', 'noise')}
    if metadata.get('schema_version') != 1 or metadata['input_sha256'] != tensor_hash(reference['prepared_x']):
        raise ValueError('Paired input identity differs')
    for key, value in paired.items():
        if tensor_hash(value) != metadata['tensor_hashes'][key] or not torch.isfinite(value).all():
            raise ValueError('Paired latent/noise identity differs')
    if not torch.equal(xyz(paired['local_latent']), reference['initial_h0_xyz']):
        raise ValueError('Paired full h0 and reference anchor differ')
    if metadata['tensor_hashes']['local_latent'] != record['initial_hashes']['local_latent']:
        raise ValueError('Initial paired/observer latent differs')
    snapshots, arrays = {}, {}
    expected_stages = {(0, 'frame')} | {(loop, stage) for loop in checkpoints for stage in ('pre_prior', 'pre_decoder', 'decoded')}
    for entry in record['snapshots']:
        key = (entry['loop_count'], entry['stage'])
        if key in snapshots or key not in expected_stages or not entry.get('rng_unchanged'):
            raise ValueError('Duplicate/unexpected snapshot or changed writer RNG')
        path = read(entry['file'])
        source.verify_identity(path, entry['identity'])
        with np.load(path, allow_pickle=False) as archive:
            if archive['index'].tolist() != ids:
                raise ValueError('Snapshot original IDs differ')
            tensors = {}
            for name, expected in entry['tensor_sha256'].items():
                value = torch.from_numpy(archive[name].copy())
                if (tensor_hash(value) != expected or list(value.shape) != entry['tensor_shapes'][name]
                        or str(value.dtype) != entry['tensor_dtypes'][name] or not torch.isfinite(value).all()):
                    raise ValueError('Snapshot tensor identity/shape/dtype/finite failure')
                tensors[name] = value
            if entry['stage'] in ('pre_prior', 'pre_decoder'):
                tensors['rng'] = load_shared_state_rng(archive)
                from legacy_trajectory import _rng_state_hash
                if _rng_state_hash(tensors['rng']) != entry['rng_state_sha256']:
                    raise ValueError('Snapshot RNG differs')
        snapshots[key], arrays[key] = entry, tensors
    if set(snapshots) != expected_stages:
        raise ValueError('Missing selected snapshot')
    source.verify_identity(read('terminal_geometry.npz'), diagnostics['terminal_geometry'])
    with np.load(read('terminal_geometry.npz'), allow_pickle=False) as archive:
        raw = archive['raw_corrupted_xyz'].copy()
        if archive['index'].tolist() != ids:
            raise ValueError('Terminal input IDs differ')
        for key in ('normalization_center', 'normalization_scale'):
            if not np.array_equal(archive[key], arrays[(0, 'frame')][key].numpy()):
                raise ValueError('Normalization frame differs')
    dataset = source.dataset_manifest(config['asset_manifest'], config)
    data_path = Path(dataset.get('root', config['dataset_root'])) / ('data_'+corruption+'_5.npy')
    source.verify_identity(data_path, dataset['files'][data_path.name])
    current = np.load(data_path, mmap_mode='r', allow_pickle=False)
    if not np.array_equal(current[ids], raw):
        raise ValueError('Archived raw input differs from current pinned dataset')
    if not torch.isfinite(arrays[(0, 'frame')]['normalization_scale']).all() or bool((arrays[(0, 'frame')]['normalization_scale'] <= 0).any()):
        raise ValueError('Invalid physical normalization scale')
    schedule = diagnostics['scheduler']['active_timesteps']
    for loop in checkpoints:
        prior, clean, decoded = arrays[(loop, 'pre_prior')], arrays[(loop, 'pre_decoder')], arrays[(loop, 'decoded')]
        t = snapshots[(loop, 'pre_prior')]['timestep']
        next_timestep(schedule, loop, t)
        if snapshots[(loop, 'pre_decoder')]['timestep'] != t or snapshots[(loop, 'decoded')]['timestep'] != t:
            raise ValueError('Prior/decoder snapshot timesteps differ')
        shapes = [(prior['noisy_local'], (2, 8192, 1, 1)), (prior['conditioning_style'], (2, 128, 1, 1)),
                  (prior['prior_t'], (2,)), (clean['predicted_clean_latent'], (2, 8192, 1, 1)),
                  (clean['original_decoder_style'], (2, 128, 1, 1)), (decoded['decoded_xyz'], (2, 2048, 3))]
        if any(v.shape != shape or v.dtype != torch.float32 for v, shape in shapes):
            raise ValueError('Captured LION shapes/dtypes differ')
        if not bool((prior['prior_t'] == t+1).all()) or not torch.equal(clean['original_decoder_style'], paired['shape_latent']):
            raise ValueError('Actual prior timestep or original decoder z0 differs')
        probe = next(p for p in record['paired_probe_hashes'] if p['loop_count'] == loop)
        if (tensor_hash(clean['predicted_clean_latent']) != probe['clean_latent_sha256'] or
                tensor_hash(decoded['decoded_xyz']) != probe['decoded_geometry_sha256']):
            raise ValueError('Captured clean/decoded state differs from actual PM probe')
    return dict(record=record, diagnostics=diagnostics, arrays=arrays, reference=reference, schedule=schedule,
                identities={name: source.file_identity(read(name)) for name in ('shared_states.json', 'diagnostics.json', 'reference.npz', 'terminal_geometry.npz')})


def boundary_gaps(points: torch.Tensor, k: int = 10) -> list:
    rows = []
    for sample in points:
        distance = torch.cdist(sample, sample, compute_mode='donot_use_mm_for_euclid_dist').square()
        distance.fill_diagonal_(float('inf'))
        nearest = distance.sort(dim=1, stable=True).values
        gap = nearest[:, k]-nearest[:, k-1]
        rows.append(dict(min=float(gap.min()), median=float(gap.median()), max=float(gap.max()),
                         exact_ties=int((gap == 0).sum()), semantics='Squared-distance nonself kth to k+1 boundary'))
    return rows


def run_case(directory: Path, loaded: dict, lion, scheduler, chamfer) -> dict:
    from graph_spectral import SpectralConfig, build_probe_spectral_target
    from sparse_heat import build_sparse_heat_target
    from legacy_trajectory import _rng_state_hash
    config = SpectralConfig()
    reference = {key: value.cuda() for key, value in loaded['reference'].items()}
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'input.json', dict(identities=loaded['identities'], original_ids=loaded['record']['original_ids'],
        source=loaded['record']['source'], frame=loaded['record']['frame'], schedule=loaded['schedule']))
    targets, sparse_targets, timings, target_reports = {}, {}, {}, {}
    for name, points in (('physical', reference['prepared_x']), ('latent', reference['initial_h0_xyz'])):
        targets[name], timings[name+'_graph_eigh_seconds'] = timed(lambda: build_probe_spectral_target(points, config))
        sparse_targets[name], timings[name+'_sparse_graph_seconds'] = timed(lambda: build_sparse_heat_target(points, config))
        target_reports[name] = save_target(directory, name+'_reference_graph', targets[name], sparse_targets[name])
        gaps, timings[name+'_knn_boundary_seconds'] = timed(lambda: boundary_gaps(points))
        write_json(directory/(name+'_knn_boundary.json'), dict(objects=gaps))
    write_json(directory/'reference_graph_scopes.json', dict(seconds=timings,
        limitation='Dense graph construction and full eigh remain; separately rebuilding sparse graph and extra probes are diagnostic costs'))
    states = []
    for loop in loaded['record']['checkpoint_steps']:
        state_dir = directory/('loop%02d' % loop)
        state_dir.mkdir(exist_ok=False)
        state_report = dict(loop_count=loop, status='running', tests={})
        try:
            prior = loaded['arrays'][(loop, 'pre_prior')]
            clean = loaded['arrays'][(loop, 'pre_decoder')]
            decoded = loaded['arrays'][(loop, 'decoded')]['decoded_xyz'].cuda()
            h, style, prior_t = (prior[key].cuda() for key in ('noisy_local', 'conditioning_style', 'prior_t'))
            latent, z0 = (clean[key].cuda() for key in ('predicted_clean_latent', 'original_decoder_style'))
            t = loaded['schedule'][loop-1]
            write_json(state_dir/'state_identity.json', dict(timestep=t,
                prior_rng_sha256=_rng_state_hash(prior['rng']), decoder_rng_sha256=_rng_state_hash(clean['rng']),
                snapshots=[entry for entry in loaded['record']['snapshots'] if entry['loop_count'] == loop]))
            try:
                current, graph_seconds = timed(lambda: build_probe_spectral_target(decoded, config))
                sparse_current, sparse_seconds = timed(lambda: build_sparse_heat_target(decoded, config))
                save_target(state_dir, 'physical_current_graph', current, sparse_current)
                objects, compare_seconds = timed(lambda: operator_comparison(targets['physical'], current,
                    sparse_targets['physical'], sparse_current, decoded-reference['prepared_x']))
                gaps, gap_seconds = timed(lambda: boundary_gaps(decoded))
                write_json(state_dir/'h2.json', dict(status='complete', objects=objects, current_knn_gaps=gaps,
                    seconds=dict(graph_eigh=graph_seconds, sparse_graph=sparse_seconds, operator_comparison=compare_seconds,
                                 knn_boundary=gap_seconds), anchor='Same original prepared X; residual Y-X for both operators',
                    refreshed_accuracy_run='not implemented; conditional on posthoc directional evidence'))
                state_report['tests']['H2'] = 'complete'
                del current, sparse_current
            except Exception as error:
                write_json(state_dir/'h2_failure.json', dict(status='failed_technical', error=type(error).__name__+': '+str(error)))
                state_report['tests']['H2'] = 'failed_technical'
            try:
                h3 = run_h3(state_dir, lion.vae, latent, z0, decoded, clean['rng'], targets['latent'],
                            targets['physical'], target_reports['latent']['bands'])
                state_report['tests']['H3'] = h3['status']
            except Exception as error:
                write_json(state_dir/'h3_failure.json', dict(status='failed_technical', error=type(error).__name__+': '+str(error)))
                state_report['tests']['H3'] = 'failed_technical'
            try:
                directions = run_directions(state_dir, lion, scheduler, loaded['schedule'], loop, t, h, style,
                    prior_t, latent, decoded, z0, prior['rng'], clean['rng'], reference['initial_h0_xyz'],
                    targets['latent'], targets['physical'], chamfer)
                state_report['tests']['H1_H6_H7'] = directions['status']
            except Exception as error:
                write_json(state_dir/'directions_failure.json', dict(status='failed_technical', error=type(error).__name__+': '+str(error)))
                state_report['tests']['H1_H6_H7'] = 'failed_technical'
            state_report['status'] = 'complete_with_unavailable' if any(s != 'complete' for s in state_report['tests'].values()) else 'complete'
        except Exception as error:
            state_report.update(status='failed_technical', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
        write_json(state_dir/'state_complete.json', state_report)
        states.append(state_report)
        print('GSD_SHARED_STATE_COMPLETE', directory.name, loop, state_report['status'], flush=True)
    return dict(states=states, status='complete' if all(s['status'] == 'complete' for s in states) else 'complete_with_unavailable')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-root', 'capture-config-sha256', 'asset-root', 'asset-manifest-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    capture = Path(args.capture_root).absolute()
    config_path = source.asset_path(capture, 'config.json')
    config_identity = source.file_identity(config_path)
    if config_identity['sha256'] != args.capture_config_sha256:
        raise ValueError('Capture config SHA-256 differs')
    config = json.loads(config_path.read_text())
    validate_capture_config(config)
    if config['asset_manifest_sha256'] != args.asset_manifest_sha256 or Path(config['asset_root']).resolve() != Path(args.asset_root).resolve():
        raise ValueError('Capture and probe must use the same pinned asset root/manifest')
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, status='running', capture_root=str(capture), capture_config=config_identity,
        frozen_ids=IDS, corruptions=CORRUPTIONS, labels_and_clean_arrays_loaded=False, cases=[],
        scope='H1/H2/H3/H6/H7 selected-state mechanism diagnostics; no full trajectories or accuracy',
        started_unix_seconds=time.time())
    write_json(output/'started.json', report)
    from legacy_trajectory import _rng_state, _restore_rng, rng_hash
    entry_rng, entry_hash = _rng_state(), rng_hash()
    models_before, lion = None, None
    try:
        if not torch.cuda.is_available() or torch.__version__ != '2.1.2+cu121' or torch.version.cuda != '12.1':
            raise RuntimeError('Requires CUDA and exact Torch2.1.2+cu121; local/native-free tests only elsewhere')
        manifest = source.read_manifest(Path(args.asset_root), args.asset_manifest_sha256)
        config['asset_manifest'] = manifest
        from scripts.diagnose_decoded_spectral import apply_backend_policy, verify_setup
        report['backend'] = apply_backend_policy(config['backend_policy'])
        torch.cuda.get_rng_state_all()
        entry_rng, entry_hash = _rng_state(), rng_hash()
        report['setup'] = verify_setup(Path(args.asset_root), manifest, config)
        for name, expected in config['executed_source_files'].items():
            if name.startswith('repo/'):
                source.verify_identity(source.asset_path(REPO, name[5:]), expected)
            elif name.startswith('assets/'):
                source.verify_identity(source.asset_path(Path(args.asset_root), name[7:]), expected)
        from scripts import eval_multibackbone_guidance as shared
        runtime_config = copy.deepcopy(config)
        runtime_config['method'] = shared.ORIGINAL_METHOD
        lion, graph, _, _, native, resolved = shared.load_guidance_runtime(runtime_config, manifest)
        if graph is not None or native.get('no_build') is not True:
            raise RuntimeError('Original no-build raw-checkpoint runtime required')
        for key in ('bindings', 'native_audit_sha256'):
            if native.get(key) != config['native_bootstrap'].get(key):
                raise ValueError('Native bootstrap identity differs from captured source')
        if resolved != config['resolved_lion_config']:
            raise ValueError('Resolved LION configuration differs')
        models_before = dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors))
        if models_before != config['lion_state_before']:
            raise ValueError('Raw/eval/EMA-off model state differs from capture')
        if any(m.training for model in (lion.vae, lion.priors) for m in model.modules()) or any(p.requires_grad for model in (lion.vae, lion.priors) for p in model.parameters()):
            raise ValueError('All model modules must be eval and parameters frozen')
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler(beta_end=.02, beta_schedule='linear', beta_start=.0001,
            clip_sample=False, num_train_timesteps=1000, prediction_type='epsilon')
        scheduler.set_timesteps(100, device='cuda')
        if float(scheduler.final_alpha_cumprod) != 1.:
            raise ValueError('Frozen final alpha must be1')
        from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
        chamfer = chamfer_3DDist()
        from pointnet2_ops import pointnet2_utils  # Bind the captured PM native extension without a classifier forward.
        for name, expected in config['native_extensions'].items():
            module = sys.modules.get(name)
            if module is None or Path(getattr(module, '__file__', '')).resolve() != Path(expected['path']).resolve():
                raise ValueError('Native extension loaded path differs from capture: '+name)
            source.verify_identity(Path(expected['path']), expected)
        from scripts.eval_chamfer_depth import scheduler_record
        report.update(native=native, model_before=models_before, runtime=dict(torch=torch.__version__, numpy=np.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()),
            executed_probe_source={name: source.file_identity(REPO/name) for name in
                ('scripts/probe_gsd_shared_states.py', 'graph_spectral.py', 'sparse_heat.py', 'semantic_trajectory.py', 'legacy_trajectory.py')})
        write_json(output/'runtime.json', {key: value for key, value in report.items() if key not in ('cases',)})
        for corruption in CORRUPTIONS:
            for batch in range(2):
                case = source.asset_path(capture, 'batches/%s/batch%04d/shared_states.json' % (corruption, batch)).parent
                loaded = load_case(case, config, corruption, batch)
                if loaded['diagnostics']['scheduler'] != scheduler_record(scheduler, 35 if corruption == 'background' else 8):
                    raise ValueError('Exact active scheduler/config/alpha identity differs')
                try:
                    result = run_case(output/corruption/('batch%04d' % batch), loaded, lion, scheduler, chamfer)
                except Exception as error:
                    result = dict(status='failed_technical', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
                    write_json(output/('failure_%s_batch%04d.json' % (corruption, batch)), result)
                report['cases'].append(dict(corruption=corruption, batch=batch, **result))
        report['status'] = 'complete' if all(r['status'] == 'complete' for r in report['cases']) else 'complete_with_unavailable'
    except Exception as error:
        report.update(status='failed_technical', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        if lion is not None and models_before is not None:
            report['model_after'] = dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors))
            report['model_unchanged'] = report['model_after'] == models_before
            if not report['model_unchanged']:
                report['status'] = 'failed_model_mutation'
        _restore_rng(entry_rng)
        report.update(entry_rng_restored=rng_hash() == entry_hash, finished_unix_seconds=time.time())
        report['wall_seconds'] = report['finished_unix_seconds']-report['started_unix_seconds']
        report['artifacts'] = {str(path.relative_to(output)): source.file_identity(path)
            for path in sorted(output.rglob('*')) if path.is_file()}
        write_json(output/'complete.json', report)
    if report['status'].startswith('failed'):
        raise RuntimeError(report.get('error', report['status'])+'; partial evidence retained at '+str(output))
    print('GSD_SHARED_PROBES_COMPLETE', str(output), report['status'], flush=True)


if __name__ == '__main__':
    main()
