#!/usr/bin/env python3
"""Bounded decoded-X gradient/correspondence diagnostic on pinned GPU assets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import eval_multibackbone_guidance as shared
from scripts import eval_multibackbone_source as source

IDS = [44, 91, 103, 137, 158, 223, 236, 265]


def configuration(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('asset-root', 'asset-manifest-sha256', 'result-root', 'paired-state-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--drive-root')
    parser.add_argument('--backend-policy', choices=('legacy', 'controlled'), default='legacy')
    parser.add_argument('--index-manifest')
    parser.add_argument('--index-manifest-sha256')
    parser.add_argument('--arm', choices=('control', 'live_control', 'diagnostic', 'physical'), default='diagnostic')
    parser.add_argument('--weight', type=float, default=0.)
    parser.add_argument('--native-decoder-audit', action='store_true')
    parser.add_argument('--freeze-decoder-residual', action='store_true')
    parser.add_argument('--count', type=int, choices=(2, 8), default=2)
    parser.add_argument('--corruptions', nargs='+', choices=('gaussian', 'background', 'shear', 'lidar'),
                        default=['gaussian', 'background'])
    args = parser.parse_args(argv)
    if args.freeze_decoder_residual and args.arm not in ('physical', 'live_control'):
        raise ValueError('Frozen decoder residual requires a physical or lightweight live-control arm')
    if args.native_decoder_audit and args.arm != 'diagnostic':
        raise ValueError('Native decoder audit requires the diagnostic arm')
    if (not math.isfinite(args.weight) or args.weight < 0
            or (args.arm == 'physical') != (args.weight > 0)):
        raise ValueError('Physical requires finite positive --weight; other arms require zero')
    if not re.fullmatch('[0-9a-f]{64}', args.asset_manifest_sha256):
        raise ValueError('A pinned asset manifest SHA-256 is required')
    if len(set(args.corruptions)) != len(args.corruptions):
        raise ValueError('Duplicate corruptions are forbidden')
    if (args.index_manifest is None) != (args.index_manifest_sha256 is None):
        raise ValueError('--index-manifest and --index-manifest-sha256 must be supplied together')
    config = vars(args)
    for key in ('asset_root', 'result_root', 'paired_state_root', 'drive_root', 'index_manifest'):
        if config[key]:
            config[key] = str(Path(config[key]).absolute())
    config.update(method='decoded_physical_spectral_diagnostic', dataset='modelnet40_c',
                  backbone='pointmae', seed=0, severity=5, batch_size=2, n_examples=2468,
                  num_classes=40, split='official_modelnet40_test', n_points=1024, input_layout='bnc',
                  pointmae_checkpoint='checkpoints/modelnet_jt.pth',
                  pointmae_config='source_snapshot/cfgs/tta_modelnet.yaml',
                  checkpoint='checkpoints/modelnet_jt.pth', selected_original_ids=IDS[:args.count],
                  guidance_protocol=dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False),
                  normal_reverse_steps=8, background_reverse_steps=35,
                  trajectory_selection=dict(normal=[2, 4, 6], background=[9, 18, 27]),
                  physical_graph=None if args.arm == 'control' else dict(k=10, delta=.1, graph_gamma=.6,
                                                                       modes=100, profile='smooth', beta=2.),
                  preprocessing='normalize -> upsample2048 -> scale3.3885 -> rotate; original encoded z0 decode',
                  correspondence_caveat='Spectral and slot residuals assume decoded/prepared vertex correspondence',
                  purpose='Gradient/correspondence diagnostic; selected-ID accuracy is exploratory',
                  backend_policy_limitation='Custom/native kernels may remain nondeterministic; deterministic algorithms disabled',
                  terminal_geometry_artifact=dict(filename='terminal_geometry.npz',
                      fields=['index', 'raw_corrupted_xyz', 'normalization_center', 'normalization_scale',
                              'terminal_decoded_xyz', 'terminal_classifier_geometry'],
                      clean_to_physical='((clean-center)/scale)*3.3885 then rotate(-z,y,x)'),
                  randomness_policy='Seed0 reset before each corruption/B2 case; paired encoded state/noise cache')
    if args.index_manifest is not None:
        selection, identity = read_index_manifest(Path(config['index_manifest']), args.index_manifest_sha256, args.count)
        config.update(selected_original_ids=selection['indices'], selection_manifest=selection,
                      selection_manifest_identity=identity)
    return config


def read_index_manifest(path: Path, expected_sha256: str, count: int) -> tuple[dict, dict]:
    identity = source.file_identity(path)
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256 or '') or identity['sha256'] != expected_sha256:
        raise ValueError('Index manifest SHA-256 mismatch')
    selection = json.loads(path.read_text())
    indices = selection.get('indices') if isinstance(selection, dict) else None
    if (not isinstance(selection, dict) or type(selection.get('schema_version')) is not int
            or selection['schema_version'] != 1 or selection.get('dataset') != 'modelnet40_c'
            or not isinstance(indices, list) or len(indices) != count
            or any(type(index) is not int or not 0 <= index < 2468 for index in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Index manifest requires schema1 ModelNet40-C sorted unique integer IDs matching count')
    return selection, identity


def apply_backend_policy(policy: str) -> dict:
    """Set flags before CUDA/model initialization; custom kernels remain uncontrolled."""
    import torch
    if policy not in ('legacy', 'controlled'):
        raise ValueError('Unknown backend policy')
    controlled = policy == 'controlled'
    torch.backends.cudnn.benchmark = not controlled
    torch.backends.cudnn.deterministic = controlled
    if controlled:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    return dict(cudnn=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                backend_determinism_guaranteed=False)


def repeat_full_decoder(vae, latent, style, decoded, original_rng) -> tuple[dict, object]:
    """Repeat the live decoder branch at the checkpoint's original RNG boundary."""
    import torch
    from legacy_trajectory import _restore_rng, model_buffer_hashes, rng_hash, tensor_hash
    from semantic_trajectory import gradient_probe
    before, buffers = rng_hash(), model_buffer_hashes({'vae': vae})
    inputs = tensor_hash(latent), tensor_hash(style)
    with gradient_probe([vae]):
        _restore_rng(original_rng)
        context, decoder_style = latent.clone().squeeze(3).squeeze(2), style.clone().squeeze(3).squeeze(2)
        repeated = vae.decoder(None, beta=None, context=context, style=decoder_style)
        output_requires_grad = repeated.requires_grad
    after = rng_hash()
    if (before != after or buffers != model_buffer_hashes({'vae': vae})
            or inputs != (tensor_hash(latent), tensor_hash(style))):
        raise RuntimeError('Full decoder repeat changed RNG, buffers or live inputs')
    difference = repeated.detach() - decoded.detach()
    if not torch.isfinite(difference).all():
        raise ValueError('Full decoder repeat difference is nonfinite')
    flat = difference.flatten(1)
    return dict(full_repeat_decoder_max_absolute_difference=float(difference.abs().max()),
                full_repeat_decoder_rms_difference=float(difference.square().mean().sqrt()),
                full_repeat_decoder_max_absolute_difference_per_object=flat.abs().max(dim=1).values.cpu().tolist(),
                full_repeat_decoder_rms_difference_per_object=flat.square().mean(dim=1).sqrt().cpu().tolist(),
                full_repeat_decoder_output_requires_grad=output_requires_grad,
                full_repeat_decoder_context_requires_grad=context.requires_grad,
                full_repeat_decoder_style_requires_grad=decoder_style.requires_grad,
                full_repeat_decoder_rng_unchanged=True, full_repeat_decoder_buffers_unchanged=True,
                full_repeat_decoder_live_inputs_unchanged=True, extra_full_repeat_decoder_forward_calls=1), repeated.detach()


def record_terminal_geometry(directory: Path, ids, raw, center, scale, decoded, geometry) -> None:
    """Retain the corrupted normalization frame for later independent clean oracles."""
    import numpy as np
    np.savez_compressed(directory / 'terminal_geometry.npz', index=np.asarray(ids, dtype=np.int64),
                        raw_corrupted_xyz=np.array(raw, copy=True),
                        normalization_center=center.detach().cpu().numpy(), normalization_scale=scale.detach().cpu().numpy(),
                        terminal_decoded_xyz=decoded.detach().cpu().numpy(),
                        terminal_classifier_geometry=geometry.detach().cpu().numpy())


def record_reference(capture, directory: Path, prepared, state) -> None:
    """Keep the exact detached h0 anchor in memory and in recoverable evidence."""
    import numpy as np
    initial_xyz = state['local_latent'].detach().reshape(len(prepared), 2048, 4)[:, :, :3].clone()
    capture['initial_h0_xyz'] = initial_xyz
    np.savez_compressed(directory / 'reference.npz', prepared_x=prepared.detach().cpu().numpy(),
                        initial_h0_xyz=initial_xyz.cpu().numpy())


def verify_setup(root: Path, manifest: dict, config: dict) -> dict:
    setup_path = source._declared_asset(root, manifest, manifest['setup_verification'])
    setup = json.loads(setup_path.read_text())
    if setup.get('status') != 'passed':
        raise ValueError('Canonical setup verification must have passed')
    pairs = [('pointnet_ckpts/modelnet_jt.pth', manifest['files'][config['pointmae_checkpoint']]),
             ('lion_ckpts/epoch_10999_iters_2100999.pt', manifest['guidance']['lion_checkpoint'])]
    dataset = source.dataset_manifest(manifest, config)
    for name in ['label.npy', *('data_' + c + '_5.npy' for c in config['corruptions'])]:
        pairs.append(('data/modelnet40_c/' + name, dataset['files'][name]))
    for name, expected in pairs:
        if any(setup.get('assets', {}).get(name, {}).get(key) != expected.get(key)
               for key in ('bytes', 'sha256')):
            raise ValueError('Setup/manifest asset identity differs: ' + name)
    return dict(path=str(setup_path), identity=source.file_identity(setup_path), report=setup)


def worker(directory: str) -> None:
    import numpy as np
    import torch
    from decoded_spectral import DecodedSpectralObserver
    from graph_spectral import SpectralConfig, build_smooth_spectral_target
    from legacy_trajectory import _rng_state, isolated_probe, rng_hash, tensor_hash
    from shared_trajectory import SharedTrajectoryObserver
    from semantic_trajectory import gradient_probe
    from scripts.analyze_legacy_trajectory import policies
    from scripts.eval_chamfer_depth import scheduler_record

    bundle = source.RunBundle(Path(directory))
    config = json.loads((bundle.path / 'config.json').read_text())
    rows, heads, lion, manifest = [], {}, None, None
    active_observer, active_directory = None, None
    started = time.monotonic()
    try:
        config['runtime'] = apply_backend_policy(config.get('backend_policy', 'legacy'))
        if config.get('index_manifest'):
            selection, identity = read_index_manifest(bundle.path / 'selection_manifest.json',
                                                      config['index_manifest_sha256'], config['count'])
            if (selection != config['selection_manifest'] or identity != config['selection_manifest_identity']
                    or selection['indices'] != config['selected_original_ids']):
                raise ValueError('Stored index manifest differs from the selected worker IDs')
        root = Path(config['asset_root'])
        manifest = source.read_manifest(root, config['asset_manifest_sha256'])
        dataset = source.dataset_manifest(manifest, config)
        config['dataset_root'] = str(Path(dataset.get('root', str(REPO / 'data/modelnet40_c'))).absolute())
        config['label_path'] = str(Path(config['dataset_root']) / 'label.npy')
        config.update(asset_manifest=manifest, dataset_provenance=dataset,
                      setup_verification=verify_setup(root, manifest, config))
        labels = source.load_array(Path(config['label_path']), dataset['files']['label.npy'], config, labels=True)
        arrays = {c: source.load_array(source.selected_data_path(config, c),
                                      dataset['files'][source.selected_data_path(config, c).name], config)
                  for c in config['corruptions']}
        if not torch.cuda.is_available():
            raise RuntimeError('Decoded spectral diagnostic requires CUDA')
        runtime_config = copy.deepcopy(config)
        runtime_config['method'] = shared.ORIGINAL_METHOD
        lion, graph, reconstruct, ops, native, resolved = shared.load_guidance_runtime(runtime_config, manifest)
        if graph is not None:
            raise RuntimeError('Original runtime must not initialize a latent graph')
        heads, guard = shared.initialize_heads({'pointmae': config}, manifest)
        from pointnet2_ops import pointnet2_utils
        from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
        chamfer = chamfer_3DDist()
        def fps(points, number):
            indices = pointnet2_utils.furthest_point_sample(points.contiguous(), number)
            return pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
        config.update(native_bootstrap=native, resolved_lion_config=resolved,
                      initialization_rng_guard=guard,
                      runtime=dict(config['runtime'], torch=torch.__version__, numpy=np.__version__,
                                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()),
                      lion_state_before=dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors)),
                      classifier_state_before_sha256=source.state_hash(heads['pointmae']))
        source.snapshot_sources(bundle, config, manifest)
        bundle.write_config(config)
        saved_filters = False
        for corruption, values in arrays.items():
            accumulated, correct, elapsed, peak = [], 0, 0., 0.
            for batch, start in enumerate(range(0, config['count'], 2)):
                ids = np.asarray(config['selected_original_ids'][start:start+2], dtype=np.int64)
                # Each case has the same pre-preparation boundary regardless of arm/order.
                before_rng = shared.seed_inference(0)
                raw_original = np.array(values[ids], copy=True)
                raw = torch.from_numpy(np.array(raw_original, dtype=np.float32, copy=True))
                normalized, center, scale = ops.normalize(raw)
                prepared = torch.from_numpy(ops.upsample_all(normalized.numpy(), 2048)).float().cuda()
                prepared = ops.rotate_pointcloud(prepared * 3.3885)
                if prepared.shape != (2, 2048, 3) or not torch.isfinite(prepared).all():
                    raise ValueError('Expected finite prepared B2 x 2048 x 3')
                steps = 35 if corruption == 'background' else 8
                checkpoints = config['trajectory_selection']['background' if steps == 35 else 'normal']
                case = bundle.path / 'batches' / corruption / ('batch%04d' % batch)
                case.mkdir(parents=True, exist_ok=False)
                active_directory = case
                capture, prior_inputs = {}, {}
                def classify(decoded):
                    capture['decoded'] = decoded.detach().clone()
                    if 'first_saved' not in capture:
                        np.savez_compressed(case / 'decoded_first_checkpoint.npz', decoded_xyz=decoded.detach().cpu().numpy())
                        capture['first_saved'] = True
                    geometry = shared.decoded_to_geometry(decoded, config['dataset'], scale, center, ops)
                    return shared.classify_geometry(geometry, heads, {'pointmae': config}, fps)[0]
                target = None
                if config['arm'] == 'control':
                    observer = SharedTrajectoryObserver(lion.vae, heads, classify, steps, 40,
                                                        checkpoint_steps=checkpoints)
                else:
                    graph_rng = rng_hash()
                    target = build_smooth_spectral_target(prepared, SpectralConfig(), profile='smooth', beta=2.)
                    if rng_hash() != graph_rng:
                        raise RuntimeError('Physical graph advanced RNG')
                    class DiagnosticObserver(DecodedSpectralObserver):
                        def clean(self, latent, style, loop_index, timestep):
                            if (loop_index + 1 in self.steps and config['native_decoder_audit']
                                    and not capture.get('decoder_native_audit_done')):
                                from decoded_spectral_native_audit import audit_decoded_gradient
                                report, evidence = audit_decoded_gradient(lion.vae, target, latent, style)
                                np.savez_compressed(case / 'decoder_native_audit.npz', **evidence)
                                (case / 'decoder_native_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                                capture['decoder_native_audit_done'] = True
                                print('DECODER_NATIVE_AUDIT ' + json.dumps(report, allow_nan=False), flush=True)
                                if not report['passed']:
                                    raise RuntimeError('Native decoder finite-difference gate failed; evidence retained')
                            repeat_rng = _rng_state() if loop_index + 1 in self.steps else None
                            loss = super().clean(latent, style, loop_index, timestep)
                            if loss is None:
                                return None
                            repeat_record, repeated = repeat_full_decoder(lion.vae, latent, style, capture['decoded'], repeat_rng)
                            self.spectral_records[-1].update(repeat_record)
                            if 'full_repeat_decoder_saved' not in capture:
                                np.savez_compressed(case / 'full_repeat_decoder_first_checkpoint.npz',
                                    decoded_xyz=repeated.cpu().numpy(), reference_decoded_xyz=capture['decoded'].cpu().numpy())
                                capture['full_repeat_decoder_saved'] = True
                            local, condition = prior_inputs['x'], prior_inputs['condition_input']
                            gradients = torch.autograd.grad(loss, (local, condition), retain_graph=True, allow_unused=True)
                            record = self.spectral_records[-1]
                            for name, gradient in zip(('local', 'conditioning_style'), gradients):
                                record[name + '_connected'] = gradient is not None
                                record[name + '_gradient_norm_per_object'] = None if gradient is None else gradient.detach().flatten(1).norm(dim=1).cpu().tolist()
                            local_xyz = latent.reshape(2, 2048, 4)[:, :, :3]
                            c1, c2, _, _ = chamfer(local_xyz.contiguous(), capture['initial_h0_xyz'].contiguous())
                            retained = int(2048 * .95)
                            scd = c1.sort(dim=1).values[:, :retained].sum() + c2.sort(dim=1).values[:, :retained].sum()
                            scd_gradients = torch.autograd.grad(scd, (local, condition), retain_graph=True)
                            update_c, update_s = [], []
                            for name, cg, sg, step_size in zip(('local', 'conditioning_style'), scd_gradients, gradients, (.03, .003)):
                                cflat = cg.detach().flatten(1).double()
                                record[name + '_scd_gradient_norm_per_object'] = cflat.norm(dim=1).cpu().tolist()
                                if sg is None:
                                    raise RuntimeError('Physical loss disconnected from ' + name)
                                sflat = sg.detach().flatten(1).double()
                                denominator = cflat.norm(dim=1) * sflat.norm(dim=1)
                                record[name + '_scd_physical_cosine_per_object'] = [float(dot / norm) if norm > 0 else None
                                    for dot, norm in zip((cflat * sflat).sum(1), denominator)]
                                update_c.append(step_size * cflat)
                                update_s.append(step_size * sflat)
                            cn = torch.cat(update_c, dim=1).norm(dim=1)
                            sn = torch.cat(update_s, dim=1).norm(dim=1)
                            record['unit_weight_scd_over_physical_update_ratio_per_object'] = [float(c / s) if s > 0 else None for c, s in zip(cn, sn)]
                            decoded = capture['decoded']
                            with gradient_probe([lion.vae]):
                                detached_decoded = lion.vae.decoder(None, beta=None,
                                    context=latent.detach().clone().squeeze(3).squeeze(2),
                                    style=style.detach().clone().squeeze(3).squeeze(2))
                            difference = detached_decoded - decoded
                            record['detached_decoder_output_requires_grad'] = detached_decoded.requires_grad
                            record['detached_decoder_max_absolute_difference'] = float(difference.abs().max())
                            record['detached_decoder_rms_difference'] = float(difference.square().mean().sqrt())
                            record['extra_detached_decoder_forward_calls'] = 1
                            if 'detached_decoder_saved' not in capture:
                                np.savez_compressed(case / 'detached_decoder_first_checkpoint.npz',
                                    decoded_xyz=detached_decoded.cpu().numpy())
                                capture['detached_decoder_saved'] = True
                            surrogate_points = decoded + (local_xyz - local_xyz.detach())
                            surrogate_loss = self.target.loss(surrogate_points)
                            surrogate_gradients = torch.autograd.grad(surrogate_loss, (local, condition), retain_graph=True)
                            surrogate_update = torch.cat([step * grad.detach().flatten(1).double()
                                for step, grad in zip((.03, .003), surrogate_gradients)], dim=1)
                            full_update = torch.cat(update_s, dim=1)
                            surrogate_norm = surrogate_update.norm(dim=1)
                            record['residual_freeze_update_cosine_per_object'] = [float(dot / norm) if norm > 0 else None
                                for dot, norm in zip((full_update * surrogate_update).sum(1), sn * surrogate_norm)]
                            record['residual_freeze_over_full_update_norm_ratio_per_object'] = [float(s / f) if f > 0 else None
                                for s, f in zip(surrogate_norm, sn)]
                            record['extra_surrogate_vjp_calls'] = 1
                            record['extra_diagnostic_scd_forward_calls'] = 1
                            record['extra_diagnostic_scd_vjp_calls'] = 1
                            decoded = capture['decoded']
                            with isolated_probe([]):
                                residual = decoded - self.target.reference_xyz
                                record['slot_mse_per_object'] = residual.square().mean((1, 2)).cpu().tolist()
                                record['spectral_loss_per_object'] = [float((r * (f @ r)).sum() / (3*2048))
                                                                      for r, f in zip(residual, self.target.filters)]
                                d1, d2, _, _ = chamfer(decoded.contiguous(), prepared.contiguous())
                                record['set_chamfer_mean_per_object'] = (d1.mean(1) + d2.mean(1)).cpu().tolist()
                            # Preserve useful checkpoint evidence even if a later native backward fails.
                            (case / 'probe_records.json').write_text(json.dumps(self.spectral_records, indent=2, allow_nan=False) + '\n')
                            return loss
                    observer_class = DiagnosticObserver if config['arm'] == 'diagnostic' else DecodedSpectralObserver
                    observer = observer_class(lion.vae, heads, classify, steps, target=target,
                        checkpoint_steps=checkpoints, record_gradients=config['arm'] == 'diagnostic',
                        freeze_decoder_residual=config['freeze_decoder_residual'])
                    filter_hashes = [tensor_hash(value) for value in target.filters]
                    (case / 'graph.json').write_text(json.dumps(dict(diagnostics=target.diagnostics,
                        reference_xyz_sha256=tensor_hash(target.reference_xyz), filter_sha256=filter_hashes), indent=2) + '\n')
                    if not saved_filters:
                        np.savez_compressed(case / 'physical_filters.npz', filters=torch.stack(target.filters).cpu().numpy())
                        saved_filters = True
                active_observer = observer
                def capture_prior(module, args, kwargs, output):
                    prior_inputs.update(x=kwargs['x'], condition_input=kwargs['condition_input'])
                hook = lion.priors[1].register_forward_hook(capture_prior, with_kwargs=True)
                ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()
                paired = Path(config['paired_state_root']) / config['dataset'] / corruption / (ids_sha + '.npz')
                state, recorder = shared.paired_preparation(paired, prepared)
                def record_state(values):
                    record_reference(capture, case, prepared, values)
                    if recorder is not None:
                        recorder(values)
                scheduler = {}
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                batch_started = time.monotonic()
                try:
                    decoded = reconstruct(prepared, lion, steps, **config['guidance_protocol'],
                        prepared_state=state, state_observer=record_state, trajectory_observer=observer,
                        decoded_spectral_weight=config['weight'],
                        scheduler_observer=lambda value: scheduler.update(scheduler_record(value, steps)))
                finally:
                    hook.remove()
                torch.cuda.synchronize()
                elapsed += time.monotonic() - batch_started
                after_rng = rng_hash()
                geometry = shared.decoded_to_geometry(decoded.detach(), config['dataset'], scale, center, ops)
                record_terminal_geometry(case, ids, raw_original, center, scale, decoded, geometry)
                logits, head_record = shared.classify_geometry(geometry, heads, {'pointmae': config}, fps)
                if rng_hash() != after_rng:
                    raise RuntimeError('Terminal geometry/classifier changed reconstruction RNG')
                observer.terminal(logits)
                payload = dict(index=ids, label=np.asarray(labels[ids], dtype=np.int64),
                               terminal_prediction=logits['pointmae'].argmax(-1).cpu().numpy(),
                               logits=logits['pointmae'].cpu().numpy(), **observer.arrays('pointmae'))
                payload.update(policies(payload['trajectory_logits'], len(observer.names)-1))
                payload['prediction'] = payload['probability_mean']
                np.savez_compressed(case / 'predictions.npz', **payload)
                diagnostics = dict(trajectory=observer.diagnostics(), scheduler=scheduler, original_ids=ids.tolist(),
                                   input_rng_before_sha256=before_rng, reconstruction_rng_after_sha256=after_rng,
                                   prepared_x_sha256=tensor_hash(prepared), geometry_sha256=tensor_hash(geometry),
                                   paired_state=dict(path=str(paired), **source.file_identity(paired)),
                                   terminal_geometry=dict(file='terminal_geometry.npz', **source.file_identity(case / 'terminal_geometry.npz')),
                                   head=head_record, elapsed_seconds=time.monotonic()-batch_started)
                (case / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False)+'\n')
                accumulated.append(payload)
                correct += int((payload['prediction'] == payload['label']).sum())
                peak = max(peak, torch.cuda.max_memory_allocated()/1024**2)
                shared.update_row(rows, config, corruption, start+2, correct, elapsed, peak, 'partial')
                bundle.write_results(rows, 'running')
                config.setdefault('completed_batches', {})[corruption] = batch+1
                bundle.write_config(config)
                shared.copy_batch_shard(bundle, config['drive_root'], corruption, batch,
                                        {path.name: path for path in case.iterdir() if path.is_file()})
                active_observer, active_directory = None, None
                print('DECODED_SPECTRAL_BATCH_COMPLETE', config['arm'], corruption, batch, flush=True)
            np.savez_compressed(bundle.path / (corruption + '_predictions.npz'),
                                **{key: np.concatenate([p[key] for p in accumulated])
                                   for key in ('index', 'label', 'prediction', 'terminal_prediction', 'logits', 'trajectory_logits')})
        config.update(status='partial', diagnostic_complete=True,
                      completion_scope='All requested selected-ID diagnostic cases; no full benchmark',
                      wall_seconds=time.monotonic()-started)
        bundle.write_results(rows, 'partial')
        (bundle.path / 'notes.md').write_text('# Decoded physical spectral diagnostic\n\n' + config['purpose']
            + '\n\n' + config['correspondence_caveat'] + '. Slot MSE and set Chamfer are recorded separately.\n'
            + 'Classifier gradients are disabled. Decoder style remains original encoded z0.\n')
    except BaseException:
        config.update(status='failed', failure=traceback.format_exc(), diagnostic_complete=False)
        if active_observer is not None and active_directory is not None:
            (active_directory / 'failure_trajectory.json').write_text(json.dumps(active_observer.diagnostics(), indent=2, allow_nan=False)+'\n')
        bundle.write_results(rows, 'failed')
        raise
    finally:
        if lion is not None:
            config['lion_state_after'] = dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors))
        if heads:
            config['classifier_state_after_sha256'] = source.state_hash(heads['pointmae'])
        if (('lion_state_before' in config and config.get('lion_state_after') != config['lion_state_before'])
                or ('classifier_state_before_sha256' in config and config.get('classifier_state_after_sha256') != config['classifier_state_before_sha256'])):
            config.update(status='failed', frozen_state_changed=True, diagnostic_complete=False)
            bundle.write_results(rows, 'failed')
        if manifest is not None:
            try:
                source.snapshot_sources(bundle, config, manifest)
            except BaseException:
                config.update(status='failed', source_snapshot_failure=traceback.format_exc(), diagnostic_complete=False)
                bundle.write_results(rows, 'failed')
        bundle.write_config(config)


def main(argv=None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ['--worker-bundle']:
        worker(arguments[1])
        return
    config = configuration(arguments)
    name = 'decoded-' + config['arm'] + '-' + hashlib.sha256(str(config['weight']).encode()).hexdigest()[:8]
    bundle = source.RunBundle.create(Path(config['result_root']), name, config,
                                    shlex.join([sys.executable, str(Path(__file__).resolve()), *arguments]))
    (bundle.path / 'environment.txt').write_text('Python: '+sys.version+'\nPlatform: '+platform.platform()
        +'\n\n'+source.command_output(['nvidia-smi'])+'\n'+source.command_output([sys.executable, '-m', 'pip', 'freeze']))
    print('Run bundle:', bundle.path, flush=True)
    process, failure, returncode = None, None, -1
    try:
        if config.get('index_manifest'):
            copied = bundle.path / 'selection_manifest.json'
            shutil.copyfile(config['index_manifest'], copied)
            source.verify_identity(copied, config['selection_manifest_identity'])
        with (bundle.path / 'stdout.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                        '--worker-bundle', str(bundle.path)], cwd=REPO,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line); log.flush()
                print(line, end='', flush=True)
            returncode = process.wait()
    except BaseException as error:
        failure = error
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
    finally:
        final = json.loads((bundle.path / 'config.json').read_text())
        if returncode or final.get('status') == 'failed':
            final.update(status='failed', worker_returncode=returncode, diagnostic_complete=False)
            if failure is not None:
                final['parent_failure'] = repr(failure)
            bundle.write_config(final)
            bundle.mark_failed()
        source.seal_bundle(bundle)
        source.save_drive_checkpoint(bundle, config['drive_root'], 'final', final=True)
        print('Final archive:', str(bundle.path)+'.zip', source.file_identity(Path(str(bundle.path)+'.zip')), flush=True)
    if failure is not None:
        raise failure
    if returncode or final.get('status') == 'failed':
        raise SystemExit(returncode or 1)


if __name__ == '__main__':
    main()
