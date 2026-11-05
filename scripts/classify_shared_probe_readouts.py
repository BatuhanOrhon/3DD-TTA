#!/usr/bin/env python3
"""Posthoc frozen Point-MAE readouts of already sealed one-next-step clouds.

No labels, clean arrays, LION load, gradients, branch selection or PM votes.
Scientific imports come only from the pinned asset-root/source_snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

IDS = [459, 545, 1229, 1435]
CORRUPTIONS = ['gaussian', 'background', 'shear', 'lidar']
ARMS = ['zero', 'C', 'S_lat', 'S_phys', 'P100_C', 'CS_lat', 'CS_phys']
FRACTIONS = [.5, 1.]


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def verify(path: Path, expected: dict) -> None:
    if identity(path) != {key: expected.get(key) for key in ('bytes', 'sha256')}:
        raise ValueError('Artifact identity mismatch: '+str(path))


def safe_file(root: Path, relative: str) -> Path:
    root, part = Path(root).absolute(), Path(relative)
    if part.is_absolute() or '..' in part.parts:
        raise ValueError('Artifact path must remain within its root')
    path = root/part
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Artifact symlinks forbidden')
    if not path.is_file() or root.resolve() not in path.resolve().parents:
        raise ValueError('Missing/outside artifact: '+str(path))
    return path


def write_json(path: Path, value: dict) -> None:
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_arrays(path: Path, values: dict) -> dict:
    import numpy as np
    import torch
    arrays = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().copy()
        value = np.asarray(value)
        if value.dtype.kind not in 'biufU' or (value.dtype.kind == 'f' and not np.isfinite(value).all()):
            raise ValueError('Expected finite numeric/Unicode arrays: '+key)
        arrays[key] = value
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return identity(path)


def branch_key(corruption: str, batch: int, loop: int, arm: str, fraction: float) -> str:
    loops = [9, 27] if corruption == 'background' else [2, 6]
    if corruption not in CORRUPTIONS or type(batch) is not int or batch not in (0, 1) or loop not in loops or arm not in ARMS or fraction not in FRACTIONS:
        raise ValueError('Invalid frozen readout key')
    return '%s/batch%04d/loop%02d/branch_%s_%s' % (corruption, batch, loop, arm, str(fraction).replace('.', 'p'))


def validate_config(config: dict, manifest_sha: str, asset_root: Path) -> None:
    expected = dict(dataset='modelnet40_c', backbone='pointmae', capture_shared_states=True,
        arm='control', weight=0., seed=0, count=4, batch_size=2, num_classes=40,
        selected_original_ids=IDS, corruptions=CORRUPTIONS, input_layout='bnc', diagnostic_complete=True)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Capture is outside the frozen four-object readout scope')
    if config.get('status') == 'failed' or config.get('frozen_state_changed') or config.get('source_snapshot_failure'):
        raise ValueError('Failed/unfrozen capture cannot be read out')
    if config.get('asset_manifest_sha256') != manifest_sha or Path(config['asset_root']).resolve() != Path(asset_root).resolve():
        raise ValueError('Capture/assets pin differs')
    if config.get('classifier_state_before_sha256') != config.get('classifier_state_after_sha256'):
        raise ValueError('Capture classifier state changed')


def collect_jobs(capture: Path, mechanism: Path, report: dict, *, tensor_hash) -> tuple:
    """Validate sealed raw metadata and enumerate every fixed key, including gaps."""
    import numpy as np
    import torch
    artifacts = report['artifacts']
    for name, expected in artifacts.items():
        verify(safe_file(mechanism, name), expected)
    if len(report.get('cases', [])) != 8 or {(r['corruption'], r['batch']) for r in report['cases']} != {(c, b) for c in CORRUPTIONS for b in range(2)}:
        raise ValueError('Final mechanism report must account for all eight B2 cases')
    jobs, coverage = [], []
    for corruption in CORRUPTIONS:
        for batch in range(2):
            prefix = '%s/batch%04d' % (corruption, batch)
            raw_case = capture/'batches'/prefix
            shared_path = safe_file(raw_case, 'shared_states.json')
            shared = json.loads(shared_path.read_text())
            diagnostics = json.loads(safe_file(raw_case, 'diagnostics.json').read_text())
            verify(shared_path, diagnostics['shared_states'])
            ids = IDS[batch*2:batch*2+2]
            loops = [9, 27] if corruption == 'background' else [2, 6]
            if (shared.get('schema_version') != 1 or shared.get('complete') is not True or shared.get('original_ids') != ids
                    or shared.get('checkpoint_steps') != loops or shared.get('completed_checkpoints') != loops):
                raise ValueError('Incomplete/wrong raw shared capture')
            snapshots = {(entry['loop_count'], entry['stage']): entry for entry in shared['snapshots']}
            expected_keys = {(0, 'frame')} | {(loop, stage) for loop in loops for stage in ('pre_prior', 'pre_decoder', 'decoded')}
            if len(snapshots) != 7 or len(shared['snapshots']) != 7 or set(snapshots) != expected_keys:
                raise ValueError('Malformed raw capture stage coverage')
            for entry in snapshots.values():
                verify(safe_file(raw_case, entry['file']), entry['identity'])
            with np.load(safe_file(raw_case, snapshots[(0, 'frame')]['file']), allow_pickle=False) as archive:
                frame = {key: torch.from_numpy(archive[key].copy()) for key in ('normalization_center', 'normalization_scale')}
                if archive['index'].tolist() != ids:
                    raise ValueError('Frame IDs differ')
            if (frame['normalization_center'].shape != (2, 1, 3) or frame['normalization_scale'].shape != (2, 1, 1)
                    or any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in frame.values())
                    or bool((frame['normalization_scale'] <= 0).any())):
                raise ValueError('Invalid captured physical frame')
            for key, value in frame.items():
                if tensor_hash(value) != snapshots[(0, 'frame')]['tensor_sha256'][key]:
                    raise ValueError('Captured frame tensor hash differs')
            input_name = prefix+'/input.json'
            case_report = next(r for r in report['cases'] if (r['corruption'], r['batch']) == (corruption, batch))
            if input_name in artifacts:
                inputs = json.loads(safe_file(mechanism, input_name).read_text())
                if inputs.get('original_ids') != ids:
                    raise ValueError('Mechanism/capture original IDs differ')
                for name, expected in inputs['identities'].items():
                    verify(safe_file(raw_case, name), expected)
            elif case_report['status'] != 'failed_technical':
                raise ValueError('Successful mechanism case lacks its capture identity receipt')
            for loop in loops:
                direction_name = prefix+'/loop%02d/directions.json' % loop
                directions = None
                if direction_name in artifacts:
                    directions = json.loads(safe_file(mechanism, direction_name).read_text())
                    entries = directions.get('branches', [])
                    keys = [(e['name'], e['fraction']) for e in entries]
                    if len(keys) != len(set(keys)) or any(a not in ARMS or f not in FRACTIONS for a, f in keys):
                        raise ValueError('Unexpected/duplicate frozen branch receipt')
                for arm in ARMS:
                    for fraction in FRACTIONS:
                        key = branch_key(corruption, batch, loop, arm, fraction)
                        row = dict(key=key, corruption=corruption, batch=batch, loop_count=loop,
                                   arm=arm, fraction=fraction, original_ids=ids)
                        entry = next((e for e in directions.get('branches', []) if (e['name'], e['fraction']) == (arm, fraction)), None) if directions else None
                        if entry is None or entry['status'] != 'complete':
                            reason = (entry.get('error', entry['status']) if entry else
                                      directions.get('unavailable', {}).get(arm, directions.get('status')) if directions else 'No completed directions receipt')
                            row.update(status='unavailable_generation', reason=reason)
                            coverage.append(row)
                            continue
                        prediction_name, state_name = key+'_prediction.npz', key+'_state.npz'
                        if input_name not in artifacts or prediction_name not in artifacts or state_name not in artifacts:
                            raise ValueError('Completed branch lacks a sealed input/state/cloud artifact')
                        timestep = snapshots[(loop, 'pre_prior')]['timestep']
                        schedule = diagnostics['scheduler']['active_timesteps']
                        if schedule[loop-1] != timestep or directions['next_timestep'] != schedule[loop]:
                            raise ValueError('Captured/next scheduler timestep differs')
                        with np.load(safe_file(mechanism, prediction_name), allow_pickle=False) as archive:
                            cloud = torch.from_numpy(archive['decoded_xyz'].copy())
                        if cloud.shape != (2, 2048, 3) or cloud.dtype != torch.float32 or not torch.isfinite(cloud).all():
                            raise ValueError('Expected finite float32 B2 physical decoded XYZ')
                        row.update(status='available', timestep=timestep, next_timestep=directions['next_timestep'],
                                   cloud_identity=artifacts[prediction_name], state_identity=artifacts[state_name],
                                   cloud_tensor_sha256=tensor_hash(cloud), frame_identity=snapshots[(0, 'frame')]['identity'])
                        jobs.append(dict(metadata=row, cloud_path=mechanism/prediction_name, frame=frame))
                        coverage.append(row)
    return jobs, coverage


def classify_one(cloud, frame, classifier, config, *, shared, ops, fps, legacy) -> tuple:
    """Use the existing physical-frame and isolated source-classifier helpers."""
    import torch
    if any(m.training for m in classifier.modules()) or any(p.requires_grad for p in classifier.parameters()):
        raise ValueError('Readout requires a frozen all-eval classifier')
    before_rng = legacy.rng_hash()
    before_cloud = legacy.tensor_hash(cloud)
    before_buffers = legacy.model_buffer_hashes({'pointmae': classifier})
    flags = []
    def check_flags(module, args):
        row = dict(global_grad_enabled=torch.is_grad_enabled(), input_requires_grad=args[0].requires_grad,
                   all_eval=all(not m.training for m in module.modules()),
                   all_parameters_frozen=all(not p.requires_grad for p in module.parameters()))
        flags.append(row)
        if row['global_grad_enabled'] or row['input_requires_grad'] or not row['all_eval'] or not row['all_parameters_frozen']:
            raise ValueError('Classifier forward violates the frozen no-grad readout contract')
    hook = classifier.register_forward_pre_hook(check_flags)
    try:
        with legacy.isolated_probe([classifier]):
            shared.seed_inference(0)
            geometry = shared.decoded_to_geometry(cloud, config['dataset'], frame['normalization_scale'], frame['normalization_center'], ops)
            logits, audit = shared.classify_geometry(geometry, {'pointmae': classifier}, {'pointmae': config}, fps)
            output = logits['pointmae'].detach().clone()
    finally:
        hook.remove()
    if len(flags) != 1:
        raise ValueError('Expected exactly one frozen classifier forward')
    if output.shape != (2, 40) or not torch.isfinite(output).all():
        raise ValueError('Expected finite B2 x40 logits')
    guards = dict(rng_before_sha256=before_rng, rng_after_sha256=legacy.rng_hash(),
        rng_unchanged=before_rng == legacy.rng_hash(), cloud_before_sha256=before_cloud,
        cloud_after_sha256=legacy.tensor_hash(cloud),
        buffers_unchanged=before_buffers == legacy.model_buffer_hashes({'pointmae': classifier}),
        classifier_audit=audit['pointmae'], classifier_forward_flags=flags[0], common_readout_seed=0)
    if not guards['rng_unchanged'] or guards['cloud_after_sha256'] != before_cloud or not guards['buffers_unchanged']:
        raise RuntimeError('Classifier readout changed its input/RNG/model buffers')
    return output, geometry.detach().clone(), guards


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-root', 'capture-config-sha256', 'mechanism-root', 'mechanism-report-sha256',
                 'asset-root', 'asset-manifest-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    capture, mechanism, assets, output = (Path(value).absolute() for value in
        (args.capture_root, args.mechanism_root, args.asset_root, args.output))
    config_path, mechanism_path = safe_file(capture, 'config.json'), safe_file(mechanism, 'complete.json')
    if identity(config_path)['sha256'] != args.capture_config_sha256 or identity(mechanism_path)['sha256'] != args.mechanism_report_sha256:
        raise ValueError('Pinned capture/mechanism report SHA differs')
    config, report = json.loads(config_path.read_text()), json.loads(mechanism_path.read_text())
    validate_config(config, args.asset_manifest_sha256, assets)
    if (report.get('schema_version') != 1 or report.get('status') not in ('complete', 'complete_with_unavailable')
            or report.get('model_unchanged') is not True or report.get('entry_rng_restored') is not True
            or report.get('labels_and_clean_arrays_loaded') is not False or not report.get('finished_unix_seconds')
            or report.get('capture_config') != identity(config_path)):
        raise ValueError('Mechanism generation must be finalized and frozen before readout')
    manifest_path = safe_file(assets, 'manifest.json')
    if identity(manifest_path)['sha256'] != args.asset_manifest_sha256:
        raise ValueError('Pinned asset manifest SHA differs')
    manifest = json.loads(manifest_path.read_text())
    # Verify all Python sources before importing scientific code from the snapshot.
    for name, expected in manifest['files'].items():
        if name.startswith('source_snapshot/') and name.endswith('.py'):
            verify(safe_file(assets, name), expected)
    snapshot = assets/'source_snapshot'
    sys.path.insert(0, str(snapshot))
    from scripts import eval_multibackbone_source as source
    if Path(source.__file__).resolve() != snapshot/'scripts/eval_multibackbone_source.py':
        raise ValueError('Scientific source helper imported outside the pinned snapshot')
    manifest = source.read_manifest(assets, args.asset_manifest_sha256)
    from scripts import eval_multibackbone_guidance as shared
    from scripts.diagnose_decoded_spectral import apply_backend_policy
    import legacy_trajectory as legacy
    import numpy as np
    import torch
    if any(snapshot.resolve() not in Path(module.__file__).resolve().parents for module in (source, shared, legacy)):
        raise ValueError('Scientific helpers must come from pinned ASSETS/source_snapshot')
    jobs, coverage = collect_jobs(capture, mechanism, report, tensor_hash=legacy.tensor_hash)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, status='running', started_unix_seconds=time.time(),
        inputs=dict(capture_config=identity(config_path), mechanism_report=identity(mechanism_path),
                    asset_manifest=identity(manifest_path), operator_source=identity(Path(__file__))),
        labels_loaded=False, clean_arrays_loaded=False, scope='Frozen one-next-prediction Point-MAE readouts; no PM vote/trajectory benchmark',
        coverage=coverage, readouts=[], limitations=['Four previously exposed original objects; corruption/checkpoint/arm rows are dependent',
            'Custom native kernel repeats can differ despite frozen inputs/RNG; no numerical equivalence claim',
            'No branch/scale selection from logits; missing/failed generation branches remain unavailable'])
    write_json(output/'started.json', {key: value for key, value in result.items() if key not in ('coverage', 'readouts')})
    write_json(output/'coverage.json', dict(expected_clouds=224, available_clouds=len(jobs), rows=coverage))
    classifier, before_state, entry_rng = None, None, None
    try:
        if torch.__version__ != '2.1.2+cu121' or torch.version.cuda != '12.1' or not torch.cuda.is_available():
            raise RuntimeError('Readouts require the pinned Torch2.1.2+cu121 CUDA environment')
        result['backend'] = apply_backend_policy(config['backend_policy'])
        torch.cuda.get_rng_state_all()
        entry_rng = legacy._rng_state()
        result['rng_before_sha256'] = legacy.rng_hash()
        hook = source._declared_asset(assets, manifest, manifest['guidance']['native_bootstrap'])
        spec = importlib.util.spec_from_file_location('_readout_native_bootstrap', hook)
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        native = bootstrap.bootstrap(assets)
        if native.get('no_build') is not True or any(native.get(key) != config['native_bootstrap'].get(key) for key in ('bindings', 'native_audit_sha256')):
            raise ValueError('Captured/readout native bootstrap differs')
        heads, initialization_guard = shared.initialize_heads({'pointmae': config}, manifest)
        classifier = heads['pointmae']
        before_state = source.state_hash(classifier)
        if before_state != config['classifier_state_before_sha256']:
            raise ValueError('Captured/readout classifier state differs')
        import utilities_3dd_tta as ops
        from pointnet2_ops import pointnet2_utils
        for name, expected in config['native_extensions'].items():
            module = sys.modules.get(name)
            if module is None or Path(getattr(module, '__file__', '')).resolve() != Path(expected['path']).resolve():
                raise ValueError('Captured/readout native extension path differs: '+name)
            source.verify_identity(Path(expected['path']), expected)
        result.update(model_before_sha256=before_state, initialization_rng_guard=initialization_guard,
            native=native, native_extensions=config['native_extensions'],
            runtime=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()),
            native_files_before={record['path']: identity(Path(record['path']))
                for record in [*native['bindings'].values(), *config['native_extensions'].values()]},
            scientific_sources={str(Path(module.__file__).resolve().relative_to(snapshot)): identity(Path(module.__file__))
                for module in tuple(sys.modules.values()) if getattr(module, '__file__', None)
                and snapshot in Path(module.__file__).resolve().parents and Path(module.__file__).suffix in ('.py', '.so')})
        write_json(output/'runtime.json', {key: value for key, value in result.items() if key not in ('coverage', 'readouts')})
        for job in jobs:
            row = dict(job['metadata'])
            path = output/(row['key']+'.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with np.load(job['cloud_path'], allow_pickle=False) as archive:
                    cloud = torch.from_numpy(archive['decoded_xyz'].copy()).cuda()
                frame = {key: value.cuda() for key, value in job['frame'].items()}
                sampled = {}
                def fps(points, number):
                    if sampled:
                        raise ValueError('Expected exactly one Point-MAE preprocessing FPS call')
                    indices = pointnet2_utils.furthest_point_sample(points.contiguous(), number)
                    points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
                    sampled.update(indices=indices.detach().clone(), points=points.detach().clone())
                    return points
                logits, geometry, guards = classify_one(cloud, frame, classifier, config, shared=shared, ops=ops, fps=fps, legacy=legacy)
                if not sampled:
                    raise ValueError('Existing preprocessing did not execute FPS')
                arrays = dict(index=np.asarray(row['original_ids'], dtype=np.int64),
                    corruption=np.asarray([row['corruption']]*2), loop_count=np.asarray([row['loop_count']]*2, dtype=np.int64),
                    next_timestep=np.asarray([row['next_timestep']]*2, dtype=np.int64), branch=np.asarray([row['arm']]*2),
                    fraction=np.asarray([row['fraction']]*2), logits=logits, prediction=logits.argmax(-1),
                    classifier_geometry=geometry, fps_indices=sampled['indices'], classifier_points=sampled['points'])
                raw_identity = write_arrays(path, arrays)
                row.update(status='complete', readout_file=str(path.relative_to(output)), readout_identity=raw_identity,
                           guards=guards, logits_tensor_sha256=legacy.tensor_hash(logits))
            except Exception as error:
                row.update(status='failed_technical', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
            write_json(path.with_suffix('.json'), row)
            result['readouts'].append(row)
        result['status'] = 'complete' if all(row['status'] == 'complete' for row in result['readouts']) and len(jobs) == 224 else 'complete_with_unavailable'
    except Exception as error:
        result.update(status='failed_technical', error=type(error).__name__+': '+str(error), traceback=traceback.format_exc())
    finally:
        if classifier is not None and before_state is not None:
            result['model_after_sha256'] = source.state_hash(classifier)
            result['model_unchanged'] = before_state == result['model_after_sha256']
            if not result['model_unchanged']:
                result['status'] = 'failed_model_mutation'
        if entry_rng is not None:
            legacy._restore_rng(entry_rng)
            result['rng_after_sha256'] = legacy.rng_hash()
            result['rng_unchanged'] = result['rng_before_sha256'] == result['rng_after_sha256']
        try:
            if 'scientific_sources' in result:
                result['scientific_sources_after'] = {name: identity(safe_file(snapshot, name)) for name in result['scientific_sources']}
                result['sources_unchanged'] = result['scientific_sources'] == result['scientific_sources_after']
                result['native_files_after'] = {name: identity(Path(name)) for name in result['native_files_before']}
                result['native_files_unchanged'] = result['native_files_before'] == result['native_files_after']
                if not result['sources_unchanged'] or not result['native_files_unchanged']:
                    result['status'] = 'failed_source_native_mutation'
        except Exception as error:
            result.update(status='failed_source_native_verification', error=type(error).__name__+': '+str(error))
        result['finished_unix_seconds'] = time.time()
        result['artifacts'] = {str(path.relative_to(output)): identity(path) for path in sorted(output.rglob('*')) if path.is_file()}
        write_json(output/'complete.json', result)
    if result['status'].startswith('failed'):
        raise RuntimeError(result.get('error', result['status'])+'; readout evidence retained at '+str(output))
    print('GSD_SHARED_READOUTS_COMPLETE', str(output), result['status'], flush=True)


if __name__ == '__main__':
    main()
