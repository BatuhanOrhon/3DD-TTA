#!/usr/bin/env python3
"""Four-arm clean-latent diagnosis; oracle information, never deployed TTA."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_chamfer_depth as depth
from scripts import eval_shape_latent as parent

shared = parent.shared
METHOD = '3dd_clean_shape_oracle_pilot'
ARMS = ('baseline', 'clean_z_fixed_h', 'clean_z_reencode_h', 'clean_pair')


def settings(arm):
    if arm not in ARMS:
        raise ValueError('Unknown fixed oracle arm')
    return dict(arm=arm, global_prior_calls=0, clean_information='oracle diagnostic only',
                posterior_draws='common eps_z and eps_h, reference-order sampling',
                clean_preprocessing='independent normalization; same fixed LION scale/rotation; isolated preprocessing RNG',
                chamfer_target='baseline corrupted-input h XYZ fixed in every arm')


def validate_config(config):
    oracle = config.get('shape_oracle', {})
    if (config.get('method') != METHOD or oracle != settings(oracle.get('arm'))
            or 'shape_latent' in config or config.get('max_batches') != 0
            or config.get('csv_policy') != 'terminal' or config.get('lion_ema') is not False
            or not config.get('oracle_state_root')
            or Path(config['oracle_state_root']).resolve() == Path(config['paired_state_root']).resolve()):
        raise ValueError('Invalid fixed oracle payload/cache roots')
    host = dict(config, method=parent.METHOD, shape_latent=parent.shape_settings('baseline'))
    parent.validate_config(host)


def configurations(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--oracle-arm', choices=ARMS, required=True)
    parser.add_argument('--oracle-state-root', required=True)
    parser.add_argument('--oracle-technical-smoke', action='store_true')
    options, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if any(v in rest for v in ('--shape-arm', '--shape-technical-smoke')):
        raise ValueError('Oracle scope cannot combine a shape-guidance arm')
    extra = ['--shape-technical-smoke'] if options.oracle_technical_smoke else []
    config, = parent.configurations(['--shape-arm', 'baseline'] + extra + rest)
    config.pop('shape_latent')
    config.update(method=METHOD, shape_oracle=settings(options.oracle_arm),
                  oracle_state_root=str(Path(options.oracle_state_root).absolute()),
                  display_name='Clean-z oracle diagnosis: '+options.oracle_arm,
                  shared_reconstruction='oracle z/h intervention on original N8/B35; common posterior draws, diffusion noise and baseline h XYZ anchor',
                  comparison_scope='64 exposed objects, seed0 exploratory oracle diagnosis, not deployable TTA',
                  randomness_policy='common global/local posterior draws and local diffusion noise; same post-preparation RNG')
    validate_config(config)
    return [config]


def batch_ids(config, path):
    import numpy as np
    ids = config['selected_original_ids']
    matches = [ids[i:i+32] for i in range(0, len(ids), 32)
               if hashlib.sha256(np.asarray(ids[i:i+32], dtype=np.int64).tobytes()).hexdigest() == Path(path).stem]
    if len(matches) != 1:
        raise ValueError('Oracle batch does not map to exact registered original IDs')
    return matches[0]


def reconstruct_geometry(*args, oracle_config, clean_array, device='cuda', method=METHOD, **kwargs):
    import numpy as np
    import torch
    from legacy_trajectory import _rng_state_hash, isolated_probe, tensor_hash
    from shape_latent_oracle import paired_states
    raw, dataset, corruption, lion, graph, reconstruct, ops = args
    path = kwargs.get('paired_state_path')
    if path is None or method != METHOD or graph is not None:
        raise ValueError('Oracle needs the original paired host without graph guidance')
    ids = batch_ids(oracle_config, path)
    with isolated_probe([]):
        _, corrupted_center, corrupted_scale = ops.normalize(raw)
        clean_raw = torch.from_numpy(np.array(clean_array[ids]))
        normalized, clean_center, clean_scale = ops.normalize(clean_raw)
        clean = torch.from_numpy(ops.upsample_all(normalized.numpy(), 2048)).float().to(device)
        clean = ops.rotate_pointcloud(clean * 3.3885)
    metrics = {}
    def sample(**parameters):
        common = Path(oracle_config['oracle_state_root']) / dataset / corruption / Path(path).name
        states, proof = paired_states(common, parameters['x'], clean, lion.vae)
        selected = states[oracle_config['shape_oracle']['arm']]
        cached = parameters.get('prepared_state')
        if cached is not None:
            if (any(tensor_hash(cached[k]) != tensor_hash(selected[k]) for k in ('shape_latent','local_latent','noise'))
                    or _rng_state_hash(cached['rng_after_prepare']) != _rng_state_hash(selected['rng_after_prepare'])):
                raise ValueError('Selected oracle cache differs from common proof')
        parameters['prepared_state'] = selected
        reference = states['baseline']['local_latent'].view(len(raw), 2048, 4)[:, :, :3]
        parameters['chamfer_reference'] = reference
        metrics.update(proof, arm=oracle_config['shape_oracle']['arm'], original_ids=ids,
                       chamfer_reference_sha256=tensor_hash(reference),
                       clean_raw_sha256=tensor_hash(clean_raw), clean_center=clean_center.tolist(),
                       clean_scale=clean_scale.tolist(), corrupted_raw_sha256=tensor_hash(raw),
                       corrupted_center=corrupted_center.tolist(), corrupted_scale=corrupted_scale.tolist())
        return reconstruct(**parameters)
    geometry, _ = depth.reconstruct_geometry(*args[:5], sample, ops, depth_config=oracle_config,
                                             device=device, method=depth.METHOD, **kwargs)
    return geometry, dict(shape_oracle=metrics)


def worker(directories):
    import shape_latent_oracle
    import shared_trajectory
    from scripts import analyze_legacy_trajectory
    if len(directories) != 1:
        raise ValueError('Oracle pilot needs exactly one frozen Point-MAE bundle')
    config = json.loads((Path(directories[0])/'config.json').read_text())
    validate_config(config)
    clean_arrays = []
    def runtime_loader(common, manifest):
        clean_path = Path(common['dataset_root'])/'data_original.npy'
        expected = manifest['datasets']['modelnet40_c']['files'][clean_path.name]
        clean_arrays.append(shared.source.load_array(clean_path, expected, common))
        common['oracle_clean_file'] = dict(path=str(clean_path), **expected)
        original = copy.deepcopy(common)
        original['method'] = shared.ORIGINAL_METHOD
        return shared.load_guidance_runtime(original, manifest)
    def geometry(*args, **kwargs):
        return reconstruct_geometry(*args, oracle_config=config, clean_array=clean_arrays[0], **kwargs)
    shared.worker(directories, runtime_loader=runtime_loader, geometry_reconstructor=geometry)


def main(argv=None):
    shared.main(argv, configuration_factory=configurations, worker_entrypoint=__file__, worker_fn=worker)


if __name__ == '__main__':
    main()
