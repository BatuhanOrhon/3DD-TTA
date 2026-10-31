#!/usr/bin/env python3
"""Fixed five-arm ModelNet shape-latent pilot on the original N8/B35 host."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_chamfer_depth as depth
from scripts import eval_multibackbone_guidance as shared

METHOD = '3dd_bounded_shape_pilot'
ARMS = ('baseline', 'updated', 'anchor', 'proposal', 'guided3')


def shape_settings(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError('Unknown fixed shape arm')
    return dict(arm=arm, global_forward_calls={'proposal': 1, 'guided3': 3}.get(arm, 0),
                global_initial_timestep=20 if arm in ('proposal', 'guided3') else None,
                global_timesteps={'proposal': [20], 'guided3': [20, 10, 0]}.get(arm, []),
                radius_fraction=None if arm in ('baseline', 'updated') else .05,
                anchor_weight=1., global_correction_radius_fraction=.01,
                local_encoding='original h sampled conditional on initial z; no reencoding',
                noise_policy='independent generator keyed by seed and paired-cache batch filename')


def validate_config(config: dict) -> None:
    settings = config.get('shape_latent', {})
    arm = settings.get('arm')
    selection = config.get('selection_manifest', {})
    ids = config.get('selected_original_ids')
    smoke = config.get('technical_smoke')
    if (config.get('method') != METHOD or config.get('dataset') != 'modelnet40_c'
            or config.get('backbone') != 'pointmae' or config.get('batch_size') != 32
            or config.get('seed') != 0 or not config.get('chamfer_depth_study')
            or config.get('normal_reverse_steps') != 8 or config.get('background_reverse_steps') != 35
            or config.get('guidance_protocol') != dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False)
            or settings != shape_settings(arm)
            or config.get('decoder_style') != ('original' if arm == 'baseline' else 'updated')
            or config.get('scheduler_final_alpha_to_one') is not True
            or config.get('num_classes') != 40 or config.get('primary_policy') != 'probability_mean'
            or config.get('chamfer_experiment') or not config.get('trajectory_observation')
            or selection.get('stage') != ('smoke' if smoke else 'development')
            or config.get('evaluation_n_examples') != (32 if smoke else 64)
            or not isinstance(ids, list) or len(ids) != (32 if smoke else 64)
            or ids != selection.get('indices') or ids != sorted(set(ids))
            or config.get('corruptions') != (['gaussian', 'background'] if smoke else list(shared.source.CORRUPTIONS))):
        raise ValueError('Shape pilot differs from the fixed arm/host/selection protocol')
    expected = dict(normal_loop_counts=[2, 4, 6, 8], background_loop_counts=[9, 18, 27, 35])
    if any(config.get('trajectory_selection', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('Shape pilot changed its fixed probability-mean checkpoints')


def configurations(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--shape-arm', choices=ARMS, required=True)
    parser.add_argument('--shape-technical-smoke', action='store_true')
    options, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if '--technical-smoke' in rest:
        raise ValueError('Use the distinct --shape-technical-smoke scope')
    fixed_defaults = ['--normal-reverse-steps', '8', '--gamma', '.03', '--eta', '.003', '--p', '.95']
    config, = depth.configurations(fixed_defaults+rest)
    arm = options.shape_arm
    config.update(method=METHOD, technical_smoke=options.shape_technical_smoke,
                  shape_latent=shape_settings(arm), decoder_style='original' if arm == 'baseline' else 'updated',
                  display_name='Bounded global shape pilot: '+arm,
                  comparison_scope='fixed exploratory five-arm global intervention, no tuning or unseen-data claim',
                  shared_reconstruction='original N8/B35 with opt-in bounded shape adapter; same initial z/h/local noise',
                  randomness_policy='common prepared z/h/local epsilon/post-prepare RNG; independent global generator')
    if options.shape_technical_smoke:
        config['corruptions'] = ['gaussian', 'background']
    config['trajectory_selection']['style'] = ('original encoded shape_latent' if arm == 'baseline'
                                              else 'current updated style at each clean probe and final decode')
    validate_config(config)
    return [config]


def reconstruct_geometry(*args, shape_config, method=METHOD, **kwargs):
    from shape_latent_guidance import ShapeLatentGuidance
    raw, dataset, corruption, lion, graph, reconstruct, ops = args
    arm = shape_config['shape_latent']['arm']
    paired_path = kwargs.get('paired_state_path')
    if paired_path is None or method != METHOD:
        raise ValueError('Shape pilot requires its paired cache and method')
    key = (str(shape_config['seed'])+':'+Path(paired_path).name).encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63)
    adapter = None if arm == 'baseline' else ShapeLatentGuidance(arm, noise_seed=seed)
    def sample(**parameters):
        if adapter is not None:
            parameters['shape_guidance'] = adapter
        return reconstruct(**parameters)
    geometry, metrics = depth.reconstruct_geometry(raw, dataset, corruption, lion, graph, sample, ops,
                                                  depth_config=shape_config, method=depth.METHOD, **kwargs)
    metrics['shape_latent'] = (adapter.diagnostics() if adapter is not None else
                               dict(arm='baseline', global_forward_calls=0, global_backward_calls=0))
    return geometry, metrics


def worker(directories: list) -> None:
    # Import the global method before the common runner freezes executed sources.
    import shape_latent_guidance
    import shared_trajectory
    from scripts import analyze_legacy_trajectory
    if len(directories) != 1:
        raise ValueError('Shape pilot produces exactly one Point-MAE bundle')
    config = json.loads((Path(directories[0])/'config.json').read_text())
    validate_config(config)
    def runtime_loader(common, manifest):
        original = copy.deepcopy(common)
        original['method'] = shared.ORIGINAL_METHOD
        return shared.load_guidance_runtime(original, manifest)
    def geometry(*args, **kwargs):
        return reconstruct_geometry(*args, shape_config=config, **kwargs)
    shared.worker(directories, runtime_loader=runtime_loader, geometry_reconstructor=geometry)


def main(argv=None) -> None:
    shared.main(argv, configuration_factory=configurations, worker_entrypoint=__file__, worker_fn=worker)


if __name__ == '__main__':
    main()
