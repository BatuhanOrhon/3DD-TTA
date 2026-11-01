#!/usr/bin/env python3
"""Fixed N8/B35 baseline followed by one full posterior reencode and local N3."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_shape_latent as parent
from scripts import eval_chamfer_depth as depth

shared = parent.shared
METHOD = '3dd_lion_reencode_n3_pilot'


def settings():
    return dict(second_reverse_steps=3, global_diffusion=False,
        second_chamfer_target='new h1 XYZ', decoder_style='each pass own initial z',
        posterior_noise='same explicit eps_z and eps_h', local_noise='same Gaussian, new depth alpha',
        bridge_frame='decoder LION Bx2048x3, no normalize/FPS/rotation',
        combined_weighting='equal8 candidates, equal4 per pass', clean_information_used=False)


def validate_config(config):
    if (config.get('method') != METHOD or config.get('reencode') != settings()
            or config.get('csv_policy') != 'probability_mean'
            or not config.get('first_trajectory_selection') or 'shape_latent' in config):
        raise ValueError('Invalid fixed reencode payload')
    original = dict(config, method=parent.METHOD, shape_latent=parent.shape_settings('baseline'),
                    trajectory_selection=config['first_trajectory_selection'], csv_policy='terminal')
    parent.validate_config(original)
    expected = copy.deepcopy(config['first_trajectory_selection'])
    for scope in ('normal', 'background'):
        expected[scope + '_loop_counts'] = [1, 2, 3, 3]
        expected[scope + '_candidate_names'] = ['clean_01', 'clean_02', 'clean_03', 'terminal']
    if config['trajectory_selection'] != expected:
        raise ValueError('Second-pass readout must use clean1/2/3 plus distinct terminal3')


def configurations(argv=None):
    config, = parent.configurations(['--shape-arm', 'baseline'] + list(sys.argv[1:] if argv is None else argv))
    config.pop('shape_latent')
    config.update(method=METHOD, reencode=settings(), csv_policy='probability_mean',
        first_trajectory_selection=copy.deepcopy(config['trajectory_selection']),
        display_name='N8/B35 → full posterior reencode → local N3',
        comparison_scope='64 exposed development objects, seed0; second PM vs same-run first PM',
        shared_reconstruction='two local trajectories, one global+local reencode, global prior off',
        timing_policy='full two-pass reconstruction including bridge/cycle/probes; first and second dispatch recorded separately')
    for scope in ('normal', 'background'):
        config['trajectory_selection'][scope + '_loop_counts'] = [1, 2, 3, 3]
        config['trajectory_selection'][scope + '_candidate_names'] = ['clean_01', 'clean_02', 'clean_03', 'terminal']
    validate_config(config)
    return [config]


def reconstruct_geometry(*args, reencode_config, method=METHOD, **kwargs):
    from legacy_trajectory import tensor_hash
    from shape_latent_refresh import prepare_reference, save_proof
    from lion_reencode import reencode
    if (method != METHOD or args[4] is not None or kwargs.get('paired_state_path') is None
            or not reencode_config.get('executed_source_sha256')):
        raise ValueError('Reencode requires original host, explicit cache and executed source')
    reconstruct = args[5]
    proof_path = Path(kwargs['paired_state_path']).with_suffix('.reencode.npz')
    metrics = {}
    def sample(**parameters):
        if parameters.get('prepared_state') is not None or proof_path.exists():
            raise ValueError('Use a new paired cache; never silently replay a two-pass attempt')
        observer = parameters['trajectory_observer']
        base, initial = prepare_reference(parameters['x'], parameters['lion'].vae)
        first_parameters = dict(parameters, prepared_state=base, trajectory_observer=observer.first)
        first = reconstruct(**first_parameters)
        observer.first.terminal(observer.first.classify_decoded(first))
        selected, proof = reencode(first, base, initial, parameters['lion'].vae)
        observer.cycle(observer.first.classify_decoded(proof['cycle_points']))
        binding = dict(executed_source_sha256=reencode_config['executed_source_sha256'],
            asset_manifest_sha256=reencode_config['asset_manifest_sha256'],
            input_sha256=tensor_hash(parameters['x']))
        save_proof(proof_path, proof, binding)
        stage2_scheduler = {}
        second_parameters = dict(parameters, x=first.detach(), steps_back_local=3,
            prepared_state=selected, state_observer=None, trajectory_observer=observer.second,
            scheduler_observer=lambda scheduler: stage2_scheduler.update(depth.scheduler_record(scheduler, 3)))
        second_parameters.pop('chamfer_reference', None)
        second = reconstruct(**second_parameters)
        metrics.update(second_reverse_steps=3, first_reverse_steps=parameters['steps_back_local'],
            second_scheduler=stage2_scheduler, proof=dict(path=str(proof_path), **shared.source.file_identity(proof_path)),
            first_decoded_sha256=tensor_hash(first), second_decoded_sha256=tensor_hash(second),
            second_chamfer_reference_sha256=tensor_hash(selected['local_latent'].view(len(first), 2048, 4)[:, :, :3]),
            first_z_sha256=tensor_hash(base['shape_latent']), second_z_sha256=tensor_hash(selected['shape_latent']))
        return second
    geometry, _ = depth.reconstruct_geometry(*args[:5], sample, args[6],
        depth_config=reencode_config, method=depth.METHOD, **kwargs)
    return geometry, dict(reencode=metrics)


def worker(directories):
    import json
    from lion_reencode import TwoPassObserver
    # The common worker freezes source identities before its first policy readout.
    import scripts.analyze_legacy_trajectory
    if len(directories) != 1:
        raise ValueError('This pilot requires one Point-MAE bundle')
    config = json.loads((Path(directories[0]) / 'config.json').read_text())
    validate_config(config)
    actual = []
    def runtime_loader(common, manifest):
        actual.append(common)
        return shared.load_guidance_runtime(dict(common, method=shared.ORIGINAL_METHOD), manifest)
    def geometry(*args, **kwargs):
        return reconstruct_geometry(*args, reencode_config=actual[0], **kwargs)
    def observer_provider(**options):
        options['checkpoint_steps'] = depth.relative_checkpoints(options['horizon'])
        return TwoPassObserver(**options)
    shared.worker(directories, runtime_loader=runtime_loader,
                  geometry_reconstructor=geometry, observer_provider=observer_provider)


def main(argv=None):
    shared.main(argv, configuration_factory=configurations, worker_entrypoint=__file__, worker_fn=worker)


if __name__ == '__main__':
    main()
