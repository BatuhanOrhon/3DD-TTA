#!/usr/bin/env python3
"""Single-job original-Chamfer depth study with isolated probability-mean probes."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_multibackbone_guidance as shared

METHOD = '3dd_chamfer_depth_study'


def relative_checkpoints(horizon: int) -> list[int]:
    """Unique predicted-clean quarters, including the final clean step if rounded."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Expected a positive integer horizon')
    return sorted({(horizon + 3) // 4, (horizon + 1) // 2, (3 * horizon + 3) // 4})


def configurations(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root', 'label-path', 'asset-root', 'result-root', 'index-manifest',
                 'paired-state-root', 'pointmae-config', 'pointmae-checkpoint'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--asset-manifest-sha256')
    parser.add_argument('--drive-root')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--normal-reverse-steps', type=int, choices=range(1, 11), default=5)
    parser.add_argument('--p', type=float, default=.95)
    parser.add_argument('--gamma', type=float, default=.01, help='local latent update rate')
    parser.add_argument('--eta', type=float, default=.01, help='conditioning-style update rate')
    parser.add_argument('--technical-smoke', action='store_true')
    parser.add_argument('--corruptions', nargs='+', default=['all'])
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args(argv)
    if args.technical_smoke:
        if (args.corruptions != ['gaussian', 'background'] or args.normal_reverse_steps not in (1, 5, 10)
                or args.seed != 0 or (args.p, args.gamma, args.eta) != (.95, .01, .01)):
            raise ValueError('Technical smoke requires gaussian/background, N1/5/10, seed0 and default parameters')
    elif args.corruptions != ['all']:
        raise ValueError('Research depth jobs require the fixed all15 scope')
    for name in ('gamma', 'eta', 'p'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == 'p' and not 0 < value <= 1):
            raise ValueError('Invalid depth-study parameter: ' + name)
    original_args = ['--dataset', 'modelnet40_c', '--method', 'original3dd', '--backbones', 'pointmae']
    original_args.extend(['--corruptions', *args.corruptions])
    for name in ('dataset_root', 'label_path', 'asset_root', 'result_root', 'index_manifest',
                 'pointmae_config', 'pointmae_checkpoint', 'seed', 'asset_manifest_sha256', 'drive_root'):
        value = getattr(args, name)
        if value is not None:
            original_args.extend(['--' + name.replace('_', '-'), str(value)])
    if args.preview:
        original_args.append('--preview')
    config, = shared.configurations(original_args)
    if args.technical_smoke and (config['evaluation_n_examples'] != 32 or config['selection_manifest']['stage'] != 'smoke'):
        raise ValueError('Technical smoke requires exactly32 explicitly selected smoke IDs')
    normal, background = relative_checkpoints(args.normal_reverse_steps), relative_checkpoints(35)
    config.update(method=METHOD, chamfer_depth_study=True, technical_smoke=args.technical_smoke, trajectory_observation=True,
                  normal_reverse_steps=args.normal_reverse_steps, background_reverse_steps=35,
                  guidance_protocol=dict(gamma=args.gamma, eta=args.eta, p=args.p, total=100, scd_normalize=False),
                  guidance_rate_mapping=dict(gamma='local_latent', eta='conditioning_style'),
                  paired_state_root=str(Path(args.paired_state_root).absolute()),
                  display_name='Original Chamfer depth study + frozen Point-MAE',
                  shared_reconstruction='original tta.tta_reconstruct with paired preparation and isolated original-style clean probes',
                  primary_policy='probability_mean', csv_policy='terminal',
                  comparison_scope='bounded normal-depth and p/local/style-rate search; Background remains35; not a global optimum or paper reproduction',
                  randomness_policy='pair input/z0/h0/epsilon and restore post-preparation RNG; initial noisy h_t changes with depth',
                  timing_policy='CSV runtime=whole reconstruction including probes/cache IO + terminal FPS/classifier; batch post-inference policy/local/Drive shard IO and worker wall including corruption checkpoint IO recorded separately; final parent sealing outside worker',
                  trajectory_selection=dict(normal_loop_counts=normal + [args.normal_reverse_steps],
                                            background_loop_counts=background + [35],
                                            normal_candidate_names=['clean_%02d' % step for step in normal] + ['terminal'],
                                            background_candidate_names=['clean_%02d' % step for step in background] + ['terminal'],
                                            policies=['terminal', 'minimum_entropy', 'probability_mean'],
                                            temperature=1., entropy='class-count normalized',
                                            candidate_weighting='equal probability per unique clean checkpoint and terminal',
                                            style='original encoded shape_latent for every clean and terminal decode',
                                            tie_policy='terminal first, then earliest checkpoint',
                                            labels_used=False, early_stopping=False))
    return [config]


def scheduler_record(scheduler, steps: int) -> dict:
    """Audit the distinct study horizon without relaxing original3dd's5/35 guard."""
    if steps not in (*range(1, 11), 35):
        raise ValueError('Depth study requires1..10 or Background35')
    record = shared.original_scheduler_record(scheduler, 35)
    record['active_timesteps'] = record['timesteps'][-steps:]
    record.update(reverse_steps=steps, initial_timestep=record['active_timesteps'][0],
                  initial_alpha_cumprod=float(scheduler.alphas_cumprod[record['active_timesteps'][0]]))
    if not math.isfinite(record['initial_alpha_cumprod']) or not 0 < record['initial_alpha_cumprod'] <= 1:
        raise ValueError('Invalid actual initial scheduler alpha')
    return record


def reconstruct_geometry(raw, dataset, corruption, lion, graph, reconstruct, ops, *,
                         depth_config, device='cuda', method=METHOD, scheduler_observer=None,
                         observer_factory=None, protocol=None, paired_state_path=None):
    import torch
    if dataset != 'modelnet40_c' or method != METHOD or graph is not None or observer_factory is None or paired_state_path is None:
        raise ValueError('Depth study requires ModelNet, original host, observer and paired states')
    normalized, center, scale = ops.normalize(raw)
    prepared = torch.from_numpy(ops.upsample_all(normalized.numpy(), 2048)).float().to(device)
    prepared *= 3.3885
    prepared = ops.rotate_pointcloud(prepared)
    if not torch.isfinite(prepared).all():
        raise ValueError('Nonfinite prepared depth-study input')
    steps = 35 if corruption == 'background' else depth_config['normal_reverse_steps']
    seen = []
    def observe(scheduler):
        record = scheduler_record(scheduler, steps)
        seen.append(record)
        if scheduler_observer is not None:
            scheduler_observer(record)
    state, recorder = shared.paired_preparation(paired_state_path, prepared)
    decoded = reconstruct(x=prepared, lion=lion, steps_back_local=steps, scheduler_observer=observe,
                          prepared_state=state, state_observer=recorder,
                          trajectory_observer=observer_factory(center, scale), **protocol)
    if len(seen) != 1:
        raise RuntimeError('Depth study must report its actual scheduler exactly once')
    geometry = shared.decoded_to_geometry(decoded, dataset, scale, center, ops)
    if geometry.shape != (len(raw), 2048, 3) or not torch.isfinite(geometry).all():
        raise ValueError('Depth study requires finite decoded B x2048 x3 geometry')
    return geometry.detach(), {}


def worker(directories: list) -> None:
    # Import probe/policy bytes before the runner freezes its executed-source map.
    import shared_trajectory
    from scripts import analyze_legacy_trajectory
    if len(directories) != 1:
        raise ValueError('Depth study produces exactly one frozen Point-MAE bundle')
    config = json.loads((Path(directories[0]) / 'config.json').read_text())
    if config.get('method') != METHOD or not config.get('chamfer_depth_study') or config.get('backbone') != 'pointmae':
        raise ValueError('Invalid depth-study worker bundle')
    def runtime_loader(common, manifest):
        original = copy.deepcopy(common)
        original['method'] = shared.ORIGINAL_METHOD
        return shared.load_guidance_runtime(original, manifest)
    def reconstruct(*args, **kwargs):
        return reconstruct_geometry(*args, depth_config=config, **kwargs)
    shared.worker(directories, runtime_loader=runtime_loader, geometry_reconstructor=reconstruct)


def main(argv=None) -> None:
    shared.main(argv, configuration_factory=configurations, worker_entrypoint=__file__, worker_fn=worker)


if __name__ == '__main__':
    main()
