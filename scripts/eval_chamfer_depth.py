#!/usr/bin/env python3
"""Original-Chamfer depth study and opt-in frozen N8 transfer across source heads."""
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
TRANSFER_BACKBONES = ('pointmae', 'dgcnn', 'curvenet', 'pointnext')
TRANSFER_PROTOCOL = dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False)


def relative_checkpoints(horizon: int) -> list[int]:
    """Unique predicted-clean quarters, including the final clean step if rounded."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Expected a positive integer horizon')
    return sorted({(horizon + 3) // 4, (horizon + 1) // 2, (3 * horizon + 3) // 4})


def configurations(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root', 'label-path', 'asset-root', 'result-root', 'index-manifest',
                 'paired-state-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--pointmae-config')
    parser.add_argument('--pointmae-checkpoint')
    parser.add_argument('--transfer-study', action='store_true')
    parser.add_argument('--dataset', choices=('modelnet40_c', 'scanobjectnn_c'), default='modelnet40_c')
    parser.add_argument('--backbones', nargs='+', choices=TRANSFER_BACKBONES)
    parser.add_argument('--asset-manifest-sha256')
    parser.add_argument('--drive-root')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--normal-reverse-steps', type=int, choices=range(1, 11))
    parser.add_argument('--p', type=float)
    parser.add_argument('--gamma', type=float, help='local latent update rate')
    parser.add_argument('--eta', type=float, help='conditioning-style update rate')
    parser.add_argument('--technical-smoke', action='store_true')
    parser.add_argument('--corruptions', nargs='+', default=['all'])
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args(argv)
    defaults = dict(normal_reverse_steps=8 if args.transfer_study else 5,
                    **(TRANSFER_PROTOCOL if args.transfer_study else shared.ORIGINAL_PROTOCOL))
    for name in ('normal_reverse_steps', 'gamma', 'eta', 'p'):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    args.backbones = args.backbones or (list(TRANSFER_BACKBONES) if args.transfer_study else ['pointmae'])
    if not args.transfer_study and (args.dataset != 'modelnet40_c' or args.backbones != ['pointmae']):
        raise ValueError('Other datasets/heads require explicit --transfer-study')
    if args.transfer_study and (args.normal_reverse_steps != 8 or
            (args.p, args.gamma, args.eta) != (.95, .03, .003)):
        raise ValueError('Transfer study requires frozen N8 / eta=.003 / gamma=.03 / p=.95')
    if args.technical_smoke and args.transfer_study:
        if args.corruptions != ['gaussian', 'background'] or args.seed != 0:
            raise ValueError('Transfer technical smoke requires gaussian/background and seed0')
    elif args.technical_smoke:
        if (args.corruptions != ['gaussian', 'background'] or args.normal_reverse_steps not in (1, 5, 10)
                or args.seed != 0 or (args.p, args.gamma, args.eta) != (.95, .01, .01)):
            raise ValueError('Technical smoke requires gaussian/background, N1/5/10, seed0 and default parameters')
    elif args.corruptions != ['all']:
        raise ValueError('Research depth jobs require the fixed all15 scope')
    for name in ('gamma', 'eta', 'p'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == 'p' and not 0 < value <= 1):
            raise ValueError('Invalid depth-study parameter: ' + name)
    original_args = ['--dataset', args.dataset, '--method', 'original3dd', '--backbones', *args.backbones]
    original_args.extend(['--corruptions', *args.corruptions])
    for name in ('dataset_root', 'label_path', 'asset_root', 'result_root', 'index_manifest',
                 'pointmae_config', 'pointmae_checkpoint', 'seed', 'asset_manifest_sha256', 'drive_root'):
        value = getattr(args, name)
        if value is not None:
            original_args.extend(['--' + name.replace('_', '-'), str(value)])
    if args.preview:
        original_args.append('--preview')
    configs = shared.configurations(original_args)
    config = configs[0]
    if args.technical_smoke and (config['evaluation_n_examples'] != 32 or config['selection_manifest']['stage'] != 'smoke'):
        raise ValueError('Technical smoke requires exactly32 explicitly selected smoke IDs')
    normal, background = relative_checkpoints(args.normal_reverse_steps), relative_checkpoints(35)
    for config in configs:
        config.update(method=METHOD, chamfer_depth_study=True, technical_smoke=args.technical_smoke, trajectory_observation=True,
                      normal_reverse_steps=args.normal_reverse_steps, background_reverse_steps=35,
                      guidance_protocol=dict(gamma=args.gamma, eta=args.eta, p=args.p, total=100, scd_normalize=False),
                      guidance_rate_mapping=dict(gamma='local_latent', eta='conditioning_style'),
                      paired_state_root=str(Path(args.paired_state_root).absolute()),
                      display_name=('Frozen N8 Chamfer transfer + ' + config['backbone'] if args.transfer_study
                                    else 'Original Chamfer depth study + frozen Point-MAE'),
                      shared_reconstruction='original tta.tta_reconstruct with paired preparation and isolated original-style clean probes',
                      primary_policy='probability_mean', csv_policy='terminal',
                      comparison_scope=('fixed selected setting transfer; independent per-head results; no model ensemble or model-specific optimum'
                                        if args.transfer_study else 'bounded normal-depth and p/local/style-rate search; Background remains35; not a global optimum or paper reproduction'),
                      randomness_policy='pair input/z0/h0/epsilon and restore post-preparation RNG; initial noisy h_t changes with depth',
                      timing_policy=('CSV runtime=shared reconstruction/' + str(len(configs)) if args.transfer_study
                                     else 'CSV runtime=whole reconstruction') + ' including probes/cache IO + terminal FPS/classifier; batch post-inference policy/local/Drive shard IO and worker wall including corruption checkpoint IO recorded separately; final parent sealing outside worker',
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
        if args.transfer_study:
            config['transfer_study'] = True
            validate_transfer_config(config)
    return configs


def validate_transfer_config(config: dict) -> None:
    """Reject modified worker payloads before loading any assets or GPU code."""
    dataset = config.get('dataset')
    classes, examples = (15, 581) if dataset == 'scanobjectnn_c' else (40, 2468)
    expected = dict(method=METHOD, chamfer_depth_study=True, transfer_study=True,
                    normal_reverse_steps=8, background_reverse_steps=35, batch_size=32,
                    decoder_style='original', scheduler_final_alpha_to_one=True, lion_ema=False,
                    graph=None, trajectory_observation=True, max_batches=0,
                    primary_policy='probability_mean', csv_policy='terminal',
                    num_classes=classes, n_examples=examples, guidance_protocol=TRANSFER_PROTOCOL)
    if (dataset not in ('modelnet40_c', 'scanobjectnn_c')
            or config.get('backbone') not in TRANSFER_BACKBONES
            or any(config.get(key) != value for key, value in expected.items())
            or any(config.get(key) is not None for key in
                   ('chamfer_experiment', 'component_arm', 'spectral_loss_kind', 'semantic_guidance'))):
        raise ValueError('Invalid frozen transfer candidate or variant combination')
    selection = config.get('trajectory_selection', {})
    if (selection.get('normal_loop_counts') != [2, 4, 6, 8]
            or selection.get('background_loop_counts') != [9, 18, 27, 35]
            or selection.get('normal_candidate_names') != ['clean_02', 'clean_04', 'clean_06', 'terminal']
            or selection.get('background_candidate_names') != ['clean_09', 'clean_18', 'clean_27', 'terminal']
            or selection.get('temperature') != 1. or selection.get('labels_used') is not False
            or selection.get('early_stopping') is not False or not config.get('paired_state_root')
            or not config.get('selected_original_ids')):
        raise ValueError('Transfer requires explicit IDs and frozen probability-mean checkpoints')
    manifest = config.get('selection_manifest', {})
    ids = config['selected_original_ids']
    if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < examples for i in ids)
            or ids != sorted(set(ids)) or config.get('evaluation_n_examples') != len(ids)
            or manifest.get('indices') != ids or manifest.get('dataset') != dataset
            or manifest.get('n_examples') != examples):
        raise ValueError('Transfer original IDs must agree with the explicit selection manifest')
    if config.get('technical_smoke'):
        if (config.get('corruptions') != ['gaussian', 'background'] or config.get('seed') != 0
                or config.get('evaluation_n_examples') != 32
                or config.get('selection_manifest', {}).get('stage') != 'smoke'):
            raise ValueError('Transfer technical smoke requires exactly32 smoke IDs and seed0')
    elif (config.get('corruptions') != list(shared.source.CORRUPTIONS)
            or manifest.get('stage') not in ('development', 'heldout')):
        raise ValueError('Research transfer requires the fixed all15 scope')


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
    allowed_dataset = dataset == 'modelnet40_c' or (dataset == 'scanobjectnn_c' and depth_config.get('transfer_study'))
    if not allowed_dataset or method != METHOD or graph is not None or observer_factory is None or paired_state_path is None:
        raise ValueError('Depth study requires ModelNet or opt-in Scan transfer, original host, observer and paired states')
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
    configs = [json.loads((Path(path) / 'config.json').read_text()) for path in directories]
    if not configs:
        raise ValueError('Depth study requires worker bundles')
    config = configs[0]
    if config.get('transfer_study'):
        names = [c.get('backbone') for c in configs]
        if len(set(names)) != len(names):
            raise ValueError('Transfer worker requires unique heads')
        common_keys = ('dataset', 'num_classes', 'n_examples', 'seed', 'corruptions', 'technical_smoke',
                       'guidance_protocol', 'trajectory_selection', 'paired_state_root', 'selected_original_ids',
                       'selection_manifest_identity', 'asset_root', 'asset_manifest_sha256', 'dataset_root', 'label_path')
        for current in configs:
            validate_transfer_config(current)
            if any(current.get(key) != config.get(key) for key in common_keys):
                raise ValueError('Transfer worker bundles must share data, candidate and reconstruction')
    elif (len(configs) != 1 or config.get('method') != METHOD or not config.get('chamfer_depth_study')
            or config.get('backbone') != 'pointmae'):
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
