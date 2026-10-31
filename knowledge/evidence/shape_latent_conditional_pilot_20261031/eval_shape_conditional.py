#!/usr/bin/env python3
"""Bounded exploratory conditional guidance, encoder versus SCD initialization."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_shape_proximal as parent
shared, depth = parent.shared, parent.depth
METHOD = '3dd_shape_conditional_pilot'
ARMS = ('firstpass', 'encoder_none', 'scd_none', 'encoder_prior3', 'scd_prior3',
        'encoder_guided3', 'scd_guided3')


def settings(arm):
    if arm not in ARMS:
        raise ValueError('Unknown conditional arm')
    return dict(arm=arm, start='encoder' if arm == 'firstpass' else arm.split('_')[0],
                mode='none' if arm == 'firstpass' else arm.split('_')[1],
                local_times=[99, 299, 599, 899], global_times=[20, 10, 0],
                step_fraction=.02, clean_information_used=False,
                target='shared detached firstpass terminal h',
                scale='per-object .02 times original encoder z norm',
                local_restart='original h0/noise/RNG/anchor; selected own-z decoder')


def validate(config):
    arm = config.get('conditional', {}).get('arm')
    if (config.get('method') != METHOD or config.get('conditional') != settings(arm)
            or config.get('seed') != 0 or config.get('batch_size') != 32
            or config.get('evaluation_n_examples') != 32
            or config.get('decoder_style') != 'original'
            or config.get('guidance_protocol') != dict(gamma=.03, eta=.003, p=.95, total=100, scd_normalize=False)
            or config.get('corruptions') != (['gaussian', 'background'] if config.get('technical_smoke') else parent.CORRUPTIONS)
            or config.get('normal_reverse_steps') != 8 or config.get('background_reverse_steps') != 35
            or config.get('backbone') != 'pointmae' or config.get('dataset') != 'modelnet40_c'
            or config.get('lion_ema') is not False or config.get('max_batches') != 0
            or config.get('primary_policy') != 'probability_mean'
            or config.get('trajectory_selection', {}).get('normal_loop_counts') != [2, 4, 6, 8]
            or config.get('trajectory_selection', {}).get('background_loop_counts') != [9, 18, 27, 35]
            or not config.get('conditional_state_root') or 'shape_proximal' in config
            or config.get('oracle_target_root')):
        raise ValueError('Conditional pilot differs from the fixed scope')


def configurations(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--conditional-arm', choices=ARMS, required=True)
    parser.add_argument('--conditional-state-root', required=True)
    parser.add_argument('--conditional-smoke', action='store_true')
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    extra = ['--global-technical-smoke'] if args.conditional_smoke else []
    config, = parent.configurations(rest + ['--global-arm', 'original', '--global-state-root',
                                           str(Path(args.conditional_state_root)/'unused')] + extra)
    config.pop('shape_proximal')
    config.update(method=METHOD, conditional=settings(args.conditional_arm),
                  conditional_state_root=str(Path(args.conditional_state_root).absolute()),
                  display_name='Exploratory conditional z guidance: '+args.conditional_arm,
                  comparison_scope='32exposed/four corruptions/seed0; two starts x three modes; shared firstpass',
                  timing_policy='Replay includes proposal and second local pass; shared firstpass cost separate; not whole deployment timing')
    validate(config)
    return [config]


def geometry(raw, dataset, corruption, lion, graph, reconstruct, ops, *, config,
             device='cuda', method=METHOD, scheduler_observer=None, observer_factory=None,
             protocol=None, paired_state_path=None):
    import torch
    import numpy as np
    from legacy_trajectory import tensor_hash, rng_hash
    from shape_latent_proximal import load_reference
    from shape_latent_refresh import load_proof, save_proof
    from shape_latent_conditional import propose
    if method != METHOD or graph is not None or paired_state_path is None or observer_factory is None:
        raise ValueError('Expected paired original host and isolated observer')
    root = Path(config['asset_root'])
    manifest_path = root/'references/manifest.json'
    shared.source.verify_identity(manifest_path, config['asset_manifest']['files']['references/manifest.json'])
    inputs = json.loads(manifest_path.read_text())
    ids = config['selected_original_ids']
    name = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()+'.npz'
    if (inputs['original_ids'] != ids or Path(paired_state_path).name != name
            or inputs['lion_state'] != config['lion_state_before']
            or inputs['classifier_state'] != config['classifier_state_before_sha256']
            or inputs['dataset_provenance'] != config['dataset_provenance']):
        raise ValueError('Imported identity/model/data differs')
    entry = inputs['cases'][corruption]
    base, meta = load_reference(root/entry['common']['path'], entry['common'], device)
    shared.source.verify_identity(root/entry['proof']['path'], entry['proof'])
    imported = load_proof(root/entry['proof']['path'], device, entry['proof_binding'])
    x = imported['prepared_input']
    if tensor_hash(x) != meta['input_sha256'] or len(x) != 32:
        raise ValueError('Imported input mismatch')
    arm = config['conditional']['arm']
    state_root = Path(config['conditional_state_root'])
    reference_path = state_root/'firstpass'/corruption/name
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    binding = dict(executed_source_sha256=config['executed_source_sha256'],
                   common=entry['common'], input_sha256=tensor_hash(x))
    selected = dict(base)
    output = state_root/'global'/arm/corruption/name
    proof = None
    if arm != 'firstpass':
        if not (state_root/'firstpass.complete').is_file():
            raise ValueError('Require completed firstpass worker before all proposal arms')
        reference = load_proof(reference_path, device, binding)
        start = base['shape_latent'] if arm.startswith('encoder_') else reference['scd_z']
        seed = int.from_bytes(hashlib.sha256((name+':conditional').encode()).digest()[:8], 'big') % (2**63-3)
        before = rng_hash()
        z, proof = propose(start, base['shape_latent'], reference['h_star'], lion.priors[0],
                           lion.priors[1], parent.make_scheduler(device),
                           config['conditional']['mode'], seed)
        if rng_hash() != before:
            raise RuntimeError('Conditional proposal changed shared RNG')
        selected['shape_latent'] = z
        output.parent.mkdir(parents=True, exist_ok=True)
        save_proof(output, proof, dict(binding, arm=arm, firstpass=shared.source.file_identity(reference_path)))
    cached, record = shared.paired_preparation(paired_state_path, x)
    if cached is not None:
        raise ValueError('Never overwrite/reuse an output attempt')
    _, center, scale = ops.normalize(raw)
    seen = []
    steps = 35 if corruption == 'background' else 8
    def observe(scheduler):
        entry = depth.scheduler_record(scheduler, steps)
        seen.append(entry)
        if scheduler_observer is not None:
            scheduler_observer(entry)
    parameters = dict(x=x, lion=lion, steps_back_local=steps,
                      scheduler_observer=observe, prepared_state=selected, state_observer=record,
                      trajectory_observer=observer_factory(center, scale),
                      chamfer_reference=base['local_latent'].view(32, 2048, 4)[:, :, :3], **protocol)
    if arm == 'firstpass':
        trajectory = reconstruct(return_trajectory=True, **parameters)
        save_proof(reference_path, dict(h_star=trajectory.final_local_latent,
                                       scd_z=trajectory.updated_style,
                                       encoder_z=trajectory.original_style), binding)
        with torch.no_grad():
            decoded = lion.vae.decoder(None, beta=None, context=trajectory.final_local_latent.flatten(1),
                                       style=selected['shape_latent'].flatten(1))
    else:
        decoded = reconstruct(**parameters)
    if len(seen) != 1:
        raise RuntimeError('Local scheduler record missing')
    result = shared.decoded_to_geometry(decoded, dataset, scale, center, ops)
    if result.shape != (32, 2048, 3) or not torch.isfinite(result).all():
        raise RuntimeError('Invalid final geometry')
    metrics = dict(arm=arm, selected_z_sha256=tensor_hash(selected['shape_latent']),
                   input_sha256=tensor_hash(x), common=entry['common'],
                   h0_sha256=tensor_hash(base['local_latent']), noise_sha256=tensor_hash(base['noise']),
                   firstpass=dict(path=str(reference_path), **shared.source.file_identity(reference_path)),
                   global_proof=None if proof is None else dict(path=str(output), **shared.source.file_identity(output)))
    return result.detach(), dict(conditional=metrics)


def worker(directories):
    import shape_latent_conditional
    import shape_latent_proximal
    import shape_latent_refresh
    import shared_trajectory
    from scripts import analyze_legacy_trajectory
    if len(directories) != 1:
        raise ValueError('One Point-MAE bundle only')
    config = json.loads((Path(directories[0])/'config.json').read_text())
    validate(config)
    actual = []
    def loader(common, manifest):
        actual.append(common)
        original = copy.deepcopy(common)
        original['method'] = shared.ORIGINAL_METHOD
        return shared.load_guidance_runtime(original, manifest)
    def reconstruct(*args, **kwargs):
        return geometry(*args, config=actual[0], **kwargs)
    shared.worker(directories, runtime_loader=loader, geometry_reconstructor=reconstruct)


if __name__ == '__main__':
    shared.main(configuration_factory=configurations, worker_entrypoint=__file__, worker_fn=worker)
