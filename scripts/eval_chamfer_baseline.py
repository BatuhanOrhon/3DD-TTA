#!/usr/bin/env python3
"""Operational proximal N8+PM baseline, with explicit historical Chamfer profile."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_chamfer_depth as depth

BASELINE_ID = depth.PROXIMAL_BASELINE_ID
BASELINE_METHOD = depth.PROXIMAL_BASELINE_METHOD
PROXIMAL_GUIDANCE = dict(arm='cheap', tau=.1, retained_fraction=.95,
                         style_updates=False, denoiser_backward=False)
EVIDENCE_SCOPE = 'ModelNet40-C/Point-MAE/all2468/s5/seed0/B128; other heads/datasets not empirically confirmed'
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def legacy_configurations(arguments) -> list[dict]:
    configs = depth.configurations(['--transfer-study', *arguments])
    for config in configs:
        config.update(method=depth.BASELINE_METHOD, operational_baseline=depth.BASELINE_ID,
                      csv_policy='probability_mean',
                      display_name='Historical N8/35 Chamfer + probability mean + ' + config['backbone'],
                      prediction_contract='prediction and CSV use probability_mean; terminal_prediction and logits retain terminal diagnostics',
                      baseline_evidence='knowledge/chamfer_depth_transfer_results_20261031.md')
        depth.validate_transfer_config(config)
    return configs


def configurations(argv=None) -> list[dict]:
    """Promoted profile is fixed; the historical policy is an explicit option."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--baseline-profile', choices=(BASELINE_ID, depth.BASELINE_ID), default=BASELINE_ID)
    parser.add_argument('--batch-size', type=int, choices=BATCH_SIZES)
    selected, arguments = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    configs = legacy_configurations(arguments)
    if selected.baseline_profile == depth.BASELINE_ID:
        if selected.batch_size not in (None, 32):
            raise ValueError('Historical baseline batch size is locked to32')
        return configs
    batch = selected.batch_size or (128 if len(configs) == 1 and
        configs[0]['dataset'] == 'modelnet40_c' and configs[0]['backbone'] == 'pointmae' else 32)
    for config in configs:
        selection = config['trajectory_selection']
        selection['background_loop_counts'] = list(selection['normal_loop_counts'])
        selection['background_candidate_names'] = list(selection['normal_candidate_names'])
        config.update(method=BASELINE_METHOD, operational_baseline=BASELINE_ID,
                      background_reverse_steps=8, batch_size=batch,
                      proximal_guidance=dict(PROXIMAL_GUIDANCE), backend_policy='controlled',
                      cudnn_policy='controlled: benchmarkFalse/deterministicTrue/TF32off; custom native determinism unproven',
                      shared_reconstruction='Frozen cheap latent Chamfer solve/DDIM bridge; original encoded decoder style; no denoiser backward/style update',
                      display_name='Thesis baseline proximal N8 + probability mean + ' + config['backbone'],
                      baseline_evidence='knowledge/lion_guidance_uniform_horizon_results_20261108.md',
                      baseline_evidence_scope=EVIDENCE_SCOPE,
                      guidance_rate_semantics='gamma/eta are compatibility reference values; cheap uses tau=.1, no gradient rate or style update',
                      baseline_host_validation='CPU contracts only; audit/cache/RNG schedule and float64 PM differ from the speed runner; GPU timing and near-tie parity unverified',
                      comparison_scope='User-promoted common-horizon operational policy; one-seed exposed-data evidence, not universal validation')
        validate_config(config)
    return configs


def validate_config(config: dict) -> None:
    expected = dict(method=BASELINE_METHOD, operational_baseline=BASELINE_ID,
                    normal_reverse_steps=8, background_reverse_steps=8,
                    proximal_guidance=PROXIMAL_GUIDANCE, backend_policy='controlled',
                    csv_policy='probability_mean', primary_policy='probability_mean',
                    baseline_evidence_scope=EVIDENCE_SCOPE)
    selection = config.get('trajectory_selection', {})
    if (any(config.get(k) != v for k, v in expected.items())
            or type(config.get('batch_size')) is not int or config['batch_size'] not in BATCH_SIZES
            or selection.get('background_loop_counts') != [2, 4, 6, 8]
            or selection.get('background_candidate_names') != ['clean_02', 'clean_04', 'clean_06', 'terminal']):
        raise ValueError('Invalid frozen proximal baseline profile')
    # Reuse the existing data/head/ID/decoder/PM guards after checking every changed field.
    historical = copy.deepcopy(config)
    historical.update(method=depth.BASELINE_METHOD, operational_baseline=depth.BASELINE_ID,
                      background_reverse_steps=35, batch_size=32)
    historical['trajectory_selection'].update(background_loop_counts=[9,18,27,35],
        background_candidate_names=['clean_09','clean_18','clean_27','terminal'])
    depth.validate_transfer_config(historical)


def proximal_reconstruct(x, lion, steps_back_local, gamma, eta, p, total=100, *,
                          prepared_state=None, state_observer=None, trajectory_observer=None,
                          scheduler_observer=None, scd_normalize=False):
    """Adapt the already measured cheap kernel to the shared inference/artifact host."""
    import torch
    from legacy_trajectory import _rng_state, _restore_rng
    from lion_guidance_pilot import sample
    if ((steps_back_local, gamma, eta, p, total, scd_normalize) != (8,.03,.003,.95,100,False)
            or trajectory_observer is None or scheduler_observer is None):
        raise ValueError('Proximal baseline requires frozen N8 policy and trajectory/scheduler observers')
    with torch.no_grad():
        if prepared_state is None:
            encoded = lion.vae.encode(x)
            z, h = (encoded[2][i][0].unsqueeze(2).unsqueeze(3) for i in (0, 1))
            lion.vae.global2style(z)
            state = dict(shape_latent=z, local_latent=h, noise=torch.randn_like(h))
            state['rng_after_prepare'] = _rng_state()
        else:
            state = dict(prepared_state)
        for key, width in (('shape_latent',128),('local_latent',8192),('noise',8192)):
            value = state[key]
            if (value.shape != (len(x),width,1,1) or value.dtype != x.dtype or value.device != x.device
                    or not torch.isfinite(value).all()):
                raise ValueError('Invalid paired proximal latent/noise tensor')
    _restore_rng(state['rng_after_prepare'])
    if state_observer is not None:
        state_observer({key:state[key] for key in ('shape_latent','local_latent','noise')})
    trajectory_observer.initial(input_points=x, **{key:state[key] for key in ('shape_latent','local_latent','noise')})
    def readout(h, z, loop):
        if loop == 8:
            trajectory_observer.final_state(h, z, 0)
        else:
            trajectory_observer.clean(h, z, loop-1, (8-loop)*10)
    snapshots, records = sample(lion, state, 8, 'cheap', tau=.1, callback=readout,
                                scheduler_observer=scheduler_observer)
    if len(snapshots) != 4 or len(records) != 8 or any(r['full_backward'] for r in records):
        raise RuntimeError('Proximal baseline changed its fixed readout/backward schedule')
    with torch.enable_grad():
        final = snapshots[-1].detach().requires_grad_(True)
        return lion.vae.decoder(None, beta=None, context=final.squeeze(3).squeeze(2),
                                style=state['shape_latent'].detach().squeeze(3).squeeze(2))


def worker(directories: list) -> None:
    configs = [json.loads((Path(path)/'config.json').read_text()) for path in directories]
    if not configs:
        raise ValueError('Baseline worker requires bundles')
    if all(c.get('operational_baseline') == depth.BASELINE_ID for c in configs):
        depth.worker(directories)
        return
    for config in configs:
        validate_config(config)
    common = configs[0]
    if len({c['backbone'] for c in configs}) != len(configs):
        raise ValueError('Baseline requires unique source heads')
    keys = ('dataset','num_classes','n_examples','seed','corruptions','technical_smoke',
            'guidance_protocol','trajectory_selection','paired_state_root','selected_original_ids',
            'selection_manifest_identity','asset_root','asset_manifest_sha256','dataset_root','label_path',
            'batch_size','proximal_guidance','backend_policy')
    if any(any(c.get(k) != common.get(k) for k in keys) for c in configs):
        raise ValueError('Baseline worker bundles must share data and reconstruction policy')
    # Import before the shared worker freezes the executed-source map.
    import lion_guidance_pilot
    from scripts import analyze_legacy_trajectory
    from scripts.diagnose_decoded_spectral import apply_backend_policy
    def load(config, manifest):
        return depth.shared.load_guidance_runtime(dict(config, method=depth.shared.ORIGINAL_METHOD), manifest)
    def reconstruct(*args, **kwargs):
        values=list(args);values[5]=proximal_reconstruct
        return depth.reconstruct_geometry(*values, depth_config=common, **kwargs)
    depth.shared.worker(directories, runtime_loader=load, geometry_reconstructor=reconstruct,
                        backend_policy=lambda:apply_backend_policy('controlled'))


def main(argv=None) -> None:
    depth.shared.main(argv, configuration_factory=configurations,
                      worker_entrypoint=__file__, worker_fn=worker)


if __name__ == '__main__':
    main()
