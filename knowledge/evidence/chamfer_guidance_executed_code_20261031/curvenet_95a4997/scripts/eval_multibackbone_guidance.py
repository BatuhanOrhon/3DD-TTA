#!/usr/bin/env python3
"""Shared reconstruction with isolated frozen heads and locked method policies."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_multibackbone_source as source
from spectral_guidance_losses import SPECTRAL_LOSS_KINDS
from chamfer_calibration_report import validate_calibration_report

METHOD = "gsd_inspired_latent_spectral_guidance"
ORIGINAL_METHOD = "3dd_original"
BACKBONES = ("curvenet", "pointnext")
LION_SHA256 = "807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2"
LION_CONFIG_SHA256 = "9d4a826f35fe9b3053210105eb4857811c8bfd53cd8c6a998f8ea7953e975816"
PROTOCOL = dict(M_max=800, beta=2.0, gamma=0.01, eta=0.01, p=0.95, total=100,
                loss_weights=dict(spectral=1.17, invariant=0.0, chamfer=1.0),
                use_static_style=False, dynamic_graph=False, graph_update_interval=3)
ORIGINAL_PROTOCOL = dict(gamma=0.01, eta=0.01, p=0.95, total=100, scd_normalize=False)
COMPONENT_WEIGHTS = dict(combined=(1.0, 1.17), chamfer=(1.0, 0.0), spectral=(0.0, 1.17), unguided=(0.0, 0.0))


def fixed_calibration_reference(path, expected_sha256, coefficient_key, *, guidance_head='pointmae'):
    """Bind the immutable two-dataset gradient report before any GPU execution."""
    path = Path(path).resolve()
    identity = source.file_identity(path)
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256 or '') or identity['sha256'] != expected_sha256:
        raise ValueError('Calibration report SHA-256 mismatch')
    report = json.loads(path.read_text())
    if guidance_head == 'curvenet':
        from scan_curvenet_entropy_report import validate_report
        validate_report(report)
    else:
        validate_calibration_report(report)
    proofs = report.get('calibration_bundles', {})
    if set(proofs) != ({'scanobjectnn_c'} if guidance_head == 'curvenet' else {'modelnet40_c', 'scanobjectnn_c'}):
        raise ValueError('Calibration proof requires both datasets')
    for proof in proofs.values():
        if (not proof.get('executed_source_files') or not proof.get('native_extensions')
                or not re.fullmatch('[0-9a-f]{64}', proof.get('asset_manifest_sha256', ''))
                or not re.fullmatch('[0-9a-f]{64}', proof.get('config_identity', {}).get('sha256', ''))):
            raise ValueError('Calibration source/native/asset identities missing')
        for mapping in (proof['executed_source_files'], proof['native_extensions']):
            for identity_value in mapping.values():
                if (not isinstance(identity_value, dict) or type(identity_value.get('bytes')) is not int
                        or identity_value['bytes'] < 0 or not re.fullmatch('[0-9a-f]{64}', identity_value.get('sha256', ''))):
                    raise ValueError('Malformed calibrated source/native identity')
    scalar = report.get('coefficients', {}).get(coefficient_key)
    if not isinstance(scalar, (int, float)) or isinstance(scalar, bool) or not math.isfinite(scalar) or scalar <= 0:
        raise ValueError('Invalid fixed calibrated coefficient')
    result = dict(path=str(path), identity=identity, purpose=report['purpose'], schema_version=1,
                coefficient_key=coefficient_key, resolved_scalar=float(scalar), formula=report['rule'],
                calibration_bundles=proofs)
    if guidance_head == 'curvenet':
        result['teacher'] = report['teacher']
    return result


def configurations(argv=None) -> list:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if sum(value == '--chamfer-experiment' or value.startswith('--chamfer-experiment=') for value in arguments) > 1:
        raise ValueError('Select exactly one independent Chamfer-only experiment')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("modelnet40_c", "scanobjectnn_c"), required=True)
    for name in ("dataset-root", "label-path", "asset-root", "result-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--asset-manifest-sha256")
    parser.add_argument("--drive-root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=("gsd", "original3dd"), default="gsd")
    parser.add_argument("--backbones", nargs="+", choices=(*BACKBONES, "dgcnn", "pointmae"), default=list(BACKBONES))
    parser.add_argument("--trajectory-observe", action="store_true")
    parser.add_argument("--component-arm", choices=tuple(COMPONENT_WEIGHTS))
    parser.add_argument('--chamfer-experiment', choices=('sparse', 'semantic', 'weighted'), default=argparse.SUPPRESS)
    parser.add_argument('--sparse-interval', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--density-k', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--semantic-weight', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--semantic-kind', choices=('kl', 'marginal_entropy'), default=argparse.SUPPRESS)
    parser.add_argument('--semantic-guidance-head', choices=('pointmae', 'curvenet'), default=argparse.SUPPRESS)
    parser.add_argument('--calibration-report', type=Path, default=argparse.SUPPRESS)
    parser.add_argument('--calibration-report-sha256', default=argparse.SUPPRESS)
    parser.add_argument("--spectral-loss-kind", choices=SPECTRAL_LOSS_KINDS, default=argparse.SUPPRESS)
    for name in ("eta", "gamma", "p", "spectral-weight", "chamfer-weight"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--index-manifest", type=Path)
    parser.add_argument("--paired-state-root", type=Path)
    parser.add_argument("--pointmae-config")
    parser.add_argument("--pointmae-checkpoint")
    parser.add_argument("--batch-size", type=int, choices=(32, 70))
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--corruptions", nargs="+", default=["all"])
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(arguments)
    experiment = getattr(args, 'chamfer_experiment', None)
    experimental_options = {'sparse_interval': 'sparse', 'density_k': 'weighted',
                            'semantic_weight': 'semantic', 'semantic_kind': 'semantic', 'semantic_guidance_head': 'semantic'}
    semantic_head = getattr(args, 'semantic_guidance_head', 'pointmae')
    for option, variant in experimental_options.items():
        if hasattr(args, option) and experiment != variant:
            raise ValueError(option + ' requires its separate Chamfer-only experiment')
    experiment_protocol = None
    if experiment:
        if args.method != 'gsd' or args.component_arm not in (None, 'chamfer'):
            raise ValueError('Experiments require the separate Chamfer-only GSD host')
        if getattr(args, 'spectral_loss_kind', None) is not None or args.spectral_weight not in (None, 0.) or args.chamfer_weight not in (None, 1.):
            raise ValueError('Chamfer-only experiments lock C=1, S=0 and disable spectral losses')
        args.component_arm, args.trajectory_observe = 'chamfer', True
        experiment_protocol = dict(variant=experiment)
        if experiment == 'sparse':
            interval = getattr(args, 'sparse_interval', 3)
            if interval < 1:
                raise ValueError('Sparse interval must be positive')
            experiment_protocol['interval'] = interval
        elif experiment == 'weighted':
            density_k = getattr(args, 'density_k', 10)
            if density_k < 1:
                raise ValueError('Density k must be positive')
            experiment_protocol['density_k'] = density_k
        else:
            if semantic_head not in args.backbones:
                raise ValueError('Semantic experiment requires selected frozen Point-MAE guidance head')
            if semantic_head == 'curvenet' and (args.dataset != 'scanobjectnn_c' or getattr(args, 'semantic_kind', 'kl') != 'marginal_entropy'):
                raise ValueError('CurveNet guidance is restricted to Scan marginal entropy')
            if not args.preview and not hasattr(args, 'semantic_weight'):
                raise ValueError('Actual semantic execution requires explicit --semantic-weight (0 is sham)')
            weight = getattr(args, 'semantic_weight', 1.)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('Semantic weight must be finite and nonnegative')
            experiment_protocol['semantic_weight'] = weight
    calibration = None
    if hasattr(args, 'calibration_report') or hasattr(args, 'calibration_report_sha256'):
        if experiment not in ('semantic', 'weighted') or not (hasattr(args, 'calibration_report') and hasattr(args, 'calibration_report_sha256')):
            raise ValueError('Both calibration report and SHA are required for semantic/weighted experiments')
        key = ('semantic_h_curvenet' if semantic_head == 'curvenet' else 'weighted_scale' if experiment == 'weighted' else
               'semantic_kl' if getattr(args, 'semantic_kind', 'kl') == 'kl' else 'semantic_h')
        options = {'guidance_head': 'curvenet'} if semantic_head == 'curvenet' else {}
        calibration = fixed_calibration_reference(args.calibration_report, args.calibration_report_sha256, key, **options)
        scalar = calibration['resolved_scalar']
        if experiment == 'semantic' and not math.isclose(experiment_protocol['semantic_weight'], scalar, rel_tol=1e-12):
            raise ValueError('Semantic weight differs from fixed calibration report')
        if experiment == 'weighted' and any(value is not None and not math.isclose(value, .01 * scalar, rel_tol=1e-12)
                                             for value in (args.eta, args.gamma)):
            raise ValueError('Weighted eta/gamma must equal baseline .01 times calibrated scale')
    if semantic_head == 'curvenet' and not args.preview and experiment_protocol['semantic_weight'] > 0 and calibration is None:
        raise ValueError('Positive CurveNet H execution requires its frozen label-free calibration report')
    if args.seed < 0 or args.max_batches < 0:
        raise ValueError("Seed and max-batches must be nonnegative")
    original = args.method == "original3dd"
    spectral_kind = getattr(args, 'spectral_loss_kind', None)
    if spectral_kind is not None and (original or args.component_arm not in ('combined', 'spectral')):
        raise ValueError('Spectral loss kinds require an explicit active-spectral component study')
    if args.trajectory_observe and original:
        raise ValueError("Trajectory observation is restricted to GSD-inspired guidance")
    if args.component_arm and (original or not args.trajectory_observe):
        raise ValueError("Component arms require GSD-inspired trajectory observation")
    overrides = {name: getattr(args, name) for name in ('eta', 'gamma', 'p', 'spectral_weight', 'chamfer_weight')}
    if any(value is not None for value in overrides.values()) and not args.component_arm:
        raise ValueError("Parameter overrides require an explicit component arm")
    for name, value in overrides.items():
        if value is not None and (not math.isfinite(value) or value < 0 or (name == 'p' and not 0 < value <= 1)):
            raise ValueError("Invalid component parameter: " + name)
    if args.paired_state_root and not args.component_arm:
        raise ValueError("Paired states require an explicit component study arm")
    if "pointmae" in args.backbones and not (original or args.trajectory_observe):
        raise ValueError("Point-MAE requires original3DD or GSD --trajectory-observe")
    if "pointmae" in args.backbones and not (args.pointmae_config and args.pointmae_checkpoint):
        raise ValueError("Point-MAE requires --pointmae-config and --pointmae-checkpoint")
    batch_size = 32 if original else 70
    if args.batch_size is not None and args.batch_size != batch_size:
        raise ValueError("Method batch size is locked to " + str(batch_size))
    if len(set(args.backbones)) != len(args.backbones):
        raise ValueError("Select unique backbone names")
    if not args.preview and not re.fullmatch(r"[0-9a-f]{64}", args.asset_manifest_sha256 or ""):
        raise ValueError("Actual guidance requires --asset-manifest-sha256")
    result = []
    for backbone in args.backbones:
        base = source.parse_args(["--preview", "--backbone", backbone, "--dataset", args.dataset,
                                  "--dataset-root", args.dataset_root, "--label-path", args.label_path,
                                  "--asset-root", args.asset_root, "--result-root", args.result_root,
                                  "--corruptions", *args.corruptions])
        config = source.resolve_config(base)
        if args.index_manifest:
            selection = json.loads(args.index_manifest.read_text())
            ids = selection.get('indices')
            if (selection.get('dataset') != args.dataset or selection.get('n_examples') != config['n_examples']
                    or selection.get('stage') not in ('smoke', 'development', 'heldout') or not isinstance(ids, list)
                    or not ids or any(type(i) is not int or not 0 <= i < config['n_examples'] for i in ids)
                    or ids != sorted(set(ids))):
                raise ValueError('Invalid sorted original-ID selection manifest')
            config.update(selected_original_ids=ids, evaluation_n_examples=len(ids), selection_manifest=selection,
                          selection_manifest_identity=source.file_identity(args.index_manifest))
        if backbone == "pointmae":
            config.update(pointmae_config=args.pointmae_config, pointmae_checkpoint=args.pointmae_checkpoint,
                          checkpoint=args.pointmae_checkpoint)
        if "original" in config["corruptions"]:
            raise ValueError("Guidance uses the predeclared corrupted scope only")
        config.update(vars(args), method=METHOD, batch_size=batch_size, lion_loaded=True,
                      corruptions=config["corruptions"], guidance_protocol=copy.deepcopy(PROTOCOL),
                      display_name="GSD-inspired latent spectral guidance + " + backbone,
                      source_classifier_preprocessing=config["preprocessing"],
                      guidance_postprocessing="rotate back; Scan: divide3.3885 and inverse input bbox; ModelNet: bbox normalize",
                      shared_reconstruction="classifier-independent tta_gsd_reconstruct; observer off; decoded2048 retained",
                      decoder_style="updated", graph=dict(k=10, delta=0.1, gamma=0.6, use_4d_gft=False, isolate_policy="legacy"),
                      scheduler_final_alpha_to_one=False, lion_ema=False,
                      cudnn_policy="benchmark=False and deterministic=False, deterministic algorithms off; legacy fresh-process defaults; source-only benchmark=True",
                      seed_boundary="after all models/native/graph initialization; each head preprocess+forward isolated",
                      timing_policy="CSV runtime=shared reconstruction/2 + own preprocessing/classification; shared seconds count once across2heads",
                      memory_policy="peak whole shared process including LION and both heads; repeated perhead, never sum")
        if len(args.backbones) != 2:
            config["timing_policy"] = ("CSV runtime=shared reconstruction/" + str(len(args.backbones))
                                       + " + own preprocessing/classification; shared seconds count once across"
                                       + str(len(args.backbones)) + "heads")
            config["memory_policy"] = "peak whole shared process including LION and selected heads; repeated perhead, never sum"
        if original:
            config.update(method=ORIGINAL_METHOD, guidance_protocol=ORIGINAL_PROTOCOL,
                          display_name="Original 3DD-TTA (repository eval/raw) + " + backbone,
                          shared_reconstruction="classifier-independent unchanged tta.tta_reconstruct; scheduler audited; decoded2048 retained",
                          decoder_style="original", graph=None, scheduler_final_alpha_to_one=True,
                          normal_reverse_steps=5, background_reverse_steps=35,
                          cudnn_policy="benchmark=True and deterministic=False, deterministic algorithms off; validated repository original eval/raw policy",
                          comparison_scope="whole reconstruction policy comparison; not isolated spectral attribution or paper reproduction")
        if backbone == "dgcnn":
            config["backbone_caveat"] = ("Published alternative, not paper checkpoint identity; ScanObjectNN DGCNN has weaker clean/source accuracy; "
                                          "exploratory comparison only to this fixed checkpoint's own source row")
        if args.trajectory_observe:
            config.update(trajectory_observation=True,
                          shared_reconstruction="classifier-independent tta_gsd_reconstruct; isolated shared clean-output probes; decoded2048 retained",
                          trajectory_selection=dict(normal_loop_counts=[3, 5, 8, 10], background_loop_counts=[8, 15, 23, 30],
                                                    normal_candidate_names=["clean_03", "clean_05", "clean_08", "terminal"],
                                                    background_candidate_names=["clean_08", "clean_15", "clean_23", "terminal"],
                                                    policies=["terminal", "minimum_entropy", "probability_mean"], temperature=1.0,
                                                    entropy="class-count normalized", tie_policy="terminal first, then earliest checkpoint",
                                                    primary_comparator="same-run terminal", labels_used=False, early_stopping=False),
                          timing_policy="CSV runtime=shared reconstruction including all probe work/" + str(len(args.backbones))
                                        + " + own terminal preprocessing/classification; shared seconds count once",
                          randomness_policy="seed-controlled shared diffusion; observer and head probes restore RNG; no common-draw claim across runs")
        if args.component_arm:
            protocol = config['guidance_protocol']
            chamfer, spectral = COMPONENT_WEIGHTS[args.component_arm]
            protocol.update(skip_inactive_components=True)
            protocol['loss_weights'].update(chamfer=chamfer, spectral=spectral)
            for key in ('eta', 'gamma', 'p'):
                if overrides[key] is not None:
                    protocol[key] = overrides[key]
            for key in ('spectral', 'chamfer'):
                value = overrides[key + '_weight']
                if value is not None:
                    if (protocol['loss_weights'][key] == 0) != (value == 0):
                        raise ValueError('Weight override must preserve active components of the named arm')
                    protocol['loss_weights'][key] = value
            config.update(display_name='GSD-host component study ' + args.component_arm + ' + ' + backbone,
                          component_study=True, component_parameters_modified=any(v is not None for v in overrides.values()) or spectral_kind is not None,
                          comparison_scope='same GSD host; matched states/noise only when verified from batch identities')
            if not (protocol['loss_weights']['spectral'] or protocol['loss_weights']['invariant']):
                config['graph'] = None
            if spectral_kind is not None:
                protocol['spectral_loss_kind'] = spectral_kind
            if args.paired_state_root:
                config['randomness_policy'] = 'immutable shared z0/h0/noise cache; restore recorded post-preparation RNG; validate prepared-input and actual latent/noise hashes'
                config['reconstruction_host'] = 'GSD paired-state component study; reference arm encodes, replay arms reuse exact encoded states/noise'
        if experiment:
            config['guidance_protocol']['chamfer_experiment'] = dict(experiment_protocol)
            config.update(chamfer_experiment=dict(experiment_protocol), graph=None,
                          display_name='Chamfer-only ' + experiment + ' guidance + ' + backbone,
                          comparison_scope='independent intervention versus unchanged Chamfer-only probability mean')
            config['calibration_status'] = 'calibrated' if calibration else 'uncalibrated explicit experimental coefficient'
            if calibration:
                config['calibration_reference'] = copy.deepcopy(calibration)
                if experiment == 'weighted':
                    config['guidance_protocol'].update(eta=.01 * calibration['resolved_scalar'], gamma=.01 * calibration['resolved_scalar'])
            if experiment == 'semantic':
                config['semantic_guidance'] = dict(kind=getattr(args, 'semantic_kind', 'kl'), head=semantic_head,
                                                 weight=experiment_protocol['semantic_weight'],
                                                 normal_loop_counts=[5, 8], background_loop_counts=[15, 23],
                                                 reference='cached detached corrupted raw X; existing Point-MAE preprocessing and native FPS start index 0; no alternate normalization',
                                                 reference_forwards_per_batch=1, reference_votes=False, batch_marginal=False,
                                                 loss_reduction='sum',
                                                 loss_definition='symmetric KL between decoded baseline and detached X' if getattr(args, 'semantic_kind', 'kl') == 'kl' else 'per-object H((p_X+p_decoded_baseline)/2), detached X')
                config['shared_reconstruction'] = 'Point-MAE-head-dependent Chamfer-only reconstruction; baseline probes reused; cached corrupted-X FPS reference for guidance only'
                if semantic_head == 'curvenet':
                    config['semantic_guidance']['reference'] = 'cached detached corrupted raw X; unchanged CurveNet centroid/L-infinity normalization and native FPS1024'
                    config['shared_reconstruction'] = 'CurveNet-head-dependent Chamfer-only reconstruction; baseline probes reused; cached corrupted-X normalized FPS reference for guidance only'
                    config['display_name'] = 'Chamfer-only marginal entropy with CurveNet guidance + ' + backbone
        for key in ("dataset_root", "label_path", "asset_root", "result_root", "drive_root", "paired_state_root", "index_manifest", "calibration_report"):
            if config.get(key):
                config[key] = str(Path(config[key]).absolute())
        result.append(config)
    if experiment == 'semantic' and semantic_head == 'curvenet':
        teacher = next(config for config in result if config['backbone'] == 'curvenet')
        for config in result:
            config['semantic_guidance'].update(teacher_checkpoint=teacher['checkpoint'], teacher_preprocessing=teacher['source_classifier_preprocessing'],
                                               teacher_input_layout=teacher['input_layout'])
    return result


def seed_inference(seed: int) -> str:
    import numpy as np
    import torch
    from legacy_trajectory import rng_hash
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return rng_hash()


def initialize_heads(configs: dict, manifest: dict, loader=None):
    from legacy_trajectory import isolated_probe, rng_hash
    before = rng_hash()
    heads = {}
    for name, config in configs.items():
        with isolated_probe([]):
            heads[name] = (loader or source.load_classifier)(config, manifest)
    after = rng_hash()
    if before != after:
        raise RuntimeError("Classifier initialization changed reconstruction RNG")
    return heads, dict(before_sha256=before, after_sha256=after, rng_unchanged=True)


def original_scheduler_record(scheduler, steps: int) -> dict:
    """Assert the actual original DDIM defaults before any latent inference."""
    config = dict(scheduler.config)
    expected = dict(beta_end=0.02, beta_start=0.0001, beta_schedule="linear", clip_sample=False,
                    num_train_timesteps=1000, prediction_type="epsilon", set_alpha_to_one=True)
    if any(config.get(key) != value for key, value in expected.items()) or config.get("steps_offset", 0) != 0:
        raise ValueError("Original DDIM scheduler config differs from locked policy")
    timesteps = scheduler.timesteps.detach().cpu().tolist()
    if timesteps != list(range(990, -1, -10)):
        raise ValueError("Original DDIM timesteps differ from locked 100-step policy")
    final_alpha = float(scheduler.final_alpha_cumprod)
    if final_alpha != 1.0 or steps not in (5, 35):
        raise ValueError("Original DDIM final alpha/horizon differs from locked policy")
    return dict(config=config, timesteps=timesteps, active_timesteps=timesteps[-steps:], final_alpha_cumprod=final_alpha)


def decoded_to_geometry(decoded, dataset: str, scale, center, ops):
    """Use the same dataset frame for predicted-clean probes and terminal output."""
    # The legacy rotation negates its input in place; probes must retain decoder bytes.
    geometry = ops.rotateback_pointcloud(decoded.clone())
    if dataset == "scanobjectnn_c":
        return ops.unnormalize_data(geometry / 3.3885, scale, center)
    return ops.normalize(geometry)[0]


def reconstruct_geometry(raw, dataset: str, corruption: str, lion, graph, reconstruct, ops, *, device="cuda", method=METHOD, scheduler_observer=None, observer_factory=None, protocol=None, paired_state_path=None):
    """Exact existing dataset pre/postprocessing, before any terminal head FPS."""
    import torch
    normalized, center, scale = ops.normalize(raw)
    prepared = torch.from_numpy(ops.upsample_all(normalized.numpy(), 2048)).float().to(device)
    prepared *= 3.3885
    prepared = ops.rotate_pointcloud(prepared)
    if method == ORIGINAL_METHOD:
        steps = 35 if corruption == "background" else 5
        observed = []
        def observe(scheduler):
            record = original_scheduler_record(scheduler, steps)
            observed.append(record)
            if scheduler_observer is not None:
                scheduler_observer(record)
        decoded = reconstruct(x=prepared, lion=lion, steps_back_local=steps,
                              scheduler_observer=observe, **ORIGINAL_PROTOCOL)
        if len(observed) != 1:
            raise RuntimeError("Original reconstruction must report its actual scheduler once")
        metrics = {}
    else:
        observation = {} if observer_factory is None else {"trajectory_observer": observer_factory(center, scale)}
        if paired_state_path is not None:
            state, recorder = paired_preparation(paired_state_path, prepared)
            observation.update(prepared_state=state, state_observer=recorder)
        decoded, metrics = reconstruct(x=prepared, lion=lion, graph_spectral_module=graph,
                                       steps_back_local=30 if corruption == "background" else 10, **observation, **(PROTOCOL if protocol is None else protocol))
    geometry = decoded_to_geometry(decoded, dataset, scale, center, ops)
    if geometry.shape != (len(raw), 2048, 3) or not torch.isfinite(geometry).all():
        raise ValueError("Reconstruction must preserve finite decoded Bx2048x3 geometry")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite guidance metrics")
    return geometry.detach(), metrics


def paired_preparation(path: Path, prepared):
    """Persist exact encoded states/noise and RNG without pickled objects."""
    import numpy as np
    import torch
    from legacy_trajectory import _rng_state, _rng_state_hash, rng_hash, tensor_hash
    path = Path(path)
    expected_input, expected_rng = tensor_hash(prepared), rng_hash()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Paired state symlinks are forbidden')
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata'].item()))
            if metadata.get('schema_version') != 1 or metadata.get('input_sha256') != expected_input or metadata.get('rng_before_sha256') != expected_rng:
                raise ValueError('Paired prepared-input/RNG identity mismatch')
            state = {key: torch.from_numpy(archive[key].copy()).to(prepared.device) for key in ('shape_latent', 'local_latent', 'noise')}
            py = metadata['python_rng']
            rng = ((py[0], tuple(py[1]), py[2]),
                   (metadata['numpy_rng'][0], archive['numpy_rng_keys'].copy(), *metadata['numpy_rng'][1:]),
                   torch.from_numpy(archive['torch_rng_cpu'].copy()),
                   [torch.from_numpy(archive['torch_rng_cuda_%d' % i].copy()) for i in range(metadata['cuda_devices'])])
            if _rng_state_hash(rng) != metadata['rng_after_sha256']:
                raise ValueError('Paired cached post-preparation RNG hash mismatch')
            for key, value in state.items():
                if tensor_hash(value) != metadata['tensor_hashes'].get(key):
                    raise ValueError('Paired cached tensor hash mismatch: ' + key)
            state['rng_after_prepare'] = rng
        return state, None
    path.parent.mkdir(parents=True, exist_ok=True)
    def record(state):
        rng = _rng_state()
        py, numpy_rng, cpu, cuda = rng
        metadata = dict(schema_version=1, input_sha256=expected_input, rng_before_sha256=expected_rng,
                        rng_after_sha256=_rng_state_hash(rng), python_rng=py,
                        numpy_rng=[numpy_rng[0], *numpy_rng[2:]], cuda_devices=len(cuda),
                        tensor_hashes={key: tensor_hash(value) for key, value in state.items()})
        arrays = {key: value.detach().cpu().numpy() for key, value in state.items()}
        arrays.update(metadata=np.asarray(json.dumps(metadata)), numpy_rng_keys=numpy_rng[1], torch_rng_cpu=cpu.numpy())
        arrays.update({'torch_rng_cuda_%d' % i: value.cpu().numpy() for i, value in enumerate(cuda)})
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
    return None, record


def classify_geometry(geometry, heads: dict, configs: dict, fps):
    from legacy_trajectory import isolated_probe, rng_hash, tensor_hash
    logits, diagnostics = {}, {}
    geometry_sha = tensor_hash(geometry)
    for name, classifier in heads.items():
        before = rng_hash()
        with isolated_probe([classifier]):
            points, features = source.preprocess(geometry, configs[name], fps)
            logits[name] = classifier(points, features=features).detach()
        after = rng_hash()
        if before != after or tensor_hash(geometry) != geometry_sha:
            raise RuntimeError("Isolated classifier changed RNG or shared geometry: " + name)
        diagnostics[name] = dict(rng_before_sha256=before, rng_after_sha256=after,
                                 rng_unchanged=True, geometry_sha256=geometry_sha)
    return logits, diagnostics


def classify_semantic_geometry(geometry, heads: dict, configs: dict, fps, guidance_head='pointmae'):
    """Preserve baseline preprocessing and retain only guidance-head input grads."""
    from semantic_trajectory import gradient_probe
    from legacy_trajectory import tensor_hash
    logits = {}
    geometry_hash = tensor_hash(geometry)
    for name, classifier in heads.items():
        if name == guidance_head:
            with gradient_probe([classifier]):
                points, features = source.preprocess(geometry, configs[name], fps)
                logits[name] = classifier(points, features=features)
        else:
            logits.update(classify_geometry(geometry, {name: classifier}, configs, fps)[0])
    if tensor_hash(geometry) != geometry_hash:
        raise RuntimeError('Semantic preprocessing mutated shared geometry')
    return logits


def semantic_reference(geometry, heads: dict, configs: dict, fps, guidance_head='pointmae'):
    """One frozen deterministic-FPS corrupted-X forward, shared by both probes."""
    from legacy_trajectory import tensor_hash
    started = time.perf_counter()
    logits, diagnostics = classify_geometry(geometry, {guidance_head: heads[guidance_head]}, configs, fps)
    reference = logits[guidance_head].detach()
    # Hashing synchronizes the returned logits before recording elapsed time.
    record = dict(geometry_sha256=tensor_hash(geometry), logits_sha256=tensor_hash(reference),
                  forward_calls=1, source='corrupted raw X, unchanged source preprocessing/FPS',
                  rng_guard=diagnostics[guidance_head], reference_dispatch_seconds=time.perf_counter() - started)
    return reference, record


def write_batch_shard(bundle, corruption: str, batch: int, arrays: dict, diagnostics: dict) -> dict:
    import numpy as np
    directory = bundle.path / "batches" / corruption / ("batch%04d" % batch)
    directory.mkdir(parents=True, exist_ok=False)
    predictions = directory / "predictions.npz"
    with predictions.open("xb") as file:
        np.savez_compressed(file, **arrays)
    diagnostic_path = directory / "diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"predictions.npz": predictions, "diagnostics.json": diagnostic_path}


def copy_batch_shard(bundle, drive_root, corruption: str, batch: int, files: dict):
    """Copy compact immutable shards, never a per-batch full bundle ZIP."""
    if not drive_root:
        return None
    config = json.loads((bundle.path / "config.json").read_text())
    directory = Path(drive_root) / config["dataset"] / config["method"] / config["run_id"] / "batches" / corruption / ("batch%04d" % batch)
    directory.mkdir(parents=True, exist_ok=False)
    identities = {}
    for name, path in files.items():
        expected = source.file_identity(path)
        destination = directory / name
        with path.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        source.verify_identity(destination, expected)
        identities[name] = expected
    receipt = dict(status="complete", dataset=config["dataset"], method=config["method"],
                   run_id=config["run_id"], corruption=corruption, batch=batch, files=identities,
                   receipt_path=str(directory / "receipt.json"), auto_resume=False)
    pending = directory / "receipt.pending.json"
    pending.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    pending.rename(directory / "receipt.json")
    return receipt


def matched_source(config: dict, manifest: dict, guidance: dict) -> dict:
    if config["backbone"] == "pointmae":
        suffix = '; primary comparator is same-run terminal' if config.get('trajectory_observation') else '; comparator: separate original3DD terminal whole-policy comparison'
        return dict(status="unavailable", reason="No validated matched full Point-MAE source-only row bound in this study for " + config['dataset'] + suffix)
    path = source._declared_asset(Path(config["asset_root"]), manifest, guidance["matched_source_results"])
    validation = json.loads(path.read_text())
    rows = [row for row in validation.get("macro", []) if row["dataset"] == config["dataset"] and row["backbone"] == config["backbone"]]
    if validation.get("status") != "complete" or len(rows) != 1:
        raise ValueError("Require exactly one independently validated matched source row")
    row = rows[0]
    checkpoint_sha = manifest["files"][config["checkpoint"]]["sha256"]
    if row["checkpoint_sha256"] != checkpoint_sha or row["preprocessing"] != config["source_classifier_preprocessing"]:
        raise ValueError("Guidance checkpoint/preprocessing differs from matched source")
    if not re.fullmatch(r"[0-9a-f]{64}", row.get("archive_sha256", "")) or not 0 <= row["accuracy_percent"] <= 100:
        raise ValueError("Invalid matched source archive/accuracy identity")
    if row.get("total_examples") != config["n_examples"] * 15:
        raise ValueError("Matched source must contain all15 full original-ID rows")
    baseline_path = source._declared_asset(Path(config["asset_root"]), manifest, guidance["matched_source_manifest"])
    if source.file_identity(baseline_path)["sha256"] != validation.get("source_manifest_sha256"):
        raise ValueError("Matched source protocol manifest SHA-256 mismatch")
    baseline = source.dataset_manifest(json.loads(baseline_path.read_text()), config)
    current = source.dataset_manifest(manifest, config)
    names = [Path(config["label_path"]).name] + [source.selected_data_path(config, name).name for name in config["corruptions"]]
    if any(current["files"][name] != baseline["files"].get(name) for name in names):
        raise ValueError("Guidance data differs from matched source protocol")
    return row


def load_guidance_runtime(config: dict, manifest: dict):
    root = Path(config["asset_root"])
    guidance = manifest["guidance"]
    paths = {}
    for key, expected_sha in (("lion_checkpoint", LION_SHA256), ("lion_config", LION_CONFIG_SHA256)):
        entry = guidance[key]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = source._declared_asset(root, manifest, entry["path"])
        source.verify_identity(path, entry)
        if entry["sha256"] != expected_sha:
            raise ValueError("Guidance requires the locked original " + key)
        paths[key] = path
    hook = source._declared_asset(root, manifest, guidance["native_bootstrap"])
    spec = importlib.util.spec_from_file_location("multibackbone_native_bootstrap", hook)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    native_report = module.bootstrap(root)
    if native_report.get("no_build") is not True:
        raise RuntimeError("Native readiness must explicitly attest no-build bootstrap")
    from default_config import cfg
    from models.lion import LION
    import utilities_3dd_tta as ops
    options = cfg.clone()
    options.merge_from_file(str(paths["lion_config"]))
    lion = LION(options)
    lion.load_model(str(paths["lion_checkpoint"]), use_ema=False)
    lion.vae.eval().requires_grad_(False)
    lion.priors.eval().requires_grad_(False)
    if config["method"] == ORIGINAL_METHOD:
        from tta import tta_reconstruct
        graph, reconstruct = None, tta_reconstruct
    else:
        from tta_gsd_v2 import tta_gsd_reconstruct
        active = config['guidance_protocol']['loss_weights']
        if config.get('component_study') and not (active['spectral'] or active['invariant']):
            graph = None
        else:
            from graph_spectral_v2 import GraphSpectralDNA
            graph = GraphSpectralDNA(k=10, delta=0.1, gamma=0.6, use_4d_gft=False, device="cuda")
        reconstruct = tta_gsd_reconstruct
    return lion, graph, reconstruct, ops, native_report, options.dump()


def update_row(rows: list, config: dict, corruption: str, n: int, correct: int, seconds: float, memory: float, status: str):
    row = source.corruption_row(config["run_id"], config["seed"], corruption, n, correct, seconds, memory,
                                status, method=config.get("method", METHOD), severity=5, dataset=config["dataset"])
    if rows and rows[-1]["corruption"] == corruption:
        rows[-1] = row
    else:
        rows.append(row)


def worker(directories: list, *, observer_provider=None) -> None:
    bundles = {json.loads((Path(path) / "config.json").read_text())["backbone"]: source.RunBundle(Path(path)) for path in directories}
    configs = {name: json.loads((bundle.path / "config.json").read_text()) for name, bundle in bundles.items()}
    rows = {name: [] for name in bundles}
    heads, lion, active = {}, None, None
    try:
        common = next(iter(configs.values()))
        root = Path(common["asset_root"])
        manifest = source.read_manifest(root, common["asset_manifest_sha256"])
        guidance = manifest["guidance"]
        dataset = source.dataset_manifest(manifest, common)
        for name, config in configs.items():
            config.update(asset_manifest=manifest, dataset_provenance=dataset, matched_source=matched_source(config, manifest, guidance))
        labels_path = Path(common["label_path"])
        labels = source.load_array(labels_path, dataset["files"][labels_path.name], common, labels=True)
        arrays = {}
        for corruption in common["corruptions"]:
            path = source.selected_data_path(common, corruption)
            arrays[corruption] = source.load_array(path, dataset["files"][path.name], common)
        import numpy as np
        import torch
        from legacy_trajectory import rng_hash, tensor_hash
        if not torch.cuda.is_available():
            raise RuntimeError("Guidance requires CUDA; no CPU inference fallback")
        lion, graph, reconstruct, ops, native_report, lion_config = load_guidance_runtime(common, manifest)
        heads, guard = initialize_heads(configs, manifest)
        if common.get('chamfer_experiment', {}).get('variant') == 'semantic':
            # Include semantic implementation bytes even in an interrupted bundle.
            import semantic_trajectory
        from pointnet2_ops import pointnet2_utils
        def fps(points, number):
            indices = pointnet2_utils.furthest_point_sample(points.contiguous(), number)
            return pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
        torch.backends.cudnn.benchmark = common["method"] == ORIGINAL_METHOD
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        seed_hash = seed_inference(common["seed"])
        lion_before = dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors))
        classifier_before = {name: source.state_hash(head) for name, head in heads.items()}
        for name, config in configs.items():
            config.update(initialization_rng_guard=guard, inference_seed_boundary_sha256=seed_hash,
                          native_bootstrap=native_report, lion_state_before=lion_before,
                          resolved_lion_config=lion_config, lion_checkpoint_load="legacy raw state loader, strict default; exact pinned bytes verified before unpickling; EMA off",
                          classifier_state_before_sha256=classifier_before[name],
                          runtime=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                                       cudnn=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark,
                                       cudnn_deterministic=torch.backends.cudnn.deterministic,
                                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                       gpu=torch.cuda.get_device_name(), python=sys.version), shared_reconstruction_seconds=0.)
            source.snapshot_sources(bundles[name], config, manifest)
            if config.get('calibration_reference'):
                reference = config['calibration_reference']
                source.verify_identity(bundles[name].path / 'calibration_report.json', reference['identity'])
                proof = reference['calibration_bundles'][config['dataset']]
                if config.get('semantic_guidance', {}).get('head') == 'curvenet':
                    teacher = reference['teacher']
                    if (teacher['checkpoint'] != config['semantic_guidance']['teacher_checkpoint']
                            or teacher['checkpoint_identity'] != manifest['files'][teacher['checkpoint']]
                            or teacher['preprocessing'] != config['semantic_guidance']['teacher_preprocessing']):
                        raise ValueError('Calibrated CurveNet teacher checkpoint/preprocessing differs')
                if (proof['asset_manifest_sha256'] != config['asset_manifest_sha256']
                        or (config['chamfer_experiment']['variant'] == 'semantic'
                            and proof.get('classifier_state_sha256') != classifier_before.get(config['semantic_guidance']['head']))
                        or proof.get('lion_state') != lion_before):
                    raise ValueError('Calibrated assets or frozen classifier/LION states differ')
                for extension, expected in proof['native_extensions'].items():
                    current = config['native_extensions'].get(extension)
                    if current is None:
                        current = source.file_identity(Path(expected['path']))
                    if any(current.get(key) != expected.get(key) for key in ('bytes', 'sha256')):
                        raise ValueError('Calibrated native extension bytes differ: ' + extension)
                for source_name, expected in proof['executed_source_files'].items():
                    current = config['executed_source_files'].get(source_name)
                    if source_name.startswith('repo/') and current is not None and current != expected:
                        raise ValueError('Calibrated source bytes differ: ' + source_name)
            bundles[name].write_config(config)
        for corruption, values in arrays.items():
            active = corruption
            accumulated = {name: [] for name in heads}
            elapsed = {name: 0. for name in heads}
            correct = {name: 0 for name in heads}
            seen, peak = 0, 0.
            selected_ids = np.asarray(common.get('selected_original_ids', list(range(common['n_examples']))), dtype=np.int64)
            for batch, start in enumerate(range(0, len(selected_ids), common["batch_size"])):
                if common["max_batches"] and batch >= common["max_batches"]:
                    break
                end = min(start + common["batch_size"], len(selected_ids))
                batch_ids = selected_ids[start:end]
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                before_rng = rng_hash()
                started = time.monotonic()
                raw = torch.from_numpy(np.array(values[batch_ids]))
                scheduler_record = {}
                observers = []
                reference, reference_record = None, None
                if common.get('chamfer_experiment', {}).get('variant') == 'semantic':
                    reference, reference_record = semantic_reference(raw.to('cuda'), heads, configs, fps,
                                                                    guidance_head=common['semantic_guidance']['head'])
                def observer_factory(center, scale):
                    from shared_trajectory import SharedTrajectoryObserver
                    def classify_decoded(decoded):
                        geometry = decoded_to_geometry(decoded, common["dataset"], scale, center, ops)
                        return classify_geometry(geometry, heads, configs, fps)[0]
                    options = {}
                    observer_class = SharedTrajectoryObserver
                    if common.get('chamfer_experiment', {}).get('variant') == 'semantic':
                        from semantic_trajectory import SemanticTrajectoryObserver
                        observer_class = SemanticTrajectoryObserver
                        def classify_semantic_decoded(decoded):
                            geometry = decoded_to_geometry(decoded, common['dataset'], scale, center, ops)
                            return classify_semantic_geometry(geometry, heads, configs, fps, guidance_head=common['semantic_guidance']['head'])
                        options.update(classify_semantic_decoded=classify_semantic_decoded,
                                       kind=common['semantic_guidance']['kind'],
                                       reference_logits=reference, reference_record=reference_record)
                        if common['semantic_guidance']['head'] != 'pointmae':
                            options['guidance_head'] = common['semantic_guidance']['head']
                    if observer_provider is None:
                        observer = observer_class(lion.vae, heads, classify_decoded,
                                                  30 if corruption == "background" else 10,
                                                  num_classes=common['num_classes'], **options)
                    else:
                        observer = observer_provider(vae=lion.vae, heads=heads, classify_decoded=classify_decoded,
                                                     horizon=30 if corruption == 'background' else 10,
                                                     num_classes=common['num_classes'], original_ids=batch_ids.tolist(),
                                                     corruption=corruption, **options)
                    observers.append(observer)
                    return observer
                observer_options = {"observer_factory": observer_factory} if common.get("trajectory_observation") else {}
                paired_path = None
                preparation_mode = 'fresh_encode'
                if common.get('paired_state_root'):
                    ids_sha = hashlib.sha256(batch_ids.tobytes()).hexdigest()
                    paired_path = Path(common['paired_state_root']) / common['dataset'] / corruption / (ids_sha + '.npz')
                    if paired_path.exists():
                        preparation_mode = 'cached_replay'
                geometry, metrics = reconstruct_geometry(raw, common["dataset"], corruption, lion, graph, reconstruct, ops,
                                                         method=common["method"], scheduler_observer=scheduler_record.update,
                                                         protocol=common['guidance_protocol'], paired_state_path=paired_path, **observer_options)
                torch.cuda.synchronize()
                reconstruction_seconds = time.monotonic() - started
                reconstruction_rng = rng_hash()
                all_logits, head_diagnostics, head_seconds = {}, {}, {}
                for name in heads:
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    logits, diagnostic = classify_geometry(geometry, {name: heads[name]}, configs, fps)
                    torch.cuda.synchronize()
                    head_seconds[name] = time.monotonic() - started
                    all_logits.update(logits); head_diagnostics.update(diagnostic)
                if rng_hash() != reconstruction_rng:
                    raise RuntimeError("Frozen heads advanced the reconstruction RNG")
                if observers:
                    observers[0].terminal(all_logits)
                peak = max(peak, torch.cuda.max_memory_allocated() / 1024 ** 2)
                shared = dict(geometry_sha256=tensor_hash(geometry), geometry_shape=list(geometry.shape),
                              reconstruction_rng_before_sha256=before_rng, reconstruction_rng_after_sha256=reconstruction_rng,
                              shared_reconstruction_seconds=reconstruction_seconds, guidance_metrics=metrics,
                              whole_process_peak_gpu_memory_mb=peak, seed=common["seed"], index_start=start, index_end=end)
                shared['preparation_mode'] = preparation_mode
                shared['paired_cache_io_in_reconstruction_time'] = paired_path is not None
                if scheduler_record:
                    shared["scheduler"] = scheduler_record
                    for config in configs.values():
                        recorded = config.setdefault("scheduler_records", {}).setdefault(corruption, scheduler_record)
                        if recorded != scheduler_record:
                            raise RuntimeError("Original scheduler changed within a corruption")
                if observers:
                    shared["trajectory"] = observers[0].diagnostics()
                if paired_path is not None:
                    shared['paired_state'] = dict(path=str(paired_path), **source.file_identity(paired_path))
                pending = {}
                seen += end - start
                for name, logits in all_logits.items():
                    prediction = logits.argmax(-1).cpu().numpy()
                    payload = dict(index=batch_ids.copy(), label=np.array(labels[batch_ids], dtype=np.int64),
                                   prediction=prediction, logits=logits.cpu().numpy())
                    accumulated[name].append(payload)
                    correct[name] += int((prediction == labels[batch_ids]).sum())
                    elapsed[name] += reconstruction_seconds / len(heads) + head_seconds[name]
                    configs[name]["shared_reconstruction_seconds"] += reconstruction_seconds
                    configs[name].setdefault("completed_batches", {})[corruption] = batch + 1
                    diagnostics = dict(shared, **head_diagnostics[name], head_seconds=head_seconds[name])
                    batch_payload = dict(payload, **observers[0].arrays(name)) if observers else payload
                    pending[name] = write_batch_shard(bundles[name], corruption, batch, batch_payload, diagnostics)
                    update_row(rows[name], configs[name], corruption, seen, correct[name], elapsed[name], peak, "partial")
                    bundles[name].write_results(rows[name], "running")
                    bundles[name].write_config(configs[name])
                for name, files in pending.items():
                    copy_batch_shard(bundles[name], common.get("drive_root"), corruption, batch, files)
                print(common["dataset"], corruption, "batch", batch, "IDs", start, end, "shared reconstruction seconds", reconstruction_seconds, flush=True)
            for name in heads:
                combined = {key: np.concatenate([part[key] for part in accumulated[name]]) for key in accumulated[name][0]}
                np.savez_compressed(bundles[name].path / (corruption + "_predictions.npz"), **combined)
                rows[name][-1]["status"] = "complete" if seen == common["n_examples"] else "partial"
                after = source.state_hash(heads[name])
                if after != classifier_before[name]:
                    raise RuntimeError("Frozen classifier state changed: " + name)
                configs[name]["classifier_state_after_sha256"] = after
                bundles[name].write_results(rows[name], "running")
                bundles[name].write_config(configs[name])
                source.save_drive_checkpoint(bundles[name], common.get("drive_root"), "corruption-" + corruption)
            active = None
        lion_after = dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors))
        if lion_before != lion_after:
            raise RuntimeError("LION weights/buffers changed")
        for name, config in configs.items():
            config.update(lion_state_after=lion_after, status="complete" if len(common["corruptions"]) == 15 and not common["max_batches"] and len(selected_ids) == common['n_examples'] else "partial")
            source.snapshot_sources(bundles[name], config, manifest)
            bundles[name].write_results(rows[name], config["status"])
    except BaseException:
        failure = traceback.format_exc()
        for name, config in configs.items():
            config.update(status="failed", failure=failure)
            if active and rows[name] and rows[name][-1]["corruption"] == active:
                rows[name][-1]["status"] = "failed"
            bundles[name].write_results(rows[name], "failed")
        raise
    finally:
        if lion is not None and any(config["status"] == "failed" for config in configs.values()):
            try:
                after = dict(vae=source.state_hash(lion.vae), priors=source.state_hash(lion.priors))
                for config in configs.values():
                    config["lion_state_after"] = after
            except Exception as error:
                for config in configs.values():
                    config["lion_state_audit_error"] = repr(error)
        for name, config in configs.items():
            if name in heads and config["status"] == "failed":
                try:
                    config["classifier_state_after_sha256"] = source.state_hash(heads[name])
                except Exception as error:
                    config["classifier_state_audit_error"] = repr(error)
            bundles[name].write_config(config)


def main(argv=None, *, worker_script=None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--worker-bundles"]:
        worker(json.loads(arguments[1]))
        return
    configs = configurations(arguments)
    if configs[0]["preview"]:
        print(json.dumps(configs, indent=2, sort_keys=True))
        return
    bundles = []
    for config in configs:
        short_name = config['backbone'] + '-seed' + str(config['seed'])
        if config.get('component_study'):
            short_name = 'guidance-components-' + config['component_arm'] + '-' + short_name
        if config.get('chamfer_experiment'):
            tag = config['chamfer_experiment']['variant']
            if tag == 'semantic':
                weight_tag = str(config['semantic_guidance']['weight']).replace('.', 'p').replace('+', '').replace('-', 'm')
                tag += '-' + config['semantic_guidance']['kind'] + '-w' + weight_tag
            short_name = 'chamfer-' + tag + '-' + config['backbone'] + '-seed' + str(config['seed'])
        bundle = source.RunBundle.create(Path(config["result_root"]), short_name, config,
                                          shlex.join([sys.executable, str(Path(__file__).resolve()), *arguments]))
        bundles.append(bundle)
        print("Run bundle:", bundle.path, flush=True)
        if config.get('index_manifest'):
            shutil.copyfile(config['index_manifest'], bundle.path / 'selection_manifest.json')
        if config.get('calibration_reference'):
            reference = config['calibration_reference']
            destination = bundle.path / 'calibration_report.json'
            shutil.copyfile(reference['path'], destination)
            source.verify_identity(destination, reference['identity'])
        (bundle.path / "notes.md").write_text("# Published alternative matched guidance study\n\n" + config["display_name"] + ". Not full GSDTTA or paper reproduction.\n"
            + config["timing_policy"] + ".\nBatch shards preserve interrupted-corruption progress; no auto resume.\nDrive corruption checkpoints capture log prefixes.\n")
        (bundle.path / "environment.txt").write_text(sys.version + "\n" + source.command_output(["nvidia-smi"]) + "\n" + source.command_output([sys.executable, "-m", "pip", "freeze"]))
    group = bundles[0].path.name + "_" + configs[0]["dataset"]
    for bundle in bundles:
        config = json.loads((bundle.path / "config.json").read_text())
        config["reconstruction_group_id"] = group
        bundle.write_config(config)
    process, failure, logs = None, None, []
    try:
        logs = [(bundle.path / "stdout.log").open("w") for bundle in bundles]
        process = subprocess.Popen([sys.executable, "-u", str(Path(worker_script or __file__).resolve()), "--worker-bundles", json.dumps([str(bundle.path) for bundle in bundles])],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(source.REPO))
        with process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)
                for log in logs:
                    log.write(line); log.flush()
        returncode = process.wait()
    except BaseException as error:
        failure = error
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        returncode = (process.returncode if process is not None else -1) or -1
    finally:
        for log in logs:
            log.close()
    delivery_error = None
    for bundle in bundles:
        if returncode:
            config = json.loads((bundle.path / "config.json").read_text())
            config.update(status="failed", worker_returncode=returncode)
            if failure is not None:
                config["parent_failure"] = repr(failure)
            bundle.write_config(config)
            bundle.mark_failed()
        try:
            source.seal_bundle(bundle)
            source.save_drive_checkpoint(bundle, configs[0].get("drive_root"), "final", final=True)
        except BaseException as error:
            delivery_error = delivery_error or error
    if delivery_error is not None:
        raise delivery_error
    if failure is not None:
        raise failure
    if returncode:
        raise SystemExit(returncode)


def experiment_main(variant, argv=None):
    """Thin standalone entrypoints share the locked runner without globals."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value == '--chamfer-experiment' or value.startswith('--chamfer-experiment=') for value in arguments):
        raise ValueError('The standalone script already selects exactly one experiment')
    return main(['--chamfer-experiment', variant, *arguments])


if __name__ == "__main__":
    main()
