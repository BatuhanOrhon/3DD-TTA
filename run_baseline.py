"""Colab smoke runner for the original 3DD-TTA path, with auditable artifacts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
import platform
import random
import shlex
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace

from research_artifacts import (CLEAN_CONTROL, CORRUPTIONS, RunBundle, corruption_row,
                                decoder_control_row, validate_selection)

REPO = Path(__file__).resolve().parent
IDENTITY_PREPROCESSING = (
    "direct corruption-file loading -> normalize -> interpolate/upsample(2048) -> "
    "scale(3.3885) -> rotate -> rotateback -> output normalize -> FPS(1024) -> "
    "frozen classifier"
)
PURE_VAE_METHODS = frozenset(("pure_vae_encode_decode", "pure_vae_seed_stability"))
PREPROCESSING_IDENTITY_METHODS = frozenset(("preprocessing_identity", "preprocessing_identity_seed_stability"))
SHARED_DECODER_METHOD = "shared_trajectory_decoder_control"
SHARED_DECODER_PILOT_CORRUPTIONS = ("gaussian", "impulse")


def is_preprocessing_identity_method(method: str) -> bool:
    """Return whether a method uses the LION-free preprocessing route."""
    return method in PREPROCESSING_IDENTITY_METHODS


@contextmanager
def preserve_classifier_rng(numpy_module, torch_module):
    """Preserve RNG streams consumed by the extra decoder-control call."""
    numpy_state = numpy_module.random.get_state()
    torch_cpu_state = torch_module.get_rng_state()
    torch_cuda_states = (torch_module.cuda.get_rng_state_all()
                         if torch_module.cuda.is_available() else None)
    try:
        yield
    finally:
        numpy_module.random.set_state(numpy_state)
        torch_module.set_rng_state(torch_cpu_state)
        if torch_cuda_states is not None:
            torch_module.cuda.set_rng_state_all(torch_cuda_states)


def decoder_control_batch_metrics(target, original_pred, updated_pred,
                                  original_points, updated_points,
                                  original_style, updated_style) -> dict:
    """Compute paired decoder diagnostics for one batch."""
    target = target.detach().view(-1)
    original_pred = original_pred.detach().view(-1)
    updated_pred = updated_pred.detach().view(-1)
    if not (target.shape == original_pred.shape == updated_pred.shape):
        raise ValueError("Decoder-control predictions and targets must have equal shapes.")
    batch_size = target.numel()
    output_diff = (updated_points.detach() - original_points.detach()).abs().reshape(batch_size, -1)
    style_diff = (updated_style.detach() - original_style.detach()).reshape(batch_size, -1)
    return dict(
        n_examples=int(batch_size),
        original_style_n_correct=int((original_pred == target).sum().item()),
        updated_style_n_correct=int((updated_pred == target).sum().item()),
        disagreement_n=int((original_pred != updated_pred).sum().item()),
        decoder_output_difference=float(output_diff.mean().item()),
        style_displacement=float(style_diff.norm(dim=1).mean().item()),
    )


def classify_decoder_variants(base_model, original_points, updated_points, label,
                              torch_module, numpy_module):
    """Classify both decoded branches while isolating the second call's RNG."""
    with torch_module.no_grad():
        original_pred = base_model.module.classification_only(
            original_points, only_unmasked=False).argmax(-1).view(-1)
    with preserve_classifier_rng(numpy_module, torch_module):
        with torch_module.no_grad():
            updated_pred = base_model.module.classification_only(
                updated_points, only_unmasked=False).argmax(-1).view(-1)
    return label.cuda().view(-1), original_pred, updated_pred


def selected_data_path(dataset_root: str, name: str, severity: int = 5) -> Path:
    """Resolve one repository-format input without changing benchmark filenames."""
    filename = "data_original.npy" if name == CLEAN_CONTROL else "data_" + name + "_" + str(severity) + ".npy"
    return Path(dataset_root) / filename


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=120)
        return result.stdout.strip() if result.returncode == 0 else (
            "UNAVAILABLE (exit %s): %s" % (result.returncode, result.stdout.strip()))
    except (OSError, subprocess.TimeoutExpired) as error:
        return "UNAVAILABLE: " + str(error)


def file_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def module_inventory(module) -> dict:
    # Read-only: no forward passes, mode changes or random draws.
    return {
        "training": module.training,
        "submodule_modes": {name: child.training for name, child in module.named_modules()},
        "dropout": [{"name": name, "class": type(child).__name__, "p": child.p,
                     "training": child.training}
                    for name, child in module.named_modules()
                    if type(child).__name__.startswith("Dropout")],
        "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
    }


def extension_inventory() -> dict:
    observed = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name in {"diffusers", "knn_cuda", "pointnet2_ops", "scipy"} or (
            path and ("pointnet2" in name or "chamfer" in name.lower())
            and Path(path).suffix in {".so", ".pyd"}):
            observed[name] = dict(
                version=getattr(module, "__version__", "not exposed"),
                file=file_identity(Path(path)) if path else None)
    return observed


def fps_with_diagnostics(data, number: int, baseline, torch_module):
    """Run legacy FPS/gather and return aggregate, read-only diagnostics."""
    pointnet2_utils = baseline.misc.pointnet2_utils
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

    index_cpu = fps_idx.detach().cpu().long()
    unique_counts = [int(torch_module.unique(row).numel()) for row in index_cpu]
    data_cpu = data.detach().cpu()
    finite_point_mask_cpu = torch_module.isfinite(data_cpu).all(dim=-1)
    finite_coordinate_unique_counts = [
        int(torch_module.unique(data_cpu[index, finite_point_mask_cpu[index]], dim=0).shape[0])
        if bool(finite_point_mask_cpu[index].any()) else 0
        for index in range(data_cpu.shape[0])
    ]
    finite_point_mask = torch_module.isfinite(data).all(dim=-1)
    origin_mask = data.square().sum(dim=-1) <= 1e-3
    selected_origin = torch_module.gather(origin_mask, 1, fps_idx.long())
    selected_finite = torch_module.gather(finite_point_mask, 1, fps_idx.long())
    return fps_data, dict(
        batch_examples=int(data.shape[0]),
        input_points=int(data.shape[1]),
        target_points=int(number),
        unique_index_min=min(unique_counts),
        unique_index_max=max(unique_counts),
        unique_index_sum=sum(unique_counts),
        duplicate_index_sum=sum(number - count for count in unique_counts),
        input_origin_sum=int(origin_mask.sum().item()),
        selected_origin_sum=int(selected_origin.sum().item()),
        input_point_finite_sum=int(finite_point_mask_cpu.sum().item()),
        input_point_nonfinite_sum=int((~finite_point_mask_cpu).sum().item()),
        input_scalar_nan_sum=int(torch_module.isnan(data_cpu).sum().item()),
        input_scalar_inf_sum=int(torch_module.isinf(data_cpu).sum().item()),
        finite_coordinate_unique_min=min(finite_coordinate_unique_counts),
        finite_coordinate_unique_max=max(finite_coordinate_unique_counts),
        finite_coordinate_unique_sum=sum(finite_coordinate_unique_counts),
        selected_point_finite_sum=int(selected_finite.sum().item()),
        selected_point_nonfinite_sum=int((~selected_finite).sum().item()),
        selected_index_min=int(index_cpu.min().item()),
        selected_index_max=int(index_cpu.max().item()),
    )


def merge_fps_diagnostics(config: dict, corruption: str, batch_stats: dict) -> None:
    """Accumulate diagnostics without storing per-example FPS indices."""
    diagnostics = config.setdefault("fps_diagnostics", {}).setdefault(corruption, dict(
        batches=0, examples=0, input_points_min=None, input_points_max=None,
        target_points=None, unique_index_min=None, unique_index_max=None,
        unique_index_sum=0, duplicate_index_sum=0, input_origin_sum=0,
        selected_origin_sum=0, input_point_finite_sum=0,
        input_point_nonfinite_sum=0, input_scalar_nan_sum=0,
        input_scalar_inf_sum=0, finite_coordinate_unique_min=None,
        finite_coordinate_unique_max=None, finite_coordinate_unique_sum=0,
        selected_point_finite_sum=0, selected_point_nonfinite_sum=0,
        selected_index_min=None, selected_index_max=None))
    diagnostics["batches"] += 1
    diagnostics["examples"] += batch_stats["batch_examples"]
    diagnostics["target_points"] = batch_stats["target_points"]
    for field, key in (("input_points", "input_points_min"),
                       ("unique_index_min", "unique_index_min"),
                       ("finite_coordinate_unique_min", "finite_coordinate_unique_min"),
                       ("selected_index_min", "selected_index_min")):
        value = batch_stats[field]
        diagnostics[key] = value if diagnostics[key] is None else min(diagnostics[key], value)
    for field, key in (("input_points", "input_points_max"),
                       ("unique_index_max", "unique_index_max"),
                       ("finite_coordinate_unique_max", "finite_coordinate_unique_max"),
                       ("selected_index_max", "selected_index_max")):
        value = batch_stats[field]
        diagnostics[key] = value if diagnostics[key] is None else max(diagnostics[key], value)
    for field in ("unique_index_sum", "duplicate_index_sum", "input_origin_sum", "selected_origin_sum",
                  "input_point_finite_sum", "input_point_nonfinite_sum", "input_scalar_nan_sum",
                  "input_scalar_inf_sum", "finite_coordinate_unique_sum", "selected_point_finite_sum",
                  "selected_point_nonfinite_sum"):
        diagnostics[field] += batch_stats[field]


def preprocessing_identity_points(data, baseline, args, torch_module):
    """Apply the TTA preprocessing/output contract while bypassing LION."""
    data_sample, data_center, data_max = tta_preprocess_points(data, baseline, args, torch_module)
    return tta_postprocess_points(data_sample, data_center, data_max, baseline, args)


def tta_preprocess_points(data, baseline, args, torch_module):
    """Prepare one batch for the TTA/VAE input contract."""
    data_sample, data_center, data_max = baseline.normalize(data)
    data_sample = baseline.upsample_all(data_sample.detach().cpu().numpy(), 2048)
    data_sample = torch_module.from_numpy(data_sample).float().to(args.device)
    data_sample *= 3.3885
    data_sample = baseline.rotate_pointcloud(data_sample)
    return data_sample, data_center, data_max


def tta_postprocess_points(pred_points, data_center, data_max, baseline, args):
    """Restore the classifier-facing output contract after VAE/TTA."""
    pred_points = baseline.rotateback_pointcloud(pred_points)
    if args.dataset_name == "scanobjectnn-c":
        pred_points /= 3.3885
        pred_points = baseline.unnormalize_data(pred_points, data_max, data_center)
    else:
        pred_points, _, _ = baseline.normalize(pred_points)
    return baseline.misc.fps(pred_points, 1024)


def pure_vae_encode_decode_points(data, baseline, lion, args, torch_module):
    """Encode and decode through LION's VAE without prior or guidance."""
    data_sample, data_center, data_max = tta_preprocess_points(data, baseline, args, torch_module)
    encoded = lion.vae.encode(data_sample)
    decomposed_eps = lion.vae.decompose_eps(encoded[0])
    pred_points = lion.vae.sample(num_samples=data_sample.shape[0],
                                  decomposed_eps=decomposed_eps)
    return tta_postprocess_points(pred_points, data_center, data_max, baseline, args)


def shared_trajectory_decoder_points(data, baseline, lion, args, torch_module,
                                     steps, scheduler_observer=None):
    """Run one trajectory, then decode its final local state with both styles."""
    data_sample, data_center, data_max = tta_preprocess_points(data, baseline, args, torch_module)
    with torch_module.enable_grad():
        trajectory = baseline.tta_reconstruct(
            data_sample, lion, steps, args.gamma, args.eta, args.lambdaa, 100,
            scheduler_observer=scheduler_observer, return_trajectory=True)
    with torch_module.no_grad():
        original_points, updated_points = baseline.decode_shared_trajectory(lion, trajectory)
    return (
        tta_postprocess_points(original_points, data_center, data_max, baseline, args),
        tta_postprocess_points(updated_points, data_center, data_max, baseline, args),
        trajectory,
        original_points,
        updated_points,
    )


def notes_for_run(args: SimpleNamespace) -> str:
    """Describe the evaluated path without classifying source-only runs as smoke tests."""
    if args.method == "source_only":
        coverage = ("All selected corruptions are evaluated completely."
                    if args.max_batches == 0 else
                    "Only a file-order prefix is evaluated; this is not a full benchmark result.")
        diagnostic_note = (" Read-only FPS diagnostics record index uniqueness, duplicate slots, "
                           "near-origin candidates, finite/non-finite points, scalar NaN/Inf counts, "
                           "and finite coordinate uniqueness; classifier inputs remain legacy FPS outputs."
                           if args.fps_diagnostics else "")
        return (
            "# Source-only run\n\n"
            "Purpose: evaluate corrupted input without LION/TTA. Direct corruption-file loading -> "
            "FPS(1024) -> frozen Point-MAE classifier; no LION, diffusion, or guidance is loaded.\n"
            + coverage + diagnostic_note + "\n"
            "Seed-controlled evaluation; no resume or test-set hyperparameter search. "
            "CUDA runtime excludes asset hashing and model loading; peak allocated memory includes models.\n"
            "User: add Colab runtime/GPU type and anomalies.\n"
        )
    if args.method == "preprocessing_identity":
        return (
            "# Preprocessing identity control\n\n"
            "Purpose: measure the TTA preprocessing/output chain with LION fully bypassed. "
            "Direct corruption-file loading -> per-shape normalize -> interpolation/upsampling "
            "to 2048 -> scale 3.3885 -> rotate -> rotateback -> ModelNet output normalize -> "
            "FPS(1024) -> frozen Point-MAE classification_only.\n"
            "This is the locked ModelNet40-C severity-5 all-15 seed-0 batch-32 control; "
            "no dataset mutation, alternate FPS policy, EMA, scheduler, or guidance is used.\n"
            "CUDA runtime excludes asset hashing and model loading; peak allocated memory includes models.\n"
            "User: add Colab runtime/GPU type and anomalies.\n"
        )
    if args.method == "preprocessing_identity_seed_stability":
        return (
            "# Preprocessing identity seed-stability control\n\n"
            "Purpose: repeat the locked preprocessing identity control at a second seed; "
            "this isolates stochastic preprocessing variability without LION, VAE, diffusion, "
            "or guidance. Direct corruption-file loading -> per-shape normalize -> "
            "interpolation/upsampling to 2048 -> scale 3.3885 -> rotate -> rotateback -> "
            "ModelNet output normalize -> FPS(1024) -> frozen Point-MAE classification_only.\n"
            "This is locked to ModelNet40-C severity 5, all-15, batch 32, seed 1 or 2, "
            "and the same assets as the archived seed-0 preprocessing identity run. No "
            "LION, EMA, alternate FPS policy, dataset mutation, or TTA guidance is used. "
            "Combine this run with the archived preprocessing_identity seed-0 ZIP for mean/std analysis.\n"
            "CUDA runtime excludes asset hashing and model loading; peak allocated memory includes models.\n"
            "User: add Colab runtime/GPU type and anomalies.\n"
        )
    if args.method == "pure_vae_encode_decode":
        return (
            "# Pure VAE encode/decode control\n\n"
            "Purpose: measure VAE reconstruction after the TTA preprocessing chain, "
            "without LION priors, diffusion scheduler, or guidance. Direct corruption-file "
            "loading -> per-shape normalize -> interpolation/upsampling to 2048 -> scale "
            "3.3885 -> rotate -> VAE encode/decompose_eps/sample -> rotateback -> ModelNet "
            "output normalize -> FPS(1024) -> frozen Point-MAE classification_only.\n"
            "This is the locked ModelNet40-C severity-5 all-15 seed-0 batch-32 control; "
            "raw VAE weights are used and the VAE is eval mode. No EMA, alternate FPS policy, "
            "scheduler, guidance, GSD, or PxP is used.\n"
            "CUDA runtime excludes asset hashing and model loading; peak allocated memory includes models.\n"
            "User: add Colab runtime/GPU type and anomalies.\n"
        )
    if args.method == "pure_vae_seed_stability":
        return (
            "# Pure VAE seed-stability control\n\n"
            "Purpose: repeat the locked pure VAE encode/decode control at a second seed; "
            "this is a stochastic stability diagnostic, not a new TTA method. Direct "
            "corruption-file loading -> per-shape normalize -> interpolation/upsampling "
            "to 2048 -> scale 3.3885 -> rotate -> VAE encode/decompose_eps/sample -> "
            "rotateback -> ModelNet output normalize -> FPS(1024) -> frozen Point-MAE "
            "classification_only.\n"
            "This is locked to ModelNet40-C severity 5, all-15, batch 32, seed 1 or 2, "
            "raw VAE eval, and the same assets as the archived seed-0 pure VAE run. "
            "No priors, EMA, diffusion scheduler, guidance, GSD, PxP, alternate FPS "
            "policy, or dataset mutation is used. Combine this run with the archived "
            "pure_vae_encode_decode seed-0 ZIP for mean/std analysis.\n"
            "CUDA runtime excludes asset hashing and model loading; peak allocated memory includes models.\n"
            "User: add Colab runtime/GPU type and anomalies.\n"
        )
    if args.method == SHARED_DECODER_METHOD:
        return (
            "# Shared-trajectory decoder-style control\n\n"
            "Purpose: compare original versus updated final decoder style on one shared "
            "eval/raw LION trajectory; this is a reproduction control, not a new TTA method. "
            "Each batch runs one VAE encode, one initial noise draw, one DDIM/SCD trajectory, "
            "then decodes the same final local latent with original shape_latent and updated "
            "style_cond.\n"
            "Locked scopes: ModelNet40-C severity 5 Gaussian/Impulse pilot or complete "
            "canonical all-15 confirmation, complete files, batch 32, seed 0/1/2, "
            "raw LION weights, LION eval mode, EMA disabled, and "
            "unchanged gamma/eta/lambda/scheduler settings. The second classifier call "
            "snapshots/restores NumPy and Torch CPU/CUDA RNG state.\n"
            "The artifact records both accuracies, paired percentage-point delta, prediction "
            "disagreement, decoder output difference, style displacement, commit and asset "
            "hashes in config.json and the extended CSV fields.\n"
            "User: add Colab runtime/GPU type and anomalies.\n"
        )
    return (
        "# Smoke run\n\nPurpose: validate execution and artifact schema, NOT benchmark accuracy.\n"
        "Only a file-order prefix is evaluated. Seed-controlled, not fully paired.\n"
        "Original TTA math, legacy LION modes, 5/35 steps, gamma/eta mapping and static decode style retained.\n"
        "CUDA runtime excludes asset hashing and model loading; peak allocated memory includes models.\n"
        "No resume or test-set hyperparameter search. User: add Colab runtime/GPU type and anomalies.\n"
    )


def run_worker(directory: str) -> None:
    """Internal subprocess entry: parent captures Python/native stdout and stderr."""
    bundle = RunBundle(Path(directory))
    config = json.loads((bundle.path / "config.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**config["cli_args"])
    rows = []
    counters = {"n": 0, "correct": 0}
    active = None
    started = None
    torch = None
    try:
        print("Run directory:", bundle.path, flush=True)
        environment = [
            "UTC run ID: " + config["run_id"], "Python: " + sys.version,
            "Platform: " + platform.platform(),
            "Colab runtime/GPU type: user must confirm in notes.md",
            "Git branch: " + config["git_branch"], "Git commit: " + config["git_commit"],
            "Git status:\n" + config["git_status"],
            "Git diff stat:\n" + command_output(["git", "diff", "--stat", "HEAD"]),
            "nvidia-smi:\n" + command_output(["nvidia-smi"]),
            "pip freeze:\n" + command_output([sys.executable, "-m", "pip", "freeze"]),
        ]
        (bundle.path / "environment.txt").write_text("\n\n".join(environment) + "\n", encoding="utf-8")

        # Hash the actual assets before loading; never hash only a filename.
        print("Hashing checkpoints and selected data files...", flush=True)
        assets = {"classifier_checkpoint": args.pointmae_ckpt,
                  "pointmae_config": args.pointmae_config, "labels": args.label_path}
        if args.method in {"3dd_original", *PURE_VAE_METHODS, SHARED_DECODER_METHOD}:
            assets.update(lion_checkpoint=args.diff_ckpt, lion_config=args.diff_config)
        identities = {name: file_identity(Path(value)) for name, value in assets.items()}
        data_files = {name: file_identity(selected_data_path(args.dataset_root, name, config["severity"]))
                      for name in args.corruptions}
        config.update(asset_manifest=identities, dataset_hash_manifest=data_files,
                      classifier_checkpoint_sha256=identities["classifier_checkpoint"]["sha256"])
        if "lion_checkpoint" in identities:
            config["lion_checkpoint_sha256"] = identities["lion_checkpoint"]["sha256"]
        bundle.write_config(config)
        with (bundle.path / "environment.txt").open("a", encoding="utf-8") as file:
            file.write("\nAsset hashes:\n" + json.dumps({"assets": identities, "data": data_files}, indent=2) + "\n")

        import numpy as np
        import torch as torch_module
        import main_3dd_tta as baseline
        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("Original baseline requires CUDA; CPU fallback is not supported.")
        args.use_gpu = True
        args.distributed = False
        # Exactly the baseline's cuDNN benchmark setting. Do not force determinism.
        torch.backends.cudnn.benchmark = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        config["randomness"] = dict(
            policy="seed-controlled, NOT common-draw-paired",
            seed=args.seed, cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
            torch_version=torch.__version__, numpy_version=np.__version__,
            gpu=torch.cuda.get_device_name(), data_order="file order; shuffle=False",
            num_workers=0, drop_last=False)
        config["checkpoint_loads"] = {}
        bundle.write_config(config)
        with (bundle.path / "environment.txt").open("a", encoding="utf-8") as file:
            file.write("\nRuntime settings:\n" + json.dumps(config["randomness"], indent=2) + "\n")

        def checkpoint_observer(name, keys):
            config["checkpoint_loads"][name] = dict(
                strict=name != "pointmae",
                missing_keys=list(keys.missing_keys), unexpected_keys=list(keys.unexpected_keys))
            bundle.write_config(config)
            print("Checkpoint load:", name, config["checkpoint_loads"][name], flush=True)

        if args.method == "source_only":
            point_config = baseline.cfg_from_yaml_file(args.pointmae_config)
            point_config.model.cls_dim = 15 if args.dataset_name == "scanobjectnn-c" else (55 if args.dataset_name == "shapenet-c" else 40)
            base_model = baseline.load_base_model(args, point_config, None, checkpoint_observer=checkpoint_observer)
            base_model.eval()
            lion = None
            source_name = "original input" if args.clean_control else "corrupted input"
            config.update(lion_loaded=False, preprocessing=source_name + " -> FPS(1024) -> frozen classifier",
                          resolved_pointmae_config=point_config)
            if args.fps_diagnostics:
                config["preprocessing"] = (source_name +
                                            " -> legacy FPS(1024) with read-only diagnostics -> frozen classifier")
                config["fps_diagnostics_schema"] = "legacy_fps_v2_finite_coordinate_unique"
        elif is_preprocessing_identity_method(args.method):
            point_config = baseline.cfg_from_yaml_file(args.pointmae_config)
            point_config.model.cls_dim = 40
            base_model = baseline.load_base_model(args, point_config, None, checkpoint_observer=checkpoint_observer)
            base_model.eval()
            lion = None
            config.update(
                lion_loaded=False,
                preprocessing=IDENTITY_PREPROCESSING,
                lion_mode_policy="bypassed",
                final_decode_style="identity; no decode",
                resolved_pointmae_config=point_config)
        else:
            base_model, lion = baseline.configure_model(args, checkpoint_observer=checkpoint_observer)
            if args.lion_eval_mode or args.method in PURE_VAE_METHODS:
                lion.vae.eval()
                lion.priors.eval()
            config["resolved_lion_config_yaml"] = baseline.diff_config.dump()
            if args.method in PURE_VAE_METHODS:
                config.update(
                    preprocessing=IDENTITY_PREPROCESSING,
                    lion_loaded=True,
                    lion_mode_policy="raw VAE eval; priors bypassed",
                    vae_contract="encode -> decompose_eps -> sample",
                    prior_used=False,
                    final_decode_style="VAE decoder from encoded latents")
            elif args.method == SHARED_DECODER_METHOD:
                config.update(
                    lion_loaded=True,
                    lion_mode_policy="raw LION eval; EMA disabled",
                    final_decode_style="shared final local latent; original and updated style",
                    decoder_control_contract="one trajectory, two final-style decodes",
                    rng_control="snapshot/restore NumPy and Torch CPU/CUDA around second classifier call")
        config["extension_inventory"] = extension_inventory()

        def record_modes(key):
            config[key] = dict(classifier=module_inventory(base_model))
            if lion is not None:
                config[key].update(lion_vae=module_inventory(lion.vae), lion_priors=module_inventory(lion.priors))

        record_modes("module_inventory_before")
        bundle.write_config(config)

        def scheduler_observer(scheduler):
            observed = dict(scheduler.config)
            if config["scheduler_config"] and config["scheduler_config"] != observed:
                raise RuntimeError("Scheduler config changed during the run.")
            config["scheduler_config"] = observed
            config["scheduler_class"] = type(scheduler).__name__
            config.setdefault("scheduler_timesteps", {})[active] = scheduler.timesteps.cpu().tolist()

        def batch_observer(target, pred):
            target, pred = target.detach().cpu().view(-1), pred.detach().cpu().view(-1)
            if target.shape != pred.shape:
                raise RuntimeError("Target/prediction counts differ.")
            counters["n"] += int(target.numel())
            counters["correct"] += int((target == pred).sum().item())
            config.setdefault("observed_batch_sizes", {}).setdefault(active, []).append(int(target.numel()))

        for active in args.corruptions:
            counters = {"n": 0, "correct": 0}
            decoder_totals = None
            dataset = baseline.PointDataset(args.dataset_root, args.label_path, active, severity=config["severity"])
            if len(dataset.data) != len(dataset.labels) or len(dataset) == 0:
                raise ValueError("Empty data or data/label count mismatch.")
            if dataset.data.ndim != 3 or dataset.data.shape[-1] != 3:
                raise ValueError("Expected data shape [examples, points, 3].")
            if dataset.labels.size != len(dataset) or not np.issubdtype(dataset.labels.dtype, np.integer):
                raise ValueError("Expected one integer class label per example.")
            if dataset.labels.min() < 0 or dataset.labels.max() >= config["num_classes"]:
                raise ValueError("ModelNet40 class label outside [0,39].")
            config.setdefault("dataset_inventory", {})[active] = dict(
                data_shape=list(dataset.data.shape), data_dtype=str(dataset.data.dtype),
                label_shape=list(dataset.labels.shape), label_dtype=str(dataset.labels.dtype),
                total_examples=len(dataset), min_label=int(dataset.labels.min()),
                max_label=int(dataset.labels.max()))
            loader = baseline.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            batches = loader if args.max_batches == 0 else itertools.islice(loader, args.max_batches)
            steps = 35 if active == "background" else 5
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            if args.method == "source_only":
                targets, predictions = [], []
                with torch.no_grad():
                    for data, label in batches:
                        data = data.float().cuda()
                        if args.fps_diagnostics:
                            points, fps_stats = fps_with_diagnostics(data, 1024, baseline, torch)
                            merge_fps_diagnostics(config, active, fps_stats)
                        else:
                            points = baseline.misc.fps(data, 1024)
                        pred = base_model.module.classification_only(points, only_unmasked=False).argmax(-1).view(-1)
                        target = label.cuda().view(-1)
                        batch_observer(target, pred)
                        targets.append(target.cpu())
                        predictions.append(pred.cpu())
                targets, predictions = torch.cat(targets), torch.cat(predictions)
            elif is_preprocessing_identity_method(args.method):
                targets, predictions = [], []
                with torch.no_grad():
                    for data, label in batches:
                        points = preprocessing_identity_points(data.float(), baseline, args, torch)
                        pred = base_model.module.classification_only(points, only_unmasked=False).argmax(-1).view(-1)
                        target = label.cuda().view(-1)
                        batch_observer(target, pred)
                        targets.append(target.cpu())
                        predictions.append(pred.cpu())
                targets, predictions = torch.cat(targets), torch.cat(predictions)
            elif args.method in PURE_VAE_METHODS:
                targets, predictions = [], []
                with torch.no_grad():
                    for data, label in batches:
                        points = pure_vae_encode_decode_points(
                            data.float(), baseline, lion, args, torch)
                        pred = base_model.module.classification_only(points, only_unmasked=False).argmax(-1).view(-1)
                        target = label.cuda().view(-1)
                        batch_observer(target, pred)
                        targets.append(target.cpu())
                        predictions.append(pred.cpu())
                targets, predictions = torch.cat(targets), torch.cat(predictions)
            elif args.method == SHARED_DECODER_METHOD:
                targets, predictions = [], []
                decoder_totals = dict(
                    n_examples=0, original_style_n_correct=0,
                    updated_style_n_correct=0, disagreement_n=0,
                    decoder_output_difference_sum=0.0,
                    style_displacement_sum=0.0)
                with torch.no_grad():
                    for data, label in batches:
                        original_points, updated_points, trajectory, original_decoder, updated_decoder = shared_trajectory_decoder_points(
                            data.float(), baseline, lion, args, torch, steps,
                            scheduler_observer=scheduler_observer)
                        target, original_pred, updated_pred = classify_decoder_variants(
                            base_model, original_points, updated_points, label, torch, np)
                        batch_metrics = decoder_control_batch_metrics(
                            target, original_pred, updated_pred, original_decoder, updated_decoder,
                            trajectory.original_style, trajectory.updated_style)
                        decoder_totals["n_examples"] += batch_metrics["n_examples"]
                        decoder_totals["original_style_n_correct"] += batch_metrics["original_style_n_correct"]
                        decoder_totals["updated_style_n_correct"] += batch_metrics["updated_style_n_correct"]
                        decoder_totals["disagreement_n"] += batch_metrics["disagreement_n"]
                        decoder_totals["decoder_output_difference_sum"] += (
                            batch_metrics["decoder_output_difference"] * batch_metrics["n_examples"])
                        decoder_totals["style_displacement_sum"] += (
                            batch_metrics["style_displacement"] * batch_metrics["n_examples"])
                        batch_observer(target, updated_pred)
                        targets.append(target.cpu())
                        predictions.append(updated_pred.cpu())
                targets, predictions = torch.cat(targets), torch.cat(predictions)
            else:
                targets, predictions = baseline.process_batches(
                    batches, base_model, lion, args, steps,
                    scheduler_observer=scheduler_observer, batch_observer=batch_observer)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if counters["n"] != targets.numel() or counters["correct"] != int((predictions == targets).sum().item()):
                raise RuntimeError("Batch counters disagree with returned predictions.")
            status = "complete" if counters["n"] == len(dataset) else "partial"
            row = corruption_row(config["run_id"], args.seed, active,
                                 counters["n"], counters["correct"], elapsed,
                                 torch.cuda.max_memory_allocated() / (1024 ** 2), status,
                                 method=args.method, severity=config["severity"], dataset=config["dataset"])
            if decoder_totals is not None:
                decoder_metrics = decoder_control_row(
                    decoder_totals["n_examples"],
                    decoder_totals["original_style_n_correct"],
                    decoder_totals["updated_style_n_correct"],
                    decoder_totals["disagreement_n"],
                    decoder_totals["decoder_output_difference_sum"] / decoder_totals["n_examples"],
                    decoder_totals["style_displacement_sum"] / decoder_totals["n_examples"])
                row.update(decoder_metrics)
                config.setdefault("decoder_control", {})[active] = decoder_metrics
            rows.append(row)
            started = None
            bundle.write_results(rows, "running")
            record_modes("module_inventory_after")
            bundle.write_config(config)
            print("Corruption result:", rows[-1], flush=True)
        final_status = "partial" if any(row["status"] == "partial" for row in rows) else "complete"
        config.update(status=final_status, execution_status="complete",
                      completed_corruptions=[row["corruption"] for row in rows])
        bundle.write_results(rows, final_status)
        bundle.write_config(config)
        (bundle.path / "notes.md").write_text(notes_for_run(args), encoding="utf-8")
    except BaseException:
        config.update(status="failed", execution_status="failed",
                      failed_corruption=active,
                      completed_corruptions=[row["corruption"] for row in rows])
        if active is not None and started is not None:
            memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch is not None else 0.0
            rows.append(corruption_row(config["run_id"], args.seed, active,
                                       counters["n"], counters["correct"],
                                       time.perf_counter() - started, memory, "failed",
                                       method=args.method, severity=config["severity"], dataset=config["dataset"]))
            if args.method == SHARED_DECODER_METHOD and decoder_totals is not None:
                n = decoder_totals["n_examples"]
                rows[-1].update(decoder_control_row(
                    n, decoder_totals["original_style_n_correct"],
                    decoder_totals["updated_style_n_correct"],
                    decoder_totals["disagreement_n"],
                    decoder_totals["decoder_output_difference_sum"] / n if n else 0.0,
                    decoder_totals["style_displacement_sum"] / n if n else 0.0))
                config.setdefault("decoder_control", {})[active] = {
                    key: value for key, value in rows[-1].items()
                    if key not in {"run_id", "dataset", "method", "seed", "severity", "corruption"}}
        bundle.write_results(rows, "failed")
        bundle.write_config(config)
        with (bundle.path / "notes.md").open("a", encoding="utf-8") as file:
            file.write("\nFAILED: incomplete evidence; inspect stdout.log traceback. No resume.\n")
        traceback.print_exc()
        raise


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--pointmae_config", default="./cfgs/tta_modelnet.yaml")
    parser.add_argument("--pointmae_ckpt", default="./pointnet_ckpts/modelnet_jt.pth")
    parser.add_argument("--diff_config", default="./lion_ckpts/unconditional_all55_cfg.yml")
    parser.add_argument("--diff_ckpt", default="./lion_ckpts/epoch_10999_iters_2100999.pt")
    parser.add_argument("--dataset_root", default="./data/modelnet40_c")
    parser.add_argument("--label_path", default="./data/modelnet40_c/label.npy")
    parser.add_argument("--gamma", type=float, default=0.01, help="Original local-latent step size")
    parser.add_argument("--eta", type=float, default=0.01, help="Original style-conditioning step size")
    parser.add_argument("--lambdaa", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=("3dd_original", "source_only", "preprocessing_identity",
                                             "preprocessing_identity_seed_stability",
                                             "pure_vae_encode_decode", "pure_vae_seed_stability",
                                             SHARED_DECODER_METHOD),
                        default="3dd_original")
    parser.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS + (CLEAN_CONTROL,), default=["gaussian"])
    parser.add_argument("--max-batches", type=int, default=2, help="0 evaluates all batches; otherwise a prefix")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--result-root", default="./result")
    parser.add_argument("--dataset-name", choices=("modelnet-c", "shapenet-c", "scanobjectnn-c"), default="modelnet-c")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--fps-diagnostics", action="store_true",
                        help="Record legacy FPS index diagnostics without changing classifier inputs.")
    parser.add_argument("--lion-eval-mode", action="store_true", help="Set LION VAE and priors to eval mode; default preserves legacy mode.")
    parser.add_argument("--lion-ema-mode", action="store_true", help="Load prior EMA parameters from the LION checkpoint.")
    args = parser.parse_args(argv)
    validate_selection(args.corruptions)
    args.clean_control = args.corruptions == [CLEAN_CONTROL]
    if CLEAN_CONTROL in args.corruptions and not args.clean_control:
        parser.error("original is a standalone clean control and cannot be mixed with corruptions.")
    if args.clean_control and args.method != "source_only":
        parser.error("original clean control supports only --method source_only.")
    if args.clean_control and args.max_batches != 0:
        parser.error("original clean control requires --max-batches 0 for a complete evaluation.")
    if args.fps_diagnostics and args.method != "source_only":
        parser.error("--fps-diagnostics is currently supported only with --method source_only.")
    if args.method == "preprocessing_identity":
        if args.dataset_name != "modelnet-c" or args.severity != 5:
            parser.error("preprocessing_identity is locked to ModelNet40-C severity 5.")
        if args.batch_size != 32 or args.seed != 0 or args.max_batches != 0:
            parser.error("preprocessing_identity is locked to batch 32, seed 0, and complete evaluation.")
        if args.corruptions != list(CORRUPTIONS):
            parser.error("preprocessing_identity requires the complete canonical all-15 corruption set.")
        if args.lion_eval_mode or args.lion_ema_mode:
            parser.error("preprocessing_identity bypasses LION; LION mode flags are invalid.")
        if (args.gamma, args.eta, args.lambdaa) != (0.01, 0.01, 0.95):
            parser.error("preprocessing_identity uses the locked baseline gamma, eta, and lambda values.")
    if args.method == "preprocessing_identity_seed_stability":
        if args.dataset_name != "modelnet-c" or args.severity != 5:
            parser.error("preprocessing_identity_seed_stability is locked to ModelNet40-C severity 5.")
        if args.batch_size != 32 or args.seed not in (1, 2) or args.max_batches != 0:
            parser.error("preprocessing_identity_seed_stability is locked to batch 32, seed 1 or 2, and complete evaluation.")
        if args.corruptions != list(CORRUPTIONS):
            parser.error("preprocessing_identity_seed_stability requires the complete canonical all-15 corruption set.")
        if args.lion_eval_mode or args.lion_ema_mode:
            parser.error("preprocessing_identity_seed_stability bypasses LION and rejects LION mode flags.")
        if (args.gamma, args.eta, args.lambdaa) != (0.01, 0.01, 0.95):
            parser.error("preprocessing_identity_seed_stability uses the locked baseline gamma, eta, and lambda values.")
    if args.method == "pure_vae_encode_decode":
        if args.dataset_name != "modelnet-c" or args.severity != 5:
            parser.error("pure_vae_encode_decode is locked to ModelNet40-C severity 5.")
        if args.batch_size != 32 or args.seed != 0 or args.max_batches != 0:
            parser.error("pure_vae_encode_decode is locked to batch 32, seed 0, and complete evaluation.")
        if args.corruptions != list(CORRUPTIONS):
            parser.error("pure_vae_encode_decode requires the complete canonical all-15 corruption set.")
        if args.lion_eval_mode or args.lion_ema_mode:
            parser.error("pure_vae_encode_decode sets raw VAE eval mode and rejects LION mode flags.")
        if (args.gamma, args.eta, args.lambdaa) != (0.01, 0.01, 0.95):
            parser.error("pure_vae_encode_decode uses the locked baseline gamma, eta, and lambda values.")
    if args.method == "pure_vae_seed_stability":
        if args.dataset_name != "modelnet-c" or args.severity != 5:
            parser.error("pure_vae_seed_stability is locked to ModelNet40-C severity 5.")
        if args.batch_size != 32 or args.seed not in (1, 2) or args.max_batches != 0:
            parser.error("pure_vae_seed_stability is locked to batch 32, seed 1 or 2, and complete evaluation.")
        if args.corruptions != list(CORRUPTIONS):
            parser.error("pure_vae_seed_stability requires the complete canonical all-15 corruption set.")
        if args.lion_eval_mode or args.lion_ema_mode:
            parser.error("pure_vae_seed_stability sets raw VAE eval mode and rejects LION mode flags.")
        if (args.gamma, args.eta, args.lambdaa) != (0.01, 0.01, 0.95):
            parser.error("pure_vae_seed_stability uses the locked baseline gamma, eta, and lambda values.")
    if args.method == SHARED_DECODER_METHOD:
        if args.dataset_name != "modelnet-c" or args.severity != 5:
            parser.error("shared_trajectory_decoder_control is locked to ModelNet40-C severity 5.")
        if args.batch_size != 32 or args.seed not in (0, 1, 2) or args.max_batches != 0:
            parser.error("shared_trajectory_decoder_control is locked to batch 32, seed 0/1/2, and complete evaluation.")
        if (args.corruptions != list(SHARED_DECODER_PILOT_CORRUPTIONS)
                and args.corruptions != list(CORRUPTIONS)):
            parser.error(
                "shared_trajectory_decoder_control requires the Gaussian/Impulse pilot "
                "or the complete canonical all-15 corruption set.")
        if args.lion_ema_mode:
            parser.error("shared_trajectory_decoder_control requires raw LION weights; EMA is disabled.")
        if (args.gamma, args.eta, args.lambdaa) != (0.01, 0.01, 0.95):
            parser.error("shared_trajectory_decoder_control uses the locked baseline gamma, eta, and lambda values.")
        args.lion_eval_mode = True
    if args.batch_size < 1 or args.max_batches < 0 or not 0 <= args.seed < 2 ** 32:
        parser.error("Batch size must be positive, max-batches non-negative; seed must be in [0,2**32).")
    if not all(math.isfinite(v) and v >= 0 for v in (args.gamma, args.eta, args.lambdaa)) or not 0 < args.lambdaa <= 1:
        parser.error("Invalid rates or SCD percentile.")
    args.device = "cuda"
    default_name = "clean-control" if args.clean_control else (
        "source-only" if args.method == "source_only" else
        "preprocessing-identity" if args.method == "preprocessing_identity" else
        "preprocessing-identity-seed-stability" if args.method == "preprocessing_identity_seed_stability" else
        "pure-vae-encode-decode" if args.method == "pure_vae_encode_decode" else
        "pure-vae-seed-stability" if args.method == "pure_vae_seed_stability" else
        "shared-decoder-s5-gaussian-impulse" if args.method == SHARED_DECODER_METHOD else
        "baseline-smoke")
    args.run_name = args.run_name or (default_name + "_seed%s" % args.seed)
    return args


def build_config(args: argparse.Namespace) -> dict:
    """Build the immutable run configuration before any GPU work starts."""
    dataset = {"modelnet-c": "modelnet40_c", "shapenet-c": "shapenet_c",
               "scanobjectnn-c": "scanobjectnn_c"}[args.dataset_name]
    num_classes = {"modelnet-c": 40, "shapenet-c": 55, "scanobjectnn-c": 15}[args.dataset_name]
    stage = ("clean_control" if args.clean_control else
             "source_identity" if args.method == "source_only" else
             "preprocessing_identity" if args.method == "preprocessing_identity" else
             "preprocessing_identity_seed_stability" if args.method == "preprocessing_identity_seed_stability" else
             "pure_vae_encode_decode" if args.method == "pure_vae_encode_decode" else
             "pure_vae_seed_stability" if args.method == "pure_vae_seed_stability" else
             "shared_trajectory_decoder_control" if args.method == SHARED_DECODER_METHOD else "smoke")
    config = dict(
        stage=stage, dataset=dataset,
        severity=0 if args.clean_control else args.severity, method=args.method,
        seed=args.seed, batch_size=args.batch_size, corruptions=args.corruptions,
        classifier="pointmae", num_classes=num_classes, num_input_points=2048,
        num_classifier_points=1024, scale_factor=3.3885, ddim_total_steps=100,
        normal_reverse_steps=5, background_reverse_steps=35,
        gamma=args.gamma, eta=args.eta, lambda_cd=args.lambdaa,
        guidance_mapping=dict(gamma="local latent", eta="style condition"),
        final_decode_style=("identity; no decode" if args.method in PREPROCESSING_IDENTITY_METHODS else
                            "VAE decoder from encoded latents" if args.method in PURE_VAE_METHODS else
                            "shared final local latent; original and updated style" if args.method == SHARED_DECODER_METHOD else
                            "original shape_latent"),
        lion_mode_policy=("bypassed" if args.method in PREPROCESSING_IDENTITY_METHODS else
                          "raw VAE eval; priors bypassed" if args.method in PURE_VAE_METHODS else
                          "raw LION eval; EMA disabled" if args.method == SHARED_DECODER_METHOD else
                          "legacy; unchanged"),
        scheduler_config={}, spectral={}, projection={}, cli_args=vars(args),
        git_branch=command_output(["git", "branch", "--show-current"]),
        git_commit=command_output(["git", "rev-parse", "HEAD"]),
        git_status=command_output(["git", "status", "--porcelain"]),
        runtime_source_manifest={name: file_identity(REPO / name) for name in
                                 ("run_baseline.py", "research_artifacts.py", "main_3dd_tta.py",
                                  "tta.py", "utilities_3dd_tta.py", "models/lion.py")})
    if args.method in PREPROCESSING_IDENTITY_METHODS:
        config.update(lion_loaded=False, preprocessing=IDENTITY_PREPROCESSING)
        if args.method == "preprocessing_identity_seed_stability":
            config["seed_stability_reference"] = "preprocessing_identity seed0 archive"
    elif args.method in PURE_VAE_METHODS:
        config.update(
            lion_loaded=True,
            preprocessing=IDENTITY_PREPROCESSING,
            vae_contract="encode -> decompose_eps -> sample",
            prior_used=False)
        if args.method == "pure_vae_seed_stability":
            config["seed_stability_reference"] = "pure_vae_encode_decode seed0 archive"
    elif args.method == SHARED_DECODER_METHOD:
        config.update(
            lion_loaded=True,
            preprocessing=IDENTITY_PREPROCESSING,
            decoder_control_contract="one trajectory, two final-style decodes",
            decoder_control={},
            rng_control="snapshot/restore NumPy and Torch CPU/CUDA around second classifier call")
    return config


def main() -> int:
    args = parse_arguments()
    if Path.cwd().resolve() != REPO:
        raise SystemExit("Run from the repository root so baseline relative configs resolve correctly.")
    config = build_config(args)
    config["git_dirty"] = bool(config["git_status"])
    command = "cwd: " + str(REPO) + "\n" + shlex.join([sys.executable] + sys.argv)
    bundle = RunBundle.create(Path(args.result_root), args.run_name, config, command)
    print("Run directory:", bundle.path, flush=True)
    worker_command = [sys.executable, "-u", "-c",
                      "from run_baseline import run_worker; import sys; run_worker(sys.argv[1])",
                      str(bundle.path)]
    code = 1
    process = None
    try:
        # Subprocess pipe also catches extension/compiler/native stderr, unlike redirect_stdout.
        with (bundle.path / "stdout.log").open("ab") as log:
            process = subprocess.Popen(worker_command, cwd=REPO, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            try:
                for line in iter(process.stdout.readline, b""):
                    log.write(line)
                    log.flush()
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
    except BaseException:
        with (bundle.path / "stdout.log").open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        code = 130 if isinstance(sys.exc_info()[1], KeyboardInterrupt) else 1
    if code != 0:
        final = json.loads((bundle.path / "config.json").read_text(encoding="utf-8"))
        final.update(status="failed", execution_status="failed", worker_exit_code=code)
        bundle.write_config(final)
        bundle.mark_failed()
        with (bundle.path / "notes.md").open("a", encoding="utf-8") as file:
            file.write("\nWorker exited with code %s. Artifact may be incomplete.\n" % code)
    archive = bundle.path.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as file:
        for path in sorted(bundle.path.iterdir()):
            file.write(path, arcname=bundle.path.name + "/" + path.name)
    print("ZIP to provide:", archive, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
