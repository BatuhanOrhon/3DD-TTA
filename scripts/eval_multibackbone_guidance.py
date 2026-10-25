#!/usr/bin/env python3
"""Shared reconstruction with isolated frozen heads and locked method policies."""
from __future__ import annotations

import argparse
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

METHOD = "gsd_inspired_latent_spectral_guidance"
ORIGINAL_METHOD = "3dd_original"
BACKBONES = ("curvenet", "pointnext")
LION_SHA256 = "807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2"
LION_CONFIG_SHA256 = "9d4a826f35fe9b3053210105eb4857811c8bfd53cd8c6a998f8ea7953e975816"
PROTOCOL = dict(M_max=800, beta=2.0, gamma=0.01, eta=0.01, p=0.95, total=100,
                loss_weights=dict(spectral=1.17, invariant=0.0, chamfer=1.0),
                use_static_style=False, dynamic_graph=False, graph_update_interval=3)
ORIGINAL_PROTOCOL = dict(gamma=0.01, eta=0.01, p=0.95, total=100, scd_normalize=False)


def configurations(argv=None) -> list:
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
    parser.add_argument("--pointmae-config")
    parser.add_argument("--pointmae-checkpoint")
    parser.add_argument("--batch-size", type=int, choices=(32, 70))
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--corruptions", nargs="+", default=["all"])
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.max_batches < 0:
        raise ValueError("Seed and max-batches must be nonnegative")
    original = args.method == "original3dd"
    if args.trajectory_observe and original:
        raise ValueError("Trajectory observation is restricted to GSD-inspired guidance")
    if "pointmae" in args.backbones and not args.trajectory_observe:
        raise ValueError("Point-MAE guidance is available only with --trajectory-observe")
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
        if backbone == "pointmae":
            config.update(pointmae_config=args.pointmae_config, pointmae_checkpoint=args.pointmae_checkpoint,
                          checkpoint=args.pointmae_checkpoint)
        if "original" in config["corruptions"]:
            raise ValueError("Guidance uses the predeclared corrupted scope only")
        config.update(vars(args), method=METHOD, batch_size=batch_size, lion_loaded=True,
                      corruptions=config["corruptions"], guidance_protocol=PROTOCOL,
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
        for key in ("dataset_root", "label_path", "asset_root", "result_root", "drive_root"):
            if config.get(key):
                config[key] = str(Path(config[key]).absolute())
        result.append(config)
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


def reconstruct_geometry(raw, dataset: str, corruption: str, lion, graph, reconstruct, ops, *, device="cuda", method=METHOD, scheduler_observer=None, observer_factory=None):
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
        decoded, metrics = reconstruct(x=prepared, lion=lion, graph_spectral_module=graph,
                                       steps_back_local=30 if corruption == "background" else 10, **observation, **PROTOCOL)
    geometry = decoded_to_geometry(decoded, dataset, scale, center, ops)
    if geometry.shape != (len(raw), 2048, 3) or not torch.isfinite(geometry).all():
        raise ValueError("Reconstruction must preserve finite decoded Bx2048x3 geometry")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite guidance metrics")
    return geometry.detach(), metrics


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
    if config.get("trajectory_observation") and config["backbone"] == "pointmae":
        return dict(status="unavailable", reason="No validated full ScanObjectNN-C Point-MAE source-only row in this study; primary comparator is same-run terminal")
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
        from graph_spectral_v2 import GraphSpectralDNA
        from tta_gsd_v2 import tta_gsd_reconstruct
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


def worker(directories: list) -> None:
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
            bundles[name].write_config(config)
        for corruption, values in arrays.items():
            active = corruption
            accumulated = {name: [] for name in heads}
            elapsed = {name: 0. for name in heads}
            correct = {name: 0 for name in heads}
            seen, peak = 0, 0.
            for batch, start in enumerate(range(0, len(values), common["batch_size"])):
                if common["max_batches"] and batch >= common["max_batches"]:
                    break
                end = min(start + common["batch_size"], len(values))
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                before_rng = rng_hash()
                started = time.monotonic()
                raw = torch.from_numpy(np.array(values[start:end]))
                scheduler_record = {}
                observers = []
                def observer_factory(center, scale):
                    from shared_trajectory import SharedTrajectoryObserver
                    def classify_decoded(decoded):
                        geometry = decoded_to_geometry(decoded, common["dataset"], scale, center, ops)
                        return classify_geometry(geometry, heads, configs, fps)[0]
                    observer = SharedTrajectoryObserver(lion.vae, heads, classify_decoded,
                                                        30 if corruption == "background" else 10,
                                                        num_classes=common["num_classes"])
                    observers.append(observer)
                    return observer
                observer_options = {"observer_factory": observer_factory} if common.get("trajectory_observation") else {}
                geometry, metrics = reconstruct_geometry(raw, common["dataset"], corruption, lion, graph, reconstruct, ops,
                                                         method=common["method"], scheduler_observer=scheduler_record.update, **observer_options)
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
                if scheduler_record:
                    shared["scheduler"] = scheduler_record
                    for config in configs.values():
                        recorded = config.setdefault("scheduler_records", {}).setdefault(corruption, scheduler_record)
                        if recorded != scheduler_record:
                            raise RuntimeError("Original scheduler changed within a corruption")
                if observers:
                    shared["trajectory"] = observers[0].diagnostics()
                pending = {}
                seen += end - start
                for name, logits in all_logits.items():
                    prediction = logits.argmax(-1).cpu().numpy()
                    payload = dict(index=np.arange(start, end, dtype=np.int64), label=np.array(labels[start:end], dtype=np.int64),
                                   prediction=prediction, logits=logits.cpu().numpy())
                    accumulated[name].append(payload)
                    correct[name] += int((prediction == labels[start:end]).sum())
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
            config.update(lion_state_after=lion_after, status="complete" if len(common["corruptions"]) == 15 and not common["max_batches"] else "partial")
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


def main(argv=None) -> None:
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
        run_name = (("mntraj-" if config.get("trajectory_observation") and config["dataset"] == "modelnet40_c" else "")
                    + config["backbone"] + "-seed" + str(config["seed"]))
        bundle = source.RunBundle.create(Path(config["result_root"]), run_name, config,
                                          shlex.join([sys.executable, str(Path(__file__).resolve()), *arguments]))
        bundles.append(bundle)
        print("Run bundle:", bundle.path, flush=True)
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
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker-bundles", json.dumps([str(bundle.path) for bundle in bundles])],
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


if __name__ == "__main__":
    main()
