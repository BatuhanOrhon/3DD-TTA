"""Paired Colab runner for the locked GSD composition experiment protocol."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import time
import traceback
import uuid

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gsd_composition_protocol import (
    METHOD,
    build_phase_plan,
    build_split_manifest,
    canonical_sha256,
    derive_draw_seeds,
    experiment_fingerprint,
    read_manifest,
    validate_arm_bundle,
    validate_all15_reference,
    validate_resume_block,
    validate_selection_manifest,
    write_arm_bundle,
    write_arm_zip,
    write_manifest,
)


PHASES = ("smoke", "diagnose", "scale", "routing", "projection", "replicate", "all15")


def required_reference_phase(phase: str, selection=None):
    """Return the nearest phase whose compatible completed controls may be reused."""
    if phase == "smoke":
        return None
    if phase in ("diagnose", "scale"):
        return "smoke" if phase == "diagnose" else "diagnose"
    if phase == "routing":
        return "diagnose"
    if phase == "projection":
        return "routing"
    if phase == "replicate":
        return (selection or {}).get("source_phase", "routing")
    if phase == "all15":
        return "replicate"
    raise ValueError("unknown phase: " + str(phase))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--result-root", default="./result/modelnet40_c/" + METHOD)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--selection-manifest")
    parser.add_argument("--scale-corruptions", nargs="+",
                        choices=("gaussian", "impulse", "background", "shear"),
                        help="optional subset of scale-phase corruptions")
    parser.add_argument("--scale-arms", nargs="+",
                        choices=("SCALE_0", "SCALE_1", "SCALE_100", "SCALE_1000"),
                        help="optional subset of scale-phase weight arms")
    parser.add_argument("--continue-on-arm-error", action="store_true",
                        help="scale only: record a failed arm and continue remaining selected arms")
    parser.add_argument("--execute", action="store_true",
                        help="run inference; omission plans the phase without loading models")
    parser.add_argument("--dataset-root", default="./data/modelnet40_c")
    parser.add_argument("--label-path", default="./data/modelnet40_c/label.npy")
    parser.add_argument("--pointmae-config", default="./cfgs/tta_modelnet.yaml")
    parser.add_argument("--pointmae-checkpoint", default="./pointnet_ckpts/modelnet_jt.pth")
    parser.add_argument("--lion-config", default="./lion_ckpts/unconditional_all55_cfg.yml")
    parser.add_argument("--lion-checkpoint", default="./lion_ckpts/epoch_10999_iters_2100999.pt")
    return parser


def _git_value(*args):
    try:
        return subprocess.run(["git", *args], cwd=str(_REPO_ROOT), check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": _file_sha256(path)}


def _source_manifest():
    names = ("scripts/run_gsd_composition.py", "gsd_composition_protocol.py",
             "gsd_composition.py", "gsd_paired_inputs.py", "tta_gsd.py",
             "tta.py", "graph_spectral.py", "run_baseline.py", "main_3dd_tta.py")
    result = {}
    for name in names:
        path = _REPO_ROOT / name
        if path.is_file():
            result[name] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
    return result


def _load_input_manifest(path, name):
    if path is None:
        return None
    return read_manifest(Path(path).expanduser().resolve())


def _check_reference(phase, selection, reference, reference_path=None):
    expected = required_reference_phase(phase, selection)
    if expected is None:
        if reference is not None:
            raise ValueError("smoke is a root phase and does not accept a reference manifest")
        return
    if reference is None:
        raise ValueError("{} requires --reference-manifest for completed {} controls".format(
            phase, expected))
    if reference.get("phase") != expected:
        raise ValueError("reference manifest must be a completed {} phase".format(expected))
    if reference.get("status") != "complete":
        raise ValueError("partial or failed reference manifests cannot be reused")
    if not reference.get("experiment_fingerprint"):
        raise ValueError("reference manifest has no experiment fingerprint")
    if phase == "all15":
        validate_all15_reference(selection, reference)
        if reference_path is None:
            raise ValueError("all15 requires a reference manifest path for bundle validation")
        root = Path(reference_path).expanduser().resolve().parent
        phase_arm_configs = {
            arm["arm_id"]: arm.get("config")
            for arm in reference.get("arms", [])
            if isinstance(arm, dict) and isinstance(arm.get("arm_id"), str)
        }
        for block in reference["paired_blocks"]:
            identity = block["identity"]
            for row in block["arms"]:
                if not isinstance(row, dict) or row.get("status") != "complete":
                    raise ValueError("all15 reference has an incomplete arm record")
                bundle_path = Path(row["bundle_path"]).expanduser()
                if not bundle_path.is_absolute():
                    bundle_path = root / bundle_path
                try:
                    bundle = validate_arm_bundle(bundle_path)
                except (OSError, ValueError, KeyError) as error:
                    raise ValueError("invalid all15 reference arm bundle: {}".format(
                        bundle_path)) from error
                manifest = bundle["manifest"]
                if (manifest.get("phase") != "replicate" or
                        manifest.get("status") != "complete" or
                        manifest.get("arm_id") != row["arm_id"] or
                        manifest.get("identity") != identity or
                        row.get("identity") != identity or
                        manifest.get("config") != phase_arm_configs.get(row["arm_id"]) or
                        manifest.get("experiment_fingerprint") != reference.get("config_sha256") or
                        manifest.get("runtime_fingerprint") != reference.get("runtime_fingerprint") or
                        manifest.get("locked_config_fingerprint") != reference.get("pairing_config_sha256") or
                        manifest.get("corruption") != block.get("corruption") or
                        manifest.get("seed") != block.get("seed") or
                        manifest.get("original_indices") != block["indices"] or
                        bundle["indices"].astype(int).tolist() != block["indices"] or
                        manifest.get("block_id") != block["block_id"]):
                    raise ValueError("all15 reference bundle does not match its paired block identity")


def build_plan(argv=None):
    """Resolve phase identities without model/CUDA execution or device calls."""
    args = _parser().parse_args(argv)
    selection = _load_input_manifest(args.selection_manifest, "selection")
    if args.phase in ("replicate", "all15"):
        validate_selection_manifest(selection)
    elif selection is not None:
        raise ValueError("selection manifests are only valid for replicate/all15")
    if args.phase != "scale" and (args.scale_corruptions or args.scale_arms or
                                    args.continue_on_arm_error):
        raise ValueError("scale filters and arm-error continuation are only valid for scale")
    reference = _load_input_manifest(args.reference_manifest, "reference")
    _check_reference(args.phase, selection, reference, args.reference_manifest)
    phase_plan = build_phase_plan(
        args.phase, selection, scale_corruptions=args.scale_corruptions,
        scale_arm_ids=args.scale_arms)

    source_manifest = _source_manifest()
    git_branch = _git_value("branch", "--show-current")
    git_commit = _git_value("rev-parse", "HEAD")
    resolved = {
        "method": METHOD,
        "dataset": "modelnet40_c",
        "severity": 5,
        "batch_size": 32,
        "ddim_total_steps": 100,
        "normal_reverse_steps": 5,
        "background_reverse_steps": 35,
        "gamma": .01,
        "eta": .01,
        "lambda": .95,
        "graph": {"profile": "hard-v1", "k": 10, "delta": .1,
                  "graph_gamma": .6, "modes": 100},
        "model": {"classifier": "frozen Point-MAE", "lion": "raw frozen/eval, EMA off",
                  "global_prior": "unused", "decoder_style": "original encoded shape latent"},
        "paths": {
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "label_path": str(Path(args.label_path).expanduser().resolve()),
            "pointmae_config": str(Path(args.pointmae_config).expanduser().resolve()),
            "pointmae_checkpoint": str(Path(args.pointmae_checkpoint).expanduser().resolve()),
            "lion_config": str(Path(args.lion_config).expanduser().resolve()),
            "lion_checkpoint": str(Path(args.lion_checkpoint).expanduser().resolve()),
        },
    }
    shared_config = dict(method=METHOD, dataset=resolved["dataset"], severity=5,
                         source_manifest=source_manifest, git_commit=git_commit,
                         runtime=resolved, draw_derivation_version=1)
    runtime_fingerprint = experiment_fingerprint(shared_config)
    phase_config = dict(shared_config, phase=args.phase,
                        arms=[dict(arm_id=arm.arm_id, config=asdict(arm.config))
                              for arm in phase_plan.arms],
                        corruptions=list(phase_plan.corruptions),
                        indices=list(phase_plan.indices), seeds=list(phase_plan.seeds),
                        selection_sha256=(selection or {}).get("selection_sha256"))
    fingerprint = experiment_fingerprint(phase_config)
    blocks = []
    for corruption in phase_plan.corruptions:
        for seed in phase_plan.seeds:
            selected_indices = list(phase_plan.indices)
            batches = []
            for offset in range(0, len(selected_indices), resolved["batch_size"]):
                batch_indices = selected_indices[offset:offset + resolved["batch_size"]]
                draw = derive_draw_seeds(resolved["dataset"], corruption, seed, batch_indices)
                batches.append(dict(
                    batch_index=len(batches), indices=batch_indices,
                    preparation_key=draw["preparation_key"],
                    preparation_seed=draw["preparation_seed"],
                    classification_key=draw["classification_key"],
                    classification_seed=draw["classification_seed"],
                    derivation_version=draw["derivation_version"],
                ))
            block_identity = dict(
                fingerprint=runtime_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
                locked_config_fingerprint=runtime_fingerprint,
                seed=int(seed),
                indices=selected_indices,
                input_sha256=None,
                preparation_keys=[batch["preparation_key"] for batch in batches],
                classification_keys=[batch["classification_key"] for batch in batches],
            )
            block_id = canonical_sha256(dict(corruption=corruption, seed=seed,
                                             indices=selected_indices,
                                             runtime_fingerprint=runtime_fingerprint))
            blocks.append(dict(
                block_id=block_id, corruption=corruption, seed=int(seed),
                indices=selected_indices,
                arm_ids=[arm.arm_id for arm in phase_plan.arms],
                identity=block_identity,
                preparation_key=canonical_sha256(block_identity["preparation_keys"]),
                classification_key=canonical_sha256(block_identity["classification_keys"]),
                batches=batches,
            ))

    result_root = Path(args.result_root).expanduser().resolve()
    command = [sys.executable, str((_REPO_ROOT / "scripts/run_gsd_composition.py").resolve()),
               "--phase", args.phase, "--result-root", str(result_root)]
    if args.reference_manifest:
        command.extend(("--reference-manifest", str(Path(args.reference_manifest).expanduser().resolve())))
    if args.selection_manifest:
        command.extend(("--selection-manifest", str(Path(args.selection_manifest).expanduser().resolve())))
    if args.scale_corruptions:
        command.extend(("--scale-corruptions", *args.scale_corruptions))
    if args.scale_arms:
        command.extend(("--scale-arms", *args.scale_arms))
    if args.continue_on_arm_error:
        command.append("--continue-on-arm-error")
    command.extend(("--dataset-root", resolved["paths"]["dataset_root"],
                    "--label-path", resolved["paths"]["label_path"],
                    "--pointmae-config", resolved["paths"]["pointmae_config"],
                    "--pointmae-checkpoint", resolved["paths"]["pointmae_checkpoint"],
                    "--lion-config", resolved["paths"]["lion_config"],
                    "--lion-checkpoint", resolved["paths"]["lion_checkpoint"], "--execute"))
    return dict(
        method=METHOD, phase=args.phase, status="planned", scope=phase_plan.scope,
        resolved_ref={"branch": git_branch, "commit": git_commit,
                      "source_manifest": source_manifest},
        config=resolved, config_sha256=fingerprint,
        pairing_config_sha256=runtime_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        selection_manifest=selection, reference_manifest_path=(
            str(Path(args.reference_manifest).expanduser().resolve()) if args.reference_manifest else None),
        reference_manifest=reference,
        selection_manifest_path=(
            str(Path(args.selection_manifest).expanduser().resolve()) if args.selection_manifest else None),
        split_manifest=build_split_manifest(),
        corruptions=list(phase_plan.corruptions), indices=list(phase_plan.indices),
        seeds=list(phase_plan.seeds),
        arms=[dict(arm_id=arm.arm_id, question=arm.question, config=asdict(arm.config))
              for arm in phase_plan.arms],
        blocks=blocks, command=shlex.join(command), execute_requested=args.execute,
        continue_on_arm_error=args.continue_on_arm_error,
        result_root=str(result_root),
    )


def _next_attempt(root: Path, phase: str):
    phase_root = root / phase
    attempt_index = 1
    while True:
        attempt_id = "attempt-{:04d}".format(attempt_index)
        attempt_path = phase_root / attempt_id
        try:
            attempt_path.mkdir(parents=True, exist_ok=False)
            return attempt_id, attempt_path
        except FileExistsError:
            attempt_index += 1


def _rng_seed(seed, random_module, numpy, torch):
    random_module.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def _preserve_rng_state(random_module, numpy, torch):
    """Restore all process RNG streams after an isolated seeded preparation."""
    python_state = random_module.getstate()
    numpy_state = numpy.random.get_state()
    torch_cpu_state = torch.get_rng_state()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random_module.setstate(python_state)
        numpy.random.set_state(numpy_state)
        torch.set_rng_state(torch_cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _postprocess_and_classify(points, seed, seed_rng, postprocess, classify):
    """Replay the classification RNG before FPS/postprocessing and inference."""
    seed_rng(seed)
    classifier_points = postprocess(points)
    return classify(classifier_points)


def _graph_runtime_seconds(events):
    """Sum one graph-target runtime per sample, excluding per-step events."""
    total = 0.0
    for event in events:
        if event.get("kind") != "graph":
            continue
        value = event.get("graph_runtime_seconds")
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) or value < 0):
            continue
        total += float(value)
    return total


def _draw_key_fields(block):
    return {
        "preparation_key": block["preparation_key"],
        "classification_key": block["classification_key"],
    }


def _persist_partial_arm_on_failure(path, manifest, *, labels, predictions, logits,
                                    diagnostics, command="", environment=""):
    """Persist completed batch rows using the standard arm bundle schema."""
    if len(labels) == 0:
        return None
    partial_manifest = dict(manifest, status="partial")
    bundle_path = write_arm_bundle(
        path, partial_manifest, labels, predictions, logits, diagnostics,
        command=command, environment=environment)
    return bundle_path


def _validate_reusable_block(reference, block, reference_path):
    """Return compatible completed arm records; invalid/partial sets are not reused."""
    if not reference:
        return []
    matching = next((row for row in reference.get("paired_blocks", [])
                     if row.get("block_id") == block["block_id"]), None)
    if matching is None:
        return []
    reference_arm_ids = matching.get("arm_ids", [])
    shared_arm_ids = [arm_id for arm_id in block["arm_ids"] if arm_id in reference_arm_ids]
    if not shared_arm_ids:
        return []
    records = matching.get("arms", [])
    identity = matching.get("identity", {})
    try:
        for key, value in block["identity"].items():
            if key != "input_sha256" and identity.get(key) != value:
                return []
        validate_resume_block(records, reference_arm_ids, identity)
        for record in records:
            bundle_path = Path(record["bundle_path"])
            if not bundle_path.is_absolute():
                bundle_path = Path(reference_path).resolve().parent / bundle_path
            bundle = validate_arm_bundle(bundle_path)
            manifest = bundle["manifest"]
            if (manifest.get("arm_id") != record["arm_id"] or
                    manifest.get("identity") != identity or
                    manifest.get("original_indices") != block["indices"]):
                return []
            record["resolved_bundle_path"] = str(bundle_path)
            record["validated_bundle"] = bundle
        return [record for record in records if record["arm_id"] in shared_arm_ids]
    except (KeyError, OSError, ValueError):
        return []


def _execute_colab_plan(plan):
    """Run paired GPU inference; dry-run avoids model/CUDA execution and calls."""
    import numpy as np
    import torch
    import main_3dd_tta as baseline
    import run_baseline as legacy_runner
    from types import SimpleNamespace
    from gsd_composition import CompositionConfig
    from gsd_paired_inputs import prepare_guidance_inputs
    from run_baseline import tta_preprocess_points, tta_postprocess_points
    from tta_gsd import tta_gsd_reconstruct

    if not torch.cuda.is_available():
        raise RuntimeError("GSD composition execution requires CUDA; dry-run does not")
    asset_paths = {
        "classifier_config": Path(plan["config"]["paths"]["pointmae_config"]),
        "classifier_checkpoint": Path(plan["config"]["paths"]["pointmae_checkpoint"]),
        "lion_config": Path(plan["config"]["paths"]["lion_config"]),
        "lion_checkpoint": Path(plan["config"]["paths"]["lion_checkpoint"]),
        "labels": Path(plan["config"]["paths"]["label_path"]),
    }
    for corruption in plan["corruptions"]:
        asset_paths["data_" + corruption] = (
            Path(plan["config"]["paths"]["dataset_root"])
            / ("data_{}_5.npy".format(corruption)))
    asset_manifest = {name: _file_identity(path) for name, path in asset_paths.items()}
    args = SimpleNamespace(
        dataset_name="modelnet-c", severity=5, device="cuda", batch_size=32,
        gamma=.01, eta=.01, lambdaa=.95, lion_ema_mode=False,
        use_gpu=True, distributed=False, sync_bn=False,
        pointmae_config=plan["config"]["paths"]["pointmae_config"],
        pointmae_ckpt=plan["config"]["paths"]["pointmae_checkpoint"],
        diff_config=plan["config"]["paths"]["lion_config"],
        diff_ckpt=plan["config"]["paths"]["lion_checkpoint"],
        dataset_root=plan["config"]["paths"]["dataset_root"],
        label_path=plan["config"]["paths"]["label_path"],
    )
    base_model, lion = baseline.configure_model(args)
    base_model.eval().requires_grad_(False)
    lion.vae.eval().requires_grad_(False)
    lion.priors.eval().requires_grad_(False)
    if hasattr(baseline, "PointDataset"):
        dataset_type = baseline.PointDataset
    else:
        from datasets import PointDataset as dataset_type
    phase_path = Path(plan["attempt_path"])
    environment = "\n".join((
        "Python: " + sys.version,
        "Platform: " + platform.platform(),
        "Torch: " + torch.__version__,
        "CUDA: " + str(torch.version.cuda),
        "GPU: " + torch.cuda.get_device_name(),
        "Git branch: " + str(plan["resolved_ref"].get("branch")),
        "Git commit: " + str(plan["resolved_ref"].get("commit")),
    ))
    native_extensions = legacy_runner.extension_inventory()
    runtime_identity = dict(
        asset_manifest=asset_manifest,
        native_extensions=native_extensions,
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=str(torch.version.cuda),
        gpu=torch.cuda.get_device_name(),
    )
    runtime_fingerprint = experiment_fingerprint(dict(
        planned_runtime_fingerprint=plan["runtime_fingerprint"],
        runtime_identity=runtime_identity))
    plan["runtime_identity"] = runtime_identity
    plan["runtime_fingerprint"] = runtime_fingerprint
    for block in plan["blocks"]:
        block["identity"]["fingerprint"] = runtime_fingerprint
        block["identity"]["runtime_fingerprint"] = runtime_fingerprint
        block["identity"]["locked_config_fingerprint"] = plan["pairing_config_sha256"]
    reference = plan.get("reference_manifest")
    dataset_cache = {}
    completed_blocks = []
    for block in plan["blocks"]:
        corruption, seed = block["corruption"], block["seed"]
        if corruption not in dataset_cache:
            dataset_cache[corruption] = dataset_type(
                args.dataset_root, args.label_path, corruption, severity=5)
        dataset = dataset_cache[corruption]
        if len(dataset.data) != len(dataset.labels) or len(dataset.data) != 2468:
            raise ValueError("composition phases require 2468 aligned ModelNet40-C examples")
        data_by_index = np.asarray(dataset.data)
        labels_by_index = np.asarray(dataset.labels)
        if max(block["indices"]) >= len(data_by_index):
            raise ValueError("phase manifest contains an index outside the dataset")
        prepared_batches = []
        input_hashes = []
        preparation_seconds = 0.0
        for batch in block["batches"]:
            indices = batch["indices"]
            data = torch.from_numpy(np.asarray(data_by_index[indices])).float()
            labels = torch.from_numpy(np.asarray(labels_by_index[indices]).astype(np.int64))
            preparation_started = time.perf_counter()
            with _preserve_rng_state(random, np, torch):
                _rng_seed(batch["preparation_seed"], random, np, torch)
                inputs, center, maximum = tta_preprocess_points(
                    data, baseline, args, torch)
                prepared = prepare_guidance_inputs(
                    inputs, lion, total=100,
                    steps_back_local=35 if corruption == "background" else 5)
            preparation_seconds += time.perf_counter() - preparation_started
            input_hashes.append(prepared.input_sha256)
            prepared_batches.append((indices, labels, inputs, center, maximum, prepared,
                                     batch["classification_seed"]))
        actual_identity = dict(block["identity"])
        actual_identity["input_sha256"] = canonical_sha256(input_hashes)

        matching = []
        if reference is not None:
            matching = _validate_reusable_block(reference, block, plan["reference_manifest_path"])
            # Reference identity includes the actual paired preparation hashes.
            if matching and matching[0]["identity"].get("input_sha256") != actual_identity["input_sha256"]:
                matching = []
        reused_by_id = {record["arm_id"]: record for record in matching}
        arm_records = []
        for arm in plan["arms"]:
            arm_id = arm["arm_id"]
            if arm_id in reused_by_id:
                record = reused_by_id[arm_id]
                arm_records.append({
                    "arm_id": arm_id, "status": "complete", "identity": actual_identity,
                    "bundle_path": record["resolved_bundle_path"], "reused": True,
                })
                continue
            targets, predictions, logits_rows, diagnostics = [], [], [], []
            arm_started = time.perf_counter()
            sampling_seconds = 0.0
            classification_seconds = 0.0
            graph_seconds = 0.0
            peak_memory = 0
            torch.cuda.reset_peak_memory_stats()
            config = CompositionConfig(**arm["config"])
            completed_indices = []
            run_id = "{}_{}_seed{}_{}".format(plan["attempt_id"], corruption, seed, arm_id)
            arm_path = phase_path / "arms" / run_id
            try:
                for (indices, labels, inputs, center, maximum, prepared,
                     classification_seed) in prepared_batches:
                    steps = 35 if corruption == "background" else 5
                    events = []

                    def observe(event):
                        row = dict(event)
                        row["arm_id"] = arm_id
                        row["corruption"] = corruption
                        row["seed"] = int(seed)
                        local_index = row.get("sample_index")
                        if local_index is not None and local_index < len(indices):
                            row["original_index"] = int(indices[local_index])
                        events.append(row)

                    sampling_started = time.perf_counter()
                    points = tta_gsd_reconstruct(
                        inputs, lion, steps, args.gamma, args.eta, args.lambdaa, 100,
                        composition_config=config, prepared_inputs=prepared,
                        composition_observer=observe)
                    sampling_seconds += time.perf_counter() - sampling_started
                    classification_started = time.perf_counter()
                    with torch.no_grad():
                        batch_logits = _postprocess_and_classify(
                            points, classification_seed,
                            lambda value: _rng_seed(value, random, np, torch),
                            lambda value: tta_postprocess_points(
                                value, center, maximum, baseline, args),
                            lambda value: base_model.module.classification_only(
                                value, only_unmasked=False),
                        )
                        batch_prediction = batch_logits.argmax(dim=-1).view(-1)
                    classification_seconds += time.perf_counter() - classification_started
                    if not torch.isfinite(batch_logits).all():
                        raise FloatingPointError("classifier returned nonfinite logits")
                    if len(labels) != len(batch_prediction):
                        raise RuntimeError("paired block prediction count mismatch")
                    targets.append(labels.cpu())
                    predictions.append(batch_prediction.detach().cpu())
                    logits_rows.append(batch_logits.detach().cpu())
                    completed_indices.extend(int(index) for index in indices)
                    diagnostics.extend(events)
                    graph_seconds += _graph_runtime_seconds(events)
                    peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
            except BaseException as error:
                partial_path = None
                if targets:
                    partial_manifest = dict(
                        run_id=run_id, phase=plan["phase"], arm_id=arm_id,
                        corruption=corruption, seed=seed, status="partial",
                        original_indices=completed_indices,
                        expected_indices=list(block["indices"]),
                        input_sha256=actual_identity["input_sha256"], identity=actual_identity,
                        config=arm["config"], experiment_fingerprint=plan["config_sha256"],
                        runtime_fingerprint=plan["runtime_fingerprint"], scope=plan["scope"],
                        total_runtime_seconds=time.perf_counter() - arm_started,
                        sampling_seconds=sampling_seconds, graph_seconds=graph_seconds,
                        classification_seconds=classification_seconds,
                        preparation_seconds=preparation_seconds,
                        **_draw_key_fields(block),
                        locked_config_fingerprint=plan["pairing_config_sha256"],
                        runtime_identity=runtime_identity, asset_manifest=asset_manifest,
                        native_extension_manifest=native_extensions,
                        peak_gpu_memory_mb=peak_memory / (1024 ** 2),
                        block_id=block["block_id"],
                    )
                    partial_path = _persist_partial_arm_on_failure(
                        arm_path, partial_manifest,
                        labels=torch.cat(targets).numpy(),
                        predictions=torch.cat(predictions).numpy(),
                        logits=torch.cat(logits_rows).numpy(), diagnostics=diagnostics,
                        command=plan["command"], environment=environment,
                    )
                if plan.get("continue_on_arm_error") and isinstance(error, Exception):
                    failure = {
                        "arm_id": arm_id, "status": "failed",
                        "block_id": block["block_id"],
                        "corruption": corruption, "seed": int(seed),
                        "failure_type": type(error).__name__, "failure": str(error),
                        "completed_indices": completed_indices,
                        "bundle_path": str(partial_path) if partial_path else None,
                        "identity": actual_identity,
                    }
                    arm_records.append(failure)
                    plan.setdefault("arm_failures", []).append(failure)
                    print("Scale arm failed; recording and continuing: {} / {}: {}".format(
                        corruption, arm_id, str(error)), file=sys.stderr)
                    traceback.print_exc()
                    continue
                raise
            arm_elapsed = time.perf_counter() - arm_started
            labels_all = torch.cat(targets).numpy()
            predictions_all = torch.cat(predictions).numpy()
            logits_all = torch.cat(logits_rows).numpy()
            arm_manifest = dict(
                run_id=run_id, phase=plan["phase"], arm_id=arm_id,
                corruption=corruption, seed=seed,
                status="complete", original_indices=block["indices"],
                expected_indices=list(block["indices"]),
                input_sha256=actual_identity["input_sha256"], identity=actual_identity,
                config=arm["config"], experiment_fingerprint=plan["config_sha256"],
                runtime_fingerprint=plan["runtime_fingerprint"], scope=plan["scope"],
                total_runtime_seconds=arm_elapsed,
                sampling_seconds=sampling_seconds,
                graph_seconds=graph_seconds,
                classification_seconds=classification_seconds,
                preparation_seconds=preparation_seconds,
                **_draw_key_fields(block),
                locked_config_fingerprint=plan["pairing_config_sha256"],
                runtime_identity=runtime_identity,
                asset_manifest=asset_manifest,
                native_extension_manifest=native_extensions,
                peak_gpu_memory_mb=peak_memory / (1024 ** 2),
                block_id=block["block_id"],
            )
            write_arm_bundle(arm_path, arm_manifest, labels_all, predictions_all,
                             logits_all, diagnostics, command=plan["command"],
                             environment=environment)
            zip_path = write_arm_zip(arm_path)
            arm_records.append({
                "arm_id": arm_id, "status": "complete", "identity": actual_identity,
                "bundle_path": str(arm_path), "zip_path": str(zip_path), "reused": False,
            })
        block_has_failures = any(record.get("status") != "complete" for record in arm_records)
        if not block_has_failures:
            validate_resume_block(arm_records, block["arm_ids"], actual_identity)
        completed_blocks.append(dict(
            block_id=block["block_id"], corruption=corruption, seed=seed,
            indices=block["indices"], identity=actual_identity, arms=arm_records,
            status="partial" if block_has_failures else "complete",
        ))
    has_failures = bool(plan.get("arm_failures"))
    return dict(status="partial" if has_failures else "complete",
                execution_status="finished_with_arm_failures" if has_failures else "complete",
                arm_failures=plan.get("arm_failures", []), paired_blocks=completed_blocks,
                completed_blocks=len(completed_blocks),
                runtime_fingerprint=runtime_fingerprint,
                runtime_identity=runtime_identity,
                asset_manifest=asset_manifest,
                native_extension_manifest=native_extensions,
                completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def _execute_plan(plan, worker=None):
    """Invoke the paired worker; the callback seam keeps orchestration CPU-testable."""
    return (worker or _execute_colab_plan)(plan)


def main(argv=None, *, worker=None):
    args = _parser().parse_args(argv)
    plan = build_plan(argv)
    attempt_id, attempt_path = _next_attempt(Path(plan["result_root"]), plan["phase"])
    plan["attempt_id"] = attempt_id
    plan["attempt_path"] = str(attempt_path)
    manifest = dict(plan)
    manifest.pop("reference_manifest", None)
    manifest["manifest_path"] = str(attempt_path / "phase_manifest.json")
    manifest["attempt_id"] = attempt_id
    manifest["experiment_fingerprint"] = plan["config_sha256"]
    manifest["status"] = "running" if args.execute else "planned"
    write_manifest(attempt_path / "phase_manifest.json", manifest)
    preview = {
        "method": manifest["method"],
        "phase": manifest["phase"],
        "status": manifest["status"],
        "scope": manifest["scope"],
        "manifest_path": manifest["manifest_path"],
        "resolved_ref": manifest["resolved_ref"],
        "config": manifest["config"],
        "config_sha256": manifest["config_sha256"],
        "corruptions": manifest["corruptions"],
        "indices_count": len(manifest["indices"]),
        "seeds": manifest["seeds"],
        "arms": [arm["arm_id"] for arm in manifest["arms"]],
        "paired_block_count": len(manifest["blocks"]),
        "command": manifest["command"],
    }
    print(json.dumps(preview, indent=2, sort_keys=True, allow_nan=False))
    print("Phase manifest:", attempt_path / "phase_manifest.json")
    print("Resolved implementation ref:", plan["resolved_ref"])
    print("Execute command:", plan["command"])
    if not args.execute:
        return 0
    try:
        result = _execute_plan(plan, worker=worker)
        if isinstance(result, dict):
            manifest.update(result)
        manifest["status"] = (result.get("status", "complete")
                               if isinstance(result, dict) else "complete")
        write_manifest(attempt_path / "phase_manifest.json", manifest)
        return 0
    except BaseException as error:
        manifest.update(status="failed", execution_status="failed",
                        failure_type=type(error).__name__, failure=str(error))
        if plan.get("partial_arms"):
            manifest["partial_arms"] = plan["partial_arms"]
        if plan.get("arm_failures"):
            manifest["arm_failures"] = plan["arm_failures"]
        write_manifest(attempt_path / "phase_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
