#!/usr/bin/env python3
"""Opt-in frozen source-only study; preview uses only the standard library."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from research_artifacts import CORRUPTIONS, RunBundle, corruption_row, validate_selection


def file_identity(path: Path) -> dict:
    """Stream identities without importing the heavy baseline runner."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def asset_path(root: Path, relative: str) -> Path:
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("Asset symlinks are not permitted")
    root = root.resolve(strict=True)
    relative = Path(relative)
    candidate = root / relative
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Asset path is outside asset root")
    for part in (candidate, *candidate.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Asset symlinks are not permitted")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("Asset path is outside asset root or is not a file")
    return resolved


def verify_identity(path: Path, expected: dict) -> dict:
    observed = file_identity(path)
    if observed["sha256"] != expected.get("sha256"):
        raise ValueError("SHA-256 mismatch: " + str(path))
    if observed["bytes"] != expected.get("bytes"):
        raise ValueError("Byte count mismatch: " + str(path))
    return observed


def read_manifest(root: Path, expected_sha256: str) -> dict:
    path = asset_path(root, "manifest.json")
    if file_identity(path)["sha256"] != expected_sha256:
        raise ValueError("Asset manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text())
    if not manifest.get("files") or not isinstance(manifest.get("datasets"), dict):
        raise ValueError("Manifest must declare files and datasets")
    for relative, expected in manifest["files"].items():
        verify_identity(asset_path(root, relative), expected)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("dgcnn", "curvenet", "pointnext", "pointmae"), required=True)
    parser.add_argument("--dataset", choices=("modelnet40_c", "scanobjectnn_c"), required=True)
    for name in ("dataset-root", "label-path", "asset-root", "result-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--asset-manifest-sha256")
    parser.add_argument("--drive-root", help="Verified immutable progress/final archive destination")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corruptions", nargs="+", default=["all"])
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--pointmae-config")
    parser.add_argument("--pointmae-checkpoint")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args(argv)


def resolve_config(args) -> dict:
    if args.batch_size < 1 or args.max_batches < 0 or args.seed < 0:
        raise ValueError("batch-size must be positive; max-batches and seed must be nonnegative")
    selected = list(CORRUPTIONS) if args.corruptions == ["all"] else validate_selection(args.corruptions)
    if "original" in selected:
        raise ValueError("This study runner evaluates the canonical corrupted inputs")
    if not args.preview and not re.fullmatch(r"[0-9a-f]{64}", args.asset_manifest_sha256 or ""):
        raise ValueError("Actual runs require --asset-manifest-sha256 with a pinned SHA-256")
    scan = args.dataset == "scanobjectnn_c"
    suffix = "scanobjectnn" if scan else "modelnet40"
    checkpoint = ("3deformrs_curvenet_scanobjectnn_better.t7" if scan and args.backbone == "curvenet"
                  else ("pointnext_" if args.backbone == "pointnext" else "svwa_" + args.backbone + "_") + suffix + ".pth")
    preprocessing = "raw XYZ -> native PointNet2 FPS(1024); no added normalization"
    if scan and args.backbone == "curvenet":
        preprocessing = "raw XYZ -> centroid and L-infinity normalization -> native PointNet2 FPS(1024); transfer from random sampling"
    if args.backbone == "pointnext":
        preprocessing = ("raw XYZ -> native PointNet2 FPS(1024) -> height=y-min(y) -> centroid and unit-sphere XYZ normalization -> XYZ + unscaled height features"
                         if scan else "raw XYZ -> native PointNet2 FPS(1024) -> XYZ features; no added normalization")
    if args.backbone == "pointmae" and not args.preview and not (args.pointmae_config and args.pointmae_checkpoint):
        raise ValueError("Point-MAE requires --pointmae-config and --pointmae-checkpoint")
    config = dict(vars(args))
    config.update(method="source_only", severity=5, corruptions=selected,
                  n_examples=581 if scan else 2468, num_classes=15 if scan else 40,
                  split="main_split" if scan else "official_modelnet40_test",
                  checkpoint="checkpoints/" + checkpoint,
                  input_layout="bcn" if scan and args.backbone == "curvenet" else "bnc",
                  preprocessing=preprocessing, n_points=1024, lion_loaded=False,
                  paper_checkpoint_identity_verified=False,
                  provenance_status="published alternative checkpoint; benchmark-transfer preprocessing, not exact GSDTTA reproduction",
                  pointnext_width=32 if scan else 64,
                  dgcnn_head_dataset_name="modelnet" if scan else "scanobject",
                  dgcnn_head_quirk="source dataset_name selects only linear3/linear3_ parameter name; does not specify data split",
                  sampling_policy="native PointNet2 FPS1024, including repeated indices if input has fewer points",
                  randomness_policy="seed-controlled, file order, no shuffle; not common-draw paired")
    if args.backbone == "pointmae":
        config["provenance_status"] = "existing Point-MAE control; checkpoint identity is supplied explicitly"
    for name in ("asset_root", "dataset_root", "label_path", "result_root"):
        config[name] = str(Path(config[name]).absolute())
    if config.get("drive_root"):
        config["drive_root"] = str(Path(config["drive_root"]).absolute())
    return config


def dataset_manifest(manifest: dict, config: dict) -> dict:
    dataset = manifest.get("datasets", {}).get(config["dataset"], {})
    if dataset.get("n_examples") != config["n_examples"]:
        raise ValueError("Dataset provenance example count mismatch")
    if config["dataset"] == "scanobjectnn_c" and dataset.get("split") != "main_split":
        raise ValueError("ScanObjectNN-C requires main_split provenance; nobg is forbidden")
    if not isinstance(dataset.get("files"), dict):
        raise ValueError("Dataset manifest must declare files")
    return dataset


def load_array(path: Path, expected: dict, config: dict, *, labels=False):
    import numpy as np
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Dataset symlinks are not permitted")
    verify_identity(path, expected)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != expected.get("shape") or len(values) != config["n_examples"]:
        raise ValueError("Dataset shape/count differs from pinned provenance: " + str(path))
    if labels:
        if values.shape not in ((config["n_examples"],), (config["n_examples"], 1)):
            raise ValueError("Labels must have shape N or Nx1")
        if not np.issubdtype(values.dtype, np.integer) or values.min() < 0 or values.max() >= config["num_classes"]:
            raise ValueError("Labels must be integer class indices within the declared class count")
        return values.reshape(-1)
    if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] != 3 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Point arrays must be floating BxNx3 XYZ")
    for start in range(0, len(values), 64):
        if not np.isfinite(values[start:start + 64]).all():
            raise ValueError("Nonfinite dataset coordinates")
    return values


def _declared_asset(root: Path, manifest: dict, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        try:
            relative = str(path.relative_to(root.resolve()))
        except ValueError:
            raise ValueError("Consumed asset must be inside asset-root: " + str(path))
    if relative not in manifest["files"]:
        raise ValueError("Consumed asset is absent from manifest: " + relative)
    return asset_path(root, relative)


def load_classifier(config: dict, manifest: dict):
    import torch
    from classifier_backbones import FrozenClassifier
    root = Path(config["asset_root"])
    backbone = config["backbone"]
    checkpoint = config["pointmae_checkpoint"] if backbone == "pointmae" else config["checkpoint"]
    checkpoint = _declared_asset(root, manifest, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if backbone in ("dgcnn", "curvenet"):
        scan_curve = backbone == "curvenet" and config["dataset"] == "scanobjectnn_c"
        package = "study_3deformrs" if scan_curve else "study_svwa"
        _declared_asset(root, manifest, "model_probe_sources/" + package + "/model.py")
        sys.path.insert(0, str(root / "model_probe_sources"))
        module = importlib.import_module(package + ".model")
        if scan_curve:
            model = module.CurveNet(num_classes=config["num_classes"], k=20, setting="default")
            state = payload
        else:
            options = SimpleNamespace(cls_dim=config["num_classes"], k=20, emb_dims=1024,
                                      dropout=0.5, transformer_config=SimpleNamespace(dataset_name=config["dgcnn_head_dataset_name"]))
            model = module.DGCNN_cls(options) if backbone == "dgcnn" else module.CurveNet(options)
            state = payload["base_model"]
        config["resolved_model_config"] = dict(classes=config["num_classes"], k=20, emb_dims=1024, dropout=0.5,
                                                package=package, dgcnn_head_dataset_name=config["dgcnn_head_dataset_name"])
    elif backbone == "pointnext":
        sys.path.insert(0, str(root))
        from openpoints.utils import EasyConfig
        from openpoints.models import build_model_from_cfg
        filename = "pointnext_scan_cfg.yaml" if config["dataset"] == "scanobjectnn_c" else "pointnext_modelnet_cfg.yaml"
        options = EasyConfig()
        options.load(str(_declared_asset(root, manifest, filename)), recursive=True)
        options.model.encoder_args.width = config["pointnext_width"]
        model = build_model_from_cfg(options.model)
        state = payload["model"]
        config["resolved_model_config"] = dict(options)
    else:
        from tools import builder
        from utils_mate.config import cfg_from_yaml_file
        options = cfg_from_yaml_file(str(_declared_asset(root, manifest, config["pointmae_config"])))
        options.model.cls_dim = config["num_classes"]
        model = builder.model_builder(options.model)
        state = payload["base_model"]
        config["resolved_model_config"] = dict(options)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint must contain a string-keyed state dictionary")
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    config["checkpoint_load"] = dict(weights_only=True, strict=True,
                                     missing_keys=list(incompatible.missing_keys), unexpected_keys=list(incompatible.unexpected_keys),
                                     checkpoint_sha256=file_identity(checkpoint)["sha256"])
    return FrozenClassifier(model.cuda(), backbone, config["num_classes"], input_layout=config["input_layout"])


def preprocess(points, config: dict, fps):
    """The predeclared transfer contract, shared with later matched guidance."""
    import torch
    if config["backbone"] == "curvenet" and config["dataset"] == "scanobjectnn_c":
        points = points - points.mean(dim=1, keepdim=True)
        scale = points.abs().amax(dim=(1, 2), keepdim=True)
        if torch.any(scale == 0):
            raise ValueError("Cannot L-infinity normalize a degenerate cloud")
        points = points / scale
    points = fps(points, 1024)
    features = None
    if config["backbone"] == "pointnext":
        if config["dataset"] == "scanobjectnn_c":
            height = points[:, :, 1:2] - points[:, :, 1:2].amin(dim=1, keepdim=True)
            points = points - points.mean(dim=1, keepdim=True)
            scale = points.norm(dim=-1, keepdim=True).amax(dim=1, keepdim=True)
            if torch.any(scale == 0):
                raise ValueError("Cannot unit-sphere normalize a degenerate cloud")
            points = points / scale
            features = torch.cat((points, height), dim=-1).transpose(1, 2).contiguous()
        else:
            features = points.transpose(1, 2).contiguous()
    return points, features


def state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def command_output(argv) -> str:
    try:
        return subprocess.run(argv, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=120, check=False).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        return "UNAVAILABLE: " + str(error)


def snapshot_sources(bundle: RunBundle, config: dict, manifest: dict) -> None:
    """Preserve bytes actually imported, including the distinct remote checkout."""
    root = Path(config["asset_root"]).resolve()
    if manifest.get("source_snapshot_archive"):
        relative = manifest["source_snapshot_archive"]
        archive = _declared_asset(root, manifest, relative)
        identity = verify_identity(archive, manifest["files"][relative])
        if identity["sha256"] != manifest.get("source_snapshot_sha256"):
            raise ValueError("Source snapshot archive SHA-256 mismatch")
        destination = bundle.path / "source_snapshot.zip"
        if destination.exists():
            verify_identity(destination, identity)
        else:
            with archive.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            verify_identity(destination, identity)
    sources = {Path(__file__).resolve(), REPO / "research_artifacts.py", REPO / "classifier_backbones.py"}
    # PyTorch synthetic namespaces advertise nonexistent relative __file__ values.
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and Path(module.__file__).suffix in (".py", ".so")
                   and (Path(module.__file__).is_absolute() or Path(module.__file__).is_file()))
    observed = {}
    for path in sorted(sources):
        if root in path.parents:
            relative = str(path.relative_to(root))
            _declared_asset(root, manifest, relative)
            verify_identity(path, manifest["files"][relative])
            observed["assets/" + relative] = file_identity(path)
        elif REPO in path.parents:
            relative = path.relative_to(REPO)
            destination = bundle.path / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            observed["repo/" + str(relative)] = file_identity(path)
    config["executed_source_files"] = observed
    config["native_extensions"] = {
        name: dict(path=str(Path(module.__file__).resolve()), **file_identity(Path(module.__file__)))
        for name, module in tuple(sys.modules.items())
        if getattr(module, "__file__", None) and Path(module.__file__).suffix == ".so"
        and any(fragment in name.lower() for fragment in ("pointnet2", "knn", "chamfer"))}


def run_worker(directory: str) -> None:
    import numpy as np
    import torch
    bundle = RunBundle(Path(directory))
    config = json.loads((bundle.path / "config.json").read_text())
    rows, outputs = [], {}
    active, started, classifier, manifest = None, None, None, None
    try:
        manifest = read_manifest(Path(config["asset_root"]), config["asset_manifest_sha256"])
        config["asset_manifest"] = manifest
        config["source_commit"] = manifest.get("source_commit")
        config["source_snapshot_sha256"] = manifest.get("source_snapshot_sha256")
        dataset = dataset_manifest(manifest, config)
        labels_path = Path(config["label_path"])
        labels = load_array(labels_path, dataset["files"][labels_path.name], config, labels=True)
        arrays = {}
        for corruption in config["corruptions"]:
            path = Path(config["dataset_root"]) / ("data_" + corruption + "_5.npy")
            arrays[corruption] = load_array(path, dataset["files"][path.name], config)
        config["dataset_provenance"] = dataset
        if not torch.cuda.is_available():
            raise RuntimeError("Source study requires CUDA; no CPU inference fallback")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = True
        config["runtime"] = dict(torch=torch.__version__, numpy=np.__version__, python=sys.version,
                                 cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                 gpu=torch.cuda.get_device_name(), cudnn_benchmark=torch.backends.cudnn.benchmark,
                                 cudnn_deterministic=torch.backends.cudnn.deterministic,
                                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        classifier = load_classifier(config, manifest)
        from pointnet2_ops import pointnet2_utils

        def fps(points, number):
            indices = pointnet2_utils.furthest_point_sample(points.contiguous(), number)
            return pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()

        config["classifier_state_before_sha256"] = state_hash(classifier)
        snapshot_sources(bundle, config, manifest)
        bundle.write_config(config)
        for corruption, values in arrays.items():
            active = corruption
            outputs = {"index": [], "label": [], "prediction": [], "logits": []}
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.no_grad():
                for batch, start in enumerate(range(0, len(values), config["batch_size"])):
                    if config["max_batches"] and batch >= config["max_batches"]:
                        break
                    end = min(start + config["batch_size"], len(values))
                    points = torch.from_numpy(np.array(values[start:end], dtype=np.float32)).cuda()
                    points, features = preprocess(points, config, fps)
                    logits = classifier(points, features=features)
                    outputs["index"].append(np.arange(start, end, dtype=np.int64))
                    outputs["label"].append(np.array(labels[start:end], dtype=np.int64))
                    outputs["prediction"].append(logits.argmax(-1).cpu().numpy())
                    outputs["logits"].append(logits.cpu().numpy())
                    print(corruption, "batch", batch, "examples", end, flush=True)
            torch.cuda.synchronize()
            save_corruption(bundle, config, rows, active, outputs, started, torch, failed=False)
            active = None
            bundle.write_results(rows, "running")
            config["completed_corruptions"] = [row["corruption"] for row in rows]
            config["classifier_state_checkpoint_sha256"] = state_hash(classifier)
            if config["classifier_state_checkpoint_sha256"] != config["classifier_state_before_sha256"]:
                raise RuntimeError("Frozen classifier state changed")
            bundle.write_config(config)
            save_drive_checkpoint(bundle, config.get("drive_root"), "corruption-" + corruption)
        config["classifier_state_after_sha256"] = state_hash(classifier)
        if config["classifier_state_before_sha256"] != config["classifier_state_after_sha256"]:
            raise RuntimeError("Frozen classifier state changed")
        config["status"] = "partial" if any(row["status"] == "partial" for row in rows) else "complete"
        bundle.write_results(rows, config["status"])
        snapshot_sources(bundle, config, manifest)
        (bundle.path / "notes.md").write_text(
            "# Source-only study\n\n" + config["provenance_status"] + ".\n"
            + config["preprocessing"] + "\n\nCounts/accuracy use fractions; paper comparisons require x100.\n"
            + "Indexed logits preserve original file ordinals; smoke runs are file-order prefixes.\n")
    except BaseException:
        if active is not None and started is not None:
            save_corruption(bundle, config, rows, active, outputs, started, torch, failed=True)
        config["status"] = "failed"
        config["failure"] = traceback.format_exc()
        if classifier is not None:
            config["classifier_state_after_sha256"] = state_hash(classifier)
        bundle.write_results(rows, "failed")
        raise
    finally:
        bundle.write_config(config)


def save_corruption(bundle, config, rows, corruption, outputs, started, torch, *, failed):
    import numpy as np
    arrays = {key: np.concatenate(value) for key, value in outputs.items() if value}
    count = len(arrays.get("index", []))
    correct = int((arrays["prediction"] == arrays["label"]).sum()) if count else 0
    status = "failed" if failed else ("complete" if count == config["n_examples"] else "partial")
    if arrays:
        np.savez_compressed(bundle.path / (corruption + "_predictions.npz"), **arrays)
    rows.append(corruption_row(config["run_id"], config["seed"], corruption, count, correct,
                               time.monotonic() - started, torch.cuda.max_memory_allocated() / 1024 ** 2,
                               status, method="source_only", severity=5, dataset=config["dataset"]))


def seal_bundle(bundle: RunBundle) -> None:
    identities = {str(path.relative_to(bundle.path)): file_identity(path)
                  for path in sorted(bundle.path.rglob("*")) if path.is_file() and path.name != "bundle_manifest.json"}
    (bundle.path / "bundle_manifest.json").write_text(json.dumps(identities, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(str(bundle.path) + ".zip", mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.path.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path(bundle.path.name) / path.relative_to(bundle.path)))


def save_drive_checkpoint(bundle: RunBundle, drive_root, stage: str, *, final: bool = False):
    """Publish verified immutable progress; interrupted copies have no receipt."""
    if not drive_root:
        return None
    config = json.loads((bundle.path / "config.json").read_text())
    identity = {key: config[key] for key in ("dataset", "method", "run_id")}
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) for value in (*identity.values(), stage)):
        raise ValueError("Drive checkpoint identity must contain safe path components")
    directory = Path(drive_root).absolute() / identity["dataset"] / identity["method"] / identity["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (stage + ".zip")
    receipt_path = directory / (stage + ".receipt.json")
    pending = directory / (stage + ".receipt.pending.json")
    if any(path.exists() for path in (target, receipt_path, pending)):
        raise FileExistsError("Immutable Drive checkpoint collision: " + str(target))
    with tempfile.TemporaryDirectory(prefix="drive-checkpoint-", dir=str(bundle.path.parent)) as temporary:
        if final:
            source = Path(str(bundle.path) + ".zip")
        else:
            snapshot = Path(temporary) / bundle.path.name
            shutil.copytree(bundle.path, snapshot)
            manifest = {str(path.relative_to(snapshot)): file_identity(path)
                        for path in sorted(snapshot.rglob("*")) if path.is_file()}
            (snapshot / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            source = Path(temporary) / (stage + ".zip")
            with zipfile.ZipFile(source, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(snapshot.rglob("*")):
                    if path.is_file():
                        archive.write(path, str(Path(bundle.path.name) / path.relative_to(snapshot)))
        expected = file_identity(source)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        verify_identity(target, expected)
        receipt = dict(identity, stage=stage, status="complete", result_status=config.get("status"),
                       parity_status=config.get("parity_status"), stdout_is_prefix=not final,
                       archive=dict(path=str(target), **expected))
        with pending.open("x") as file:
            json.dump(receipt, file, indent=2, sort_keys=True)
            file.write("\n")
        pending.rename(receipt_path)
    print("Verified Drive checkpoint:", receipt_path, flush=True)
    return receipt


def main(argv=None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--worker-bundle"]:
        run_worker(arguments[1])
        return
    config = resolve_config(parse_args(arguments))
    if config["preview"]:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    if (REPO / ".git").exists():
        config.update(git_commit=command_output(["git", "rev-parse", "HEAD"]).strip(),
                      git_branch=command_output(["git", "branch", "--show-current"]).strip(),
                      git_status=command_output(["git", "status", "--porcelain"]),
                      git_diff=command_output(["git", "diff", "HEAD"]))
    else:
        config["git_metadata"] = "Executed source snapshot has no .git; use pinned asset manifest source identity"
    bundle = RunBundle.create(Path(config["result_root"]), config["backbone"] + "-seed" + str(config["seed"]),
                              config, shlex.join([sys.executable, str(Path(__file__).resolve()), *arguments]))
    environment = ["Python: " + sys.version, "Platform: " + platform.platform(),
                   "nvidia-smi:\n" + command_output(["nvidia-smi"]),
                   "pip freeze:\n" + command_output([sys.executable, "-m", "pip", "freeze"])]
    (bundle.path / "environment.txt").write_text("\n\n".join(environment) + "\n")
    print("Run bundle:", bundle.path, flush=True)
    process, failure = None, None
    try:
        with (bundle.path / "stdout.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker-bundle", str(bundle.path)],
                                       cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with process.stdout:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            returncode = process.wait()
    except BaseException as error:
        failure = error
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        returncode = process.returncode if process is not None else -1
        returncode = returncode or -1
    if returncode:
        failed = json.loads((bundle.path / "config.json").read_text())
        failed.update(status="failed", worker_returncode=returncode)
        if failure is not None:
            failed["parent_failure"] = repr(failure)
        bundle.write_config(failed)
        bundle.mark_failed()
        (bundle.path / "notes.md").write_text("# Failed source-only invocation\n\nSee config.json and stdout.log.\n")
    seal_bundle(bundle)
    save_drive_checkpoint(bundle, config.get("drive_root"), "final", final=True)
    if failure is not None:
        raise failure
    if returncode:
        raise SystemExit(returncode)
    print("Finalized:", bundle.path, flush=True)


if __name__ == "__main__":
    main()
