"""Colab smoke runner for the original 3DD-TTA path, with auditable artifacts."""
from __future__ import annotations

import argparse
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

from research_artifacts import CORRUPTIONS, RunBundle, corruption_row, validate_selection

REPO = Path(__file__).resolve().parent


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
        identities = {name: file_identity(Path(value)) for name, value in {
            "classifier_checkpoint": args.pointmae_ckpt, "lion_checkpoint": args.diff_ckpt,
            "pointmae_config": args.pointmae_config, "lion_config": args.diff_config,
            "labels": args.label_path,
        }.items()}
        data_files = {name: file_identity(Path(args.dataset_root) / ("data_" + name + "_5.npy"))
                      for name in args.corruptions}
        config.update(asset_manifest=identities, dataset_hash_manifest=data_files,
                      classifier_checkpoint_sha256=identities["classifier_checkpoint"]["sha256"],
                      lion_checkpoint_sha256=identities["lion_checkpoint"]["sha256"])
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

        base_model, lion = baseline.configure_model(args, checkpoint_observer=checkpoint_observer)
        config["extension_inventory"] = extension_inventory()
        config["resolved_pointmae_config"] = baseline.cfg_from_yaml_file(args.pointmae_config)
        config["resolved_pointmae_config"]["model"]["cls_dim"] = 40
        config["resolved_lion_config_yaml"] = baseline.diff_config.dump()

        def record_modes(key):
            config[key] = dict(classifier=module_inventory(base_model),
                               lion_vae=module_inventory(lion.vae),
                               lion_priors=module_inventory(lion.priors))

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
            dataset = baseline.PointDataset(args.dataset_root, args.label_path, active)
            if len(dataset.data) != len(dataset.labels) or len(dataset) == 0:
                raise ValueError("Empty data or data/label count mismatch.")
            if dataset.data.ndim != 3 or dataset.data.shape[-1] != 3:
                raise ValueError("Expected data shape [examples, points, 3].")
            if dataset.labels.size != len(dataset) or not np.issubdtype(dataset.labels.dtype, np.integer):
                raise ValueError("Expected one integer class label per example.")
            if dataset.labels.min() < 0 or dataset.labels.max() >= 40:
                raise ValueError("ModelNet40 class label outside [0,39].")
            config.setdefault("dataset_inventory", {})[active] = dict(
                data_shape=list(dataset.data.shape), data_dtype=str(dataset.data.dtype),
                label_shape=list(dataset.labels.shape), label_dtype=str(dataset.labels.dtype),
                total_examples=len(dataset), min_label=int(dataset.labels.min()),
                max_label=int(dataset.labels.max()))
            loader = baseline.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            batches = itertools.islice(loader, args.max_batches)
            steps = 35 if active == "background" else 5
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            targets, predictions = baseline.process_batches(
                batches, base_model, lion, args, steps,
                scheduler_observer=scheduler_observer, batch_observer=batch_observer)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if counters["n"] != targets.numel() or counters["correct"] != int((predictions == targets).sum().item()):
                raise RuntimeError("Batch counters disagree with returned predictions.")
            status = "complete" if counters["n"] == len(dataset) else "partial"
            rows.append(corruption_row(config["run_id"], args.seed, active,
                                       counters["n"], counters["correct"], elapsed,
                                       torch.cuda.max_memory_allocated() / (1024 ** 2), status))
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
        (bundle.path / "notes.md").write_text(
            "# Smoke run\n\nPurpose: validate execution and artifact schema, NOT benchmark accuracy.\n"
            "Only a file-order prefix is evaluated. Seed-controlled, not fully paired.\n"
            "Original TTA math, legacy LION modes, 5/35 steps, gamma/eta mapping and static decode style retained.\n"
            "CUDA runtime excludes asset hashing and model loading; peak allocated memory includes models.\n"
            "No resume or test-set hyperparameter search. User: add Colab runtime/GPU type and anomalies.\n",
            encoding="utf-8")
    except BaseException:
        config.update(status="failed", execution_status="failed",
                      failed_corruption=active,
                      completed_corruptions=[row["corruption"] for row in rows])
        if active is not None and started is not None:
            memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch is not None else 0.0
            rows.append(corruption_row(config["run_id"], args.seed, active,
                                       counters["n"], counters["correct"],
                                       time.perf_counter() - started, memory, "failed"))
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
    parser.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS, default=["gaussian"])
    parser.add_argument("--max-batches", type=int, default=2, help="Smoke-only prefix limit per corruption")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--result-root", default="./result")
    args = parser.parse_args(argv)
    validate_selection(args.corruptions)
    if args.batch_size < 1 or args.max_batches < 1 or not 0 <= args.seed < 2 ** 32:
        parser.error("Batch size/max-batches must be positive; seed must be in [0,2**32).")
    if not all(math.isfinite(v) and v >= 0 for v in (args.gamma, args.eta, args.lambdaa)) or not 0 < args.lambdaa <= 1:
        parser.error("Invalid rates or SCD percentile.")
    args.device, args.dataset_name = "cuda", "modelnet-c"
    args.run_name = args.run_name or ("baseline-smoke_seed%s" % args.seed)
    return args


def main() -> int:
    args = parse_arguments()
    if Path.cwd().resolve() != REPO:
        raise SystemExit("Run from the repository root so baseline relative configs resolve correctly.")
    config = dict(
        stage="smoke", dataset="modelnet40_c", severity=5, method="3dd_original",
        seed=args.seed, batch_size=args.batch_size, corruptions=args.corruptions,
        classifier="pointmae", num_input_points=2048, num_classifier_points=1024,
        scale_factor=3.3885, ddim_total_steps=100,
        normal_reverse_steps=5, background_reverse_steps=35,
        gamma=args.gamma, eta=args.eta, lambda_cd=args.lambdaa,
        guidance_mapping=dict(gamma="local latent", eta="style condition"),
        final_decode_style="original shape_latent", lion_mode_policy="legacy; unchanged",
        scheduler_config={}, spectral={}, projection={}, cli_args=vars(args),
        git_branch=command_output(["git", "branch", "--show-current"]),
        git_commit=command_output(["git", "rev-parse", "HEAD"]),
        git_status=command_output(["git", "status", "--porcelain"]),
        runtime_source_manifest={name: file_identity(REPO / name) for name in
                                 ("run_baseline.py", "research_artifacts.py", "main_3dd_tta.py",
                                  "tta.py", "utilities_3dd_tta.py", "models/lion.py")})
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
