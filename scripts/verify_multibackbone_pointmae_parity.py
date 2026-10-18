#!/usr/bin/env python3
"""Prepare/verify exact Point-MAE call parity on 32 IDs, never benchmark accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import shlex
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_multibackbone_source as source_runner

CHECKPOINTS = {
    "modelnet40_c": ("modelnet_jt.pth", "507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75"),
    "scanobjectnn_c": ("scanobject_jt.pth", "53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac"),
}
CONDITIONS = ("gaussian", "density", "background")


def load_pinned_legacy_checkpoint(path, expected_sha256):
    """Trust legacy metadata only after hashing the exact handle being loaded."""
    import torch
    with Path(path).open("rb") as checkpoint:
        digest = hashlib.sha256()
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Point-MAE checkpoint SHA-256 mismatch")
        checkpoint.seek(0)
        return torch.load(checkpoint, map_location="cpu", weights_only=False)


def compare_calls(model, classifier, points):
    import numpy as np
    import torch
    def capture():
        return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    def identity(state):
        payload = pickle.dumps(state[:2]) + b"".join(s.cpu().numpy().tobytes() for s in (state[2], *state[3]))
        return hashlib.sha256(payload).hexdigest()
    before = capture()
    with torch.no_grad():
        direct = model.classification_only(points, only_unmasked=False)
        after_direct = capture()
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])
        torch.cuda.set_rng_state_all(before[3])
        adapted = classifier(points)
    after_adapted = capture()
    checks = dict(logits_equal=torch.equal(direct, adapted), predictions_equal=torch.equal(direct.argmax(-1), adapted.argmax(-1)),
                  rng_equal=identity(after_direct) == identity(after_adapted), rng_before_sha256=identity(before),
                  rng_direct_after_sha256=identity(after_direct), rng_adapted_after_sha256=identity(after_adapted),
                  max_abs_diff=float((direct - adapted).abs().max()), n_unequal_logits=int((direct != adapted).sum()))
    return direct, adapted, checks


def worker(directory):
    bundle = source_runner.RunBundle(Path(directory))
    config = json.loads((bundle.path / "config.json").read_text())
    rows, model = [], None
    try:
        root = Path(config["asset_root"])
        manifest = source_runner.read_manifest(root, config["asset_manifest_sha256"])
        if source_runner.REPO.resolve() != (root / "source_snapshot").resolve():
            raise ValueError("Execute the pinned asset-root/source_snapshot script")
        config["asset_manifest"] = manifest
        checkpoint = source_runner.asset_path(Path(config["checkpoint_root"]), CHECKPOINTS[config["dataset"]][0])
        config["checkpoint_identity"] = source_runner.file_identity(checkpoint)
        if config["checkpoint_identity"]["sha256"] != CHECKPOINTS[config["dataset"]][1]:
            raise ValueError("Point-MAE checkpoint SHA-256 mismatch")
        dataset = source_runner.dataset_manifest(manifest, config)
        config["dataset_provenance"] = dataset
        data_root = Path(config["dataset_root"])
        labels = source_runner.load_array(data_root / "label.npy", dataset["files"]["label.npy"], config, labels=True)
        arrays = {name: source_runner.load_array(data_root / ("data_" + name + "_5.npy"),
                  dataset["files"]["data_" + name + "_5.npy"], config) for name in CONDITIONS}
        import numpy as np
        import torch
        from classifier_backbones import FrozenClassifier
        from tools import builder
        from utils_mate.config import cfg_from_yaml_file
        from utils_mate import misc
        if not torch.cuda.is_available():
            raise RuntimeError("Actual parity requires CUDA; no CPU fallback")
        os.chdir(str(source_runner.REPO))
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        filename = "tta_scanobj.yaml" if config["dataset"] == "scanobjectnn_c" else "tta_modelnet.yaml"
        options = cfg_from_yaml_file(str(source_runner.REPO / "cfgs" / filename))
        options.model.cls_dim = config["num_classes"]
        model = builder.model_builder(options.model)
        config["checkpoint_load"] = dict(weights_only=False, strict=True,
                                         trust="exact pinned legacy Point-MAE SHA-256; NumPy training metadata")
        state = load_pinned_legacy_checkpoint(checkpoint, CHECKPOINTS[config["dataset"]][1])["base_model"]
        stripped = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
        if len(stripped) != len(state):
            raise ValueError("Checkpoint module-prefix stripping collides")
        model.load_state_dict(stripped, strict=True)
        model = model.cuda().eval().requires_grad_(False)
        classifier = FrozenClassifier(model, "pointmae", config["num_classes"], input_layout="bnc")
        config["resolved_model_config"] = dict(options.model)
        config["runtime"] = dict(torch=torch.__version__, cuda=torch.version.cuda, python=sys.version, gpu=torch.cuda.get_device_name())
        config["model_state_before_sha256"] = source_runner.state_hash(model)
        source_runner.snapshot_sources(bundle, config, manifest)
        config["comparisons"] = {}
        for name, values in arrays.items():
            print(config["dataset"], name, "native FPS and paired classifier calls on IDs 0:32", flush=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            points = misc.fps(torch.from_numpy(np.array(values[:32], dtype=np.float32)).cuda(), 1024)
            input_before = points.detach().clone()
            direct, adapted, checks = compare_calls(model, classifier, points)
            checks["input_unchanged"] = torch.equal(points, input_before)
            checks["model_state_after_sha256"] = source_runner.state_hash(model)
            config["comparisons"][name] = checks
            predictions = adapted.argmax(-1).cpu().numpy()
            np.savez_compressed(bundle.path / (name + "_parity.npz"), index=np.arange(32), label=np.array(labels[:32]),
                                direct_logits=direct.cpu().numpy(), adapted_logits=adapted.cpu().numpy(),
                                direct_prediction=direct.argmax(-1).cpu().numpy(), adapted_prediction=predictions)
            torch.cuda.synchronize()
            rows.append(source_runner.corruption_row(config["run_id"], config["seed"], name, 32,
                        int((predictions == labels[:32]).sum()), time.monotonic() - started,
                        torch.cuda.max_memory_allocated() / 1024 ** 2, "partial", method=config["method"], dataset=config["dataset"]))
            if not all(checks[key] for key in ("logits_equal", "predictions_equal", "rng_equal", "input_unchanged")) or checks["model_state_after_sha256"] != config["model_state_before_sha256"]:
                raise RuntimeError("Point-MAE parity failed: " + name)
            bundle.write_results(rows, "running")
            bundle.write_config(config)
            source_runner.save_drive_checkpoint(bundle, config.get("drive_root"), "parity-" + name)
            print(name, "exact parity passed; max_abs_diff", checks["max_abs_diff"], flush=True)
        config.update(status="partial", parity_status="passed")
        bundle.write_results(rows, "partial")
    except BaseException:
        config.update(status="failed", parity_status="failed", failure=traceback.format_exc())
        for row in rows:
            row["status"] = "failed"
        bundle.write_results(rows, "failed")
        raise
    finally:
        bundle.write_config(config)
        (bundle.path / "notes.md").write_text("# Point-MAE parity smoke\n\nFirst32 original IDs per condition; raw pinned XYZ -> native FPS1024.\nNot an all15 accuracy or paper-parity result. See config.json for exact-call checks/failure.\n")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--worker-bundle"]:
        worker(arguments[1])
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in ("asset-root", "result-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--asset-manifest-sha256")
    parser.add_argument("--drive-root", help="Verified immutable progress/final archive destination")
    parser.add_argument("--modelnet-root", default="/content/3DD-TTA/data/modelnet40_c")
    parser.add_argument("--checkpoint-root", default="/content/3DD-TTA/pointnet_ckpts")
    parser.add_argument("--batch-size", type=int, choices=(32,), default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", action="store_true")
    config = vars(parser.parse_args(arguments))
    config.update(method="pointmae_boundary_parity", severity=5, corruptions=list(CONDITIONS), n_smoke_examples=32,
                  preprocessing="raw pinned XYZ -> native utils_mate.misc.fps1024 once -> same tensor for both calls")
    if config["seed"] < 0:
        raise ValueError("seed must be nonnegative")
    if config["preview"]:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    if not re.fullmatch(r"[0-9a-f]{64}", config["asset_manifest_sha256"] or ""):
        raise ValueError("Actual execution requires a pinned manifest SHA-256")
    for dataset in CHECKPOINTS:
        scan = dataset == "scanobjectnn_c"
        config.update(dataset=dataset, num_classes=15 if scan else 40, n_examples=581 if scan else 2468,
                      dataset_root=str(Path(config["asset_root"]) / "datasets/scanobjectnn_c") if scan else config["modelnet_root"])
        bundle = source_runner.RunBundle.create(Path(config["result_root"]), "pointmae-seed" + str(config["seed"]), config,
                                                shlex.join([sys.executable, str(Path(__file__).resolve()), *arguments]))
        (bundle.path / "environment.txt").write_text(sys.version + "\n" + source_runner.command_output(["nvidia-smi"]) + "\n" + source_runner.command_output([sys.executable, "-m", "pip", "freeze"]))
        print("Run bundle:", bundle.path, flush=True)
        with (bundle.path / "stdout.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker-bundle", str(bundle.path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            with process.stdout:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
            returncode = process.wait()
        if returncode:
            failed = json.loads((bundle.path / "config.json").read_text())
            failed.update(status="failed", parity_status="failed", worker_returncode=returncode)
            bundle.write_config(failed)
            bundle.mark_failed()
        source_runner.seal_bundle(bundle)
        source_runner.save_drive_checkpoint(bundle, config.get("drive_root"), "final", final=True)
        print("Sealed:", bundle.path, flush=True)
        if returncode:
            raise SystemExit(returncode)


if __name__ == "__main__":
    main()
