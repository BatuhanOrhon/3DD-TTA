"""Validate and summarize the fixed full-test-set GSD candidate screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import shlex
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile
import zlib

from research_artifacts import CORRUPTIONS, PER_COLUMNS, SUMMARY_COLUMNS

FILES = {"config.json", "command.txt", "environment.txt", "stdout.log", "notes.md",
         "per_corruption.csv", "summary.csv"}
METHOD = "gsd_latent_spectral_smooth_v2"
ARMS = {"scd_only": (0.0, "hard", None, None),
        "beta_0.5_rho_0.001": (8.140161356429882, "smooth", .5, .001),
        "beta_2_rho_0.01": (113.10823980075075, "smooth", 2., .01)}
MUTABLE_CALIBRATION_SOURCE_FILES = {"run_baseline.py", "gsd_protocol.py", "gsd_calibration.py"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, field: str, *, nonnegative=False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(field + " must be numeric")
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(field + " must be finite" + (" and non-negative" if nonnegative else ""))
    return number


def _integer(value, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(field + " must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(field + " must be an integer")
    if str(result) != str(value) or result < 0:
        raise ValueError(field + " must be a non-negative integer")
    return result


def _csv(text: str, fields: tuple[str, ...], label: str, path: Path) -> list[dict]:
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != fields:
        raise ValueError(label + " header mismatch: " + str(path))
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(label + " malformed row: " + str(path))
    return rows


def _environment_identity(text: str, path: Path) -> dict:
    try:
        package_block = text.split("pip freeze:\n", 1)[1].split("\n\nAsset hashes:", 1)[0]
    except IndexError as error:
        raise ValueError("environment record has no pip-freeze section: " + str(path)) from error
    packages = tuple(line.strip() for line in package_block.splitlines() if line.strip())
    if not packages:
        raise ValueError("environment record has an empty package inventory: " + str(path))
    fields = {}
    for key in ("Python", "Platform", "Git branch", "Git commit"):
        line = next((line for line in text.splitlines() if line.startswith(key + ":")), None)
        if line is None:
            raise ValueError("environment record is missing " + key + ": " + str(path))
        fields[key] = line.partition(":")[2].strip()
    return {"fields": fields, "packages": packages}


def _bundle_contents(path: Path):
    if path.is_dir():
        if {p.name for p in path.iterdir() if p.is_file()} != FILES:
            raise ValueError("run directory must contain the exact seven-file bundle: " + str(path))
        contents = {name: (path / name).read_bytes() for name in FILES}
        return path.name, contents
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != 7 or len(set(names)) != 7:
                raise ValueError("archive must contain exactly seven unique files: " + str(path))
            if archive.testzip() is not None:
                raise ValueError("archive CRC check failed: " + str(path))
            parsed = [PurePosixPath(name) for name in names]
            if any(item.is_absolute() or ".." in item.parts or "\\" in name for item, name in zip(parsed, names)):
                raise ValueError("unsafe archive member path: " + str(path))
            if any(len(item.parts) != 2 for item in parsed) or {item.name for item in parsed} != FILES:
                raise ValueError("archive must contain one root and the exact seven bundle files: " + str(path))
            roots = {item.parts[0] for item in parsed}
            if len(roots) != 1:
                raise ValueError("archive contains multiple run roots: " + str(path))
            root = next(iter(roots))
            return root, {PurePosixPath(name).name: archive.read(name) for name in names}
    except (BadZipFile, zlib.error, EOFError, RuntimeError) as error:
        raise ValueError("invalid ZIP archive: " + str(path)) from error


def _validate_manifest(config: dict, name: str, path: Path):
    manifest = config.get(name)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("missing " + name + ": " + str(path))
    for key, value in manifest.items():
        if (not isinstance(key, str) or not isinstance(value, dict) or
                not isinstance(value.get("sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
            raise ValueError("invalid " + name + " identity: " + str(path))


def read_full_run(path: Path) -> dict:
    """Read and strictly validate one complete seven-file full-suite bundle."""
    path = Path(path)
    root, contents = _bundle_contents(path)
    try:
        config = json.loads(contents["config.json"])
        command_text = contents["command.txt"].decode("utf-8")
        per_text = contents["per_corruption.csv"].decode("utf-8")
        summary_text = contents["summary.csv"].decode("utf-8")
        environment = contents["environment.txt"].decode("utf-8")
        stdout = contents["stdout.log"].decode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("run bundle has invalid JSON or text encoding: " + str(path)) from error
    if not isinstance(config, dict) or config.get("run_id") != root:
        raise ValueError("archive root and config run ID disagree: " + str(path))
    if (config.get("stage") != "full_dataset_development" or config.get("execution_status") != "complete" or
            config.get("status") != "complete" or config.get("dataset") != "modelnet40_c" or
            config.get("severity") != 5 or config.get("method") != METHOD or
            config.get("completed_corruptions") != list(CORRUPTIONS) or
            config.get("corruptions") != list(CORRUPTIONS)):
        raise ValueError("run is not a complete canonical full-dataset development bundle: " + str(path))
    cli = config.get("cli_args")
    if not isinstance(cli, dict):
        raise ValueError("run is missing CLI metadata: " + str(path))
    required_cli = {"gsd_stage": "full_dataset_development", "dataset_name": "modelnet-c", "severity": 5,
                    "batch_size": 32, "max_batches": 0, "lion_eval_mode": True, "lion_ema_mode": False,
                    "gamma": .01, "eta": .01, "lambdaa": .95, "gsd_scd_weight": 1.,
                    "gsd_k": 10, "gsd_delta": .1, "gsd_graph_gamma": .6, "gsd_modes": 100,
                    "corruptions": list(CORRUPTIONS)}
    for key, value in required_cli.items():
        if cli.get(key) != value:
            raise ValueError("full-suite CLI protocol mismatch for " + key + ": " + str(path))
    if cli.get("seed") not in (0, 1, 2) or config.get("seed") != cli.get("seed") or config.get("batch_size") != 32:
        raise ValueError("full-suite seed/batch metadata mismatch: " + str(path))
    inventory = config.get("dataset_inventory")
    if not isinstance(inventory, dict) or set(inventory) != set(CORRUPTIONS):
        raise ValueError("run is missing complete all-corruption dataset inventory: " + str(path))
    for corruption in CORRUPTIONS:
        entry = inventory[corruption]
        if not isinstance(entry, dict) or _integer(entry.get("total_examples"), "dataset inventory count") <= 0:
            raise ValueError("invalid full-file inventory for " + corruption + ": " + str(path))
    batches = config.get("observed_batch_sizes")
    if not isinstance(batches, dict) or set(batches) != set(CORRUPTIONS):
        raise ValueError("run is missing complete per-file batch accounting: " + str(path))
    for corruption in CORRUPTIONS:
        sizes = batches[corruption]
        total = _integer(inventory[corruption]["total_examples"], "dataset inventory count")
        if (not isinstance(sizes, list) or not sizes or any(type(size) is not int or not 0 < size <= 32 for size in sizes) or
                sum(sizes) != total or any(size != 32 for size in sizes[:-1])):
            raise ValueError("observed batches do not cover the full file: " + corruption + ": " + str(path))
    timesteps = config.get("scheduler_timesteps")
    expected_timesteps = list(range(990, -1, -10))
    if not isinstance(timesteps, dict) or set(timesteps) != set(CORRUPTIONS) or any(
            timesteps[name] != expected_timesteps for name in CORRUPTIONS):
        raise ValueError("scheduler timeline mismatch: " + str(path))
    randomness = config.get("randomness")
    if (not isinstance(randomness, dict) or randomness.get("data_order") != "file order; shuffle=False" or
            randomness.get("num_workers") != 0 or randomness.get("drop_last") is not False):
        raise ValueError("data loader ordering/settings mismatch: " + str(path))
    reference = config.get("calibration_reference")
    if not isinstance(reference, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256", ""))):
        raise ValueError("run is missing calibration reference identity: " + str(path))
    if (reference.get("run_id") != "20260928-113047_gsd-cal-diagnose-reference-seed0-n64" or
            reference["sha256"] != "550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6"):
        raise ValueError("run is bound to the wrong calibration reference: " + str(path))
    profile, beta = cli.get("gsd_profile"), cli.get("gsd_beta")
    rho = cli.get("gsd_target_rho")
    arm = "scd_only" if cli.get("gsd_weight") == 0 and rho is None else (
        "beta_0.5_rho_0.001" if profile == "smooth" and beta == .5 and rho == .001 else
        "beta_2_rho_0.01" if profile == "smooth" and beta == 2. and rho == .01 else None)
    if arm is None:
        raise ValueError("run is not one of the three locked screen arms: " + str(path))
    weight, expected_profile, expected_beta, expected_rho = ARMS[arm]
    if not math.isclose(_finite(cli.get("gsd_weight"), "spectral weight"), weight, rel_tol=0, abs_tol=1e-12):
        raise ValueError("candidate coefficient mismatch: " + str(path))
    if (profile, beta, rho) != (expected_profile, expected_beta, expected_rho):
        raise ValueError("candidate profile/rho mismatch: " + str(path))
    spectral = config.get("spectral")
    if not isinstance(spectral, dict) or any(spectral.get(k) != v for k, v in {
            "weight": weight, "scd_weight": 1., "profile": expected_profile,
            "beta": expected_beta, "k": 10, "delta": .1, "graph_gamma": .6}.items()):
        raise ValueError("spectral contract disagrees with CLI: " + str(path))
    for name in ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest"):
        _validate_manifest(config, name, path)
    compatibility = config.get("calibration_source_compatibility")
    if (not isinstance(compatibility, dict) or compatibility.get("allowed_source_extensions") !=
            sorted(MUTABLE_CALIBRATION_SOURCE_FILES)):
        raise ValueError("run is missing the declared calibration-source compatibility record: " + str(path))
    if set(config["dataset_hash_manifest"]) != set(CORRUPTIONS):
        raise ValueError("run is missing full-file hashes for all 15 corruptions: " + str(path))
    expected_manifests = reference.get("expected_manifests")
    if (not isinstance(expected_manifests, dict) or
            any(not isinstance(expected_manifests.get(name), dict) or not expected_manifests[name]
                for name in ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest"))):
        raise ValueError("calibration reference is missing input manifests: " + str(path))
    for name, identity in expected_manifests.get("asset_manifest", {}).items():
        if config["asset_manifest"].get(name) != identity:
            raise ValueError("calibration asset identity mismatch: " + name + ": " + str(path))
    for name, identity in expected_manifests.get("dataset_hash_manifest", {}).items():
        if config["dataset_hash_manifest"].get(name) != identity:
            raise ValueError("calibration test-file identity mismatch: " + name + ": " + str(path))
    for name, identity in expected_manifests.get("runtime_source_manifest", {}).items():
        if name in MUTABLE_CALIBRATION_SOURCE_FILES:
            continue
        if config["runtime_source_manifest"].get(name) != identity:
            raise ValueError("calibration source identity mismatch: " + name + ": " + str(path))
    if config.get("randomness", {}).get("seed") != config["seed"]:
        raise ValueError("randomness seed mismatch: " + str(path))
    if not environment.strip() or not stdout.strip():
        raise ValueError("run bundle lacks environment or execution log: " + str(path))
    if "Run directory:" not in stdout:
        raise ValueError("execution log lacks the worker run marker: " + str(path))
    command_lines = [line for line in command_text.splitlines() if line and not line.startswith("cwd:")]
    try:
        command_tokens = shlex.split(command_lines[-1])
    except (IndexError, ValueError) as error:
        raise ValueError("command.txt has no parseable runner command: " + str(path)) from error
    logical_run_name = cli.get("run_name")
    if not isinstance(logical_run_name, str) or not root.endswith("_" + logical_run_name):
        raise ValueError("config run name disagrees with archived run ID: " + str(path))
    for flag, expected in (("--run-name", logical_run_name), ("--seed", str(config["seed"])),
                           ("--gsd-stage", "full_dataset_development"), ("--max-batches", "0")):
        if flag not in command_tokens or command_tokens[command_tokens.index(flag) + 1] != expected:
            raise ValueError("command/config mismatch for " + flag + ": " + str(path))

    rows = _csv(per_text, PER_COLUMNS, "per_corruption.csv", path)
    if len(rows) != len(CORRUPTIONS):
        raise ValueError("per-corruption CSV must contain all 15 rows: " + str(path))
    by_corruption = {}
    for row in rows:
        name = row["corruption"]
        if name not in CORRUPTIONS or name in by_corruption:
            raise ValueError("duplicate or noncanonical corruption row: " + str(path))
        expected_count = _integer(inventory[name]["total_examples"], "dataset inventory count")
        n, correct = _integer(row["n_examples"], "CSV examples"), _integer(row["n_correct"], "CSV correct")
        accuracy = _finite(row["accuracy"], "CSV accuracy")
        _finite(row["runtime_seconds"], "CSV runtime", nonnegative=True)
        _finite(row["peak_gpu_memory_mb"], "CSV peak memory", nonnegative=True)
        if (row["run_id"] != root or row["dataset"] != "modelnet40_c" or row["method"] != METHOD or
                _integer(row["severity"], "CSV severity") != 5 or _integer(row["seed"], "CSV seed") != config["seed"] or
                row["status"] != "complete" or n != expected_count or not 0 <= correct <= n or
                not math.isclose(accuracy, correct / n, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("CSV row disagrees with full file/config: " + name + ": " + str(path))
        by_corruption[name] = row
    if list(by_corruption) != list(CORRUPTIONS):
        raise ValueError("per-corruption rows must preserve canonical order: " + str(path))
    summary_rows = _csv(summary_text, SUMMARY_COLUMNS, "summary.csv", path)
    if len(summary_rows) != 1:
        raise ValueError("summary CSV must contain one row: " + str(path))
    summary = summary_rows[0]
    total_n = sum(int(row["n_examples"]) for row in rows)
    total_correct = sum(int(row["n_correct"]) for row in rows)
    macro = sum(float(row["accuracy"]) for row in rows) / len(CORRUPTIONS)
    runtime = sum(float(row["runtime_seconds"]) for row in rows)
    if (summary["run_id"] != root or summary["dataset"] != "modelnet40_c" or summary["method"] != METHOD or
            _integer(summary["severity"], "summary severity") != 5 or
            _integer(summary["seed"], "summary seed") != config["seed"] or summary["status"] != "complete" or
            _integer(summary["n_corruptions"], "summary corruption count") != 15 or
            _integer(summary["total_examples"], "summary examples") != total_n or
            _integer(summary["total_correct"], "summary correct") != total_correct or
            not math.isclose(_finite(summary["macro_accuracy"], "summary macro"), macro, abs_tol=1e-12) or
            not math.isclose(_finite(summary["micro_accuracy"], "summary micro"), total_correct / total_n, abs_tol=1e-12) or
            not math.isclose(_finite(summary["total_runtime_seconds"], "summary runtime"), runtime, abs_tol=1e-9)):
        raise ValueError("summary CSV disagrees with per-corruption rows: " + str(path))
    return {"path": str(path.resolve()), "archive_sha256": _sha256(path) if path.is_file() else None,
            "run_id": root, "arm": arm, "seed": config["seed"], "config": config,
            "rows": by_corruption, "summary": summary,
            "environment_identity": {"runtime": config["randomness"],
                                     "extensions": config["extension_inventory"],
                                     "environment": _environment_identity(environment, path)}}


def _mean_sd(values):
    values = [float(value) for value in values]
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def _calibration_identity(calibration):
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be the validated calibration payload")
    from gsd_calibration import PINNED_REFERENCE_CONFIG_SHA256, PINNED_REFERENCE_RUN_ID
    if (calibration.get("run_id") != PINNED_REFERENCE_RUN_ID or
            calibration.get("sha256") != PINNED_REFERENCE_CONFIG_SHA256 or
            not isinstance(calibration.get("report"), dict) or
            not isinstance(calibration.get("expected_manifests"), dict)):
        raise ValueError("calibration payload does not match the pinned reference")
    if any(not isinstance(calibration["expected_manifests"].get(name), dict) or
           not calibration["expected_manifests"][name]
           for name in ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest")):
        raise ValueError("pinned calibration is missing source/data/checkpoint identities")
    return calibration


def analyze_runs(paths: list[Path], calibration: dict) -> dict:
    """Validate exactly nine completed runs and compute equal-corruption summaries."""
    calibration = _calibration_identity(calibration)
    if len(paths) != 9:
        raise ValueError("provide exactly one complete run for each of 3 arms x 3 seeds")
    runs = [read_full_run(Path(path)) for path in paths]
    pairs = [(run["arm"], run["seed"]) for run in runs]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate arm/seed run pair")
    expected = {(arm, seed) for arm in ARMS for seed in range(3)}
    if set(pairs) != expected:
        raise ValueError("missing expected arm/seed run pair")
    first = runs[0]["config"]
    compare_keys = ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest", "extension_inventory")
    for run in runs[1:]:
        config = run["config"]
        if any(config.get(key) != first.get(key) for key in compare_keys):
            raise ValueError("screen runs differ in assets, test files, sources, or extensions")
        if config["dataset_inventory"] != first["dataset_inventory"]:
            raise ValueError("screen runs used different full-file example counts")
        if run["environment_identity"]["extensions"] != runs[0]["environment_identity"]["extensions"]:
            raise ValueError("screen runs used different compiled extensions")
        env_a, env_b = run["environment_identity"]["environment"], runs[0]["environment_identity"]["environment"]
        if env_a != env_b:
            raise ValueError("screen runs used different Python/platform/package environments")
        if run["config"].get("git_commit") != first.get("git_commit") or run["config"].get("git_branch") != first.get("git_branch"):
            raise ValueError("screen runs were recorded from different Git revisions")
        random_a = run["environment_identity"]["runtime"]
        random_b = runs[0]["environment_identity"]["runtime"]
        if any(random_a.get(key) != random_b.get(key) for key in
               ("torch_version", "numpy_version", "cuda_version", "cudnn_version", "gpu")):
            raise ValueError("screen runs used different package/runtime versions")
        if config["calibration_reference"].get("sha256") != first["calibration_reference"].get("sha256"):
            raise ValueError("screen runs reference different calibration configs")
    for run in runs:
        if run["config"]["calibration_reference"].get("expected_manifests") != calibration["expected_manifests"]:
            raise ValueError("screen run embeds different calibration source/data/checkpoint identities")
    report = calibration["report"]
    for arm, beta_key, rho_key in (("beta_0.5_rho_0.001", "0.5", "0.001"),
                                   ("beta_2_rho_0.01", "2.0", "0.01")):
        expected_weight = report.get("candidates", {}).get(beta_key, {}).get("weights", {}).get(rho_key)
        if expected_weight is None or not math.isclose(float(expected_weight), ARMS[arm][0], rel_tol=1e-12, abs_tol=0):
            raise ValueError("calibration report does not yield locked coefficient for " + arm)
    indexed = {(run["arm"], run["seed"]): run for run in runs}
    arm_results = {}
    for arm in ARMS:
        seeds = {}
        for seed in range(3):
            run = indexed[(arm, seed)]
            accuracies = {name: float(run["rows"][name]["accuracy"]) for name in CORRUPTIONS}
            seeds[str(seed)] = {"run_id": run["run_id"], "macro_accuracy": statistics.mean(accuracies.values()),
                                "per_corruption_accuracy": accuracies,
                                "total_runtime_seconds": float(run["summary"]["total_runtime_seconds"]),
                                "peak_gpu_memory_mb": max(float(row["peak_gpu_memory_mb"]) for row in run["rows"].values())}
        arm_results[arm] = {"seeds": seeds,
                            "macro_accuracy_across_seeds": _mean_sd(v["macro_accuracy"] for v in seeds.values()),
                            "runtime_seconds_across_seeds": _mean_sd(v["total_runtime_seconds"] for v in seeds.values()),
                            "peak_gpu_memory_mb_across_seeds": _mean_sd(v["peak_gpu_memory_mb"] for v in seeds.values())}
    deltas = {}
    for arm in ("beta_0.5_rho_0.001", "beta_2_rho_0.01"):
        seed_deltas = {}
        for seed in range(3):
            candidate = indexed[(arm, seed)]["rows"]
            baseline = indexed[("scd_only", seed)]["rows"]
            by_corruption = {name: (float(candidate[name]["accuracy"]) - float(baseline[name]["accuracy"])) * 100
                             for name in CORRUPTIONS}
            macro_delta = statistics.mean(by_corruption.values())
            seed_deltas[str(seed)] = {"macro_delta_pp": macro_delta,
                                      "direction": "positive" if macro_delta > 0 else "negative" if macro_delta < 0 else "zero",
                                      "per_corruption_delta_pp": by_corruption}
        deltas[arm] = {"seeds": seed_deltas,
                       "mean_pp": statistics.mean(row["macro_delta_pp"] for row in seed_deltas.values()),
                       "sample_sd_pp": statistics.stdev(row["macro_delta_pp"] for row in seed_deltas.values()),
                       "positive_seed_count": sum(row["macro_delta_pp"] > 0 for row in seed_deltas.values()),
                       "negative_seed_count": sum(row["macro_delta_pp"] < 0 for row in seed_deltas.values())}
    per_corruption = {}
    for arm in ARMS:
        per_corruption[arm] = {name: _mean_sd(indexed[(arm, seed)]["rows"][name]["accuracy"] for seed in range(3))
                               for name in CORRUPTIONS}
    environment = runs[0]["environment_identity"]
    runtime = environment["runtime"]
    return {"schema_version": 1,
            "evidence_status": "full-test-set development; candidate selection is descriptive, not independent confirmation",
            "calibration_reference": {"run_id": calibration["run_id"], "sha256": calibration["sha256"]},
            "environment": {"python_platform_packages": environment["environment"],
                            "runtime_versions": {key: runtime.get(key) for key in
                                                 ("torch_version", "numpy_version", "cuda_version", "cudnn_version", "gpu")},
                            "extensions": environment["extensions"],
                            "git_branch": first.get("git_branch"), "git_commit": first.get("git_commit")},
            "input_identity": {"assets": first["asset_manifest"],
                               "all_corruption_files": first["dataset_hash_manifest"],
                               "full_examples_by_corruption": {name: first["dataset_inventory"][name]["total_examples"]
                                                                for name in CORRUPTIONS},
                               "source_files": first["runtime_source_manifest"]},
            "protocol": {"dataset": "ModelNet40-C", "severity": 5, "n_corruptions": 15,
                         "corruptions": list(CORRUPTIONS), "seeds": [0, 1, 2], "batch_size": 32,
                         "n_runs": 9, "macro_definition": "unweighted arithmetic mean of 15 corruption accuracies"},
            "runs": [{"run_id": run["run_id"], "arm": run["arm"], "seed": run["seed"],
                      "path": run["path"], "archive_sha256": run["archive_sha256"]} for run in runs],
            "arms": arm_results, "candidate_deltas": deltas, "per_corruption_across_seeds": per_corruption,
            "seed_pairing_note": "Compare like seeds only; seeds are controlled separate runs, not common-draw pairs."}


def load_pinned_calibration(path: Path) -> dict:
    import gsd_calibration
    path = Path(path)
    config_path = path / "config.json" if path.is_dir() else path
    config, report = gsd_calibration.load_calibration(config_path)
    expected = {key: config[key] for key in
                ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest")}
    payload = {"run_id": config.get("run_id"), "sha256": _sha256(config_path),
               "report": report, "expected_manifests": expected}
    return _calibration_identity(payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--screen-run", required=True, nargs=9, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    calibration = load_pinned_calibration(args.calibration)
    result = analyze_runs(args.screen_run, calibration)
    output = args.output.resolve()
    if output in {path.resolve() for path in args.screen_run} or output == args.calibration.resolve():
        raise ValueError("derived output must not overwrite calibration or a raw run archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote derived full-test-set screen summary:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
