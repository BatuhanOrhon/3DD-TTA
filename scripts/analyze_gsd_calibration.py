"""Print a compact calibration summary and optionally rank completed screens.

This reads raw diagnostics locally and emits only a small JSON object. It never
prints individual per-sample rows. Accuracy ranking is exploratory development
selection, not a held-out or benchmark claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsd_calibration import BETAS, RHOS, development_indices, load_calibration
from gsd_protocol import PILOT_CORRUPTIONS, SMOOTH_METHOD


SCREEN_FILES = {"config.json", "command.txt", "environment.txt", "stdout.log", "notes.md",
                "per_corruption.csv", "summary.csv"}
PER_CORRUPTION_FIELDS = ("run_id", "dataset", "severity", "method", "seed", "corruption",
                          "n_examples", "n_correct", "accuracy", "runtime_seconds",
                          "peak_gpu_memory_mb", "status")
SUMMARY_FIELDS = ("run_id", "dataset", "severity", "method", "seed", "n_corruptions",
                  "macro_accuracy", "total_examples", "total_correct", "micro_accuracy",
                  "total_runtime_seconds", "status")
SCREEN_CONFIG = {
    "dataset": "modelnet40_c", "severity": 5, "method": SMOOTH_METHOD, "batch_size": 32,
    "corruptions": list(PILOT_CORRUPTIONS), "num_classes": 40, "num_input_points": 2048,
    "num_classifier_points": 1024, "scale_factor": 3.3885, "ddim_total_steps": 100,
    "normal_reverse_steps": 5, "background_reverse_steps": 35, "gamma": .01, "eta": .01,
    "lambda_cd": .95, "guidance_mapping": {"gamma": "local latent", "eta": "style condition"},
    "final_decode_style": "original shape_latent", "lion_mode_policy": "raw LION eval; EMA disabled",
    "lion_loaded": True, "scd_normalization": {"enabled": False, "reduction": "legacy sum"},
    "scheduler_class": "DDIMScheduler",
    "scheduler_config": {"beta_start": .0001, "beta_end": .02, "beta_schedule": "linear",
                         "num_train_timesteps": 1000, "clip_sample": False, "set_alpha_to_one": True,
                         "steps_offset": 0, "prediction_type": "epsilon", "trained_betas": None},
}
SCREEN_CLI = {
    "batch_size": 32, "corruptions": list(PILOT_CORRUPTIONS), "dataset_name": "modelnet-c",
    "eta": .01, "gamma": .01, "lambdaa": .95, "lion_eval_mode": True, "lion_ema_mode": False,
    "max_batches": 0, "method": SMOOTH_METHOD, "severity": 5, "gsd_stage": "development",
    "gsd_scd_weight": 1., "gsd_k": 10, "gsd_delta": .1, "gsd_graph_gamma": .6,
    "gsd_modes": 100, "gsd_split_seed": 20260927,
}


def _config_path(path: Path) -> Path:
    return path / "config.json" if path.is_dir() else path


def calibration_summary(path: Path) -> dict:
    path = _config_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "cli_args" in payload:
        config, report = load_calibration(path)
        run_id = config["run_id"]
        split_seed = config["cli_args"]["gsd_split_seed"]
        example_count = config["cli_args"]["gsd_development_count"]
        graph_diagnostics = config["gsd_diagnostics"]
    else:
        required = {"run_id", "candidates", "by_corruption", "state_by_step"}
        if not required.issubset(payload):
            raise ValueError("input must be a calibration config.json or phase-report JSON")
        config, report = None, payload
        run_id = report["run_id"]
        split_seed = report.get("split_seed")
        example_count = report.get("examples_per_corruption")
        if example_count is None:
            ratio_count = report["by_corruption"]["gaussian"]["hard"]["local_ratio"]["count"]
            if ratio_count % 3:
                raise ValueError("cannot infer diagnostic sample count from three probe steps")
            example_count = ratio_count // 3
        graph_diagnostics = {}

    def candidate_view(candidate, corruption_rows):
        def compact(stat, fields=("median", "p90", "count", "missing")):
            return {key: stat.get(key) for key in fields}
        return dict(local_ratio=compact(candidate["local_ratio"]),
                    style_ratio=compact(candidate["style_ratio"]),
                    local_cosine=compact(candidate["local_cosine"], ("median",)),
                    alpha_by_rho=candidate.get("weights"),
                    by_corruption={name: {
                        "local_ratio": compact(corruption_rows[name]["local_ratio"], ("median",)),
                        "style_ratio": compact(corruption_rows[name]["style_ratio"], ("median",)),
                        "local_cosine": compact(corruption_rows[name]["local_cosine"], ("median",)),
                    } for name in PILOT_CORRUPTIONS})

    beta_rows = []
    for beta in BETAS:
        key = str(beta)
        candidate = report["candidates"][key]
        graph = {}
        for corruption in PILOT_CORRUPTIONS:
            graph_stats = graph_diagnostics.get(corruption, {}).get("graph", {}).get("scalars", {})
            stat = graph_stats.get("effective_mass_beta" + key, {})
            graph[corruption] = stat.get("mean")
        beta_rows.append(dict(
            beta=beta,
            **candidate_view(candidate, {name: report["by_corruption"][name][key]
                                         for name in PILOT_CORRUPTIONS}),
            effective_mass_by_corruption=graph,
        ))
    hard = report["candidates"]["hard"]
    hard_summary = candidate_view(hard, {name: report["by_corruption"][name]["hard"]
                                         for name in PILOT_CORRUPTIONS})
    selected_state_metrics = ("local_state_rms", "local_scd_update_rms",
                              "local_scd_update_state_ratio", "local_ddim_displacement_norm",
                              "local_scd_ddim_ratio")
    state_scale = {}
    for corruption in PILOT_CORRUPTIONS:
        state_scale[corruption] = {}
        for step, values in report["state_by_step"][corruption].items():
            state_scale[corruption][step] = {
                metric: {key: values[metric].get(key) for key in ("median", "p90")}
                for metric in selected_state_metrics if metric in values}
    return dict(
        report_type="label-free spectral calibration; no accuracy-based beta selection",
        run_id=run_id, split_seed=split_seed, examples_per_corruption=example_count,
        corruptions=list(PILOT_CORRUPTIONS), target_contributions=list(RHOS),
        hard=hard_summary,
        smooth_candidates=beta_rows,
        local_state_scale_by_corruption_and_step=state_scale,
        interpretation=("Use ratios to calibrate alpha and inspect update scale. "
                        "This diagnostic alone cannot establish an accuracy-optimal beta."))


def _equal(left, right) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
    return left == right


def _parse_int(value, label: str, run_path: Path) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(label + " must be an integer: " + str(run_path))
    if str(parsed) != str(value):
        raise ValueError(label + " must be an integer: " + str(run_path))
    return parsed


def _parse_finite(value, label: str, run_path: Path, nonnegative=False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(label + " must be numeric: " + str(run_path))
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        raise ValueError(label + " must be finite" + (" and non-negative" if nonnegative else "") + ": " + str(run_path))
    return parsed


def _read_csv(text: str, fields: tuple[str, ...], label: str, run_path: Path) -> list[dict]:
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != fields:
        raise ValueError(label + " has an unexpected header: " + str(run_path))
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(label + " has malformed rows: " + str(run_path))
    return rows


def _read_screen_bundle(path: Path) -> tuple[dict, str, str, Path]:
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            files = [name for name in archive.namelist() if not name.endswith("/")]
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in files):
                raise ValueError("unsafe path in screen archive: " + str(path))
            if archive.testzip() is not None:
                raise ValueError("screen archive failed CRC validation: " + str(path))
            roots = {Path(name).parts[0] for name in files if len(Path(name).parts) == 2}
            if (len(files) != 7 or {Path(name).name for name in files} != SCREEN_FILES or len(roots) != 1 or
                    any(len(Path(name).parts) != 2 for name in files)):
                raise ValueError("screen archive must contain the complete seven-file run bundle: " + str(path))
            config_name = next(name for name in files if name.endswith("/config.json"))
            csv_name = next(name for name in files if name.endswith("/per_corruption.csv"))
            summary_name = next(name for name in files if name.endswith("/summary.csv"))
            config = json.loads(archive.read(config_name))
            csv_text = archive.read(csv_name).decode("utf-8")
            summary_text = archive.read(summary_name).decode("utf-8")
            root = next(iter(roots))
        run_path = path
    else:
        path = _config_path(path)
        files = [item.name for item in path.parent.iterdir() if item.is_file()]
        if len(files) != 7 or set(files) != SCREEN_FILES:
            raise ValueError("screen directory must contain the complete seven-file run bundle: " + str(path.parent))
        config = json.loads(path.read_text(encoding="utf-8"))
        csv_text = (path.parent / "per_corruption.csv").read_text(encoding="utf-8")
        summary_text = (path.parent / "summary.csv").read_text(encoding="utf-8")
        root = path.parent.name
        run_path = path
    if not isinstance(config, dict) or config.get("run_id") != root:
        raise ValueError("screen bundle root/run ID disagrees with config: " + str(run_path))
    return config, csv_text, summary_text, run_path


def _validate_manifest(manifest, name: str, run_path: Path) -> None:
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("screen run is missing " + name + ": " + str(run_path))
    for identity in manifest.values():
        if (not isinstance(identity, dict) or not isinstance(identity.get("path"), str) or
                not isinstance(identity.get("bytes"), int) or identity["bytes"] < 0 or
                not isinstance(identity.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"])):
            raise ValueError("screen run has an invalid " + name + " identity: " + str(run_path))


def _validate_split(config: dict, run_path: Path) -> None:
    cli = config["cli_args"]
    count = _parse_int(cli.get("gsd_development_count"), "declared count", run_path)
    split_seed = _parse_int(cli.get("gsd_split_seed"), "declared split seed", run_path)
    split = config.get("development_split")
    if not isinstance(split, dict) or set(split) != set(PILOT_CORRUPTIONS):
        raise ValueError("screen run has invalid development split: " + str(run_path))
    for corruption in PILOT_CORRUPTIONS:
        entry = split[corruption]
        if not isinstance(entry, dict):
            raise ValueError("screen run has invalid development split: " + str(run_path))
        total = _parse_int(entry.get("total_examples"), "split total examples", run_path)
        indices = entry.get("indices")
        if total <= 0 or not isinstance(indices, list) or len(indices) != count:
            raise ValueError("screen declared count/indices disagree: " + str(run_path))
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise ValueError("screen indices must be integers: " + str(run_path))
        if len(set(indices)) != len(indices) or any(index < 0 or index >= total for index in indices):
            raise ValueError("screen indices contain duplicates or out-of-range values: " + str(run_path))
        if entry.get("split_seed") != split_seed or indices != development_indices(total, count, split_seed):
            raise ValueError("screen indices disagree with the declared split seed/count: " + str(run_path))
    reference_split = config["calibration_reference"].get("development_split")
    if not isinstance(reference_split, dict) or set(reference_split) != set(PILOT_CORRUPTIONS):
        raise ValueError("screen run is missing calibration split provenance: " + str(run_path))
    for corruption in PILOT_CORRUPTIONS:
        reference_indices = reference_split[corruption].get("indices")
        if not isinstance(reference_indices, list) or config["development_split"][corruption]["indices"][:len(reference_indices)] != reference_indices:
            raise ValueError("screen development split does not preserve calibration prefix: " + str(run_path))


def _validate_protocol(config: dict, run_path: Path) -> None:
    if config.get("stage") != "development" or config.get("execution_status") != "complete":
        raise ValueError("screen run must be a fully executed development bundle: " + str(run_path))
    if config.get("status") != "partial" or config.get("completed_corruptions") != list(PILOT_CORRUPTIONS):
        raise ValueError("screen run must cover Gaussian and Impulse: " + str(run_path))
    reference = config.get("calibration_reference")
    if (not isinstance(reference, dict) or not isinstance(reference.get("run_id"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256")))):
        raise ValueError("screen run is missing calibration provenance: " + str(run_path))
    for key, expected in SCREEN_CONFIG.items():
        if not _equal(config.get(key), expected):
            raise ValueError("screen protocol mismatch for " + key + ": " + str(run_path))
    cli = config.get("cli_args")
    if not isinstance(cli, dict):
        raise ValueError("screen run is missing CLI metadata: " + str(run_path))
    for key, expected in SCREEN_CLI.items():
        if not _equal(cli.get(key), expected):
            label = "SCD weight" if key == "gsd_scd_weight" else "graph modes" if key == "gsd_modes" else key
            raise ValueError("screen protocol mismatch for " + label + ": " + str(run_path))
    if config.get("seed") != cli.get("seed") or config.get("batch_size") != cli.get("batch_size"):
        raise ValueError("screen config/CLI metadata disagree: " + str(run_path))
    for manifest_name in ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest"):
        manifest = config.get(manifest_name)
        _validate_manifest(manifest, manifest_name, run_path)
        if reference.get("expected_manifests", {}).get(manifest_name) != manifest:
            raise ValueError("screen calibration manifest disagrees for " + manifest_name + ": " + str(run_path))
    spectral = config.get("spectral")
    if not isinstance(spectral, dict):
        raise ValueError("screen run is missing spectral contract: " + str(run_path))
    for field, cli_field in (("weight", "gsd_weight"), ("scd_weight", "gsd_scd_weight"),
                             ("profile", "gsd_profile"), ("beta", "gsd_beta"), ("k", "gsd_k"),
                             ("delta", "gsd_delta"), ("graph_gamma", "gsd_graph_gamma")):
        if not _equal(spectral.get(field), cli.get(cli_field)):
            raise ValueError("screen spectral/CLI contract disagrees for " + field + ": " + str(run_path))
    _validate_split(config, run_path)
    expected_timesteps = list(range(990, -1, -10))
    if config.get("scheduler_timesteps") != {name: expected_timesteps for name in PILOT_CORRUPTIONS}:
        raise ValueError("screen protocol mismatch for scheduler timesteps: " + str(run_path))
    count = _parse_int(cli["gsd_development_count"], "declared count", run_path)
    expected_batches = [32] * (count // 32)
    if config.get("observed_batch_sizes") != {name: expected_batches for name in PILOT_CORRUPTIONS}:
        raise ValueError("screen observed batch sizes disagree with count: " + str(run_path))
    randomness = config.get("randomness")
    if not isinstance(randomness, dict) or any(randomness.get(key) != value for key, value in
                                               (("seed", config["seed"]), ("num_workers", 0), ("drop_last", False),
                                                ("data_order", "fixed shuffled development indices; loader shuffle=False"))):
        raise ValueError("screen randomness metadata disagrees with protocol: " + str(run_path))


def _read_screen(path: Path) -> tuple[dict, dict[str, dict]]:
    config, csv_text, summary_text, run_path = _read_screen_bundle(path)
    _validate_protocol(config, run_path)
    split = config["development_split"]
    rows = _read_csv(csv_text, PER_CORRUPTION_FIELDS, "per_corruption.csv", run_path)
    if len(rows) != len(PILOT_CORRUPTIONS):
        raise ValueError("screen CSV must contain exactly Gaussian and Impulse rows: " + str(run_path))
    rows_by_corruption = {}
    for row in rows:
        if row["corruption"] in rows_by_corruption:
            raise ValueError("duplicate corruption result row: " + str(run_path))
        if (row["run_id"] != config["run_id"] or row["dataset"] != config["dataset"] or
                _parse_int(row["severity"], "CSV severity", run_path) != config["severity"] or
                row["method"] != config["method"] or _parse_int(row["seed"], "CSV seed", run_path) != config["seed"] or
                row["status"] != config["status"]):
            raise ValueError("screen CSV run ID/status disagrees with config: " + str(run_path))
        if row["corruption"] not in split:
            raise ValueError("screen CSV contains an unexpected corruption: " + str(run_path))
        n_examples = _parse_int(row["n_examples"], "CSV count", run_path)
        n_correct = _parse_int(row["n_correct"], "CSV correct count", run_path)
        accuracy = _parse_finite(row["accuracy"], "CSV accuracy", run_path)
        _parse_finite(row["runtime_seconds"], "CSV runtime", run_path, nonnegative=True)
        _parse_finite(row["peak_gpu_memory_mb"], "CSV peak memory", run_path, nonnegative=True)
        if n_examples != len(split[row["corruption"]]["indices"]):
            raise ValueError("screen subset coverage/count is inconsistent: " + str(run_path))
        if n_correct < 0 or n_correct > n_examples or not math.isclose(accuracy, n_correct / n_examples, rel_tol=0, abs_tol=1e-12):
            raise ValueError("screen CSV accuracy/counts disagree: " + str(run_path))
        rows_by_corruption[row["corruption"]] = row
    if set(rows_by_corruption) != set(PILOT_CORRUPTIONS):
        raise ValueError("screen CSV is missing Gaussian or Impulse: " + str(run_path))
    summary_rows = _read_csv(summary_text, SUMMARY_FIELDS, "summary.csv", run_path)
    if len(summary_rows) != 1:
        raise ValueError("summary CSV must contain exactly one row: " + str(run_path))
    summary = summary_rows[0]
    if (summary["run_id"] != config["run_id"] or summary["dataset"] != config["dataset"] or
            _parse_int(summary["severity"], "summary CSV severity", run_path) != config["severity"] or
            summary["method"] != config["method"] or _parse_int(summary["seed"], "summary CSV seed", run_path) != config["seed"] or
            summary["status"] != config["status"] or _parse_int(summary["n_corruptions"], "summary CSV n_corruptions", run_path) != len(PILOT_CORRUPTIONS)):
        raise ValueError("summary CSV metadata disagrees with config: " + str(run_path))
    total_examples = sum(_parse_int(row["n_examples"], "CSV count", run_path) for row in rows_by_corruption.values())
    total_correct = sum(_parse_int(row["n_correct"], "CSV correct count", run_path) for row in rows_by_corruption.values())
    macro_accuracy = sum(_parse_finite(row["accuracy"], "CSV accuracy", run_path) for row in rows_by_corruption.values()) / len(PILOT_CORRUPTIONS)
    total_runtime = sum(_parse_finite(row["runtime_seconds"], "CSV runtime", run_path, nonnegative=True) for row in rows_by_corruption.values())
    if (_parse_int(summary["total_examples"], "summary CSV total examples", run_path) != total_examples or
            _parse_int(summary["total_correct"], "summary CSV total correct", run_path) != total_correct or
            not math.isclose(_parse_finite(summary["macro_accuracy"], "summary CSV macro accuracy", run_path), macro_accuracy, rel_tol=0, abs_tol=1e-12) or
            not math.isclose(_parse_finite(summary["micro_accuracy"], "summary CSV micro accuracy", run_path), total_correct / total_examples, rel_tol=0, abs_tol=1e-12) or
            not math.isclose(_parse_finite(summary["total_runtime_seconds"], "summary CSV runtime", run_path, nonnegative=True), total_runtime, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("summary CSV counts or aggregate metrics disagree: " + str(run_path))
    return config, rows_by_corruption


def _common_screen_runs(paths: list[Path], calibration_run_id=None) -> list[tuple[dict, dict[str, dict]]]:
    runs = [_read_screen(path) for path in paths]
    if not runs:
        raise ValueError("provide completed screen run directories or ZIPs")
    first = runs[0][0]
    reference_hash = first["calibration_reference"]["sha256"]
    common_split = first["development_split"]
    for config, _ in runs:
        reference = config["calibration_reference"]
        if reference["sha256"] != reference_hash:
            raise ValueError("screen runs use different calibration files")
        if calibration_run_id is not None and reference["run_id"] != calibration_run_id:
            raise ValueError("screen run references a different calibration run")
        if config["development_split"] != common_split:
            raise ValueError("screen runs use different development indices")
        for manifest in ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest"):
            if config[manifest] != first[manifest]:
                raise ValueError("screen runs use different " + manifest)
        if config["seed"] != first["seed"]:
            raise ValueError("screen runs use different seeds; rank one seed at a time")
        if config["cli_args"]["gsd_development_count"] != first["cli_args"]["gsd_development_count"]:
            raise ValueError("screen runs use different subset sizes")
    return runs


def _screen_condition(config: dict):
    args = config["cli_args"]
    weight = _parse_finite(args.get("gsd_weight"), "guidance weight", Path(config["run_id"]), nonnegative=True)
    beta = args.get("gsd_beta")
    profile = args.get("gsd_profile")
    rho = config["calibration_reference"].get("target_rho")
    if args.get("gsd_target_rho") != rho:
        raise ValueError("screen CLI rho disagrees with calibration provenance: " + config["run_id"])
    if weight == 0:
        if rho is not None or profile != "hard" or beta is not None:
            raise ValueError("SCD-only comparator has invalid spectral settings")
        return "scd_only", None, None, weight
    if rho not in RHOS:
        raise ValueError("guided run is missing its predeclared rho")
    if profile == "smooth" and beta in (.5, 2., 8.):
        return "beta_" + str(beta), beta, rho, weight
    if profile == "hard" and beta is None:
        return "hard", None, rho, weight
    raise ValueError("unexpected profile/beta in screen: " + repr((profile, beta)))


def _candidate_summary(candidate: dict, baseline: dict[str, dict]) -> dict:
    rows = candidate["rows"]
    per_corruption = {}
    for corruption in PILOT_CORRUPTIONS:
        row = rows[corruption]
        accuracy = _parse_finite(row["accuracy"], "screen accuracy", Path(candidate["run_id"]))
        per_corruption[corruption] = dict(
            accuracy=accuracy, n_examples=_parse_int(row["n_examples"], "screen count", Path(candidate["run_id"])),
            n_correct=_parse_int(row["n_correct"], "screen correct count", Path(candidate["run_id"])),
            delta_vs_scd_only_pp=100 * (accuracy - float(baseline[corruption]["accuracy"])))
    return dict(run_id=candidate["run_id"], weight=candidate["weight"], rho=candidate["rho"],
                per_corruption=per_corruption,
                total_correct=sum(item["n_correct"] for item in per_corruption.values()),
                macro_accuracy=sum(item["accuracy"] for item in per_corruption.values()) / len(PILOT_CORRUPTIONS))


def screen_ranking(paths: list[Path], calibration_run_id=None, calibrated_weights=None) -> dict:
    runs = _common_screen_runs(paths, calibration_run_id)
    first = runs[0][0]
    reference_hash = first["calibration_reference"]["sha256"]
    rho = None
    by_candidate = {}
    for config, rows in runs:
        key, _, declared_rho, weight = _screen_condition(config)
        if key != "scd_only":
            if rho is None:
                rho = declared_rho
            if rho != declared_rho:
                raise ValueError("guided runs use different rho values")
            if calibrated_weights is not None:
                alpha_by_rho = calibrated_weights[key]
                expected_weight = alpha_by_rho.get(str(declared_rho)) if alpha_by_rho else None
                if expected_weight is None or not math.isclose(
                        weight, expected_weight, rel_tol=1e-12, abs_tol=0):
                    raise ValueError("screen weight differs from the calibrated coefficient for " + key)
        if key in by_candidate:
            raise ValueError("duplicate run for candidate " + key)
        by_candidate[key] = dict(run_id=config["run_id"], rows=rows, weight=weight, rho=declared_rho)
    expected = {"scd_only", "beta_0.5", "beta_2.0", "beta_8.0", "hard"}
    if set(by_candidate) != expected:
        raise ValueError("need exactly SCD-only, beta .5/2/8 and matched hard; found " +
                         ", ".join(sorted(by_candidate)))
    ranked = []
    baseline = by_candidate["scd_only"]["rows"]
    for key, candidate in by_candidate.items():
        ranked.append(dict(candidate=key, **_candidate_summary(candidate, baseline)))
    ranked.sort(key=lambda row: row["macro_accuracy"], reverse=True)
    top = ranked[0]["macro_accuracy"]
    top_candidates = [row["candidate"] for row in ranked if abs(row["macro_accuracy"] - top) <= 1e-12]
    return dict(
        report_type="exploratory development accuracy ranking; not held-out confirmation",
        calibration_sha256=reference_hash, seed=first["seed"],
        examples_per_corruption=first["cli_args"]["gsd_development_count"], rho=rho,
        macro_definition="equal-weight mean of Gaussian and Impulse accuracies",
        ranking=ranked, best_observed_candidates=top_candidates,
        interpretation=("The top candidate is the observed development-set winner only. "
                        "Use repeated seeds/larger data and a locked held-out confirmation "
                        "before describing beta as accuracy-optimal."))


def interaction_ranking(paths: list[Path], calibration_run_id=None, calibrated_weights=None) -> dict:
    """Summarize the preregistered beta .5/2 by rho .001/.01 development cells."""
    runs = _common_screen_runs(paths, calibration_run_id)
    first = runs[0][0]
    by_condition = {}
    for config, rows in runs:
        key, beta, rho, weight = _screen_condition(config)
        if key == "scd_only":
            condition = key
        elif key in ("beta_0.5", "beta_2.0") and rho in (.001, .01):
            if calibrated_weights is not None:
                expected_weight = calibrated_weights[key].get(str(rho))
                if expected_weight is None or not math.isclose(weight, expected_weight, rel_tol=1e-12, abs_tol=0):
                    raise ValueError("interaction weight differs from the calibrated coefficient for " + key)
            condition = key + "_rho_" + str(rho)
        else:
            raise ValueError("interaction screen contains an unexpected condition: " + config["run_id"])
        if condition in by_condition:
            raise ValueError("duplicate interaction condition " + condition)
        by_condition[condition] = dict(run_id=config["run_id"], rows=rows, weight=weight, rho=rho, beta=beta)
    expected = {"scd_only", "beta_0.5_rho_0.001", "beta_0.5_rho_0.01",
                "beta_2.0_rho_0.001", "beta_2.0_rho_0.01"}
    if set(by_condition) != expected:
        raise ValueError("interaction screen needs SCD-only and beta .5/2 at rho .001/.01")
    baseline = by_condition["scd_only"]["rows"]
    cells = {"beta_0.5": {}, "beta_2.0": {}}
    for beta_key in cells:
        for rho in (.001, .01):
            condition = by_condition[beta_key + "_rho_" + str(rho)]
            cells[beta_key][str(rho)] = _candidate_summary(condition, baseline)
    return dict(
        report_type="exploratory development beta-rho interaction; not held-out confirmation",
        calibration_sha256=first["calibration_reference"]["sha256"], seed=first["seed"],
        examples_per_corruption=first["cli_args"]["gsd_development_count"],
        macro_definition="equal-weight mean of Gaussian and Impulse accuracies",
        scd_only=_candidate_summary(by_condition["scd_only"], baseline), beta_rho_cells=cells,
        interpretation=("This completes the declared beta .5/2 by rho .001/.01 development comparison. "
                        "It does not select a final coefficient or establish held-out performance."))


def screen_weight_ranking(paths: list[Path], calibration_report: dict) -> dict:
    """Compare SCD-only with beta=2 at its three preregistered rho values."""
    runs = _common_screen_runs(paths, calibration_report["run_id"])
    if len(runs) != 4:
        raise ValueError("weight screen requires exactly four bundles: SCD-only and beta2 at three rho values")
    first = runs[0][0]
    by_candidate = {}
    for config, rows in runs:
        if config["calibration_reference"].get("run_id") != calibration_report["run_id"]:
            raise ValueError("screen run references a different calibration run")
        if (config["calibration_reference"]["sha256"] != first["calibration_reference"]["sha256"] or
                config["development_split"] != first["development_split"] or
                config["seed"] != first["seed"] or
                config["cli_args"]["gsd_development_count"] != first["cli_args"]["gsd_development_count"]):
            raise ValueError("screen runs differ in calibration, split, seed or subset size")
        for manifest in ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest"):
            if config.get(manifest) != first.get(manifest):
                raise ValueError("screen runs use different " + manifest)
        key, _, rho, weight = _screen_condition(config)
        if key == "scd_only":
            key = "scd_only"
        else:
            if key != "beta_2.0" or rho not in RHOS:
                raise ValueError("weight screen contains a non-beta2 or undeclared-rho run")
            key = "beta2_rho_" + str(rho)
            expected_weight = calibration_report["candidates"]["2.0"]["weights"][str(rho)]
            if not math.isclose(weight, expected_weight, rel_tol=1e-12, abs_tol=0):
                raise ValueError("beta2 weight differs from the calibration report")
        if key in by_candidate:
            raise ValueError("duplicate weight-screen condition " + key)
        by_candidate[key] = dict(run_id=config["run_id"], rows=rows, weight=weight, rho=rho)
    expected = {"scd_only"} | {"beta2_rho_" + str(rho) for rho in RHOS}
    if set(by_candidate) != expected:
        raise ValueError("weight screen is missing one or more of baseline/beta2 rho conditions")
    baseline = by_candidate["scd_only"]["rows"]
    ranking = []
    for key, candidate in by_candidate.items():
        per_corruption = {}
        for corruption in PILOT_CORRUPTIONS:
            row = candidate["rows"][corruption]
            accuracy = float(row["accuracy"])
            per_corruption[corruption] = dict(
                accuracy=accuracy, n_examples=int(row["n_examples"]), n_correct=int(row["n_correct"]),
                delta_vs_scd_only_pp=100 * (accuracy - float(baseline[corruption]["accuracy"])))
        ranking.append(dict(candidate=key, run_id=candidate["run_id"],
                            rho=candidate["rho"], weight=candidate["weight"],
                            per_corruption=per_corruption,
                            macro_accuracy=sum(item["accuracy"] for item in per_corruption.values()) /
                            len(PILOT_CORRUPTIONS)))
    ranking.sort(key=lambda item: item["macro_accuracy"], reverse=True)
    top = ranking[0]["macro_accuracy"]
    top_candidates = [row["candidate"] for row in ranking if abs(row["macro_accuracy"] - top) <= 1e-12]
    return dict(report_type="exploratory development weight screen; seed 0, fixed subset",
                calibration_run_id=calibration_report["run_id"],
                calibration_sha256=first["calibration_reference"]["sha256"],
                seed=first["seed"], examples_per_corruption=first["cli_args"]["gsd_development_count"],
                macro_definition="equal-weight mean of Gaussian and Impulse accuracies",
                ranking=ranking, best_observed_rho_candidates=top_candidates,
                interpretation="Select rho only for the next exploratory beta comparison. This 128-example, one-seed ranking is not confirmation.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True,
                        help="diagnostic run directory or its config.json")
    parser.add_argument("--screen-run", type=Path, nargs="+", default=None,
                        help="screen ZIPs or run directories/configs")
    parser.add_argument("--weight-screen", action="store_true",
                        help="rank baseline plus beta2 at rho .0001/.001/.01 (four bundles)")
    parser.add_argument("--interaction-screen", action="store_true",
                        help="summarize SCD-only and beta .5/2 at rho .001/.01 (five bundles)")
    parser.add_argument("--rho", type=float, choices=RHOS,
                        help="optional expected screen rho, checked when --screen-run is set")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional compact summary path outside the raw run directory")
    args = parser.parse_args(argv)
    try:
        summary = calibration_summary(args.calibration)
        if args.screen_run:
            if args.weight_screen and args.interaction_screen:
                raise ValueError("choose only one of --weight-screen and --interaction-screen")
            if args.weight_screen:
                if args.rho is not None:
                    raise ValueError("--rho is not used for the three-rho weight screen")
                summary["development_weight_selection"] = screen_weight_ranking(
                    args.screen_run, {"run_id": summary["run_id"],
                                      "candidates": {"2.0": {"weights": next(
                                          row["alpha_by_rho"] for row in summary["smooth_candidates"]
                                          if row["beta"] == 2.0)}}})
            elif args.interaction_screen:
                if args.rho is not None:
                    raise ValueError("--rho is not used for the beta-rho interaction screen")
                weights = {"beta_0.5": next(row["alpha_by_rho"] for row in summary["smooth_candidates"]
                                              if row["beta"] == .5),
                           "beta_2.0": next(row["alpha_by_rho"] for row in summary["smooth_candidates"]
                                              if row["beta"] == 2.0)}
                summary["development_beta_rho_interaction"] = interaction_ranking(
                    args.screen_run, summary["run_id"], weights)
            else:
                weights = {"hard": summary["hard"]["alpha_by_rho"]}
                weights.update({"beta_" + str(row["beta"]): row["alpha_by_rho"]
                                for row in summary["smooth_candidates"]})
                ranking = screen_ranking(args.screen_run, summary["run_id"], weights)
                if args.rho is not None and ranking["rho"] != args.rho:
                    raise ValueError("screen rho differs from --rho")
                summary["development_accuracy_selection"] = ranking
        elif args.rho is not None:
            raise ValueError("--rho is only valid together with --screen-run")
        rendered = json.dumps(summary, indent=2, allow_nan=False)
        if args.output:
            source_path = _config_path(args.calibration).resolve()
            source = source_path.parent
            if source_path.name != "config.json" and not (source / "config.json").exists():
                source = source_path
            target = args.output.resolve()
            if target == source or (source.is_dir() and source in target.parents):
                raise ValueError("write compact summaries outside the immutable raw run directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(summary, separators=(",", ":"), allow_nan=False) + "\n",
                              encoding="utf-8")
            print("Compact summary written:", target)
        else:
            print(rendered)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
