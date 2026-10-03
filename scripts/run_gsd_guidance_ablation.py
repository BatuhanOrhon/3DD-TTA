"""Shared fixed-protocol launcher for the two missing guidance ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gsd_calibration import PINNED_REFERENCE_RUN_ID
from research_artifacts import CORRUPTIONS
from scripts.analyze_gsd_full_dataset_screen import read_full_run
from scripts.run_gsd_full_dataset_screen import (
    build_screen_commands, run_streamed, _method_root, _free_run_name, _resolved)


def build_commands(arm: str, calibration: Path, result_root: str) -> list[list[str]]:
    if arm not in ("unguided", "smooth_only"):
        raise ValueError("unknown guidance ablation")
    commands = build_screen_commands(calibration, result_root)
    commands = commands[:3] if arm == "unguided" else commands[3:6]
    for command in commands:
        seed = command[command.index("--seed") + 1]
        command[command.index("--gsd-stage") + 1] = "full_dataset_ablation"
        command[command.index("--gsd-scd-weight") + 1] = "0"
        command[command.index("--run-name") + 1] = (
            f"gsd-ablation-{arm}-{PINNED_REFERENCE_RUN_ID}-seed{seed}")
    return commands


def _matching_runs(root: Path, base: str, arm: str, seed: int) -> list[dict]:
    matches = []
    for path in sorted(root.glob(f"*_{base}*.zip")):
        logical = path.stem.split("_", 1)[-1]
        if logical != base and not logical.startswith(base + "-attempt"):
            continue
        try:
            run = read_full_run(path, guidance_ablation=True)
        except (OSError, ValueError) as error:
            print(f"Incomplete/invalid archive retained: {path}: {error}", flush=True)
            continue
        if (run["arm"], run["seed"], run["config"]["cli_args"]["run_name"]) != (arm, seed, logical):
            raise ValueError("archive name and ablation identity disagree: " + str(path))
        matches.append(run)
    if len(matches) > 1:
        raise ValueError("multiple complete archives; resolve duplicate before continuing: " + base)
    return matches


def _check_sources(run: dict) -> None:
    # Use the same exact byte identity function as the worker.
    from run_baseline import file_identity
    for name, identity in run["config"]["runtime_source_manifest"].items():
        if file_identity(REPO / name) != identity:
            raise ValueError("completed run uses different source; refusing to rerun or mix: " + name)


def summarize_runs(runs: list[dict], arm: str) -> dict:
    if sorted(run["seed"] for run in runs) != [0, 1, 2] or any(run["arm"] != arm for run in runs):
        raise ValueError("summary requires exactly the three requested seeds")
    first = runs[0]["config"]
    for run in runs[1:]:
        for key in ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest",
                    "extension_inventory", "dataset_inventory"):
            if run["config"][key] != first[key]:
                raise ValueError("ablation seed runs differ in " + key)
        for name in CORRUPTIONS:
            if run["config"]["per_example_predictions"][name]["labels"] != first["per_example_predictions"][name]["labels"]:
                raise ValueError("ablation seed runs differ in label order")
    def stats(values):
        return {"mean_percent": 100 * statistics.mean(values),
                "sample_sd_pp": 100 * statistics.stdev(values)}
    return {
        "arm": arm, "calibration_reference": first["calibration_reference"],
        "protocol": {"seeds": [0, 1, 2], "severity": 5, "examples_per_corruption": 2468,
                     "corruptions": list(CORRUPTIONS), "scd_weight": 0,
                     "spectral": first["spectral"]},
        "evidence_status": "full-test-set development ablation; historical controls are not common-draw paired",
        "macro_accuracy": stats([float(run["summary"]["macro_accuracy"]) for run in runs]),
        "per_corruption": {name: stats([float(run["rows"][name]["accuracy"]) for run in runs])
                           for name in CORRUPTIONS},
        "runs": [{key: run[key] for key in ("path", "archive_sha256", "run_id", "seed", "summary")}
                 for run in runs]}


def run_ablation(arm: str, calibration: Path, result_root: str) -> Path:
    commands = build_commands(arm, calibration, result_root)
    root = _method_root(result_root)
    runs = []
    for command in commands:
        base = command[command.index("--run-name") + 1]
        seed = int(command[command.index("--seed") + 1])
        existing = _matching_runs(root, base, arm, seed)
        if existing:
            _check_sources(existing[0])
            print("SKIP validated completion:", existing[0]["path"], flush=True)
            runs.extend(existing)
            continue
        name = _free_run_name(base, root)
        command[command.index("--run-name") + 1] = name
        code, directory, zip_text = run_streamed(command)
        if code or not zip_text:
            raise RuntimeError(f"ablation stopped; exit={code}, run={directory}; rerun this launcher to resume")
        run = read_full_run(_resolved(zip_text), guidance_ablation=True)
        if (run["arm"], run["seed"], run["config"]["cli_args"]["run_name"]) != (arm, seed, name):
            raise ValueError("new archive differs from requested run")
        _check_sources(run)
        runs.append(run)
    summary = summarize_runs(runs, arm)
    output = root / f"guidance_ablation_summary_{arm}_{PINNED_REFERENCE_RUN_ID}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Summary:", output, flush=True)
    return output


def main(arm: str, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"Run 3 full ModelNet40-C seeds: {arm}; resume automatically.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--result-root", default="./result")
    parser.add_argument("--dry-run", action="store_true", help="Validate calibration and print commands; no GPU runs")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != REPO:
        raise SystemExit("Run from /content/3DD-TTA in the existing 3dd_tta_env environment")
    if args.dry_run:
        for command in build_commands(arm, args.calibration, args.result_root):
            print(shlex.join(command))
    else:
        run_ablation(arm, args.calibration, args.result_root)
    return 0
