"""Run/resume the fixed nine-run ModelNet40-C full-test-set GSD screen in Colab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gsd_calibration import (PINNED_REFERENCE_CONFIG_SHA256, PINNED_REFERENCE_RUN_ID,
                             load_calibration)
from gsd_protocol import SMOOTH_METHOD
from research_artifacts import CORRUPTIONS
from scripts.analyze_gsd_full_dataset_screen import (analyze_runs, load_pinned_calibration,
                                                      read_full_run)

METHOD_ROOT_PARTS = ("modelnet40_c", SMOOTH_METHOD)
CONDITIONS = (("scd", "hard", None, 0., None),
              ("beta0p5-rho0p001", "smooth", .5, 8.140161356429882, .001),
              ("beta2-rho0p01", "smooth", 2., 113.10823980075075, .01))


def build_screen_commands(calibration_path: Path, result_root: str) -> list[list[str]]:
    """Preflight pinned calibration, then build the immutable nine-command matrix."""
    calibration_path = Path(calibration_path)
    calibration = load_pinned_calibration(calibration_path)
    if calibration["run_id"] != PINNED_REFERENCE_RUN_ID or calibration["sha256"] != PINNED_REFERENCE_CONFIG_SHA256:
        raise ValueError("full-dataset screen requires the pinned calibration identity")
    report = calibration["report"]
    for beta_key, rho_key, expected in (("0.5", "0.001", 8.140161356429882),
                                        ("2.0", "0.01", 113.10823980075075)):
        observed = report["candidates"][beta_key]["weights"][rho_key]
        if abs(float(observed) - expected) > max(abs(expected) * 1e-12, 1e-15):
            raise ValueError("calibration-derived coefficient changed for beta " + beta_key + "/rho " + rho_key)
    reference_path = calibration_path / "config.json" if calibration_path.is_dir() else calibration_path
    commands = []
    for label, profile, beta, weight, rho in CONDITIONS:
        for seed in range(3):
            run_name = "gsd-full-screen-{}-{}-seed{}".format(PINNED_REFERENCE_RUN_ID, label, seed)
            command = [sys.executable, "-u", str(REPO / "run_baseline.py"),
                       "--method", SMOOTH_METHOD, "--batch_size", "32", "--seed", str(seed),
                       "--severity", "5", "--lambdaa", ".95", "--gamma", ".01", "--eta", ".01",
                       "--max-batches", "0", "--lion-eval-mode", "--result-root", result_root,
                       "--run-name", run_name, "--corruptions", *CORRUPTIONS,
                       "--gsd-stage", "full_dataset_development", "--gsd-weight", repr(weight),
                       "--gsd-scd-weight", "1", "--gsd-profile", profile,
                       "--gsd-calibration-reference", str(reference_path)]
            if beta is not None:
                command += ["--gsd-beta", repr(beta), "--gsd-target-rho", repr(rho)]
            commands.append(command)
    return commands


def run_streamed(command: list[str]) -> tuple[int, str | None, str | None]:
    print("$ " + shlex.join(command), flush=True)
    process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    run_directory = zip_path = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("Run directory:"):
            run_directory = line.split(":", 1)[1].strip()
        elif line.startswith("ZIP to provide:"):
            zip_path = line.split(":", 1)[1].strip()
    return process.wait(), run_directory, zip_path


def _resolved(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO / path).resolve()


def _method_root(result_root: str | Path) -> Path:
    return _resolved(result_root).joinpath(*METHOD_ROOT_PARTS)


def _candidate_archives(method_root: Path, run_name: str) -> list[Path]:
    return sorted(method_root.glob("*_{0}.zip".format(run_name))) if method_root.exists() else []


def _valid_resume_archive(path: Path, run_name: str, expected_arm: str, seed: int) -> bool:
    try:
        run = read_full_run(path)
    except (OSError, ValueError):
        return False
    config = run["config"]
    return (run["arm"] == expected_arm and run["seed"] == seed and
            config.get("cli_args", {}).get("run_name") == run_name)


def _free_run_name(base: str, method_root: Path) -> str:
    suffix = 1
    while True:
        name = base if suffix == 1 else "{}-attempt{:02d}".format(base, suffix)
        if not _candidate_archives(method_root, name) and not list(method_root.glob("*_{0}".format(name))):
            return name
        suffix += 1


def run_screen(calibration_path: Path, result_root: str, *, resume=False,
               summary_output: Path | None = None) -> Path:
    calibration = load_pinned_calibration(calibration_path)
    commands = build_screen_commands(calibration_path, result_root)
    method_root = _method_root(result_root)
    completed = []
    for command in commands:
        base_name = command[command.index("--run-name") + 1]
        arm = "scd_only" if "-scd-" in base_name else (
            "beta_0.5_rho_0.001" if "beta0p5" in base_name else "beta_2_rho_0.01")
        seed = int(command[command.index("--seed") + 1])
        valid = [path for path in _candidate_archives(method_root, base_name)
                 if _valid_resume_archive(path, base_name, arm, seed)]
        if len(valid) > 1:
            raise ValueError("multiple valid archives exist for " + base_name)
        if resume and valid:
            print("RESUME SKIP (validated complete):", valid[0], flush=True)
            completed.append(valid[0])
            continue
        if valid and not resume:
            print("Existing completion retained; starting a distinct attempt:", valid[0], flush=True)
        run_name = _free_run_name(base_name, method_root)
        command[command.index("--run-name") + 1] = run_name
        print("Starting arm={} seed={} run_name={}".format(arm, seed, run_name), flush=True)
        code, run_directory, zip_text = run_streamed(command)
        if code:
            print("STOP: command failed with exit code {}: {}".format(code, shlex.join(command)), flush=True)
            print("Completed archives:", [str(path) for path in completed], flush=True)
            print("Failed run directory:", run_directory, flush=True)
            raise RuntimeError("full-dataset screen command failed")
        if not zip_text:
            raise RuntimeError("runner completed without reporting its ZIP archive")
        archive_path = _resolved(zip_text)
        run = read_full_run(archive_path)
        if run["arm"] != arm or run["seed"] != seed or run["config"]["cli_args"].get("run_name") != run_name:
            raise ValueError("new archive does not match its requested arm, seed, and run name")
        completed.append(archive_path)
        print("Validated completed ZIP:", archive_path, flush=True)
    summary = analyze_runs(completed, calibration)
    output = summary_output or (_method_root(result_root) /
                                ("full_test_screen_summary_{}.json".format(PINNED_REFERENCE_RUN_ID)))
    output = _resolved(output)
    if output == calibration_path.resolve() or output in {path.resolve() for path in completed}:
        raise ValueError("summary output cannot overwrite calibration or a raw run archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote derived summary:", output, flush=True)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Pinned raw calibration config.json or its run directory")
    parser.add_argument("--result-root", default="./result")
    parser.add_argument("--resume", action="store_true",
                        help="Skip only complete, validated matching arm/seed ZIPs")
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != REPO:
        raise SystemExit("Run from the repository root inside the existing 3dd_tta_env Colab environment.")
    run_screen(args.calibration, args.result_root, resume=args.resume, summary_output=args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
