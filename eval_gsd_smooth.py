"""Build or execute the paired GSD smooth-spectrum v2 Colab pilot matrix."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

from gsd_protocol import METHOD, PILOT_CORRUPTIONS, SMOOTH_METHOD

REPO = Path(__file__).resolve().parent


def build_commands(stage: str, *, beta: float, hard_weight: float,
                   smooth_weight: float, seeds=None,
                   arms=("v1", "hard", "smooth"),
                   smoke_corruption="gaussian", result_root="./result") -> list[list[str]]:
    """Build matched v1, v2 hard-M and v2 smooth-profile commands.

    Coefficients and beta have no defaults by design; choose them before score
    inspection and supply the recorded calibration values explicitly.
    """
    if stage not in ("smoke", "pilot"):
        raise ValueError("GSD smooth v2 supports smoke or pilot only")
    for name, value in (("beta", beta), ("hard_weight", hard_weight),
                        ("smooth_weight", smooth_weight)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name + " must be finite")
    if beta <= 0 or hard_weight < 0 or smooth_weight < 0:
        raise ValueError("beta must be positive and weights must be non-negative")
    seeds = ([0] if stage == "smoke" else [0, 1, 2]) if seeds is None else list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (0, 1, 2) for seed in seeds):
        raise ValueError("Use unique seeds from 0/1/2")
    if not arms or len(set(arms)) != len(arms) or any(
            arm not in ("v1", "hard", "smooth") for arm in arms):
        raise ValueError("Use unique arms from v1/hard/smooth")
    if smoke_corruption not in ("gaussian", "background"):
        raise ValueError("Smoke supports Gaussian or Background")
    corruptions = [smoke_corruption] if stage == "smoke" else list(PILOT_CORRUPTIONS)
    commands = []
    for seed in seeds:
        for arm in arms:
            is_v1 = arm == "v1"
            method = METHOD if is_v1 else SMOOTH_METHOD
            profile = None if is_v1 else arm
            weight = 1.0 if is_v1 else (hard_weight if profile == "hard" else smooth_weight)
            run_name = f"gsd-smooth-v2-{stage}-{arm}-seed{seed}"
            if profile == "smooth":
                run_name += f"-beta{beta}"
            command = [sys.executable, "-u", str(REPO / "run_baseline.py"),
                       "--method", method, "--batch_size", "32", "--seed", str(seed),
                       "--severity", "5", "--lambdaa", ".95", "--gamma", ".01", "--eta", ".01",
                       "--max-batches", "1" if stage == "smoke" else "0",
                       "--lion-eval-mode", "--result-root", result_root,
                       "--run-name", run_name, "--corruptions", *corruptions]
            gsd_stage = "smoke" if stage == "smoke" else "pilot"
            command += ["--gsd-stage", gsd_stage,
                        "--gsd-weight", str(weight), "--gsd-scd-weight", "1",
                        "--gsd-k", "10", "--gsd-delta", ".1",
                        "--gsd-graph-gamma", ".6", "--gsd-modes", "100"]
            if profile is not None:
                command += ["--gsd-profile", profile]
            if profile == "smooth":
                command += ["--gsd-beta", str(beta)]
            commands.append(command)
    return commands


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--beta", type=float, required=True,
                        help="Predeclared smooth-profile beta; no accuracy-selected default")
    parser.add_argument("--hard-weight", type=float, required=True,
                        help="Calibrated coefficient for v2 hard-M arm")
    parser.add_argument("--smooth-weight", type=float, required=True,
                        help="Calibrated coefficient for v2 smooth arm")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--arms", choices=("v1", "hard", "smooth"), nargs="+",
                        default=["v1", "hard", "smooth"])
    parser.add_argument("--smoke-corruption", choices=("gaussian", "background"), default="gaussian")
    parser.add_argument("--result-root", default="./result")
    parser.add_argument("--execute", action="store_true", help="Run in the existing Colab environment")
    args = parser.parse_args(argv)
    try:
        commands = build_commands(
            args.stage, beta=args.beta, hard_weight=args.hard_weight,
            smooth_weight=args.smooth_weight, seeds=args.seeds, arms=args.arms,
            smoke_corruption=args.smoke_corruption, result_root=args.result_root)
    except ValueError as error:
        parser.error(str(error))
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            completed = subprocess.run(command, cwd=REPO)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
