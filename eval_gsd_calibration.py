"""Colab common-state diagnostic and staged development-screen launcher."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from gsd_calibration import RHOS, load_calibration
from gsd_protocol import SMOOTH_METHOD

REPO = Path(__file__).resolve().parent


def build_commands(phase: str, *, calibration=None, rho=None, result_root="./result",
                   seed=0, count=None) -> list[list[str]]:
    if phase not in ("diagnose", "screen-weight", "screen-beta"):
        raise ValueError("unknown calibration phase")
    if seed not in (0, 1, 2) or (phase == "diagnose" and seed != 0):
        raise ValueError("diagnostics require seed 0; screening supports 0/1/2")
    count = (64 if phase == "diagnose" else 128) if count is None else count
    if count not in range(32, 513, 32):
        raise ValueError("count must be a multiple of 32 in [32,512]")
    split_seed = 20260927
    reference = None
    if phase == "diagnose":
        if calibration is not None or rho is not None:
            raise ValueError("diagnose takes no calibration reference or selected rho")
        conditions = [("reference", "hard", None, 0., None)]
    else:
        if calibration is None:
            raise ValueError("screening requires --calibration pointing to a completed run/config.json")
        reference = Path(calibration).resolve()
        if reference.is_dir():
            reference = reference / "config.json"
        config, report = load_calibration(reference)
        split_seed = config["cli_args"]["gsd_split_seed"]
        if phase == "screen-weight":
            if rho is not None:
                raise ValueError("screen-weight uses all three preregistered rho values")
            conditions = [("baseline", "hard", None, 0., None)]
            selected = [("smooth", 2., value) for value in RHOS]
        else:
            if rho not in RHOS:
                raise ValueError("screen-beta requires an explicit --rho from 0.0001/0.001/0.01")
            conditions = []
            selected = [("smooth", .5, rho), ("smooth", 8., rho), ("hard", None, rho)]
        for profile, beta, value in selected:
            key = "hard" if beta is None else str(beta)
            weights = report["candidates"][key]["weights"]
            if weights is None:
                raise ValueError("candidate " + key + " has no usable coefficient; review diagnostic rows")
            alpha = weights[str(value)]
            if not math.isfinite(alpha) or alpha <= 0:
                raise ValueError("calibrated coefficient must be finite and positive")
            conditions.append((profile + ("-beta" + str(beta) if beta is not None else "") + "-rho" + str(value),
                               profile, beta, alpha, value))
    commands = []
    for label, profile, beta, alpha, contribution in conditions:
        name = f"gsd-cal-{phase}-{label}-seed{seed}-n{count}".replace(".", "p").replace("+", "")
        command = [sys.executable, "-u", str(REPO / "run_baseline.py"),
                   "--method", SMOOTH_METHOD, "--batch_size", "32", "--seed", str(seed),
                   "--severity", "5", "--lambdaa", ".95", "--gamma", ".01", "--eta", ".01",
                   "--max-batches", "0", "--lion-eval-mode", "--result-root", str(result_root),
                   "--run-name", name, "--corruptions", "gaussian", "impulse",
                   "--gsd-stage", "calibrate" if phase == "diagnose" else "development",
                   "--gsd-weight", str(alpha), "--gsd-scd-weight", "1",
                   "--gsd-profile", profile, "--gsd-development-count", str(count),
                   "--gsd-split-seed", str(split_seed)]
        if beta is not None:
            command += ["--gsd-beta", str(beta)]
        if reference is not None:
            command += ["--gsd-calibration-reference", str(reference)]
            if contribution is not None:
                command += ["--gsd-target-rho", str(contribution)]
        commands.append(command)
    return commands


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("diagnose", "report", "screen-weight", "screen-beta"), required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--result-root", default="./result")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.phase == "report":
            if args.calibration is None:
                raise ValueError("report requires --calibration")
            _, report = load_calibration(args.calibration)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        commands = build_commands(args.phase, calibration=args.calibration, rho=args.rho,
                                  result_root=args.result_root, seed=args.seed, count=args.count)
    except (ValueError, KeyError, OSError) as error:
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
