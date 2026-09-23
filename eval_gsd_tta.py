"""Print or execute locked Colab GSD experiments with matched baseline controls."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from gsd_protocol import METHOD, PILOT_CORRUPTIONS
from research_artifacts import CORRUPTIONS

REPO = Path(__file__).resolve().parent


def build_commands(stage: str, *, seeds=None, arms=("baseline", "off", "on"),
                   smoke_corruption="gaussian", result_root="./result") -> list[list[str]]:
    """Construct fixed controls; custom method tuning uses the explicit runner CLI."""
    if stage not in ("smoke", "pilot", "all15"):
        raise ValueError("Unknown experiment stage")
    seeds = ([0] if stage == "smoke" else [0, 1, 2]) if seeds is None else list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in (0, 1, 2) for s in seeds):
        raise ValueError("Use unique seeds from 0/1/2")
    if not arms or len(set(arms)) != len(arms) or any(a not in ("baseline", "off", "on") for a in arms):
        raise ValueError("Use unique arms from baseline/off/on")
    if smoke_corruption not in ("gaussian", "background"):
        raise ValueError("Smoke supports Gaussian or Background")
    corruptions = ([smoke_corruption] if stage == "smoke" else
                   list(PILOT_CORRUPTIONS) if stage == "pilot" else list(CORRUPTIONS))
    commands = []
    for seed in seeds:
        for arm in arms:
            command = [sys.executable, "-u", str(REPO / "run_baseline.py"),
                       "--method", "3dd_original" if arm == "baseline" else METHOD,
                       "--batch_size", "32", "--seed", str(seed), "--severity", "5",
                       "--max-batches", "2" if stage == "smoke" else "0",
                       "--lion-eval-mode", "--result-root", result_root,
                       "--run-name", f"gsd-v1-{stage}-" +
                       (f"{smoke_corruption}-" if stage == "smoke" else "") + f"{arm}-seed{seed}",
                       "--corruptions", *corruptions]
            if arm != "baseline":
                command += ["--gsd-stage", "benchmark" if stage == "all15" else stage,
                            "--gsd-weight", "0" if arm == "off" else "1",
                            "--gsd-k", "10", "--gsd-delta", ".1",
                            "--gsd-graph-gamma", ".6", "--gsd-modes", "100"]
            commands.append(command)
    return commands


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot", "all15"), default="pilot")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--arms", choices=("baseline", "off", "on"), nargs="+",
                        default=["baseline", "off", "on"])
    parser.add_argument("--smoke-corruption", choices=("gaussian", "background"), default="gaussian")
    parser.add_argument("--result-root", default="./result")
    parser.add_argument("--execute", action="store_true", help="Run in the existing Colab environment")
    args = parser.parse_args(argv)
    commands = build_commands(args.stage, seeds=args.seeds, arms=args.arms,
                              smoke_corruption=args.smoke_corruption, result_root=args.result_root)
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            completed = subprocess.run(command, cwd=REPO)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
