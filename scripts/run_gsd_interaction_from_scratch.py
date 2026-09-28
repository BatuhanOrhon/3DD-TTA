"""Run the GSD beta/rho interaction from one fresh diagnostic reference.

Run from the repository root inside the existing 3dd_tta_env Colab environment.
By default, validate the four existing prerequisite bundles. With
``--rebuild-prerequisites``, regenerate only the required four cells under the
fresh reference before launching the single new beta .5/rho .01 arm.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gsd_calibration import load_calibration
from gsd_protocol import SMOOTH_METHOD
from scripts.analyze_gsd_calibration import _read_screen, _screen_condition


RESULT_ROOT = Path("./result")
METHOD_ROOT = RESULT_ROOT / "modelnet40_c" / "gsd_latent_spectral_smooth_v2"
ARCHIVED_CELLS = (
    ("20260927-203752_gsd-cal-screen-weight-baseline-seed0-n128.zip", "scd_only", None, None),
    ("20260927-205045_gsd-cal-screen-beta-smooth-beta0p5-rho0p001-seed0-n128.zip",
     "beta_0.5", .001, "0.5"),
    ("20260927-203909_gsd-cal-screen-weight-smooth-beta2p0-rho0p001-seed0-n128.zip",
     "beta_2.0", .001, "2.0"),
    ("20260927-203957_gsd-cal-screen-weight-smooth-beta2p0-rho0p01-seed0-n128.zip",
     "beta_2.0", .01, "2.0"),
)


def build_rebuild_commands(calibration_config: dict, report: dict,
                           calibration_config_path: str,
                           result_root: str = "./result") -> list[list[str]]:
    """Build only the prerequisite cells needed for the beta-rho interaction."""
    alpha05 = report["candidates"]["0.5"]["weights"]
    alpha2 = report["candidates"]["2.0"]["weights"]
    if not alpha05 or not alpha2:
        raise ValueError("Fresh calibration did not produce usable beta .5/beta 2 coefficients.")
    selected_weights = (alpha05["0.001"], alpha2["0.001"], alpha2["0.01"])
    if any(not math.isfinite(float(weight)) or float(weight) <= 0 for weight in selected_weights):
        raise ValueError("Prerequisite guidance coefficients must be finite and positive.")

    calibration_id = calibration_config["run_id"]
    split_seed = str(calibration_config["cli_args"]["gsd_split_seed"])
    conditions = (
        ("baseline", "hard", None, 0.0, None),
        ("smooth-beta0p5-rho0p001", "smooth", .5, alpha05["0.001"], .001),
        ("smooth-beta2p0-rho0p001", "smooth", 2.0, alpha2["0.001"], .001),
        ("smooth-beta2p0-rho0p01", "smooth", 2.0, alpha2["0.01"], .01),
    )
    commands = []
    for label, profile, beta, alpha, rho in conditions:
        name = "gsd-rebuild-{}-{}-seed0-n128".format(calibration_id, label)
        command = [
            sys.executable, "-u", str(REPO / "run_baseline.py"),
            "--method", SMOOTH_METHOD, "--batch_size", "32", "--seed", "0",
            "--severity", "5", "--lambdaa", ".95", "--gamma", ".01", "--eta", ".01",
            "--max-batches", "0", "--lion-eval-mode", "--result-root", result_root,
            "--run-name", name, "--corruptions", "gaussian", "impulse",
            "--gsd-stage", "development", "--gsd-weight", str(alpha),
            "--gsd-scd-weight", "1", "--gsd-profile", profile,
            "--gsd-development-count", "128", "--gsd-split-seed", split_seed,
            "--gsd-calibration-reference", calibration_config_path,
        ]
        if beta is not None:
            command += ["--gsd-beta", str(beta)]
        if rho is not None:
            command += ["--gsd-target-rho", str(rho)]
        commands.append(command)
    return commands


def run_streamed(command: list[str]) -> tuple[int, str | None, str | None]:
    """Run a command while preserving live Colab output and capture run paths."""
    print("$ " + shlex.join(command), flush=True)
    process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    run_directory = None
    zip_path = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("Run directory:"):
            run_directory = line.split(":", 1)[1].strip()
        elif line.startswith("ZIP to provide:"):
            zip_path = line.split(":", 1)[1].strip()
    return process.wait(), run_directory, zip_path


def _resolved(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO / path).resolve()


def verify_archived_cells(calibration_config: dict, report: dict,
                          paths: list[Path] | None = None) -> list[Path]:
    """Verify prerequisite bundles against one calibration reference."""
    expected_manifests = {key: calibration_config[key] for key in
                          ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest")}
    expected_split = calibration_config["development_split"]
    alpha05 = report["candidates"]["0.5"]["weights"]
    alpha2 = report["candidates"]["2.0"]["weights"]
    if not alpha05 or not alpha2:
        raise ValueError("Fresh calibration did not produce usable beta .5/beta 2 coefficients.")

    expected_cells = [(key, rho, beta_key) for _, key, rho, beta_key in ARCHIVED_CELLS]
    if paths is None:
        paths = [REPO / METHOD_ROOT / filename for filename, _, _, _ in ARCHIVED_CELLS]
    if len(paths) != len(expected_cells):
        raise ValueError("Interaction comparison requires exactly four prerequisite bundles.")
    for path, (expected_key, rho, beta_key) in zip(paths, expected_cells):
        if not path.is_file():
            raise FileNotFoundError("Required interaction bundle is missing: " + str(path))
        config, _ = _read_screen(path)
        if (config["seed"] != 0 or config["cli_args"]["gsd_development_count"] != 128 or
                config["cli_args"]["gsd_split_seed"] != calibration_config["cli_args"]["gsd_split_seed"]):
            raise ValueError("Interaction seed/count/split seed differs: " + path.name)
        if config["development_split"]["gaussian"]["indices"] != config["development_split"]["impulse"]["indices"]:
            raise ValueError("Gaussian/Impulse development indices differ: " + path.name)
        reference = config["calibration_reference"]
        if reference.get("development_split") != expected_split:
            raise ValueError("Calibration split differs from interaction bundle: " + path.name)
        if reference.get("expected_manifests") != expected_manifests:
            raise ValueError("Calibration source/data/checkpoint manifests differ: " + path.name)
        if any(config.get(key) != value for key, value in expected_manifests.items()):
            raise ValueError("Interaction runtime/assets differ from calibration: " + path.name)
        condition, beta, observed_rho, weight = _screen_condition(config)
        if condition != expected_key or observed_rho != rho:
            raise ValueError("Unexpected interaction condition: " + path.name)
        if condition == "scd_only":
            continue
        expected_weight = (alpha05 if beta_key == "0.5" else alpha2)[str(rho)]
        if not math.isclose(weight, expected_weight, rel_tol=1e-12, abs_tol=0):
            raise ValueError(
                "Calibration coefficient differs from " + path.name +
                ". Stop before the new guidance run; the old cells cannot be combined "
                "with this regenerated reference."
            )
    print("Baseline and three interaction cells match the calibration split, "
          "source/assets and calibrated coefficients.", flush=True)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-prerequisites", action="store_true",
        help="re-run only SCD-only and beta .5/2 rho .001/.01 cells under the fresh reference",
    )
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO:
        raise SystemExit("Run this script from /content/3DD-TTA.")
    calibration_command = [sys.executable, "-u", str(REPO / "eval_gsd_calibration.py"),
                           "--phase", "diagnose", "--execute", "--result-root", str(RESULT_ROOT)]
    code, calibration_directory, calibration_zip = run_streamed(calibration_command)
    if code:
        return code
    if not calibration_directory or not calibration_zip:
        raise RuntimeError("Could not read the diagnostic run directory/ZIP from launcher output.")

    calibration_config_path = _resolved(calibration_directory) / "config.json"
    if not calibration_config_path.is_file():
        raise FileNotFoundError("Completed diagnostic config is missing: " + str(calibration_config_path))
    calibration_config, report = load_calibration(calibration_config_path)
    if calibration_config.get("execution_status") != "complete":
        raise ValueError("Diagnostic run did not complete successfully.")

    if args.rebuild_prerequisites:
        archived_paths = []
        for command in build_rebuild_commands(
                calibration_config, report, str(calibration_config_path), str(RESULT_ROOT)):
            code, _, archive = run_streamed(command)
            if code:
                return code
            if not archive:
                raise RuntimeError("Could not read a rebuilt prerequisite ZIP path from runner output.")
            archived_paths.append(_resolved(archive))
        archived_paths = verify_archived_cells(calibration_config, report, archived_paths)
    else:
        archived_paths = verify_archived_cells(calibration_config, report)
    alpha = report["candidates"]["0.5"]["weights"]["0.01"]
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Fresh beta .5/rho .01 coefficient is not finite and positive.")
    print("Fresh beta .5/rho .01 alpha:", repr(alpha), flush=True)

    interaction_name = "gsd-cal-interaction-smooth-beta0p5-rho0p01-seed0-n128"
    if args.rebuild_prerequisites:
        interaction_name = "gsd-cal-interaction-{}-smooth-beta0p5-rho0p01-seed0-n128".format(
            calibration_config["run_id"])
    interaction_command = [
        sys.executable, "-u", str(REPO / "run_baseline.py"),
        "--method", SMOOTH_METHOD, "--batch_size", "32", "--seed", "0", "--severity", "5",
        "--lambdaa", ".95", "--gamma", ".01", "--eta", ".01", "--max-batches", "0",
        "--lion-eval-mode", "--result-root", str(RESULT_ROOT),
        "--run-name", interaction_name,
        "--corruptions", "gaussian", "impulse", "--gsd-stage", "development",
        "--gsd-weight", repr(alpha), "--gsd-scd-weight", "1", "--gsd-profile", "smooth",
        "--gsd-beta", ".5", "--gsd-development-count", "128", "--gsd-split-seed",
        str(calibration_config["cli_args"]["gsd_split_seed"]), "--gsd-target-rho", ".01",
        "--gsd-calibration-reference", str(calibration_config_path),
    ]
    code, _, interaction_zip = run_streamed(interaction_command)
    if code:
        return code
    if not interaction_zip:
        raise RuntimeError("Could not read the interaction ZIP path from runner output.")

    output_name = "beta_rho_interaction_summary.json"
    if args.rebuild_prerequisites:
        output_name = "beta_rho_interaction_summary_{}.json".format(calibration_config["run_id"])
    output_path = REPO / METHOD_ROOT / output_name
    analyzer_command = [
        sys.executable, str(REPO / "scripts" / "analyze_gsd_calibration.py"),
        "--calibration", str(calibration_config_path), "--interaction-screen",
        "--screen-run", *(str(path) for path in archived_paths), str(_resolved(interaction_zip)),
        "--output", str(output_path),
    ]
    code, _, _ = run_streamed(analyzer_command)
    if code:
        return code
    print("Fresh calibration ZIP:", _resolved(calibration_zip), flush=True)
    print("Interaction ZIP:", _resolved(interaction_zip), flush=True)
    print("Compact interaction summary:", output_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
