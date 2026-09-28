"""Run a fresh GSD calibration, then only the registered beta/rho condition.

Run from the repository root inside the existing 3dd_tta_env Colab environment.
The script checks that archived interaction cells use the same development
indices, source/assets and calibrated coefficients before launching the new arm.
"""
from __future__ import annotations

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


def verify_archived_cells(calibration_config: dict, report: dict) -> list[Path]:
    """Allow a regenerated reference only when its calibration is equivalent."""
    expected_manifests = {key: calibration_config[key] for key in
                          ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest")}
    expected_split = calibration_config["development_split"]
    alpha05 = report["candidates"]["0.5"]["weights"]
    alpha2 = report["candidates"]["2.0"]["weights"]
    if not alpha05 or not alpha2:
        raise ValueError("Fresh calibration did not produce usable beta .5/beta 2 coefficients.")

    paths = []
    for filename, expected_key, rho, beta_key in ARCHIVED_CELLS:
        path = REPO / METHOD_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError("Required earlier interaction bundle is missing: " + str(path))
        config, _ = _read_screen(path)
        paths.append(path)
        if (config["seed"] != 0 or config["cli_args"]["gsd_development_count"] != 128 or
                config["cli_args"]["gsd_split_seed"] != calibration_config["cli_args"]["gsd_split_seed"]):
            raise ValueError("Archived interaction seed/count/split seed differs: " + filename)
        if config["development_split"]["gaussian"]["indices"] != config["development_split"]["impulse"]["indices"]:
            raise ValueError("Archived Gaussian/Impulse development indices differ: " + filename)
        reference = config["calibration_reference"]
        if reference.get("development_split") != expected_split:
            raise ValueError("Fresh calibration split differs from archived reference: " + filename)
        if reference.get("expected_manifests") != expected_manifests:
            raise ValueError("Fresh source/data/checkpoint manifests differ from archived run: " + filename)
        if any(config.get(key) != value for key, value in expected_manifests.items()):
            raise ValueError("Archived runtime/assets differ from fresh calibration: " + filename)
        condition, beta, observed_rho, weight = _screen_condition(config)
        if condition != expected_key or observed_rho != rho:
            raise ValueError("Unexpected archived interaction condition: " + filename)
        if condition == "scd_only":
            continue
        expected_weight = (alpha05 if beta_key == "0.5" else alpha2)[str(rho)]
        if not math.isclose(weight, expected_weight, rel_tol=1e-12, abs_tol=0):
            raise ValueError(
                "Fresh calibration coefficient differs from " + filename +
                ". Stop before the new guidance run; the old cells cannot be combined "
                "with this regenerated reference."
            )
    print("Archived baseline and three interaction cells match the fresh split, "
          "source/assets and calibrated coefficients.", flush=True)
    return paths


def main() -> int:
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

    archived_paths = verify_archived_cells(calibration_config, report)
    alpha = report["candidates"]["0.5"]["weights"]["0.01"]
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Fresh beta .5/rho .01 coefficient is not finite and positive.")
    print("Fresh beta .5/rho .01 alpha:", repr(alpha), flush=True)

    interaction_command = [
        sys.executable, "-u", str(REPO / "run_baseline.py"),
        "--method", SMOOTH_METHOD, "--batch_size", "32", "--seed", "0", "--severity", "5",
        "--lambdaa", ".95", "--gamma", ".01", "--eta", ".01", "--max-batches", "0",
        "--lion-eval-mode", "--result-root", str(RESULT_ROOT),
        "--run-name", "gsd-cal-interaction-smooth-beta0p5-rho0p01-seed0-n128",
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

    output_path = REPO / METHOD_ROOT / "beta_rho_interaction_summary.json"
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
