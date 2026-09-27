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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsd_calibration import BETAS, RHOS, load_calibration
from gsd_protocol import PILOT_CORRUPTIONS


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


def _read_screen(path: Path) -> tuple[dict, dict[str, dict]]:
    path = _config_path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("stage") != "development" or config.get("execution_status") != "complete":
        raise ValueError("screen run must be a fully executed development bundle: " + str(path))
    if config.get("completed_corruptions") != list(PILOT_CORRUPTIONS):
        raise ValueError("screen run must cover Gaussian and Impulse: " + str(path))
    reference = config.get("calibration_reference")
    if not reference or not reference.get("sha256"):
        raise ValueError("screen run is missing calibration provenance: " + str(path))
    split = config.get("development_split", {})
    rows_by_corruption = {}
    with (path.parent / "per_corruption.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows_by_corruption[row["corruption"]] = row
    if set(rows_by_corruption) != set(PILOT_CORRUPTIONS):
        raise ValueError("screen CSV is missing Gaussian or Impulse: " + str(path))
    for corruption, row in rows_by_corruption.items():
        if row["status"] != "partial" or int(row["n_examples"]) != len(split[corruption]["indices"]):
            raise ValueError("screen subset coverage/count is inconsistent: " + str(path))
    return config, rows_by_corruption


def screen_ranking(paths: list[Path], calibration_run_id=None, calibrated_weights=None) -> dict:
    runs = [_read_screen(path) for path in paths]
    if not runs:
        raise ValueError("provide the screen-weight and screen-beta run directories")
    first = runs[0][0]
    reference_hash = first["calibration_reference"]["sha256"]
    rho = None
    common_split = first["development_split"]
    by_candidate = {}
    for config, rows in runs:
        if config["calibration_reference"]["sha256"] != reference_hash:
            raise ValueError("screen runs use different calibration files")
        if calibration_run_id is not None and config["calibration_reference"].get("run_id") != calibration_run_id:
            raise ValueError("screen run references a different calibration run")
        if config["development_split"] != common_split:
            raise ValueError("screen runs use different development indices")
        if config["seed"] != first["seed"]:
            raise ValueError("screen runs use different seeds; rank one seed at a time")
        if config["cli_args"]["gsd_development_count"] != first["cli_args"]["gsd_development_count"]:
            raise ValueError("screen runs use different subset sizes")
        args = config["cli_args"]
        weight = float(args["gsd_weight"])
        beta = args["gsd_beta"]
        profile = args["gsd_profile"]
        declared_rho = config["calibration_reference"].get("target_rho")
        if weight == 0:
            if declared_rho is not None:
                raise ValueError("SCD-only reference unexpectedly declares rho")
            key = "scd_only"
        else:
            if declared_rho not in RHOS:
                raise ValueError("guided run is missing its predeclared rho")
            if rho is None:
                rho = declared_rho
            if rho != declared_rho:
                raise ValueError("guided runs use different rho values")
            if profile == "smooth" and beta == 2.0:
                key = "beta_2.0"
            elif profile == "smooth" and beta in (.5, 8.0):
                key = "beta_" + str(beta)
            elif profile == "hard" and beta is None:
                key = "hard"
            else:
                raise ValueError("unexpected profile/beta in screen: " + repr((profile, beta)))
            if calibrated_weights is not None:
                alpha_by_rho = calibrated_weights[key]
                expected_weight = alpha_by_rho.get(str(declared_rho)) if alpha_by_rho else None
                if expected_weight is None or not math.isclose(
                        weight, expected_weight, rel_tol=1e-12, abs_tol=0):
                    raise ValueError("screen weight differs from the calibrated coefficient for " + key)
        if key in by_candidate:
            raise ValueError("duplicate run for candidate " + key)
        by_candidate[key] = dict(run_id=config["run_id"], rows=rows)
    expected = {"scd_only", "beta_0.5", "beta_2.0", "beta_8.0", "hard"}
    if set(by_candidate) != expected:
        raise ValueError("need exactly SCD-only, beta .5/2/8 and matched hard; found " +
                         ", ".join(sorted(by_candidate)))
    ranked = []
    baseline = by_candidate["scd_only"]["rows"]
    for key, candidate in by_candidate.items():
        per_corruption = {}
        for corruption in PILOT_CORRUPTIONS:
            row = candidate["rows"][corruption]
            accuracy = float(row["accuracy"])
            if not math.isfinite(accuracy):
                raise ValueError("nonfinite screen accuracy")
            per_corruption[corruption] = dict(
                accuracy=accuracy,
                delta_vs_scd_only_pp=100 * (accuracy - float(baseline[corruption]["accuracy"])))
        ranked.append(dict(candidate=key, run_id=candidate["run_id"], per_corruption=per_corruption,
                           macro_accuracy=sum(row["accuracy"] for row in per_corruption.values()) /
                           len(PILOT_CORRUPTIONS)))
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True,
                        help="diagnostic run directory or its config.json")
    parser.add_argument("--screen-run", type=Path, nargs="+", default=None,
                        help="all five screen run directories/configs at one seed and rho")
    parser.add_argument("--rho", type=float, choices=RHOS,
                        help="optional expected screen rho, checked when --screen-run is set")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional compact summary path outside the raw run directory")
    args = parser.parse_args(argv)
    try:
        summary = calibration_summary(args.calibration)
        if args.screen_run:
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
