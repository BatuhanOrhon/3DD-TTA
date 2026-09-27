"""Label-free diagnostic metrics and fixed coefficient calibration; no model imports."""
from __future__ import annotations

import math
import json
from pathlib import Path
import random
import statistics

BETAS = (.5, 2., 8.)
RHOS = (1e-4, 1e-3, 1e-2)
DENOMINATOR_EPS = 1e-12
CANDIDATES = (("hard", None),) + tuple(("smooth", beta) for beta in BETAS)


def development_indices(total: int, count: int, seed: int) -> list[int]:
    if not 0 < count <= total:
        raise ValueError("development count must be positive and <= dataset length")
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def _ratio(numerator: float, denominator: float):
    return numerator / denominator if denominator > DENOMINATOR_EPS else None


def _norms(value):
    return value.detach().flatten(1).double().norm(dim=1).cpu().tolist()


def gradient_rows(local_scd, style_scd, local_spectral, style_spectral) -> list[dict]:
    rows = [{} for _ in range(local_scd.shape[0])]
    for name, scd, spectral in (("local", local_scd, local_spectral),
                                ("style", style_scd, style_spectral)):
        scd_norms, spectral_norms = _norms(scd), _norms(spectral)
        dots = (scd.detach().flatten(1).double() * spectral.detach().flatten(1).double()).sum(1).cpu().tolist()
        for row, a, b, dot in zip(rows, scd_norms, spectral_norms, dots):
            cosine = dot / (a * b) if min(a, b) > DENOMINATOR_EPS else None
            row.update({name + "_scd_norm": a, name + "_spectral_norm": b,
                        name + "_ratio": _ratio(b, a),
                        name + "_cosine": max(-1., min(1., cosine)) if cosine is not None else None,
                        name + "_scd_near_zero": a <= DENOMINATOR_EPS,
                        name + "_spectral_near_zero": b <= DENOMINATOR_EPS})
    return rows


def state_rows(local, style, local_scd, style_scd, prev_local, gamma, eta,
               local_spectral=None, style_spectral=None) -> list[dict]:
    rows = [{} for _ in range(local.shape[0])]
    for name, state, grad, spectral, rate in (("local", local, local_scd, local_spectral, gamma),
                                             ("style", style, style_scd, style_spectral, eta)):
        size = math.sqrt(state[0].numel())
        spec_norms = _norms(spectral) if spectral is not None else [0.] * len(rows)
        total_norms = _norms(grad + spectral) if spectral is not None else _norms(grad)
        for row, state_norm, grad_norm, spec_norm, total_norm in zip(
                rows, _norms(state), _norms(grad), spec_norms, total_norms):
            update_norm = abs(rate) * grad_norm
            row.update({name + "_state_norm": state_norm, name + "_state_rms": state_norm / size,
                        name + "_scd_grad_norm": grad_norm,
                        name + "_scd_update_rms": update_norm / size,
                        name + "_scd_update_norm": update_norm,
                        name + "_scd_update_state_ratio": _ratio(update_norm, state_norm),
                        name + "_spectral_update_rms": abs(rate) * spec_norm / size,
                        name + "_spectral_update_state_ratio": _ratio(abs(rate) * spec_norm, state_norm),
                        name + "_total_update_norm": abs(rate) * total_norm,
                        name + "_total_update_rms": abs(rate) * total_norm / size,
                        name + "_total_update_state_ratio": _ratio(abs(rate) * total_norm, state_norm),
                        name + "_state_near_zero": state_norm <= DENOMINATOR_EPS})
    for row, ddim in zip(rows, _norms(prev_local - local)):
        row.update(local_ddim_displacement_norm=ddim,
                   local_scd_ddim_ratio=_ratio(row["local_scd_update_norm"], ddim),
                   local_total_ddim_ratio=_ratio(row["local_total_update_norm"], ddim),
                   local_ddim_near_zero=ddim <= DENOMINATOR_EPS)
    return rows


def scalar_summary(values) -> dict:
    values = list(values)
    valid = sorted(float(v) for v in values if v is not None)
    if any(not math.isfinite(v) for v in valid):
        raise ValueError("calibration contains nonfinite values")
    return dict(count=len(valid), missing=len(values) - len(valid),
                median=statistics.median(valid) if valid else None,
                p90=valid[max(0, math.ceil(.9 * len(valid)) - 1)] if valid else None,
                min=valid[0] if valid else None, max=valid[-1] if valid else None)


def coefficient_summary(rows: list[dict]) -> dict:
    local = scalar_summary(row["local_ratio"] for row in rows)
    style = scalar_summary(row["style_ratio"] for row in rows)
    ratio = local["median"]
    usable = bool(local["count"] and not local["missing"] and ratio > DENOMINATOR_EPS)
    return dict(local_ratio=local, style_ratio=style,
                local_cosine=scalar_summary(row.get("local_cosine") for row in rows),
                style_cosine=scalar_summary(row.get("style_cosine") for row in rows),
                weights={str(rho): rho / ratio for rho in RHOS} if usable else None,
                rule="fixed alpha=rho/median(sample,probe local spectral/SCD norm ratio)",
                unavailable_reason=None if usable else "missing SCD denominator or zero/near-zero median; inspect raw rows")


def summarize_calibration(config: dict) -> dict:
    """Reject incomplete evidence before recommending any fixed coefficient."""
    from gsd_protocol import PILOT_CORRUPTIONS, SMOOTH_METHOD
    if (config.get("execution_status") != "complete" or config.get("stage") != "calibrate" or
            config.get("method") != SMOOTH_METHOD or config.get("seed") != 0 or
            config.get("completed_corruptions") != list(PILOT_CORRUPTIONS)):
        raise ValueError("need a completed seed-0 Gaussian+Impulse calibration bundle")
    cli = config["cli_args"]
    if cli["gsd_weight"] != 0 or cli["gsd_scd_weight"] != 1:
        raise ValueError("calibration must follow the SCD-only reference")
    candidates = {str(beta) if beta is not None else "hard": [] for _, beta in CANDIDATES}
    by_corruption, state_summary, state_by_step = {}, {}, {}
    for corruption in PILOT_CORRUPTIONS:
        split = config["development_split"][corruption]
        indices = split["indices"]
        if indices != development_indices(split["total_examples"], cli["gsd_development_count"], cli["gsd_split_seed"]):
            raise ValueError("calibration split indices disagree with its declared seed/count")
        data = config["gsd_sample_diagnostics"][corruption]
        expected = {(index, step) for index in indices for step in (0, 2, 4)}
        by_corruption[corruption] = {}
        for profile, beta in CANDIDATES:
            key = str(beta) if beta is not None else "hard"
            rows = [row for row in data["probe"] if (row["profile"], row["beta"]) == (profile, beta)]
            if len(rows) != len(expected) or {(row["sample_index"], row["step_index"]) for row in rows} != expected:
                raise ValueError("missing or duplicate calibration sample/probe records")
            candidates[key].extend(rows)
            by_corruption[corruption][key] = coefficient_summary(rows)
            by_corruption[corruption][key]["by_probe_step"] = {
                str(step): {metric: scalar_summary(row.get(metric) for row in rows if row["step_index"] == step)
                            for metric in ("local_ratio", "style_ratio", "local_cosine", "style_cosine")}
                for step in (0, 2, 4)}
        expected_states = {(index, step) for index in indices for step in range(5)}
        if len(data["state"]) != len(expected_states) or {
                (row["sample_index"], row["step_index"]) for row in data["state"]} != expected_states:
            raise ValueError("missing or duplicate SCD state records")
        state_summary[corruption] = {
            key: scalar_summary(row[key] for row in data["state"])
            for key in data["state"][0] if key not in ("sample_index", "step_index", "timestep")}
        state_by_step[corruption] = {
            str(step): {key: scalar_summary(row[key] for row in data["state"] if row["step_index"] == step)
                        for key in state_summary[corruption]}
            for step in range(5)}
    return dict(run_id=config["run_id"], candidates={key: coefficient_summary(rows) for key, rows in candidates.items()},
                by_corruption=by_corruption, state_summary=state_summary, state_by_step=state_by_step,
                selection="no labels/accuracy used; global pooled fixed coefficients",
                caution="small development subset; no stability or accuracy conclusion from ratios alone")


def load_calibration(path) -> tuple[dict, dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    return config, summarize_calibration(config)


def calibration_provenance(args) -> dict:
    """Bind a development run to the measured coefficient and input identities."""
    import hashlib
    path = Path(args.gsd_calibration_reference).resolve()
    if path.is_dir():
        path = path / "config.json"
    config, report = load_calibration(path)
    for key in ("gsd_split_seed", "gsd_k", "gsd_delta", "gsd_graph_gamma", "gsd_modes",
                "gsd_scd_weight", "lambdaa", "gamma", "eta", "batch_size"):
        if config["cli_args"][key] != getattr(args, key):
            raise ValueError("calibration setting mismatch: " + key)
    if args.gsd_development_count < config["cli_args"]["gsd_development_count"]:
        raise ValueError("screening pool must contain the diagnostic pool")
    if args.gsd_weight:
        key = "hard" if args.gsd_profile == "hard" else str(float(args.gsd_beta))
        weights = report["candidates"][key]["weights"]
        if weights is None or args.gsd_target_rho is None or not math.isclose(
                args.gsd_weight, weights[str(args.gsd_target_rho)], rel_tol=1e-12, abs_tol=0):
            raise ValueError("guidance weight does not match the declared calibrated rho")
    elif args.gsd_target_rho is not None:
        raise ValueError("SCD-only comparator must not declare a spectral target rho")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                run_id=config["run_id"], target_rho=args.gsd_target_rho,
                development_split=config["development_split"],
                expected_manifests={key: config[key] for key in
                                    ("asset_manifest", "dataset_hash_manifest", "runtime_source_manifest")})


def verify_calibration_inputs(config: dict) -> None:
    reference = config.get("calibration_reference")
    if reference is None:
        return
    for kind, manifest in reference["expected_manifests"].items():
        for name, identity in manifest.items():
            if not identity.get("sha256") or config.get(kind, {}).get(name, {}).get("sha256") != identity["sha256"]:
                raise ValueError("calibration input/source hash mismatch: " + kind + "/" + name)
