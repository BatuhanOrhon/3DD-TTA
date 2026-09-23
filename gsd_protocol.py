"""GSD-only CLI, artifact and diagnostics contracts; no model imports."""
from __future__ import annotations

import math

METHOD = "gsd_latent_spectral_v1"
METHOD_NAME = "GSD-inspired latent spectral guidance"
PILOT_CORRUPTIONS = ("gaussian", "impulse")
DEFAULTS = dict(gsd_weight=1.0, gsd_k=10, gsd_delta=0.1,
                gsd_graph_gamma=0.6, gsd_modes=100, gsd_stage="pilot",
                gsd_scd_weight=1.0)


def add_arguments(parser) -> None:
    parser.add_argument("--gsd-scd-weight", type=float, default=None)
    for name, cast in (("weight", float), ("k", int), ("delta", float),
                       ("graph-gamma", float), ("modes", int)):
        parser.add_argument("--gsd-" + name, type=cast, default=None)
    parser.add_argument("--gsd-stage", choices=("smoke", "pilot", "benchmark", "benchmark_no_background"), default=None)


def validate_arguments(args, parser, all_corruptions) -> None:
    if args.method != METHOD:
        if any(getattr(args, key) is not None for key in DEFAULTS):
            parser.error("GSD options require --method " + METHOD)
        # Keep existing methods' serialized CLI dictionaries unchanged.
        for key in DEFAULTS:
            delattr(args, key)
        return
    for key, value in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.dataset_name != "modelnet-c" or args.severity != 5:
        parser.error("GSD v1 requires ModelNet40-C severity 5.")
    if args.batch_size != 32 or args.seed not in (0, 1, 2):
        parser.error("GSD v1 requires batch 32 and seed 0/1/2.")
    if args.lion_ema_mode or (args.gamma, args.eta, args.lambdaa) != (.01, .01, .95):
        parser.error("GSD v1 requires raw LION, EMA off, gamma=eta=.01, lambdaa=.95.")
    if args.fps_diagnostics:
        parser.error("GSD v1 keeps the existing FPS path.")
    if (not math.isfinite(args.gsd_weight) or args.gsd_weight < 0 or
            not math.isfinite(args.gsd_scd_weight) or args.gsd_scd_weight < 0):
        parser.error("GSD and SCD weights must be finite and non-negative.")
    if not 1 <= args.gsd_k < 2048 or not 1 <= args.gsd_modes <= 2048:
        parser.error("GSD requires 1 <= k < 2048 and 1 <= modes <= 2048.")
    if not math.isfinite(args.gsd_delta) or args.gsd_delta <= 0:
        parser.error("GSD delta must be finite and positive.")
    if not math.isfinite(args.gsd_graph_gamma) or args.gsd_graph_gamma < 0:
        parser.error("GSD graph gamma must be finite and non-negative.")
    if args.gsd_stage == "smoke":
        if args.max_batches not in (1, 2) or args.corruptions not in (["gaussian"], ["background"]):
            parser.error("GSD smoke requires 1/2 batches of Gaussian or Background.")
    else:
        expected = PILOT_CORRUPTIONS if args.gsd_stage == "pilot" else (
            tuple(name for name in all_corruptions if name != "background")
            if args.gsd_stage == "benchmark_no_background" else all_corruptions)
        if args.max_batches != 0 or args.corruptions != list(expected):
            parser.error("GSD pilot/benchmark requires complete files in its canonical scope.")
    if args.gsd_stage in ("benchmark", "benchmark_no_background") and (
            args.gsd_weight not in (0.0, 1.0) or args.gsd_scd_weight != 1.0 or args.gsd_k != 10 or
            args.gsd_delta != .1 or args.gsd_graph_gamma != .6 or args.gsd_modes != 100):
        parser.error("GSD v1 benchmark is locked to weight 0/1, k=10, delta=.1, graph gamma=.6, modes=100.")
    args.lion_eval_mode = True


def spectral_contract(args) -> dict:
    return dict(
        name=METHOD_NAME, version=1, weight=args.gsd_weight,
        scd_weight=args.gsd_scd_weight,
        graph_domain="encoded local latent XYZ", signal="predicted clean local latent XYZ",
        k=args.gsd_k, delta=args.gsd_delta, graph_gamma=args.gsd_graph_gamma,
        requested_modes=args.gsd_modes, graph="static non-self kNN; max-symmetric RBF",
        distance="squared Euclidean in exp(-d2/(2*delta^2))",
        threshold="graph_gamma * directed_adjacency.sum() / (N*k)",
        outlier_policy="both endpoints must pass directed degree threshold; exclude final isolates",
        laplacian="combinatorial, active induced subgraph; no diagonal penalty",
        band_boundary="fixed anchor; rtol=1e-5, atol=max(1e-7, 2*roundoff_floor)",
        roundoff_floor="eps(dtype) * active_vertices * Laplacian infinity norm",
        zero_mode_tolerance="max(1e-7, roundoff_floor); numerical, not exact connectivity count",
        reduction="sum_samples_mean_modes_xyz", guidance="SCD + weight * spectral; ordinary sum",
        target="detached encoded XYZ; static shared orthonormal basis",
        gradients="through clean prediction and frozen denoiser to noisy local state AND conditioning",
        zero_weight_behavior="direct original baseline trajectory; bypass graph and spectral computation",
        random_draws="no extra graph or diagnostics RNG draws; cross-run pairing not assumed",
        tuning="fixed initial configuration; exploratory pilot, no established optimal weight",
    )


def notes_for_run(args) -> str:
    return (
        "# " + METHOD_NAME + "\n\n"
        "[Code] Opt-in method " + METHOD + "; stage=" + args.gsd_stage + ". "
        "Raw LION eval, EMA off, frozen Point-MAE, original-style decoder, "
        "summed SCD with configurable SCD weight, lambda=.95, gamma=eta=.01, original 100-step DDIM / 5-35 reverse schedule.\n"
        "[Inference] Static low-frequency fidelity may preserve instance structure. "
        "This is not a reproduction of the full GSDTTA algorithm.\n"
        "[Open] Accuracy benefit, corrupted-graph bias and real CUDA operator gradients "
        "require Colab evidence. Weight zero is the exact original trajectory dispatch. "
        "Diagnostics are online aggregates per corruption in config.json; CSVs preserve "
        "the original counts/runtime/memory schema. Smoke prefixes are not accuracy evidence.\n"
        "[Code] Seed-controlled separate runs; no common-draw causal claim. "
        "No resume; no checkpoint/data/preprocessing/FPS/scheduler changes.\n"
        "[Open] Record Colab runtime type and anomalies; provide the complete seven-file ZIP.\n"
    )


def merge_diagnostics(config: dict, corruption: str, event: dict) -> None:
    """Keep bounded scalar aggregates, including absent/empty-spectrum counters."""
    kind = event["kind"]
    rows = event["samples"] if kind == "graph" else [event]
    target = config.setdefault("gsd_diagnostics", {}).setdefault(corruption, {}).setdefault(
        kind, {"records": 0, "scalars": {}})
    for row in rows:
        target["records"] += 1
        for key, value in row.items():
            if key == "kind" or isinstance(value, (str, list, dict, tuple)):
                continue
            stats = target["scalars"].setdefault(key, dict(count=0, missing=0, sum=0.0, min=None, max=None))
            if value is None:
                stats["missing"] += 1
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Nonfinite GSD diagnostic: " + key)
            stats["count"] += 1
            stats["sum"] += number
            stats["min"] = number if stats["min"] is None else min(stats["min"], number)
            stats["max"] = number if stats["max"] is None else max(stats["max"], number)
            stats["mean"] = stats["sum"] / stats["count"]


def process_batches(batches, base_model, lion, args, baseline, torch_module, steps,
                    *, scheduler_observer, batch_observer, diagnostics_observer):
    """GSD-only adapter using the existing preprocessing and postprocessing helpers."""
    from graph_spectral import SpectralConfig
    from tta_gsd import tta_gsd_reconstruct
    from run_baseline import tta_preprocess_points, tta_postprocess_points

    spectral_config = SpectralConfig(k=args.gsd_k, delta=args.gsd_delta,
                                    graph_gamma=args.gsd_graph_gamma, modes=args.gsd_modes)
    targets, predictions = [], []
    for data, label in baseline.tqdm(batches, desc="GSD Batches"):
        with torch_module.no_grad():
            inputs, center, maximum = tta_preprocess_points(data, baseline, args, torch_module)
        points = tta_gsd_reconstruct(
            inputs, lion, steps, args.gamma, args.eta, args.lambdaa, 100,
            spectral_weight=args.gsd_weight, scd_weight=args.gsd_scd_weight,
            spectral_config=spectral_config,
            scheduler_observer=scheduler_observer, diagnostics_observer=diagnostics_observer)
        with torch_module.no_grad():
            points = tta_postprocess_points(points, center, maximum, baseline, args)
            pred = base_model.module.classification_only(points, only_unmasked=False).argmax(-1).view(-1)
        target = label.view(-1)
        batch_observer(target, pred)
        targets.append(target.cpu())
        predictions.append(pred.cpu())
    return torch_module.cat(targets), torch_module.cat(predictions)
