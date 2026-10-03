"""GSD-only CLI, artifact and diagnostics contracts; no model imports."""
from __future__ import annotations

import math

METHOD = "gsd_latent_spectral_v1"
SMOOTH_METHOD = "gsd_latent_spectral_smooth_v2"
METHODS = (METHOD, SMOOTH_METHOD)
METHOD_NAME = "GSD-inspired latent spectral guidance"
SMOOTH_METHOD_NAME = "GSD-inspired smooth latent spectral guidance"
PILOT_CORRUPTIONS = ("gaussian", "impulse")
DEFAULTS = dict(gsd_weight=1.0, gsd_k=10, gsd_delta=0.1,
                gsd_graph_gamma=0.6, gsd_modes=100, gsd_stage="pilot",
                gsd_scd_weight=1.0)
DEVELOPMENT_OPTIONS = ("gsd_development_count", "gsd_split_seed",
                       "gsd_calibration_reference", "gsd_target_rho")


def is_gsd_method(method: str) -> bool:
    return method in METHODS


def add_arguments(parser) -> None:
    parser.add_argument("--gsd-scd-weight", type=float, default=None)
    for name, cast in (("weight", float), ("k", int), ("delta", float),
                       ("graph-gamma", float), ("modes", int)):
        parser.add_argument("--gsd-" + name, type=cast, default=None)
    parser.add_argument("--gsd-stage", choices=("smoke", "pilot", "benchmark", "benchmark_no_background",
                                                  "ablation_no_background", "background_completion",
                                                  "calibrate", "development",
                                                  "full_dataset_development", "full_dataset_ablation"), default=None)
    parser.add_argument("--gsd-profile", choices=("hard", "smooth"), default=None)
    parser.add_argument("--gsd-beta", type=float, default=None)
    parser.add_argument("--gsd-development-count", type=int, default=None)
    parser.add_argument("--gsd-split-seed", type=int, default=None)
    parser.add_argument("--gsd-calibration-reference", default=None)
    parser.add_argument("--gsd-target-rho", type=float, default=None)


def validate_arguments(args, parser, all_corruptions) -> None:
    if args.method not in METHODS:
        if (any(getattr(args, key) is not None for key in DEFAULTS) or
                args.gsd_profile is not None or args.gsd_beta is not None or
                any(getattr(args, key) is not None for key in DEVELOPMENT_OPTIONS)):
            parser.error("GSD options require --method " + " or ".join(METHODS))
        # Keep existing methods' serialized CLI dictionaries unchanged.
        for key in DEFAULTS:
            delattr(args, key)
        delattr(args, "gsd_profile")
        delattr(args, "gsd_beta")
        for key in DEVELOPMENT_OPTIONS:
            delattr(args, key)
        return
    is_smooth_method = args.method == SMOOTH_METHOD
    if is_smooth_method and args.gsd_weight is None:
        parser.error("GSD smooth v2 requires an explicit --gsd-weight.")
    if not is_smooth_method and (args.gsd_profile is not None or args.gsd_beta is not None):
        parser.error("--gsd-profile and --gsd-beta require --method " + SMOOTH_METHOD)
    for key, value in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if is_smooth_method:
        if args.gsd_profile is None:
            parser.error("GSD smooth v2 requires an explicit --gsd-profile.")
        if args.gsd_profile == "smooth":
            if args.gsd_beta is None or not math.isfinite(args.gsd_beta) or args.gsd_beta <= 0:
                parser.error("GSD smooth profile requires a finite positive --gsd-beta.")
        elif args.gsd_beta is not None:
            parser.error("--gsd-beta is only valid with --gsd-profile smooth.")
    else:
        delattr(args, "gsd_profile")
        delattr(args, "gsd_beta")
    if args.dataset_name != "modelnet-c" or args.severity != 5:
        parser.error("GSD requires ModelNet40-C severity 5.")
    if args.batch_size != 32 or args.seed not in (0, 1, 2):
        parser.error("GSD requires batch 32 and seed 0/1/2.")
    if args.lion_ema_mode or (args.gamma, args.eta, args.lambdaa) != (.01, .01, .95):
        parser.error("GSD requires raw LION, EMA off, gamma=eta=.01, lambdaa=.95.")
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
    reference_stages = ("development", "full_dataset_development", "full_dataset_ablation")
    if args.gsd_calibration_reference is not None and args.gsd_stage not in reference_stages:
        parser.error("Calibration reference requires a calibrated development stage.")
    if args.gsd_stage in reference_stages and args.gsd_calibration_reference is None:
        parser.error("Development stage requires --gsd-calibration-reference.")
    if args.gsd_target_rho is not None and (args.gsd_calibration_reference is None or
                                          args.gsd_target_rho not in (1e-4, 1e-3, 1e-2)):
        parser.error("Target rho requires calibration reference and value 0.0001/0.001/0.01.")
    if args.gsd_stage in ("full_dataset_development", "full_dataset_ablation"):
        if (not is_smooth_method or args.max_batches != 0 or
                args.corruptions != list(all_corruptions)):
            parser.error("Full-dataset development requires smooth-v2, canonical all-15 and max-batches 0.")
        if args.gsd_stage == "full_dataset_ablation":
            if args.gsd_scd_weight != 0:
                parser.error("Guidance ablation requires SCD weight zero; reuse existing SCD runs.")
            condition = (args.gsd_weight, args.gsd_profile, args.gsd_beta, args.gsd_target_rho)
            if condition not in ((0., "hard", None, None),
                                  (8.140161356429882, "smooth", .5, .001)):
                parser.error("Guidance ablation is locked to unguided or beta .5/rho .001 smooth-only.")
            if (args.gsd_k, args.gsd_delta, args.gsd_graph_gamma, args.gsd_modes) != (10, .1, .6, 100):
                parser.error("Guidance ablation preserves the reference graph settings.")
        elif args.gsd_scd_weight != 1:
            parser.error("Full-dataset development preserves SCD weight 1.")
        if args.gsd_development_count is not None or args.gsd_split_seed is not None:
            parser.error("Full-dataset development does not accept subset options.")
    elif args.gsd_stage == "background_completion":
        if (is_smooth_method or args.max_batches != 0 or args.corruptions != ["background"]):
            parser.error("Background completion requires GSD v1, the full Background file, and max-batches 0.")
        if (args.gsd_weight, args.gsd_scd_weight, args.gsd_k, args.gsd_delta,
                args.gsd_graph_gamma, args.gsd_modes) != (1., 0., 10, .1, .6, 100):
            parser.error("Background completion is locked to v1 spectral-only M100 settings.")
    elif args.gsd_stage in ("calibrate", "development"):
        if not is_smooth_method or args.max_batches != 0 or args.corruptions != list(PILOT_CORRUPTIONS):
            parser.error("GSD calibration/development requires smooth-v2, Gaussian+Impulse and max-batches 0.")
        if args.gsd_scd_weight != 1:
            parser.error("Calibration/development preserves SCD weight 1.")
        if args.gsd_stage == "calibrate" and (args.gsd_weight != 0 or args.seed != 0 or args.gsd_profile != "hard"):
            parser.error("Calibration requires spectral weight 0, seed 0 and hard placeholder profile.")
        if args.gsd_development_count is None:
            args.gsd_development_count = 64 if args.gsd_stage == "calibrate" else 128
        if args.gsd_split_seed is None:
            args.gsd_split_seed = 20260927
        if not 32 <= args.gsd_development_count <= 512 or args.gsd_development_count % 32:
            parser.error("Development count must be a multiple of 32 in [32,512].")
    elif args.gsd_development_count is not None or args.gsd_split_seed is not None:
        parser.error("Development subset options require calibrate/development stage.")
    elif args.gsd_stage == "background_completion":
        pass  # The single-corruption full-file contract is checked above.
    elif args.gsd_stage == "smoke":
        if args.max_batches not in (1, 2) or args.corruptions not in (["gaussian"], ["background"]):
            parser.error("GSD smoke requires 1/2 batches of Gaussian or Background.")
    else:
        expected = PILOT_CORRUPTIONS if args.gsd_stage == "pilot" else (
            tuple(name for name in all_corruptions if name != "background")
            if args.gsd_stage in ("benchmark_no_background", "ablation_no_background") else all_corruptions)
        if args.max_batches != 0 or args.corruptions != list(expected):
            parser.error("GSD pilot/benchmark requires complete files in its canonical scope.")
    if is_smooth_method and args.gsd_stage not in ("smoke", "pilot", "calibrate", "development",
                                                    "full_dataset_development", "full_dataset_ablation"):
        parser.error("GSD smooth v2 is restricted to smoke/pilot until promotion review.")
    if not is_smooth_method and args.gsd_stage in ("benchmark", "benchmark_no_background") and (
            args.gsd_weight not in (0.0, 1.0) or args.gsd_scd_weight != 1.0 or args.gsd_k != 10 or
            args.gsd_delta != .1 or args.gsd_graph_gamma != .6 or args.gsd_modes != 100):
        parser.error("GSD v1 benchmark is locked to weight 0/1, k=10, delta=.1, graph gamma=.6, modes=100.")
    args.lion_eval_mode = True


def spectral_contract(args) -> dict:
    smooth_version = args.method == SMOOTH_METHOD
    method_name = SMOOTH_METHOD_NAME if smooth_version else METHOD_NAME
    contract = dict(
        name=method_name, version=2 if smooth_version else 1, weight=args.gsd_weight,
        scd_weight=args.gsd_scd_weight,
        graph_domain="encoded local latent XYZ", signal="predicted clean local latent XYZ",
        k=args.gsd_k, delta=args.gsd_delta, graph_gamma=args.gsd_graph_gamma,
        requested_modes=(args.gsd_modes if not smooth_version or args.gsd_profile == "hard" else None),
        graph="static non-self kNN; max-symmetric RBF",
        distance="squared Euclidean in exp(-d2/(2*delta^2))",
        threshold="graph_gamma * directed_adjacency.sum() / (N*k)",
        outlier_policy="both endpoints must pass directed degree threshold; exclude final isolates",
        laplacian=("combinatorial active induced subgraph divided by mean active degree; no diagonal penalty"
                   if smooth_version else "combinatorial, active induced subgraph; no diagonal penalty"),
        band_boundary=("fixed anchor; rtol=1e-5, atol=max(1e-7, 2*roundoff_floor)"
                       if not smooth_version or args.gsd_profile == "hard"
                       else "not used by smooth weighting; retained only for graph diagnostics"),
        roundoff_floor="eps(dtype) * active_vertices * Laplacian infinity norm",
        zero_mode_tolerance="max(1e-7, roundoff_floor); numerical, not exact connectivity count",
        reduction=("sum_samples_fixed_original_point_count_xyz" if smooth_version
                   else "sum_samples_mean_modes_xyz"),
        guidance=(f"{args.gsd_scd_weight} * SCD + {args.gsd_weight} * spectral; ordinary sum"),
        target=("detached encoded XYZ; static shared-basis spectral filter operator"
                if smooth_version else "detached encoded XYZ; static shared orthonormal basis"),
        gradients="through clean prediction and frozen denoiser to noisy local state AND conditioning",
        zero_weight_behavior="direct original baseline trajectory; bypass graph and spectral computation",
        random_draws="no extra graph or diagnostics RNG draws; cross-run pairing not assumed",
        tuning="fixed initial configuration; exploratory pilot, no established optimal weight",
    )
    if smooth_version:
        contract.update(
            profile=args.gsd_profile,
            beta=args.gsd_beta,
            hard_requested_modes=args.gsd_modes if args.gsd_profile == "hard" else None,
            operator="full active graph spectral filter; dense fixed operator")
    if args.gsd_stage == "calibrate":
        contract.update(name="SCD-only reference with read-only spectral probes",
                        profile="shared hard/smooth probes", beta=None,
                        zero_weight_behavior="instrumented SCD-only trajectory; no spectral gradient applied",
                        operator="one retained eigensystem per graph; fixed 3*N probe losses",
                        tuning="label-free pooled development calibration; candidates hard and beta .5/2/8")
    elif args.gsd_stage == "development":
        contract.update(tuning="exploratory fixed shuffled subset; not a benchmark",
                        zero_weight_behavior="instrumented SCD-only trajectory; graph bypass")
    elif args.gsd_stage == "full_dataset_development":
        contract.update(tuning="fixed calibration-derived candidate; full-test-set development screen",
                        zero_weight_behavior="calibrated-reference-bound SCD-only comparator; graph bypass")
    elif args.gsd_stage == "full_dataset_ablation":
        contract.update(
            name="Unguided LION diffusion" if args.gsd_weight == 0 else "Smooth spectral-only latent guidance",
            tuning="fixed existing coefficient; SCD disabled; no new calibration or parameter search",
            zero_weight_behavior="explicit unguided DDIM; no graph, Chamfer loss, or guidance gradients",
            target_rho_interpretation="ratio on archived SCD reference only; no active SCD denominator")
        if args.gsd_weight == 0:
            contract.update(operator="none", gradients="none; inference-only diffusion", graph="not constructed")
    return contract


def notes_for_run(args) -> str:
    if args.gsd_stage == "background_completion":
        return ("# GSD v1 spectral-only Background completion\n\n"
                "[Code] Completes the missing Background severity-5 file for the existing seed-0, "
                "14-corruption spectral-only M100 run. All 2,468 examples are processed; "
                "spectral weight=1, SCD weight=0, k=10, delta=.1, graph gamma=.6, raw/eval LION, "
                "EMA off, lambda=.95 and gamma=eta=.01. This single-corruption artifact is not "
                "a standalone all-15 run. The companion analyzer combines it with the archived "
                "14-corruption ZIP and labels the reconstructed mean as a cross-run composite.\n")
    if args.gsd_stage == "full_dataset_ablation":
        return ("# Fixed guidance ablation\n\n"
                "[Code] All15 severity5, full files, raw/eval LION, no EMA, batch32, lambda .95, "
                "gamma=eta=.01, original style, 5/35 reverse steps. SCD weight is zero. "
                "Spectral weight zero selects true unguided diffusion with constant conditioning; "
                "positive weight applies beta .5 smooth-only using the archived alpha unchanged. "
                "Rho describes the archived SCD reference, not an active SCD ratio.\n"
                "[Code] Per-example labels/predictions and file indices are in config.json. "
                "Labels are used only after prediction. Seven-file ZIP; seeded runs, no common-draw claim. "
                "Existing SCD results are reused, not rerun. Provide each complete ZIP and derived summary.\n")
    if args.gsd_stage == "full_dataset_development":
        return ("# GSD full-test-set development screen\n\n"
                "[Code] All examples from the canonical ModelNet40-C severity-5 test files are evaluated "
                "for all 15 corruptions. Inputs and ordering match the existing all-corruptions test files. "
                "Adaptation receives point clouds only; labels are read for post-prediction accuracy metrics.\n"
                "[Code] Calibration-derived coefficients are fixed before this run. Comparing and selecting "
                "candidates on these test-set scores is descriptive development evidence, not independent "
                "confirmation. Do not treat the selected score as an unbiased final estimate.\n"
                "[Code] Aggregate diagnostics are bounded per corruption; no sample-by-step diagnostic rows. "
                "The raw seven-file bundle is the evidence unit. Record Colab runtime details and anomalies.\n")
    if args.gsd_stage in ("calibrate", "development"):
        return ("# GSD development experiment\n\n"
                "[Code] Stage=" + args.gsd_stage + ". Fixed shuffled index subset; raw/eval LION, "
                "EMA off, original-style decoder, SCD weight 1, lambda .95, gamma=eta=.01. "
                "Calibration probes hard and beta .5/2/8 at shared SCD-only states; no candidate update applied. "
                "Development applies its configured fixed guidance weight.\n"
                "[Code] config.json contains development_split and per-sample gsd_sample_diagnostics. "
                "CSV coverage stays partial. Pooled calibration creates cross-example dependence.\n"
                "[Open] Index correspondence is not verified object identity. No held-out or full-benchmark claim. "
                "Separate seeded screening runs are not proven common-draw paired. "
                "Provide the complete seven-file ZIP.\n")
    name = SMOOTH_METHOD_NAME if args.method == SMOOTH_METHOD else METHOD_NAME
    method = args.method
    profile = (f" Profile={args.gsd_profile}; beta={args.gsd_beta}; fixed 3*N reduction."
               if args.method == SMOOTH_METHOD else "")
    return (
        "# " + name + "\n\n"
        "[Code] Opt-in method " + method + "; stage=" + args.gsd_stage + ". "
        "Raw LION eval, EMA off, frozen Point-MAE, original-style decoder, "
        "summed SCD with configurable SCD weight, lambda=.95, gamma=eta=.01, original 100-step DDIM / 5-35 reverse schedule."
        + profile + "\n"
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
    if kind in ("state", "probe"):
        config.setdefault("gsd_sample_diagnostics", {}).setdefault(corruption, {}).setdefault(
            kind, []).extend(event["samples"])
        return
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
    smooth_method = args.method == SMOOTH_METHOD
    targets, predictions = [], []
    offset = 0
    for data, label in baseline.tqdm(batches, desc="GSD Batches"):
        with torch_module.no_grad():
            inputs, center, maximum = tta_preprocess_points(data, baseline, args, torch_module)
        extra = {}
        if args.gsd_stage == "full_dataset_ablation":
            extra["allow_unguided"] = True
        if args.gsd_stage in ("calibrate", "development"):
            def observe(kind, event):
                metadata = {key: value for key, value in event.items() if key != "samples"}
                rows = [dict(row, **metadata, sample_index=args.gsd_sample_indices[offset + i])
                        for i, row in enumerate(event["samples"])]
                diagnostics_observer(dict(kind=kind, samples=rows))
            extra["sample_observer"] = lambda event: observe("state", event)
            if args.gsd_stage == "calibrate":
                extra["probe_observer"] = lambda event: observe("probe", event)
        points = tta_gsd_reconstruct(
            inputs, lion, steps, args.gamma, args.eta, args.lambdaa, 100,
            spectral_weight=args.gsd_weight, scd_weight=args.gsd_scd_weight,
            spectral_config=spectral_config,
            scheduler_observer=scheduler_observer, diagnostics_observer=diagnostics_observer,
            spectral_profile=args.gsd_profile if smooth_method else None,
            spectral_beta=args.gsd_beta if smooth_method else None, **extra)
        with torch_module.no_grad():
            points = tta_postprocess_points(points, center, maximum, baseline, args)
            pred = base_model.module.classification_only(points, only_unmasked=False).argmax(-1).view(-1)
        target = label.view(-1)
        batch_observer(target, pred)
        targets.append(target.cpu())
        predictions.append(pred.cpu())
        offset += len(data)
    return torch_module.cat(targets), torch_module.cat(predictions)
