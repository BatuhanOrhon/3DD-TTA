"""Validate and summarize paired GSD composition arm artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gsd_composition_protocol import METHOD, validate_arm_bundle


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _discover(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    if (root / "experiment_manifest.json").is_file():
        return [root]
    paths = sorted(path.parent for path in root.rglob("experiment_manifest.json"))
    paths.extend(sorted(root.rglob("*.zip")))
    unique = []
    seen = set()
    for path in paths:
        marker = str(path.resolve())
        if marker not in seen:
            seen.add(marker)
            unique.append(path)
    return unique


def _load_arms(root: Path) -> List[dict]:
    if not root.exists():
        raise ValueError("result root does not exist: " + str(root))
    paths = _discover(root)
    if not paths:
        raise ValueError("no composition arm bundles found under " + str(root))
    arms = [validate_arm_bundle(path) for path in paths]
    identities = set()
    complete_identities = set()
    for arm in arms:
        manifest = arm["manifest"]
        if manifest.get("method") != METHOD:
            raise ValueError("foreign method in result root")
        key = (manifest.get("block_id"), manifest.get("corruption"),
               manifest.get("seed"), manifest.get("arm_id"))
        if key in identities:
            raise ValueError("duplicate arm for a paired corruption/seed block")
        identities.add(key)
        if manifest.get("status") == "complete":
            complete_key = (manifest.get("phase", "unknown"), manifest.get("corruption"),
                            manifest.get("seed"), manifest.get("arm_id"))
            if complete_key in complete_identities:
                raise ValueError("multiple complete attempts exist for the same phase/seed/corruption/arm")
            complete_identities.add(complete_key)
    return arms


def _paired_identity(manifest: dict) -> dict:
    return {
        "method": manifest.get("method"),
        "corruption": manifest.get("corruption"),
        "seed": manifest.get("seed"),
        "block_id": manifest.get("block_id"),
        "runtime_fingerprint": manifest.get("runtime_fingerprint"),
        "locked_config_fingerprint": manifest.get("locked_config_fingerprint"),
        "input_sha256": manifest.get("input_sha256"),
        "preparation_key": manifest.get("preparation_key"),
        "original_indices": manifest.get("original_indices"),
    }


def _groups(arms: Sequence[dict]) -> Dict[Tuple, List[dict]]:
    groups = {}
    for arm in arms:
        manifest = arm["manifest"]
        key = (manifest.get("block_id"), manifest.get("corruption"), manifest.get("seed"))
        groups.setdefault(key, []).append(arm)
    pairable = {}
    for key, entries in groups.items():
        required = _paired_identity(entries[0]["manifest"])
        fingerprints = ("runtime_fingerprint", "locked_config_fingerprint", "input_sha256")
        if (any(not isinstance(required[name], str) or
                not re.fullmatch(r"[0-9a-f]{64}", required[name]) for name in fingerprints) or
                not isinstance(required["block_id"], str) or not required["block_id"] or
                not isinstance(required["original_indices"], list) or not required["original_indices"]):
            continue
        identities = [_paired_identity(entry["manifest"]) for entry in entries]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError("paired arms have mixed runtime/config/input/index identities: {}".format(key))
        if len({entry["manifest"].get("arm_id") for entry in entries}) != len(entries):
            raise ValueError("paired block has duplicate arm ID")
        reference_labels = entries[0]["labels"]
        if any(not (entry["labels"] == reference_labels).all() for entry in entries[1:]):
            raise ValueError("paired arms have mismatched labels for the same original indices")
        pairable[key] = entries
    return pairable


def _accuracy(arms: Sequence[dict]) -> dict:
    grouped = {}
    for arm in arms:
        manifest = arm["manifest"]
        phase = manifest.get("phase", "unknown")
        if manifest.get("status") == "complete":
            grouped.setdefault(phase, {}).setdefault(manifest["arm_id"], {}).setdefault(manifest["corruption"], []).append(arm)
    report = {}
    for phase, phase_arms in grouped.items():
        report[phase] = {}
        for arm_id, corruptions in phase_arms.items():
            by_corruption = {}
            for corruption, entries in sorted(corruptions.items()):
                values = [entry["accuracy"] for entry in entries if entry["accuracy"] is not None]
                n = sum(entry["n_examples"] for entry in entries)
                correct = sum(entry["n_correct"] for entry in entries)
                by_corruption[corruption] = dict(
                    n_examples=n, n_correct=correct,
                    accuracy=(correct / n if n else None),
                    seed_accuracies=values,
                    per_seed=[dict(seed=entry["manifest"]["seed"],
                                   accuracy=entry["accuracy"],
                                   n_examples=entry["n_examples"],
                                   n_correct=entry["n_correct"])
                              for entry in entries],
                )
            macro_values = [item["accuracy"] for item in by_corruption.values()
                            if item["accuracy"] is not None]
            result = dict(by_corruption=by_corruption,
                          macro_accuracy=(statistics.mean(macro_values) if macro_values else None),
                          macro15=None, macro14_excluding_background=None,
                          background_accuracy=by_corruption.get("background", {}).get("accuracy"))
            if len(by_corruption) == 15:
                result["macro15"] = result["macro_accuracy"]
                without_background = [item["accuracy"] for name, item in by_corruption.items()
                                      if name != "background" and item["accuracy"] is not None]
                result["macro14_excluding_background"] = (statistics.mean(without_background)
                                                           if without_background else None)
            report[phase][arm_id] = result
    return report


def _cluster_bootstrap_delta(clusters: dict, *, seed: int = 20261004,
                             iterations: int = 2000) -> dict:
    """Bootstrap original sample IDs, averaging repeated seeds within each ID."""
    object_deltas = [statistics.mean(values) for values in clusters.values() if values]
    if not object_deltas:
        return dict(mean_delta_pp=None, ci95=None, resamples=0)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        resampled = [object_deltas[rng.randrange(len(object_deltas))] for unused in object_deltas]
        samples.append(100.0 * sum(resampled) / len(resampled))
    samples.sort()
    return dict(
        mean_delta_pp=100.0 * statistics.mean(object_deltas),
        ci95=[samples[int(.025 * (iterations - 1))], samples[int(.975 * (iterations - 1))]],
        resamples=iterations,
        n_object_ids=len(object_deltas),
        method="paired per-corruption bootstrap clustered by original index across seeds; no cross-corruption pooling",
    )


def _transitions(groups: dict, arms: Sequence[dict]) -> Tuple[dict, dict]:
    output, uncertainty = {}, {}
    pairs = set()
    for entries in groups.values():
        ids = sorted(entry["manifest"]["arm_id"] for entry in entries)
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                pairs.add((left, right))
    for left, right in sorted(pairs):
        key_name = left + "__vs__" + right
        for block_key, entries in groups.items():
            mapping = {entry["manifest"]["arm_id"]: entry for entry in entries}
            if left not in mapping or right not in mapping:
                continue
            lrow, rrow = mapping[left], mapping[right]
            if lrow["manifest"].get("status") != "complete" or rrow["manifest"].get("status") != "complete":
                continue
            labels = lrow["labels"]
            lp = lrow["predictions"]
            rp = rrow["predictions"]
            lc, rc = (lp == labels), (rp == labels)
            both = int((lc & rc).sum())
            left_only = int((lc & ~rc).sum())
            right_only = int((~lc & rc).sum())
            both_wrong = int((~lc & ~rc).sum())
            n = int(labels.size)
            oracle = (both + left_only + right_only) / n if n else None
            corruption = lrow["manifest"]["corruption"]
            phases = sorted({entry["manifest"].get("phase", "unknown") for entry in entries})
            phase = phases[0] if len(phases) == 1 else "cross_phase:" + "__".join(phases)
            output.setdefault(phase, {}).setdefault(key_name, {}).setdefault(corruption, []).append(dict(
                seed=lrow["manifest"]["seed"], n_examples=n,
                both_correct=both, left_only_correct=left_only,
                right_only_correct=right_only, both_wrong=both_wrong,
                oracle_accuracy=oracle,
                oracle_gap_from_better=(oracle - max(lrow["accuracy"], rrow["accuracy"])) if n else None,
                prediction_disagreement=int((lp != rp).sum()),
            ))
            uncertainty.setdefault(phase, {}).setdefault(key_name, {}).setdefault(corruption, []).append(
                (lc.tolist(), rc.tolist(), lrow["indices"].tolist()))
    # Convert transitions across seeds to count totals; bootstrap stays split by corruption.
    for phase, phase_pairs in output.items():
        for pair, corruptions in phase_pairs.items():
            for corruption, rows in list(corruptions.items()):
                output[phase][pair][corruption] = {
                    key: sum(row[key] for row in rows)
                    for key in ("n_examples", "both_correct", "left_only_correct",
                                "right_only_correct", "both_wrong", "prediction_disagreement")
                }
                n = output[phase][pair][corruption]["n_examples"]
                output[phase][pair][corruption].update(
                    oracle_accuracy=((output[phase][pair][corruption]["both_correct"] +
                                      output[phase][pair][corruption]["left_only_correct"] +
                                      output[phase][pair][corruption]["right_only_correct"]) / n if n else None),
                    seed_rows=rows,
                )
                clusters = {}
                for left_correct, right_correct, original_indices in uncertainty[phase][pair][corruption]:
                    for index, left_ok, right_ok in zip(original_indices, left_correct, right_correct):
                        clusters.setdefault(int(index), []).append(int(right_ok) - int(left_ok))
                seed_deltas = [row["right_only_correct"] - row["left_only_correct"] for row in rows]
                uncertainty[phase][pair][corruption] = dict(
                    seed_delta_pp=[100.0 * delta / row["n_examples"] if row["n_examples"] else None
                                   for delta, row in zip(seed_deltas, rows)],
                    per_seed_directions_positive=all(delta > 0 for delta in seed_deltas),
                    paired_accuracy_delta=_cluster_bootstrap_delta(clusters),
                    cross_corruption_object_bootstrap="omitted; correspondence unverified",
                )
    return output, uncertainty


def _geometry(arms: Sequence[dict]) -> dict:
    numeric = ("style_gradient_cosine", "local_gradient_cosine",
               "local_guidance_ddim_norm_ratio", "style_update_norm",
               "local_update_norm", "style_routed_direction_norm",
               "local_routed_direction_norm")
    output = {}
    for arm in arms:
        arm_id = arm["manifest"]["arm_id"]
        values = {name: [] for name in numeric}
        conflict_n = valid_cosine_n = skipped_n = 0
        for row in arm["diagnostics"]:
            for name in numeric:
                value = row.get(name)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values[name].append(float(value))
            cosine = row.get("style_gradient_cosine")
            if isinstance(cosine, (int, float)) and math.isfinite(float(cosine)):
                valid_cosine_n += 1
                conflict_n += int(cosine < 0)
            skipped_n += int(bool(row.get("style_projection_skipped")))
        phase = arm["manifest"].get("phase", "unknown")
        output.setdefault(phase, {})[arm_id] = dict(
            metrics={name: dict(count=len(seq), mean=(statistics.mean(seq) if seq else None),
                                median=(statistics.median(seq) if seq else None),
                                p90=(sorted(seq)[int(.9 * (len(seq) - 1))] if seq else None))
                     for name, seq in values.items()},
            style_conflict_fraction=(conflict_n / valid_cosine_n if valid_cosine_n else None),
            valid_style_cosines=valid_cosine_n,
            style_projection_skipped=skipped_n,
        )
    return output


def _runtime(arms: Sequence[dict]) -> dict:
    by_arm = {}
    for arm in arms:
        m = arm["manifest"]
        phase = m.get("phase", "unknown")
        row = by_arm.setdefault(phase, {}).setdefault(
            m["arm_id"], {"preparation_seconds": [], "graph_seconds": [],
                          "sampling_seconds": [], "classification_seconds": [],
                          "total_seconds": [], "peak_gpu_memory_mb": []})
        for key, field in (("preparation_seconds", "preparation_seconds"),
                           ("graph_seconds", "graph_seconds"),
                           ("sampling_seconds", "sampling_seconds"),
                           ("classification_seconds", "classification_seconds"),
                           ("total_runtime_seconds", "total_seconds"),
                           ("peak_gpu_memory_mb", "peak_gpu_memory_mb")):
            value = m.get(field)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                row[key].append(float(value))
    return {phase: {arm: {key: dict(count=len(values), mean=(statistics.mean(values) if values else None))
                          for key, values in metrics.items()}
                    for arm, metrics in phase_arms.items()}
            for phase, phase_arms in by_arm.items()}


def analyze_root(result_root: Path) -> dict:
    arms = _load_arms(Path(result_root))
    groups = _groups(arms)
    transitions, uncertainty = _transitions(groups, arms)
    statuses = [entry["manifest"].get("status") for entry in arms]
    report = dict(
        schema_version=1,
        method=METHOD,
        validation=dict(complete_arms=statuses.count("complete"),
                        partial_arms=statuses.count("partial"),
                        failed_arms=statuses.count("failed"),
                        total_arms=len(arms), paired_blocks=len(groups),
                        complete_arms_without_pair_identity=sum(
                            1 for arm in arms if arm["manifest"].get("status") == "complete" and
                            (not arm["manifest"].get("input_sha256") or
                             not arm["manifest"].get("runtime_fingerprint") or
                             not arm["manifest"].get("locked_config_fingerprint") or
                             not arm["manifest"].get("block_id") or
                             not arm["manifest"].get("original_indices"))),
                        all_arm_bundles_valid=True),
        accuracy=_accuracy(arms),
        paired_transitions=transitions,
        uncertainty=uncertainty,
        complementarity={key: value for key, value in transitions.items()},
        gradient_geometry=_geometry(arms),
        runtime=_runtime(arms),
        object_correspondence="cross-corruption object IDs unverified; no pooled object bootstrap",
        accuracy_semantics="Equal-weight corruption macro accuracy; per-corruption Background retained separately. Micro counts are also recorded.",
        execution_timing_semantics="Shared preparation/graph work and per-arm sampling/classification timings are separate where available; suite timing is not standalone deployment timing.",
    )
    return _safe(report)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze_root(args.result_root)
    output = args.output
    if output.suffix.lower() == ".json":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        report_path = output
    else:
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "analysis.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote validated composition analysis:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
