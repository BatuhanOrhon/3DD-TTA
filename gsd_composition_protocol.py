"""Versioned experiment, pairing, and artifact contracts for GSD composition."""
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from gsd_composition import CompositionConfig
from research_artifacts import CORRUPTIONS, PER_COLUMNS, SUMMARY_COLUMNS


METHOD = "gsd_guidance_composition_v1"
PROTOCOL_SCHEMA_VERSION = 1
SPLIT_SEED = 20261004
SPLIT_SIZE = 2468
PILOT_CORRUPTIONS = ("gaussian", "impulse", "background", "shear")
SMOKE_CORRUPTIONS = ("gaussian", "background")
EXPECTED_ARM_FILES = frozenset((
    "command.txt", "config.json", "environment.txt", "stdout.log", "summary.csv",
    "per_corruption.csv", "notes.md", "predictions.npz",
    "gradient_diagnostics.jsonl.gz", "experiment_manifest.json",
))


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    config: CompositionConfig
    question: str = ""


@dataclass(frozen=True)
class PhasePlan:
    phase: str
    arms: Tuple[ArmSpec, ...]
    corruptions: Tuple[str, ...]
    indices: Tuple[int, ...]
    seeds: Tuple[int, ...]
    scope: str


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def experiment_fingerprint(config: dict) -> str:
    return canonical_sha256(config)


def build_split_manifest() -> dict:
    indices = list(range(SPLIT_SIZE))
    random.Random(SPLIT_SEED).shuffle(indices)
    pilot = indices[:64]
    replication = indices[64:320]
    return dict(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        seed=SPLIT_SEED,
        source_size=SPLIT_SIZE,
        indices=indices,
        indices_sha256=canonical_sha256(indices),
        pilot_indices=pilot,
        pilot_indices_sha256=canonical_sha256(pilot),
        replication_indices=replication,
        replication_indices_sha256=canonical_sha256(replication),
        correspondence="index correspondence across corruptions is unverified",
    )


_SPLIT = build_split_manifest()


def derive_draw_seeds(dataset: str, corruption: str, model_seed: int,
                      ordered_batch_indices: Sequence[int]) -> dict:
    """Derive phase/arm-independent preparation and classifier RNG keys."""
    base = dict(dataset=dataset, corruption=corruption,
                model_seed=int(model_seed), ordered_batch_indices=list(ordered_batch_indices))
    preparation_key = canonical_sha256(dict(schema_version=1, purpose="prepared-inputs", **base))
    classification_key = canonical_sha256(dict(schema_version=1, purpose="classification", **base))
    return dict(
        derivation_version=1,
        preparation_key=preparation_key,
        preparation_seed=int(preparation_key[:16], 16) % (2 ** 32),
        classification_key=classification_key,
        classification_seed=int(classification_key[:16], 16) % (2 ** 32),
    )


def _config(local: str = "off", style: str = "off", *, scd: float = 1.0,
            spectral: float = 1.0) -> CompositionConfig:
    return CompositionConfig(
        local_mode=local, style_mode=style,
        local_scd_weight=scd, local_spectral_weight=spectral,
        style_scd_weight=scd, style_spectral_weight=spectral)


_ARM_LIBRARY = {
    "C_SCD": ArmSpec("C_SCD", _config("scd", "scd"), "matched SCD reference"),
    "C_SPEC": ArmSpec("C_SPEC", _config("spectral", "spectral"), "matched v1 spectral reference"),
    "C_SUM": ArmSpec("C_SUM", _config("sum", "sum"), "matched additive reference"),
    "R_S0": ArmSpec("R_S0", _config("scd", "off"), "SCD local with style guidance off"),
    "R_SG": ArmSpec("R_SG", _config("scd", "spectral"), "local SCD and style spectral"),
    "R_G0": ArmSpec("R_G0", _config("spectral", "off"), "spectral local with style off"),
    "R_GS": ArmSpec("R_GS", _config("spectral", "scd"), "local spectral and style SCD"),
    "P_SUM": ArmSpec("P_SUM", _config("scd", "sum"), "style additive control"),
    "P_PC": ArmSpec("P_PC", _config("scd", "pcgrad"), "norm-capped symmetric style projection"),
    "P_NORM": ArmSpec("P_NORM", _config("scd", "sum_norm_pcgrad"), "same-state norm-matched sum"),
    "P_SCD_PRIORITY": ArmSpec("P_SCD_PRIORITY", _config("scd", "scd_priority"), "one-way style projection"),
}


def _scale_arms() -> Tuple[ArmSpec, ...]:
    return tuple(ArmSpec("SCALE_{}".format(weight),
                         _config("sum", "sum", spectral=float(weight)),
                         "scalar hard-v1 spectral weight {}".format(weight))
                 for weight in (0, 1, 100, 1000))


def make_selection_manifest(source_phase: str, candidate_id: str, *, rationale: str,
                            relevant_comparators: Sequence[str]) -> dict:
    if candidate_id in _ARM_LIBRARY:
        candidate = _ARM_LIBRARY[candidate_id]
    elif candidate_id.startswith("SCALE_"):
        candidate = next((arm for arm in _scale_arms() if arm.arm_id == candidate_id), None)
        if candidate is None:
            raise ValueError("unknown scale candidate")
    else:
        raise ValueError("unknown candidate arm")
    selection = dict(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        selection_type="frozen_development_candidate",
        source_phase=source_phase,
        candidate_id=candidate_id,
        candidate=dict(arm_id=candidate.arm_id, **asdict(candidate.config)),
        relevant_comparators=list(relevant_comparators),
        rationale=rationale,
    )
    selection["selection_sha256"] = canonical_sha256(selection)
    validate_selection_manifest(selection)
    return selection


def validate_selection_manifest(selection) -> dict:
    if not isinstance(selection, dict):
        raise ValueError("selection manifest is required")
    required = {"schema_version", "selection_type", "source_phase", "candidate_id",
                "candidate", "relevant_comparators", "rationale", "selection_sha256"}
    if set(selection) != required or selection.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("selection manifest schema is invalid")
    if selection["selection_type"] != "frozen_development_candidate":
        raise ValueError("selection must be frozen from a development phase")
    valid_sources = {"routing", "projection", "scale", "replicate"}
    if not isinstance(selection["source_phase"], str) or selection["source_phase"] not in valid_sources:
        raise ValueError("selection source_phase must name a completed development phase")
    expected_hash = selection["selection_sha256"]
    body = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if expected_hash != canonical_sha256(body):
        raise ValueError("selection manifest fingerprint mismatch")
    candidate = selection["candidate"]
    if not isinstance(candidate, dict) or candidate.get("arm_id") != selection["candidate_id"]:
        raise ValueError("selection candidate identity is invalid")
    if not isinstance(selection["candidate_id"], str):
        raise ValueError("selection candidate ID must be a string")
    known_ids = set(_ARM_LIBRARY) | {arm.arm_id for arm in _scale_arms()}
    if selection["candidate_id"] not in known_ids:
        raise ValueError("selection names an unknown candidate")
    candidate_id = selection["candidate_id"]
    allowed_candidates = {"R_SG", "R_GS", "P_PC", "P_SCD_PRIORITY"} | {
        arm.arm_id for arm in _scale_arms()}
    if candidate_id not in allowed_candidates:
        raise ValueError("selection candidate is not a promotable experiment arm")
    expected_source = ("routing" if candidate_id in {"R_SG", "R_GS"} else
                       "projection" if candidate_id in {"P_PC", "P_SCD_PRIORITY"} else "scale")
    if selection["source_phase"] not in {expected_source, "replicate"}:
        raise ValueError("selection candidate does not belong to its source phase")
    canonical_arm = (_ARM_LIBRARY[candidate_id] if candidate_id in _ARM_LIBRARY else
                     next(arm for arm in _scale_arms() if arm.arm_id == candidate_id))
    canonical_candidate = dict(arm_id=canonical_arm.arm_id, **asdict(canonical_arm.config))
    if candidate != canonical_candidate:
        raise ValueError("selection candidate config must match the canonical named arm")
    try:
        config = CompositionConfig(**{key: candidate[key] for key in (
            "local_mode", "style_mode", "local_scd_weight", "local_spectral_weight",
            "style_scd_weight", "style_spectral_weight", "norm_floor", "schema_version")})
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection contains an invalid composition route") from error
    if (not isinstance(selection["relevant_comparators"], list) or
            not selection["relevant_comparators"] or
            any(not isinstance(value, str) for value in selection["relevant_comparators"])):
        raise ValueError("selection must name relevant matched comparators")
    if (len(set(selection["relevant_comparators"])) != len(selection["relevant_comparators"]) or
            set(selection["relevant_comparators"]) - known_ids):
        raise ValueError("selection contains unknown or duplicate comparators")
    required_comparators = {
        "R_SG": {"C_SCD", "R_S0"},
        "R_GS": {"C_SPEC", "R_G0"},
        "P_PC": {"C_SCD", "P_SUM", "P_NORM"},
        "P_SCD_PRIORITY": {"C_SCD", "P_SUM", "P_PC"},
    }
    if selection["source_phase"] == "replicate":
        required = {"C_SCD"}
    elif candidate_id.startswith("SCALE_"):
        required = {"SCALE_0", "SCALE_1"}
    else:
        required = required_comparators[candidate_id]
    if not required.issubset(set(selection["relevant_comparators"])):
        raise ValueError("selection omits a predeclared mechanism comparator")
    if not isinstance(selection["rationale"], str) or not selection["rationale"].strip():
        raise ValueError("selection rationale is required")
    return dict(selection, parsed_config=config)


def _selected_arm(selection: dict) -> ArmSpec:
    resolved = validate_selection_manifest(selection)
    values = resolved["candidate"]
    config = CompositionConfig(**{key: values[key] for key in (
        "local_mode", "style_mode", "local_scd_weight", "local_spectral_weight",
        "style_scd_weight", "style_spectral_weight", "norm_floor", "schema_version")})
    return ArmSpec(selection["candidate_id"], config, selection["rationale"])


def validate_all15_reference(selection: dict, reference: dict) -> dict:
    """Require completed replication evidence for the exact all15 candidate."""
    validated = validate_selection_manifest(selection)
    if selection["source_phase"] != "replicate":
        raise ValueError("all15 selection must come from completed replication")
    if not isinstance(reference, dict) or reference.get("phase") != "replicate" or reference.get("status") != "complete":
        raise ValueError("all15 reference must be a completed replicate manifest")
    if not isinstance(reference.get("experiment_fingerprint"), str) or not reference["experiment_fingerprint"]:
        raise ValueError("all15 reference has no experiment fingerprint")
    replicated_selection = reference.get("selection_manifest")
    if not isinstance(replicated_selection, dict):
        raise ValueError("all15 reference has no embedded replicate selection")
    replicated_selection = validate_selection_manifest(replicated_selection)
    if (replicated_selection["source_phase"] not in ("routing", "projection", "scale") or
            replicated_selection["candidate_id"] != selection["candidate_id"]):
        raise ValueError("all15 candidate does not match the replicated candidate")
    expected_arms = (set(replicated_selection["relevant_comparators"]) |
                     {replicated_selection["candidate_id"]})
    expected_keys = {(corruption, seed) for corruption in PILOT_CORRUPTIONS for seed in (0, 1, 2)}
    blocks = reference.get("paired_blocks")
    if not isinstance(blocks, list) or len(blocks) != len(expected_keys):
        raise ValueError("all15 reference must contain exactly 12 completed replication blocks")
    observed = set()
    expected_indices = _SPLIT["replication_indices"]
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("all15 reference contains an invalid paired block")
        key = (block.get("corruption"), block.get("seed"))
        if (not isinstance(block.get("seed"), int) or isinstance(block.get("seed"), bool) or
                key not in expected_keys or key in observed):
            raise ValueError("all15 reference has duplicate or unexpected replication blocks")
        observed.add(key)
        block_identity = block.get("identity")
        if (not isinstance(block.get("block_id"), str) or not block["block_id"] or
                block.get("indices") != expected_indices or
                not isinstance(block_identity, dict) or
                block_identity.get("indices") != expected_indices or
                block_identity.get("seed") != block.get("seed")):
            raise ValueError("all15 reference block does not match the locked replication split")
        for fingerprint in ("runtime_fingerprint", "input_sha256", "locked_config_fingerprint"):
            value = block_identity.get(fingerprint)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("all15 reference block has an invalid paired identity fingerprint")
        arms = block.get("arms")
        if not isinstance(arms, list):
            raise ValueError("all15 reference block has no completed arm records")
        arm_ids = [row.get("arm_id") for row in arms if isinstance(row, dict)]
        if len(arm_ids) != len(arms) or len(set(arm_ids)) != len(arm_ids) or set(arm_ids) != expected_arms:
            raise ValueError("all15 reference block has an unexpected or duplicate arm inventory")
        complete = {row.get("arm_id") for row in arms
                    if isinstance(row, dict) and row.get("status") == "complete"}
        if not expected_arms.issubset(complete):
            raise ValueError("all15 reference block is missing a completed candidate or comparator")
        for row in arms:
            if (row.get("status") != "complete" or row.get("identity") != block_identity or
                    not isinstance(row.get("bundle_path"), str) or not row["bundle_path"].strip()):
                raise ValueError("all15 reference arm has a mismatched identity or missing bundle_path")
    if observed != expected_keys:
        raise ValueError("all15 reference is missing a required corruption/seed block")
    return dict(selection=selection, reference=reference, candidate_id=validated["candidate_id"])


def build_phase_plan(phase: str, selection: Optional[dict] = None, *,
                     scale_corruptions: Optional[Sequence[str]] = None,
                     scale_arm_ids: Optional[Sequence[str]] = None) -> PhasePlan:
    if phase not in ("smoke", "diagnose", "scale", "routing", "projection", "replicate", "all15"):
        raise ValueError("unknown phase: {}".format(phase))
    if phase != "scale" and (scale_corruptions is not None or scale_arm_ids is not None):
        raise ValueError("scale filters are only valid for the scale phase")
    if phase in ("replicate", "all15") and selection is None:
        raise ValueError("{} requires a reviewed selection manifest".format(phase))
    if phase == "smoke":
        arm_ids, corruptions, indices, seeds = ("C_SCD", "C_SPEC", "C_SUM"), SMOKE_CORRUPTIONS, _SPLIT["pilot_indices"][:32], (0,)
        scope = "first32 of locked pilot indices; technical smoke"
    elif phase in ("diagnose", "routing", "projection"):
        arm_ids = {"diagnose": ("C_SCD", "C_SPEC", "C_SUM"),
                   "routing": ("R_S0", "R_SG", "R_G0", "R_GS"),
                   "projection": ("P_SUM", "P_PC", "P_NORM")}[phase]
        corruptions, indices, seeds = PILOT_CORRUPTIONS, _SPLIT["pilot_indices"], (0,)
        scope = "first64 of locked pilot indices; development diagnostics"
    elif phase == "scale":
        arm_ids, corruptions, indices, seeds = _scale_arms(), PILOT_CORRUPTIONS, _SPLIT["pilot_indices"], (0,)
        scope = "isolated scalar hard-v1 scale pilot; first64 development indices"
        if scale_corruptions is not None:
            selected_corruptions = tuple(dict.fromkeys(scale_corruptions))
            if (not selected_corruptions or
                    set(selected_corruptions) - set(PILOT_CORRUPTIONS)):
                raise ValueError("scale_corruptions must select known pilot corruptions")
            corruptions = selected_corruptions
        if scale_arm_ids is not None:
            selected_arm_ids = tuple(dict.fromkeys(scale_arm_ids))
            scale_library = {arm.arm_id: arm for arm in arm_ids}
            if not selected_arm_ids or set(selected_arm_ids) - set(scale_library):
                raise ValueError("scale_arm_ids must select known scale arms")
            arm_ids = tuple(scale_library[arm_id] for arm_id in selected_arm_ids)
        if scale_corruptions is not None or scale_arm_ids is not None:
            scope += "; selected corruptions={} arms={}".format(
                ",".join(corruptions), ",".join(arm.arm_id for arm in arm_ids))
    elif phase == "replicate":
        chosen = _selected_arm(selection)
        comparators = tuple(selection["relevant_comparators"])
        is_scale = chosen.arm_id.startswith("SCALE_")
        if not is_scale and "C_SCD" not in comparators:
            raise ValueError("replication selection must include C_SCD")
        arm_library = dict(_ARM_LIBRARY)
        arm_library.update((arm.arm_id, arm) for arm in _scale_arms())
        arm_ids = tuple(arm_library[value] for value in comparators) + (chosen,)
        if len({arm.arm_id for arm in arm_ids}) != len(arm_ids):
            raise ValueError("selection candidate cannot also be a comparator")
        corruptions, indices, seeds = PILOT_CORRUPTIONS, _SPLIT["replication_indices"], (0, 1, 2)
        scope = "next256 locked indices; three-seed development replication"
    else:
        chosen = _selected_arm(selection)
        if selection["source_phase"] != "replicate":
            raise ValueError("all15 requires a frozen selection from completed replication")
        if "C_SCD" not in selection["relevant_comparators"]:
            raise ValueError("all15 selection must include C_SCD")
        arm_ids = (_ARM_LIBRARY["C_SCD"], chosen)
        corruptions, indices, seeds = CORRUPTIONS, tuple(range(SPLIT_SIZE)), (0, 1, 2)
        scope = "complete ModelNet40-C severity-5 development set; not independent confirmation"

    arms = tuple(arm_ids) if arm_ids and isinstance(arm_ids[0], ArmSpec) else tuple(_ARM_LIBRARY[key] for key in arm_ids)
    return PhasePlan(phase, arms, tuple(corruptions), tuple(indices), tuple(seeds), scope)


def write_manifest(path: Path, manifest: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Path) -> dict:
    path = Path(path)
    payload = _strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object: " + str(path))
    return payload


def _strict_json(value):
    def reject_constant(name):
        raise ValueError("non-finite JSON constant: " + name)
    return json.loads(value, parse_constant=reject_constant)


def validate_resume_block(records: Sequence[dict], expected_arm_ids: Sequence[str],
                          expected_identity: dict) -> bool:
    if len(records) != len(expected_arm_ids):
        raise ValueError("paired block has incomplete arm inventory")
    by_id = {}
    for record in records:
        if not isinstance(record, dict) or record.get("arm_id") in by_id:
            raise ValueError("paired block has duplicate or invalid arm records")
        by_id[record.get("arm_id")] = record
    if set(by_id) != set(expected_arm_ids):
        raise ValueError("paired block arm inventory does not match the requested block")
    for arm_id in expected_arm_ids:
        record = by_id[arm_id]
        if record.get("status") != "complete":
            raise ValueError("partial or failed paired blocks cannot be resumed")
        if record.get("identity") != expected_identity:
            raise ValueError("paired block runtime, config, seed, indices, or input identity mismatch")
    return True


def _csv_text(columns: Sequence[str], row: dict) -> str:
    import csv
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue()


def write_arm_bundle(path: Path, manifest: dict, labels, predictions, logits,
                     diagnostics: Iterable[dict], *, command: str = "", environment: str = "") -> Path:
    """Write a method-scoped ten-file arm directory from completed arrays."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    indices = np.asarray(manifest.get("original_indices", []), dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1:] != (40,):
        raise ValueError("logits must have shape [N,40]")
    n = len(labels)
    if len(predictions) != n or len(indices) != n or logits.shape[0] != n:
        raise ValueError("labels, indices, predictions, and logits must have matching counts")
    status = manifest.get("status", "complete")
    if status not in ("complete", "partial", "failed"):
        raise ValueError("invalid arm status")
    corruption, seed = manifest["corruption"], int(manifest["seed"])
    if corruption not in CORRUPTIONS:
        raise ValueError("unsupported corruption")
    correct = int((labels == predictions).sum())
    run_id = manifest.get("run_id", path.name)
    config = dict(
        run_id=run_id, dataset="modelnet40_c", severity=5, method=METHOD, seed=seed,
        batch_size=32, corruption=corruption, status=status,
        composition_config=manifest.get("config", {}),
        experiment_fingerprint=manifest.get("experiment_fingerprint"),
        runtime_fingerprint=manifest.get("runtime_fingerprint"),
    )
    per_row = dict(run_id=run_id, dataset="modelnet40_c", severity=5, method=METHOD,
                   seed=seed, corruption=corruption, n_examples=n, n_correct=correct,
                   accuracy=(correct / n if n else ""),
                   runtime_seconds=manifest.get("total_runtime_seconds", 0.0),
                   peak_gpu_memory_mb=manifest.get("peak_gpu_memory_mb", 0.0), status=status)
    summary = dict(run_id=run_id, dataset="modelnet40_c", severity=5, method=METHOD,
                   seed=seed, n_corruptions=1,
                   macro_accuracy=(correct / n if n else ""), total_examples=n,
                   total_correct=correct, micro_accuracy=(correct / n if n else ""),
                   total_runtime_seconds=manifest.get("total_runtime_seconds", 0.0), status=status)
    (path / "command.txt").write_text(command + "\n", encoding="utf-8")
    (path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (path / "environment.txt").write_text(environment + "\n", encoding="utf-8")
    (path / "stdout.log").write_text("", encoding="utf-8")
    (path / "notes.md").write_text("# Composition arm\n\n{}\n".format(manifest.get("scope", "Paired GSD composition run.")), encoding="utf-8")
    (path / "per_corruption.csv").write_text(_csv_text(PER_COLUMNS, per_row), encoding="utf-8")
    (path / "summary.csv").write_text(_csv_text(SUMMARY_COLUMNS, summary), encoding="utf-8")
    np.savez_compressed(path / "predictions.npz", labels=labels, original_indices=indices,
                        predictions=predictions, logits=logits)
    with (path / "gradient_diagnostics.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in diagnostics:
                compressed.write((canonical_json(row) + "\n").encode("utf-8"))
    record = dict(manifest, method=METHOD, run_id=run_id,
                  n_examples=n, n_correct=correct, status=status)
    record["artifacts"] = {
        name: dict(bytes=(path / name).stat().st_size, sha256=_file_sha256(path / name))
        for name in sorted(EXPECTED_ARM_FILES - {"experiment_manifest.json"})
    }
    write_manifest(path / "experiment_manifest.json", record)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_bytes(path: Path) -> Tuple[dict, str]:
    path = Path(path)
    if path.is_dir():
        files = [item for item in path.iterdir() if item.is_file()]
        if len(files) != len(EXPECTED_ARM_FILES) or {item.name for item in files} != EXPECTED_ARM_FILES:
            raise ValueError("arm directory must contain the exact ten-file schema: " + str(path))
        return {item.name: item.read_bytes() for item in files}, path.name
    if path.suffix.lower() != ".zip":
        raise ValueError("arm input must be a directory or ZIP: " + str(path))
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if archive.testzip() is not None:
            raise ValueError("arm ZIP failed CRC validation")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("unsafe path in arm ZIP")
        roots = {Path(name).parts[0] for name in names if len(Path(name).parts) == 2}
        if (len(names) != len(EXPECTED_ARM_FILES) or len(roots) != 1 or
                any(len(Path(name).parts) != 2 for name in names) or
                {Path(name).name for name in names} != EXPECTED_ARM_FILES):
            raise ValueError("arm ZIP must contain one exact ten-file run directory")
        return {Path(name).name: archive.read(name) for name in names}, next(iter(roots))


def validate_arm_bundle(path: Path) -> dict:
    files, root_name = _bundle_bytes(Path(path))
    manifest = _strict_json(files["experiment_manifest.json"].decode("utf-8"))
    config = _strict_json(files["config.json"].decode("utf-8"))
    if manifest.get("method") != METHOD or config.get("method") != METHOD:
        raise ValueError("bundle method identity is not {}".format(METHOD))
    if manifest.get("run_id") != root_name or config.get("run_id") != root_name:
        raise ValueError("bundle root and run identity disagree")
    if manifest.get("status") not in ("complete", "partial", "failed"):
        raise ValueError("bundle has invalid status")
    phase = manifest.get("phase")
    phase_scopes = {
        "smoke": (SMOKE_CORRUPTIONS, tuple(_SPLIT["pilot_indices"][:32])),
        "diagnose": (PILOT_CORRUPTIONS, tuple(_SPLIT["pilot_indices"])),
        "scale": (PILOT_CORRUPTIONS, tuple(_SPLIT["pilot_indices"])),
        "routing": (PILOT_CORRUPTIONS, tuple(_SPLIT["pilot_indices"])),
        "projection": (PILOT_CORRUPTIONS, tuple(_SPLIT["pilot_indices"])),
        "replicate": (PILOT_CORRUPTIONS, tuple(_SPLIT["replication_indices"])),
        "all15": (CORRUPTIONS, tuple(range(SPLIT_SIZE))),
    }
    if phase not in phase_scopes:
        raise ValueError("bundle phase is missing or unknown")
    expected_corruptions, expected_indices = phase_scopes[phase]
    if manifest.get("corruption") not in expected_corruptions:
        raise ValueError("bundle corruption is outside its phase scope")
    if (not isinstance(manifest.get("arm_id"), str) or not manifest["arm_id"] or
            manifest.get("corruption") not in CORRUPTIONS or
            isinstance(manifest.get("seed"), bool) or not isinstance(manifest.get("seed"), int) or
            manifest["seed"] < 0):
        raise ValueError("bundle arm/corruption/seed identity is invalid")
    route = manifest.get("config")
    if not isinstance(route, dict) or config.get("composition_config") != route:
        raise ValueError("config and manifest route records disagree")
    try:
        CompositionConfig(**route)
    except (TypeError, ValueError) as error:
        raise ValueError("bundle contains an invalid composition config") from error
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != EXPECTED_ARM_FILES - {"experiment_manifest.json"}:
        raise ValueError("bundle has incomplete artifact hashes")
    for name, identity in artifacts.items():
        payload = files[name]
        if not isinstance(identity, dict):
            raise ValueError("invalid artifact identity: " + name)
        if identity.get("bytes") != len(payload) or identity.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError("artifact hash mismatch: " + name)
    import io
    arrays = np.load(io.BytesIO(files["predictions.npz"]), allow_pickle=False)
    required_arrays = {"labels", "original_indices", "predictions", "logits"}
    if set(arrays.files) != required_arrays:
        raise ValueError("predictions archive has unexpected arrays")
    labels = np.asarray(arrays["labels"])
    indices = np.asarray(arrays["original_indices"])
    predictions = np.asarray(arrays["predictions"])
    logits = np.asarray(arrays["logits"])
    n = labels.size
    if labels.ndim != 1 or indices.shape != labels.shape or predictions.shape != labels.shape:
        raise ValueError("prediction arrays have inconsistent shapes")
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("labels, original indices, and predictions must be integers")
    if len(np.unique(indices)) != n:
        raise ValueError("duplicate original indices")
    if logits.shape != (n, 40) or not np.issubdtype(logits.dtype, np.floating) or not np.isfinite(logits).all():
        raise ValueError("logits must be finite and have shape [N,40]")
    if n and not np.array_equal(logits.argmax(axis=1), predictions):
        raise ValueError("predicted classes disagree with logits")
    if labels.size and (labels.min() < 0 or labels.max() >= 40):
        raise ValueError("label outside ModelNet40 class range")
    if predictions.size and (predictions.min() < 0 or predictions.max() >= 40):
        raise ValueError("prediction outside ModelNet40 class range")
    csv = __import__("csv")
    per_reader = csv.DictReader(files["per_corruption.csv"].decode("utf-8").splitlines())
    summary_reader = csv.DictReader(files["summary.csv"].decode("utf-8").splitlines())
    if tuple(per_reader.fieldnames or ()) != PER_COLUMNS or tuple(summary_reader.fieldnames or ()) != SUMMARY_COLUMNS:
        raise ValueError("CSV artifacts do not match the standard seven-file schema")
    per_rows = list(per_reader)
    summary_rows = list(summary_reader)
    if len(per_rows) != 1 or len(summary_rows) != 1:
        raise ValueError("each arm bundle must contain one corruption and one summary row")
    row, summary = per_rows[0], summary_rows[0]
    correct = int((labels == predictions).sum())
    if int(row["n_examples"]) != n or int(row["n_correct"]) != correct:
        raise ValueError("per-corruption counts disagree with predictions")
    if int(summary["total_examples"]) != n or int(summary["total_correct"]) != correct:
        raise ValueError("summary counts disagree with predictions")
    if row["method"] != METHOD or row["corruption"] != manifest.get("corruption"):
        raise ValueError("CSV rows disagree with experiment identity")
    if int(row["seed"]) != int(manifest.get("seed")):
        raise ValueError("CSV seed disagrees with experiment identity")
    if (row["run_id"] != root_name or summary["run_id"] != root_name or
            summary["seed"] != row["seed"] or summary["dataset"] != row["dataset"] or
            row["dataset"] != "modelnet40_c" or row["severity"] != "5"):
        raise ValueError("CSV run/dataset identity is invalid")
    if row["status"] != manifest.get("status") or summary["status"] != manifest.get("status"):
        raise ValueError("CSV and experiment statuses disagree")
    if manifest.get("original_indices") != indices.astype(int).tolist():
        raise ValueError("manifest indices disagree with predictions archive")
    index_positions = {value: position for position, value in enumerate(expected_indices)}
    observed_indices = indices.astype(int).tolist()
    if any(value not in index_positions for value in observed_indices):
        raise ValueError("bundle indices are outside their phase scope")
    positions = [index_positions[value] for value in observed_indices]
    if positions != sorted(positions):
        raise ValueError("bundle indices do not preserve phase order")
    if manifest.get("status") == "complete" and positions != list(range(len(expected_indices))):
        raise ValueError("complete bundle does not match its phase scope")
    if manifest.get("n_examples") != n or manifest.get("n_correct") != correct:
        raise ValueError("manifest counts disagree with predictions")
    if int(summary["n_corruptions"]) != 1 or summary["method"] != METHOD:
        raise ValueError("summary identity or corruption count is invalid")
    expected_accuracy = correct / n if n else ""
    try:
        csv_accuracy = float(row["accuracy"]) if row["accuracy"] else ""
        summary_accuracy = float(summary["macro_accuracy"]) if summary["macro_accuracy"] else ""
    except ValueError as error:
        raise ValueError("CSV accuracy fields must be numeric") from error
    if csv_accuracy != expected_accuracy or summary_accuracy != expected_accuracy:
        raise ValueError("CSV accuracy fields disagree with predictions")
    if (summary["micro_accuracy"] != str(expected_accuracy) and
            not (expected_accuracy == "" and summary["micro_accuracy"] == "")):
        raise ValueError("summary micro accuracy disagrees with prediction counts")
    try:
        runtime = float(row["runtime_seconds"])
        memory = float(row["peak_gpu_memory_mb"])
    except (TypeError, ValueError) as error:
        raise ValueError("runtime and memory fields must be numeric") from error
    if not np.isfinite(runtime) or runtime < 0 or not np.isfinite(memory) or memory < 0:
        raise ValueError("runtime and memory must be finite and nonnegative")
    if manifest.get("status") == "complete" and n == 0:
        raise ValueError("a complete arm cannot contain zero predictions")
    diag_data = gzip.decompress(files["gradient_diagnostics.jsonl.gz"]).decode("utf-8")
    diagnostics = [_strict_json(line) for line in diag_data.splitlines() if line.strip()]
    if any(not _finite_tree(row) for row in diagnostics):
        raise ValueError("diagnostics contain a non-finite number")
    return dict(path=str(path), root_name=root_name, manifest=manifest, config=config,
                labels=labels, indices=indices, predictions=predictions, logits=logits,
                diagnostics=diagnostics, n_examples=int(n), n_correct=correct,
                accuracy=(correct / n if n else None), per_corruption=row, summary=summary)


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return np.isfinite(value).item()
    return True


def write_arm_zip(directory: Path, archive_path: Optional[Path] = None) -> Path:
    directory = Path(directory)
    validate_arm_bundle(directory)
    archive_path = Path(archive_path or directory.with_suffix(".zip"))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED) as archive:
        for name in sorted(EXPECTED_ARM_FILES):
            archive.write(directory / name, arcname=directory.name + "/" + name)
    return archive_path
