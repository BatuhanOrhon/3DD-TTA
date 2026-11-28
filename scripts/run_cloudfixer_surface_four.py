#!/usr/bin/env python3
"""Run and independently recount the four frozen heads on a shared surface trajectory."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import time
from uuid import uuid4
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudfixer_store import MountStore, canonical, hard_deadline, sha
from research_artifacts import CORRUPTIONS

MODELS = ("dgcnn", "curvenet", "pointnext", "pointmae")
N_OBJECTS = 2468
BLOCK_SIZE = 32
DIGEST = re.compile("[0-9a-f]{64}")
REASONS = ("accepted", "unstable", "surface_support", "surface_veto")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def sealed(path: Path, data: bytes) -> None:
    store = MountStore(path.parent)
    store.put(path.name, data)
    store.put(sidecar(path).name, (sha(data) + "\n").encode())


def checked(read, name: str) -> bytes:
    value = read(name)
    if read(name + ".sha256") != (sha(value) + "\n").encode():
        raise ValueError("SHA-256 mismatch: " + name)
    return value


def disk_reader(root: Path):
    return lambda name: (root / name).read_bytes()


def protocol() -> dict:
    return dict(schema=1, method="pose_pca_surface_t50", dataset="modelnet40_c",
                severity=5, seed=0, models=list(MODELS), corruptions=list(CORRUPTIONS),
                objects=N_OBJECTS, block_size=BLOCK_SIZE, steps=8, start=80,
                tau=.1, retained=.95, t50_policy="surface-confirmed PCA",
                probe_indices=[20, 10, 0], denoiser_calls=9, pm="float64 softmax, equal mean")


def ranges():
    for condition in CORRUPTIONS:
        for start in range(0, N_OBJECTS, BLOCK_SIZE):
            stop = min(start + BLOCK_SIZE, N_OBJECTS)
            yield condition, start, stop, f"blocks/{condition}/{start:04d}-{stop:04d}.npz"


def read_profile(data: bytes, condition: str, campaign_sha: str) -> dict:
    profile = json.loads(data)
    size = profile["selected_batch_size"]
    eligible = {t["size"] for t in profile["trials"] if t["equivalent"] is True}
    controls = profile["independent_b1"].values()
    if (profile["corruption"] != condition or profile["provenance_sha256"] != campaign_sha
            or type(size) is not int or size not in (1, 2, 4, 8) or not {1, size} <= eligible
            or not controls or any(set(v["heads"]) != set(MODELS)
                                   or not all(h["exact"] is True for h in v["heads"].values()) for v in controls)):
        raise ValueError("Wrong calibration identity or selected batch for " + condition)
    return profile


def predictions(logits: np.ndarray) -> tuple:
    shifted = logits.astype(np.float64)
    shifted -= shifted.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs.mean(axis=1).argmax(axis=-1), logits[:, -1].argmax(axis=-1)


def batches(start: int, stop: int, size: int):
    while start < stop:
        # A shorter non-singleton tail was never calibrated.
        end = start + (size if start + size <= stop else 1)
        yield list(range(start, end))
        start = end


def encode(runtime, condition: str, start: int, stop: int, profile: dict, profile_sha: str, *, writer=None) -> bytes:
    rows = []
    for ids in batches(start, stop, profile["selected_batch_size"]):
        result = runtime.predict(condition, ids)
        n = len(ids)
        logits = np.asarray(result["logits"])
        if logits.shape != (n, len(MODELS), 3, 40) or logits.dtype != np.float32:
            raise ValueError("Runtime must return three float32 logit vectors per object and model")
        head_seconds = np.asarray(result["head_seconds"])
        if head_seconds.shape != (len(MODELS),):
            raise ValueError("Unexpected head timing shape")
        row = dict(ids=ids, labels=runtime.labels[ids], pose=result["pose"],
                   reason=result["reason"], angle_deg=result["angle_degrees"],
                   input_sha256=result["input_sha256"], trajectory_sha256=result["trajectory_sha256"],
                   batch_start=np.full(n, ids[0]), batch_size=np.full(n, n),
                   trajectory_ms=np.full(n, 1000*result["denoise_seconds"]/n),
                   wall_ms=np.full(n, 1000*result["wall_seconds"]/n),
                   peak_mib=np.full(n, result["peak_mib"]))
        row["accepted"] = np.asarray(row["reason"]) == "accepted"
        for h, model in enumerate(MODELS):
            values = logits[:, h]
            row[model+"_logits"] = values
            row[model+"_pm"] = np.asarray(result["pm"])[:, h]
            row[model+"_terminal"] = np.asarray(result["terminal"])[:, h]
            row[model+"_head_ms"] = np.full(n, 1000*head_seconds[h]/n)
        rows.append(row)
    arrays = {field: np.concatenate([r[field] for r in rows]) for field in rows[0]}
    if writer is not None:
        arrays["writer_sha256"] = writer
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays, corruption=condition, seed=0, severity=5,
                        models=MODELS, probe_indices=[20, 10, 0], profile_sha256=profile_sha,
                        provenance_sha256=profile["provenance_sha256"])
    return stream.getvalue()


def validate(data, condition, start, stop, labels, profile, profile_sha, writer=None) -> dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        if len(saved.files) != len(set(saved.files)):
            raise ValueError("Duplicate NPZ member")
        v = dict(saved)
    n = stop - start
    expected = dict(ids=np.arange(start, stop), labels=labels[start:stop], models=MODELS,
                    corruption=condition, seed=0, severity=5, probe_indices=[20, 10, 0],
                    provenance_sha256=profile["provenance_sha256"], profile_sha256=profile_sha)
    if writer is not None:
        expected["writer_sha256"] = writer
    expected_batches = list(batches(start, stop, profile["selected_batch_size"]))
    expected["batch_start"] = [b[0] for b in expected_batches for _ in b]
    expected["batch_size"] = [len(b) for b in expected_batches for _ in b]
    if any(not np.array_equal(v[k], wanted) for k, wanted in expected.items()):
        raise ValueError(f"Block identity, labels, IDs or calibrated batch differs at {condition}/{start}")
    if any(v[k].dtype.kind not in "iu" for k in ("ids", "labels", "batch_start", "batch_size")):
        raise ValueError("Block IDs, labels and batch fields must be integers")
    if any(v[k].shape != (n,) or not all(DIGEST.fullmatch(str(s)) for s in v[k])
           for k in ("input_sha256", "trajectory_sha256")):
        raise ValueError("Invalid input or trajectory digest")
    reason, accepted, angle, pose = (v[k] for k in ("reason", "accepted", "angle_deg", "pose"))
    if (reason.shape != (n,) or not np.isin(reason, REASONS).all()
            or accepted.dtype != np.bool_ or not np.array_equal(accepted, reason == "accepted")
            or angle.shape != (n,) or not np.isfinite(angle).all()
            or np.any(angle[~accepted] != 0) or np.any(angle[accepted] < 5-1e-3)
            or np.any(angle[accepted] > 30+1e-3) or pose.shape != (n, 3, 3)
            or not np.isfinite(pose).all()):
        raise ValueError("Invalid t50 pose, acceptance decision or applied angle")
    timing = ["trajectory_ms", "wall_ms", "peak_mib", *(m+"_head_ms" for m in MODELS)]
    for key in timing:
        arr = v[key]
        if (arr.shape != (n,) or not np.isfinite(arr).all() or np.any(arr < 0)
                or not np.array_equal(arr, arr[v["batch_start"]-start])):
            raise ValueError("Invalid per-batch timing or peak allocation: " + key)
    for model in MODELS:
        logits = v[model+"_logits"]
        if logits.shape != (n, 3, 40) or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError("Invalid classifier logits: " + model)
        for suffix, decision in zip(("_pm", "_terminal"), predictions(logits)):
            if v[model+suffix].dtype.kind not in "iu" or not np.array_equal(v[model+suffix], decision):
                raise ValueError("PM or terminal differs from independent logit recount: " + model)
    return v


def quarantine(root: Path, path: Path) -> None:
    evidence = [p for p in (path, sidecar(path)) if p.exists()]
    if evidence:
        folder = root / "quarantine" / uuid4().hex / path.parent.relative_to(root)
        folder.mkdir(parents=True)
        for p in evidence:
            p.replace(folder / p.name)


INVALID = (FileNotFoundError, KeyError, ValueError, EOFError, zipfile.BadZipFile)


def seal_identity(root: Path, path: Path, data: bytes) -> None:
    checksum = (sha(data) + "\n").encode()
    existing = path.read_bytes() if path.exists() else None
    receipt = sidecar(path).read_bytes() if sidecar(path).exists() else None
    if existing is not None or receipt is not None:
        if existing != data and receipt != checksum:
            raise ValueError(f"Existing {path.name} identity differs from the loaded runtime")
        if existing != data or receipt != checksum:
            quarantine(root, path)
    sealed(path, data)


def inventory(root: Path) -> None:
    allowed = {s for *_, name in ranges() for s in (name, name+".sha256")}
    observed = {p.relative_to(root).as_posix() for p in (root/"blocks").rglob("*")
                if p.is_file() and not p.name.startswith(".pending_")}
    if observed - allowed:
        raise ValueError("Unexpected blocks or duplicate ranges: " + str(sorted(observed - allowed)))


def calibration(runtime, root: Path, condition: str, campaign_sha: str):
    read = disk_reader(root)
    active = f"profiles/{condition}.json"
    mirror = f"profile_backups/{condition}.json"
    missing = []
    valid = {}
    for name in (active, mirror):
        try:
            raw = checked(read, name)
            profile = read_profile(raw, condition, campaign_sha)
            valid[raw] = profile
        except INVALID:
            missing.append(name)
    if len(valid) > 1:
        raise ValueError("Conflicting measured calibrations for " + condition)
    if valid:
        raw, profile = next(iter(valid.items()))
    else:
        profile = dict(runtime.calibrate(condition), corruption=condition, provenance_sha256=campaign_sha)
        raw = canonical(profile)
        read_profile(raw, condition, campaign_sha)
    for name in missing:
        quarantine(root, root/name)
        sealed(root/name, raw)
    return profile, sha(raw)


STRESS = ("uniform", "background", "upsampling", "rotation")
ORIGINAL_WRITER = "9a27805a6086dc10bdc08d0f727748ec751b82df429882aef52073fe3b3bcc5d"
TRANSFER = "transfer.json"
PARITY = "cross_vm_parity.json"


def check_campaign_pair(old, new, writer):
    def science(campaign):
        runtime = dict(campaign["runtime"])
        hardware = dict(runtime["hardware"])
        for key in ("gpu", "total_bytes"):
            hardware.pop(key)
        runtime["hardware"] = hardware
        return (campaign["labels"], campaign["protocol"], runtime)

    if (old["runner_sha256"] not in (ORIGINAL_WRITER, writer) or
            science(old) != science(new) or old["protocol"] != protocol()):
        raise ValueError("Source/target campaigns differ in labels, scientific sources, assets or numeric runtime")


def check_parity(receipt, old_id, new_id, old_profiles, new_profiles):
    if (receipt.get("schema") != 1 or receipt.get("status") != "passed" or receipt.get("source_campaign_sha256") != old_id
            or receipt.get("target_campaign_sha256") != new_id or old_id == new_id
            or set(receipt.get("profiles", {})) != set(STRESS)):
        raise ValueError("Cross-VM parity receipt does not match the registered campaigns and four controls")
    for condition in STRESS:
        a, old_sha = old_profiles[condition]
        b, new_sha = new_profiles[condition]
        link = receipt["profiles"][condition]
        if (link.get("source_sha256") != old_sha or link.get("target_sha256") != new_sha
                or a["ids"] != b["ids"] or a.get("reason_b1") != b.get("reason_b1")
                or set(a["independent_b1"]) != {"321", "1146"}
                or set(b["independent_b1"]) != {"321", "1146"}):
            raise ValueError("Cross-VM parity sample or profile identity differs: "+condition)
        for ident, left in a["independent_b1"].items():
            right = b["independent_b1"][ident]
            if not DIGEST.fullmatch(left["input_sha256"]) or left["input_sha256"] != right["input_sha256"]:
                raise ValueError("Cross-VM input parity failed: "+condition+"/"+ident)
            for head in MODELS:
                lhs, rhs = left["heads"][head], right["heads"][head]
                keys = ("clouds_sha256", "logits_sha256", "pm", "probe_predictions", "reason", "angle_degrees")
                if (any(lhs[k] != rhs[k] for k in keys) or
                        not all(DIGEST.fullmatch(lhs[k]) for k in ("clouds_sha256", "logits_sha256"))):
                    raise ValueError("Cross-VM cloud, logits or decision parity failed: "+condition+"/"+ident+"/"+head)


def metadata_names(root):
    return sorted(p.relative_to(root).as_posix() for folder in ("profiles","attempts")
                  for p in (root/folder).glob("*.json") if sidecar(p).is_file())


def profiles_from(files, campaign_id):
    return {name[9:-5]: (read_profile(raw, name[9:-5], campaign_id), sha(raw))
            for name, raw in files.items() if name.startswith("profiles/")}


def prepare_import(source, root, campaign_data, parity_path):
    if source.resolve() == root.resolve() or (source/TRANSFER).exists():
        raise ValueError("Expected a distinct, original source campaign")
    inventory(source)
    read = disk_reader(source)
    parity = checked(disk_reader(parity_path.parent), parity_path.name)
    if (root/TRANSFER).is_file() and sidecar(root/TRANSFER).is_file():
        index = json.loads(checked(disk_reader(root), TRANSFER))
        if (sha(checked(read, "campaign.json")) != index["source_campaign_sha256"]
                or sha(campaign_data) != index["target_campaign_sha256"] or sha(parity) != index["parity_sha256"]):
            raise ValueError("Resume sources differ from the sealed transfer")
        return index
    files = {key: checked(read, key) for key in ["campaign.json", *metadata_names(source)]}
    if (source/"frozen.json").is_file() and sidecar(source/"frozen.json").is_file():
        files["frozen.json"] = checked(read, "frozen.json")
    old_id = sha(files["campaign.json"])
    new_id = sha(campaign_data)
    writer = file_sha(Path(__file__))
    old, new = json.loads(files["campaign.json"]), json.loads(campaign_data)
    check_campaign_pair(old, new, writer)
    profiles = profiles_from(files, old_id)
    fresh = profiles_from({f"profiles/{c}.json": checked(disk_reader(root), f"profiles/{c}.json") for c in STRESS}, new_id)
    check_parity(json.loads(parity), old_id, new_id, profiles, fresh)
    # Freeze membership before publishing any old bytes at canonical target paths.
    blocks, invalid = {}, []
    available = {p.relative_to(source).as_posix() for p in (source/"blocks").glob("*/*.npz")}
    labels = np.asarray(old["labels"])
    for c, start, stop, key in ranges():
        if key in available:
            try:
                raw = checked(read, key)
                validate(raw, c, start, stop, labels, *profiles[c])
            except INVALID:
                invalid.append(key)
            else:
                blocks[key] = sha(raw)
    if not blocks:
        raise ValueError("No verified source blocks are available for import")
    index = dict(schema=1, source_campaign_sha256=old_id, target_campaign_sha256=new_id,
                 writer_sha256=writer, parity_sha256=sha(parity), blocks=blocks, invalid_source_blocks=invalid,
                 files={name:sha(raw) for name,raw in files.items()})
    for name, raw in files.items():
        sealed(root/"segments"/old_id/name, raw)
    sealed(root/PARITY, parity)
    seal_identity(root, root/TRANSFER, canonical(index))
    return index


def verify_transfer(consume, index, campaign, target_id, profiles):
    old_id = index["source_campaign_sha256"]
    writer = index["writer_sha256"]
    if (index.get("schema") != 1 or not DIGEST.fullmatch(old_id) or not DIGEST.fullmatch(writer)
            or index["target_campaign_sha256"] != target_id or not index["blocks"]):
        raise ValueError("Invalid archived campaign transfer identity")
    legal = {key for *_,key in ranges()}
    if (not set(index["blocks"]) <= legal or any(not DIGEST.fullmatch(h) for h in index["blocks"].values())
            or set(index["invalid_source_blocks"]) & set(index["blocks"])
            or not set(index["invalid_source_blocks"]) <= legal):
        raise ValueError("Unexpected duplicate, invalid or foreign transfer block")
    prefix = "segments/"+old_id+"/"
    files = {}
    for name,digest in sorted(index["files"].items()):
        allowed = (name in ("campaign.json","frozen.json") or name in {f"profiles/{c}.json" for c in CORRUPTIONS}
                   or re.fullmatch(r"attempts/[a-f0-9]{32}[.](start|end)[.]json",name))
        if not allowed:
            raise ValueError("Unexpected source campaign metadata: "+name)
        raw = consume(prefix+name)
        if sha(raw) != digest:
            raise ValueError("Imported metadata SHA mismatch: "+name)
        files[name] = raw
    if sha(files["campaign.json"]) != old_id:
        raise ValueError("Imported campaign SHA mismatch")
    old = json.loads(files["campaign.json"])
    check_campaign_pair(old, campaign, writer)
    old_profiles = profiles_from(files, old_id)
    raw = consume(PARITY)
    if sha(raw) != index["parity_sha256"]:
        raise ValueError("Cross-VM parity receipt SHA mismatch")
    parity = json.loads(raw)
    check_parity(parity, old_id, target_id, old_profiles, profiles)
    for replay in parity.get("uniform_raw_b1_replays", {}).values():
        if (replay["exact"] is not True or index["blocks"].get(replay["source_block"]) != replay["source_block_sha256"]):
            raise ValueError("Uniform parity replay lacks a matching original sealed block")
    if "frozen.json" in files:
        receipt = json.loads(files["frozen.json"])
        if receipt != dict(provenance_sha256=old_id,states=old["runtime"]["frozen_states"]):
            raise ValueError("Invalid source frozen receipt")
    segment = dict(hardware=old["runtime"]["hardware"], blocks=sorted(index["blocks"]),
                   attempts=sorted(prefix+k for k in files if k.startswith("attempts/")),
                   selected_batch_sizes={c:p["selected_batch_size"] for c,(p,_) in old_profiles.items()},
                   final_frozen_receipt="frozen.json" in files and set(index["blocks"]) == legal)
    return old_profiles, segment


def restore_imported(root, source, name, expected_sha, args):
    try:
        data = checked(disk_reader(root), name)
    except INVALID:
        if source is None:
            raise ValueError("Original --resume-from campaign required to restore "+name)
        data = checked(disk_reader(source), name)
        if sha(data) != expected_sha:
            raise ValueError("Original block SHA changed: "+name)
        validate(data, *args)
        quarantine(root, root/name)
        sealed(root/name, data)
        return
    if sha(data) != expected_sha:
        raise ValueError("Conflicting sealed blocks at "+name)
    validate(data, *args)


def run(runtime, root: Path, *, calibrate_only: bool = False, conditions=None, resume_from=None, parity_receipt=None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    inventory(root)
    labels = np.asarray(runtime.labels)
    if (labels.shape != (N_OBJECTS,) or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels >= 40))):
        raise ValueError("Wrong label mapping")
    campaign = dict(protocol=protocol(), labels=labels.tolist(), runtime=runtime.identity,
                    runner_sha256=file_sha(Path(__file__)))
    # A fresh target may already hold an original-writer calibration. Keep that signed registration.
    if (resume_from or (root/TRANSFER).exists()) and (root/"campaign.json").is_file():
        previous = json.loads(checked(disk_reader(root), "campaign.json"))
        if previous.get("runner_sha256") == ORIGINAL_WRITER:
            campaign["runner_sha256"] = ORIGINAL_WRITER
    campaign_data = canonical(campaign)
    campaign_sha = sha(campaign_data)
    read = disk_reader(root)
    path = root/"campaign.json"
    if not path.exists() and not sidecar(path).exists():
        for prior in (root/"profiles").glob("*.json"):
            recorded = json.loads(checked(read, prior.relative_to(root).as_posix()))
            if recorded["provenance_sha256"] != campaign_sha:
                raise ValueError("Previous calibration identifies a different campaign")
    seal_identity(root, path, campaign_data)
    def verify_frozen():
        states = runtime.verify_frozen()
        if states != runtime.identity["frozen_states"]:
            raise ValueError("Frozen states changed before publication")
        return states

    transfer = None
    if resume_from is not None or parity_receipt is not None:
        if not resume_from or not parity_receipt or calibrate_only:
            raise ValueError("A mixed run needs both --resume-from and --parity-receipt")
        for condition in STRESS:
            calibration(runtime, root, condition, campaign_sha)
        transfer = prepare_import(resume_from, root, campaign_data, parity_receipt)
    elif (root/TRANSFER).exists() or sidecar(root/TRANSFER).exists():
        transfer = json.loads(checked(read, TRANSFER))
    elif (root/"segments").exists() or (root/PARITY).exists():
        raise ValueError("Interrupted transfer; supply --resume-from and --parity-receipt")
    if transfer:
        if transfer["writer_sha256"] != file_sha(Path(__file__)):
            raise ValueError("Mixed-run writer source changed")
        controls = profiles_from({f"profiles/{c}.json": checked(read, f"profiles/{c}.json") for c in STRESS}, campaign_sha)
        old_profiles, _ = verify_transfer(lambda p: checked(read,p), transfer, campaign, campaign_sha, controls)
    for condition in conditions or CORRUPTIONS:
        profile, profile_sha = calibration(runtime, root, condition, campaign_sha)
        if calibrate_only:
            continue
        for corr, start, stop, name in ranges():
            if corr != condition:
                continue
            if transfer and name in transfer["blocks"]:
                restore_imported(root, resume_from, name, transfer["blocks"][name],
                                 (condition, start, stop, labels, *old_profiles[condition]))
                continue
            writer = transfer["writer_sha256"] if transfer else None
            args = (condition, start, stop, labels, profile, profile_sha, writer)
            try:
                validate(checked(read, name), *args)
                continue
            except INVALID:
                quarantine(root, root/name)
            payload = encode(runtime, condition, start, stop, profile, profile_sha, writer=writer)
            validate(payload, *args)
            verify_frozen()
            sealed(root/name, payload)
            print(f"{condition}: {stop}/{N_OBJECTS} IDs sealed on Drive", flush=True)
    frozen = canonical(dict(provenance_sha256=campaign_sha, states=verify_frozen()))
    seal_identity(root, root/"frozen.json", frozen)


def recount(read, names, archive=None) -> dict:
    """Check the closed evidence through either the Drive mount or a downloaded ZIP."""
    names = set(names)
    expected_blocks = {p for *_, name in ranges() for p in (name, name+".sha256")}
    if {p for p in names if p.startswith("blocks/")} != expected_blocks:
        raise ValueError("Unexpected or missing block paths (duplicate or incomplete coverage)")
    def consume(name):
        value = checked(read, name)
        if archive is not None:
            archive.writestr(zipfile.ZipInfo(name), value)
            archive.writestr(zipfile.ZipInfo(name+".sha256"), (sha(value)+"\n").encode())
        return value
    try:
        campaign_data = consume("campaign.json")
        campaign = json.loads(campaign_data)
        if campaign["protocol"] != protocol():
            raise ValueError("Different scientific protocol")
        labels = np.asarray(campaign["labels"])
        if labels.shape != (N_OBJECTS,) or labels.dtype.kind not in "iu" or not np.isin(labels, range(40)).all():
            raise ValueError("Invalid campaign labels")
        digest = sha(campaign_data)
        frozen = json.loads(consume("frozen.json"))
        if (frozen["provenance_sha256"] != digest or not frozen["states"]
                or frozen["states"] != campaign["runtime"]["frozen_states"]):
            raise ValueError("Missing frozen model-state receipt")
        profiles = {}
        for condition in CORRUPTIONS:
            raw = consume(f"profiles/{condition}.json")
            profiles[condition] = (read_profile(raw, condition, digest), sha(raw))
        transfer = json.loads(consume(TRANSFER)) if TRANSFER in names else None
        segments = {}
        if archive is None:
            expected_segment = {f"segments/{transfer['source_campaign_sha256']}/{k}{suffix}"
                                for k in transfer["files"] for suffix in ("", ".sha256")} if transfer else set()
            if {k for k in names if k.startswith("segments/")} != expected_segment:
                raise ValueError("Unexpected or missing source segment metadata")
        if transfer:
            old_profiles, segment = verify_transfer(consume, transfer, campaign, digest, profiles)
            segments[transfer["source_campaign_sha256"]] = segment
        primary_blocks = []
        cells = {m: {} for m in MODELS}
        for cond, start, stop, name in ranges():
            raw = consume(name)
            if transfer and name in transfer["blocks"]:
                if sha(raw) != transfer["blocks"][name]:
                    raise ValueError("Imported raw block SHA mismatch: "+name)
                item_profile = old_profiles[cond]
            else:
                item_profile = profiles[cond]
                primary_blocks.append(name)
            writer = transfer["writer_sha256"] if transfer and name not in transfer["blocks"] else None
            values = validate(raw, cond, start, stop, labels, *item_profile, writer=writer)
            for model in MODELS:
                cell = cells[model].setdefault(cond, dict(n=0, pm_correct=0, terminal_correct=0, accepted=0))
                cell["n"] += stop - start
                cell["pm_correct"] += int(np.sum(values[model+"_pm"] == values["labels"]))
                cell["terminal_correct"] += int(np.sum(values[model+"_terminal"] == values["labels"]))
                cell["accepted"] += int(np.sum(values["accepted"]))
        attempts = sorted(p for p in names if p.startswith("attempts/") and p.endswith(".json"))
        for p in attempts:
            json.loads(consume(p))
    except (KeyError, FileNotFoundError) as exc:
        raise ValueError("Missing required evidence: "+str(exc)) from exc
    sizes = {c:p["selected_batch_size"] for c,(p,_) in profiles.items()}
    segments[digest] = dict(hardware=campaign["runtime"].get("hardware", {}), blocks=primary_blocks,
                            attempts=attempts, selected_batch_sizes=sizes, final_frozen_receipt=True)
    return dict(provenance_sha256=digest, hardware_segments=segments, model_object_records=N_OBJECTS*len(CORRUPTIONS)*len(MODELS),
                cells=cells, selected_batch_sizes=sizes)


def verify_archive(path: Path) -> dict:
    actual = file_sha(path)
    if (sidecar(path).read_bytes() != (actual+"\n").encode()
            or (path.name.startswith("raw-") and path.stem != "raw-"+actual)):
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member")
        return recount(archive.read, names)


def finalize(root: Path) -> Path:
    inventory(root)
    names = [p.relative_to(root).as_posix() for folder in ("blocks", "attempts")
             for p in (root/folder).rglob("*") if p.is_file() and not p.name.startswith(".pending_")]
    if (root/TRANSFER).exists() or sidecar(root/TRANSFER).exists():
        names.append(TRANSFER)
    # Stage on local disk; a closed, verified archive is the publication boundary.
    with tempfile.TemporaryDirectory() as temp:
        staged = Path(temp)/"raw.zip"
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
            counts = recount(disk_reader(root), names, archive)
        digest = file_sha(staged)
        target = root/("raw-"+digest+".zip")
        if not target.exists() or file_sha(target) != digest:
            quarantine(root, target)
            pending = root/(".pending_"+uuid4().hex)
            try:
                with staged.open("rb") as src, pending.open("xb") as dest:
                    shutil.copyfileobj(src, dest)
                    dest.flush()
                    os.fsync(dest.fileno())
                if file_sha(pending) != digest:
                    raise OSError("Drive readback of raw ZIP differs from staged archive")
                pending.replace(target)
            finally:
                pending.unlink(missing_ok=True)
        MountStore(root).put(sidecar(target).name, (digest+"\n").encode())
    if verify_archive(target) != counts:
        raise ValueError("Raw archive recount differs from live blocks")
    sealed(root/("verified-"+digest+".json"), canonical(dict(archive=target.name, sha256=digest, **counts)))
    return target


@contextmanager
def owned_writer(root: Path):
    # This campaign owns one Colab VM. Protect restart clicks on that VM.
    lock = Path(tempfile.gettempdir())/("surface-four-"+sha(str(root).encode())+".lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("calibrate", "run"):
        part = commands.add_parser(action)
        part.add_argument("--store", required=True, type=Path)
        if action == "run":
            part.add_argument("--resume-from", type=Path, help="Original mounted campaign with sealed blocks")
            part.add_argument("--parity-receipt", type=Path, help="SHA-sealed four-condition cross-VM parity JSON")
        for option in ("manifest", "asset-root", "official-root", "diffusion", "dataset-root", "classifier"):
            part.add_argument("--"+option, required=True, type=Path)
        part.add_argument("--deadline-epoch", type=float, required=True,
                          help="Absolute UTC deadline registered for this GPU phase; retain on retry")
        part.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS, default=list(CORRUPTIONS))
    commands.add_parser("finalize").add_argument("--store", required=True, type=Path)
    commands.add_parser("verify").add_argument("--archive", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == "verify":
        print(json.dumps(verify_archive(args.archive), indent=2))
        return
    root = args.store.resolve()
    if args.action == "finalize":
        with owned_writer(root):
            print(finalize(root))
        return
    drive = Path("/content/drive")
    if not drive.is_mount() or drive/"MyDrive" not in root.parents:
        raise ValueError("GPU blocks require the mounted MyDrive destination")
    if len(args.corruptions) != len(set(args.corruptions)):
        raise ValueError("Duplicate corruption selection")
    remaining = args.deadline_epoch - time.time()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError("Registered whole-phase UTC deadline elapsed")
    with owned_writer(root), hard_deadline(remaining):
        for field in ("manifest", "asset_root", "official_root", "diffusion", "dataset_root", "classifier"):
            setattr(args, field, getattr(args, field).resolve())
        source = getattr(args, "resume_from", None)
        parity = getattr(args, "parity_receipt", None)
        source = source.resolve() if source else None
        parity = parity.resolve() if parity else None
        from scripts.cloudfixer_surface_four_runtime import SurfaceRuntime
        attempt = root/"attempts"/uuid4().hex
        now = time.time()
        sealed(attempt.with_suffix(".start.json"), canonical(dict(
            action=args.action, start_epoch=now, started_utc=datetime.now(timezone.utc).isoformat(),
            deadline_epoch=args.deadline_epoch, command=shlex.join(sys.argv))))
        status = "failed"
        try:
            runtime = SurfaceRuntime(args)
            run(runtime, root, calibrate_only=args.action == "calibrate", conditions=args.corruptions,
                resume_from=source, parity_receipt=parity)
            status = "complete"
        finally:
            sealed(attempt.with_suffix(".end.json"), canonical(dict(
                status=status, end_epoch=time.time(), elapsed_seconds=time.time()-now)))


if __name__ == "__main__":
    main()
