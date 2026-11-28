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


def encode(runtime, condition: str, start: int, stop: int, profile: dict, profile_sha: str) -> bytes:
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
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays, corruption=condition, seed=0, severity=5,
                        models=MODELS, probe_indices=[20, 10, 0], profile_sha256=profile_sha,
                        provenance_sha256=profile["provenance_sha256"])
    return stream.getvalue()


def validate(data, condition, start, stop, labels, profile, profile_sha) -> dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        if len(saved.files) != len(set(saved.files)):
            raise ValueError("Duplicate NPZ member")
        v = dict(saved)
    n = stop - start
    expected = dict(ids=np.arange(start, stop), labels=labels[start:stop], models=MODELS,
                    corruption=condition, seed=0, severity=5, probe_indices=[20, 10, 0],
                    provenance_sha256=profile["provenance_sha256"], profile_sha256=profile_sha)
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


def run(runtime, root: Path, *, calibrate_only: bool = False, conditions=None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    inventory(root)
    labels = np.asarray(runtime.labels)
    if (labels.shape != (N_OBJECTS,) or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels >= 40))):
        raise ValueError("Wrong label mapping")
    campaign = dict(protocol=protocol(), labels=labels.tolist(), runtime=runtime.identity,
                    runner_sha256=file_sha(Path(__file__)))
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

    for condition in conditions or CORRUPTIONS:
        profile, profile_sha = calibration(runtime, root, condition, campaign_sha)
        if calibrate_only:
            continue
        for corr, start, stop, name in ranges():
            if corr != condition:
                continue
            args = (condition, start, stop, labels, profile, profile_sha)
            try:
                validate(checked(read, name), *args)
                continue
            except INVALID:
                quarantine(root, root/name)
            payload = encode(runtime, condition, start, stop, profile, profile_sha)
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
        cells = {m: {} for m in MODELS}
        for cond, start, stop, name in ranges():
            values = validate(consume(name), cond, start, stop, labels, *profiles[cond])
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
    return dict(provenance_sha256=digest, model_object_records=N_OBJECTS*len(CORRUPTIONS)*len(MODELS),
                cells=cells, selected_batch_sizes={c: p["selected_batch_size"] for c,(p,_) in profiles.items()})


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
        from scripts.cloudfixer_surface_four_runtime import SurfaceRuntime
        attempt = root/"attempts"/uuid4().hex
        now = time.time()
        sealed(attempt.with_suffix(".start.json"), canonical(dict(
            action=args.action, start_epoch=now, started_utc=datetime.now(timezone.utc).isoformat(),
            deadline_epoch=args.deadline_epoch, command=shlex.join(sys.argv))))
        status = "failed"
        try:
            runtime = SurfaceRuntime(args)
            run(runtime, root, calibrate_only=args.action == "calibrate", conditions=args.corruptions)
            status = "complete"
        finally:
            sealed(attempt.with_suffix(".end.json"), canonical(dict(
                status=status, end_epoch=time.time(), elapsed_seconds=time.time()-now)))


if __name__ == "__main__":
    main()
