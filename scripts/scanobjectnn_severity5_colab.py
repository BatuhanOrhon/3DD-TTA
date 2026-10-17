"""Pinned author-code Gaussian5 generation and verified Colab/Drive delivery.

CPU data preparation only. No model, inference, normalization or GPU setup.
Python 3.8 compatible; notebook embeds this file and the author source.
"""
import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import traceback
import uuid
import zipfile

AUTHOR_COMMIT = "107305fd7baf40b359f31c07d235599198be7324"
AUTHOR_SHA = "3c18c0b3a7018d84f4517a5b431eefab8a52427dba49385d294b69fa645cd8bb"
SCHEMA = "scanobjectnn-author-gaussian5-v1"
ARRAYS = ("data_original.npy", "data_gaussian_5.npy", "label.npy")
DEFAULT_DRIVE = "/content/drive/MyDrive/thesis/dataset/scanobjectnn_c_s5_seed0"


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_members(archive):
    names = set()
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (not info.filename or "\\" in info.filename or ":" in info.filename
                or path.is_absolute() or ".." in path.parts
                or info.filename in names
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError("Unsafe or duplicate ZIP member: " + info.filename)
        names.add(info.filename)
    if archive.testzip() is not None:
        raise ValueError("ZIP CRC failure")
    return names


def author_functions(path):
    """Execute only unchanged author functions; no mesh/GUI/PyGeM imports."""
    import numpy as np
    source = Path(path).read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != AUTHOR_SHA:
        raise ValueError("Author generator source does not match pinned upstream")
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("normalize", "uniform_noise", "gaussian_noise")]
    if len(selected) != 3:
        raise ValueError("Missing author corruption functions")
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def validate_raw(data, labels):
    import numpy as np
    labels = np.asarray(labels).reshape(-1)
    if data.shape != (581, 2048, 3) or data.dtype != np.float32:
        raise ValueError("Expected main_split test data float32[581,2048,3]")
    if (labels.shape != (581,) or not np.issubdtype(labels.dtype, np.integer)
            or set(labels.tolist()) != set(range(15)) or not np.isfinite(data).all()):
        raise ValueError("Invalid raw data or 15-class labels")
    return labels


def generate_dataset(data, labels, generator_path, session, source_identity):
    import numpy as np
    session = Path(session)
    audit = session / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    labels = validate_raw(data, labels)
    functions = author_functions(generator_path)
    dataset = session / "dataset"
    dataset.mkdir()  # Never merge into an existing generated dataset.
    source = Path(generator_path).read_text(encoding="utf-8")
    with (audit / "author_generator.py").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(source)
    shutil.copyfile(Path(__file__), audit / "scanobjectnn_severity5_colab.py")
    selections = {}
    saved_rng = np.random.get_state()
    try:
        np.random.seed(0)
        # Preserve the published MAP order through Gaussian: Uniform draws first.
        for corruption in ("uniform", "gaussian"):
            indices = np.random.choice(data.shape[1], 2048, replace=False)
            selections[corruption] = indices.tolist()
            clouds = data[:, indices, :]
            function = functions[corruption + "_noise"]
            if corruption == "uniform":
                for cloud in clouds:
                    function(cloud, 5)  # Output discarded; original RNG order retained.
            else:
                gaussian = np.stack([function(cloud, 5) for cloud in clouds])
    finally:
        np.random.set_state(saved_rng)
    # Independent replay verifies sigma, point permutation, dtype and clipping.
    replay = np.random.RandomState(0)
    replay.choice(2048, 2048, replace=False)
    for cloud in data:
        replay.uniform(-0.05, 0.05, cloud.shape)
    replay_indices = replay.choice(2048, 2048, replace=False)
    for index, cloud in enumerate(data):
        expected = np.clip((cloud[replay_indices] + replay.normal(size=cloud.shape)
                            * 0.03).astype("float32"), -1, 1)
        if not np.array_equal(expected, gaussian[index]):
            raise ValueError("Author Gaussian output failed independent sigma0.03 replay")
    np.save(dataset / ARRAYS[0], data, allow_pickle=False)
    np.save(dataset / ARRAYS[1], gaussian, allow_pickle=False)
    np.save(dataset / ARRAYS[2], labels, allow_pickle=False)
    np.testing.assert_array_equal(np.load(dataset / ARRAYS[0]), data)
    np.testing.assert_array_equal(np.load(dataset / ARRAYS[2]), labels)
    manifest = {
        "schema": SCHEMA, "dataset": "scanobjectnn_c", "scope": "gaussian",
        "severity": 5, "seed": 0, "sigma": 0.03, "clip": [-1, 1],
        "split": "main_split/test_objectdataset.h5", "n_examples": 581,
        "point_selection": selections, "generation_order": ["uniform", "gaussian"],
        "saved_corruptions": ["gaussian"], "normalization": "none added",
        "source_h5": source_identity, "author_commit": AUTHOR_COMMIT,
        "author_source_sha256_lf": AUTHOR_SHA,
        "author_run_seed": "unpublished; seed0 is our reproducibility choice",
        "original_author_dataset_byte_identity": "not established",
        "gaussian_replay_exact": True,
        "raw_min": float(data.min()), "raw_max": float(data.max()),
        "raw_fraction_outside_unit_cube": float(np.mean(np.abs(data) > 1)),
        "gaussian_fraction_at_clip_boundary": float(np.mean(np.abs(gaussian) == 1)),
        "label_counts": np.bincount(labels, minlength=15).tolist(),
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "platform": platform.platform()},
        "files": {name: identity(dataset / name) for name in ARRAYS},
    }
    write_json(dataset / "manifest.json", manifest)
    verify_dataset(dataset)
    write_json(audit / "status.json", {"status": "complete", "schema": SCHEMA})
    print("Validated Gaussian severity5 + clean: 581 examples, exact replay passed", flush=True)
    return manifest


def verify_dataset(dataset):
    import numpy as np
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    locked = {"schema": SCHEMA, "dataset": "scanobjectnn_c", "scope": "gaussian",
              "severity": 5, "seed": 0, "sigma": 0.03, "clip": [-1, 1],
              "split": "main_split/test_objectdataset.h5", "n_examples": 581,
              "author_commit": AUTHOR_COMMIT, "author_source_sha256_lf": AUTHOR_SHA,
              "generation_order": ["uniform", "gaussian"], "gaussian_replay_exact": True}
    if any(manifest.get(key) != value for key, value in locked.items()):
        raise ValueError("Dataset does not match the locked author-code Gaussian5 protocol")
    if set(manifest.get("files", {})) != set(ARRAYS):
        raise ValueError("Incomplete dataset manifest")
    if {p.name for p in dataset.iterdir()} != set(ARRAYS) | {"manifest.json"}:
        raise ValueError("Unexpected or missing files in dataset directory")
    for name in ARRAYS:
        if identity(dataset / name) != manifest["files"][name]:
            raise ValueError("Dataset content identity mismatch: " + name)
    data = np.load(dataset / ARRAYS[0], allow_pickle=False)
    labels = np.load(dataset / ARRAYS[2], allow_pickle=False)
    validate_raw(data, labels)
    gaussian = np.load(dataset / ARRAYS[1], allow_pickle=False)
    if (gaussian.shape != data.shape or gaussian.dtype != np.float32
            or not np.isfinite(gaussian).all() or np.any(np.abs(gaussian) > 1)):
        raise ValueError("Invalid Gaussian array")
    if np.bincount(labels, minlength=15).tolist() != manifest["label_counts"]:
        raise ValueError("Label counts changed")
    return manifest


def publish(session, drive_root):
    session, drive_root = Path(session), Path(drive_root)
    status = json.loads((session / "audit/status.json").read_text(encoding="utf-8"))
    complete = status.get("status") == "complete"
    if complete:
        verify_dataset(session / "dataset")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    archive_path = session / ("handoff_" + stamp + ".zip")
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for folder in (session / "dataset", session / "audit"):
            for path in sorted(folder.rglob("*")) if folder.exists() else []:
                if path.is_file():
                    archive.write(path, path.relative_to(session).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        checked_members(archive)
    relative = "runs/" + stamp + "/dataset.zip"
    destination = drive_root / relative
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(archive_path, destination)
    if identity(destination) != identity(archive_path):
        raise ValueError("Drive backup hash/size mismatch; latest pointer not changed")
    receipt = dict(schema=SCHEMA, severity=5, seed=0, scope="gaussian", complete=complete,
                   archive_relative_path=relative, archive=identity(destination),
                   local_archive=str(archive_path), drive_archive=str(destination))
    write_json(destination.parent / "receipt.json", receipt)
    write_json(session / "backup_receipt.json", receipt)
    if complete:
        write_json(drive_root / "latest_gaussian.json", receipt)
    print("Drive backup verified: " + str(destination), flush=True)
    return receipt


def restore(drive_root, local_root, if_missing=False):
    drive_root, local_root = Path(drive_root), Path(local_root)
    pointer = drive_root / "latest_gaussian.json"
    if not pointer.exists() and if_missing:
        print("No verified ScanObjectNN Gaussian5 package on Drive yet. Run its generation section.")
        return None
    receipt = json.loads(pointer.read_text(encoding="utf-8"))
    if (receipt.get("schema") != SCHEMA or receipt.get("severity") != 5
            or receipt.get("seed") != 0 or receipt.get("scope") != "gaussian"
            or receipt.get("complete") is not True):
        raise ValueError("Invalid ScanObjectNN5 Drive receipt")
    relative = PurePosixPath(receipt["archive_relative_path"])
    if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3
            or relative.parts[0] != "runs" or relative.name != "dataset.zip"
            or "\\" in str(relative) or ":" in str(relative)):
        raise ValueError("Unsafe Drive archive path")
    archive_path = drive_root.joinpath(*relative.parts)
    if identity(archive_path) != receipt["archive"]:
        raise ValueError("Drive archive content identity mismatch")
    target = local_root / receipt["archive"]["sha256"]
    if not target.exists():
        with zipfile.ZipFile(archive_path) as archive:
            names = checked_members(archive)
            if not all(name.startswith(("dataset/", "audit/")) for name in names):
                raise ValueError("Unexpected archive layout")
            # A separate temporary destination prevents reuse of interrupted extraction.
            temporary = local_root / ("extract_" + uuid.uuid4().hex)
            temporary.mkdir(parents=True)
            archive.extractall(temporary)
        verify_dataset(temporary / "dataset")
        temporary.rename(target)
    verify_dataset(target / "dataset")
    author_functions(target / "audit/author_generator.py")
    if json.loads((target / "audit/status.json").read_text())["status"] != "complete":
        raise ValueError("Attempt is not complete")
    print("Verified ScanObjectNN Gaussian5 dataset: " + str(target / "dataset"), flush=True)
    return target / "dataset"


def locate_h5(source, search_root, session):
    """Accept a specific H5/ZIP, otherwise require one unambiguous main_split input."""
    candidates = []
    if source:
        paths = [Path(source)]
    else:
        paths = list(Path(search_root).rglob("test_objectdataset.h5"))
        paths = [p for p in paths if p.parent.name == "main_split"]
        if not paths:
            paths = [p for p in Path(search_root).rglob("*.zip")
                     if "scanobject" in p.name.lower() or p.name.lower() == "h5_files.zip"]
    for path in paths:
        if path.suffix.lower() == ".h5":
            if path.parent.name != "main_split" or path.name != "test_objectdataset.h5":
                raise ValueError("Select main_split/test_objectdataset.h5")
            candidates.append((path.resolve(), None))
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                matches = [n for n in archive.namelist()
                           if n.endswith("main_split/test_objectdataset.h5")
                           and PurePosixPath(n).parts[-2:] == ("main_split", "test_objectdataset.h5")]
                candidates.extend((path.resolve(), n) for n in matches)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValueError("Set ORIGINAL_SOURCE to one original main_split H5 or ZIP. Candidates: "
                         + repr(candidates))
    path, member = candidates[0]
    provenance = dict(identity(path), path=str(path), archive_member=member)
    if member:
        if ".." in PurePosixPath(member).parts or "\\" in member:
            raise ValueError("Unsafe H5 member")
        target = Path(session) / "input/main_split/test_objectdataset.h5"
        target.parent.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive, archive.open(member) as original, target.open("xb") as out:
            shutil.copyfileobj(original, out)
        path = target
    return path, dict(identity(path), path=str(path), container=provenance)


def generate_from_h5(args):
    import h5py
    session = Path(args.session)
    session.mkdir(parents=True, exist_ok=False)
    (session / "audit").mkdir()
    write_json(session / "audit/status.json", {"status": "running"})
    with (session / "audit/stdout.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                (session / "audit/command.txt").write_text(repr(sys.argv), encoding="utf-8")
                (session / "audit/environment.txt").write_text(
                    sys.version + "\nh5py=" + h5py.__version__ + "\n" + platform.platform(), encoding="utf-8")
                path, source_identity = locate_h5(args.source, args.search_root, session)
                print("Original H5:", path, source_identity, flush=True)
                with h5py.File(path, "r") as original:
                    data, labels = original["data"][:], original["label"][:]
                generate_dataset(data, labels, args.generator, session, source_identity)
            except Exception:
                traceback.print_exc()
                write_json(session / "audit/status.json", {"status": "failed"})
                raise
    print((session / "audit/stdout.log").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--session", required=True)
    generate.add_argument("--generator", required=True)
    generate.add_argument("--source", default="")
    generate.add_argument("--search-root", default="/content/drive/MyDrive/thesis/dataset")
    backup = commands.add_parser("backup")
    backup.add_argument("--session", required=True)
    backup.add_argument("--drive-root", default=DEFAULT_DRIVE)
    load = commands.add_parser("restore")
    load.add_argument("--drive-root", default=DEFAULT_DRIVE)
    load.add_argument("--local-root", default="/content/3DD-TTA/data/scanobjectnn_c_s5_seed0_gaussian")
    load.add_argument("--if-missing", action="store_true")
    load.add_argument("--state", default="/content/scanobjectnn_s5_loaded.json")
    args = parser.parse_args()
    if args.command == "generate":
        generate_from_h5(args)
    elif args.command == "backup":
        publish(args.session, args.drive_root)
    else:
        root = restore(args.drive_root, args.local_root, args.if_missing)
        write_json(Path(args.state), {"dataset_root": str(root) if root else None,
                                     "severity": 5, "scope": "gaussian", "schema": SCHEMA})


if __name__ == "__main__":
    main()
