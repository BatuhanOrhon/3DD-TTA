import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


repo = Path("/content/3DD-TTA")
state_path = Path("/content/scanobjectnn_s1_profile_state.json")
assert not state_path.exists(), "Profile state exists; inspect it instead of duplicating the run"
assert sha256(repo / "utilities_3dd_tta.py") == "6f21ab34cd0537115c4eadbcaa411948661cb8acfdb92bcbed4fd5dd9945226a"

generator_path = Path("/content/create_corrupted_dataset.py")
assert sha256(generator_path) == "e3b2d5154903775ab5f7fab01a52e72bb57e4be4bba9041ac6c771cfdc0986b3"
spec = importlib.util.spec_from_file_location("scanobjectnn_corruptions", generator_path)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)

source = Path(json.loads(Path("/content/scanobjectnn_s5_loaded.json").read_text())["dataset_root"])
data = np.load(source / "data_original.npy", allow_pickle=False)
labels = np.load(source / "label.npy", allow_pickle=False)
assert data.shape == (581, 2048, 3) and labels.reshape(-1).shape == (581,)
assert sha256(source / "data_original.npy") == "47c641b3d32e342e5a4b7260a5166826ea9d4771942d2552f971eceae1c175fe"

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
root = Path("/content/scanobjectnn_s1_profile_" + stamp)
dataset = root / "dataset"
dataset.mkdir(parents=True)
np.save(dataset / "data_original.npy", data, allow_pickle=False)
np.save(dataset / "label.npy", labels, allow_pickle=False)

# Scale is generated only to preserve the released generator's RNG order; it is
# not one of the 13 non-rendered corruptions in the published ScanObjectNN-C row.
generation_order = (
    "uniform", "gaussian", "background", "impulse", "scale", "upsampling",
    "shear", "rotation", "cutout", "density", "density_inc", "distortion",
    "distortion_rbf", "distortion_rbf_inv",
)
evaluated = tuple(name for name in generation_order if name != "scale")
np.random.seed(0)
files = {}
for corruption in generation_order:
    generator.ORIG_NUM = 4096 if corruption == "density_inc" else 2048
    sample_num = min(generator.ORIG_NUM, data.shape[1])
    indices = np.random.choice(data.shape[1], sample_num, replace=False)
    sampled = data[:, indices, :]
    corrupted = np.stack([generator.MAP[corruption](cloud, 1) for cloud in sampled])
    assert corrupted.shape[0] == 581 and corrupted.shape[2] == 3
    assert np.isfinite(corrupted).all()
    if corruption in evaluated:
        path = dataset / ("data_" + corruption + "_1.npy")
        np.save(path, corrupted, allow_pickle=False)
        files[path.name] = {"shape": list(corrupted.shape), "sha256": sha256(path)}
    print("GENERATED", corruption, corrupted.shape, flush=True)

manifest = {
    "schema": "scanobjectnn-severity1-profile-v1",
    "purpose": "diagnose the unpublished severity/provenance mismatch in 3DD-TTA Table 3",
    "seed": 0,
    "actual_severity": 1,
    "source_original_sha256": sha256(source / "data_original.npy"),
    "generator_sha256": sha256(generator_path),
    "generation_order": list(generation_order),
    "evaluated_corruptions": list(evaluated),
    "files": files,
}
(dataset / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

checkpoint_source = Path("/content/drive/MyDrive/thesis/weights/scanobject_jt.pth")
checkpoint = repo / "pointnet_ckpts/scanobject_jt.pth"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(checkpoint_source, checkpoint)
expected_checkpoint = "53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac"
assert sha256(checkpoint) == expected_checkpoint

result_root = root / "results"
run_name = "scanobjectnn-severity1-profile-seed0"
command = [
    sys.executable, "-u", "run_baseline.py",
    "--method", "source_only", "--dataset-name", "scanobjectnn-c",
    "--pointmae_config", "./cfgs/tta_scanobj.yaml",
    "--pointmae_ckpt", "./pointnet_ckpts/scanobject_jt.pth",
    "--dataset_root", str(dataset), "--label_path", str(dataset / "label.npy"),
    "--max-batches", "0", "--batch_size", "32", "--seed", "0", "--severity", "1",
    "--corruptions", *evaluated, "--run-name", run_name, "--result-root", str(result_root),
]
state = {
    "status": "running",
    "dataset_root": str(dataset),
    "manifest": manifest,
    "checkpoint_sha256": expected_checkpoint,
}
state_path.write_text(json.dumps(state, indent=2) + "\n")
subprocess.run(command, cwd=repo, check=True)

run_dirs = list((result_root / "scanobjectnn_c/source_only").glob("*_" + run_name))
assert len(run_dirs) == 1, run_dirs
with (run_dirs[0] / "per_corruption.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == len(evaluated) and all(row["status"] == "complete" for row in rows)
observed = {row["corruption"]: 100 * float(row["accuracy"]) for row in rows}
paper = {
    "uniform": 59.0,
    "gaussian": 59.0,
    "background": 57.3,
    "impulse": 66.6,
    "upsampling": 64.4,
    "distortion_rbf": 73.6,
    "distortion_rbf_inv": 72.9,
    "density": 81.2,
    "density_inc": 82.4,
    "shear": 71.9,
    "rotation": 59.7,
    "cutout": 78.3,
    "distortion": 70.7,
}
comparison = {
    name: {"observed_accuracy_percent": observed[name],
           "paper_accuracy_percent": paper[name],
           "difference_pp": observed[name] - paper[name]}
    for name in evaluated
}
archive = run_dirs[0].with_suffix(".zip")
handoff = Path("/content/scanobjectnn_s1_profile_handoff.zip")
shutil.copyfile(archive, handoff)
drive_dir = Path("/content/drive/MyDrive/thesis/result/preparation/scanobjectnn_s1_profile") / stamp
drive_dir.mkdir(parents=True, exist_ok=False)
drive_handoff = drive_dir / handoff.name
shutil.copyfile(handoff, drive_handoff)
assert sha256(drive_handoff) == sha256(handoff)
state.update({
    "status": "complete",
    "run_directory": str(run_dirs[0]),
    "run_archive": str(archive),
    "handoff": str(handoff),
    "handoff_sha256": sha256(handoff),
    "drive_handoff": str(drive_handoff),
    "drive_handoff_sha256": sha256(drive_handoff),
    "comparison": comparison,
    "macro_observed_percent": sum(observed.values()) / len(observed),
    "macro_paper_percent": sum(paper.values()) / len(paper),
    "mean_absolute_difference_pp": sum(abs(observed[name] - paper[name]) for name in evaluated) / len(evaluated),
})
state_path.write_text(json.dumps(state, indent=2) + "\n")
drive_state = drive_dir / state_path.name
shutil.copyfile(state_path, drive_state)
assert sha256(drive_state) == sha256(state_path)
print(json.dumps(state, indent=2), flush=True)
print("SCANOBJECTNN_SEVERITY1_PROFILE_COMPLETED", flush=True)
