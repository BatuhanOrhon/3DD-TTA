# Phase 5 P_PC Colab handoff — 2026-10-05

[Run/Inference] Completed and ingested on 2026-10-06 at `replicate/attempt-0001`:48 complete arms; all registered Phase5 development gates pass. See the [validation report](../result/modelnet40_c/gsd_guidance_composition_v1/replicate/attempt-0001/analysis_ingested/validation.md). Do not rerun the execution cell. Actual Colab commit is `86fb155`, not preparation ref `fa03d50`; all nine inference sources are identical between them and match archived hashes. Cells below remain the historical handoff. No automatic all15 evaluation follows.

[Code/Run] Prepared against `gsd-smooth-spectrum@fa03d5032a63cd2c283b2f07236f4201fc221de8`, the current local and `origin/gsd-smooth-spectrum` ref observed on 2026-10-05. The user-provided last-known ref was `24b641d`. The existing runner already supports `replicate`, requires its selection manifest, and validates that manifest. No runner/protocol code change was needed.

The frozen selection is `result/modelnet40_c/gsd_guidance_composition_v1/replicate/selection_manifest.json`, SHA-256 field `56a256da5b4c5d789f83ce8ea4230b8746ec4f9db242ad22597c436decceaa2d`. It selects P_PC and keeps C_SCD, P_SUM, P_NORM. The schema's source phase is `projection`, so Phase 5 references the complete `projection/attempt-0001/phase_manifest.json`; the separately validated `control_completion/attempt-0001` provides the final Phase 4 gate evidence. Selection rationale records both.

The plan must resolve to 4 arms × 4 corruptions × 256 indices × 3 seeds = 12 paired blocks, 48 arm ZIPs and 12,288 classifications. `build_plan` preview is pure planning; running the CLI without `--execute` writes a planned attempt manifest. Cell 2 therefore calls `build_plan` directly.

## Cell 1 — sync branch and check environment

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
git status --short
git diff --quiet || { echo 'Tracked edits present; inspect before updating.'; exit 2; }
git diff --cached --quiet || { echo 'Staged edits present; inspect before updating.'; exit 2; }
git fetch origin gsd-smooth-spectrum
git checkout gsd-smooth-spectrum
git pull --ff-only origin gsd-smooth-spectrum
EXPECTED_REF=fa03d5032a63cd2c283b2f07236f4201fc221de8
test "$(git rev-parse HEAD)" = "$EXPECTED_REF"
test -f scripts/run_gsd_composition.py
test -f scripts/analyze_gsd_composition.py
conda env list
conda run --no-capture-output -n 3dd_tta_env python - <<'PY'
import sys, numpy, torch, scipy, diffusers
import chamfer, knn_cuda, pointnet2_ops._ext
assert "3dd_tta_env" in sys.prefix, sys.prefix
assert torch.cuda.is_available(), "CUDA is not available"
print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("torch/cuda:", torch.__version__, torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("scipy/diffusers:", scipy.__version__, diffusers.__version__)
print("native imports: chamfer, knn_cuda, pointnet2_ops._ext OK")
PY
```

This is environment/import validation only; it does not construct a model or run inference. Keep NumPy, PyTorch and native extensions in `3dd_tta_env`.

## Cell 2 — write selection manifest and preview plan

The schema-generated JSON must print the frozen fingerprint below. The preview does not write an attempt directory or call CUDA/model code.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
conda run --no-capture-output -n 3dd_tta_env python - <<'PY'
import json
from pathlib import Path
from gsd_composition_protocol import make_selection_manifest, validate_selection_manifest
from scripts.run_gsd_composition import build_plan
selection = make_selection_manifest(
    "projection", "P_PC",
    rationale=(
        "Phase 4 candidate nominated after the complete matched control-completion block: "
        "result/modelnet40_c/gsd_guidance_composition_v1/control_completion/attempt-0001/phase_manifest.json "
        "and its analysis_ingested/validation.md. P_PC passed the registered positive-mean gates "
        "versus C_SCD, P_SUM, and P_NORM on the locked first64/seed0 pilot. The prior mechanism "
        "pilot is result/modelnet40_c/gsd_guidance_composition_v1/projection/attempt-0001/phase_manifest.json; "
        "it provides the required completed projection reference only. Phase 5 retains all three "
        "comparators and applies the original frozen thresholds. This is development replication, "
        "not thesis-level confirmation or automatic all15 promotion."
    ),
    relevant_comparators=["C_SCD", "P_SUM", "P_NORM"],
)
assert selection["selection_sha256"] == "56a256da5b4c5d789f83ce8ea4230b8746ec4f9db242ad22597c436decceaa2d"
path = Path("result/modelnet40_c/gsd_guidance_composition_v1/replicate/selection_manifest.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n")
validate_selection_manifest(selection)
root = Path("result/modelnet40_c/gsd_guidance_composition_v1")
plan = build_plan([
    "--phase", "replicate", "--result-root", str(root),
    "--selection-manifest", str(path),
    "--reference-manifest", str(root / "projection/attempt-0001/phase_manifest.json"),
])
arm_ids = [arm["arm_id"] for arm in plan["arms"]]
assert arm_ids == ["C_SCD", "P_SUM", "P_NORM", "P_PC"], arm_ids
assert plan["corruptions"] == ["gaussian", "impulse", "background", "shear"]
assert plan["seeds"] == [0, 1, 2]
assert len(plan["indices"]) == len(set(plan["indices"])) == 256
assert len(plan["blocks"]) == 12
assert all(b["arm_ids"] == arm_ids and b["indices"] == plan["indices"] for b in plan["blocks"])
count = sum(len(b["indices"]) * len(b["arm_ids"]) for b in plan["blocks"])
assert count == 12288 and not plan["execute_requested"]
assert plan["reference_manifest"]["phase"] == "projection"
assert plan["reference_manifest"]["status"] == "complete"
print("selection_sha256:", selection["selection_sha256"])
print("resolved ref:", plan["resolved_ref"])
print("reference: projection / complete")
print("corruptions:", plan["corruptions"], "seeds:", plan["seeds"])
print("indices:", len(plan["indices"]), "unique")
print("arms:", arm_ids, "paired blocks:", len(plan["blocks"]))
print("classifications:", count, "; no model/CUDA/inference call")
PY
```

Review the output before choosing to run Cell 3. The same four-arm batch is prepared once and cloned across arms by the existing runner.

## Cell 3 — GPU execution (only after review)

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
ROOT=result/modelnet40_c/gsd_guidance_composition_v1
N=1
while test -e "$ROOT/replicate/$(printf 'attempt-%04d' "$N")"; do N=$((N+1)); done
ATTEMPT_ID=$(printf 'attempt-%04d' "$N")
LOG="$ROOT/replicate/${ATTEMPT_ID}_colab_execution.log"
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/run_gsd_composition.py \
  --phase replicate \
  --result-root "$ROOT" \
  --selection-manifest "$ROOT/replicate/selection_manifest.json" \
  --reference-manifest "$ROOT/projection/attempt-0001/phase_manifest.json" \
  --execute 2>&1 | tee "$LOG"
cp "$LOG" "$ROOT/replicate/$ATTEMPT_ID/colab_execution.log"
```

Expected output is 12 paired blocks, 48 arm ZIPs, 12,288 classifications. If execution stops or an arm is partial/failed, preserve the attempt and logs; do not report a partial score as completed Phase 5 accuracy.

## Cell 4 — validate, analyze, package and download

Use this as a **normal Python Colab cell**, without `%%bash`. The validation and analyzer still run in `3dd_tta_env`; packaging and `files.download` run in the notebook kernel, where `google.colab` is available. Replace the attempt path if needed. The raw output is packaged and downloaded even if the audit finds a partial/failed attempt; analyzer runs only after a complete audit.

```python
from pathlib import Path
from shutil import copy2
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import hashlib
import subprocess

from google.colab import files

repo = Path("/content/3DD-TTA").resolve()
attempt = repo / "result/modelnet40_c/gsd_guidance_composition_v1/replicate/attempt-0001"
root = repo / "result/modelnet40_c/gsd_guidance_composition_v1"
assert attempt.is_dir(), attempt

log_source = root / "replicate" / (attempt.name + "_colab_execution.log")
log_copy = attempt / "colab_execution.log"
if log_source.is_file() and not log_copy.exists():
    copy2(log_source, log_copy)

audit_code = r'''
import hashlib, json, subprocess, sys
from pathlib import Path
from gsd_composition_protocol import build_split_manifest, validate_arm_bundle, validate_selection_manifest
p = Path(sys.argv[-1]).resolve()
m = json.loads((p / "phase_manifest.json").read_text())
assert m["phase"] == "replicate" and m["status"] == "complete" and m["execution_status"] == "complete"
assert m["resolved_ref"]["branch"] == "gsd-smooth-spectrum"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == m["resolved_ref"]["commit"]
s = validate_selection_manifest(m["selection_manifest"])
assert s["candidate_id"] == "P_PC" and s["relevant_comparators"] == ["C_SCD", "P_SUM", "P_NORM"]
assert s["selection_sha256"] == "56a256da5b4c5d789f83ce8ea4230b8746ec4f9db242ad22597c436decceaa2d"
cfg = m["config"]
assert (cfg["batch_size"], cfg["ddim_total_steps"], cfg["normal_reverse_steps"], cfg["background_reverse_steps"]) == (32, 100, 5, 35)
assert (cfg["gamma"], cfg["eta"], cfg["lambda"]) == (0.01, 0.01, 0.95)
assert cfg["graph"] == {"profile": "hard-v1", "k": 10, "delta": 0.1, "graph_gamma": 0.6, "modes": 100}
phase_arm_configs = {a["arm_id"]: a["config"] for a in m["arms"]}
indices = build_split_manifest()["replication_indices"]
corr = {"gaussian", "impulse", "background", "shear"}
arms = {"C_SCD", "P_SUM", "P_NORM", "P_PC"}
component_names = {"input_points", "shape_latent", "local_latent", "style_conditioning", "local_noise", "timesteps", "alpha_bar"}
source_names = {"scripts/run_gsd_composition.py", "gsd_composition_protocol.py", "gsd_composition.py", "gsd_paired_inputs.py", "tta_gsd.py", "tta.py", "graph_spectral.py", "run_baseline.py", "main_3dd_tta.py"}
assert set(m["resolved_ref"]["source_manifest"]) == source_names
assert m["asset_manifest"] and m["native_extension_manifest"]
blocks = m["paired_blocks"]
assert len(blocks) == 12
assert {(b["corruption"], b["seed"]) for b in blocks} == {(c, seed) for c in corr for seed in (0, 1, 2)}
reference_runtime = None
checked = 0
for b in blocks:
    assert b["status"] == "complete" and b["indices"] == indices and b["identity"]["indices"] == indices
    assert len(b["arms"]) == 4 and {r["arm_id"] for r in b["arms"]} == arms
    assert all(r["status"] == "complete" for r in b["arms"])
    manifests = []
    for r in b["arms"]:
        bundle = validate_arm_bundle(Path(r["zip_path"]))
        a = bundle["manifest"]
        assert a["phase"] == "replicate" and a["status"] == "complete"
        assert (a["arm_id"], a["corruption"], a["seed"], a["block_id"]) == (r["arm_id"], b["corruption"], b["seed"], b["block_id"])
        assert a["original_indices"] == indices and a["identity"] == b["identity"] == r["identity"]
        assert a["config"] == phase_arm_configs[r["arm_id"]]
        manifests.append(a)
        checked += 1
    assert all(a["identity"] == manifests[0]["identity"] for a in manifests)
    for a in manifests:
        assert a["runtime_identity"] == m["runtime_identity"]
        assert a["asset_manifest"] == m["asset_manifest"]
        assert a["native_extension_manifest"] == m["native_extension_manifest"]
        assert {"classifier_checkpoint", "classifier_config", "lion_checkpoint", "lion_config", "labels"}.issubset(a["asset_manifest"])
        runtime = (a["runtime_identity"], a["asset_manifest"], a["native_extension_manifest"])
        if reference_runtime is None:
            reference_runtime = runtime
        assert runtime == reference_runtime
        batches = a["identity"]["prepared_components"]
        assert len(batches) == 8 and [i for x in batches for i in x["indices"]] == indices
        assert all(set(x["components"]) == component_names for x in batches)
        assert all(len(v) == 64 for x in batches for v in x["components"].values())
        assert all(len(x["scheduler_config_sha256"]) == 64 for x in batches)
for name, digest in m["resolved_ref"]["source_manifest"].items():
    data = Path(name).read_bytes()
    assert len(data) == digest["bytes"] and hashlib.sha256(data).hexdigest() == digest["sha256"], name
assert checked == 48
print("validated:", checked, "complete arm ZIPs; 12 complete paired blocks")
print("prepared component hashes, runtime/assets/native identities and source hashes: OK")
'''

audit = subprocess.run(
    ["conda", "run", "--no-capture-output", "-n", "3dd_tta_env", "python", "-c", audit_code, str(attempt)],
    cwd=repo,
    check=False,
)
audit_status = audit.returncode
analysis_status = 0

if audit_status == 0:
    existing_analysis = attempt / "analysis" / "analysis.json"
    if existing_analysis.is_file():
        print("Keeping existing analysis:", existing_analysis)
    else:
        analysis = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", "3dd_tta_env", "python",
             "scripts/analyze_gsd_composition.py", "--result-root", str(attempt),
             "--output", str(attempt / "analysis")],
            cwd=repo,
            check=False,
        )
        analysis_status = analysis.returncode
else:
    print("Audit failed or the phase is partial; analysis skipped. Raw output will still be packaged.")

archive = attempt.with_suffix(".zip")
with ZipFile(archive, "w") as z:
    for item in sorted(attempt.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(attempt)
        if relative.parts[0] == "arms":
            if len(relative.parts) == 2 and item.suffix.lower() == ".zip":
                z.write(item, relative.as_posix(), compress_type=ZIP_STORED)
                continue
            arm_dir_name = relative.parts[1]
            if (attempt / "arms" / (arm_dir_name + ".zip")).is_file():
                continue  # the corresponding complete arm ZIP already contains these files
        z.write(item, relative.as_posix(), compress_type=ZIP_DEFLATED)

print("Downloading attempt package:", archive)
files.download(str(archive))
if audit_status != 0 or analysis_status != 0:
    print("Attempt downloaded, but it is not a validated complete Phase 5 result.")

A non-complete manifest or any failed audit means no complete paired accuracy report: retain and download the failed attempt for diagnosis. After local ingestion, calculate paired accuracy and seed deltas only from validated complete arms. Within each seed, calculate each arm's per-corruption accuracy over the same 256 indices, then take the equal-weight mean of the four corruption accuracies. For every seed report the paired macro deltas P_PC−C_SCD, P_PC−P_SUM and P_PC−P_NORM; report each three-seed delta mean and sample SD. Separately report P_PC−C_SCD on Background by seed and its three-seed mean. Apply the locked thresholds exactly: mean P_PC−C_SCD ≥ +0.2 pp; P_PC−C_SCD positive at each seed; mean Background delta ≥ −1.0 pp; and three-seed means P_PC−P_SUM and P_PC−P_NORM both positive. Report paired uncertainty per corruption only; cross-corruption object correspondence is unverified, so omit a pooled bootstrap. Timestep rows are not samples. No result automatically promotes P_PC to `all15`.

