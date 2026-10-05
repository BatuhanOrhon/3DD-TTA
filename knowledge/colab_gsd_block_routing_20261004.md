# Colab scenarios: local/style routing and conditional composition

[Implementation/Run handoff updated] 2026-10-05. The runner and analyzers are
on branch gsd-smooth-spectrum at commit 83e06b3 and pushed to origin. The scale
continuation flags described below are available at that ref; the remaining
Colab execution examples are runnable after pulling it. See
[design](gsd_block_routing_design_20261004.md) for rationale and settings.

[Run handoff] Routing attempt-0001 completed and is validated; see its
`result/modelnet40_c/gsd_guidance_composition_v1/routing/attempt-0001/validation.md`.
All route arms used fresh preparations: diagnose reuse was rejected by runtime,
native-extension and prepared-input identity differences. A conditional
projection pilot is justified by frequent nonterminal hard-v1 style conflicts.
It compares P_PC to P_SUM/P_NORM, but cannot establish gain versus ordinary
C_SCD because that matched practical control is unavailable.

[Run handoff updated] Projection attempt-0001 is complete and validated.
P_PC is slightly ahead of P_SUM/P_NORM in the four-corruption macro, but
effects are uncertain and paired intervals touch/include zero. The full
replication gate is unevaluated because matched C_SCD is missing. The
2026-10-05 [review and action plan](gsd_pilot_review_action_plan_20261005.md)
supersedes the earlier stop recommendation and proposes one four-arm control
block. Its runner support and executable cells follow below.
Actual prepared-input hashes differ
across routing/projection despite common draw keys, so only within-phase
pairing is accepted. See `projection/attempt-0001/validation.md`.

## Phase 4b - Matched projection control completion

[Code] The local runner now has an explicit `control_completion` phase. It
uses the fixed first64 indices, seed0, all four pilot corruptions, and the
ordered arms C_SCD/P_SUM/P_PC/P_NORM. It prepares each batch once and runs
all four arms from cloned shared inputs. There is no reference manifest and
no arm reuse. The planned size is16 arm archives /1,024 classifications.
Batch identities record SHA-256 values for input points, shape/local latents,
style conditioning, noise, timesteps, alpha-bar and scheduler configuration.

The Colab runtime must receive these three updated source files first:

- `gsd_paired_inputs.py`
- `gsd_composition_protocol.py`
- `scripts/run_gsd_composition.py`
- `scripts/analyze_gsd_composition.py`

### Cell 1 - Upload and install the updated runner files

Download the four linked source files from the local workspace, then select
all four when this cell opens the upload dialog. It writes only these named
runner files inside the repository checkout.

```python
from google.colab import files
from pathlib import Path

uploaded = files.upload()
required = {
    "gsd_paired_inputs.py": Path("/content/3DD-TTA/gsd_paired_inputs.py"),
    "gsd_composition_protocol.py": Path("/content/3DD-TTA/gsd_composition_protocol.py"),
    "run_gsd_composition.py": Path("/content/3DD-TTA/scripts/run_gsd_composition.py"),
    "analyze_gsd_composition.py": Path("/content/3DD-TTA/scripts/analyze_gsd_composition.py"),
}
missing = set(required) - set(uploaded)
if missing:
    raise RuntimeError(f"Missing uploaded source files: {sorted(missing)}")
for name, destination in required.items():
    destination.write_bytes(uploaded[name])
    print(f"Installed {name}: {destination}")
```

### Cell 2 - Plan-only preview (no model loading or GPU inference)

```python
import importlib
import os
os.chdir("/content/3DD-TTA")
import gsd_composition_protocol
importlib.reload(gsd_composition_protocol)
import scripts.run_gsd_composition as runner
importlib.reload(runner)

ROOT = "/content/3DD-TTA/result/modelnet40_c/gsd_guidance_composition_v1"
plan = runner.build_plan([
    "--phase", "control_completion",
    "--result-root", ROOT,
])
assert plan["status"] == "planned"
assert plan["corruptions"] == ["gaussian", "impulse", "background", "shear"]
assert plan["indices"] and len(plan["indices"]) == 64
assert plan["seeds"] == [0]
assert [arm["arm_id"] for arm in plan["arms"]] == [
    "C_SCD", "P_SUM", "P_PC", "P_NORM"
]
assert len(plan["blocks"]) == 4
assert sum(len(block["arm_ids"]) * len(block["indices"]) for block in plan["blocks"]) == 1024
print("Preview valid:", len(plan["blocks"]), "blocks,",
      len(plan["arms"]), "arms, 1,024 classifications")
print("Commit:", plan["resolved_ref"]["commit"])
print("Source hashes (these identify the uploaded working files):")
for name, identity in sorted(plan["resolved_ref"]["source_manifest"].items()):
    print(name, identity["sha256"])
```

Check that the preview lists exactly those four arms/corruptions and the
intended commit and uploaded-file hashes. The commit can still show the base
revision because these source files were uploaded as working-tree edits. The
manifest records both the commit and the source-file hashes. If correct, run
the next cell once.

### Cell 3 - Execute the matched control block

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
ROOT="result/modelnet40_c/gsd_guidance_composition_v1"
LOG_TMP="$(mktemp)"
set +e
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/run_gsd_composition.py \
  --phase control_completion \
  --result-root "$ROOT" \
  --execute 2>&1 | tee "$LOG_TMP"
RUN_STATUS=${PIPESTATUS[0]}
set -e
MANIFEST_PATH="$(sed -n 's/^Phase manifest: //p' "$LOG_TMP" | tail -n 1)"
if [[ -n "$MANIFEST_PATH" && -d "$(dirname "$MANIFEST_PATH")" ]]; then
  cp "$LOG_TMP" "$(dirname "$MANIFEST_PATH")/colab_console.log"
fi
rm -f "$LOG_TMP"
exit "$RUN_STATUS"
```

After execution, download the entire newly created
`result/modelnet40_c/gsd_guidance_composition_v1/control_completion/attempt-NNNN/`
directory as a ZIP, including `phase_manifest.json`, all arm ZIPs and Colab
console output. A partial or failed block is not an accuracy comparison.

### Cell 4 - Validate and summarize the new attempt

Run after Cell3, including when the phase manifest is partial, to produce a
validation report that distinguishes complete and failed arms.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
ROOT="result/modelnet40_c/gsd_guidance_composition_v1/control_completion"
ATTEMPT="$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -name 'attempt-*' | sort | tail -n 1)"
test -n "$ATTEMPT"
python scripts/analyze_gsd_composition.py \
  --result-root "$ATTEMPT/arms" \
  --output "$ATTEMPT/analysis"
python - "$ATTEMPT/analysis/analysis.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as source:
    report = json.load(source)
print("Bundle validation:", report["validation"])
for phase, arms in report["accuracy"].items():
    for arm, row in arms.items():
        print(phase, arm, row["macro_accuracy"], row["by_corruption"])
PY
```

### Cell 5 - Download the complete attempt

```python
from google.colab import files
from pathlib import Path
import shutil

phase_root = Path("/content/3DD-TTA/result/modelnet40_c/gsd_guidance_composition_v1/control_completion")
attempts = sorted(phase_root.glob("attempt-*"))
if not attempts:
    raise FileNotFoundError("No control_completion attempt found")
attempt = attempts[-1]
archive = Path(shutil.make_archive(
    str(Path("/content") / (attempt.name + "_control_completion")),
    "zip", root_dir=attempt.parent, base_dir=attempt.name))
print("Downloading", archive, "from", attempt)
files.download(str(archive))
```

## Fixed conditions and sample manifests

- Method: `gsd_guidance_composition_v1`; hard v1, weight1, M100, k10,
  delta .1, graph gamma .6; SCD weight1, lambda .95, gamma=eta=.01.
- ModelNet40-C severity5, raw/eval LION, EMA off, frozen Point-MAE, batch32.
- DDIM grid100; actual reverse steps5 normally and35 for Background.
- Original encoded global latent at final decode; global prior off.
- Main diagnostic corruptions: Gaussian, Impulse, Background, Shear.
  First two connect to existing probes; Background tests the SCD exception;
  Shear checks a geometric distortion. Report all four separately.
- Manifest splitter: `random.Random(20261004).shuffle(list(range(2468)))`.
  Diagnose/routing/projection use the first64 indices; replication the next256.
  Freeze the lists/hash once; retain original indices in every output.
  Use the same index lists across corruptions. Verify object correspondence
  before calling them common objects; labels alone do not prove correspondence.
- Model draw seeds: pilot0; replication0/1/2. Seed is not a substitute for
  shared actual preparation/noise. Phase reuse requires identical source,
  environment/native hashes, configuration, indices, prepared tensors and
  random-draw fingerprints. A missing match prevents a paired claim.
- These datasets were already inspected in historical full evaluations.
  The replication subset is disjoint from this pilot, but is **not a pristine
  held-out benchmark**. All selection here is development research.

## Named arms

`local / style` specifies the two composed gradient directions. Every arm
still runs local DDIM; `off` disables guidance on that block, not diffusion.
Diagnostic probes may evaluate a loss even when its update is masked.

| ID | Local | Style | Question |
|---|---|---|---|
| C_SCD | scd | scd | Existing SCD behavior in matched new host |
| C_SPEC | spectral | spectral | Existing v1-only behavior in matched host |
| C_SUM | sum | sum | Existing additive behavior in matched host |
| R_S0 | scd | off | Is SCD style adaptation useful at all? |
| R_SG | scd | spectral | User's proposed local-SCD/style-spectral routing |
| R_G0 | spectral | off | Style-off anchor for reverse routing |
| R_GS | spectral | scd | Reverse assignment |
| P_SUM | scd | sum | Same local route as style projection |
| P_PC | scd | pcgrad | Per-example symmetric style projection, norm-capped |
| P_NORM | scd | sum_norm_pcgrad | Sum direction at its own state's hypothetical capped-PCGrad norm |
| P_SCD_PRIORITY | scd | scd_priority | Optional one-way interpretation control |

P_PC's cap and P_NORM's zero cases are specified in the design. Never compare
P_PC only with C_SUM and attribute the difference entirely to projection;
the local route differs. Inference routing uses no ground-truth labels.
P_NORM matches candidate norms at a common state within that arm; diverged
P_PC/P_NORM trajectories need not have equal actual norm sequences. Log both.

## Phase 0 - Technical smoke

- Corruptions Gaussian/Background, first32 manifest indices, seed0.
- Run C_SCD/C_SPEC/C_SUM: 3 arms x 2 corruptions x32 =192 classifications.
- Instrument and compare each with the corresponding old sampler using the
  same prepared states/draws; these are technical parity checks, not accuracy
  promotion. Include a small routing/projection dry integration fixture on GPU
  if the native adapter cannot be exercised on CPU.
- Accept only finite losses/gradients/states, expected connected paths,
  actual5/35 steps, frozen/eval model modes, prediction/count agreement,
  and prepared-input/arm-order replay checks.
- Same-runtime identical-mode replay should match predictions; if nondeterministic
  CUDA behavior exists, record maximum tensor difference and prediction flips,
  investigate before claiming small effects. Do not silently widen tolerance.
- Export raw smoke ZIPs even when failing; stop later phases on technical failure.

## Phase 1 - Missing hard-v1 diagnostics and complementarity

- Four corruptions,64 examples each, seed0, C_SCD/C_SPEC/C_SUM.
- 3 arm runs covering4 corruptions each: 768 classifications, no full-file run.
- Record both objective gradients along each trajectory; compare their geometry
  only at the same sample/state/timestep within a trajectory. C_SCD versus
  C_SPEC state distributions are separate, not pairwise gradient measurements.
- All actual timesteps are logged. Report first/middle/last plus full step
  distributions; final style update is marked unused by later computation.
- Save logits and paired prediction transitions. Quantify complementarity
  and oracle gap; no oracle selection in deployed inference.
- Gate: technical validity; diagnose whether v1 style conflicts exist, whether
  spectral style contribution is effectively tiny, and whether predictions
  differ usefully. Routing can proceed without conflicts; PCGrad needs them.

## Phase 2 - GSD scale pilot (separate from routing)

- This phase tests whether low applied spectral magnitude limits its accuracy
  effect. It changes one scalar coefficient only; no routing, projection,
  smoothing or reduction changes are allowed in these runs.
- Same four corruptions,64 examples,seed0. Run SCD + v1 spectral weights
  `0,1,100,1000` on the same prepared state/noise: four arms x four x64 =
  1,024 predictions. Keep lambda .95, SCD coefficient1 and all settings fixed.
- Weight0 is the matched SCD-only control. Do not call 100/1000 calibrated optima.
- Report per corruption/sample/block/timestep spectral/SCD norm ratios,
  guidance/DDIM displacement ratios, prediction transitions versus weight0,
  finite-state checks, geometry change and accuracy change separately.
- Flag 1000 unstable if nonfinite or >2x guidance/DDIM on >10% nonterminal
  example-steps; do not replicate it. These are a predeclared safety gate,
  not an accuracy-derived threshold.
- Positive pilot evidence nominates at most one non-unstable scale for seeds
  0/1/2 and next256 indices, with weight0 and weight1 controls: 9 arm/seed
  runs x4x256 =9,216 predictions. Require positive mean paired change over0;
  preserve per-corruption deltas. This is development, not confirmation.
  Do not launch all15 automatically.
- If stronger guidance changes predictions but not accuracy, gradient
  smallness alone is insufficient. Do not combine this phase with routing.

### Continue an interrupted scale pilot

[Code] After `scale/attempt-0001` stopped at Background/SCALE_100, do not rerun
its completed cells. The CLI now accepts scale-only corruption/arm filters and
`--continue-on-arm-error`; each invocation creates a fresh attempt, records
failed arms (and any completed-batch partial ZIP), then proceeds through the
selected matrix. A handled arm failure leaves the phase manifest status
`partial`, even when remaining selected cells finish.

From `/content/3DD-TTA`, after pulling the updated branch, run these as separate
Colab cells. They use the same seed0 indices and preparation keys:

```bash
conda run -n 3dd_tta_env python scripts/run_gsd_composition.py \
  --phase scale \
  --result-root result/modelnet40_c/gsd_guidance_composition_v1 \
  --reference-manifest result/modelnet40_c/gsd_guidance_composition_v1/diagnose/attempt-0001/phase_manifest.json \
  --scale-corruptions background \
  --scale-arms SCALE_1000 \
  --continue-on-arm-error --execute
```

```bash
conda run -n 3dd_tta_env python scripts/run_gsd_composition.py \
  --phase scale \
  --result-root result/modelnet40_c/gsd_guidance_composition_v1 \
  --reference-manifest result/modelnet40_c/gsd_guidance_composition_v1/diagnose/attempt-0001/phase_manifest.json \
  --scale-corruptions shear \
  --scale-arms SCALE_0 SCALE_1 SCALE_100 SCALE_1000 \
  --continue-on-arm-error --execute
```

The failed Background/SCALE_100 observation stays failed; do not rerun it or
change its coefficient. Check each new `phase_manifest.json` and retain any
per-arm failure bundle. Cross-attempt pairing/aggregation is allowed only
after comparing prepared-input hashes, asset/runtime identities, indices and
the unchanged inference-source hashes. A new runner commit alone does not
prove compatibility. These are still seed0 development cells, not scale
replication.

## Phase 3 - Routing pilot

- Same64 examples/corruption, seed0; add R_S0/R_SG/R_G0/R_GS.
- 4 new arms x4x64 =1,024 classifications. Reuse the three Phase1 controls
  only under exact matched manifest/runtime/prepared-draw checks.
- Primary contrasts:
  - `C_SCD - R_S0`: SCD style contribution with local SCD fixed.
  - `R_SG - R_S0`: spectral style contribution with local SCD fixed.
  - `R_SG - C_SCD`: replacing SCD style with spectral style.
  - `R_GS - R_G0`: SCD style contribution with local spectral fixed.
  - `R_GS - C_SPEC`: replacing spectral style with SCD style.
  - Routed arms versus C_SUM: whole-system practical comparison.
- No global-prior or decoder change. If R_SG matches R_S0, don't claim spectral
  style carries useful information merely because both beat C_SCD.
- Small-sample pilot gate: record all outcomes; a routing candidate needs
  positive aggregate paired gain over C_SCD and its relevant style-off arm
  to be accuracy-promoted. Otherwise stop that accuracy hypothesis or propose
  the single separate scale-calibration follow-up; do not search a broad grid.
  A positive pilot is only a reason to replicate, not a thesis result.

## Phase 4 - Style projection pilot (completed, inconclusive)

- Run only if Phase1 shows negative style cosine on at least one valid,
  nonterminal update and the projected direction differs numerically.
- Same64/four-corruption/seed0 set; add P_SUM/P_PC/P_NORM.
- 3 new arms x4x64 =768 classifications. Reuse a comparator only after exact
  runtime/prepared-input validation; diagnose controls currently fail that gate.
- Primary `P_PC-P_SUM` and `P_PC-P_NORM`; compare with C_SCD for practical gain.
  A gain reproduced by P_NORM supports scale change rather than direction repair.
- If projection never activates, report a no-op and omit this GPU phase.
- Add P_SCD_PRIORITY only to explain a nonzero symmetric result or an explicit
  SCD-versus-spectral trade-off; it is not part of the initial grid.
- Candidate gate: positive aggregate gain versus C_SCD and both P_SUM/P_NORM.
  Null/negative or scale-only results do not promote conflict repair.

### Executed Colab cell (archive for provenance; do not rerun)

This uses the completed routing manifest as the phase reference. It runs
P_SUM/P_PC/P_NORM sequentially as one paired suite: 3 arms x 4 corruptions
x64 =768 predictions. It does not reuse old diagnose controls or include a
matched C_SCD practical comparator.

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
ROOT="result/modelnet40_c/gsd_guidance_composition_v1"
REF="$ROOT/routing/attempt-0001/phase_manifest.json"
test -f "$REF"
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/run_gsd_composition.py \
  --phase projection \
  --result-root "$ROOT" \
  --reference-manifest "$REF" \
  --execute
```

## Phase 5 - Replicate one selected routing/projection mechanism

- Freeze a `selection_manifest.json` before this run: candidate ID, exact
  weights/modes, relevant comparators, prior artifacts and decision rationale.
- Next256 indices, same four corruptions, seeds0/1/2.
- Routing: candidate + C_SCD + corresponding style-off control, 9 arm/seed
  runs x4x256 =9,216 classifications. All three contrasts remain interpretable.
- Projection: P_PC + C_SCD + P_SUM + P_NORM, 12 arm/seed runs x4x256 =12,288
  classifications. Do not omit P_SUM when claiming a projection gain.
- If pilot candidates tie within one net correct prediction, prefer fewer
  objective backpropagations, then lower measured runtime; choose only one.
- Predeclared development promotion rule: mean candidate-minus-C_SCD >=.2 pp,
  all three seed deltas positive, and mean Background delta >=-1.0 pp.
  Its mechanism-specific contrast must also have positive three-seed mean
  (routing versus style-off; projection versus both P_SUM and P_NORM).
  These thresholds are exploratory practical gates, not statistical proofs
  or measured optima; change them only before inspecting the replication data.
- Report per-corruption paired deltas, disagreement/correctness transitions,
  means/sample SD and bootstrap interval with2,000 resamples over verified
  object IDs, grouping that object's corruptions and seed repeats. If object
  correspondence is unverified, use separate per-corruption intervals and
  omit a misleading pooled interval. Do not treat timestep rows as samples.

## Phase 6 - Optional full15 development evaluation

- Trigger only after Phase4 and result review; freeze candidate configuration.
- Canonical15 corruptions,2,468 examples each, seeds0/1/2.
- Candidate and C_SCD are the minimal matched performance comparison: six
  arm/seed runs if no fingerprint-compatible full controls exist. This is a
  genuinely new paired composition experiment; the user must see why new
  controls are needed before launch. Reuse compatible completed controls.
- Historical controls remain useful descriptively if exact matching fails;
  do not relabel them as paired or automatically rerun all archived variants.
- Full mechanism claims require their mechanism comparator at matching scope;
  otherwise limit the claim to candidate-versus-baseline performance.
- Report macro15, macro14 without Background, Background separately, and
  each corruption. Give runtime/memory and identity limitations. This is
  full-test-set development evaluation, not independent confirmation.

## Optional global diffusion study: separate after routing evidence

[Inference/Planning] A style-guidance gain does not establish a global-prior
gain. Before any global code, audit raw z/global2style/decoder representations,
global-prior checkpoint/prediction parametrization and scheduler; confirm
whether mixed prediction is handled by the model. Legacy dual code is only
a reference. Do not feed noisy z_t directly as clean local conditioning.

Start with **sequential frozen global diffuse-denoise**, then local diffusion.
Predeclare one shallow setting, global5 actual reverse steps on a100-step grid;
validate its betas/prediction conventions against the actual global prior.
Use global-step0 as an exact bypass. Share original encoding, local noise,
global noise and classification RNG across the applicable arms. Do not reencode
local h under a changed z in this first experiment; report this potential
conditioning mismatch explicitly. Local guidance SCD; manual style updates off
throughout, so added conditioning drift is isolated from manual guidance.

| Global arm | Local-prior condition | Final decode style |
|---|---|---|
| G00 | original z transformed once | original decoder style |
| G10 | globally denoised z transformed once | original decoder style |
| G01 | original z transformed once | denoised decoder-compatible style |
| G11 | globally denoised z transformed once | denoised decoder-compatible style |

Implement only after exact decoder contract is verified; do not guess whether
it expects raw or transformed style for a changed configuration. G00 shares
the R_S0 conceptual control but may need a new matched runtime block.
First64/four-corruption/seed0 screen has1,024 classifications; this is a new
mechanism study. A matched global-step0 replay must reproduce G00 exactly.
Only a stable benefit over G00 motivates seeds0/1/2 replication or deeper
global schedules. Negative/null results close this branch. A later guided
global or synchronized method requires a new differentiation/time-alignment
design; this plan does not authorize copying the legacy synchronized sampler.

## Colab handoff after a reviewed code ref is available

Use these two cells after the reviewed implementation has an accessible Git
ref. Do not use the planning commit as if it contained the runner. Cell 1
checks out that exact ref without discarding tracked edits; Cell 2 starts with
the technical smoke. The runner's default dataset/checkpoint paths match the
repository's expected Colab layout, but override them if the mounted assets
are elsewhere.

**Cell 1 responsibilities:** check the existing clone and conda environment;
fetch/select the supplied implementation ref without discarding tracked edits;
verify `scripts/run_gsd_composition.py` exists; record commit and environment;
perform smoke dry-run. Keep data/checkpoints in their configured locations.

**Cell 2 — technical smoke:**

```bash
%%bash
set -euo pipefail
cd /content/3DD-TTA
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/run_gsd_composition.py --phase smoke \
  --result-root result/modelnet40_c/gsd_guidance_composition_v1 --execute
```

Later phases use the same entry point with the preceding printed absolute
`--reference-manifest` path; replication/all15 additionally require the
reviewed `--selection-manifest`. The launcher must print the exact next
command with real generated filenames. It must not auto-advance phases.
Dry-run is obtained by omitting `--execute`. Analyzer contract:

```bash
conda run --no-capture-output -n 3dd_tta_env python \
  scripts/analyze_gsd_composition.py \
  --result-root result/modelnet40_c/gsd_guidance_composition_v1 \
  --output result/modelnet40_c/gsd_guidance_composition_v1/analysis
```

## Result delivery

After each executed phase, provide the complete run ZIPs and suite/selection
manifests, not only percentages/screenshots. Each arm has:

```text
command.txt, config.json, environment.txt, stdout.log,
summary.csv, per_corruption.csv, notes.md,
predictions.npz, gradient_diagnostics.jsonl.gz, experiment_manifest.json
```

Include the analyzer output and preserve failed attempts. Remove credentials
from command/environment logs before sharing. Follow `result/README.md` with
the new method-specific schema added by the implementing agent. Archive
validation and research interpretation precede any next-phase decision.
