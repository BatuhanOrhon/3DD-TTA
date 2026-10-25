# ScanObjectNN-C GSD-inspired trajectory selection — 2026-10-19

[User report/Paper/Open] 2026-10-25 reading follow-up: see the
[full-title paper list and primary links](gsd_trajectory_attribution_20261025.md#reading-list-requested-by-the-user--2026-10-25)
for timestep ensembling and entropy selection. The requested
[logit-mean versus probability-mean comparison](open_questions.md#2026-10-25--logit-mean-versus-probability-mean)
can reuse this study's saved candidate logits; it has not been evaluated.
The accepted terminal/minimum-entropy/probability-mean results below are unchanged.

Status: **complete and independently accepted** for the selected four-head Scan
GSD-only trajectory study. Historical preparation/running entries are retained.
Separate DGCNN original3DD/ModelNet completion was subsequently accepted at
18:33UTC; r9 and task monitors were closed at18:40UTC. A20:44UTC server/process
recheck confirms no active Colab session or task background worker. Earlier
pending/live-session paragraphs below are historical, superseded by the
[final synthesis](gsd_multibackbone_final_findings_20261019.md) and
[residual acceptance](gsd_multibackbone_dgcnn_residual_20261019.md).

The user selected **only the GSD-inspired guidance pipeline** for this study.
Evaluate CurveNet, PointNeXt, DGCNN and Point-MAE on severity5, seed0,
all15 corruptions of the 581-example ScanObjectNN-C `main_split`.
The primary comparison is each head's same-run terminal prediction.
Published alternative backbone checkpoints remain distinct from exact GSDTTA
paper reproduction. DGCNN's weak source checkpoint remains exploratory.

Fixed candidates: predicted-clean outputs at reverse loop counts3/5/8 of10,
or8/15/23 of30 for Background, plus the ordinary terminal output.
Decode each selected latent once, then apply each frozen head's established
preprocessing. Scan outputs must use the terminal coordinate transform:
rotate back, divide3.3885, inverse the input bounding box; no output normalization.
Policies: terminal, minimum normalized entropy (ties favor terminal then the
earliest checkpoint), equal class-probability mean at temperature1.
No labels, threshold fitting, early stopping, or extra diffusion per policy.
Existing Point-MAE ModelNet trajectory results do not establish Scan performance.

## Batch 1 — observer and isolated runner integration

- Goal: collect four-head, four-candidate logits without changing GSD updates.
- Scope: opt-in Scan trajectory mode, shared decoder observation, trusted exact
  Point-MAE checkpoint loading, indexed batch trace persistence, pure analysis.
- Touched areas: `scripts/eval_multibackbone_guidance.py`, a small shared observer,
  focused tests and a Scan trajectory analyzer. Default old modes stay unchanged.
- Stack context: local CPU checks; actual runtime Python3.8/PyTorch2.1.2/cu121.
- Dependencies: existing `tta_gsd_v2` hooks, `isolated_probe`, source preprocessing,
  immutable bundle and Drive shard helpers, class-count-independent `policies`.
- Implementation notes for development-agent: preserve classifier/LION state and
  RNG; preserve the live autograd graph; decode once per checkpoint; require all
  finite B×4×15 logits; preserve original IDs including last21 examples. Do not
  invent a full Point-MAE matched source row; its primary comparator is terminal.
- Verification: targeted CPU tests for purity, geometry transforms, default-off
  behavior, policy ties and complete indexed coverage; then genuine GPU smoke.
- Knowledge artifact to update: this note (agent implementation section).
- User review gate: selected scope is already authorized; automatically verify
  acceptance before the GPU phase, with explicit approval only if its complete
  expected duration exceeds60minutes.
- Out of scope: original3DD trajectory selection, hyperparameter tuning, completing
  the separate pending full DGCNN ModelNet/original3DD phase.

## Batch 2 — sealed source and GPU pilot

- Goal: validate actual four-head observations and estimate the whole GPU phase.
- Scope: fresh immutable source/asset manifest; existing r9 environment, no build;
  Gaussian and Background one-batch pilot, per-batch Drive receipts and SHA checks.
- Touched areas: preparation artifacts and this note; never mutate prepared_v7.
- Stack context: existing r9 A10080GB, same four pinned native binaries and LION.
- Dependencies: accepted Batch1 CPU checks and source provenance.
- Implementation notes: record the executed snapshot/hash; validate states/RNG,
  candidate counts and same terminal transforms; account for pilot plus fullrun.
- Verification: actual GPU shapes, purity, peak memory, finite losses/logits,
  independent Drive/local hash round trip and policy analyzer.
- Knowledge artifact to update: this note, findings_log, open_questions.
- User review gate: no extra approval under60minutes; if estimate exceeds60,
  request one comprehensive approval before the dependent full GPU phase.
- Out of scope: runtime replacement/reinstallation or other Colab sessions.

## Batch 3 — all15 evaluation and acceptance

- Goal: obtain12 accuracy rows (four heads × three selection policies).
- Scope: one shared full Scan diffusion trajectory, immutable candidate shards
  copied to Drive after every batch, per-head/corruption and final bundles.
- Touched areas: results, independent analysis, research records and commits.
- Stack context: fixed B70, normal10/Background30 reverse updates, seed0.
- Dependencies: accepted pilot and whole-phase duration/approval gate.
- Implementation notes: analyze all8715 original-ID examples per head, report
  all15 macro and paired terminal changes; save prefix on interruption.
- Verification: independent indexed predictions and SHA-256 checks, all135
  shared batches, four classifier and LION state guards, policy recomputation.
- Knowledge artifact to update: this note, findings_log, open_questions, README.
- User review gate: report concrete evidence and limitations; push accepted batches.
- Out of scope: single-seed generalization or a claim of full GSDTTA reproduction.

## Inventory and runtime baseline

CurveNet and PointNeXt source/original3DD/GSD comparisons are complete on both
datasets. DGCNN source-only is complete on both; its full guidance/original3DD
comparison remains incomplete. The four newly accepted DGCNN GPU smoke tests
are408 predictions and are not full accuracy rows. No validated four-head Scan
trajectory trace exists in the prior terminal-only archives.

The prior shared Scan CurveNet+PointNeXt GSD job took1004.737seconds (16.746min),
including924.644seconds shared reconstruction for135 batches. This is an
empirical baseline, not yet the new four-head observer duration estimate.
The live r9 session is reused. Remote checkout and sealed executed research
source have distinct provenance; see the DGCNN smoke note and runtime handoff.

## Shared observer and independent analyzer implementation — 2026-10-19

`shared_trajectory.SharedTrajectoryObserver` implements the existing GSD hooks
without editing GSD updates. Each selected pre-update clean latent is decoded
once under `isolated_probe`, then a callback performs the Scan terminal coordinate
transform and classifies that shared geometry with all selected frozen heads.
The callback does not receive labels. Three clean candidates and the already
computed terminal logits form finite B×4×15 arrays, with explicit candidate names
and loop counts. Missing/reordered checkpoints, mismatched heads, changed RNG,
buffers or live latent/style are rejected. Diagnostics preserve initial/final
hashes and each probe's RNG, buffer and decoded-geometry identities. Component
times are unsynchronized dispatch wall times; the runner owns GPU wall timing.

`scripts/analyze_scan_trajectory.py` independently verifies every sealed bundle
member's SHA-256/size and accepts the existing `run_id/` ZIP prefix. It requires
the declared GSD-inspired, severity5/seed0, 581-example `main_split`, 15-class,
B70 protocol and unchanged classifier/LION state guards. Full acceptance requires
all15 corruptions, all original IDs0–580 exactly once in order, and nine batches
per corruption including the final21 examples. Candidate logits must contain
the exact stored terminal logits. Policies reuse the existing class-agnostic
temperature1 implementation and report per-corruption/macro accuracy, changes
against same-run terminal, corrected and harmed predictions. `--allow-partial`
permits a clearly labelled complete-batch prefix for the pilot; it cannot create
a full accuracy row from an incomplete trace. No Point-MAE source-only result is
invented by this analysis.

Local verification: `uv run --offline --no-project --with torch --with numpy
python -m unittest tests.test_shared_trajectory
tests.test_legacy_trajectory_observer -v` passed all15 tests. The new tests cover
stochastic single-decoder observation, RNG/buffer restoration with a live BN
autograd graph, missing checkpoints, four-head/15-class policy ties, all135
synthetic batch records including remainder21, duplicate IDs, partial scope,
prefixed sealed ZIPs and member tampering. These are synthetic CPU checks;
actual r9 CUDA memory, model/checkpoint compatibility and full trajectory
results remain pending the real GPU pilot and evaluation.


## Sealed source and actual r9 CPU proof — 15:37UTC

[Code/Run] Implementation commit `746bc3024f26de67a3cab410c1e15cfbdfa2eb2a`
is pushed. A fresh `prepared_scan_trajectory_v8` preserves v7 and declares
manifest `7c3b55fe7cf179a63ee47c4c62a1a02cacd68372a645b534dd31f4d8e616806c`,
sourceZIP `f30c224d99e8d62263b55c47b0e6bfb6ac8e5575efcdae875a51aedce058948c`.
Existing diffusion/guidance/native consumer source bytes are unchanged.
Scan Point-MAE checkpoint351525407bytes SHA53f6c2e…b9535dac is now declared
with the three existing frozen Scan checkpoints. The actual r9 strict CPU
load passes all4 models, with no forward or CUDA initialization; the pinned
four-backend no-build bootstrap also passes. CPUproof wall14.703seconds.

Immutable CPUevidence51514bytes SHA
`c15bbe0ccd25663d760ae314b646b8efcfa15be29fffbae68262ece075506c9a`
and delta5487366bytes SHA
`a050f0dcf0cc360354d6f0d1b2504abde7499d6d0efecfdc138a21e16005e008`
pass actual Drive/local roundtrip. Local artifacts:
`result/preparation/scan_trajectory_20261019/`; actual Drive prefix:
`MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261019T153709Z_scan_trajectory_v8/`.

[Run/Open] Pilot started15:37:49UTC in r9, shared Gaussian+Background first70
IDs, four heads and four candidates. Phase
`/content/gsd_multibackbone_study_20261018/gpu_phases/20261019T153749Z_scan_trajectory_pilot`.
Controller SHA27b7e74a…2d699e7; actual GPU result acceptance remains pending.
Planned whole new GPU phase25–45min based on the16.75min prior two-head job;
pilot recalibrates before fullrun. One shared3600s job-wall cap includes pilot
and fullrun, never a per-call loophole. No long DGCNN completion launch.


## Actual four-head GPU pilot accepted — 15:41UTC

[Run] Pilot completed15:38:54UTC:65.652680296seconds, two shared70-example
batches,560 terminal predictions and2240 four-candidate predictions. Gaussian
shared reconstruction9.959955540s; Background16.086486018s. Peakallocated
36335.369MiB (35.484GiB) on A10080GB. Allfour strict classifier states and
LION states remain equal; each probe preserves RNG, modelbuffers and live
latent/style bytes. The four heads share identical indexed trajectory hashes.
Original main_split labels match the declared local dataset; all15-class
candidate shapes, terminal equality, policy recomputation and sealed manifests
pass. Prefix140examples/head is smoke evidence, not benchmark accuracy.

All4rawZIPs are under `result/scanobjectnn_c/gsd_inspired_latent_spectral_guidance/`:
`20261019-153750_curvenet-seed0.zip` SHA153e1b9b…9906150;
`20261019-153751_pointnext-seed0.zip` SHA10d4b0a0…c1e0c91;
`20261019-153752_dgcnn-seed0.zip` SHA967db8cb…730c1b9;
`20261019-153753_pointmae-seed0.zip` SHAbc268fa2…ed94f2.
Exact full identities are in `result/preparation/scan_trajectory_20261019/gpu_pilot_receipt.json`.
Allfour are downloaded from their actual Drive finalZIP and hash-verified.

Pilot phase evidence48595bytes SHA380a8046…1fcb036b verifies Drive/locally.
Accepted analysis/physical-receipt binder37464bytes SHA
`182a8f2415a3b4bbd3b5b04d07c70733c7dc37b2401d5ef2dceac47e02e9cdd9`
verifies actual Drive/local roundtrip and every sealed member, including
8physical batch receipts,8corruption receipts and4final receipts.
Local `pilot_accepted_evidence_drive_roundtrip.zip`; Drive
`MyDrive/thesis/result/preparation/gsd_multibackbone_study/scan_trajectory_pilot_accepted_20261019T154120Z/`.

[Inference/Run] Whole-phase estimate includes pilot,126normal+9Background
batches: central1795.385s (29.923min), conservative2550.279s (42.505min).
Upper adds40percent compute margin,3s/batch persistence and120s fixed overhead.
Thus expected duration stays under60min; no new long-run approval required.
The shared3600s job-wall cap still includes pilot and fullrun.

Full all15 started15:41:44UTC in the same r9 session, source/manifest unchanged.
Phase `/content/gsd_multibackbone_study_20261018/gpu_phases/20261019T154144Z_scan_trajectory_all15`;
controllerSHA94c512da…a80dba. Expected135shared batches,8715examples/head,
139440candidate predictions. Batch receipts copy allcandidates to Drive,
followed by per-corruption and per-head final immutable ZIPs.
No full accuracy claim until independent local acceptance.

## Full all15 results accepted — 16:07UTC

[Run/Artifact] Severity5, seed0, ScanObjectNN-C main_split581×15:8715
original-ID prediction events per head. Terminal and both fixed selection
policies use the exact same four candidates in this run.

| Backbone | Terminal % | Minimum entropy % | Probability mean % | Entropy delta pp | Mean delta pp |
|---|---:|---:|---:|---:|---:|
| CurveNet |55.134825|56.408491|56.477338|+1.273666|+1.342513|
| PointNeXt |45.358577|46.769937|46.815835|+1.411360|+1.457258|
| DGCNN |36.075731|37.131383|37.326449|+1.055651|+1.250717|
| Point-MAE |34.125072|35.720023|35.685600|+1.594951|+1.560528|

Both policies exceed terminal in all four heads for this seed. Probability
mean is highest for CurveNet, PointNeXt and DGCNN; minimum entropy is highest
for Point-MAE by0.034423pp. This ranking is descriptive, not a post-hoc policy
recommendation or a seed-stability claim.

Probability mean corrects/harms320/203 CurveNet terminal events (net+117),
334/207 PointNeXt (+127),354/245 DGCNN (+109),258/122 Point-MAE (+136).
Minimum entropy corrects/harms364/253,364/241,379/287,285/146
(net+111,+123,+92,+139 respectively). Repeated objects across corruptions
are not independent binomial samples. Probability mean improves/worsens/ties
12/1/2 CurveNet corruptions,11/3/1 PointNeXt,12/3/0 DGCNN,14/1/0 Point-MAE;
minimum entropy counts10/4/1,12/3/0,10/4/1,14/1/0 respectively.

### Matched references and limits

[Run/Inference] Source remains52.071142% CurveNet,49.087780% PointNeXt and
32.782559% DGCNN. Probability mean changes these by+4.406196pp,-2.271945pp
and+4.543890pp. DGCNN remains an exploratory weak published alternative,
not an exact paper checkpoint. No complete Point-MAE source/original3DD
main_split row was supplied by this study; its accepted comparison here
is solely the same-run terminal.

CurveNet probability mean56.477338% is1.239243pp above the prior accepted
original3DD55.238095%; PointNeXt46.815835% is2.868617pp below its original
49.684452%. These are different reconstruction-policy runs/batches/schedules;
this does not isolate spectral effects. No original3DD trajectory selection
was run, following the user's GSD-only scope.

New CurveNet/PointNeXt terminals55.134825/45.358577 differ from prior separate
r7 terminals55.008606/45.209409 by+0.126219/+0.149168pp. Same seed does not
establish bitwise trajectory equality across processes; prior native encoder
replay already demonstrated nondeterminism. Probe RNG, buffers and live inputs
are unchanged in this run. Use new terminals for selection deltas; preserve
prior raw results and never substitute the new attempt into their old trace.
All325 executed project source files match implementation746bc30 exactly;
none differ or are undeclared in that commit.

DGCNN Scan GSD terminal is now a complete accepted row36.075731%,+3.293173pp
above its own source. DGCNN original3DD on both datasets and ModelNet GSD
Occlusion/LiDAR completion remain pending. The old59386-event proposal includes
the now-covered8715Scan GSD events; do not run that proposal unchanged.
Residual50671 prediction events need a freshly reviewed scope and duration
estimate. No further GPU phase was activated.

### Runtime and immutable evidence

Full GPU job15:41:44–16:05:18UTC:1413.941804s (23.565697min). Shared
reconstruction1227.026896s includes observer work and is counted once across
heads. Pilot+full GPU job wall1479.594484s (24.659908min), below the whole-phase
42.505min upper estimate and3600s cap. Peak allocated36335.369MiB (35.484GiB);
135 shared batches,540 head-batch traces,139440 candidate predictions.
No early stopping or pure inference speedup is measured.

All frozen classifier/LION states, checkpoint/native/source identities, probe
RNG/buffers/live latent-style preservation, finite15-class logits, per-corruption
0..580 indices and labels, candidate timing/order and standard terminal
CSV/aggregate NPZ/batch consistency independently pass. The four heads share
identical batch trajectory and terminal geometry hashes. All seven standard
files and complete bundle-member SHA manifests pass, including ZIP CRC.

Local raw ZIPs under `result/scanobjectnn_c/gsd_inspired_latent_spectral_guidance/`:

| Run ID | Bytes | SHA-256 |
|---|---:|---|
|20261019-154146_curvenet-seed0|10204604|`acb85642c3635275512cfca70a2da80f758c0f559b96641a9004a71c31805d14`|
|20261019-154147_pointnext-seed0|10049066|`07ea076d8cff7da65cc0002364ed9c9ac50409f298b25865cf1617512e1826de`|
|20261019-154148_dgcnn-seed0|10189173|`c50efad82090e8c777e219b3647defffa294f30ff1b0dbcd6317ff343dc2be58`|
|20261019-154148_pointmae-seed0|10166160|`62bbbaa47a4ad6517177f315929bd677ada546d63dd14b9c0337e78bd731db64`|

Drive raw root:
`MyDrive/thesis/result/gsd_multibackbone_study/scanobjectnn_c/gsd_inspired_latent_spectral_guidance/<run_id>/final.zip`.
Each is physically Drive-verified and downloaded to canonical local storage
with identical size/SHA.540 batch,60 corruption and4 final physical Drive
receipts were independently audited and preserved.

Full-phase evidence153027bytes SHA
`2090393e53395c21d764e00de1ce86da84bfdcaee1a27c977dbdd603f32725e5`;
local `result/preparation/scan_trajectory_20261019/all15_phase_evidence_drive_roundtrip.zip`.
Final analysis/physical-receipt binder659325bytes SHA
`a85ca9c6e2925daf50e010b5ac9f688fcef99eab0300479bc9cb78c176852e97`;
local `all15_accepted_evidence_drive_roundtrip.zip` in the same preparation root;
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/scan_trajectory_all15_accepted_20261019T160659Z/scan_trajectory_all15_accepted_20261019T160659Z.zip`.
Nested local analysis package130110bytes SHA
`ed60fbc1b1be633218be9c1af263692f2fa399fbf240b194ac16beb3480d01de`.
Actual Drive/local roundtrip, all member manifests and nested analysis pass.

Derived `all15_analysis/{validation.json,macro.csv,per_corruption.csv}` and
`matched_readout_comparison.csv` retain exact values and paired errors.
Final report backup stays separate from raw/accepted evidence.

The same `tta-multibackbone-dgcnn-a100-r9-20261019` session remains preserved;
no stop/release/teardown was issued. Monitor70119/local session63241 remains
sole execution owner. Bounded host awake assertion PID79381/session29840
expires7200seconds after15:38UTC; managed Colab lifetime is not guaranteed.

## Final cross-study interpretation and closure — 20:44 UTC

[Run] The subsequently accepted Scan DGCNN original3DD row is36.442915%.
The existing same-run GSD minimum-entropy37.131383% and probability-mean
37.326449% therefore exceed that separate original run by0.688468pp and
0.883534pp respectively. No trajectory was rerun to obtain these differences.
The historical `matched_readout_comparison.csv` was sealed before the DGCNN
original completed and contains blank original fields for that head; preserve
that artifact. The final synthesis joins its accepted trace with the later
original result explicitly rather than silently modifying the old CSV.

[Inference] Probability mean exceeds original3DD for CurveNet and DGCNN but
remains below it for PointNeXt. Both fixed readouts improve all four own
terminals in seed0; this does not isolate spectral guidance, prove repeatable
gains or supply missing full Point-MAE source/original Scan rows. Preserve the
new CN/PN terminal values solely as the comparator for their same-run readouts.

[Run/Open] Fresh local count-based verification reproduces all12 policy rows
from180 corruption rows. The accepted659325byte binder retains its recorded
SHA. Server inventory is empty, all task monitors/evaluation/host-awake workers
are absent, and no extra trajectory or seed test was launched. Same-flow
spectral-off controls and additional seeds remain unexecuted future studies.
[Detailed findings, evidence and remaining gates](gsd_multibackbone_final_findings_20261019.md).
