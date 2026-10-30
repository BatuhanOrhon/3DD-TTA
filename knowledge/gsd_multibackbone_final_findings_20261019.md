# Multibackbone study: final findings and closed-runtime handoff

## Agent A fixed ModelNet readouts integrated — 2026-10-30

[Run/Inference] CompletedOct25 all15/2468-ID/seed0 three-head observation:
probability mean exceeds own terminals by+0.7347/+0.9130/+0.8860pp, yet
remains0.9887–1.0778pp below historical separate original3DD. Background
regressions and other negative/zero rows are retained. No isolated spectral,
seed-stability, speed or novelty claim. Agent A closed with permissionOct25;
Agent B's subsequent studies/closure below remain unchanged.
[Full results, protocol and compact evidence](modelnet_multibackbone_trajectory_handoff_20261025.md).


**2026-10-25 next-scope addendum [User report/Code/Open]:** the user now selects
ModelNet40-C three-head minimum-entropy/probability-mean completion as the
next session's task. These6 readouts remain missing; historical accepted
terminal/Scan results below are unchanged. Current observer/analyzer are
Scan15-class-specific and terminal archives lack candidate logits. See the
[next-agent handoff](modelnet_multibackbone_trajectory_handoff_20261025.md).
No new implementation or Colab run has started.

**2026-10-25 attribution addendum [Paper/Inference/Open]:** trajectory averaging
and entropy-based candidate selection have relevant prior literature; the
earlier local user proposal is not a first-invention claim. Cite the primary
works and distinguish their tasks/operators from this pipeline using the
[attribution review](gsd_trajectory_attribution_20261025.md). This changes no
accuracy, source identity or runtime closure evidence reported below.

## Status and evidence boundary — 2026-10-19 20:44 UTC

[User report] The user requested a fresh check that no Colab runtime or task
background terminal remains, followed by a detailed knowledge update. This is
an archival/interpretation task; no new experiment or allocation was requested.

[Run] At 20:44 UTC, bounded `REQUEST_TIMEOUT=20 colab sessions` returned exit0
and `No active sessions found on server.` Local process inspection found no
Colab CLI worker, runtime monitor, multibackbone evaluation worker or
`caffeinate` process. The known former task PIDs34633/63286/70119/83891/85667
are absent. One pre-existing interactive zsh on `ttys001` has no child
processes and was preserved; it is not an active research background job.
No process termination or VM allocation was needed in this check.

[Code] The audit starts from `gsd-multibackbone-study` at
`422cb3c7d3b8daeaac3877a26e303df84ac91c86`. Pre-existing local edits to the
Colab skill/setup note and monitor tests/scripts are outside this documentation
batch and remain protected. This local HEAD is not the executed remote source.

[Run] Fresh local verification checks23 ZIP identities against the accepted
SHA-256 records, ZIP CRC and safe paths. The six-case matrix is recalculated
from90 per-corruption count rows; the Scan trajectory table is recalculated
from180 count rows. All18 method accuracy values and12 readout values agree.
This is a recheck of archived evidence, not new GPU accuracy or a fresh Drive
read. Prior actual Drive/local verification remains recorded below.

## Completed scope and comparison units

[Run] All explicitly requested cases are covered:

- DGCNN, CurveNet and PointNeXt on ModelNet40-C and ScanObjectNN-C:
  source-only, original3DD-TTA and GSD-inspired latent spectral guidance
  terminal accuracy, yielding18 accuracy rows across six matched cases.
- ScanObjectNN-C only: CurveNet, PointNeXt, DGCNN and Point-MAE with terminal,
  minimum entropy and equal probability mean from the same GSD trajectory,
  yielding12 policy rows. Original3DD trajectory selection was explicitly
  excluded when the user selected the GSD-only pipeline.
- Actual Point-MAE boundary parity, source-only smokes, six full source rows,
  six clean controls, guidance/original smokes, strict loads and frozen-state
  checks were completed before accepting the relevant full evaluations.

[Run] All benchmark numbers below are severity5, seed0, equal-weight all15
corruption macro accuracies in percent. Deltas are absolute percentage points
(pp). ModelNet has2468 objects per corruption; Scan uses581 `main_split`
objects per corruption, never `main_split_nobg`. Each full method therefore
has37020 or8715 prediction events. The matrix joins411615 method-role events;
these are repeated objects and method outputs, not411615 independent objects.
The Scan trace contains139440 candidate predictions from135 shared batches.

## Source-only, original3DD and GSD terminal

[Run] The accepted historical terminal attempts remain unchanged. In
particular, the later Scan CurveNet/PointNeXt trajectory terminals do not
replace their earlier GSD values in this table.

| Dataset | Backbone | Source % | Original3DD % | GSD terminal % | Original−source pp | GSD−source pp | GSD−original pp |
|---|---|---:|---:|---:|---:|---:|---:|
| ModelNet40-C | CurveNet |64.011345|67.071853|65.186386|+3.060508|+1.175041|−1.885467|
| ModelNet40-C | DGCNN |58.797947|65.586170|63.833063*|+6.788223|+5.035116|−1.753106|
| ModelNet40-C | PointNeXt |65.831983|68.292815|66.472177|+2.460832|+0.640194|−1.820637|
| ScanObjectNN-C | CurveNet |52.071142|55.238095|55.008606|+3.166954|+2.937464|−0.229489|
| ScanObjectNN-C | DGCNN |32.782559|36.442915|36.075731|+3.660356|+3.293173|−0.367183|
| ScanObjectNN-C | PointNeXt |49.087780|49.684452|45.209409|+0.596672|−3.878371|−4.475043|

*ModelNet DGCNN GSD combines13 complete recovered corruption files with fresh
complete Occlusion and LiDAR files. All37020 events are covered, but this is
two-attempt coverage, not one uninterrupted run or exact RNG continuation.
The old13 final interrupted LION state hash is missing. All their
per-corruption classifier states were verified; this does not supply the
missing final LION proof. Old partial Occlusion910 events are excluded.

[Inference] Original3DD exceeds its own source row in all six cases. GSD
terminal exceeds source in five cases but is below original3DD in all six.
Thus the terminal results support neither a universal backbone improvement
nor superiority over original3DD. Scan PointNeXt is the clearest negative
transfer case. These are fixed-checkpoint, single-seed observations.

[Run] Per-corruption and event-level heterogeneity versus original3DD:

| Dataset/backbone | Improved/worsened/tied corruptions | Corrected/harmed events | Largest observed gain | Largest observed loss |
|---|---:|---:|---|---|
| ModelNet / CurveNet |4/11/0|2250/2948|Impulse +6.7261pp|RBF-inverse −5.2674pp|
| ModelNet / DGCNN |4/11/0|2460/3109|Impulse +6.2399pp|Rotation −6.3209pp|
| ModelNet / PointNeXt |3/12/0|2273/2947|Impulse +4.2950pp|Density −5.6726pp|
| Scan / CurveNet |5/9/1|524/544|Impulse +12.3924pp|Background −4.8193pp|
| Scan / DGCNN |5/10/0|635/667|Impulse +6.1962pp|Density and RBF-inverse −4.3029pp|
| Scan / PointNeXt |4/11/0|453/843|Impulse +6.5404pp|Density −11.8761pp|

These maxima are descriptive inspection of existing results, not a basis for
corruption-specific inference routing or selecting new hyperparameters.
Full counts are in `result/preparation/dgcnn_residual_20261019/complete_matrix/`.

## Scan same-trajectory selection and averaging

[Code/Run] Fixed candidates are predicted-clean outputs at reverse loop
counts3/5/8 for the10-step path and8/15/23 for Background's30-step path, plus
ordinary terminal. Each probe is decoded once and shared across four frozen
heads. Scan coordinates match terminal: rotate back, divide3.3885, inverse
the input bounding box, without output normalization. Minimum normalized
entropy uses temperature1 and ties favor terminal, then the earliest probe.
Probability mean equally averages the four class-probability vectors.
Labels never enter selection; there is no threshold fitting or early stop.

| Backbone | Same-run terminal % | Minimum entropy % | Probability mean % | Entropy−terminal pp | Mean−terminal pp |
|---|---:|---:|---:|---:|---:|
| CurveNet |55.134825|56.408491|56.477338|+1.273666|+1.342513|
| PointNeXt |45.358577|46.769937|46.815835|+1.411360|+1.457258|
| DGCNN |36.075731|37.131383|37.326449|+1.055651|+1.250717|
| Point-MAE |34.125072|35.720023|35.685600|+1.594951|+1.560528|

[Run] Both policies improve all four same-run terminals in seed0. Probability
mean corrects/harms320/203,334/207,354/245 and258/122 events respectively;
minimum entropy corrects/harms364/253,364/241,379/287 and285/146. Probability
mean improves/worsens/ties12/1/2,11/3/1,12/3/0 and14/1/0 corruption rows.
Point-MAE minimum entropy exceeds its mean by only0.034423pp; this is not
evidence for selecting a different policy for each backbone after inspection.

[Run/Inference] Compared descriptively with the separately run original3DD:

| Scan backbone | Probability mean % | Original3DD % | Mean−original pp | Mean−source pp |
|---|---:|---:|---:|---:|
| CurveNet |56.477338|55.238095|+1.239243|+4.406196|
| DGCNN |37.326449|36.442915|+0.883534|+4.543890|
| PointNeXt |46.815835|49.684452|−2.868617|−2.271945|

The positive CurveNet/DGCNN differences are exploratory whole-policy results.
PointNeXt averaging recovers part of the GSD terminal loss but remains below
both source and original3DD. No full matched Point-MAE source/original3DD
Scan main_split rows were produced by this study; do not fill them using a
Gaussian-only, no-background or ModelNet result.

[Run/Inference] New CurveNet/PointNeXt terminals differ from their historical
r7 terminals by+0.126219/+0.149168pp. Same seed does not prove identical draws
or trajectories across processes. Prior native VAE replay established
nondeterminism; the new observation still passes RNG/buffer/live-input purity
and shared-geometry guards. Readout deltas must use their own terminal.
No online latency saving or early-stop speedup was measured.

## Why this comparison does not isolate spectral guidance

[Code/Run] Both methods use raw/eval LION with EMA off and a frozen classifier,
but their complete reconstruction policies differ:

| Setting | GSD-inspired guidance | Original3DD path |
|---|---|---|
| Batch |70|32|
| Reverse updates, normal/Background |10/30|5/35|
| Decoder style |updated|original|
| DDIM `set_alpha_to_one` |False|True|
| cuDNN benchmark |False|True|
| Guidance configuration |M800, beta2, spectral1.17, chamfer1|p.95, gamma/eta.01, SCD sum|

[Inference/Open] A spectral contribution requires the same GSD flow with
spectral weight0 and SCD/chamfer1, holding batch, horizon, scheduler, decoder,
assets, preprocessing and randomness fixed. That control has not been run
for the new backbones. The legacy Point-MAE configuration was transferred
unchanged; it is not demonstrated to be optimal for these heads. No target-label
hyperparameter search was performed to repair the observed negative rows.

## Source discrepancy and paper reproduction boundary

[Run] Accepted source measurements versus the user-supplied printed GSDTTA
macro targets are retained in the
[source audit](gsd_multibackbone_source_results_20261018.md):

| Dataset | Backbone | Measured % | Paper target % | Difference pp |
|---|---|---:|---:|---:|
| ModelNet40-C | DGCNN |58.797947|66.51|−7.712053|
| ModelNet40-C | CurveNet |64.011345|71.38|−7.368655|
| ModelNet40-C | PointNeXt |65.831983|66.99|−1.158017|
| ScanObjectNN-C | DGCNN |32.782559|55.72|−22.937441|
| ScanObjectNN-C | CurveNet |52.071142|53.06|−0.988858|
| ScanObjectNN-C | PointNeXt |49.087780|50.66|−1.572220|

[Run/Inference] Clean Scan accuracy is68.6747% DGCNN,77.4527% CurveNet and
92.0826% PointNeXt. DGCNN weakness therefore predates TTA. Numerical closeness
for Scan CurveNet/PointNeXt does not prove paper equivalence; ModelNet CurveNet
also has a substantial source gap and must not be described as reproduced.

[Code/Open] Exact GSDTTA checkpoint bytes, variant/FPS/normalization contract
and corruption bytes remain unavailable. Published alternatives include Scan
PointNeXt PB_T50_RS training and a CurveNet OBJ_ONLY trainer with its recorded
random sampling/L-infinity normalization; exact SVWA Scan training-log linkage
is unresolved. Class counts, labels, strict state loads and main_split
membership are verified, but they cannot identify the unique cause of each
paper gap. Preserve per-backbone preprocessing in the pinned provenance JSON.

[Run/Inference] The earlier Point-MAE/3DD Scan diagnostic does not establish a
severity-only explanation: severity1 Gaussian60.59% resembles the reported
59.0%, yet the full13 severity1 profile49.28% remains below the corresponding
reported69.0%. The no-background diagnostic only partly closes its Gaussian
gap. Those tests concern different assets/reference tables and are not a
controlled severity sweep for these three new heads. Do not silently lower
severity, replace main_split or claim all deficits are due to severity.

## Reproducibility, persistence and runtime cost

[Code/Run] Executed source is pinned separately from local orchestration:

- Original/residual v7 manifest:
  `78cdb53891bc0714dacd100eb220a5f6deb26bfe36f4e10b0aaa16f9aced60f9`;
  source ZIP `f82cf5052b8100591df679b159acec9f172144c5f6a0a916c5a5255cadeed4cf`.
- Scan observer implementation `746bc30`; v8 manifest:
  `7c3b55fe7cf179a63ee47c4c62a1a02cacd68372a645b534dd31f4d8e616806c`;
  source ZIP `f30c224d99e8d62263b55c47b0e6bfb6ac8e5575efcdae875a51aedce058948c`.
- Remote global checkout `dev@a458cd46bba1288a3f41302548bc777700cbc7d4`
  is not the local research branch. All325 executed v8 project source files
  match the declared implementation; exact checkpoint hashes are in the
  complete matrix validation and
  [asset provenance](gsd_multibackbone_provenance_20261018.json).

[Run] Results persist indexed IDs/labels/logits, standard configuration,
environment and logs, frozen-state/geometry/RNG guards and member manifests.
Completed batch shards, corruption ZIPs and final model ZIPs were saved to
Drive and physically checked; they are not exact RNG-resume snapshots or
copies of every reconstructed cloud/model weight. Scan readout archival adds
all four candidate logits per sample. No raw accepted attempt was overwritten.

| Accepted evidence | Local path under `result/preparation/` | Bytes | SHA-256 |
|---|---|---:|---|
| Scan analysis and physical receipts |`scan_trajectory_20261019/all15_accepted_evidence_drive_roundtrip.zip`|659325|`a85ca9c6e2925daf50e010b5ac9f688fcef99eab0300479bc9cb78c176852e97`|
| ModelNet DGCNN GSD composition |`dgcnn_residual_20261019/modelnet_gsd_composite_drive_roundtrip.zip`|5487419|`6298278eb5f0b0eab5c7adcbbacbf32a157b7d8ee03b94ce4049e0f9a47cd4ea`|
| Residual analysis and physical receipts |`dgcnn_residual_20261019/final_acceptance_drive_roundtrip.zip`|1105309|`227af6582cc8fa0371db71a261f3d6392e783a22e7e73b44804d7ec81ead693f`|
| Final pre-teardown report |`dgcnn_residual_20261019/completed_report_drive_roundtrip.zip`|238340|`22ca24428687516bc82ddf5d002b76480c22a83479bac32fcde223f02e27735f`|

Actual Drive root is
`MyDrive/thesis/result/preparation/gsd_multibackbone_study/`.
Exact subpaths/raw ZIP identities and physical receipts are documented in the
[Scan study](gsd_scan_trajectory_selection_20261019.md) and
[residual acceptance](gsd_multibackbone_dgcnn_residual_20261019.md).
The new20:44 audit rechecks local copies; it does not assert a new Drive upload.

[Run] Scan pilot+full GPU job wall is24.659908min, full alone23.565697min;
peak allocated memory35.484GiB. The last three DGCNN jobs took45.441790min
within the single3600s cap. These are phase/job wall times, not total study
cost, billing time or summed per-head reconstruction cost. Setup, losses,
previous runs and runtime lifetime cannot be inferred from these two totals.

[Run] Fresh immutable local audit:
`result/preparation/multibackbone_final_audit_20261019T204359Z.zip`,11731bytes,
SHA-256 `f277f3d56e7219ce8c5cc2c2c178efb25a479356d0df143677a60a0677102000`.
Its `audit.json`, count tables and manifest preserve the fresh server/process
check,23 local ZIP checks and count-based recomputation. This post-teardown
audit is local/Git-referenced; research results were verified on Drive before
shutdown. No new VM was created just to archive a documentation update.

## Remaining studies, decisions and falsifiers

[Open] Two concrete follow-ups were recorded but not executed:

1. Same-flow spectral0/SCD1 versus spectral1.17/SCD1, with all other settings
   fixed. This tests spectral attribution rather than the current whole-policy
   comparison. Paired input/noise evidence must be recorded; equal seeds alone
   cannot remove native nondeterminism.
2. Repeat the locked method/readout policy with additional seeds, initially1/2,
   reporting seed variation, per-corruption changes and cost. Existing
   single-seed event resampling cannot replace this replication.

[Open] Exact-paper reproduction is a separate asset/protocol identification
problem. A full uninterrupted ModelNet DGCNN GSD rerun could remove its
composite-state limitation, but full corruption coverage is already present;
the user's no-redundant-runs preference means this is not a mandatory missing
accuracy test. Point-MAE matched Scan source/original rows, original3DD
trajectory selection, three-head ModelNet trajectory selection and measured
online early stopping are outside the completed recent requested scope.
Broader historical ideas remain in the dated backlog, not new promised runs.

[Inference/Decision] Treat the current evidence as exploratory frozen-head
transfer plus label-free readout observations. Prioritize the matched
spectral-off control, then replication, if further experiments are requested.
Do not promote an optimum, a universal gain, full GSDTTA reproduction or a
spectral causal effect. No further GPU work is active or authorized by this
archival task; a future phase must use a new runtime and the setup/run gates.

Falsifiers: changed/mismatched checkpoint, split, labels, original IDs, source
snapshot, frozen state, ZIP identity or count recomputation would invalidate
the affected comparison. The20:44 local checks pass for the archived evidence;
unseen seed outcomes or same-flow controls may overturn any broader benefit
or attribution hypothesis. The scientific uncertainty remains open even
though the requested operational task and runtime cleanup are complete.
