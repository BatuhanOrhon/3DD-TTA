# Bounded 25% PM M100 screen — 2026-11-07

[User report/Decision/Open] User replaces the long full benchmark with a High-RAM
A100, larger batches and a four-hour cap for the whole new GPU phase. Evaluate
617/2468 official ModelNet40-C test objects, all15 corruptions/s5/seed0. Training
data are excluded. The old full study is incomplete and superseded; its three
cache and eight inference archives remain preserved. New batch results form a
separate run. The initial standard r2 status below is superseded by the authorized replacement
record. The peer High-RAM runtime remains protected.

[Decision] IDs are frozen without labels: sorted
NumPy RandomState(20261107).choice(2468,617,replace=False), saved in
result/preparation/gsd_lowband_quarter_fast_20261107/evaluation617.json.
Historical benchmark exposure remains disclosed. This is a budgeted screen,
not a previously untouched confirmation or a causal proof.

[Decision] Three matched arms: operational SCD, live half-SCD/GSD0, and hardM100
halfmix. Retain prepared input/state/noise/RNG and graph/basis pairing. Keep
N8/Background35, DDIM100/finalalpha1, gamma.03/eta.003/p.95, PM checkpoints,
terminal behavior, raw/eval/frozen models and full decoder derivatives.
Profile B8/16/32/64 using throughput and actual memory without accuracy-based
selection; fix one batch across all arms and retain the real final tail.
Verify actual GPU VRAM separately from the High-RAM request. Calibrate alpha100
with label-free actual local/style update norms for the chosen batch, then freeze.

[Decision] The four-hour deadline covers new GPU allocation, canonical setup,
technical checks, calibration, inference and archiving. CPU preparation before
allocation is outside this clock. Require measured conservative completion of
617/all15/three arms within the remaining budget before inference. Otherwise
retain the feasibility failure; do not silently shrink coverage or controls.
No automatic full2468 or Scan stage belongs to this four-hour phase.

[Decision] At the completed617 screen, PM M100 minus operational <=0 stops
further accuracy expansion. Positive differences with a CI containing zero are
inconclusive. Report paired original-object cluster CIs and the half0 contrast.
The earlier native-repeat failure remains unresolved; a small positive difference
does not establish isolated GSD contribution. Any subsequent study needs its own
bounded scope rather than an automatic full scan.

## Ordered batches

1. Opt-in batch configuration and ModelNet617 tail; preserve default B2 guards
   and all sampler/loss/graph/head math. Broaden native audit batch-shape support
   with unchanged numeric gates. Run meaningful CPU tests.
2. Minimal profile/calibration/screen operators and actual-batch raw acceptance;
   freeze source, IDs and stopping rule; CPU checks and independent review.
3. Own High-RAM canonical10-cell setup, actual hardware verification, throughput
   feasibility gate, technical controls and frozen unsupervised calibration.
4. Complete matched617/all15/three arms within the remaining four hours; verify
   Drive/Mac archives and preserve all outcomes; publish completed code/knowledge.

[User report/Run] Userexplicitlyauthorizedcompletingandclosingonlyoldstandard
`tta-gsd-lowband-full-a100-20261107-r2`; CLIreportedSessionterminated20:29UTC.
Itsmonitor92199andoldlocalprogresswatcher45615stopped;accepteddata unchanged.
PeerHighRAMr3leftuntouched. NewownHighRAMA100session
`tta-gsd-quarter-fast-a100-20261107-r3` allocated20:33:04UTC. Userrequested
TerminalDriveMountcommandnow; commandprovided anduserowns thatserialsetup
operation. No competingkernelcell/monitorprobeisstartedbeforemountcompletion.
Fourhourwhole-new-GPUphaseclock20:33:04UTC→2026-11-08T00:33:04UTC; this
includesallocation/setup/consent/profile/calibration/inference/archive.
Canonical10cellsstillrequired. ActualGPUVRAMandcapacitynotyetverified.

## Batch1 CPU boundary — verified 2026-11-07

[Code/Verified CPU] `scripts/diagnose_decoded_spectral.py` now accepts opt-in
`--batch-size {2,4,8,16,32,64}` (default2) and records it in config. Both CLI
and worker selection guards require the pinned sorted unique hashed ID manifest
for benchmark coverage. ModelNet B2 retains even2..2468; larger batches permit
actual tails, including the fixed617 IDs (B32:19×32+9; B64:9×64+41).
Worker stride, selection, prediction/class cardinality and cumulative CSV count
use the actual batch. Scan B2 singleton semantics remain covered by CPU regression.

[Code/Verified CPU] Native decoder audit now accepts matching B1..64 latent/style
shapes with unchanged finite-difference epsilon, relative/absolute tolerances,
RNG/state/buffer guards and derivatives. The archive-only opt-in
`--skip-dense-filter-witness` defaultsfalse: it skips the optional dense
`physical_filters.npz` export while retaining the full resident target/loss,
all `physical_bases.npz` witnesses, exact filter/graph/reference hashes and
active indices. A rank257 fixture verifies that modes are not cropped to100.
Default dense export behavior is preserved. No sampler/loss/graph/head math changed.

[Test] TDD RED→GREEN covered configuration/tails and optional dense export.
`/private/tmp/tta-lowband-halfmix-20261107-venv/bin/python -m pytest -q tests --tb=short`
passed826 tests plus562 subtests, skipped2, with7 warnings (52.93s). The actual
frozen617 manifest was independently parsed with B32/B64 and exact tails9/41.
Four changed Python files parse with Python3.8 syntax; `git diff --check` passed.
Patch, full CPU log, source hashes and machine-readable verification are in
`result/preparation/gsd_lowband_quarter_fast_20261107/batch1_cpu_*`. No commit or
remote/GPU action was performed by this implementation batch.

[Limit/Open] CPU shape/cardinality tests establish no GPU speed, VRAM fit or
native numerical-repeat claim. Larger-batch calibration must be performed and
frozen for the chosen batch; old B2 alpha is not reused. Native audit capture can
retain dense per-object matrices, so technical/profile archive sizes require
actual measurement; the optional inference dense-export flag does not change
audit payloads. Reconstructible bases retain actual expanded ranks.

[Run/Open] Driveconsentreportedcomplete; MyDrive/thesislistedbeforecell1.
Canonicalcell1passed20:36:05UTCand sameVMCondareconnectpassed20:39:34UTC.
ActualGPUconfirmedNVIDIAA100-SXM4-80GB,81920MiB. Cell2launched20:39:51UTC.
Solemonitor77268; rawsetupstream result/preparation/gsd_lowband_quarter_fast_20261107/setup/monitor.jsonl.
Drivewasmountedbeforecell1atuserrequest; orderedcanonicalcellsremainunchanged.
No617inferenceornewaccuracyyet.

[Review/CPU] Independent Batch1 review found no correctness blocker. Default B2
behavior and numerical gates remain compatible; larger batches retain exact
selected IDs, real final tails and complete basis ranks. Only dense witness
export is optional. Native audit still retains dense filters, so its technical
witness must remain bounded and its cost included in feasibility.

[Run/Open] Canonical cell2 ended20:46:32UTC only at the previously authorized
fastrlock0.8.4 pip-wheel failure. The raw output identifies no other package
build failure. This single known failure is deferred; environment/Drive checks
run separately before cell3. Source code918780b was published as e1d445f after
29 integrated focused tests and84 subtests passed. File-only deployment package
7747906 bytes/SHA4eaa62df9cd66e91d32f989ab1b80e4599548a3c63157b286d76f96f001bf488
passed five CPU trust-boundary checks, including failed Drive backup causing
zero native replacements. Deployment and GPU runtime checks remain pending.

[Run/Open] Post-cell2 verification passed20:47:30UTC: Python3.8.20 and Drive
mount true. Cell3 finished20:49:51UTC after delayed CLI output, verifying the
already mounted Drive and byte-identical KNN wheel copy. No duplicate cell or
remount was submitted. Server inventory and a task-owned read-only console
process inspection were used while the original stream was pending; neither
peer nor kernel execution was changed. Cell4 began20:50:37UTC. New source
payload and budget were uploaded via Contents API; deployment waits for cell10.

[User decision] GPU runs need no repeated permission. On any subsequent Colab
session/connection problem, preserve the current state and wait for a new user
command before retry, recovery or new runtime allocation. This supersedes
previous autonomous connection-recovery authorization. The existing cell4
continues; no new incident was reported at this instruction.

[CPU/Open] The separate fast617 acceptance/analyzer passed seven focused CPU
checks locally (4.17s): actual B8/tail1 and rank257 preservation, stale RNG and
unsafe ZIP rejection,384-row alpha median, conflict-preserving cache materialization,
617-object/9255-pair aggregation and partial/missing-arm/SHA rejection. Source:
result/preparation/gsd_lowband_quarter_fast_20261107/{accept_fast617.py,analyze_fast617.py,test_accept_fast617.py}.
This is archive/statistics validation, not a new accuracy observation. Remote
source/native mirrors are explicit verified copies in the deployment context.

[Run/Open] Cell4 requirements installed with the known nonfatal pip resolver
warnings: dask2023.5.0 requires importlib-metadata>=4.13.0 but4.10.1 is present;
scikit-image0.20.0 requires networkx>=2.8 but2.6.3 is present. These are preserved,
not a claim of globally consistent dependencies. CUDA12.1/GCC11.4 package
installation continues after its solver passed20:56:58UTC. Native/model/data
checks remain mandatory; no new accuracy or throughput observation follows.

[CPU/Review/Open] Fast acceptance/analysis checks expanded to12 and passed;
pre-ACK sibling matching now includes exact complete graph/basis/support/rank,
IDs, labels, common source/native/model identity and paired RNG. The local
consumer's standalone checks pass for three parallel32MiB parts, whole ZIP
SHA/CRC, corruption rejection, durable receipt/ACK replay, true tails, hard
00:33:04UTC deadline and WAIT_USER on connectivity failure. Parent independently
ran the consumer selfcheck. Current files stay in the new quarter preparation
namespace; old artifacts are unchanged. Current hashes:
accept_fast617.py f1f238287d72f820e4ad988d4b12dd0775796cc9297c8094a83909f35b570d6d;
analyze_fast617.py 6c4d0b666d7a07f2c4b2a0d3b66c638e7a0ef181e2d1faf18ce6fea34332e0bd;
transfer_daemon.py dc89fa3d9b1ccedbfe6c30519d623c51b8229ae3194034cd44d2c9139c8e2f9f.
These CPU passes do not establish CUDA compatibility or speed. Cell4 passed
21:00:35UTC with architecture8.0 and expected source patches; canonical EMD
cell5 started21:02:17UTC, build output began21:02:36UTC.

[CPU/Review/Run/Open] Final operator payload138102bytes/SHA
9f117b79ae207d8ddf10b59337bb48f68c0a2127a441d626eead06d382ecb960;
manifest813720627faf8f4d034d3bd8d5ee32f095f338ff4c04e3fb58a392641fd7fbb0.
Fourteen operator contract checks and the40-job controller simulation passed.
Phase packaging retains each validated sealed ZIP once, with a SHA-bound
recovery map proving recovery of every omitted duplicate. Corrupt/unsealed,
changed and uncovered raw files remain preserved. Main archives and method
math are unchanged. Seventeen wrapper CPU checks passed, including actual
payload/native/source context, failed-phase preservation,32MiB reassembly,
Drive identity and success guards. Cell6 completed21:07:07UTC; cell7 completed
21:12:12UTC; cell8 PVCNN started21:13:18UTC. All native setup phases remain
serial on monitor77268. No new research inference result yet.
New local evidence root:
result/modelnet40_c/lowband_quarter_fast/20261107-203304_a10080_screen617/.

## WAIT_USER — recurrent CLI connection loss, 21:22:08UTC

[Run/Failed/User decision/Open] Cell8 completed21:20:42UTC: PyTorch2.1.2+cu121,
CUDA12.1/A10080GB, FPS and gather forward/backward passed. Cell9 completed
21:21:49UTC after Drive copy, ZIP CRC and ModelNet extraction. Cell10 was
submitted21:21:56UTC; the CLI emitted `RuntimeError: Connection was lost.` at
21:22:08UTC. No exact cell10 completion marker or exec-finished event has been
observed. This establishes a client connection failure, not VM termination or
proof of whether cell10 executed. Fresh full setup is incomplete/unverified.

The user's explicit recurrent-session boundary now applies: WAIT_USER, no
retry, recovery, new session, reconnect or new remote inspection. Parent issued
only a local monitor stop request21:23:23UTC to suppress further autonomous
launch/probe work; it is deferred while the existing CLI process remains
unresolved. Runtime was not closed; its current execution/assignment state is
unknown after the failure. Peer remains untouched. Research asset deployment,
batch profile, fresh calibration and617 screen have NOT started. The four-hour
allocation deadline remains2026-11-08T00:33:04UTC; waiting does not reset it.
Incident record: result/preparation/gsd_lowband_quarter_fast_20261107/cell10_connection_loss_20261107T212208Z.json.
Raw primary stream remains setup/monitor.jsonl. Completed CPU packages, wrappers,
acceptors and existing negative evidence are retained. Last published integration
is5fb5b5f; later local progress/incident records await the next user command
before publication or any other new external action.
