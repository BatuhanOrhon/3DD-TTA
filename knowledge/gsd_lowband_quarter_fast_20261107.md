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
