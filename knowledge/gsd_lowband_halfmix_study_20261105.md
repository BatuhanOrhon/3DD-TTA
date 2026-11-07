# PM low-band half mixture — 2026-11-05

[User report/Code/Open] Execution of the explicitly supplied
[handoff](gsd_lowband_halfmix_agent_handoff_20261105.md), starting at d5869c2.
Own branch `research/gsd-lowband-halfmix-20261105`, own worktree
`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-lowband-halfmix-20261105`.
Peer H5 files, branch, runtime and monitor are protected. Evidence labels:
[Code], [CPU], [Run], [Inference], [Open].

**Current: 2026-11-07 — all planned GPU tests and raw/local recount complete.**
[Final results and interpretation](gsd_lowband_halfmix_results_20261107.md).
Earlier allocation/user-wait entries below are historical. Background repeat
instability remains unresolved; no reliable GSD gain or default promotion.

## Implementation and execution plan

- [x] CPU: reuse `build_smooth_spectral_target(profile='hard')` with M100/200;
  preserve its active-support/isolate and complete boundary-eigenspace rules.
  Add `decoded_scd_weight=1` to tta.py; only change SCD when a live clean-PM
  loss exists. Existing additive guidance/defaults remain unchanged.
- [x] CPU: optional observer update probes use existing SCD and physical loss
  graphs, two extra VJPs, actual gamma=.03/eta=.003, float64 norm arithmetic.
  Record norms/cosine/mixed norm/cancellation. Reject nonfinite or <=1e-12
  component update norms during calibration; no clamping or silent rescaling.
  Freeze alpha as NumPy median of all predeclared per-object/checkpoint ratios.
- [x] CPU: driver flags for hard modes, checkpoint SCD coefficient and optional
  probes; all five arms use one locked DDIM100/finalalpha1/raw/eval/EMAoff host.
  Tests: handoff formula/gradient/FD, repeated eigenspace/isolate, exact prepared
  frame, half-SCD only on clean PM, no terminal update, frozen scalar validation.
- [x] Colab: own A100 runtime and sole monitor; canonical ten setup cells,
  cell1 restart/reconnect, then2–10, exact markers, known cell2 fastrlock only.
- [x] Freeze source bytes/dirty hashes/native bytes/assets, concrete IDs and
  whole GPU estimate before science. Target16 new IDs/all15/s5/seed0/B2,
  calibration4 separate IDs/Gaussian+Background, no labels/clean in calibration.
  Estimate30–45min, hard55min from first scientific GPU work, subject to
  measured projection. The Nov7 user instruction below supersedes the original
  >60min permission-question rule; actual total science40.33min.
- [x] Technical: grad/FD/projection/frozen states, same cached preparedX,z0,h0,
  noise and RNG; operational/live0 plus same-observer fresh-process repeats.
  Fixed gates: logit max .05, per-object/checkpoint RMS .01, first decoded RMS
  1e-4. Preserve failures and limit accuracy comparisons to descriptive ones.
- [x] Pilot: operational full SCD, live full SCD/weight0, live half SCD/weight0,
  halfmix M100, halfmix M200. All caches prebuilt before measured arms; inspect
  actual ranks/eigengaps/support and alias filters. No alpha/accuracy selection.
- [x] Archive seven files/prediction NPZ/cache/graph/norm/source/native bytes and
  successful/partial/failed ZIPs; verify local/Drive SHA. Independent PM primary,
  terminal secondary recount and object-cluster bootstrap, all15/per corruption,
  versus operational/full-live/half-SCD controls. Separate fresh pipeline,
  preparation, graph/eigh, loss/extra VJP, reconstruction+PM, terminal head,
  IO/worker wall and peak scopes. Update memory and publish necessary completed
  code/compact evidence via clean latest thesis-integration normal push.

## Scope and risks

No H8, online exact norm matching, beta/rate search, classifier gradients,
clean-reference tuning, extra seeds or default promotion. Existing Background
same-observer instability is unresolved; controlled cuDNN is not determinism.
A hard rank expansion can alias M100/M200; record it without treating aliases
as independent methods. Retaining U_M does not reduce full eigh setup cost.

## CPU result and allocation blocker — 2026-11-05 18:10UTC

[Code/CPU] The opt-in implementation is complete. `tta.py` accepts the fixed
scalar `decoded_scd_weight` (default1); it acts only when the decoded observer
returns a clean-PM loss. The existing spectral coefficient remains the additive
coefficient, so the new pilot passes `--checkpoint-scd-weight .5 --weight
(0.5*alpha_M)`. The other steps retain SCD1, terminal gets no new update.
One joint backward remains per step. Optional component VJPs retain the graph,
never accumulate `.grad`, and are explicitly counted/timed separately.

[Code] `--lowband-modes 100|200` selects `profile='hard', beta=None`; the smooth
profile remains the historical default for this diagnostic entrypoint. Hard
runs reject the H8 flag. Existing graph/isolate/eigenspace rules are reused.
Opt-in `retain_bases=True` forms F=sym(U_M U_M^T) from the selected padded basis
and archives that basis for **every batch**, with exact tensor/F hashes and
active vertex indices. This uses a selected-basis matrix product instead of
historical full-size zero-weight multiplication; it is the same mathematical
projector but carries **no cross-device/native bitwise equivalence claim**.
Full eigh is still computed. Dense F is retained in memory and the first raw
filter witness remains archived. Basis storage and graph build peak are recorded;
reconstruction peak excludes earlier graph-build temporaries but includes the
resident filters/bases. Instrumented replay is not fresh deployment throughput.

[CPU] Full suite514 tests /513pass /1historical skip in41.540s; independent
read-only review68 focused tests and an entirely isolated hard-graph self-check
passed. New combined positive-weight tests cover normal8 and Background35,
checkpoint-only halfmix, exact probe-on/off local/style update equality and one
joint backward per step. Hard loss/gradient reconstruction, repeated-boundary
expansion, empty support, frame checks and default sampler contracts are covered.
Existing warnings and historical fixture skip are retained in local full logs.
[Compact verification/allocation receipt](evidence/gsd_lowband_halfmix_20261105/cpu_and_allocation.json).

[Run/Open] Four allocation attempts produced no owned runtime:
`tta-gsd-lowband-halfmix-a100-20261105-r1` A100/HighRAM ->503ServiceUnavailable;
`r2`, `r3` and user-requested `r4` same A100/Standard ->precondition failed (capacity/usage/assignment
limit unspecified by server). Inventory showed three protected peer assignments.
No peer cell, reconnect, teardown, monitor, branch or file mutation occurred.
No task monitor was started, no setup cell or scientific GPU job ran, and no
alpha/accuracy/timing/native-gradient/repeatability result exists. Scientific
GPU time is0. This is an external allocation blocker, **not a failed guidance
result**. Prior Background instability remains unresolved. Do not promote the
CPU implementation as a completed Colab experiment.

Local preparation: `result/preparation/gsd_lowband_halfmix_20261105/` contains
source-hashed ten canonical cell wrappers, copied monitor source, test logs and
per-role selection manifests. Future remote `/content/studies/gsd-lowband-halfmix-20261105`;
Drive `thesis/result/gsd_lowband_halfmix_20261105`. No Drive copy is claimed yet.
Next available session name is `tta-gsd-lowband-halfmix-a100-20261105-r5`;
reconcile inventory first, allocate own A100 only, and follow canonical setup.

## Frozen selection and continuation details

[Code/Proposal] [Selection manifest](evidence/gsd_lowband_halfmix_20261105/selection.json)
uses SHA256 salt `gsd-lowband-halfmix-20261105-v1` after excluding582 IDs in the
previous physical study's exclusion union and32-ID cohort. Calibration IDs
411,672,777,2247; pilot IDs197,502,536,1008,1064,1240,1285,1422,1549,1670,1738,
1824,1858,2100,2285,2310. These remain development IDs with prior full-benchmark
exposure, not unseen confirmation. Two fixed8-ID pilot shards reuse the existing
runner's count8 contract; calibration uses two fixed2-ID shards.

After setup, strict CPU/source/native/asset verification and source seal:

1. Prebuild all preparedX/z0/h0/noise/post-preparation-RNG caches before measured
   arms; archive actual NPZ bytes and hashes, preserving per-B2 preparation cost.
   Do not allow one measured arm to populate missing caches while others replay.
2. Technical control/live_control fresh-process repeats on Gaussian44/91 and
   Background495/512, three processes each, all weight0. Use hard100 live path.
   Native full-decoder FD with fixed target, projection equality, frozen state
   and the original numerical thresholds remain mandatory. Failed repeats do
   not become accepted because PM classes happen to agree.
3. Calibration uses **only** new4IDs ×Gaussian/Background ×three clean PM
   checkpoints =24per-object ratios per M. Run `--arm live_control
   --lowband-modes M --update-norm-probes`, SCD1/weight0/controlled backend.
   Require all24 declared rows per M and their ID/corruption/checkpoint/source/
   cache identities before calling `freeze_update_alpha`; it deliberately
   rejects inactive/nonfinite ratios. Freeze each alpha and manifest SHA before
   any main accuracy result is read. No old802.5 coefficient or accuracy choice.
   Compute frozen-alpha mixture norms/cancellation from those saved component
   norms/cosines; the measurement VJPs do not belong to deployment timing.
4. Pilot seed0/B2/s5/all15, same caches and backend, five fixed arms below;
   no update-norm probes during measured pilot. Rotate arm order across the two
   shards: listed order then reverse. Reassess measured whole-phase projection
   before pilot; estimate30–45min/hard55min including technical/calibration and
   fresh-pipeline timing. Do not restart the phase clock. If expected >60min,
   obtain explicit whole-phase approval before further research GPU work.

| Arm | CLI arm | lowband-modes | checkpoint-scd-weight | weight |
|---|---|---:|---:|---:|
| Operational full SCD+PM | control | omit |1|0|
| Matched live full SCD | live_control |100|1|0|
| Matched live half SCD | live_control |100|.5|0|
| M100 halfmix | physical |100|.5|.5*alpha100|
| M200 halfmix | physical |200|.5|.5*alpha200|

All invocation prefixes use `scripts/diagnose_decoded_spectral.py --asset-root
ASSETS --asset-manifest-sha256 VERIFIED_SHA --result-root NEW_ROOT
--paired-state-root CACHE_ROOT --drive-root NEW_DRIVE_ROOT --backend-policy
controlled --index-manifest ROLE_JSON --index-manifest-sha256 ROLE_SHA --count
2_OR_8 --corruptions ...`. Resolve absolute paths and hashes after deployment;
these placeholders are **not launchable commands**. Seal the concrete commands,
actual native bytes and source/dirty identities before starting science.

5. Retain all rank/support/eigengap/basis/F evidence; report M100/200 aliases
   explicitly. Native numerical failure leaves any accuracy comparison
   descriptive; no gate relaxation or default promotion. PM primary, terminal
   secondary; all15/per-corruption correct/n, paired operational/live-full/half
   differences and10,000 original-object cluster bootstrap draws, seed20261101.
6. Complete all seven-file raw bundles/prediction NPZ/source/native/caches,
   successful/partial/failed attempts and SHA-verified Drive/local backup before
   inference or publication. Keep fresh pipeline, common preparation, kNN/eigh,
   loss/VJP, decoder+PM, terminal head, IO/worker wall and peak scopes distinct.

Remaining: all Colab setup/technical/calibration/pilot/analysis/archive phases.
Only completed CPU code, tests and compact preparation evidence are selected
for the current integration publication; raw setup wrappers/archives are local.

## User-directed wait — 2026-11-05

[User report/Decision] After standardA100r4 refusal, the user directs code
preparation now and will announce when an existing session becomes available.
**Do not retry allocation or begin setup/research GPU work until that notice.**
No monitor, retry queue or unattended GPU job exists for this task. Preserve
all peer assignments; this direction gives no teardown authorization.

## CPU publication and local source seal

[Code/CPU] Research source`89e58e9820796bec56ba9a1917a1f7b1e88bb9fc` was selected into clean
latest integration3fb7f96 as **1ccda41c04405a1508141b30a1c4132447ad1719**, normal push to origin/thesis-integration
and actual remote ref verified. All514tests/1skip pass in41.221s in that clean
checkout; independent68focused tests and whitespace checks pass. Reencode and
spectral-z knowledge additions were retained during conflict reconciliation.
[Selected/excluded scope, test hashes and publication receipt](evidence/gsd_lowband_halfmix_20261105/publication.json).

Local248-file source overlay684674bytes, SHA256
`6f5e4cb4d076204eec282bec4aafc0c4d4ca3527f202f3cdfddc3f1795198a69`,
under `result/preparation/gsd_lowband_halfmix_20261105/operator_v1/`. Exact
setup wrappers, source/deployment templates and CPU logs stay local, outside
shared Git. Native source is explicitly preserved from the pinned historical
asset base; future freshly compiled native bytes must be archived before any
declared pinned-native restoration. No native/source identity is silently
claimed from an uncreated runtime. No Drive copy, GPU result, alpha or
repeatability acceptance exists. User-directed wait remains in effect.

## Authorized resumption — 2026-11-07 09:19UTC

[User report/Run/Open] User explicitly resumes allocation and testing, revoking
the Nov5 wait. Own standardA100 `tta-gsd-lowband-halfmix-a100-20261107-r1`,
endpoint `gpu-a100-s-usc1c0-gz8ghzlq66bg`; actual A100-SXM4-40GB. Inventory
was empty before allocation. Own sole monitor exec91825 controls all cells.
Cell1 and same-VM Conda/hardware reconnect passed; cell2 started09:21UTC.
No scientific GPU run yet. Canonical source/wrapper hashes are preserved.

Own branch merged current accepted integrationa9b807a into8846b06; source
tree equals that shared revision. Prior halfmix code remains, latest H0/H5
opt-in diagnostics and negative results are reconciled. No default math or
native repair is introduced. Latest H0 brackets the first covered same-input/
full-RNG drift at PVCNN voxelization feature output, not a uniquely proven
kernel. H5 repeat floors fail beyond Background too. Existing numerical gates
remain fixed, and any small pilot differences must stay descriptive if failed.

Fresh local/remote/Drive task roots use `gsd_lowband_halfmix_20261107` /
`/content/studies/gsd-lowband-halfmix-20261107`. Original Nov5 records remain.
Initial local test rerun used an expired temporary environment without NumPy/
Torch:201tests/97importerrors/2dependency-induced failures, retained in local
`cpu_environment_failure.log`; a new task-specific venv restores CPU dependencies.
This is an environment failure, not a changed algorithm result. Fresh full
CPU suite is rerunning before scientific source seal. Setup remains separate.

[CPU] New task-local venv full pytest:807passed/2skipped,500subtests,51.02s.
Missing-pytest intermediate attempt is preserved separately; no code fix was
needed. Latest accepted source tree and optional halfmix paths pass together.
Source bytes will be resealed after this dated setup record; scientificGPU
clock has not started.

### Canonical fresh setup passed — 2026-11-07 10:06UTC

[Run] All ten canonical setup cells were executed sequentially on the owned
standard A100. Cell2's known fastrlock wheel failure alone was deferred as
previously authorized; Python3.8.20 was separately verified before cell3.
User completed Drive consent. Cells4–8 passed pinned CUDA12.1/GCC11.4/native
preparation and their persistent completion markers; cell9 prepared ModelNet40-C.
Cell10 passed native CUDA sum/FPS/KNN/Chamfer/EMD,15severity5 arrays, clean
array and checkpoint identities. This is a complete fresh replay of
the ten-cell split on this task's VM; no research notebook cells were run.
Actual setup checkout is dev@a458cd46bba1288a3f41302548bc777700cbc7d4,
separate from scientific overlayd6761c895ef6fa38d882bb2d4ca3f3205c1a1fa3.
Setup verifier SHA256c67cab91f47735e6e616669ca6aba587f91082e6256e3546fd844dc8e6b312ba,
local `result/preparation/gsd_lowband_halfmix_20261107/setup/cli_setup_verification.json`.
Scientific phase has not started. CPU deployment is preserving fresh native bytes
before the declared pinned-native restoration; Drive/local acceptance remains
pending. Whole scientific phase estimate30–45minutes, one hard55minute clock,
not reset between technical/calibration/pilot/fresh timing stages. Failure gates
remain unchanged. All setup/phase sources and monitor logs are retained locally.

### Technical gates — 2026-11-07 10:17UTC

[Run] Shared126prebuilt caches completed30.69s and per-cache Drive byte/SHA
checks passed. All12repeat processes and2native FD jobs completed. Full decoder
FD checks passed M100 and M200 at the fixed0.001/0.01scales and original
absolute1e-5+relative0.1gate; this does not certify the denoiser/whole trajectory.
CPU sealed-bundle repeat comparison: Gaussian control-within3/3,live-within3/3,
cross9/9pass; maximum cross logit difference1.8596649169921875e-5.
Background control-within0/3,live-within0/3,cross0/9pass; respective maximum
logit differences2.2772207260131836,3.4207286834716797,3.150510787963867.
Thresholds were not relaxed. Native numerical instability remains unresolved;
any subsequent pilot accuracy differences are descriptive. This failure is
preserved, not used to tune IDs/weights. Calibration starts on the predetermined
4IDs because both decoder FD gates passed. Scientific clock began10:08:09UTC;
technical phase finished10:16:08UTC, elapsed478.54s for its14jobs. Raw
bundle/Drive checks are per-run complete; consolidated local acceptance pending.

### Frozen calibration and pilot started — 2026-11-07 10:20UTC

[Run] Calibration4jobs completed;24active finite per-object actual-update norm
ratios per M, using gamma0.03/local and eta0.003/style. Median alpha100=
4433.612924010353,alpha200=4321.537946627087. All48unique(ID,corruption,PM,M)
rows and medians independently rechecked locally before pilot. Frozen JSON
SHA256fb121b127d0ac78ed8ac1ee8dc500d52c5f2a631c4dabaac2a932d6595eca9a1;
Drive/phase copy verified. Freeze code reads no labels or clean data. Calibration
cosine ranges M100[-0.352534,0.109953],M200[-0.327201,0.111400]; reconstructed
half-mix cancellation fractions18.18–30.93% and18.45–38.49%, respectively.
These are zero-weight calibration trajectory diagnostics, not applied outcomes.

[Run/Open] Pilot10jobs started10:20:03UTC, two8-ID shards,15corruptions each,
all five predeclared arms; second shard reverses arm order. Alpha never retuned,
update norm VJPs disabled in pilot. First operational-control job completed;
first live-control job is running. Full cohort comparison and consolidated
local/raw acceptance remain pending. Background repeat-gate failure means all
small accuracy differences remain descriptive.

[User report] During the running pilot, user explicitly authorizes all necessary
GPU tests in sequence without further GPU-run permission requests and requests
intermediate knowledge records until all tests finish. This supersedes the
previous >60-minute approval-question requirement for this task. Current frozen
scientific plan and single55-minute safety clock remain sufficient; no scope
expansion, reset, or additional GPU job was introduced by this authorization.

### Pilot progress — 2026-11-07 10:27UTC

[Run/Open] First8-ID shard's three controls completed all15corruptions:
operational SCD,live full-SCD/zero-GSD,live half-SCD/zero-GSD. Completed
live/half outer process times147.69s/149.14s, including sealing/Drive IO;
these are not isolated GPU kernel timings. M100 applied halfmix is now running,
M200 follows. Second shard uses the same five arms in reversed order. No weights,
IDs, thresholds or inclusion rules changed. Pilot accuracy has not been selected
or interpreted from logs. Final sealed local recount/paired acceptance pending.

### First common shard complete — 2026-11-07 10:33UTC

[Run] First8-ID/all15corruptions shard completed all five arms without process
failure. M200 outer wall170.22s; second shard started in reversed order at
10:32:31UTC. No accuracy-based selection; full planned16-ID cohort still running.

[Run] Independent local acceptance exercised on an actual sealed Gaussian
technical live0 bundle SHAffad913e728dcfc24a98a0817477c848533caf00ed7ee1a6c6c3a12e902f6d3b,
34,282,579bytes, downloaded via CLI along with exact manifest/cache/native bytes.
Source/native identities, actual latent/noise/RNG cache tensors, selected IDs,
PM/terminal predictions, CSV summary and retained-basis checks passed. CPU
float64 basis projection loss0.012023248691596378 vs native0.01202324777841568,
absolute error9.13180697725835e-10 below fixed1e-6. This validates one real
acceptance path, not the full campaign; consolidated raw acceptance remains
pending. Local `result/preparation/gsd_lowband_halfmix_20261107/preview_acceptance.json`.

### Both applied arms complete — 2026-11-07 10:38UTC

[Run/Open] Both M100 and M200 applied halfmix arms now completed the full planned
16-ID/all15corruptions coverage, as two8-IDshards. Second-shard M100 outer
wall160.30s; second shard's three controls remain, with half-SCD running.
No process failure yet. Still no final common-cohort accuracy claim until all
controls finish and raw/prediction recount passes. Frozen coefficients unchanged;
Background numerical repeat failure persists as the interpretation limit.

### Full limited pilot completed — 2026-11-07 10:45UTC

[Run] All10pilot jobs completed with rc0, all five arms on the same16original
IDs and15corruptions,severity5,seed0,B2. Frozen alpha identity remained exactly
unchanged. Scientific clock elapsed2210.44s at pilot end; last control outer
wall118.17s. All per-job sealed Drive checkpoints reported verified. No accuracy
claim yet: full raw package/source/native/cache validation and independent PM/
terminal recount remain pending. Five fresh-preparation timing jobs now start,
one B2pair on Gaussian/Background per arm, separate caches; timing only, excluded
from pilot accuracy. Negative Background repeat gates remain in the record.

### All planned GPU phases complete — 2026-11-07 10:48:30UTC

[Run] Five fresh-preparation timing jobs completed rc0, preserving10distinct
fresh caches. All34logical jobs (1cache preparation +33inference/diagnostic runs)
finished with no timeout or process failure. Single scientific clock2419.940985s
=40.33235minutes, within30–45minute estimate and55minute bound; setup excluded.
Pilot16-ID/all15/five-arm coverage complete. Consolidated raw/Drive/local package
acceptance, independent all-run PM/terminal recount and knowledge publication are
now CPU/archive work. No more GPU computation planned. Own runtime and monitor
remain live; no teardown authorized. Background repeat failure remains unresolved.

### Raw Drive package verified — 2026-11-07 10:53UTC

[Run] All33sealed inference/diagnostic ZIPs validated for CRC/member SHA during
collection; no broken archives. Actual126common+10fresh caches and native ELF
bytes preserved,383payloads. Raw archive2329149449bytes,
SHA256e2f2e4c4025fc4b7a64b25b08504760432a3ffdbcfa64ee1594101c1712d1898; Drive byte/SHA verified at
`/content/drive/MyDrive/thesis/result/gsd_lowband_halfmix_20261107/raw_package_20261107T105026Z.zip`. Local CLI transfer is running as70parts of at most32MiB,
with exact whole-archive reconstruction/SHA required. Destination
`result/modelnet40_c/lowband_halfmix_study/20261107-105300_raw/`.
Full local scientific acceptance/recount remains pending; raw bytes are immutable.
All GPU work completed40.33minutes; own runtime/monitor remain live.

### Independent local recount complete — 2026-11-07

[Run/Inference] Local/Drive raw SHA matches;383payloads,33canonical rawZIPs,
10nativeELFs,126common+10fresh caches validated. All33runs/632B2cases accepted;
504basis-loss checks pass,max8.5741e-8. PM primary operational165/240,live0164,
half0169,M100171,M200166. M100 vs half0+0.8333pp,CI[-.8333,2.5]; M200 vs
half0-1.25pp,CI[-2.9167,.4167]. These conditional16object-cluster intervals do
not contain native-repeat variability. Zero-guidance pilot compatibility fails
30/120B2cases across13corruptions; M100/M200 graph+reconstruction cost about23%
above operational. No causal gain/default promotion. All raw/negative evidence
retained. [Complete result and compact evidence](gsd_lowband_halfmix_results_20261107.md).
Derived Drive backup and clean integration publication are the final pending steps.


[Run] Final derived402,060byte/SHA61583a1d613837eb711cd51b10b759ef316ec923671cddfb8bf113ab3e4c2bc8
is Drive/local/actualDrive-download verified. Clean integration full807tests/2skip,
500subtests/61.61s and identical33-run recount passed. Independent review found
no concrete blocker. Only normal integration publication/final receipt remains.


[Run] Necessary completed results/analysis/knowledge published by normal push as
`ab56a007fad95ce806cf68c50f7067a2c1ee6bd7` to`origin/thesis-integration`, preserving latest target79445c0
and peers. [Selected/excluded scope and checks](evidence/gsd_lowband_halfmix_20261107/publication.json).
Scientific task complete; final metadata receipt is additive. Runtime/monitor
remain live; no close authorization.


[Run] Final metadata Drive/local roundtrip and latest live runtime proof accepted;
[final receipt](evidence/gsd_lowband_halfmix_20261107/final_evidence_receipt.json).
All requested bounded tests, acceptance, backups and scientific publication complete.
No active research workers; runtime/monitor preserved without teardown.
