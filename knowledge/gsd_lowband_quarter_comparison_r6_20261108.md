# PM M100 comparison on owned r6 — 2026-11-08

## Current disposition — FAIL / accuracy-improvement candidate closed

[User decision/Run/Inference,2026-11-08] The user requests that the method be
recorded as failed based on the observed results. PM low-frequency SCD/GSD
halfmix is **FAIL as an accuracy-improvement candidate under this tested M100
configuration**. The latest accepted paired comparison covers448 ModelNet40-C
test objects/all15 corruptions/severity5/seed0/B32: operational SCD4438/6720
(66.041667%), M1004426/6720 (65.863095%), half04410/6720 (65.625%).
M100−SCD is−.178571pp, conditional object-cluster95%CI[−.610119,+.238095].
M100−half0 is+.238095pp,CI[−.133929,+.610119]; neither contrast establishes
the intended operational accuracy gain. The earlier16-object+2.50pp pilot
gain was not reproduced in this accepted follow-up.

The candidate is closed and is not promoted to the baseline. The conditional
different-dataset/separate-source-model replication is not advanced. No further
GPU continuation or parameter search is scheduled by this decision. Negative
results, pilot positives, failed numeric gates, transport failures and immutable
raw evidence remain preserved; this disposition does not alter their values.

Limits remain explicit:617/2468 was the planned25% screen, while only448/2468
(18.15%) have accepted paired accuracy. Shutdown salvage of extra raw archives
does not constitute512/final617 acceptance. Native repeat gates failed29/30;
the v2 scalar-comparator correction changed acceptance post-hoc and did not
resolve sampling instability. The interval conditions on one observed seed and
excludes native numerical variability. Thus FAIL records the unsuccessful
development candidate, not a completed full-data benchmark or a general proof
of harm/ineffectiveness. Runtime and monitor remain closed. Historical execution
and next-action entries below are superseded by this disposition.

[User decision/Run/Open] User confirms the unknown A100 is not theirs and must
not be used; explicitly authorizes a new owned runtime if prior r5 is gone and
forbids closing another agent's runtime. At10:49UTC, server inventory lacks old
r5 endpoint, lists unknown+peer. Unknown was never attached, executed or closed.
A new own r6 A100/High-RAM request at10:52 returned503 Service Unavailable.
Read-only inventory confirmed no r6 assignment. One same-name allocation retry
at10:55 succeeded by10:56:29UTC. No other runtime was stopped.

Owned session: tta-gsd-quarter-fast-a100-20261108-r6.
Endpoint: gpu-a100-hm-ass1c0-uycluss04t2v.
Worktree: /Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-lowband-halfmix-20261105.
Branch: research/gsd-lowband-quarter-fast-retry-20261108.
First monitor66517 started10:56:48UTC with supported local REQUEST_TIMEOUT=60;
cell1 passed10:57:14UTC, same-VM Conda reconnect10:58:11UTC confirms A10080GB,
81920MiB. Known-idle local monitor stopped10:58:36UTC solely for serial Drive
mount; no VM teardown. User reports new Drive consent completed. Sole owner
is pending drivemount terminal40681; monitor restarts after mount completion.

[User decision] The user explicitly removed the global time window and asked
for the comparison. Original r5 budget5986cce and hashed waiver32421e2 remain
historical immutable anchors. R6 has separate runtime_assignment.json; no
arbitrary new total cutoff. Worker/CLI timeouts, WAIT_USER on connection loss,
source/native/cache/FD/repeat/archive gates and finite40-job scope remain.

[Protocol] Scientific source918780b, assetmanifest e3f280/source a62a7e unchanged.
Fixed617/2468 official ModelNet40-C test IDs/all15/s5/seed0; three arms
operational SCD, livehalf0, hardM100 halfmix. Validated B32 resource selection
from r5 is reused only after actual A10080GB verification; old resource raw
archive f94fc37 remains provenance, not new accuracy. No old alpha or B2 states.
R6 prepares all64 calibration IDs outside617 anew, freezes384 actual gamma.03/
eta.003 weighted local/style ratios, then runs unchanged native FD/repeat/tail
checks and matched617 inference. No accuracy-dependent selection or silent
coverage reduction; native instability remains and small gains stay descriptive.

[Code/CPU/Open] Completed r5_unbounded fork is CPU-reviewed with explicit waiver,
None globaldeadline, preserved source/numeric/ACK/timeout guards. R6 retarget is
local CPU-only during fresh canonical setup. Initial deployment will explicitly
resolve the already verified mounted thesis shortcut before unchanged strict
native guards; actual fresh R6 native bytes are Drive-preserved before any pinned
restore. Scientific donor ZIP4eaa unchanged. No science launch until setup,
source/context/Drive/Mac identities and required technical gates pass.

Current evidence: result/preparation/gsd_lowband_quarter_fast_20261108_r6/.
R5 canonical/environment/context and successful B8/16/32/64 resource profile
archives remain in their original roots. No617 accuracy result or fresh alpha
exists as this record starts. Prior transport/backend failures remain preserved.

[Run] 11:04UTC Drive/mount audit: user consent propagated. Prior local mount
terminal was no longer present, so a read-only raw-console audit first verified
fuse.drive at /content/drive, readable project wheel, both kernels idle and no
active build. Console detached without interrupting kernel. Sole monitor23310
restarted11:04:39 with REQUEST_TIMEOUT60 and scoped local caffeinate -i; actual
cell2 launched11:04:48. No new allocation, duplicate job or peer operation.
Audit receipt: r6/setup/drive_mount_verified.json. No new research accuracy.

[Run/Code/CPU] 11:12UTC canonical cell2 reached only the expressly deferred
fastrlock0.8.4 wheel failure (full log retained); separate Python3.8/Drive
verification passed LOWBAND_CELL2_ENV_VERIFIED. Cells3–10 continue serially
through sole monitor with exact markers and immediate stop on any new failure.
Final R6 operator/bootstrap independent review found/fixed five stale R5 Drive
constants before deployment. Final checks:16 operator contracts,40-job simulator,
4 bootstrap tests,5 native-preservation guards,33 consumer tests plus transfer
checks. Active producer/consumer ROOT and Drive now R6; historical R5 metadata
stays byte-exact. CPU receipt r6/cpu_verification.json; sealed operator ZIP
143380B SHAe5fd72d180d7e4a198dd91ad29ed6cf23b79ca0f0616c9d21f0a9b7fa4499642;
bootstrap9853B SHA13e43385bd1edbb85080f4a507f8d28beaacc69f58a6f9d0559182f8aec4a6e8.
Staging uploads use only ContentsAPI while setup owns kernel; no science launch.

[Run] Canonical cells3–10 completed11:14:53/11:25:09/11:27:11/11:30:01/
11:34:00/11:41:16/11:42:25/11:43:58UTC, each exact marker+exit0; native phases
also check on-disk completion markers. Sole monitor advanced serially without
inter-stage idle gaps. Cell10 actual GPU/native/data/checkpoint verification
passed; Mac9005B proof SHA2b5bba7087d63a1c1243ee71c4a5564ec4c9e7a913e3ce5a5a2c08dd8112ea01
retained at result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/
fresh_canonical_setup.json. Python3.8.20/Torch2.1.2+cu121/NumPy1.21.2/A10080;
15s5 arrays/2468test, prep dev a458cd46. Passed tensor/FPS/KNN/Chamfer/EMD;
known two nonfatal pip dependency conflicts remain recorded, no global pip
consistency claim. File-only deployment now starts with pinnedPython wrapper
SHAf622a87ae45deb1f41d12c54d554c3dc96497f8af48800aeb58c0526c617521e.
No fresh alpha, technical result or617 accuracy yet.

[Publication] Completed scoped preparation/failed R5 evidence from sourcebranch
commits aebfcbd+f06c86e selected into clean publication checkout; normalFF push
origin/thesis-integration0cd5691 succeeded. Included5 knowledge files; raw
archives/weights/temp setup wrappers stay out of Git. Scientific batching code
918780b was already integrated as e1d445f; no new scientific code in this note.
Validation: independent CPU receipts noted above, namespace/seal review,
integration diff/check; no accuracy completion claim. Own branches/raw retained.

[Run/Accepted] File-only initial deployment completed11:45:30UTC; narrow
configured Drive alias resolved correctly before first mutation. No deliberately
failed attempt. Fresh10-native/proof backup Drive-verified before replacement,
all400 scientific asset identities checked; only missing multimethod1.9.1 may
be added by pinned dependency guard. Context8,218,583B SHA882e5879d687369dfa6c6433cc508f6dfea61e08c3f7ac4e4bf3c13d8994cab6
and fresh-native6,455,290B SHA260c7e98e02d9c3a8cf1d046e84881dca7d4547d83ed4f29a9020917ab948f0e
are both Drive+Mac accepted.124 context files/allCRC/memberSHA, old+active
operator manifests/source/native10 verified with existing strict acceptor; all
10 fresh binaries were present and preserved. No old alpha/prepared caches.
Canonical local study root:
result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/
(deployment_context_acceptance.json; raw two ZIPs; reconstructed downloaded/).
Fresh calibration now starts only after these gates:64 disjoint IDs,384 actual
gamma/eta-weighted ratios, B32. GPU permission standing/user time waiver; no new
accuracy interpretation. Next accept calibration archive then unchanged technical
FD/repeat/actual41-tail gates before617screen. Peer/unknown untouched.

[Run/Accepted] Fresh calibration launched11:47:23UTC, cache22.1265s, actual
norm job103.8195s, phase completed11:49:42UTC. All384 positive/finite ratios
independently validated on Mac,64ID×2corruption×3cleanPM checkpoints, B32.
Frozen alpha100=4398.3843101977745, freeze178240B SHA6f293cc40763ed3486076c1d478b8cdce976cdb9c977db4cc51e41df4bb0714f.
Labels/clean target not used in alpha; both actual gamma.03/eta.003-weighted
local/style gradients included. Fivepart raw142115354B ZIP SHA61c948f7e2495d59e435a54b2ef6c5e89dd19be0f6cceadbacef1ef0bd795ceb
Drive+Mac/CRC/memberSHA accepted146new+132recovered files; receipts at canonical
study root calibration_phase_acceptance.json and calibration_validation.json.
No617 accuracy used or produced. Nexttechnical begins only after this acceptance;
finite-difference hard gate, alternating repeat failures preserved/descriptive,
actual41tail and appliedhalf0/M100 timings unchanged. Alpha stays frozen.

[Run/Accepted/Failed numerical repeats] Technical phase11:52:26–12:03:13UTC
completed FD/actual41tail [32,9]/all6repeat+2appliedtiming jobs. Raw431604836B
SHA8d12f79833673d51121edcd51a23a9665e4ad411b5f3e39467cb3129e85774fd
Drive+Mac/13parts/CRC/memberSHA accepted1220new+1140recovered files.
Finite-difference passes original relative.1/absolute1e-5 at eps.1233658981/
1.2336589813, relative errors.09840418/.07295051; local decoder only.
Repeat gate FAILS: control-within0/6 batchcases; live0-within0/6; cross1/18.
Total1/30 B32 batchcases pass (3+3+9 comparisons, Gaussian+Background).
Original logit max.05/checkpointRMS.01/first-decodedRMS1e-4 thresholds retained.
Operational SCD itself drifts. All failures preserved; no isolated causal-GSD
claim, unchanged descriptive617 comparison remains authorized/predeclared.
Actual measured per-case seconds Gaussian/Background: control5.684696/16.624911;
half010.914317/21.508307; M10011.439704/21.973138. Cold/postworker costs separate
~19–20s/~7–10s. No norm probes in applied timing/inference. These two-corruption
case timings are not a measured all15 duration. Alpha4398.3843101977745 unchanged.
Compact accepted receipts/617+cal64 IDs/CPU pins copied under
knowledge/evidence/gsd_lowband_quarter_fast_20261108_r6/ (no raw/weights/wrappers).
Next: start bounded local consumer, then finite40-job617/all15/three-arm screen;
Drive+Mac SHA/CRC/state/basis/RNG/cross-arm pre-ACK guards remain mandatory.

[Run] Full fixed617 screen submitted12:07:12UTC, controller binding12:07:16;
first shard000 cache now active. Local finite consumer PTY48160/process69638
started before launch, confirmed live with accepted frozen context. Sole kernel
owner monitor23310; current endpoint unchanged; no competing cell. Global
deadlineNone only via exact waiver;40finite jobs,7200sworker,180sCLI transfer
and pre-ACK identity/CRC/Drive+Mac guards retained. PrimaryCLI86400s is a transport
watchdog rather than a four-hour research stop. No full2468/Scan expansion in
controller, no early accuracy-based truncation. All617×15×3 outputs required.
Actual OP-source archive b313b95386ca474231ec2469d3293a94038911d0803aa282c9b3cdb9307a8dac
is separately bound from stagingpayload e5fd72 (same pinned member identities,
different ZIP metadata). No617 accuracy accepted yet at this launch record.

[Run/Partial] 2026-11-08 12:31:16 UTC: first64 matched objects/960predictions perarm accepted
Drive+Mac with all4receipts/ACK, native/source/cache/RNG/half0-M100 filters matched.
SCD629/960=65.520833%;half0623/960=64.895833%;M100631/960=65.729167%.
M100-SCD+0.208333pp (14corrected,12harmed), conditional64object95%CI
[-1.25,+1.666667]pp; M100-half0+.833333pp,CI[-.104167,+1.770833].
Native29/30repeat failures still unresolved; no causal/seed/unseen-data claim.
HardM100 effective ranks100–257 for960object-corruption graphs perlivearm,
identicalfilters/ranks acrosshalf0/M100. Local derived/partial_064.json; compact
repositoryreceipt evidence/gsd_lowband_quarter_fast_20261108_r6/partial_064.json.
Coverage64/2468=2.59319%officialtest,64/617=10.3728%screen. Other oldB2 runs
notpooled. Continue all617; no partial-accuracy futility/expansion decision.
GPU corefirstjobs:cache36.6645s,SCD217.6936s,half0422.4759s,M100436.3106s;
raw archive/transfer creation costs are additional and remain in wholeelapsed.

[Publication] Sourcea7cdfaa selected14knowledge/compactreceipt files and pushed
normalFF as origin/thesis-integration916ef24. JSON/IDSHA/diff checks pass;
no raw archives/weights/temp wrappers or uncompletedaccuracy claim promoted.

[Run/Failed hard gate] 12:35:43UTC shard001_half0 worker completed0/420.2177s
and sealed957625938B rawZIP a51c70bdfd743efce1a2b8b164b3c2ad456b6e7bc56abbbf5dc4aedd43fb5ea7
(Drive final receipt verified), then independent basis-loss acceptance failed
at accept_fast617_unbounded.py:256, absolute1e-6. Campaign stopped before
shard001M100/control; no accepted job receipt/ACK for rejectedhalf0. Source
journal failed; exactprimaryexec end12:35:44/rc0/markerfalse (remotechilderror).
This is NOT a Colab connection/runtime loss; no newruntime/research retry.
First64/all15/all3 remain accepted; second64 cache accepted, half0 scientifically
rejected, no128pairedaccuracy. Local transferconsumer stopped while idle after
all5existinggoodarchives/ACK to avoid endless polling of failedcampaign;
ownkernelmonitor/runtime remain live, peer untouched.
Rootcause investigation follows systematic-debugging: no tolerance relaxation
or guessing. Acceptedshard000 background loss~8.7 hasFP32ULP9.54e-7, fixedgate1e-6;
its maxobservedFP64basiserror5.05e-7 passes. Firstdecodedhashes match all60
acceptedlivecases. FP32sumvsFP64reference is a hypothesis, not confirmedcause
until rejectedartifact is independently reproduced. File-only salvage now
preserves failed archive/sourcejournal/log; no newloss/GPU inference launched.

[Run/CPU/Independent review] Failed957625938B/a51c archive is now wholeSHA/
29parts/CRC Drive+Mac preserved at rejected/shard001_half0/payload/; noACK or
retroactivev1acceptance. Localfailedjournal+workerlog also downloaded. CPUaudit
andindependentreviewer reproduce exactlyonefailure/30: Background/batch0001,
32IDs398–499; raw8.811872482299805 vsFP64basis8.81187358757228,
abs1.10527247443315e-6 (1.159FP32ULP). Allfirstdecoded/probe/reference/basis
hashes match; no support/rank/unit/reduction mismatch, no rawbytes altered.
Per-objectFP64energies roundedtoFP32 and accumulatedleft-to-rightFP32 give
rawEXACT; FP64sum ofFP32per-objectterms differsidealonly7.71e-8. This identifies
batch-accumulation arithmetic asdominantcause, notproofeachnative term/gradient
is bit-exact.90actualcases qualify source-orderreplay atunchanged1e-6 number,
max9.536743e-7. Fullaudit underdiagnostics_cpu_20261108T124719Z.

[Decision/Code planned] Independentreview recommends explicitvalidator-v2
comparator correction, NOT producer/guidance change. This IS a posthoc
acceptance-rule amendment: v1 rejection/source/journal/raw remain immutable;
neverclaimv1passed orunchangedacceptanceregion. RetainFP64ideal/error diagnostics
forallcases, exactsource-dtype/object-order replay, same1e-6numericthreshold,
allnativegates unchanged, nofallback/min/ULPtolerance/droppedcase/specialBGbranch.
NewCPUregression/negativecorruption tests anduniformfullvalidator replay on90
actualcases precede anyv2acceptance. Separatev2source/verdict/bindingarchives and
reviewedexplicitresume required; no completedGPUworker rerun ordata/alpha/source
change. CurrentGPU remains stopped for this verification.

[Code/CPU/Independent review] Explicitvalidator-v2 source27478B/7f082803c10ff55300ed993bb1b32606e574ffcffc032fb0369609719b9e50bc
passes21targeted tests (withTDDRED/GREEN),859existingconsumer/projecttests2skip
562subtests7warnings. Fullv2load_run qualifies90spectral+30control cases and
2cachearchives/60preparedstates; v1 exactrejection remainsreproduced. Reviewer
independentlypassed21andPython3.8syntax withnoblocker. Sourcefrozen918/assets e3/
alpha4398 unchanged; currentruntime stillnotresumed. Portablepublicationbundle
underknowledge/evidence/gsd_lowband_quarter_fast_20261108_r6_basis_fp32_v2/ includes
source/tests/actual32energyregression/fixturebuilders/fixed617IDmanifest/qualification.
Published-location21tests pass; omittedfixturemanifest foundandadded before
publication. Broadernewconsumer/controller recovery verification stillpending;
nooriginalacceptance/source/ACKpaths overwritten.

### 2026-11-08 13:53–13:56 UTC — actual v2 admission and same-VM resume

[Run/CPU/Independent review] Reviewed provenance bundle13784478B/fecb316b,
operator manifest4085B/1ef63cee and installer13144B/c7e67296 are deployed in
separate v2 paths. File-only admission ran13:52:48–13:53:34UTC with exact
completion marker and CLI0. All six current Drive raw ZIPs were rehashed;
the original v1 files/failed journal/rejected manifest remain unchanged.
Actual import receipt9923B/d2a92a53 and v1 metadata ZIP10121768B/5925dd4d
were downloaded. Mac finalization passed actual snapshot wholeSHA/CRC,
all six raw SHA, full sidecars and six genuine new remote ACKs. It created
six accepted_basis_fp32_v2 receipts; old v1 receipts remain historical.
Compact actual receipts and explicit acceptance amendment are preserved under
knowledge/evidence/gsd_lowband_quarter_fast_20261108_r6_basis_fp32_v2/.

[Run/Running/Open] Mac proof uploaded successfully; new consumerPTY16246
replaces stopped v1 consumer48160. Sole kernel owner remains monitor23310.
Same-VM explicit screen --resume started13:55:51UTC, operator binding13:55:55.
Parent phase SHA8144ca085d1f638b53c1de22ef1f2d842dfb4c66cfd226460401558e168b8cde.
Expected ledger40 comprises six retained imports and34 remaining workers
(26 inference/eight caches), starting shard001_M100 then control. No completed
GPU job is rerun. Source918/assets e3/alpha4398.3843101977745/617 IDs/B32/seed0/
native capsule/global time waiver are unchanged. Allocation start1794135389
still anchors whole-phase elapsed time, including diagnosis/recovery overhead.
Resume is running, not617 completion; first64 accuracy remains unchanged.
Native29/30 repeat failures remain and forbid causal attribution from small gains.

### 2026-11-08 14:09 UTC — 128 paired objects accepted; no SCD accuracy gain

[Run/Partial] Shard001 M100 completed440.245s; its957624316B/f4428f15 raw
archive passed v2 basis/source/pairing checks and Drive+Mac acceptance. SCD raw
122979901B/ac193533 also passed and was acknowledged. All eight receipts across
two complete shards are present. V2 analyzer emits derived/partial_128_v2.json,
copied byte-exact to current v2 compact knowledge evidence. No old B2 run pooled.

128/2468 official test objects=5.186386%;128/617 planned=20.745543%.
Each arm has1920 predictions, all15 corruptions/s5/seed0/B32:
SCD1250/1920=65.104167%, half01236/1920=64.375%, M1001250/1920=65.104167%.
M100-SCD0pp; conditional object-cluster95%CI[-.885417,+.885417], corrected24/
harmed24. M100-half0+.729167pp, CI[+.104167,+1.354167]. Thus M100 currently
recovers the half-SCD control's deficit without improving operational SCD;
the old16-object +2.5pp is not replicated in this partial cohort. This does not
establish a causal spectral effect: native repeat29/30 failure persists and the
interval excludes that uncertainty. Partial cohort never triggers the fixed617
stopping decision. Shard002 cache completed37.160s and M100 now runs; continue
unchanged. Exactly one legacy-v1 basis failure remains visible in v2 diagnostics.

[Publication] Completed v2 admission milestone source36aad5b was integrated and
normally pushed asf94e3d8 to origin/thesis-integration. Curated scope: research
notes, explicit amendment and compact genuine admission/provenance receipts;
temporary deployment/phase/consumer wrappers, full sidecars and raw ZIPs stay
out of Git, preserved under result/ and the verified Drive provenance bundle.

### 2026-11-08 14:32 UTC — 192 paired objects; primary delta −.034722pp

[Run/Partial/Open] All three arms/three shards accepted:192/2468 objects
(7.779579% official test),2880 predictions per arm. SCD1861=64.618056%,
half01845=64.0625%,M1001860=64.583333%. M100-SCD−.034722pp,
conditional object-cluster95%CI[-.729167,+.659722]. M100-half0+.520833pp,
CI approximately[0,+1.041667]. Original v1 failure remains visible exactly once.
No operational SCD gain established; no causal/native-stability claim. Fixed617
continues; next paired report256. Full derived partial_192_v2.json SHA
d379998b0d1e9892c068b3bc22cfc09ac342d922a39e15f3ac04963b3fe2caba is
copied byte-exact to current v2 knowledge evidence. Dedicated local-only CPU
supervisor records each future complete64-object group in
gsd_lowband_r6_progress_cpu_20261108.md; no remote/CLI/PTY access by that supervisor.

### 2026-11-08 14:53 UTC — 256 paired objects; no operational SCD gain

[Run/Partial/Open] Four complete paired shards/16 accepted job receipts:
256/2468 official test objects=10.372771%; planned617 coverage41.491086%.
All15/s5/seed0/B32 gives3840 predictions per arm. SCD2528=65.833333%,
half02513=65.442708%,M1002528=65.833333%; primarydelta0pp,
conditional object-cluster95%CI[-.572917,+.572917]. M100-half0+.390625pp,
CI[-.052734,+.833333]. Original-v1 failurecount1/native29/30 remain; no causal
or+2.5pp claim. Fifthshard half0 running, fixed617 continues without tuning.
Full partial_256_v2.json SHA484b0b8f58ca22cc44a4ce2f8e1a15b0148cb0a6dba2cda72a618d519aa9593e
is preserved in derived and current v2 compact evidence; local CPU progress
note records the same accepted counts and intervals. No oldB2 pooling.

### 2026-11-08 15:11 UTC — 320 paired objects; primary−.0625pp

[Run/Partial/Open] Five complete paired shards/20 accepted jobs,320 objects,
4800 predictions per arm: SCD3176=66.166667%,half03158=65.791667%,
M1003173=66.104167%. Primary−.0625pp,conditional object-cluster95%CI
[-.5625,+.4375]; M100-half0+.3125pp,CI[-.104167,+.729167]. No operational
accuracy gain established. Original v1 failure remains once; native failure
29/30 unresolved. Keep617 fixed, no partial stopping or expansion. Full derived
partial_320_v2.json3588903f9739b7545422b47fc33042f50bdfd1bfe98a38f8fd143ac51711100a
and per-stage logs remain under result/; CPU progress note preserves the metrics
and SHA. Later overlapping partial dumps stay local, with final617 evidence
selected for integration. Sixthshard M100 running.

### 2026-11-08 15:34 UTC — 384 paired objects; primary−.086806pp

[Run/Partial/Open] Six complete all15 paired shards/24 accepted jobs:
384 objects,5760 predictions/arm. SCD3745=65.017361%,half03726=64.6875%,
M1003740=64.930556%. Primary−.086806pp,conditional object-cluster95%CI
[-.538194,+.364583]; M100-half0+.243056pp,CI[-.15625,+.642361]. Still no
operational gain or oldpilot+2.5pp replication. Fixed617 continues with original
native29/30 and posthoc-v2 caveats. Full derived partial384 SHA
a28831c771ff1b7c98dc19b042511f4747c01420936b342a4009f8ea5dfa7c82;
metrics/coverage/CI/SHA preserved in CPU progress note; full overlapping derived
JSON remains local. Seventhshard SCD runs. No new GPU retry or source change.

[Code/Run/Partial,15:51–15:54UTC] Reviewed file-only finalizer0a75212a
staged successfully under /content. Two auxiliary wrapper PUT uploads into the
new ROOT/finalization_inputs directory returned HTTP500/CLI1. Installed CLI
upload code does not create parents; no prior phase created this directory.
Expected cause is missing staging parent, not proven by a server traceback.
Same-endpoint primary GPU and normal artifact consumer continue (eighthshard
half0 active); no session loss/reconnect/reallocation or scientific retry.
Sanitized failure receipt final_staging_failure.json preserves status/paths
without proxy credentials. Defer repair until scan completes: bounded CPU
finalization creates/verifies the owned input directory and copies SHA-guarded
leaf-staged wrapper bytes. Sealed finalizer/scientific source unchanged.

### 2026-11-08 15:55 UTC — 448 paired objects; primary−.178571pp

[Run/Partial/Open] Seven complete all15 paired shards/28 accepted jobs:
448 objects,6720 predictions/arm. SCD4438=66.041667%,half04410=65.625%,
M1004426=65.863095%. Primary−.178571pp,conditional object-cluster95%CI
[-.610119,+.238095]; M100-half0+.238095pp,CI[-.133929,+.610119]. No
operational gain. Fixed617 continues with169 objects remaining. Full partial448
SHA2ef8a1e9b79298c080fcb80c6c528dce9b3d8b7e645c8bf1394c315ce662b5fb
remains under derived/; CPU progress note records complete metrics and SHA.

Legacy-v1 scalar-comparator failures now2: the original runtime rejection at
shard001_half0 Background1 plus a new counterfactual legacy check at
shard006_M100 Background1 (idealFP64error1.3879207383382663e-6, orderedFP32
replayerror0). The latter is a retained diagnostic under deployed v2, not a
claim that v1 was rerun or passed. V2's numeric1e-6 remains unchanged and all
current v2 cases pass; acceptance-rule amendment remains explicitly posthoc.
Native29/30 repeat failure is unrelated and unresolved. Eighthshard half0 runs.

### 2026-11-08 16:11:46 UTC — transport503, WAIT_USER; no512/final617 acceptance

[Run/Failed transport/User instruction/Open] New consumer16246 exited3 after
Contents API returned503 ServiceUnavailable while reading existing
acks_basis_fp32_v2/shard007_M100.json before ACK publication. Raw local failure
WAIT_USER_20261108T161146.json SHA5566058ace74e821094cf1e48d07257fcef7248a8e2de461262be7910ae2f1cb
retains the masked error. Compact sanitized transport receipt is in v2 knowledge
evidence. This is separate from the earlier auxiliary missing-directory PUT500.
No new Colab CLI action/retry/reconnect/reallocation or GPU worker launch by
the parent after detecting WAIT_USER; explicit user rule requires a new command.
Runtime23310/owned r6 was not closed or interrupted. Local CPU supervisor stopped.

[Run/Preserved] There are31 exact local job receipts,30 ACKed and
shard007_M100 locally validated with ACK pending. Its949021975-byte raw ZIP
SHAf2690b46bc6863df989371b0ba1d933756a9106113955258404f3d9cd700293c
is physically retained on Mac and independently rehashed after this incident.
Seven complete paired shards448 objects remain the latest accepted accuracy;
no512/576/final617 analysis exists. Primary stream continued after503: shard007
SCD completed218.0475s and sealed122840940B/5ccadb5e; shard008 cache also sealed.
At16:14:29UTC actual controller reports max3 pendingACKs:
shard007_M100,shard007_control,shard008_cache. It waits at its safeguard;
this is no proof of VM loss and not permission to start a duplicate controller.
The existing monitor still owns the running primary stream and is preserved.

[Next action/User gate] Wait for a new user command. If authorized, first
reconcile the same endpoint/current primary stream and pending transport state;
recover the ACK/consumer path before any consideration of restarting inference.
All source918/7f/operator1ef/alpha4398/B32/617 assets and waiver remain frozen.
Preserve this failed attempt and all old v1/legacy/native diagnostics. Final
source/metadata archiving remains prepared, not executed: finalizer0a75212a,
parent wrapper9c16184c. Auxiliary input directory repair is deferred until scan
completion; no final617 proof/report/archive claim. Source640b60f/integration
81cd13c published the448 milestone while retaining peer integration2ce4e7f.

### 2026-11-08 17:35–17:41 UTC — user-directed archival and owned-r6 closure

[User decision/Run/Closed/Partial] Main owner thread
01a1b552-e131-7291-8f18-9d23a5153108 conveyed the user's explicit request to
close its completed speed runtime and any other Colab session. Coordinated
ownership assigned only this r6 to this agent. Authorization lifted WAIT_USER
for archival/shutdown only, without research recovery/resume/new GPU work.

[Run/Failed/Preserved] Primary controller had already failed17:13:30UTC on
`Local archive ack deadline exceeded; preserve all pending data`; CLI0 with
missing completion marker remains failed. It held no campaign lock during the
bounded file-only shutdown phase. Three pending remote/Drive ZIPs and transport
parts were hash checked without emitting ACKs. New Mac salvage:
007control122840940B/5ccadb5e and008cache59224783B/f0b21ddf, both wholeSHA/CRC.
Previously retained pendingM100949021975B/f2690b46 remains unchanged.

Shutdown metadata phase7e284d3a completed17:39:25UTC with exact marker/CLI0:
5732301B SHA14471f1341d80eb443673b4c798d503b959f6cef2b7599a9bc818f79dbd90e18,
118 members. Drive copy downloaded and wholeSHA/CRC/allmember hashes verified
on Mac. It preserves failed journal, compact33-archive index, pending receipts/
manifests, phase logs, source/assignment/alpha/amendment and exact archival code.
Local base result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/shutdown_20261108T173539Z/;
Drive result/gsd_lowband_quarter_fast_20261108_r6/shutdown_20261108T173539Z/.
All33 raw Mac ZIPs (16789346159bytes) were independently rehashed against the
remote shutdown index after retrieval. No old raw/receipt/rejection bytes changed.

[Run/Closed] Monitor23310 exited0 at17:40:59.565599UTC. Exact ownr6
`tta-gsd-quarter-fast-a100-20261108-r6` stop returned0/`Session terminated.`;
17:41:30UTC server inventory reports no active sessions. This owner did not
touch peer r5; its owner closed it separately. Compact runtime_closure_20261108.json
and shutdown_archive_acceptance.json are in current v2 knowledge evidence.
617 screen/final decision/external replication remain incomplete/unrun; latest
scientifically accepted paired result is still448 objects, primary−.178571pp,
conditional95%CI[-.610119,.238095]. No512/final617 analysis or positive claim.
Runtime is closed; any future scientific continuation requires a new user task.
