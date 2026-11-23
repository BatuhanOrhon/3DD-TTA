# CloudFixer XYZ baseline: three heads, paired PM and true terminal

**Güncel durum 2026-11-23:** Bilimsel all15 sonuçlar kabul edildi; [tam tablolar ve yorum](cloudfixer_baseline_multibackbone_results_20261123.md). Son406.9MB arşiv Drive→freshhost SHA/CRC doğrulamasını geçti; yayın, knowledge/source packet yedeği, cloudflush ve yetkili r25 kapatma bekliyor. Aşağıda başlangıçtan itibaren tarihlenmiş tarihçe korunur.

2026-11-23 [User report/Code/Open]. New task; no GPU results yet.
User requests a new Colab evaluation of the established CF-denoiser baseline
with different source heads, with/without Probability Mean, and comparison to
originalCF. Scope DGCNN/CurveNet/PointNeXt, full ModelNet40-C2468×15/s5, seed0,
B64/final36. Policy lockedN8/K80/tau.1/retained.95, pre-clean20/10+true-t0 equal
float64 probability mean. No-PM is the same trajectory's true t0, not N32.

Plan: `docs/superpowers/plans/2026-11-23-cloudfixer-baseline-heads.md`.
Approach chosen by operator under user's explicit no-extra-permission instruction:
reuse the existing multihead asset/Drive runner and exact proximal_policy via
an opt-in profile. Separate per-head trajectories would duplicate nine prior
calls; separate PM-off reconstruction would weaken the paired readout comparison.
The oldCF path and LION historical results remain separate and unchanged.

Current sourceHEAD2340685, clean at task start. Latest shared targetf4be3d4;
no active Colab sessions,460.52CU/0CUh at initial read. A100/High-RAM requested
as the continuing preference. New owned session planned
`tta-cloudfixer-baseline-heads-a100-20261123-r1`; no account switch/CLI upgrade.
Colab0.7.4. User-operated Drive consent may be the only external blocker.

Prelaunch estimate (not measured new-method throughput): canonicalsetup1–1.5h,
full baseline1–2.5h, total2–4h at8.88CU/h ≈18–36CU. Wholephase cap8h/80CU,
reserve5CU plus180s margin; all setup/retries/transfer charged. Measured B64
profile and maxpoints2048 capacity before full; independent deadline persists
through finalization. Reassess on material change, no automatic GPU-type fallback.

Fresh destinations: Drive `MyDrive/thesis/result/cloudfixer_baseline_heads_20261123/`;
local `result/modelnet40_c/cloudfixer_baseline_heads/`; preparation/evidence use
this taskdate. Every64 inputs plus full Python/NumPy/CPU/CUDA evidence persisted,
per-corruption/head ZIP/cursor published last, SHA/CRC/ID/label/logit/PM host
recount before reporting. Disconnect is not job death; inspect sameVM/process
and receipts before retry. Never mix profiles/campaigns or overwrite a run.

Accepted originalCF comparator: same checkpoints, frozen states, data, originalIDs,
s5/seed0/B64; campaign3596befa489b/source228555f. OriginalCF all15 results:
DGCNN71.277688%, CurveNet75.931929%, PointNeXt75.018909%. Original65.5MB raw/source
bundle SHA bcb64ea0c5c0165e43c4498dfc428af6c1e4804f82889e17d4924b76e38cfe07
exists on host/Drive. Do not rerun CF to replace available raw results. Pair by
actual IDs/labels/head states; nativeCF random optimizer and newN8 unit-fork
streams are not common draws. SameN8 PM/terminal do share geometry/noise.

This is a new B64/seed0 multihead measurement; previous Point2VecB1/seed2 and
Point-MAE development results are not numbers for these heads. No labels used
to choose the PM window or policy. No extraN32, LION, entropy/pose tuning or
Scan evaluation. Record per-readout cost; PM uses3 classifier forwards/head,
terminal only1, while the shared acquired trace includes all3. Saving terminal
from that trace does not measure a separate terminal-only wall-time run.

Standing user authorization permits all requested GPU/code/recovery work without
further approval and closing the owned runtime after completed results. Preserve
all peer/unknown/old-account runtimes. Academic records use [Code]/[Run]/[Inference]/
[Open] labels; final results/knowledge, independentreview, Drive cloudflush and
ownedteardown are all required before completion.

## 2026-11-23 — allocation and ownership observations

[Run/Open] r1A100HighRAM request503/noownedVM. Before r2 request, read-only
serverinventory showed new peer `tta-rotation-feedback-a100-20261123-r4`,
endpointgpu-a100-hm-ass1c0-ba9wk54844zv. It is protected; no execution/teardown
rights inferred from visibility. r2A100HighRAM request in progress. Shared
account use/balance must be freshly reread after allocation and before fullgate.

[Run/Decision] r2HighRAM also503/noownedVM. r3 requests sameA100 with standard
RAM under user's prior explicit low-RAM A100 allowance; no GPU-type fallback.
PhysicalCPU/RAM/capacity dictate nativebuildparallelism; scientificB64 remains
locked after its actual capacity gate, cost/time caps unchanged.

## 2026-11-23 06:06 UTC — seven A100 failures, authorized hardware revision

[Run/Decision] r1–r7 A100 allocation requests (HM andstandard) returned503;
currentaccountonlyprotectedpeerrotation-feedbackr4. NoownedVM/noGPUresults.
Operator retrylogger printedquota basedon traceback sourcecontext; only503 is
confirmed, not assignmentquota. OnefinalA100r8 attempt precedes preauthorized
L4 use if unavailable: user explicitly said “L4 ile de deneme yapalım” and
“benden GPU koşuları veya farklı bir konu için izin almana gerek yok”. These
standinginstructions cover the choice; no newpermissiongate. This is recorded
operatorjudgment, not a claimthat userselectedL4for thisexact newmeasurement.
Newmanifesthardware/native8.9/capacity/profile must be validated; oldA100CF
comparison is descriptive with hardwarevariationqualified. SamepolicyB64 and
head/data/prior fixed; noL4accuracy tuning. Rootowns allColab.

## 2026-11-23 — implementation and CPU verification

[Code] Source `8dfad44d8a475cf5ca9fb4f69364b8b1df2c56b5` adds the opt-in
`cloudfixer_xyz_proximal_pm_n8_k80_late_v1` package/profile. It calls the existing
Point2Vec proximal policy with a three-head classifier in the author output frame.
One nine-call trajectory produces pre-correction probes 20 and 10, then the
true extra-call t0. The archived terminal logits are the last candidate. PM uses
the float64 mean of the three candidate softmax vectors. The raw trajectory
fingerprint binds the input and mask hashes and the ordered candidate cloud
hashes to the unit, seed and policy.

[Run/CPU] 56 focused and 244 impacted checks passed, including synthetic B64
final-block/recovery, float64 PM versus terminal disagreement, and malformed
payload/cursor rejection. Independent recount of the 15 accepted original CF
ZIPs reproduced all 135 counts and matched 111,060 original ID, label and head
observations. Local L4/B64 package campaign
`505412b7853065655246a509be86eb54e84f333eb1050d98e7e565cac2704461` passed source
and archive integrity checks. [Full implementer handoff](../result/preparation/cloudfixer_baseline_multibackbone_20261123/task-1-report.md).

[Open] GPU memory, replay, native FPS/head execution and runtime remain
technical admission gates. A terminal readout from the retained PM trajectory
gives a paired accuracy measure and component timings, while a standalone
terminal-only wall time would require a separate timed run.

[User report/Open] User offers creating a newVM through browser instead ofCLI.
CLIretriesheld toavoidduplicates. Installed0.7.4has no builtinattach/import,
get_session requireslocalregistration; currentserverinventoryonlyprotectedpeer.
Same-accountbrowserassignment adoption needs exactuserprovidednotebook/endpoint
ownership andverifiedconnection beforeclaiming success. No browserVM attached yet.

## 2026-11-23 06:24 UTC — requested final threeCLIretries failed

[Run/Open] At userrequest A100r12HM/r13standard/r14HM attempted90s apart.
All3returned503, noownedVM. Retryingbatchfinished; no backgroundallocatorleft
tocreateanidleVM. Onlyknownpeerrotation-feedbackr4 observed; untouched.
Browserprovisioning offered as next externalresource path; notebook/ownership
andactualconnection mustbe verified. Code244CPUchecks passed8dfad44; independent
review found2ImportantRNG-boundary/expected-manifest acceptance gaps, fixround1
active. BothL4/A100packages built against8dfare pre-review snapshots andmust
notdeploy afterfixchanges withoutrebuilding. No newbaselineGPUresults.

### Review round 1 implementation

[Code/Run-CPU] Fix `ccd0a7c3c4bdf2f6259079a52ad3dad233d22fb5` binds every
baseline unit RNG boundary to the full seed-0 digest captured by the live
registered runtime. Host analysis requires the independently prepared
campaign SHA and validates registered digests across artifact shards. The
worker also checks that sequential classifier timings fit within measured
wall intervals. Whole-corruption wrong-seed and truncated Torch state
receipts, an internally consistent wrong source campaign and impossible
head durations are covered by regression tests. 61 focused and
249 impacted CPU tests passed. Earlier package bytes were preserved and
require rebuilding from the reviewed source before GPU use.
[Review fix handoff](../result/preparation/cloudfixer_baseline_multibackbone_20261123/task-1-report.md#review-round-1).

## 2026-11-23 — reviewed code preparation complete, runtime blocked

[Code/CPU/Open] Sourceccd0a7c fixes registeredseed0RNG boundary checks in raw/
prefix/completedZIP/host validation, independent operator `--expected-campaign`
binding and enclosingwalltime constraints. 61focused/249impactedCPUtests passed;
round1 independentreview SpecCompliance/Quality bothPass, noopenfindings.
Task1codecomplete; Task2GPUadmission/evaluation notstarted becauseallCLI
allocationsincluding userrequestedlast3returned503. Pre-reviewpackagespreserved
andmustnotdeploy. Freshpackagingfromacceptedsource follows. No scientific
PM/terminal accuracy claim andno ownedruntime toclose. Peer preserved.
Next: userbrowserA100sameCLIaccount/notebooklink or successfulnewallocation,
identifyownedendpoint/registerifneeded, read-onlyhardwareproof, canonical10
withcell1alone/sameVMreconnect, Driveconsent, realnative/smoke/replay/B64/maxpoints
profile andfreshresourcegate beforefullall15. Hostanalysisrequires expected
**canonicalmanifestcampaign**, notprettyJSONfilesha; comparewithsavedCFraws.

[User report/Run/Open] Latestuser steering explicitly renews CLIallocation and
authorizes both A100standardRAM andL4. Freshbounded alternatingrequestbatch
startsA100r15 thenL4r2; onlyoneownedVM,90sspacing, haltatfirstsuccess. Both
reviewedsource packagesv2 areavailable; codeccd0a7c, documentsnapshotd15a3c5,
allfileidentities/ZIPCRC checked. Exactcanonicalcampaigns in reviewed_packages_v2.json.
No newGPUresults or implicitpermissionquestion; browseroption deferred peruser.

## 2026-11-23 06:43 UTC — authorized A100standard/L4 batch exhausted

[Run/Open] Userexplicitly renewed allocation and authorizedA100normalRAM/L4.
Requests A100r15,L4r2,A100r16,L4r3 (90sspacing) all returned503. NoownedVM,
allallocatorprocessesended; noidleGPUleft. Protectedrotation-feedbackr4
untouched. 249CPUchecks and cleanindependentreview; final A100/L4packagesv2
source/document snapshotd15a3c5 (reviewedcodeccd0a7c) fullyhash/CRCverified.
A100canonicalcampaign810bb93f4037450d47254ba54f8722cce6d4aa4770fb0d253c630402476bc915;
L4canonicalcampaignb4964a9d9ade18038f0927f8083879c3dc840347157f336b022036e04dec701d.
Oldpackages8dfmarkedobsolete. GPUadmission and accuracy runs neverstarted.
Nextresourcepath: successfulnewallocation or same-accountusercreatedbrowserVM,
with exactendpointownership/connectionverification beforecanonical10. CLIbrowser
adoptionnotyetverified; neverexecuteprotectedpeer to test it.

## 2026-11-23 07:04 UTC — user-authorized three-minute A100 polling

[User report/Run/Open] User explicitly requests periodicnewA100 allocation
at3minuteintervals. Watched single-writer allocator will request standardRAM
A100 at180s start-to-start, haltatfirstobservedownedassignment, and preserve
all peer/unknown sessions. Rootdoesnotyieldanunwatchedallocator; allocation
result/profile/cost andcanonical/consent transition remainactivework. No new
GPUaccuracy yet; reviewedcode/packages unchanged. Request failuresdo notmean
aVMwascreated; inspectinventoryaftereachrequest andstoponambiguity.

## 2026-11-23 07:41 UTC — A100 r25 allocated, canonical setup active

[Run/Open] Three-minute standard-RAM A100 polling stopped automatically after
attempt9 succeeded at07:32:15UTC. Owned session
`tta-cloudfixer-baseline-heads-a100-20261123-r25`, endpoint
`gpu-a100-s-ass1c0-2al4sshtzsyaj`, root task/branch/worktree unchanged.
Actual NVIDIA A100-SXM4-40GB/40960MiB,12CPUs,87528500kB physical RAM verified.
Server peer rotation-feedbackr4 remains protected and untouched. Allocation
receipt is `result/preparation/cloudfixer_baseline_multibackbone_20261123/allocation_success.json`.
First read-only Jupyter probe lost its localWebSocket before writing its
receipt; sameVM console and transport60 reconnect verified the runtime. No
duplicate allocation. Canonical cell1 exact DONE marker observed after Conda
restart; cell2 sameVM condacolab.check passed and environment creation is active.
Drive mount user consent pending on localTTY30053; no inference yet.
New task Drive root `MyDrive/thesis/result/cloudfixer_baseline_heads_20261123`.
Prepared A100v2 reviewedcodeccd0a7c/source snapshotd15a3c5, canonical campaign
810bb93f4037450d47254ba54f8722cce6d4aa4770fb0d253c630402476bc915.
Whole-phase8h/80CU caps count from07:32:11UTC allocation request, reserve5CU
and180s; fresh balance446.69CU/total15.61CUh (twoassignments), owned rate
approximately6.73CUh if peer rate unchanged. Actual all15 ETA remains unknown
until measured B64/maxpoints smoke/capacity; no full run before that gate.
Setup MAX_JOBS=4 chosen for measured12CPU/83.47GiBRAM. Each B64 unit and complete
corruption will be persisted to Drive and independently retrieved/validated.
User authorizes closing this owned runtime after completed tests, archived
results and knowledge; account consent remains user-operated.

## 2026-11-23T07:50:21.629806+00:00 — exact remote package and canonical2 environment verified

[Run/Open] All seven A100v2 package members match reviewed receipt SHA
0e745c7598c6317ba8cf0cc7892d3732f90faad2c2c24a39fa8ea5a7acb18413
on ownedr25; both ZIP CRCs pass and remote receipt was downloaded to host.
98MB single-file upload failed BrokenPipe before a usable target appeared;
12pieces of8MiB(orfinalshortpiece) were joined without replacing existing
files and exact originalheadZIP SHA16cca72d...cb8e6 verified. Immutable source
package/manifest/campaign unchanged. Canonical2 finished only at known
fastrlock wheel error (calmsize/clip built); actual conda Python3.8.20 and
preparationcheckouta458cd46bba1288a3f41302548bc777700cbc7d4 independently
verified. Only this user-authorized error is deferred; canonical3–10 and
GPU/native/data/checkpoint admission remain required.
Sharedtarget fetched to d93d9c4; peerrotation changes must survive latercurated
integration. Latestserverinventory contains onlyownedr25; root neverintervened
inpeer lifetime. Freshaccount444.43CU/6.73CUh/1assignment.
No newaccuracy; nextcanonical3 requires freshphysicalDrive mount after
user-operated Googleconsent alreadyrequested (TTY30053). Ownedr25 preserved.
Proof `knowledge/evidence/cloudfixer_baseline_multibackbone_20261123/r25_package_cell2_verified.json`;
fullcanonical/upload logs and exactsources remainunder taskpreparation.

## 2026-11-23T08:01:18.847809+00:00 — operational preparation reviewed; blocked on Drive consent

[Code/CPU/Open] Seven local operator scripts prepared from accepted earlier
workflow and bound to exact newN8 profile/registeredseed0/frozenstates. First
independent review found3Important recovery gaps: alternate-label bootstrap
concurrency, Drive-only old fullclaim reconciliation, and expiredwindow or
transient terminaldownloads. Round1 fixes all3; four targeted regression
cases and Python3.8 AST/compile pass. Scoped independentre-review SpecPass/
QualityApproved, zeroopenfindings. Root readback verifies six uploadedremote
script SHA256s against final reviewedlocalbytes; nooperator/native/GPUrun
entrypoint invoked. Canonical3–10 stillnotstarted.
Local source/log/receipt preparation ZIP `result/preparation/cloudfixer_baseline_multibackbone_20261123/operational_preparation_r25_v2.zip`
(196647B/SHA195f0ab976cde4436ef25751a8457acb009f55bdd26b96b23b9527df1b8668cc) passesCRC; Drivebackup
pending mount. NewCF-XYZ threehead accuracy remains unmeasured.
Actual user-only blocker: freshDriveGoogleconsent requested via question;
localTTY30053 waits foruserconfirmation, rootmusthandleEnter, thenphysical
mount/knownsharedasset checks andtaskwrite-readback. Next execute canonical3–10
MAX_JOBS4; backupphase1/2 receipts andoperationalZIP, copyonlySHA-verified old
diffusioncache ifavailable, thenreviewedbootstrap→technical→measuredresource
gate→all15→hostrecount/Drivearchive/knowledge/publication→authorizedclose.
Ownedr25 remainslive; nofurtherallocationpoller.8h/80CU countsfrom07:32:11UTC,
includingallwait/setup; reassesstime/balance beforeinference,neverresetspent.

[Run/Open] 2026-11-23T08:15:07.685468+00:00: user requested renewed Drive link and live-status proof.
Freshserverinventory lists exact ownedr25 endpointgpu-a100-s-ass1c0-2al4sshtzsyaj;
sameVMconsole confirms own setupreceipt and physicalDrive is notmounted.
Stale localmountCLI PID31935 retired onlyafter exactargumentidentity check;
VM/kernel notterminated. Fresh `colab drivemount -s` consent waitsTTY33601;
oldTTY30053 isobsolete. User mustcomplete newGoogleconsent, root handlesEnter,
thenphysicalDriveverification beforecanonical3. No newinference or accuracy.
OAuthlink/challenge deliberatelyexcludedfromknowledge. Setup/costclocknotreset.

## 2026-11-23T08:17:22.399057+00:00 — user requires measured batch efficiency

[User report/Decision/Open] User explicitly asks to use memory efficiently
when choosing batch_size. B64 is the current verified-code registration, not
a measured hardware optimum. Current ownedVM has40GiB GPU memory and
83.47GiB systemRAM; GPU working-set and end-to-end examples/sec determine
batch choice, with hostRSS/availableRAM also checked.
After user-operatedDriveconsent and canonical/native setup, first measure
the registered B64 Gaussian and worst-cardinality Upsampling2048 path.
If measured headroom allows, compare boundedB96/B128 resource-only probes
using the exact fixedN8/threehead computation (warmup then timed repetitions).
Record allocated/reserved peak VRAM, actual free/total VRAM, hostRSS/availableRAM,
wallseconds andexamples/sec. Choose by throughput withmemoryheadroom; no
accuracy/label-based selection andnochange toN8/K80/tau/.95/PMprobes.
A higher batch is admitted only after real native/capacity measurements and
reviewed batch-aware manifest/runner/analyzer/recovery support; currentB64
checks cannot just be bypassed. If largerbatch is chosen, newimmutable
manifest/campaign and full all15 atone commonbatch are mandatory; keepB64
technical artifacts separate and disclose originalCFreference usedB64.
No fullbenchmark has started; noRAM-efficiency or optimum claim is established.
DriveconsentTTY33601 remains the currentuser-operated blocker.

[Run/Open] 2026-11-23T08:37:29.955614+00:00: user saidcontinue. Enter on oldmountTTY33601 produced
TimeoutError, no successmarker; sameVMconsole physicalDriveFalse/MyDriveFalse.
Sameownedr25 serverassignment live. FreshmountTTY68525 asksrenewedGoogleconsent;
asyncquestion delivered, no accountapproval inferredfromcontinue. PriorMountCLI
exited1; no research/nativephase started or duplicated. Newprotectedpeer
`tta-pca-all15-a100-20261123-r4` endpointgpu-a100-s-usc1b0-21ic5pfeiqb4w
observedread-only, untouched. Shared436.65CU/13.46CUh/twoassignments; ownlast
confirmed6.73CUh. Allocationanchor07:32:11UTC and8h/80CU unchanged.
Resource-onlyprobe preparation nowassignedbaseline_batch_probe from ignored
batch-probe-brief.md; existingreviewedproductionB64package untouched until
real measurements justify separately reviewednewregistration. Noaccuracy.

## 2026-11-23T08:56:56.721134+00:00 — Drive consent resolved and canonical setup proceeds

[Run] User confirmed newGoogleconsent. Credentials propagated, first mount
reportedmountfailed; sameVM inspectedphysicalFalse/noactiveDriveFS research
work, renewedmountreusedcompletedconsent withoutnewprompt andreturnedMounted.
Independentphysicalmount/taskwrite-readback/sharedassetchecks pass. SavedCF
65528957BarchiveSHA bcb64ea0...cfe07 verified; noresultsrerun.
Initial40member196647B operational/source/logZIP persistedDrive, downloaded
freshhost, SHA195f0ab9...68cc/CRC pass. ThisisDriveFSreadback, notindependent
cloudsyncproof. Exact159801477B priorb64841d0...5a5a copied fromownedoldcache
tonewtaskDrive withhashchecks. Canonical3 exactDONE/exit0/phasebackup pass;
4running onprimarydriver15316, MAX_JOBS4. The firsthostdriverfailed onlyat
metadatakey lookup beforesubmittinganyremotephase; preservedfailedsource/log,
v2schemafix afterall10sources/wrappers byteexactcheck. No duplicatecell3.
NoGPUaccuracyyet. Next4–10/native/smoke/replay/B64capacity, independently
reviewedresourceprobes, measuredbatchdecision/newcampaignifneeded, fulltest.
Source/currentcontinuation andDriveevidence recordedinr25_drive_and_canonical3.json.

## 2026-11-23T09:12:47.261474+00:00 — native setup and resource probe preparation

[Run/Code/CPU] Canonical4 prepare,5 EMD,6 Chamfer passed exactwrappermarkers
andphysicalcompletionfiles; sources/stdout/receiptsbackedDrive. Canonical7
firstCLI lostWS inCWDprelude(execution.py217), beforephasebody. Registered
kernel independentlyidle viaJupyterGET, noactivecompiler/PointNetmarker;
retiredonlyownlocalfailedCLI44247, preservedfailurelog/exit−15. SameVM
retry01 ofexact7SOURCE passedpointnet2.complete andDONE;8PVCNNactive.
NoVMreset/peerintervention orduplicatedPointNetbuild.
Resourceprobe initially3reviewissues, round1addressedall3 butnewpre-spawn
no-launchreconciliationgap; round2fixedandindependentreviewSpec/QualityPass,
zeroopenfindings.14CPUchecks. AcceptedprobeSHA
a62fb66fc2c4c0777a102d3a913e05fb1758e00c5f1efce9b438a1c7394a6a25.
7-member25449B source/test/reviewZIP SHA
6404f00f1bb1c5e897da6d3b699cc26d742c8579da0f75476a4de143b6ff2f0d
passesDrive→freshhost SHA/CRC. Actualresourceprobes requirecanonical10/native/
B64technicalfirst; no newGPUaccuracyorbatchoptimalityclaim.
At09:05–09:06 freshaccount429.94CU/13.46CUh/twoassignments; own6.73CUh
lastconfirmed. NativebuildMAX_JOBS4.8h/80CUallocationanchor unchanged.

## 2026-11-23T09:23:34.913347+00:00 — canonical10 verified, extra native bootstrap active

[Run] All10 exactnotebook SOURCE bodies completed inrequiredorder, sameVM
aftercell1 Condarestart; onlycell2knownfastrlockfail deferred afteractual
Python3.8.20 proof. EMD/Chamfer/PointNet++/PVCNN/FPS-gatherforward/backward
physicalmarkersandchecks pass.15severity5 arrays+labels (16registeredfiles)
allSHA-matchA100v2; Torch2.1.2+cu121,NumPy1.21.2,CUDA12.1,A10040GB.
Canonical9firstpreludelostWS beforebody; noactiveunzip/datafile, registered
kernelidle. Oldlocalhardwareprobe31642and9CLI44767 wereleftaliveaftertheir
exceptions; retiredonlytheseLOCALclients, preservedfailurelogs. SameVM9–10
retry02 passed. ExtraorphanWSclientinterference ispossible, notproven.
Canonical53member133133Bsource/log/receiptZIP SHA
2d13ececafe750a447c6a1d1787fd65279ee77a74782762bc10f4c6297661729
passesDrive→freshhostSHA/CRC; noglobalpipconsistencyclaim. Generalnotebook
LION/PointMAE assetchecks do notmakeLION partofCF-XYZ experiments.
Source-bound shortstarterlaunchedbootstrap01 PID64845, detachedConda3ddenv;
log `/content/cf_baseline_bootstrap01_driver.log`. ExactA100v2package/ownsession
checks precedecompile; pinnedPyTorch3D2f11ddc andChamferf86f6f7 build/actualGPU
checks precedesetupreceipt/deployment. CurrentPyTorch3Dcompileactive.
Nextactualnativecompletion/exit thenstart_technical01.py → B64smoke/replay/
profile/maxpointscapacity → resource-onlybatchchoice → fullgate/all15.
Noaccuracyoroptimalbatch claim. Global8h/80CUanchor unchanged.

## 2026-11-23T09:35:31.175845+00:00 — real B64 technical admission and resource screen

[Run] Extraofficialnativebackward/KNNGPUprobe passed; actualbootstrapchild
exit0. Detachedtechnical01 PID74494 passed sixsmokeunits (Background/Rotation/
Gaussian,4objectblocks), freshprocesscache replay withidenticalpayloads and
zeronewreservations, thenGaussianB64 andUpsampling2048×B64 validation.
Allprior/headstates,seed0boundaries,candidate20/10/true0,9denoisercalls,
float64PM andtrue-terminal checks passed inrawunits. Runtime signatureSHA
7d35f4704bee15fa149069de47c18522773ae371c8cb60ea0f404991dcadf250.
Gaussian64 operation7.628720s,allocatedpeak5102.585MiB; Upsampling64
operation15.624514s,allocatedpeak18173.398MiB. These are initialtechnical
capacity observations, notall15 throughputoraccuracy.
Resource-onlysteadyB64Gaussianpassedactualexit/settlement/sourcebindings;
timedwall7.053s/7.073s,peakreserved9.16GiB. B64Upsamplingprobe active.
Eachcandidateusesfreshsharedaccountsnapshot; currenttestedsourceunchanged.
No fulltestlaunch oraccuracy-basedselection; 96/128stillunmeasured.

## 2026-11-23T09:58:56.570089+00:00 — Ölçüme dayalı B96 seçimi ve yeni kayıt

[Run/Code/CPU] Gaussian ve en büyük Upsampling girdisinde her aday için bir
ısınma ve iki zamanlı tekrar kullanıldı. Süre; veri aktarımı, üç donmuş model,
N8 proximal akış, ham doğrulama ve Drive kaydını kapsıyor. Seçim yalnızca
süre ve bellek değerlerinden yapıldı.

| Batch | Gaussian örnek/s | Upsampling örnek/s | En düşük GPU bellek payı | Karar |
| --- | ---: | ---: | ---: | --- |
| 64 | 9.061610 | 4.258317 | %48.3006 | Uygun referans |
| 96 | 9.853039 | 4.358689 | %23.7565 | Seçildi |
| 128 | Gaussian tamamlandı | CUDA OOM | Ölçülemedi | Uygun değil |

B96 için geometrik ortalama hız6.553345 örnek/s, B64 için6.211861 örnek/s;
artış yaklaşık%5.50. Önceden kaydedilen%15GPU/%10sistemRAM payı ve%5B64
yakınlık kuralı kullanıldı. B96 sistemRAM payı en az%95.17. B128Upsampling
ısınmasında16.02GiB tahsis isteği CUDAOOM verdi; gerçekçocuk çıkışı1, zaman
aşımıFalse, rezervasyon/settlement/hata kaydı korundu. Aynı başarısız etiket
yeniden çalıştırılmadı. Bu karar test edilen üç aday ve bu ortam içindir.

Yeni üretim kodu `a979bce`: B64/B96/B128 profil kapsamı, manifestten türetilen
blok/arşiv sayısı, eskiB32/B64 uyumluluğu ve yanlışbatchcache reddi.
BağımsızSPEC/QUALITYPASS;0bulgu.272ilgili ve1110CPUtesti geçti,2skip.
N8/K80/tau.1/.95, PM20/10/true0, üçhead, checkpointler ve matematik sabit.
B96 tamtest26blok/corruption,390blok/15corruption,sonblok68örnek.
Seçili manifest kampanyası `e8df74eb3976517f74452c49363df953ac5d103e4ff0cc6fb265d327769b0427`,
sourcea979bce ve taşıma makbuzuSHA`48f93eab2f5891eafca6e985c18a7a51fc17fea7ab1a5884a1d3f1293dcae988`.
98MBheadZIP'in tüm üye baytları yeni paketle karşılaştırılıp aynı olduğu
doğrulandı; mevcut VM kopyası yeni taşıma makbuzunda kullanıldı. Manifest
kimliği bu taşıma tercihinden etkilenmedi. YeniROOT/Drive `selected_b96`
alanları eskiB64 teknik ve kaynak ölçümlerinden ayrı. Nativekaynak/ELF/ortam
kimlikleri kontrol edilerek aynıVM kurulum fazı yeniden kullanılacak;
paketin gerçeknativeGPU kontrolü ve yeniB96smoke/replay/profil/kapasite
kontrolleri tamtestten önce zorunlu.

OriginalCF referansıB64; yeniB96 blokları geometri RNG seed'ini değiştirir.
Kıyas, aynınesne/etiket/başlık/checkpoint üzerinden farklıyöntem akışlarını
karşılaştırır; ortaknoise iddiası yoktur. AynıB96 koşusunun PM/terminal
okumaları tek geometri/noise akışını paylaşır. Tamtest henüz başlamadı.

## 2026-11-23T10:13:52.885463+00:00 — B96 tam test başladı

[Run] SeçilenB96 kampanyasıe8df74eb...b0427, sourcea979bce, sameownedr25.
Yeni nativeGPU kontrolü/gerçekchildexit0, altıB4smoke, ayrıprocessaynıcache
tekrarı,Gaussian96 veUpsampling2048×96 geçerli. YeniGaussian96profil ile
öncekiB96kaynakölçümünün source/candidate/terminal logits, PMve adayprob,
input/mask/cloudhash ve geometryseed alanları birebir aynı; manifest/unit
kimliği ayrı tutuldu. RuntimeSHA7d35f470...f250aynı.
Sonkapasite:Gaussian10.331127s/96,peak7542.946MiB;Upsampling22.555571s/96,
peak27149.165MiB. Kaynakbellekölçümü75üye/8287577B arşiviSHA
0682646cf50fdb82a99526da411e0a4b4a4dfefc208c079c4f21721574d380a5
Drive→hostSHA/CRC ve15hamunit/1344tekrar+ısınma gözlemiCPUdoğrulamasını geçti.

Tamworker10:10:53.725143UTC başladı. GerçekworkerPID106919, supervisory
CondaPID105783; yerellog `/content/cf_baseline_heads_a100_b96/full01_worker.log`.
Tam2468×15/s5/seed0, üçhead,390blok,sonblok68. Her96girdi/RNGDrivecommit,
hercorruption26bloktamamlanıncaSHAZIP/3headreceipt/cursorlast; hostcollector
aktif (tool4699), `result/modelnet40_c/cloudfixer_baseline_heads/e8df74eb3976/retrieval_20261123T101143Z`.
Başlangıçkapısı:417.03CU,6.73CUh/tekassignment,temkinliETA11430.84s≈3.18h;
fundeddeadline13:20:32UTC.8h/80CU/reserve5CU+180s başlangıç07:32:11UTC
korunuyor; önceki aşamalar bütçeye dahildir. Öncekimeslektaşruntime'ına
müdahale edilmedi; güncelaggregate6.73CUh gözlendi.

İlkUniformblokları kaydedildi; henüz tamcorruption/all15 accuracy yok.
Asıl çıkışdosyası full01_worker_exit.json, DriveI/O'danönceyazılır.
CLI kopması yenidenbaşlatma nedeni değildir; aynıPID/log/receipt incelenir.
Tamtest/recount/kıyas/arşiv/knowledge/publication tamamlanınca kullanıcının
mevcut talimatıyla yalnızownedr25 cloudflush ve kapatma yapılacak.

## 2026-11-23T10:33:08.155034+00:00 — ilk dört corruption güvence altında

[Run/Partial] Aynı owned A100r25/full01 devam ediyor. Uniform, Gaussian,
Background ve Impulse: her biri 26 blok, 2468 özgün ID ve üç donmuş head;
Drive ZIP/cursor/head receipt, SHA/CRC, logit/probability/etiket/ID/RNG/frozen
kontrolleri host collector tarafından bağımsız doğrulandı. Toplam 104 blok,
9872 corruption gözlemi/head güvence altında; bu all15 sonucu değildir.
Upsampling aktif; yeni inference veya tekrar koşusu başlatılmadı.

[Code/CPU] Latest shared target 24be2e9 üzerinde yeni task-owned integration
checkout hazırlandı; yalnız beş gerekli runner/packager/analyzer/setup/test
dosyası seçildi. Ortak cloudfixer_diffusion.py ve peer çalışmaları korundu.
159 ilgili CPU testi geçti; eski deney sampler'ı ile shared default-path
arasında sekiz synthetic CPU durumda cloud/metadata/RNG birebir eşit.
Bu gerçek CUDA/checkpoint replay değildir; production GPU source a979bce
değiştirilmedi. Evidence: integration_default_sampler_parity.json.
Tam test/recount/arşiv/knowledge/publication ve yalnız ownedr25 kapatma bekliyor.

## 2026-11-23T11:02:29.007306+00:00 — 10/15 tam corruption ve son kanıt betikleri

[Run/Partial] Aynı full01/A100r25 devam ediyor. Uniform, Gaussian, Background,
Impulse, Upsampling, Distortion RBF, RBF inverse, Density, Density increase
ve Shear: her biri 26 blok/2468 ID/üç head; toplam 260 blok ve 24680
corruption gözlemi/head Drive→host bağımsız SHA/CRC/ham logit/etiket/ID/RNG
doğrulamasını geçti. Rotation aktif; all15 ortalaması henüz yok.

[Code/CPU] Son arşiv/rapor betiklerinde ilk bağımsız incelemenin beş önemli
bulgusu tek düzeltme turunda giderildi; kapsamlı yeniden sayımın sealed ZIP
ve tabloya bağlanması, sabit source/probe kimlikleri, güvenli allowlist ve
kesintiye dayanıklı değişmez yayın kontrolleri eklendi. Sekiz CPU regresyon
kontrolü, Python3.8 AST ve gerçek Uniform ham-arşiv kontrolü geçti. Yeniden
inceleme Spec/Quality PASS, açık bulgu0. Betikler henüz tam-test sonrası
çalıştırılmadı. VM'ye yüklenen capture source SHA
99e265f55746fb8e7213ccc1a538065f4af4690f26e51f49a7931f13d85a5b58
yerel kaynakla eşleşiyor. Full worker/collector ve bilimsel yöntem değişmedi.

[Run/CPU] Ortak branch hazırlığında 159 ilgili +15 rotation CPU kontrolü
geçti; peer diffusion/pose dosyaları korundu. Yalnız beş gerekli kod/test
dosyasının integration hazırlık commit'i6ae0338; yayın henüz yapılmadı.
Kanıtlar: partial10_validation.json, final_evidence_helpers_acceptance.json.
Sonraki adım: kalan beş corruption ve actual exit0, terminal kaynak/receipt
arşivi, tekrar ham sayım/final arşiv, Drive→freshhost doğrulaması, ayrıntılı
knowledge, son bağımsız review/yayın, official cloudflush ve yetkili ownedr25 kapatma.

## 2026-11-23T11:31:46.284956+00:00 — all15 bilimsel sonuç kabulü, son Drive taşıması bekliyor

[Run/Validated] Full01 gerçek exit0;15/15 corruption,390blok,37020gözlem/head,
135accuracy+135pairedsatır hostcollector ve ikinci ZIP'ten ham yeniden sayım
ile doğrulandı. Denetimli tam tur4500.207551s≈75.0dk. Kaynaka979bce,
B96/seed0/s5, önceki bütün GPU hazırlık süreleri allocation07:32:11'e dahildir.
DGCNN source58.835764/terminal70.680713/PM71.139924/CF71.277688%;
CurveNet64.011345/73.841167/74.243652/75.931929%;
PointNeXt65.837385/73.987034/74.278768/75.018909%.
PM kendi N8 terminalini üçünde de yükseltir; CF toplamda üçünde de öndedir.
Diğer13'te PM net+689/+120/+451; Background+Rotation CF avantajı740/745/725.
Ayrıntılar: cloudfixer_baseline_multibackbone_results_20261123.md.

Son bundle406927870B SHA c392dce804c213a21e97769590e3a036ee80deefb9b0d6379689958530646265,
`result/modelnet40_c/cloudfixer_baseline_heads/e8df74eb3976/final_20261123T112717Z.zip`;
7zorunludosya,15değişmez ZIP,exactsource/head/native/komparator/probe/OOM
kanıtları içeriyor.154üyelik terminalops1920976B ZIP SHA
03731758a98355a7233863f756d02f8ab76efc25b5c06da23b9bd5f7ccec7dbc
Drive→hostSHA/CRCveactualcaptureexit0 geçti. Nihai bundle tekparçaCLIupload
HTTP500 döndü; aynıVMls ohedefinoluşmadığını doğruladı. Bilimsel hamlar
vehostfinal güvenli.8MiBparçalı immutabletaşıma hazırlığı sürüyor; sonbundle
Drive→freshhostSHA/CRC,knowledgepublication,officialcloudflushveyalnızownedr25
kapatma henüz bekliyor. Yeniinference/tuningyok.

## 2026-11-23T11:59:34.084239+00:00 — nihai Drive→host arşiv ve bütün-branch review geçti

[Run/Validated]49parça+manifest+receipt+publisher,52CLIinvocationactualexit0.
VMherparçayı ve406927870B birleşimSHA'sını kontrol etti; stagedimmutable
Drivepublishactualexit0. Freshhostilk90sGETsınırınatakıldı; aynıimmutable
Drivekaynağından600sboundileikincideneme137.494s'deCLIexit0 döndü.
SonhostZIP406927870B/SHA c392dce804c213a21e97769590e3a036ee80deefb9b0d6379689958530646265,
141safeuniqueüyedeCRC geçti. Yerelsonkopya:
`result/modelnet40_c/cloudfixer_baseline_heads/e8df74eb3976/drive_download_20261123T1155Z/final_20261123T112717Z.zip`.
Drive `MyDrive/thesis/result/cloudfixer_baseline_heads_20261123/selected_b96/`
altındaaynısonZIP+originalreceipt+DriveFSreadbackproof mevcut.
Bu fizikselDriveFSreadbackvefreshhostdoğrulamasıdır; officialcloudflush henüzbekliyor.

[Code/Review] Bütüncuratedbranch6f4abdd..ebe787f bağımsızreview:
0Critical/0Important/1Minor. Seed_for yorumununprofilegöreifadesi
koddan doğrulanıp yalnızyorumdüzeltildi(root4d89c14/sharedprep464f547);
ASTparentilebirebir, scopedrereview0Critical/Important/Minor,ReadyToMerge.
Bilimsel sonuç/kod davranışı değişmedi. Kaynak/Gitbilimselkayıtlarhazır;
yayın, knowledge/sourcepacketDrivefreshproof,officialflushveyetkiliownedr25
kapatma sıradadır. Kanıtlar: final_drive_verification.json,whole_branch_acceptance.json.
