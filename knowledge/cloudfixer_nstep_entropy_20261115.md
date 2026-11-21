# Parallel CloudFixer N-neighborhood and entropy exploration — 2026-11-15

[User decision] Two parallelColab roles: N6/7/8/9/10/11/12 aroundN8, and dense
N8perstep classifier/entropy/terminal error exploration. PreserveincrementalDrive
andresume. No additionalduration approval withinthisscope. PreviousCPU-close
permission does notauthorize teardown of these newGPUs.

[Design/Plan] docs/superpowers/specs/2026-11-15-cloudfixer-nstep-entropy-design.md
anddocs/superpowers/plans/2026-11-15-cloudfixer-nstep-entropy.md record choices,
interfaces, caps, recovery andseparateownership. Operatorcall: same128IDs/all15/
s5/seed0/B32,K80,tau.1,p.95. N sweep420units; denseN8 60units with9cleanx0
candidates. FulloldN8canonical4logit parity required; no entropyweightedadaptive
policy/weight tuning. This is exploratory reusedtest-subset analysis.

[Evidence] AcceptedoldN8K80tau.1 PM65.104167%, terminal66.093750%,208.114310s;
bestoldN32PM65.9375%,598.874860s. MeasuredN8vsN32pipeline~2.88x, not4x;
N+1actualdenoisercalls9vs33 andfixedclassifier/storagecosts matter.

[Wholephase] RootsoleCLIowner; intendednewA10080HighRAM sessionnames
`tta-cloudfixer-nstep-a100-20261115-r1` and
`tta-cloudfixer-entropy-a100-20261115-r1`. Perrole8hcapincludesallsetup/recovery/
archival; conservativewholephaseA2–4h,B1.5–3h; provisionalcombined31–62CU
at8.88CU/hour/GPU, actualtariffunknown. Canonical10setupcells pernewVM,
onlyknowncell2fastrlockfailuredeferred;CFnative/assets afterward. Drive mounts
userconsentifneeded. No newVM allocated ornewscience result atthisentry.

[Delegation] DeveloperA sampler/runtimeopttrace; DeveloperB worker/analyzer.
Rootsource/technicalreview, twoVMsetup, execution, archival andknowledge.
No agentmaytouchmainusercheckoutorpeerVM. Sourcebranchremainsownresearch
worktree at524ab70beforethisplanningcommit; code snapshotsealed afterCPUgates.


[Run/Availability] Initial inventory0active; account481.05CU,0CU/hr.
First allocation attempts forbothplannedA100HighRAM sessions returned503Service
Unavailable; inventoryafterward0active, no source/environment/science executed.
Root retriesbounded allocations while implementation proceeds, preserving
session ownership andrecordingfailure. Denseclassifiercanonical4callorder must
be preserved before extras, then logitsrestored to densechronological order;
revieweridentified authorunusedmaskRNG consumption as aparityrisk. AgentA
adds a stochasticclassifier fixture to enforce this rather than assume purity.


[Availability/Operatorcall] SecondA100HighRAM requests forbothroles also503;
no activeassignment. TryA100withstandardhostRAM under distinctnstepr2name,
preservingGPUmodelandscientificidentity; hostRAM must be checkedbeforelarge
setup/build. Ifsuccessful, entropy getsitsowndistinctA100standardname too.
No automaticGPUarchitecturechange orparity-tolerance relaxation.

[CPU] Task1opt-indensetrace/canonical-firstclassification passed62focused,
existing-core962pass/2skip (worker/analyzerinprogress excluded); noGPUevidence.
Densehelperisminimal, defaultoldfourcand contractunchanged.


[Run] FirststandardhostRAM allocation succeeded:
Nstepownedtta-cloudfixer-nstep-a100-20261115-r2,
endpointgpu-a100-s-ass1c0-24gmixfofyxkt,A100/Standard/GPU;
rootmonitor6704started09:35:48UTC, soleCLIexecowner, livenessafter36000.
Canonicalcell1submitted throughmonitor; verifyexpectedConda restart andsameVM
beforecell2. EntropystandardA100r2allocationinflight. TwoearlierHighRAM503s
perrole preserved. GPUarchitectureunchanged; no parity tolerance relaxation.


[Run] Nstepcell1and sameVMreconnectverified09:39:30UTC. ActualGPUis
A100-SXM4-40GB(40960MiB), hostMemTotal87528500kB(~83.5GiB), sufficientfor
prior9.3GiBpeak. Compute-architectureSM80unchangedbutoldgridwasA10080GB;
all60oldN8parity remainsmandatory andcrossdevice latencywillnotbeassumed.
Condacolabcheckpasses; rootkernelPython3.13.16/conda26.3.2; canonicalcell2
creates isolatedPython3.8env andisnowactive. Entropy standardA100attempt2
also503; noentropyassignmentyet. Standardfallbackaddresses80GBpool shortage
withoutlooseningnumerical/readoutgate.


[Run] Entropy secondstandardA100 allocationattempt3succeeded:
ownedtta-cloudfixer-entropy-a100-20261115-r2,
endpointgpu-a100-s-usc1b0-1c2f26jgoty7n,A100/Standard/GPU.
Rootmonitor57884started09:42:50UTC; canonicalcell1active09:43:37UTC.
Currentaccount480.34CU/2assignments,total13.46CU/hr (~6.73each inferred).
Newhardwarelatencywillbecomparedwithinrole, notassumedmatchedwithold80GBgrid.
NoGPUinferenceyet; canonicalsetupbothroles underway.

[Code/CPU] Bothimplementationtaskscomplete; freshindependentreviewfound no
Critical/Important/Minor issues aftersource/N8smoke+provenance fixes. Root
fullproject tests977pass/2skip/5existingwarnings in49.37s; focusedreview44pass.
Default4candidate behavior retained; densechronological9withcanonical4first
classifiercalls; old4PMargmaxneartie parityexplicit. Newworker420/60roleunits,
separateimmutableledgers, strictsource/N8repeat, all60oldN8parity, standard
artifacts, offlinefullcoverage/entropy/calibration/oracle diagnostics. No weighted
entropyPMpolicy orcalibrationfit. CPUreview witness inknowledge/evidence.

[Existingartifact exploration, notnewdenseexperiment] ReadacceptedoldN8four
logits only: tindices70/50/30/0 correct1119/1224/1265/1269of1920;
meanentropycorrect .37325/.34003/.31149/.34789 vswrong1.17304/1.15634/1.14095/1.12530.
LowerentropycorrectnessAUROC .84976/.86148/.87405/.86037, pooledbytime over15
corruptions. This is association, not calibration orproofweightedPMhelps.
Label-dependentnondeployableany-candidateoracle1369/1920; currentold4PM1250,
terminal1269. Newdense9 experiment stillneededforothersteps. ExactJSONpreserved.


[Run/Setup] Nstepcanonicalcell2finishedwithonlytheknownfastrlockwheel failure
at09:46:38UTC; completeCLIexecutionexport confirmscalmsize/CLIPbuilt and
fastrlocksolefailedwheel. This is the user-explicitly deferredcell2failure,
notacceptanceofotherpip/nativefailures. Python3.8envverification follows.
CLIhistoricalsafesession_created events pinallocation09:34:44.633339UTC for
nstep and09:42:18.980422UTC forentropy; wholephasebudgets chargefromthese
ratherthancell1receipt times. Entropycell2active09:49:00UTC.

[Code/Package] Source66e4b8a, new8-fileoverlay andold513-fileauthorarchive;
manifestfileSHAa335a8023a55952482672d58b5ae2f3c3d9d52e064eace8a1ac607e23cb22905,
taskSHA3fd2c4fb71f7309432aa5df822ac36c3f4ee261f315ddea587016fe4b1577a17.
Bundle1779495B SHA419956820bbdea174c16e3517c995c96160f5a930a9d1e22fafbc5d03974bf07.
Extractedisolatedoverlayimports/help andbothrolepreviews420/60pass; allPython
modulesAST3.8pass. No oldrunadapters bundled; rootrole-specificadapters next.


[Run/Ownershiphandoff] NstepenvPython3.8.20verifiedafterknowncell2defer.
At09:51:09UTC roottemporarilyretiredlocalmonitor6704 (runtime_teardown=False)
foranexclusiveinteractiveTTYCLI drivemount; no competingexec/probe whileauth
command50144ownsroot'ssameVM. This routineconsenthandoffdoesNOTcloseVM.
User'sfirstNstepDriveconsentpanelpending; noautomaticEnter/consentassumption.
AftermountCLIends rootrestartsmonitoronsameendpoint thenruns canonicalcell3.
Entropyprimarymonitor57884continuescell2; no authmessages/URLs stored in
researchrecords. Operatoradapters, packageisolation/AST andsourcehashesready;
bundlesuploadingindependentlythroughcontentsAPI (notexperimentexecution).


[Run/Setup/Consent] 2026-11-15 10:19UTC. Both task-owned A100/Standard
r2 sessions remain assigned; no runtime teardown issued. User reports both
Drive consents approved. Nstep actual mount succeeded; canonicalcell3 exact
completion10:06:41UTC, canonicalcell4 active since10:09:19UTC (Torch2.1.2cu121
and KNN installed, remaining pinned packages/toolchain pending). Root monitor
94324 owns nstep execution. Entropy cell2 had only known user-deferred fastrlock
failure and Python3.8 verification passed. Its previous mount failed after
consent: no credential-propagation success and remote ValueError mount failed;
CLIexit0 did not establish success. Exclusive local CLI mount retry99963 now
waits for fresh entropy-only interactive consent, bounded600s remote/900s CLI.
Local diagnostic wrapper logs HTTPstatus/exceptiontype only, bounds transport
requests90s; no credential/header/body logging. GET and dry-run POST both200,
explicit redirect required. Auth links/nonces excluded from research records.
No new science/accuracy yet. Next: mount evidence then canonical3-10 entropy;
finish canonical4-10 nstep, official deployment and strict source/N8 parity
smokes before420/60 units. Separate Drive ledgers and8h wholephase caps retained.

[Run/Setup] 10:25:02UTC nstep canonicalcell4 exact marker+CLI0: CUDA12.1,
GCC11.4 and sourcepatches verified (SM80). Cell5 EMD complete10:27:22UTC;
canonicalcell6 Chamfer next. Entropy retry's new prompt was not confirmed
within600s; remote mount kernel verified idle10:24:24UTC. Expired local CLI
99963 retired after exact ownedPID check; runtime preserved. Root restarted
passive entropy monitor18751 at10:26:03UTC (no competing execution). Previous
authlink explicitly marked expired to user; defer another nonce until user
returns. No fabricated mount success/new accuracy. Nstep monitor94324 remains
sole executor. Wholephase budgets still charge all elapsed setup/auth waits.

[Run/Failure/Recovery] Nstep canonicalcell6 first CLI stream lost connection
10:28:08UTC; no success marker. Inventory retained same endpoint; first kernel
HTTP probe timed out, subsequent readonly probe recovered primaryidle with
lastactivity10:28:26UTC. Hung localCLI8587 retired only after traceback (no
remote interrupt/teardown). Root monitor94324 recordedfailedRC-15. Readonly
audit sameprimary10:30:36UTC: prepare+EMD markers true, Chamfer/PointNet++/PVCNN
false, no native build process; exactauditmarker+CLI0. Consequently bounded
canonicalcell6 retry on sameVM is safe; no competing/repeated active build.
Scientific source, protocol and phase markers unchanged; no accuracy yet.

[Run/Setup] Nstep retry canonicalcell6 Chamfer passed10:32:53UTC; cell7
PointNet++ passed10:37:13UTC. Both exact wrapper markers and on-disk native
markers required by scripts; no numerical/scientific changes. Canonicalcell8
PVCNN/native GPU smoke follows. Entropy same owned endpoint preserved awaiting
new interactive consent; passive monitor18751 only, no competing job.

[Run/Setup] Nstep canonicalcell8 passed10:44:31UTC: native CUDA import, FPS
and gather forward/backward passed on A100-SXM4-40GB withTorch2.1.2+cu121.
Canonicalcell9 ModelNet40-C Drive archive copy/extraction completed10:46:27UTC.
Canonicalcell10 checkpoint/native/data verification next; this remains setup,
not scientific inference. Dataset/source identities and parity gates unchanged.

[Run/Setup/Validation] Nstep canonicalcell10 passed10:48:43UTC. Downloaded
verificationJSON parsed locally: Python3.8.20, Torch2.1.2+cu121, A10040GB,
CUDA/FPS/KNN/Chamfer/EMD checks pass,15corruption arrays; classifiercheckpoint
SHA507e0b... matchesacceptedprior. Compact fullverification inknowledge/evidence.
Only known canonicalcell2 fastrlock exception remains explicitlydeferred.
Packageunpack firstattempt encountered CLIprelude Connectionwaslost10:49:28UTC
before payload; readonly ContentsAPI target absent andprimaryidle confirmed.
HunglocalCLI9490 retired; bounded sameVM retry passed10:50:15UTC exactmarker.
Scientific deployment package unpacked with bundleSHA/CRC/path validation.
Official CF dependencies and strict asset/native deployment now follow.

[Run/Unexpected runtime loss] Entropy r2 assignment no longer listed at
10:56:35UTC audit; passive monitor reports LOST already10:46:04.918039UTC (last assigned10:45UTC; exact disappearance lies between probes). No
runtime teardown/remote interrupt issued. Mount never validated; no scientific
units/accuracy were computed, hence no result loss. Prior preparation remains
local; failed-attempt allocated walltime must be charged to entropy's8h cap
before replacement. Do not claim two live sessions after this audit. Nstep
assignment continues; pinned PyTorch3D compile57/67 observed10:56UTC. Account
usage now464.79CU/6.73CUhr/1active assignment. Replacement entropy allocation
deferred until interactive Drive consent can actually be completed, avoiding
another unproductive waiting VM; no research duration/approval gate added.

[Run/Plan amendment] 11:02UTC: Nstep smoke accepted10:59:35UTC; fresh source
repeat and oldN8 canonical-four logits/argmax/PM parity pass in bounded first
unit and repeat/reopen check. Full N6–12 run active11:00:15UTC,27/420units
committed by11:02:28UTC; no aggregate claim until420/420 validation.
[Decision] Entropy r2 loss prevented requested parallel execution. Continue
scientific scope sequentially on root's prepared nstep A100 after its final
archive/validation; this reuses its valid Drive mount and avoids another
interactive-consent/native-setup cycle. User informed; no new GPU approval
introduced. Role/campaign/config/source and separateDrivefolder unchanged.
Prepared entropy_reuse_nstep operator adapters only (no sampler/worker edits),
AST3.8/self-check substitutions pass; receipt inknowledge/evidence. Reused
assignment explicitly handed to entropy only after nstep worker+archive finish;
never overwrite shared task/setup receipt while nstep active. Prior grid ZIP
verified/reused, entropy-specific setup-evidence filename. Entropy8h ledger
conservatively charges lostB3825.937617s plus whole reusedA elapsed since
09:34:44UTC, including Nstep setup/run time. All60 N8 golden parity still hard
gated; no entropy-weighted policy fit. Estimated remaining nstep~28min, entropy
~6–10min including smoke/report/archive, based on first~4s/batch units.

[Artifact/Independent validation] Nstep smoke raw archive retrievedto
result/modelnet40_c/cloudfixer_exploration/20261115-105859_nstep_smoke.zip;
240736B SHA f206dd3c355f4733d28099f0e2b9811f3c74a5498c79acc2e5acd21bf9e57e84.
Local SHA matches DriveFS receipt; ZIP CRC/path/symlink/required7files and
exitstatus0 pass. Smoke/source repeat max_abs0, no argmaxchange; firstoldN8
logits max_abs0, PM probabilitydifference0, no candidate/PM predictionchanges.
Coverage1/420 explicitlypartial, notaggregate accuracy. CompactvalidationJSON
records rawreceipts/gates. Fullgrid50/420committed11:04:03UTC.

[Ownership/Recovery] 11:16UTC root retired only local entropy monitor18751
after confirmed missing assignment, as ownership hands to active nstep monitor
94324 for sequential fallback. This is not VM teardown; entropy VM had already
been lost by10:46UTC. No peer monitor/runtime touched. Fallback receipt now
includes prior-attempt offset in budget handoff (self-check confirms100s gap
still charged despite priorB debit). Operator source12files ZIP15381B SHA
dbb732ed79bc212b70282184906970548f4ec97c9def94aa3cf33eec0260dc0d
uploaded to survivingA; backuptobothroleDrivefolders queued only after nstep
archive. Scientificsource unchanged. Nstep190/420units by11:16:09UTC.

[Run/Accepted N-grid] Fullrun20261115-110019_nstep_run completed420/420units
11:34:19UTC; exact workercomplete11:34:29UTC, archive/outermarker11:34:35UTC.
Raw ZIP27625392B SHA2a4c544f2f08de453205f556689a47a1aedabf37be4b87e07d8c9348e68d58f7
inresult/modelnet40_c/cloudfixer_exploration/ and taskDrivecloudfixer_nstep...
Local SHA/CRC/safe-path/required7files/exit0 and420NPZ checks pass. Offline
registeredanalyzer equalsworkerJSON exactly. Independent NumPyfloat64softmax
recount verifies1920predictions perN,128originalIDs per15corruptions, true
labels,60pairedinput/RNG groups acrossall7N, all60oldN8 actualpriorNPZ checks
max_abs0/candidatechanges0/PMchanges0. Compactwitness and7-grid/105corruption
CSV rows inknowledge/evidence. FixedK80/tau.1/retained.95/seed0/B32/s5.

| N | PMcorrect/1920 | PM% | terminalcorrect/1920 | terminal% | pipeline s | candidate diffusion indices |
|---:|---:|---:|---:|---:|---:|:---|
|6|1223|63.697917|1262|65.729167|177.840986|80,53,40,0|
|7|1235|64.322917|1259|65.572917|194.461539|80,57,34,0|
|8|1250|65.104167|1269|66.093750|211.079649|70,50,30,0|
|9|1247|64.947917|1275|66.406250|227.783756|71,53,36,0|
|10|1254|65.312500|1251|65.156250|244.311435|72,48,32,0|
|11|1249|65.052083|1253|65.260417|261.093594|73,51,29,0|
|12|1257|65.468750|1235|64.322917|277.719108|67,47,27,0|

[Inference/Limits] BesttestedPMN12 adds7/1920=.364583pp vsN8, with31.57%more
measuredinferencepipeline time. BesttestedterminalN9 adds6/1920=.3125pp vsN8,
with7.91%more pipeline time. SameGPU comparisons include4classifiercalls;
notterminal-onlylatency. PreviouscoarsebestPMN32K80tau.03=65.9375 remainshigher,
not a newglobaloptimum. ChangingN alsochanges cumulativeperstep correction
and actualprobe times (especiallyN6/7firstprobe80), so PMdifferences cannotbe
attributedsolelyto DDIMdiscretization. One exploratory label-tunedsubset/seed,
no confirmation/paper-levelclaim. DenseN8 entropy phase next.

[Operator] Sourcebackup firstRC0 wasflaggedFAILED solelybecause monitoring
expected baremarker whereas scriptprints marker+SHA+size. Correct exactline
registered; idempotent samebytes retry passed11:35:55UTC, sourcesbackedboth
Drivefolders. No scientificchange/loss. Entropy handofffirstattempt11:36UTC
failedCLIprelude beforepayload; primaryidle verified11:36:50UTC, localclient
11578retired (noVMinterrupt), sameVM boundedretryactive11:37:36UTC.

[Run/Transport diagnosis] Entropy handoff attempt2 alsofailedCLIprelude
11:38:00UTC beforepayload; primaryidlelastactivity11:38:17UTC confirmed. No
science duplicated. Root retiredhunglocalCLI11638 andlocalmonitor94324 only,
noVMinterrupt/restart/teardown. Sameendpointnewmonitor87036 started11:41:19UTC,
soleCLIowner. It invokesinstalledColabCLI0.7.4main througha localdiagnostic
facade that adds only WSerrorclass/HTTPstatus/closecode/reasonlength logging;
no token/header/body/URL logging, no dependency/account/CLIversionchanges.
Diagnostic retryofsameunchangedoperatorprep follows. Codebytes preservedlocal
result/preparation/.../diagnostic_colab_cli.py; science66e4b8a unchanged.

[Code/Run/Transport root cause] SanitizedinstalledCLI log flags showWS
connections became ready afterCLIprelude hadalreadyraisedConnectionwaslost:
11:36:31failure→11:36:50ready;11:38:00→11:38:17;11:42:06→11:42:11. Inspected
SDKstart_channels waitsitsdefaulttimeout thenreturns evenifconnection_ready
false; prelude thenfails. Local CLI facade now waitsbounded90s forinitial
ready afterSDKstartbeforeissuingrequests; ready/timeoutbranches self-check.
No kernelID/account/version/scientificsource changes. SameVM primaryidle proof
11:42:11 before retiringoldclient11878. Bpreparewithfacade passed11:44:40UTC,
exactoutermarker+CLI0; next denseN8 source/canonical4 golden/repeat smoke.
Setup/handoffreceipts retainlostB3825.937617s andallreusedA elapsed; rootmonitor
87036 remainssoleCLIowner. Diagnosticfacade SHA recorded separately.

[Run] Dense entropy smoke20261115-114556_entropy_smoke completed11:46:33UTC
(exactmarker/CLI0), preserving canonical4logits/PM andsource repeat. Full60unit
phase launched11:47:07UTC on sameendpoint bysolemonitor87036 withsameimmutable
scientificsource. Cleanprobes9 at80,70,60,50,40,30,20,10,terminal0; no noisyzt
classification, no weightedpolicy. Metadata/task receipts switched onlyafter
Nstep completearchive; roleDrivefolders separate. Newaccountconsent unnecessary.


## Accepted dense N8 entropy exploration — 2026-11-15

[Run/Artifact] `20261115-114749_entropy_run` completed60/60units11:53:37UTC;
workercomplete11:53:40UTC, outer/archive marker+CLI0 at11:53:45UTC.
RawZIP8869630B SHA509ea9699cd2fcde8542fc52564c03146a1f2f30db0d4e20d176e1aa479506df
at`result/modelnet40_c/cloudfixer_exploration/20261115-114749_entropy_run.zip`,
Drive`MyDrive/thesis/result/cloudfixer_entropy_20261115_3fd2c4fb71f7/`samebasename.
LocalrawSHA matchesDriveFSreceipt; CRC/safe-path/required7files/exit0/60NPZ
checks pass. Offline analyzer equalsworkerJSON. Independent true-label/
float64softmax/entropy/AUROC/paired-case recount passes; all60oldN8 logits
max_abs0, andcanonical4 exactlyequalfreshN-gridN8. Actualsource+sampler
smoke/repeat NPZ equality rechecked. All15corruptions×128sameoriginalIDs,
s5/seed0/B32/N8/K80/tau.1/retained.95, authorXYZdiffusion+frozenPoint-MAE,
LIONnotloaded. This is1920object-corruption pairs, not1920independentobjects.

All probes are clean x0 estimates; nonterminal probes are pre-correction,
terminalt0 is deterministiccleanestimate. Classifiercalls happen after
trajectory collection, canonical4first thenextras, preserving prior RNG/calls.
No extra denoiser evaluations:9perbatch inbothsparse4anddense9N8,540total.
Densepipeline248.424065s includes9classifiercalls perbatch, compared with
211.079649s forfreshsparse4N8 (~17.69%observeroverhead); notterminal-onlytiming.

| Diffusion index | correct/1920 | accuracy% | H correct(nats) | H wrong(nats) | low-H correctnessAUROC | intermediatecorrect/terminalwrong | intermediatewrong/terminalcorrect |
|---:|---:|---:|---:|---:|---:|---:|---:|
|80|1031|53.697917|0.406331|1.151927|0.828458|41|279|
|70|1119|58.281250|0.373249|1.173043|0.849761|54|204|
|60|1184|61.666667|0.359991|1.166376|0.854760|56|141|
|50|1224|63.750000|0.340029|1.156336|0.861480|62|107|
|40|1250|65.104167|0.321136|1.168139|0.873159|66|85|
|30|1265|65.885417|0.311492|1.140950|0.874052|61|65|
|20|1284|66.875000|0.311787|1.153802|0.876149|59|44|
|10|1266|65.937500|0.313228|1.140726|0.876266|46|49|
|0|1269|66.093750|0.347889|1.125298|0.860368|0|0|

Readouts: canonical4PM1250/1920=65.104167%; denseall9equalPM1256/1920=65.416667%;
terminal1269/1920=66.093750%. DenseequalPM adds6correct=.3125pp, still13correct
(.677083pp) belowterminal. Canonical4PM repairs43terminalerrors but breaks62
terminalcorrect; all9PM repairs49 andbreaks62. No entropyweights applied.

[Finding] Best observed N8 singlecleanprobe is t20:1284/1920=66.875%,
15correct/.78125pp above terminal. It repairs59terminalerrors andbreaks44
terminalcorrect. Per-corruption:10improve,4decline,1tie; notablyrbf+6,uniform+4,
lidar/upsampling+3 each, butocclusion-5 anddistortion-3 (counts/128percorruption).
This is a label-selected exploratory timestep, not a locked deployed method.

[Finding/Limit] Among651terminalerrors,141haveatleastonecorrectearlierclean
probe. Any9candidatecorrect=1410/1920=73.4375% is a label-dependent nondeployable
diagnostic, only relevant to hypotheticalsingle-candidate selection; it isnot
an upperbound on probability averaging. Old4anycorrect1369, so5extraobserved
probes reveal41additionalrecoverableterminalerrors. Full185-case CSV includes
all141terminalwrong/earliercorrect plus44t20wrong/terminalcorrect countercases.

[Inference] Earlier t80/t70 probes have lower accuracy than lateprobes; equal
averaging can mix weak andstrong estimates. This supports exploring temporal
windows/entropy weighting later, but does not prove a causal explanation of
PM's deficit or validate a weightfunction. Entropycorrectness separation is
stronger att20(AUROC.876149) thanterminal(.860368), whileconfidencealone isnot
correctness. Allmax-probabilitycalibrationbins remaininrawanalysis; no entropy
threshold/temperature/weight fitted andno newPM policy selected. No independent
validation/extra seed/full2468-object benchmark run. Timestep/N choices are
label-tuned exploratory onthe reused development subset.

Detailed evidence:9candidate CSV rows,135candidate×corruption rows,15t20-vs-
terminal corruption rows,185object/corruption cases andindependentrecountJSON
in`knowledge/evidence/cloudfixer_nstep_entropy_20261115/`. Original predictions
and fullcalibrationbins remaininrawZIP; offlineanalysis storedseparately.

[Archive/Completion] Verifiedanalysis/operator/knowledgebundle31files479136B
SHA6f8a361a68f1d5a3046f7d89037df140d6e7afc7f8eaa3b50000e3df74742a04,
20261115-120141_verified_analysis.zip, backedto BOTHroleDrivefolders with
exactmarker/CLI0 at12:02:51UTC. ContentsAPIDrive→Mac readback SHA matches.
Includes actualCLItransportfacade source andoperator sources, detailedresearch
records/evidence; no OAuthlinks/tokens. DriveFSverification doesnot independently
prove server synchronization; localrawarchives andderivedbundle provideanother
copy. Finalentropybudget handoff chargesunaccountedanalysiswalltime including
priorBoffset;8h caps not reset. Accountsnapshot481.05→458.06CU (22.99CU account
delta, notisolatedbillingaudit); remainingA100rate6.73CU/hr. Source66e4b8a;
no scientificcodechange afterreviewed977CPUtest pass/2skip. Operator selfchecks,
GPUtechnicalgates, actual120oldN8parity, fullcoverage andindependentrecounts
pass. User-requested exploratorysciencecomplete; futureweightedPMplanning remains
separate. Survivingtta-cloudfixer-nstep-a100-20261115-r2/endpointgpu-a100-s-ass1c0-24gmixfofyxkt
remainsassigned; monitor87036 live; no runtime teardown authorization/action.

[Final formatting verification] StagedGitcheck caughtCRLF lineends innew
derivedCSV files; normalizedsixderivedCSVstoLF withparsedfield-by-field
equality assertions. Allnumbers andrawZIPbytes unchanged. Freshstagedcheck
passes. Finalnormalizedknowledge/operator snapshot32files480288B,
20261115-120638_final_verified_analysis.zip,
SHAc36ae9184070cca735be9459a7c67462fb2864989ccd028a958095cc680a8ef2,
backedbothroleDrivefolders12:07:09UTC andDrive→Mac SHAverified. Earlier
formatting snapshot retained; finalreceipt inknowledge/evidence. Surviving
A100assignment recheckedafterbackup; no teardown.


## Follow-up — entropy-weighted PM evaluated later2026-11-15

[Run/Independent validation] Theearlier "not weighted" status describes this
originalexploration only. Usernowrequested weightingN8/N12; see
[fullweightedPM study](cloudfixer_entropy_weighted_pm_20261115.md). Fixed
expbeta1 N8four1258/1920=65.520833% (+8vsequal), N12four1253/1920=65.260417%
(-4vsequal); parameter-freelinear rule alsohasmixed direction. No newGPU/model
inference, no entropyweights learned or baselinepromoted. NewCPUDrive→Mac
rawSHAretransfer provespriorarchivedresults persistedafteroldGPUloss.
# Later diagnostic — 2026-11-15

[Run/Inference] [Terminal/PM audit](cloudfixer_terminal_diagnosis_20261115.md)
adds saved-logit posthoc windows: N8 [20,10,0] equalPM1283/1920=66.822917%
versus ownterminal1269=66.093750%; conditionalobjectCI includes0. N12[27,0]
PM1272=66.25% versus ownterminal1235=64.322917%. All48subset rows retained;
no deployedwindowselection, newinference or independentconfirmation. Pooled
entropyAUC~.86 contrasts with within-object~.62–.64; timing and confidentearly
errors are better-supported hypotheses than a universalfailure of PM.

## 2026-11-18 — subsequent held-out confirmation and transfer status

[Run/Code/Inference] The held-out256/all15/s5/three-seed Point-MAE study tests N8/.1, N12/.1 and matched N12/(.8/12) at K80. Late equal PM beats each own terminal. Reducing N12 tau restores +1.510pp terminal (CI[+.955,+2.075]); matched N12 vsN8 terminal is -.078pp (CI[-.417,+.243]), with differing individual repairs/harms. N*tau matching does not establish equivalent dynamics or an optimal N. Historical128 canonical/dense N8 logits replay exactly before confirmation. [Full evidence and limitations](cloudfixer_pm_hypotheses_confirmation_20261118.md). [Current execution](cloudfixer_two_colab_execution_20261118.md).
