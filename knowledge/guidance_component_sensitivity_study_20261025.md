# Agent B parameter and loss sensitivity — 2026-10-25

Status: CPU preparation; no new GPU research authorized or launched.
Required evidence labels: [User report], [Code], [Run], [Inference], [Open].

## Question and evidence boundary

[User report] Extend the fixed component study with different GSD parameters
and several loss calculations, and assess whether Chamfer/original3DD is more
reliably supported. This reopens research; preserve own live A100 and monitor.
The earlier optional close question is not teardown authorization.

[Run/Inference] Existing C probability mean56.8929% versus C+S56.7222% has
conditional interval[-0.0397,+0.3651]pp. That is not proof of superiority or
equivalence. At transferred coefficient1.17, standalone spectral local updates
are several orders smaller than Chamfer's. Chamfer-only in the GSD host also
differs from original3DD in batch, horizon, final style and DDIM endpoint.
Historical Scan readouts exceed original3DD in some cases. No finite menu of
experiments can establish universal superiority over every GSD parameter/loss.

Falsifiable question: after a declared finite search and label-free scale
calibration, does a frozen spectral-containing candidate outperform a matched
GSD-host Chamfer comparator and the distinct original3DD whole-policy baseline
on newly reserved objects across three seeds? Retain all negative/null/failure
results; the conclusion must be allowed to favor either family.

## Selected design and alternatives

The existing four-arm matrix is accepted evidence and will not be rerun merely
for a new launcher. Exhaustive tuning or trying variants until Chamfer wins
would not answer the question. The selected extension is a bounded exploratory
screen followed by a newly frozen confirmation. This is the smallest useful
way to test the weak-scale explanation, a few distinct losses, parameter
sensitivity and the actual original host together.

Use the same two datasets, four heads, all15 severity5 corruptions, native
binaries/checkpoints and GSD B70/M800/beta2/10-or30-step/update-style policy.
Graph construction, isolate policy, eigensolver, frequency cutoff and scheduler
remain fixed. New objectives are named experimental variants, not repairs or
full GSDTTA reproduction. No model training or gradient projection is added.

## Loss definitions and calibration

For predicted/reference graph coefficients H and Q, N latent points, and the
unchanged weights w_i=exp(-2 lambda_i) below800 (zero above), use:

1. `legacy_l2`: sum(w*(H-Q)^2)/(3N), preserving the exact default arithmetic.
2. `huber`: sum(w*smooth_l1(H,Q,beta=1))/(3N). Delta1 is fixed before labels.
3. `power`: sum_i w_i*(sum_c H_ic^2 - sum_c Q_ic^2)^2/(3N).
   This is per-mode squared energy, not amplitude or a claim of invariance
   under arbitrary mixing of degenerate eigenvectors.

For each loss, compute actual first-step gradients through the frozen local
prior and DDIM predicted-clean path on all30 existing development caches,
seed0. Use per-object RMS local updates at eta=.01; compare spectral coefficient1
with Chamfer1,p=.95. Freeze one scalar per loss from the median of per-object
Chamfer/spectral update ratios across2100 input events, equally represented
across both datasets and15 corruptions. No labels/classifier scores enter this
calibration. Style updates are recorded separately; one scalar is not asserted
to equalize both blocks. Ratios rho=.1 and1 multiply that frozen scalar.

Reject nonfinite gradients; record zero/tiny norms. A loss needs at least90%
valid local-gradient pairs; no silent coefficient clipping. Candidate scalar
must be finite and in[1e-12,1e12], otherwise stop for diagnosis. No calibration
per corruption/backbone/target label, or recalibration for a trial's rates.

## Fixed development trial budget

Development remains the original70 objects/dataset, all15, seed0. Each family
has six configurations INCLUDING its already accepted baseline. Each new
configuration sees both datasets/four heads and the same prepared states/noise.
The accepted baseline may be reused only after source/numerical/state parity;
its original source identity remains explicit rather than being relabeled v2.

| Family | Configurations, in registered order |
|---|---|
|C|accepted base; eta=gamma=.003; eta=gamma=.03; eta=.003/gamma=.01; p=.80; p=.99|
|C+S|accepted base1.17; calibrated L2 rho.1; calibrated L2 rho1; Huber rho.1; power rho.1; calibrated L2 rho.1 with eta=gamma=.003|
|S|accepted base1.17; calibrated L2 rho.1; calibrated L2 rho1; Huber rho.1; power rho.1; calibrated L2 rho.1 with eta=gamma=.003|

Other rates remain.01,p=.95. A coupled eta/gamma scale is one declared factor.
The trial count is equal, not equal wall time or exhaustive search-space
coverage. There are15 genuinely new configurations/30 dataset jobs; failures
consume their trial slot and are archived. No adaptive replacement or broad
Cartesian grid. Terminal/entropy/probability readouts are retained; the fixed
selection score is probability-mean balanced all15 across eight cases.
Ties favor earlier registered configurations; no per-case/corruption routing.

## Original baseline and prospective confirmation

Enable terminal-only original3DD for all four existing frozen heads, including
the currently blocked Point-MAE path, with its explicit checkpoint/config.
Preserve batch32,5/35 reverse updates, original final style, DDIM finalalphaTrue
and original tta.py. A missing matched-source row stays explicitly unavailable.
This is a new matched-subset/native study control, not a full accuracy rerun.
GPU smoke must verify the newly enabled boundary before broad execution.

Reserve140 NEW original IDs/dataset now, excluding BOTH previously used70-ID
sets; use the same new IDs across all corruptions/seeds. They are reserved from
this development stage, not an externally unseen benchmark. Do not inspect
their predictions until configuration freeze.

After screen acceptance, freeze the best spectral-containing configuration
across the six C+S and six S entries; this larger family-level choice opportunity
must be disclosed. Also freeze the best C configuration. Confirm:

- the chosen spectral candidate;
- a GSD-host Chamfer-only comparator with the candidate's eta/gamma/p;
- the best tuned C configuration if it differs from that comparator;
- locked original3DD as a separate whole-policy comparator.

Maximum four conditions, both datasets/all15/four heads/seeds0/1/2 on140 IDs.
If the spectral candidate is C+S, its matched C contrast isolates removing
spectral guidance at the chosen setting. If it is S-only, the contrast replaces
the objective; it is not a conditional spectral increment. Confirm equal
probability readouts for GSD-host conditions; original has ordinary terminal.
Report same-policy terminal/entropy diagnostics, all15/macro14, every case,
per-seed signs/ranges, corrected/harmed events and measured cost. Original's
different host permits whole-policy outcome comparison, not shared-latent or
isolated-loss causal claims.

Resample original object IDs jointly across corruptions, methods, heads and
all three seeds; independent draws between datasets. There are280 reserved
object clusters, not840 independent objects. Failed/null cases and unequal
family selection opportunities remain visible. No optimum/equivalence claim.

## Batches, compute and gates

### Batch 1 — explicit losses and original Point-MAE boundary

- Goal/scope: minimal opt-in objectives and missing terminal original head.
- Touched areas: tta_gsd_v2.py, runner, shared spectral-loss function and tests.
- Stack: existing CPU PyTorch contracts; no GPU or dependency installation.
- Dependencies: accepted fixed-source artifacts and ownership contract.
- Development notes: preserve default math/RNG/configs; pin new dependency.
- Verification: old-mode regressions, finite-difference/robust/power checks,
  strict CLI and original-policy tests; Python3.8 syntax and diff checks.
- Knowledge target: this note; append implementation and actual test evidence.
- User gate: authorized CPU preparation; no new GPU before duration approval.
- Outside scope: graph changes, original trajectory readouts, paper reproduction.

### Batch 2 — concrete sealed campaign and calibration

- Goal/scope: label-free gradient calibration, immutable trial/ID registry,
  complete commands, source/assets hashes and single whole-campaign ledger.
- Touched areas: own preparation root and isolated new remote source directory.
- Stack: existing Colab CLI/monitor, pinned environment/native assets.
- Dependencies: Batch1 tests; safe reuse/parity controls and reviewed preflight.
- Verification: CPU dry commands, trial counts/IDs/hash guards, calibration
  formula checks, strict expected trial parameters and timeout/recovery tests.
- Knowledge target: this note, findings/open questions/README and phase plan.
- User gate: one comprehensive >60min research-run approval BEFORE any GPU.
- Outside scope: experiments selected to force a desired conclusion.

### Batch 3 — bounded search, new freeze and confirmation

- Goal/scope: approved calibration/smokes/30 development jobs/original control,
  then evidence-gated frozen confirmation (maximum24 condition/dataset/seed jobs).
- Dependencies: exact sealed proposal approval, CPU/source/Drive acceptance.
- Verification: same-state/noise within GSD, source/native/checkpoint/ID/state/
  readout checks, physical Drive/local SHA, paired object-cluster analysis.
- Knowledge target: separate exploratory and confirmation result notes.
- User gate: keep the entire approved campaign budget, not per-CLI subdivision;
  any material scope/duration increase needs reassessment.
- Outside scope: full2468/581 benchmark, unlimited tuning, automatic teardown.

[Open] Preliminary whole campaign estimate3.5–5 GPU hours, provisional hard cap
5.5h, includes calibration, all smokes, bounded retries, development, original
baseline, confirmation and evidence persistence. Exact measured-cost proposal
will replace this estimate before approval; no GPU launch is authorized by
this document. Setup/transfer/billing lifetime is distinct. Existing own session
`tta-guidance-components-a100-20261025-r1` and monitor52794 are preserved.

## Batch 1 implementation and CPU verification — 2026-10-25

[Code/CPU/Open] Working base `3bc8ebb05911a76af1b831d5a295d1d59be51eed`;
this batch is an uncommitted source change, not an executed Colab revision.
`spectral_guidance_losses.py` now owns `legacy_l2`, fixed-delta-1 `huber`,
and per-mode squared-energy `power`, all with the declared `3N` reduction and
unchanged exponential weights/cutoff. Logging and applied spectral guidance
use the same helper. Invariant loss, graph/eigh/isolate behavior and original
`tta.py` are unchanged. Exact old-inline legacy values, gradients, reconstructed
outputs/style, metrics and RNG are checked. Power tests cover XYZ orthogonal
rotation and per-mode sign changes and explicitly demonstrate that arbitrary
mode mixing is not an invariance.

The opt-in `--spectral-loss-kind` requires an active-spectral component arm;
explicit selections record the kind and modified flag. Omitting the option
preserves existing configuration dictionaries. Terminal-only original3DD now
accepts Point-MAE with both explicit checkpoint/config paths, retaining B32,
5/35 updates, original style and finalalphaTrue. Its unbound source row stays
unavailable; no original trajectory observation is enabled. The prior observed
Point-MAE unavailable-source reason string is preserved.

[Code/CPU] The root's unrestricted focused suite passes **54 tests in 3.672s**;
the new 11-test module separately passes in 0.527s. All 12 pre/post full default
configuration groups compare exactly equal; Python 3.8 AST checks and
`git diff --check` pass. Tests include finite differences for all three losses,
Huber outlier gradients, exact legacy parity, CLI/original policy guards and
an actual `snapshot_sources` copy/hash check for the new helper. The first child
suite's five legacy temporary-directory errors were sandbox write denials;
the unrestricted rerun resolves that verification limitation.

[Open] No GPU/Colab execution, calibration result or accuracy/speed improvement
is established by this batch. The next sealed source ZIP must include the new
helper and its declared identity; the generic multibackbone source inventory
already captures it. Historical legacy entrypoints/inventories are outside
this campaign and remain unchanged. A new GPU smoke must verify the original
Point-MAE boundary and selected losses after the whole-campaign approval gate.
Source/helper hash, gradient/RNG or locked-original-policy mismatches invalidate
the affected future comparison.


## Concrete CPU readiness and approval proposal — 2026-10-25

[Code/Run/Open] Consumer commit`8169271c9058995e8dc128062848a253baa13f10`,
source ZIP`73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8`,
manifest`599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef`.
Recoverable inherited v1 overlay is explicitly dirty relative to Git; the exact
source ZIP and four-file delta are retained. Original tta.py, graph/native/model
sources and checkpoint hashes remain unchanged.16actualstrict CPU loads cover
GSD/original ×2datasets×4heads, including original Point-MAE. No forward orCUDA
initialization occurred. Actual Drive/local CPU evidence11715bytes SHA256
`eb7af9e0c8ae2e460f6307b071538a4832e8f9c3c4c3760991dc3553b137cde3` passes.
54CPUcontracts,12exactdefaultgroups,2calibrationintegrationtests,5controller
integrationtests/selfchecks and184head-config previews pass.

Registered plan:`result/preparation/guidance_component_sensitivity_20261025/campaign_plan.json`,
SHA256`4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1`.
Nominal stages: calibration5min,smoke14min,development111.90min,
maximumconfirmation120.71min: total4.1934h.20%margin plus1200s packing/retry
reserve gives5.3654h; one cumulative19800s/5.5h hard cap covers every GPU stage.
Original180/270s job estimates are provisional until actual smoke; reestimate
before broader execution and after freezing candidates, preserving the cap.
The controller stops if spent+1.2*remaining stage estimates+reserve cannot fit.
Source, native, ID, state, RNG and receipt failures stop and preserve partial
outputs; no silent new grid or resets. Caches persist after every accepted job.

Actual calibration will also require exact scalar and coefficient-gradient
legacy parity. Through the native prior, two old-inline backward repetitions
measure a null floor; helper relativeL2 difference must be≤max(1e-5,3*floor)
and≤1e-3, with floor≤1e-3. This predeclared repeatability check does not claim
bitwise native/backbone/decoder/eigenspace equality across processes.

[Open] One comprehensive duration approval was requested for this exact scope;
no new GPU calibration, smoke or accuracy experiment has run. The CPU-only
calibration preview uses synthetic in-memory coefficients and cannot authorize
a GPU launch. Actual coefficients will be derived without labels and sealed.
Own`tta-guidance-components-a100-20261025-r1`/monitor52794 remain live.

[Paper/Code] Primary attribution was refreshed: original3DD uses selective
Chamfer guidance with frozen classifier, whereas full GSDTTA includes learned
spectral shifting and model self-training. These three new loss variants are
GSD-inspired latent guidance, not implementation of full GSDTTA. Primary sources:
[3DD-TTA](https://arxiv.org/abs/2411.14495),
[GSDTTA](https://arxiv.org/abs/2507.18225). This bounds attribution, not accuracy.


[Run/Open] Final CPU acceptance: real VM plan/registry/ID/source/16-load receipt
integration passes, as do controller/calibrator selfchecks under Python3.8.
The complete5,570,596byte readiness packageSHA256
`e2af93ccaca4a6415489fb92a19ded2253ca916293fbffbb7e322e64568b7741`
is physically verified on Drive and after local download, including exact plan
bytes and ZIPCRC. Independent integration review found no material blocker.
`readiness_validation.json` records all checks and the next exact action.
User GPU-duration approval remains pending; no new research GPU executed.


## Explicit approval and launch — 2026-10-26

[User report/Run/Open] User replied **“Tamam onaylıyorum”** to the exact
registered whole-campaign5.5h duration request. Recorded beforeGPU at
2026-10-26T05:53:05.669817+00:00 with planSHA
`4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1`;
nominal4.1934h/margin5.3654h/hard19800s unchanged. Approval466bytes SHA
`2f3e2a4f9efe9bdfe6a68e1345506f96f2a2f8aff848da849959f5e6005313da`
exists identically in own runtime and Drive, local`gpu_approval.json`.
SameA10080GB/endpoint and16-load CPU proof reverified; free disk151784640512bytes.
Calibration wrapper submitted through solemonitor52794; actualGPU outcome and
raw archive pending. No new accuracy result. Preserve single cumulative ledger,
source/identity failure stops and runtime lifecycle contract. Next: accept
calibration evidence, then registered14-job smoke beforedevelopment.


## Calibration preserved; unexpected runtime loss — 2026-10-26

[Run/Open] Actual calibration phase state completed in155.460578s; physical
Drive archive5,806,352bytes SHA256
`3431f2a15e67892429fcc663b7fd2c0d6e3d93083daee078dc4b3e884d61c8f2`
was downloaded before runtime loss and passes localSHA/ZIPCRC. The retained
ledger records155.866607s spent and no activeworker. Report coefficients:
legacyL2=9373.767440639811,Huber=19729.987388376365,power=13.76292805636293.
These are gradient scale coefficients, not accuracy outcomes; independent
2100-row calibration/parity review remains pending. PrimaryCLI stream never
returned the exactwrappermarker; it later reported Connection was lost.
Repeated serverinventories showed ownr1A100 absent at07:36/07:37UTC; monitor
52794 subsequently reportsLOST. No teardown was issued. No smoke/development
orconfirmation from this campaign ran. Localhost scheduling/sleep gaps occur;
servertermination cause remains unknown. A bounded21600s localcaffeinate
assertion is active to reduce idlehostsleep; it does not guarantee Colablifetime.

Newr2 allocation printedREADY but later inventory/status found no suchassignment;
no setup/researchcell was submitted. A sameA100/HighRAM r3 allocation is being
checked. No peer/unknownCPUassignment is touched. NewVM requiresall10setupcells
and exactnative/assets/Drive recovery before campaigncontinuation. Approved
19800s cumulativecap and trialscope persist; never resetspenttime or rerun
acceptedcalibration silently. Rawlocal:`calibrate_evidence_drive_roundtrip.zip`
and`calibrate_receipt.json` under sensitivitypreparationroot.


## Independent calibration acceptance; recovery waits for stable host — 2026-10-26

[Run/Code/Open] IndependentCPU validation and rootrevalidation accept actual
5,806,352byte calibrationarchive. Every ZIPmember/source/native/frozenLION
binding and registeredapproval verifies. All30physical cachedNPZ files were
rehashed and joined to accepted latent/noise/input/RNG metadata. All2100/2100
ratios are valid for eachloss, no tiny/nonfinitepair. Recomputed globalmedians
exactly equal L2=9373.767440639811,Huber=19729.987388376365,power=13.76292805636293.
30/30 scalar and directcoefficientgradients are exact, RNG unchanged; native
helperrelativeL2 maxima local0.00025963/style0.00035591 pass individualdeclared
repeatabilityfloor/caprules. These are first-stepgradientcalibrationresults,
not accuracy evidence or fulltrajectoryequivalence. Rawreport99029bytes SHA
`82ebc1fdb2784a0663968a20d46909e3020e127310f237971382c78c40e211e0`.
Local`calibration_local_acceptance.json`, `calibration_root_revalidation.json`,
`validate_calibration_local.py` retain independentchecks. No GPU repeat needed.
Post-seal ledger downloaded beforeloss preserves155.866606843s spent; archive
ledger155.460578268s is the earlierpre-seal snapshot. Use the largerknowncost.

[Run/Open] r3A100HighRAM was actualserver-verified at10:06UTC, endpoint
`gpu-a100-hm-euw4a0-ws8p7xzfs0gu`, solemonitor9752. Cell1installedConda and
restarted; primaryCLI laterlostconnection, serverinventory at10:42UTC showed
noA100. No reconnect or cell2+ passed; freshsetup is incomplete. r2also
vanished beforeanysetupcell. At11:06UTC server reportedzero activeassignments;
no runtime was stopped bythisagent. Bothmonitorprocesses remainrunning/LOST;
no activeevaluationworker. No smoke/development/confirmationaccuracy exists.

[Run/Inference/Open] `recovery_20261026/host_sleep_evidence.txt` retains161
Macsleep/wakeevents today, including longMaintenancesleep/deepidle onbattery
duringCLIstreamgaps. Local21600s caffeinate-i assertion didnotpreventthese
sleeps. This establishesinterruptedlocal supervision, not the server'scause
oftermination. Repeatedallocation alone is not a soundrecovery. An asynchronous
question now asksuser toconfirmMacawake/stable beforeanotherallocation;
existing5.5hGPUbudgetapproval staysvalid and is NOT beingrequestedagain.
Oncehoststable: allocate sameA100HighRAM, replayall10setupcells, exactpinned
native/assets/DriveCPUproof, restoreacceptedcalibration andspentledger with
explicitmigrationprovenance; then registered14-jobsmoke. Do nottouchunknown
orpeerassignments. Fullscopeandhardcap19800s remain unchanged.


## User-directed recovery r4 — 2026-10-28

[User report/Run/Open] User directs “Yeni oturum açıp çalışmaya devam et”.
Existing whole-campaign approval/cap19800s and spent155.866607s remain binding.
Server inventory initially0active; new ownA100HighRAM session
`tta-guidance-components-a100-20261028-r4`, endpoint
`gpu-a100-hm-euw4a0-3s9zrjduwd03k`, actualA100-SXM4-80GB/175166460kBRAM
verified. Solemonitor execsession5314; bounded21600s localhost-awake assertion
execsession85660. These supervise legitimatework and do not guaranteeVMlifetime.
Canonicaltennotebookcells unchanged; receipt-onlysessionliteral adapted in
newsetupwrappers under`recovery_20261028_r4/setup`. Cell1ranseparately, restarted,
and sameVMreconnectexactmarker passed13:21:02UTC. Cell2environment creation
submitted13:21:20UTC. No researchGPU or newaccuracytest yet.

Next: finishcells2–10 (onlyknownfastrlockfailuredeferred), mountDrive through
CLI/userconsent ifrequested, exactpinnedassets/native restore and freshCPUproof,
explicitplan/session/ledger migration preservingacceptedcalibrationbytes and
approvedscope/spenttime, then14-jobsmoke. No teardown authorization.


[Run/Code/Open] r4cell2 createdPython3.8.20 and failedonlyatknownfastrlockwheel;
explicitdefer remainsapplicable. Independentenvironmentmarker passed13:29:11UTC.
Cell3Drive/KNN submitted13:30:04UTC. Ownership-onlycontinuationplanSHA
`1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42`,
controllerSHA`0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc`;
changesonlysession/ledgerownership andstrengthenedacceptedreport/spentfloor
guards. Sealedassets/source/IDs/objectives/trials/readouts/capunchanged.
TwoCPU migrationtests, controller selfcheck andPython3.8syntaxpass.
Fresh8+16CPUproofs and actualDrive/recoveryinstallation remainpending.
No newresearchGPUexecuted andnoaccuracyclaimed.


[Run/User action required/Open] r4cell3 firstattempt failed`ValueError:mountfailed`
at13:32:07UTC (exactmarkerabsent). TTYCLI`colab drivemount` execsession37218
received actualDriveOAuthrequest for ownendpoint; actualGoogleconsentscreen
opened inuserbrowser andasynchronousquestionasksuser tocompleteauthorization.
Noaccountpermissionclickedbyagent; noauthURI/tokeniscopiedtorepository.
Solemonitor5314remainsrunning, withprobesinhibitedafterfailedcell3; nosecond
notebookcellwillbesubmittedwhilemountautomationwaits. Thisisinteractive
accountconsent, notnewGPUbudgetapproval. Runtimepreserved/liveatlastinventory.
Nextafteruserconfirms: sendnewline to37218, requiremountsuccess, reruncell3
andverifyDrive/KNN; thencells4–10inorder. ResearchGPUstillnotstarted.


[Code/CPU/Open] r4localanalysis preparation isready. Sharedhelper nowaccepts
explicitmigratedplan+hash whileoriginalcalibration isverifiedagainstitsown
oldplan/session/archive/report/acceptance; seed0defaultsretained, registered
seed1/2 andseed-specificpreparationpairingvalidated. Confirmationanalyzer
requiresactual140-IDlabels, 180physicalcaches, exactdevelopmentfreeze and
280objectclusters jointlyacrossseeds. Root9CPUchecks pass0.172s, previous5
fixturechecks0.033s, selfcheck+Python3.8ASTpass; actualmigratedplan+original
calibrationingestionpasses. Noactualnewsmoke/dev/confirmationdataexists, so
thoseacceptancepathsremainunverified. ModelNetfullsealedlabel.npy mustbe
downloadedaftercell9; Scanlabelsareinassets. Source, checksandCLIhandoffunder
`result/preparation/guidance_component_sensitivity_20261025/`; nosealed
researchconsumer/source/objectivechange. R4stillawaitsactualDriveuserconsent
inTTY37218; asyncquestionpending. No newGPUapprovalrequest oraccuracyresult.


## Drive consent completed; same r4 resumes — 2026-10-28

[User report/Run/Open] User says “Başla”, then explicitlyconfirms
“Drive erişim iznini tamamladım”. OldmountTTY37218exitedwithtimeout after
itslongwait; remote2kernelswereidle andDriveMyDrive404, so no duplicateactive
cellwasstarted. SameVMnewmountTTY57259receivedactualfreshconsent; usercompleted
it, CLIreportedCredentialspropagated and`Mounted at /content/drive`, exit0.
Cell3retry submitted14:37:23UTC viaunchangedsolemonitor5314; markerpending.
Accountpermissionwasuser-operated; existingGPUbudgetapprovalunchanged.
No researchaccuracyrun yet. Next: acceptcell3Drive/KNN andrun4–10inorder,
exactpinnedassets/freshCPUproofs, installreviewedownership-onlyrecovery,
thenregistered14-jobsmoke. Oldcalibrationand155.866607sspentremainbinding.


[Code/CPU/Run/Open] Independentanalysisreviewfoundtwofutureacceptanceblockers:
ledgerkey`resolved`vsproducer`resolved_model_config`, andNumPy2pickleRNGmodule
spellingdiffersfromremoteNumPy1. Both reproducedbeforefix: actualcontracttest
failed; genuinecachehashmismatchfirstfile, independentreview0/30. Fixedexact
producerfield; isolatedCPUanalysisPython3.10.18+NumPy1.21.6 nowmatches30/30
genuineoldcacheRNGhashes, withoutrewritingrawbytesorweakeningchecks. Explicit
NumPy1guardpreventsfalsecorruptionreports. Full11CPUtestspass(.292sNP1,
.168sNP2unsupportedguard). GPUconsumer/source/manifest/planunchanged.
Actualconfirmationacceptanceisstillunexecuted. Seeanalysis_environment.json
andanalysis_review_fixes.json under recovery_20261028_r4.

[Run/Open] r4EMDpassed14:53:16UTC,Chamfer14:55:21,PointNet++14:59:47, each
withdiskmarker+physicalDriveprogressreceiptchecks. Cell8PVCNN/nativeGPUchecks
submitted15:00:39UTC via5314. Dataset/checkpoints/setup10 andfreshCPUproofs
remainpending; no newresearchaccuracyrun.


## Accepted r4 recovery; registered GPU smoke launches — 2026-10-28

[Run/Code/Open] All10canonicalsetupcellspassed (onlyknowncell2fastrlock
deferred). ActualA10080GB,PyTorch2.1.2cu121/CUDA12.1/NumPy1.21.2/nativeGPU
operations,ModelNetall15andcheckpointsverify. Setup56157byteDrive/localZIP
SHA`924076c28b64af0c1b121703cacbf55fa8400a92e6c8deb0925d4015fcb2a517`,
637filedeployment1421391bytesSHA`ee84885876c9008dc6b9a66ddc072f3fe92b6d1b1a185141832fa089b85f7c57`,
8-headCPU24937bytesSHA`3058021bb9b91f7d517eece7899a11dcf244cf60f16ac42f12d207901032bd50`,
16-loadCPU11716bytesSHA`d90e6308a985e4a25b85ff9c99d15b82ff11c76db7e94be2e5ced5c898172e9f`
allactualDrive/localSHA,CRC,source/stateverify. Freshnativebyteswerepreserved
beforeexactaudited4binaryrestore; nohiddenfresh-buildparityclaim.

Recovery11845364bytesSHA`03c86fa0bfc15a84abe7b0e7d845b96d9f9f004816262f54655372b3b3bd954d`
passedindependentlocal32membermanifest/CRC/plan/statechecks. Oldcalibration
reportbytes/sessionretained,30actualdevelopmentNPZsrestoredfromDrive;
post-seal155.866606843sspent/historycarried. Newownershipplan14033563…c4d42,
consumer8169271/source73d9fbe1…b37b8/manifest599b8dcb…8eef unchanged.
No newGPUapprovalneededforunchangedscopeand19800scap.

Registered14jobGPUtechnicalsmoke now submittedthroughsolemonitor5314 in
`tta-guidance-components-a100-20261028-r4`:7conditions×2datasets,Gaussian/
Background,70fixedIDs/dataset,4heads,seed0; freshdistinctsmokecachenamespace.
WrapperSHA`539b14b2fb540f0de44b73eb4d14f53cb84787090e8f16f8019f06eee52fed30`.
Actualsuccess/accuracy/rawarchivepending. Completionneedsexactdirectmarkers,
physicalDrive/localevidence+cachebytes, independentanalysis andmeasured
wholecampaignprojectionbeforedevelopment. Localbounded21600sawakeprocess
64591supportsCLI supervision; noVMlifetimeguarantee/teardownauthorization.


## Sensitivity smoke independently accepted; development running — 2026-10-28

[Run/Code/Open] All14registeredtechnicaljobs completed in916.655641s;
post-sealwholecampaignspent1105.891987s includingoldacceptedcalibration.
Bothdatasets/Gaussian+Background/70developmentIDs/fourheads/seed0,7conditions
(C/CS/Sbase,calibratedCSL2/Huber/power,actualoriginal3DD) yield56rawZIPs.
Rawphase627301980bytesSHA`523c2e634d6af682157b4e1e3e597e3173523c067b777a77d1d2e3243128fad3`
and4physicalcacheNPZcollection17198671bytesSHA
`7bdd9ba956c742fc8b57795cb14768a69f9ffd38c9069069ab581d16db5857a3`
matchactualDrive/localbytes. Independentlocalanalyzeracceptsall56rawsource/
config/checkpoint/state/ID/label/trajectory/pairing/receiptguards,all4physical
newcachesand30oldcalibrationcaches; rootNP1reconstructsall4newRNGhashes.
Technicaltwo-corruptionresultsarenotall15ranking/finalaccuracyproof.

Localacceptance37355bytesSHA
`f285c4e5e32c5d5b4e910a0cdbbdc1d1c9c1d85e18272bb03e85d0eb6041c74e`
physicallyDrive/localroundtripverified; ROOT/accepted_sensitivity_smoke.json
passedcurrentplan/Drive/cumulativeledgergates15:48:13UTC. Conservative
post-smokeprojectionusesmaxmeasurednew/oldjobratio1.025455,20%margin,1200s
reserveandincreasedoriginalfutureallowances200s/devdataset,300s/confirmation
datasetseed. Upper19743.614653s=5.48434h fitsunchanged19800scap; thisis
anestimate,notruntimeguarantee. Recheckafterdevelopment/freeze.

Registered32-jobdevelopment submitted15:48:34UTC viaownr4/solemonitor5314.
30newGSDdatasetjobs(5newsettingsperfamily×2datasets),2actualoriginalcontrols,
all15/s570IDs/dataset/4heads/seed0;3baselinesreusedonlywithdeclaredparity.
FirstC-step-lowModelNetjobhasproducedseveralcorruptionreceiptsusingoldpaired
preparations. Actualdevelopmentacceptance/ranking/freezeand140-IDthree-seed
confirmationremainpending. NoGPUscope/trial/readout/source/ID/capexpansion.


[Code/CPU/Open] Futureconfirmation180NPZcollector preparedbyadaptingexisting
artifactworkflow, sourceSHA`694f55a0e138c3d6af34be4f92ef5e44c8b3ac88537ef22aa90efb9e11d15c4f`.
Itrequirescompletedconfirmation/inactiveledger/exactfreezeand180unique
source+Drivecachepaths, neverGPUrecalculation. ThreeCPUfixturespass0.105s,
Python3.8syntaxchecked. Actualcollectionunexecuted; actualconfirmationoutputs
do notexist yet. Source/test/handoff underrecovery_20261028_r4.


### Development acceptance preparation — 2026-10-28 16:51 UTC

[Code/CPU/Open] The CPU acceptance/freeze template is prepared and independently
reviewed (SHA8366ab2e…158ff6); root reran its three synthetic fixture tests, all
passed in0.004s. It cannot execute until bound to a genuine accepted development
package. Gates retain32new jobs/152raw ZIPs/19trial rows/30physical caches,
source/history separation, exact freeze binding, immutable Drive evidence and
the conservative maximum24-job confirmation budget under19800s. Independent
read-only review also accepted24 genuine old ZIPs/30physical caches and checked
all72 possible freeze pairs using synthetic scores (12three-condition,60four).
Evidence: recovery_20261028_r4/development_acceptance_preparation.json plus
source/tests/handoff and development_review_check_development.py/stdout.json.
The GPU development phase is still running; no new ranking, winner or accuracy
claim is established. Next: genuine completed archive acceptance, frozen choices
and measured budget gate before the existing confirmation wrapper.


### Bounded-memory artifact transport prepared — 2026-10-28 17:17 UTC

[Code/CPU/Open] Localhost has16GiB RAM; installed ColabCLI0.7.4 download loads
the entire base64 JSON response. Genuine smoke archive sizes project about4.7GiB
of development raw ZIPs, so direct whole-archive retrieval risks excess memory.
Prepared a CPU phase to stream the completed immutable Drive ZIP into<=256MiB
transport parts and a local joiner requiring exact original receipt/plan/session,
part SHA/order/offset/coverage and aggregate original SHA. Originals stay intact.
Source SHA d3550929…62723 (split),667deec2…8773 (join). Root reviewed source,
reran three synthetic fixtures (0.013s), and checked Python3.8 AST; independent
review found no material blocker. Genuine transport remains unexecuted. See
recovery_20261028_r4/chunk_transfer_preparation.json and chunk_transfer_HANDOFF.md.
Scientific source, trial registry, frozen-ID plan and19800s cap are unchanged.


## Sensitivity development accepted; frozen confirmation running — 2026-10-28 17:49 UTC

[Run/Code/Decision/Open] All32new development jobs completed in6584.800419s;
post-seal cumulative campaign spent7782.756118s. Raw phase1614733520bytes SHA
`2c4424e3cf017b8c1da99e77eb233fddb6c1d472efeb5df9552fd7b3c74fdea8`
was physically checked from Drive through7bounded parts and reconstructed with
identical local SHA. Independent NumPy1 analysis accepted152raw ZIPs (128new,
24historical),19registered trial rows,30physical dev/calibration caches and all
source/state/ID/label/trajectory/policy guards. Raw and derived data stay separate.

Development balanced all15/s5/70IDs per dataset/four heads/seed0 probability-mean
scores: best tuned C(`c-step-high`,eta=gamma=.03)56.785714%; best spectral-containing
CS(`cs-l2-r010`,calibratedL2rho=.1,eta=gamma=.01)55.345238%; actual original3DD
terminal55.238095%. These are development selection scores, not confirmatory
accuracy evidence or proof that original3DD is superior. Selected CS versus C
adds the spectral term at matched settings; tunedC and original are distinct
comparators. Original host/readout policy differs from GSD.

The exact accepted JSON SHA f94afc83…54ed and frozen4condition JSON SHA
ea245f8a…c360 bind spectral-candidate,matched-chamfer,tuned-chamfer,original3dd.
Compact90496byte acceptance SHA37f528f6…61b8d passed physical Drive/local
roundtrip and remote source/cache/freeze checks17:46:53UTC. Timing review
includes doubled four-head cost and maximum24futurejobs: remaining7741.187194s;
wholecampaign upper18272.180750s including20%futuremargin+1200reserve, below
unchanged approved19800s. Actualdevmaxslowdown1.001292 is below retained1.025455.

Registered confirmation submitted17:47:55UTC through solemonitor5314 in owned
`tta-guidance-components-a100-20261028-r4`; wrapper SHAb507da40…ebe8. All15/four
heads/new140IDs per dataset/seeds0,1,2,4frozenconditions/24jobs. First fresh-cache
ModelNet batch is observed; actual confirmation results remain pending. No new
GPU scope/ID/readout/trial/cap expansion, so existing Oct26 approval applies.
No runtime teardown authorized. Next: retain all180physical new NPZs and raw
confirmation Drive/local bytes, then object-cluster paired analysis across all
three seeds (280objects, not840), all15/macro14 and qualified policy costs.
Evidence under recovery_20261028_r4/local_development_analysis_v1 and
development_local_acceptance_verification.json; actual source remains8169271
plus sealed73d9fbe1…b37b8/manifest599b8dcb…8eef, separate from local branch HEAD.


[Run/Open] 2026-10-28 18:28UTC: frozen confirmation seed0 completed all8jobs
(fourconditions×two datasets); seed1 spectral-candidate ModelNet is now active.
No partial accuracy ranking has been selected or used to modify the freeze.
Ledger retained locally as recovery_20261028_r4/confirmation_live_ledger_seed0_end.json;
all per-corruption/final worker receipts remain on owned Drive. Solemonitor5314
and bounded host-awake support remain active. Seeds1/2 and independent final
96ZIP/180physical-cache acceptance are pending; no new superiority claim.

[Code/Open] Confirmation transport copy prepared with only STAGE changed from
development to confirmation; SHA d1f31063…79de2,10456bytes,Python3.8 syntax
checked. Shared logic already passed three fixtures and genuine7-part development
transport. Actual confirmation split is not run and requires completion/inactive
ledger; original raw ZIP and Drive identities must remain unchanged.


[Code/CPU/Open] 2026-10-28 18:49UTC: final confirmation acceptance template
prepared (SHAc6a4456a…3312), using actual analyzer/collector metadata schema.
Root read source, reran three synthetic guard fixtures (0.025s), checkedPy3.8
syntax; independent review found no blocker and confirmed the final exclusive-write
delta by reversing it to reproduce the previously reviewed source hash. The CPU-only phase
requires genuine accepted24-job/96-ZIP/180-cache/280-object results, exact frozen
development/calibration/source bindings, original Drive raw/cache bytes and an
inactive under-cap ledger. No estimator rerun, tuning, result selection or GPU
operation; package identity is unbound until actual final local analysis passes.
Source/tests/handoff and final_acceptance_preparation.json are inrecovery_20261028_r4.
Actual confirmation remains running; final binding/acceptance is unexecuted.


[Run/Open] 2026-10-28 19:09UTC: confirmation seeds0and1 completed all16jobs;
third/final seed2 is now running, starting spectral-candidate ModelNet. Actual
cumulative campaign spent12571.930780s at this boundary, below19800s cap; no
configuration/readout/trial change or partial-score-based decision. Local ledger:
recovery_20261028_r4/confirmation_live_ledger_seed1_end.json. Allremaining8jobs,
96rawZIP/180physicalNPZ final acceptance and object-cluster analysis remain
pending. Same ownedA100/solemonitor5314/host-awake support are preserved.


## Confirmation numeric failure; reviewed recovery preparation — 2026-10-28 19:43 UTC

[Run/Code/Open] Registeredslot20 (tunedC`c-step-high`,ModelNet,seed2,eta=gamma=.03)
failed19:32:22UTC with`ValueError: Classifier input contains nonfinite XYZ coordinates`.
Uniform/Gaussian both70-batches and Backgroundbatch0 completed; the failure is
in Backgroundbatch1 (selectedpositions70–139), during a clean trajectory decode
probe before classification. ModelNet/DGCNN preprocessing usesFPS with no
normalization. Existing logs do not identify the exact first invalid tensor or
original object; no unsupported detailed causal claim. Other source/state/data/
native/RNG mismatch is not reported. No numerical clipping/fallback/parameter
change or rerun was made. The finite-input guard correctly stopped the worker.

20registeredjobs complete,slot20partial(status`running` retained in immutable
producer state, overallattempt`failed`),slots21–23unrun. Raw failed phase
1159246521bytes SHA`f586870206961f88d38daa7ed9da632fddfa51356075e114c5f3b98f020ebe5e`
sealed on Drive19:33:34UTC; local raw roundtrip pending. Exact failedreceipt
SHA5ab88805…9754,post-sealfailedledger60c8ba55…736b retains14114.660198s spent.
No complete wrapper marker or confirmation validation exists; previous running
headlines are superseded by this failure. Runtime andsolemonitor5314remain live.

Independent policy review permits a separately reviewed new-output continuation
of onlyUNRUN21(tunedC/Scan/seed2),22–23(originalMN/Scan/seed2), keeping failed20
failed, immutable archives/freeze/source and one cumulative19800s cap. Conservative
upper16384.444877s includes891.487232s futurework,20%margin and1200reserve.
No continuation GPU has started: first preserve physical raw failedattempt+all180
cachebytes; source/native/data/state/RNG/receipt mismatch remains a stop condition.
The original complete-only controller/analyzer/templates stay unchanged and may
not accept this as complete four-condition confirmation. Prepare separate
failure-aware analysis: candidate vs matchedC already has all3seeds; candidate
vsoriginal requires the two unrun original jobs. TunedC's three-seed balanced
endpoint remains unavailable; no dropping the failedcase, two-seed primary
average, setting substitution or hidden failed-slot replacement.
Exact local evidence: recovery_20261028_r4/confirmation_failed_receipt.json,
confirmation_failed_ledger.json,confirmation_failed_job020.log and
confirmation_numeric_failure_diagnosis.json. The numerical failure is research
evidence, not proof of either method's universal superiority.


## Failure preserved; reviewed unrun-only continuation active — 2026-10-28 20:12 UTC

[Run/Code/Open] Failedraw1159246521B/f5868702…be5e physicallydownloaded through
5boundedparts and recombined with identicalSHA. Failure/cacheproof774792352B
SHA2a003500c74feafbd6aa6ac03363ec57e6330a67707fc22c803865081d15b898 matches
actualDrive/localbytes,190memberSHA/CRC and180physicalNPZ extraction. Remote
CPU verification checked all180input/latent/noise/post-RNG hashes underPy3.8/
NumPy1.21.2/Torch2.1.2 withCUDAuninitialized before/after. Scoped to20completed
workers plus180preparations; failedslot20 remains unvalidated for accuracy.
Allfourfailedrawconfigs also matchknownnumericstack and unchangedclassifier/
LIONstates, without audit-error flags. No source or scientificparameter changed.

Reviewedremaining-onlyplan2362B SHA13ff25249aec97eafd6c1fb6db33cd5e3d8462b79331a7cfdfba0e4a06540f6c
binds actual2900B preservationreceipt866135f4…486a and sourcewrapperac5bb60d…9931/
runner eb44e15a…0b09d. Onlyslots21–23; failed20excluded. Native no-build
bootstrap omission was found by review beforelaunch, fixed with targeted
RED→GREEN regression; root8fixturespassed0.196s and independentdelta review
accepted. Originalscientificcontroller/plan/freeze/failedarchive stayunchanged.
NewCPUpreflight actuallyloaded12strictselectedheadconfigs withoutforward/CUDA
andcheckedsource/data/checkpoint/nativeidentities; downloadedproofSHAd64cc5af…4ef8.

Operatorwrapper submitted20:05:33UTC via5314; freshCPUproof passed and first
remainingGPUworker began20:06:42UTC in newimmutableROOT/Driveoutput
`sensitivity_confirmation_remaining_20261028T200610144036Z`. Slot21(tunedC/Scan/
seed2)completed; slot22(originalMN/seed2)nowruns, then23(originalScan/seed2).
Nofailed-slotretry, resultreplacement orwholefour-condition-complete claim.
Conservativewholecampaignupper16384.444877s retains existing19800scap; original
failedledger/history and postsealspent14114.660198s formthepreservedprefix.

[Code/Open] Separatefailure-awareanalysis is beingvalidated. A rawZIPprefix
I/Obug was found before scoring and confined to newadapter Bundle normalization.
Failedraws omit onlydynamicexecutedsourceinventory entryshared_trajectory.py;
allotherrequiredentriesmatch. Rootverifiedall4 embeddedwhole sourceZIPs exactly
73d9fbe1…b37b8 and member6524B/dce0e8b2…f2fcda plusknowntracepaths. Explicit
failed-provenance-only recovery may retainthisone missingentry fromexactarchived
bytes, withmissingoriginalkey/recoveryproof recorded; anywrongbytes orother
missingentry stops. Rawbytes/status andcomplete-onlyguards stayunchanged. No
failedprefixaccuracy accepted, no estimatormath/readout change.
Localproofs: failure_preservation_receipt.json,failure_preservation_local_validation.json,
failed_worker_partial_metadata_local_validation.json and
failed_shared_trajectory_source_inventory_proof.json underrecovery_20261028_r4.
Finalremainingarchive/postsealledger/failure-aware independentanalysis and
Drive/localpartialacceptance remainpending.


## Sensitivity campaign accepted with explicit numerical failure — 2026-10-28 20:35 UTC

[Run/Decision/Inference] All24registeredconfirmation slotsare recorded:23complete,
failed20retained; reviewed21–23allpassed. Independentfailure-awareanalysisaccepts
92full+4failedZIPs,180physicalNPZs and280objectclusters. Originalterminal53.763889%,
matchedCprobmean53.013889%,CS-L2rho.1probmean52.962302%. Original−CS+.801587pp,
conditional95%CI[+.148760,+1.557589]; wholepolicy/host/readoutcomparison,5/8cases.
MatchedC−CS+.051587pp,CI[-.125,+.234127]crosses0; noisolatedCDsuperiority/equivalence.
TunedCthree-seedendpointnullbecauseoneModelNetseed2Backgroundprobe numerical
failure; no2seedmean/sampledrop/replacement. No universalwinnerclaim.

Actualpartialacceptancebinder722266B/dc5b854e…68df storedonfreshDrive and
physicallydownloaded withsameSHA/CRC/analysisbytes. Statusaccepted_partial_confirmation_with_numerical_failure,
notfullfour-conditioncomplete. Finalpostsealspent14836.554263s<19800s. Original
failedrawf5868702…be5e,remaining7b676a7b…ccd6d,cacheproof2a003500…b898 remainimmutable.
AllplannedGPUworkrecorded; runtime+5314preservedwithoutteardownauthorization.
Cleanresults,percasevalues,source/coverage/failure/CIlimits andexactproofpaths:
knowledge/guidance_component_sensitivity_results_20261028.md and
recovery_20261028_r4/final_local_verification.json.


## Agent B runtime closed after accepted campaign — 2026-10-29 local / 2026-10-28 22:04 UTC

[User report/Run/Decision] The user authorized closing the completed campaign and
recording its results: “Colab oturumu kapatıp sonuçları knowledge dosyalarına
ekleyebilirsin”. Monitor5314 exited0 at22:02:52.511932UTC, then the explicit CLI
stop targeted `tta-guidance-components-a100-20261028-r4` (A100 High RAM,
endpoint `gpu-a100-hm-euw4a0-3s9zrjduwd03k`). A fresh server inventory by22:04:19UTC
returned `No active sessions found on server.` The exact server release timestamp
is not available. Four existing Drive-roundtrip archives were rehashed locally
before teardown; all bytes/SHA-256 matched `final_local_verification.json`.

Accepted scientific status remains `accepted_partial_confirmation_with_numerical_failure`:
23 successful jobs plus preserved numerical failure20. No retry, replacement,
new GPU work or changed metric/inference occurred during closure. Campaign time
remains14836.554263s under the19800s approved cap. Original whole-policy mean
53.763889% versus C+S52.962302% does not establish universal or isolated-loss
Chamfer superiority; the matched C−CS interval still crosses zero. No scheduled
GPU work remains. This entry supersedes prior live-runtime/monitor statements;
future generalization or failure diagnosis remains separately scoped.

[Accepted results](guidance_component_sensitivity_results_20261028.md) ·
[Closure evidence](../result/preparation/guidance_component_sensitivity_20261025/recovery_20261028_r4/runtime_closure_20261029.json). Scientific source remains
`8169271c9058995e8dc128062848a253baa13f10` plus the unchanged sealed snapshot;
closure documentation starts from operator HEAD `03416e8`.
