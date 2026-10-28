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
