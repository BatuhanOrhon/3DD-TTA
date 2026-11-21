# CloudFixer entropy-weighted Probability Mean — 2026-11-15

**Current status: complete and independently verified (CPU postprocessing).**
Using each N's canonical four-candidate readout, entropy weighting improves N8 slightly but
reduces N12 equal-PM accuracy. Exponential beta1 gives1258/1920=65.520833%
for N8 (+8 correct/.416667pp) and1253/1920=65.260417% for N12 (-4/-.208333pp).
The primary parameter-free1-H/log40 rule gives1253/1252correct respectively.
No new diffusion/classifier inference, baseline promotion or independent
confirmation. Detailed27-readout/405-corruption tables and predictions/weights
are archived; CPU Drive-transfer session remains assigned.

[User request] Test lower-entropy weighted PM for selected N8 andN12; retain
allfindings inknowledge. User additionallyrequests a newColab session and
Drive mount consent. No runtime teardown authorization.

[Predeclared design, before weighted accuracy inspection] Reuse pinned
N-grid420unitZIP SHA2a4c544f2f08de453205f556689a47a1aedabf37be4b87e07d8c9348e68d58f7
anddenseN8 ZIP509ea9699cd2fcde8542fc52564c03146a1f2f30db0d4e20d176e1aa479506df.
Same128originalIDs/all15corruptions/s5/seed0/B32/K80/tau.1/retained.95.
Primarycomparison usescanonical4candidates forbothN (N8:70/50/30/0;
N12:67/47/27/0). Secondary N8dense9 only; N12dense13 notavailable orrequested.
Primaryparameter-freeweight w∝1-H/log40 reuses existinglegacy helper.
Secondarypredeclared sensitivity w∝exp(-beta*H), Hinnats, beta0,.5,1,2,4,8;
beta0 mustreproduceequalPM exactly. Weights computed perobject/timewithout
labels; labels onlyfor reporting. No per-corruption adaptation, temporal
windowselection, weight fitting or new model inference. Rankingsareexploratory
onreuseddevelopmentdata, not independentconfirmation.

[Scope] CPUofflineanalysis ofsavedlogits; no sampler/classifier modification
or GPUre-evaluation needed. NewCPUColab isDrive-transfer/backup only, retaining
previousCPUtransfer workflow ratherthan running GPU-specific10-cell setup on
a CPU. No model/research cells executed onthatVM. WholeCPUtransferphase cap
60min includingmount/retries; estimate5–15min plusinteractiveconsent; GPUcost0
(newCPUonly), actualCLIaccountusage tobechecked. Fresh task-specific owned
sessionplannedtta-cloudfixer-weighted-pm-drive-cpu-20261115-r1.

[Run/Inventory] ColabCLI0.7.4 serverinventorycurrently0assignments; previous
nstepA100 is nolongerassigned, no agentteardownissued. No oldruntime reused.
Sourcecheckoutcb6925a; officialscientificsource remains66e4b8a. Newrawanalysis
output freshUTC directory underresult/modelnet40_c/cloudfixer_entropy_weighted/.
Keep7standardfiles+weightedpredictions/config/source/rawSHA/pairedcases and
per-corruptionresults. BackuptofreshDrivefolder afteruseroperatedmount;
verifySHA/CRC/readback andpreserveCPUVM until separatelyauthorizedclosure.

[Run/Transfer] NewCPUcreated15:20UTC, endpointm-s-usc1c9-1lut8c3l2e92z,
userTerminalCLI drivemount consent completed; solekernelb3d4bb1b-6bd3-43e7-ad8a-65610bad8bf3
verifiedidle15:21:38UTC andpinnedforfutureCLI. Monitor93130 started15:23:55UTC
asrootsoleowner afteruserhandoff, CPU/Standard; no VMrestart/modelinference.
Drive rawN-grid anddenseN8 ZIPs downloadedthroughnewVM, SHAexactlymatchaccepted
2a4c544f.../509ea969... . This provides independentsecondVM evidence thatprior
Drive rawarchives persisted afteroldGPU loss. Localtransfercopies in
result/preparation/cloudfixer_entropy_weighted_20261115/.

[Code/Verification] Newofflineanalyzer scripts/analyze_cloudfixer_weighted_pm.py
reusesexistingnormalized1-Hhelper andcompleteoriginal420/60registeredarchive
validator. Predeclaredbeta0,.5,1,2,4,8 fixedbeforeaccuracy inspection. Fournew
behavior tests initiallyfailedagainststub; implementationpassed13focused
new+legacy+archive-analysis tests. Python3.8AST passes. No sampler/model edits.
Initialanalysis20261115-152910_saved_logits complete27policies/405corruption
rows; subsequentfreshattempt repairsmissingrun_id/dataset/severity/seedfields
inper-corruption CSV only. Firstattempt retainedimmutable, no weight/grid
changes afterresultsinspection; compareallrawpredictions/weights beforeacceptance.


## Accepted weighted-PM screen

[Run] Accepted artifact20261115-153117_saved_logits. Local CPU analysis took
1.793042s including input validation; this is not TTA inference latency.
All15 ModelNet40-C severity5 corruptions, same128 original development IDs,
1920object-corruption pairs perview, seed0/B32/K80/SCDtau.1/retained.95;
frozen CloudFixer XYZ diffusion andPoint-MAE predictions reused.

For each object and candidate t, p_t=softmax(logits_t),
H_t=-sum_c p_tc log(p_tc) in nats; weighted PM=sum_t w_t p_t.
Primary rule: w_t=(1-H_t/log40)/sum_s(1-H_s/log40), equal fallback ifallHmax.
Sensitivity: w_t=exp(-beta H_t)/sum_s exp(-beta H_s), beta0,.5,1,2,4,8
registered before seeing weighted accuracies. Beta0 exactly equals original PM.
The classifier softmax temperature and diffusion/SCD parameters do not change.
Only the label-free weighting function is reused from the legacy macro14
analyzer; its macro14/corruption-selection protocol is not reused. All15
corruptions, including Background, are retained here.
Weights use no labels or corruption-dependent routing. Evaluation labels are
used for these development summaries and post-hoc coefficient comparison.

Primary candidates: N8[70,50,30,0], N12[67,47,27,0]. Secondary N8dense9 uses
[80,70,60,50,40,30,20,10,0]; N12dense13 was not generated. Changing candidate
count/time distribution is a separate condition from changing weights.
All percent values below are equal-corruption means, identical to micro
accuracy because eachcorruption has128examples; not class-balanced accuracy.

| Rule | N8fourcorrect/1920 | N8four% | N12fourcorrect/1920 | N12four% | N8ninecorrect/1920 | N8nine% |
|:---|---:|---:|---:|---:|---:|---:|
| equal_pm | 1250 | 65.104167 | 1257 | 65.468750 | 1256 | 65.416667 |
| linear_entropy | 1253 | 65.260417 | 1252 | 65.208333 | 1255 | 65.364583 |
| exp_beta_0 | 1250 | 65.104167 | 1257 | 65.468750 | 1256 | 65.416667 |
| exp_beta_0.5 | 1253 | 65.260417 | 1251 | 65.156250 | 1257 | 65.468750 |
| exp_beta_1 | 1258 | 65.520833 | 1253 | 65.260417 | 1254 | 65.312500 |
| exp_beta_2 | 1258 | 65.520833 | 1252 | 65.208333 | 1255 | 65.364583 |
| exp_beta_4 | 1257 | 65.468750 | 1252 | 65.208333 | 1253 | 65.260417 |
| exp_beta_8 | 1258 | 65.520833 | 1251 | 65.156250 | 1248 | 65.000000 |
| terminal | 1269 | 66.093750 | 1235 | 64.322917 | 1269 | 66.093750 |

[Paired results] Primarylinear rule repairs4equal-PM errors andbreaks1correct
forN8 (net+3/.15625pp); N12 repairs2/breaks7 (net-5/-.260417pp).
Atbeta1, N8 repairs11/breaks3 (net+8/.416667pp), N12 repairs5/breaks9
(net-4/-.208333pp). N8beta1 improves6corruptions/ties9, none declineinaggregate;
N12 improves3/declines6/ties6. See405per-corruption rows for negative cases.
N8beta1 still11correct/.572917pp below its terminal1269/1920.

[Secondary] N8dense9 parameter-free weighting loses1correct; beta.5 gains1;
beta8 loses8 versus own1256-correct equal9PM. No reliable general improvement
from increasing entropy strength. N8four bestnonzero beta1/2/8 tieat1258, yet
all nonzero N12 exponential settings are below its1257-correct equalPM.

[Inference/Decision] Low entropy is correlated with correctness but thisdoes
not guarantee better probability aggregation. The positiveN8four andnegative
N12four/N8dense9 cases are both retained. No automaticreadout/baselinechange.
Coefficients and candidate counts remain exploratory choices onreusedlabels;
small differences do not establish statisticalsignificance or generalization.
No new temporalwindow, inverseH rule, per-corruption coefficient or learned
selector was introduced after seeing the screen. Independentvalidation/seeds
are needed before promoting a weightedPM policy.

## Independent verification and provenance

[Verification] Both input ZIP SHAs match accepted immutable files after
Drive→new CPU VM→Mac transfer. Completeoriginal420/60registeredunit validation
(source/environment/model identities, coverage andprobability/entropy checks)
passes. Original-label SHA matches; allthreeviewbaselines reproduce priorcounts.
An independent NumPy implementation rederivedsoftmax/entropy from rawlogits,
linearconfidence usinglog40-H andunshiftedexponentials, then verifiedall27
prediction/weight arrays,405per-corruption rows, pairedrepair/break counts and
source snapshots. No newanalyzer/helper imported inindependentrecount.
Firstmetadata-onlyattempt andacceptedrun haveidenticalall63NPZ arrays.
13focused tests pass; Python3.8AST passes. Scientificsampler/runner unchanged.

[Artifacts] Localresult/modelnet40_c/cloudfixer_entropy_weighted/:

- Preliminary20261115-152910_saved_logits.zip,1719379B,
  SHA8ca90bb8e7a040efa35ec3db3f211d54f4b2598d4db319e27058f97a44fb7801;
  superseded onlyfor missing per-corruptioncontext columns, retained.
- Accepted20261115-153117_saved_logits.zip,1723588B,
  SHA9af29a2d70e5ad778775a9c0943f24c07f7bf2b93cda352256e607f84694fbd8.
Bothinclude7standardfiles, source snapshots, completeinputvalidation analyses
andpredictions/weights. Acceptedalsoincludesindependentrecount+source/tests.
Compact evidence inknowledge/evidence/cloudfixer_entropy_weighted_pm_20261115/.

[Runtime] NewCPUcreated15:20UTC, mountcompletedbyuser; no GPUallocated.
CLIaccount snapshot454.68CU/0.08CUhour/1CPUassignment (CPUrate is nonzero;
priorGPUcost isnot thisphasecost). PlannedGPUcost0 meant noGPU, not freeCPU.
Wholetransferphasecap60min retained; recordallmount/retry/backupwalltime.
Ownedtta-cloudfixer-weighted-pm-drive-cpu-20261115-r1,
endpointm-s-usc1c9-1lut8c3l2e92z, primarykernelb3d4bb1b-6bd3-43e7-ad8a-65610bad8bf3,
solemonitor93130. NoGPU/native setup ormodelcells executed on transferVM.
No completion-and-close authorization; preserveCPU until explicitclosure.


## Drive backup and final handoff

[Run/Archive] Transfer completed2026-11-15T15:43:30UTC (exact outermarker+CLI0).
Drivefolder`MyDrive/thesis/result/cloudfixer_weighted_pm_20261115_6b791dad1d16/`
contains both immutable attempt ZIPs plus434020B knowledge/sourcebundle
`20261115-154230_weighted_knowledge.zip`,
SHAd884b82c139580335484ac36250342713e20537b93b4fd27305fc36d8ffcec2a.
Acceptedanalysis1723588B/SHA9af29a2d... downloadedbackthroughCPUDriveFS→Mac;
size/SHA matchesoriginal. DriveFSfsync/rename/readback andlocalredundancy pass;
separateVM sync fornewweightedfiles isnotproven. PriorN-grid/denseinput raw
files were separatelyvalidated throughthisnewVM andtheiroriginalSHAs match.

Operator reuse ofexactcloudfixer_store.py, task/owner/sourceSHA manifest,
persistedwholeCPUsetupbudget and300s transferoperation allow idempotentretry
without overwritingdifferentcontent. No failedoperationorbudget reset.
Allocation-to-backup ~23min, within60min cap; atobserved.08CU/hr ~.031CU
rate-derivedCPUestimate, notisolatedbillingaudit. Numericalanalysisitself1.793s.
CPUremainsassignedunderrootmonitor93130; no runtime teardownauthorization
or action. Code/math tests13pass, Python3.8AST, all27/405independentrecounts
andDriveacceptedZIP readback pass. No newdiffusion/classifierinference.

## 2026-11-18 — subsequent held-out confirmation and transfer status

[Run/Code/Inference] On the separate256/all15/s5/three-seed confirmation cohort, pooled entropy AUC~.867–.868 falls to~.635–.647 for within-trajectory correct/wrong pairs. Fixed linear and exp-beta rules do not consistently improve locked late equal PM; stronger weighting harms all-step PM in allthree schedules. All registered beta/window results are retained without confirmation-based selection. This supersedes neither the old128 data nor their posthoc status. [Full evidence and limitations](cloudfixer_pm_hypotheses_confirmation_20261118.md). [Current execution](cloudfixer_two_colab_execution_20261118.md).
