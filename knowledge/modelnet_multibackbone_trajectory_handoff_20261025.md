# ModelNet three-head GSD trajectory readouts — completed study

Date:2026-10-25. Evidence: [User report], [Code], [Run], [Open], [Inference].
Audit base:`gsd-multibackbone-study@7017fa8` (inspect actual HEAD/status on resume).
Status: **requested9 policy rows complete and verified; owned runtime/monitor closed with user authorization; Agent B preserved**.

## Accepted full results — 2026-10-25 15:09 UTC

[Run/Artifact/Code] **Agent A's requested scope is complete.** Full phase
`20261025T133443Z_mntraj_all15` completed without interruption on the owned
A100 runtime. All15 severity5 corruptions,seed0,2468 original IDs each,
40 classes,3 frozen heads:540 shared batches,111060 head-input events and
444240 candidate predictions. These repeat2468 original objects across
corruptions/heads/candidates; they are not independent sample counts.

| Backbone | Same-run terminal % | Minimum entropy % | Probability mean % | Entropy Δpp | Mean Δpp |
|---|---:|---:|---:|---:|---:|
| DGCNN | 63.814155 | 64.219341 | 64.548892 | +0.405186 | +0.734738 |
| CurveNet | 65.170178 | 65.829281 | 66.083198 | +0.659103 | +0.913020 |
| PointNeXt | 66.329011 | 66.820637 | 67.215019 | +0.491626 | +0.886008 |

[Run/Inference] Both fixed readouts improve their own terminal for all3 heads
in this seed. Probability mean ranks highest of the3 registered policies,
with +0.734738/+0.913020/+0.886008pp (DGCNN/CurveNet/PointNeXt). It corrects/
harms1352/1080,1312/974 and1304/976 events. Minimum entropy gives
+0.405186/+0.659103/+0.491626pp, correcting/harming1417/1267,1410/1166 and
1367/1185 events. No target label fitted a rule, threshold or hyperparameter.
These gains do not establish the prospective1–2pp target or seed stability.

[Run] Probability mean improves/worsens/ties11/3/1,11/3/1 and12/3/0
corruption rows. Minimum entropy has8/7/0,9/5/1 and12/3/0. Mean's largest
loss is DGCNN Occlusion−0.931929pp, CurveNet Background−0.931929pp,
PointNeXt Background−3.606159pp. Minimum entropy loses4.740681pp on
PointNeXt Background. Preserve this heterogeneity; no corruption-specific
readout/horizon tuning or universal per-corruption win is inferred.

[Run/Inference] Against historical separately run original3DD, the new means
remain lower by1.037277/0.988655/1.077796pp. This is a descriptive whole-policy
comparison; original changes batch/horizon/style/scheduler and does not
isolate spectral contribution. Source/paper checkpoint gaps remain. New
terminals differ from historical GSD terminals by−0.018909/−0.016207/
−0.143166pp; equal seed does not establish cross-process identity. All readout
deltas above use the new same-run terminal. The new DGCNN trace is complete
and has its own final LION proof; it does not repair/relabel the historical
13+2 composite or its missing old13 final state.

### Exact protocol and validation

[Code/Run] Locked combined GSD B70/M800/beta2/spectral1.17/chamfer1,
eta(local)=gamma(style)=.01,p=.95,total100,normal10/Background30,
raw/eval LION/EMAoff,updated style,DDIM finalalphaFalse,legacy graph/isolate
policy. Clean probes3/5/8 or8/15/23 plus distinct ordinary terminal.
Temperature1, class-normalized entropy with terminal-first/earliest-probe ties;
equal probability mean. One head-independent diffusion trajectory and one
decode per candidate are shared across3 isolated frozen heads. ModelNet's
rotate-back/output-normalization path is retained. No logit mean, entropy
weighting, tuning, extra seed, spectral-off arm, early stop, source/original
rerun, Point-MAE or Scan full run was added.

[Run] Independent raw/member SHA/CRC/path checks, all IDs/counts/labels,
B70/last18 remainder,40-class finite candidates, terminal agreement,
CPU/GPU classifier hashes, frozen classifier/LION states, shared geometry/
trajectory, probe buffers/RNG/live inputs, actual native no-build bindings and
source snapshot all pass. Physical Drive audit verifies1620 batch receipts,
45 corruption receipts and3 final model receipts; actual Drive/local raw and
analysis ZIP identities match. Accepted binder1696973bytes SHA
`ff6837972ed38043d98430630af78342d2b26b8d323e61624aba08488a245308`,
1673 safe/CRC/member-verified files. Standard seven-file inference schemas
are retained; full candidate logits are stored in batch shards.

### Source, artifacts and measured cost

[Code/Run] Implementation665ed07, source snapshot
`a04717337d6732b74bef18a956381f6979a83f208ca248dd2a113aeab7979015`,
manifest`2326e8e07f30bd46f5e29a7d9d169b45fa71d5a3aac7fd9c5ac4fb3f56e3a3a8`.
Actual remote base remains`dev@a458cd46bba1288a3f41302548bc777700cbc7d4`;
the pinned snapshot/overlay identifies executed code separately. Four legacy
native binaries are pinned, and fresh setup natives were preserved before
restoration. Exact checkpoints/data/source identities are in the sealed
manifest and raw configs. Code/CPU gates:62 focused/332 full tests and an
independent read-only review; actual sealed-source strict CPU/native proof
and1680-candidate GPU smoke precede full launch.

| Backbone | Raw local ZIP under `result/modelnet40_c/gsd_inspired_latent_spectral_guidance/` | Bytes | SHA-256 |
|---|---|---:|---|
| DGCNN | `20261025-133445_mntraj-dgcnn-seed0.zip` | 44709995 | `1710fbe36deaffa42fa3a6bb495ea9cbeca6c9ab77bab6486b162c611ca12996` |
| CurveNet | `20261025-133446_mntraj-curvenet-seed0.zip` | 43734611 | `6bce346cc41de5045b3fb0ef378aa20be1db8bfecd0ea952e52e82d6e62b1340` |
| PointNeXt | `20261025-133447_mntraj-pointnext-seed0.zip` | 43108376 | `8c8309f735ef02a094e4a3052a390446567537c908bb7ec5b9ef4539b23f8314` |

Local preparation root:
`result/preparation/modelnet_trajectory_study_20261025/`.
Read `all15_analysis/{validation.json,macro.csv,per_corruption.csv}`,
`all15_local_acceptance.json`, `all15_final_local_acceptance.json`,
`full_readout_findings.json`, `gpu_all15_receipt.json` and
`all15_accepted_evidence_drive_roundtrip.zip`. Raw CSV accuracies use fractions;
derived `accuracy_pct` is percent, deltas are absolute percentage points.

Drive root:`MyDrive/thesis/result/modelnet_trajectory_study_20261025/`.
Raw finals are under`runs/modelnet40_c/gsd_inspired_latent_spectral_guidance/`
and each run ID's`final.zip`. Accepted binder is under
`preparation/mntraj_all15_accepted_20261025T150536Z/`.

[Run] Full job wall time=86.943923min; smoke+full=87.885149min.
Whole-process peak allocated memory=35.383849GiB. These include this phase's
job overhead and probe work; they exclude setup, user authorization wait,
later CPU analysis and runtime lifetime/billing. No matched speedup or early-
stop latency was measured. User explicitly removed the earlier150min duration
cap with “Onaylıyorum ne kadar sürdüğü önemli değil başlat”; original approval,
stopped duration estimate and ledger are preserved. No research scope changed.

### Completed report backup

[Run/Artifact] The final report/source/operator/analysis/knowledge evidence
bundle is7242077bytes, SHA
`004f7999933d25a60e109bbe2521ccba2aaf279d9894a4adddfbf46c198f4227`.
Actual Drive/local SHA, CRC, safe paths and34 member identities pass.
Local:`result/preparation/modelnet_trajectory_study_20261025/completed_report_drive_roundtrip.zip`.
Drive:`MyDrive/thesis/result/modelnet_trajectory_study_20261025/preparation/20261025T151222Z_completed_report/mntraj_completed_report.zip`.
The bundle contains the sealed executed source snapshot and the accepted
physical-receipt binder; raw3 model ZIPs remain separately immutable.

### Lifecycle, remaining uncertainty and next action

[Run/User report] At17:11 UTC on2026-10-25, after the completed task and
verified archives, the user authorized closure: “Kapatabiliriz, Agent B'nin
oturumunu kapatmamaya dikkat et”. Only own session
`tta-mn-trajectory-a100-20261025-r1`, endpoint
`gpu-a100-hm-euw4a0-axkeusdcat4y`, was stopped through the named Colab CLI
stop command. MonitorPID25910/TTY30907 exited0 at17:11:34UTC; its pending
liveness probe ended with returncode1 after closure, not a research failure.
Fresh server/process verification confirms own runtime, monitor and child47297
absent, while Agent B session`tta-guidance-components-a100-20261025-r1`,
endpoint`gpu-a100-hm-kkb-ass1c2-36gbz99z2hf98`, and monitor28301 remain.
Peer worktree/Drive and unknown CPU assignment were untouched. Local
`closure_authorization.json`, `closure_verification.json`, updated
`runtime_owner.json` and monitor log under the preparation root retain the
audit. Final report bytes/SHA were rechecked locally before recording closure.
Accepted results commit:`9d0d1700deb4afcc018d3d769f17e17163884905`.
No runtime execution remains for Agent A; future GPU work needs a new runtime.
The immutable Drive report predates closure; closure evidence is local/Git.
Controllercell153 is finished; do not rerun it.

[Open/Decision] No requested readout row remains missing. Additional seeds,
same-host spectral attribution and broader method/tuning experiments remain
separate future work (Agent B's study is unchanged). Existing saved logits
support a separately requested offline logit-mean comparison without new
diffusion. One seed, prior benchmark exposure and head-specific source gaps
limit generalization/novelty claims. Changed/missing IDs, labels, checkpoints,
source/native state or receipt hashes would invalidate affected acceptance;
replication or same-host component controls may overturn broader hypotheses.
The historical records below remain dated history, superseded for current
Agent A status by this completed result.

## Live full observation — 2026-10-25 13:36 UTC

[User report/Run/Code/Open] User explicitly states “Onaylıyorum ne kadar
sürdüğü önemli değil başlat”. This supersedes the150min cap and the pending
160min question: elapsed-time restriction is removed for the unchanged
Agent A scope, not for tuning or new experiments. Original authorization and
ledger are preserved; `authorization_v2.json` records the exact instruction.
The actual ledger retains56.473518321s of smoke and records no duration cap.

Smoke physical acceptance is complete:6 batch/6 corruption/3 final receipts
and all data/state/source checks pass. Accepted26172byte evidence ZIP SHA
`da5799856d76d3d0f05716ab94cfd817fac70cc88c2bc9636ea2d530d91b9126`
passes actual Drive/local SHA, CRC/path/member checks. New full driver SHA
`b2e530d046c6dbc1b6ae20e3dc501e855dd54d48f4ffa7c21354968875af048e`
has no elapsed-time kill timer; all source/data/state/receipt failure guards
remain. The source manifest2326e8e0…e3a3a8 and locked guidance math are unchanged.

Full phase `20261025T133443Z_mntraj_all15` starts13:34:43UTC in the same owned
A100 session/endpoint, sole monitor25910/TTY30907. GSD-only ModelNet all15,
severity5/seed0/B70, DGCNN+CurveNet+PointNeXt;540 shared batches/37020 inputs
per head/444240 candidate predictions/9 policy rows expected. No full result
yet. Batch/corruption/model evidence persists under the contracted own Drive
root. Current live counters are in `all15_live_progress.json`; controller
cell153 drains the same monitor and saves its primary output. Read that
controller/progress before any new runtime execution; never start a duplicate.

Local/Drive preparation roots and lifecycle contract below stay valid. Own
runtime/monitor remain live and peer state is untouched; no teardown right.
Next: monitor to completion, retrieve raw+physical Drive evidence, validate
all original IDs/labels/states/source/native identities, independently score
same-run terminal/minimum entropy/probability mean, archive and update memory.
This record supersedes the stopped-budget gate below; it remains as history.

## Current gate — 2026-10-25 13:10 UTC

[Run/Code/User report/Open] Own A100 setup and actual-source CPU proof are
complete. Canonical cells1,3–10 passed; only the known cell2 fastrlock failure
was deferred after Python3.8 verification. Actual device is NVIDIA
A100-SXM4-80GB, Python3.8.20/PyTorch2.1.2+cu121/CUDA12.1, remote base
`dev@a458cd46bba1288a3f41302548bc777700cbc7d4`. Setup's5 native markers,
CUDA/FPS/KNN/Chamfer/EMD,15 ModelNet arrays and checkpoint identities pass.
Drive/local setup ZIP21427bytes SHA
`56b4b7e224fa7c0eaf6615837590ecd7fd63b07c638643e6367c9c52af26e5ce`;
CRC/safe paths/all17 member identities pass. Fresh native binaries were
preserved in own VM before restoring4 pinned reference binaries. Actual
sealed-source3 strict classifier loads and matched-source bindings pass with
CUDA uninitialized and no JIT build. CPU ZIP59303bytes SHA
`8c39279aed60173f7d9a74890697d1b3744da86e976e3c8e62ef1680893728c1`
passes actual Drive/local SHA and9-member/schema checks.

The first smoke launch failed before model/job launch because the fresh
`gpu_phases` parent was absent. The failed script is preserved asv1; creating
parents with `exist_ok=False` for the attempt fixes it, and an actual fresh-root/
overwrite-rejection self-check passes. No guidance code or protocol changed.
Successful smoke phase `20261025T125937Z_mntraj_pilot` took56.473518321s:
Gaussian+Background, IDs0–69 each,3 heads,1680 candidate predictions. All3
raw ZIPs,6 physical batch receipts, indexed labels,40-class candidates,
terminal agreement, frozen classifier/LION states, CPU/GPU classifier state
identity, native bindings and cross-head shared trajectory/RNG/purity pass.
This is execution evidence only, not benchmark accuracy.

**Do not start all15 yet.** The registered duration formula estimates
6414.616657s=106.910278min centrally and9101.873912s=151.697899min
conservatively (40percent compute margin plus I/O). The latter exceeds the
user's existing9000s/150min cap; the gate stopped correctly. A specific
9000→9600s/160min amendment has been asked asynchronously but is not yet
approved. The original approval remains valid only at150min; elapsed time
is not consent. Scope/source/seed/protocol stay fixed, and56.473518321s
remains consumed in the common ledger. No all15 job has launched.

Data validation is independently sealed in
`pilot_data_validation_drive_roundtrip.zip`,13587bytes SHA
`996e95250c36007a5d9e2cc84488e5244d35e26aae87c050b8a5d648e4da9584`;
actual Drive/local SHA and member checks pass. Drive root:
`MyDrive/thesis/result/modelnet_trajectory_study_20261025/`, data package
under `preparation/20261025T130859Z_pilot_data_validation/`.
All local receipts, raw ZIPs, partial analyzer tables, timer inspection and
failure logs remain under the task-local preparation root and canonical
`result/modelnet40_c/gsd_inspired_latent_spectral_guidance/`.

Own runtime/endpoint and sole monitor25910/TTY30907 remain live. The bounded
host-awake process36287 is still active. Next action depends on the user's
cap reply: if160min is explicitly approved, preserve original authorization,
amend authorization+ledger+driver cap together without resetting consumed
seconds, promote smoke only after physical Drive acceptance, then run all15.
If the user retains150min, do not launch full under this registered gate.
Never terminate this runtime/monitor or touch peer state without authorization.

## Live Agent A state — 2026-10-25 12:25 UTC

[User report/Code/Open] Source batch `665ed07e6443f491a7dd127a768e3fa3e7998eb7`
is pushed to `origin/gsd-modelnet-trajectory`. User explicitly approved:
“90–120 dk fazı, 150 dk üst sınırla onaylıyorum”. This covers the single
smoke+all15+bounded-retry GPU phase with a shared9000s cap; no repeated
approval is needed inside the unchanged scope. Setup/allocation are separate
from research-job wall time; no research GPU job has started.

The new sealed manifest is
`2326e8e07f30bd46f5e29a7d9d169b45fa71d5a3aac7fd9c5ac4fb3f56e3a3a8`,
source snapshot SHA
`a04717337d6732b74bef18a956381f6979a83f208ca248dd2a113aeab7979015`,
delta payload5501084bytes SHA
`f202a8d4dffdb523fca34325175355bab6d46818503d4c7b69f6f52acdc1fe84`.
Local proof validates627 asset files,325 source files, exact snapshot members
and all3 matched source/checkpoint links. Old3 ModelNet GSD archives pass
SHA/CRC and lack candidate logits. Failed local packaging attemptv1 is
preserved; correctedv2 resolves the DGCNN extracted-directory reference to
its accepted sealed ZIP. Actual remote source/native/data/checkpoint proof
and new results remain pending.

Own session/endpoint and sole monitor remain the same as below (PID25910,
TTY30907). Cell1 completed; cell2 failed only at the previously deferred
fastrlock build and the resulting Python3.8 environment was independently
verified. First cell3 mount failed; a nonconcurrent interactive CLI Drive
mount received explicit user consent and completed, then the unchanged cell3
passed. Cell4 is installing dependencies; controllercell96 sequences4–10,
verifying exact markers and native `.complete` state before advancing.
A separate12:24UTC server inventory confirms the owned A100 assignment
exists; peer A100 and unknown CPU assignments are untouched. Host sleep is
boundedly inhibited for14400s (own local exec36287); it is not a guarantee
against Colab-managed loss.

Task-local `authorization.json`, `gpu_phase_proposal.json`,
`preparation_identity.json`, `sealed_local_validation.json`,
`existing_coverage_audit.json`, `runtime_owner.json`, `progress.md`,
`monitor_output.log` and `setup/` preserve exact controls. Remote root is
`/content/studies/mn-trajectory/`; Drive writes stay under
`MyDrive/thesis/result/modelnet_trajectory_study_20261025/`.
Next: finish setup, deploy/verify actual pinned source and native binaries,
retrieve CPU proof through Drive/local SHA, then smoke and independently
accepted full observation. Runtime and monitor must remain live; user has
not authorized teardown.

## Agent A execution — 2026-10-25 12:02 UTC

[Code/Open] User assigned this session to Agent A and requested continuous
monitoring. Own worktree is `/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/modelnet-trajectory`,
branch `gsd-modelnet-trajectory`, derived from committed base
`4c5be0eaf0ac933923129fe26738e0db5f96744a`. Original uncommitted operator
skill/monitor files are read-only operational inputs; peer worktree/runtime
remain untouched.

The minimal ModelNet extension now passes 62 focused CPU tests and the full
332-test suite after two ignored historical test prerequisites were copied
from the original workspace and hash-checked. New tests first failed on the
missing40-class observer argument and Scan-only runner restriction. Strict
analyzer support is explicit (`--dataset modelnet40_c`); default Scan15/581
acceptance is retained. ModelNet candidates are40 classes,2468 ordered IDs,
B70 with18 in the last batch; wrong split/count/labels/dimensions/terminal
and probe-state evidence are rejected. Guidance/graph/scheduler equations,
geometry contracts and checkpoint/preprocessing paths are unchanged. Read-only
independent review reported no material findings; no GPU accuracy yet.

Own A100 High-RAM session `tta-mn-trajectory-a100-20261025-r1`, endpoint
`gpu-a100-hm-euw4a0-axkeusdcat4y`, is executing the ten-cell canonical setup.
Current original-workspace `monitor_colab_runtime.py` owns execution in local
TTY session30907; cell1 completed and cell2 is active on the same endpoint.
Task-local audit root: `result/preparation/modelnet_trajectory_study_20261025/`.
`runtime_owner.json` records the actual monitor/source identities. Full-suite
logs and SHA-verified test prerequisite receipt are in that root. No research
GPU phase has started. Whole-phase approval is pending; prior69.421min
ModelNet reconstruction implies the >60min gate for smoke+all15+retries.
Preserve this runtime and monitor; no teardown authorization exists.

## Parallel assignment amendment — Agent A

[User report/Decision] A second agent now owns guidance-component experiments.
Follow the [shared isolation contract](parallel_colab_agent_contract_20261025.md).
The earlier instruction to continue directly in the base branch is superseded:
derive `gsd-modelnet-trajectory` from the committed `gsd-multibackbone-study`
base in `/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/modelnet-trajectory`.
Use only task-owned Colab sessions named
`tta-mn-trajectory-a100-<UTCdate>-r<N>`, not Agent B's
`tta-guidance-components-a100-<UTCdate>-r<N>` sessions.

Preserve the locked combined-guidance policy in this task. Do not run Agent B's
Chamfer/spectral ablations or change its branch, runtime, monitor, environment,
Drive files or approval. Use explicit own-session CLI operations and separate
task-local preparation/Drive roots defined in the contract. A new worktree does
not include the user's uncommitted operator skill/scripts; read their current
original-workspace versions without modifying them. No workspace/session is
created by this amendment. [Updated copyable prompt](parallel_agent_prompts_20261025.md).

## User-selected next task and accepted coverage

[User report] Prepare a concise next-agent prompt after checking research
memory. The next session should obtain ModelNet40-C GSD minimum-entropy and
equal-probability-mean accuracy for DGCNN, CurveNet and PointNeXt on Colab.
This turn prepares the handoff; it does not start that experiment.

[Run/Code] The current six-case source/original/GSD matrix has18 accuracy
values. Scan has12 accepted policy rows for four heads. ModelNet Point-MAE
already has terminal64.592112%, minimum entropy65.051324%, probability
mean65.156672% on its same trace. The missing scope is exactly the following:

| ModelNet backbone | Accepted historical GSD terminal % | Minimum entropy | Probability mean |
|---|---:|---|---|
| DGCNN |63.833063*|not evaluated|not evaluated|
| CurveNet |65.186386|not evaluated|not evaluated|
| PointNeXt |66.472177|not evaluated|not evaluated|

*DGCNN terminal is the qualified13+2 complete-corruption composite, not an
uninterrupted run. Its old13 final LION state/RNG continuation proof is absent.

[Run/Code] The2026-10-25 archive inventory checks SHA/CRC for the accepted
CurveNet/PointNeXt ModelNet GSD ZIPs and inspects their NPZ members; neither
contains `trajectory_logits`. The DGCNN derived composite NPZs also lack them.
Historical terminal logits cannot reconstruct missing candidate predictions.
New observation is necessary unless matching archived decoded probe clouds
or complete candidate logits are independently located. Do not rerun existing
source-only/original3DD, Scan policies or Point-MAE all15 merely to fill this gap.

## Locked scope and interpretation

[Code/Open] Extend the existing observation path with the smallest reviewed
ModelNet change, preserving the accepted Scan path and default terminal mode.
Use the same published-alternative weights, preprocessing and dataset identities;
this remains **GSD-inspired latent spectral guidance**, not full GSDTTA.

- ModelNet40-C, severity5, seed0, all15 corruptions,2468 IDs each,40 classes.
- Locked GSD B70/M800/beta2/spectral1.17/chamfer1, normal10/Background30,
  raw/eval LION, EMAoff, updated style, final DDIM alphaFalse. Verify resolved
  configurations against prior artifacts rather than relying on this shorthand.
- Three predicted-clean candidates at loops3/5/8, or8/15/23 for Background,
  plus the distinct ordinary terminal. Equal probability mean at temperature1;
  minimum normalized entropy, terminal-first then earliest-probe tie handling.
- One classifier-independent diffusion trajectory, each selected latent decoded
  once, classified by three isolated frozen heads. Expected540 shared batches,
  37020 inputs/head,111060 head-input events and444240 candidate predictions.
- Accept9 policy rows:3 terminal controls plus6 new selection/mean values.
  Compare policies to their **new same-run terminal**. Old terminals remain
  historical references; same seed does not imply cross-process identity.
- Preserve no-label selection, classifier/LION states, RNG/buffers/live-input
  guards, checkpoint/data/native/source identities and indexed coverage.
  Single-seed uncertainty and whole-policy comparison limits still apply.

[User report/Open] Logit mean is recorded as a separate open question. It is
not part of this requested GPU scope; save full logits so later offline
comparison is possible. No new tuning, entropy weighting, early stopping,
source-checkpoint search, seed replication or spectral-off arm is requested here.

## Actual code gates found during the audit

[Code] This is not currently a runnable change of CLI dataset alone:

| Reference | Relevant responsibility / required review |
|---|---|
| `scripts/eval_multibackbone_guidance.py` | `configurations` rejects ModelNet trajectory observation; existing shared reconstruction, observer factory, shards and Drive receipts should be reused. |
| `shared_trajectory.py` | `SharedTrajectoryObserver._logits` hardcodes15 classes. Keep strict dataset/config-derived class validation when supporting40; do not remove shape guards. |
| `scripts/analyze_scan_trajectory.py` | Strict Scan581/15-class/schema validator; adapt/reuse carefully for ModelNet2468/40-class data without accepting mismatched Scan artifacts. |
| `scripts/analyze_legacy_trajectory.py` | Class-agnostic `probabilities` and `policies`; reuse their existing mathematical rules. |
| `tta_gsd_v2.py` | Existing `trajectory_observer` hooks; preserve guidance equations and probe timing. |
| `scripts/eval_multibackbone_guidance.py:decoded_to_geometry` | ModelNet rotates back then normalizes; Scan divides3.3885 and inverse-unormalizes. Never substitute the Scan coordinate contract. |
| `scripts/eval_multibackbone_source.py`, `classifier_backbones.py` | Existing frozen-head loading, source contracts and per-backbone preprocessing. Verify checkpoint hashes from accepted validation. |
| `tests/test_shared_trajectory.py`, `tests/test_multibackbone_scan_trajectory.py` | Existing15-class/Scan restrictions are deliberate tests; extend with40-class coverage and retain Scan/default-mode regressions. |

[Open] Follow CPU/schema/purity verification with genuine Colab smoke and
then full coverage, subject to the whole-phase approval gate. Native encoder
nondeterminism prevents treating independent processes as bitwise parity tests.
Retain strict within-probe state/RNG and same-run terminal evidence.

## Runtime, persistence and approval

[Run] Last independently verified lifecycle:19Oct2026, runtimes/monitors/
host-awake workers closed; the subsequent bounded20:50UTC inventory was empty.
No runtime was queried or allocated in this handoff turn. The next agent must
inspect current server state; do not assume r7/r8/r9 is still available or
terminate unrelated assignments. Preserve local user edits listed by Git.

[Run/Inference/Open] The previous two-head ModelNet GSD full run took
4165.243seconds=69.421min, with3934.254seconds shared reconstruction, before
this new three-head probe work. This is a historical reference, not a new
runtime guarantee. Budget the entire planned GPU phase including smoke/full
observation and retries. A comparable full phase is reasonably expected to
exceed60min; prepare one comprehensive approval request before starting GPU
research. Do not reuse the old campaign approval or split calls to avoid it.

[Code/Open] Use Colab CLI and the operator skill. For a new runtime use the
ten-cell setup: cell1 alone/reconnect to same VM, cells2–10 in order; require
native/data/checkpoint checks. Pin the actually executed snapshot/assets and
remote identity separately from local HEAD. Persist batch candidate logits,
per-corruption archives and each final model bundle to Drive; verify actual
Drive/local SHA-256 before accepting results. Use fresh immutable run IDs and
preserve partial progress. Do not auto-close a runtime without the user's
explicit task-completion and teardown authorization.

## Reading order and exact evidence references

1. `AGENTS.md`, `/Users/batuhan/.codex/RTK.md`, both research/Colab skills;
   `knowledge/README.md`, `thesis_scope.md`, `experiment_protocol.md`.
2. [Accepted final coverage and limits](gsd_multibackbone_final_findings_20261019.md),
   [Scan observer protocol](gsd_scan_trajectory_selection_20261019.md),
   [ModelNet Point-MAE observation](gsd_trajectory_implementation_20261011.md).
3. [Source contracts](gsd_multibackbone_source_results_20261018.md),
   [provenance](gsd_multibackbone_provenance_20261018.json),
   [ModelNet timing and old terminals](gsd_multibackbone_guidance_results_20261018.md),
   [DGCNN composition](gsd_multibackbone_dgcnn_residual_20261019.md).
4. [Paper names/links and attribution](gsd_trajectory_attribution_20261025.md),
   [open questions](open_questions.md), [dated findings](findings_log.md),
   `knowledge/colab_cli_setup_20261017.md`, `result/README.md`.

Existing evidence roots, relative to repo:

- `result/preparation/dgcnn_residual_20261019/complete_matrix/{summary.csv,validation.json}`
  identifies historical source/original/GSD values and exact input archives.
- `result/preparation/scan_trajectory_20261019/all15_analysis/` and
  `all15_accepted_evidence_drive_roundtrip.zip` retain the four-head pattern.
- `result/modelnet40_c/legacy_trajectory/` retains Point-MAE candidate traces.
- Local immutable literature evidence:
  `result/preparation/trajectory_attribution_review_20261025T105306Z.zip`.

Acceptance: validate IDs/counts/class dimensions, labels, candidate names/
ordering, terminal agreement, frozen states/purity, source/native hashes and
Drive/local receipts before reporting6 new accuracy values. Append findings,
update open questions/current study note/README, then commit and push only
accepted task files. A mismatch invalidates the affected result; never fill
missing cells from paper targets or different runs. No result exists yet for
the new ModelNet three-head readout scope.
