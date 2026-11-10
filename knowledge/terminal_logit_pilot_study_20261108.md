# Terminal acceleration and same-logit LM–PM pilot

## 2026-11-10 — bounded pilot complete

[Run/CPU/Inference] [Accepted results and all tables](terminal_logit_pilot_results_20261110.md).
Same-session recovery/setup/smoke/pilot/archive/local admission complete.
Native floor55/80fails remains; descriptive evidence only. Runtime retained.
The dated preparation/loss entries below are history, not current blockers.

[User decision/Code/Open] Execute the [Nov8 handoff](terminal_logit_pilot_agent_handoff_20261108.md)
without another GPU approval question. No full-data expansion. On runtime or
connection loss, preserve evidence and WAIT_USER; no replacement, retry or new
remote cell while state is unknown. Runtime teardown is not authorized.

Owner: terminal/logit pilot agent. Worktree
`/Users/batuhan/.codex/worktrees/c117/3DD-TTA`; branch
`experiment/terminal-logit-pilot-20261110`; fetched base
`bb1526d461081a0cae5401dbfee3c5745624cb1a`. Other checkouts and runtimes are protected.
Evidence labels: [Paper], [Code], [Run], [User report], [Inference], [Open].

### Frozen design and implementation sequence

The user's execution request adopts the five-arm design in the handoff; no
additional design approval is pending. All15 ModelNet40-C corruptions, s5,
frozen40-class Point-MAE, seed0,32 common IDs/B32, three fresh operational
repeats. Repeat0 primary, repeats1/2 reported separately. Two fixed warmups
per arm/corruption. Smoke uses four disjoint IDs/B2 on Gaussian, Background,
Density and LiDAR. IDs come from one default_rng(20261108) permutation before
accuracy, stored in [registration](evidence/terminal_logit_pilot_20261110/registration.json).

T5/P5 and T8/P8 are matched terminal comparisons. Classic uses real `tta.py`
with rates(.01,.01)/(.03,.003); cheap uses tau.1/retained.95, no backward or
style update. P8-PM alone reads clean2/4/6+terminal8; offline LM and PM share
its B×4×40 logits and float64 arithmetic. Saved GPUfloat32 PM differences are
counted separately. No baseline defaults change.

- [x] Extend `lion_guidance_pilot.sample(..., terminal_only=True)` to N5 and
  omit intermediate readouts/snapshots; retain old N8/35 defaults.
- [x] Add bounded runner and CPU reducer; test count/grad/style/policy/finite/ID
  guards and same-logit reductions; run impacted suite and independent review.
- [x] Seal source/registration/actual weights/data/native receipts before GPU.
- [x] Allocate a new uniquely named High-RAM A100; canonical cells1–10 with
  exact markers; only known cell2 fastrlock exception may be deferred.
- [x] Technical smoke: actual scheduler/update and1/4readout counts, frozen
  model hashes, finite outputs, three same-state/RNG repeats; disclose failures.
- [x] Pilot with 45minute soft science budget; initial estimate15–45minutes,
  revise after smoke. Fresh preprocessing/encode per timed arm, balanced order,
  synchronized runtime and allocated/reserved peaks; exclude loading/warmup/
  RNG seeding/hashes/D2H/artifact I/O equally. B32ms/input is not B1 latency.
- [x] Checkpoint each corruption, verify immutable rawZIPs Drive/Mac SHA/CRC,
  then report counts/all15/macro14/per-corruption, paired corrected/harmed/net,
  object-cluster10000draw/seed20261108 CIs per repeat and logit diagnostics.
- [ ] Reconcile latest integration and normal fast-forward push only necessary
  code/tests/knowledge/compact evidence; exclude rawZIPs, setup wrappers/weights.

[Historical preparation/Open] The following source/session/measurement items
were pending at preparation time; actual receipts/results now appear below
and in the accepted report. No strict repeatability or isolated causal
improvement is claimed. Small exposed
cohort/one inference seed and native variation limit conclusions. No −.3pp
noninferiority proof is expected from32objects.

[Code/CPU] Preparation checks:13 new pilot tests pass, including a complete225-case synthetic grid with known same-logit LM/PM decisions and changed-ID rejection. Latest broad project run:887 passed/2skipped/564subtests (the subsequent full-grid fixture separately passes). Python3.8 syntax and diff checks pass. Independent review identified source-seal binding and missing host-contract tests; both corrected. Duplicate cheap-only discarded style projection removed; eval global2style is pure and called once by each sampler.

Registration identity: `{"bytes": 1801, "sha256": "ac18925b95bf88d5ef4e95c7b16ea4b9fd780a579a058fadfed4d3c2a995e3b7"}`. Accuracy not inspected when IDs selected.

[CPU/Open] Unrestricted root `pytest -q` additionally collects unsupported vendor tests and fails collection in Pointnet2_PyTorch/tests (hydra absent), extensions/emd/test_emd_loss.py and third_party/PyTorchEMD/test_emd_loss.py (emd_cuda absent), and third_party/ChamferDistancePytorch/unit_test.py (Python3.12 importlib.find_loader removed). Raw log: `result/preparation/terminal_logit_pilot_20261110/root_pytest.log`. Project `pytest tests -q` passes; no Linux/CUDA model environment was installed on Mac.

[Run/Open] 2026-11-10T17:50:13.995571+00:00 — New owned session `tta-terminal-logit-pilot-a100-20261110-r1`, endpoint `gpu-a100-hm-ass1c0-vcmn4mvmkhfd`, A100/High-RAM allocated. Sole CLI owner is this task; first setup CLI exec session6579, no peer assignments observed. Source commit `c76d6b967ea63a6efd2988ec674fbb196756cfad`; source overlay `51a40ab644bfddce7ab63fb2d70389e8ceba6382ab89283bb092a1ccc6b12ac3` (791710bytes/263files), native source bytes to be bound to actual fresh build. Local preparation `result/preparation/terminal_logit_pilot_20261110/`; planned Drive task root `/content/drive/MyDrive/thesis/result/terminal_logit_pilot_20261110_r1/`. Cell1 in progress; all science unrun.

[Run/Open] 2026-11-10T17:59:25.902040+00:00 — Canonical cell1 exact marker and planned Conda restart/reconnect pass; actual A100-SXM4-80GB/81920MiB. Cell2 completed Conda transaction but only known deferred fastrlock pip-wheel failure; no cell2 success claimed. Independent exact `TERMINAL_CELL2_PYTHON38_VERIFIED` marker passes. Awaiting user-run `colab drivemount -s tta-terminal-logit-pilot-a100-20261110-r1 /content/drive`; do not send cell3 before consent. Cell3–10/deployment/smoke/pilot remain unrun. Session live/idle awaiting consent, not closed. All latest project CPU checks888passed/2skipped/564subtests,13 new targeted tests; compact preparation receipt preserved.

### Completed CPU publication scope; science pending

[Code/CPU/Decision] Publish the necessary opt-in sampler/runner/reducer,13tests,
frozen ID registration, compact preparation receipt and current knowledge only.
Source branch `experiment/terminal-logit-pilot-20261110`, code source commit
`c76d6b967ea63a6efd2988ec674fbb196756cfad`, shared base/latest fetched target
`bb1526d461081a0cae5401dbfee3c5745624cb1a`. The isolated worktree is based on that
latest target; no shared conflicts or peer changes to reconcile. Verification:
888passed/2skipped/564subtests in51.66s;13new pilot tests, Python3.8 parse,
source overlay CRC/hash and independent review. Root vendor collection failures
are preserved above. No algorithm default promotion or GPU validation follows.

Excluded: all `result/` raw logs/sourceZIPs, setup/deployment/science wrappers,
large assets and weights. These are preserved in the owned preparation folder;
Drive backup is pending actual consent. Current deploy captures fresh canonical
native source/binary identities (including exact installed `.so` import origins),
refreshes actual runtime metadata and restores no historical binaries. This
operator has not run and remains gated on full canonicalsetup. Publication of
this completed CPU batch does not mark the requested scientific pilot complete.

Next: user Terminal `colab drivemount -s tta-terminal-logit-pilot-a100-20261110-r1 /content/drive`;
then verify actual mount/shortcut/assets, run canonicalcell3–10 in order, deploy
current source/fresh native receipts, archive and independently validate smoke,
update duration estimate and run only the fixed pilot. User already authorizes
GPU phase; no additional research-run approval needed. On connection/runtime
loss, WAIT_USER without a new cell/retry/replacement. Runtime is live/idle
awaiting consent and must not be closed.

[Code/Publication/Open] Completed CPU scope normal-pushed and remote-ref verified at `2f71f0845e328d3ed381f4d83e57cf652fedc3ee` (includes code source c76d6b9). [Compact receipt](evidence/terminal_logit_pilot_20261110/preparation_receipt.json). This records the CPU integration, not scientific pilot completion. Drive consent remains pending; no further remote cell has been submitted.

## 2026-11-10 — connection loss after user Drive authorization

[User report/Run/Failure/Open] 2026-11-10T18:14:44.400734+00:00 — User reports completing Drive
consent. A read-only server inventory confirmed owned endpoint
`gpu-a100-hm-ass1c0-vcmn4mvmkhfd`. The next CLI command attempted
`verify_drive.py`, but failed in `colab_cli/commands/execution.py:217`, the
CLI's initial working-directory prelude, with **RuntimeError: Connection was
lost**. The verification file was not submitted and no Drive gate marker
appeared. Do not attribute this to invalid consent, Drive files, FUSE, an
installer, quota or VM termination; the cause is unknown.

Status **WAIT_USER_CONNECTION_LOSS**. Per user's no-automatic-recovery boundary,
no new remote cell, retry, replacement, restart or teardown was attempted.
Post-failure read-only inventory still lists the SAME A100High-RAM assignment;
that does not prove kernel responsiveness or idle state. The primary local
exec session92895 emitted the traceback but had not returned a final exit when
archived; preserve its identity and do not launch a duplicate owner. No science
subprocess was created. Cell3–10, deployment, smoke and pilot are **NOT_RUN**.
Last confirmed gates remain canonicalcell1/reconnect/A10080 and independent
cell2Python3.8 (only the known deferred fastrlock wheel failure).

Local immutable raw archive:
`result/preparation/terminal_logit_pilot_20261110/connection_loss_evidence_20261110T1813Z.zip`,
1,659,233bytes/SHA256
`10ff7f87de3b39ef262835e1d377052301dad4ac3e0c5d5e76a8711d26b0d963`;
42files, ZIPCRC/safe paths/unique members pass. Contains actual failure log,
exact submitted verification script, canonicalcells/wrappers/source hashes,
CPU logs, sealed source overlay and prepared/unexecuted operator. No weights
or credentials. [Compact receipt](evidence/terminal_logit_pilot_20261110/connection_loss_receipt.json).
Drive backup remains pending because access is unverified; no automatic retry.

Next gated action: **wait for an explicit user command** before same-session
reconnection/inspection/recovery. Preserve the same assignment and all local
artifacts. No new GPU duration approval is needed for unchanged pilot scope,
but that standing authorization does not override the explicit loss/retry stop.
No new accuracy, per-corruption paired result, timing or memory measurement.
This failed connection is not pilot evidence. Source c76d6b9/frozen32+4IDs,
completed888CPUtests/2skip/564subtests and default baseline remain unchanged.

[User decision/Run/Open] 2026-11-10T18:17:03.441203+00:00 — User explicitly requests reconnect attempt. Stopped only identity-verified stale local CLI31297/parent31287; no kernel/runtime signal. New primary CLI with supported per-process REQUEST_TIMEOUT60 emitted exact TERMINAL_RECONNECT_AUDIT_COMPLETE: same kernelPID896, project exists, Drive mounted, A100-SXM4-80GB/81920MiB, no native setup markers yet. Same endpoint vcmn4mvmkhfd retained; no new allocation/restart. Prior loss remains preserved history; user command reopens same-session work. Actual Drive file gate running before cell3; research remains NOT_RUN.

[Run/Open] 2026-11-10T18:21:24.478254+00:00 — Same-session user-authorized recovery: exact reconnect and Drive file-access gates pass. Kernel896 unchanged; KNN5717B, ModelNetZIP1970686633B, PointMAE351656113B, LION1508051672B readable through MyDrive/thesis, local230GBfree. Canonicalcell3 exact TERMINAL_SETUP_CELL_3_COMPLETE passes; wheel copied. Initial setup evidence ZIP40640B/SHAabf9ae8e1943d1af72dd35c866d1833d9cd670abef40d6401965ddc80dff13cf uploaded; verified Drive backup in progress before cell4. No science yet. Temporary science-wrapper review found no blocker; fresh paths/GPU remain actual-run gates.

[Run/Open] 2026-11-10T18:33:10.197308+00:00 — Cell4 exact original source runs through primary CLI68047. Torch2.1.2+cu121/vision/audio and KNN-CUDA installs completed; requirements stage silent. One bounded read-only independent TTY process audit confirms pipPID9298 under conda9283/kernel896 (elapsed538s/CPU50s); no process interrupted or second notebook cell submitted. Separate audit shell exited normally. Same owned endpoint persists; peer CloudFixer assignment observed only in server inventory and untouched. Setup-before-native archive Drive copy hash verified. No native completion marker or science yet.

[Run/Code/Open] 2026-11-10T18:39:17.662106+00:00 — Canonicalcell4 exact TERMINAL_SETUP_CELL_4_COMPLETE and downloaded prepare.complete pass. nvcc12.1.105/GCC11.4 and A100arch8.0 validated; PointNet++/EMD source-patch hashes emitted. Requirements+CLIP completed with only known pip dependency-conflict warnings, no failed shell command. Original cells/wrappers preserved; new checkpointed wrappers5–8 execute original bytes and require phase markers before writing fresh Drive marker/build-source receipts. Source/modified-wrapper SHA manifest saved locally. Cell5EMD now running; no research yet.

[Run/Open] 2026-11-10T18:42:27.272808+00:00 — Cell5EMD completed18:41:20UTC: exact original/checkpointed wrapper markers and Drive after_cell5 receipt verified. emd.complete retrieved locally before cell6. Fresh binary installed into actual emd_ext egg; deployment captures setup-report exact import origins rather than assuming flat site-packages. Cell6Chamfer running; actual GPU/native smoke still pending.

[Run/Open] 2026-11-10T18:44:33.164455+00:00 — Cell6Chamfer completed18:43:39UTC with exact original/checkpointed markers, on-disk marker and verified Drive after_cell6 receipt; marker downloaded locally. Cell7PointNet++ now running. No skipped/rebuilt completed phase, no old binary restoration or research inference.

[Run/Open] 2026-11-10T18:49:32.814147+00:00 — Cell7PointNet++ completed18:47:54UTC; exact markers/Drive checkpoint pass, local pointnet2.complete retrieved. Actual installed package is pointnet2_ops3.0.0 egg as anticipated. Cell8PVCNN/nativeGPU smoke now running. Source/weights/data gates and research smoke remain required.

[Run/Open] 2026-11-10T18:57:07.692516+00:00 — Cell8 completed18:55:57UTC, exact original/checkpointed markers plus pvcnn-smoke.complete/Drive after_cell8 pass. Native GPU checks report Torch2.1.2+cu121/CUDA12.1/A10080/FPS+gather forward/backward. All five native phase markers locally retrieved. Cell9ModelNet archive preparation running; final setup/native/data/weights gate, deployment and research still pending.

[Run/Open] 2026-11-10T18:58:28.475582+00:00 — Cell9 exact marker passes after Drive ModelNetZIP copy/CRC/path-safe extraction; clean and corrupted files prepared. Cell10weights/nativeGPU/dataset verification running. New source operatorv2 sealed790893bytes/SHAcaa1c2a8f7f9f2925252c7ecf8a17618cae06e228d93655d828a73d1a9f0b82e; upload/deploy not yet executed. Core scientific source remains c76d6b9 and frozen registration unchanged.

[Run/CPU/Open] 2026-11-10T19:01:19.580159+00:00 — All canonicalten setup phases accepted (cell2 known fastrlock exception only). Cell10 SETUP_ENVIRONMENT_VERIFIED + exact wrapper marker, actual statuspassed/Torch2.1.2+cu121/NumPy1.21.2/A10080/175166460kBhostRAM/15arrays/clean/weights. Remote preparation checkout dev@a458cd46bba1288a3f41302548bc777700cbc7d4 plus logged native patches is distinct from scientific sourcec76d6b9. Local setup_verification.json downloaded and independently checked; canonical setup evidence105811bytes/SHAb7622715ff93bc6cb2acd83bcd184ffafaad3e621425aebdd7a022c39df837de. Operatorv2 and setuparchive uploaded; exact Drive backup/unpack wrapper running. Science deployment/smoke/pilot still pending.

[Run/CPU/Archive/Open] 2026-11-10T19:10:47.615887+00:00 — Canonicalsetup and operator Drive hashes pass. Current-source deployment exact marker + strict CPU no-CUDA/no-forward PointMAE load pass; actual fresh7native libraries captured, no historical restore. Pre-science supplement independently downloaded from Drive,5720910bytes/SHAc94a11bea7931ee660d8fb5b945273485aaa23dbbe4602d9745e5e9993ad2a36/109members/all member hashes/CRC/nested-source checks pass. Asset manifest8bc589db3dd6eb5cc83522cd716d7b59ff3ef5101d567d8708edb280821a09d5; actual source snapshotb5ff4e7284274aab24be890a932a0df7261b7d04f50ec126986f5c78f32d7d43/c76d6b9. Fresh setup receipt e3a0523bb89f19ba11c2c9e946a31a400ac4f300e843a6c22f2be51d1946ed92. Technical4-ID/B2/four-corruption/five-arm/three-same-state-repeat smoke submitted via primary CLI, no pilot accuracy yet. Scientific phase remains bounded15–45min planning estimate/45min soft pilot budget; user explicitly waives another GPU question.

## 2026-11-10 — technical smoke accepted with failed native repeat floor

[Run/CPU/Inference] 2026-11-10T19:15:41.329333+00:00 — Raw smoke
`20261110-191050_smoke_seed0` complete:4disjoint IDs/Gaussian/Background/Density/
LiDAR/B2/fivearms/three same-state+RNG repeats,120case files. Source/registration,
finite B×1/4×40 class scores, actual5/8prior/scheduler and1/4decoder/head counts,
cheap zero backward steps/snapshot counts, all frozen model hashes and seven-file
artifact/CRC/exact evidence membership checks pass independent local admission.
Raw Drive→Mac1763623bytes/SHA
`ad6b46a65237a0ba9faee5fd486a84fd57679566735f761ab7362d0f4b0c1b6e`.
Local `result/modelnet40_c/terminal_logit_pilot/raw/20261110-191050_smoke_seed0.zip`
and extracted sibling; Drive ownedtaskroot/raw/samefilename.
[Compact acceptance](evidence/terminal_logit_pilot_20261110/smoke_acceptance.json).

**Native repeat floor fails55/80:**25/80 within fixed maxabs-logit1e-5,
maximum1.1585958003997803, zero prediction changes on these4IDs. No tolerance
relaxation or repeat-parity/isolated causal claim. The handoff explicitly allows
only the descriptive fresh-repeat pilot under this failure. Its LM–PM reductions
will still share each exact saved four-logit tensor. These smoke outcomes are
technical controls, not pilot accuracy/speed evidence.

Updated planning estimate for32/all15/B32/three operational repeats: roughly
15–35minutes, uncertain; hard evidence is B2 cached-state smoke only and earlier
accepted B32/B128 lightweight-host timing, so no new B32 latency is claimed.
The current five-arm policy includes fewer readouts than that old timing host;
retain the registered45minute elapsed softcap and reassess actual B32 progress.
No extra profiler arm, tuning, full data, new seed or automatic rescue/retry.
CPU loading/warmup/checkpoint/archive costs remain outside measured inference
but count toward the conservative elapsed phase budget. Repeat0 primary,1/2
separate. User GPU authorization remains in force; no new approval question.

[Run/CPU/Review/Open] 2026-11-10T19:17:04.049272+00:00 — Independent reviewer confirms qualified descriptive-only pilot admission: no source/model/finite/count blocker, native floor25/80 remains failed. All120raw labels match freshly downloaded SHA-pinned original label.npy; actual GPU classifier state matches strict CPU load. Acceptance uploaded to same ROOT/smoke_accepted.json. Fixed32/all15/seed0/B32/3fresh-repeat/five-arm pilot submitted via single primary CLI; no tuning, extra seed, extra diffusionLM arm or full-data expansion. Initial actual B32 timing and all accuracy still pending; estimate15–35min/45min elapsed softcap.

[Run/Inference/Open] 2026-11-10T19:19:10.724606+00:00 — Pilot actual firstB32 Uniform measured calls: T5≈2.803s,P5≈1.905s,T8≈4.060s,P8≈2.627s,P8-PM≈3.728s. These are initial per-arm batch measurements, not accepted final speed results. Two B32 warmups/arm pass; no OOM/finite error. Updated projected fullpilot elapsed≈20–25min including225measured+150warmup calls and fixed overhead, below45min softcap. Same seed0/32IDs/three operational repeats, no outcome-based scope change; all15 pending complete raw validation.


[Run/Inference/Open] 2026-11-10T19:29:48UTC — Pilot nine corruptions complete,
Shear running through original primary CLI54894. No connection/OOM/finite/count
failure; measured B32 arm times remain near T5 2.8s/P5 1.9s/T8 4.1s/P8 2.6s/
P8-PM 3.7s. These are progress observations, not accepted final results.
Two warmups per arm/corruption and three fresh operational repeats unchanged;
all15 completion/Drive raw archive/local admission still pending. Independent
review adds strict original-label SHA and actual executed-source-map vs ZIP
member assertions to local acceptance tool before seeing pilot accuracy;
174 source identities already checked on smoke. Core/GPU source unchanged.
Latest origin fetch has no new integration commits to reconcile.


## 2026-11-10 — complete pilot admitted independently

[Run/CPU/Inference] T5329/480=68.5417%; P5324/480=67.5000%; T8334/480=69.5833%; P8325/480=67.7083%; P8-PM344/480=71.6667%; P8-LM345/480=71.8750%.
T5→P5 -1.0417pp,95%CI[-3.5417,+1.2500],32.1167%less time; T8→P8 -1.8750pp,95%CI[-4.3750,+0.6250],35.3848%less time; P8-PM→P8-LM +0.2083pp,95%CI[+0.0000,+0.6250]; P8→P8-PM +3.9583pp,95%CI[+2.0833,+6.0417],-42.0284%less time.
Raw 20261110-191704_pilot_seed0:4777541B/SHA256`8570672072e7bbcb9b62e4c06246f33dbfb14863b20f7f9e7612f5b8d0fc8111`.
Full225cases/7200rawGPU/8640derived predictions, all labels/source/native/model
identities/counts/CRC/member hashes verified. Independent object-cluster
bootstrap and local complete recount agree with remote analysis.
[Full report](terminal_logit_pilot_results_20261110.md),
[acceptance](evidence/terminal_logit_pilot_20261110/pilot_acceptance.json).
Repeat0primary;1/2separate. No accuracy-driven scope change or automatic next
phase. Native floor55/80failed is retained, no causal/noninferiority claim.
Documentation/derived-analysis Drive backup and final normal publication follow;
runtime retained and no teardown authorized.


[CPU/Diagnostic/Correction] Local admission first failed because the newly
added validator expected live call counters in timed pilot metadata. Sealed
runner lines223–236 make hooks/audits smoke-only; pilot intentionally records
audit=null and zero placeholder counters to exclude hook overhead. Root cause
is local validator expectation, not a run failure. Failed validator source
preserved as validate_pilot_failed_counter_expectation.py. Corrected validator
enforces null/zero and byte-validates already-extracted files before reuse;
raw ZIP/extracted source bytes unchanged. No GPU retry, rerun or parameter
change. Actual smoke call/update counts remain accepted; no instrumented pilot
call-count claim. Complete local admission/recount/bootstrap passed afterward.

[Run/Inference/Decision] Actual pilot runner1185.561316736seconds/19.759minutes,
within2700second softcap. Primary proximal terminal speed gains32.1167%/35.3848%
come with point accuracy losses1.0417pp/1.8750pp and intervals below−.3pp;
planned retention criterion not established. LM+1/480 (+.2083pp),CI[0,.625]pp;
repeat1+1 and repeat2+2, everyCI touches0. No reliable LM benefit established.
Retain current proximalN8+PM baseline; no full-data expansion/newseed/tuning.


[CPU/Review] 2026-11-10T19:39:40UTC — Independent reviewer accepts complete
DESCRIPTIVE pilot with no remaining admission/report blocker. Independently
checked archive member hashes/174 executed-source identities/smoke-native-model
bindings/labels/all225cases/outcome counts/timings/primary clustered intervals.
Agrees terminal accuracy retention and reliable LM gain are unestablished;
smoke instrumentation versus uninstrumented pilot timing and native floor
limits correctly disclosed. No reviewer edits or runtime operations.


[CPU/Archive] 2026-11-10T19:41:30UTC — Accepted derived analysis, exact local
validator (including preserved failed expectation), operator sources/logs,
original label.npy, updated knowledge and independent review archived as
`documentation_20261110T193956Z.zip`,639522B/SHA256
`6496ec9d85bfe76fa2b886cf90f184e189d144674b7359d1667431f91833e09e`.
Drive taskroot/samefilename; local preparation/samefilename plus independent
Drive-roundtrip copy.43members/exact manifest/everySHA/ZIPCRC pass. Raw pilot,
smoke, source/deployment and canonicalsetup archives remain separate and
immutable. [Backup receipt](evidence/terminal_logit_pilot_20261110/documentation_archive_receipt.json).
This receipt/final publication status follow the prepublication archive and
are recorded externally; archive scope is explicit. Kernel896 still responds;
no new model inference, retry, replacement, restart or teardown.

[CPU/Publication scope] Latest origin/thesis-integration still f84faef4c8e9bb364d7648bba82860b3a9f45f72;
no peer commits/conflicts. Core c76d6b9 already published/tested888passed/2skip/
564subtests. This batch adds accepted subset results, all corruption/repeat/
paired/time/memory tables, setup/reconnect/smoke/pilot/archive/review receipts
and reconciles README/findings/open questions/plans/method/setup/study records.
Source unchanged, independent full real-output recount/bootstrap and new
JSON/CSV/link/diff checks pass. Excludes result ZIPs/logs/operators/weights,
full benchmark/extra seed, defaults and peer checkouts/runtimes. Publish via
normal fast-forward only; no force. Worktree and live runtime retained.
