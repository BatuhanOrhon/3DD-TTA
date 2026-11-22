# Original CloudFixer: matched alternative source models, 2026-11-22

[Code/Open/User report] Preparation and failed allocation record, **not new
accuracy evidence**. Working checkout is `/Users/batuhan/.codex/worktrees/0e7f/3DD-TTA`,
branch `research/cloudfixer-transfer-20261114`, starting HEAD
`656ac65e62d0a982c4ba44ba2fc4e1bc999db8a1`. Main checkout and peer work are protected.
See [machine phase plan](evidence/cloudfixer_multibackbone_20261122/phase_plan.json).

## Question and checkpoint decision

Measure unchanged author CloudFixer on ModelNet40-C with the **same frozen
DGCNN, CurveNet and PointNeXt** used in the accepted LION proximalN8 PM fulltest.
Do not rerun LION or change its accepted results. Its three means are respectively
63.9411129119%, 65.6780118855%, 70.6266882766%; exact counts and all per-corruption
references are read from `complete_results_20261114.json` in the shared branch.
The prior source is `b651e259db9ec6341f39e606b01eef5254d90456`.

[Decision/Inference] For the later **ScanObjectNN-C cross-dataset transfer**
experiment use the already hash-verified **ModelNet40 XYZ CloudFixer prior**,
SHA `b64841d022e08b1a89f58dd9b16d9023e00be52fbca3dca735d5798362bb5a5a`,
with Scan-specific frozen classifiers. This is a usable transfer checkpoint,
not a claim that it was trained on ScanObjectNN or uniquely optimal.
[The preceding source-domain audit](cloudfixer_scannet_scanobjectnn_lion_provenance_20261122.md)
found no released ScanObjectNN-trained prior. PointDA-10 ScanNet weights are not
ScanObjectNN weights. A ShapeNetCore prior is a separate candidate requiring its
own provenance/overlap audit. **No Scan experiment is launched in this phase.**

## Locked protocol and comparison limits

- 2,468 original IDs × all15 corruptions × severity5, seed0, B32, final4 retained.
  1,170 native reconstruction blocks; each reconstruction serves all3 heads.
- Author clone `f043e24414fcf4165fdc892abff21215c9989141` is archived unchanged
  except the documented commented-out PointNet++ import repair in `utils/misc.py`.
  `adapt.cloudfixer` performs its original30 delta/rotation Adamax updates,
  plainCF/vote1; classifier parameters/BN remain frozen. No PM or accuracy tuning.
- Seed0 is initialized once before each corruption's author dataset construction;
  geometry draws continue across batches. Python/NumPy/CPU/CUDA RNG is committed
  with every block. Classifier RNG is forked/restored around all3 readouts.
  This is not the author's full shuffled-loader random stream or common-draw
  pairing with the older LION runs.
- Input is author `ModelNet40C(rotate=True)`, unitSTD, mask/indices with a2048-point subsample upper bound; actual Uniform/Gaussian inputs have1024points, background1075, upsampling2048. Unlike LION, CF does not upsample every input to2048.
  Output uses author's inverse x-axis rotation and `scale_to_unit_cube_torch`:
  despite its name, this function centers and scales by **max radius**, not bbox.
  Accepted head-specific FPS1024 follows. Source-raw and CF-frame-source are
  separate controls. LION's bbox output transform is preserved in its history.
- The comparison is between **whole pipelines with different priors, output
  preprocessing and native random streams**. It cannot isolate optimizer or
  denoiser quality, and is not paper-head benchmark replication.
- Head files, states, model implementations/configuration and PointNeXt native
  extension are pinned through accepted provenance and base-manifest SHA
  `599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef`.
  Head checkpoints: `svwa_dgcnn_modelnet40.pth`,
  `svwa_curvenet_modelnet40.pth`, `pointnext_modelnet40.pth`.
  Dataset/label identities come from `cfgs/cloudfixer_transfer_assets.json`.

## Resilience, partial funding and evidence

[User report] Initial GPU instruction was intermittent A100 retries; the user
then temporarily chose code preparation only. No request was submitted during
that hold. Latest instruction **resumes A100 attempts and explicitly accepts
continuation across compute-unit exhaustion**. This supersedes the earlier
requirement to fund the full remaining campaign in one session.

1. Every32 original objects, publish compressed raw logits for all3 heads and
   all3 readouts, labels, IDs, timings, geometry hashes and full RNG continuation.
   Payload is read back before publishing its immutable SHA/length receipt.
2. Each completed corruption produces its own raw ZIP plus separate completion
   JSON for **each source model**, containing the3 correct/total readouts and
   pointing to that ZIP's SHA. An immutable corruption cursor is published last.
   Recovery verifies cursor/archive hashes and skips finished corruptions;
   the active corruption resumes only after its accepted contiguous prefix.
3. Save pinned native ELF files to Drive before results, restore them on a
   replacement canonical Python3.8/Torch2.1.2+cu121/A100 environment and run fresh
   native/head/geometry checks. Runtime signature mismatch stops recovery;
   do not silently combine different native builds. Fresh smoke replay must
   reopen two B4 batches each for background/rotation/Gaussian without rescoring.
4. A fresh account receipt and B32 profile admit a **funded execution window**,
   not necessarily the full test. A5CU reserve and all active assignments' rate
   determine its size. The window expires from the account-reading timestamp,
   is single-use and covers runtime checks, recovery, inference and ZIP/Drive
   work. Worker stops before another block when near the window end; a hard
   deadline also bounds stuck native/Drive work. Incomplete final blocks have
   no accepted receipt and can be recomputed after loss.
5. Task Drive root:
   `MyDrive/thesis/result/cloudfixer_multibackbone_20261122`.
   Host raw downloads: `result/modelnet40_c/cloudfixer_multibackbone/`.
   Stream CLI/worker stdout to a separate attempt log; retain it on success,
   partial completion and failure. Corruption ZIPs deliberately contain scoped
   tables and `campaign_complete=false`; they do not pretend to be a full run.
   Capture stdout and the final seven-file inference bundle separately.
6. After each corruption marker, CLI-download to a fresh host destination and
   verify its published SHA. Independent analyzer validates raw receipts,
   IDs/labels/RNG chains, per-corruption and headline CSV counts, then computes
   matched-head/corruption CF-minus-LION pp from pinned historical counts.
   A paired object-win count requires the actual old raw predictions, not just
   scalar counts. Final full-data status requires all15 validated shards.

DriveFS readback can be cached; it is not remote durability proof. Verify
host downloads, retain raw bytes, and flush Drive through the supported mount
flow before user-authorized teardown. A disconnected browser or exhausted CU
can still lose the current uncommitted block; no keep-alive guarantee exists.

## Cost, estimates and allocations

[Run/Inference] Actual old officialCF B32 timing extrapolates to
**2.8878694736 hours** for78 batches/corruption, before extra-head/setup overhead.
[Recalculation evidence](evidence/cloudfixer_multibackbone_20261122/historical_runtime_estimate.json)
retains source CSV SHA and per-corruption variability. Plan **3–4h science +
1–1.5h setup, total4–5.5h**, prospective campaign cap6h/55CU and reserve5CU;
profile and current usage govern each window. The initial30–60min science
estimate and3h/20CU proposal were wrong and are retained as superseded in the
machine plan; no owned VM or inference ran under that proposal.

[Run] A100 requests r1–r10 returned **HTTP503**: r3/r8 High-RAM, others standard.
All names have prefix `tta-cloudfixer-multibackbone-a100-20261122-`.
No owned assignment, remote setup or inference exists. At12:58UTC balance was
37.00CU, usage8.88CU/h, one assignment. The visible protected peer is
`tta-cloudfixer-background-a100-20261122-r7`, endpoint
`gpu-a100-hm-euw4a0-39enm5hdhexf8`; inventory does not authorize using/stopping it.
No accelerator substitution, peer action or account-authentication change.

## Runnable preparation and next gates

Sources: `prepare_cloudfixer_multibackbone.py`,
`setup_cloudfixer_multibackbone_colab.py`, `run_cloudfixer_multibackbone.py`,
`analyze_cloudfixer_multibackbone.py` under `scripts/`.
Package contains accepted assets, exact official tar, source overlay, bootstrap,
reference and SHA/size receipt. Weights/raw results stay out of Git.

- Rebuild into a **fresh** preparation directory after source commit using the
  packager's `--base-assets`, `--official-checkout`, `--output` arguments.
  Accepted local head asset root is the read-only peer artifact
  `/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components/result/preparation/guidance_component_sensitivity_20261025/assets_v2`.
- Allocate a distinct owned A100 via CLI; record endpoint first. Run the exact
 10 canonical notebook code cells, cell1 alone/reconnect, cells2–10 in order,
  user-operated Drive consent, required markers and `/content/3DD-TTA` cwd.
  Exact-source wrappers/hashes are in ignored preparation artifacts; regenerate
  from the canonical notebook if unavailable. Only known cell2 fastrlock failure
  is deferred. No whole-notebook run.
- Upload package members and run its bootstrap with `--package`, `--root`,
  `--drive-folder`; wait for `CF_MULTIBACKBONE_DEPLOYED`. It installs pinned
  PyTorch3D/author Chamfer extras and multimethod1.9.1 after canonical10.
- Use deployed `code/scripts/run_cloudfixer_multibackbone.py` under
  `conda run --no-capture-output -n 3dd_tta_env python`, supplying manifest,
  asset-root, official-root, diffusion, dataset-root, drive-folder and a fresh
  output for each attempt. Run `--phase smoke` twice in independent interpreters,
  then profile; full also needs `--full-gate-key full_gate_<contentSHA>.json`.
  Gate JSON fields are enforced by `validate_full_gate`; use actual profile,
  account timestamp/rate/balance, full remaining ETA and affordable execution
  window. Do not invent an account receipt or treat planning as a measurement.
- Resume only missing work after runtime/native/mount/profile/resource gates;
  inspect unknown remote worker state before launching anything.

[Code/CPU/Open] Focused synthetic tests cover full/tail coverage, corrupt
receipts, RNG-prefix gaps, Drive namespace, renewed/expired partial funding,
raw recount/CSV tampering, immutable large-file copy, native restore path checks
and completed-corruption cursors. GPU/native/head accuracy, real Drive recovery,
actual timing and all new CF accuracy values remain **unmeasured**.
The preparation is falsified by any native/head parity, identity or recovery
failure; stop and retain evidence rather than weakening the checks.

[Code/CPU] Final preparation check:22 focused tests and a synthetic two-shard
CLI analysis validate combination across independent funded windows, disjoint
coverage, summary totals and matched historical deltas. Pass all15 separately
extracted shard directories after `--run`; the collector rejects overlap or
mixed campaigns. A final attempt directory is not assumed to contain previously
completed corruptions. Scope ZIP extraction must first verify published SHA,
CRC and safe paths (existing `analyze_cloudfixer_point2vec.members` helper).
Shared-branch promotion of this unvalidated GPU runner is deferred until real
canonical/native/smoke admission; reviewed preparation remains on the task branch.

[CPU verification receipt](evidence/cloudfixer_multibackbone_20261122/cpu_verification.json):
168 related tests passed (22 focused), syntax and diff checks passed; independent
source review found no remaining CPU-preparation blocker. No GPU admission.

[Code/Artifact] Source commit `d62252d48b79e162a8e7d5e7e3951d43a37e2213`; fresh delivery is
`result/preparation/cloudfixer_multibackbone_20261122/package_v2/`. All receipt
SHA/lengths, ZIP CRCs and all source/head member identities independently checked.
[Compact package receipt](evidence/cloudfixer_multibackbone_20261122/package_v2_receipt.json).
Old `package_v1` is superseded and must not be deployed. No remote upload occurred.
Latest server inventory still shows only the protected background peer; this
task has no live runtime. Current blocker is repeated Colab allocation503,
not a new GPU-duration approval requirement. Resume with ownedA100 allocation
when available; then canonical setup/user consent and native gates above.

## 2026-11-22 13:15UTC — renewed standard-RAM A100 attempts

[User report/Run/Open] User requested allocation again and explicitly accepted
standard/lower host RAM. CLI requests r11 and r12 used `--gpu A100` without
`--high-mem`; both returned HTTP503/Service Unavailable. No owned VM/setup or
inference. At13:13UTC account read34.77CU/8.88CUh/one protected peer assignment;
503 does not establish a cause such as RAM, balance or quota. Raw local outputs
are preserved under `result/preparation/cloudfixer_multibackbone_20261122/`
`allocation_r11/stdout.log` and `allocation_r12/stdout.log`. Source/code package
unchanged from d62252d; current task worktree was clean at b021920 before these
metadata updates. No peer runtime action, authentication change or GPU fallback.
Next: retry a distinct **standard-RAM A100** assignment when service permits,
then the previously recorded canonical setup/consent/native/partial funding
gates; GPU-run authorization remains valid, no new duration approval needed.

## 2026-11-22 13:31UTC — L4 allocated; interactive Drive consent pending

[User report/Run/Code/Open] User explicitly requested L4. Owned session
`tta-cloudfixer-multibackbone-l4-20261122-r1`, endpoint
`gpu-l4-s-ass1b0-arzn2htwd6mw`, standard RAM. Native `nvidia-smi` confirms
**NVIDIA L4 /23034MiB / compute capability8.9**. Exact canonical cell1 completed,
Conda restarted its kernel, sameVM reconnect/condacolab check passed. Initial
CLI process2320 completed; Drive mount CLI61730 currently waits for the user's
Google consent. Link provided through interactive question; credentials/link
state are deliberately excluded from Git and artifact logs. Do not press Enter
until the user confirms; do not launch a second kernel cell during mount.
No inference, native completion or source-head parity result yet.

[Run] Allocation account snapshot33.30CU, total10.82CU/h, protected background
peer8.88CU/h; marginal observed L4 rate1.94CU/h. L4 performance and full ETA
remain unknown. Initial setup/native/smoke/profile window at most2h, ending
15:26UTC absent reassessment; global6h/55CU and account reserve5CU remain.
Waiting/failed setup counts toward owned runtime cost. Later full evaluation
uses only affordable windows after profile/current-rate checks.

[Code/CPU/Open] Packager now explicitly registers accelerator A100(default) or
L4; guards require exact model/capability pair8.0 or8.9. Additional CF natives
build for the registered actual capability. L4 PointNeXt native compiles the
pinned source into a **separate** directory and is explicitly loaded before
OpenPoints; original accepted A100 asset bytes remain pinned/preserved. New
L4 campaign/Drive subdirectory `.../cloudfixer_multibackbone_20261122/l4_trial`
prevents accidental signature/prediction mixing with A100. Weights, frozen state,
input/output transforms, CF30/seed0/B32 and all readouts remain as registered.
No cross-GPU numerical equality or speed claim follows from CPU checks.

Canonical notebook file remains unchanged. Executed wrappers preserve its first
nine source cells exactly; cell10's embedded verifier has **only** its GPU-name
assertion A100→L4 adapted to the user's hardware request. All native/CUDA/data/
weights checks remain. Original/executed/wrapper SHA identities are in
[canonical L4 wrapper record](evidence/cloudfixer_multibackbone_20261122/canonical_wrappers_l4.json).
171 related CPU tests (25focused) and syntax/diff checks passed; real L4 native
compilation, B32 memory/time profile and resumed Drive correctness are open.
[Owned phase/monitor record](evidence/cloudfixer_multibackbone_20261122/l4_trial_plan.json).

[Code/CPU] 13:32UTC L4 cache restore now validates the exact Python3.8/Linux
module basename, positive length and SHA **before** forming destination paths.
Four malformed-name/digest regressions observed failing then passed. Final
175 related tests (29focused), syntax and diff checks pass.
[L4 CPU receipt](evidence/cloudfixer_multibackbone_20261122/cpu_verification_l4.json).

[Code/Artifact] L4 source237d7b2 packaged at
`result/preparation/cloudfixer_multibackbone_20261122/l4_trial/package_v3/`;
all receipt/member identities and ZIP CRCs independently pass.
[L4 delivery receipt](evidence/cloudfixer_multibackbone_20261122/package_v3_l4_receipt.json).
Do not deploy old A100-only v2 on this L4. No package upload/inference yet;
active user-consent mount process61730 is preserved, owned L4 remains live.

## 2026-11-22 13:39UTC — L4 Drive mounted, canonical cell2 running

[Run/User report/Open] First consent CLI61730 returned `mount failed` after
the user confirmed; cause unproven. SameVM retry88591 obtained a new consent,
user confirmed, CLI reported credentials propagated and `Mounted at /content/drive`.
No authentication configuration/account change. Independent remote check passed
physical `os.path.ismount`, fresh task JSON write/fsync/readback SHA
`f7fcf7f3068a0c8ce44f8440383839d7ec0c740217fa5fa7c82d99c6d91e723e`;
ModelNet ZIP cache exists, local free disk235.03GiB. Mount process88591 complete.
Canonical cell2 now running through primary CLI65127; source exact from saved
notebook wrapper. Raw output preserved locally under `l4_trial/cell_02.stdout.log`.
Do not launch another cell until its actual marker or the specifically allowed
fastrlock failure and separate Python3.8 verification. No inference yet.

[Run] 13:49UTC canonicalcell2 exited after the explicitly deferred fastrlock
wheel build failure. Local full-log scan found exactly that failed wheel,
`Successfully built calmsize clip`, and nocell2 success marker. Separate
Python3.8.20 conda-env probe passed `CF_CELL2_DEFERRED_ENV_VERIFIED`; receipt
written remotely. No other failure waived. Canonicalcell3 CLI41758 now running;
primary log retained, no concurrent kernel cell. Current server inventory
confirms both ownedL4 and protectedpeer assignments, not kernel completion.

[Run] 14:13UTC canonical4 exactmarker/prepare.complete passed. Independent
phase04 audit verifies Python3.8.20/Torch2.1.2+cu121, actual L4/SM8.9 CUDA tensor
sum496, 12CPUs/MemTotal55530568kB (MemAvailable52735744kB). All package parts
coalesced in fresh `/content/cf_mb_package_l4_v3`, original SHA/member CRCs pass;
Drive bootstrap ZIP exactSHA readback passes. Canonical pip retains documented
dask/importlib-metadata and scikit-image/networkx warnings; no global pipcheck
claim. Native builds use explicit MAX_JOBS4 within observed host resources.

[Run/Open] First canonical5 CLI21946 failed on HTTP kernel metadata GET with
10s timeout **before execute(code)**. Failed log/command receipt uploaded to
Drive. Server kept ownL4/peer. Independent remote process/marker probe confirms
no active compiler/research process, prepare=true/emd=false and emits
CF_L4_IDLE_AFTER_PRELUDE_PASS. Only after this evidence, a fresh retry01 driver
reuses the exact source cells with a process-local HTTP REQUEST_TIMEOUT60
wrapper around the installed CLI (no package/auth-state edit). Retry outputs
have fresh filenames; no failed directory silently reset. No inference.

[Run/Open] 14:19UTC nativeEMD/step5 passed in88.7s with exactmarker and
physical completed marker; fresh source/log/command receipt uploaded toDrive.
Step6 connection lost specifically in `execution.py:217` CWD prelude before
the selected setup body. Hung LOCALCLI PID73838/parent73600 was verified as
owned and terminated; no VM/kernel/build process stopped. Subsequent remote
probe reports no active build/research, prepare=true/emd=true/chamfer=false,
CF_L4_IDLE_BEFORE_NATIVE06_PASS. Only then retry02 starts6–10, fresh outputs.
Process-local CLI wrapper keeps HTTPmetadata and unspecified CWDprelude
timeouts at60s; explicit phase2700s unchanged. Native source/math and installed
CLI/auth configuration untouched. No new inference or lost accepted data.

[Run] 14:31UTC selected canonical native phases5–8 completed, exact wrapper
markers + physical completion markers asserted; receipt wall durations: EMD88.7s,
Chamfer111.5s, PointNet++122.5s, PVCNN/native-smoke194.1s. Real L4 FPS/gather
forward/backward passed. Each source/command/stdout artifact retained locally;
completed phase log+receipt uploaded to fresh Drive names. Canonical9 now
prepares ModelNet40-C;10 and separate officialCF/head native gates still pending.
Transport retry recovery completed without duplicate native build or peer action.

[Run/Code/CPU] 14:43UTC first supplemental CFbootstrap failed **before writes**
on Drive namespace comparison. Actual `MyDrive/thesis` is a Google Drive
shortcut into `.shortcut-targets-by-id/.../thesis`; resolving only the child
and comparing with the literal taskroot falsely rejects valid storage.
Readonly diagnostics confirm physical mount and no CFnative root. Intermittent
CLI CWD-prelude connection loss affected diagnostics too; only ownedLOCAL
failed client retired; readonly retry succeeded, no kernel/peer changes.

Shared `require_task_drive` now resolves both approved root and target, verifies
approved root remains beneath the actual Drive mount, rejects escaping child
symlinks/prefix siblings and returns the resolved target. Setup and worker use
the same function. Two regressions observed failing then fixed;177 related CPU
checks pass and independent source review has no namespace blocker. Source/
overlay identities change, so v3 remains preserved as failed-prewrite; freshv4
is required. No scientific predictions exist; no frozen checkpoint/protocol
changed. Verified v3 asset container can be reused only after full member/CRC
identity checks and binding its actual SHA in the v4 receipt.

[Run/Code/CPU] 15:11UTC supplemental L4 kernels pass: actual official Chamfer
CUDA backward/KNN finite, PointNeXt SM8.9 native compile/config import.
Finalization initially failed because caching `setup_...py` polluted budget
`setup_` JSON namespace. Repo producer fix uses `package_` for all package
cache objects; integrated Budget regression observed failure then178tests pass.
Registered on-VM **v4 science bytes remain source663d756/campaign68d538...**.
Reviewed operational `recover_setup_cache02.py` preserves/verifies original
source artifact, migrates only its cache name and temporarily redirects that
exact producer key during original-v4 finalization. No worker/model/native
file changes. It replays native probe, records original setup + append-only
nonoverlapping repair intervals, and archives its own source SHA. Immediate
second invocation validates cached completion without rebuilding/recharging.
Both recovery/repeat-safe markers pass. Old failure, old source and all
durations retained; no scientific cache reset.

[Run/Open] First model smoke `20261122T151211Z_smoke01` completes6units:
background/rotation/Gaussian each2×4objects. All strict head checkpoint/state,
wrapped/direct outputs and RNG, input-axis/unit-sphere frame and frozen model
checks pass. Model/readout logits are finite;24object reconstructions are
technical coverage, **not full-benchmark accuracy evidence**. New process
replay now verifies all6cached units and original RNG chain without rescoring.

[Run] 15:22UTC all technical gates pass. Newprocess smoke02 reopens all6units
without rescoring; smoke_gate published. B32Gaussian native20.94456s, whole
head-forward sum1.75328s, peak2647.57MiB. Largest corruption upsampling is
2048points (Gaussian1024/background1075); B32 native58.67185s, peak9183.82MiB
and all3 frozen heads finite/state-equal. No batch/protocol change needed.
All technical labels remain readiness data, no full-test accuracy claim.

[Inference/Run/Open] Measured L4 max ratio2.79147 of oldA100 Gaussian/upsampling
profile + measuredheadcost +15percentmargin estimates9.96h science. This
supersedes A100-based4–5.5h whole-phase ETA **for L4**. Six-hour globalcap and
55CU bound remain for current admitted work; extension before reaching it
requires a recorded operational amendment, never a new duration permission.
No change to science/manifest or silently reset budget. Current scope executes
only a funded segment; full remaining campaign cannot be promised with current
account balance. User already authorizes interrupted-credit continuation.

[Run/Planned] 15:24UTC snapshot11.81CU/10.82CUh/twoassignments; ownedL4 marginal
1.94CUh, protectedpeer8.88CUh. Gate a5f781... binds current profileSHA940c3e...,
campaign68d538...; reserve5CU,2085s from account-reading timestamp,180s cost
margin, expires15:58:49UTC. First full1170unit phase now launched; scope
2468×all15/s5/seed0/B32 unchanged. Per32object payload/RNG and perhead/corruption
shards/cursors plus live Drive stdout apply; host copy/recount follows each
completed corruption. This launch is not completion.


[Run/Code/Open] 2026-11-22T15:47UTC — actual full02 started15:42:30UTC;
worker74350 / independent supervisor74349. Downloaded Drive stdout confirms
first7uniform blocks (224objects), all3 frozen heads, of1170fullcampaign blocks.
Full01 synchronous attempt failed in installedCLI CWD prelude before scientific
body; no full cache/attempt/usedgate then. Local process-only absolute-path
CLI adaptation avoids that redundant RPC; detached worker is independent of WS.
Supervisor always waits for actual worker exit (including SIGKILL) before final
status; user-approved per32object Drive receipts/RNG remain recovery authority.

Account read15:35:51UTC10.00CU/allassignments10.82CUh/2assignments; currentgate
c2a3b356... binds measuredprofile and alloperator hashes. Operational reserve
amended5→1CU because user explicitly accepts CU-exhaustion continuation and
actual atomic/replay gates passed;180s margin retained, funded deadline16:22:45UTC.
Existing6h/55CU caps remain. Source663d756 numeric/model/manifest bytes unchanged;
resourcepolicy/runner/supervisor/launcher sources separately archived. No prior
failed output overwritten; no peer runtime modified.178CPU checks pass.
[Compact receipt](evidence/cloudfixer_multibackbone_20261122/l4_full02_launch_receipt.json).
No complete-corruption or full-benchmark accuracy is yet claimed. Next: retrieve
and independently recount each completed corruption, preserve partial prefix at
funded stop, refresh truthful resource snapshot before any continuation window.


[Run] 2026-11-22T15:52UTC first additional recovery capture:
21uniform blocks/672objects copied read-only while detached GPU worker continues.
DriveZIP1390049B/SHA13501433a8aee539e19944ea0cf807e2195261541d7e2e913f72b8984f6ad67b
independently CLI-downloaded toMac. SafeZIP paths/CRC, exactmanifest/data/label,
all rawpayloadSHA/finite shapes/originalIDs, contiguousprefix and interblock RNG
chain pass existing CPU validator. Raw extraction and validation report under
`result/modelnet40_c/cloudfixer_multibackbone/68d53814491e/partial_capture_20261122T155130Z/`.
[Compact proof](evidence/cloudfixer_multibackbone_20261122/l4_partial_prefix_155130_receipt.json).
This is partial preserved coverage, not a complete-corruption accuracy result.
The independently downloaded operational source archive also passed exactbyte/CRC
identity (11711B/SHA54125a7f...0d73f). Source/knowledge handoff committed128c453;
scientific v4 source remains663d756. Read-only final review finds no active-window
blocker; residual supervisor evidence risk on Drive failure before local exit
receipt is recorded for nextwindow amendment, not patched inside active process.


## 2026-11-22 16:15UTC — İlk tam corruption: Uniform (diğer14 henüz tamam değil)

[Run/Code/Inference/Open] Owned L4r1/full02 çalışmasında Uniform severity5,
özgünID0..2467, seed0/B32, tüm3donmuşbaşlık tamamlandı.78blokluk ham ZIP,
Drive completioncursor/2468 kapsamı, byteboyutu/SHA, güvenliZIP/CRC, bütün
payload/hash/etiket/ID/RNGzinciri ve hem per_corruption hem summaryCSV bağımsız
CPU yeniden sayımı geçti. Gaussian GPU'da devam ediyor. All15 makro sonucu yok.

Accuracy değerleri yüzde; farklar yüzde puanıdır. Eski referans **LION
proximalN8PM**'dir, CloudFixer-prior PM değildir. Aynı kabul edilmiş donmuş
başlıklar kullanılır; önceki sourceb651e259... referansı ve arşivSHA'ları
manifestte kilitlidir. Değişen prior/ön işleme/RNG nedeniyle bütün-pipeline
kıyasıdır; optimizer veya PM etkisini tek başına kanıtlamaz.

| Model | Source raw | Source CF frame | PlainCF doğru/2468 | PlainCF | Eski LION PM | CF−LION(pp) |
|---|---:|---:|---:|---:|---:|---:|
| dgcnn | 68.192869 | 79.376013 | 2179/2468 | 88.290113 | 78.484603 | +9.805511 |
| curvenet | 67.301459 | 82.941653 | 2173/2468 | 88.047002 | 76.701783 | +11.345219 |
| pointnext | 45.259319 | 69.286872 | 2174/2468 | 88.087520 | 80.632091 | +7.455429 |

[Inference] Uniform için üç pipeline kıyası da CF lehine; net doğru sayısı
farkı DGCNN+242/CurveNet+280/PointNeXt+184. Bunlar **eşleşmiş nesne bazında
CF-only/LION-only kazanım sayıları değildir**: eski scalarreferans yalnız net
sayı verir. SourceCFframe kontrolünün raw'a göre büyük farkı, ön işlemenin
önemini gösterir; CF'nin frame'e ek farkları da pozitif. Bu tekcorruption/
tekseed sonuçları all15 üstünlüğü, seedkararlılığı, rotation/background sonucu
veya bağımsız PM üstünlüğü olarak genellenemez. Testetiketleriyle tuning yok.

HamDriveZIP `corruption_20261122T154232Z_full_window02_uniform.zip`,
SHA065cf7fb7530858164c9d00d67db624a8470e7e345ec1359cbc59fca0d79a9a3.
Hostcollector/rawscope/analysis:
`result/modelnet40_c/cloudfixer_multibackbone/68d53814491e/retrieval_20261122T154804Z/`.
Kanonik doğrulanmış yerel bundle:
`result/modelnet40_c/cloudfixer_multibackbone/68d53814491e/uniform_validated_20261122T1615Z/`.
HamZIP korunmuştur; orijinal6standartdosyaya ayrıDrive stdoutsidecar'ı
belgelenerek `stdout.log` eklendi, scientific dosyalar değiştirilmedi.
Yeni bundleZIP4648978B/SHAbda64deb...5b8d16 Drive'a yüklendi; bağımsız
readbackdurumu [bundle receipt](evidence/cloudfixer_multibackbone_20261122/uniform_validated_bundle_receipt.json).
[Sayısal kanıt](evidence/cloudfixer_multibackbone_20261122/uniform_full2468_validated.json).
Sonraki kapı: fonlanan pencere sonuna kadar Gaussian blokları, sonra soncache
snapshot/yerelprefixdoğrulama ve güncel kaynaklara göre devam kararı.


## 2026-11-22 16:23UTC — Kapasite engeli; son3076 örnek güvenli, task tamam değil

[Run/Open] GPUworker16:21:33UTC `partial_window_complete`, gerçekexit0;
ayrı supervisor aynı çıkışı doğruladı,1completedcorruption/15. Bu bir fonlanan
bölümün tamamlanmasıdır, tüm deneylerin tamamlanması değildir.97/1170blok:
Uniform2468/2468 + Gaussian608/2468 =3076/37020 özgünörnek, her3başlık için.
33944örnek kaldı; sonraki yeni Gaussian ID608. Uniform completioncursor/ZIP
korunarak atlanır;19Gaussian cacheblok yalnız RNGreplay ile okunur, yeniden
inference yapılmaz. Bilimsel manifest/campaign/source663d756 sabit tutulur;
yeni128c453 package hazırlayıp eski cache ile karıştırma.

Son capture `cf_mb_partial_20261122T162159Z.zip`5923249B,
SHA0299db66dcb0a435fb1224cfa13c88952419ff7a5515f0c7db7a1059824675fd.
Drive→Mac bağımsızCLI indirme, güvenliZIP/CRC/sizeSHA,97payload/hash/etiket/ID/
RNGchain/contiguousprefix ve önceki114dosyanın byteaynılığı geçti. Host hamveri
`result/modelnet40_c/cloudfixer_multibackbone/68d53814491e/partial_capture_20261122T162159Z/`.
[Sonprefixkanıtı](evidence/cloudfixer_multibackbone_20261122/l4_final_prefix_162159_receipt.json).
Uniform tamsonuçlarının yedekbundle'ı da yeniden Drive'dan indirilip byteaynılığı,
CRC ve yedistandartdosya kapsamıyla doğrulandı. Gaussian partialaccuracy
raporlanmadı; global/all15 accuracy veya yeni baseline üstünlüğü seçilmedi.

[Run/Open]16:23:04UTC Colabusage0.95CU, ikiaktifassignment toplam10.82CU/h;
ownedL4 marjinal1.94CU/h. Currentreserve1CU +180s pay +asgari120s yeniwindow
karşılanmıyor (hesaplanan−197s); yeni GPU bölümü açılmadı. Bu kullanıcıdan
GPUonayı isteme değil, gerçek hesapkapasitesi engelidir. [Kaynakkanıtı](evidence/cloudfixer_multibackbone_20261122/l4_capacity_final_receipt.json).
Owned `tta-cloudfixer-multibackbone-l4-20261122-r1` son16:22:16sunucuenvanterinde
canlı; tamamlanmamış task için teardown yapılmadı. PeerA100 korunuyor. Bakiyenin
bitmesi VM'yi kaybettirebilir; sonveri ikiortamda güvenli. Hesap yenilenirse
önce gerçekenvanter/Driveprefix/native/campaign/güncelkaynak doğrulanır,
aktifiş olmadığı görülür, yeni immutablegate/window03 etiketi oluşturulur.
Yeni supervisor sürümünde localexitreceipt Driveglob'dan önce kaydedilmelidir;
aktif02 bilimsel/controller dosyaları sonradan değiştirilmedi.

6h/55CU backstop ve bütünspent korunur. L4 tam bilimsel faz tahmini9.96h olduğundan
6h aşılmadan ayrı kayıtlı operasyonel capuzatımı gerekir; kullanıcıdan yeni
süre/GPU onayı gerekmez. Başka runtime'a müdahale veya labeltuning yok.
Kodun GPUhazırlığı tamam,32odaklıCPUcheck ve önceki178ilişkiliCPUcheck geçiyor;
fulltest/görev tamamlanmadığı için yeni bütünbranch paylaşılanentegrasyona
aktarılmadı. Sonraki eylem CUyenilenmesi ve aynıcacheden devamdır.

[Run]16:25UTC additional recovery-source/knowledge ZIP48944B/SHA2133d6f279945a1742a5c2e23da05b9b74d58a235574efafff79462ddf8b0113 also uploaded to taskDrive, independently downloaded and byte/CRC verified. [Receipt](evidence/cloudfixer_multibackbone_20261122/l4_recovery_knowledge_archive_receipt.json). No newGPUphase; capacityblocker persists.


[Code/Run/Clarification] 2026-11-22T18:34:30.344147+00:00 — Prior/preprocessing question:
The current alternative-head comparison is ShapeNet-all55 LION latent
VAE/diffusion+proximalPM versus ModelNet40-trained CF XYZdiffusion+author
rotation/delta optimizer/terminal vote1. Classifiercheckpoint/state and final
FPS1024 are matched. LION input bboxcenter/(longestedge/2), upsample2048,
×3.3885 and fixedaxis transform; output inverseaxis and bboxnormalize.
CF input mean-center/globalcoordinateSTD and fixedaxis transform, original
cardinality up to2048 (Uniform1024), mask/indices; output inverseaxis plus
mean-center/maxEuclideanradius. Both final scalings are isotropic; neither
rescales each axis independently. Protocol wording corrected above:2048 is
CF's upper bound, not a constant count for every corruption. Actual capacity
receipt already recorded these shapes; no code/experiment/data change.
Raw→CFframe→CF Uniform controls are68.192869→79.376013→88.290113% DGCNN,
67.301459→82.941653→88.047002% CurveNet,45.259319→69.286872→88.087520%
PointNeXt. These quantify the recorded CFframe control versus raw and CF,
not the isolated preprocessing contribution to CF-minus-LION or isolated
prior quality. A matched crossed-prior/preprocessing ablation was not run.
No newGPUwork; existing capacityblocker/coverage unchanged.


## 2026-11-22T18:42:32.900786+00:00 — User resumes on A100 High-RAM / larger batch

[User report/Code/Run/Planned/Open] User explicitly requests newA100HighRAM,
largerbatch for speed, autonomouscontinuation and closeaftercompletedresults.
ActualHEAD197551e/cleanbeforebatchpatch; sourcebranch/worktree unchanged.
Freshserverinventory0; oldL4r1externallylost, noagentteardown.500CU,0CU/h;
previous0.95CUblocker resolved. A100HighRAMr13/r14 requestsHTTP503, noVM;
boundedsequentialr15/r16 retry inprogress, noGPUtypefallback.

Design/implementation scope: only batchregistration32/64, packagerbudget6/8h,
workerprofile/recovery/options usemanifestbatch, CPUanalyzer derivesfullscope
countfrommanifest. LegacyB32/default remains; nooptimizer/math/checkpointchange.
64 means585blocks/39percorruption/tail36; smoke remains6×4, profile64, maxpoints64
capacity mustpass beforefullphase. Testnewbatch observedTypeError then34focused
CPUchecks pass; relatedsuite/review inprogress. No testlabels-basedselection.

A100newnativehardware/batch/RNG differs, soL4B32Uniform2468+Gaussian608 remains
immutablepreliminarycohort, notimported intoA100B64finalmacro. Newfull15coherent
A100B64manifest/campaign isneeded; Uniformretest is~1/15 cost toavoidmixedprotocol.
Accepted3headfiles/states/FPS1024/officialCFcheckpoint/optimizer remaincommon.
Newdestination `MyDrive/thesis/result/cloudfixer_multibackbone_20261122/a100_highram_b64`.
RawL4partial/prefix/Uniformcomparisons retained unchanged.

Operationalbudget amendment recordedbeforework: priorL4~3assignedhours retained;
wholemultibackbonetask12h/120CU cap, newA100phase8h/80CU includingcanonical10/
CFnative/setup/technical gates, reserve5CU and180s margin. This replacesold6h/
55CU operatorbackstop undernewuserrequest, not a hiddenreset; oldspent archived.
Setup1–1.5h+science1.5–3.5h, whole4–6h provisional; confirm actualA100profile/
maxcardinality andfreshallassignmentresourcegate beforefull. NohumanGPUduration
approvalneeded. Canonical10 bodies exactagain; cell1separate/reconnect, Drive
interactiveconsent useroperated; previousfastrlockonlydefer preserved.
Per64/RNGimmutableDrivepayloads, permodel/corruptionZIP+cursors, safehostrecount/
readback backup apply. Explicitusercloseaftertaskcompletion authorization includes
thisownednewruntime only, nopeerintervention. [Plan](evidence/cloudfixer_multibackbone_20261122/a100_highram_resume_plan.json).


[Run/Open] 2026-11-22T18:55:43.568846+00:00 — OwnedA100r16 established on `gpu-a100-hm-ass1c0-1smilmkaz90s4`,
A100-SXM4-80GB81920MiB/HighRAM12CPU/MemTotal175166460kB. Singleassignment
observed8.88CUh, balance500CU. Canonical1 exactbody passed; Condarestart then
tworeadonlyWSdisconnects, vanillaCLIprobe03 confirmed sameVM/Conda/A1008.0.
No cell1rerun/kernelreset. UsernewDriveconsent completed, PTY35358 Enterapplied,
physicalmount/readwrite/fsync/readback passed. SharedDriveold97blocks sum
Uniform2468+Gaussian608, expectedcampaign68d538... unchanged. Newdestination
MyDrive/thesis/result/cloudfixer_multibackbone_20261122/a100_highram_b64.
NoGoogleauthURL/accountfileinventory added toknowledge/archive.

Canonical2isrunning onprimaryCLI17679 (Condaenvironmentsolver); nootherkernel
cellqueued.3–10prepared exactbodies, perphasephysicalmarker assertions precede
completionmarker, MAX_JOBS8 justified12CPU/167GiBRAM. Packagev5/source228555f,
campaign3596befa489bc8f05cd5f139a976c7a00b7429d390a19fec850595814cd3f224,
A100/B64/8h registered. OriginalacceptedA100headassetZIP reused only after
allmemberSHA/CRC equality; L4rebuiltELFs excluded. Smallpackagefiles uploaded,
largeimmutablecachecontainers torestore fromverifiedoldDrive aftercanonical2.
131relatedCPUchecks pass. Independentreview confirmsbatch64scope/recovery/
profile/analyzer; requires runtime-bound maxpoint64 capacity and remaining80CU
operatorgate beforefull, both concreteoperatorchecks beingprepared. No new
GPUinference/accuracy yet. [Liveplan](evidence/cloudfixer_multibackbone_20261122/a100_highram_resume_plan.json).


[Code/Run/Correction] 2026-11-22T19:07UTC: CLI account changed beforethefresh
500CU inventory. Therefore currentaccount0sessions did **not** establishthat
oldL4r1wasexternallyterminated. Earliernew-account wording "oldL4externallylost"
is superseded: oldruntimepost-switch status isunverified, noagentteardown;
we didnot switchauth, touchpeer/oldruntime or claimcleanup. Lastoldaccount
observations and oldspent remainpreserved; sharedDrive97blocks/3076objects
werepositivelyverified onnewaccount. NewA100r16ownership/80GBHighRAM/8.88CUh
and its canonical3passed124.5s aredirectlyobserved; canonical4running.
Newphase8h/80CU bound applies toownedr16fromrecordedallocationtime andwill
be enforcedbyfreshgate issuer togetherwithaccountall-assignmentfunding;
oldL4execution/spent remainsseparate, notresetormergedintonewscientificcampaign.


## 2026-11-22T20:11:41.194121+00:00 — A100 B64 fulltest running after all gates

[Run/Code/Open] Canonical10complete: originalbodies, physicalnative/actualGPU
checks andSETUP_ENVIRONMENT_VERIFIED. Cell4 975.9s; EMD77.4s, Chamfer101.8s,
PointNet++physicalmarker andPVCNN155.0s;9dataand10verification89.7s pass.
CLIphase7CWDprelude failedbeforebody; onlyfailedLOCALclient93257 terminated
byownership+traceproof, CLIconsolereadonly verifiednoactivebuild;7–10freshretry02
passed. Earlier4-stageconnectionwording correctedbyactualphase07trace; noVM/
remotejobterminated. BootstrapWSalso disconnected but remotePID41895 continued;
itsDrive logconfirms CF_MB_EXTRA_NATIVE_GPU_PASS + CF_MULTIBACKBONE_DEPLOYED,
physicalnative_setup.complete/deployment.json. No duplicatedbootstrap.

Aftermandatorynotebooksetup, purePythontechnicaloperators wereexecuted as
revieweduploadedfiles through **ColabCLI console** toavoidunstableJupyterWS.
NoRunAll, notebookresearchcell, authchangeorinstalledCLIupdate. SameConda3.8.20,
scientificoverlay228555f, allcommands/bytes/Drive/localstdout retained. Detached
technicalPID49954 completed smoke01, newprocesssmoke02cachedRNGreplay, profile64,
maxheadercapacity64 andmetadata withstatuspassed19:55:14UTC. Headcheckpoint/
state/directwrapper/RNG/geometrycontrols pass. Full64Gaussian native15.779777s,
wholebudgetoperation17.956019s, peak5089.01MiB. Upsampling64×2048points native
41.639718s, peak18163.37MiB (=17.74GiB), all3headsfinite/frozen; runtimeSHA
397d6d99221fee4bac4e90a90eae9888654fc8bd5ff4e87fd7ff41b7160471f3.
[Technicalproof](evidence/cloudfixer_multibackbone_20261122/a100_b64_technical_gates.json).

Sourcecontrolindependentreview found/fixed3operatorissues beforefull: gate
selectionimmutableDrive-first/localcreate-or-verify and explicitkey/deadline
argvthroughsupervisor→worker; localstatusatomictemp/fsync/replace; independent
supervisor deadline kill/reap even nativeworkerholdsGIL. CPUfailureinjection
passed concurrent100JSONupdates, conflictpreservesoldselection, nativeGILsleep
childkilled andreaped, andstart/terminalDrivewritefailures stillwaitactualchild.
Finalreviewnolaunchblocker. [Controlproof](evidence/cloudfixer_multibackbone_20261122/a100_full_control_verification.json).

Fresh19:57:58UTC account489.36CU/8.88CUh/oneassignment. Gatefull_gate_d695904649420376cd552ff4aa1632cca9a2ddde6fc3a81663535e55f18a1664.json
includesmeasuredcapacity/runtime/profile/sourceSHA andallresourcebounds.
EstimatedownedCUaccrued11.392 includeswholeallocationelapsed, notclaimedbilled.
8h/80CUnewphase,12h/120CUtask; priorL4persistedbudget8805.354s retained,
old55CUallowance reserved(notactualspend) underglobal120 givesnew65CUmaximum;
180s costmargin/5CUaccountreserve retained. Noreset. EmpiricalnativeA100B32
all15 extrapolation scaledbynewB64Gaussian/maxpoints, new3headcost andmeasured
leaseoverhead floor2.5s,25%margin+120s ⇒4.392751h conservativefullETA.
Admittedwindow15813sfromaccountread, expires2026-11-23T00:21:31.323761Z;
earlierpartialstopretainscache ifneeded. Native3headreconstructionisshared,
not3independentCFsimulations. Technicalprofileisnotaccuracyselection.

Fullparent56566 started20:00:40UTC; worker56962, run`20261122T200042Z_full_window01`.
Scopeall2468×15/s5/seed0/B64/585blocks/tail36;3acceptedfrozenheads andoriginal
CFoptimizer30/vote1; source_raw/source_cf_frame/plainCFcontrols. Immutableper64
Drivepayload/receipt/RNG andpercorruption/headarchives/cursors apply. OldL4B32
pilot3076preserved separately, notmixedintonewA100macro. Hostcollector69593
readsrealprogress viaCLIContentsAPI, downloadscompletedDriveZIPtofreshdirs,
safeZIP/SHA/CRC/IDs/labels/RNG/CSVrecounts andcombinesscopeuniquely. Sevenfile
bundleaddsdocumentedrawstdoutsidecar whileoriginalZIP remainsunchanged.
Noall15accuracyorcompletedtaskclaimyet. [Launchproof](evidence/cloudfixer_multibackbone_20261122/a100_full_launch_receipt.json).

Rawlocalroot `result/modelnet40_c/cloudfixer_multibackbone/3596befa489b/`;
Drive `MyDrive/thesis/result/cloudfixer_multibackbone_20261122/a100_highram_b64`.
Canonical+operationalsourcearchive110976B/SHA216646e1...e278af60 uploaded;
Independent Drive-to-host readback passed at20:15UTC: size/SHA/CRC and62members match; receipt
`a100_canonical_operational_archive_receipt.json`. Finalall15raw/CPUrecount/Drive
andhostvalidation/knowledge precede explicitlyuserauthorizedownedr16close.
OlderaccountL4runtimepost-switchstatus remainsunverified; nootheraccount/runtime
wasmodifiedorclosed. Next: continuefullworker, ingesteachcompletedcorruption,
recordall15outcomes andpublishnecessarycuratedthesisresults thenauthorizedclose.

### 2026-11-22 20:15 UTC — İlk A100/B64 corruption doğrulandı

[Run] Uniform/s5/seed0, özgün ID0–2467, 39blok ve üçdonmuşbaşlık için
Drive ZIP → yeni yerel dizin SHA/CRC, etiket/ID, RNG zinciri ve hamlogit
accuracy sayımı geçti. Bu yeni A100/B64 campaign sonucudur; önceki L4/B32
pilotunu üzerine yazmaz ve iki koşu aynı deneymiş gibi birleştirilmez.

| Başlık | Source raw (%) | CF frame kontrolü (%) | Plain CF (%) | Eski LION N8 PM (%) | CF−LION (pp) |
|---|---:|---:|---:|---:|---:|
| DGCNN | 68.192869 | 79.376013 | 88.168558 | 78.484603 | +9.683955 |
| CurveNet | 67.301459 | 82.941653 | 88.452188 | 76.701783 | +11.750405 |
| PointNeXt | 45.259319 | 69.286872 | 87.965964 | 80.632091 | +7.333874 |

Ham ZIP SHA `70e95ab8666b793f241023bc1f97157a544313111cd87fd642008a5b4b3711e3`.
Yerel kanıt: `result/modelnet40_c/cloudfixer_multibackbone/3596befa489b/`
`retrieval_20261122T200631Z/analysis_1/recount.json`; yalnız Uniform, all15 değildir.
Gaussian devam ediyor; her64 örnek Drive'a sonuç+RNG, corruption tamamlandığında
ZIP ve tamamlanma imleci yazılıyor. Farklı prior/ön işleme/RNG nedeniyle kıyas
bütün pipeline'lar arasındadır; bu tablodan PM'in veya prior'ın tekbaşına etkisi çıkarılamaz.

### 2026-11-22 20:23 UTC — Gaussian ve yayın incelemesi

[Run] Gaussian39blok/2468ID tamamlandı. Ham ZIP SHA
`10080c6fb9c05623fbe17b945c536d9fb038ed325c18b0e9fa67568249d65f65`;
Drive→host SHA/CRC ve bağımsız ID/etiket/RNG/logit sayımı geçti. DGCNN/CurveNet/
PointNeXt plain CF doğru sayıları2166/2184/2168, accuracy%87.763371/%88.492707/
%87.844408. Eski LION PM'e +9.359806/+12.155592/+7.171799pp; yalnızGaussian.
İlkiki corruption78blok doğrulandı, `analysis_2/recount.json`; Background aktif.

[Code] Bağımsız yayın incelemesinde gelecekteki paketlemeye ait tek açık
giderildi: hareketli integration referansı okununca kabul edilmiş LION kaydının
94668bayt/SHA199e52a5...bcc48 kimliği artık zorunlu. Aynıuzunlukta yanlışSHA ve
yanlışuzunluk vakaları önce kırmızı, düzeltme sonrası133ilgiliCPUcheck4.59s
geçti; gerçek referans pozitifkimlik kontrolü de geçti. Bu yalnız yeni paketler
içindir: çalışan v5/source228555f ve manifest değişmedi; mevcutreferans zaten
doğruydu. Tamall15 sonuçları için yenidenGPU koşusu gerekmez. Kaynak paket
kimliği korunacak, paylaşılan gelişmiş `cloudfixer_diffusion.py` geriye alınmayacak.

[Code/Interpretation] Payload `denoiser_calls=30` yapılandırılmış resmî güncelleme
sayısıdır, bağımsız dinamikçağrı sayacı değildir. `peak_mib` bütünblokpeak'idir;
başlıkzamanları paylaşılan reconstruction + senkronön işleme/forward toplamıdır,
yalıtılmış end-to-end model latency ölçümü olarak sunulmayacaktır.

### 2026-11-22 20:35 UTC — Background tamtest kontrol ayrımı

[Run] Üç corruption117blok/7404girdi/başlık doğrulandı. Background/s5/seed0/
2468ID/B64 hamZIP SHA `c0cbac468aa3ea768f7db928c6eb1c3f66d12043c75fdbb2f98e035f430c8be1`,
Drive→host SHA/CRC ve hamlogit/ID/etiket/RNG doğrulaması geçti.

| Başlık | Source raw (%) | CF-frame source (%) | Plain CF (%) | Eski LION PM (%) | CF−LION (pp) |
|---|---:|---:|---:|---:|---:|
| DGCNN | 13.614263 | 24.473258 | 55.348460 | 15.802269 | +39.546191 |
| CurveNet | 7.576985 | 4.538088 | 49.918963 | 14.586710 | +35.332253 |
| PointNeXt | 85.251216 | 49.392220 | 75.607780 | 70.299838 | +5.307942 |

[Inference] CF eski LION PM'i üçbaşlıkta geçiyor; fakat source-only'ye göre
herbaşlıkta faydalı değildir. PointNeXt raw2104doğru → CF-frame1219 → CF1866:
ön işleme885netdoğru kaybediyor, CFoptimizasyonu aynıframekontrolüne647 ekliyor,
amasonuç rawsource'a göre238netdoğru (%9.643436pp) kayıp. Bunlar net sayılardır;
CF-only/raw-only çiftlerinin ayrı büyüklükleri değildir. Bu gözlem CFframe'in
sebepseltekbileşenetkisini izole etmez; ölçek/eksen/FPS işlemleri birlikte değişir.
DGCNN veCurveNet için aynı kontrol üçlüsü farklı davranıyor. Accuracy'ye bakıp
corruption'a veya başlığa göre yöntemi değiştirmiyoruz; tüm15kilitliprotokol sürer.

Eski karşılaştırıcı, LION proximalN8/tau.1/retained.95, equalPM aday2/4/6/8,T1,
DDIM100 ve orijinal encoded decoder style'dır. Bu, CloudFixer XYZprior'lı
N8/K80/t20-t10-t0 yeni tezbaseline'ımızla aynı yöntem değildir. Bu deney
yalnız eskiLIONpipeline ile plainCFpipeline kıyasıdır; yeniN8-CF yönteminin bu
üçbaşlıktaki başarısını ölçmemiştir. [Ham sayım özeti](evidence/cloudfixer_multibackbone_20261122/a100_first_three_corruptions.json).
Impulse aktif, tam15sonucu henüz yok.

### 2026-11-22 21:13 UTC — İlk beş corruption güvenli

[Run] Uniform/Gaussian/Background/Impulse/Upsampling:195blok, herbirinde2468ID
ve9raporsatırı; toplam12340girdi/başlık. Host `analysis_5/recount.json` tamolmayan
5/15kapsamı olarak doğruluyor. Impulse plainCF doğruDGCNN2170/CurveNet2177/
PointNeXt2197; Upsampling2183/2204/2179. Upsampling hamZIP
SHA `6db712b56c90386c8255bc2c004ea90548b7e3f50edf95daf5db4abe990ec15c`,
Drive→host SHA/CRC/hamtahmin/ID/etiket/RNG doğrulaması geçti. UpsamplingB64
max2048nokta kapasiteprovası tamkoşuda da OOM olmadan tamamlandı. Çalışanmodel/
ayar/PMseçimi değiştirilmedi; DistortionRBF aktif. Bağımsız inceleyici
kabul edilmiş LION referansını sabitleyen gelecektekipaket düzeltmesini
pozitifkimlik ve tekbaytdeğişimiyle de doğruladı; açık yayınkodengeli yok.
Kalan10corruption ve tamall15sayımı tamamlanmadan deney tamamlandı denmeyecek.

### 2026-11-22 21:42 UTC — Density olumsuz sonucu

[Run] Sekizcorruption312blok/19744girdi/başlık doğrulandı; `analysis_8/recount.json`.
Density/s5/seed0/tam2468/B64 hamZIP
SHA `4934eb7bae9f9638588a1c252cc806248a2ce1d6b5ece0e6c2e53ee809fb1e46`.

| Başlık | Raw source (%) | CF-frame source (%) | Plain CF (%) | CF−eskiLIONPM (pp) |
|---|---:|---:|---:|---:|
| DGCNN | 51.620746 | 50.931929 | 50.526742 | −15.640194 |
| CurveNet | 79.132901 | 78.119935 | 74.027553 | +4.173420 |
| PointNeXt | 85.615883 | 85.170178 | 77.431118 | −2.188006 |

[Inference] CF bütüncorruption'larda üstün değildir. Density'de DGCNN eskiLIONPM'e
386netdoğru, PointNeXt54netdoğru geriler; CurveNet103netdoğru kazanır. Ayrıca
CF, rawsource'a göre üçbaşlıkta27/126/202netdoğru kaybeder (DGCNN/CurveNet/
PointNeXt). Aynıframekontrolüne göre de üçünde kayıp vardır. Bunlar gözlenen
netfarklar; ayrıpairedwin/loss sayıları veya bileşenin tekbaşına nedenselliği
değildir. Density ve Background istisnaları nihai raporda korunacak; test
etiketleriyle politika değiştirilmez. Önceki tamamlananDistortionRBFInverse'de
CurveNet CF%86.466775, CF-frame source%87.844408: optimizasyonun herbaşlık/
corruption'da fayda sağlamadığına ek gözlem. Kalan7corruption, all15sayımı ve
yedekler tamamlanmadan tamsonuç veya yeni modelüstünlüğü iddia edilmeyecek.

### 2026-11-22 22:08 UTC — On corruption tamam, Rotation aktif

[Run] Tam10/15kapsam,390blok/24680girdi/başlık host `analysis_10/recount.json`
ile doğrulandı. DensityIncrease CF(DGCNN/CurveNet/PointNeXt)2073/2131/2110doğru,
%83.995138/%86.345219/%85.494327; eskiLIONPM'e +2.755267/+7.171799/−2.269044pp.
Shear1918/2009/1865doğru, %77.714749/%81.401945/%75.567261; eskiLIONPM'e
+5.713128/+7.212318/+1.782820pp. Shear CF-frame kontrolü%80.915721/%83.833063/
%78.970827 ile üçündeCF'den yüksek. Bu kontrollerin olumsuz sonuçları korunuyor.
DensityIncrease hamZIP SHA09406f07...1f539a3; Shear e387b20e...d2a93e,
Drive→host SHA/CRC/ID/etiket/RNG/hamlogit tekrar sayımı geçti. Rotation aktif;
all15sonuçları, son arşiv/Driveflush ve kullanıcıyetkiliownedr16teardown bekler.

### 2026-11-22 22:20 UTC — Rotation tamtest sonucu

[Run] Rotation/s5/seed0/B64/2468ID; SHA83f0eff9...ab2f2c2 hamZIP Drive→host,
CRC/ID/etiket/RNG/hamlogit sayımı geçti. Tam11/15corruption429blok doğrulandı;
Cutout aktif. [Rotation kanıtı](evidence/cloudfixer_multibackbone_20261122/a100_rotation_validated.json).

| Başlık | Raw source (%) | CF-frame source (%) | Plain CF (%) | Eski LION PM (%) | CF−LION (pp) |
|---|---:|---:|---:|---:|---:|
| DGCNN | 57.252836 | 62.844408 | 75.972447 | 53.930308 | +22.042139 |
| CurveNet | 68.800648 | 75.405186 | 83.306321 | 66.450567 | +16.855754 |
| PointNeXt | 39.059968 | 42.585089 | 63.006483 | 41.815235 | +21.191248 |

[Inference] Rotation'da plainCF rawsource'u, aynıframekontrolünü ve eskiLIONPM'i
üçbaşlıkta da geçiyor. CF−frame netdoğru324/195/504 (DGCNN/CurveNet/PointNeXt).
Bu resmîrotation+delta paketinin gözlenen toplam katkısıdır; yalnızrotasyon
bileşenini veya yalnızprior'ı izole etmez. BackgroundPointNeXt rawsource
istisnası ve Density/DensityIncrease olumsuzları hâlâ geçerlidir; tümsonucu
Rotation'daki kazanımlara indirgemiyoruz. YeniCF-XYZ N8/K80/t20-t10-t0 tez
baseline'ımız buüçbaşlıkta ölçülmedi; eskiLIONPM ile aynı yöntem sayılmaz.
Kalan4corruption/nihai135satır/585blok sayımı, Drivebackup/flush ve yetkili
ownedruntimekapatma tamamlanınca all15raporu kesinleştirilecek.

### 2026-11-22 23:06 UTC — All15 tam sonuç, host ve Drive readback geçti

[Run/Complete] Kaynak228555f/v5 campaign3596befa489b, A100/B64, all2468×15/s5/
seed0;585blok/tail36,135corruption-output-head satırı,9overall satırı doğrulandı.
Workerfullflagtrue/15 ve gerçeksupervisorreturncode0; bağımsızdeadline dolmadı.
Gerçeksüre20:00:42.921245–23:01:08.928929UTC,10826.007684s=3.007224h;
8.88CU/h'de yalnızfullsüre26.704152CU tahmini, setup/finalization ayrıca ve gerçek
faturalama değildir. Bağımsız CPU, tümID/etiket/logit/RNG ve CSVleri doğruladı.

PlainCF DGCNN26387/37020=%71.277688, CurveNet28110=%75.931929,
PointNeXt27772=%75.018909. Kabul edilmiş eskiLION N8 PM'e sırasıyla
+7.336575/+10.253917/+4.392220pp, net+2716/+3796/+1626doğru. CurveNet15/15,
DGCNN12/15, PointNeXt12/15corruption'da yüksek. Rawsource veCFframe all15
ortalamaları da raporda; yerelnegatifkontroller korunuyor. Bu eskiLIONPM2/4/6/8
ilebütünpipeline kıyasıdır, yeniCF-XYZ N8t20/t10/t0 veya PMtekbaşına testi değil.

Son ham/analiz/kaynak paketi
`result/modelnet40_c/cloudfixer_multibackbone/3596befa489b/final_20261122T230335Z.zip`,
65528957B/SHA `bcb64ea0c5c0165e43c4498dfc428af6c1e4804f82889e17d4924b76e38cfe07`.
15orijinalZIP değiştirilmeden, tam7dosyalıderivedreport, actualworker/supervisor
çıkışları, tamv5overlay/resmîkaynak ve canonical/operatör/CPUanalizkaynağı içerir.
Drive aynıtaskklasörüne yükleme + yenihedefe bağımsızCLIindirme SHA/CRC/41üye
kontrolü geçti. [Paket makbuzu](evidence/cloudfixer_multibackbone_20261122/a100_final_bundle_receipt.json).

[Ayrıntılı tüm15 tablo, kontroller, negatifler ve çıkarımlar](cloudfixer_multibackbone_results_20261122.md).
GPUdeneyi tamamlandı; sonbilimselinceleme, gerekli kod/knowledge'ın güncel
thesis-integration'a aktarımı, resmiDrivecloudflush ve kullanıcının açıkça
yetkilendirdiği yalnızownedr16kapatma sürüyor. Diğer account/runtime'lara dokunulmadı.
