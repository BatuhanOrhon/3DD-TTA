# PM M100 quarter-screen retry r5 — 2026-11-08

[User decision/Run/Open] User says “Tekrar dene” after r4 allocation rejection.
The single new High-RAM A100 request succeeded. Owned session:
`tta-gsd-quarter-fast-a100-20261108-r5`, endpoint
`gpu-a100-hm-euw4a0-1bhkcvsfl7lu9`. Foreground monitor21273 started07:08:48UTC,
raw stream result/preparation/gsd_lowband_quarter_fast_20261108_r5/setup/monitor.jsonl.
No peer operation or runtime teardown. Prior stale local terminals were closed
and verified in the r4 record. No active server session existed before r5.

[Decision] This is a separate authorized four-hour attempt:
2026-11-08T07:08:21UTC→11:08:21UTC, including allocation/setup/authentication,
profile/calibration/technical checks/617 inference and archiving. Prior r3/r4
records and clocks stay unchanged. GPU permission need not be requested again;
a recurrent Colab connection/session problem still triggers WAIT_USER before
any further retry/recovery/new allocation.

[Protocol] Fixed617/2468 official ModelNet40-C test IDs, all15/s5/seed0;
operational SCD/livehalf0/hardM100. Fourteen normal N8 corruptions and
Background35; DDIM100/finalalpha1; gamma.03/eta.003/p.95; raw/eval/frozen
PointMAE+LION. True hard100 complete eigenspaces and unchanged numerical gates.
Profile B8/16/32/64 using only measured resources; verify actual GPU VRAM.
Fresh actual-batch median384 local/style update-norm ratios on64 fixed IDs
outside617; no labels/clean geometry, then freeze before accuracy. Three-arm
completed617 PM comparison with object-cluster CIs; nonpositive M100-minus-SCD
stops further expansion. Positive CI spanning zero remains inconclusive;
native instability persists. No silent scope shrink or automatic Scan/full run.

[Code/Open] Scientific source918780b and accepted asset identities unchanged.
Only controllers, clock bindings and output namespaces are retargeted into new
preparation/Drive/remote roots, followed by relevant CPU checks and independent
review. Worktree /Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-lowband-halfmix-20261105;
branch research/gsd-lowband-quarter-fast-retry-20261108. Last integration5fb5b5f;
local e66c035 preserves prior CPU preparation and r3 connection failure.

[Run/Open] Canonical cell1 submitted07:09:30UTC. All ten cells remain required,
with same-VM Conda restart/reconnection, separate native phases and exact
completion markers. Actual GPU memory and fresh setup are not yet verified.

[Run/Open] Cell1 passed07:09:43UTC and same-VM Conda reconnect passed07:10:35UTC;
actual GPU is NVIDIA A100-SXM4-80GB,81920MiB. At07:11:05UTC the known-idle
local monitor was stopped solely for serial CLI Drive authentication ownership;
runtime was not closed. Sole owner is now pending drivemount terminal95569.
Google's new-runtime Drive consent is genuinely pending and the user was asked
to complete it. No OAuth URL/state/token is written to repository knowledge or
published artifacts. After mounting, the same endpoint's monitor will restart
before cell2. No competing kernel cell or fake keep-alive is run during consent.
The clock is unchanged. CPU-only controller retargeting continues independently.

[User report/Run/Open] User completed the real Google Drive consent; CLI
credentials propagated and `Mounted at /content/drive` returned exit0 by
07:13:06UTC. The same endpoint monitor restarted07:13:12UTC, sole foreground
owner74712, appending to the same raw setup log. Cell2 submitted07:13:23UTC.
Drive mount precedes cell2 as an explicit serial setup adaptation; canonical
cell sources/order are unchanged. No competing CLI/browser kernel execution.

[CPU/Review/Open] R5 retarget batch completed and independently reviewed:
14 operator contract checks,40 simulated screen jobs,27 acceptance tests
(15 phase/12 screen),17 wrapper cases,5 deployment trust-boundary cases,
transfer/deadline/WAIT_USER checks,25 Python3.8 syntax checks. Retarget audit
confirms59 prior inputs unchanged. Scientific asset payload4eaa62df and source
918780b/a62a7e unchanged; no method math, ID selection or control change.
New operator payload138072bytes/SHA5bf6b8231ba1a7fec9fcf3adc465a90f74fdc9d75df774e69224001e9494971a;
operator manifest52ae65ff39e2a7733363720fb8e50af5847d5df0a7d5bc55f5d1a88214c5c9c5;
portable acceptorcb6daa2d2567e83915b46ce10042388bc88c7683974b9fbef6e4b871ad743a27.
Exact staged paths are upload_map.json; all10 initial uploads returned0.
Canonical ten source cells and wrappers were compared against the actual
checkout and match every recorded SHA. Cell2 solver passed07:17:33UTC; package
installation continues. These are CPU/setup observations, not speed/accuracy.
Last publicationca0f015 includes completed r3/r4/r5 history; subsequent evidence
will be integrated under the same thesis-integration policy.

[Run/Open] Canonical cell2 ended07:19:45UTC at the single previously authorized
fastrlock0.8.4 pip-wheel failure. Raw errors identify no other failed wheel.
Only this known failure is deferred; environment/Drive verification is separate
before cell3. No connection/session failure occurred at this stage.

[Run/Open] Python3.8.20 and Drive verification passed07:20:20UTC. Cell3 passed
07:22:43UTC after its delayed original CLI stream returned the already-mounted
Drive and byte-identical KNN wheel. No repeated cell or forced remount. Cell4
submitted07:23:07UTC, preserving /content/3DD-TTA and checked shell statuses.
This is still canonical setup; research deployment/profile/calibration have
not started. Own monitor74712 remains the sole execution owner.

[Run/Open] Cell4 PyTorch2.1.2+cu121 installed07:24:04UTC, KNN-CUDA07:24:10UTC,
CLIP07:29:31UTC; CUDA-toolchain solver passed07:30:47UTC. Requirements emitted
the same nonfatal resolver warnings as r3: dask2023.5.0/importlib-metadata4.10.1
and scikit-image0.20.0/networkx2.6.3. They remain recorded; no global dependency
consistency claim. Native GPU and final full setup checks remain required.
Local archive space precheck recorded209.54GiB free in local_disk_precheck.json.

[Run/Open] Canonical cell4 passed07:34:21UTC, GPU architecture8.0, with expected
PointNet++ setup and complete-EMD source patches. Cell5 EMD submitted07:34:49UTC.
Local r5 archive root is
result/modelnet40_c/lowband_quarter_fast/20261108-070821_a10080_screen617/;
prior B2/failed-attempt roots remain unchanged. No research inference yet.

## WAIT_USER — recurrent CLI connection loss at cell6

[Run/Failed/User decision/Open] Cell5 EMD completed07:36:52UTC with exact
completion marker. Cell6 Chamfer was submitted07:37:12UTC; the CLI raised
`RuntimeError: Connection was lost.` at07:37:23UTC. No exact cell6 completion
marker or exec-finished event has been observed. Its actual execution state
remains unknown; this is not proof that the VM stopped or that Chamfer ran.
The previously successful r3 native compilation does not complete fresh r5 setup.

The user's explicit recurrent-connection boundary applies again: WAIT_USER.
No retry, recovery, new allocation, reconnect or new remote inspection after
this error. Only a local monitor stop request was issued to suppress further
autonomous launches/probes; it is deferred while the current CLI remains
unresolved. Runtime not closed, peer untouched. Research asset deployment,
resource profile,384-row calibration and617 screen never started in r5.
New runtime allocation did not prevent recurrence of the observed CLI error;
root cause and assignment state have not been established. Raw setup stream
and incident JSON are retained in the r5 preparation folder. The existing
07:08:21→11:08:21UTC clock remains unchanged; waiting does not reset it.
Last published integrationca0f015; later local setup/failure records await a new
user command before external publication or any other new external action.

[User request/Run/Open] At08:45:17UTC the user requested current liveness only.
Read-only `colab sessions` lists owned r5 at its original endpoint as A100,
High-RAM. Server assignment still exists; this does not establish kernel
responsiveness or cell6 completion. No cell, reconnect, retry or research
resume was requested or performed. WAIT_USER for further work remains.

## Explicit live-session continuation — 08:46UTC

[User decision/Run/Open] User instructs continuation on the existing live r5.
Old hung local CLI50549 was ownership-verified and terminated locally08:47UTC;
monitor74712 then recorded returncode-15 and stopped, without runtime teardown.
Task-owned raw console40653 read native markers/processes/GPU and local Jupyter
kernel metadata. Only prepare+EMD markers exist; no nvcc/ninja/setup build is
active. A10080GB responds with0MiB/0% usage, Drive exists. Jupyter API on its
actual bound172.28.0.12:9000 returns both kernels idle (last activities
07:37:34 and07:12:50UTC). This resolves the duplicate-build ambiguity: Chamfer
is incomplete and no build/kernel work remains active. No research job existed.
Read-only console was closed before returning notebook ownership to the monitor.
The same VM will continue from canonical cell6; completed cells remain preserved.
Original deadline11:08:21UTC is unchanged; user waiting time was not hidden or
reset. Connection failure cause is not established by this audit.

[Code/Run/Open] Offline installed CLI inspection found supported per-process
REQUEST_TIMEOUT=60 (default10): it governs WebSocket startup/readiness and
silent chdir prelude timeouts, unlike real-cell --timeout. It does not enable
automatic reconnection or allocate a kernel/runtime. Existing failures occurred
about11seconds after submission while the VM later showed idle kernels, but
this timing is not proof of a single root cause. New monitor30443 started
08:49:56UTC on the same endpoint with this single configuration change; it is
the sole owner after read-only console closure. Canonical cell6 is retried only
after proving both kernels idle, no active build and no Chamfer completion marker.
No global CLI update/patch, authentication change or model/math change.

[Run/Open] Supported local REQUEST_TIMEOUT=60 was independently import-tested
against the installed CLI: constants/default client execution/HTTP/WS startup
are60, WebSocket reconnect interval remains0. Existing HTTP startup retry
behavior was not changed. This is a timing-handling configuration, not a proven
root-cause fix. Same-VM cell6 began08:50:17UTC and Chamfer completed08:52:13UTC
with both native phase and exact wrapper markers. Cell7 PointNet++ submitted
08:52:56UTC. No cell replay while busy or fresh runtime allocation occurred.

[Run/Open] PointNet++ native phase reported completed08:57:02UTC; original
stream/wrapper completion arrived08:57:09UTC. Cell8 PVCNN was submitted
09:13:59UTC. Raw monitor records the intervening elapsed time; it is not removed
from the allocation clock or reclassified as GPU inference. Deadline11:08:21UTC
still applies. Publication6696bc6 preserves completed failure/recovery evidence.

[Run/Failed/Open] Cell8 native GPU smoke completed09:21:21UTC; cell9 completed
09:38:39UTC; cell10 full environment verification passed09:40:03UTC. Downloaded
fresh proof9005bytes/SHA9f2d7f3f1a5bec56408518906fe2f8d7e63ed1f70849b25170c628ae0892fbda:
Python3.8.20/Torch2.1.2+cu121/CUDA12.1/NumPy1.21.2/A10080GB,15arrays,prepdev a458.

[Run/Failed/Code/Open] Initial file-only deployment failed09:40:11UTC at the
symlink guard, before asset directory/native mutation. Read-only audit passed
09:42:20UTC: configured MyDrive/thesis is a Google Drive shortcut entirely
within the mounted Drive; resolved base archive exactly247002524bytes and
SHAce9c797ccacb4eeffcc0a2836c65109213c9c7920a40a5145ef598b2adac1333.
ROOT/prepared/assets and ROOT/deployment_receipt.json do not yet exist. The
failed stage/payload/proof is preserved. A narrowly scoped file-only recovery
is being CPU-checked: resolve only the audited configured Drive paths, keep
general/native symlink guards and durable Drive-before-native backup unchanged.
No research model/loss/code, IDs, calibration or inference has changed or run.

## User removes the time-window constraint — 10:05UTC

[User decision] User explicitly says to disregard the time window and begin
the comparison. This revokes the11:08UTC global stop; it does not waive source,
cache, alpha, finite-difference, repeat, archive or matched-control gates.
Original budget/failed time projection stay immutable. A new versioned
administrative controller will require an explicit hashed waiver, keep original
allocation start/deadline for history, and impose no new arbitrary total window.
Per-child/transfer safeguards and finite40-job scope remain; no automatic full
or Scan expansion. No accuracy-based batch/weight choice.

[Run/Accepted resources] Profile raw ZIP269037151bytes/SHA
f94fc37ad66656fdd2d4ac1e89b51e7d8fda4f7f3068c45439bdd3fc126d4d03
is Drive+Mac verified, with514 direct and464 recovered files accepted. All
B8/16/32/64 jobs completed. B32 chosen by the predeclared resource rule:
peak36485.22MiB; Gaussian case11.38154s, Background21.93959s; mixed startup/
interim-copy19.73374s and post-worker9.77925s per two-corruption profile.
B64 peak72413.01MiB exceeds85% of81152.75MiB despite slightly better throughput.
The historical early gate extrapolated24157.7s before margin/reserves by
conservatively scaling unseparated startup/copy costs; this is not a measured
whole617 duration and must not be presented as one. Its failure alone stopped
profile status before calibration, while all resource measurements remain valid.
No617 inference accuracy or frozen new alpha exists yet. New controllers will
reuse this validated selection and caches only at their declared scope, then
perform fresh384-row calibration/technical checks before the comparison.

## R5 assignment loss and owned R6 continuation

[Run/Failed/Open] Original R5 monitor inventory timed out10:22:47UTC, then
server inventory reports assignment missing10:26:34 through10:31:35. Local
monitor stopped10:32:33 with runtime_teardown=false; no VM stop was issued.
The accepted resource profile/context remain preserved; new calibration,
technical checks and617 inference never launched on R5. No OOM/root-cause
evidence exists, so this loss is not labeled a RAM failure.

[User decision/Run] User rejects ownership/use of the unknown A100 and explicitly
authorizes a new own session if R5 is gone, protecting peer runtimes. Read-only
10:49 inventory lacks R5. Own r6 subsequently allocated10:56:29 after an audited
503/no-assignment attempt and same-name retry. Current continuation is
[owned R6 comparison](gsd_lowband_quarter_comparison_r6_20261108.md); historical
R5 time projection/native/profile/source evidence stays unchanged. R6 reuses
only the validated B32 resource choice, never R5 alpha/prepared states.
