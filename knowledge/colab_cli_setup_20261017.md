## 2026-10-30 — New depth-study A100 ten-cell replay passes

[Run/Open] Owned tta-chamfer-depth-tuning-a100-20261030-r1 completes all10
canonical cells by12:35:03UTC, including native GPU checks, dataset and weights.
Only the known cell2 fastrlock build failure was deferred; requirements later
installed fastrlock0.8. Existing dask/importlib-metadata and scikit-image/networkx
pip conflict warnings are recorded, without claiming global pip consistency.
Intermittent CLI prelude connection losses were audited before retries; no
duplicate setup/research process, CLI update, auth-state edit or peer operation.
Cells6–8 preserve canonical code and append physical-marker asserts.
The actual dev@a458cd4 preparation checkout, full runtime versions,17dataset
identities and both checkpoint identities are preserved in the
[depth study record](chamfer_depth_tuning_study_20261030.md) and its downloaded
setup verification JSON. Scientific sealed-source CPU deployment is pending;
this setup replay is not an accuracy result or research-GPU launch approval.

# Colab CLI setup on Mac — 2026-10-17

Evidence: [Code], [Run], [User decision], [Open].

## 2026-10-25: Agent B fresh ten-cell replay passed

[Run/User decision] Isolated session `tta-guidance-components-a100-20261025-r1`
on endpoint `gpu-a100-hm-kkb-ass1c2-36gbz99z2hf98` completed all ten canonical
setup cells on A100-SXM4-80GB/High RAM. Cell1 ran separately and reconnected to
the same VM. Cell2's known fastrlock wheel failure was the only deferred error;
Python3.8.20 was verified. User-completed Drive consent and a Cell3 retry passed.
Cell4's first CLI connection failed before sending setup code; an idle kernel
audit permitted a same-VM retry with REQUEST_TIMEOUT60, which passed. Separate
EMD/Chamfer/PointNet++/PVCNN completion markers and actual native GPU smoke passed.
Cells9/10 verified ModelNet data, CUDA12.1/GCC11.4/PyTorch2.1.2+cu121 and weights.
No full research notebook was executed during setup.

[Run/Code] Setup evidence56156bytes SHA256
`13338f05cf2fc67418911e3eafba7723a746f948ef6997aa36ee05f0acaa132a`
matches actual Drive and downloaded local bytes, member identities and CRC.
Receipt and exact extracted-cell wrappers are under
`result/preparation/guidance_component_study_20261025/setup/`.
For the later matched study, fresh setup binaries were first archived and then
the accepted pinned four native binaries restored with no-build identity checks;
that deliberate study binding is distinct from fresh-build setup success.
[Current study](guidance_component_study_handoff_20261025.md) records complete
source/native identities and the live runtime lifecycle. Historical r7 status
below is retained as dated evidence, not the current Agent B status.

## Later 2026-10-18: Cell4 split for future runtimes

### First fresh replay: r7 active

[Run/User decision/Open] The earlier r6 session, which still used the combined
six-cell layout, disappeared from the server inventory during Cell4 at about27
minutes without a completion marker. The still-waiting local executor was then
interrupted and exited130; the CLI pruned the stale r6 record. No native success
or research result is attributed to r6.

The user directed an immediate retry with the split components. Fresh
A100/High-RAM session `tta-scanobjectnn-nobg-a100-r7-20261018` was created from
the canonical ten-cell notebook. Cell1 emitted its wrapper marker and the CLI
reconnected to the same endpoint
`gpu-a100-hm-kkb-euw4a2-1d8qxj97k4u94`; Cell2 created the environment and ended
only at the explicitly deferred `fastrlock` wheel failure. The first Cell3
attempt ended with `ValueError: mount failed`; the user then authorized a
TTY-backed CLI `drivemount`. `/content/drive/MyDrive` and the source KNN wheel
were listed, and the Cell3 retry copied the wheel and emitted
`SETUP_WRAPPER_CELL_3_COMPLETED`. Next run separate Cells4–8 with exact
wrapper/phase markers. Cells9–10 and no-background inference remain
pending. Keep r7 live until explicit task-complete and close authorization.

[User decision/Code] The canonical extracted setup notebook now contains ten
code cells instead of six. The old combined Cell4 is split into five phases:
package/toolchain/source preparation, EMD, Chamfer, PointNet++, and PVCNN plus
native GPU smoke. The shell helper accepts `prepare`, `emd`, `chamfer`,
`pointnet2`, `pvcnn-smoke`, or backward-compatible `all`; it rejects an unknown
phase before touching Conda. Each successful phase writes an on-disk marker,
checks its prerequisites, streams build output and emits UTC heartbeats every
roughly two minutes while a compile remains active.

[Code validation] `tests.test_colab_environment_setup` passes three tests;
`bash -n`, `jq empty`, `git diff --check`, and byte comparison of the embedded
and standalone helper pass. Helper SHA-256 is
`bd2baffb888de47e14f4119bd69df999450fd6d26d2ea2898e73d03c816a3bb6`.
The complete local suite passes266 tests in an isolated uv environment with
numpy/torch/scipy/h5py/PyYAML, and the official skill validator reports valid.

[Code/Diagnosis] CLI0.7.4 `colab status` reads the CLI's local session-state
`running` field; it is not an independent remote-kernel probe. Separate CLI
processes can therefore show a transient IDLE/BUSY contradiction, while a
silent cell may still be healthy. Monitoring now keeps one primary `colab exec`
process, preserves its raw output, uses `colab sessions` only for server
assignment existence, and requires the exact wrapper marker plus the native
phase `.complete` marker before advancing. The new skill contract test failed
before this rule and all6 tests plus official validation pass after it.
The full research notebook is intentionally preserved. Operational cell mapping
is now Cell1 restart, Cell2 environment, Cell3 Drive/KNN, Cells4–8 native setup,
Cell9 ModelNet40-C, Cell10 weights/final verification.

[Run/Open] The split notebook's first fresh replay is active on r7, but no split
native phase has completed. Do not claim that the structural split has passed
Colab execution until Cells4–8 finish with their exact markers.

## Final validated status

[Run/Code/User decision/Open] 2026-10-17, final handoff at15:12 UTC. Successful
agent-created VM remains ready/idle: `tta-setup-a100-20261017-r2`, endpoint
`gpu-a100-hm-kkb-euw4a1-2xjsmbu8imvgc`. Actual GPU: A100-SXM4-80GB; host
MemTotal175,166,472kB. Python3.8.20, PyTorch2.1.2+cu121, environment-local
nvcc12.1.105 and GCC/G++11.4.0. System CUDA/driver was not replaced.

- Cell1/reconnection, Drive/KNN wheel, repaired native build, final ModelNet40-C
  cell5 and final cell6 including its embedded verifier completed on this VM.
  Cell2's known fastrlock pip failure was explicitly deferred; it is not
  relabeled as a successful original cell2 execution.
- Native checks passed: CUDA tensor sum, FPS, KNN, Chamfer identical-input and
  EMD finite-output; cell4 also passed gather forward/backward. Imports include
  PointNet++, CLIP, Open3D, diffusers, models.pvcnn2 and utils.eval_helper.
- ModelNet40-C clean array and15 severity5 arrays (2468 examples each) plus labels
  were checked. All24 recorded assets include SHA256 identities. Point-MAE is
  351,656,113 bytes; LION is1,508,051,672 bytes. See verification_final.json.
- User confirmed only `modelnet40_c.zip` is needed: it contains data_original.npy.
  The separate zero-byte Drive original ZIP was deleted on explicit request.
  Cell5 verifies CRC/paths and extracts only the dataset folder; a harmless
  __MACOSX member is excluded. The expired original-data signed URL is removed.
- ScanObjectNN restore exposed `Invalid ScanObjectNN5 Drive receipt`. It remains
  unaccepted; its existing dedicated research section is separate from base
  setup. No dataset was relabeled, regenerated or declared valid to bypass it.

### Delivered flow and execution evidence

Historical r2 validation used the then-current six-cell version. The canonical
new-runtime file is still `scripts/colab_environment_setup.ipynb`, but its later
2026-10-18 revision has ten cells as described above. The full research notebook
remains unchanged. Helper sources are
`scripts/build_cuda_extensions_colab.sh` and `scripts/verify_colab_environment.py`.
The notebook embeds both helpers so the original dev checkout requires no push
of these local files before setup. README, AGENTS and Colab skill route here.

Fixes verified: cu121 toolkit instead of system13.0; GCC11.4 instead of the
GCC12.2/pybind11 compile failure; preserve actual PointNet++ GPU architecture;
reuse the existing complete EMD kernel instead of the truncated extension copy;
stream/check subprocess output; use ModelNet40-C only; correct verifier sys.path.
A diagnostic controller initially mistook a marker echoed in a traceback for
success. Its raw evidence is preserved; final acceptance separately required
exact output lines, no traceback, JSON status=passed and valid remote evidence.
The skill now explicitly requires exact-line markers after ANSI stripping.

Local raw evidence: `result/preparation/cli_setup_20261017T130314Z_r2/`.
Canonical archive: `setup_attempt.zip`,461,401 bytes, SHA256
`0cc4a67a67ab1a53ef97dc89cc4abbbdd2bafa373d72060c4f295536deff798a`.
Drive: `MyDrive/thesis/result/preparation/cli_setup_20261017T130314Z_r2/setup_attempt.zip`.
Both copies match size/SHA; ZIP CRC passed on both. Adjacent archive_receipt.json
records verification. Local CLI logs originate on the Mac, so the complete
canonical archive was assembled locally and then uploaded/hash-verified on Drive.
It includes failed/partial attempts, exact phase snapshots, final source copies,
resolved environment and actual remote commit/patch. Remote evidence ZIP SHA256:
`51a1a15e4ff086aac92a6eb9822a827e6536a40a5e097413dabc5282c8e1cb43`.

Actual remote source is `dev@a458cd46bba1288a3f41302548bc777700cbc7d4`; local
research checkout is `gsd-smooth-spectrum@e955799de89476a5d58105c4f8622d0481bdecdc`.
These are recorded separately; this setup did not upload/validate a full new GSD
research checkout or execute a model accuracy experiment.

### Limits and next action

The final consolidated notebook was not replayed on a second clean VM; its
constituent corrected steps and the final embedded verifier passed on r2.
Only the known cell2 fastrlock failure is deferred. Existing pip conflict
warnings are preserved in logs; no claim of a globally conflict-free environment.
CLI0.7.4 cannot select runtime2026.07; the default-image A100 setup was approved.
Allocation began13:03:46 UTC; final checks/backup finished around15:12 UTC. This
wall time includes consent, package installation and debugging, not measured GPU
compute duration. Compute-unit cost was not measured.

Next: reuse this named VM while it remains available; inspect transferred
ScanObjectNN artifacts/receipt and align the intended research source before
that experiment. Ask approval only for GPU phases reasonably expected to exceed
60 minutes, treating the whole phase together. No further setup approval is
pending. User-operated Terminal contents and OAuth/token files were not read.
Historical entries below describe earlier states and are superseded here.

## Initial configuration snapshot (superseded by final status)

The user authorized CLI configuration, not a new VM or research run.
Installed executable: `/Users/batuhan/.local/bin/colab`, version0.7.4.
Use its existing OAuth2 provider; no gcloud installation or custom OAuth client
is needed because the package includes the client configuration.

Initial checks inspected file existence only: no cached Colab OAuth token,
local session registry or ADC credentials were present. No credential contents
were read. A Mac Terminal tab was opened with:

```bash
/Users/batuhan/.local/bin/colab --auth oauth2 whoami
```

The installed CLI displays a Google authorization URL, then waits for the user
to paste Google's authorization code into that Terminal. Keep this interaction
in Terminal; never send the code or token through chat or research records.
The CLI stores credentials in its own user configuration directory.

**Status: OAuth identity and backend listing verified.**
The user completed sign-in in Terminal. A separate
`colab --auth oauth2 whoami` returned success, with Colab and Drive-file scopes;
`colab --auth oauth2 sessions` returned one A100 High-RAM GPU assignment,
labelled `[?]` because it is not locally registered. No code was executed on it.
The user-facing Terminal window was not read or scraped. Only credential-file
existence was checked; no OAuth authorization code or credential-file contents
were read by the agent. The separate CLI identity output included account
email, OAuth client audience, expiry and scopes in conversation tool output;
it included no access token or refresh token. Account identifiers and OAuth
values are deliberately omitted from this repository note.

## Session boundary

The installed CLI can list backend assignments, but only sessions already
registered locally can be targeted by its normal execution commands. It has
no supported `attach`/`adopt` command for a browser-created session. Do not
manually edit the credential/session registry to work around this limitation.
An existing browser session must remain untouched. If a new VM is needed,
prepare the requested hardware/setup and obtain separate approval first.
The user can open a CLI-created runtime in a browser with `colab url -s NAME`.

## Requested runtime: 2026.07 / A100 / High-RAM

[User decision/Code/Open] The user requests runtime version2026.07, A100 and
High-RAM if version selection is supported. Installed CLI0.7.4 `new()` exposes
`--gpu A100` and `--high-mem`, but no runtime-image/version parameter;
`client.assign()` / `_build_assign_url()` likewise send only notebook identity,
variant, accelerator and shape. Therefore do not allocate a default-image VM
as a substitute for the requested version. No new session was created.
Google lists2026.07 as a past runtime, but browser availability does not imply
that this CLI supports selecting it. Sources inspected on2026-10-17:
[CLI session documentation](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md),
[past-runtime inventory](https://github.com/googlecolab/backend-info/blob/main/README.md).

Next action: present a concrete new-runtime/setup proposal for separate user
approval if CLI execution is needed. Existing A100 access, Drive mounting,
project dependencies and experiment execution remain unverified.

No VM allocation, Drive mount, dependency installation, remote execution or
GPU run is authorized or performed in this setup phase. Follow the
[operator skill](../.codex/skills/colab-research-operator/SKILL.md) for execution.

## Later: six-cell setup explicitly authorized

[User decision/Code/Open] The user now explicitly authorizes a fresh A100
High-RAM session and execution/verification of the first six preparation code
cells, reconnecting after cell1's Conda kernel restart. This supersedes the
earlier setup-only authorization and conditional2026.07 allocation request:
the latest instruction requests A100/High-RAM without an image pin. No model
evaluation is included; leave successful setup ready for later work.

Extracted exact code-cell sources from notebook indices1–6 into
`scripts/colab_environment_setup.ipynb`; full source notebook SHA-256:
`dd3146508be8a717d260418a312d00c127291672cb2dc1e20c40b642cd1b9315`.
Execution receipt/source snapshots:
`result/preparation/cli_setup_20261017T104109Z/`.
Planned command: `colab --auth oauth2 new -s tta-setup-a100-20261017 --gpu A100 --high-mem`.
Runtime cost is unknown; it consumes account compute units while allocated.
Stop on allocation or verified cell failure, preserve outputs, and never
substitute another GPU. Expected cell1 disconnect requires same-VM kernel
reconnection, not fresh VM allocation. Drive consent is user-interactive.
Current state: CLI reported `Session READY`; cell1 installed Conda and printed
`Restarting kernel`. Reconnection to the same session succeeded: cell2's
`condacolab.check()` passed, the repository cloned and checked out `dev`.
Cell2 failed in `conda env create -f env.yaml`: its pip phase downloaded
`fastrlock0.8.4` as source and could not build the wheel, ending with
`CondaEnvException: Pip failed`. The lower-level compiler message is absent
from captured conda output, so the root cause is not established. Cells3–6
and full environment verification were not run. Raw cell1/cell2 logs and
per-cell execution wrappers are in the receipt directory above. The wrappers
retain exact cell sources through IPython, set an explicit working directory
and raise on nonzero shell exit status to avoid false notebook success.

The CLI nevertheless exited0 for the remote Python error. Cell2's completion
marker is absent, confirming failure rather than setup completion.
A diagnostic script was prepared but did not execute: the CLI's initial
connection returned404/401 and it pruned the local session registration.
The creation event identifies our VM as
`gpu-a100-hm-kkb-ass1c2-jah5ciy1qndo` (allocated2026-10-17T10:42:38Z).
That exact endpoint is absent from the subsequent assignment list; the two
visible A100 assignments have different identities and were left untouched.
[User report] The user changed only their own browser sessions, not this VM.
No cause of the missing VM, token expiry or user-caused termination is claimed.

The exact six code-cell sources were saved unchanged in the standalone setup
notebook; source equality and empty historical outputs were checked. The
notebook's introduction states the failed validation status. Local failed
attempt archive:
`result/preparation/cli_setup_20261017T104109Z/setup_attempt.zip`,72,382 bytes,
SHA-256`2f512b3aaf492c305151fa4283e1f66ce3dbd416e49c6b0f9ac1968726e98c2d`;
ZIP CRC passes. No Drive backup was possible because cell3 was not reached.

Next gate: obtain approval for a fresh A100/High-RAM diagnostic/setup attempt;
capture a minimal Python3.8 `fastrlock0.8.4` wheel build with verbose compiler
output before proposing a dependency change. Preserve this failed attempt and
do not silently rerun the six cells or substitute a package version. No model
evaluation was performed and no six-cell success is claimed.

## Current revision: CUDA13 / PyTorch cu121 build mismatch

[User decision] Defer cell2 fastrlock diagnosis unless later GPU work fails.
This supersedes the preceding fastrlock-first proposal; CLIP installation
alone is not evidence that all missing dependencies have been repaired.

[User report/Code] Supplied cell4 traceback confirms nvcc13.0 versus PyTorch
CUDA12.1, a fatal major-version mismatch. PyTorch's
[compatibility check](https://github.com/pytorch/pytorch/blob/v2.1.2/torch/utils/cpp_extension.py#L365)
must remain enabled. Archived12.8/12.1 was a different, nonfatal warning.

[Code] Only cell4 changed in both preparation notebooks. It installs
`cuda-toolkit=12.1.1`, GCC/G++12.2.0 into `3dd_tta_env`, using NVIDIA's
[12.1.1 label](https://anaconda.org/nvidia/cuda-toolkit/manage) and conda-forge;
then runs the embedded `scripts/build_cuda_extensions_colab.sh`. This reuses
the earlier build-script pattern on `fix/pin-colab-environment`. Embedding
supports the existing dev checkout without a push. CLIP now precedes build_pkg.

The helper validates nvcc/torch CUDA12.1, uses environment-local headers/libs
and GCC, persists that environment's CUDA settings for later conda-run/JIT,
and targets the actual GPU. It repairs dev's forced architecture override,
clears generated extension build objects, rebuilds EMD/Chamfer/PointNet++ and
PVCNN, then checks imports plus CUDA FPS/gather forward/backward. System driver,
system CUDA and PyTorch compatibility checking are unchanged.

[Local/Open] Bash/Python3.8 syntax, notebook/helper identity, preservation of
all32 other original cells, first-six-cell equality, rejection of13.0 versus
acceptance of12.1, and dev/current setup A100-target preservation pass locally.
No new VM, native build or GPU execution occurred. Helper SHA-256:
`b5f8ec8f5bcb1ea383457c5e8168f2830c6e3639ac4c2853b3bd3dc389aba7b8`.
Earlier failed-attempt evidence remains unchanged.

[Next approval] Proposed retry: fresh default-image A100/High-RAM, cell1 then
same-VM reconnect, cells2–6 (only known cell2 fastrlock failure deferred),
interactive Drive mount, revised cell4 GPU checks, then data/checkpoint
verification. No model evaluation. Fresh UTC logs/source hashes and Drive/local
ZIP; runtime/CU cost unknown. Leave success ready; on failure retrieve evidence
and close only the new VM. Explicit allocation/retry approval remains pending.

[Later User decision/Code] The user explicitly answered "A100 kurulumunu
onaylıyorum" for that bounded retry. Prepared immutable notebook/helper and
per-cell snapshots at `result/preparation/cli_setup_20261017T130314Z_r2/`.
Session name: `tta-setup-a100-20261017-r2`. The local starter allocates once,
records the creation event, runs cell1, reconnects after restart, then runs
cell2; only the specifically known fastrlock pip failure may be deferred.
Drive authorization and cells3–6 follow separately. Allocation not yet started
at this entry; no second approval is required for these unchanged steps.

### Current approval rule and r2 progress

[User decision] Ask only for GPU runs reasonably expected to exceed 60 minutes.
Short GPU runs, CPU work, setup, allocation and retries within the requested
task need no additional research-run approval. Estimate the complete GPU phase;
do not split calls to evade the threshold. Shell sandbox escalation and the
user's interactive account consent remain separate. AGENTS and both research/
Colab skills now use this rule; earlier every-run instructions are superseded.

[Code/Open] r2 allocated at2026-10-17T13:03:46Z, endpoint
`gpu-a100-hm-kkb-euw4a1-2xjsmbu8imvgc`. Hardware preflight reports
NVIDIA A100-SXM4-80GB,81,920MiB, driver580.82.07; host MemTotal175,166,472kB.
Cell1 passed. The first cell2 invocation lost its connection in the CLI prelude
before its source executed; the VM remained present and registered. Explicit
`colab restart-kernel -s tta-setup-a100-20261017-r2` succeeded. The subsequent
cell2 invocation passed Conda check, clone/dev checkout and reached Conda
the known fastrlock pip failure, deferred under the user's instruction; no
duplicate VM was created. Drive mounting was opened in the user's Terminal;
interactive authorization is pending and Terminal contents are not read. Logs are
`cell_02.log` and `cell_02_reconnected.log` in the r2 attempt directory.

The build helper now supports both NVIDIA root and targets/ header/library
layouts for the same12.1.1 toolkit. Both embedded notebook copies are updated;
the attempt retains original and layout-revised cell4 snapshots separately.
Use `cell_04_layout_revision.py` for this attempt. No native rebuild has run yet.

[Run/Open, later update] User completed Terminal Drive authorization. Cell3
passed its completion marker, mounted-Drive check and KNN wheel check. Cell4
first lost connection in the CLI prelude before source execution; a same-VM
kernel restart succeeded. The retry is running and has installed the first
dependency group and KNN-CUDA. See `cell_03.log`, `cell_04.log` (prelude failure),
`restart_before_cell4.log` and `cell_04_reconnected.log`. Native compilation and
cells5–6 are not yet validated; no new VM or research evaluation was started.

### r2 native compiler diagnosis and GCC11 revision

[Run] Cell4 completed pip/CLIP and installed environment-local CUDA12.1.105.
A read-only probe confirms Python3.8, PyTorch2.1.2+cu121, A100 CUDA availability,
CC/CXX activation and header/runtime libraries in the environment root. The
initial native helper returned1 without subprocess stdout reaching CLI output.
A streamed retry (`native_build_logged.py`) installed EMD but failed compiling
Chamfer at torch/include/pybind11/cast.h:45 (`expected template-name`). See
`native_diagnosis.log` and `native_build_logged.log` under the r2 directory.

[Code/Inference/Open] Preserve CUDA12.1 and torch; change only the host GCC/G++
pins from12.2.0 to11.4.0 in both cell4 copies. A bounded same-VM compiler/native
retry is running via `cell_04_compiler_revision.py`, under the revised short-run
approval rule. The same CUDA12.1/GCC12 pybind11 failure and GCC11 workaround
are reported in the project's [upstream issue discussion](https://github.com/VAST-AI-Research/TripoSR/issues/3).
This is a working hypothesis until the actual compiler retry passes. Both
notebooks now stream subprocess stdout/stderr and preserve nonzero failure;
local success/nonzero-exit checks pass. Historical executed snapshots remain
unchanged in the attempt directory. Cells5–6 have not run; no model inference
or completed environment validation is claimed.

[Run/Code/Open, later r2 update] GCC11.4 resolves the Chamfer pybind11 compile
failure: EMD, Chamfer, PointNet++ and PVCNN built, and build_pkg.py imports
completed. The final helper import instead exposes undefined
`MatchCostBackward` in emd_cuda. Downloaded remote source proves the extension's
kernel defines only ApproxMatchForward; both cost functions are absent. The
existing third_party/PyTorchEMD kernel defines all three interfaces, matches
local bytes (SHA256 f15ce49d662ccf1d169127db9e905fa09ea47b29485b2a4ee7e8ad7452707dc3)
and compiled during build_pkg. The setup helper now reuses that complete kernel
only when cost definitions are absent; local repair/idempotence checks pass.
Both notebook payloads match the new helper (SHA256
8e0b58b812fd86fb3b95627dd2dd8f1556b0a779e83314060b2eb7cf040cbd30).
A bounded same-VM repair rebuilds only EMD, then executes the exact helper GPU
checks (`cell_04_emd_revision.py` / .log). Previously successful other builds
are retained; no model run, GPU-check success or cell5/6 completion is claimed.

## 2026-10-18 r3 fresh-VM replay of the single-PyTorch Cell4

[Code/Run] After two later setup VMs disappeared during the old Cell4, both
notebooks were synchronized to install the final pinned
torch2.1.2+cu121/torchvision0.16.2+cu121/torchaudio2.1.2+cu121 stack first,
then KNN, requirements and CLIP. This removes the observed unpinned2.4.1/CUDA
dependency installation followed by downgrade; it does not claim that the
duplication caused either server-side VM loss. The guard test is
`tests/test_colab_environment_setup.py`.

[Run] Fresh A100-SXM4-80GB/High-RAM session
`tta-scanobjectnn-s1-a100-r3-20261018` completed cells1–6 in order after the
required Cell1 restart/reconnect and user Drive authorization. Cell2 had only
the explicitly deferred fastrlock failure. Cell4 built EMD, Chamfer, PointNet++,
PVCNN and the complete EMD backend with environment CUDA12.1/GCC11.4, then
emitted `SETUP_WRAPPER_CELL_4_COMPLETED`. Cells5–6 emitted their completion
markers; final verification reported Python3.8.20, torch2.1.2+cu121,
CUDA12.1, A100, native FPS/KNN/Chamfer/EMD and verified ModelNet40-C/checkpoint
assets. This is the previously missing fresh-VM replay and supersedes the
unverified status of the single-install repair.

[Identity] Current notebook SHA-256 values after this repair are
`7878e5170344d745f366ef7a7e8c44c7366617118dd657fc1bf57dc89cf98c33`
for `scripts/colab_environment_setup.ipynb` and
`8473cc1684ddd69a8b889963607a26c5b4a9dbb41a10bf20ab6a5aa889a17213`
for `scripts/gsd_guidance_ablations_colab.ipynb`. The setup was immediately
used for the separately recorded ScanObjectNN severity1 diagnostic.

## 2026-10-18 r4 runtime loss during Cell4

[Run/Failed] A later fresh no-background diagnostic session,
`tta-scanobjectnn-nobg-a100-r4-20261018`, completed Cell3 and began the same
validated Cell4. It installed pinned PyTorch2.1.2+cu121 and KNN-CUDA, then
stopped producing output. Server inventory subsequently returned no active
sessions and the local record was pruned at13:43:49 UTC. There was no Cell4
completion marker. The orphaned local client was interrupted after server
absence was confirmed; it does not authorize or imply runtime teardown.

[Diagnosis] The prior successful r3 history spans about23m25s between Cell3
and Cell4 completion records. Therefore the 2400-second timeout is only a guard,
not an expectation that Cell4 should take40 minutes. Fresh Cell4 downloads the
complete dependency/toolchain stack and serially (`MAX_JOBS=1`) rebuilds EMD,
Chamfer, PointNet++ and PVCNN. Warm/partial retries can be materially faster.
The r4 VM nevertheless disappeared at roughly27 minutes without a server-side
reason, so its delay cannot be attributed solely to compilation. Recovery
requires a new runtime and the full six-cell sequence.

[Run/Failed, later r5] r5 completed Cell1 and Drive mount, then disappeared
during Cell2 without a completion marker. The server listed zero sessions while
the local client remained blocked for about2h24m;106.16 compute units remained.
Periodic second print cells are not a recovery mechanism for a busy kernel.
Future long phases should emit real timestamped phase/progress heartbeats from
the running process and use a separate CLI connection only for server inventory.

## 2026-10-18 r7 ten-cell replay and monitoring corrections

[Run] Session `tta-scanobjectnn-nobg-a100-r7-20261018` completed the first full
fresh replay of the ten-cell setup. Cells4–8 passed their exact wrapper and
on-disk native markers; Cell10's verifier passed CUDA12.1, A100, native
FPS/KNN/Chamfer/EMD, datasets and checkpoints. This supersedes the earlier
unvalidated status of the split setup notebook. r7 remains live pending explicit
task-complete and close authorization.

[Code/Diagnosis] CLI0.7.4 `colab url -s SESSION` generated a URL containing the
correct r7 A100 endpoint, but the user's browser showed CPU and the server
inventory gained an unrelated `[?]` CPU assignment. Therefore an endpoint in
the generated URL is only an attachment request, not proof that the frontend
honored it. `empty.ipynb` also does not show prior CLI cell output. Verify the
browser-reported hardware and server inventory; do not run browser cells in
parallel with a CLI-managed cell. The unrelated CPU assignments were not
modified.

[Code/Diagnosis] The native helper emitted two-minute heartbeats, but Cells5–8
used bare `subprocess.run`, whose inherited child file descriptors bypassed the
Jupyter IOPub/CLI stream. This explained why only Python start/end prints were
visible. The canonical notebook now reuses Cell4's `_tta_run_checked` helper to
pipe combined stdout/stderr through flushed Python prints. A RED/GREEN test
requires all four native phase cells to use that streaming path, and the Colab
operator skill records the same constraint. This improves observability but
does not prevent Colab-managed quota, lifetime or backend termination.

[Run/History] Local CLI history lists seven named sessions created on
2026-10-18 plus unregistered server assignments. Several terminations followed
27–144 minute gaps after the last completed logged action and are compatible
with inactivity; r6 disappeared during an uncompleted long Cell4, so inactivity
does not explain every loss. The backend exposed no definitive termination
reason. Keep primary `colab exec` output, exact completion markers, persistent
phase markers and server inventory as separate evidence rather than claiming a
single liveness signal.

[Run/Decision] 2026-10-30 depth-study r2 canonicalten-cellsetup/native/CPU checks supported a completed44-attempt research campaign. Solemonitor31219 observed streamedprogress and read-only genuinehealth, never queuedfakecellswhileGPUbusy. Final1,345,860,484-byte evidenceZIP was splitinto256MiBparts for CLIContentsAPI downloads on16GBMac; all7267member identities/CRC and422physicalcachefiles verify locally. Derivedanalysis/knowledge alsoDrive-roundtrips. Onlyowneddepth-tuningr2closed18:06UTC underexplicitcompletion-and-closeauthorization; freshinventorypreservedotheragentguidance-variantsr1endpoint. See[accepteddepthresults](chamfer_depth_tuning_results_20261030.md) andclosure receipts. No claimthatmonitorpreventsbackendquota/lifetimeloss; originalr1loss remainsunexplained.
