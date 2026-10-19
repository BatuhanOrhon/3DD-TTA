# r7 interruption recovery through fresh r8 — 2026-10-19

**Current09:39UTC [Run/Code/Open]:** TTY44538 awaits user-requested renewed consent.
The previous timely confirmation failed in the CLI auth hook with a hidden
ConnectionResetError; extending the auth window alone did not fix it.
The new scoped helper clears its HTTP pool before the final propagation POST,
uses45s HTTP deadlines, and surfaces exception types. Scope/header checks pass;
actual mount remains unverified. Monitor63286/69569 stays suspended until this
external mount completes. No r7 data recovered or new researchGPU inference.
Earlier pending request IDs below are historical and superseded by44538.

[User decision/Run/Open] User explicitly authorizes creating a fresh Colab CLI
session and mounting Drive to retrieve the interrupted study's evidence. This
is the selected route; browser Drive access was not retried. The user's UI
shows no Drive block, while the browser tool reported a saved block; the
origin of that disagreement remains unknown. Manual user downloads are not
required while the authorized CLI route is available.

Current session `tta-multibackbone-recovery-a100-r8-20261019`, endpoint
`gpu-a100-hm-kkb-ass1c0-xawnnd6jnsdw`, A100 High-RAM, created08:15UTC.
CLI0.7.4 uses existing OAuth2 credentials, no account changes. Usage at
allocation481.18 compute units,6.77CU/hour. Sole r8 monitorPID63286,
local exec-session69569. Old r7 monitorPID34633/session97631 remainsLOST;
no runtime stop/release was requested for either session.

Setup input `scripts/colab_environment_setup.ipynb`, SHA256
`f60db956118c25196f43ae02a00667485ddffbc73691e601134cb194f885559b`; exact10 cell sources and wrappers are retained
under `result/preparation/cli_multibackbone_recovery_20261019_r8/`.
Cell1 passed its exact wrapper marker and same-endpoint reconnection after
Conda restart. Cell2 completed at the sole user-deferred `fastrlock0.8.4`
wheel/Pip failure. No other setup exception is accepted. Source clone follows
the setup notebook's `dev` checkout; it must not be equated to local research
HEAD`bf135e9`. Actual remote source identity is still to be recorded.

At08:24UTC official CLI `drivemount` requested interactive Google consent;
its local TTY exec-session80871 waits for the user to finish, then Enter.
No OAuth URL, transient state or credential is stored in this research note.
No Drive file has yet been retrieved and no new research GPU run launched.

Next steps: finish Drive consent; verify Cell3 and mounted expected research
path; run CPU-only inventory and scoped immutable recovery before Cells4–10.
Do not modify `20261019-070402_dgcnn-seed0` on Drive. Verify receipt-bound
batch bytes and corruption ZIP manifests, copy loss evidence into the new
recovery archive, back it up to a fresh Drive preparation path, then download
and independently verify local SHA/CRC/all members. Only then reassess a
fresh remaining DGCNN GPU phase against the existing4h cap including lost
work. Real RNG state was not saved; do not claim exact mid-run resume.

Prepared CPU-only phases:
- `inspect_drive_recovery.py`: SHA256
  `76439c4bb233f2a1b1bc26b6cc46be51bc1bd6d0670c27f889e514a1fa727bf0`.
- `recover_r7_drive.py`:6146bytes SHA256
  `9ecddac23f968213aa11cf7a05475e24474ed57c11d65cebd9815ec079180986`.
  Uploaded to `/content/recover_r7_drive.py`; loss ZIP62808bytes also uploaded.
These are prepared commands, not completed recovery evidence.

[Accepted comparisons and last r7 evidence](gsd_multibackbone_method_comparison_20261019.md).

## Partial setup evidence and interactive handoff — 08:33 UTC

[Run/Open] Local setup evidence archive50859bytes SHA256
`9b803b9f53078050ee1d9bcd6655ba2df285ab8afdf85b9aa82536f92cf38f35`,
`result/preparation/cli_multibackbone_recovery_20261019_r8/r8_setup_cells12_evidence_v1.zip`.
CRC/27safe members/all26member identities verify; includes exact ten saved
cells/wrappers, notebook, receipt manifest, public Cell1/2 execution events and
prepared recovery code. Google consent URL/transient OAuth state are excluded.
Drive backup remains pending consent; Cells3–10 and recovery are not complete.

The monitor knows only its own exec commands; a separate interactive
`drivemount` is not tracked as active work. Before its idle probe threshold,
local r8 monitorPID63286 was temporarily suspended withSIGSTOP at08:32UTC to
prevent a second notebook cell queueing behind Drive consent. It was not
terminated, and no runtime lifecycle command was issued. Server inventory
remains observable via CLI without notebook cells. After CLI80871 finishes,
resume this exact local PID withSIGCONT, retain exec-session69569 and submit
Cell3 through its control protocol. Do not start a second r8 monitor or cell
while the interactive mount is pending. r8 remains allocated, waiting for
Google consent; this is not an idle-runtime closure authorization.

## Drive mount timeout diagnosis — 08:52 UTC

[Code/Run/Open] Installed CLI automation uses600s, but installed Colab VM
`drive.mount` has an independent120000ms `request_auth` timeout. First mount
80871 reached local reply timeout after user confirmation. Second40124
returned remoteValueError/mountfailed (CLIexit0 is not success). Sanitized
DriveFS diagnostics show account-context auth failure08:38:49UTC before user
confirmation around08:41UTC; this is consistent with the120s remote deadline.
Physical Drive directory remained absent. No data was deleted or replaced.
Two CLI consoles inspected only mounts/process names/redacted error categories,
then detached; no notebook research cell or credential file was read.

A scoped local wrapper calls the installed CLI's existing auth flow, overriding
only this exact runtime/path to `drive.mount(...,timeout_ms=900000)` and local
execution1200s. Scope self-test passes and CLI public automation history proves
the third actual remote invocation contains900000ms. Installed CLI, OAuth
account and project notebook sources remain unchanged. Current TTY83156 is
waiting for the newest Google consent; older URLs do not complete this request.
No consent URL/transient OAuth state is archived.

Immutable local diagnosis ZIP12956bytes SHA256
`1fb90f4db8a829637349b291de0134a49c919fa12055cbfcc6e29fd75b42f070`,
`result/preparation/cli_multibackbone_recovery_20261019_r8/r8_drive_timeout_evidence_v1.zip`;
CRC/10members/all9member identities pass. Drive backup pending. Actual original
local traceback is unavailable beyond tool output; remote mount error/public
commands and installed module source are preserved. The timeout change is not
claimed successful until the expected Drive path and Cell3 receipts verify.

An unchanged full-v7 asset payload is prepared:247002524bytes SHA256
`ce9c797ccacb4eeffcc0a2836c65109213c9c7920a40a5145ef598b2adac1333`,
`result/preparation/cli_multibackbone_recovery_20261019_r8/r8_assets_v7_full.zip`.
All622 declared file identities plus the exact manifest verify;623ZIPmembers
and CRC pass. All8upload parts reached r8; assembly/deployment/CPUpreflight
remain unexecuted. Asset manifest78cdb538…60f9 and source snapshotf82cf505…deed4cf
are unchanged. Prepared `deploy_full_v7_r8.py` reuses the prior CPU-preflight
flow with fresh full-asset restore and pinned native archive verification.

## 09:05 UTC — Fresh30-minute auth request; Drive remains unverified

[Run/Code/Open] User confirmation for third request arrived09:01:55UTC,
after its15-minute remote deadline around08:59; it returned mountfailed.
The assistant also identified a transcription error in one parameter of its
last displayed consent URL; no causal claim that this alone explains failure.
New URLs must be copied exactly from CLI output, with no manual scope edits.

A new scoped wrapper retains the installed auth flow, remote1800000ms /
local2100s. Scope self-test passes; source SHA256
`605fc98dea93f0016971a976402571abd7c7f459f440c1cf2f6bf383c218c4a2`,
`result/preparation/cli_multibackbone_recovery_20261019_r8/mount_drive_with_human_timeout_v2.py`.
Its first invocation97774 lost the local WebSocket before presenting a URL.
Server/file service remained available. Read-only kernel API audit showed
setup kernel idle and only that abandoned auth kernel busy. A guarded kernel
interrupt targeted only44896965-5879-40e6-9b5e-e277254b60f2; subsequent audit
showed all kernels idle. No runtime restart/shutdown or setup-kernel change.
Current fresh request **TTY77142**, started09:05UTC, awaits Google consent;
remote deadline approximately09:35/local09:40. Resume the same suspended
monitor63286/69569 after mount; then Cell3 and CPU recovery. No Drive metric,
file recovery or researchGPU inference is accepted. Earlier pendingTTY IDs
in this note are superseded by77142. OAuth URLs/state are not archived.


## 09:21 UTC — Timely consent exposed hidden connection reset

[User report/Run/Code/Open] User approved the09:05 request within its30-minute
window. At09:09:56.523UTC the final credential propagation raised
ConnectionResetError / Connection reset by peer. Installed CLI runtime catches
auth-hook exceptions at DEBUG level, leaving the screen at “Authorizing VM”
and the remote mount waiting. Whitelist extraction from the diagnostic log
confirms those constants; no credentials, consent URL or raw auth state is
included. Thus the previous timeout diagnosis explains expired attempts but
is insufficient for this timely attempt. A stale persistent HTTP connection
is a hypothesis; the server-side reason for reset is not established.

CPU-only same-request rescue used a fresh challenge; dry-run successfalse,
so it sent no stdin reply and ran no cell/mount. Guarded interrupt targeted
onlyed674de6-1858-4042-936e-cf54c2f3bbb3; setup kernel and runtime preserved,
all kernels idle before retry. No account/scope/installed-package change.

`mount_drive_fresh_connection.py` retains the installed official consent flow,
remote1800s/local2100s, clears only its existing AuthorizedSession HTTP pool
before final propagation, and bounds HTTP requests45s. Scope/header-preservation
assert checks pass; SHA256839737e3a48cc3e90f6b67f4e831270dbb51b46171fc706379ff7167b27c0fb2.
CurrentTTY87017 started09:20UTC; fresh consent expires approximately09:50UTC
(12:50 Istanbul). The exact CLI URL was sent directly to the user, not archived.
Do not send Enter until the user confirms this request; do not start a second
mount or cell. Expected Drive path, Cell3, recovery and Cells4–10 remain gates.

Immutable local diagnosis ZIP12625bytes/9members SHA256
`a0a3fff927224cf1fac7a4a02ad3b3011b3cbaedf325ba0f71fa8c593601ae38`,
`result/preparation/cli_multibackbone_recovery_20261019_r8/r8_drive_connection_reset_evidence_v1.zip`;
CRC/all8manifest identities pass, Drive backup pending. Includes exact scoped
helpers, safe request IDs, sanitized diagnosis and installed public CLI sources.
No model evaluation rerun; accepted accuracy results remain unchanged.


## 09:32 UTC — User-requested consent renewal

[User decision/Run/Open] User requests a newly generated link. Interrupted only
old waiting-auth kernel9452e052, verified all kernels idle, then terminated
its local blocked CLI66108 (not the VM). Fresh unchanged helper now waits in
TTY53559, remote30min/local35min, approximate consent deadline10:02UTC/13:02
Istanbul. CLI returned a new URL; URL/state are not archived. Resume the same
suspended63286/69569 after mount. r8 endpoint remains responsive; setup retained,
no researchGPU run or artifact recovery. Earlier pendingTTY87017 superseded.


## 09:39 UTC — Latest user-requested link renewal

[User decision/Run/Open] New CLI-generated consent inTTY44538; previous53559
auth kernel1420426e interrupted after exact identity audit, all kernels idle
before retry; only blocked local CLI66639 terminated. Same r8/setup preserved.
Remote deadline approximately10:09UTC/13:09 Istanbul. Current URL/state excluded.
No new research/artifact evidence; resume63286/69569 only after mount finishes.
