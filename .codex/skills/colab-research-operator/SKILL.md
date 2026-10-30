---
name: colab-research-operator
description: Use when preparing, approving, executing, monitoring, or archiving this project's Google Colab work through the Colab CLI, including dataset generation, source-only evaluation, CUDA smoke runs, and research knowledge updates.
---

# Colab Research Operator

Use the installed `colab` CLI for remote execution and artifact retrieval.
Keep research continuity with `.agents/skills/iterative-thesis-researcher/SKILL.md`.
Resolve paths below relative to the repository root. Follow
`knowledge/experiment_protocol.md` and `result/README.md` for artifact schemas.

## Prepare without starting a run

Read `knowledge/README.md`, the latest relevant dated handoff, findings and open
questions; inspect actual branch, HEAD, status and files. A historical claim that
a script or ZIP exists does not establish its presence in this checkout.
Inspect existing result coverage before proposing another evaluation. When
artifacts are on another machine or awaiting Drive transfer, record that fact;
do not reconstruct results from prose or repeat the experiment to replace them.

Discover the installed interface with `colab --help`, `colab version`,
`colab skill` and the relevant subcommand's `--help`. Installed help and code
take precedence over stale bundled examples. In the inspected 0.7.4 install,
top-level help/code default to OAuth2 while bundled skill prose claims ADC.
Use the user's configured authentication; do not change accounts or install
another CLI. Never read or print OAuth/token/session-state file contents.
If authentication needs browser interaction, ask the user to complete it.

CLI help/version can write local logs and encounter sandbox restrictions.
Request the required tool escalation; do not redirect credentials or bypass
permissions. Shell approval for a CLI command is separate from research-run
approval. Do not request a reusable unrestricted `colab exec`/`run` prefix.

## Approval for GPU runs longer than one hour

**Obtain explicit user approval only for GPU runs reasonably expected to take
more than 60 minutes.** This user decision supersedes approval for every run.
Within the requested task, proceed with shorter GPU runs, CPU work, VM
allocation, environment setup, retries, inspection and artifact retrieval
without an additional research-run approval. Use existing logs or small pilots
to estimate uncertain durations; request approval when the GPU phase is expected
to exceed one hour. Assess the whole planned GPU phase, not each CLI call;
do not split a long run into short commands to avoid approval.

The approval request must identify:

- Question/hypothesis and stage; exact script/commands or selected notebook cells.
- Source commit plus snapshot/dirty-file hashes; dataset, split, severity,
  corruptions, indices/counts, method, seeds, batch size and checkpoint identities.
- Existing or new session, requested accelerator, environment, runtime/cost
  estimate (or explicitly unknown), and stopping/failure conditions.
- Fresh output/Drive destinations, required archive contents and recovery plan.

For a run over one hour, record the user's reply and approved scope before
launch. Do not ask again for unchanged execution, monitoring or downloads.
Reassess duration after material changes or failures. A distinct GPU run over
one hour needs approval; a short retry or smoke-to-pilot transition does not
need approval solely because it is a new invocation. Record estimates and
actual duration, and obtain approval if continuing work is expected to make
the GPU phase longer than one hour.

Local preparation and CLI help need no run approval. Inspecting session status,
existing logs/files and downloading available artifacts also need no new run
approval. Remote inspection code must be bounded to reading state and must not
invoke experiment entrypoints. Uploads, installs, Drive mounting and environment
changes may proceed within the user's setup task; interactive account consent
remains user-operated. Runtime teardown follows the separate lifecycle contract
below; creating a VM does not authorize its later cleanup.

## Task-specific session names and agent ownership

Before allocating a runtime, choose a unique session name that identifies the
agent's actual task: `tta-<task>-<accelerator>-<UTCdate>-r<N>`. For example,
`tta-scan-dgcnn-checkpoint-a100-20261030-r1`. Use distinct task slugs for parallel
agents; generic names such as `test`, `research` or `colab` are insufficient.
Check the server inventory for name collisions without executing cells on other
sessions. Record the session name, endpoint, task/agent owner, branch/worktree,
monitor/process identity and output/Drive destinations in the task's knowledge
note. Always target the recorded name explicitly with `-s SESSION`.

Another agent's runtime is protected: do not stop, release, delete, reconnect for
execution, interrupt its kernel, or stop its monitor/background processes. Do
not clean up all visible sessions or free GPU quota by terminating peer work.
An idle indicator, `[?]`, an unfamiliar name or missing local bookkeeping does
not establish ownership or authorize cleanup; preserve unknown sessions.
Historical or broad cleanup instructions do not authorize closing another
agent's current task. If the user explicitly requests intervention in a peer
runtime, first identify the exact session and coordinate ownership handoff with
that agent; teardown still requires the completion-and-close authorization below.
Read-only server inventory does not transfer ownership. Reusing a prepared
runtime requires a recorded assignment to the current task, not just visibility
in the inventory. If no runtime is assigned, allocate a distinctly named one.

## Runtime lifecycle and teardown

Treat every prepared Colab runtime as expensive, reusable task state. Do not
stop, release, delete, or otherwise terminate a runtime until the user explicitly
states both that the current task is complete and that the runtime may be closed.
This applies to agent-created and shared runtimes alike.

Successful setup, completed experiments, artifact retrieval, an idle runtime, cost concerns, or the end of the current turn are not teardown authorization.
A request to pause work is not permission to close the runtime. If the user
states only that the task is complete or only that a runtime may be closed,
preserve it until both conditions are explicit. Do not infer permission from a
cleanup step in an earlier plan. Before yielding after Colab work, report the
session name and whether it remains live; unexpected runtime loss is a failure
to record, not a reason to claim cleanup.

## Idle-runtime reminder

While a Colab runtime is live, track whether any setup, experiment, transfer,
or inspection command is still running. If the named runtime has had no active
work for roughly 10 minutes and no immediate next command is being prepared,
remind the user that the named runtime is idle and may continue to incur cost,
then ask whether they want it closed. Ask for both task-completion and close permission in the same prompt so that an affirmative reply can satisfy the
teardown contract unambiguously.

Never close it merely because the reminder was sent or went unanswered. Send at most one reminder per continuous idle period; reset the reminder only after
substantive runtime work resumes. Waiting for Drive authorization, a running
remote process, artifact transfer, or a planned near-term follow-up is not idle.
If idle duration cannot be established, report the observed state without
inventing a timestamp and preserve the runtime.

## Connection resilience and progress preservation

Colab-managed maximum lifetime, quota, or compute-unit termination cannot be prevented
or treated as a guaranteed connection. Keep legitimate setup, experiment, transfer,
or inspection work observable with bounded monitoring. Do not use fake clicks, JavaScript keep-alive loops, empty busy loops, or similar anti-idle workarounds.

Do not submit a second notebook cell merely to print while the kernel is busy;
it will normally queue behind the running cell and is not a keep-alive. Instead,
make long-running scripts emit periodic meaningful progress heartbeats from the running phase itself with a UTC timestamp, current phase, and last confirmed unit of progress.
A heartbeat improves observability; it does not guarantee runtime retention.
For notebook-launched subprocesses, forward child stdout and stderr into the notebook's IOPub stream and flush each line so the primary `colab exec` receives those heartbeats. Bare `subprocess.run` output may bypass the CLI stream even when the child prints correctly; verify visibility during a timed phase before relying on it.
When the kernel is busy or silent, query the server session inventory through a separate CLI connection without launching another notebook cell.

In the inspected CLI0.7.4, `colab status` reports CLI-local bookkeeping from
the session-state record; it is not an independent remote-kernel probe and can
briefly disagree across separate CLI processes. Do not infer remote execution or completion from one `status` snapshot or silence in the streamed output.
Monitor one cell through the same primary `colab exec` process and retain its
raw output. Treat a setup phase as successful only after the exact wrapper completion marker and, for split native phases, the phase's on-disk `.complete` marker are both confirmed. Use `colab sessions` only to confirm that the server assignment still exists; it does not report kernel busy state or cell completion.

`colab url -s SESSION` requests that the browser attach to an existing CLI-created runtime.
The generated `empty.ipynb` page does not display prior CLI cell output or
automatically open a remote notebook file. The URL containing the expected
endpoint is not proof that the browser attached to the requested endpoint.
Confirm the browser-reported hardware and compare the server inventory before
trusting it; an unexpected CPU assignment means the attachment failed or fell
back to another runtime. Do not run browser cells concurrently with a CLI-managed cell;
use the primary CLI process and remote completion markers as execution evidence.

A local CLI or WebSocket disconnect is not evidence that the remote kernel or process stopped.
Before retrying, query the server inventory, reconnect to the same named session,
and inspect remote processes, completion markers, and artifacts. Never launch a
duplicate job while the prior state is unknown. If the server no longer lists the
runtime, record the unexpected loss and preserve the last available local evidence.

Persist checkpoints, manifests, logs, and intermediate results to Drive after each
material phase, and design long phases to be restartable when practical. Verify each
copy before relying on it for recovery. A mobile device or browser tab does not need to remain open after interactive authorization for CLI-managed execution; tell the
user when a new consent prompt is actually waiting instead of asking them to keep a
screen awake.

## Execute the planned phase

For a new runtime, use the ten preparation code cells in
`scripts/colab_environment_setup.ipynb`. Execute cell1 alone, reconnect to the
same VM after its Conda kernel restart, and continue with cells2–10 in order.
Restore `/content/3DD-TTA` as the working directory before cells4–10 because CLI
calls begin in `/content`. Complete interactive Drive authorization through the
user's Terminal, and verify the mounted path before cell3. Preserve the exact
saved cells; record any execution adaptation and stop on failed shell commands
even if IPython displays a final success message. Verify real environment,
native CUDA imports, datasets and weights before declaring setup complete.
Cell4 prepares pinned packages, CUDA12.1/GCC11.4 and native-source patches.
Cells5–8 run the EMD, Chamfer, PointNet++ and PVCNN/smoke phases separately;
require each phase's exact completion marker before advancing, and reuse its
on-disk marker rather than rebuilding a completed phase. Cell9 uses only
`modelnet40_c.zip`, including its clean `data_original.npy`. Cell10 runs the
embedded `scripts/verify_colab_environment.py` checks. The
original notebook's separate ScanObjectNN section handles that dataset;
an invalid receipt is not evidence of a valid restored dataset.
Read `knowledge/colab_cli_setup_20261017.md` for known setup failures before
launch. CLI `exec` can return exit0 for remote Python errors; require the
per-cell completion marker and inspect remote errors as well as CLI exit status.
Match the marker as an exact output line after stripping ANSI formatting;
marker text echoed inside a traceback or source listing is not completion.

Prefer the prepared session, always with explicit `-s SESSION`. Verify it is
the intended VM and that no conflicting work is active. Use absolute remote
paths and the project's existing conda environment, not the kernel's incidental
Python. Confirm Drive, disk space, source, data/checkpoint hashes and native
dependencies before computation; stop on a mismatch.

Use a reviewed local Python phase script with
`colab exec -s SESSION -f PHASE.py --timeout SECONDS`, choosing the timeout for
the approved phase. Kernel variables persist between calls; make required state
explicit in files/manifests. `exec` begins in `/content`; do not rely on a prior
working directory. Preserve exit codes when launching subprocesses.

**Never execute the entire project notebook or use Run all.** CLI execution of
an `.ipynb` runs every code cell. Extract only the approved cells into a reviewed
phase, preserving notebook magic semantics (e.g. execute `%%bash` bodies with
Bash) and record their indices/source hashes. Preserve the original notebook.
`colab run` creates a fresh VM and normally releases it afterward; use it only
when provisioning and artifact persistence are part of the requested task,
with approval if the planned GPU run exceeds one hour.
Do not automatically choose another GPU on allocation failure.

Monitor status/logs with bounded calls and keep the user informed. A local
timeout/disconnect is not proof that the remote process stopped. Inspect the
same session/process/run files first; reconnect to an active job. If state is
unknown, preserve it and pause launch decisions. Never start a duplicate job.
On failure, retain partial outputs, traceback, exit/status and completed counts;
review them before proposing another attempt; seek approval if its GPU phase
is expected to exceed one hour. Do not reset, overwrite,
silently resume, or automatically rerun a failed directory.

## Archive evidence before interpreting it

Use the existing runner's new UTC run ID and immutable `result/` hierarchy.
For inference retain the seven required files: `command.txt`, `config.json`,
`environment.txt`, `stdout.log`, `summary.csv`, `per_corruption.csv`, `notes.md`,
plus method-specific predictions/manifests/diagnostics. Dataset preparation
uses its declared dataset/audit/receipt schema; do not fabricate accuracy CSVs.

Record actual remote commit and dirty/untracked source identities separately
from local HEAD; include the exact executed phase, CLI version, session/hardware,
environment/native extensions, full resolved configuration, RNG settings,
data/checkpoint hashes and selected original IDs. A commit alone cannot identify
working-tree overrides. Archive the executed source snapshot, or a pinned base
plus reconstructible patch and relevant untracked file contents; hashes alone
cannot recover source bytes. Keep credentials out of commands and captured output.

Package successful, partial and failed attempts. Validate ZIP paths, symlinks,
CRC and reasonable sizes before safe extraction. Back up to a fresh approved
Drive path, verify size/SHA-256 there, and download through `colab download` to
a new local destination; verify local SHA matches. Preserve raw ZIP bytes and
put derived analysis separately. CLI history exported with
`colab log -s SESSION -o FILE.jsonl` can supplement evidence; it does not
replace runner logs. Review history for credentials before sharing it.

If Drive backup fails, salvage available artifacts to a fresh local destination
immediately, verify their hashes, and record Drive backup as pending. Preserve
the runtime until evidence is persisted and its cleanup authorization is resolved.

Validate required files, matching run IDs, command/config/log agreement,
hashes, counts/indices, CSV units and completion status. Missing evidence stays
incomplete; request only what cannot be retrieved through CLI/Drive. Do not ask
the user to manually ZIP/download files that the agent can retrieve. Finalize
artifact retrieval before user-authorized VM cleanup; otherwise preserve the
runtime for follow-up tests.

## Update knowledge for every material phase

Append a dated entry to `knowledge/findings_log.md` and update the existing
experiment note. Use this compact record, including pending/failed outcomes:

```text
Date/time (UTC; local date if useful), evidence labels:
Question/stage; duration estimate; approval reference if GPU time exceeds1h:
Run/session ID; source commit + snapshot/dirty hashes:
Dataset/checkpoint identity; config/seeds/coverage:
Status (planned / awaiting approval / running / complete / partial / failed):
Raw local result path, Drive path, archive SHA-256 (pending if unavailable):
Validation, metrics with units, runtime/memory, limitations:
Decision; next gated action; evidence that would falsify the interpretation:
```

Use `[Code]`, `[Run]`, `[User report]`, `[Inference]`, `[Open]` and `[Paper]`
according to `knowledge/README.md`. A planned command is not a run; process exit
is not validated completion; a complete subset is not full dataset coverage.
Recompute metrics from validated artifacts; distinguish fractions, percentages,
percentage points, macro14/all15, same-trace and separate-run comparisons.

Update `knowledge/open_questions.md` gates and the relevant method/reproduction
note. Update the headline in `knowledge/README.md` when the current task,
blocker or reading order changes. Preserve superseded dated history with a
pointer to newer evidence. Before yielding, leave a precise next action and
approval state so another session can continue without chat history.
