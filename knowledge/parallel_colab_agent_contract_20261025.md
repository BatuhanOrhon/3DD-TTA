# Parallel agent isolation contract — 2026-10-25

[User report/Decision] Two future agents may run independently. Each must
derive its own branch from the committed `gsd-multibackbone-study` base, use a
separate worktree and an exclusively owned Colab session. This document plans
ownership; no worktree, session or background process is created here.

| Assignment | Branch | Worktree | Colab session prefix |
|---|---|---|---|
| A: missing ModelNet trajectory readouts |`gsd-modelnet-trajectory`|`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/modelnet-trajectory`|`tta-mn-trajectory-a100-<UTCdate>-r<N>`|
| B: guidance component ablations/development |`gsd-guidance-components`|`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components`|`tta-guidance-components-a100-<UTCdate>-r<N>`|

## Git, files and evidence

1. Inspect the original workspace's Git status and worktree list. Record the
   full committed base hash, then create the assigned branch/worktree from
   `gsd-multibackbone-study`. Do not reset/stash/clean or commit the user's
   existing edits. If the assigned name/path exists, inspect ownership first;
   reuse only the correct task checkout or choose a new suffix without overwrite.
2. Work and commit only in the assigned worktree; push only that task branch.
   Do not switch the original workspace's branch, modify the peer worktree,
   merge into the base branch, or cherry-pick unreviewed peer changes mid-run.
3. The original workspace has uncommitted Colab skill/setup/monitor changes.
   Read original `AGENTS.md`, `/Users/batuhan/.codex/RTK.md` and current
   `/Users/batuhan/Desktop/projects/3DD-TTA/.codex/skills/colab-research-operator/SKILL.md`
   before operating. A new worktree's committed skill may be older. Current
   monitor scripts under the original skill's `scripts/` may be read/invoked
   by absolute path, with identity recorded; never move, overwrite or commit
   those user files as a side effect of preparing the task worktree.
4. Previously accepted raw results/assets in the original workspace are
   read-only references. Do not move them, reuse output directories for new
   attempts, change their permissions, or overwrite manifests. Download/copy
   needed inputs into task-local storage with SHA verification. Only immutable
   accepted peer bundles may be shared as input evidence; no live state/cache.

Separate preparation and Drive roots:

| Owner | Local preparation root inside its worktree | Drive study root |
|---|---|---|
| A |`result/preparation/modelnet_trajectory_study_20261025/`|`MyDrive/thesis/result/modelnet_trajectory_study_20261025/`|
| B |`result/preparation/guidance_component_study_20261025/`|`MyDrive/thesis/result/guidance_component_study_20261025/`|

Use fresh UTC run IDs containing `mntraj` for A and `guidance-components` plus
arm/config for B; preserve canonical dataset/method result schemas. In each
assigned VM use a distinct study root, e.g. `/content/studies/mn-trajectory/`
or `/content/studies/guidance-components/`. Remote source snapshots, manifests,
logs and receipt paths are task-specific even if the underlying code is shared.

## Colab ownership and lifecycle

- Read the server inventory without mutating peer assignments. Reuse only an
  explicitly verified runtime owned by this task; otherwise allocate a new
  task-named session. Never attach/exec/install/upload/mount/stop/release on
  the other agent's runtime or an unowned `[?]` assignment.
- Use an explicit `-s <own-session>` for applicable CLI operations. Record
  actual session, endpoint, hardware, UTC start, source/asset hashes, monitor
  PID/local exec-session ID and running phase in the task's own
  `runtime_owner.json`. A prefix alone is not proof of endpoint ownership.
- One monitor is the sole execution owner of each runtime; submit phases
  through its documented control protocol. Do not send concurrent notebook
  cells or start a second monitor for the same runtime. Record status changes
  and preserve completed artifacts on interruption.
- No global `stop all`, broad `pkill`/`killall`, shared-environment rebuilds,
  global authentication reset or peer monitor cleanup. If GPU capacity/quota
  is unavailable, report the limitation; do not evict the peer or silently
  substitute different hardware. Coordinate any required account change.
- Estimate each whole proposed GPU phase, including its pilot/full work and
  retries; obtain one comprehensive approval before phases expected to exceed
  60min. Separate CLI calls do not reset the budget or approval scope.
- Never terminate the peer's runtime. Close even your own runtime only when
  the user explicitly confirms task completion and authorizes that closure.
  Authorization for a past session/task is not global teardown permission.

## Cross-task code and numerical controls

[Code/Decision] Both tasks may need ModelNet40-class observer/analyzer support.
They may develop independently in their branches; preserve Scan15-class and
default terminal behavior. Reuse a peer implementation only through an explicit,
reviewed commit/snapshot, with tests and executed hash recorded before a new run.
Do not wait on or mutate the peer's live VM to obtain code or intermediate state.

[Decision] Agent A preserves the locked combined guidance policy. Agent B owns
all guidance-component changes and exploratory tuning. Neither agent changes
the other's defaults, run settings, knowledge files or pending approval record.
Both update their own branch's study note/findings/open questions and push
accepted batches; integration of diverged knowledge files happens after review.

Related scopes: [A handoff](modelnet_multibackbone_trajectory_handoff_20261025.md),
[B study](guidance_component_study_handoff_20261025.md),
[copyable prompts](parallel_agent_prompts_20261025.md).
