# Agent A selective integration — 2026-10-30

[User report/Code/Run/Decision] Target is only `origin/thesis-integration`.
Source: `gsd-modelnet-trajectory@11a89c63be80e3f58940cbf9a55643f6520f1b37`,
initially clean; implementation665ed07, accepted results9d0d170. Prepared in
`integrate/agent-a-modelnet-20261030` from shared2778a66, preserving the source
worktree, peer worktrees and root's uncommitted Colab changes.

## Selected and already integrated work

Shared Agent B code already implements ModelNet40 observation, dataset-aware
strict analysis and runner class wiring. Retain its newer component/selection/
seed controls and automatic dataset detection. No production file, dependency
or configuration needs replacement. Agent A's historical `--dataset` analyzer
option and naming-only `mntraj-` prefix are unnecessary on this shared tree.

Add five missing behavior tests for40-class RNG/buffer/gradient preservation,
wrong/nonfinite logits, three frozen head configurations, ModelNet coordinate
normalization and tampered schema/labels/terminal/IDs. Extend existing full-set
coverage with the last18-example batch assertion. Existing Agent B tests remain.

Reconcile the accepted9-row study and relevant README/findings/backlog/method/
scope notes without replacing peer findings. Preserve135 corruption rows,
including regressions and ties, and the fact that all3 probability means are
below historical separate original3DD. One-seed/whole-policy/provenance limits
remain explicit. Compact evidence includes original CSVs, source/checkpoint/
dataset/native identities, acceptance receipts, failed/stopped preparation
outcomes and dated closure. Agent B's later campaign/closure remains separate.

## Verification and exclusions

[Run] Integrated350CPU suite tests pass;18 targeted observer/analyzer/runner
tests pass. Shared production analysis freshly reproduces all9 macro and135
corruption rows from raw3 ZIPs within1e-10 for every original column. Raw3,
accepted binder and final report SHA/CRC checks pass locally. The first test
attempts lacked NumPy/Torch; an existing cached dependency environment passed
without adding project dependencies. These are CPU integration checks, not a
new model accuracy run or external replication.

Raw ZIPs, weights, arrays/caches, large per-batch/member dumps, setup wrappers
and repetitive live-state histories stay in the source worktree/Drive. Necessary
negative/null/failed results remain in the compact evidence. The original
Drive-local verification is historical; no Drive or runtime operation was
performed during integration. The immutable executed snapshot is the execution
anchor; current shared code contains later Agent B extensions.

[Selection manifest](evidence/thesis_integration_20261030/agent_a_selection.json) ·
[Test receipt](evidence/thesis_integration_20261030/agent_a_verification.json) ·
[Raw reanalysis](evidence/modelnet_trajectory_20261025/integration_reanalysis.json) ·
[Full study and raw archive locations](modelnet_multibackbone_trajectory_handoff_20261025.md).

Publish a normal fast-forward push to `origin/thesis-integration`; report its
verified remote SHA after publication. This curated commit does not mark the
excluded Agent A history as merged. No force-push, `main`/`upstream` write,
runtime cleanup or worktree deletion is authorized by this integration.
