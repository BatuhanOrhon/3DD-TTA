# Integration scope: proximal pilot handoff

2026-10-31. [User report/Decision] User requested a standalone fresh-session
prompt and explicit recording of the current research state. This publication
completes that documentation task; the new scientific pilot remains unrun.

- Source worktree: `/Users/batuhan/.codex/worktrees/7bf0/3DD-TTA`.
- Source branch: `integration/chamfer-guidance-variants-20261030`.
- Source HEAD: `a82200adbe117ce56a134dc3c22601cb73dc2654`, dirty state preserved.
- Integration base: `1bacabfe20a70bf94e9a3c8a5988084d0ae48d3f`.
- Target: only `origin/thesis-integration`, normal fast-forward push.
- Selected: complete [handoff prompt](shape_latent_proximal_handoff_20261031.md),
  proximal design and updated execution plan, this scope note, and dated
  README/findings/open-question updates reconciled onto the latest target.
- Excluded: uncompleted proximal scientific code/tests, existing dirty tta/model
  and unrelated edits, setup/operator wrappers, raw archives, caches and weights.
  Those files remain available in the source worktree; this documentation
  publication does not claim to integrate or validate their GPU execution.

[Run/Code] Git worktree/branch/HEAD and prior remote refresh publication were
verified. CLI0.7.4 server inventory showed the owned replacement A100 session
`tta-shape-proximal-a100-20261031-r1`, endpoint`gpu-a100-hm-ass1c0-zwl345mh6780`.
Cell2 exact completion marker and deferred fastrlock failure, both completed
uploads, sealed asset/operator size/SHA/CRC and real v2 registration were checked.
The outside install wrapper's old-session assertion was confirmed and recorded
as pending correction. No new notebook cell or scientific GPU run was launched;
no runtime teardown was issued. Only knowledge and local progress records changed.

Validation for this documentation scope: referenced local files exist;
new Markdown relative links resolve in the publication checkout; original
shared-doc content is preserved; staged diff contains only the selected
knowledge files; `git diff --check` passes. No inference tests were rerun for
documentation edits. Final commit/remote SHA is recorded in the local
publication receipt and delivery message after the push.
