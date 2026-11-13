# GPU task authorization — 2026-11-13

[User decision] The user explicitly requests removal of the separate GPU-run
approval rule and instructs recreation of Colab and launch of the existing
full-test evaluation. Requested GPU work now proceeds without a separate
confirmation, regardless of duration. Historical 60-minute approval gates in
dated research notes are superseded; their execution/result history remains.

[Code] AGENTS.md, the Colab operator skill and the thesis research skill now
share this rule. Prepare and record whole-phase scope, source/data/checkpoint
identities, time/cost estimates, stopping limits and recovery before execution.
A machine authorization receipt may bind the user instruction to the actual
plan; it must not become a new human approval gate. Interactive Google account
consent and explicit task-completion/runtime-close authorization stay separate.

[Code] Selected source commit f043b4c from
`experiment/fulltest-n5-proximal-n8-20261112`; four policy/test files only.
Eight Colab policy regression checks pass on latest shared base45b7aaf.
Unfinished full-test experiment code, setup wrappers, raw artifacts, weights and
unrelated active-project edits are excluded from this policy integration.
No research accuracy, runtime closure or expanded scientific scope is claimed.
