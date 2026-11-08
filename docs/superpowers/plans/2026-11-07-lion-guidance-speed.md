# LION Guidance Speed Continuation Implementation Plan

> Execute inline with executing-plans; user has authorized necessary sequential research tests without further run-permission questions.

**Goal:** Determine whether fixed cheap/hybrid guidance offers a defensible speed/accuracy tradeoff against operational N8+PM.
**Architecture:** Reuse experimental sampler, source-bound assets, canonical setup, and archive validation. Add one speed runner and one small paired-analysis module; leave production defaults unchanged.
**Tech Stack:** Existing Python3.8/Torch2.1.2/CUDA12.1, NumPy and ColabCLI; no new dependencies.
**Spec:** knowledge/lion_guidance_speed_20261107.md

## Global constraints
Fixed tau.1, N8/B35, gamma.03/eta.003/p.95,4equalPMreadouts. All15/s5 included in operational comparisons. No new B/C tuning. Primary>=15% saving/-0.3pp paired95%lowerbound. Original sampler is the reference. Native stochastic repeats do not pass historical exact gates.

## Review focus
Timing includes fresh encoding and PM consistently; observer hash costs excluded equally. Repeats are not an ensemble. Order effects balanced. Full shim/terminal decoder grad branch audited. Exposed official-test status and object-cluster uncertainty explicit.

## Tasks
- [x] Setup and record fresh owned A100 canonical10cells, hashes and exact markers; preserve failed attempts and peer runtime.
- [x] Add scripts/run_lion_guidance_speed.py with --phase repeat/screen/confirm (warmed timing included), explicit pinned selection/manifest, balanced orders and fresh inference timing; reuse sample and original runtime. Add scripts/analyze_lion_guidance_speed.py for paired clustered summaries. Tests pin PM correctness, unmatched rows rejection, repeated-unit accounting and order balance before implementation.
- [ ] Run source-bound native characterization, accept archives, then all15screen and warmed measurements; record every phase decision.
- [ ] Lock survivor and run conditional256×3×2confirmation/transfer; stop expansion only on declared scientific criteria or an explicit external blocker.
- [ ] Independent whole-change/evidence review, CPU suite, knowledge reconciliation, fresh integration to thesis-integration; preserve runtime.

## Ledger
2026-11-07: Reused isolated publication worktree on new task branch; original dirty checkout preserved. User opted into autonomous execution, so no design/permission stall. Current remote0b6faeb verified. Allocation succeeded. Firstcell1 CLI lost connection before submitting setup; same-VM inspection found no conda and no installer process, so local stale client was stopped and same canonicalcell1 retried. No science has run.

2026-11-07: CPU implementation/review fixes verified823pass/2skip/500subtests. Added actual readout grad/RNG and all9cross-control pairs tests; candidate capture no longer duplicated. Stage1 remains pending user Drive consent beforecell3; no science launched. Conditionalcheap survivor requires a reviewed unguided control beforeconfirmation; operator explicitly refuses that premature launch.

2026-11-07: Canonicalsetup10cells, deployment and7ELFpre-run supplement accepted; v3current. Repeat20261107T163252Z accepted288rows/96own/144cross; independent reviewer permits operationalall15screen without relabeling failed historical numerical parity. Raw/source-bound advancement frozen before screening.

2026-11-07 recovery: r1unexpectedloss after121/360; finalintegrityunmeasured, no confirmationpass. R2A10080 allocated/cell1+reconnectpassed, Driveaccountconsentpending. Fullr2confirmationrerun withper-corruptionhashproof; sourcev5sealedlocalonly,827CPUtests pass. Preserveallr1evidence; noaccuracy-basedcase selection.
