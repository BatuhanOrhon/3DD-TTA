# PM M100 quarter-screen retry — 2026-11-08

[User decision/Run/Failed/Open] User authorizes closing background terminal
processes, creating one fresh Colab runtime, starting its legitimate persistent
monitor, and retrying the unfinished experiment. GPU runs need no repeated
approval; recurrent session/allocation problems still require a new command
before another retry. The original four-hour bound remains the retry scope.

[Run] At06:54UTC, server inventory reported no active sessions. No old or peer
runtime was terminated. Exact own local groups53959 (old r1 monitor, child
CLI90567 and tee),9683 (obsolete r2 Drive mount) and86618 (own caffeinate) were
ownership-verified and sent SIGTERM. Follow-up local ps confirms all five known
old PIDs absent. No broad pkill, peer process or peer file was changed.

[Run/Failed] `colab new -s tta-gsd-quarter-fast-a100-20261108-r4 --gpu A100
--high-mem` was submitted06:55:07UTC and exited1. Exact backend message:
“Backend rejected accelerator 'A100'. You may not have quota or entitlement
for this accelerator on your account.” This is an allocation rejection; the
specific quota/entitlement cause is not independently established. No endpoint
or VM was allocated, so no runtime monitor or setup/research job was launched.
No alternative accelerator or further allocation was attempted. WAIT_USER.

[Code/Open] CPU preparation and unchanged fixed617/all15/s5/seed0 three-arm
protocol remain in the prior quarter preparation namespace. Calibration64 lies
outside617; batch-specific norm calibration and all GPU gates remain unrun.
The old B2 and small-pilot results remain historical evidence, not results of
this retry. The old r3 connection failure and expired four-hour budget are
preserved. New allocation request timing in the r4 budget is an audit record,
not evidence of an active or completed GPU phase.

Worktree: /Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-lowband-halfmix-20261105.
Branch: research/gsd-lowband-quarter-fast-retry-20261108. Scientific source918780b;
completed r3 local record committede66c035. Last published integration5fb5b5f.
Raw local evidence: result/preparation/gsd_lowband_quarter_fast_20261108_r4/
(local_cleanup.json,local_cleanup_verification.txt,allocation_failure.json,
gpu_budget.json). Canonical setup wrappers copied unchanged for a future
explicitly authorized allocation; no new scientific controller retargeting or
math change was needed or executed. Later records remain local while waiting
for the user's next command; no external publication after the rejection.
