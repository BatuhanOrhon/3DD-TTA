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
