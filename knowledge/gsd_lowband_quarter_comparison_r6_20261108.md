# PM M100 comparison on owned r6 — 2026-11-08

[User decision/Run/Open] User confirms the unknown A100 is not theirs and must
not be used; explicitly authorizes a new owned runtime if prior r5 is gone and
forbids closing another agent's runtime. At10:49UTC, server inventory lacks old
r5 endpoint, lists unknown+peer. Unknown was never attached, executed or closed.
A new own r6 A100/High-RAM request at10:52 returned503 Service Unavailable.
Read-only inventory confirmed no r6 assignment. One same-name allocation retry
at10:55 succeeded by10:56:29UTC. No other runtime was stopped.

Owned session: tta-gsd-quarter-fast-a100-20261108-r6.
Endpoint: gpu-a100-hm-ass1c0-uycluss04t2v.
Worktree: /Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-lowband-halfmix-20261105.
Branch: research/gsd-lowband-quarter-fast-retry-20261108.
First monitor66517 started10:56:48UTC with supported local REQUEST_TIMEOUT=60;
cell1 passed10:57:14UTC, same-VM Conda reconnect10:58:11UTC confirms A10080GB,
81920MiB. Known-idle local monitor stopped10:58:36UTC solely for serial Drive
mount; no VM teardown. User reports new Drive consent completed. Sole owner
is pending drivemount terminal40681; monitor restarts after mount completion.

[User decision] The user explicitly removed the global time window and asked
for the comparison. Original r5 budget5986cce and hashed waiver32421e2 remain
historical immutable anchors. R6 has separate runtime_assignment.json; no
arbitrary new total cutoff. Worker/CLI timeouts, WAIT_USER on connection loss,
source/native/cache/FD/repeat/archive gates and finite40-job scope remain.

[Protocol] Scientific source918780b, assetmanifest e3f280/source a62a7e unchanged.
Fixed617/2468 official ModelNet40-C test IDs/all15/s5/seed0; three arms
operational SCD, livehalf0, hardM100 halfmix. Validated B32 resource selection
from r5 is reused only after actual A10080GB verification; old resource raw
archive f94fc37 remains provenance, not new accuracy. No old alpha or B2 states.
R6 prepares all64 calibration IDs outside617 anew, freezes384 actual gamma.03/
eta.003 weighted local/style ratios, then runs unchanged native FD/repeat/tail
checks and matched617 inference. No accuracy-dependent selection or silent
coverage reduction; native instability remains and small gains stay descriptive.

[Code/CPU/Open] Completed r5_unbounded fork is CPU-reviewed with explicit waiver,
None globaldeadline, preserved source/numeric/ACK/timeout guards. R6 retarget is
local CPU-only during fresh canonical setup. Initial deployment will explicitly
resolve the already verified mounted thesis shortcut before unchanged strict
native guards; actual fresh R6 native bytes are Drive-preserved before any pinned
restore. Scientific donor ZIP4eaa unchanged. No science launch until setup,
source/context/Drive/Mac identities and required technical gates pass.

Current evidence: result/preparation/gsd_lowband_quarter_fast_20261108_r6/.
R5 canonical/environment/context and successful B8/16/32/64 resource profile
archives remain in their original roots. No617 accuracy result or fresh alpha
exists as this record starts. Prior transport/backend failures remain preserved.

[Run] 11:04UTC Drive/mount audit: user consent propagated. Prior local mount
terminal was no longer present, so a read-only raw-console audit first verified
fuse.drive at /content/drive, readable project wheel, both kernels idle and no
active build. Console detached without interrupting kernel. Sole monitor23310
restarted11:04:39 with REQUEST_TIMEOUT60 and scoped local caffeinate -i; actual
cell2 launched11:04:48. No new allocation, duplicate job or peer operation.
Audit receipt: r6/setup/drive_mount_verified.json. No new research accuracy.

[Run/Code/CPU] 11:12UTC canonical cell2 reached only the expressly deferred
fastrlock0.8.4 wheel failure (full log retained); separate Python3.8/Drive
verification passed LOWBAND_CELL2_ENV_VERIFIED. Cells3–10 continue serially
through sole monitor with exact markers and immediate stop on any new failure.
Final R6 operator/bootstrap independent review found/fixed five stale R5 Drive
constants before deployment. Final checks:16 operator contracts,40-job simulator,
4 bootstrap tests,5 native-preservation guards,33 consumer tests plus transfer
checks. Active producer/consumer ROOT and Drive now R6; historical R5 metadata
stays byte-exact. CPU receipt r6/cpu_verification.json; sealed operator ZIP
143380B SHAe5fd72d180d7e4a198dd91ad29ed6cf23b79ca0f0616c9d21f0a9b7fa4499642;
bootstrap9853B SHA13e43385bd1edbb85080f4a507f8d28beaacc69f58a6f9d0559182f8aec4a6e8.
Staging uploads use only ContentsAPI while setup owns kernel; no science launch.
