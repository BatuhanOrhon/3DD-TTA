# PM low-band half mixture — 2026-11-05

[User report/Code/Open] Execution of the explicitly supplied
[handoff](gsd_lowband_halfmix_agent_handoff_20261105.md), starting at d5869c2.
Own branch `research/gsd-lowband-halfmix-20261105`, own worktree
`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-lowband-halfmix-20261105`.
Peer H5 files, branch, runtime and monitor are protected. Evidence labels:
[Code], [CPU], [Run], [Inference], [Open]. No accuracy outcome yet.

## Implementation and execution plan

- [x] CPU: reuse `build_smooth_spectral_target(profile='hard')` with M100/200;
  preserve its active-support/isolate and complete boundary-eigenspace rules.
  Add `decoded_scd_weight=1` to tta.py; only change SCD when a live clean-PM
  loss exists. Existing additive guidance/defaults remain unchanged.
- [x] CPU: optional observer update probes use existing SCD and physical loss
  graphs, two extra VJPs, actual gamma=.03/eta=.003, float64 norm arithmetic.
  Record norms/cosine/mixed norm/cancellation. Reject nonfinite or <=1e-12
  component update norms during calibration; no clamping or silent rescaling.
  Freeze alpha as NumPy median of all predeclared per-object/checkpoint ratios.
- [x] CPU: driver flags for hard modes, checkpoint SCD coefficient and optional
  probes; all five arms use one locked DDIM100/finalalpha1/raw/eval/EMAoff host.
  Tests: handoff formula/gradient/FD, repeated eigenspace/isolate, exact prepared
  frame, half-SCD only on clean PM, no terminal update, frozen scalar validation.
- [ ] Colab: own A100 runtime and sole monitor; canonical ten setup cells,
  cell1 restart/reconnect, then2–10, exact markers, known cell2 fastrlock only.
- [ ] Freeze source bytes/dirty hashes/native bytes/assets, concrete IDs and
  whole GPU estimate before science. Target16 new IDs/all15/s5/seed0/B2,
  calibration4 separate IDs/Gaussian+Background, no labels/clean in calibration.
  Estimate30–45min, hard55min from first scientific GPU work, subject to
  measured projection; >60min needs explicit user approval.
- [ ] Technical: grad/FD/projection/frozen states, same cached preparedX,z0,h0,
  noise and RNG; operational/live0 plus same-observer fresh-process repeats.
  Fixed gates: logit max .05, per-object/checkpoint RMS .01, first decoded RMS
  1e-4. Preserve failures and limit accuracy comparisons to descriptive ones.
- [ ] Pilot: operational full SCD, live full SCD/weight0, live half SCD/weight0,
  halfmix M100, halfmix M200. All caches prebuilt before measured arms; inspect
  actual ranks/eigengaps/support and alias filters. No alpha/accuracy selection.
- [ ] Archive seven files/prediction NPZ/cache/graph/norm/source/native bytes and
  successful/partial/failed ZIPs; verify local/Drive SHA. Independent PM primary,
  terminal secondary recount and object-cluster bootstrap, all15/per corruption,
  versus operational/full-live/half-SCD controls. Separate fresh pipeline,
  preparation, graph/eigh, loss/extra VJP, reconstruction+PM, terminal head,
  IO/worker wall and peak scopes. Update memory and publish necessary completed
  code/compact evidence via clean latest thesis-integration normal push.

## Scope and risks

No H8, online exact norm matching, beta/rate search, classifier gradients,
clean-reference tuning, extra seeds or default promotion. Existing Background
same-observer instability is unresolved; controlled cuDNN is not determinism.
A hard rank expansion can alias M100/M200; record it without treating aliases
as independent methods. Retaining U_M does not reduce full eigh setup cost.

## CPU result and allocation blocker — 2026-11-05 18:10UTC

[Code/CPU] The opt-in implementation is complete. `tta.py` accepts the fixed
scalar `decoded_scd_weight` (default1); it acts only when the decoded observer
returns a clean-PM loss. The existing spectral coefficient remains the additive
coefficient, so the new pilot passes `--checkpoint-scd-weight .5 --weight
(0.5*alpha_M)`. The other steps retain SCD1, terminal gets no new update.
One joint backward remains per step. Optional component VJPs retain the graph,
never accumulate `.grad`, and are explicitly counted/timed separately.

[Code] `--lowband-modes 100|200` selects `profile='hard', beta=None`; the smooth
profile remains the historical default for this diagnostic entrypoint. Hard
runs reject the H8 flag. Existing graph/isolate/eigenspace rules are reused.
Opt-in `retain_bases=True` forms F=sym(U_M U_M^T) from the selected padded basis
and archives that basis for **every batch**, with exact tensor/F hashes and
active vertex indices. This uses a selected-basis matrix product instead of
historical full-size zero-weight multiplication; it is the same mathematical
projector but carries **no cross-device/native bitwise equivalence claim**.
Full eigh is still computed. Dense F is retained in memory and the first raw
filter witness remains archived. Basis storage and graph build peak are recorded;
reconstruction peak excludes earlier graph-build temporaries but includes the
resident filters/bases. Instrumented replay is not fresh deployment throughput.

[CPU] Full suite514 tests /513pass /1historical skip in41.540s; independent
read-only review68 focused tests and an entirely isolated hard-graph self-check
passed. New combined positive-weight tests cover normal8 and Background35,
checkpoint-only halfmix, exact probe-on/off local/style update equality and one
joint backward per step. Hard loss/gradient reconstruction, repeated-boundary
expansion, empty support, frame checks and default sampler contracts are covered.
Existing warnings and historical fixture skip are retained in local full logs.
[Compact verification/allocation receipt](evidence/gsd_lowband_halfmix_20261105/cpu_and_allocation.json).

[Run/Open] Four allocation attempts produced no owned runtime:
`tta-gsd-lowband-halfmix-a100-20261105-r1` A100/HighRAM ->503ServiceUnavailable;
`r2`, `r3` and user-requested `r4` same A100/Standard ->precondition failed (capacity/usage/assignment
limit unspecified by server). Inventory showed three protected peer assignments.
No peer cell, reconnect, teardown, monitor, branch or file mutation occurred.
No task monitor was started, no setup cell or scientific GPU job ran, and no
alpha/accuracy/timing/native-gradient/repeatability result exists. Scientific
GPU time is0. This is an external allocation blocker, **not a failed guidance
result**. Prior Background instability remains unresolved. Do not promote the
CPU implementation as a completed Colab experiment.

Local preparation: `result/preparation/gsd_lowband_halfmix_20261105/` contains
source-hashed ten canonical cell wrappers, copied monitor source, test logs and
per-role selection manifests. Future remote `/content/studies/gsd-lowband-halfmix-20261105`;
Drive `thesis/result/gsd_lowband_halfmix_20261105`. No Drive copy is claimed yet.
Next available session name is `tta-gsd-lowband-halfmix-a100-20261105-r5`;
reconcile inventory first, allocate own A100 only, and follow canonical setup.

## Frozen selection and continuation details

[Code/Proposal] [Selection manifest](evidence/gsd_lowband_halfmix_20261105/selection.json)
uses SHA256 salt `gsd-lowband-halfmix-20261105-v1` after excluding582 IDs in the
previous physical study's exclusion union and32-ID cohort. Calibration IDs
411,672,777,2247; pilot IDs197,502,536,1008,1064,1240,1285,1422,1549,1670,1738,
1824,1858,2100,2285,2310. These remain development IDs with prior full-benchmark
exposure, not unseen confirmation. Two fixed8-ID pilot shards reuse the existing
runner's count8 contract; calibration uses two fixed2-ID shards.

After setup, strict CPU/source/native/asset verification and source seal:

1. Prebuild all preparedX/z0/h0/noise/post-preparation-RNG caches before measured
   arms; archive actual NPZ bytes and hashes, preserving per-B2 preparation cost.
   Do not allow one measured arm to populate missing caches while others replay.
2. Technical control/live_control fresh-process repeats on Gaussian44/91 and
   Background495/512, three processes each, all weight0. Use hard100 live path.
   Native full-decoder FD with fixed target, projection equality, frozen state
   and the original numerical thresholds remain mandatory. Failed repeats do
   not become accepted because PM classes happen to agree.
3. Calibration uses **only** new4IDs ×Gaussian/Background ×three clean PM
   checkpoints =24per-object ratios per M. Run `--arm live_control
   --lowband-modes M --update-norm-probes`, SCD1/weight0/controlled backend.
   Require all24 declared rows per M and their ID/corruption/checkpoint/source/
   cache identities before calling `freeze_update_alpha`; it deliberately
   rejects inactive/nonfinite ratios. Freeze each alpha and manifest SHA before
   any main accuracy result is read. No old802.5 coefficient or accuracy choice.
   Compute frozen-alpha mixture norms/cancellation from those saved component
   norms/cosines; the measurement VJPs do not belong to deployment timing.
4. Pilot seed0/B2/s5/all15, same caches and backend, five fixed arms below;
   no update-norm probes during measured pilot. Rotate arm order across the two
   shards: listed order then reverse. Reassess measured whole-phase projection
   before pilot; estimate30–45min/hard55min including technical/calibration and
   fresh-pipeline timing. Do not restart the phase clock. If expected >60min,
   obtain explicit whole-phase approval before further research GPU work.

| Arm | CLI arm | lowband-modes | checkpoint-scd-weight | weight |
|---|---|---:|---:|---:|
| Operational full SCD+PM | control | omit |1|0|
| Matched live full SCD | live_control |100|1|0|
| Matched live half SCD | live_control |100|.5|0|
| M100 halfmix | physical |100|.5|.5*alpha100|
| M200 halfmix | physical |200|.5|.5*alpha200|

All invocation prefixes use `scripts/diagnose_decoded_spectral.py --asset-root
ASSETS --asset-manifest-sha256 VERIFIED_SHA --result-root NEW_ROOT
--paired-state-root CACHE_ROOT --drive-root NEW_DRIVE_ROOT --backend-policy
controlled --index-manifest ROLE_JSON --index-manifest-sha256 ROLE_SHA --count
2_OR_8 --corruptions ...`. Resolve absolute paths and hashes after deployment;
these placeholders are **not launchable commands**. Seal the concrete commands,
actual native bytes and source/dirty identities before starting science.

5. Retain all rank/support/eigengap/basis/F evidence; report M100/200 aliases
   explicitly. Native numerical failure leaves any accuracy comparison
   descriptive; no gate relaxation or default promotion. PM primary, terminal
   secondary; all15/per-corruption correct/n, paired operational/live-full/half
   differences and10,000 original-object cluster bootstrap draws, seed20261101.
6. Complete all seven-file raw bundles/prediction NPZ/source/native/caches,
   successful/partial/failed attempts and SHA-verified Drive/local backup before
   inference or publication. Keep fresh pipeline, common preparation, kNN/eigh,
   loss/VJP, decoder+PM, terminal head, IO/worker wall and peak scopes distinct.

Remaining: all Colab setup/technical/calibration/pilot/analysis/archive phases.
Only completed CPU code, tests and compact preparation evidence are selected
for the current integration publication; raw setup wrappers/archives are local.

## User-directed wait — 2026-11-05

[User report/Decision] After standardA100r4 refusal, the user directs code
preparation now and will announce when an existing session becomes available.
**Do not retry allocation or begin setup/research GPU work until that notice.**
No monitor, retry queue or unattended GPU job exists for this task. Preserve
all peer assignments; this direction gives no teardown authorization.
