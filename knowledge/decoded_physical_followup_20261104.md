# Decoded physical spectral follow-up — 2026-11-04

Status: **prepared**, measurement support passed CPU verification. This continues the next
tests in [the preceding results](decoded_physical_spectral_results_20261103.md).
Evidence labels: [Code], [Run], [Inference], [Open]. The user's instruction
“Tamam şimdi planımızdaki bir sonraki testleri çalıştır” authorizes execution.
No research GPU job in this phase has started yet.

## Ownership and fixed scope

- Branch: `research/gsd-physical-followup-20261104`, derived from
  `origin/thesis-integration` at `72803180be4be628f12dbb284910755f628c4a60`.
- Worktree: `/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-guidance-math-audit-20261103`.
- Reuse assigned session `tta-gsd-physical-spectral-a100-20261103-r1`, endpoint
  `gpu-a100-hm-ass1c0-i933qghuewn8`, A100-SXM4-80GB. Canonical setup already
  passed. Monitor exec 97932 / PID 34871 remains sole cell owner. The same VM
  answered its CPU liveness probe at 2026-11-03 23:57:19 UTC.
- Fresh remote root `/content/studies/gsd-physical-followup-20261104`;
  Drive `/content/drive/MyDrive/thesis/result/gsd_physical_followup_20261104`;
  local preparation `result/preparation/gsd_physical_followup_20261104`.
- Whole GPU phase estimate 25–45 minutes, hard cap 55 minutes, measured from
  its first research GPU job. Setup, CPU analysis and retrieval are separate.
  No phase over 60 minutes is planned. Runtime teardown is not authorized.

Frozen protocol: PointMAE / LION raw eval / EMA off, seed 0, batch 2,
severity 5, DDIM 100 / final alpha 1, N8 except Background N35,
PM clean checkpoints 2/4/6 (Background 9/18/27), terminal 8/35;
gamma .03, eta .003, p .95, SCD sum. No physical guidance is applied in
these diagnostic arms. Previously frozen coefficient 802.5000841164167 is
not refitted using accuracy. No full benchmark or applied accuracy claim.

## P0 — numerical repeat / backend protocol

Three fresh processes each for legacy and controlled backend settings, same
Gaussian IDs 44/91, same previously sealed paired preparation cache. Legacy
retains benchmark=True and deterministic=False. Controlled disables benchmark
and both TF32 flags, enables CuDNN deterministic, and leaves PyTorch strict
deterministic algorithms disabled because custom native atomics remain.
**Controlled does not promise determinism.** Record all actual flags.

Record full/full and full/detached decoder forward differences at the exact
same state, preserving RNG and buffers. Cross-process gates, fixed before
results: trajectory logits max absolute <= .05 and RMS <= .01, first decoded
RMS <= 1e-4, identical prepared inputs / initial states / RNG / source / native
and checkpoint identities. Same-state forward witness gate: RMS <= 1e-5 and
max absolute <= 5e-4. Report all pairs and worst cases; do not relax gates.
Failure prevents attributing separate-run small accuracy shifts to guidance.

## P1 — H8 direction transfer

Fixed controlled backend, eight stage-disjoint objects (459, 545, 1229, 1435,
1441, 1557, 1881, 2095) x Gaussian / Background / Shear / LiDAR = 32 object
conditions, 16 batches, 96 checkpoint comparisons. IDs were selected by
SHA-256 ordering with fixed salt, excluding the prior pilot and the five
recorded development/confirmation cohorts; the exact manifest includes their
hashes. Historical full-benchmark exposure remains: this is a mechanism pilot,
not unseen confirmation.

Compare exact full-decoder and H8 VJPs at the same state after denoiser
propagation and the .03/.003 update scales. Gates: cosine >= .99 and H8/full
norm ratio within [.9, 1.1]. Report fraction passing, minima and extrema by
corruption. P0 noise failure still allows descriptive same-state P1, with
explicit qualification. No detached-input production decoder optimization.

## P2 — slot correspondence and corrupted-anchor oracle

Save raw corrupted inputs, normalization center/scale, physical decoded points
and terminal classifier geometry for every batch. CPU artifact analysis uses
pinned clean `data_original.npy` **only as an oracle**, after all GPU trajectories
are fixed. Map clean points into the corrupted input's prepared physical frame
using its exact center/scale, factor 3.3885 and rotation (-z,y,x); classifier
renormalization is not this frame. Set distances tolerate unequal point counts.

For each object's physical decoded Y and prepared X, F is the fixed input heat metric;
g = 2 F(Y-X)/(3N), d = -g/||g||, epsilon = .001 ||Y||. Compare spectral
fidelity and clean symmetric nearest-neighbor squared Chamfer before/after this
one point-space perturbation. Report simultaneous spectral decrease / clean
distance increase as an objective-bias counterexample, not proof about the
sampler update. Also report slot versus nearest-neighbor discrepancies and a
fixed Y-only permutation control: set distance must stay invariant while slot-based
loss may change. Joint relabeling (X,Y,F -> PX,PY,PFP^T) must preserve the loss.
Check F symmetry and record the exact finite-step quadratic change
`-epsilon*||g|| + epsilon^2*(d^T F d)/(3N)` to distinguish finite-step overshoot
from a derivative error. Clean data never select a coefficient, checkpoint or arm.
Reconstructed F/loss must agree with saved GPU evidence to absolute 1e-6
(otherwise report analysis mismatch and stop that comparison).

## Completion requirements

Preserve successful and failed raw bundles, source overlay, exact manifests,
native payloads, execution scripts, logs, clock/cap and Drive/local SHA receipts.
Use fresh attempt directories; do not overwrite prior pilot evidence. Validate
required archive files and completion markers before analysis. Update this note,
findings/open questions and the current README headline. Review and integrate
only completed necessary source, tests, compact evidence and notes using the
standing `origin/thesis-integration` authorization. Keep the runtime monitored.

[Code] Before deployment: full CPU suite **492 tests, 491 passed / 1 skipped**
in 26.868 seconds. The two analysis-gate tests also passed after extending locked
native/bootstrap, scheduler and initial-state controls. Independent review found
and prompted stronger analyzer identity checks; no production guidance change.
