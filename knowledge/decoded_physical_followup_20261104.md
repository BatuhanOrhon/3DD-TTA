# Decoded physical spectral follow-up — 2026-11-04

Status: **P0/P1 and mixed-origin P2 complete; raw/derived backup and integration published**. This continues the next
tests in [the preceding results](decoded_physical_spectral_results_20261103.md).
Evidence labels: [Code], [Run], [Inference], [Open]. The user's instruction
“Tamam şimdi planımızdaki bir sonraki testleri çalıştır” authorizes execution.
First research phase started 2026-11-04 00:06:24 UTC.

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

[Run] Deployment completed at 00:05:02 UTC without model forward or CUDA init in
the strict CPU preflight. Source `2109a5d67715001ac7820322aab8ace0530b011b`;
overlay 682700 bytes / SHA256
`855f467cbcebd8b45d2a6b91d505b991a678dab54d701c86a55095d3f280cae3`;
asset manifest 181762 bytes / SHA256
`b485ea97be7693c690e25746fdb767a01b88c08bd99b5f7e52848c6b16eb2cd7`.
Operator package 684634 bytes / SHA256
`7528ad1450ef6c047d489daffbb1dd1f3518d210bb986b893fcc544445f995ef`.
P0 legacy1 began at 00:06:24 UTC with a new immutable 3300-second phase clock.

## Accepted P0/P1 results

[Run] All seven GPU runner bundles completed. Whole phase elapsed **326.766 s
(5.446 min)**, 00:06:24–00:11:50 UTC, below the immutable 3300-second cap.
The conservative 25–45 minute estimate was not consumed by extra experiments.
P0 and P1 remained zero-weight diagnostics; no applied accuracy arm launched.

| P0 worst observed discrepancy | Legacy | Controlled |
|---|---:|---:|
| Cross-process logit max absolute | 2.424399853 | 0.0000557899475 |
| Cross-process per-object/checkpoint logit RMS | 0.749611909 | 0.0000243000839 |
| First decoded per-object RMS | 0.0107397983 | 0.000000112835125 |
| Same-state full/full decoded RMS | 0.0000116168048 | 0.0000000234470061 |
| Same-state full/detached decoded RMS | 0.0000113744982 | 0.0000000235956943 |
| All cross-process / full-repeat / detached gates | fail / fail / fail | pass / pass / pass |

[Run] Three pairs per backend, all locked source/native/model/cache/input/state/RNG
controls accepted. Controlled actual flags matched the declared policy. These
settings jointly changed benchmark, deterministic and CuDNN TF32 behavior; the
individual flag responsible is not isolated. Custom native kernels remain outside
a strict determinism guarantee.

[Inference] The previous V5 full/detached equivalence rejection under legacy
settings does **not isolate detachment as its cause**: a repeated ordinary full
forward also fails that same RMS gate here. Under controlled settings both forms
pass. V5 remains a recorded rejected attempt and its optimization remains reverted;
this is evidence for a future controlled optimization test, not retrospective
acceptance of its unapplied arms.

| P1 corruption | Passing object/checkpoint comparisons | Minimum cosine | H8/full norm ratio range |
|---|---:|---:|---:|
| Gaussian | 24/24 | .998278310 | .999690252–1.001727793 |
| Background | 24/24 | .999838596 | .999255786–1.000541019 |
| Shear | 24/24 | .999174524 | .999588137–1.001267228 |
| LiDAR | 24/24 | .995611972 | .999096588–1.004202485 |

[Run] **96/96** passed; the minimum occurred at LiDAR ID1229, clean6. All48B2
checkpoint forward witnesses also passed; worst full/full RMS3.54055167e-8,
full/detached RMS3.58671812e-8 and max absolute2.38418579e-7. These are selected
zero-guidance trajectories, not a bound along future guided paths.

[Run] Backend comparison cost, measured on the same three B2 diagnostic repeats:
median reconstruction **2.654118s legacy / 2.961973s controlled** (+11.60%).
Median peak **21756.4458MiB / 4336.5571MiB** (approximately21.25/4.24GiB).
These timings include extra diagnostic forwards/VJPs and exclude graph setup;
the memory includes their process allocations. They do not measure H8 deployment
speed or attribute the difference to a single flag. No speed/accuracy gain claimed.

## P2 reconstruction failure and exact saved-filter audit

[Run] CPU V1 source `6996d50` preserved an **analysis_mismatch** result: CPU
reconstruction of first Gaussian B2 F differed from the saved GPU F by maximum
**5.95178045e-6**, exceeding fixed1e-6. That B2 case's four first/terminal rows
were aborted; 60 rows from30other object-conditions passed their first-loss
scalar checks (max absolute1.96350611e-7). Their metric directions are explicitly
CPU-reconstructed, not GPU-operator certified. No tolerance was relaxed.

[Run] A separate CPU audit directly used the archived GPU F for IDs459/545.
Gaussian ID459 terminal gives a concrete output-space counterexample:
spectral loss **.0143945200835 -> .0140846361175**, while clean symmetric mean-NN
squared Chamfer **.0505244935070 -> .0505442634157**. The exact quadratic change
matches the gradient/curvature calculation within1.20e-18. Both permutation
controls passed. No sampler update or classification improvement follows from
this point-space oracle direction.

[Run] V2 source `644bcfa` used archived GPU F directly where available, verified
against `graph.json` tensor hashes, and retained the failed CPU-reconstruction
gate separately. The other30object-conditions retain CPU-reconstructed
scalar-gated metrics. This new mixed-origin CPU analysis has fresh output; V1
failure and source bytes remain archived. The numerical gradient uses
`(F+F.T)E/(3N)`, exactly evaluating the stored quadratic including float32 symmetry
roundoff; it reduces to `2FE/(3N)` for exactly symmetric F.

[Run] V2 status **complete_with_reconstruction_mismatch**,64/64 rows. Both
available CPU/GPU F reconstruction comparisons failed (ID4595.95178045e-6,
ID5453.54647636e-6 > fixed1e-6); they were not repaired or certified. Exact
archived F nonetheless supports the two corresponding object-conditions/four
checkpoint rows. The other30conditions/60rows have only a scalar loss gate.

| Corruption | Spectral decrease / tested steps | Clean CD increase, first | Clean CD increase, terminal |
|---|---:|---:|---:|
| Gaussian | 16/16 | 0/8 | 5/8 |
| Background | 16/16 | 5/8 | 7/8 |
| Shear | 16/16 | 6/8 | 7/8 |
| LiDAR | 16/16 | 6/8 | 6/8 |

[Run] **42/64** finite physical steps decreased spectral loss and increased clean
CD: **1/4** archived-GPU-F rows, **41/60** CPU-reconstructed rows. All64 spectral
losses decreased, all steps resolved, no finite-step overshoot. Maximum exact
quadratic prediction error7.72e-17, joint permutation error5.56e-17; set-CD
permutation invariance passed. These64rows reuse eight objects and are not
independent samples or a population failure-rate estimate.

[Inference] This establishes that corrupted-X spectral fidelity can oppose clean
geometry at an observed physical output, even when H8 approximates its derivative
well. It neither identifies slot misalignment as the unique cause nor proves an
applied sampler/accuracy effect. A fixed common U is still a valid representation;
the objective's target and correspondence assumptions remain the substantive
question. Y-only permutation sensitivity is an expected property of indexed
quadratic loss, not itself an implementation bug.

[Decision] Keep all production/default guidance unchanged. Next gated work is
(1) controlled-backend performance/forward-graph optimization with matched
lightweight controls, (2) separate correspondence from corrupted-anchor bias
before proposing a replacement objective, and only then (3) applied paired
accuracy/development/confirmation. No new GPU work is authorized by this note.

## Raw archive acceptance and operator history

[Run] Drive/local raw package **163569211 bytes**, SHA256
`0f3fceb1aed50c0d23023a389439abf83955eec4174f45e11f2ba498b882e0e1`.
73 declared payloads plus its manifest; all CRC/SHA checks, seven inner runner
manifests,248source files/Python3.8 syntax, six physical ELF payloads, exact
prepared normalization frames, predictions/CSV counts and complete trajectories
passed local and independent review. Raw ZIPs live under
`result/modelnet40_c/decoded_physical_spectral_diagnostic/`; package and reports
under `result/preparation/gsd_physical_followup_20261104/`, with the matching
Drive root recorded above. Clean subset84715bytes/SHA256
`16db79b9c969e122d3c8277e61b0834a11ef90603aa189ea51632b108effddd9` was extracted
only after GPU research completion from the pinned30326912-byte clean asset.

[Run] Operator bookkeeping issue: the research phase printed its completion
marker with JSON on the same line, so monitor's exact-line check correctly
reported `marker_found=false`, despite CLI exit0. No experiment was rerun.
The following CPU collection phase re-read the complete on-disk receipt,
verified all seven raw archive identities, backed up the package and emitted its
exact `GSD_FOLLOWUP_ARTIFACTS_COMPLETE` marker at00:12:49 UTC (monitor COMPLETE).
An earlier empty-hash local control request was rejected before executing any
cell; its corrected submission launched the sole research phase. Both operator
events are preserved, without treating them as scientific run failures.

## Compact evidence

[P0 repeats](evidence/decoded_physical_followup_20261104/p0_repeats.json),
[P1 all96 directions](evidence/decoded_physical_followup_20261104/p1_transfer.json),
[P2 summary and identities](evidence/decoded_physical_followup_20261104/p2_summary.json),
[P2 all64 physical steps](evidence/decoded_physical_followup_20261104/p2_directions.csv),
[preserved V1 failure](evidence/decoded_physical_followup_20261104/p2_v1_failure.json),
[raw acceptance](evidence/decoded_physical_followup_20261104/local_acceptance.json).
CPU artifact math ran on macOS arm64/Python3.12.10/Torch2.15.0/NumPy1.26.4;
GPU runs used the earlier pinned Linux/Python3.8/Torch2.1.2+cu121 environment.
This backend/version difference is a possible source of reconstruction discrepancy,
not evidence that either eigensolver or the production implementation is incorrect.

## Final verification and publication

[Run/Code] Clean integration checkout `integrate/gsd-physical-followup-20261104`
accepted **500 CPU tests:499passed/1skipped,39.889s**; final incoming diff has no
whitespace errors. Scientific source/notes/compact evidence were merged and
normally pushed to `origin/thesis-integration` as
**45fad1db612f55dd9d3a488e353039bd56e7fb55**, based on7280318, from research source
head51d3465. The full incoming19-file diff was necessary; raw archives, setup and
operator wrappers, caches, ELF binaries, weights and unrelated main-worktree
edits were excluded from Git and preserved in their existing locations.

[Run] Derived archive **137396bytes**, SHA256
`ddfb569fadf162d13db31c0e082309616861a05b74db3c278fb41c90fce6f16f`, verified at
`/content/drive/MyDrive/thesis/result/gsd_physical_followup_20261104/derived/followup_derived_v1.zip`.
It includes V1/V2 raw CPU reports, source-recovery identities, numerical recounts,
compact evidence, executed analyzer sources, CPU test logs and primary streams.
The [final receipt](evidence/decoded_physical_followup_20261104/final_evidence_receipt.json)
records same-VM CPU audit00:26:41UTC: A100-SXM4-80GB, no active owned research
worker, raw and derived Drive hashes accepted, exact final marker observed.
Session `tta-gsd-physical-spectral-a100-20261103-r1` and monitor97932 remain open;
runtime teardown is not authorized. This metadata follow-up does not change code
or scientific conclusions and needs no repeated model/GPU tests.
