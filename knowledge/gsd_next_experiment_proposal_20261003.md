# Next experiment proposal: separate diffusion, SCD and spectral effects

> **Superseded execution recommendation, 2026-10-03:** The user requests
> reuse of completed tests. The [archive inventory and correction](gsd_existing_runs_audit_20261003.md)
> confirms SCD-only/all15 and substantial spectral-only/combined coverage.
> Do not execute the historical 12-new-run proposal below. The minimal new
> main condition is unguided diffusion, seeds0/1/2 (three all15 runs), using
> existing controls with their pairing/provenance limits. Retain this document
> as the scientific comparison design, not the current execution queue.

[Inference/Proposal/Code] 2026-10-03, `gsd-smooth-spectrum` at
`4be6afd86629f3d52648fb543fc2eb350a5f9f1d`. User requested a recommendation
and useful missing evidence that can be produced in parallel. This document
registers a proposal; no launcher implementation or Colab execution occurred.

## Primary question and fixed four-arm comparison

[Run/Inference] Spectral-only v1 preserves Gaussian/Impulse performance and
has a recorded +.2296 pp mean versus its separate SCD pilot controls. This
does not prove that the spectral increment is responsible: a matched unguided
diffusion arm is missing. Pure VAE has no diffusion and cannot replace it.
The smooth SCD-plus-spectral null does not refute the spectral-only result.

[Proposal] All 15 ModelNet40-C severity-5 corruption files, including Background,
2,468 examples each, seeds 0/1/2. Four conditions x three seeds = 12 full runs.
Preserve the current host: raw/eval LION, EMA off, original final style,
batch32, lambda .95, gamma=eta=.01, 100 DDIM schedule, normal/background
reverse steps5/35, same checkpoints, preprocessing and classifier.

| Arm | SCD coefficient | Spectral coefficient | Spectral definition |
|---|---:|---:|---|
| A: unguided diffusion | 0 | 0 | None; same encoding, noising and reverse trajectory setup |
| B: SCD-only | 1 | 0 | None |
| C: spectral-only | 0 | 1 | Existing v1 hard-M100, actual-rank normalization |
| D: SCD + spectral | 1 | 1 | Exactly the same v1 term as C |

[Proposal] V1 is chosen to reproduce the positive spectral-only observation;
do not silently replace it with smooth v2 or alter its graph/rank policy.
This is the recent archived v1, not the unverified historical dev M400/mid-band
configuration. Do not retune coefficients after inspecting these outcomes.

[Proposal] Compare C-A (spectral contribution without SCD), B-A (SCD
contribution), D-B (spectral contribution with SCD), C-B (replacement), and
(D-B)-(C-A) (descriptive interaction). Report per-seed and per-corruption
paired differences, mean/sample SD, paired uncertainty and corrected-versus-
broken predictions. Overlapping intervals or tiny differences do not prove
equivalence. Positive C-A but no D-B would support limited incremental benefit
in SCD's presence; D below B and C would motivate an interference investigation.
Mixed corruption patterns motivate a mechanism study, not automatic tuning.

## Pairing and missing observations

[Proposal] Replay identical preprocessing/interpolation outputs, encoded
latents and diffusion-noise draws across arms for each seed/example or batch.
Use the same input order, batching and checkpoint/runtime identities. A seed
alone is not proof of shared random draws or deterministic CUDA execution.
Verify pairing using captured inputs/draw identities. Keep all four arms for
a seed on the same runtime/device; independent runtimes may handle separate
seed blocks if assets/software identities are verified and hardware is logged.
Do not assign each method exclusively to a different GPU/environment.

[Proposal] Record original sample index, corruption, seed, predictions and
logits (or margins), labels only for evaluation, local/style raw and weighted
gradient scales, cosine/negative fractions, update-to-state/DDIM ratios, and
runtime/memory. Preserve the standard seven-file bundle; attach per-example
arrays as a separately named sidecar with a recorded hash rather than silently
breaking existing strict ZIP readers. Aggregate diagnostic summaries should
cover all samples; bulky tensors may use a fixed declared subset.

## Parallel evidence, prioritized

1. [Proposal] Shared-state mechanism diagnostic on 64 fixed examples from each
   of all 15 corruptions, seed0 (960 examples; not another full accuracy run).
   Use common SCD-reference states and compare v1 hard, v2 hard and smooth
   beta .5/2. Record raw and matched-update-norm virtual directions separately;
   never apply all candidates to one live reference trajectory. Measure
   direction cosines between candidates, local/style conflict, spectral mode
   contributions, actual graph components/degree, reference displacement and
   decoded effects on a bounded declared capture set. This addresses direction
   and graph quality without opening another accuracy-based parameter grid.
   Smooth-versus-hard accuracy causality remains open after this diagnostic.
2. [Proposal] Recover the successful historical legacy command/config/logs and
   per-corruption bundle. Only then define original-host, legacy-host spectral-off
   and legacy-host spectral-on controls. Defaults alone are insufficient to
   recreate the user's old result. This is independent of the main four-arm
   study and currently has an evidence dependency, not a runnable job.
3. [Proposal; lower priority] Later append a fixed smooth-only and the identical
   smooth-plus-SCD condition to the same paired block. A possible operational
   setting is the already tested beta .5/rho .001 coefficient, retained from
   reference113047 rather than recalibrated with SCD disabled. This would add
   six full runs if contemporaneous A/B controls and common draws can be reused;
   otherwise matching controls must also be repeated. Comparison with v1 is a
   locked-setting method comparison, not a pure profile effect. A pure profile
   claim needs v2 hard-versus-smooth at matched normalization and declared scale.

[Decision] Prioritize the 12-run factorial plus parallel frozen-state diagnostic.
The main block supplies the missing no-guidance baseline, all-15 spectral-only,
three-seed coverage and Background evidence together. Source-only, preprocessing
identity, pure VAE, EMA and lambda screens need no wholesale repetition for
this question. These previously inspected test files provide mechanistic
development evidence; future independent confirmation remains a separate step.

## Implementation readiness

[Code] `tta_gsd.py` rejects spectral_weight=0 with scd_weight != 1; the usual
zero-spectral branch delegates to original SCD. `gsd_protocol.py` locks v1
benchmark SCD weight to1, while spectral-only's dedicated ablation stage
excludes Background. Smooth full_dataset_development also requires SCD1.
The proposed A/C all-15 conditions therefore require an explicit isolated
ablation protocol and truthful metadata. No valid ready-to-run command for
the complete four-arm experiment is claimed at this revision.

[Proposal] Before full execution, verify the unguided arm applies zero guidance
and retains diffusion; establish SCD-only parity with the existing path;
check shared starting tensors/draws, prediction logging and a Background smoke.
These are correctness checks, not accuracy-based promotion or parameter tuning.

Sources: `gsd_smoothing_diagnosis_20261001.md`,
`modelnet40_c_all15_results_matrix_20261001.md`,
`findings_log.md` spectral-only pilot entry, `tta_gsd.py`, `gsd_protocol.py`,
and [PyTorch 2.1 reproducibility](https://docs.pytorch.org/docs/2.1/notes/randomness.html).

## 2026-10-03 implementation update after explicit user request

[User decision/Code] The old 12-run block and implementation-readiness section
above are superseded by the audited two-condition plan. Implemented unguided
diffusion and smooth-only beta=.5/alpha8.140161356429882, three full seeds each.
Reuse existing controls; do not establish new common-draw pairing by assertion.
See [Colab handoff](gsd_guidance_ablation_handoff_20261003.md). GPU outcomes pending.
