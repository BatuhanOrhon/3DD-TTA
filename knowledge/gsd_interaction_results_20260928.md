# Rebuilt GSD beta-rho interaction: accepted development evidence

[Run/Code/Inference/Open] Ingested 2026-09-28 on local
`gsd-smooth-spectrum@ee12d91`; all six runs record runtime commit
`c1c74672493284c323a66a558f31b57ca03c35f3`. No GPU run was launched here.
This supersedes the pending-ingestion state in the restart review/handoff.

## Evidence identity and acceptance

[Run] Canonical artifact directory:
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/`.
Reference ID: `20260928-113047_gsd-cal-diagnose-reference-seed0-n64`.
Raw reference config SHA-256:
`550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6`.

Exact six ZIP names in that directory:

```text
20260928-113047_gsd-cal-diagnose-reference-seed0-n64.zip
20260928-113124_gsd-rebuild-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-baseline-seed0-n128.zip
20260928-113155_gsd-rebuild-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-smooth-beta0p5-rho0p001-seed0-n128.zip
20260928-113242_gsd-rebuild-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-smooth-beta2p0-rho0p001-seed0-n128.zip
20260928-113330_gsd-rebuild-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-smooth-beta2p0-rho0p01-seed0-n128.zip
20260928-113417_gsd-cal-interaction-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-smooth-beta0p5-rho0p01-seed0-n128.zip
```

[Verification] Separate derived artifacts live in
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/analysis_20260928_113047/`:
`audit.py`, `audit.json`, `calibration_config.byte_identical.json`, and
`interaction_summary.recomputed.json`. The CPU-only audit checks all six
archives without extracting arbitrary paths. It verifies safe unique paths,
CRC, one root/seven files, run identities, completion, CSV counts/accuracy,
command/config agreement, logged corruption results, finite state metrics,
sample/step coverage, and common source/data/checkpoint identities.
The five development bundles additionally pass the existing strict analyzer.
The reference is validated by `load_calibration` on a byte-identical copy.
The analyzer was invoked with `--interaction-screen` and precisely the five
new ZIPs; its parsed output equals the supplied
`beta_rho_interaction_summary_20260928-113047_gsd-cal-diagnose-reference-seed0-n64.json`.

[Run] All five development configs reference the exact ID and config hash
above. All six record Gaussian/Impulse, severity5, batch32, seed0, split seed
20260927, raw/eval LION, EMA off, lambda .95, gamma=eta=.01, SCD weight1.
Counts are 64/corruption for diagnostics and 128/corruption for development.
Indices are unique, in range, deterministic, and preserve the diagnostic
prefix; identical indices across corruptions do not prove object identity.
Successful execution is `complete`; subset coverage correctly remains `partial`.

[Run] Within the block, pip-freeze lists, extension inventories (including
recorded binaries), scheduler, model-load/mode inventories, preprocessing,
resolved LION config and source/assets match. Environment: Python3.8.20,
Torch2.1.2+cu121, CUDA12.1, cuDNN8902, Diffusers0.11.1, Hub0.11.1,
NumPy1.24.4, SciPy1.9.1, A100-SXM4-80GB. Runtime Git is dirty with recorded
build/cache artifacts; it must not be described as a clean checkout.
cuDNN benchmark is on and deterministic algorithms are off. Seed-controlled
does not mean deterministic or common-draw paired.

[Open] `environment.txt` runtime settings retain the generic string
`file order; shuffle=False`; config randomness and selected indices instead
record the development shuffle. This is an artifact metadata inconsistency;
the explicitly recorded selected sample/step coverage passes. No raw evidence
was edited. The inventory hashes some package entry files rather than every
possible loaded binary, so this is recorded-environment agreement, not a full
machine-image identity proof.

## Accuracy and the interaction contrast

[Run] Equal-weight macro over Gaussian and Impulse, each with 128 examples:

| Condition | Gaussian correct | Impulse correct | Total correct | Macro % | Delta vs new SCD-only, pp |
|---|---:|---:|---:|---:|---:|
| SCD-only | 99 | 92 | 191/256 | 74.609375 | 0 |
| beta .5, rho .001 | 99 | 94 | 193/256 | 75.390625 | +0.781250 |
| beta .5, rho .01 | 100 | 91 | 191/256 | 74.609375 | 0 |
| beta 2, rho .001 | 98 | 91 | 189/256 | 73.828125 | -0.781250 |
| beta 2, rho .01 | 100 | 94 | 194/256 | 75.781250 | +1.171875 |

[Run] Increasing rho from .001 to .01 changes macro accuracy by -.781250 pp
at beta .5 and +1.953125 pp at beta2. Their difference, defined as the beta2
rho effect minus the beta .5 rho effect, is +2.734375 pp. Per-corruption
rho effects are (+.781250, -2.343750) pp for beta .5 and
(+1.562500, +2.343750) pp for beta2, in Gaussian/Impulse order; the interaction
contrasts are +.781250 and +4.687500 pp respectively.

[Inference] These are descriptive contrasts on one small development pool,
not significant interactions or independent confirmation. The best observed
condition exceeds SCD-only by three total correct predictions and the other
promising condition by one. Per-example correctness pairs are unavailable;
no paired classification significance test can be reconstructed from totals.

## Diagnostic acceptance and coefficient drift

[Run] Each diagnostic candidate has 384 local/style ratio observations
(64 examples x 3 probes x 2 corruptions), with zero missing denominators.
State records cover all 640 example-step pairs; recorded local DDIM and
local/style state near-zero flags are all false. Beta .5 local ratio
median/p90/max: .0001228477/.0001458231/.0001744185; beta2:
.0000884109/.0000975669/.0001112605. Full tails and stepwise summaries
are retained in `audit.json`.

| Quantity | Old report | New raw reference | Signed change | Relative change % |
|---|---:|---:|---:|---:|
| beta .5 median R | 0.00012286363268864894 | 0.00012284768768251798 | -1.5945006131e-8 | -0.01297781 |
| beta .5 alpha at rho .001 | 8.139104941932809 | 8.140161356429882 | +0.001056414497 | +0.01297949 |
| beta2 median R | 0.00008841542256107535 | 0.00008841088869931848 | -4.5338617569e-9 | -0.00512791 |
| beta2 alpha at rho .001 | 11.310243971397897 | 11.310823980075076 | +0.000580008677 | +0.00512817 |

[Run/Inference] The new rho .01 coefficients are 81.40161356429881 and
113.10823980075075. These measured differences exceed the original
`rel_tol=1e-12` equality gate, although they are small in relative magnitude.
This quantifies the supplied 113047 reference versus the old report; it does
not recover the exact earlier failed-comparison reference or prove its cause.
The original raw diagnostic ZIP remains missing; the new one is now complete.

[Run] Across reference steps0..3, local SCD/DDIM median ratios are .227--.265
for Gaussian and .200--.234 for Impulse. At step4 they rise to 2.469/2.332
(p90 3.156/2.909, maxima 4.057/3.691). Local SCD/state medians remain
.00144--.00178. No missing or near-zero DDIM denominator was found. A final-step
ratio above one is not by itself an overshoot diagnosis; retain SCD weight1.
Style SCD/state median ranges are .000021--.000718 and .000024--.001339;
the largest recorded style SCD/state ratio is .008440 (Impulse step1).

[Run] Actual guided median spectral/SCD ratios across all 640 sample-step
rows per corruption (these are already weighted):

| Condition | Local Gaussian | Local Impulse | Style Gaussian | Style Impulse |
|---|---:|---:|---:|---:|
| beta .5, rho .001 | .000968294 | .001032978 | .000910355 | .000651171 |
| beta2, rho .001 | .000986086 | .001011376 | .001144837 | .000737114 |
| beta .5, rho .01 | .009680235 | .010328967 | .009050270 | .006556164 |
| beta2, rho .01 | .009854074 | .010114729 | .011702590 | .007364740 |

[Inference] Local medians remain within 3.30% of target rho; no broken local
calibration scale is apparent. Style is not simultaneously matched: for
beta2/rho .01 its Gaussian/Impulse p90 is .024986/.016300 and maxima
.076052/.055580. Preserve this difference when interpreting profile effects.
Finite ratios and matching medians alone are not proof of trajectory stability.

[Run] Corruption runtime totals: SCD-only 11.734s, guided 28.654--28.733s.
Peak allocated memory: baseline14603 MiB, guided approximately21009--21012 MiB.
These instrumented subset timings exclude setup/model loading and should not
be extrapolated to all corruptions as a benchmark cost.

## Historical comparison and remaining uncertainty

[Run] The old beta2/rho .001 result was 100/93 versus old baseline98/92;
the new values are 98/91 versus new baseline99/92. Thus its relative macro
effect changes from +1.171875 to -.781250 pp. Source/data/checkpoint manifests
and selected host fields match across these blocks, but environment identity
does not: NumPy1.21.2 -> 1.24.4, SciPy1.8.0 -> 1.9.1, plus many other package
changes and different hashes for the recorded Chamfer, Chamfer3D and PointNet2
extension binaries. Both blocks have nondeterministic CUDA settings.

[Open] Changes in coefficients, packages/binaries and stochastic numerical
execution are confounded. The observations do not isolate which caused the
prediction/count differences; even different binary hashes do not prove
different kernel behavior. Do not pool the blocks as matched replicates,
attribute the change to NumPy/eigensolver/CUDA, or rebuild the working environment.

## Next bounded development comparison (registered proposal, not executed)

[Inference/Decision] Accept this calibration for continued exploratory
development. Keep two promising conditions: beta .5/rho .001 and beta2/rho .01,
alongside a matched SCD-only control. Use 512 examples per corruption,
seeds0/1/2 and unchanged split seed20260927: three arms x three seeds = nine
new development runs. 512 is within the existing runner's supported limit.
This economically narrows the screen; it cannot reconfirm the whole 2x2
interaction or establish that the omitted conditions are inferior.

[Plan] Reuse the exact 113047 raw reference config and its accepted alpha
values for every model seed. Preserve batch32, severity5, Gaussian/Impulse,
lambda .95, gamma=eta=.01, raw/eval LION, EMA off and SCD weight1. Use the
existing direct `run_baseline.py --gsd-stage development` path with count512
and the reference; do not call the from-scratch launcher, refit calibration
per seed, edit recorded sources, or mix in the 128-example baseline.
Archive all nine complete seven-file ZIPs according to `result/README.md`.

[Plan/Falsifier] Primary report: same-seed candidate minus SCD-only macro
percentage-point deltas, all three deltas, their mean and sample SD,
per-corruption counts/deltas, actual local/style ratios, runtime and memory.
Positive mean and positive deltas at all three seeds would support advancing
a candidate to separately designed confirmation, without establishing
statistical significance. A nonpositive mean, mixed seed signs or material
corruption regressions weaken the claim and prevent an automatic winner/freeze.
Do not increase the grid or change selection rules after inspecting these
results. If both candidates remain close, retain both as unresolved rather
than picking the largest single-seed percentage.

[Open] The 512 pool contains earlier examples; it remains development data.
Object-disjoint confirmation, cross-corruption correspondence, native versus
instrumented SCD CUDA prediction/RNG parity, and per-example prediction
evidence remain separate unresolved gates. No new GPU work has been started.
