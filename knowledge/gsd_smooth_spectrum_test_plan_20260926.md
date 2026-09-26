# GSD Smooth Spectrum v2 Test Plan

**[Code/Run] Review/correction update:** The [2026-09-26 review](gsd_smooth_review_20260926.md)
recorded three blockers; all three are repaired. New persistent smooth-loss,
gradient, tolerance, protocol and launcher tests pass as part of the 111-test
full CPU suite after the second review. This does not validate GPU execution or accuracy.
Beta/alpha calibration and Colab smoke/pilot remain prerequisites.

**Status:** V2 implementation and command-matrix runner are present on `gsd-smooth-spectrum`, based on `gsd-development@9650770`. CPU formula/protocol checks pass; calibration, common-draw verification and Colab model evaluation remain pending. No model evaluation was performed locally. V1 uses its historical alpha=1; the two v2 coefficients and smooth beta are explicit inputs and must be fixed before evaluation.

**Method:** `gsd_latent_spectral_smooth_v2`; retain `gsd_latent_spectral_v1` as the historical comparator. The hypothesis and derivation are in [the mathematical decision](gsd_guidance_math_20260926.md#9-difference-from-the-current-gsd-code-and-next-test-case).

## 1. CPU mathematical and protocol plan

Use deterministic float64 synthetic coordinates and tiny graphs, so expected values can be calculated independently. These checks verify algebra and dispatch; they say nothing about ModelNet40-C accuracy or native CUDA operators.

| Case | Fixture | Expected behavior |
|---|---|---|
| Hard profile operator | Two vertices joined by weight `a=exp(-2)`; `k=1`, `delta=1`, `gamma=0`, `M=1` | The selected low eigenvector gives `F=0.5*[[1,1],[1,1]]`; loss equals `sum(E^2)/(3*N)` after projection, with original N in the denominator |
| Smooth profile operator | Same two-node graph; `mu=a`, scaled eigenvalues `[0,2]`, beta `2` | Weights `[1, exp(-4)]`; `F=U diag(weights) U^T`; effective spectral mass `1+exp(-4)` |
| Loss/gradient | Nonzero fixed residual E | `ell=<E,FE>/(3*N)` and `grad=2FE/(3*N)`; autograd and central finite differences agree within float64 tolerance |
| Basis sign/rotation invariance | Flip signs or rotate a repeated-eigenvalue basis | F, loss and gradient unchanged |
| Vertex permutation | Permute reference and prediction together | Loss invariant; gradient permutes with vertices |
| Isolates/empty graph | Threshold all edges or leave one active component | Removed rows have zero operator/gradient; empty graph gives finite connected zero loss |
| Invalid inputs | beta `<=0`, NaN/Inf, malformed XYZ, materially negative eigenvalues | Fail early with a useful error; never emit NaN/Inf artifacts |
| V1 compatibility | Existing v1 CLI/config and zero-weight route | v1 serialized contract stays unchanged; zero weight still bypasses graph construction and preserves RNG/output |
| V2 protocol | Missing profile/beta, invalid profile, v2 benchmark attempt, v2 flags on non-GSD | Reject unsupported config; accept only declared v2 smoke/pilot contracts |
| Artifact schema | Complete and failed synthetic protocol runs | Preserve seven-file contract, explicit v2 method/profile/beta/normalization and finite diagnostics |
| Launcher | Seeds 0/1/2, two pilot corruptions, three arms | Exactly 9 commands/run bundles; each covers both corruptions, for 6 per-corruption/seed cells per arm and 18 cells total; each seed has one unchanged v1, one v2 hard and one v2 smooth command |

CPU verification commands:

```powershell
python -m unittest tests.test_graph_spectral tests.test_graph_spectral_smooth tests.test_gsd_protocol tests.test_gsd_launcher tests.test_gsd_smooth_launcher -v
python -m unittest tests.test_gsd_trajectory -v
```

**Acceptance:** Each mathematical case must have an independently derived expectation. Compare the loss to the direct eigensystem formula, the gradient to `2FE/(3N)`, and synthetic per-mode weights to literals; do not use helper code under test to compute expected values.

## 2. Colab exploratory pilot

The pilot tests whether the smooth profile produces a useful cleaning direction at matched guidance scale. It is a pilot, not a final benchmark or a claim of improvement.

| Factor | Locked value |
|---|---|
| Dataset/corruptions | ModelNet40-C severity 5; Gaussian and Impulse |
| Seeds | 0, 1, 2; pair all arms by seed and common random draws where runner supports it |
| Batch / LION | 32; raw LION in eval mode; EMA disabled |
| Classifier | Frozen Point-MAE, same checkpoint/hash across all arms |
| 3DD-TTA host | lambda `.95`, gamma=eta `.01`, original 100-step DDIM with 5/35 reverse steps, original-style decoder, same preprocessing and FPS |
| Graph | v1 rules unchanged: k=10, delta=.1, graph gamma=.6, one reference graph per encoded sample |
| Arms | A: unchanged v1 hard-M; B: v2 hard-M with `3*N`; C: v2 smooth with `3*N` |
| M | 100 requested modes for A/B; v1 boundary expansion remains part of the historical arm |
| Beta / weights | Explicit in command/config; no implicit defaults. Smooth beta candidates are fixed before score inspection. Calibrate B and C's spectral coefficients to a predeclared equal aggregate gradient scale on calibration data, freeze weights before pilot evaluation. |
| Repetitions | 3 arms × 3 seeds = 9 complete run bundles; each bundle evaluates both corruptions, yielding 6 per-corruption/seed cells per arm and 18 cells total |

Arm A vs B reports the operational effect of normalization/refactoring plus any external coefficient change. It isolates the denominator only if alpha is held fixed; do not attribute a calibrated-alpha comparison solely to normalization. Arm B vs C is the profile comparison at calibrated aggregate scale. The calibration statistic and calibration data must be specified before launch; report local and style spectral/SCD gradient norms separately because one aggregate match may hide a route imbalance. The beta candidate set and selection rule must not be selected using these evaluation scores.

### Required measurements

- Macro accuracy per corruption and seed; paired percentage-point deltas for A/B/C.
- Summary mean and sample standard deviation across the three paired seed deltas; show all seed signs.
- Per-corruption paired differences, not only the two-corruption aggregate.
- Effective spectral weight mass `trace(F)`, active vertices/components, eigenvalue range, and any empty graph count.
- Spectral and SCD gradient norms for local and style routes before/after weighting; update norms and finite/zero counts.
- Graph/eigendecomposition time, total runtime, and peak GPU memory; record runtime type and package environment.
- Exact source commit, dirty status, CLI command, model/data hashes, and seven-file run bundle under `result/modelnet40_c/gsd_latent_spectral_v1/` for A and `result/modelnet40_c/gsd_latent_spectral_smooth_v2/` for B/C.

### Decision rule

First require all 9 bundles complete, finite, schema-valid, and on the same intended source/assets/config. A/B/C are not interpretable if graph, schedule, decoder, preprocessing, host or paired draw differs. A null/negative mean B-vs-C paired delta or unstable per-seed signs does not support accuracy promotion; retain v2 only as a mechanism/scale diagnostic or stop it. A positive pilot is exploratory and may justify a separately declared validation/confirmation design; it does not authorize tuning on or claiming an all-15 gain.

Do not start an all-15 evaluation until the pilot is ingested, the decision rule is reviewed, and beta/alpha are frozen independently of final benchmark scores.

Command matrix template (replace placeholders with predeclared calibration results):

```powershell
python eval_gsd_smooth.py --stage pilot --beta <BETA> --hard-weight <ALPHA_HARD> --smooth-weight <ALPHA_SMOOTH>
```

## 3. Run handoff checklist

For each run, save `command.txt`, `config.json`, `environment.txt`, `stdout.log`, `summary.csv`, `per_corruption.csv`, and `notes.md`. Use `result/README.md` schemas. Include calibration source/statistic, paired draw details, interruptions, OOMs and deviations. Preserve raw ZIPs unchanged when ingesting them.
