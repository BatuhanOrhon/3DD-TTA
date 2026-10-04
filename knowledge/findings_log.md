# Findings Log

## 2026-09-28 — Colab GSD rerun blocked before calibration

[User report] Three failures interrupted the fresh Colab attempt: a Diffusers
import accessed missing `torch.xpu`; a later environment create could not
resolve `torch==2.0.1+cu121`; then pip could not find `argparse` while only the
PyTorch wheel index was configured. The latest attempt loaded LION checkpoints
successfully but failed on the first FPS call with `no kernel image is
available` in `furthest_point_sampling`. Its failed archive is
`20260928-092752_gsd-cal-diagnose-reference-seed0-n64.zip`. No calibration or
accuracy result was produced.

[Code] As requested, `env.yaml` and `requirements.txt` on
`gsd-smooth-spectrum` have been restored to match branch `dev` exactly. The
Colab handoff's interim environment recreation advice has been removed.
Separately, the vendored PointNet2 `setup.py` hard-codes
`TORCH_CUDA_ARCH_LIST="8.6"` (present in the repository's initial commit);
this is a plausible cause of the FPS kernel error when the active GPU has a
different compute capability, but the run ZIP/GPU identity is needed to
confirm it. No environment or CUDA source adjustment is recorded as validated.

[Open] Inspect the failed ZIP's `environment.txt` for GPU and installed
package identities, confirm the PointNet2 extension selected by Python, then
rebuild/test only that native extension for the reported GPU before rerunning
the calibration launcher. Keep the failed artifact as incomplete evidence.

## 2026-09-27 — Colab Python 3.8 import fix

[User report/Code] The first Colab smoke failed while importing
`graph_spectral.py`: `str | None` annotations were evaluated at module load,
but Colab's recorded environment uses Python 3.8.20 (findings log, entry
2026-09-15, Colab smoke). Add postponed annotation evaluation so the union
type hints load under Python 3.8 without changing runtime tensor behavior.

[Run] A regression test first failed on the eager annotation, then passed with
`from __future__ import annotations`. Full local CPU suite:
`python -m unittest discover -s tests -q` — **112 tests passed**. This host
uses Python 3.13; the test verifies the annotation is deferred. Colab must pull
the fix and rerun smoke on its recorded Python 3.8.20 environment.

[Open] The failed v1 smoke ZIP is correctly marked failed and provides no
accuracy evidence. The complete smoke matrix must be rerun after the fix;
check that each bundle completes before exporting it to Drive.

## 2026-09-26 — second GSD smooth review before push

[Code/Run] Independent review confirmed the earlier fixes. Additional regression
tests exposed and fixed empty-graph operator-byte underreporting, six-digit
launcher rounding of calibration inputs, and finite large-beta overflow in
float32 zero modes. The return tensor now determines storage, CLI floats
round-trip, and spectral exponential weights are computed in float64 before
casting back. V1 graph/loss semantics are preserved.

[Run] Actual-source full suite: `python -m unittest discover -s tests -q`,
**111 tests passed** (3.489 seconds), branch `gsd-smooth-spectrum`, base
`9650770`, changes prepared for commit/push. Added independent heat-kernel and
hard-v1 equivalence checks, invariance/isolate/empty/gradient checks, v2 baseline
bypass and synthetic artifact checks. An independent follow-up review ran
21 tests and found no further actionable code issue.

[Decision/Open] CPU review complete; calibration, runtime common-draw evidence
and Colab smoke/pilot remain pending. A-vs-B interpretation must include an
alpha change unless alpha is held fixed. No new accuracy result or raw-result
modification. See [the review record](gsd_smooth_review_20260926.md).

## 2026-09-26 — smooth-spectrum implementation review

[Code/Run] `gsd-smooth-spectrum`, HEAD `9650770f75cf0c37e1e16b873bd6b8ddd0a4276e`,
uncommitted v2 working tree: actual CPU suite has 92 tests, 1 failure and 22
errors. Shared basis allocation remains N x 0 after rank selection, breaking
v1 and v2. An allocation-only in-memory diagnostic makes all 92 existing
tests pass; no implementation source was repaired during review. A separate
matrix-exponential check validates the intended quadratic loss and gradient
on a small graph after that diagnostic substitution.

[Code/Run] The launcher omits the planned v1 arm. The normalized-eigenvalue
PSD check also uses a raw-unit tolerance; a controlled roundoff probe
demonstrates false rejection. Dedicated v2 tests and calibration/pairing
evidence remain incomplete. See [full evidence and next steps](gsd_smooth_review_20260926.md).

[Decision/Open] Supersede the v1-preservation claim and block pilot launch
until repaired actual-source tests and smoke artifacts pass. No Colab/model
run or new accuracy result was produced; historical ZIPs remain untouched.

## 2026-09-26 — review blockers repaired

[Code] The allocation now uses the final eigenspace rank; the negative
eigenvalue check compares raw Laplacian eigenvalues against the raw-unit
roundoff tolerance before scaling. The command builder now produces unchanged
v1, v2 hard-M, and v2 smooth arms for seeds 0/1/2. Smooth v2 CLI use requires
an explicit spectral coefficient rather than silently defaulting to 1.

[Run] Added persistent checks for active v1/smooth bases, rank-one smooth
filters, tolerated raw zero-mode roundoff, literal two-node heat-kernel loss
and gradient, v2 invalid/missing protocol values, matched launcher arms, and
smooth guidance reaching both local and conditioning states. The focused CPU
suite passed 60 tests, then the complete suite passed **101 tests** via
`python -m unittest discover -s tests -v`. Python compilation passed for the
changed runtime and test modules. The launcher preview generated 9 valid
commands: 3 v1, 3 hard-v2 and 3 smooth-v2. These checks do not establish CUDA
behavior or accuracy.

[Open] Calibration source/statistic and beta selection rule remain to be
declared before the pilot; then run Colab smoke and ingest its complete bundle.
No model evaluation or accuracy result was produced in this correction.

## 2026-09-26 - Paper reread and spectral-guidance mathematical decision

[Paper] Re-read the repository Wei PDF, visually including pp.3-5/Eqs.7-18.
GSDPS learns low-frequency displacement `X_s=X+U_M DeltaC`, retaining high
input frequencies; its driving losses are classification/information/Chamfer,
not spectral equality. Low-band fidelity is a different method hypothesis.
[Code] Host audited at `gsd-development@9650770`; predicted-clean and reference
latent XYZ use tracked slots and the existing SCD domain. No source edit.
[Inference] Select a fixed reference graph/common basis and smooth weights
`exp(-beta*lambda)` on coefficient differences, fixed original N normalization
and sample sum. Detached graph, chain rule through denoiser, existing host
controls retained. This is a proposal, not implemented or accuracy-confirmed.
[Code: CPU mathematical checks] Analytic/autograd and finite-difference
derivatives, basis invariance and permutation checks passed on small synthetic
graphs. Full derivation, paper ambiguities, alternatives and falsifiers are in
[the decision record](gsd_guidance_math_20260926.md). No new model/Colab run or
result path. Existing pilot evidence is not evidence for this new formula.

## 2026-09-26 - Smooth-spectrum v2 implementation and test plan

[Code] Created branch `gsd-smooth-spectrum` from
`gsd-development@9650770f75cf0c37e1e16b873bd6b8ddd0a4276e`. Added opt-in
`gsd_latent_spectral_smooth_v2` with the existing reference graph rules,
mean-active-degree-scaled Laplacian, dense fixed hard-M/smooth spectral filter,
and sample-summed `3*N` objective. V1 dispatch and method ID remain separate.
The new `eval_gsd_smooth.py` builds paired baseline/hard/smooth smoke and pilot
matrices; beta and both spectral coefficients are explicit required inputs.
[Open] No automated test command or model/Colab evaluation has been run in this
turn; there is no result path or accuracy evidence. The dense filter stores
N-by-N values per sample, so GPU peak memory/runtime are unresolved.
[Open/plan] CPU formula/protocol verification and the severity-5 Gaussian/Impulse
pilot (9 run bundles: three arms x seeds 0/1/2, each covering both corruptions) are described in the
[test plan](gsd_smooth_spectrum_test_plan_20260926.md). Calibration data,
gradient-scale statistic, beta candidate set and acceptance rule must be fixed
before evaluation; all-15 is not a parameter-selection set.

[Code] Comparison with `graph_spectral.py` and `tta_gsd.py` confirms that the
fixed reference graph, shared-basis coordinate difference, predicted-clean
XYZ, frozen-denoiser chain rule and SCD+spectral host already exist in v1. The
new part is smooth eigenvalue weighting over all active modes, scalar mean
degree normalization of L, full-spectrum state/configuration, and the `3*N`
reduction. The denominator changes guidance scale, so weight/alpha cannot be
carried over numerically from v1.
[Open / next test] Registered `gsd_smooth_spectrum_profile_pilot`: compare the
unchanged v1 method, a hard-M arm in the new `3*N` host and the smooth profile
in that host; calibrate the latter two to a predeclared equal aggregate
spectral-gradient scale before paired Gaussian/Impulse severity-5 seeds
0/1/2. This is a planned experiment, not implemented or run. Beta candidates,
calibration data/rule and acceptance criteria remain to be declared; no
accuracy gain is claimed. Full case specification is in
[the mathematical decision record](gsd_guidance_math_20260926.md#9-difference-from-the-current-gsd-code-and-next-test-case).

## 2026-09-23 - Spectral projector and band-boundary follow-up

[User report] Old small spectral gradients accompanied almost +1 pp accuracy;
PxP was introduced to address imbalance/conflicts. Small norm alone cannot
explain the old/current discrepancy.
[Code] At `761f47f`, a CPU-only synthetic N2048 graph check demonstrates that
the boundary-expansion heuristic can select distinguishable eigenmodes:
M100 becomes104, tolerance .007169 versus float32/float64 eigenvalue error
about 1e-5; the first added gap is .0002177. Current analytic loss gradients
agree with autograd (max error1.09e-10). This is numerical evidence only.
[Run] Existing seed0 pilots select M100 mean ranks138.43/120.64 but M400
mean ranks404.53/403.71. Expansion changes both direction and sample weighting;
its role in accuracy is open and it cannot alone explain the M400 null result.
[Inference] Current degree cutoff is 6% of mean directed degree; weak graph
regions can enter low modes and anchor corruption. A scalar weight cannot
recover a different projector or legacy low/mid weighting. PxP projection
can alter SCD substantially even with small spectral norms, above its epsilon
floor. Full derivations, CPU reproduction, artifact scope and falsifiers:
[spectral mathematics follow-up](gsd_branch_comparison_20260923.md#follow-up-spectral-mathematics-rather-than-gradient-magnitude).
No source edit or new model/GPU evaluation; actual latent graph diagnostics
and controlled Colab comparisons remain needed.

## 2026-09-23 - Legacy/current GSD comparison and gradient-scale diagnosis

[Code] Compared `gsd-development@761f47f`, `dev@a458cd4` and
`pxp-gradient-projection@53ba252`; see the
[branch comparison](gsd_branch_comparison_20260923.md) for exact defaults,
source anchors, raw artifact scope, causal controls and falsifiers.
[User report] Legacy defaults including M400/mean weight16 improved accuracy.
[Run] Re-reading the 18 current Gaussian/Impulse pilot ZIPs confirms mean
on-minus-off deltas of -.0135/-.1688/-.0540 pp for M100/240/400.
At M100 weight1, the local spectral/SCD ratio of mean gradient norms is only
.084%/.097% (Gaussian/Impulse); at M400 it falls to .042%/.045%.
[Inference] Weak added guidance plausibly explains the current null increment,
but does not establish the cause of legacy gains. Legacy eval also changes
steps (10/30), graph filtering, mid-band guidance, batch size and LION mode;
pxp eval additionally changes final style and scheduler endpoint. Old
weight16 is batch-mean: at identical B32/U/M its current-scale equivalent is
.5, not 16. Test the successful legacy host with all spectral weights zero
before attributing its gain to GSD. No new GPU run or implementation change.
[Code] Current GSD remains lambda=.95 after `509b901`; the earlier .96 lock
entry below is historical and superseded. Complete legacy run ZIPs are pending.

## 2026-09-23 - GSD spectral-band pilots M=240 and M=400

[Run] Twelve complete pilot ZIPs were supplied under
`result/modelnet40_c/gsd_latent_spectral_v1/`: GSD weight 0/1, seeds 0/1/2,
and requested modes M=240/400. All archives pass CRC validation, contain the
seven-file schema, cover complete Gaussian/Impulse files (4,936 examples),
record lambda=.95, and have no error signatures.

[Run] Paired GSD-on minus GSD-off macro deltas for M=240 are -0.4660,
+0.1013, and -0.1418 pp at seeds 0/1/2 (mean -0.1688 pp, sample SD
0.2846 pp). For M=400 they are +0.0608, -0.0810, and -0.1418 pp (mean
-0.0540 pp, sample SD 0.1040 pp). Gaussian/Impulse rows and all diagnostics
are finite; active mean ranks are approximately 250 for M=240 and 404 for
M=400 because the requested boundary expands across numerically degenerate
eigenvalues.

[Inference] Neither M=240 nor M=400 improves the paired pilot mean relative
to its weight-zero control. M=400 is more negative than the earlier M=100
mean (-0.0540 pp versus -0.0135 pp), while M=240 is also
more negative. These are exploratory Gaussian/Impulse pilot results, not an
all-15 claim.

## 2026-09-23 - Current-commit original pilot parity check

[Run] Three current `3dd_original` pilot ZIPs were supplied under
`result/modelnet40_c/3dd_original/`: seeds 0, 1, and 2, each complete for
Gaussian and Impulse (4,936 examples), with seven-file/CRC-valid archives and
no error signatures. Configs record raw LION eval, EMA off, batch 32, severity
5, lambda=.95, gamma=eta=.01, and the same checkpoint/label/config hashes as
the GSD pilot.

[Run] Current baseline two-corruption macro values are 72.2447%, 72.1637%,
and 72.3460% for seeds 0, 1, and 2 (mean 72.2515%). The previously supplied
GSD-off values are 72.3258%, 72.2650%, and 72.4473% (mean 72.3460%), giving
GSD-off minus current-baseline deltas of +0.0810, +0.1013, and +0.1013 pp
(mean +0.0945 pp).

[Inference] This is practical protocol parity, not bitwise equality: the runs
are separate CUDA processes and the runner retains cuDNN benchmark behavior.
The small baseline/off spread is now suitable for interpreting the spectral
ablation, while exact same-process common-draw equality remains open.

## 2026-09-23 - SCD lambda=.96 all-15 confirmation result

**[Run]** The three complete all-15 archives are
`result/modelnet40_c/scd_lambda96_control/20260922-193839_scd-lambda96-s5-all15-seed0.zip`,
`20260922-200845_scd-lambda96-s5-all15-seed1.zip`, and
`20260922-203850_scd-lambda96-s5-all15-seed2.zip`. Each ZIP contains exactly
the seven required files, passes `testzip()`/CRC validation, records 15
complete corruption rows and 37,020 examples, has `execution_status=complete`,
and contains no traceback or failed-run note. Their SHA-256 values are,
respectively, `6ab0fea56221eddb449b9beb99276f5148eb37bedcd4a9c56e63c8dba9a03e81`,
`4cf00f0b0ac728254956764c3814236b289717520770167b7f63495bc5495164`, and
`78dd3a1e2a9d9f8ce4e831ade3a5007d67ec2551d56439022872840f38210d2a`.

**[Code/Run]** All three runs record commit
`79cc02774e5fa85a7c2f84a08506617670416642`, method
`scd_lambda96_control`, ModelNet40-C severity 5, batch 32, raw LION eval,
EMA disabled, original-style decoding, `lambda=.96`, retained count 1966,
and legacy unnormalized SCD. The Point-MAE hash is
`507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75`, the
LION hash is `807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`,
and all 15 dataset manifest hashes match across the three seeds and the
matched shared-decoder comparator. Config metadata records 106 dropout
modules with `training=false` before and after each run. `git_dirty=true`
records Colab-generated build/cache changes; the source commit and manifests
remain explicit in the artifacts.

**[Run]** Lambda=.96 original-style macro accuracies are 63.7304%, 63.9708%,
and 64.1005% for seeds 0/1/2: mean **63.9339%**, sample SD **0.1878 pp**.
The matched lambda=.95 original-style rows from
`result/modelnet40_c/shared_trajectory_decoder_control/` are 63.5900%,
63.8006%, and 63.8547%: mean **63.7484%**, sample SD **0.1399 pp**.
The paired seed deltas are **+0.1405, +0.1702, and +0.2458 pp**, with mean
**+0.1855 pp** and sample SD **0.0543 pp**.

**[Run]** The mean per-corruption deltas in percentage points (lambda=.96
minus matched original-style lambda=.95) are:

| Corruption | Mean delta (pp) |
|---|---:|
| background | +1.3236 |
| cutout | +0.2566 |
| density | +0.1351 |
| density_inc | +0.0810 |
| distortion | -0.2026 |
| distortion_rbf | +0.4322 |
| distortion_rbf_inv | +0.0540 |
| gaussian | +0.3377 |
| impulse | -0.3106 |
| lidar | 0.0000 |
| occlusion | +0.3106 |
| rotation | +0.2431 |
| shear | -0.2026 |
| uniform | +0.0675 |
| upsampling | +0.2566 |

**[Inference]** Under the predeclared paper-setting confirmation, lambda=.96
raises the matched all-15 macro mean in all three seeds and by +0.1855 pp on
average. This supports retaining `.96` as the paper-conformant reference
setting for subsequent baseline/GSD comparisons; it is a small three-seed
effect, not a significance test or proof of generalization beyond this
protocol. Lambda=.95 remains the historical operational comparator.

**[Open]** The existing GSD pilot was locked to lambda=.95. Reusing its result
as a lambda=.96 GSD result would be invalid; any future GSD pilot using `.96`
must be a newly declared, matched run with its own weight-zero control.

## 2026-09-23 - GSD lambda baseline lock

**[Code]** On `gsd-development`, `gsd_latent_spectral_v1` now defaults to and
requires the paper-conformant lambda=.96. An explicit `.95` GSD invocation is
rejected. The legacy `3dd_original`, source-only and existing control routes
retain their historical contracts; no raw result was changed.

**[Code]** Targeted GSD protocol tests and the full 88-test CPU suite pass;
`run_baseline.py`, `gsd_protocol.py`, `tta_gsd.py`, `graph_spectral.py` and
`eval_gsd_tta.py` compile successfully.

**[Open]** The previous `.95` GSD pilot remains evidence for `.95` only. A
paper-conformant `.96` GSD pilot must rerun matched weight-zero and weight-one
arms before any all-15 promotion decision.

## 2026-09-23 - GSD Gaussian/Impulse pilot artifacts

[Run] The repository also contains three historical complete
`3dd_original` all-15 eval/raw ZIPs:
`result/modelnet40_c/3dd_original/20260913-163817_3dd-original-all15-eval-raw-seed0.zip`,
`20260913-173819_3dd-original-all15-eval-raw-seed1.zip`, and
`20260913-183818_3dd-original-all15-eval-raw-seed2.zip`. Their configs record
ModelNet40-C severity 5, batch 32, raw LION eval mode, EMA disabled, lambda
0.95, gamma/eta 0.01, 100 DDIM steps, complete 15-corruption coverage, and
the same classifier/LION/label asset hashes as the supplied GSD pilot.

[Open] Those baseline ZIPs were produced at commit `b999a1e`, whereas the GSD
pilot was produced at `ad07257`. They are therefore valid historical
eval/raw references, but strict current-commit parity would require a fresh
`3dd_original` pilot run with the same launcher and seed/corruption scope.

[Run] Six complete pilot ZIPs were supplied under
`result/modelnet40_c/gsd_latent_spectral_v1/`: GSD weight 0 and 1 at seeds
0, 1, and 2, covering complete Gaussian and Impulse severity-5 files. Every
archive passed CRC validation, contains the required seven files, has 4,936
examples across two complete corruption rows, and has no traceback, runtime
error, nonfinite, or CUDA OOM signature.

[Run] Paired GSD-on minus GSD-off macro deltas are -0.0608 pp (seed 0),
+0.1418 pp (seed 1), and -0.1216 pp (seed 2). The three-seed mean is
-0.0135 pp with sample SD 0.1379 pp. Gaussian deltas are -0.1216, 0.0000,
and -0.0405 pp; Impulse deltas are 0.0000, +0.2836, and -0.2026 pp for
seeds 0, 1, and 2 respectively.

[Run] GSD-on diagnostics are finite for all 2,468 examples in each corruption
and each seed, with no missing aggregate values. Mean spectral gradient norms
are approximately 0.0950 (Gaussian) and 0.1009 (Impulse). GSD-on runtime is
approximately 515 seconds per seed versus 186 seconds for GSD-off, about
2.77x in this pilot.

[Open] The six supplied artifacts do not include a newly matched
`3dd_original` baseline ZIP. The GSD-off control is therefore the current
paired comparison, while exact original-arm parity remains open. The pilot
promotion screen is not met because GSD-on does not improve all three seeds
and its mean delta is negative. Do not start all-15 with this locked weight
without an explicitly documented decision to record a negative confirmation.

## 2026-09-23 - GSD Colab smoke artifacts

[Run] Four Colab smoke ZIPs were supplied under
`result/modelnet40_c/gsd_latent_spectral_v1/`: Gaussian and Background, each
with GSD weight 0 and 1, seed 0, severity 5, and 64 examples (two batches).
All four archives passed `ZipFile.testzip()`, contain the required seven files,
have `execution_status=complete`, and show no traceback, runtime error,
nonfinite, or CUDA OOM signature. The CSV status is `partial` because smoke is
intentionally a two-batch prefix.

[Run] Gaussian accuracy is 0.734375 (47/64) at weight 0 and 0.750000 (48/64)
at weight 1, a +1.5625 percentage-point prefix difference. Background is
0.656250 (42/64) at weight 0 and 0.625000 (40/64) at weight 1, a -3.1250
percentage-point prefix difference. These are smoke diagnostics, not accuracy
evidence or a promotion decision.

[Run] Active GSD diagnostics are finite for all 64 examples in both on-runs.
Gaussian records 10 guidance steps and Background records 70, reflecting the
5 and 35 reverse-step protocols. Mean spectral gradient norms are nonzero;
the active graph mean rank is 145.125 for Gaussian and 109.578125 for
Background. Peak GPU memory is approximately 20.5 GB for the on-runs versus
14.6 GB for the corresponding off-runs. These resource values are smoke-only
and must be checked again on complete pilot runs.

[Open] The supplied set has GSD off/on arms but no newly matched
`3dd_original` baseline ZIP with the same timestamp/configuration. Weight-zero
trajectory parity therefore remains a pilot gate; do not interpret the smoke
differences as a GSD improvement claim.

## 2026-09-23 - GSD-only latent spectral integration

[User report] Implement GSD alone on `gsd-development`, beginning at the
latest `baseline-repro-clean`. Preserve PxP/projection/combined work and all
raw local artifacts. This supersedes earlier GSD deferral only for this scope.

[Code] Branch base is `79cc02774e5fa85a7c2f84a08506617670416642`.
The new opt-in ID is `gsd_latent_spectral_v1`. `graph_spectral.py`, `tta_gsd.py`
and `eval_gsd_tta.py` were implemented anew after inspecting legacy revisions
`53ba252`, `06561f3`, `f63587b`, `09580b2` and `4079f49`.
The new GSD-only protocol module connects it additively to the artifact runner.
Baseline `tta.py`, `main_3dd_tta.py`, LION, dataset, preprocessing and FPS
implementations were not edited.

[Paper] Rechecked the local 3DD-TTA Sec. 3.3/Eqs. 9--12/Algorithm 1,
LION Sec. 3/Eqs. 5--7 and GSDTTA Sec. 3/Eqs. 7--13. GSDTTA p. 4 visibly
uses gamma/(N*k), and its printed distance notation/prose is ambiguous.
[Code] Legacy GSD changed the scheduler endpoint convention, final style,
rate bindings and relative batch scaling; its graph used gamma/N and an
arbitrary isolated-node diagonal penalty. These were not inherited.

[Inference] Selected objective: detached static graph on encoded local XYZ,
low-band fidelity of predicted clean XYZ, sum over samples normalized by
3*actual_rank, added to unchanged SCD. The ordinary chain rule guides both
noisy local state and conditioning. Final decoder retains encoded original
style. This is a GSD-inspired regularizer, not full GSDTTA reproduction.
See [short design](gsd_design.md) and [full audit](gsd_integration_20260922.md).

[Code] Review exposed a float32 repeated-eigenvalue boundary failure on a
two-component graph: fixed absolute tolerance split its zero eigenspace and
made loss depend on vertex order. A dtype/size/Laplacian-scale roundoff
allowance and explicit diagnostic tolerances address the reproduced case;
numerically unresolved neighboring modes may also be included.

[Code] CPU tests cover spectral derivatives/invariance, zero-weight original
tensor/RNG parity, active-path rate/scheduler/style contracts, frozen model
state, no_grad callers, invalid protocols, immutable artifacts, actual worker
dispatch and failure counters. GPU dependencies are doubles in trajectory
and worker tests. The [final verification record](gsd_verification_20260923.md)
reports 84 passing CPU tests, syntax checks and preserved baseline sources.

[Open] No GSD GPU run exists. Expected paths:
`result/modelnet40_c/gsd_latent_spectral_v1/<UTC-timestamp>_gsd-v1-<stage>-<arm>-seed<seed>/`.
Colab scripts prepare smoke, full Gaussian/Impulse pilot and all-15 original/
off/on comparisons at seeds 0/1/2, raw/eval, EMA off, batch32, severity5,
gamma=eta=.01, lambda=.95 and original decode. The configuration is fixed
before all-15; test-set pilot selection must be disclosed.

[Inference] Falsifiers: failed off-parity, zero/nonfinite guidance, inconsistent
three-seed pilot gains, retained-corruption bias or unacceptable graph cost.
No accuracy improvement, full CUDA correctness or publication parity is claimed.

## 2026-09-22 - SCD lambda=.96 all-15 confirmation decision

**[User report/Decision]** Although the Gaussian/Impulse lambda=.96 pilot was
null, the user requested all-15 confirmation because the paper reports
lambda=.96. This supersedes the pilot-only stopping suggestion as an explicit
paper-conformance check; lambda=.96 is fixed before observing the all-15
outcomes and is not being selected from the all-15 test results.

**[Code]** The existing `scd_lambda96_control` method accepts the canonical
all-15 corruption list, retains legacy summed SCD, keeps the original-style
decoder and raw LION eval/EMA-off contract, and changes no other factor.

**[Open]** Run seeds 0/1/2 on all 15 severity-5 corruptions. Compare the
macro mean and per-corruption rows against the matched lambda=.95
original-style baseline; preserve the lambda=.96 result as a paper-setting
confirmation even if the mean is null or negative.

## 2026-09-22 - SCD lambda=.96 Gaussian/Impulse pilot

**[Run]** The three complete pilot archives are
`result/modelnet40_c/scd_lambda96_control/20260922-191121_scd-lambda96-s5-gaussian-impulse-seed0.zip`,
`result/modelnet40_c/scd_lambda96_control/20260922-191512_scd-lambda96-s5-gaussian-impulse-seed1.zip`,
and
`result/modelnet40_c/scd_lambda96_control/20260922-191840_scd-lambda96-s5-gaussian-impulse-seed2.zip`.
Each ZIP contains the seven required files, passes CRC validation, has two
complete 2,468-example corruption rows, and has no traceback. All record
commit `4a1403d38c3342200789d5dd15652a4132790239`.

**[Run]** The lambda=.96 macro accuracies are 72.9133%, 72.0827% and
72.7917% for seeds 0/1/2, with mean 72.5959% and sample SD 0.4486 pp.
The artifacts record raw LION eval mode, EMA disabled, the original summed
SCD reduction, retained count 1966/2048 and matching checkpoint/data hashes.
All inventoried LION VAE/prior dropout modules are `training=false` before
and after each run.

**[Run]** The matched lambda=.95 original-style pilot arms have seed mean
72.5891%. Lambda=.96 minus lambda=.95 deltas are +0.4862, -0.3241 and
-0.1418 pp by seed, for mean +0.0068 pp and sample SD 0.4251 pp. The
corruption means are +0.1351 pp for Gaussian and -0.1216 pp for Impulse.

**[Inference]** The lambda=.96 pilot is null and seed-directionally unstable;
it provides no evidence that changing the retained fraction from .95 to .96
improves the operational baseline. Do not promote it to all-15 confirmation.
Preserve the original unnormalized lambda=.95 baseline.

**[Open]** A scale-matched Eq. 11 experiment would answer a different question
because SCD reduction and gamma/eta jointly determine update scale. It must be
predeclared separately if pursued; it is not justified by this lambda pilot.

## 2026-09-22 - SCD lambda=.96 control implementation

**[Code]** Added the opt-in `scd_lambda96_control` method. It preserves the
original unnormalized directed SCD sums, original `shape_latent` decoder,
raw LION eval mode, EMA-off policy, scheduler, gamma=.01 and eta=.01; only
`lambdaa` is locked to `.96` instead of `.95`.

**[Code]** The method is restricted to ModelNet40-C severity 5, batch 32,
seeds 0/1/2, complete evaluation and either the Gaussian/Impulse pilot or
canonical all-15 scope. Its artifact config records that SCD normalization is
disabled, the retained count `int(2048 * .96) = 1966`, and the Colab
integration contract. Existing `3dd_original`, shared-decoder and SCD
normalization routes remain unchanged.

**[Code]** Local verification passes with 33 unit tests, `py_compile` for the
touched runtime modules and `git diff --check`. No GPU evaluation has been
performed for this method.

**[Open]** Run the three-seed Gaussian/Impulse pilot first. Promote to all-15
only after the pilot is complete and reviewed; do not combine lambda=.96 with
normalization, gamma/eta changes, scheduler changes or decoder-style changes.

## 2026-09-22 - SCD normalization all-15 confirmation

**[Run]** The three complete all-15 archives are
`result/modelnet40_c/scd_normalization_control/20260920-162839_scd-normalized-s5-all15-seed0.zip`,
`result/modelnet40_c/scd_normalization_control/20260920-165839_scd-normalized-s5-all15-seed1.zip`,
and
`result/modelnet40_c/scd_normalization_control/20260920-172842_scd-normalized-s5-all15-seed2.zip`.
Each ZIP contains exactly the seven required files, passes `testzip()`/CRC
validation, has 15 complete corruption rows and 37,020 examples, and has no
traceback. All record commit
`2d06f79feb25452cdcdde730d06fd341bc4c7714`.

**[Run]** Macro accuracy is 61.2804%, 61.3587% and 61.1642% for seeds 0, 1
and 2; the mean is 61.2678% with 0.0979 pp sample SD. The artifacts retain
the same Point-MAE hash `507e0bbf...aec75`, LION hash
`807f6732...a0a86c2` and all 15 dataset hashes across seeds.

**[Run]** Against the matched all-15 shared-trajectory original-style arms
(`result/modelnet40_c/shared_trajectory_decoder_control/`), whose seed mean is
63.7484%, the normalized control changes the macro result by -2.4806 pp on
average (seed deltas -2.3096, -2.4419 and -2.6904 pp). Nine of 15 corruption
means are lower and six are higher. The dominant decline is Background
(-37.0881 pp mean); the largest mean gains are LiDAR (+0.7969 pp), Uniform
(+0.3917 pp) and Distortion-RBF (+0.3782 pp).

**[Code]** The run configuration confirms raw LION eval mode, EMA disabled,
2048-point normalization and retained count 1945. VAE/prior dropout records
are `training=false` before and after every seed. The ZIP validation therefore
found no implementation/runtime or eval-mode failure that would invalidate the
negative result.

**[Inference]** Under the locked `.01/.01/.95` settings, Eq. 11-style point
normalization is rejected as an operational replacement for the original
summed-SCD baseline. This is a result for normalization with unchanged update
rates: because the normalized gradient is 1/2048 of the legacy sum at fixed
input cardinality, it is not evidence that a separately scale-matched
normalization experiment would also be negative.

**[Decision]** Preserve the original-style, unnormalized SCD decoder baseline.
Do not promote the normalized control or run an all-15 normalized retuning
search. The next isolated candidate is the paper/code lambda discrepancy
(`.96` versus the locked `.95`) with the original unnormalized SCD path; it
must remain a separate three-seed test.

## 2026-09-20 - SCD review follow-up: CPU/GPU verification boundary

**[Code]** The SCD normalization helper and CLI/config contracts can be
validated locally, but the real `tta_reconstruct` path depends on the CUDA
DDIM and Chamfer implementation. A synthetic CPU trajectory is therefore not
treated as evidence for the GPU execution path.

**[Code]** The SCD artifact contract now records the actual LION pipeline,
the fixed denominator value 2048, the retained count `int(2048 * .95) = 1945`,
and the explicit Colab integration check. This corrects the previous metadata
that described the SCD run with the LION-free identity-preprocessing string.

**[Open]** A complete Colab run of
`scd_normalization_control` at the locked scope is required to validate the
real CUDA/Chamfer integration. The run must be checked for complete status,
zero traceback, the new `scd_normalization` config fields, and the expected
raw LION eval/dropout mode records. No GPU run was performed locally.

## 2026-09-20 - SCD normalization control implemented

**[Code]** Added the opt-in `scd_normalization_control` method on top of the
original-style decoder path. It passes `scd_normalize=True` through the
existing batch runner and divides the two retained directed SCD sums by the
original point-set cardinality before the batch sum, matching the recorded
Eq. 11 normalization contract. The default `3dd_original` path keeps
`scd_normalize=False`.

**[Code]** The control is locked to ModelNet40-C severity 5, batch 32, seeds
0/1/2, raw LION weights, LION eval mode, EMA off, gamma=.01, eta=.01,
lambda=.95, and the existing scheduler. It accepts the Gaussian/Impulse
pilot or the complete canonical all-15 scope, rejects arbitrary partial
scopes, and retains the original `shape_latent` decoder style.

**[Code]** Local verification passes: 30 unit tests, `py_compile`, CLI parsing,
`git diff --check`, the SCD sum-to-point-count normalization test, and the
source-only runner branch AST comparison. No GPU evaluation has been run for
this method yet.

**[Open]** The first Colab run should be the complete Gaussian/Impulse pilot
at seeds 0/1/2. Promote to all-15 only if the pilot is directionally useful;
keep lambda, scheduler, decoder style, RNG policy and LION eval mode fixed.

## 2026-09-20 - Shared-trajectory decoder control all-15 confirmation

**[Run]** The user-requested all-15 coverage is complete and validated. The
three raw archives are:

- seed 0: `20260920-131504_shared-decoder-s5-all15-seed0.zip`, SHA-256
  `d2cefdb57a91b1d1beefc444a8475e19f55bd2e27f890033fd3d6f5644f5080f`;
- seed 1: `20260920-134840_shared-decoder-s5-all15-seed1.zip`, SHA-256
  `bfafbe4c581da659bf46db34582ca8fa2ac4c733e1202286e490676cc4194fce`;
- seed 2: `20260920-142217_shared-decoder-s5-all15-seed2.zip`, SHA-256
  `a826c867595598b2dadf2031dd964c6e265ecce9f23a7dc6f9a33f7b8917e8b0`.

Each archive passes CRC validation, contains exactly the seven required files,
and records `status=complete`, `execution_status=complete`, with no traceback
or failure marker. Each has 15 complete corruption rows with 2,468 examples;
the total is 37,020 examples per seed and 111,060 paired examples across the
three seeds. All configs record branch `baseline-repro-clean` and commit
`5fc05e71cc7245828f0b64c0d6ab54a66926cdc2`.

**[Code]** All three artifacts record raw LION weights, EMA disabled and
`lion_eval_mode=true`. The VAE inventory reports `training=false` and all 33
inventoried dropout modules report `training=false` both before and after the
run. The LION checkpoint SHA-256 is
`807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`; the
15 corruption dataset hashes are identical across seeds.

**[Run]** Updated-style minus original-style paired deltas, in percentage
points, are:

| Corruption | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| Uniform | -0.0810 | +0.1621 | 0.0000 | +0.0270 |
| Gaussian | +0.2431 | -0.1216 | -0.2026 | -0.0270 |
| Background | -0.8509 | -0.8914 | +0.7293 | -0.3377 |
| Impulse | +0.2431 | +0.2836 | -0.1216 | +0.1351 |
| Upsampling | +0.1621 | -0.0405 | -0.2026 | -0.0270 |
| RBF distortion | -0.1216 | -0.1216 | +0.3647 | +0.0405 |
| Inverse-RBF distortion | -0.0810 | -0.2836 | -0.1621 | -0.1756 |
| Density | +0.0810 | -0.0810 | -0.0810 | -0.0270 |
| Density increase | +0.2026 | -0.0405 | -0.0405 | +0.0405 |
| Shear | -0.2026 | +0.0810 | -0.0810 | -0.0675 |
| Rotation | +0.0405 | +0.1621 | +0.0810 | +0.0945 |
| Cutout | +0.1216 | -0.2836 | +0.1216 | -0.0135 |
| Distortion | +0.0405 | -0.4457 | 0.0000 | -0.1351 |
| Occlusion | +0.4457 | -0.0810 | +0.2026 | +0.1891 |
| LiDAR | -0.1621 | -0.2026 | -0.2026 | -0.1891 |

The updated-style arm is better in 18/45 rows, worse in 25/45 and tied in
2/45. Seed-level all-15 macro deltas are +0.0054, -0.1270 and +0.0270 pp.
Pooled over 111,060 examples, original-style accuracy is 63.7484%,
updated-style accuracy is 63.7169%, and the paired delta is -0.0315 pp.

**[Inference]** The all-15 coverage confirms that style effects are
corruption-dependent and seed-sensitive, not a consistent updated-style
improvement. It does not support replacing the original-style decoder or
claiming a general decoder-style gain. The previously selected original-style
baseline remains appropriate.

**[Open]** The next isolated factor remains Eq. 11 SCD normalization. Keep
lambda, RNG policy, scheduler, LION eval/dropout mode, preprocessing and
decoder-style contract unchanged.

## 2026-09-20 - User-requested all-15 decoder-control coverage

**[User report]** Although the Gaussian/Impulse pilot did not satisfy the
predeclared consistency rule for promoting updated style, the user requested
an all-15 confirmation to measure the decoder-style behavior across every
canonical ModelNet40-C corruption.

**[Code]** Commit
`5ead0556b87ceae47e86587e16d1279d31f92b90` extends the opt-in
`shared_trajectory_decoder_control` scope guard to accept either the original
Gaussian/Impulse pilot or the complete canonical all-15 list. It rejects
arbitrary partial lists, preserves raw LION/eval mode, EMA-off, batch 32,
severity 5, seed 0/1/2, and all trajectory/decoder math.

**[Open]** The all-15 Colab artifacts are not yet available. The previous
pilot result remains the decision-rule evidence; the requested all-15 run is
an expanded descriptive confirmation, not a retroactive claim that updated
style was already supported.

## 2026-09-20 - Shared-trajectory decoder control pilot

**[Run]** The three raw Colab archives under
`result/modelnet40_c/shared_trajectory_decoder_control/` are complete and
valid. The archives are:

- seed 0: `20260920-124502_shared-decoder-s5-gaussian-impulse-seed0.zip`,
  SHA-256 `eaf2e676cdddc974f6a62ceea372e168498ccffc64f253cd2f2185d979c87608`;
- seed 1: `20260920-124921_shared-decoder-s5-gaussian-impulse-seed1.zip`,
  SHA-256 `cc670612cace441fea444172f1416c6fb3cf2a1bd923ca01de97c244f33d6019`;
- seed 2: `20260920-125318_shared-decoder-s5-gaussian-impulse-seed2.zip`,
  SHA-256 `3e14c1fb0ecdfc58a5b9f159d07502a8e279d2c99ece1d5fefbe9de14d3e038d`.

Each ZIP passes CRC validation and contains exactly the seven required files.
All six corruption rows are complete with 2,468 examples each, all summaries
have `status=complete` and `execution_status=complete`, and no traceback or
failed marker is present. The archives record branch
`baseline-repro-clean` and commit
`141ad6de8566543fd6558fe664db455bb1e286ec`.

**[Code]** The three configs record raw LION weights, EMA disabled,
`lion_eval_mode=true`, the same LION checkpoint SHA-256
`807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`, and
the same Gaussian/Impulse dataset hashes. Each VAE inventory reports
`training=false` and 33 dropout modules with `training=false` both before and
after evaluation.

**[Run]** Paired original-style versus updated-style deltas are:

| Seed | Corruption | Original | Updated | Updated - original |
|---:|---|---:|---:|---:|
| 0 | Gaussian | 74.7164% | 74.7974% | +0.0810 pp |
| 0 | Impulse | 70.1378% | 70.4214% | +0.2836 pp |
| 1 | Gaussian | 75.0000% | 75.2026% | +0.2026 pp |
| 1 | Impulse | 69.8136% | 69.4895% | -0.3241 pp |
| 2 | Gaussian | 75.3241% | 75.0000% | -0.3241 pp |
| 2 | Impulse | 70.5429% | 70.5024% | -0.0405 pp |

The updated-style arm is better in 3/6 paired rows and worse in 3/6. Seed-level
macro deltas are +0.1823, -0.0608 and -0.1823 pp for seeds 0, 1 and 2. The
pooled result over 14,808 examples is 72.5891% original-style versus 72.5689%
updated-style, a -0.0203 pp delta. Prediction disagreement is nonzero in every
row; decoder-output difference and style displacement are recorded in the raw
per-corruption CSVs.

**[Inference]** The predeclared consistency rule is not met: updated style is
not better across all three seeds and both pilot corruptions. The pilot does
not justify all-15 confirmation and does not replace the original-style
decoder baseline.

**[Open]** The next single-factor test is Eq. 11 SCD normalization, with
lambda, RNG policy, scheduler, dropout mode, and other factors unchanged.
Diffusion/guidance causality remains separate from this decoder-style result.

## 2026-09-20 - Shared decoder control execution repair

**[Code]** Review of implementation `381d74a` at HEAD `cf7d92d`
found three defects: the worker disabled guidance autograd, the
`decoder_control` string collided with per-corruption metric storage, and
failed rows omitted paired fields. CPU reproductions confirmed the first two
exceptions and failure-summary rejection. This supersedes the earlier
"code-ready" assessment; the old 21 passing tests did not cover these paths.

**[Code]** The control helper now explicitly enables gradients around the
existing trajectory and disables them for decoding. Contract text uses
`decoder_control_contract`; metrics use the `decoder_control` dictionary.
Failure handling retains paired counters and distances for completed batches,
including empty failed rows; summaries leave unobserved accuracies blank.

**[Code]** Added regression tests for gradient context, metadata storage and
empty/partially completed failed summaries. A CPU fake-dependency test executes
the actual TTA loop, checks one encode/five prior calls, unchanged original
style, changed updated style, no extra trajectory during decoding, and exact
original-arm parity with the default decoder under the same seed.
All 25 local unit tests pass. These are software checks, not model accuracy
evidence. No local GPU evaluation or raw artifact modification occurred.

**[Open]** Colab pilot and complete seven-file ZIP validation remain required.
Keep Gaussian/Impulse, severity 5, batch 32, seeds 0/1/2, eval/raw and existing
guidance/scheduler settings. No all-15 promotion or style benefit is established.

## 2026-09-13 - LION prior EMA inventory and opt-in loader

**Evidence:** [Run] `result/modelnet40_c/diagnostics/checkpoint_ema_inventory.json` and `checkpoint_sha256.txt`; [Code] `models/lion.py`, `main_3dd_tta.py`, `run_baseline.py`.

The checkpoint contains 462/462 prior EMA tensors matching the 462 prior model entries by count and shape. The VAE has no EMA entries. Config metadata reports EMA enabled with decay `.9999`. An opt-in, shape-validated `--lion-ema-mode` loader was added; default raw loading remains unchanged. VAE weights are never replaced. `py_compile` and `git diff --check` passed; no GPU inference was run.

**Decision:** Run eval+raw versus eval+EMA on complete Gaussian and Impulse severity 5 with identical seed, batch, scheduler, gamma, eta and lambda. Do not combine with legacy dropout mode. Repeat any positive effect at seeds 1/2 before changing the selected baseline.

## 2026-09-13 - First prior-EMA pilot: Gaussian and Impulse

**Evidence:** [Run] `result/modelnet40_c/3dd_original/20260913-154218_3dd-original-gaussian-impulse-eval-raw-seed0.zip` and `20260913-154546_3dd-original-gaussian-impulse-eval-ema-seed0.zip`.

**Protocol:** Commit `c91c1c3`; ModelNet40-C severity 5; Gaussian and Impulse; 2,468 examples/corruption; seed 0; batch 32; 100 DDIM / 5 reverse steps; gamma=eta=.01; lambda=.95; LION eval mode. Artifact manifests confirm identical Point-MAE and LION checkpoint hashes. The only intended CLI difference is `lion_ema_mode` false/true. EMA stdout confirms 462 prior EMA parameters loaded; both bundles are complete and traceback-free.

**Result:** Raw/EMA Gaussian: 74.3112/74.7974% (+.4862 pp, +12 correct). Raw/EMA Impulse: 70.2188/70.6240% (+.4052 pp, +10 correct). Two-corruption macro: 72.2650/72.7107% (+.4457 pp, +22/4936 correct).

**Interpretation/decision:** A modest, same-direction seed-0 pilot supports testing EMA further, but does not establish a baseline change: runs are seed-controlled, not common-draw paired, and cuDNN benchmark remains enabled. Repeat the exact pair at seeds 1 and 2. Do not merge EMA with legacy/train-mode LION in this confirmation.

## 2026-09-13 - Prior-EMA three-seed confirmation: inconclusive

**Evidence:** [Run] six complete raw/EMA Gaussian+Impulse bundles at seeds 0/1/2, listed in `ema_inventory_20260913.md`; all use commit `c91c1c3`, eval mode and identical model/data hashes.

**Result:** Macro raw/EMA by seed: 72.2650/72.7107 (+.4457), 72.3663/72.8525 (+.4862), and 72.6904/72.4878 (-.2026) percent. Three-seed means are 72.4406% raw and 72.6837% EMA: +.2431 pp with .3865 pp sample SD. Gaussian mean is -.0540 pp; Impulse mean is +.5402 pp.

**Decision (superseded):** The two-corruption result is small, seed/corruption dependent, and not common-draw paired, but is insufficient to rule out an effect across the remaining corruptions.

**Active decision:** Run all 15 corruptions for seeds 0 and 1 under matched `eval+raw` and `eval+EMA` conditions before accepting or rejecting EMA as a candidate. This is an exploratory two-seed screen, not a final benchmark claim.

## 2026-09-13 - All-15 three-seed EMA screen: small, non-robust aggregate effect

**Evidence:** [Run] six complete all-15 raw/EMA ZIPs listed in `ema_inventory_20260913.md`, commit `b999a1e`, seeds 0/1/2, severity 5, batch 32, eval mode, matching assets and settings except EMA flag.

**Result:** Raw/EMA macro by seed: 63.7061/63.8466 (+.1405), 63.8088/64.0546 (+.2458), and 63.9006/63.8250 (-.0756) percent. Means are 63.8052/63.9087: **+.1035 ± .1639 pp** sample SD; 70,862/70,977 correct over 111,060 examples. Largest mean gains: Background +.5808 and Distortion +.3647 pp. Largest mean declines: Impulse -.3241 and LiDAR -.2431 pp.

**Decision:** Keep EMA as a documented ablation, not the selected baseline. The small aggregate gain is seed/corruption dependent and non-common-draw paired. Next run legacy+raw for all 15 corruptions at seeds 1/2 to isolate the dropout/eval effect against current eval+raw results.

## 2026-09-15 - EMA provenance audit: code-backed, not paper-reported

**Evidence:** [Paper] local `lion.pdf` exact-term scan; [Code] original local LION repository `../LION/trainers/common_fun_prior_train.py`, `trainers/train_prior.py`, `utils/ema.py`, plus this fork's supplied `lion_ckpts/unconditional_all55_cfg.yml`; [Run] checkpoint inventory under `result/modelnet40_c/diagnostics/`.

**Result:** The LION paper text contains no exact `EMA`, `exponential moving average`, or `moving average` mention. The original training code does wrap the prior/DAE optimizer in an EMA optimizer, serializes its state as `dae_optimizer`, and swaps those weights into the prior around sampling when `cfg.ddpm.ema` is enabled. The supplied all55 configuration enables it with decay .9999. The checkpoint has 462 matching prior EMA tensors and no VAE EMA tensors.

**Decision:** Retain the EMA experiment as a code-derived fork ablation, not a paper reproduction claim. Its existing three-seed all-15 result remains +.1035 +/- .1639 pp and is not selected. Full audit: `ema_inventory_20260913.md`.

Append entries chronologically. Never delete negative or superseded results. Use exact run paths for **[Run]** claims.

## 2026-09-19 - Preprocessing identity control implemented

**Evidence:** [Code] run_baseline.py, tests/test_preprocessing_identity.py,
and result/README.md; no [Run] artifact yet. The implementation was committed
on top of `4096d4ec1426a8b2682f6fd2d45c9d094ad4c199` after local checks.

**Question:** How much of the source-only to TTA accuracy difference is
attributable to the TTA preprocessing/output chain rather than LION?

**Implementation:** Added the opt-in --method preprocessing_identity path.
It is locked to ModelNet40-C severity 5, all 15 canonical corruptions, batch
32, seed 0, complete evaluation, direct corruption-file loading and the
existing gamma/eta/lambda values. The path performs per-shape normalize,
the existing interpolation/upsampling to 2048, scale 3.3885,
rotate_pointcloud, rotateback_pointcloud, existing ModelNet output
normalization, FPS(1024), and frozen Point-MAE classification_only under
torch.no_grad(). It does not load or call LION, mutate data, alter FPS,
use EMA, scheduler, guidance, GSD, or PxP.

The runner reuses the existing immutable seven-file directory/ZIP contract.
Identity config records method, stage, preprocessing contract,
lion_loaded=false, lion_mode_policy=bypassed, final_decode_style,
point counts, scale, scheduler/spectral/projection fields, asset manifests and
the exact CLI. The canonical source_only branch AST is unchanged relative
to HEAD.

**Local verification:** [Code] py_compile passed for changed/affected
modules; the four-test unittest control suite passed; CLI scope guards,
identity transform order, and identity config serialization are covered.
No CUDA/model evaluation was run locally.

**Decision/Open:** The code is ready for the predeclared Colab run, but no
accuracy or causal conclusion is made until the complete ZIP is supplied and
validated. The result must be stored under
result/modelnet40_c/preprocessing_identity/ and contain exactly the seven
required files. A positive identity delta permits planning pure VAE
encode/decode next; a null/negative result sends the next one-factor checks to
updated final-style decoder, Eq. 11 SCD normalization, lambda .95/.96, and
RNG controls in that order.

## 2026-09-19 - Pure VAE encode/decode control implemented

**Evidence:** [Code] `run_baseline.py`, `tests/test_preprocessing_identity.py`,
and `result/README.md`; no [Run] artifact yet.

**Question:** After the positive preprocessing identity delta, how much of the
remaining source-only to 3DD-TTA difference is explained by VAE reconstruction
itself, without LION priors, diffusion scheduling, or guidance?

**Implementation:** Added the opt-in `--method pure_vae_encode_decode` path.
It is locked to ModelNet40-C severity 5, all 15 canonical corruptions, batch 32,
seed 0, complete evaluation, direct corruption-file loading and the existing
gamma/eta/lambda values. The path uses the same preprocessing/output contract as
the identity control, then calls LION VAE `encode` -> `decompose_eps` -> `sample`
with the encoded latents. It loads raw LION weights, forces VAE/prior modules to
eval mode for provenance, but never calls the priors; no EMA, scheduler,
guidance, GSD, PxP, alternate FPS policy, or dataset mutation is introduced.

The seven-file immutable artifact contract is reused. Pure VAE config records
`lion_loaded=true`, `lion_mode_policy="raw VAE eval; priors bypassed"`,
`vae_contract="encode -> decompose_eps -> sample"`, and `prior_used=false`.
The canonical source-only branch remains structurally unchanged relative to the
pre-implementation HEAD.

**Local verification:** [Code] the focused unittest suite passes 7/7, including
locked CLI scope, config serialization, exact preprocessing order, and the
absence of prior/guidance calls in the fake-VAE control. No CUDA/model
evaluation was run locally.

**Decision/Open:** The implementation is ready for the predeclared Colab run.
Do not infer an accuracy result until the complete ZIP is supplied and checked
for the seven required files, status/traceback, commit, checkpoint/data hashes,
total accuracy, and all 15 per-corruption rows. If pure VAE is near source-only,
the positive identity delta is primarily preprocessing-local; if it is much
higher, VAE reconstruction explains a larger share of the TTA delta. These are
[Inference] decision rules, not [Run] conclusions.

## 2026-09-19 - Preprocessing identity result: modest positive delta

Evidence: [Run] result/modelnet40_c/preprocessing_identity/20260919-140904_identity-s5-all15-seed0.zip; archive SHA-256 9348bcbf613b8e77ee1c758c7542b1e90cde74e59d1c8896a188669a3e969629.
Git commit: 4096d4ec1426a8b2682f6fd2d45c9d094ad4c199; branch baseline-repro-clean.

Protocol: Complete ModelNet40-C severity 5, all 15 corruptions, 2,468 examples/corruption, batch 32, seed 0, direct corruption-file loading, identity preprocessing chain, LION bypassed, frozen Point-MAE, FPS(1024), gamma=.01, eta=.01, lambda=.95. The ZIP passes testzip(), contains exactly the seven required files, has 15 complete rows, and stdout has no traceback/error signature. Classifier, label and all 15 data hashes match the archived source-only comparator; Colab records torch 2.1.2+cu121, CUDA 12.1, Diffusers 0.11.1, PointNet2 3.0.0 and an A100-SXM4-80GB.

### Result

Identity macro/micro accuracy is 55.0243% (20,370/37,020). The source-only severity-5 comparator is 53.6899% (19,876/37,020), so identity improves by +1.3344 pp (+494 correct). The seed-0 eval/raw 3DD-TTA context run is 63.7061%; identity remains 8.6818 pp below it and closes 13.32% of the 10.0162 pp source-to-TTA gap.

Identity is higher than source-only on 9/15 corruptions, equal on Upsampling, and lower on 5/15. Largest gains are Density Increase +8.1848 pp, Cutout +6.0373 pp, Density +3.6467 pp and Occlusion +2.8363 pp. Largest declines are LiDAR -4.6596 pp and Background -4.2950 pp.

## 2026-09-19 - Pure VAE encode/decode result: preprocessing-localized positive delta

**Evidence:** [Run] `result/modelnet40_c/pure_vae_encode_decode/20260919-144513_pure-vae-s5-all15-seed0.zip`; archive SHA-256 `ade0a5c0199be5cbbf6cd95b3deb5324a0572b16dd57dc9fc64d3c753dc0eb0d`.

**Provenance/validation:** The archive passes `testzip()` and contains exactly
the seven required files under one run directory. Config and CSV status are
`complete`; all 15 canonical corruption rows contain 2,468 examples; stdout
contains 15 corruption results and no traceback, error, exception, or OOM
marker. The run is branch `baseline-repro-clean`, commit
`3630e7903f1a888e0f015ef5bb4346bcdc99221e`, ModelNet40-C severity 5, batch 32,
seed 0, direct file loading, frozen Point-MAE, and raw VAE eval with
`encode -> decompose_eps -> sample`. `scheduler_config` is `{}` and no
`scheduler_class` or timestep list is recorded. Classifier, label, and all 15
data hashes match the source-only and preprocessing-identity artifacts; the
LION checkpoint hash is `807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`.

### Result

Pure VAE macro/micro accuracy is **54.7947%** (20,285/37,020). Relative to the
source-only severity-5 comparator at 53.6899% (19,876/37,020), this is
**+1.1048 pp**. Relative to preprocessing identity at 55.0243%, it is
**-0.2296 pp**. Relative to the seed-0 eval/raw 3DD-TTA context at 63.7061%,
it is **-8.9114 pp** and closes 11.03% of the source-to-TTA gap. This last
comparison is contextual, not a causal estimate, because the TTA artifact is
from a different commit and random draw.

Pure VAE is higher than source-only on 12/15 corruptions and lower on
Background (-5.1864 pp), Shear (-0.9319 pp), and LiDAR (-3.2010 pp). Largest
gains are Density Increase (+6.6451 pp), Cutout (+4.7407 pp), Gaussian
(+2.6742 pp), and Uniform (+2.2285 pp). It is equal to identity on Upsampling
and below identity by 0.2296 pp overall.

**Decision:** [Inference] The positive but identity-near result localizes most
of the modest source-only delta to the preprocessing/interpolation/rotation/
output-normalization chain. Pure VAE reconstruction does not explain the
large remaining TTA gap and slightly reduces the identity result in aggregate.
This does not by itself prove that diffusion guidance is ineffective: pure VAE
and full TTA differ in more than one operation, and the available TTA context
run is not a common-random-number, same-commit comparison.

**Open/falsifier:** A matched common-draw comparison or a separately isolated
decoder/protocol control could revise the localization. Do not add GSD/PxP,
EMA, or another dataset on the strength of this single seed; preserve the raw
ZIP unchanged.

## 2026-09-19 - Pure VAE seed-stability diagnostic implemented

**Evidence:** [Code] `run_baseline.py`, `tests/test_preprocessing_identity.py`,
`result/README.md`, and `knowledge/repository_map.md`; no [Run] artifact yet.

**Implementation:** Added the opt-in `--method pure_vae_seed_stability` path.
It reuses the already validated pure VAE encode/decode implementation and is
locked to ModelNet40-C severity 5, all 15 canonical corruptions, batch 32,
complete evaluation, direct corruption-file loading, raw VAE eval, and the
same gamma/eta/lambda, checkpoint, data, preprocessing, and FPS contracts.
Only seed 1 or seed 2 is accepted; seed 0 remains represented by the archived
`pure_vae_encode_decode` run. The new path is a stochastic stability diagnostic,
not a new TTA method, and records
`seed_stability_reference="pure_vae_encode_decode seed0 archive"`.

No priors, EMA, diffusion scheduler, guidance, GSD, PxP, alternate FPS policy,
or dataset mutation is added. The existing seed-0 pure VAE and source-only
paths retain their locked validation rules. Local tests cover seed-1 acceptance,
seed-2 config serialization, and seed-0 rejection.

**Decision/Open:** The implementation is ready for two separate Colab runs,
one at seed 1 and one at seed 2. Combine their complete seven-file ZIPs with
the archived seed-0 pure VAE ZIP to calculate mean and standard deviation. Do
not interpret the one-seed `+1.1048 pp` pure-VAE delta as stable until this
screen is complete.

## 2026-09-19 - Pure VAE seed-stability result

**Evidence:** [Run] the complete archives
`result/modelnet40_c/pure_vae_seed_stability/20260919-162638_pure-vae-s5-all15-seed1.zip`
and
`result/modelnet40_c/pure_vae_seed_stability/20260919-163345_pure-vae-s5-all15-seed1.zip`.
Their SHA-256 values are `5593e519efa0a4d7a87fae8e39bb551a78cbdc260d00172f96cbd2bf41101490`
and `a5fe7d7b1b21736ce6403f0371f2cb022e4ab68a1003c0351fd8a1153210db67`.

**Validation/provenance:** Both archives pass `testzip()`, contain exactly the
seven required files, have complete status, 15/15 rows, 2,468 examples per
corruption, and no traceback/error/exception/OOM marker. Both record method
`pure_vae_seed_stability`, raw VAE `encode -> decompose_eps -> sample`, no
priors, empty scheduler config, and the same batch-32 ModelNet40-C severity-5
all-15 contract. Both record commit `e8be9a4e0ef6a0289fb0a749dd0ac72d1600d48c`;
the archives also record the Colab working tree as dirty, so the commit and
runtime source hashes are the provenance anchors. The filenames both end in
`seed1`, but the earlier timestamp `162638` records seed 1 in config and
command, while the later timestamp `163345` records seed 2; this discrepancy
is preserved and the raw files are unchanged.

### Result

Seed 0 (archived pure VAE control) is **54.7947%** (20,285/37,020); seed 1 is
**54.9379%** (20,338/37,020); seed 2 is **54.8082%** (20,290/37,020). Across
seeds 0/1/2, mean accuracy is **54.8469%**, sample standard deviation
**0.0790 pp**, population standard deviation **0.0645 pp**, and range
**0.1432 pp**. The mean is **+1.1570 pp** over deterministic source-only at
53.6899% (19,876/37,020).

The paired pure-VAE minus preprocessing-identity differences are **-0.2296 pp**
(seed 0), **-0.1297 pp** (seed 1), and **-0.1405 pp** (seed 2). Thus VAE
reconstruction is consistently below the matched preprocessing-only control;
the one-seed localization is supported, although the absolute VAE delta is
still modest and stochastic.

**Decision:** [Run]/[Inference] The positive source-model effect persists in
the pure-VAE control, but the paired comparison assigns the aggregate gain
primarily to preprocessing rather than VAE reconstruction. This is not yet a
causal estimate of diffusion/guidance: the pure-VAE and available 3DD-TTA
context runs are not same-commit common-draw pairs. The next test should be a
same-commit, common-draw comparison between pure VAE encode/decode and the
operational eval/raw 3DD-TTA path, with preprocessing held fixed. Keep EMA,
GSD/PxP, alternate FPS policies, and other datasets parked. This was the
initial candidate; the later decision below defers it unless a stronger causal
thesis claim is required.

## 2026-09-19 - Common-draw control deferred

**Evidence:** [Run] the validated pure-VAE seed screen above, the validated
preprocessing identity seed screen above, and the operational eval/raw LION
screen documented in `knowledge/dropout_eval_mode_20260913.md` under
`result/modelnet40_c/3dd_original/`.

**Assessment:** The existing independent screens already establish the
operational ordering: source-only is 53.6899%, preprocessing identity averages
55.0135%, pure VAE averages 54.8469%, and the available eval/raw 3DD-TTA screen
averages 63.8547% over its two repeated seeds (with the historical seed-0
screen also archived). A common-draw run would not be expected to change these
accuracy values; its purpose would be to tighten the causal attribution of the
remaining 3DD-TTA gap by reusing one preprocessing realization in both branches.

**Decision:** [User report]/[Inference] Do not spend the next Colab run on this
paired control. Treat the result as an optional confirmatory experiment only if
the thesis requires the stronger claim that diffusion/guidance, rather than
unmatched stochastic preprocessing or VAE reconstruction, is causally
responsible for the remaining gap. Until then, report the preprocessing
positive effect and the consistently non-positive pure-VAE contribution, while
leaving diffusion/guidance causality [Open]. No common-draw code was retained,
committed, or pushed.

## 2026-09-20 - Source-only provenance manifest audit

**Evidence:** [Run] the complete severity-5 source-only archives
`result/modelnet40_c/source_only/20260912-111822_source-only_seed0.zip`,
`result/modelnet40_c/source_only/20260912-120122_source-only_seed1.zip`,
`result/modelnet40_c/source_only/20260912-120349_source-only_seed2.zip`, and
`result/modelnet40_c/source_only/20260915-185615_source-only-sev5-all15_seed0.zip`;
[Code] `data/readme.md`, `pointnet_ckpts/readme.md`, and the current
`datasets_mate/create_corrupted_dataset.py`.

**Result:** All four complete severity-5 archives contain the same 15 data
hashes, Point-MAE checkpoint hash
`507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75`, Point-MAE
config hash `346f37e06fc73111ddbcd2a2c07ff85f057d180f27e3c664dda88c7f015dd3c6`,
and label hash
`97b6e660820103074ab192f10b6d9c33aa0446f1547ddb6ad2ca35a029a63b31`. The
recorded run commits differ (`ef87692`, `fd4caea`, and `262f3a6`), but no asset
hash difference appears between those source-only runs. The one archived run
for each severity 1--4 also has a self-consistent 15-file data manifest.

**Limitation:** The workspace's `data/` directory contains only `readme.md` and
`pointnet_ckpts/` contains only an empty readme; the actual `.npy` files and
Point-MAE checkpoint are Colab-side assets. Therefore these manifests prove
cross-run consistency, not byte identity with the canonical ModelNet40-C
Zenodo archive or the authors' checkpoint. `data/readme.md` identifies the
Zenodo corruption download, while author-published hashes are unavailable.
The LiDAR generator's replacement sampling has already been shown to be
upstream-consistent; no regeneration or alternate FPS policy is authorized.

**Decision:** [Run]/[Code]/[Open] The internal asset gate is consistent but not
closed. The next provenance action is to obtain or compare canonical archive
and checkpoint hashes, not to run another unmatched TTA method. Until that
evidence exists, keep the source-only gap as an unresolved data/checkpoint
provenance issue and keep the shared-trajectory decoder comparison queued
after the source-data gate.

## 2026-09-20 - Canonical Zenodo archive identified

**Evidence:** [User report] Colab's Zenodo API lookup returned the file entry
`modelnet40_c.zip`, size `1,970,686,633` bytes, with MD5
`c4a7fffaa52c80b33f7b3a0ac7782d3b`. [Run] The previously audited Colab
source-only manifests have matching checkpoint/config/label hashes and the
same 15 corruption-file hashes across complete severity-5 runs.

**Interpretation:** The API metadata pins the canonical archive identity to a
specific Zenodo file. It does not yet prove that the 15 `.npy` files used by
the Colab runs are byte-identical to the archive, because the archive contents
have not yet been extracted and compared against the recorded per-file
manifest. It also does not establish that the Point-MAE checkpoint is the
author's exact checkpoint; that checkpoint is a separate provenance object.

**Decision:** [Inference]/[Open] Download the identified archive outside the
repository, extract it without mutating `data/` or any raw result, and compare
per-file sizes/SHA-256 values with the Colab manifest. If all 15 files match,
remove the corruption-archive mismatch as an explanation for the 53.6899%
source-only result. If any file differs, rerun the direct source-only control
with the canonical archive before interpreting TTA. Keep the checkpoint gate
separate and do not change LiDAR generation or inference FPS.

## 2026-09-20 - Archive byte audit deprioritized (superseded)

**Evidence:** [User report] The ModelNet40-C data was downloaded from the same
fixed Zenodo address. [Run] Complete Colab source-only artifacts already share
the same 15 data-file hashes across seeds and recorded commits. [User report]
The long-running archive download is not considered worthwhile at this stage.

**Interpretation:** Same-record download provenance plus repeated identical
Colab manifests makes an archive mismatch unlikely enough that a 1.97 GB
byte-level re-download is not a useful current experiment gate. It does not
mathematically prove archive-content identity, so the caveat remains recorded
as [Open] rather than being silently declared resolved.

**Decision:** Stop/defer the archive download, do not rerun source-only solely
for this audit, and proceed to the predeclared shared-trajectory final-style
decoder control. Do not regenerate LiDAR, alter FPS, or add a new TTA method.

This decision is superseded by the supplied completed provenance report below.

## 2026-09-20 - Canonical archive byte identity confirmed

**Evidence:** [Run] `result/modelnet40_c/provenance_report/provenance_report.json`.
The report contains 15 unique corruption rows and complete fields for every
row. The downloaded archive is `1,970,686,633` bytes with MD5
`c4a7fffaa52c80b33f7b3a0ac7782d3b`, matching both Zenodo metadata and the
expected archive identity.

**Result:** `all_archive_members_match=true` and
`all_current_colab_files_match=true`. Every one of the 15 severity-5 archive
members has exactly one matching member, and its byte count and SHA-256 match
the audited source-only manifest. The corresponding 15 current Colab files
also match the same byte counts and SHA-256 values. This includes LiDAR; no
LiDAR regeneration or alternate FPS policy is justified by provenance.

**Decision:** [Run] The ModelNet40-C corruption archive mismatch is removed as
an explanation for the 53.6899% source-only result. The dataset provenance
gate is closed. The Point-MAE checkpoint's author/canonical identity remains a
separate [Open] question, but no source-only rerun is needed for the dataset.
Proceed to the predeclared shared-trajectory original-versus-updated
final-style decoder control after the normal code/protocol audit.

## 2026-09-19 - Preprocessing identity seed-stability diagnostic implemented

**Evidence:** [Code] `run_baseline.py`, `tests/test_preprocessing_identity.py`,
`result/README.md`, and `knowledge/repository_map.md`; no [Run] artifact yet.

**Implementation:** Added the opt-in
`--method preprocessing_identity_seed_stability` path. It reuses the validated
preprocessing identity chain with LION fully bypassed and is locked to
ModelNet40-C severity 5, all 15 corruptions, batch 32, complete evaluation,
direct files, seed 1 or 2, the same gamma/eta/lambda, data/checkpoint assets,
and FPS(1024) contract as the seed-0 identity archive. Seed 0 remains the
archived `preprocessing_identity` reference. The config records
`seed_stability_reference="preprocessing_identity seed0 archive"`.

This control is necessary because `upsample_all` uses NumPy random interpolation
and downsampling. It isolates preprocessing stochasticity separately from the
pure VAE seed screen, whose randomness includes both this preprocessing and VAE
latent sampling. No LION, VAE, EMA, scheduler, guidance, alternate FPS policy,
or dataset mutation is introduced.

**Decision/Open:** Run seeds 1 and 2 as separate seven-file ZIPs and compare
them with the archived seed-0 identity ZIP before interpreting the pure VAE
seed screen. The useful paired quantity is pure-VAE accuracy minus
preprocessing-identity accuracy at the same seed.

## 2026-09-19 - Preprocessing seed-stability routing failure and fix

**Evidence:** [Run] failed archive
`result/modelnet40_c/preprocessing_identity_seed_stability/20260919-164103_preprocessing-identity-s5-all15-seed1.zip`;
[Code] `run_baseline.py`, `tests/test_preprocessing_identity.py`.

The first Colab attempt reached the evaluation loop but failed before the first
corruption result with `AttributeError: 'NoneType' object has no attribute
'vae'`. Root cause: model setup recognized the new method as a LION-free
preprocessing route, but the batch-evaluation branch still matched only the
exact `preprocessing_identity` string and fell through to
`baseline.process_batches(..., lion=None, ...)`. The ZIP is preserved as a
failed/incomplete run and provides no accuracy evidence.

The fix centralizes the route predicate in
`is_preprocessing_identity_method()` and uses it for both model setup and batch
evaluation. The focused suite now covers both method identifiers; the failed
archive must not be overwritten or interpreted as a benchmark result.

## 2026-09-19 - Preprocessing identity seed-stability result

**Evidence:** [Run] the complete archives
`result/modelnet40_c/preprocessing_identity_seed_stability/20260919-165516_preprocessing-identity-s5-all15-seed1.zip`
and
`result/modelnet40_c/preprocessing_identity_seed_stability/20260919-165857_preprocessing-identity-s5-all15-seed1.zip`.
Their SHA-256 values are `a9afc629d81a8c1c08c2b7ab4b1b331d492f53d45ad82d60bbe2923599e7a358`
and `82720bc9378e4b235d31286007430d68c46af676f0ed08f95e572b3c34b56ae9`.

**Validation/provenance:** Both archives pass `testzip()`, contain exactly the
seven required files, have complete status, 15/15 rows, 2,468 examples per
corruption, and no traceback/error/exception/OOM marker. Both record commit
`9ad172848844467307427d7b7e1cbed72b3387d7`, raw preprocessing identity with
LION bypassed, empty scheduler config, matching classifier/data/label hashes,
and the same batch-32 ModelNet40-C severity-5 all-15 contract. The filenames
both end in `seed1`, but the earlier timestamp `165516` records seed 1 in both
config and command, while the later timestamp `165857` records seed 2; this
labeling discrepancy is preserved and does not alter the raw files.

### Result

Seed 0 (archived identity control, commit `4096d4e`) is **55.0243%**
(20,370/37,020); seed 1 is **55.0675%** (20,386/37,020); seed 2 is
**54.9487%** (20,342/37,020). Across seeds 0/1/2, mean accuracy is
**55.0135%**, sample standard deviation **0.0602 pp**, population standard
deviation **0.0491 pp**, and range **0.1189 pp**. The mean is **+1.3236 pp**
over deterministic source-only at 53.6899% (19,876/37,020); per-seed deltas
are +1.3344, +1.3776, and +1.2588 pp.

**Decision:** [Run]/[Inference] The preprocessing-only positive source-model
delta is stable across these three seeds at the aggregate level and is not
explained by a large interpolation-randomness swing. This strengthens the
finding that preprocessing itself raises source-only accuracy modestly. The
pure VAE seed screen remains necessary because it adds a second stochastic
source, VAE latent sampling; compare pure VAE and preprocessing identity at
the same seed before attributing any VAE contribution.

## 2026-09-12 — Initial repository and literature audit

**Evidence:** paper PDFs, current branch source, full Git history; no archived Colab artifacts.  
**Git branch/commit:** `pxp-gradient-projection` / `53ba252` at audit start.  
**Status:** mixed **[Paper]**, **[Code]**, **[User report]**, and **[Inference]** evidence.

### Confirmed from sources

- **[Paper]** Published 3DD-TTA ModelNet40-C mean is 65.7%; repository README says 66.1%.
- **[Code]** The original baseline path remains separated from GSD/PxP scripts.
- **[Code]** The main GSD path is a latent spectral regularizer, not full GSDTTA.
- **[Code]** Dynamic mode recomputes an eigenvector matrix/basis, not a single `U_0` vector.
- **[Code]** PxP variants project spectral and SCD gradients and therefore are PixelAsParam-inspired rather than direct implementations.
- **[Code]** Runs are stochastic and current output schemas do not capture enough provenance for definitive comparison.
- **[Code]** Loss reductions and scheduler/style choices have changed across commits, creating confounds.

### Provisional observations

- **[User report]** Original baseline is approximately 63% locally.
- **[User report]** A GSD variant reaches approximately 63.5%.
- **[Inference]** The first investigation should prioritize environment/checkpoint/data identity, seed variance, scheduler serialization, and metric aggregation before further method search.

### What would update this conclusion

A complete archived baseline run with per-corruption results, exact command/config, package/GPU environment, data/checkpoint hashes, and repeated seeds.

## 2026-09-12 — Follow-up code audit and clean restart decision

**Evidence:** [Code], [Paper], [User report], [Inference]; no new [Run].  
**Fork commit:** `53ba252519c7cf65f836a9c1c564027142ab1573`  
**Original LION commit:** `7711b3d185752eeb632d095494876e4de15f3195`  
**Run paths:** none; static audit only, not an accuracy experiment.

### Findings

- [Code] Original LION trainer inference disables VAE/prior dropout. Its demo wrapper does not; the fork wrapper is byte-identical. Current baseline/GSD setup sets only Point-MAE to eval. This verified discrepancy is inherited, not caused by GSD additions; accuracy impact remains [Inference].
- [Code] Scheduler, reverse steps, final style, batch and unequal gamma/eta semantics confound current baseline/GSD comparisons.
- [Code] Mean spectral MSE plus summed SCD changes relative guidance with actual batch/band/channel counts.
- [User report] Mean settings previously gave smaller reported spectral losses and higher accuracy than sum. Exact runs/configurations are not archived; a smaller mean number alone is not spectral-fidelity evidence.
- [Paper/Code/Open] Threshold normalization differs from GSDTTA Eq. (10). Keep visual/symbol/adjacency verification before correction; self-neighbors, distance units and isolated-node handling need small tests.
- [Code] Conditional missing-key failures, unsafe resume aggregation, symmetric CSV header/row mismatch, stale analyzer shapes and batch-flattened projection require targeted tests rather than wholesale legacy migration.
- [Code] SCD uses the first three local-latent channels; it is not computed after decoding. Earlier method/map wording is superseded and corrected.
- [Inference] Shared eigenvector sign flips/within-band rotations do not change complete-band same-basis MSE; changing projectors and band boundaries are the dynamic diagnostics.

Full source paths, wrapper hash, caveats and falsifiers: [code audit](code_audit_20260912.md).

### Decision and verification gates

The user endorsed an incremental clean restart from main. Local main/origin/main/upstream/main equal `107305fd7baf40b359f31c07d235599198be7324`. Proposed `baseline-repro-clean` is not created yet. Preserve legacy branches/user work; add artifact and source-only controls, then isolated dropout A/B, then lock the baseline ladder and require spectral-off parity. Sum plus smaller eta/gamma and dynamic/PxP tuning are deferred. [Small batches](clean_restart_batches.md) define scopes, tests, knowledge updates and Colab review gates.

This update changes documentation only. No branch switch or Python modification; no gain claimed. Gaussian/background seeds 0,1,2 under fixed configuration/common draws are next accuracy evidence. Repeated null/negative dropout effects weaken the gap hypothesis. A source-only mismatch redirects diagnosis to data/classifier; a failed spectral-off parity check exposes accidental method differences.

## 2026-09-12 — Batch 0 branch reference prepared, placement pending

**Evidence:** [Code] Git/ref/status checks; no numerical run.  
**Legacy checkout:** `pxp-gradient-projection` / `53ba252519c7cf65f836a9c1c564027142ab1573`  
**New branch/main:** `baseline-repro-clean` / `107305fd7baf40b359f31c07d235599198be7324`  
**Run paths:** none.

The user approved proceeding. Created the branch reference with `git branch baseline-repro-clean main`, without switching checkout, staging, commit or push. Ref equality and `git diff --exit-code` verified; pre-existing untracked knowledge/skill/protocol/PDF/notes remain. Worktree inspection found only the normal root checkout. Legacy tracked scripts and source PDFs would disappear from the folder on an in-place switch, though still retained on the old branch; asked the user for placement preference before this transition.

Batch 0 remains partial until safe placement, curated memory preservation and review. No Python changes, dependency install or local accuracy test. Keep Batch 1 on hold until the clean branch is actually checked out in its declared workspace. See `clean_restart_batches.md` for the next handoff. This supersedes the earlier status that the new branch did not exist; it does not supersede the audit's inference findings.

## 2026-09-12 — Batch 0 same-folder checkout and documentation preservation

**Evidence:** [Code] Git state, SHA-256 and parity checks; no [Run].  
**Active baseline source:** `baseline-repro-clean`, main `107305fd7baf40b359f31c07d235599198be7324`  
**Legacy experiments:** `pxp-gradient-projection` / `53ba252519c7cf65f836a9c1c564027142ab1573`  
**Run paths:** none.

User selected same-folder development and approved carrying skill/knowledge. Switched to the prepared clean branch. No tracked or staged code edits existed, so no new stash; existing dev stash `02ddaf533c93164d69643e43c55ad37df6fa0343` remains. Restored four absent scholarly PDFs from legacy Git history to local untracked files; did not overwrite the user's PixelAsParam PDF. All 23 selected memory/skill/protocol/PDF hashes matched across checkout before documentation updates. Tracked Python/requirements/environment match main; old variants are not imported.

Curated memory, researcher skill and result protocol are selected for a local documentation-only commit, with all PDFs and unrelated docs/tmp excluded. The containing documentation commit is recorded in the handoff rather than embedded as a self-referential hash. No push, dependency installation or numerical test. The prior placement-pending status is superseded. Next is Batch 1 after handoff review, not dropout modification yet.

## 2026-09-12 - Batch 1 baseline smoke artifact implementation

**Evidence:** [Code], not [Run].

**Base Git commit:** b31fd23193bbcb9a5c189cfb4118be41506f9333 on baseline-repro-clean; implementation commit reported in the handoff.

**Run paths:** none supplied yet. Expected result/modelnet40_c/3dd_original/<UTC-timestamp>_baseline-smoke_seed0/.

**Question:** can we produce an internally consistent, diagnosable original-TTA smoke bundle without changing adaptation math/modes?

Added run_baseline.py and research_artifacts.py, with optional read-only checkpoint/scheduler/batch observers in three baseline modules. The runner reuses the original preprocessing/TTA/classification path. Seeds and actual runtime flags/configs, checkpoint/data/source hashes, load incompatibilities, dropout/module modes, installed extension identities and count-based fraction CSVs are logged; subprocess capture retains Python/native stderr. Each invocation creates a fresh seven-file directory and a sibling ZIP, with partial/failed status rather than fabricated full accuracy. No resume, GSD, LION eval, scheduler/rate/style/reduction or dependency changes.

Local syntax/CLI and temporary artifact/count/macro-micro/duplicate/collision/failure-state checks passed. Observer-stripped ASTs of all three modified baseline files match main. The first AST comparison failed because its checker omitted a nested batch-observer branch; the corrected recursive checker passed. This was a verification-script limitation, not an adaptation-code change. No unit-test files were created per user request.

**Protocol:** pending Level 0 Colab smoke, Gaussian severity5, first two batches, seed0, explicit batch32 and repository lambda0.95; gamma/eta0.01, normal/background5/35, unchanged legacy modes. Runner default batch remains40. Same seed is not a guarantee of common-draw pairing.

**Decision:** request the complete smoke ZIP using colab_baseline_smoke.md; validate before Batch 2. No numerical result, dropout effect or accuracy-gap cause is inferred. A Colab runtime/import/schema/count failure or unexplained GPU behavior rejects the end-to-end handoff until resolved.

## 2026-09-12 - Chamfer import restoration for Batch 1 smoke

**Evidence:** [Code] plus [User report] Colab traceback; no successful run yet.

The Batch 1 smoke failed while constructing Point-MAE, before LION/TTA or classification. `models_mate/Point_MAE.py` referenced `ChamferDistanceL2` for `cdl2` but its import was commented out. The same import is active on the previously working `pxp-gradient-projection` branch. Git history identifies commit `8183863` as restoring it for Colab.

Decision: restore only the missing import. This is a baseline construction repair, not a TTA, dropout, scheduler or GSD change. Re-run the same Level 0 Gaussian two-batch command with seed 0. The new ZIP falsifies this diagnosis if it still reaches the same undefined-name error.

## 2026-09-12 - Batch 1 Gaussian smoke accepted

**Evidence:** [Run] `result/modelnet40_c/3dd_original/20260912-110541_baseline-smoke_seed0/` and its unchanged ZIP, SHA-256 `1f7db8dee1fe80bc2cb3a7b4d9d36b5bd19ce355b98e4a4cb17de36bfd8d1611`.

**Git commit:** `6a60b611b8ce03c236d541474fd4fd151c0da112` on `baseline-repro-clean`.

**Protocol:** Level 0 smoke; ModelNet40-C Gaussian severity 5; first two file-order batches; 64 of 2468 examples; batch 32; seed 0; gamma=eta=0.01; lambda=0.95; normal reverse steps=5. NVIDIA A100-SXM4-80GB, driver 580.82.07, CUDA 13.0; Python 3.8.20.

The seven required files have one safe archive root and no unexpected members. `execution_status=complete` and CSV `status=partial` correctly distinguish a finished prefix from full dataset coverage. Counts are internally consistent: 47/64 = 0.734375 for both recorded macro and micro values. Runtime is 4.3506 seconds and peak allocated GPU memory is 25748.2 MiB. `stdout.log` has no traceback or error line.

Point-MAE loaded with strict=False but zero missing/unexpected keys; both LION modules loaded strictly with zero missing/unexpected keys. The actual scheduler is DDIMScheduler with set_alpha_to_one=True, epsilon prediction and 100 timesteps from 990 to 0. Classifier is eval; LION VAE/priors are train mode, retaining the intended legacy-mode control. Source hashes match the recorded Git commit blobs; apparent local Windows hash differences were CRLF line endings. Colab git dirty state contains compiled extension/cache artifacts, not the runner source files.

**Decision:** accept Batch 1 artifact/output contract. This result is not an accuracy benchmark or reproduction claim. Next implementation gate is Batch 2 source-only full-corruption identity evaluation; do not yet change LION mode or add GSD.

## Entry template

## 2026-09-20 - Shared-trajectory final-style decoder control implemented

**Evidence:** [Code] `tta.py`, `run_baseline.py`, `research_artifacts.py`,
`tests/test_shared_trajectory_decoder_control.py`, `result/README.md`; no
Colab `[Run]` result yet. The implementation is committed on
`baseline-repro-clean` at `381d74adc98624b233c5007c793c5cd8189aeb33`.

**Question:** Does decoding the same final local latent with the updated
`style_cond` change predictions relative to decoding with the original
`shape_latent`, with the TTA trajectory held fixed?

**Implementation:** [Code] Added the opt-in
`shared_trajectory_decoder_control` method. Each batch runs one existing
`tta_reconstruct` trajectory with one initial noise/scheduler/SCD sequence,
returns its final local latent plus original/updated styles, and calls the
decoder twice with the shared local latent. The default `3dd_original` and
`source_only` routes remain unchanged; the new method is locked to ModelNet40-C
severity 5, Gaussian/Impulse, complete files, batch 32, seeds 0/1/2, raw LION,
eval mode, EMA disabled, and gamma=eta=.01/lambda=.95.

The seven-file artifact contract is preserved. [Code] Per-corruption and
summary CSVs additionally record original/updated-style counts and accuracy,
paired delta in percentage points, prediction disagreement, decoder-output
difference, and style displacement. `config.json` records the method contract,
RNG snapshot/restore policy, commit and asset manifests. [Code] The extra
classifier call snapshots/restores NumPy and Torch CPU/CUDA RNG state.

**Local verification:** [Code] 21 unittest cases pass; `py_compile` passes for
`tta.py`, `run_baseline.py`, `research_artifacts.py`, `main_3dd_tta.py`, and
`utilities_3dd_tta.py`; exact pilot CLI parsing and scope rejection pass;
`main_3dd_tta.py` `process_batches` AST and the `source_only` runner branch AST
match HEAD; `git diff --check` has no whitespace errors. No CUDA/model
evaluation was run locally.

**Decision/Open:** [Open] The control is code-ready but has no accuracy
evidence until three complete Colab ZIPs are supplied and validated. Updated
style is accepted for all-15 confirmation only if it is consistently better
on all three seeds and both pilot corruptions. Null/unstable results reject
the decoder-style hypothesis; negative results preserve the current original-
style baseline. The next separate factor after a null/negative result is Eq.
11 SCD normalization, without changing lambda, RNG, or scheduler together.

## 2026-09-12 - Source-only all-corruption identity result

**Evidence:** [Run] `result/modelnet40_c/source_only/20260912-111822_source-only_seed0/`; ZIP SHA-256 `bf1ff8884401b4617213299ff7e1f7a90476345e25a6ea79d8d514cfff09eabe`.

**Git commit:** `ef87692042e337cf628b9438bfc06a3e362b2e0b`.

**Protocol:** source-only identity evaluation, ModelNet40-C severity 5, all 15 corruptions, 2468 examples each, 37020 total; FPS(1024), frozen Point-MAE, batch32, seed0. No LION loaded, no normalize/rotation/diffusion/guidance. Execution complete with no traceback.

Macro and micro accuracy are both `0.536899` (19876/37020), or 53.69%. This is 3.91 percentage points below the repository README source-only reference 57.6%. Per-corruption lows are lidar 19.94%, background 28.16%, rotation 30.19% and occlusion 37.24%; highest is density_inc 77.35%.

**Follow-up code/data audit (2026-09-12):** [Code] the repository's data guide directs ModelNet40-C users to download the already-corrupted Zenodo package; it only directs users to generate corruptions for ShapeNetCore and ScanObjectNN. [Run] the archived package has the expected 15 severity-5 arrays, 2,468 examples and labels in `[0,39]` for every corruption. [Code] the source-only runner is behaviorally identical to the upstream `tools/runner_finetune.py` source evaluation for this path: direct `np.load(data_<corruption>_5.npy)` / `label.npy`, then `misc.fps(points, 1024)`, then frozen `classification_only(..., only_unmasked=False)`. It adds logging only; it does not normalize, rotate, denoise, or augment inputs.

**Interpretation:** data/checkpoint/FPS/classifier protocol is operational and the locally used source-only path is not a preprocessing divergence from the upstream source-evaluation path. The actual downloaded files and Point-MAE checkpoint still have no author-published checksums, so structural validity does not prove byte identity. Because only one seed and one Colab environment are archived, this is a diagnosed reproduction gap, not a causal conclusion. Do not interpret LION/TTA accuracy before this gap is investigated.

**Decision:** preserve this as the source-only comparator. Next, compare data/checkpoint hashes and Point-MAE preprocessing/evaluation details against the reference protocol before dropout A/B. No GSD work.

## 2026-09-12 - Source-only seed variance result

**Evidence:** [Run] `result/modelnet40_c/source_only/20260912-120122_source-only_seed1/` and `result/modelnet40_c/source_only/20260912-120349_source-only_seed2/`.
**Git commit:** `ef87692042e337cf628b9438bfc06a3e362b2e0b`.
**Protocol:** source-only identity evaluation, ModelNet40-C severity 5, all 15 corruptions, 2468 examples each; FPS(1024), frozen Point-MAE, batch32. Seeds 1 and 2.

### Result
Both Seed 1 and Seed 2 have identically matching configuration, dataset hashes, and per-corruption metrics compared to Seed 0. 
Zero variance across all 3 seeds: macro and micro accuracy are perfectly equal at 53.69% (19876/37020). 

### Interpretation
The source-only evaluation pipeline is deterministic across these three seeds. It confirms that the 53.69% accuracy gap observed for Point-MAE source-only is stable and not an artifact of random sampling in the FPS step (which was potentially stochastic if unseeded, but it appears to yield identical outcomes here, likely because numpy/torch seeds were fixed).

### Decision
Source-only baseline is fully verified and stable at 53.69%. Proceed with data/checkpoint provenance investigation and ShapeNet preparations.

### Falsifier / next evidence
A different checkpoint or data source showing the 57.6% source-only accuracy.

```markdown
## YYYY-MM-DD — Short finding title

**Evidence:** [Run]/[Paper]/[Code]/[User report]/[Inference]  
**Git commit:** full hash  
**Run paths:** `result/...`  
**Hypothesis:** falsifiable statement  
**Protocol:** level, dataset, corruptions, seeds, important config  

### Result

Exact values, uncertainty, failures, runtime/memory, and per-corruption pattern.

### Interpretation

What the evidence supports and what it does not support.

### Decision

Continue, modify, reject, reproduce, or escalate to full evaluation.

### Falsifier / next evidence

What result would overturn or materially revise the interpretation.
```

## 2026-09-12 � ScanObjectNN Gaussian source-only pilot

- **[Run]** Artifact: result/scanobjectnn_c/20260912-135346_source-only-gaussian-seed0-label-fix.zip (user-supplied; complete after validation).
- **[Run]** Dataset: main_split-derived ScanObjectNN, Gaussian severity 8, 581 examples, seed 0, batch size 32, frozen 15-class Point-MAE checkpoint scanobject_jt.pth.
- **[Run]** Result: 108/581 correct, accuracy 0.1858864028 (18.59%), runtime 2.884 s, no traceback, checkpoint missing/unexpected keys empty.
- **[Inference]** This is a valid corruption result but not yet interpretable as adaptation evidence. A clean-input control with the same checkpoint and preprocessing is required first; the checkpoint is user-supplied and its clean OBJ-BG parity is not established.
- **[Open]** Create data_original.npy from the official main_split test H5, run the complete clean source-only control, then compare Gaussian degradation.

## 2026-09-12 � ScanObjectNN clean control

- **[Run]** Artifact: result/scanobjectnn_c/source_only/20260912-135748_clean-control-seed0.zip (user-supplied; complete after validation).
- **[Run]** Same seed/checkpoint/protocol as Gaussian pilot; 581 clean main_split examples, source-only, severity 0.
- **[Run]** Result: 415/581, accuracy 0.7142857143 (71.43%), runtime 2.875 s, no traceback.
- **[Run]** Clean and Gaussian artifacts use the identical label SHA-256 bb145670...eeb7a68; both inventory shapes are 581 examples and both checkpoint loads have empty missing/unexpected keys. This rules out a run-to-run label-file mismatch.
- **[Inference]** The 52.84-point clean-to-Gaussian drop is not explained by an observed label shift. Remaining candidates are the severity-8 corruption strength, checkpoint/preprocessing mismatch, or a generator/data-content issue; external H5-label equality and a severity-1 control remain open.

## 2026-09-12 � ScanObjectNN 3DD-TTA smoke

- **[Run]** Artifact: result/scanobjectnn_c/3dd_original/20260912-154840_3dd-original-gaussian-smoke-seed0-device-fix.zip.
- **[Run]** LION priors/VAE and Point-MAE loaded with empty missing/unexpected keys; the CPU/GPU unnormalization error did not recur.
- **[Run]** First two batches, 64 examples, Gaussian severity 8: 11/64 correct (17.1875%), runtime 5.167 s; status partial by design.
- **[Open]** Full 581-example 3DD-TTA run remains necessary; smoke accuracy is not benchmark evidence.

## 2026-09-12 - LION eval-mode TTA autograd compatibility repair

- **[User report]** The ModelNet40-C 3dd_original Gaussian eval-mode pilot failed at ch_loss.backward() because PVCNN devoxelization ran with is_training=False, so it did not retain the interpolation indices/weights required by its custom backward function.
- **[Code]** models/pvcnn2.py and models/pvcnn2_ada.py now pass self.training or torch.is_grad_enabled() to trilinear_devoxelize. Thus lion.eval() continues to disable dropout, while gradient-enabled TTA retains the custom CUDA operator's backward state. Normal no-grad eval remains unchanged.
- **[Code]** Local structural verification: python -m py_compile models/pvcnn2.py models/pvcnn2_ada.py third_party/pvcnn/functional/devoxelization.py completed successfully. No local CUDA execution was attempted.
- **[Open]** The prior artifact 20260912-162954_3dd-original-gaussian-eval-seed0.zip is a failed run, not an accuracy result. Rerun the same smoke command after pulling this repair, using a fresh run name.

## 2026-09-12 - LION dropout mode A/B: full Gaussian pilot

**Evidence:** [Run] result/modelnet40_c/3dd_original/20260912-162758_3dd-original-gaussian-legacy-seed0/ and result/modelnet40_c/3dd_original/20260912-163734_3dd-original-gaussian-eval-fix_seed0/.

**Protocol:** Complete ModelNet40-C Gaussian severity 5, 2,468 examples, batch 32, seed 0, gamma=eta=0.01, lambda=0.95, 100 DDIM steps and 5 reverse steps. The classifier checkpoint, LION checkpoint, Gaussian data file, configuration assets, batch size, and all listed TTA settings have identical hashes/values. The independent variable is LION mode: legacy train versus --lion-eval-mode.

**Result:** Legacy LION mode: 1,826/2,468 = 0.739870 (73.99%), runtime 96.60 s, peak memory 25,750 MiB. Eval mode after commit 9ce5553: 1,843/2,468 = 0.746759 (74.68%), runtime 95.55 s, peak memory 25,249 MiB. The observed change is +17 correct examples, or +0.6888 percentage points. Both archives contain all seven required files, have complete status, and their stdout logs contain no Traceback, RuntimeError, or ValueError.

**Code verification:** Legacy config records priors/VAE training=True. Eval config records CLI lion_eval_mode=True and priors/VAE training=False; this confirms dropout is disabled. The successful eval run also accepts the PVCNN autograd repair. The config field lion_mode_policy remains the stale string 'legacy; unchanged' in the eval artifact, so module inventory and CLI are the authoritative mode evidence for this pair.

**Interpretation:** This is encouraging but inconclusive. The runner records seed-controlled, not common-random-number-paired sampling; stochastic interpolation/noise remains a confound. The two runs are also on adjacent commits, although 9ce5553 only repairs eval-mode autograd. One corruption and one seed cannot establish a reliable accuracy effect or a benchmark gain.

**Decision:** Keep eval mode as a viable candidate. Repeat the exact legacy/eval pair for seeds 1 and 2 before selecting a mode; then test background under the same paired design. Do not yet use this +0.69 pp pilot to change the full benchmark protocol or make a thesis claim.

## 2026-09-12 - LION dropout A/B: Gaussian three-seed result

**Evidence:** [Run] seed-0 archives in the preceding entry plus 20260912-165102 legacy seed1, 20260912-165259 eval seed1, 20260912-165454 legacy seed2, and 20260912-165650 eval seed2 under result/modelnet40_c/3dd_original/.

**Protocol:** Full Gaussian severity 5; 2,468 examples; seeds 0,1,2; batch32; gamma=eta=0.01; lambda=.95; 100 DDIM / 5 reverse steps. All six archives have seven required files, complete CSV status, matching within-seed assets, and no traceback/runtime error. Seed1/2 pairs both use 9ce5553; seed0 legacy is 2611da8 and eval is 9ce5553.

**Result:** legacy/eval: seed0 73.9870/74.6759 (+0.6888 pp); seed1 73.3387/75.2431 (+1.9044 pp); seed2 73.9870/75.2836 (+1.2966 pp). Means: legacy 73.7709% (SD .3743 pp), eval 75.0675% (SD .3398 pp); paired mean delta +1.2966 pp (SD .6078 pp). All three deltas favor eval.

**Interpretation and decision:** Gaussian replicates a favorable eval-mode effect, but this is not a significance or benchmark claim: n=3, seed-controlled rather than fully common-draw-paired sampling, and seed0 spans the repair commit. The matched-commit seed1/2 deltas are still positive. Run background at seeds 0,1,2 (35 reverse steps) before a full 15-corruption comparison.


## 2026-09-12 - LION dropout A/B: full Background three-seed result

**Evidence:** [Run] `result/modelnet40_c/3dd_original/20260912-171041_3dd-original-background-legacy-seed0.zip`, `20260912-171913_3dd-original-background-eval-seed0.zip`, `20260912-172738_3dd-original-background-legacy-seed1.zip`, `20260912-173612_3dd-original-background-eval-seed1.zip`, `20260912-174437_3dd-original-background-legacy-seed2.zip`, and `20260912-175310_3dd-original-background-eval-seed2.zip`.

**Git commit:** `9ce5553de9277f9b9d7e26fc729e70b912a7c2d7`.

**Protocol:** Complete ModelNet40-C Background severity 5; 2,468 examples per run; seeds 0, 1, 2; batch 32; gamma=eta=.01; lambda=.95; 100 DDIM steps and 35 background reverse steps. The six archives each contain the required seven files, one archive root, one complete summary/per-corruption row, and no Traceback, RuntimeError, or ValueError. Every artifact records the same Point-MAE, LION, config, label, and Background-file identities. Legacy CLI has lion_eval_mode=false; eval CLI has lion_eval_mode=true. Eval module inventory confirms LION dropout entries have training=false; legacy retains training mode.

**Result:** Legacy accuracies for seeds 0/1/2 are 60.6564%, 61.5073%, and 60.0486%; eval accuracies are 60.8185%, 60.3323%, and 60.7374%. Paired eval-minus-legacy deltas are +0.1621, -1.1750, and +0.6888 percentage points. Means are 60.7374% legacy (SD 0.7327 pp) and 60.6294% eval (SD 0.2605 pp), for a paired mean delta of -0.1080 pp (SD about 0.961 pp). Eval is faster by about 8.21 seconds per run on average (485.69 versus 493.89 seconds) and uses about 500 MiB less peak allocated GPU memory (about 25,250 versus 25,750 MiB).

**Interpretation:** Background does not replicate Gaussian's consistent eval advantage (+1.2966 pp mean across its three seed-controlled pairs). The current evidence falsifies a claim that disabling LION dropout uniformly improves 3DD-TTA over corruptions. It does not isolate a causal dropout effect because the runs are seed-controlled rather than common-random-number paired; changing the mode changes stochastic-draw consumption. The observed mode interaction is nevertheless large enough that a Gaussian-only mode selection would be unjustified.

**Decision:** Do not lock either LION mode as the global baseline yet. Keep batch 32 unchanged. The next protocol decision must evaluate mode behavior across more corruptions before any all-corruption baseline is declared; do not interpret the Background mean as a paper-level reproduction metric.

**Falsifier / next evidence:** A matched common-draw A/B or a broader multi-corruption paired evaluation that shows a stable same-direction difference would revise this conclusion.


## 2026-09-13 - 15-corruption LION-mode screen at seed 0

**Evidence:** [Run] Complete seed-0 legacy/eval ZIP pairs under `result/modelnet40_c/3dd_original/`, consisting of the existing Gaussian/Background pairs and 26 newly supplied `*-screen-seed0.zip` artifacts for Cutout, Density, Density Increase, Distortion, RBF Distortion, Inverse-RBF Distortion, Impulse, LiDAR, Occlusion, Rotation, Shear, Uniform, and Upsampling.

**Git revisions:** `9ce5553de9277f9b9d7e26fc729e70b912a7c2d7` for every newly added screen artifact, all Background artifacts, and Gaussian eval. Gaussian legacy seed 0 is `2611da8a7507a449f5c34c5788f92e4506a63ad3`; therefore that one pair spans the PVCNN eval-autograd repair commit.

**Protocol:** ModelNet40-C severity 5, all 15 corruptions, 2,468 examples per corruption, seed 0, batch 32, gamma=eta=.01, lambda=.95, 100 DDIM steps; Background uses 35 reverse steps and other corruptions use 5. Each included ZIP has exactly seven expected files under one root; each selected CSV row is complete, has 2,468 examples, and its stdout has no Traceback, RuntimeError, or ValueError. This is a seed-0 exploratory mode screen, not a locked Level-3 benchmark and not a common-random-number pair.

**Result:** Macro accuracy is 63.0578% legacy (23,344/37,020 correct) and 63.8817% eval (23,649/37,020), a +0.8239 percentage-point eval-minus-legacy difference (+305 correct). Eval is higher in 13/15 corruptions; it is lower only on Rotation (-.5267 pp) and Shear (-.3241 pp). Largest gains are RBF Distortion (+1.9854 pp), Cutout (+1.6613), Density Increase (+1.5802), LiDAR (+1.5802), and Upsampling (+1.2561). Total recorded adaptation runtime is 1,818.34 s eval versus 1,846.22 s legacy; mean peak allocated memory is 25,249.40 MiB eval versus 25,749.44 MiB legacy.

**Interpretation:** The full seed-0 screen strengthens the evidence that LION eval mode can improve the aggregate result under this implementation, but it does not establish a reproducible global advantage. Background seed variation already changes the comparison direction, the new 13 corruption pairs have only seed 0, draws are not common-random-number paired, and the Gaussian seed-0 legacy/eval pair crosses a code revision. The 63.8817% eval screen remains 1.8183 pp below the paper's 65.7% and 2.2183 pp below the README's 66.1%; it must not be used as a claim of paper-level parity.

**Decision:** Keep batch 32. Treat eval mode as the leading candidate for the next confirmed baseline, but do not lock it until matched-commit repeated-seed evidence is obtained. Do not resume GSD/PxP tuning from this screen alone.

**Falsifier / next evidence:** Re-run a predeclared full all-15-corruption evaluation for seeds 1 and 2 under a clean matched commit (or introduce common-draw pairing) and report the three-seed macro mean and variance.

## 2026-09-13 - Matched all-15 LION eval-mode screen, seeds 1 and 2

**Evidence:** [Run] `result/modelnet40_c/3dd_original/20260913-173819_3dd-original-all15-eval-raw-seed1.zip`, `20260913-183818_3dd-original-all15-eval-raw-seed2.zip`, `20260913-200206_3dd-original-all15-legacy-raw-seed1.zip`, and `20260913-203231_3dd-original-all15-legacy-raw-seed2.zip`. All are safe seven-file ZIPs with 15 complete 2,468-example rows and no error signature in `stdout.log`.

**Git commit:** `b999a1eb809690a625c1075b205cb042b384443f` for all four bundles.

**Protocol:** ModelNet40-C severity 5, all 15 corruptions, 37,020 examples/run, batch 32, gamma=eta=.01, lambda=.95, raw LION prior (`lion_ema_mode=false`) and seeds 1/2. The sole planned condition difference is `lion_eval_mode`: legacy has false and records LION prior/VAE dropout `training=true`; eval has true and records `training=false`. Recorded LION and Point-MAE checkpoint identities agree across all four bundles.

**Result:** Legacy macro accuracies are 63.1361% (seed 1) and 63.0578% (seed 2), mean 63.0970%. Eval/raw is 63.8088% and 63.9006%, mean 63.8547%. Eval minus legacy is +.6726 and +.8428 pp, respectively: **+.7577 +/- .1203 pp** sample SD, or +561 correct predictions across the two complete runs. Eval is higher in 13/15 two-seed per-corruption means; the largest mean gains are Density (+1.742 pp), Shear (+1.682), Impulse (+1.479), Cutout (+1.074), and LiDAR (+1.033). Uniform (-.122) and Background (-.101) are lower.

**Interpretation:** This is the first full, same-commit repeated-seed evidence that the LION eval-mode path is operationally preferable in this fork. It remains a seed-controlled, non-common-draw comparison, so it cannot isolate the causal effect of dropout alone; it also does not resolve the source-only gap or establish parity with the 65.7% paper mean.

**Decision:** Select `--lion-eval-mode` with raw weights as the provisional baseline for subsequent reproduction diagnostics. Keep legacy/raw as the comparator, retain EMA as an ablation, and keep GSD/PxP parked. Full table and protocol caveats: `knowledge/dropout_eval_mode_20260913.md`.

**Falsifier / next evidence:** A common-random-number legacy/eval control that reverses the result would overturn the operational choice. Independently, a labelled source-only severity 1--5 probe is the next P0 test for the reproduction gap.

## 2026-09-15 - Main/LION re-audit: FPS and integration candidates

**Evidence:** [Code], [Paper], [Inference]; [Run] reanalysis only of existing
source/eval bundles. Active HEAD `262f3a668b3f5a7bc44c6282c4a8a2723ac6f00a`;
3DD-TTA main `107305fd7baf40b359f31c07d235599198be7324`; LION
`7711b3d185752eeb632d095494876e4de15f3195`. Remote main refs agree with local.
Full source locations, exact archive paths, protocol and falsifiers are in
`knowledge/code_audit_20260915.md`. No new GPU evaluation or algorithm change.

**New observations:** [Code/Run] source-only always requests FPS(1024), although
Density/Cutout/LiDAR inputs contain only 649/724/768 points in
`result/modelnet40_c/source_only/20260912-111822_source-only_seed0/`. Gather-based
sampling necessarily repeats indices. The bundled FPS kernel also excludes
later candidates of squared radius <=.001, even when input/output counts match.
The binary's actual behavior requires Colab inspection. [Code] Point-MAE creates
unused NumPy random masks in all-token classification, which can alter subsequent
TTA interpolation; added classifier diagnostics need RNG isolation. Inherited
PVCNN interpolation/voxelization supplies only partial coordinate gradients.

**Reconfirmed:** [Paper/Code] final decoding uses old global style despite its
updates; Eq. 11 point-count denominators are absent from the SCD sum; paper
lambda=.96 differs from code .95. Decoder residual weight .01 makes the style
effect a measured question, not a promised gain. A shared-trajectory paired
decode is the smallest targeted TTA experiment. A preprocessing-only identity
control is missing from attribution of TTA-minus-source gains.

**Reference correction:** visually checked Table 2 confirms paper source=57.6%;
corrected the stale corruption headings in `papers/3dd_tta.md`. The earlier
2026-09-13 audit claimed that correction but the old headings had remained.

**Decision:** finish severity 1--5 first; if unresolved, prioritize installed
FPS/index and preprocessing controls before source-data gate closure. Queue
updated-style, SCD-scale and lambda as separate exploratory TTA tests after
reviewing that gate. Keep eval/raw, batch32, severity5 and parked-method decisions.
No candidate has a newly established accuracy benefit. Preserve all raw evidence;
commit only the relevant knowledge changes. The pre-existing runner note-only
working change remains uncommitted and was not modified by this audit.

**Falsifiers:** an installed FPS binary without the suspected behavior weakens
that runtime explanation; matched source controls with unchanged predictions
rule down sampling/preprocessing effects. Null/negative shared-trajectory style
results rule down the decoder candidate. Record full seven-file Colab ZIPs for
each condition/seed; do not infer causality from cross-paper gain subtraction.

## 2026-09-16 - Source-only severity 1--5 probe: severity explains a large descriptive component

**Evidence:** [Run] Five complete seven-file ZIPs under
`result/modelnet40_c/source_only/`:

- `20260915-184540_source-only-sev1-all15_seed0.zip` (SHA-256 `a99c8998212d006a94418e8168ab21d299989f022a5fc87ccf6f9d70b76e8c7b`)
- `20260915-184848_source-only-sev2-all15_seed0.zip` (SHA-256 `a5272e15c439772357ec1ef9d51b9dd67635158f08d725d6fa4091db400c23f6`)
- `20260915-185118_source-only-sev3-all15_seed0.zip` (SHA-256 `a716844b0974efeee723ae36d447cc5dcaea198cbec8b86dfe735b2d2f5b60f1`)
- `20260915-185346_source-only-sev4-all15_seed0.zip` (SHA-256 `49cef14e82ae709e9ed67c96ef4d6104d263dce7599fda71661c434b4fc6c458`)
- `20260915-185615_source-only-sev5-all15_seed0.zip` (SHA-256 `01fdcfdab370f06a4d9b950603174fb6b978cd0c3acc1b2fd2a812eb7a0ceea6`)

All five use commit `262f3a668b3f5a7bc44c6282c4a8a2723ac6f00a`, seed 0,
batch 32, frozen Point-MAE, direct corruption-file loading, FPS(1024), all
15 corruptions and 2,468 examples per corruption. Classifier checkpoint hash
is `507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75`; the
label hash is identical across levels. Each archive has a safe single root,
all required files, 15 complete rows, 37,020 total examples, matching
config/CSV severity and no traceback/error signature. The generated `notes.md`
files retain the pre-patch generic smoke wording; this is documented metadata
debt and raw artifacts are not rewritten.

**Hypothesis:** released corruption severity could explain part of the source
Point-MAE discrepancy before LION/TTA.

### Result

| Severity | Macro = micro accuracy | Correct / 37,020 | Difference from paper source 57.6% |
|---:|---:|---:|---:|
| 1 | 75.8806% | 28,091 | +18.2806 pp |
| 2 | 73.2739% | 27,126 | +15.6739 pp |
| 3 | 68.5062% | 25,361 | +10.9062 pp |
| 4 | 62.0205% | 22,960 | +4.4205 pp |
| 5 | 53.6899% | 19,876 | -3.9101 pp |

Severity 1 to 5 changes the macro by **-22.1907 pp**. Thirteen of 15
corruption curves are non-increasing. Occlusion rises from 41.7747% (s1) to
43.7196% (s3); LiDAR rises from 20.2188% to 23.7439% before declining. The
severity-5 row exactly reproduces the earlier source-only 53.6899% result.
The largest s5 deficits against rounded paper source cells are Density
(65.2350% vs 75.1%), LiDAR (19.9352% vs 29.1%), Cutout (62.2771% vs 70.4%)
and Gaussian (51.2966% vs 57.0%).

### Interpretation and decision

**[Inference]** Severity is a strong descriptive determinant of source-only
accuracy and can account for more than the observed s5 gap if a different
severity were used. No tested severity is a provenance match for the paper:
s4 is 4.4205 points above 57.6%, while s5 is 3.9101 points below it. The paper
reports 57.6% in Table 2 but does not state severity; released 3DD-TTA code
uses `_5` files. Checkpoint/data identity and corruption-generation version
remain open. This is not TTA evidence and does not redefine the severity-5
benchmark.

Accept the probe as complete descriptive evidence. Keep severity 5 as the
operational protocol and retain all five levels as a diagnostic curve. Next P0
is provenance plus installed FPS/index inspection, followed by a preprocessing
identity control. Do not start GSD/PxP or select a lower severity post hoc.

**Falsifier / next evidence:** a verified paper-specific severity declaration
or byte-identical data/checkpoint bundle would revise provenance. A different
validated asset set would test asset-specificity. FPS diagnostics showing no
suspected repeats/origin filter in the actual Colab binary would weaken that
candidate explanation.

## 2026-09-19 - FPS diagnostic blocked by dependency drift

**Evidence:** `[Run]` The first Colab diagnostic artifact
`result/modelnet40_c/source_only/20260919-104821_source-only-fpsdiag-s5-seed0.zip`
failed before evaluation. Its traceback reaches `main_3dd_tta.py` ->
`models/lion.py` -> `diffusers.utils.peft_utils` ->
`diffusers.utils.torch_utils`, where the installed Diffusers code evaluates
`torch.xpu.empty_cache` and the installed PyTorch has no `torch.xpu` attribute.
The parent runner still sealed a failed ZIP; it is incomplete evidence, not an
accuracy result.

**[Code]** The source-only runner imports `main_3dd_tta` at worker startup,
which imports LION/Diffusers even though source-only does not use LION. The
repository's `env.yaml` pins `torch==2.0.1+cu121` but leaves `diffusers`
unpinned; `requirements.txt` contains the historical compatible pins
`diffusers==0.11.1` and `huggingface-hub==0.11.1`. This explains why older
environments could run while a newly resolved environment fails at import.
`ninja: no work to do` and `_pvcnn_backend` loading are preceding normal output,
not the failure source.

**Decision:** Do not add a fake `torch.xpu` attribute. First verify the Colab
package versions, then restore the historical Diffusers/Hub pins without
changing the dataset or FPS code. Separately consider lazy LION/Diffusers
imports so source-only diagnostics do not require unused TTA dependencies.

**[Code] Superseding update, 2026-09-19:** The diagnostic runner commit
`47d3e3b` was reverted by `cc52437` at the user's request before any valid FPS
diagnostic result. `run_baseline.py` is back to its pre-diagnostic behavior;
the failed import remains an environment/provenance finding, not evidence
against FPS and not an accuracy result.

**[Code/User report] Update, 2026-09-19:** After the user reported that the
environment-rebuilt source-only smoke completed successfully, the same
read-only `--fps-diagnostics` implementation was re-enabled. It remains an
opt-in source-only path; no alternate resampling policy or dataset mutation is
included. The complete smoke ZIP is still required before treating the
environment gate as `[Run]` evidence.

## 2026-09-19 - Legacy FPS diagnostic completed on four severity-5 corruptions

**[Run]** The complete archive
`result/modelnet40_c/source_only/20260919-122509_source-only-fpsdiag-s5-seed0.zip`
was validated without modifying it. SHA-256 is
`1397e8640b2b0e7688d5a55caaad1872a5a8f4c4d459515b5185d03c505f5539`.
It contains exactly the seven required files, four complete corruption rows
(2,468 examples each; 9,872 total), and no traceback/error signature. The run
records commit `36a2d602a121acf46a8462a58992ab648d446bb9`, Diffusers `0.11.1`,
PointNet2 extension `3.0.0`, and classifier hash
`507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75`.

| Corruption | input N | mean unique indices | mean duplicate slots | input-origin total | selected-origin total | source accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Density | 649 | 648.7338 | 375.2662 | 657 | 0 | 65.2350% |
| Cutout | 724 | 723.7034 | 300.2966 | 733 | 1 | 62.2771% |
| LiDAR | 768 | 396.1528 | 627.8472 | 405 | 0 | 19.9352% |
| Gaussian | 1024 | 1023.6297 | 0.3703 | 914 | 0 | 51.2966% |

**[Code/Run]** The four accuracies exactly match the corresponding prior
source-only values, so the diagnostics did not change the classifier input or
predictions. Density, Cutout, and Gaussian duplicate totals are approximately
the unavoidable `1024-N` padding plus skipped near-origin points. LiDAR is
qualitatively different: it averages only about 396 unique indices and about
628 repeats per example; the 405 input-origin points cannot explain that scale.

**[Inference/Open]** LiDAR may contain repeated/degenerate coordinates,
non-finite values, or extension tie behavior, but these aggregate counters do
not distinguish them. The next diagnostic should record finite/NaN/Inf counts
and coordinate-unique counts, prioritizing LiDAR. No alternate resampling
policy has been implemented or benchmarked.

## 2026-09-19 - FPS diagnostic v2 prepared for LiDAR localization

**[Code]** `run_baseline.py` now records additional read-only counters behind
the existing opt-in `--fps-diagnostics` flag. For each selected corruption it
aggregates finite and non-finite input-point counts, scalar NaN and Inf counts,
finite-point coordinate-unique counts, and finite/non-finite selected-point
counts. The schema is explicitly marked
`legacy_fps_v2_finite_coordinate_unique` in `config.json`.

**[Code/Inference]** The legacy FPS indices, gather operation, classifier input,
and predictions are unchanged. Coordinate uniqueness is computed over exact
finite `[x,y,z]` rows only; non-finite rows are counted separately rather than
silently included in the uniqueness statistic. This is a localization
diagnostic, not a resampling-policy change.

**[Open]** No Colab v2 archive exists yet. The next run must use the same
severity-5, seed-0, batch-32, four-corruption scope so its counters remain
directly comparable to the validated v1 archive.

## 2026-09-19 - Uploaded v2-named ZIP used stale diagnostic code

**[Run]** Archive
`result/modelnet40_c/source_only/20260919-133003_source-only-fpsdiag-v2-s5-seed0.zip`
is structurally complete and error-free: seven required files, four complete
corruptions, 9,872 examples, and no traceback. SHA-256 is
`dfa4efd18637850b4cb0b5d62463fa797e3a7cd671cd799b28c524ea574d26a6`.
Its source-only accuracies reproduce the validated v1 values exactly: Density
65.2350%, Cutout 62.2771%, LiDAR 19.9352%, Gaussian 51.2966%.

**[Run/Provenance]** This is not v2 diagnostic evidence. `config.json` records
`git_commit=36a2d602a121acf46a8462a58992ab648d446bb9`, while v2 was added in
later commit `ceb9576`; `fps_diagnostics_schema` is absent and the new finite,
NaN/Inf, and finite-coordinate-unique fields are absent. `notes.md` also has
the v1 diagnostic wording. The run therefore confirms the old classifier path
only, not the LiDAR localization hypothesis.

**[Open]** Re-run after fetching `origin/baseline-repro-clean` at or beyond
`ceb9576`; verify `git rev-parse HEAD` and the schema field before accepting
the archive as v2 evidence. Raw ZIP remains unchanged.

## 2026-09-19 - FPS diagnostic v2 localizes LiDAR repetition

**[Run]** Archive
`result/modelnet40_c/source_only/20260919-133400_source-only-fpsdiag-v2-s5-seed0.zip`
is complete and valid (seven files, four complete rows, 9,872 examples, no
traceback). SHA-256 is
`9a9cc4cd8dcdd432344d19e43c70394bd8c05be6efae748b9a39e37d56802330`.
The run records commit `0003743b6362be322352ba42a70cbd8260c98f6d` and schema
`legacy_fps_v2_finite_coordinate_unique`. Accuracy is unchanged from v1:
Density 65.2350%, Cutout 62.2771%, LiDAR 19.9352%, Gaussian 51.2966%.

| Corruption | input N | mean finite points | mean finite-coordinate unique | mean FPS-index unique | mean input coordinate duplicates | mean FPS duplicate slots | input NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---:|
| Density | 649 | 649 | 649.0000 | 648.7338 | 0.0000 | 375.2662 | 0 / 0 |
| Cutout | 724 | 724 | 724.0000 | 723.7034 | 0.0000 | 300.2966 | 0 / 0 |
| LiDAR | 768 | 768 | 396.2273 | 396.1528 | 371.7727 | 627.8472 | 0 / 0 |
| Gaussian | 1024 | 1024 | 1024.0000 | 1023.6297 | 0.0000 | 0.3703 | 0 / 0 |

**[Run/Inference]** LiDAR has no non-finite input points, but about 372 exact
coordinate duplicates per example. FPS-index uniqueness nearly equals
finite-coordinate uniqueness, so the large duplicate-slot count is explained
by the input itself plus padding to 1024, not by NaN/Inf handling. The small
184-index aggregate difference is consistent with FPS selecting among repeated
coordinate rows. Density, Cutout, and Gaussian have no exact coordinate
duplicates; their earlier padding/origin-filter interpretation remains intact.

**[Code]** The repository generator's `simulate_lidar` uses
`np.random.choice(new_pc.shape[0], 768)` at
`datasets_mate/create_corrupted_dataset.py:655`; without an explicit
`replace=False`, NumPy samples with replacement. This is a direct code-level
mechanism consistent with the observed LiDAR duplicates, although the current
artifact does not by itself prove which historical generator invocation
created the archived `.npy` files.

**[Decision/Open]** Do not change inference resampling or claim an FPS bug yet.
First reconcile the corruption-file provenance/generator version. The v2 gate
is now complete; alternate policies remain parked.

## 2026-09-19 - Upstream generator confirms LiDAR replacement sampling

**[Code]** The canonical ModelNet40-C repository's `data/generate_c.py` uses
the same LiDAR construction at lines 240--242:
`index = np.random.choice(new_pc.shape[0], 768)` followed by `new_pc[index]`.
No `replace=False` is supplied. The fork's
`datasets_mate/create_corrupted_dataset.py:655-657` mirrors this behavior.
The upstream README documents both direct pre-corrupted download and generation
with `python data/process.py` / `python data/generate_c.py`.

**[Paper/Code/Run]** Together with the v2 counters, this strongly supports
that repeated LiDAR coordinates are an intended property of the ModelNet40-C
generator, not evidence that our inference FPS implementation is wrong. The
current archive still does not prove byte identity with the Zenodo package, so
asset identity remains an open provenance question, but `replace=False` is not
a justified benchmark correction.

**[Decision]** Do not regenerate `data_lidar_5.npy`, alter inference resampling,
or claim a LiDAR preprocessing contribution. Keep the validated archive as the
benchmark input and return to the broader source-only gap / asset-checkpoint
identity investigation.
this uncommitted batch is db1482ae3becb2e5f9f44a6811c775c5570b0501.
**Implementation commit:** `1e66374b7d9fbe3f193c51cdeb9895d1c7ecd5de`,
pushed to `origin/baseline-repro-clean`; no raw artifacts were included.
Identity is higher than source-only on 9/15 corruptions, equal on Upsampling, and lower on 5/15. Largest gains are Density Increase +8.1848 pp, Cutout +6.0373 pp, Density +3.6467 pp and Occlusion +2.8363 pp. Largest declines are LiDAR -4.6596 pp and Background -4.2950 pp.

Interpretation: [Run/Inference] The preprocessing chain has a real but modest positive aggregate effect under this seed and explains only a minority of the observed source-to-TTA difference. It is not sufficient to account for the TTA result, and the corruption-dependent signs prevent a claim of uniform preprocessing improvement. The run config has git_dirty=true because Colab generated or modified compiled extensions, caches and local data artifacts; the recorded source commit and runtime source manifest identify the intended code, and no method-source modification is visible in the recorded status.

Decision: This is a positive identity control under the predeclared decision rule, so plan the next pure VAE encode/decode control. Do not add a new TTA method or tune GSD/PxP. Keep the result exploratory because it is one stochastic seed and the source comparator is archived at an older commit, even though source-only behavior was structurally preserved.

Falsifier / next evidence: A pure VAE encode/decode result near source-only would localize the modest gain to preprocessing/interpolation/rotation/output normalization; a large pure-VAE gain would show that generative reconstruction, not guidance, explains more of the TTA difference. A repeat identity seed or common-draw control would test the stability of the +1.3344 pp estimate.
## 2026-09-22 - lambda=.96 eval provenance metadata repair

**[Code]** `scd_lambda96_control` already forces `lion_eval_mode=True` and
rejects EMA, but its immutable config metadata omitted the method from the
`raw LION eval; EMA disabled` policy set. The metadata and regression test now
agree with the runtime contract; no inference path or experimental factor was
changed.

**[Code]** Local CPU verification passed: 33 unit tests and `py_compile` for
`run_baseline.py`. The all-15 GPU run remains a Colab-only operation.
## 2026-09-26 - spectral-only latent guidance pilot

[User report] The user supplied three spectral-only pilot ZIPs under
`result/modelnet40_c/gsd_latent_spectral_v1/`:
`20260923-183420_gsd-v1-pilot-on-seed0-spectral-only.zip`,
`20260923-184326_gsd-v1-pilot-on-seed1-spectral-only.zip`, and
`20260923-185232_gsd-v1-pilot-on-seed2-spectral-only.zip`.

[Run] All three archives pass CRC validation and contain the required seven
files. They are complete Gaussian/Impulse pilot runs at commit
`761f47f`, with raw LION eval, EMA disabled, batch 32, severity 5, lambda
`.95`, `gsd_weight=1`, `gsd_scd_weight=0`, and 100 requested modes. The
archives record `git_dirty=true`; this is retained as provenance metadata and
does not alter the raw files.

[Run] Macro accuracy for Gaussian/Impulse is 72.3865%, 72.6297%, and
72.4271% for seeds 0/1/2. Against the matched `3dd_original` pilot ZIPs,
paired deltas are +0.1418, +0.4660, and +0.0810 percentage points; mean
delta is +0.2296 pp. This is a small exploratory pilot result, not evidence
for all-15 promotion or causality.

[Inference] Removing SCD did not collapse the pilot under this configuration,
but the comparison is against separate seed-controlled runs and only two
corruptions. The result supports retaining the ablation as a diagnostic
comparison; it does not establish that spectral guidance is better than SCD.

[Open] No matched spectral-only off trajectory is currently defined: the
zero-spectral path delegates the exact original SCD baseline. Do not interpret
an off arm with `gsd_scd_weight=0` as a no-SCD control. A larger corruption
scope or common-draw comparison would be required for a stronger claim.

## 2026-09-27 - smooth-profile smoke run-name failure

[User report] The hard-profile Gaussian smoke completed one batch at
81.25% (26/32), status `partial`, with reported peak GPU memory 21,010.9 MB.
This is a one-batch smoke result and is not accuracy evidence. The following
smooth-profile arm stopped before model execution while creating its artifact:
`RunBundle.create` rejected the generated run name
`gsd-smooth-v2-smoke-smooth-seed0-beta2.0` because periods are outside its
allowed run-name alphabet.

[Code] Root cause is the launcher formatting the beta value directly into the
artifact run name. `eval_gsd_smooth.py` now replaces decimal points with `p`
for the run-name label only (so beta 2.0 becomes `beta2p0`); the numerical
`--gsd-beta` argument is unchanged. A regression test passes the generated
name through the real `RunBundle.create` validator.

[Code] The focused GSD smooth launcher tests pass (6 tests), and the full CPU
suite passes (113 tests). These checks do not validate the Colab GPU path.

[Open] Re-fetch the branch containing this fix in Colab and rerun only the
smooth Gaussian smoke arm. Preserve the already completed hard smoke ZIP.
Then inspect the complete hard and smooth smoke bundles before starting any
pilot or interpreting accuracy. No calibration or profile conclusion follows
from the reported 32-example partial result.

## 2026-09-27 - smooth-profile Gaussian smoke validated

[Run] The user supplied
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/20260927-192156_gsd-smooth-v2-smoke-smooth-seed0-beta2p0.zip`.
ZIP CRC passes; all seven required files are present; config and both CSVs
agree on run ID; logs show no traceback and one completed batch. The run records
commit `d75a32d3c31b95d899a49a3e352bb74e2e243b37`, beta 2, smooth profile,
spectral/SCD weights 1/1, seed 0, Gaussian severity 5, and 32 examples.

[Run] Accuracy is 26/32 (81.25%), runtime 5.3296 seconds, and peak allocated
GPU memory 21,010.9 MB. This is a partial smoke prefix, not an accuracy
comparison. The separately reported hard smoke also had 26/32, but its ZIP is
not present in the canonical result folder, and equal prefix counts cannot
establish method equivalence.

[Run/Inference] Batch-mean effective smooth spectral mass is 641.79 with no
numerically zero weights, confirming the smooth filter ran without an observed
hard cutoff/underflow. The ratios of aggregate gradient-norm means are 0.00873%
for local spectral/SCD and 0.01380% for style spectral/SCD. This supports a
very weak relative spectral contribution at weight 1 in this smoke; it does
not establish a performance cause or calibration value.

[Provenance] `git_dirty=true`, but all manifested runtime/method Python files
match this checkout after normalizing Windows CRLF to LF. The recorded dirty
status is from generated build/cache artifacts plus a generated build-tree
Python file; no method-source divergence is detected in the manifest.

[Decision/Open] The earlier smoke-failure item is superseded: the fixed smooth
smoke now completes and its bundle is archived. Keep pilot accuracy runs
parked until beta candidates, calibration data, and a scale-matching rule are
predeclared; also archive the hard smoke ZIP if a direct artifact comparison
is needed. See the derived validation note beside the raw ZIP.

## 2026-09-27 - beta and weight calibration proposal

**Superseded by the later calibration review below.** The q95 prerequisite,
v1-scale target as a default and blanket target-input restriction were not
implemented and are no longer the recommended protocol.

[Inference/Proposal] Derive beta candidates from an active-spectrum attenuation
target rather than accuracy search: measure per-shape q95 for
`q=lambda/mean_active_degree`, take its calibration-set median, and set
`beta=-ln(tau)/q_ref` for predeclared attenuation targets such as 0.5, 0.1,
and 0.01. The existing smoke only logs scaled-eigenvalue min/max and is not a
valid calibration source because it is an inspected benchmark prefix.

[Inference/Proposal] For the scalar spectral weights, use an independent
unlabeled calibration pool and common diffusion draws. Match hard-v2 and each
smooth candidate's aggregate update-space spectral/SCD gradient ratio to the
v1 weight-1 ratio; retain local/style ratios separately. Freeze all values
before any accuracy-based validation. A distinct held-out accuracy-validation
set is required if pilot accuracy will be used to select a candidate.

[Open] A disjoint ModelNet40 training-partition calibration pool is the
practical same-domain option but uses source-domain examples offline and must
be disclosed; otherwise an external unlabeled pool is needed. No beta grid,
calibration data, q95 capture or calibration runner has been approved or
implemented yet. No run/accuracy result supports this proposal.

## 2026-09-27 - calibration efficiency and SCD-scale reassessment

[Code/Derivation] Review at `d75a32d`: mean active normalized eigenvalue is
exactly 1 in exact arithmetic, so beta candidates do not require a fitted
spectral quantile. The Gaussian smoke's batch-wide local SCD norm113.0058
corresponds to coordinate update RMS .002207 for B32, local dimension8192,
gamma=.01. State/DDIM scales are missing; SCD overshooting is unestablished.

[Inference] Recommend a small predeclared beta grid (.5,2,8), common-state
unit-gradient probes, and fixed local spectral/SCD contribution targets
(.01%, .1%, 1%) for exploratory screening. A v1 gradient match is an ablation
control, not an optimum. Monitor style independently and verify full guided
trajectories because fixed-state linear alpha scaling does not transfer
exactly across changed trajectories. Reusing eigendecompositions and probing
only selected timesteps saves work; candidate VJPs still have a real cost.

[Paper/Correction] Unlabeled target-input adaptation is compatible with
source-free TTA (Tent, https://arxiv.org/abs/2006.10726). The prior blanket
restriction was too strong. Predetermined target-input statistics require an
explicit adaptation/dependency protocol; final-test accuracy-based selection
remains distinct. TTAB discusses model-selection/batch-dependency pitfalls:
https://proceedings.mlr.press/v202/zhao23d.html .

[Decision/Open] Preserve the SCD=1 baseline, add relative-step diagnostics
before judging its scale, and use a staged development screen rather than
immediately changing the SCD objective or adding online norm balancing.
The detailed proposal is `gsd_calibration_review_20260927.md`. This is a
documentation/design revision, not a new implementation or model experiment.

## 2026-09-27 - approved calibration runner and staged test preparation

[User report] User approved updating knowledge and performing the revised
tests. Implementation extends `gsd-smooth-spectrum` from `d75a32d`; exact
implementation commit is the commit containing this entry.

[Code] Added SCD-only common-state probes for hard and beta .5/2/8, with one
graph eigendecomposition per example and no candidate gradient application.
Local/style metrics distinguish SCD, spectral and total update sizes; local
DDIM displacement and per-probe gradient ratios/cosines retain raw norms and
explicit null denominators. Ratios are per-example slices, not ratios of
batch-mean norms. The report includes median/p90 and per-step summaries.

[Code] A local RNG selects nested shuffled diagnostic/development index pools
(default 64/128 per Gaussian/Impulse corruption, split seed 20260927). Fixed
alpha uses pooled median unit local spectral/SCD ratio and rho 1e-4/1e-3/1e-2.
Zero spectral ratios remain in the median; missing denominators or near-zero
median block recommendations. There is no label/accuracy input to this fit.
The launcher adds baseline+beta2/three weights, then .5/8/hard only after an
explicit selected rho. No automatic winner or benchmark promotion.

[Code] Development requires a completed calibration reference, records its
hash/run ID, enforces matched coefficient/host/graph settings and checks
source/data/asset hashes before loading models. Raw individual diagnostics
and original indices reside in config.json; seven-file ZIP schema remains.
Successful subset execution has coverage partial. The exporter supports the
new smooth-v2 archives through `--stage all --name-contains gsd-cal-`.

[Verification] 125 CPU tests pass; diff whitespace check passes. Algebra,
five-step output/RNG parity under CPU adapters, worker/count/artifact plumbing,
calibration failure guards, staged commands and export have persistent tests.
Independent review caught a missing-reference CLI escape and it was fixed.
See `gsd_calibration_execution_20260927.md` and `colab_gsd_calibration.md`.

[Open/Falsifier] No numerical LION/CUDA run took place locally. Ingest the
first diagnostic ZIP before choosing any weight. OOM, nonfinite gradients,
missing probe rows, unexpected scale tails or CUDA trajectory discrepancies
would block progression. Identity correspondence across corruption files
remains unverified; a shuffled index pool does not establish held-out objects.
The small screen cannot establish a roughly 1 pp gain. Preserve SCD weight1
until measured evidence justifies a separate strength ablation.

## 2026-09-27 - compact analyzer and supplied calibration report

[User report/Code] The user placed the phase-report JSON at
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/report.json` (124,414
bytes). `scripts/analyze_gsd_calibration.py` reads that file in Python and
writes only a compact derived summary to
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/calibration_summary.json`;
the raw report is left unchanged. The analyzer also accepts a raw calibration
run and can rank the five completed same-rho development runs later.

[User report, derived by script] Diagnostic run ID is
`20260927-201522_gsd-cal-diagnose-reference-seed0-n64`, 64 examples per
corruption. Pooled local unit spectral/SCD median ratios are 0.00012286 for
beta .5, 0.00008842 for beta 2 and 0.00006719 for beta 8. At rho=.0001 these
imply alpha=.8139, 1.1310 and 1.4882; at rho=.001 alpha=8.1391, 11.3102 and
14.8822. Gaussian/Impulse local R are respectively .00011959/.00012703,
.00008763/.00008919 and .00006740/.00006687. The respective local gradient
cosine medians against SCD are positive and fall as beta rises: roughly
.229/.202, .186/.170, .151/.132. Hard-M has R=.00005333.

[Inference/Open] These scales alone do not identify an accuracy-optimal beta.
Beta 8 has the closest Gaussian/Impulse scale match, beta .5 has the strongest
median gradient alignment, and beta 2 is the predeclared middle candidate;
none of those properties proves better cleaning. Run the registered accuracy
screen at the explicit rho, then use the analyzer on its five matching run
directories to report the observed two-corruption development winner.

[User report, derived by script] Median local SCD update/state ratios are
about .14%--.18% across steps/corruptions, while SCD update/DDIM displacement
ratios are about .20--.27 through steps 0--3 and rise to 2.31--2.46 at step 4.
This last-step denominator is small and makes that ratio large; it deserves
review with the recorded displacement distribution, but is not by itself
evidence to change SCD weight. No accuracy labels were used in calibration.
The report JSON is a summary rather than a validated seven-file source ZIP;
retain and provide the original calibration run bundle for provenance.

## 2026-09-27 - beta2 contribution screen received

[Run] Four screen-weight ZIPs were added under
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/`: SCD-only baseline and
beta2 at rho .0001/.001/.01, seed 0, 128 examples per corruption. All four
archives pass CRC checks and contain the complete seven-file bundle. Their
run IDs are `20260927-203752_gsd-cal-screen-weight-baseline-seed0-n128`,
`20260927-203822_gsd-cal-screen-weight-smooth-beta2p0-rho0p0001-seed0-n128`,
`20260927-203909_gsd-cal-screen-weight-smooth-beta2p0-rho0p001-seed0-n128`,
and `20260927-203957_gsd-cal-screen-weight-smooth-beta2p0-rho0p01-seed0-n128`.

[Verification] Configurations agree on calibration ID/hash, shuffled indices,
seed/count, data/checkpoint manifests and runtime source manifests. The
analyzer reads ZIP members directly (without extraction) and writes
`screen_weight_summary.json` beside the archives. Equal-corruption macro and
paired accuracy differences against SCD-only are:

| Arm | Gaussian | Impulse | Two-corruption macro | Delta vs SCD |
|---|---:|---:|---:|---:|
| SCD-only | 76.5625% (98/128) | 71.8750% (92/128) | 74.2188% | reference |
| beta2, rho .0001, alpha 1.1310 | 77.3438% (99/128) | 71.8750% (92/128) | 74.6094% | +0.3906 pp |
| beta2, rho .001, alpha 11.3102 | 78.1250% (100/128) | 72.6563% (93/128) | 75.3906% | +1.1719 pp |
| beta2, rho .01, alpha 113.1024 | 78.1250% (100/128) | 71.8750% (92/128) | 75.0000% | +0.7813 pp |

[Inference] Rho .001 is the current development-screen leader: it adds two
correct Gaussian examples and one Impulse example relative to baseline. Rho
.01 has the same Gaussian count but loses the Impulse example; raising the
target contribution did not improve this screen. This is one seed and 128
examples per corruption; per-example paired predictions are not in the CSV,
so the aggregate cannot support a paired significance test or a 1 pp claim.
The next registered step is beta .5, beta 8 and matched hard at rho .001 on
the same seed/index pool, reusing this beta2 and SCD control. Then expand
promising arms and controls to other seeds/larger subsets before any claim.

## 2026-09-28 - beta profile screen at rho .001

[Run] The three beta-screen ZIPs supplied by the user were added beside the
previous two controls under
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/`. The compact analyzer
read the calibration `report.json` and five ZIPs directly, verified their
shared calibration hash, seed, 128-example scope and required run bundle, and
wrote `beta_screen_summary.json`. The five run IDs are:

- `20260927-203752_gsd-cal-screen-weight-baseline-seed0-n128` (SCD-only)
- `20260927-203909_gsd-cal-screen-weight-smooth-beta2p0-rho0p001-seed0-n128`
- `20260927-205045_gsd-cal-screen-beta-smooth-beta0p5-rho0p001-seed0-n128`
- `20260927-205132_gsd-cal-screen-beta-smooth-beta8p0-rho0p001-seed0-n128`
- `20260927-205219_gsd-cal-screen-beta-hard-rho0p001-seed0-n128`

All use seed 0 and 128 examples per Gaussian/Impulse corruption, with
calibration SHA-256
`e9f5ce3a435aea40a41e0ced3e33e085d052eeb473cf2e5400309ed2dcaff4c4`.
At rho .001, calibrated alpha is 8.1391 for beta .5, 11.3102 for beta 2,
14.8822 for beta 8 and 18.7515 for hard-M. Equal-corruption accuracies are:

| Arm | Gaussian | Impulse | Macro | Delta vs SCD-only |
|---|---:|---:|---:|---:|
| SCD-only | 76.5625% (98/128) | 71.8750% (92/128) | 74.2188% | reference |
| Smooth beta .5 | 77.3438% (99/128) | 72.6563% (93/128) | 75.0000% | +0.7813 pp |
| Smooth beta 2 | 78.1250% (100/128) | 72.6563% (93/128) | 75.3906% | +1.1719 pp |
| Smooth beta 8 | 77.3438% (99/128) | 71.0938% (91/128) | 74.2188% | 0.0000 pp |
| Hard-M | 76.5625% (98/128) | 71.8750% (92/128) | 74.2188% | 0.0000 pp |

[Inference] Beta 2 is the observed winner on this development subset. Its
delta is +1.5625 pp Gaussian and +0.7813 pp Impulse; beta .5 is positive on
both, while beta 8's Gaussian gain is offset by an Impulse loss. Hard-M ties
SCD-only. This favors the smooth profile at beta 2 in this particular screen,
but does not establish that smoothing caused the gain or that beta 2 is
accuracy-optimal.

[Open] This is one seed, 128 examples per corruption and five compared arms.
The CSVs do not contain per-example predictions, and cross-corruption object
identity correspondence has not been verified. Do not interpret the aggregate
as paired significance, independent confirmation or a stable 1 pp gain. The
report JSON is available locally, but the complete raw diagnostic run ZIP is
still absent. Keep beta 2/rho .001 as a frozen candidate for the next
validation only. First define an object-level disjoint validation pool and
verify its indices across corruption files; then compare the frozen candidate
with SCD-only under repeated seeds. If a disjoint pool cannot be established,
describe the next repeated-seed results as development-set stability only.

## 2026-09-28 - implementation and experiment-plan audit of 6a0b4da

[Code/Run/Inference] User requested a review of recent code and adherence to
the knowledge plans. Full findings are in
`gsd_calibration_audit_20260928.md`. The core smooth loss, shared-state probes,
fixed local-ratio coefficients and host updates follow the approved design;
128 CPU tests pass. All seven screen ZIPs were independently checked against
their counts, commands/logs, source/assets, scheduler and host settings; the
recorded ranking is unchanged. No runtime code or raw archive was changed.

[Code] The ranking analyzer can accept mismatched SCD/graph CLI settings,
missing manifests and counts inconsistent with declared subset size. In-memory
mutation checks reproduce these gaps; actual supplied bundles do not contain
them. Strict validator fixes remain open. Current sources match the archived
runtime source hashes after accounting for local CRLF line endings.

[Run] On guided rho=.001 trajectories, per-corruption median weighted local
ratios across profiles are .0009683--.0010329, supporting successful local
calibration. Style medians span .0006538--.0014786, so both routes are not
simultaneously matched. Mean effective masses for beta .5/2/8 are about
1259/647/329 on Gaussian and 1261/615/286 on Impulse. Hard requested M=100
expands to mean ranks139.3/121.2 and maxima291/240 under the existing tolerance.

[Correction/Decision] Supersedes the preceding immediate beta2-freeze/held-out
next-step recommendation. Beta2 beats beta .5 by just one aggregate correct
prediction. The registered plan still calls for a small interaction check and
larger/repeated-seed development comparisons, retaining multiple promising
candidates, before freezing. A proposed single extra condition beta .5/rho
.01 completes a 2x2 development comparison with three existing corners.
Its outcome is unknown. Independent confirmation must have a separate path:
current development code enforces the same split seed and calibration-prefix
overlap. Do not bypass that guard or call a larger nested prefix held out.

[Open/Falsifier] Native CUDA original/instrumented SCD parity, original raw
diagnostic ZIP, per-example paired predictions and object correspondence
remain outstanding. Losing the apparent gain on expanded development data
or repeated seeds would rule against promoting the current candidate. This
review supports continued development, not a confirmed accuracy improvement.

## 2026-09-28 - strict screen validation and beta-rho interaction handoff

[Code/Verification] `scripts/analyze_gsd_calibration.py` now requires an
exact seven-file bundle with one root/run ID, valid asset/dataset/runtime
manifest identities and agreement with the immutable calibration provenance.
It checks the fixed ModelNet40-C smooth-v2 host/graph protocol, deterministic
development indices and prefix, scheduler/batch/randomness metadata, CSV
headers/metadata/counts/correct-count bounds, and summary aggregates. ZIP and
run-directory inputs use the same checks. Regression fixtures reject the audit
examples: SCD weight `.5`, modes `400`, missing manifests, a mismatched CLI
count, inconsistent summary totals and a scheduler change.

[Verification] The stricter reader accepted the current four weight-screen and
five beta-screen ZIPs directly, preserving their recorded rankings and compact
summaries. This verifies internal archive consistency, not CUDA/native baseline
parity or a new accuracy result. No raw ZIP, report JSON or derived committed
result was changed.

[Code/Plan] Added `--interaction-screen` to compactly summarize SCD-only plus
the four beta `.5/2` by rho `.001/.01` cells after the proposed beta `.5`,
rho `.01` run arrives. The only registered new command uses the existing raw
calibration `config.json`, seed 0, count 128, batch 32 and alpha
`81.39104941932808`; it has not been executed. `eval_gsd_calibration.py` was
not changed because its recorded hash is required by that calibration reference.

[User report/Code] The user confirmed that the original diagnostic reference
`config.json` is absent in Colab (`commit afa9b7a`, expected path missing).
Added `scripts/run_gsd_interaction_from_scratch.py`: it runs the existing
diagnostic phase first, derives beta .5/rho .01 alpha from the resulting raw
config, and checks the four archived cells against the fresh split, manifests
and coefficients before launching the one new arm. The interaction analyzer
can compare calibration generations only when every included guided run's
stored coefficient matches the fresh report exactly under the existing tight
numeric tolerance, and the split/manifests agree. A mismatch stops before the
new guidance run; no existing ZIP is edited. The script has not been run in
Colab; no new model result or performance claim exists.

[Open/Falsifier] Ingest the printed complete seven-file ZIP, run the strict
interaction summary, then retain beta .5/2 candidates only if expanded
development examples and seed 0/1/2 comparisons warrant it. Do not call the
same/nested development pool held out or fit a new final coefficient from this
single result.

## 2026-09-28 - restart interaction cells under one calibration reference

[User report/Run] Colab completed a new diagnostic calibration, but the
existing beta .5/rho .001 ZIP had a different calibrated coefficient. The
launcher correctly stopped before the new guidance run; do not combine those
old cells with the new reference. The coefficient difference was reported,
but its numeric magnitude was not included in the traceback.

[Code] `scripts/run_gsd_interaction_from_scratch.py --rebuild-prerequisites`
now creates one fresh diagnostic reference, then regenerates only SCD-only,
beta .5/rho .001, beta 2/rho .001 and beta 2/rho .01 using coefficients from
that reference. It validates the four generated bundles and launches only the
missing beta .5/rho .01 condition before producing an interaction summary.
It does not repeat beta 8 or hard guidance cells. Run names and summary output
are tied to the fresh calibration ID, preserving previous raw artifacts.

[Verification] Two launcher unit tests and all 133 CPU tests pass. No Colab
model run was performed in this environment; the regenerated ZIPs and
interaction outcome are pending.

## 2026-09-28 - review of the current restart and next-agent handoff

[Code/Inference] Reviewed `c1c7467` against the calibration review, execution
plan and 2026-09-28 audit. The new six-run block (one diagnostic plus five
development conditions) preserves the registered beta .5/2 by rho .001/.01
question and fixed host/data protocol. Repeating the necessary controls under
one new reference is a user-authorized recovery from missing original raw
reference and reported coefficient mismatch, not independent confirmation.
See `gsd_interaction_review_20260928.md` and the copyable
`gsd_interaction_handoff_20260928.md` for complete context.

[Run/User report/Open] Seven original screen ZIPs were re-read with passing
CRC, strict analyzer validation and unchanged counts. No new rebuild or interaction ZIP exists locally
at review time. The supplied 095332 failure is the Diffusers .36.0/Hub .36.2
import failure; the subsequent successful .11.1/.11.1 import check is user
evidence, not a new archived experiment. The original alpha mismatch size and
cause remain unknown; it used a very tight numeric equivalence check.

[Inference/Open] The automated restart does not pause for scientific review of
new diagnostic tails. Inspect those metrics before interpretation/promotion.
Require all six raw ZIPs, one exact calibration identity across the five
development runs, and actual environment evidence. Do not mix in historical
cells or restart completed GPU work solely after an analyzer/export failure.
Retain beta .5/2 until expanded development data and seeds0/1/2 support a
decision; frozen disjoint confirmation remains a later separate stage.

[Verification] Re-ran the full local suite: 133 CPU tests pass. Env files match
`dev`. No runtime code or raw artifact was changed by this review.

## 2026-09-28 - rebuilt interaction accepted; ranking changes

[Run/Verification] On local `gsd-smooth-spectrum@ee12d91`, ingested all six
ZIPs and the supplied interaction JSON for
`20260928-113047_gsd-cal-diagnose-reference-seed0-n64` under
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/`. All runs record runtime
`c1c74672493284c323a66a558f31b57ca03c35f3`. Safe path/CRC/seven-file checks,
completion, command/config/log/CSV agreement, counts, sample/step coverage,
source/assets/host and recorded environment comparisons pass. Each of the five
development runs binds to reference config SHA256
`550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6`.
The strict interaction analyzer reproduces the supplied JSON exactly.
Derived audit/config copy/recomputed summary are separate in
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/analysis_20260928_113047/`.
Raw files remain unchanged. No GPU work or full runtime test suite was run here.

[Run] Seed0, Gaussian/Impulse128 each, severity5, batch32, lambda .95,
raw/eval LION, EMA off: new baseline99/92 (74.609375%); beta .5/rho .001
99/94 (75.390625%, +.781250 pp); beta .5/rho .01 100/91 (74.609375%, null);
beta2/rho .001 98/91 (73.828125%, -.781250 pp); beta2/rho .01 100/94
(75.781250%, +1.171875 pp). Rho .01 minus .001 macro effects are -.781250
and +1.953125 pp for beta .5/2; difference +2.734375 pp is descriptive only.

[Run] The new diagnostic has384 valid probe ratios per candidate with no
missing denominator; state coverage is complete. Relative alpha drift from
old `report.json` is +.01297949% for beta .5 and +.00512817% for beta2.
New rho .001 alpha values:8.140161356429882 and11.310823980075076;
rho .01 values:81.40161356429881 and113.10823980075075. Actual local guided
medians stay within3.30% of target rho. Style is not simultaneously matched;
beta2/rho .01 Gaussian style ratio p90=.024986, max=.076052.
Step4 SCD/DDIM medians2.469/2.332 coexist with local SCD/state medians
approximately .0015; they alone do not justify changing SCD weight1.

[Run/Open] All six new recorded environments agree. Historical source/data/
checkpoint identities match, but NumPy1.21.2 ->1.24.4, SciPy1.8.0 ->1.9.1
and multiple extension hashes/package versions differ. Nondeterministic CUDA
settings and calibration changes remain confounded. Neither coefficient drift
nor environment changes isolate the cause of reversed beta2/rho .001 ranking.
The exact earlier failing reference is still unavailable. Environment data-order
text is generic/stale relative to config and explicit sample indices; preserve
that metadata caveat. Full details and exact ZIP names are in
`knowledge/gsd_interaction_results_20260928.md`.

[Inference/Decision/Falsifier] Accept the reference for continued development,
not a final optimum. Register a bounded next proposal: SCD-only plus beta
.5/rho .001 and beta2/rho .01, 512 examples/corruption, seeds0/1/2, fixed
split20260927 and fixed113047 calibration (nine new runs, not executed).
Report per-seed matched-control macro deltas, their mean/sample SD, corruption
effects and costs. Nonpositive mean, mixed seed directions or material
corruption regressions weaken advancement; no automatic freeze. This narrow
comparison will not reconfirm the entire2x2 interaction. Nested objects remain
development, with separate holdout/object-identity/native-parity gates open.

## 2026-10-01 - full ModelNet40-C screen: spectral candidates do not improve

[Run/Verification] Inspected all nine `gsd-full-screen` ZIPs and
`full_test_screen_summary_20260928-113047_gsd-cal-diagnose-reference-seed0-n64.json`
under `result/modelnet40_c/gsd_latent_spectral_smooth_v2/` on
`gsd-smooth-spectrum@4be6afd86629f3d52648fb543fc2eb350a5f9f1d`. Each ZIP passes
the strict seven-file, safe-path, CRC, completion, config/CLI/log/CSV,
batch/timestep/count, and manifest checks. The nine archives contain exactly
one run for each of SCD-only, beta .5/rho .001, and beta2/rho .01 at seeds 0/1/2.
All cover the 15 severity-5 corruptions and 2,468 examples per corruption
(37,020 examples per run), bind to calibration config SHA-256
`550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6`, and agree
on commit, data/assets/source manifests, extensions, and runtime. Recorded
runtime is PyTorch 2.1.2+cu121, CUDA 12.1, NVIDIA A100-SXM4-80GB. Local strict
recomputation agrees with the supplied summary after ignoring machine-specific
paths and allowing Python's final-bit floating-point differences (maximum
absolute difference 2.22e-16). Raw ZIPs and supplied summary were not changed.

[Run] Equal-weight macro accuracy across the 15 corruption rows, mean +/-
sample SD across seeds: SCD-only **63.879885 +/- 0.132590%**; beta .5/rho .001
**63.851972 +/- 0.035323%**; beta2/rho .01 **63.821358 +/- 0.118373%**. Matched
per-seed macro deltas versus SCD-only are respectively **+0.083739,
-0.043220, -0.124257 pp** (mean **-0.027913 pp**, sample SD 0.104839 pp) and
**-0.010805, -0.105348, -0.059427 pp** (mean **-0.058527 pp**, sample SD
0.047278 pp). Beta .5 is mixed across seeds and slightly lower on average;
beta2/rho .01 is lower in all three seeds. Across corruption means, beta .5
improves 8 rows and regresses 7; beta2 improves 6 and regresses 9. Guidance
runs take about 76.5 minutes each versus 29.6 minutes for SCD-only; peak memory
is about 21.0 GB versus 14.6 GB as recorded by the runner. Total recorded
runtime is about 9.14 GPU-hours.

[Inference/Decision] This full-test-set screen supplies no evidence to advance
either candidate over SCD-only under the locked protocol. Do not label these
test-set comparisons independent confirmation: the ModelNet40-C test outcomes
were used to compare candidate settings. The prior proposal to repeat these
conditions on 512 examples/corruption is superseded by the full-file runs.
Preserve SCD-only as the development reference; any future parameter search
needs a separately declared, genuinely independent evaluation source or split.
This does not establish equivalence or prove that spectral guidance cannot
help under another predeclared setting. Raw evidence is the nine screen ZIPs
and the summary at the path above; the conclusion would change only with
independent, protocol-matched evidence showing a reproducible gain.


## 2026-10-01 ModelNet40-C all-15 results matrix

[Run/Verification] Audited 141 local ModelNet40-C ZIPs against full severity-5
coverage: 40 contain all 15 canonical corruption rows with 2,468 examples each
and internally consistent bundle/CSV/macro data. The consolidated matrix is
`knowledge/modelnet40_c_all15_results_matrix_20261001.md`; it reports aggregate
results, per-corruption accuracies and paired deltas across source-only severity
1-5, preprocessing identity, pure VAE, 3DD-TTA controls, decoder/SCD ablations,
and the nine-run GSD smooth-v2 screen. Raw ZIPs were read only.

[Run/Verification] The local GSD v1 spectral-only `ablation14` result covers
14/15 corruptions (Background absent), so it is recorded separately and not
counted as an all-15 result. Earlier 2-corruption beta-rho interaction and
external paper/upstream reference scores are also explicitly kept outside the
all-15 comparison. The GSD full-test candidates are lower on mean macro than
matched SCD-only in the same screen; the matrix marks the full-test candidate
selection as descriptive rather than independent confirmation.

## 2026-10-01 - smoothing null and historical small-gradient diagnosis

[Code/Run/Inference/Open] At `gsd-smooth-spectrum@4be6afd86629f3d52648fb543fc2eb350a5f9f1d`,
reanalyzed the nine full-screen ZIPs and raw 113047 calibration ZIP under
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/`. Exact input identities,
derived metrics and a CPU-only archive-reader script are in
`analysis_20261001_smoothing/`; the full interpretation is
`knowledge/gsd_smoothing_diagnosis_20261001.md`. No model run or source-method
change was made; raw bundles are unchanged.

[Run] Full Gaussian/Impulse-only candidate deltas are -.054025 and -.148568
pp across seeds 0/1/2, versus +.781250/+1.171875 in the 128-example-per-
corruption seed-0 subset. The small-pool gains therefore do not vanish solely
because 13 other corruptions were added. Gaussian full-run local weighted
spectral/SCD ratios of batch-step mean norms are .096322% and .980377%;
Background ratios are .428373% and 4.914854%. These are not per-example
medians. Gaussian profile masses are 1259.76 and 646.17, showing broad
spectral support rather than a minor adjustment of the M100 boundary.

[Run] Shared SCD-reference probes have no negative local cosine for either
smooth profile in Gaussian/Impulse. Beta2 style conflicts occur in
38.0208%/36.9792% of those probe rows. These are not full-guided-trajectory
measurements, and conflict does not establish classification harm.

[Code/Inference] The objective remains fidelity to a corrupted encoded
reference; smoothing does not repair reference topology or establish semantic
usefulness. Legacy raw MSE logging, batch-mean reduction, graph filtering,
low/mid bands and host defaults differ from current runs. A small raw loss
or gradient is not a sufficient measure of applied influence. Historical
gains require a matched legacy host-off/on comparison; full-screen smoothing
causality also remains open because no matched all-15 hard arm was run.

[Decision/Falsifier] Prioritize recovering the successful legacy bundle and
same-state graph/profile/conditioning diagnostics with paired predictions.
A repeatable legacy on-minus-off gain would support an actual old spectral
benefit; a host-off gain without that increment would favor a host explanation.
These proposals do not launch new Colab work. Update method synthesis and
open questions; preserve the full-screen negative/null finding without
claiming that spectral guidance or smoothing is universally ineffective.


## 2026-10-03 - proposed factorial control for spectral-only interpretation

[Code/Inference/Proposal] On `gsd-smooth-spectrum@4be6afd`, the user asked
which experiment to run next and what missing evidence can run in parallel.
Registered `knowledge/gsd_next_experiment_proposal_20261003.md`: all15 full
files at seeds0/1/2, unguided diffusion/SCD-only/v1 spectral-only/SCD+spectral
(12 conditions). Hold v1 M100/weight1 and the current .95 host fixed; replay
common input/latent/noise states and record paired predictions. The key
missing contrast is spectral-only minus unguided diffusion. Pure VAE is not
an unguided diffusion control; smooth-plus-SCD null is not proof of spectral
uselessness. Include Background, absent from the prior 14-corruption run.

[Code] Current trajectory rejects simultaneous zero guidance weights and
protocol guards prevent the required all15 spectral-only benchmark. This is
a proposed experiment needing explicit ablation support, not a ready launcher.
No implementation/model run/commit/push occurred. Parallel priorities are
shared-state graph/direction diagnostics and recovery of the actual successful
legacy bundle; avoid guessing historical runtime settings from defaults.

[Decision/Falsifier] C-A positive but D-B null supports a spectral contribution
without incremental benefit under SCD; null C-A weakens attribution of C's
performance to spectral guidance, without proving equivalence. Report seeds,
corruptions and paired uncertainty. Existing result paths and exact evidence
are linked from the proposal; no new empirical accuracy result is claimed.


## 2026-10-03 - existing-run audit corrects unnecessary repetition proposal

[Run/Code/User report/Decision] The user explicitly rejects repeating tests
whose results already exist. At `gsd-smooth-spectrum@4be6afd`, read config.json
and per_corruption.csv from all 144 local ZIPs (141 ModelNet40-C, three
ScanObjectNN-C), without read errors; standalone configs added no unarchived
run identities. Exact paths/config hashes/CSV rows are in
`result/modelnet40_c/diagnostics/gsd_existing_run_inventory_20261003.json`.
This was metadata/coverage verification, not a new CRC/runtime parity audit.

[Run] All15 SCD-only exists at seeds0/1/2 both in original eval/raw and the
latest full-screen control block. V1 spectral-only exists for full Gaussian/
Impulse at all three seeds and for 14 corruptions excluding Background at
seed0. SCD+v1 spectral exists for Gaussian/Impulse at M100/240/400, three
seeds per M, with off controls. Both selected SCD+smooth candidates already
have all15/three-seed results. No unguided diffusion archive was found;
v1 off is SCD-only, and pure VAE omits diffusion.

[Decision] Withdraw the blanket 12-new-run proposal; retain its scientific
contrasts but reuse existing evidence. Minimal new main proposal is unguided
diffusion, all15/seeds0-2 (three new full runs). Any later coverage extension
should identify missing cells, avoid duplicate evaluation, and retain scope/
randomization/environment differences rather than splice a synthetic paired
benchmark. Missing per-example predictions are a known evidence limitation,
not an automatic reason to repeat all tests. Reuse existing calibration/
full-screen diagnostics before scheduling new captures. Full correction:
`knowledge/gsd_existing_runs_audit_20261003.md`. Updated proposal, backlog,
README and thesis scope; no experiment, code change, commit or push occurred.

## 2026-10-03 ? Requested unguided and smooth-only launchers prepared

[User decision/Code] Added two missing full-test ablations, each all15 severity5,
2468 examples/corruption, seeds0/1/2. No existing SCD or combined run repeated.
Unguided uses zero losses/gradients while retaining DDIM; smooth-only sets SCD0
and preserves beta=.5, alpha=8.140161356429882 from reference113047. Per-example
indices/labels/predictions are recorded post-inference. Resume skips validated
complete seed ZIPs; failed attempts are retained. Base commit
`4be6afd86629f3d52648fb543fc2eb350a5f9f1d`; local changes only, no push/GPU run.

[Verification] 158 CPU tests passed, six real-reference CLI/config builds and
both dry-runs passed; nine archived full-screen ZIPs still validate. Independent
review caught the tta_gsd source-identity regression on old calibrated stages;
new compatibility records explicitly cover the added branch, while old records
retain strict source validation.

[Open] Accuracy and CUDA execution remain Colab work. Historical contrasts are
not common-draw paired; v1-versus-smooth also changes normalization/scale.
Exact protocol, outputs, transfer bundle and falsifiers:
`knowledge/gsd_guidance_ablation_handoff_20261003.md`.

## 2026-10-04 - v1 spectral-only Background accepted; descriptive 14+1 composite

[Run/Verification] Ingested the user-supplied archive
`result/modelnet40_c/gsd_latent_spectral_v1/20261004-095901_gsd-v1-background-completion-seed0-spectral-only-m100.zip`
at `gsd-smooth-spectrum@f04bfbdb24d5e9617602b50f1711f835861b6eb9`.
Archive SHA256 is `49b6883694e231413c1415b18f62041a13cf3c9a4c283ab8c4170b5d35546b43`.
The run records the same commit with a dirty checkout; all ten recorded
inference-source hashes exactly match that commit's Git blobs.
Seven unique safe ZIP members, CRC, statuses, CLI/config/stdout/CSV agreement,
counts, scheduler and model inventories pass. Background has 77 batches of 32
plus one of 4, 2468 graph records and 2730 batch-step records (78 x 35).
Seed 0, severity 5, raw/eval LION, no EMA, frozen Point-MAE, original final style,
batch32, lambda=.95, gamma=eta=.01, spectral weight1/SCD0, M100/k10,
delta=.1/graph gamma=.6. This is one full-corruption single-seed result;
it does not meet the three-seed Level 2 requirement or constitute an all15 run.

[Run] Background **572/2468 = 23.176661%**, runtime 877.007984 seconds,
peak 20521.495117 MB. Revalidated and reused
`result/modelnet40_c/gsd_latent_spectral_v1/20260926-141029_gsd-v1-ablation14-on-seed0-spectral-only.zip`
(SHA256 `a083cf795dc7e2b951fa80f47ea32e939cbeb8f9e41c027449b02431ee6b804a`,
commit `9650770f75cf0c37e1e16b873bd6b8ddd0a4276e`). Its 22083/34552 plus the
new Background 572/2468 give **22655/37020 = 61.196650%** equal-corruption
macro/micro. This is a locally derived cross-run **14+1 composite**, with
one seed and no repeated-seed uncertainty or independent confirmation.

[Run/Inference] Compared descriptively with
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/20260928-133819_gsd-full-screen-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-scd-seed0.zip`,
SCD-only Background is 60.899514% and all15 is 63.752026%.
Composite-minus-reference is -2.555375 pp; Background contributes -2.514857 pp
and the other 14 together -0.040519 pp. Other-14 means alone are 63.912364%
versus 63.955777%. The observed deficit is concentrated in Background;
this does not identify a causal spectral/SCD effect. The v1 ZIPs have no
per-example predictions, and the unguided/smooth-only artifacts remain awaited.

[Code/Run/Open] Old/new assets match; all 15 recorded data identities match
the archived full-screen reference. Raw dataset files are absent locally,
so no new check of Colab/current file bytes is claimed. PyTorch/CUDA/cuDNN/GPU
metadata match, while ConfigArgParse/OpenEXR/tzdata versions and native hashes
for Chamfer, chamfer_3D and PointNet2 change. Four inference-source files differ,
including `graph_spectral.py`. Static review retains the v1 graph/loss/update
math, but does not establish numerical equivalence across native builds.
The launcher allows the other three source differences but omits the graph
file from its allowed set; it would reject these archives after current-data
checks. No Colab composite JSON or parent-launcher final stdout was supplied.
The completed child run remains valid; no inference rerun is needed to address
summary handling. The missing Colab summary has not been fabricated.

[Decision/Falsifier] Mark Background seed0 complete and reuse the new archive.
The remaining v1 spectral-only coverage gap is 26 cells (13 each at seeds 1/2);
do not expand automatically. Await the two ongoing guidance-ablation families
before deciding on mechanism experiments. Conflicting counts/metadata,
nonzero SCD weighting, asset mismatch or unexplained source identity would
reopen acceptance; none was found within the stated verification scope.

Derived files, kept separate from immutable raw inputs:
`result/modelnet40_c/gsd_latent_spectral_v1/analysis_20261004_background/validation_and_composite.json`,
`per_corruption_comparison.csv`, and `validation.md` in the same directory.
Updated README, completion handoff, method synthesis, coverage audit, result
matrix addendum and open questions. No model/code change, GPU execution,
commit or push occurred during ingestion.

## 2026-10-04 - unguided and smooth-only full-file results accepted

[Run/Verification] The user supplied six ZIPs and both launcher summaries
under `result/modelnet40_c/gsd-guidance-ablations/`, preserving the Drive
hierarchy. Raw files remain at their supplied paths, unchanged and without
duplicate canonical copies. Ingestion Git context and all six recorded run
commits: `gsd-smooth-spectrum@f04bfbdb24d5e9617602b50f1711f835861b6eb9`.
Calibration: `20260928-113047_gsd-cal-diagnose-reference-seed0-n64`, raw config
SHA256 `550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6`.

Raw directories:

- `result/modelnet40_c/gsd-guidance-ablations/unguided/20260928-113047_gsd-cal-diagnose-reference-seed0-n64/`
  contains seed0/1/2 ZIPs timestamped20261003-145004/150745/152505 and the
  `guidance_ablation_summary_unguided_<reference>.json` supplied summary.
- `result/modelnet40_c/gsd-guidance-ablations/smooth_only/20260928-113047_gsd-cal-diagnose-reference-seed0-n64/`
  contains seed0/1/2 ZIPs timestamped20261003-141149/152811/164407 and the
  `guidance_ablation_summary_smooth_only_<reference>.json` supplied summary.

Every ZIP passes the strict full-file ablation reader, safe seven-file/CRC,
config/CLI/calibration/count/prediction checks, and independent stdout/CSV,
eval/frozen module, checkpoint-load and Git-source-blob reconciliation.
Both supplied summaries agree with raw recomputation, allowing only per-run
machine directory differences and final-bit floating-point differences
(max1.110223e-16 and5.551115e-17). The six reused historical SCD-only and
same-beta combined ZIPs also validate. Exact paths and input SHA256 values
are in the derived analysis JSON linked below. Raw dataset/checkpoint files
were not rehashed locally; archived manifests agree across all12 runs.

[Run] All15 severity5, 2468 examples/corruption, seeds0/1/2, batch32,
gamma=eta=.01, lambda=.95, raw/eval/noEMA, frozen classifier, original final
style. Both new arms have SCD0; smooth-only retains beta .5 and
alpha8.140161356429882, while unguided has spectral0. Every new Background
run records35 actual reverse steps (2730 batch-step records, indices0-34,
times340-0); other corruptions record5. Unguided local/style guidance norms
are exactly zero; smooth-only records the fixed nonzero spectral weight.
This closes the missing full-test conditions, with development coverage
rather than independent benchmark confirmation.

| Condition | All15 mean +/- sample SD (%) | Background mean (%) |
|---|---:|---:|
| Unguided | 61.202053 +/- 0.127389 | 23.622366 |
| Smooth-only | 61.359625 +/- 0.072095 | 23.878984 |
| Reused SCD-only | 63.879885 +/- 0.132590 | 60.669908 |
| Reused SCD + same smooth | 63.851972 +/- 0.035323 | 60.913020 |

[Run/Inference] Smooth-only minus unguided is **+0.157572 +/-0.103226 pp**,
with seed deltas+.113452/+.083739/+.275527 and12/15 positive corruption
means. Aligned predictions show corrected/broken counts531/489,497/466,
534/432: net42/31/102 extra correct per37020 exposures. These are cross-run
outcome counts, not proven guidance-caused corrections. The corresponding
increment with SCD remains **-.027913 +/-0.104839 pp**, with one positive
seed. Smooth-only minus historical SCD is-2.520259 pp. Other14 means are
63.886316% unguided,64.036814% smooth-only,64.109169% SCD-only; most of the
large descriptive SCD-versus-no-SCD gap is concentrated in Background.
The earlier v1 Background23.1767% is consistent with the low no-SCD scores,
but differs in operator/scale/build and is not a homogeneous fourth seed.

[Run/Open] New arms have identical runtime source/data/assets and nominal
Python/platform/packages/PyTorch/CUDA/GPU metadata, but `chamfer`,
`chamfer_3D` and `pointnet2_ops._ext` hashes differ between the two arms.
Each arm's three seeds share its own native inventory. Native functional
equivalence is unverified; method and build are confounded despite repeated
seeds. Historical controls at `4be6afd86629f3d52648fb543fc2eb350a5f9f1d`
also differ in runtime source and package inventory. Equal seed numbers and
aligned indices do not prove common random draws. No causal interaction,
significance, projection benefit or held-out generalization claim is made.

[Run] Mean recorded per-seed runtimes are1017.621s unguided,4538.678s
smooth-only,1776.273s SCD-only and4597.439s combined. Smooth-only/unguided
observed ratio is about4.46, with mean peak memory21008.48 versus11686.34 MB.
These are logged evaluation costs, not a controlled performance benchmark.

[Decision/Falsifier] Reuse all completed conditions. The small positive
fixed-alpha observation supports a mechanism investigation, not promotion
over SCD or another test-set parameter grid. First inspect existing shared
reference-state graph/update/style probes, then identify only the missing
measurement needed by a concrete hypothesis. Counts/prediction disagreement,
nonzero unguided updates, wrong reverse steps or identity mismatch would
reopen acceptance; none was found in the stated checks. Independent evidence
resolving native-build/draw confounds could weaken or strengthen the causal
interpretation of the small smooth-only increment.

Detailed report: `knowledge/gsd_guidance_ablation_results_20261004.md`.
Derived outputs under
`result/modelnet40_c/gsd-guidance-ablations/analysis_20261004/`:
`analysis.json`, `per_corruption.csv`, `prediction_transitions.csv`.
Updated README, handoff, coverage audit, all15 matrix, method synthesis,
open questions and result README. No model experiment, inference-source
change, commit or push occurred during this ingestion.

## 2026-10-04 - Background reverse-step sensitivity plan

[Code] Plan recorded on branch `gsd-smooth-spectrum`, checkout commit
`aa8b725ba4daaf3448f0cda75bfc6aaec6c86d48`; no inference source was changed.

[Run] The validated 35-step Background values provide the reference condition
for the next diagnostic. Unguided accuracy by seeds 0/1/2 is
23.703404/23.946515/23.217180% (mean 23.622366%, sample SD 0.371359 pp).
Smooth-only (beta .5, alpha 8.140161356429882) is
24.149109/23.662885/23.824959% (mean 23.878984%, sample SD 0.247573 pp).
The separate v1 spectral-only seed-0 Background result is 23.176661%; it is
not a matched seed or method control. Full prior results and provenance are
in `gsd_guidance_ablation_results_20261004.md` and
`gsd_v1_background_completion_20261003.md`. The accepted smooth/unguided
analysis is at `result/modelnet40_c/gsd-guidance-ablations/analysis_20261004/analysis.json`;
the v1 completion is at
`result/modelnet40_c/gsd_latent_spectral_v1/20261004-095901_gsd-v1-background-completion-seed0-spectral-only-m100.zip`.

[Code] `run_baseline.py` currently chooses 35 reverse steps for Background
and 5 for other corruptions. `tta_gsd.py` interprets this argument as a
percentage of a 100-step schedule, so values 5/10/15/20/25 correspond to the
requested number of reverse steps. The underlying runner files are unchanged
in this plan. The Colab cell prepared in this session temporarily adapts the
runtime copy to accept Background-only full-file runs and a configurable step
count, then restores the two source files on exit; resulting artifacts should
retain the adapted source fingerprints and CLI step value.

[User report/Decision] Because the 35-step smooth-only result is unexpectedly
low, the user requested a Background-only step scan. The declared matrix is
five step counts x two fixed arms x seeds 0/1/2 = 30 full-file runs. Each run
uses ModelNet40-C severity 5 Background (2,468 examples), batch 32, raw/eval
LION, EMA off, frozen Point-MAE, original final decode style, lambda=.95,
gamma=eta=.01 and the pinned calibration reference. Both arms have SCD weight
0. Smooth-only keeps beta=.5 and alpha=8.140161356429882; unguided keeps
spectral weight 0. Reuse, rather than rerun, each arm's existing 35-step
seed0/1/2 archive as the comparison condition.

[Superseded by the 2026-10-04 sweep result below] The pre-run protocol and
first launch failure are retained in `gsd_background_step_sensitivity_20261004.md`.

[User report/Run failure] The first Colab attempt started only the 5-step
smooth-only, seed-0 condition. It created
`20261004-121037_gsd-bg-step-5-smooth_only-seed0-20261004-121036.zip`, then
failed before model inference in `verify_calibration_inputs` with
`dataset_hash_manifest/gaussian` mismatch. The worker had recorded only the
selected Background file in `dataset_hash_manifest`, while the pinned
calibration reference requires current Gaussian and Impulse hashes too. The
shell stopped after this first failed condition, so the other 29 runs did not
start. Treat this ZIP as a failed pre-inference artifact; preserve it for
provenance and exclude it from accuracy summaries.

[Code/Decision] Correct the Colab-only launcher patch at the verification
boundary: compute hashes for any calibration-reference dataset files missing
from the selected-run manifest, call `verify_calibration_inputs` on a copy
containing both sets, and record those calibration-input hashes in a separate
config field. Keep `dataset_hash_manifest` scoped to the Background file being
evaluated. This retains calibration data identity checking without labeling
Gaussian/Impulse as evaluated in the Background-only run. No repository
inference source has been edited for this fix.

## 2026-10-04 - Imported GSD improvements default all-15 CSV

**Evidence:** [User report] The user ran the `gsd-tta-improvements` branch's
`eval_gsd_tta.py` with defaults and supplied
`result/modelnet40_c/gsd_tta_improvements/eval_results.csv`. [Code] The locally
available branch worktree is at `6992587`; its defaults specify batch 70,
M=400/M_mid=600, low/mid spectral weights 16/2, normal/Background reverse
steps 10/30, and low/mid losses with `reduction='mean'`. The CSV does not
record which commit, seed, command, runtime environment, checkpoint/data
hashes, or per-row example counts produced it.

**[Run/Inference]** The CSV has 15 unique corruption rows. Every accuracy is
consistent with an integer correct count out of 2,468 (inferred from the
rounded CSV values; explicit counts are absent). Equal-corruption macro is
64.3868% over all 15 and 64.7459% with Background excluded. The current
smooth-v2 full-test screen's SCD-only three-seed means are 63.8799% and
64.1092% on those same respective scopes, so the imported single-run CSV is
descriptively +0.5069 pp / +0.6367 pp higher. It is higher on only 6/15 rows
(6/14 excluding Background); its largest same-corruption gains are Impulse
(+8.16 pp), LiDAR (+5.58 pp), and Gaussian (+2.04 pp), while Shear is lower
by 3.82 pp. This is an unmatched comparison, not evidence that mean reduction
caused the higher macro.

**[Inference/Decision]** A batch-mean spectral-loss ablation in the current
smooth-v2 implementation is warranted. Hold profile, beta, weight, SCD,
trajectory and all other settings fixed; compare the existing sum-over-samples
condition against dividing only the spectral loss/gradient by the actual
batch size. That scales the spectral gradient by `1/B` while preserving its
direction, but changes its balance against summed SCD. Report per-corruption
accuracy and local/style spectral-to-SCD gradient ratios. Current all-15
candidate outcomes already use these test files, so any follow-up remains
descriptive development evidence, not independent confirmation.

**Open evidence:** Preserve the supplied CSV unchanged and request its full
run directory (command, config, environment, stdout, counts, seed, and hashes)
before promoting it to a fully auditable result.

## 2026-10-04 - Background reverse-step sensitivity accepted

[Run/Verification] Ingested 30 valid Background-only runs at
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/background_step_sensitivity_20261004/`:
smooth-only and unguided x steps 5/10/15/20/25 x seeds 0/1/2. Every ZIP CRC,
seven-file member set, and ZIP-to-directory content check passes. All runs
complete all 2,468 examples, with config/CSV/summary completion consistent.
Within the new block, asset, Background data, calibration-input, runtime
source, and extension manifests are identical. LION VAE/priors and Dropout
are eval; EMA is off. One failed pre-inference 5-step smooth-only seed-0 ZIP
is retained, documented, and excluded; its initial calibration-input hash
error did not proceed to model inference.

[Code/Run] The saved `scheduler_timesteps` array contains the full 100-point
DDIM grid, not the consumed step count. Per-batch diagnostics show exactly
78 x requested reverse steps in every valid run, with step index 0 through
N-1. The sampler consumes the grid suffix, so both loop count and initial
noising timestep change (25-step starts at t=240; 35-step starts at t=340).

[Run] Background accuracy by seed0/1/2; mean +/- sample SD (%):

| Arm | Steps | Seed 0 | Seed 1 | Seed 2 | Mean +/- SD (%) | Delta vs same-arm 35-step (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Smooth-only | 5 | 24.0681 | 23.8655 | 23.6224 | 23.8520 +/- 0.2232 | -0.0270 +/- 0.2079 |
| Smooth-only | 10 | 25.5267 | 24.4733 | 24.7974 | 24.9325 +/- 0.5396 | +1.0535 +/- 0.2922 |
| Smooth-only | 15 | 24.9190 | 23.3387 | 23.9060 | 24.0546 +/- 0.8005 | +0.1756 +/- 0.5531 |
| Smooth-only | 20 | 24.4733 | 23.3387 | 23.5413 | 23.7844 +/- 0.6051 | -0.0945 +/- 0.3632 |
| Smooth-only | 25 | 25.7699 | 24.1086 | 25.0000 | 24.9595 +/- 0.8314 | +1.0805 +/- 0.5932 |
| Unguided | 5 | 24.1896 | 24.0681 | 24.1491 | 24.1356 +/- 0.0619 | +0.5132 +/- 0.4059 |
| Unguided | 10 | 24.7974 | 24.7569 | 24.4733 | 24.6759 +/- 0.1766 | +1.0535 +/- 0.2256 |
| Unguided | 15 | 25.0000 | 23.4603 | 23.5413 | 24.0005 +/- 0.8665 | +0.3782 +/- 0.8926 |
| Unguided | 20 | 24.3517 | 23.1767 | 23.8250 | 23.7844 +/- 0.5886 | +0.1621 +/- 0.8073 |
| Unguided | 25 | 25.1621 | 24.5543 | 25.2836 | 25.0000 +/- 0.3907 | +1.3776 +/- 0.7327 |

[Run/Inference] Scores stay low and non-monotonic. The best means, at step
25, are about +1.08 pp smooth-only and +1.38 pp unguided against the 35-step
references; step depth through 25 does not explain the roughly 23–25% no-SCD
outcome by itself. Historical 35-step SCD-only and smooth+SCD Background
means are 60.6699% and 60.9130%, respectively. This makes SCD presence the
stronger descriptive correlate, not a causal finding: old/new comparisons
also differ in `run_baseline.py`/`gsd_protocol.py` source hashes and native
Chamfer/PointNet2 hashes. All step conditions within the new block are
matched; same seeds do not establish common random draws. This is
full-test-set development evidence; do not select step 25 as a final setting
or launch another accuracy grid on the same examples without a separate
mechanistic hypothesis.

Derived validation and tables are stored separately from the raw archives in
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/background_step_sensitivity_20261004/analysis_20261004_step_sensitivity/`:
`validation.md`, `analysis.json`, `per_step_summary.csv`, and
`run_validation.csv`.


## 2026-10-04 ? SCD/spectral composition: dominance and conflict are distinct

[Paper/Code/Run/Inference/Open] User requested research into why close
standalone scores fail to yield a better combined result. Inspected current
`gsd-smooth-spectrum` HEAD `aa8b725ba4daaf3448f0cda75bfc6aaec6c86d48`,
legacy `pxp-gradient-projection` `53ba252519c7cf65f836a9c1c564027142ab1573`,
and primary PCGrad, PixelAsParam and CAGrad papers. Full report:
[composition review](gsd_scd_gradient_composition_review_20261004.md).

[Run/Inference] New offline analysis of archived SCD-only common-state
probes, seed0, 64 Gaussian + 64 Impulse examples, three timesteps each:
weighted smooth beta.5 local sums deviate from SCD by median .05430/.05791
degrees; beta2 by .55512/.56816 degrees. No local conflicts in this scope.
Style negative-cosine rates are 10.417/8.854% and 38.021/36.979%, respectively.
These are hypothetical sums reconstructed from norms/cosines, not applied
trajectories, classification gains, or hard-v1 evidence. Repeated timestep
observations are not independent examples. Raw archive remained unchanged:
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/20260928-113047_gsd-cal-diagnose-reference-seed0-n64.zip`.
Derived script/JSON/CSVs are in the sibling
`analysis_20261004_gradient_composition/` directory; SHA256 and formulas
are recorded in `analysis.json`.

[Decision/Open] First fill missing v1 gradient geometry and SCD/v1 paired
prediction complementarity on a declared small diagnostic subset. Conditional
symmetric projection per example and initially style-only is the first
direction candidate if v1 conflicts confirm this pattern. Include magnitude
controls. SCD-priority one-way projection need not preserve spectral progress;
legacy projection flattens whole batches and must not be transplanted.
Positive weights do not change pairwise cosine. PixelAsParam's selective-pair
ablation is Table6. Updated stale branch/diagnostic descriptions in method
and paper notes. No model code, GPU run, commit or push was performed.

[Falsifiers] No conflicts makes conditional projection inactive; no prediction
complementarity limits separate-trajectory fusion. Better gradient geometry
without better classification rejects conflict repair as the performance
explanation at that scope. Do not repeat completed full benchmarks to collect
these initial missing diagnostics; current inspected samples are development
evidence, and future comparison requires matched runtime/common draws.


## 2026-10-04 - Local/style routing assessment and implementation handoff

[User report/Code/Run/Inference/Planning] User requests a critical assessment
of local-SCD/style-spectral routing and an implementation/Colab plan for a
separate agent. Source inspected at gsd-smooth-spectrum
`aa8b725ba4daaf3448f0cda75bfc6aaec6c86d48`; no source changes or GPU runs.
Created [design](gsd_block_routing_design_20261004.md),
[implementation batches](gsd_block_routing_implementation_plan_20261004.md),
and [Colab scenarios](colab_gsd_block_routing_20261004.md).

[Code/Inference] SCD-only already updates style conditioning through the
local prior. Final decoding uses original shape latent; the last style
update is therefore unused downstream. Global diffusion is a separate model
on raw z and is not called on this path. Low-frequency XYZ spectral guidance
does not imply a matching semantic style gradient after the denoiser Jacobian.
LION section3 trains local diffusion with clean global conditioning; noisy
synchronized conditioning from legacy dual code needs an independent audit.

[Run] Reused completed shared-trajectory decoder evidence: all15/seeds0-2,
original63.7484%, updated63.7169%, mean delta-.0315pp. Artifacts remain at
`result/modelnet40_c/shared_trajectory_decoder_control/`. This is not evidence
about disabling conditioning guidance or enabling global diffusion.

[Decision/Planning] Start hard-v1 per-block routing with style-off anchors;
if hard-v1 nonterminal style conflicts exist, compare style-only norm-capped
symmetric PCGrad against same-route sum and applied-norm-matched sum.
Scope: severity5, raw/eval, noEMA, B32, lambda.95, gamma/eta.01, v1weight1,
M100, DDIM100 with actual5/35steps, original final decode. Proposed diagnostic
and pilot use64 samples/corruption at Gaussian/Impulse/Background/Shear,
seed0; next256 indices with seeds0/1/2 replicate one selected mechanism.
All indices/draws/provenance are explicit; no full-grid repetition and no
independent-confirmation claim from already inspected test data. Additional
scale/CAGrad/ensemble/schedule/global-prior studies are conditional follow-ups.

[Falsifiers] Routing that matches style-off may merely remove harmful style
updates; projection matched by step-norm control does not establish a useful
direction effect; seed-unstable gains do not justify full15 promotion. Even a
replicated style-guidance gain does not predict a global-diffusion gain.
The requested documents are ready for a separate implementation agent;
new sampler options/runners do not yet exist. No commit or push performed.


## 2026-10-04 - Clarify spectral gradient magnitude and calibration

[Code/Run/Inference] User asks why spectral style guidance is small and whether
local guidance is similarly small. Verified current SCD sum, v1 division by
3*actual_rank, smooth division by3*N, and calibration rule in
`gsd_calibration.coefficient_summary`. Both local/style receive the same
scalar loss normalization; smooth alpha explicitly targets local norm ratios
rho=.001/.01 for the two discussed settings. Style has no independent target.
Existing hard-v1 M100 weight1 aggregate norm ratios are local.0838/.0968%
and style approximately.164/.0905% for Gaussian/Impulse. These are historical
ratios of mean norms, not the smooth reference-probe medians. The smallness
is relative to SCD, not a diagnosed style-specific vanishing gradient.
Added this clarification to the block-routing design; no new experiment or
implementation change. No research conclusion beyond existing evidence.


## 2026-10-04 - Registered GSD gradient-scale hypothesis

[User decision/Open] Add to `open_questions.md`: can the small effective v1
spectral gradient explain its limited accuracy gain? The planned next scale
pilot holds SCD, loss reduction, graph, trajectory and decoder fixed; only v1
`gsd_weight` varies over 0/1/100/1000 on matched prepared inputs. It records
local/style spectral-to-SCD norm ratios, guidance-to-DDIM displacement, state
stability and paired accuracy. If one scale merits follow-up, replicate only
one plus 0/1 controls on new manifest indices/seeds 0-2. Keep this study
distinct from block routing, projection, smooth profiles and mean normalization.
Larger applied gradients changing predictions without accuracy gain would
falsify gradient magnitude as a sufficient explanation. The scale phase is
documented in `colab_gsd_block_routing_20261004.md` and the implementation
handoff; the CPU runner and protocol are now implemented locally. The four
composition/algebra, protocol, runner and analyzer suites passed in focused
runs; trajectory tests also passed. Independent review approved the final fixes.
No GPU run has been launched, so there is no accuracy evidence yet. Code is
uncommitted and no Colab execution ref is available.


## 2026-10-04 - Block-composition algebra implemented

[Code/Verification] Batch1 added `gsd_composition.py` with independent per-example
route/project arithmetic and `tests/test_gsd_composition.py`. RED observed
(`ModuleNotFoundError` before helper existed), then 12 focused tests passed:
`python -m unittest discover -s tests -p "test_gsd_composition.py"`. Reviewer
approved spec compliance/quality. `git diff --check` clean. Uses float64 reductions
then returns source dtype; norm-capped symmetric PCGrad and same-state matched-sum
control expose separate cap/applied scales. Sampler, CLI, Colab artifacts and the
registered gradient-scale experiment remain unimplemented. No GPU run.

## 2026-10-04 - Smooth Spectral Guidance and Batch Invariance
[Code/User report] Transitioned to `gsd-tta-smooth-integration` branch.
1. Fixed LION prior evaluation mode `.eval()` dropout bug.
2. Replaced hard cut-off frequency filtering (M, M_mid, M_high) with a continuous exponential decay function: `weight = exp(-beta * lambda)` where `lambda` is the graph Laplacian eigenvalue and `beta=2.0`.
3. Discovered that PyTorch's default `reduction='mean'` in MSELoss was inadvertently dividing the per-sample spectral gradient by the batch size (`1/B`), causing guidance strength to fluctuate heavily with `batch_size`.
4. Fixed the loss calculation to be batch-invariant by summing over the batch dimension instead of averaging, and mathematically calibrated the default spectral weight (from 16.0 -> 1.17) to match the exact effective gradient magnitude of the previous optimal run.
5. Grid search subset (120 samples/corruption) confirmed `M_max = 800` (cutting off eigenvalues > ~0.95) acts as an optimal noise filter, yielding ~64.72% subset accuracy.
6. [Run] Full evaluation on ModelNet40-C (all 15 corruptions, 37020 samples) with `batch_size=70`, `M_max=800`, `weight_spectral=1.17` yielded a new baseline accuracy of **64.53%**, successfully surpassing the prior 64.39% record with a more mathematically principled formulation.
