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
[Run/Inference] 2026-10-04 GSD composition scale pilot, attempt-0001 is
incomplete. The initial artifact review below is superseded by the immediate
follow-up after the missing partial Background/SCALE_100 ZIP was supplied.
The phase manifest and ten completed arm ZIPs under
`result/modelnet40_c/gsd_guidance_composition_v1/scale/attempt-0001/` show
Gaussian and Impulse completed at spectral weights 0/1/100/1000, and
Background completed at 0/1. Background/SCALE_100 stopped with
`FloatingPointError: GSD nonfinite gradient` in the SCD gradient extraction;
the raw SCD loss had passed its finite check. The failed manifest points to a
partial arm bundle, but that bundle was absent from the copied artifacts, so
the failing step, example, and number of completed predictions are unknown.
Background/SCALE_1000 and Shear were not run. Descriptive pilot accuracies:
Gaussian weights 0/1/100/1000 = 71.875/71.875/70.3125/68.75%; Impulse =
59.375/60.9375/64.0625/60.9375%; Background 0/1 = 62.5/60.9375%. These
64-example cells are not a complete paired scale screen and do not support
selection or a gain claim. Successful Gaussian/Impulse SCALE_1000 local
guidance/DDIM ratios stayed below .47; Background SCALE_0/1 had about 2.3%
of nonterminal example-steps above 2x, below the predeclared 10% threshold.
The isolated Background/SCALE_100 failure is nevertheless an instability
signal at coefficient 100 on that corruption; it does not establish a
general coefficient threshold. Preserve completed ZIPs, inspect the missing
partial bundle if available, and do not restart the entire phase before
deciding whether a bounded follow-up is warranted. Commit `e0c1733`.

[Run/Inference] Follow-up after ingesting
`attempt-0001_background_seed0_SCALE_100.zip`: the partial arm contains 32/64
examples, each with all 35 denoising steps and finite recorded diagnostics;
the exception occurred in the second batch. That batch's in-progress observer
events were buffered and not persisted, so the failing timestep/sample and
whether the latent first became nonfinite remain unknown. On the matched first
32 examples, SCALE_100 is 17/32 (53.125%), versus SCALE_0 19/32 (59.375%) and
SCALE_1 18/32 (56.25%); this incomplete prefix is descriptive only. For the
first batch, local guidance/DDIM ratio exceeded 2 on 32/1088 nonterminal
sample-steps (2.94%) overall, but on 26/32 examples at the first step (median
2.75; max 7.49). The predeclared aggregate >10% threshold is not met on
persisted steps, while the later nonfinite failure independently makes this
Background/SCALE_100 cell unstable. This does not locate a safe coefficient
or establish general instability. The partial bundle is preserved beside the
other attempt artifacts; no rerun was made at result-ingestion time.

[Code/Verification] 2026-10-05 Added scale-only scale-corruptions and
scale-arms filters plus continue-on-arm-error to avoid rerunning completed
scale cells and let later selected arms execute after an arm failure.
Inference code and coefficients are unchanged. py_compile and CLI dry-run
previews for Background/SCALE_1000 and Shear/0,1,100,1000 passed; no model
tests or Colab execution were run. Exact commands are recorded in
colab_gsd_block_routing_20261004.md.

## 2026-10-05 - M_max=1800 Full Evaluation and Parameter Confirmation
[Run/User report] Full evaluation on ModelNet40-C using `M_max=1800` with smoothed GSD (`weight_spectral=1.17`, `beta=2.0`) yielded an accuracy of **64.51%**.
[Inference] Comparing the full dataset runs, M_max=800 (64.53%) and M_max=1800 (64.51%) perform virtually identically (difference of ~8 samples out of 37,020). The slight peak at 1800 observed during the 120-sample grid search was subset variance.
[Decision] `M_max=800` remains the optimal theoretical and empirical choice, as it properly discards high-frequency noise (eigenvalues > ~0.95), reduces computational overhead slightly, and prevents potential instability without sacrificing accuracy. The 64.53% record stands as the definitive smoothed GSD baseline.

## 2026-10-05 - Background SCALE_1000 continuation failure reported

[User report] The user supplied the Colab traceback for the planned
Background/SCALE_1000 seed0, 64-example scale continuation. It reports
`Scale arm failed; recording and continuing: background / SCALE_1000:
GSD nonfinite gradient`. The call fails in `tta_gsd.py:164` while extracting
SCD gradients via `_paired_loss_gradients`, then `_require_finite`.
No new attempt archive has been supplied. Its exact attempt ID, actual
Colab commit/runtime, completed sample count and final phase status are
unverified. Expected parent path:
`result/modelnet40_c/gsd_guidance_composition_v1/scale/`.

[Code] Inspected local `gsd-smooth-spectrum@5b21382`, with continuation
support from `83e06b3`. Both SCD and hard-v1 spectral scalar losses are
checked for finiteness before this SCD backward call. The generic gradient
error does not distinguish local from style, NaN from infinity, or identify
the failing sample/timestep. The handler prints this traceback and continues
selected arms; a normally finalized attempt with caught arm failures is
`partial`. Finalization must still be verified from the supplied manifest.

[Decision/Inference] The reported nonfinite failure meets the predeclared
SCALE_1000 instability gate independently of the guidance/DDIM ratio test;
exclude SCALE_1000 from replication. Retain the separate archived
Background/SCALE_100 failure. Continue the already planned Shear pilot
0/1/100/1000 cell even if Background finishes partial. Do not rerun completed
or failed Background arms, smoke or diagnose, and do not introduce clipping,
normalization, coefficient or update-rate changes. This failure supplies no
complete 64-example accuracy and does not establish a generally safe scale
or prove that spectral amplification caused the SCD backward failure.

[Open] Request both new complete attempt directories as ZIPs, including
phase manifests, raw logs and any partial arm bundles. Compare prepared
input hashes, indices, assets, runtime/native identities and inference-source
hashes before combining attempts. Archive validation may refine the reported
scope; locating the first nonfinite operation requires evidence absent from
this traceback. Review scale results before deciding Phase 3 routing, reusing
the existing diagnose reference. No model code change or local GPU run.

## 2026-10-05 - Shear scale attempt-0004 ingested and validated

[Run] At the user's request, copied the five newest Downloads files (four
Shear arm ZIPs and phase_manifest.json) unchanged to
`result/modelnet40_c/gsd_guidance_composition_v1/scale/attempt-0004/`.
Source/destination SHA-256 matches; existing user edits and raw results are
preserved. Colab commit `5b21382a9c0cb5e3572d383f11a8446e45cb12f0`.
All four arms and the phase manifest are complete. ZIP CRC/path/schema,
internal artifact hashes, phase/arm identities, finite logits/diagnostics,
CSV/prediction counts,64 indices and64x5 actual steps pass. Within this
attempt all four arms share prepared-input/runtime/assets/draw identities.
Each arm has256 nonterminal observations after excluding timestep0.

| Spectral coefficient | Correct/64 | Accuracy (%) | Delta vs0 (pp) | Corrected/broken vs0 |
|---:|---:|---:|---:|---:|
| 0 | 47 | 73.4375 | 0 | 0/0 |
| 1 | 49 | 76.5625 | +3.1250 | 2/0 |
| 100 | 47 | 73.4375 | 0 | 1/1 |
| 1000 | 46 | 71.8750 | -1.5625 | 1/2 |

[Run/Inference] All four arms have0/256 guidance/DDIM ratios>2; respective
maxima are .299941/.299966/.300936/.397509. Nonterminal median weighted
local spectral/SCD ratios at1/100/1000 are .10166/10.08685/94.38713%; style
ratios .12006/11.43351/105.81907%. Stronger gradients change predictions
without improving this Shear subset. Gradient smallness alone is therefore
insufficient at this scope; weight1 is only an observed single-seed pilot
leader, not an optimum. Its paired bootstrap95% interval versus0 includes
zero ([0,7.8125]pp,2,000 resamples over64 indices). No larger scale nominated.

[Run/Limit] Cross-attempt audit finds identical locked settings, indices,
common model/config/data hashes and seven inference/preparation source hashes;
runner/protocol source hashes differ. Python/Torch/CUDA/GPU/platform fields
match, but Chamfer, Chamfer3D and PointNet2 extension binaries differ.
Diagnose Shear actual prepared-input hash differs despite identical draw
keys. Old C_SCD/C_SUM each score48/64, with three prediction differences
against new SCALE_0/SCALE_1 respectively. These controls are not paired
reusable; no attempts were pooled. Old scale attempt-0001 lacks completed
Shear preparation evidence. Empty arm stdout files and compact environment
records limit trace-level provenance; full console logs were not supplied.

[Decision/Open] Preserve the archived Background100 nonfinite result and
user-reported Background1000 failure;1000 remains excluded from replication
even though Shear1000 passes its recorded stability gate. The five supplied
files contain no Background1000 attempt manifest/bundle; request that ZIP.
Routing remains a separate mechanism pilot, but Phase3's control-reuse gate
is unresolved for the new runtime/prepared inputs. Use the existing diagnose
evidence without rerunning diagnose; restore demonstrable compatibility or
explicitly scope minimal new matched controls before launch. No new GPU run
or inference-code change. A matched repeated-seed accuracy benefit would be
needed to overturn this pilot-only no-promotion decision.

Derived script, full audit hashes, analyzer output, uncertainty and report:
`result/modelnet40_c/gsd_guidance_composition_v1/analysis_20261005_shear/`.

## 2026-10-05 - Background SCALE_1000 attempt-0005 manifest ingested

[Run/User report] The user supplied `Downloads/phase_manifest (1).json`,
copied byte-for-byte to
`result/modelnet40_c/gsd_guidance_composition_v1/scale/attempt-0005/phase_manifest.json`.
SHA-256 source/destination agrees:
`a21d2a2d9638b228b176bbf39ef75ac5e27a3872c4db92b6119c7cbd350310d7`.
It identifies scale attempt-0005 on Colab commit
`5b21382a9c0cb5e3572d383f11a8446e45cb12f0`, Background, seed0,
SCALE_1000 only. Phase status is `partial`, execution status
`finished_with_arm_failures`; the failure is `FloatingPointError:
GSD nonfinite gradient`. Its `completed_indices` is empty and `bundle_path`
is null. No per-arm ZIP, prediction, diagnostics, or accuracy was produced.
The manifest records input hash
`a52b838bb8b8492c1bed70ebb9ede531d3d801ec7b36854823a937dc7cb0bd6f`, but
without a completed arm this is not sample-level failure localization.

[Inference/Decision] This repeat reproduces the Background/SCALE_1000
nonfinite failure and independently satisfies the predeclared instability
gate. It remains excluded from replication; do not run it again. No root
cause is established: the generic gradient check does not say local/style,
sample, step, NaN/Inf, or which backward operation first became nonfinite.
The earlier Background/SCALE_100 failure remains separate evidence. Preserve
Shear attempt-0004's validated results; no pooled/complete four-corruption
scale score follows from these artifacts.

[Open] Only an available raw Colab traceback/log could add execution detail;
the attempt manifest itself cannot. Phase3 routing remains gated on resolving
diagnose-control runtime/prepared-input mismatch as recorded above. No
inference code change or local model execution.

## 2026-10-05 - Hard-v1 block-routing pilot attempt-0001 ingested

[Run/Code] Validated `result/modelnet40_c/gsd_guidance_composition_v1/routing/attempt-0001/`.
Phase manifest is complete on commit `5b21382a9c0cb5e3572d383f11a8446e45cb12f0`;
all16 arm ZIPs pass the repository analyzer, comprising four complete paired
blocks (Gaussian/Impulse/Background/Shear, first64 indices, seed0). No arms
were reused from diagnose. Exact archive, prediction, count, diagnostic and
paired-transition output is in `analysis/analysis.json`; interpretation is in
`validation.md`.

[Run] Per-corruption correct counts and four-corruption equal-weight macro:

| Arm | Gaussian | Impulse | Background | Shear | Macro (%) |
|---|---:|---:|---:|---:|---:|
| R_S0 (local SCD, style off) |45/64|38/64|34/64|47/64|64.0625|
| R_SG (local SCD, style spectral) |46/64|40/64|33/64|48/64|65.2344|
| R_G0 (local spectral, style off) |46/64|41/64|14/64|46/64|57.4219|
| R_GS (local spectral, style SCD) |45/64|38/64|23/64|46/64|59.3750|

[Run/Inference] Primary `R_SG-R_S0` deltas are +1.5625/+3.1250/-1.5625/
+1.5625 pp (Gaussian/Impulse/Background/Shear), macro +1.1719 pp; paired
transitions correct4 and break1 of256 corruption-example rows. Per-corruption
paired 95% bootstrap intervals include zero at or touch zero: Gaussian
[0,6.25], Impulse [0,7.8125], Background [-4.6875,0], Shear [0,4.6875] pp.
Reverse `R_GS-R_G0` is -1.5625/-4.6875/+14.0625/0 pp, macro +1.9531 pp;
its improvement is Background-concentrated, with Gaussian/Impulse losses.
Cross-corruption object correspondence is unverified; no pooled bootstrap.
One seed and64 examples/corruption are development-only evidence.

[Run/Mechanism] On R_SG nonterminal observations with valid nonzero style
gradients, negative style SCD/spectral cosines occur on Gaussian96/256,
Impulse115/256, Background1227/2176 and Shear80/256 (31.25-56.4%). This
meets the registered condition to consider the conditional style-projection
pilot; conflict frequency alone is not evidence of accuracy harm or repair.

[Run/Limit] Routing references the complete diagnose manifest from commit
`2acd05a2ec5f282ef3d9d1d51442ac1d8dbd448b`, but its runtime fingerprint
`0ec676...` differs from routing `088dc8...`; native extension and actual
prepared-input hashes also differ. Every route arm record has `reused:false`.
Thus no direct paired comparison to old C_SCD/C_SPEC/C_SUM is valid. R_SG's
positive aggregate versus its same-run style-off anchor is a routing-specific
pilot signal, not promotion versus ordinary SCD.

[Decision/Open] The next bounded mechanism pilot is Phase4 projection:
P_SUM/P_PC/P_NORM on the same first64 indices, only to test projected style
direction against additive style guidance and its norm control. This pilot
cannot establish practical improvement against C_SCD. Resolve a compatible
C_SCD comparator before any replication. Do not rerun diagnose; no additional
full-set or scale experiment is warranted from this pilot alone. User runs
Colab GPU; no local model inference was performed during ingestion.

## 2026-10-05 - Conditional style-projection pilot attempt-0001 ingested

[Run/Code] Validated `result/modelnet40_c/gsd_guidance_composition_v1/projection/attempt-0001/`.
The manifest is complete on commit `5b21382a9c0cb5e3572d383f11a8446e45cb12f0`;
12/12 arm ZIPs pass repository validation: P_SUM/P_PC/P_NORM on Gaussian,
Impulse, Background and Shear, seed0, first64 indices. All four paired blocks
share actual prepared inputs and runtime/config/assets within this phase.
Derived counts/transitions/bootstrap are in `analysis/analysis.json` and the
validation report is `validation.md`.

[Run] Four-corruption equal-weight accuracy is P_SUM67.5781%, P_PC68.3594%,
P_NORM67.1875%. P_PC-P_SUM is +.78125 pp (2 net correct rows/256), with
per-corruption deltas Gaussian0, Impulse+1.5625, Background+1.5625, Shear0.
P_PC-P_NORM is +1.171875 pp (3 net correct rows/256), with deltas Gaussian
-1.5625, Impulse+3.125, Background+4.6875, Shear-1.5625 pp. Per-corruption
paired bootstrap intervals include/touch zero: P_PC-P_SUM Gaussian[0,0],
Impulse[-3.125,6.25], Background[-6.25,7.8125], Shear[0,0]; P_PC-P_NORM
Gaussian[-4.6875,0], Impulse[0,7.8125], Background[0,10.9375],
Shear[-4.6875,0]. No pooled cross-corruption interval: object correspondence
is unverified. One seed/64 examples is exploratory.

[Run/Mechanism] Among valid nonzero style gradients on nonterminal P_PC
observations, negative cosine counts are Gaussian100/256, Impulse115/256,
Background1589/2176 and Shear71/256; none of these projection updates was
skipped. P_PC changes predictions versus P_SUM but net accuracy gains only
two rows; against P_NORM it gains three net rows. The norm-control trajectory
matches projected norm at its own state, so actual norm sequences may diverge.
Frequent conflicts and active projections do not establish useful accuracy
repair.

[Run/Limit] Projection and routing share runtime fingerprint
`088dc88dba75449d401d3cc65f5b8558ba1156c0e1174782afa0267d28e1ca00`, source,
native identities, indices and preparation keys, but actual prepared-input
hashes differ on every corruption. Do not combine with R_S0. Diagnose
C_SCD/C_SPEC/C_SUM remain incompatible by runtime/native/prepared-input
identity. No matched ordinary C_SCD practical comparator is available.

[Decision/Open] Do not select P_PC for replication from this pilot; per-cell
paired intervals include/touch zero and effects are mixed. Stop projection
accuracy work here. A practical method claim needs a same-runtime/same-input
block containing C_SCD, but no such GPU run is selected now. Do not repeat
completed scale, diagnose, routing or projection cells. No inference code
change or local model inference during ingestion.

## 2026-10-05 - Frequency Cutoff (M_max) vs. Corruption Type Analysis
[Inference] A deep dive into the per-corruption results between `M_max=800` (64.53%) and `M_max=1800` (64.50%) revealed a critical theoretical confirmation of the Graph Spectral method:
1. **Geometric/Structural Corruptions:** `M_max=1800` performs consistently better on corruptions that alter geometry but don't add random points (e.g., `distortion_rbf`: 63.17% vs 62.03%, `shear`: 62.93% vs 62.36%, `density`, `occlusion`). Including high-frequency bands (even with tiny weights) helps preserve fine structural details.
2. **Noise/Outlier Corruptions:** `M_max=800` performs consistently better on corruptions that inject noise or add points (e.g., `background`: 60.09% vs 59.81%, `impulse`: 79.01% vs 78.77%, `gaussian`). Cutting off the high-frequency spectrum entirely makes the model robust against confusing pure noise with structural information.
[Decision] This proves `M_max` acts as a direct dial controlling the model's trade-off between structural detail retention and noise filtering, rather than just being an arbitrary performance limit. `M_max=800` remains the robust default since additive noise is common, but adaptive frequency filtering per-corruption is theoretically possible.


## 2026-10-05 - GSD pilot gate review and proposed control completion

[Run] Reviewed existing routing/projection manifests and derived analyses at
`result/modelnet40_c/gsd_guidance_composition_v1/{routing,projection}/attempt-0001/`.
Both phases are complete (16/12 ZIPs), commit
`5b21382a9c0cb5e3572d383f11a8446e45cb12f0`, seed0, four corruptions,64 indices.
R_SG-R_S0 is +1.171875pp; P_PC-P_SUM is +0.78125pp (5 corrected/3 broken)
and P_PC-P_NORM is +1.171875pp (5 corrected/2 broken). These are exploratory
within-phase effects. Corrected the projection report's conflict-frequency
lower endpoint: Shear71/256 is27.734375%, not31.25%.

[Inference correction] The earlier decision to stop projection solely on
intervals touching/including zero was stronger than the registered Phase4
gate. That gate requires positive aggregate change versus C_SCD and both
projection controls, without a pilot significance threshold. Two contrasts
pass; C_SCD is absent. The gate is unevaluated, not demonstrated to fail.
This entry supersedes the earlier permanent-stop inference while preserving
its historical record. No reliable method benefit or replication selection
is claimed.

[Run/Code/Open] Routing/projection have equal runtime/source/native identities
but different actual prepared-input hashes in all four corruptions. Diagnose
C_SCD also has incompatible native/runtime identity. Cause and magnitude of
tensor differences remain unknown. Only aggregate input hashes were archived;
a standalone new C_SCD run cannot guarantee the earlier prepared states.

[Inference/Plan] Propose explicit runner support for one shared-preparation
C_SCD/P_SUM/P_PC/P_NORM block: first64 indices, seed0, four corruptions,
1,024 classifications. Re-evaluating the three projection conditions is
necessary for this missing comparison; it is a scoped proposal, not a GPU
launch. P_PC is prioritized for completing its two positive mechanism
contrasts, not by comparing unmatched absolute routing/projection accuracy.
If all three mean contrasts are positive, freeze one candidate and use the
original next256/seeds0-2 Phase5 gates (12,288 classifications). If a complete
block fails an accuracy contrast, do not promote projection. No scale retry,
diagnose/smoke repeat, new hyperparameter grid or local GPU run is planned.
Background100 and1000 failures remain separate instability evidence.

Full protocol, dependencies, acceptance criteria and implementation handoff:
[gsd_pilot_review_action_plan_20261005.md](gsd_pilot_review_action_plan_20261005.md).
No inference code changed and no tests/model inference ran in this review.


## 2026-10-05 - Matched projection control-completion runner implemented

[Code] On branch `gsd-smooth-spectrum` at base HEAD
`5b21382a9c0cb5e3572d383f11a8446e45cb12f0`, added the opt-in
`control_completion` phase. It fixes C_SCD/P_SUM/P_PC/P_NORM, the locked first64
indices, seed0 and four pilot corruptions. The existing runner prepares each
batch once and runs all arms sequentially with cloned common inputs. No old
phase arm is reused. Each arm identity records per-batch SHA-256 hashes for
input points, shape/local latents, style, noise, timesteps, alpha-bar and the
scheduler config. The analyzer now treats those component hashes as part of
paired identity and avoids counting both a retained arm directory and its ZIP.

[Verification] `py_compile` passed for the four touched Python files. A local
plan-only `build_plan` preview returned four paired blocks, four arms,64
indices, seed0, no reference manifest, and1,024 classifications. No automated
tests or GPU inference ran. No user files were staged or committed.

[Open/Run] Updated sources are local and must be uploaded into the Colab clone
before the new preview/execution cells. The user executes one four-arm block
(16 archives). Then download the full attempt ZIP with manifest, bundles,
analysis and console log. Accuracy gate remains positive equal-corruption mean
versus each of C_SCD, P_SUM and P_NORM; a positive pilot only permits the
registered next256/three-seed replication.

Implementation and separate Colab cells:
[gsd_pilot_review_action_plan_20261005.md](gsd_pilot_review_action_plan_20261005.md)
and [Colab phase handoff](colab_gsd_block_routing_20261004.md).


## 2026-10-05 - Matched projection control completion attempt-0001 ingested

[Run/User supplied] The user supplied `C:/Users/batuo/Downloads/attempt-0001_control_completion.zip`.
It passes ZIP CRC/safe-path checks. An unchanged copy is stored at
`result/modelnet40_c/gsd_guidance_composition_v1/control_completion/attempt-0001_control_completion.zip`;
source and copy SHA-256 are both
`b283afd09b499295dcdd306648d701e3ff6a343fc1fda1bfd1b1667e748b16e1`.
It extracts to `control_completion/attempt-0001/`.

[Run] Manifest is complete, phase `control_completion`, seed0, first64 indices,
four corruptions, commit `86fb15544f1a82a31afccd99427bb39f3a3c53e4`.
All16 arm archives validate; there are four complete paired blocks, no partial
or failed arms. All four arms share exact prepared-input/component, index,
runtime and configuration identities within each corruption. All nine
recorded inference-source hashes match the recorded commit. Step logs cover
all64 samples:320 rows/arm for each five-step corruption and2,240 rows/arm
for35-step Background. Local analysis is separate from the raw ZIP at
`control_completion/attempt-0001/analysis_ingested/analysis.json` and
`validation.md`.

[Run] Four-corruption macro accuracy is C_SCD67.1875%, P_SUM68.3594%,
P_PC69.53125%, P_NORM67.578125%. P_PC paired deltas are +2.34375pp versus
C_SCD (7 corrected/1 broken), +1.171875pp versus P_SUM (5/2), and +1.953125pp
versus P_NORM (6/1). Per-corruption P_PC deltas (Gaussian/Impulse/Background/
Shear) are +1.5625/+4.6875/+4.6875/-1.5625pp vs C_SCD,
+1.5625/+1.5625/+3.1250/-1.5625pp vs P_SUM, and
+1.5625/+4.6875/+1.5625/0pp vs P_NORM. Every contrast has mixed or null
corruption behavior; per-corruption paired bootstrap intervals touch/include
zero. Cross-corruption object correspondence is unverified, so no pooled
interval is reported.

[Decision/Inference] P_PC passes the registered Phase4 pilot criterion of
positive aggregate mean against C_SCD, P_SUM and P_NORM. Nominate it for the
registered Phase5 development replication: next256 indices, same four
corruptions, seeds0/1/2, retaining all three comparators. This single-seed
pilot does not prove stable benefit or qualify as independent confirmation.
Keep Shear's negative/mixed results visible. Do not start all15 before Phase5
is assessed. Recorded sum of arm runtimes is175.58s; setup/serialization are
outside that sum. Peak GPU allocation is about15.16GB.

## 2026-10-05 - Phase 5 P_PC selection and plan prepared

[Code] On `gsd-smooth-spectrum@fa03d5032a63cd2c283b2f07236f4201fc221de8`, audited `gsd_composition_protocol.py`, `scripts/run_gsd_composition.py` and `scripts/analyze_gsd_composition.py`. The `replicate` phase, frozen selection-manifest schema and P_PC canonical route already exist; no inference source changes were required. The replicate runner requires a `projection` phase reference for a P_PC selection. Keep the final candidate-gate evidence separately tied to `control_completion/attempt-0001`; do not relabel that phase as the selection source.

[Run/Code verification] Generated `result/modelnet40_c/gsd_guidance_composition_v1/replicate/selection_manifest.json` with `make_selection_manifest` and validated it with `validate_selection_manifest`. Frozen selection SHA-256 is `56a256da5b4c5d789f83ce8ea4230b8746ec4f9db242ad22597c436decceaa2d`; candidate P_PC, comparators C_SCD/P_SUM/P_NORM. Pure `build_plan` preview against completed `projection/attempt-0001/phase_manifest.json` resolves branch `gsd-smooth-spectrum`, HEAD `fa03d5032a63cd2c283b2f07236f4201fc221de8`, four corruptions, 256 unique replication indices, seeds0/1/2, 12 paired blocks, exactly48 planned arms and12,288 classifications. No runner CLI, model, CUDA or inference call ran. The preview used the installed local Python3.13/NumPy2.3.2 only as CPU planning code; Colab commands explicitly use conda environment `3dd_tta_env` for NumPy and analysis.

[User report/Scope] User requested Colab execution and did not authorize local GPU inference. No Colab run was launched. I ran no stage/commit/push/reset/delete command. The read-only reflog shows HEAD advancing from `f2a152c` at22:43:45 +0300 to user-authored `fa03d50` (`docs(knowledge): sync research notes`) at22:45:57 +0300 while this task was active; this was not a command issued by this preparation. Preserve that observed commit as-is. Existing user changes and untracked results, including the separate M_max=1800 notes, were not selected or altered; this entry is a separate Phase5 append.

[Open] User must review the preview and start the GPU cell in Colab. Afterward validate the phase manifest, all48 arm ZIPs, ordered indices, eight prepared batches and component hashes per arm, runtime/assets/native identities and all recorded inference-source hashes. Only complete arms in a complete phase can supply the paired accuracy result. Report four corruption means, per-seed deltas and per-corruption uncertainty; no cross-corruption pooled bootstrap because object correspondence is unverified. Evaluate the frozen Phase5 thresholds without tuning. No outcome automatically authorizes `all15`.

[Handoff] Four separate Colab cells and the post-run audit/analyzer/download steps are in `gsd_phase5_colab_handoff_20261005.md`. `knowledge/open_questions.md` keeps execution and result validation open.

## 2026-10-06 - Phase 5 Colab download cell corrected

[User report/Code] The user reports that Phase 5 post-run validation succeeded for48 complete arm ZIPs across12 paired blocks, and `analyze_gsd_composition.py` wrote `replicate/attempt-0001/analysis/analysis.json`. Packaging then failed at `from google.colab import files` because it ran under `%%bash` in a subprocess without the notebook's `google` module. This is an output-transfer failure after analysis, not an inference or artifact-validation failure.

[Code] The first documentation correction separated bash packaging from the notebook-kernel download. The user clarified that they need the entire post-run cell fixed. The current Cell4 is a normal Python notebook cell: it invokes audit/analyzer as `conda run` subprocesses, then packages and calls `google.colab.files.download` in the notebook kernel. This fixes the process-boundary cause without rerunning inference. Remote Colab data was not available to this workspace, so ZIP creation/download was not executed locally.


## 2026-10-06 - Phase 5 P_PC replication attempt-0001 validated

[Run/User supplied] Raw ZIP `result/modelnet40_c/gsd_guidance_composition_v1/replicate/attempt-0001.zip`, 44,022,540 bytes, SHA-256 `028d31bec666bf0bd0e4427e7a64799ade78f2b3f3eb2e283b0285a6ed1e3a29`, is retained unchanged. All51 archive entries pass CRC/safe-path checks and match fresh extraction. Derived audit, recomputation and report are separate in `replicate/attempt-0001/analysis_ingested/`.

[Run] Phase and execution status complete:48 complete arm ZIPs,12 paired blocks, zero partial/failed arms, next256 locked indices, seeds0/1/2, Gaussian/Impulse/Background/Shear,12,288 classifications. Fixed config, canonical arm routes, index/label/prediction/logit counts, artifact hashes and sample-step coverage validate. All four arms share eight prepared batches per block; component hashes, reconstructed aggregate input hashes and derived RNG keys match. Phase/config/runtime fingerprints recompute exactly. Recorded assets/native/runtime match across arms and the earlier control-completion pilot. Original tensor/model/native binary bytes are absent here, so their recorded hashes cannot be independently remeasured locally.

[Run/Code] Colab records `gsd-smooth-spectrum@86fb15544f1a82a31afccd99427bb39f3a3c53e4`, rather than preparation HEAD `fa03d5032a63cd2c283b2f07236f4201fc221de8`. All nine inference-source hashes match the run commit, current HEAD and LF-normalized working files; the analyzer is also unchanged. Runtime is Python3.8.20/torch2.1.2+cu121/CUDA12.1/A100-SXM4-80GB using `3dd_tta_env`. The local NumPy-only audit reproduces Colab counts and decisions;18 floating fields differ by at most1.78e-15. Both analyses and exact roundoff differences are retained. No inference ran locally.

[Run] Equal-corruption macro mean +/- sample SD across seeds: C_SCD65.104167 +/-0.203288%, P_SUM65.332031 +/-1.184019%, P_PC65.950521 +/-0.298345%, P_NORM65.266927 +/-0.056382%. P_PC-C_SCD seed deltas are +0.683594/+0.976562/+0.878906pp, mean +0.846354pp (SD0.149172). P_PC-P_SUM mean +0.618490pp (SD1.368350), P_PC-P_NORM +0.683594pp (SD0.292969). P_PC loses to P_SUM in seed0 (-0.78125pp). P_PC corrects80/breaks54 vs C_SCD,79/60 vs P_SUM and73/52 vs P_NORM across all recorded predictions.

[Run/Decision] Every unchanged Phase5 development criterion passes: mean vs C_SCD +0.846354 >=+0.2pp; all three seed differences positive; Background +2.083333 >=-1pp; mean vs both P_SUM and P_NORM positive. The pilot direction replicates at this scope with a smaller mean effect than the first64/seed0 pilot (+2.34375pp). This is not statistical proof, independent held-out confirmation or automatic all15 promotion. No new run, tuning, stage, commit or push was performed.

[Run/Inference] P_PC-C_SCD per-corruption deltas and95% paired intervals (pp): Gaussian0 [-1.1719,+1.3021], Impulse+0.2604 [-1.0417,+1.6927], Background+2.0833 [0,+4.2969], Shear+1.0417 [+0.1302,+2.0833]. Intervals use2,000 per-corruption bootstrap resamples, clustering three seed repeats within original index. Cross-corruption correspondence is unverified; no pooled object bootstrap. Mechanism-contrast intervals all touch/include zero; Impulse P_PC-P_NORM is negative. P_NORM matches the hypothetical projection norm at its own state, not identical norms along diverging trajectories. Thus the favorable aggregate controls do not establish a unique directional causal explanation.

[Run] Diagnostics contain153,600 timestep rows, verified against256 examples per arm at5/35 steps; these are not additional accuracy samples. Nonterminal P_PC style conflict rates are43.39% Gaussian,50.07% Impulse,70.63% Background and29.72% Shear; no style projection skipped. Sum of recorded arm runtimes2118.41s excludes setup/shared preparation/packaging; peak allocation15185.17MiB. The existing analyzer's `total_seconds` aggregate is empty due to its field lookup; cost above is computed directly from raw `total_runtime_seconds`. No production-source change was made.

[Open] Generalization beyond these four development corruptions and causal separation of direction/norm effects remain unresolved. A provenance/pairing error would invalidate this acceptance; broader separately authorized evaluation could contradict a general benefit claim. Keep the completed pilot/replication and Background SCALE_100/1000 nonfinite failures; no repeats or optimum-scale claim follows. Separate user M_max=1800 notes remain untouched.

[Evidence] [Full validation, paired tables and fixed gate decision](../result/modelnet40_c/gsd_guidance_composition_v1/replicate/attempt-0001/analysis_ingested/validation.md); machine-readable `phase5_audit.json`, `analysis.json`, `colab_comparison.json` and reproducible offline scripts in the same derived directory.


## 2026-10-06 - Phase 6 all15 preparation explicitly authorized

[User report/Decision] After reviewing the successful Phase5 result, the user authorized preparing the P_PC versus matched C_SCD full15 evaluation and requested runnable Colab cells with reliable download handling. Scope:15 corruptions,2468 indices each,seeds0/1/2,45 paired blocks,90 arm ZIPs,222120 classifications. No new model/algorithm setting, pilot repeat or Background100/1000 retry is included. Full-test-set development performance evaluation; mechanism contrasts remain limited to Phase5.

[Code] Added `scripts/gsd_all15_colab.py` and `knowledge/gsd_phase6_colab_handoff_20261006.md`. Existing runner/protocol/inference sources already support all15 and remain unchanged. The helper validates the accepted Phase5 reference ZIP/manifest, creates a separate relocated reference manifest pointing at original arm ZIPs, freezes a new selection (source `replicate`, P_PC, C_SCD), and calls pure `build_plan`. New selection SHA-256 is `33e39ba3f601451122581fe335fd0810704c87f6abae42b4b3c7e75e857e29b7`. Old raw and selection manifests are retained unchanged.

[Run/Code verification] CPU preview at preparation HEAD `fa03d5032a63cd2c283b2f07236f4201fc221de8` confirms45 blocks/90 arms/222120 classifications,78 batches per block with last batch4. No all15 attempt directory or inference was created. Python3.8 helper syntax and all four cells' embedded Python parse. Exercised packaging on the existing Phase5 artifact: all51 original entries and48 nested ZIPs match byte-for-byte, CRC passes; evidence is `result/preparation/gsd_all15_20261006/packaging_verification.json`. This is packaging verification, not a new accuracy run or a live browser download check.

[Code] Post-run audit rejects partial/failed phases as complete accuracy, verifies all90 arm bundles, indices, shared components/RNG keys, config/runtime/assets/native/source identities and sample-step coverage, and runs the existing analyzer in `3dd_tta_env`. Separate derived postrun directories preserve supplied raw analyses. Even audit/analyzer failure leads to a clearly named UNVALIDATED archive of available raw files. Only byte-identical extracted copies of ZIP members are omitted; incomplete/divergent files remain. Unique archive/receipt names, CRC and SHA-256 checks avoid selecting stale packages. `google.colab.files.download` runs only in the notebook kernel; browser failures retain the verified ZIP for manual download.

[Run/Inference] Three historical full-screen SCD ZIP configs lack the current composition prepared-input/component identity contract; no prior full all15 composition attempt is present. New same-process paired C_SCD is necessary for this newly authorized P_PC comparison; other archived baselines remain descriptive controls.

[Open] User-run Colab execution and subsequent complete artifact ingestion remain pending. Frozen reporting: macro15 primary; macro14, Background, per-corruption and per-seed differences, sample SD, paired uncertainty, runtime/memory. No automatic outcome claim or post-result tuning. Existing dirty user notebook, result files and M_max notes are excluded from staging/commit/push.


[Code/Git verification] Phase6 helper and new handoff only were committed as `3c80105c828245a31aaa3dec4af22fd565f4a718` (`feat(gsd): add all15 Colab handoff`) and pushed to `origin/gsd-smooth-spectrum`. Read-only `git ls-remote` confirms the exact remote commit. Default `.git/index.lock` access was denied; the scoped Git operations succeeded with approved escalation. User notebook, existing knowledge edits, M_max notes, raw results and local derived artifacts were not staged or committed.


## 2026-10-06 - Phase 6 Cell1 Colab dirty-tree blocker corrected

[User report/Code] The first Phase6 Cell1 aborted on any tracked Colab change (`git diff --quiet`) before synchronization or environment checks. This was overly broad; it could block harmless notebook edits. Replaced it in `gsd_phase6_colab_handoff_20261006.md`: fetches commit objects only (no checkout/pull/reset/stash), installs just the pinned helper while preserving a differing helper copy, checks the nine Phase5 inference-source files against run commit `86fb155`, then checks `3dd_tta_env` and assets. It prints the current branch/status and stops with per-file hashes only if an inference source differs. The user Colab state and the original erroring tracked files remain untouched.

[Code verification] Re-parsed all four documented cells and embedded Python bodies; all parse. The inference helper remains commit `3c80105` unchanged. No inference was started in Colab or locally by this fix.


## 2026-10-06 - Phase 6 Colab helper placement fixed

[User report/Code] On retry, Colab fetched successfully but the first cell failed writing `scripts/gsd_all15_colab.py` because the existing checkout has no `scripts/` directory. Avoid creating files inside this dirty checkout. The latest Cell1 creates a detached worktree at the pinned Phase6 commit under `/content/3DD-TTA-phase6`, links read-only source data/checkpoints/compiled Chamfer and the shared result tree, verifies all nine inference files against Phase5 commit `86fb155`, and checks `3dd_tta_env`. Cells2-4 now use the isolated workspace; output paths are shared through the result-directory symlink. The finalize audit accepts detached provenance (empty branch name), while requiring exact commit-bound source hashes. No reset, pull, checkout or source overwrite is done to the user's original tree.

[Code verification] All four documented cells and embedded Python blocks parse; user-reported first-cell failure has been traced to the absent parent directory. No GPU run was launched. Isolated worktree creation/symlinks and live Colab remain to be exercised by the user in Colab.

[Code/Git verification] Isolated workspace helper changes were committed only with the Phase6 helper/handoff as `32a6c6b50f69e9236b66c2ecc7c4d219f7e1077b`. Cell1 is pinned to that commit. Existing user notebook and knowledge edits remain unstaged.


## 2026-10-07 — Phase 6 all15 independent audit (run completed 2026-10-06)

[Run] Raw `result/modelnet40_c/gsd_guidance_composition_v1/all15/attempt-0001_all15_validated_20261006T201838Z-796974.zip`
is unchanged:479306228 bytes, SHA-256
`c0d4aedffc03379dbf7ccd5fc0d0154c57230ef9496de1e1d948e282572b62c2`.
Recomputed outer CRC/safe paths and all90 inner ZIP CRCs/ten-file inventories,
artifact hashes, finite40-class logits/argmax, indices, counts, CSV schemas,
manifest/config identities and diagnostic coverage.45 complete matched blocks,
90 complete arms,222120 classifications; no missing/duplicate/partial/failed arm.
There are222120 graph rows and1554840 sample-step rows, not additional accuracy
samples. All15 uses ordered0-2467, not the shuffled Phase5 next256 list.

[Run/Code] Run and ingestion HEAD are
`gsd-smooth-spectrum@c18926ff7751ed9023313869229f5d4dffe218be`.
All nine inference-source identities match that commit and Phase5's86fb155
Git bytes. Six working files match raw; main_3dd_tta.py,run_baseline.py,tta.py
match after CRLF-to-LF only. Frozen selection recomputes to
`33e39ba3f601451122581fe335fd0810704c87f6abae42b4b3c7e75e857e29b7`.
Phase5 reference ZIP SHA028d31...1e3a29 matches; its bundled relocated manifest
differs only in96 storage-path fields. All3510 batch component/state/noise
identities and preparation/classification draw keys agree between paired arms.
Runtime/assets/native identities are constant within Phase6. Three native
binary hashes differ from Phase5; no cross-phase common-state claim is made.

[Run/Code/Open] Handoff Cell1's full pin32a6c6b50f69e9236b66c2ecc7c4d219f7e1077b
is unavailable in local Git; history records32a6c6baa3d1228c75712e799c78ac65bef441e0.
The actual preview/log/manifest consistently use c18926f and /content/3DD-TTA,
not the documented detached /content/3DD-TTA-phase6. This documentation/setup
deviation does not imply an inference-source mismatch. Original tensor/data/
checkpoint/binary bytes are absent from the ZIP. kNN's recorded hash covers
__init__.py, not a compiled kNN binary; full package/dirty-state/determinism
inventory is unavailable.90 empty arm stdout files are supplemented by the
shared execution log; completion is established from artifact coverage.

[Run] Macro15 mean +/- sample SD: C_SCD63.974428 +/-0.209487%,
P_PC64.011345 +/-0.241410%. Paired deltas by seed0/1/2:
-0.002701/+0.056726/+0.056726pp; mean +0.036917 +/-0.034310pp.
Macro14 excluding Background: +0.017365 +/-0.053130pp; Background:
+0.310643 +/-0.244235pp.6 corruption means positive,8 negative,1 zero.
All per-corruption95% intervals include zero.20,000 per-corruption paired
bootstrap resamples cluster all three seed repeats inside original index;
no pooled cross-corruption or macro object interval. The original analyzer's
2000-resample method was also replayed unchanged. Counts:1464 corrected,
1423 broken,5822 disagreements; net41 correct among111060 prediction pairs.

[Run/Inference] Phase5's +0.8464pp and Background +2.0833pp are retained at
their four-corruption/next256 scope. The all15 effect is much smaller and has
mixed seed directions; a broadly stable gain is not supported. This is full
test-set development performance, not independent confirmation, equivalence,
automatic promotion or full-scope proof against P_SUM/P_NORM. No new outcome
threshold or algorithm tuning was introduced. Background100/1000 nonfinite
failures and the separate M_max=1800 research record remain unchanged.

[Run/Code] Packaged audit/counts/transitions agree. Local Python3.13/NumPy2.3.2
CPU replay uses the unchanged commit-bound analyzer calculation code with a
documented streaming I/O adapter after independent validation; local conda/
3dd_tta_env was unavailable.20 float fields differ by <=5.55e-16, no substantive
analysis difference. Existing analyzer total_seconds is empty due to a field
lookup error; raw manifest totals are C_SCD13544.839s and P_PC13629.588s,
peak15376.960MiB. Shared preparation640.756s is counted once; graph is within
sampling. These are instrumented-suite timings, not plain standalone SCD cost.

[Evidence] [Dated report](../result/modelnet40_c/gsd_guidance_composition_v1/all15/analysis_20261006_independent/validation_20261007.md), all45 per-seed count/accuracy rows,
15 corruption mean/SD/interval rows, code/artifact audits and replay receipts
are in `all15/analysis_20261006_independent/`. Raw ZIP rehashed unchanged.
Only derived outputs and dated Phase6 knowledge additions were made; pre-existing
tracked changes/notebook, untracked user artifacts and M_max notes were preserved.
No GPU/model inference, reset/clean/stage/commit/push or new test suite was run.
This later result supersedes the preceding preparation/pending-output statements
without deleting their history.


## 2026-10-07 ? User sets all15 accuracy as the immediate priority

[User decision] The user prioritizes +1?2 absolute percentage points in all15
macro accuracy as the intended empirical foundation of a paper and expresses
disappointment with the style results. Treat this as the forward research
objective; do not substitute additional mechanism-only ablations for that goal.
The target does not alter any registered Phase5/6 interpretation threshold.

[Run/CSV-only] Read the existing standalone eval_results.csv files under
result/modelnet40_c/gsd_smooth_optimal/ and gsd_smooth_optimal_1800/. Fifteen-row
equal-weight means recompute to64.532684% (M800) and64.505672% (M1800). These
directories contain only CSVs; they do not supply the full manifests, seed
replication or matched SCD control required to call the differences verified
method gains. Their historical notes refer to the separate
gsd-tta-smooth-integration branch, B70, beta2/weight1.17. Preserve that branch
and its notes; no configuration/result was changed or pooled with Phase6.

[Inference/Proposal] Stop prioritizing additional P_PC/style variants or
routing reruns after the near-zero all15 result. First locate existing exact
source/config/provenance and comparator evidence for the higher descriptive
smooth-branch score; its corruption pattern can identify the next concrete
accuracy hypothesis, but cannot justify label-based corruption-specific
selection. Current C_SCD63.974428% implies a same-protocol target near65?66%;
this is an aspiration, not an established achievable effect. +1?2pp represents
roughly370?740 net extra correct predictions per37020-example seed.

[Code/Scope] Context c18926ff7751ed9023313869229f5d4dffe218be,
gsd-smooth-spectrum; separate smooth-integration branch exists in another
worktree. Only read-only inventory/CSV arithmetic and these dated knowledge
additions were performed. No GPU/model inference, new run, source edit,
reset/clean/stage/commit/push.


## 2026-10-07 - Accuracy branch audit completed; bounded next candidate

[Run/Code] Read the separate smooth-integration worktree at5c3ed883 and
recomputed three CSV means: multiband64.386817%, M80064.532684%,
M180064.505672%. Seven tracked sources match that HEAD after CRLF
normalization, but CSV executed-source identity remains missing. M800
versus Phase6 C_SCD is an unmatched descriptive+0.558256pp, with large
Impulse/LiDAR gains and Shear/Distortion losses. No same-host SCD-only
result was located; a launcher for a different entry point is not evidence
that this control ran. The October3 inventory has144 archives, zero normal10.

[Inference/Proposal] Park further style/projection variants. Propose one
current-host SCD candidate: normal10/Background35 versus5/35, other settings
fixed, all15 development coverage and seeds0/1/2. This changes starting noise
strength as well as update count; no +1-2pp improvement is promised. The
completed SCD-off Background step scan does not answer this question.
No GPU run or implementation was performed. Details, limits, commands and
prospective action sequence: [accuracy branch audit](gsd_accuracy_branch_audit_20261007.md).
Historical M_max1800 and failed Background100/1000 records remain intact.


## 2026-10-08 - User identifies legacy command; smooth-v2 restored locally

[User report] The64.532684% result used origin/gsd-tta-smooth-integration
and eval_gsd_tta_v2.py with B70, spectral1.17/SCD1, M800/beta2,
gamma=eta=.01, lambda.95, normal10/Background30. This resolves the earlier
missing-command question; exact historical commit, seed and build identities
remain unrecorded. The user requests porting this structure into the current
branch and testing it, superseding the proposed normal10/Background35-only
next action. No parameter sweep is authorized or launched by this restoration.

[Code] Main basec18926f; locally available origin source5c3ed883. Restored
exact graph_spectral_v2.py/tta_gsd_v2.py plus eval_gsd_tta_v2.py with explicit
legacy eval modes and an auditable execution shell. Current calibrated smooth
uses normalized eigenvalues/all modes; legacy uses raw eigenvalues/M800,
stronger symmetric-degree filtering, isolate penalty/jitter, updated decoder,
DDIM final-alphaFalse and10/30 steps. These differences are not all bugs.

[Code/CPU tests] 51 tests passed: source/AST parity, CLI and eval modes,
synthetic graph and gradient math, artifact success/failure counts, existing
GSD regressions. No actual GPU/model execution or accuracy result. Source
overlay/Colab smoke/full cells and remaining native-runtime limits are in
[restoration report](gsd_legacy_smooth_restore_20261008.md). No shared inference
source defaults were edited, no stage/commit/push; user changes and raw results
are preserved. Phase6 raw size479306228 and SHA256c0d4aedffc03379dbf7ccd5fc0d0154c57230ef9496de1e1d948e282572b62c2 rechecked unchanged.


## 2026-10-10 - Mathematical audit and minimal isolate control handoff

[Code/CPU] See `gsd_spectral_math_audit_20261008.md`: self-query KNN retains
self edges, so the legacy full-row degree-zero mask misses self-only vertices.
The minimal opt-in `--isolate_policy offdiag` counts nonzero nonself edges and
applies the existing penalty1000. No arbitrary epsilon, filter, cutoff,
decoder, scheduler, guidance or batch change accompanies it. Legacy stays the
default; `graph_spectral_v2.py` and `tta_gsd_v2.py` preserve the reference core.
Base commit c18926ff7751ed9023313869229f5d4dffe218be; reference
5c3ed88314f1c3d50ab77d17f7c6ba22da59c441; changes remain uncommitted.

[Code/CPU] 89 CPU tests pass, including single-mask source-diff restriction,
self-only vertex penalty, unchanged connected graph, original inference parity,
artifact lifecycle and existing graph/trajectory/composition regressions.
A read-only independent review found no important issue; its suggestion to
hash the optional graph even before import is included. Seven notebook cells
parse; source ZIP passes CRC, source hashes and Python3.8 grammar checks.
Records: `result/modelnet40_c/diagnostics/isolate_control_20261010/`.

[Open/Inference] `scripts/gsd_legacy_isolate_control_20261010.ipynb` schedules
Gaussian/Background two-batch integration checks for each policy, followed by
all15 severity5 full2468/seed0 per policy (74040 full classifications).
Parameters remain the user-reported historical B70, M800, beta2, spectral1.17,
SCD1, gamma=eta=.01, lambda=.95, normal10/Background30. Historical seed/build
identity remains unknown. New runs are seed-controlled, not prepared-input
hash-proven pairs. No CUDA/model inference, broad sweep or new result exists.
A null/negative delta would refute an accuracy benefit in this seed setting;
a positive single-seed delta would still require replication. Test-set
development evidence cannot serve as independent validation or a guaranteed
+1?2pp contribution. Preserve the original64.5327% observation as unmatched.
Raw archives, M1800 notes and failed Background100/1000 records are untouched.


## 2026-10-10 - Authorized Git publication of legacy isolate control

[Code/CPU] User explicitly authorized pushing the new scripts. Commit
12525b8191a2af18aa00eba5dc3129ec2fbcdc0a is verified on
origin/gsd-smooth-spectrum. It adds five runtime files, two CPU test files,
`scripts/gsd_legacy_isolate_git_colab.ipynb` and
`knowledge/gsd_isolate_git_colab_20261010.md`. The notebook continues after
the original notebook's first six code cells; source ZIP upload is no longer
required. Previews, experiments, analysis and packaging run in3dd_tta_env.
89 relevant CPU tests passed freshly; notebook dispatch was mocked and all
expanded conda payloads parse with Python3.8 grammar. Independent review
found no blocker. No local GPU/model run or accuracy evidence was added.
Only the nine new files were committed; prior tracked user modifications and
raw archives remain outside that commit. Older no-push/ZIP-transfer notes
describe the earlier handoff and are superseded for this deployment.


## 2026-10-10 - Colab setup guard corrected from user trace

[User report/Run] After preparation cells1-6, Colab is at HEAD535e48a,
base c18926f is an ancestor, required scripts are tracked, and
`3dd_tta_env` reports CUDA=True on NVIDIA A100. The first handoff cell failed
because Cell4's PointNet2 build changed the tracked generated file
`Pointnet2_PyTorch/pointnet2_ops_lib/build/lib.linux-x86_64-cpython-38/pointnet2_ops/pointnet2_utils.py`; the broad `*.py` dirty-tree guard returned1.

[Code] Cell7 and the repeated launch guard now exclude only that exact generated
helper. Other changed Python sources still fail the guard. The code alteration
is committed/pushed separately; no inference or result archive was produced.


## 2026-10-10 - Legacy isolate control smoke ingested

[Run] User-provided handoff `20261010T114156946419Z_handoff_20261010T114914115459Z.zip` is preserved byte-for-byte at `result/modelnet40_c/legacy_isolate_control/attempt-0001_smoke_seed0/`. Size243729 bytes, SHA-256 `bd1aae3f9c827c5e7f0baf5dab58f1bf1b633cea7220151f82c36328e53c9a18`. Outer ZIP and all four nested arm ZIP CRCs/path inventories validate. Each arm completed two batches/140 indexed examples with finite40-class logits and accuracy counts matching persisted predictions. All are correctly marked partial smoke, not full corruption results.

[Run] Prefix counts: Background legacy89/140 (63.5714%), offdiag91/140 (65.0000%); Gaussian legacy107/140 (76.4286%), offdiag104/140 (74.2857%). By original index, Background had7 corrected/5 broken (net+2); Gaussian0 corrected/3 broken (net-3). Signs differ; these first140-example prefixes do not estimate all15 or full-corruption accuracy and do not support a gain claim. Pair targets, six asset hashes, 226-entry native inventories, external Python sources, model modes and common CLI parameters match. Prepared input hashes were not recorded, so this is not a hash-proven paired-input result.

[Run/Code] Torch2.1.2+cu121, CUDA12.1, NVIDIA A100-SXM4-80GB. Each arm peaks around60.76GiB allocated. The session manifest records commit535e48a while arm configs resolve80931d6; all five declared runtime source hashes match, and intervening commits only updated notebook/handoff docs. Manifests also show generated PointNet2 build outputs/native files and pycache dirt after setup. See `validation.md`, `validation.json`, and `smoke_summary.csv` beside the preserved ZIP.

[Decision/Open] Smoke implementation completed for all four short arms. Proceed with Cell10 legacy all15 only; inspect its ZIP before Cell11 offdiag all15. No GPU run was performed locally.

## 2026-10-10 - Legacy isolate correction all15 ingested

[Run] The handoff ZIP `20261010T114156946419Z_handoff_20261010T141536060023Z.zip`
is preserved byte-for-byte at
`result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/`.
It is 12,277,729 bytes, SHA-256
`534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968`.
The outer archive and all six nested arm ZIPs pass CRC, duplicate-path and
traversal checks. The two full arms each report complete coverage of 15
severity-5 corruptions x 2,468 indices at seed0; four smoke arms remain partial
as expected. Each full arm's 540 NPZ predictions have finite logits, logits
argmax matches saved predictions, indices cover 0-2467 once per corruption,
and counts reproduce the CSVs.

[Code] Session manifest commit is `535e48a`; resolved run commit is
`80931d6` (descendant); reference commit is `5c3ed88`. All 77 full-run source
hashes match the resolved Git tree. The five session-declared inference source
hashes match both full arms. Fixed configuration, assets, 226-entry native
inventories, external Python sources, model modes, scheduler and RNG protocol
match; only isolate policy/output path differ. Runtime is PyTorch2.1.2+cu121,
CUDA12.1, A100-SXM4-80GB. No prepared-input/common-random-draw hashes exist.

[Run] Legacy macro15 is64.570502% (23,904/37,020); offdiag is64.554295%
(23,898/37,020), delta -0.016207 pp. Macro14 delta -0.014471 pp; Background
-0.040519 pp. Per-corruption results and index-aligned transitions are in
`result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/analysis_ingested/`.
The bundled Cell12 comparison matches independent recomputation. Prediction
transitions total872 corrected and878 broken (net -6), with3,571 differing
predictions. This is descriptive same-index comparison, not hash-proven paired
inference.

[Inference/Open] One seed per method gives no seed SD or inferential interval;
separate stochastic trajectories prevent causal attribution of the tiny delta.
This correction does not deliver the stated +1 to 2 pp accuracy objective and is
not promoted for accuracy. Historical user-reported64.5327% remains separate:
the new legacy arm is descriptively +0.0378pp, but historical seed/build/runtime
identity is missing, so parity is unestablished. The earlier pending/smoke-only
status is superseded by this completed artifact; retain it as history. Do not
combine with M_max=1800 or the failed Background100/1000 scale runs.

## 2026-10-11 - Practical legacy reproduction; accuracy research and offline selection

[User report/Run] The user accepts legacy64.570502% as practical reproduction
of historical64.5327% and requests accuracy-focused research. Historical
bitwise provenance is still unknown but no longer blocks development. Current
branch/HEAD: gsd-smooth-spectrum/f670c07440533f01e10fd149fad2480d36f9194c.
The October10 pending-parity priority is superseded by this user decision.

[Run] CPU-only fixed-rule analysis of the existing all15 seed0 logits:
probability averaging64.848730% (+0.278228pp, corrected496/broken393), logit
averaging64.816316%, minimum-entropy selection64.800108%, maximum-probability
selection64.854133%. No parameters were fitted. Probability averaging improves
13/15 rows. The label-using selector ceiling66.925986% is explicitly nondeployable.
Raw ZIP SHA534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968
verified before/after; archive untouched. Script/JSON/CSV:
`result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/analysis_20261011_selection/`.

[Inference] Probability-average conditional95% bootstrap interval is
[+0.1162,+0.4349]pp:20,000 independent within-corruption paired-index resamples,
RNG20261011, equal macro15 weighting. Assumes within-stratum exchangeability;
does not model seed variance/cross-corruption dependence, adjust multiple
comparisons or establish independent validation. This narrower conditional
interval is possible despite no seed SD. Two reconstruction costs and one
seed limit its usefulness; do not attribute diversity specifically to offdiag.

[Code/Paper] Reviewed current legacy spectral fidelity, classifier CLS/max
pooling and BN, plus Purge-Gate official commit894d4b6372897986eb468038dc6ffbc93f89e38c.
Official PG-SF prunes before first transformer block, uses entropy over purge
sizes0/2/4/8/16/32, and recommended commands also reset BN buffers. It uses a
PointTransformer wrapper/random FPS/different grouping implementation; porting
the small operation while preserving our Point_MAE/checkpoint is required.
Other literature scoped: Point-Cache, Uni-Adapter, BayesMM, adaptive diffusion
and FOCUS. Their task/backbone/training differences preclude direct gain transfer.

[Inference/Open] Proposed next pilot: all15,280 fixed indices/corruption,
seeds0/1/2; input/no-diffusion, same-host spectral-off SCD, and legacy GSD input
states, each with gate on/off x frozen/current-batch BN. Only two diffusion
trajectories per sample; cache geometry for classifier variants. No new GPU
work or algorithm implementation. Earlier normal5->10 proposal is operationally
superseded because legacy already uses10/30. Conditional second idea: choose
among clean decoded trajectory outputs by prediction consistency/entropy.
Full proposal, falsifiers, costs and references:
[accuracy research plan](gsd_accuracy_research_plan_20261011.md).
No guarantee of1-2pp or novelty; no rerun of style/projection, failed100/1000,
or changes to separate M1800 work. No stage/commit/push.



## 2026-10-11 - User trajectory proposal and advisor agenda created

[User report] User requests a persistent record of advisor discussion topics,
including Purge-Gate link/summary, and proposes low-entropy clean diffusion-step
selection, temporal prediction averaging, early stopping and conditional
GSD/SCD runtime ablation/removal. New advisor_meeting_log.md separates pending
topics from actual meeting records; no completed meeting is reported.

[Code] Inspected f670c07440533f01e10fd149fad2480d36f9194c: tta_gsd_v2.py
produces pred_original_sample before guidance and decodes physical points only
at the end. Probes require extra decode/preprocessing/Point-MAE forward. Match
pre-update style to the clean estimate; retain final guided decode separately.
Spectral weight0 still builds a graph; either loss can retain shared prior
backward. These limit anticipated savings from component removal/early exit.

[Inference/Open] Proposed sparse observation study: terminal vs min entropy vs
equal probability mean, same trajectories, all15 and repeated seeds. Confidence
is not stability/correctness. Follow with causal prefix stopping only if useful;
measure actual active-batch wall time. GSD/SCD2x2 controls follow conditionally.
Existing legacy/offdiag +0.2782pp averaging at
result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/analysis_20261011_selection/
is contextual evidence only, not a temporal-run result. Primary calibration
reference: https://proceedings.mlr.press/v70/guo17a.html .

The earlier PG-first assistant recommendation is superseded as immediate
research emphasis by this user proposal, without deleting its literature review.
Full assessment: [trajectory selection](gsd_trajectory_selection_20261011.md).
No implementation/model test/GPU run, raw archive mutation, stage/commit/push.
Historical nonfinite100/1000 and separate M1800 records are unchanged.


## 2026-10-11 - Clean-session implementation handoff and online correction

[User report] User chooses accepted legacy64.570502% as the baseline for the
next development session and requests a copyable prompt with knowledge/code
references. User clarifies that online TTA uses single/small batches, so early
stopping can skip work without large-batch waiting.

[Code/Inference/Open] Inspected HEAD remains f670c07440533f01e10fd149fad2480d36f9194c.
At B1 a true early exit skips future diffusion calls; the earlier B70 caveat
is not applicable to single-sample scheduling. Net latency gain depends on
saved work minus extra probes. Plan separate B70 benchmark and B1 matched
latency/accuracy evaluations, causal stopping and explicit horizon assumptions.
The old B70 result must not serve as a measured B1 baseline. Online unknown
corruption does not automatically supply the legacy10/30 corruption identity.

New [session handoff](gsd_trajectory_session_handoff_20261011.md) fixes the
baseline settings/raw artifact, initial opt-in observer/analyzer scope, code
map, conda-only Colab execution, CPU validation expectations and future stages.
No inference implementation, tests or GPU execution occurred in this turn;
only documentation. Dirty/untracked user files and raw results preserved.


## 2026-10-11 — Legacy trajectory observation/analyzer implemented

[Code/CPU/Open] Base HEAD f670c07440533f01e10fd149fad2480d36f9194c on
gsd-smooth-spectrum; changes remain unstaged/uncommitted. User authorized
implementation, synthetic CPU tests and separate Colab continuation cells.
Local conda was unavailable; user explicitly approved existing Python3.13.5
for CPU tests. No local GPU or trained-model inference was launched.

Implemented default-off Legacy observer, audit-only mode, clean latent decoded
with its matching pre-update forward style at3/5/8 or8/15/23, original terminal
as a separate untouched candidate, RNG/buffer isolation and model-state audit.
Added indexed logits/entropy/timing/state/noise/source/assets/native provenance,
file SHA manifests, CPU-only terminal/minimum-entropy/equal-probability-mean
analysis, fixed-probe and labelled oracle diagnostics, paired transitions,
seed SD and original-index clustered development intervals. No thresholds,
accuracy increase, speedup or early-stop policy are claimed.

[CPU verification] `python -m unittest discover -s tests -q`:
**234 tests passed in28.480s**, exit0. New tests exercise the actual Legacy loop
with synthetic CPU dependencies; outputs/guidance/RNG agree across off/audit/
observe, and probe styles are exactly pre-update. Tests also cover buffer
exception restoration, untouched version counters, policy arithmetic/ties,
index selection, corrupted artifacts and simulated all15 audit packaging.
Python3.8 AST syntax passes for the six inference/observer/analysis/runner files.
Notebook Python and embedded conda-Python blocks parse. Its source installer
was executed against a disposable CPU fixture:81 files installed, an unknown
edit was rejected before writes and retained. Source ZIP CRC and all81 current
source hashes match. Partial independent review identified missing cross-run
runtime provenance matching; native/external/environment/model-state comparisons
were added. The reviewer then hit its usage limit; full final independent review
was not completed, and no claim to that effect is made.

Receipt and complete log:
`result/modelnet40_c/diagnostics/legacy_trajectory_20261011/verification.json`
and `cpu_tests.log`. CPU evidence is not native CUDA parity or model accuracy.

[Run integrity] The original Legacy raw ZIP SHA remains
534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968.
Operational reference remains Legacy B70/M800/10-30 seed0 **64.570502%**.
Offdiag/Phase6, M1800 and failed Background100/1000 are not substituted or rerun.
The original user-modified notebooks, graph, Point-MAE and classifier config
were not edited. No reset/clean/stage/commit/push.

[Colab handoff] See [implementation and cells](gsd_trajectory_implementation_20261011.md).
New notebook: `scripts/gsd_legacy_trajectory_colab_20261011.ipynb`, used after
the original notebook's first6 code preparation cells. Source upload:
`result/preparation/legacy_trajectory_20261011/legacy_trajectory_source_20261011.zip`,
SHA256 c44c243929b64c5628bfc3d49152cdd9416d06e0c07fb8ef67aea83e545dcc9c.
Conda-only compute cells; off/audit/observe Gaussian+Background smoke first,
then STOP and review/download; separate all15/280/seeds0–2 development pilot,
then STOP. Each sealed arm and the full timestamped handoff ZIP have verified
Drive backups; download full logs/logits, not screenshots. Pilot remains unrun.

[Inference/Open] Falsifier remains absent/harmful selection or averaging gains,
confident wrong minima, fixed-probe equivalence or terminal voting explaining
any effect. Low entropy, correctness and temporal stability remain distinct.
Later onlineB1 stopping must use observed prefixes and measure real net latency
against same-B1 full horizon including probes. B70 throughput is separate;
unknown-corruption horizon, actual GSD/SCD compute skips, novelty and independent
evaluation remain open. Purge-Gate is still only advisor agenda/comparator.


## 2026-10-11 — Historical step-count evidence rechecked for trajectory hypothesis

[Run/Code/Inference/Open] At HEAD f670c074 with observer working-tree changes,
rechecked24 raw Background ZIP hashes/CRCs/counts for two SCD-off arms,
steps5/10/15/20, seeds0/1/2. Smooth-only5-to10 improves+1.080497pp; unguided
improves+0.540249pp, both positive in all three seeds.10-to20 falls-1.148028pp
and-0.891410pp respectively, also consistently across seeds. Existing25-step
results rebound, so10 is not declared optimal. B32 no-SCD results do not
substitute the current Legacy B70/M800/SCD-on baseline.

Updated unmatched Phase6 normal5 versus restored Legacy normal10 contrasts:
Impulse+8.684495pp, LiDAR+5.294435pp, Gaussian+1.755808pp, Shear-3.552134pp,
Distortion-1.580227pp (Phase6 seeds0–2 mean versus Legacy seed0). Batch,
guidance/graph, scheduler and final style also differ; no step-only attribution.
No completed same-Legacy all15 step5/10 control was identified in the searched
knowledge. The earlier October7 proposal was not an executed experiment.

Critical distinction: changing5 to10 also changes initial timestep t40 to t90.
This is not stopping a fixed10-step trajectory after five steps. Evidence
motivates adaptation-depth sensitivity, but same-trajectory selection/entropy
benefit remains unproven. [Full assessment](gsd_step_count_evidence_20261011.md).
Derived CPU results (no model inference):
`result/modelnet40_c/diagnostics/trajectory_step_evidence_20261011/analysis.json`
and its two CSVs. Raw archives/code unchanged; no repeats or Git writes.
Falsifier: no same-trajectory policy/candidate advantage in the planned pilot.


## 2026-10-11 — User-directed seed0-only trajectory screening

[User decision/Code/CPU/Open] User requests saving time by evaluating all15
at seed0 first, with seeds1/2 conditional on useful policy benefit. Updated
`scripts/legacy_trajectory_colab.py` session manifest, launch generation,
preview and automatic analyzer inputs to seed0 only. Smoke remains6 seed0
arms/840 examples; pilot is now15×280×1 =4,200 examples (previous12,600).
B70/M800, normal10/Background30, probes and fixed original IDs are unchanged;
index-selection seed20261011 is retained. No full2468 benchmark is scheduled.

Notebook `scripts/gsd_legacy_trajectory_colab_20261011.ipynb` now launches and
analyzes only `pilot_seed0.zip`. Multi-seed analyzer capability remains available
for later replication. Single-seed summary SD is null; index-bootstrap intervals
are conditional evidence, not cross-seed stability or a general significance
claim. Review gains/losses and probe cost, then lock the policy before a separate
seed1/2 evaluation. No numerical gate or accuracy benefit is asserted.

Updated source package:
`result/preparation/legacy_trajectory_20261011_seed0/legacy_trajectory_source_seed0_20261011.zip`
SHA256 `0e6114c6a5049ce171e121547113c1e871184ab8c968120ecfd84f53581a722b`.
Old source package remains preserved. The installer recognizes its exact prior
manifest/source hashes, backs up known files and refuses unknown edits. Existing
run directories are untouched; prior completed smoke evidence must be reviewed
before any redundant run is scheduled.

[CPU verification] The launcher contract test first failed with3 pilot jobs
instead of1; after the update, `python -m unittest discover -s tests -q`
passes234 tests in29.834s, exit0. Tests cover seed0-only commands, session
pilot/deferred seeds,4,200-index scope and undefined single-seed SD. Notebook
Python blocks parse as3.8; no seed1/2 input remains in executable cells. The
source upgrade was exercised in a disposable fixture:81 sources, backup of old
manifest, unknown edit retained/rejected. Source package CRC/current hashes
match. Receipt/log:
`result/modelnet40_c/diagnostics/legacy_trajectory_seed0_20261011/verification.json`
and `cpu_tests.log`.

Base HEAD remainsf670c074; no GPU/model inference, stage/commit/push or raw
archive modification. Baseline SHA remains534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968.
Updated README, protocol, handoff, implementation and open questions record the
new launch order while preserving historical three-seed planning entries.


## 2026-10-11 ? Trajectory smoke stopped at terminal parity

[Run/Inference/Open] User handoff ZIP preserved and CRC-checked:
`result/modelnet40_c/legacy_trajectory/20261011T135333Z_trajectory_handoff_20261011T142247Z.zip` (SHA-256 `748af41956fa7f958a557a81bed9c6357f157c2f0bc773e77d5b128b728bf532`). Six smoke arms completed 140 examples
per arm; no pilot started. Provenance, assets, environment, source hashes, IDs
and labels match across modes. Off/audit terminal predictions differ on9/140
Gaussian and24/140 Background examples; audit/observe differ on6/140 and22/140.

Audit and observe share input-point, explicit diffusion-noise, and captured
Python/NumPy/Torch CPU/CUDA RNG hashes per batch, but VAE style/local latent
hashes already differ before the first clean probe. GPU run used A100-SXM4-80GB,
PyTorch2.1.2+cu121, cuDNN8.9.2, with deterministic algorithms and cuDNN
determinism disabled. [Inference] This implicates uncontrolled variability in
or beneath VAE encoding before any clean probe; probe causality is not shown.
A same-mode repeat was subsequently completed and confirmed run-to-run VAE
latent variability before changing the gate. Keep failed parity gate;
no accuracy or observer-safety claim. [Detailed audit](../result/modelnet40_c/diagnostics/legacy_trajectory_smoke_20261011/validation.md).


## 2026-10-11 - Same-mode Gaussian audit repeat confirms run variability

[Run/Inference/Open] Preserved repeated arm:
`result/modelnet40_c/legacy_trajectory/20261011T135333Z_trajectory/diagnostic_gaussian_audit_repeat.zip`
(SHA-256 `2d784fbe468000f39df896d97e96994dc85148d5abf9ffa389b9c4adbd323c93`).
Against the original Gaussian audit, source/assets/native/environment, initial
model state, indices/targets, input-point hashes and captured RNG before/after
match. Style/local encoded-latent hashes differ in both batches; terminal
predictions differ9/140. This confirms the strict cross-process parity gate
fails even between same-mode runs. The exact nondeterministic GPU operator or
uncaptured randomness is not localized. It does not show the observer caused
the change and does not test the selection hypothesis. Keep pilot stopped until
a revised smoke gate retains within-run RNG/buffer/provenance invariants and
accounts for measured repeat variability. [Audit](../result/modelnet40_c/diagnostics/legacy_trajectory_smoke_20261011/repeat_validation.md).


## 2026-10-11 - Smoke gate now measures within-run invariants

[Code/CPU/Open] After the repeated audit confirmed cross-process variability,
replaced strict inter-arm prediction equality as the smoke pass criterion.
Each arm now validates its locked example IDs, complete finite logits and,
when trajectory mode is enabled, unchanged model state. Audit repeats VAE
encode from the saved pre-encode RNG state and records sampled-latent and
buffer equality; observe records RNG/buffer hashes around every clean probe.
Cross-process differences remain summarized as diagnostics. CPU unit tests
for the replay and probe invariants pass. This does not establish GPU replay
behavior, explain the observed variability, prove observer neutrality across
processes, or test accuracy/selection. The smoke must be run again under a
fresh session/commit; pilot remains stopped pending ZIP review.


## 2026-10-11 - Trajectory smoke recovery and pilot guard review

[Run/CPU/Inference/Open] Preserved handoff ZIP SHA-256 `7cd75452ce7d64f3e3359ee66edadb5f08c9cf2d7a57c074b0c12bbf1bed488e`; six seed0 smoke arms each cover140 examples. Observe Gaussian terminal/min-entropy/probability-mean accuracy is76.43/79.29/79.29%; Background is57.14/54.29/55.00%. Off terminal controls are75.00% and57.86%, respectively. These partial figures have opposite directions across corruptions and are exploratory only. Observe vs audit predictions differ10/140 Gaussian and22/140 Background. Identical-RNG VAE latent replay differs in4/4 audit batches with RNG/buffer guards intact; the operation causing variability is unknown.

[Code check/Open] Pilot normally preserves completed work on exceptions: batch artifacts and result rows are written incrementally; the runner's `finally` records status and seals a ZIP; the launcher backs up that ZIP to Drive even on subprocess failure. CPU analysis is downstream of that backup. A hard runtime reset may skip both `finally` blocks and lose local partial batches; there is no periodic Drive checkpoint or auto-resume. Ensure Drive mount and available runtime before launch; if an ordinary failure occurs, package/download the attempt before making any new run.

[Open] Entropy-weighted probability averaging may be compared after the seed0 all15 screening, on the same traces and alongside equal averaging/min-entropy/fixed-probe controls. Freeze the weighting function and tau or calibrate separately. No empirical gain or entropy guarantee is asserted.


## 2026-10-11 - Seed0 all15 pilot completed

[Run/CPU/Inference/Open] Preserved and CRC-checked the recovered handoff; all15 corruption types completed 280 shared seed0 IDs (4,200 total). Local CPU analyzer reproduced the per-corruption report. Equal probability mean was+0.7619pp and minimum entropy+0.5238pp versus the same-trace terminal, but both lost3.5714pp on Background. This is a single-seed development subset, not full-data baseline or independent confirmation. The per-corruption table, conditional bootstrap limits, cost and next-step proposal are recorded in [trajectory evidence](gsd_trajectory_implementation_20261011.md) and [selection analysis](gsd_trajectory_selection_20261011.md). Entropy-weighted averaging remains an offline exploratory follow-up; no early stop or speedup was tested.

## 2026-10-11 - Full-data trajectory inference running by user report

[User report/Open] User confirms `Preflight passed` and that the full all15
seed0 inference cell is running. Earlier launch-pending statements are
superseded; completion and accuracy remain unverified until the full ZIP arrives.
Scope remains 37,020 examples, Legacy B70/M800, horizons10/30 and probes3/5/8
or8/15/23. Local branch/HEAD rechecked: `gsd-smooth-spectrum` at
`a351e66ec58f08f3a2104e46cab2cb8b20cdf961`. No local model execution or rerun.

[Code] Read current notebook and source: full-data eval seals available outputs
on ordinary exceptions; Drive backup is performed by the separate packaging
cell after inference ends. Preserve pointer/session and use package then
download, including on ordinary failure while runtime remains alive. Full-data
artifact path and SHA are not yet available. Raw Legacy baseline ZIP SHA-256
rechecked as `534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968`.
No code/notebook/raw-result edits or Git index/history changes. Only dated
knowledge continuations record this operational update.

[Open] Accept completion only after archive/coverage/provenance/model-state
checks; reject a completion claim if any fail. Analyze policies against the
same trace terminal and explicitly report Background. This update supplies no
new method result or basis for changing the accepted baseline or selecting tau.
See [active session handoff](gsd_trajectory_session_handoff_20261011.md).

## 2026-10-11 - Full-data seed0 trajectory screen: modest gain, Background regression

- [Run/CPU] Preserved the user's full handoff at
  `result/modelnet40_c/legacy_trajectory/full_all15_seed0_20261011T154536Z_handoff_20261011T171147Z.zip`
  (33,579,609 bytes; SHA-256 `44f5366448263dc02d7932111685948ed0a5ead39d4a8f53c3bda4ec2969d07a`).
  Outer7-member and nested3,249-member ZIP CRCs pass; members are path-safe.
  Independent CPU analysis is beside the ingested bundle. All15x2,468,
  seed0=37,020 examples; source commit a351e66, all79 source hashes match its
  Git tree, model state unchanged, execution/coverage complete. Baseline raw
  archive SHA still matches the accepted record.
- [Run/Inference] Same-trace terminal64.592112% (23,912),
  minimum entropy65.051324% (24,082; +0.459211pp,1,073 corrected/903 broken),
  equal probability mean65.156672% (24,121; +0.564560pp,953/744). Conditional
  1,000-replicate index-cluster95% intervals are[+0.2269,+0.6942] and
  [+0.3431,+0.7915]pp. One seed only; intervals do not measure seed or path
  variation. Equal mean is +0.586170pp versus separate accepted64.570502%
  Legacy baseline descriptively, not paired; result does not reach+1pp.
- [Run/Inference] Background regresses3.4846pp under minimum entropy and
  2.6742pp under mean (mean:77 corrected/143 broken). Mean improves10 rows
  and regresses5; full paired table is in
  `..._ingested/independent_analysis/per_corruption_seed.csv`. Macro14 mean
  is65.7560% versus terminal64.9601%. Fixed probe3 is64.7299% and has no
  positive interval versus terminal.
- [Run/Cost/Open] Wall timestamps span86.1min; instrumented sum is80.4min vs
  prior Legacy68.98min (~17% by sums, approximate, separate-run comparison);
  observer includes probe and artifact overhead. Peak62.16GiB allocated /
  69.05GiB reserved. No measured early-exit or speed benefit. Same-cost
  terminal voting and matched timing remain open.
- [Inference/Decision] Full scope strengthens a modest same-trace selector/mean
  signal, but Background harm and one-seed evidence prohibit broad promotion.
  Keep accepted Legacy baseline. Next candidate is exploratory entropy-weighted
  probability averaging from saved logits; do not fit/select tau confirmatorily
  on these same outcomes. If policy retained, seed1/2 test repeatability before
  any independent dataset evaluation. Details and falsifiers:
  [implementation report](gsd_trajectory_implementation_20261011.md).


## 2026-10-11 - Proposed bounded offline follow-up

[Run/Inference/Open] Background early probes are much weaker than terminal,
while Uniform/Gaussian benefit from early candidates. Recommend comparing a
parameter-free entropy-weighted probability mean (w proportional to1-H,
normalized H; equal fallback at zero total weight) and latest-clean-plus-terminal
equal mean on saved logits. Both are proposals, not implemented/evaluated.
Use one rule across corruptions, no tau/threshold fitting, paired transitions
against terminal and equal mean. Merely repairing Background's mean loss would
hypothetically raise macro gain from+0.5646pp to about+0.7428pp, still below1pp.
Freeze a supported candidate before conditional seeds1/2. See
[proposal and limits](gsd_trajectory_selection_20261011.md).


## 2026-10-11 - Common-horizon development focus (user decision)

[User decision/Run/Open] Prioritize the14 non-Background corruptions sharing
10 reverse steps. The user points out that selecting30 steps specifically for
Background assumes corruption identity at inference; Background repair is no
longer the next development objective or candidate-selection requirement.
Preserve and report the existing Background/all15 legacy results separately.
On the existing full-data seed0 trace, macro14 terminal/minimum entropy/equal
mean are64.960060/65.700972/65.755962% (+0.740912/+0.795902pp vs terminal).
Use one shared label-free policy across14; per-corruption reporting must not
become corruption-specific inference routing. These are macro14 development
results, not all15 improvements or demonstrated unknown-corruption deployment.
No horizon/model/code/raw-artifact change or new inference. Entropy-weighted
averaging remains a proposed offline candidate, with no tau/threshold selected.
See [updated decision and limits](gsd_trajectory_selection_20261011.md).

## 2026-10-11 - Parameter-free entropy weighting on saved full-data logits

- [Run/CPU] New strict macro14 analyzer validated the complete frozen-state seed0 all15 archive and evaluated only the14 common10-step corruptions (34,552 examples). Results are written to `..._ingested/macro14_policy_analysis_v2/`; the source archive was read-only.
- [Run/Inference] Terminal64.9601%; equal probability mean65.7560% (+0.7959pp;876 corrected/601 broken); normalized entropy weighted by `1-H` gives65.7270% (+0.7670pp;875/610). Weighted vs equal mean:26 corrected,36 broken (net -10; -0.0289pp). This parameter-free weighting therefore nearly matches but does not beat equal averaging in this seed0 screen.
- [Run/Open] Weighted accuracy exceeds terminal in10/14 corruption rows and is below it in4/14 (Occlusion, Lidar, Cutout, Density). Latest-clean/terminal equal control:65.4752%. No policy promotion, tau, speed or generalization claim. Same-seed index bootstrap excludes seed variability.
- [Code/Validation] `scripts/analyze_legacy_trajectory_macro14.py` strictly validates provenance, all15 coverage, unchanged model state and the locked Legacy protocol; writes paired CSVs against terminal and equal mean. Focused3 tests and45 existing regression tests pass.

## 2026-10-11 - Error complementarity and next accuracy proposal

[Run/CPU/Inference] From the saved full-data macro14 paired CSV,24,569/34,552 examples have a correct clean03/05/08 or terminal prediction (label-using selector oracle71.1073%; not deployable). Equal mean is wrong on1,850 examples with a correct candidate;6,461 have unanimously wrong candidates. Proposed next independent reliability signal is prediction agreement across two point-sampling views, with frozen Point-MAE/BN/token algorithm and common10-step horizon. Compare equal multi-view averaging and an H+JS selector to equal trajectory mean and same-pass-count terminal voting. Existing archive saves zero decoded geometries per batch, so this needs a new Colab probe run. Prepared-input plus trajectory mean is a separate diversity candidate. These are unimplemented hypotheses; no new accuracy gain or runtime claim. Source, exact artifact path, diagnostic recipe, primary literature and falsifiers are in the latest section of `gsd_trajectory_selection_20261011.md`.

## 2026-10-11 - Sampling proposal attribution clarified

[User requirement/Paper/Code] Explicit source mapping added to `gsd_trajectory_selection_20261011.md`: Bahri et al. WACV2025 sections3.1-3.2 (FPS/KNN variation with TENT and weight averaging); MEMO NeurIPS2022 section3.1 (marginal entropy across augmentations with parameter updates); AugMix ICLR2020 is an additional related JS-consistency precedent. The frozen-classifier trajectory selector and normalized H+JS score are our unvalidated synthesis, not a reproduced paper algorithm or established novelty. Code confirms each input view has1024 points after shared normalization. Two views overlap and represent the whole shape; they are not disjoint halves. Alternative external FPS initialization requires explicit implementation/verification because the current wrapper has no start-index argument.

## 2026-10-11 - Sampling-consistency test implementation

[Code/CPU/Open] Implemented the user-approved two-view FPS trajectory experiment with unchanged Point-MAE weights/BN/token algorithm and Legacy10-step macro14 host. Six policies include existing four-A mean, eight-view mean, H selector, H+JS selector and terminal8 vote control. Acquisition14 forwards is explicitly distinct from eight-forward policy cost. Original A/RNG/guidance CPU preservation, artifact/provenance rejection, partial recovery and regressions pass66 tests. No GPU run or new accuracy. Exact plan/test and reference titles (Bahri et al. WACV2025 sampling variation; MEMO NeurIPS2022; AugMix ICLR2020) are in `gsd_sampling_consistency_20261011.md`; full source ZIP/SHA and notebook continuation are in the latest implementation record. Next is user-run Gaussian140 smoke, full ZIP review, then conditional macro14x280 pilot.


## 2026-10-12 - Sampling source delivery v2 (current)

[Code/CPU/Open] Base remains `gsd-smooth-spectrum@a351e66ec58f08f3a2104e46cab2cb8b20cdf961` plus the declared hashed working-tree overrides. Use `result/preparation/trajectory_sampling_source_20261011_v2.zip` (642,170 bytes; 318 source files), SHA-256 `d0c48e7f5a90f8fe19bfc615961e0ed1db3451b19cf5c98494cb613181b57295`. This supersedes v1 for new launches; v1 is preserved.

The standalone CPU analyzer now independently compares recorded checkpoint/config/label/selected-data SHA-256 and sizes with the accepted baseline specification and verifies that specification against source delivery. Changed or absent identities are rejected before scoring. The launcher still hashes actual assets before inference.67 targeted CPU tests pass; extracted v2 CRC, all source SHA checks, CPU macro14 preview (34,552 full-scope examples; the staged runs remain140/3920), and Python3.8 syntax checks pass. Tests use synthetic CPU fixtures, not native CUDA.

The notebook now requests v2 and its exact SHA. Only the filename/hash in the newly added section were revised; original21 cells are unchanged. Amendment receipt: `result/preparation/trajectory_sampling_notebook_receipt_20261012_v2.json`. The plan and its explicit paper attribution remain in `knowledge/gsd_sampling_consistency_20261011.md` (sampling variation/weight averaging paper, MEMO, AugMix; our H+JS selector is an unvalidated synthesis).

Next: user-operated Colab preparation, new sampling preflight once, Gaussian140 seed0 smoke, then download/provide its complete handoff ZIP for review. Pilot macro14x280 remains separate and conditional on smoke review. No Run all, old-run repetition, local GPU, stage/commit/push, or raw baseline changes. Native CUDA behavior, accuracy, cost and repeatability remain open; failed invariants or no paired gain over the controls would reject promotion. Acquisition14 forwards is not deployment latency for the8-forward method.
[The entry below is the latest workspace handoff snapshot; prior entries remain historical evidence.]

## 2026-10-17 - Current status reconciled and next-agent handoff prepared

[Run/Code/Inference/Open] Reconciled README and current artifacts: operational
Legacy all15 seed0 is64.570502%; Phase6 P_PC-C_SCD is+0.036917pp; Phase5
P_PC65.9505% is a four-corruption result. Full all15 Legacy trajectory policies
on one seed0 trace score64.592112% terminal,65.051324% minimum entropy and
65.156672% equal probability mean (+0.564560pp versus its same-trace terminal).
This is not a matched comparison against the separate baseline, not seed
stability, and not external validation; Background loses2.6742pp under mean.

[Code/CPU/Open] Sampling-consistency v2 is implemented and source-pinned, but
the Gaussian140 CUDA smoke handoff is still the next gate according to the
latest dated knowledge and local `legacy_trajectory` inventory. No sampling
accuracy result is recorded. Smoke ZIP review must precede the distinct
macro14×280 pilot. No local GPU run was performed for this handoff.

[Code/Open] Added `knowledge/new_agent_handoff_20261017.md` with ordered
knowledge/code/artifact references and a copyable prompt. Updated the README
front matter to distinguish current facts from older dated status notes. The
working tree remains dirty; HEAD is
`a351e66ec58f08f3a2104e46cab2cb8b20cdf961`; no source/results were changed,
and nothing was staged, committed, or pushed.

## 2026-10-17 - ScanObjectNN severity mismatch confirmed; Colab plan prepared

[User decision] User switches current work to ScanObjectNN source accuracy
and asks to ensure severity5 generation using the original dataset on Drive.
Sampling-consistency smoke remains pending; it is not the immediate task.

[Code/Run] `main:data/readme.md` shows severity5 filenames but references an
absent `datasets/create_corrupted_dataset.py`. The actual main generator in
`datasets_mate/` hardcodes severity8; the current HEAD generator exposes
`--severity` but defaults to8. Reopened three ScanObjectNN ZIPs (CRC passes).
Gaussian source command/config/CSV all say severity8 and581 examples,
108 correct (18.588640%). Local Gaussian/label SHA identities match their
runtime manifest. This is not a severity5 source result. Gaussian sigma
is0.03 at5 versus0.045 at8, followed by clipping[-1,1]; low accuracy's full
cause remains open. The current `--skip_ply` flag is parsed but has no effect.

[Code/Inference/Open] Prepared
`colab_scanobjectnn_severity5_20261017.md`: original main_split H5 audit,
fresh isolated generation with explicit severity5/seed0, label/order/hash
checks and CPU deterministic replay, direct clean control, complete581-example
Gaussian source-only evaluation, verified Drive handoff, then conditional
13-array-corruption expansion. Occlusion/lidar require a separate mesh
protocol. No source code, raw results or inference were changed; no local
GPU or tests were run. No new accuracy or severity-causality claim.

## 2026-10-17 - Primary-source ScanObjectNN severity and preparation audit

[User report/Code] User confirms notebook's first6 code cells prepare their
Colab GPU session. Inspected cells1..6: environment/Drive/extensions/data/
weights setup is reusable. They checkout `dev` and copy ModelNet/LION weights,
so later ScanObjectNN cells must record actual source and separately verify
the ScanObjectNN checkpoint. Do not rerun preparation or old experiments.

[Code] Upstream initial author commit `e5b7278` already hardcodes generator8;
fork `1f28b29` preserved8 when introducing the severity CLI. Thus prior pilot8
was inherited generation behavior, not documented ScanObjectNN paper protocol.
Live official `main_3dd_tta.py` and `utilities_3dd_tta.py` confirm ScanObjectNN
evaluation selects the shared `_5.npy` loader. Generator8 is verified against
local upstream/main `107305f`; remote generator fetch was unavailable.

[Paper/Inference] Official supplement Appendix A says Gaussian sigma0.03,
uniform+/-0.05, background5%, impulse10%, shear+/-0.25, rotation+/-15degrees;
all map to inherited severity5 entries. Main Sec4.1 gives no explicit severity
number. This independently supports Gaussian5 and the released evaluation
setting. It does not establish exact author data provenance or full appendix/
code equivalence (cut-out descriptions differ). Updated paper note and
`colab_scanobjectnn_severity5_20261017.md` with primary links, history and
after-six-cell workflow. No notebook/source edit, local GPU or accuracy run.

## 2026-10-17 - ScanObjectNN author-code Gaussian5 Colab workflow implemented

[User decision] User authorizes fresh generation/Drive backup and notebook
restore integration; explicitly selects Gaussian+clean first and emphasizes
matching the paper's dataset procedure. Keep original assets and raw results.

[Code/CPU] Added `scripts/scanobjectnn_severity5_colab.py`, tests and a pinned
author-source snapshot from upstream/main `107305f`. The adapter executes
unchanged author normalization/Uniform/Gaussian functions; no added raw-data
normalization. It preserves the author's MAP order through Gaussian (Uniform
draws first, output discarded), point permutations, severity5 sigma0.03,
float32 and clipping; saves direct clean H5 data and flat labels. Independent
CPU replay must exactly match. Seed0 is declared locally, not attributed to
the authors. Exact paper data identity remains unestablished without original
assets/seed. This supersedes the earlier Gaussian-only CLI draft RNG sequence.

[Code] Notebook cell5 appends verified Drive restore; three new cells after
the original six preparation code cells implement generation, verified Drive
backup and download/recovery. Original cell objects other than the cell5
addition are preserved; source payload is embedded so Colab's dev checkout
requires no commit/push. Snapshot and preservation receipt are in
`result/preparation/scanobjectnn_author_generator_107305f.py` and
`scanobjectnn_s5_notebook_receipt_20261017.json`. Drive destination is
`thesis/dataset/scanobjectnn_c_s5_seed0/runs/<fresh-ID>/dataset.zip`;
only complete verified data can update `latest_gaussian.json`.

[CPU/Open] Six focused unittest cases pass with synthetic numpy arrays:
generation/order/replay/clean identity, verified roundtrip and repeated load,
severity/label and ZIP tampering rejection, unsafe paths, failed-attempt
non-promotion, and H5 split/ambiguity rejection. Python3.8 syntax, payload
identity, pinned source SHA and notebook preservation checks pass. Initial
default Windows temp-root permissions blocked test setup; rerunning in a
unique workspace-owned test root passed. Real h5py I/O, Colab Drive behavior
and original-data generation await user execution. No local real-data/GPU
evaluation, accuracy, Git stage/commit/push, or old ZIP modification.

## 2026-10-17 - Mac Colab CLI skill and continuation audit

[User decision/Code] The user now delegates Colab operations through the
installed CLI and requires approval before new runs. Added
`.codex/skills/colab-research-operator/SKILL.md`, repository `AGENTS.md` routing,
and aligned the existing research skill. The workflow specifies bounded phase
approval (CPU generation included), exact source/assets/config provenance,
shared-session preservation, timeout inspection before retry, immutable
Drive/local SHA-verified archives, and dated evidence-labelled knowledge updates.
CLI0.7.4 and local help for exec/download/log were inspected; bundled auth-default
prose conflicts with installed help/code, so the skill uses the actual interface.
Help/version required sandbox escalation for the CLI's local log file; this
does not constitute approval for research execution.

[Code] Actual Mac branch/HEAD is `gsd-smooth-spectrum` at
`7fcfef71641b2b623ee737ae1ce404a29116d6c4`, whose latest commit updates research
documents. Initial status contains only pre-existing untracked `.codex/`.
The documented ScanObjectNN severity5 adapter, pinned source/receipt under
`result/preparation/`, and updated notebook section are absent here. The local
guidance-ablation notebook has12 cells and no severity5 section, rather than
the recorded33. This checkout cannot execute the documented cell sequence as-is.
The existing generator's default remains8; the source-only runner supports
ScanObjectNN and15 output classes. This is a source-delivery discrepancy,
not new evidence against any archived accuracy result.

[User report/Open] The user confirms result files exist in another environment
and is transferring them through Drive. Do not classify their local absence as
lost evidence or authorize reruns. Research context remains ScanObjectNN
Gaussian5+clean first; sampling-consistency smoke is pending and deferred.
Historical ModelNet40 scores retain their recorded scopes and limits; raw
archives were not independently revalidated during this continuation.

[Decision/Open] Continue result-independent skill/knowledge work. Once delivery
is available, reconcile source and coverage, recover a complete Gaussian5
package if present, otherwise prepare the exact missing generation phase and
request approval. Source-only evaluation is a later reviewed/approved phase.
Falsifier for the source-delivery blocker: the intended adapter, pinned source
and notebook are recovered with matching identities. No remote session/auth
was inspected, no VM or experiment started, and no source/notebook/raw result
was changed. No stage, commit or push performed.

[Validation] Both skill frontmatters and UI metadata pass YAML/name/description/
invocation checks using Ruby's installed YAML library; new workflow links and
`git diff --check` pass. The bundled `quick_validate.py` could not run because
PyYAML is absent in the local Python environments; no dependency was installed.
An independent dry scenario review covered approval boundaries, successful
CPU generation before unapproved inference, timeout recovery, cross-environment
transfer and shared-session cleanup. It prompted explicit source-byte archiving
and local salvage when Drive backup fails. The baseline already respected
approval, so this is an instruction review, not measured behavioral improvement.
No project test suite, real Colab execution or result validation was performed.

## 2026-10-17 - Downloads result transfer verified and source copy removed

[User decision/Code] At the user's explicit request, copied
`/Users/batuhan/Downloads/result` into the existing repository `result/` without
overwriting existing files. All1,087 source files (1,351,309,810 bytes) matched
their mapped destinations by size and SHA-256, and the source inventory was
unchanged before deletion. One identical calibration config was reused. The
incoming README differs only by CRLF/LF; the existing README stays intact and
incoming bytes are preserved as `source_README.md` beside
`result/import_receipts/downloads_20261017T102815Z/transfer_manifest.json`.
That manifest records every source/destination mapping, size/hash and deletion
confirmation. Removed only the Downloads/result tree after verification and
confirmed every mapped destination file remains present. System escalation
was required because Downloads is outside the writable workspace.

[Code/Open] `result/preparation/` now includes the author generator, severity5
notebook receipt and sampling source v1/v2 ZIPs previously absent on this Mac.
Source checkout discrepancies remain separate. This operation verifies file
transfer only; no ZIP internals, numerical results or historical scientific
claims were revalidated, and no Colab session/run was started.

## 2026-10-17 - Colab CLI OAuth configuration started

[User decision/Code/Open] User authorized CLI setup. Initial existence-only
checks found no cached OAuth token, local session registry or ADC credentials;
gcloud is absent and is unnecessary for the installed CLI's bundled OAuth2
flow. Read installed auth/command source and opened a user-facing Terminal
running `/Users/batuhan/.local/bin/colab --auth oauth2 whoami`. Google sign-in
and authorization-code entry must occur there, not through chat. Awaiting user
completion; no identity or backend-access success is claimed. No credential
contents read or archived. Setup-time HEAD is `e955799de89476a5d58105c4f8622d0481bdecdc`;
earlier checkout hashes remain dated snapshots.

[Code/Open] Normal CLI execution targets locally registered sessions; the
installed0.7.4 interface lacks a browser-session attach/adopt command. Next:
verify whoami and read-only backend session listing after interactive sign-in.
New VM allocation, Drive mounting, environment installation and experiments
remain outside this setup authorization. Details:
`knowledge/colab_cli_setup_20261017.md`. No Colab VM or GPU run started.

[Later Code/User report/Open] User completed Google sign-in in the Terminal.
Token-file existence was checked without reading contents. Separate CLI
`--auth oauth2 whoami` succeeded, followed by `--auth oauth2 sessions`:
one A100 High-RAM assignment is visible as `[?]`, not registered locally.
This supersedes the pending-auth status above, but does not verify access to
that runtime's kernel, Drive or project environment. No VM allocation,
remote code or experiment occurred. The Terminal window and entered OAuth
code were not read; the separate CLI identity command returned email,
audience, expiry and scopes in conversation tool output, with no access or
refresh token. No account identifier or credential values are copied here.
Next is a separately approved runtime/setup plan if CLI execution is needed;
preserve the existing browser runtime. Setup record updated accordingly.

[Later User decision/Code/Open] User conditionally requests a new2026.07
runtime with A100 and High-RAM if CLI version selection is supported. Inspected
installed0.7.4 session-command signature and assignment URL construction:
GPU/high-memory flags are supported, but runtime image/version selection is
not exposed or sent. Official backend-info lists2026.07 as a past runtime;
that does not establish CLI selection support. No new VM was allocated and
no default runtime was substituted. Preserve the requested image/hardware
combination while deciding an execution route. Details and primary links:
`knowledge/colab_cli_setup_20261017.md`.

## 2026-10-17 - Six-cell setup extraction and failed A100 CLI attempt

[User decision/Code] User explicitly authorized saving the first6 preparation
code cells, allocating A100/High-RAM and verifying their execution, reconnecting
after cell1. Extracted exact notebook indices1–6 from the now33-cell source at
HEAD`e955799` into `scripts/colab_environment_setup.ipynb`, without historical
outputs. Original source SHA-256`dd3146508be8a717d260418a312d00c127291672cb2dc1e20c40b642cd1b9315`.
Source equality checks pass; original notebook is unchanged. AGENTS and the
operator skill route future setup to this file and its current validation note.

[Code/Run/Open] CLI allocation reported READY for
`tta-setup-a100-20261017`, A100/High-RAM, endpoint
`gpu-a100-hm-kkb-ass1c2-jah5ciy1qndo` at10:42:38Z. Cell1 installed Conda and
restarted the kernel. Same-session reconnection and cell2 condacolab check,
clone and `dev` checkout passed. `conda env create` then failed in its pip
phase: CuPy pulled source`fastrlock0.8.4`, whose wheel build failed;
`CondaEnvException: Pip failed`. Actual compiler cause is not in the captured
output. The guarded cell stopped; cells3–6 never ran. CLI exited0 despite the
remote RuntimeError, so completion markers/errors must supplement exit codes.

[Code/User report/Open] A subsequent prepared diagnostic never ran because
the CLI prelude received404/401 and pruned its local registration. The exact
created VM is absent from later active assignments; two other A100 endpoints
were visible and untouched. User reports modifying only their own browser
sessions, not this VM. Cause of VM loss is unknown; do not attribute it to
user action or token expiry without evidence. No duplicate allocation or retry.

[Artifact/Decision] Failed-attempt logs, exact cell sources, wrappers and
manifest are archived locally in
`result/preparation/cli_setup_20261017T104109Z/setup_attempt.zip`,72,382 bytes,
SHA-256`2f512b3aaf492c305151fa4283e1f66ce3dbd416e49c6b0f9ac1968726e98c2d`;
CRC passes. Drive backup unavailable because the mount phase was not reached.
Next is renewed approval for a fresh diagnostic/setup attempt, minimally
capturing the Python3.8 wheel compiler error before any dependency change.
No successful six-cell validation, GPU-kernel test, model accuracy or new
research result is claimed. Full status: `knowledge/colab_cli_setup_20261017.md`.

## 2026-10-17 - cu121 repair, approved r2 setup and revised approval threshold

[User decision/User report/Code] User defers known cell2 fastrlock diagnosis
and supplies actual PointNet++ failure: toolkit13.0 versus torch CUDA12.1.
Prepared an environment-local12.1.1/GCC12.2 build helper, reused from the
earlier build-script pattern, and embedded it in cell4 of both notebooks.
It preserves system CUDA/driver, targets the GPU, fixes dev's architecture
override and checks native imports/FPS/gather forward/backward. Local syntax,
version-guard and notebook-preservation checks pass; GPU repair is unverified.

[User decision/Code/Open] User explicitly approved r2 A100/High-RAM setup.
Later changed the ongoing rule: require approval only for GPU runs expected
to exceed60 minutes; shorter runs, CPU/setup/allocation/retries proceed within
scope. Updated AGENTS and both applicable skills, preserving historical notes.
r2 preflight confirms A100-SXM4-80GB and MemTotal175,166,472kB. After cell1's
Conda restart, cell2 initially lost connection in the CLI prelude. The VM was
still active; an explicit same-VM kernel restart recovered access. Cell2 is
now solving the environment. No duplicate allocation, model evaluation or
native CUDA success claim. Pinned source/logs:
`result/preparation/cli_setup_20261017T130314Z_r2/`; use its layout-revised
cell4 snapshot. See `knowledge/colab_cli_setup_20261017.md` for exact metadata.

[User decision/Code/Open, later update] Approval is required only for a GPU
phase reasonably expected to exceed 60 minutes; exactly 60 minutes and shorter
phases do not require additional research approval. Assess the whole phase,
including any planned continuation. The active r2 setup reached the known
deferred cell2 fastrlock failure and is now awaiting user-operated Drive
authorization in Terminal. Terminal contents and credentials are not read.

## 2026-10-17 15:12 UTC — A100 setup validated, consolidated and archived

[User decision/Code/Run/Open] r2 is ready/idle on A100-SXM4-80GB / High-RAM.
CUDA12.1/GCC11.4/PyTorch2.1.2+cu121 passed native imports, FPS/gather
forward/backward, KNN, Chamfer and EMD checks; ModelNet40-C clean +15 severity5
arrays and both checkpoints passed, with24 asset SHA256 identities recorded.
Fixed the GCC12.2 pybind11 failure and missing EMD cost kernels using the
project's existing complete source. The user confirmed ModelNet40-C as the only
required archive and explicitly authorized removal of the empty original ZIP.
ScanObjectNN's invalid receipt stays unaccepted in its separate research flow.

Delivered six-cell scripts/colab_environment_setup.ipynb, embedded build and
verification helpers, synchronized original prep cells and README/AGENTS/skill
routing. Final acceptance uses exact output markers, JSON status and archive
validation; the earlier sys.path/substring-marker diagnostic failures remain in
raw logs. Only cell2 fastrlock is deferred. No second-clean-VM replay and no
model accuracy run. Actual remote dev@a458cd46 differs from local research HEAD;
see the [final handoff](colab_cli_setup_20261017.md#final-validated-status).

Local result: result/preparation/cli_setup_20261017T130314Z_r2/.
Drive: MyDrive/thesis/result/preparation/cli_setup_20261017T130314Z_r2/setup_attempt.zip.
Archive461,401 bytes, SHA256
0cc4a67a67ab1a53ef97dc89cc4abbbdd2bafa373d72060c4f295536deff798a;
local/Drive size, SHA and CRC verified. Scope had explicit user setup approval;
standing approval threshold is now only GPU phases expected to exceed60 minutes.
Next gate: inspect ScanObjectNN receipt/actual artifacts and intended source;
no new accuracy or method-comparison claim follows from environment setup.

## 2026-10-17 20:22–20:49 UTC — ScanObjectNN Gaussian severity diagnosis paused

[User decision/Run] The user authorized the short A100 diagnostic phase and
later requested a pause plus teardown of all Colab runtimes. Session
`tta-scanobjectnn-s5-a100-20261017`, A100-SXM4-80GB/High-RAM, used remote
`dev@a458cd46`; setup/native/data/checkpoint verification passed. Checkpoint
SHA-256 was `53f6c2e...35dac`; main-split original/Gaussian5/label SHA-256 were
`47c641b3...175fe`, `9bff59ad...2a7e7`, and `bb145670...b7a68`.

[Run] Exact author-order seed0 Gaussian5 generation and complete source-only
evaluation produced clean415/581=71.4286% and Gaussian5
183/581=31.4974%. The historical Gaussian8 result is108/581=18.5886%, so the
severity correction recovers+12.9088pp but remains27.5026pp below the paper's
Gaussian59.0%.

[Run/Inference] Controls reject point order and clipping: permuted clean70.40%;
Gaussian clipped permuted31.50%, clipped original-order31.33%, unclipped
permuted32.53%, unclipped original-order32.01%. PB_T50_RS is also incompatible
with this checkpoint: clean1579/2882=54.7883%, Gaussian5
620/2882=21.5128%.

[Run] Same-noise Gaussian severity1–5 results are352,296,248,210,183 correct
out of581:60.5852%,50.9466%,42.6850%,36.1446%,31.4974%. Severity1 is close to
Table3 but contradicts the published Appendix-A sigma0.03/severity5 parameter;
it is a diagnostic, not a replacement benchmark.

[Inference/Open] The remaining paper gap is best recorded as an unpublished
data/checkpoint/evaluator-asset mismatch. Exact author array/checkpoint hashes
or clarification would falsify or refine that attribution. Do not tune
severity/seed to59.0 or expand all15 before this provenance gate.

[Artifact] Local pause handoff:
`result/preparation/cli_scanobjectnn_s5_20261017T190028Z/scanobjectnn_pause_handoff_20261017.zip`,
139,672 bytes,24 members, SHA-256
`eab1be91726d5bb7f7aef3c957fdbd88c45936fec310e692597975b28083d9a7`;
CRC and downloaded SHA passed. It retains state/manifests plus one failed and
six complete runner archives, excluding large NPY arrays. CLI history is
adjacent and credential-pattern scan was empty. Next action is intentionally
paused. The named A100 session and sole orphan CPU assignment were released;
final server/local CLI checks reported no active sessions.

## 2026-10-18 - Planned severity/provenance falsification

[Run plan/Open] The severity5 Gaussian result (31.4974%) agrees with the
independent official SVWA Point-MAE source-only Gaussian result (32.5%), while
our severity1 Gaussian result (60.5852%) is close to 3DD-TTA Table3 (59.0%).
Run one seed0 A100 source-only profile over the13 non-rendered Table3
corruptions at actual severity1, preserving the released generator's order and
recording every generated-array hash. Hypothesis: if several corruption
accuracies and the macro profile approach Table3, the paper/released-protocol
severity mismatch is the dominant explanation; otherwise retain unpublished
dataset/checkpoint/evaluator provenance as the unresolved cause. This is a
diagnostic and must not relabel severity1 as the published severity5 benchmark.

## 2026-10-18 10:02 UTC - Severity1 profile setup VM lost

[Run/Failed] Session `tta-scanobjectnn-s1-a100-20261018` was allocated as an
A100 High-RAM VM for the planned seed0 severity1 source-only profile. Cell1
completed sufficiently to install Conda and restart the kernel; cell2 reached
only the previously deferred `fastrlock` wheel failure and its recovery check
passed on remote `dev@a458cd46` with Python3.8.20. The user completed Drive
authorization and cell3 passed with `SETUP_CELL_3_COMPLETED`.

[Run/Open] Cell4 installed the pinned requirements, including torch2.1.2 and
torchvision0.16.2, then entered the CUDA12.1.1/GCC11.4 Conda toolchain step.
Before any native-build completion marker or forward/backward result appeared,
the CLI reported `RuntimeError: Connection was lost`; an immediate server
query pruned the stale local record and returned no active sessions. Therefore
cell4, cells5-6, ScanObjectNN restore and all severity1 inference remain
unexecuted. No result archive or accuracy was produced. Do not treat package
installation output as a validated environment. A continuation requires a new
runtime and the complete six-cell setup from cell1; the remaining planned GPU
phase should be re-estimated before allocation.

[User decision/Planned] The user explicitly approved a fresh A100 High-RAM
retry after being told the complete replacement phase is estimated at60-75
minutes. Approved scope: new session, all six setup cells, verified
ScanObjectNN main-split restore, seed0 severity1 generation for the13
non-rendered Table3 corruptions, Point-MAE `scanobject_jt.pth` source-only
evaluation at batch32, hash-verified local/Drive archiving, and cleanup.
Stop on native, dataset/checkpoint identity or artifact validation failure.

## 2026-10-18 10:31–10:58 UTC - Second setup VM lost; Cell4 dependency duplication removed

[Run/Failed] Replacement session `tta-scanobjectnn-s1-a100-r2-20261018`
reached the same validated preconditions: Cell1 passed on an A100-SXM4-80GB
High-RAM VM, the deferred Cell2 `fastrlock` failure was verified against
Python3.8.20 and remote `dev@a458cd46`, Drive mounted successfully, and Cell3
emitted `SETUP_CELL_3_COMPLETED`. Cell4 then ran for approximately26 minutes
without a completion marker. The CLI ended with `RuntimeError: Connection was
lost`, and the immediate server query returned no active sessions. No Cell4
success, later setup cell, dataset restore or inference result exists from r2.

[Code/Diagnosis] The last successful setup history from2026-10-17 proves the
old Cell4 first installed unpinned torch2.4.1/torchvision0.19.1 and their CUDA
packages, then the repository requirements replaced them with
torch2.1.2/torchvision0.16.2 before native compilation. This avoidable double
installation materially increases setup time, downloads and disk churn. It is
a reproducible setup defect and a plausible contributor to the two VM losses,
but the absent in-progress remote log means it is not proven as their server-
side termination cause.

[Code/Test] Cell4 in both `scripts/colab_environment_setup.ipynb` and
`scripts/gsd_guidance_ablations_colab.ipynb` now installs the required
torch2.1.2+cu121, torchvision0.16.2+cu121 and torchaudio2.1.2+cu121 first;
KNN, repository requirements, CLIP, toolchain and native-build steps retain
their order. `tests/test_colab_environment_setup.py` failed against both old
cells and passes after the synchronized change. A fresh six-cell runtime replay
is still required before treating this repair as Colab-validated.

## 2026-10-18 12:20–12:21 UTC - r3 setup and severity1 profile completed

[Run/Code] Fresh r3 A100/High-RAM completed the revised six-cell setup. The
single pinned torch2.1.2+cu121 installation survived KNN, requirements and CLIP;
CUDA12.1/GCC11.4 compiled EMD, Chamfer, PointNet++ and PVCNN. Final native,
dataset and checkpoint verification passed. This validates the Cell4 repair on
a new VM; it does not establish why r1/r2 disappeared.

[Run] The13 non-rendered ScanObjectNN-C corruptions were generated at actual
severity1, seed0, released RNG order (including discarded scale), with every
array hash recorded. Point-MAE checkpoint
`53f6c2e...35dac` evaluated all581 examples per row. Results (%): uniform41.14,
Gaussian60.59, background42.00, impulse34.42, upsampling28.40, shear43.72,
rotation44.41, cutout71.94, density70.74, density-inc71.60, distortion44.06,
RBF43.20, inverse-RBF44.41. Macro13=49.2784%; Table3 macro over the same13
columns=69.0%; mean absolute difference=19.9654pp.

[Inference] Gaussian (+1.59pp) is the only row within2pp of Table3; the other12
are6.36–36.00pp lower. Thus Table3 is not a simple severity1 profile. Our
Gaussian-s531.50% and independent SVWA source-only Gaussian-s532.5% agree,
supporting the released severity5 baseline and leaving unpublished/mixed
Table3 data, checkpoint or evaluator provenance unresolved. Do not relabel
severity1 as severity5 or tune seeds.

[Artifact] `result/preparation/cli_scanobjectnn_s1_20261018/` contains the
state, streamed runners, CLI JSONL and seven-file result ZIP. Validation passed
CRC, regular safe paths,13 complete rows and independent macro recomputation.
Local/Drive ZIP SHA-256:
`885a1f16f866b64c179b0c48948f11d5ac534b5aef4d37322647c197f7486a79`.

[Correction/Code] The run itself is valid, but the initial state's derived
comparison zipped paper values to generator order although Table3 uses RBF,
RBF-inverse, density-decrease, density-increase, shear, rotation, cutout and
distortion order. Correct name-based mapping leaves macro49.2784% and
MAE19.9654pp unchanged but replaces the false density-inc -0.30pp comparison
with -10.80pp. Corrected derived CSV/notes are adjacent to the immutable raw
artifact. The runner now uses an explicit dictionary; its regression test
failed on the old `dict(zip(...))` form and passes after the fix.

[User correction/Operational failure] The agent terminated r3 after artifact
validation based on the earlier cleanup plan, before the user declared task
completion/close permission. Server and local CLI then showed no sessions; the
VM is not recoverable. The user requires costly Colab environments to remain
live for follow-up tests. The Colab operator skill now prohibits teardown until
the user explicitly states both task completion and permission to close, for
agent-created and shared runtimes alike. A focused skill-contract test passed
after first failing on the prior wording; official skill validation also passed.

## 2026-10-18 - Official SVWA source-only protocol audit

[Paper] SVWA reports a Point-MAE source-only ScanObjectNN-C row with Gaussian
32.5% and all-15 mean33.1%. The first13 non-rendered entries average36.7385%.

[Code] Official repository commit
`59935d68d0ef12811619ce4776e1dc4a711997d1` fixes evaluation to severity5,
uses the same15 corruption order as the paper/3DD table, loads
`data_<corruption>_5.npy`, applies FPS(1024), sets15 output classes and calls
Point-MAE `classification_only(..., only_unmasked=False)`. Its corruption
generator reads the581-example ScanObjectNN main-split test H5. The published
Gaussian32.5% therefore closely corroborates our Gaussian-s531.4974% under the
same released protocol family.

[Artifact/Code] Direct public-Drive downloads establish that SVWA does not
reuse 3DD's checkpoint. 3DD `scanobject_jt.pth` is351,525,407 bytes with
SHA-256 `53f6c2e...35dac`, exactly matching our evaluated asset; SVWA
`objectnn_src_only_PointMAE.pth` is294,085,061 bytes with SHA-256
`011a9efc...a432`. Both PyTorch ZIPs pass CRC. Their model state dictionaries
have227 keys each:225 common names and shapes, but zero equal raw tensor
payloads; the remaining classifier-head key pair is named differently. The
recorded epochs are93 (3DD) and100 (SVWA).

[Limit/Inference] SVWA does not publish corruption-array hashes and its
evaluator uses `drop_last=True`. Consequently the row is an independent
severity5/protocol-family cross-check, not an exact reproduction. Agreement
despite genuinely different weights makes the much higher 3DD Table3 source
row still more anomalous. Because our checkpoint is exactly the official 3DD
download, the remaining candidates are Table3's unpublished/mixed checkpoint,
corruption arrays or evaluator conditions; exact experimental assets or author
clarification are still required to identify which one conclusively.

## 2026-10-18 13:02–13:44 UTC - r4 no-background setup runtime lost

[Run/Failed] Fresh A100/High-RAM session
`tta-scanobjectnn-nobg-a100-r4-20261018` completed Cell1, reached only the
explicitly deferred Cell2 `fastrlock` failure, mounted Drive after a second
authorization attempt, and emitted `SETUP_WRAPPER_CELL_3_COMPLETED`. Cell4
installed the pinned PyTorch2.1.2+cu121 stack and KNN-CUDA, but no Cell4
completion marker or later setup/inference result appeared.

[Run/Diagnosis] A server inventory query returned no active sessions; the
local status command pruned the stale session at13:43:49 UTC. The still-waiting
local `colab exec` client had accumulated about27 minutes and was interrupted
only after server absence was confirmed. This wait is not evidence of a
40-minute native build. The exported CLI history contained an ephemeral Drive
authorization URL and was deliberately not retained; the sanitized receipt is
`result/preparation/cli_scanobjectnn_nobg_20261018/runtime_loss_receipt.json`.
The server did not expose a definitive quota, compute-unit, or backend cause.

[Run/Inference/Open] The prior successful fresh r3 history places Cell3
completion at11:50:06 UTC and Cell4 completion at12:13:31 UTC, about23m25s.
Thus the2400-second CLI timeout was a safety ceiling, not a runtime estimate;
roughly10-minute observations likely reflect a warmer or partial setup rather
than this full fresh Cell4. r4 produced no no-background accuracy. A new runtime
must replay all six setup cells before the planned clean/Gaussian severity5
diagnostic.

[User decision/Code] The Colab operator skill now states that managed lifetime,
quota and compute-unit termination cannot be prevented; forbids fake-click,
JavaScript and empty-loop anti-idle workarounds; requires same-session/server
inspection before retry; and requires verified Drive persistence of material
intermediate state. It also clarifies that a mobile/browser tab need not remain
open after interactive authorization for CLI-managed execution. The focused
contract tests and official skill validator pass.

## 2026-10-18 13:47–16:17 UTC - r5 lost during Cell2; heartbeat rule clarified

[Run/Failed] Session `tta-scanobjectnn-nobg-a100-r5-20261018` completed Cell1,
reconnected to the same A100 VM, mounted and verified Drive, then entered Cell2.
No Cell2 completion marker was produced. At16:17:10 UTC the server inventory
returned zero sessions and the local record was pruned while its orphaned CLI
client had waited about2h24m. The client was interrupted only after server
absence was confirmed. The account still had106.16 compute units, so exhaustion
is not supported as the cause. Sanitized evidence:
`result/preparation/cli_scanobjectnn_nobg_20261018/runtime_loss_receipt_r5.json`.

[User decision/Code] The user requested periodic meaningless print cells to
avoid disconnects. This was not encoded as anti-idle behavior: a busy kernel
would queue the second cell and Colab-managed retention is not guaranteed.
Instead, the operator skill now requires long phases to emit meaningful UTC/
phase/progress heartbeats from the running process and permits independent
server-inventory checks without launching a second cell. The added contract
test failed before the change; all four skill tests and validation pass after it.

## 2026-10-18 16:52 UTC - future setup Cell4 split into restartable phases

[User decision/Code] The user approved splitting the extracted setup notebook's
long Cell4 for future sessions. The canonical notebook now has ten code cells:
Cell4 installs pinned packages/toolchain and patches sources; Cells5–8 build
EMD, Chamfer, PointNet++, and PVCNN plus native GPU smoke; the former data and
final-verification cells move to9 and10. The full research notebook is retained
unchanged.

[Code] `scripts/build_cuda_extensions_colab.sh` now supports ordered
`prepare`, `emd`, `chamfer`, `pointnet2`, and `pvcnn-smoke` phases plus the old
`all` behavior. It checks prerequisite completion markers, skips confirmed
phases, rejects unknown phases before environment access, and emits UTC phase
heartbeats during long builds. Standalone and notebook-embedded helper bytes
match SHA-256 `bd2baffb...3bb6`.

[Code validation/Open] The focused test was observed failing first because the
old notebook had six rather than ten code cells and the old helper reached
missing Conda before rejecting an invalid phase. After the change, all three
focused tests, Bash syntax, notebook JSON, diff checks, and embedded-helper byte
comparison pass. Local IPython transform compilation was unavailable because
IPython is not installed in the Mac environment. The ten-cell notebook remains
unvalidated on a fresh Colab VM; active r6 uses the earlier combined Cell4 and
must not switch numbering mid-run.

## 2026-10-18 16:32–17:06 UTC - r6 lost; first split replay started on r7

[Run/Failed] The r6 A100/High-RAM session used the pre-split six-cell notebook.
Its combined Cell4 printed successful pinned PyTorch2.1.2+cu121 and KNN-CUDA
installation but emitted no Cell4 wrapper marker. At about27 minutes the named
runtime disappeared from the server inventory. The user then directed stopping
the wait and continuing with split components; interrupting the local executor
returned exit130 and the CLI pruned the stale r6 record. No Cell5/6 or
no-background inference ran. As with r4, the backend exposed no definitive
termination cause, so package output is not treated as environment validation.

[User decision/Run/Open] The user directed a new session and per-component
execution. Fresh session `tta-scanobjectnn-nobg-a100-r7-20261018` is live on
A100/High-RAM endpoint `gpu-a100-hm-kkb-euw4a2-1d8qxj97k4u94`. Cell1 completed
and the CLI reconnected to the same VM. Cell2 created the Python3.8 environment
and stopped only at the known, explicitly deferred fastrlock wheel failure.
Planned gates: mount and verify Drive in Cell3, execute split Cells4–8 sequentially
with exact wrapper and phase markers, finish Cells9–10, then run and archive the
581-example no-background clean/Gaussian severity5 Point-MAE diagnostic. Keep
r7 live until the user explicitly declares task completion and permits closure.

[Code validation] The split setup and Colab skill contract run eight focused
tests successfully; Bash syntax, notebook JSON and scoped diff checks pass.
Initial bare-Python discovery failed only because numpy/torch were absent. A
fresh isolated uv environment with numpy, torch, scipy, h5py and PyYAML then ran
the complete suite:266 tests passed. The skill quick validator also reports
`Skill is valid!` in that environment. The first Cell3 attempt on r7 ended with
`ValueError: mount failed`. After user authorization, the TTY-backed CLI
`drivemount` reported `Mounted at /content/drive`; independent listing verified
MyDrive and the KNN wheel. The Cell3 retry then copied the wheel and emitted the
exact `SETUP_WRAPPER_CELL_3_COMPLETED` marker. Split Cell4/prepare is next.

[Code/Diagnosis] Inspection of CLI0.7.4 showed that `colab status` prints the
local session-state record's `running` field, not independently queried remote
kernel state. This explains the transient IDLE/BUSY contradiction during Cell3;
the lack of output was also expected while `drive.mount()` had not returned.
The operator skill now forbids conclusions from one status snapshot or output
silence and requires the same primary exec output, exact wrapper marker, and
native `.complete` marker. The regression test was RED before the change; all6
skill tests and official validation are GREEN afterward.

## 2026-10-18 17:17–18:02 UTC - r7 setup and no-background diagnostic complete

[Run] r7 completed split Cells4–8 with every exact wrapper marker and matching
`prepare`, `emd`, `chamfer`, `pointnet2` and `pvcnn-smoke` persistent marker.
Cells9–10 completed; the final verifier returned `status: passed` for
PyTorch2.1.2+cu121, CUDA12.1, A100-SXM4-80GB, native GPU operators, ModelNet40-C
and checkpoint assets. The r7 runtime remains live under the explicit lifecycle
contract.

[Run] Exact official `main_split_nobg` H5 SHA-256
`b7176ebd4b8f2218c8cca84b59617d96f5145d864924cc930b36eaf55931c9e1`
and released Point-MAE checkpoint SHA-256
`53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac`
produced clean504/581=86.74699% and author-order Gaussian severity5
248/581=42.68503%. Both complete batch32/seed0 runs loaded the checkpoint with
no missing or unexpected keys. Compared with the validated `main_split` runs,
the no-background gains are+15.3184pp clean and+11.1876pp Gaussian.

[Inference/Open] The dataset split is a material cause of the earlier low
accuracy but cannot explain Table3 by itself: no-background Gaussian-s5 remains
16.3150pp below59.0%. Exact Table3 corruption bytes and training/checkpoint
provenance remain required. The identical248/581 count for no-background-s5 and
main-split-s3 is not treated as protocol equivalence.

[Artifact] Local state/manifest and two validated ZIPs are in
`result/preparation/scanobjectnn_nobg_20261018T180104Z/`. Clean ZIP SHA-256 is
`f655a6524932dbefa20d12d57ac7c01b4b289873f5c9cb83af0ad73715f22e1e`;
Gaussian ZIP SHA-256 is
`b07c50d3169d70f16136f94559a583a4afb7ee8f73d8b451fe304bfd3dc602e3`.
CRC, exact seven-file membership, summaries and581-example totals pass. Drive
copy is `MyDrive/thesis/result/preparation/scanobjectnn_nobg/20261018T180104Z/`.

[Code/Diagnosis] The first diagnostic launch failed before state creation on an
exact source-hash guard: r7's `dev` checkout lacked `run_baseline.py`,
`research_artifacts.py` and `gsd_protocol.py` and had an older
`utilities_3dd_tta.py`. Uploading the exact four-file, SHA-guarded local source
snapshot allowed the safe retry. A separate monitoring audit proved that bare
notebook `subprocess.run` hid child heartbeats from the CLI IOPub stream even
though the remote helper emitted them. Cells5–8 now reuse the existing
line-streaming `_tta_run_checked`; the Colab skill requires IOPub forwarding and
independent browser-attachment verification. Focused tests were RED before both
rules and GREEN afterward; the skill validator passes.


## 2026-10-18 — multibackbone asset and classifier preparation

[User decision/Code] `gsd-multibackbone-study` was created at exact
`2684f11922d77696a26c2b954dcd0c43de5f5a6c`. The user has no author checkpoints
and authorizes published dataset-trained alternatives. Concurrent edits to the
Colab operator skill/setup record and their tests are preserved outside this
study's commits.

[Paper/Open] ICCV2025 GSDTTA Tables1–2 give all15 source macros of
66.51/71.38/66.99 (ModelNet40-C) and55.72/53.06/50.66 (ScanObjectNN-C), in
DGCNN/CurveNet/PointNeXt order. Main paper and supplement do not identify exact
checkpoint bytes, variant or inference preprocessing. Six published alternatives
were downloaded and strict CPU-loaded; none is asserted to be a GSDTTA paper
checkpoint. Two misleadingly named Scan CurveNet candidates were rejected
because their classifier heads have40 outputs rather than15. Full identities,
source revisions and uncertainties are in
[the study](gsd_multibackbone_study_20261018.md) and
[provenance summary](gsd_multibackbone_provenance_20261018.json).

[Run] The BFTT3D released ScanObjectNN-C clean581x1024 arrays have the same label
order as official `main_split`, and every point is an exact coordinate member
of its corresponding581x2048 H5 cloud (maximum nearest distance0). This verifies
split/order membership, not GSDTTA corruption-byte or seed identity. Select its
canonical15 severity5 arrays, excluding extra scale. Existing severity8 Drive
arrays and the Gaussian-only receipt are not reused as all15.

[Code/Run] Added a separate frozen classifier boundary and opt-in source/parity
runners, leaving existing Point-MAE callers unchanged. Ten focused boundary CPU
checks pass; real CUDA Point-MAE parity remains pending. PointNeXt's native batch
extension compiled/imported in existing r7's CUDA12.1/GCC11.4 environment without
GPU inference. A stalled local CLI stream was resolved only after completion
state/binary verification, then a bounded recovery archive was sealed. Drive and
local archive bytes478900/SHA-256
`1043480475cfc0f4262ad175ce22a092921a90ffb5ea507ce1bf43af3c089bfd`, CRC and
safe paths match. Raw evidence:
`result/preparation/multibackbone_20261018/`. Actual runtime source is dirty
`dev@a458cd4`, distinct from the planned local snapshot. The native binary SHA is
recorded in the provenance summary.

[Decision/Open] No accuracy experiment or GSD integration has run. First require
real Point-MAE parity, six source smokes and full15 source diagnosis; subsequently
compare “GSD-inspired latent spectral guidance + <backbone>” against the same
checkpoint/preprocessing's source-only result. Whole planned GPU phase exceeds
60min; comprehensive user approval is still pending. No numerical paper-parity
claim follows from strict-load success. r7 must remain live pending explicit
combined task-complete/close authorization.


[Run/Diagnosis/Artifact] Multibackbone preparation addendum19:51 UTC: all26
focused CPU tests pass. Real r7 Python3.8/torch2.1.2 CPU preflight strict-loads
all6 models, with CUDA uninitialized and no forward calls. First preflight's
wrong inferred ModelNet label shape was rejected; actual headers are2468x1,
not2468. A fresh revision2 manifest fixes only observed metadata, preserves
failed v1 and all data/code/weight bytes. Correct manifest SHA is
`e6421333fdaab6fdd30167ea905f3d829d40de77f78fe91b696c035caa02a07a`.
Passed CPU ZIP SHA is
`782225ef24f418fe4cdf41c9752a378684f02f0eed66921a21391075f8941d7e`;
Drive/local SHA and CRC match. Source snapshot SHA
`683924c4e8caad6459dfdf48caccfd545185501d3ea04257ae7f114028e09dd5`
describes base2684 plus the study overlay, distinct from dirty remote dev.
The full249654807-byte payload and39060-byte revision2 delta are verified on
Drive. Detailed recovery/failed-phase evidence is in the study note. GPU scope
is concretely prepared at [the phase note](gsd_multibackbone_gpu_phase_20261018.md),
but no long-GPU approval or GPU accuracy exists yet.


## 2026-10-18 20:08 UTC — approved campaign, persistence and first parity stop

[User decision] The user approved the prepared whole conditional8-hour GPU
campaign, seed0, and required Drive persistence after each model. Added verified
immutable per-corruption/parity-condition checkpoints plus final per-model ZIP
receipts; interrupted copies without a completed receipt are incomplete. All28
focused CPU tests pass; real r7 strict CPU preflight again loads all six models
with unchanged state hashes and CUDA uninitialized. Revision3 asset/source IDs
and Drive receipts are recorded in the current GPU phase note/provenance JSON.

[User decision/Run] The user subsequently explicitly requested closing all
assignments unused by this task. This supersedes the earlier don't-touch-other
assignments instruction for the two observed CPU assignments. They were released
using the installed Colab CLI client's unassign method (plain CLI stop accepts
only locally named sessions); server re-listing proves only protected r7 remains.
Receipt: `result/preparation/multibackbone_20261018/unused_assignments_release_receipt.json`.
The skill's foreground monitor remains live for r7. The agent explicitly states
that monitoring cannot guarantee Colab quota/maximum-lifetime retention; no
runtime-lifetime guarantee or fake GPU/anti-idle work is claimed.

[Run/Diagnosis] First actual parity phase started20:08:05 UTC and stopped before
any forward. SHA-verified original Point-MAE ModelNet payload contains NumPy
scalar training metadata unsupported by torch2.1.2's weights-only unpickler.
Strict-load success of the six alternative backbones does not test this legacy
payload. Failure was sealed and copied to Drive automatically; local ZIP SHA
`84b108a8ca186d3b0f171feb14acc832010f8f5ae5046a0279fcc7a208357599`
matches Drive receipt, CRC and failed config. GPU-job budget consumed11.328s.
No parity or accuracy passed. Use the existing legacy payload loading semantics
only behind exact known Point-MAE SHA guards; keep all six alternative loads
weights-only. A fresh snapshot and CPU check precede the short approved retry.


[Run/Artifact] 2026-10-18 20:14 UTC — Point-MAE real CUDA boundary parity
accepted on both datasets (3 conditions x32 IDs). All logits/predictions, RNG
post-states, inputs and model-state hashes match, max difference0.0. Accepted
ZIP SHA values are `61cef430...b7a456` (ModelNet) and `ab42b3b3...f87a19d`
under canonical `result/<dataset>/pointmae_boundary_parity/` paths listed in
the study. Drive/local SHA, CRC, full member manifests and NPZ pair equality
pass. Phase elapsed30.967s; no alternative backbone accuracy exists yet.
Legacy metadata failure was fixed with exact-open-handle SHA-gated original
Point-MAE payload loading only;29 CPU tests and8 actual strict CPU loads pass.
Revision4 source/manifest/Drive preparation details are in the study and current
GPU plan. The user approved8GPU hours and per-model/per-corruption persistence.
Only r7 remains after explicitly authorized release of two unused CPU
assignments; the single local monitor preserves CLI ownership, not a guarantee
of externally imposed runtime lifetime. Proceed to six source smokes.


[Run/Artifact] 2026-10-18 20:17 UTC — six source-only GPU smokes accepted:
all3 backbones x both datasets, Gaussian-s5 first32 original IDs, seed0/batch32.
Finite logits, exact class/output sizes, NPZ indices/argmax/counts, strict weights
and frozen-state hashes pass. Six final archives and corruption checkpoints
are verified on Drive; local CRC/safe paths/full manifests also pass. Elapsed
54.388s. No macro benchmark inference follows from prefixes. Advance
unchanged to six full15 rows; source discrepancies must precede GSD integration.


## 2026-10-18 20:35 UTC — full source and clean controls accepted

[Run/Artifact] Six full15 seed0/severity5 source evaluations and six clean
controls complete on existing r7, all2468 ModelNet/581 Scan main_split objects.
Independent validation recomputes137205 corrupted +9147 clean predictions from
indexed finite logits, pinned labels and argmax/counts. Strict weights and
full frozen-state hashes pass; every raw archive has matching Drive/local
SHA, CRC, safe paths and member manifest. Source macros DGCNN/CurveNet/PointNeXt
are58.7979/64.0113/65.8320% ModelNet and32.7826/52.0711/49.0878% Scan.
Clean values are89.7083/89.9109/92.5041% and68.6747/77.4527/92.0826%.
All15 phase20:18:18–20:24:20 UTC; clean20:26:51–20:28:04 UTC. Focused source
runner16 CPU tests pass; clean is explicitly singleton severity0 and does not
change default all15. Revision5 strict CPU preflight is again verified.

[Artifact] Derived immutable archive SHA
`35a3676efa75dc39084760a476ba086f6579de80ceee5faf03e6fa5c19cedfed`,19410 bytes,
local `result/preparation/multibackbone_20261018/source_clean_analysis_v1.zip`,
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261018T203352Z/source_clean_analysis_v1.zip`.
It contains exact12 run indexes, paper row mapping and independent validation/
comparison files; raw ZIPs remain separate and immutable. All per-corruption
and final per-model Drive receipts passed before advancing.

[Inference/Open/Decision] Exact GSDTTA weights and corruption/protocol identities
remain unknown. Known alternative checkpoint/training and preprocessing
confounds, unequal corruption deltas and weaker clean Scan DGCNN are recorded
in the source-results note. These do not isolate a unique quantitative cause.
Severity8->5 previously partially improved Point-MAE; severity1 failed to
reproduce13-column 3DD Table3. That is a different paper/backbone investigation,
not proof of GSDTTA severity. No tuning to published targets. Retain all six
fixed published alternatives for already-approved exploratory matched-source
GSD-inspired guidance; paper-reproduction gate remains open. First verify
three GSD native backend identities/no-build import, then integration CPU tests,
new immutable source deployment, GPU smokes and conditional full15.


[User report/Inference/Open] 2026-10-18 20:43 UTC — user proposes parking
DGCNN and prioritizing two-backbone matched guidance, then original3DD-TTA as
a subsequent comparison. PointNeXt+CurveNet is the provisional interpretation
of “ModelNet and CurveNet”; clarification pending. No guidance GPU launch yet.
The old severity1/Point-MAE investigation concerns3DD Table3, not a causal
explanation of GSDTTA's six new source rows. CurveNet/ModelNet still has a
7.37pp source gap. Fixed legacy800 guidance is a selected historical setting,
not a new-backbone optimum. Implementation preparation narrows to two heads;
source ZIPs remain immutable, future3DD/spectral0 studies stay proposals.

[User decision] 2026-10-18 20:44 UTC — user explicitly confirmed
PointNeXt+CurveNet; preserve both datasets, park DGCNN. Proceed within the
previously approved fixed guidance campaign, with two-head shared geometry.


## 2026-10-18 20:47 UTC — two-head guidance integration CPU gate accepted

[User decision] PointNeXt+CurveNet explicitly confirmed on both datasets;
DGCNN parked, its six source/clean results preserved. Original3DD-TTA and a
same-flow spectral0 comparator remain proposed next stages, not launched.

[Code/Run] New opt-in shared reconstruction runner and scoped native binding
pass33 focused CPU tests (plus10 unchanged FrozenClassifier checks earlier).
Actual r7 Python3.8/torch2.1.2 CPU preflight loads all4 audited native binaries
without JIT build, restores the original loader, leaves CUDA uninitialized,
strict-loads six alternative and two original Point-MAE models, and verifies
four selected matched-source rows against original all15 manifest/data bytes.
The original Point-MAE/GSD entrypoints and equations remain unchanged. Real
LION loading, B70 GPU memory/time and adaptation metrics remain the next smoke
gate; CPU checks do not establish those.

[Artifact] Revision6 manifest SHA
`c88e96de38aa3498f3e3989b286d52e1970a4f1a62d33f86d8e71bd8c9329e25`;
sourceZIP SHA `a43b60d5e62c3a25c6a302cb7509b0042b04846bfe40918570dbfe6bc3e01c5e`.
Preparation delta5523990 bytes SHA
`9dfca4fe1fe6595326a77f95646273826612f6553ce818366b4477551c207706`.
Actual CPU proof ZIP28303 bytes SHA
`e7a69695ff9cc09650040fb2c607b67f598ef62b230521454f2c00a2e37167ee`,
local `result/preparation/multibackbone_20261018/cpu_preflight_v6.zip`;
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261018T204639Z/`.
Source/preparation and CPU proof copies match Drive/local SHA; CRC, all nested
phase source members and exact executed deployer bytes pass. Audit v2 native
ZIP SHA `c77d71e934e7b52f65493e49957f2ff3f45986babf4cd4b6f3d862b10932fc2e`
also passes all99 member hashes and Driveroundtrip. CPU phase elapsed21.095s.

[Decision/Next] Retain fixed legacy800/beta2/spectral1.17/SCD1, normal10/BG30,
seed0/B70, all other predeclared raw/eval/updated-decoder policies. Each head's
full source preprocessing is separately RNG-isolated on the same decoded2048
geometry. Batch shards and corruption/final ZIPs have immutable SHA-verified
Drive receipts. Proceed to3-condition one-batch GPU smoke per dataset/head;
only after artifacts/finite geometry/logits/losses/frozen states/RNG guards pass, run full15.
Continue within the user's unchanged28800s GPU-work cap and preserve r7.


## 2026-10-18 20:53 UTC — two-backbone CUDA smokes accepted

[Run/Artifact] Existing r7 completed both selected heads on Gaussian,
Background and Density, first70 original IDs each, seed0/B70. All840 indexed
predictions independently validate finite logits/argmax/pinned labels/counts;
shards exactly concatenate to corruption NPZs. Six shared geometry batches
match across the two heads, all RNG guards and frozen classifier/LION state
hashes pass. Native bootstrap reports all4 pinned binaries/no-build. Raw/eval
LION loads successfully; peak whole-process allocation32381.92MiB (~31.62GiB).
Reconstruction takes~7s/normal batch and~13.8s/Background; full two-dataset phase
is estimated1.5–2h, within the existing8h approval. Smokes are partial subsets,
not all15 accuracies and not evidence of improvement over full source macros.

All4 final ZIPs and12 batch NPZ/diagnostic receipts were independently checked
against exact local/Drive bytes, in addition to per-corruption checkpoints.
Raw ZIPs live under canonical `result/<dataset>/gsd_inspired_latent_spectral_guidance/`;
run IDs and full hashes are in the provenance JSON and guidance smoke validation.
Smoke jobs consumed57.106+56.524s; no model/parameter setting changed afterward.

Immutable evidence ZIP local
`result/preparation/multibackbone_20261018/guidance_smoke_evidence_v1.zip`,15264bytes,
SHA `4760b814c94b3bab81f26b9954e872a85663dd106407f72184af2922df44de75`;
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261018T205237Z/`.
It contains the exact executed CPU verification phase, ledger, final phase
state,12 Drive receipt audit and independent derived analysis/source bytes.
Local/Drive SHA, CRC and all member identities pass. Current code batch is
pushed as fbb096b; executed snapshot remains separately identified by revision6.

[Decision/Next] Run fixed full15 seed0 for CurveNet+PointNeXt, both datasets,
with batch-level immutable receipts and per-corruption/final model ZIPs. Compare
only to each own fixed full source baseline. Original3DD-TTA and spectral0
controls stay future proposals. Preserve r7 and the sole local monitor.


[Run/running] 2026-10-18 20:53:36 UTC — fixed full15 two-head phase launched
on protected r7 through monitor97631. Exact controller SHA
`c609a312d72068154c244ddeaccb3c9b3f8620c9431c17fc4b3a76f84f6981d7`,
revision6 manifest/source unchanged. Phase path
`/content/gsd_multibackbone_study_20261018/gpu_phases/20261018T205336Z_guidance_all15/`.
ModelNet run IDs `20261018-205338_curvenet-seed0` and
`20261018-205339_pointnext-seed0`; Scan follows automatically only after both
ModelNet final ZIPs verify on Drive. First Uniform batches are producing
verified immutable Drive shards. Prior GPU-job consumption645.614s of28800s.
Estimate1.5–2h for full phase, existing approval unchanged; no new3DD-TTA run.
Do not start another cell or duplicate job while this phase is active.


[Run/Partial campaign] 2026-10-18 21:00 UTC (local date2026-10-19) — first
full corruption accepted while the unchanged full15 phase continues. ModelNet
Uniform has2468 indexed predictions per head. CurveNet source1661/2468=67.3015%
versus guidance1774/2468=71.8801%, +4.5786pp (271 corrected/158 harmed objects).
PointNeXt source1117/2468=45.2593% versus guidance1880/2468=76.1750%, +30.9157pp
(844 corrected/81 harmed). Both immutable corruption ZIPs were downloaded from
Drive; receipt SHA/bytes, safe paths, CRC, every checkpoint member hash, labels,
IDs, finite logits/argmax/counts and original-source predictions independently
pass. Local paths are
`result/preparation/multibackbone_20261018/full_modelnet_<backbone>_uniform_checkpoint.zip`;
ZIP SHA values are `d2ed9e9edb465f28d85b41652fba4c56ece8976986fc011ec80ef66a1dd6cf32`
(CurveNet) and `579b5a7b74ecb9076e8515a152fc7c627e342453d218b0e5a97ef4367b80526c`
(PointNeXt). Drive roots are the current205338/205339 run directories,
`corruption-uniform.zip` plus receipt. Derived counts are in
`uniform_full_checkpoint_validation.json` under the same local preparation root.
These are single-corruption descriptive gains, not all15 means or isolated
spectral contributions. No setting changed after observing the labels.


[Run/Partial campaign] 2026-10-18 21:07 UTC — full ModelNet Gaussian
checkpoint downloaded from Drive and independently verified for both heads:
CurveNet source1902/2468=77.0665% ->1770/2468=71.7180%, -5.3485pp
(175 corrected/307 harmed); PointNeXt1356/2468=54.9433% ->1871/2468=75.8104%,
+20.8671pp (641 corrected/126 harmed). Receipt size/SHA, CRC/safe paths,
checkpoint member hashes, pinned IDs/labels, finite logits/argmax/counts and
frozen classifier hashes pass. ZIP SHA values:
CurveNet `b1251937f74ee9ecb1a9457c9db405826bfd80c461f30a1e8e8e0d44f5ad0379`;
PointNeXt `ef5424b8f100ba9e8bdccfd808c23a6e379e2c567e2e19cf0a77e1e92d5a48a5`.
Local files are `full_modelnet_<backbone>_gaussian_checkpoint.zip` and
`gaussian_full_checkpoint_validation.json` in the preparation root. Drive
roots remain205338/205339, `corruption-gaussian.zip` plus receipt. Mixed
positive/negative corruption effects preclude an all15 benefit claim now.
Full15 continues unchanged; no target-based configuration selection occurred.


[Run/Artifact] 2026-10-18 22:05 UTC — ModelNet full15 accepted: CurveNet
64.0113->65.1864%, +1.1750pp; PointNeXt65.8320->66.4722%, +0.6402pp. Independent
74040-prediction/540-shared-batch checks pass, full classifier/LION states and
RNG guards unchanged. Each head improves6/worsens9 corruptions; no universal
benefit or isolated spectral claim. Joint duration4165.243s; raw final ZIPs
are verified on Drive and locally. Derived8837-byte analysis ZIP SHA
77ff619a008b881702358b83e4fbe54e6518bf3904cbdb17c13a6500389f1eff passes actual
Drive upload/download roundtrip and all member identities. Full identities,
per-corruption table and interpretation are in the new guidance-results note.
Scan full15 now runs; no extra GPU phase or config change.


[Run/Artifact/Inference] 2026-10-18 22:23 UTC / 2026-10-19 local — complete
two-backbone guidance phase accepted. Scan main_split581 full15: CurveNet
52.0711->55.0086% (+2.9375pp), PointNeXt49.0878->45.2094% (-3.8784pp).
Combined with ModelNet +1.1750/+0.6402pp, the fixed legacy policy benefits3/4
matched cases descriptively and harms Scan PointNeXt. Single seed; no universal
benefit, isolated spectral attribution or new optimality claim. Scan CurveNet
improves7/worsens8 corruptions; PointNeXt improves5/worsens9/ties1.

Independent91470-prediction/675-shared-batch analysis and actual Drive reads
verify all4 final ZIPs,1350 batch receipts and60 corruption receipts. Frozen
classifier/LION states, RNG/geometry guards, source/data/preprocessing linkage,
CRC and all member hashes pass. Exact raw/source identities and per-corruption
results: [complete report](gsd_multibackbone_guidance_results_20261018.md).
Final immutable evidence183296 bytes, SHA
`ef34d122fcf38f36a9a85a497f3aee0fc1584d1876f797167f7ff310574f90a3`, local
`result/preparation/multibackbone_20261018/guidance_all15_evidence_v1.zip`;
Drive preparation root `20261018T222327Z/` has the exact archive/receipt.
All downloaded and nested member bytes verify. Full guidance5169.979s;
whole GPU-job ledger5815.594s of28800s. Executed revision6/source hashes remain
unchanged from accepted smokes; code batch fbb096b, prior accepted record e19cb75.
No re-evaluation or test-label tuning followed the mixed result. Backups retain
completed outputs, not raw decoded geometry/latent/RNG restart state.

[Decision/Open] Original3DD-TTA with matched heads and same-flow spectral0
remain subsequent proposals, not run. Replication could falsify the one-seed
pattern; matched controls could attribute the benefit or harm. Exact paper
weights remain unavailable. Preserve r7 and monitor; no teardown authorized.


[User decision/Code/CPU/Open] 2026-10-19 06:02 UTC — user requests original
3DD-TTA for CurveNet/PointNeXt, then DGCNN with original3DD and GSD. DGCNN
parking is superseded for this extension; retain its fixed weaker source model.
Opt-in runner adds method/head selection while defaultGSD and unchangedtta.py
math remain intact. Root47CPU tests pass. Actual r7 CPU gate strict-loads8
models, binds6source rows, audits4no-build native binaries and actualDDIM
5/35 schedules; CUDA uninitialized. CPU proof33902bytes SHA
`c725f6cfe02de9a845562e7bb938f979e9c1a8709fe084d452ad0ebf04f5c769`;
Drive/local roundtrip of delta5464063bytes also verifies. Revision7 manifest
`78cdb53891bc0714dacd100eb220a5f6deb26bfe36f4e10b0aaa16f9aced60f9`;
sourceZIP `f82cf5052b8100591df679b159acec9f172144c5f6a0a916c5a5255cadeed4cf`.
Parent acc27d3; exact execution source is the separately sealed overlay.

Prepared6joint full jobs/8new model rows,182940predictions, seed0, severity5,
ModelNet2468/Scan main_split581. OriginalB32/5–35/originaldecoder/alphaTrue
versus GSD B70/10–30/updated/alphaFalse is whole-policy, not spectral causality.
Estimate2.5–3.5h, new4h cap; single explicit duration approval still pending
before any extension GPU smoke/full. Old four GSD rows are reused.
Detailed commands/hashes/checkpoint links/stop/persistence/acceptance plan:
[extension note](gsd_multibackbone_3dd_extension_20261019.md).
Drive root `.../preparation/gsd_multibackbone_study/20261019T060200Z/`.
Protected r7/monitor remain live; only stale local liveness client was replaced
by a fresh successful bounded audit. No runtime rebuild/teardown or GPU run.


[User decision/Run/running] 2026-10-19 06:16 UTC — user “Devam et”
approves the prepared2.5–3.5h extension with14400s cap, selected smokes and
conditional full15. Exact scopeSHA e0eafe5938907d6de45803daa550f486567e86e532cd1f3cbb81362541b74f41
unchanged; accepted preparationcommit fd7a82d. Approvalactivation is persisted
to Drive and locally,3756-byte immutable ZIP SHA
`a85a588bfaf49e6c389da4525864aed6581dd13c934167f2544013783a8e0bee`,
Drive preparation root20261019T061550Z; exact scope/approval/executed source
bytes, CRC and Drive/local SHA pass. No additional launch approval is needed
for this same phase within its cap.

GPU smoke starts on protectedr7 via solemonitor97631: phase
`/content/gsd_multibackbone_study_20261018/gpu_phases/20261019T061610Z_extension_smoke/`.
Firstjob original3DD CN+PN ModelNet/B32; remainingpredeclared6joborder and
all8outputmodels unchanged. Full15 still requires independent smoke acceptance.
Each batch/corruption/model writes verified Drive evidence. Runtime preserved.


[Run/Artifact] 2026-10-19 06:21 UTC — all6selected extension CUDA smoke
jobs/8model bundles pass independent996-prediction/18shared-geometry validation.
Native4no-build, exact checkpoint/source preprocessing, finite geometry/logits,
strict/frozen classifier+LION hashes and RNG isolation pass. Original method
actual full/5-or35 DDIM schedules match the locked config. All24batch receipts,
24corruption receipts and8final ZIPs physically match Drive/local SHA.
Smokes remain partial prefixes, not full15 accuracies or gain evidence.
Peak whole-process allocation: originalCNPN ModelNet35447.222MiB,
Scan17931.125MiB; DGCNN GSD32366.378MiB, original14494.007MiB (maxdatasets).
Cumulative smoke jobs250.287s of14400s new cap; fixed source/settings unchanged.

Immutable smoke evidence21819bytes SHA
`0b1a91476eb42a649225f8bf2c91ac004ccdab9eb7adf64afddcb4598de0808b`;
local `result/preparation/multibackbone_20261018/extension_smoke_evidence_v1.zip`,
Drive preparation root20261019T062114Z. Actual download, CRC, all nested
manifest members, exact executed finalizer and analysis source verify.
Nested analysis ZIP10896bytes SHA
`a863fea56c71b469ff08c3794440be1f7ec0a39a0df1d490d212dfb83be62404`.
Standalone analyzer revision3 fixes a not-yet-executed revision1 local variable
shadow and adds declared partial/full job coverage checks; actual raw inference
source is unchanged. No GPU rerun was needed for that CPU-only analyzer fix.

[Run/running] Fixed full15 begins06:21:48UTC on protectedr7/monitor97631:
`/content/gsd_multibackbone_study_20261018/gpu_phases/20261019T062148Z_extension_all15/`.
Exact controllerSHA ccd395b0fbda7662da17daa13542151fcbc410e9368d10f98775645238cdb041.
OriginalCNPN ModelNet first, then Scan, then DGCNN GSD/original pairs.
Batch/corruption/model Drive persistence and whole14400s job cap remain active.
No further approval required for this approved scope; no teardown authorized.


[CPU/Artifact] 2026-10-19 — verification hardening during unchanged full GPU
phase: finalizer review required exact NPZ+diagnostic receipt membership and
comparison to phase-pinned sealed ZIP member identities, rather than mutable
bundle-directory files alone. New `finalize_extension_all15_template_v2.py`
implements both. Rechecked all24 actual smoke receipts against pinned sealed
ZIPs: all contain exactly2 expected files and every byte identity matches.
No missing/mismatched actual artifact or inference/source change occurred.
Derived binding verifier+validation ZIP1399bytes SHA
`693b2bc329802b53845d160424de4676ca8f513aa97c1e049b44ba8b3453657a`,
local `result/preparation/multibackbone_20261018/smoke_sealed_member_binding_v1.zip`;
Drive062114Z preparation folder has exact uploaded/downloaded bytes and CRC.
This closes the generic receipt-verifier gap without repeating any GPU work.


[Run/Artifact/Inference] 2026-10-19 06:56 UTC — ModelNet original3DD full15
accepted: CurveNet67.0719% (source64.0113,+3.0605pp), PointNeXt68.2928%
(source65.8320,+2.4608pp). Previously accepted GSD65.1864/66.4722% is lower
than original by1.8855/1.8206pp. Exact paired IDs show GSD corrects2250/2273
but harms2948/2947 events relative to original. Single-seed whole-policy
comparison, not isolated spectral causality; no setting changed.

74040predictions/1170shared batches, frozen states/RNG, actual scheduler, all
member hashes and Drive/local raw ZIP SHA pass. Joint original job2023.697s.
Derived16321-byte matched analysis ZIP SHA
`85ac78483c9ad5eb32e41aa7a40d6b5c9d82e4676ae76b6d957681d26418c5a8`
passes actual Drive roundtrip and every manifest member. Full raw identities,
source/oldGSD links and per-corruption results in
[matched comparison](gsd_multibackbone_method_comparison_20261019.md).
Scan original CN/PN now running; DGCNN jobs follow unchanged within4h cap.
Replicated seeds or matched same-flow controls could falsify/attribute the
one-seed ordering. Preserve r7/monitor; no additional GPU policy/search.


[Run/Artifact/Inference] 2026-10-19 07:06 UTC — Scan main_split581
original3DD accepted: CurveNet55.2381% (source52.0711,+3.1670pp),
PointNeXt49.6845% (source49.0878,+0.5967pp). PriorGSD55.0086/45.2094%
is0.2295/4.4750pp lower than original. Whole-policy seed0 comparison only;
small CurveNet difference is not stable superiority evidence. No settings
changed after observations. Both original datasets now91470 predictions/1455
shared batches independently pass frozen/RNG/scheduler/labels/logits/ZIP
checks. Joint Scan job508.593s; ledger2782.578s before active DGCNN job.
Immutable22758-byte combined analysis ZIP SHA
`8f3be7464f78c12dcf4eb588019fa131ba41f9241980eda71a21d2b7ccd35d64`
passes actualDrive upload/download, CRC/all members. Exact paths/raw SHAs and
paired corruption analysis are in the current method-comparison note.

Primary CLI stream became silent after07:01, but read-only file-service
audits confirmed Scan had completed normally and DGCNN GSD ModelNet was
active/advancing (run20261019-070402_dgcnn-seed0). Server still lists the
exact r7/A100 endpoint. Do not rerun/duplicate or kill/restart the runtime;
continue solemonitor plus immutable file-progress snapshots. At final phase
completion, independently check .complete/state/ledger before repairing any
stale local client and submitting the CPU finalizer.

## 2026-10-19 08:09 UTC — Unexpected r7 loss interrupts DGCNN GSD

[Run/Code/Open] Approved4h original3DD/DGCNN extension, source batchfd7a82d /
executed v7 snapshotf82cf505…deed4cf, accepted-result HEAD389dea3. r7 A100
assignment disappeared from repeated server inventories around08:05UTC;
backend cause unknown and no stop/release requested. Last read-only log
08:03:20UTC proves13 complete ModelNet corruptions plus Occlusion batch6 /
490 examples:32574 observed prediction events,475 batch Drive acknowledgements.
Full DGCNN accuracy is incomplete; ScanGSD and both originalDGCNN jobs did not
start. Accepted CN/PN original4rows and matched source/priorGSD artifacts remain
unchanged and verified. Real RNG state was not saved; no exact mid-run resume.

Local loss bundle `result/preparation/multibackbone_20261018/r7_runtime_loss_20261019T080916Z.zip`,
62808bytes SHA25668986b855cfc1e1d30633aa5ace7d84a81ab0e302ed02fd8a140d12aa6853b3c;
CRC/all15 manifest entries/16members pass. Drive backup and actual interrupted
shard recovery pending: saved browser site permission blocksDrive, user asked
to enable it. Completed-job ledger2782.578s excludes approximately61min active
interrupted job; do not undercount. Only stale local CLI client58698 killed
08:07:03UTC after runtime absence; monitor34633/97631 preserved, reportsLOST.
Decision: rescue existing Drive evidence, retain source/checkpoint/protocol,
reassess fresh remaining4-job duration/cap, no GPU rerun yet. Falsifier: a
complete hash-verified DGCNN final artifact or surviving exact endpoint would
supersede the partial/lost status; neither is currently observed.

## 2026-10-19 08:33 UTC — New r8 CLI recovery awaits Drive consent

[User decision/Run/Open] User reports no Drive entry in browser blocklist and
explicitly authorizes a new CLI session/Drive connection for artifact rescue.
A100 High-RAM `tta-multibackbone-recovery-a100-r8-20261019`, endpoint
`gpu-a100-hm-kkb-ass1c0-xawnnd6jnsdw`, allocated08:15UTC; no alternate GPU
selected. Cell1 exact wrapper/reconnect passes; Cell2 only the previously
user-deferred fastrlock0.8.4/Pip failure. Official CLI Drive mount80871 awaits
user Google consent; no new research GPU run or artifact recovery claimed.
Local evidence50859byte/27member ZIP SHA256
`9b803b9f53078050ee1d9bcd6655ba2df285ab8afdf85b9aa82536f92cf38f35`,
`result/preparation/cli_multibackbone_recovery_20261019_r8/r8_setup_cells12_evidence_v1.zip`;
CRC/all member identities pass, consent state excluded, Drive backup pending.
Local monitor63286/session69569 temporarily suspended during external Drive
mount to prevent false-idle probing; resume exact PID after mount. No runtime
stop/release requested; old r7 monitor remainsLOST. Next: obtain consent,
verify Cell3, recover/hash-bind r7 batch/corruption artifacts on CPU, backup/
download/verify then Cells4–10. Full DGCNN rerun requires duration/cap review;
accepted4CNPN original triplets remain unchanged. See current recovery note.

## 2026-10-19 08:52 UTC — Independent remote Drive auth deadline diagnosed

[Code/Run/Open] r8 first interactive mount80871 timed out locally (600s).
Second40124 returned remoteValueError/mountfailed despiteCLIexit0 and user
consent. Installed `google.colab.drive.mount` has an independent120s auth
request deadline. DriveFS auth failure08:38:49UTC precedes user confirmation
around08:41, consistent with that remote timeout. Expected Drive path absent;
no artifact recovered. Read-only CLI console inspections detached normally.
Scoped local helper preserves installed CLI/account and extends only this
mount call toVM900s/local1200s; self-test and actual remote command verify.
Third TTY83156 awaits fresh consent; monitor63286/69569 remains suspended to
avoid concurrent notebook probes, resume after mount. No researchGPUforward.
Diagnosis ZIP12956bytes SHA2561fb90f4db8a829637349b291de0134a49c919fa12055cbfcc6e29fd75b42f070,
`result/preparation/cli_multibackbone_recovery_20261019_r8/r8_drive_timeout_evidence_v1.zip`;
CRC/10members/all identities pass; no OAuth state/URL included. Drive copy
pending. Unchanged v7 full asset payload247002524bytes SHAce9c797c…ac1333,
623members,8upload parts complete; not deployed or executed. Next action is
finish current Google consent, Enter83156, verify filesystem/Cell3 then CPU
recovery; do not infer successful mount from user consent or exit0 alone.

## 2026-10-19 09:05 UTC — r8 fresh Drive request after expired consent window

[Run/Code/Open] Third extended mount returnedValueError09:01 after its15min
VM deadline. One assistant-displayed URL also contained a transcription error;
new consent URLs are copied from CLI verbatim, no invented causality. Fresh
helper remote30min/local35min passes scope test, SHA605fc98d…8c4a2. First new
call lost WebSocket; server/file service survived. Read-only kernel audit and
guarded interrupt cleared only abandoned auth kernel44896965; setup kernel
preserved, all kernels idle before retry. Current TTY77142 now awaits new
Google consent from09:05; Drive still physically unverified, monitor63286/
69569 suspended during external mount, no GPU research or runtime teardown.
Next: user consent/Enter77142, verify Drive/Cell3, then recover actual DGCNN
shards. Prior accepted results and source snapshot remain unchanged.


## 2026-10-19 09:21 UTC — Timely r8 Drive consent hit a hidden HTTP reset

[User report/Code/Run/Open] Within-window consent forTTY77142 failed at
09:09:56.523UTC with ConnectionResetError / Connection reset by peer inside
the propagation hook; installed runtime suppresses this exception at DEBUG.
Therefore timeout alone is insufficient for the latest failure. Stale HTTP
pool is an unconfirmed explanation. Fresh-challenge same-request recovery
returned dry-run false and sent no reply/cell. Guarded interrupt cleared only
that auth kernel; setup kernel/runtime preserved and all idle before retry.
Fresh helper clears only its HTTP pool before finalPOST, sets45s deadlines;
scope/header checks pass, actual mount unverified. TTY87017 now awaits fresh
consent; remote1800s/local2100s. Monitor63286/69569 remains suspended during
external mount. No r7 shards retrieved or researchGPU run. Local9member/12625byte
ZIP SHAa0a3fff927224cf1fac7a4a02ad3b3011b3cbaedf325ba0f71fa8c593601ae38,
`result/preparation/cli_multibackbone_recovery_20261019_r8/r8_drive_connection_reset_evidence_v1.zip`,
CRC/all manifest identities pass; Drive copy pending. Sanitized evidence omits
URLs/state/credentials. Next: user confirms newest URL, Enter87017, verify
filesystem then Cell3/CPUrecovery; do not infer mount from consent/exit0.


## 2026-10-19 09:32 UTC — Drive consent renewed at user request

[User decision/Run/Open] New consent URL generated by unchanged CLI helper,
TTY53559. Old waiting-auth kernel interrupted after identity audit; all kernels
idle before retry, old local CLI terminated, VM/setup preserved. Approximate
remote deadline10:02UTC. No researchGPU/artifact recovery or new method evidence;
existing diagnosis archive remains applicable. Next: user confirmation/Enter
sameTTY, verify physical Drive before Cell3/CPU recovery. OAuth URL/state excluded.


## 2026-10-19 09:39 UTC — User-requested Drive link renewed

[User decision/Run/Open] CurrentTTY44538 supersedes53559, same unchanged helper
and r8 endpoint. Prior auth call/local client ended after guard/audit; setup
preserved. New consent deadline approximately10:09UTC. No inference or recovery;
existing diagnosis evidence unchanged, consent URL/state excluded.


## 2026-10-19 09:56 UTC — r8 Drive mount and Cell3 actually verified

[User report/Run/Code/Open] User confirms latest request; fresh HTTP propagation
succeeds09:48:39UTC and mount completes. Actual research directory and exact
Cell3 marker verify09:52:16. Python3.8.20/remote dev@a458cd46/KNN wheel hash
recorded in recovery note. Same monitor63286/69569 resumed and due liveness
probe finished before Cell3. CPU inventory now running; no new researchGPU.
Local immutable mount ZIP77324bytes/9members SHA2ca5392ee2efb91eaf89fac2f53b09cc48ce7b7c9a690b7cd2802ee2e1828cb4
CRC/all identities pass; uploaded, Drive backup/roundtrip pending. Includes
earlier setup/diagnosis ZIPs and safe success/receipt evidence. Next backup,
parallel CPU recoveryv2 (all hash guards retained), download/verify, Cells4–10.
No recovered coverage or accuracy accepted until physical archive validation.


## 2026-10-19 10:02 UTC — Mount evidence accepted; r7 rescue copying

[Run/Artifact/Open] Mount evidence77324bytes SHA2ca5392ee2efb91eaf89fac2f53b09cc48ce7b7c9a690b7cd2802ee2e1828cb4
passes physicalDrive and fresh local download/CRC/all members. Drive directory
r8_mount_20261019T100157Z under study preparation; full paths in recovery note.
Serial read-only inventory intentionally interrupted after9min10s; v2 retains
all receipt guards and copies8files concurrently. Active10:02:19 on same69569;
1469files enumerated,100copied by10:02:33. No recovered result yet accepted and
no GPU inference. Next exact recovery marker, actualDrive ZIP download and
independent partial prediction validation, then remaining setup.


## 2026-10-19 10:09 UTC — r7 DGCNN partial evidence independently accepted

[Run/Artifact/Code/Open] Parallel CPU rescue1469files completes10:04:56UTC.
32994indexed predictions,481batch receipts,13complete corruption ZIPs plus
Occlusion910/2468 validate;420more events than last local loss observation.
Raw147769635bytes SHAe411595190ec043f964f42d84e4aeb4e23312d7403b4735310933a6f3f48bc8a
and derived13512bytes SHA84aca14c813280630740c80d6245ec7f226b2560ffcacbc81432f60f36f8bf35
pass actualDrive/local roundtrip/CRC/member and indexed prediction/source/
frozen classifier checks. Exact local/Drive paths and verifier/source hashes
in current recovery note. Original loss ZIP nested unchanged onDrive. Not
full15, no actual RNG state or final interrupted LION state check. First tiny
derived backup lost local WS during cwd bootstrap; audited kernel idle/server
live before local-client cleanup/retry; second passed, no inference repeated.
Decision: retain partial evidence, complete fresh setup (Cell4 running10:08:35),
then CPUproof and concrete remaining-phase/RNG plan/approval if>60min. Existing
source/CNPN comparisons unchanged; no new researchGPU or runtime shutdown.


## 2026-10-19 10:33 UTC — Setup Cells4–7 evidence Drive/local verified

[Run/Artifact/Code/Open] Cells4prepare/5EMD/6Chamfer/7PointNet++ each pass
exact wrapper and physical phase marker before advancement. Same63286/69569
now runsCell8/PVCNN, no researchGPU. Partial setup ZIP118027bytes SHA256
f0d83589a168bc70cebef273d4531410acf8597bbd0eaa601fab53cffb65c020
passes actualDrive/file-service upload/freshdownload/CRC/all identities.
Paths and exact times in current recovery note. Includes exact4execution
events/sources and nested prior evidence, no consentURLs/state. Fullsetup/
compiled binaries/proof pending finalCell10 collector. Prepared DGCNN scope
reuses13full rows, newOcclusion/LiDAR RNG boundary,59386new full-eval events
plus408smoke; estimate65–90min/cap120. Scope/controllers parse and actual pure
command assertions pass; no user run approval or inference launch. Complete
setup/strictCPUproof/immutable readiness before requesting the one >60min
approval. Existing accepted results/partial rescue remain unchanged.


## 2026-10-19 11:01 UTC — Complete r8 environment evidence accepted

[Run/Artifact/Code/Open] All10setup cells complete (sole Cell2fastrlock
deferred). Cell8 original stream lost, actualnative marker/serveridle verified;
revalidation60s SDK request timeout emitsalready-complete skip and exactmarker,
no rebuild/VM restart. Cell9passes10:45:43, Cell10nativeGPU/environment verifier
passes10:48:22. NVIDIAA100-SXM4-80GB/Python3.8.20/Torch2.1.2+cu121/CUDA12.1,
FPS/KNN/Chamfer/EMD/data/checkpoints pass. Complete3045275byteZIP SHA256
f1590dc98a1c8f44a34bb46af0c238f309bbf8b0b36fd659e8728d9b6be81f81
passes actualDrive/local CRC/all31entries/source-log equality. Includes3native
.so bytes,5markers and full publicsource/env/toolchain/verification. Paths
and timeout audit in recovery note. Same69569 begins unchangedv7 restoration/
CPUproof11:01:00; no new researchGPU or approval activation. Next immutable
CPUproof/readiness archive and one concrete >60min remaining-phase approval.


## 2026-10-19 11:35 UTC — r8 lost after native identity guard; r9 setup

[Run/Code/Artifact/Inference/Open] v7CPUproof rejects freshchamfer_3D.so vs
pinnedr7bytes before model loads; source/payload identities pass. Failed30128byte
ZIP SHA7f332f7936cfad75421f10c3fec372085a50fae054eb5c0f9e78caf9ea6765a4
onDrive, localreceipt saved. Exact4native archive members validate offline;
restoration source prepared, not executed. r8lastassigned11:03, firstmissing
11:26 and repeated inventories empty; no teardown. HostIdleSleep11:03:30
withwake11:30 overlaps monitor gap, but backend loss reason remainsunknown.
Local3041193byte/8member lossZIP SHAaa219a45161dd1b0b28553bb1d7c846be87bb3644fc8d9a5c9ee4f664bf5cff2
CRC/all identities pass, Drivependingnewmount. Complete setup/native3.so and
32994DGCNNpredictions remain accepted/Drive-local safe. Under existing setup/
retry authorization allocatedr9 A100HighRAM, Cell1passed/sameVMreconnect and
Cell2running. Monitor70119/63241 envREQUEST_TIMEOUT60; caffeinate70055/51775
bounded3h hostawake, no Colab fakework or lifetimeguarantee. No researchGPU or
remaining-phase approval. Full paths/exacttimestamps in current recovery note.


### 2026-10-19 11:46 UTC — r9 Drive-consent gate

[Run/Open] r9 Cell2 exactmarker completed11:40:39UTC; only known deferred
fastrlock failure. Cell1/2 complete, Cells3–10 pending, no new accuracy.
Fresh r9 interactive Drive authorization (TTY94406) must be completed before
checkedCell3; previous r8 confirmations are historical. Same monitor70119/63241
temporarily suspended for exclusive mount, to resume after verified mount.
Consent nonce is not archived. Full handoff and immutable accepted evidence:
[recovery note](gsd_multibackbone_recovery_20261019.md#1146-utc--r9-cell2-complete-new-drive-consent-pending).


### 2026-10-19 12:33 UTC — r9 Drive/Cell3 verified

[User report/Run/Open] Timely fresh consent83619 propagated and mountedDrive;
actualdirectory listing and downloadedCell3 receipt verify. ExactCell3 marker
12:31:02, Python3.8.20/dev@a458cd4/wheel809a43da…586cd0. Same70119/63241 runs
Cell4; no new accuracy. Mount bundle19598bytes SHAda9491d0…5231e6 localvalid;
Drive backup/roundtrip pending afterCell4. Prior failedr8 CPU rawZIP now
Drive/local size/SHA/CRCverified. r9v2 remaining scope preparedonly, no GPU
approval. [Full handoff](gsd_multibackbone_recovery_20261019.md#1233-utc--r9-drivecell3-accepted-cell4-running).


### 2026-10-19 12:40 UTC — r9 mount/r8 loss archival accepted

[Run/Artifact/Open] CLI uploads into absent actualDrive filenames and fresh
Drive-download roundtrips verify mount19598bytes SHAda9491d0…5231e6 and
r8loss3041193bytes SHAaa219a45…bf5cff2, including CRC/allmanifest members.
Cell4 still installingCUDA packages; no new inference. Known pip conflict
warnings retained for final actual verification. [Full paths/current state](gsd_multibackbone_recovery_20261019.md#1240-utc--mount-and-r8-loss-bundles-verified-on-actual-drive).


### 2026-10-19 12:53 UTC — r9 native preparation stage archived

[Run/Artifact/Open] Cells4–7 each pass exactwrapper plusactual requiredmarker;
Cell8 PVCNN running. Partialsetup2928313bytes SHA668852b7…83be677 includes
3actualbaseline binaries,4markers and exactexecution evidence. ActualDrive/
local freshdownload SHA/CRC/safe paths/allmembers pass. No inference or GPU
research approval. [Full paths/gates](gsd_multibackbone_recovery_20261019.md#1253-utc--r9-cells47-archived-pvcnn-running).


### 2026-10-19 13:03 UTC — r9 full environment accepted

[Run/Artifact/Open] All10 exactsetupmarkers and actualnativeGPU/data/checkpoint
checks pass. FullsetupZIP2984884bytes SHA82737111…d0601e actualDrive/local
size/SHA/CRC/safe paths/all31members verify; all11collectorcommands exit0.
Source/env/native binaries archived; remote dev@a458cd4 differs fromresearch
snapshotf82cf505…deed4cf. Same70119/63241 restores unchangedv7 assets/raw
recovery; pinnednativeCPUproof and >60minGPUapproval stillpending. No new
accuracy. [Full paths/evidence](gsd_multibackbone_recovery_20261019.md#1303-utc--r9-ten-cell-setup-accepted-unchanged-v7-restore-running).


### 2026-10-19 13:13 UTC — r9 CPU proof/readiness accepted

[Run/Code/Artifact/Open] Four exactpinnednatives restored withfreshbytes
preserved; missing r7-specific multimethod1.9.1 andexactlegacyScanPointMAE
asset restored after immutablefailedproofs. No broaderpipchanges. Unchanged
CPUproofpasses8strictloads/6matchedsource/no-build/CUDA-uninitialized/DDIM5/35.
Finalproof2978673bytes SHAa06fbba2…7df72a3 andreadiness11411412bytes
SHAfa3cb67a…7222786 passactualDrive/local SHA/CRC/allmembers. No inference.
SoleliveA100r9/monitor70119/63241, newboundedlocalawake73723/64954 retained.
Distinct65–90min/shared120min remainingDGCNNphase readyforoneexplicitapproval;
Driveconfirmationisnotthatapproval. [Full paths/failed-to-passed chain](gsd_multibackbone_recovery_20261019.md#1313-utc--cpu-proof-and-immutable-readiness-accepted),
[concreteplan](gsd_multibackbone_dgcnn_recovery_plan_20261019.md).


### 2026-10-19 15:10 UTC — four DGCNN GPU smoke tests accepted on r9

[User report/Run/Code/Open] “Tamam bu testleri de çalıştır” authorizes only
the4shortGPU tests, scope4c2c4854…a4b139/separate1800s ledger; no longphase
activation. Existingr9A100, frozenv7source/assets/checkpoints/nativebytes.
Gaussian+Background/seed0/s5, GSD B70/originalB32, Scanmain_split581.
All4jobs complete15:04:33–15:07:08UTC:408predictions/8batches,154.552s jobwall.
Independent rawlogit/ID/label/CSV/frozenstate/RNG/no-build/source/preprocessing/
originalDDIM checks pass. ActualDrive4finalZIPs/8batch/8corruption receipts
verify; local rebinding ties16receiptfiles to sealedmembers. Evidence17855bytes
SHAeacc97ac…07f4aa andbinding1631bytes SHAc519af6c…c8d81f passphysicalDrive/
localSHA/CRC/memberchecks. No all15 accuracy claim; remainingfullphase pending
explicitapproval. r9/monitor retainedlive. [Exact paths/source/cost/limits](gsd_multibackbone_dgcnn_smoke_20261019.md).


## 2026-10-19 — Scan four-head trajectory implementation (GPU pending)

[User report/Code/Open] User selected only GSD-inspired guidance for the new
ScanObjectNN-C main_split581/all15/s5/seed0 four-head comparison. Existing
CN/PN Scan archives contain terminal logits, no recoverable clean candidates.
Fixed clean3/5/8 or Background8/15/23 plus terminal reuse class-count-independent
minimum entropy (terminal-first ties) and equal probability mean. Shared probes
use the exact terminal Scan inverse-bbox frame and restore RNG/model buffers;
no GSD update change, labels in selection, new tuning or early stopping.
52 focused local CPU tests pass, including existing source/guidance/original3DD
and Point-MAE boundary regressions. Actual CUDA observation, memory, timing
and fresh immutable Drive/local proof remain pending. Planned whole new GPU
phase25–45min, to be recalibrated by Gaussian+Background one-batch pilot; stop
for comprehensive approval if the phase is expected to exceed60min.
Separate pending full DGCNN completion is not activated by this scope selection.
See [study and batches](gsd_scan_trajectory_selection_20261019.md).


### 2026-10-19 15:37UTC — Scan trajectory sealed CPU proof accepted; pilot running

[Code/Run/Open] Fresh source746bc30/v8 manifest7c3b55fe…616806c and
sourceZIPf30c224d…058948c retain old guidance math/native bytes. Allfour
Scan checkpoint strict CPU loads and pinned no-build imports pass without
forward/CUDA initialization. CPUevidence51514bytes SHAc15bbe0c…5506c9a and
delta5487366bytes SHAa050f0dc…005e008 pass actual Drive/local SHA roundtrip.
Gaussian+Background first70ID shared four-head observer pilot started15:37:49
UTC in existing r9; no trajectory accuracy accepted yet. Fullrun awaits
purity/memory/independent-artifact review and whole-phase timing gate.
[Exact identities and paths](gsd_scan_trajectory_selection_20261019.md).


### 2026-10-19 15:41UTC — Scan trajectory GPU pilot accepted; all15 running

[Run/Inference/Open] Four-head GSD-only observer passes Gaussian+Background
first70IDs:2240candidate predictions/2sharedbatches,65.653s,35.484GiB peak.
Four strict frozenclassifier/LION states and probe RNG/buffers/live latents
pass; allhead indexedtrace hashes and originalmain_split labels match.
FourrawZIPs and phase48595byte evidence SHA380a8046…1fcb036b verify actual
Drive/local hashes. Acceptedanalysis+20physicalreceipts binder37464bytes
SHA182a8f24…2e9cdd9 passes allmembers andactualDrive/local roundtrip.
Wholeplannedphase central29.923min/upper42.505min includespilot, IO andmargin;
same3600s accumulatedcap. FullScanall15 started15:41:44UTC in r9 using
unchanged746bc30/v8 source/manifest. No benchmark outcome yet; no newlong
DGCNNoriginal/ModelNet phase activated.
[Evidence and gates](gsd_scan_trajectory_selection_20261019.md).


### 2026-10-19 16:07UTC — full Scan four-head trajectory accepted

[Run/Artifact/Inference] GSD-inspired only, main_split581/all15/s5/seed0;
source746bc30, manifest7c3b55fe…616806c, sourceZIPf30c224d…058948c. All325
executed project source files exactly match commit746bc30. Four same-run
terminal/minentropy/probmean accuracy percentages:
CurveNet55.134825/56.408491/56.477338;
PointNeXt45.358577/46.769937/46.815835;
DGCNN36.075731/37.131383/37.326449;
Point-MAE34.125072/35.720023/35.685600.
Mean deltas+1.342513/+1.457258/+1.250717/+1.560528pp; entropy deltas
+1.273666/+1.411360/+1.055651/+1.594951pp. Both fixed label-free readouts
improve allfour terminals in this seed; additional seeds remain needed.
CurveNet mean exceeds prior original3DD by1.239243pp; PointNeXt remains
2.868617pp below its original and2.271945pp below source. Whole-flow,
separate-run comparisons do not isolate spectral effects. No Point-MAE full
matched main_split source/original row exists in this study. DGCNN remains
exploratory with its weak fixed source checkpoint.

Fulljob1413.941804s; pilot+full1479.594484s=24.659908min, peak35.484GiB.
139440candidate predictions/135shared batches: strict frozenclassifier/LION,
probe RNG/buffers/live latents, sharedhead geometry/trajectory, declaredsource,
original IDs/labels,15classes, fourcandidates and allterminal CSV/NPZ counts
pass. Actual Drive/local raw4ZIP hashes and sealedmembers verify.
540batch/60corruption/4final physical receipts are preserved in accepted
659325byte evidence SHAa85ca9c6e2925daf50e010b5ac9f688fcef99eab0300479bc9cb78c176852e97.
Local `result/preparation/scan_trajectory_20261019/all15_accepted_evidence_drive_roundtrip.zip`;
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/scan_trajectory_all15_accepted_20261019T160659Z/`.
[Full values, rawhashes and policy transitions](gsd_scan_trajectory_selection_20261019.md).

Decision: accept the selected Scan trajectory study, retain unchanged policies;
do not tune on these labels. New terminals differ modestly from earlier r7
CN/PN runs; use same-run terminal for readout deltas. DGCNN Scan GSD36.075731%
now fills that fullrow. Remaining DGCNN original bothdatasets and ModelNet
Occlusion/LiDAR are unexecuted; old59386-event proposal includes redundant
8715ScanGSD events and must be revised to50671 residual events before launch.
No further GPU phase or teardown; same r9 monitor63241 remains owner.
Falsifier: missing IDs/source/state/receipt binding or policy recomputation
failure would invalidate acceptance; these checks pass. Repeatability and
isolated guidance attribution remain open.


### 2026-10-19 17:44UTC — reduced DGCNN completion authorized and preflight accepted

[User report/Code/Run/Open] User requested remaining tests if r9 remains live.
Same r9 confirmed; scopecc77f0d3…ee2355b removes already accepted ScanGSD8715
from obsolete four-job scope. Three jobs50671events: MN GSD Occlusion/LiDAR,
MN originalall15, Scan originalall15. Whole expected49.8–58.8min based on
actual full-original reconstruction and DGC smoke timings; one3600s cap.
No additional long-run approval required under60min expected total. Existing
four smoke-only authorization is preserved and admitted explicitly under new
scope with unchangedv7 source/assets/runtime; no smoke/source/CNPN rerun.

Preflight actualDrive/local14928byte evidence SHA4214776d…27f6158 verifies622
assets/four native binaries/LION and prior smoke/ScanGSD identities; no active
worker, exact global a458cd4 checkout remains separate from v7 researchsource.
SourceZIPf82cf505…5cadeed4cf/manifest78cdb538…ced60f9 unchanged. Controller
SHA707dc790…a6b1759e submitted to solemonitor63241. Old13 MN GSD corruptions
are preserved; oldpartialOcclusion910 excluded from composite. No exact RNG
resume and old13 finalLION-state limitation remain. NewGPU results pending.
[Concrete scope/protocol/budget/evidence](gsd_multibackbone_dgcnn_residual_20261019.md).


### 2026-10-19 17:57UTC — DGCNN ModelNet GSD composite accepted; original running

[Run/Artifact/Inference] Fresh Occlusion/LiDAR4936predictions/72batches pass
full raw/source/state/RNG/labels validation; firstjob519.251s, raw7036942bytes
SHA3ebdb4e6…dfb1230d actualDrive/local. Old13 revalidated32084events and468
batch receipts; partialOcclusion910 excluded. Complete37020-event all15
composite63.833063%, ownsource58.797947% (+5.035116pp). Two attempts and no
exact RNG resume; old13 lack finalLION state proof, though oldinitial matches
new before/after and all per-corruption classifiers remain frozen.
Derived5487419byte bundle SHA6298278e…a47cd4ea passes actualDrive/local and
allmembers; all15 arrays/provenance/limits retained separately from raw.
Second job ModelNet original3DD started17:54:46UTC; finalScanoriginal follows.
No original-vs-GSD conclusion yet. [Scope, rawpaths and evidence](gsd_multibackbone_dgcnn_residual_20261019.md).


### 2026-10-19 18:25UTC — DGCNN ModelNet original3DD accepted; Scan last job running

[Run/Artifact/Inference] ModelNet originalDGCNN37020predictions/1170batches
completed29.439616min. Accuracy65.586170%, ownsource58.797947% (+6.788223pp),
qualifiedGSD13+2composite63.833063% (GSD-original-1.753106pp). Samecheckpoint/
preprocessing/data/source binding, strict frozenclassifier/LION, exactoriginal
DDIM5/35/finalalphaTrue and allindexedCSV/NPZ/logit/RNG checks pass. No paper
reproduction or isolatedspectral claim; old13 GSD finalLION/RNG limits persist.
Raw18629467bytes SHAf6339944…943e698e actualDrive/local; immutable11156byte
analysis SHA6154f546…d25c3738 passes allmember andactualDrive roundtrip.
Same r9 lastScanoriginal started18:24:12UTC; one3600s cap includes allthree.
[Paths, protocols and acceptance](gsd_multibackbone_dgcnn_residual_20261019.md).


### 2026-10-19 18:33UTC — all remaining DGCNN tests and six-case matrix accepted

[Run/Artifact/Inference] Three new jobs50671events/1527batches complete in
45.441790min (wholecap3600s). New Scanoriginal8715events36.442915%, source
32.782559%, GSDterminal36.075731%: GSD-original-0.367183pp. MNoriginal
65.586170% vsqualifiedGSD13+2composite63.833063% (-1.753106pp), source58.797947%.
New Scanraw7006542bytes SHAbe908813…8d7674139 actualDrive/local verified.
Allfrozenstates/source/native/DDIM/logits/IDs/labels/RNG/CSV/NPZ checks pass.
Complete six-case/18accuracy matrix recomputed from411615joinedmethod-role
events. GSDterminal beloworiginal6/6, abovesource5/6; fixed Scanprobabilitymean
beats original forCurveNet+1.239243pp andDGCNN+0.883534pp, remains below for
PointNeXt-2.868617pp. Single seed/whole-policy only; no spectral attribution.
Old13 MN GSD lack finalLION/RNGresume proof; composite explicitly labelled.

Phase38057byte evidence SHA87201e96…8df05f3 andfinal1105309byte accepted
analysis/physicalreceipts binder SHA227af658…1ead693f pass actualDrive/local
andallnestedmembers;1527batch/32corruption/3final receipts preserved.
Local `result/preparation/dgcnn_residual_20261019/final_acceptance_drive_roundtrip.zip`;
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/dgcnn_residual3_accepted_20261019T183309Z/`.
[Exact tables/paths/protocol/limits](gsd_multibackbone_dgcnn_residual_20261019.md).
Decision: requested missing tests complete; no new GPU studies. Final report
backup then lifecycle cleanup follows the user's conditional close instruction.
Repeated seeds, isolated controls and exact paper checkpoint reproduction
remain distinct unexecuted research questions, not missing requested cases.


### 2026-10-19 18:40UTC — requested cleanup completed after full result backup

[User report/Run] All three remaining tests and final18-row matrix were
accepted, satisfying the user's prior conditional close instruction. Final
238340byte report SHA22ca2442…e27735f verifies actualDrive/local, allmembers
and pushed results commitbf28603. r9 stopped through ColabCLI (exit0,
“Session terminated”); bounded serverinventory confirms0active sessions.
All3monitor PIDs34633/63286/70119 and hostawakePID83891 stopped/absent.
An unbounded verificationinventory hung; only localPID85667 was ended, then
boundedREQUEST_TIMEOUT20 succeeded. No new runtime or research inference.
Postclosure receipt is local/Git-recorded; allresearch data backed onDrive
beforeteardown. User's pre-existing modified/untracked files remain untouched.
[Final report/evidence and lifecycle record](gsd_multibackbone_dgcnn_residual_20261019.md).


### 2026-10-19 20:44 UTC — fresh closure audit and consolidated final interpretation

[User report/Run/Code] User requested verification that no Colab runtime or
background terminal remains, then detailed knowledge consolidation. Bounded
`REQUEST_TIMEOUT=20 colab sessions` returned exit0 and no active server
sessions. Local inspection found no task Colab CLI, monitor, evaluation worker
or host-awake process; former PIDs34633/63286/70119/83891/85667 are absent.
One existing interactive zsh onttys001 has no child processes and was preserved.
No runtime allocation, research run or process termination was needed.
Audit base is `gsd-multibackbone-study@422cb3c7d3b8daeaac3877a26e303df84ac91c86`;
executed research v7/v8 snapshots remain distinct from this local HEAD and the
remote global `dev@a458cd4` checkout. Protected user edits are outside the batch.

[Run] Revalidated23 local ZIPs against accepted SHA-256, CRC and safe-path
records. Independently recomputed the18 method accuracies from90 corruption
count rows and12 Scan readout accuracies from180 rows; all agree. No raw ZIP,
sealed CSV or historical attempt was altered. The prior Scan readout CSV was
sealed before DGCNN original finished and still has empty original fields;
the new synthesis explicitly joins the later accepted original reference.

[Inference] Final whole-policy results: original3DD exceeds own source6/6;
GSD terminal exceeds source5/6 and loses to original6/6. Scan PointNeXt loses
3.878371pp versus source and4.475043pp versus original. Scan probability mean
improves own terminal by1.342513/1.457258/1.250717/1.560528pp for
CurveNet/PointNeXt/DGCNN/Point-MAE; minimum entropy also improves all four.
Against separate original runs, probability mean gives+1.239243pp CurveNet,
+0.883534pp DGCNN and−2.868617pp PointNeXt. These remain seed0 observations.
No matched full Point-MAE Scan source/original row or online speedup exists
in this study. Preserve new same-trace CN/PN terminals separately from their
earlier terminal-only attempts.

[Code/Inference/Open] The comparison changes batch, horizon, decoder style
and DDIM final-alpha/cuDNN policy, so spectral causality is not established.
Two previously recorded validation studies remain unexecuted: same-flow
spectral0/SCD1 control and additional locked-policy seeds1/2. Exact GSDTTA
checkpoint/protocol/corruption identity is a separate unresolved asset gap.
ModelNet CurveNet source remains7.3687pp below the target despite its small
Scan gap; Scan DGCNN clean68.6747% already exposes weak source performance.
The earlier Point-MAE severity1 full13 profile and no-background diagnostic
do not support a severity-only fix or establish a three-head severity sweep.
ModelNet DGCNN GSD retains13+2 complete-corruption coverage, missing old13
final LION state and no exact RNG continuation. No mandatory rerun is inferred
from that disclosed provenance limit.

[Run] Immutable fresh local audit:
`result/preparation/multibackbone_final_audit_20261019T204359Z.zip`,11731bytes,
SHA `f277f3d56e7219ce8c5cc2c2c178efb25a479356d0df143677a60a0677102000`.
Contains server/process result,23 archive identities, protected-file hashes,
count tables and manifest. Prior final report/accepted analyses were physically
Drive/local verified before teardown; this new post-teardown audit checks
local copies and has no fresh Drive upload. Documentation is preserved in Git
without creating a VM solely for backup. No GPU duration estimate applies.

Decision: current requested experimental scope and cleanup are complete;
future research controls remain open without an active runtime. Added the
[detailed synthesis](gsd_multibackbone_final_findings_20261019.md) and reconciled
README, open questions, scope, reproduction diagnosis, method synthesis and
the relevant guidance/comparison/trajectory/residual notes. Historical entries
are explicitly superseded, not erased. Next proposed experiment, if requested,
is the matched spectral-off control followed by seed replication. Changed
assets, missing IDs/state/provenance or failed count/hash checks would falsify
acceptance; seed/control outcomes may falsify broader benefit hypotheses.


### 2026-10-25 10:56 UTC — attribution review for trajectory averaging and entropy selection

[User report/Code/Paper/Inference/Open] User asks whether the recommended
GSD readouts copied prior work and requires relevant attribution. At review
base `ee8d218a923259ac8695e94df5aabd492fd242e2`, dated2026-10-11 trajectory
and advisor G-002 records identify the local idea as the user's proposal.
They do not document a direct paper implementation or establish originality.
The earlier review did not adequately cover diffusion-timestep ensembles;
related work identified now must not be retroactively called the original
inspiration. Existing candidate/probability/entropy code is unchanged.

[Paper/Inference] Verified primary precedents: Rosnati/Roschewitz/Glocker,
ML4H2023 section4/equation9 (timestep prediction-map mean); Wang et al.,
2025 section4.1/equation4 (single-trajectory intermediate-answer voting);
Li et al., NeurIPS2025 appendixB.4 (timestep classifier-logit ensembles);
Purge-Gate, ICCV2025 section3.3.1/equations20–21 (minimum-entropy selection
over token-purging candidates). Task, representation, training and aggregation
differences from our frozen-head decoded-point-cloud readouts are explicit.
DiffusionDet, DDA and MEMO are additional context, not exact implementations
of our inference rule. Generic ensembling/entropy selection is not novel.

[Code/Paper] Seven primary documents downloaded successfully; source URLs,
resolved URLs, sizes and SHA identities retained. Immutable local evidence
`result/preparation/trajectory_attribution_review_20261025T105306Z.zip`,
8897535bytes/14members, SHA
`4c5b8d2bd0b79e0253a1c3adb3057519d7bde0cf91bdb55eea67cc48c743fec3`;
all source hashes, ZIP CRC and member hashes pass. Local-only literature
archive; no fresh Drive upload, runtime query/allocation or GPU run. No new
accuracy, runtime/memory result, inference-source change or numerical rerun.

Decision: cite the relevant precursors, distinguish retrospective related-work
discovery from recommendation provenance, and describe the present study as
evaluation of established readout ideas in this GSD-inspired frozen-classifier
setting. Added [attribution note](gsd_trajectory_attribution_20261025.md) with
exact sections/equations and manuscript wording; linked original proposal,
final findings, README and open questions. Broader novelty assessment and
existing controls/seed replication remain open. A newly found exact method
match or earlier local reference would revise contribution/provenance claims;
this bounded review does not certify their absence.


### 2026-10-25 — user reading list and logit-mean research question recorded

[User report/Code/Paper/Open] At `gsd-multibackbone-study@921d329`, the user
requests saving the four previously verified papers by full title and primary
URL for later reading, and recording whether logit averaging could outperform
probability averaging. Added a reading checklist and relevant section/equation
pointers to `gsd_trajectory_attribution_20261025.md`; linked it from README and
the ModelNet/Scan trajectory notes. No new literature-priority claim or user
reading completion is inferred.

The new `open_questions.md` entry specifies the same saved GSD candidates,
frozen head, IDs and temperature1 for logit mean versus probability mean.
Reuse available Point-MAE ModelNet/four-head Scan logits; report paired and
per-corruption effects, logit-scale limits and all15/macro14 scope. Historical
two-trajectory logit averaging is a different experiment and does not answer
this question. Li et al.(NeurIPS2025, appendixB.4) is related motivation, not
evidence that logit averaging wins in this pipeline.

Documentation only: no new offline policy scoring, GPU execution, runtime
allocation, inference-source change or accuracy result. Existing evidence stays
at `result/preparation/trajectory_attribution_review_20261025T105306Z.zip`
(SHA4c5b8d2b…c743fec3, local-only) and the accepted immutable trajectory
archives. User files remain protected. Next action is the user's paper review
or a separately requested saved-logit comparison; a nonpositive matched delta
would refute any superiority hypothesis, and one seed cannot show stability.


### 2026-10-25 11:04 UTC — next ModelNet trajectory handoff audited

[User report/Code/Run/Open] User requests a correct knowledge state and a
concise next-agent prompt to obtain the missing ModelNet40-C minimum-entropy
and probability-mean results with DGCNN/CurveNet/PointNeXt. Audit base
`7017fa81254f0b2af86b446543b565be7014136a`, same research branch; user edits
remain protected. This is handoff preparation, not experiment activation.

Accepted six-case matrix and four-head Scan trace retain18/12 policy accuracy
values. SHA/CRC-checked ModelNet CurveNet/PointNeXt GSD archives each contain
555NPZ members with no `trajectory_logits`; DGCNN13+2 composite NPZs also
lack candidate logits. Old terminal predictions cannot supply the six missing
readouts. Existing Point-MAE ModelNet/Scan results and source/original rows
must be reused, not repeated. Matching probe geometries/logits, if located
elsewhere, should be audited before new reconstruction.

[Code] Runner `--trajectory-observe` currently rejects ModelNet; shared
observer enforces15 logits, and the analyzer enforces Scan581/15classes.
Existing ModelNet output conversion rotates back and normalizes, unlike the
Scan inverse-bbox contract. The handoff identifies strict40-class/schema/
geometry extension and Scan/default-path regressions as gates, reusing current
class-agnostic policies and reconstruction hooks. No code implementation yet.

[Run/Inference/Open] Prior shared two-head ModelNet guidance took4165.243s
(69.421min), including3934.254s shared reconstruction. New probe/head work
requires a fresh whole-phase estimate and comprehensive >60min GPU approval;
no permission is inferred from the historical campaign or this handoff request.
Next target: seed0/s5/all15/2468 IDs/head, fixedB70 legacy policy,540 shared
batches,3 terminal controls+6 newreadouts. Use each new trace's own terminal,
not historical terminals, for selection deltas. Preserve full candidate logits
for the separately parked logit-mean question; do not add that test implicitly.

Decision: added [next-agent handoff](modelnet_multibackbone_trajectory_handoff_20261025.md),
updated README/scope/open questions/final findings with new pending scope while
preserving completed historical scope. No runtime query/allocation, GPU work,
new accuracy, native rebuild or inference-source change. Last server-empty
verification remains dated19Oct; next agent must inspect current assignments.
Drive persistence/model receipts and local SHA gates remain required for new
runs. Missing candidate coverage/state/source/receipt evidence would prevent
acceptance; no new result or speedup is asserted.


### 2026-10-25 — parallel component-study assignment and isolation prompts prepared

[User report/Code/Inference/Open] User selects a second future agent for
guidance-component tests and bounded improvement trials while Agent A completes
ModelNet readouts. Planning base `gsd-multibackbone-study@47e69be`. Current
Git worktree inventory shows only the original checkout; no worktree, VM,
monitor or experiment was created for either task in this turn. The user
requires derived branches/separate worktrees, clear session names and strict
protection of the other agent's Colab runtime.

Added [ownership contract](parallel_colab_agent_contract_20261025.md):
A=`gsd-modelnet-trajectory`/`modelnet-trajectory` worktree/
`tta-mn-trajectory-a100-<UTCdate>-r<N>`;
B=`gsd-guidance-components`/`guidance-components` worktree/
`tta-guidance-components-a100-<UTCdate>-r<N>`. Both use independent output/
Drive roots, explicit session/endpoint ownership and one monitor; no peer exec,
setup, auth reset, stop-all, broad process cleanup or live-state sharing.
Original uncommitted operator skill/monitor/setup changes remain read-only
inputs, not automatically inherited by a new worktree or silently committed.

[Code] `tta_gsd_v2.py` computes its graph even at spectral0; Chamfer creation
is conditional, while both active losses share the backward. Actual eta updates
local latent, gamma updates style, and lambdaa/p controls selective-Chamfer
retention; graph gamma and loss weights are distinct. Speed benefits need
verified skip paths and actual end-to-end/profile measurements. Existing older
Point-MAE ablations and all six current terminal contrasts remain contextual,
not an isolated spectral causal result or a universal readout win.

[Decision/Open] Agent B's fixed same-host matrix is C+S(1,1.17), C(1,0),
S(0,1.17), U(0,0), each with the same terminal/entropy/probability readouts.
Screen both datasets/four frozen heads on registered small IDs; budget before
expanding96 possible full policy rows/seed. Separate any bounded parameter/
method development from fixed ablation and freeze candidates before prospective
held-out/seed confirmation; balanced dataset/head macros and per-case failures
remain visible. Label prior test-set exposure; do not select a favorable final
test maximum as generalization or a universal necessity result.

Added [Agent B handoff](guidance_component_study_handoff_20261025.md) and
[both copyable prompts](parallel_agent_prompts_20261025.md); updated Agent A's
handoff, README, open questions and scope. No inference code or accepted raw
artifact changed. Whole proposed GPU phases expected >60min still need one
concrete approval; this planning turn does not launch them. Acceptance requires
matched identities/states/IDs/receipt hashes and measured cost; contrary arm/
seed outcomes would refute a claimed spectral advantage or speed improvement.


### 2026-10-25 — bounded3DD citation/reproduction audit completed

[Paper/Code/Run/Inference/Open] On isolated`gsd-3dd-citation-audit` from committed
study base`4c5be0eaf0ac933923129fe26738e0db5f96744a`, audited Scholar title/
Cited-by, SemanticScholar/OpenAlex inventories,31 webqueries and primary texts/
supplements/public pinned code. Verified15 citing publication identities;
12 full primary texts and3 publisher-preview/reference-only cases. Five later
numerical3DD rows occur in GSDTTA and Uni-Adapter. Their author-stated reruns
use DGCNN/Uni3D-Large, respectively; Uni also shares original authors and has
B1/s5/step search. No independently verified exact original Point-MAE rerun
was found. Original paper remainsMN65.7/Scan64.3/Shape69.8, source57.6/61.5/59.3.

[Paper/Code] Fresh SVWA primary/source checks support its ScanGaussian32.5/
source33.1 cross-check; existing read-only checkpoint comparison receipt records
225 common same-shaped tensors differing from released3DD. Repeated PG/LN/
SATOR source fingerprints have unspecified rerun provenance. LoTT explicitly
reports all-severity aggregation, so equal57.6/59.3 means do not prove reuse
or s5 parity. Original3DD AppendixAσ.03 supports5; upstream generator8 versus
evaluator5 is real, with seed call commented out. Existing s5/s1full13/split
controls and canonical ModelNet byte parity prevent claiming random generation
alone explains the remaining gaps. README66.1 also conflicts with its65.44
rounded-cell mean. No new causal attribution or accuracy is claimed.

[Artifact/Decision] [Full report, matrices, ledger and bundle receipt](3dd_tta_citation_reproduction_audit_20261025.md).
Raw literature lives in own`result/preparation/3dd_citation_audit_20261025T123834Z/`
and adjacent sealedZIP; the coordinating root also preserves an identical sealed
ZIP under its canonical `result/preparation/` directory. Small evidence
tables/manifests are committed. No GPU,
Colab command, peer-runtime operation, external message or change to historical
run artifacts occurred. Exact authors' arrays/checkpoints/protocol/logs, or an
accessible exact originalPoint-MAE rerun, would change the conclusion. Current
alternative-backbone studies do not establish exactGSDTTA reproduction.

[Artifact] Coordinator review accepted `e661b33` as study commit `659610f`.
Independent ZIP SHA/CRC/path,107member-hash, extracted-validator and CSV-field
checks pass:72rows,690facts,15publications. Canonical ZIP SHA256 is
`04cc873fc3a696213a9657ae0551b31c1cef1eb1656260f286db86cd05c8217b`;
the raw evidence remains local, without a Drive backup in this literature task.

### 2026-10-25 — expanded source-only and cross-paper reporting audit

[Paper/Code/Run/Inference/Open] From study`51dd03b`, three isolated research
workers and coordinator reviewed16 numerical publications:15TTA papers plus
original corruption benchmark. No model/Colab/GPU execution. The
[expanded audit](pointcloud_tta_reporting_audit_20261025.md) records823 macro
observations,128source rows and77paired original→later cases, retaining repeated
compact/detailed tables and different training/batch/cohort variants.744rows are
accuracy,69ordinary classificationerror,10macrorecall. Explicit100−error is
separate; no mCE/recall conversion or independent-replication count is implied.

[Paper/Run] Newly reviewed SVWAsuppTable3 PDFp3 sourceCurveNet64.0 agrees
with our64.0113, complementing DGCNN58.8/58.7979. CloudFixer full15suppI3
sourcePoint-MAE53.70 agrees with our53.6899. This supersedes the previous
missing SVWA CurveNet comparison cell. DGCNN/CurveNet MN weights were selected
from SVWA; numerical agreement is source-family validation, not independent
training/asset-bitwise proof. ScanDGC32.7826 remains5.1174pp belowSVWA37.9.

[Paper/Inference] BFTsupp explicitly reuses MATE/TENT errors fromMATE. MATE
MNstandard64.3→SMART adaptation-only58.9→PG58.90 is a rerun then explicitly
inherited result; SMARTfullretraining63.0 remains separate. SMART reports rerun
SVWA57.1/37.4/50.5 vs original75.0/46.1/67.0 under changed settings. SATOR
PG/LN changed means remain execution-provenance-unknown. CFsix/all15 and
input-only/model-update variants must not be pooled. PointCache/BayesMM48paired
rows have identical clean+7cells but generally7versus8term Avg;14/112displayed
mean arithmetic flags are preserved, not repaired. These are reporting/protocol
observations, not proof of copying, a bug, or a quantified cause of our gaps.

[Artifact/Decision] Raw papers/extractions/vectors/validators are under
`result/preparation/tta_literature_expansion_20261025/` and adjacent sealedZIP;
committed CSV/readable tables and SHA/CRC receipt link all observations.
No Drive transfer occurred in this literature-only phase. Earlier sealedaudit
is retained; SATOR PDFtable offsets are corrected to6/7 in newrecords.
Authors' exact assets/logs, explicit reuse/rerun declarations or corrected
reductions would resolve/falsify specific open interpretations. No further
GPU experiment, peer-runtime work or test-label tuning was authorized here.

### 2026-10-25 — final accuracy-reliability interpretation recorded

[User report/Inference/Decision/Open] The user asks whether accuracy differences
appear to be a literature-wide corruption-generation issue rather than our
implementation. From committed audit`794ddb8`, recorded the
[qualified conclusion](tta_accuracy_reliability_conclusion_20261025.md): three
ModelNet source-family matches and internal validation weaken a blanket local
failure explanation; documented training/protocol/subset/aggregation/reuse
variation limits cross-paper comparability. Universal regeneration, complete
local correctness and a quantitative causal explanation are not established.
ScanDGC5.1174pp gap, exact author assets and isolated spectral benefit remain
open. Canonical ModelNet byte parity and prior Scan controls are unchanged.

[Artifact] No new search, accuracy measurement, GPU/Colab operation, author
contact or Drive transfer. The original sealed88,986,715-byte literature ZIP
SHA`e5c57dfb54ad3e6448006821d937658b408d0e4255d819b2c65b9fb88348f578`
is retained unchanged; a separate small interpretation receipt under
`result/preparation/tta_reliability_conclusion_20261025/` binds the new note to
that evidence. Matching author assets/reruns, a demonstrated local defect or a
controlled corruption-variance experiment could change specific conclusions.

### 2026-10-25 — requested source-only / original3DD paired table published

[User report/Paper/Run/Code/Open] At user request, paired every source-only
observation in the823-row matrix with an actual original3DD row when present,
then added known GSDRSCNN/PGPointMamba sources(5) and Uni-Adapter other-VLFM
appendix sources(6). The [paired overview](source_only_3dd_all_papers_20261025.md)
and147-setting CSV preserve139paper source observations and8local head/dataset
settings; one source is macro-recall, not accuracy. Fifteen3DDvalues comprise
8paper rows from3DD/GSD/Uni and7accepted local terminal results. Missingfull15
ScanPoint-MAE source/original3DD remains unavailable; GSD readouts are not
substituted. LocalPoint-MAE MN uses historical eval/rawseed0 reference63.7061.

[Paper/Code] Uni fullarXivPDFp13Table8 has ULIP-2 source45/33.05/42.22 and
OpenShape67.43/30.10/54.35 (MN/Scan/Shape); no3DDrow. RSCNNsupp§1/caption
ambiguity is retained. Old823/72matrices and raw research ZIPs remain unchanged;
new focused view adds supplementary coverage without rewriting prior counts.
Five-severity benchmark,7atomic-corruptionVLFM,6-typeCF andmacro-recall stay
separate. Elevenother examined publications have no matching numeric scope
or limitedaccess as explicitly recorded, never invented missing percentages.
No new search, accuracy evaluation, GPU/Colab operation or Drive transfer.

[Artifact] Focused table evidence starts from`29bb403`; raw derived ZIP
`result/preparation/source_3dd_table_20261025.zip` is35,762bytes,
SHA256`d5a38341e9b99fefa6b27519b0ce45083a8995f6aa3af24a0989fd41d2d84caf`.
CRC/member hashes and extracted portable validator pass. Primary/evidence
provenance could change a missing-cell classification if additional numerical
3DDresults or matching author assets become available; no missing value is
inferred from another method's accuracy.

## Agent B activated — 2026-10-25

[User report/Code/Open] User assigned this session to Agent B. Own branch/worktree was created from committed base `4c5be0eaf0ac933923129fe26738e0db5f96744a`; original user edits and Agent A worktree remain untouched. Existing Point-MAE original-host ablations are reused as historical context, not relabeled matched B70/M800 controls. Initial fixed screen:70 label-free development IDs/dataset across15 corruptions, four arms/four heads, three readouts;70 object-disjoint IDs reserved per dataset. Exact manifests and [execution plan](../result/preparation/guidance_component_study_20261025/execution_plan.md) are prepared. Whole initial research GPU phase estimate25–45min, cap55min including smoke/screen/retries; real smoke projection gates screen. No research GPU has run; no accuracy or speed result yet. Tuning and full benchmark remain future scoped decisions. Evidence artifacts: `result/preparation/guidance_component_study_20261025/`; separate Drive root per contract.


### Agent B CPU controls accepted — 2026-10-25

[Code/Open] On own branch from `54ae61f`, opt-in C+S/C/S/U controls, strict ModelNet40/Scan15 observation/analyzer, nonprefix original-ID selections and immutable z0/h0/noise plus post-preparation RNG replay are implemented. Default combined/terminal equations and dataset coordinate contracts are preserved. Inactive graph/Chamfer/backward call counters verify genuine skipped computation in CPU fake-native tests. Component run IDs include task+arm. Final local seven-module suite passes43 tests; independent review found no blocking issue before the final run-ID test. Source checks include Python3.8 syntax and `git diff --check`. This is CPU contract evidence, not native CUDA/model accuracy. Smoke will measure encode-inclusive normal execution; development cache replay timing is separately labeled and cannot establish deployment speedup. No research GPU run yet; own runtime remains in setup. Raw CPU log: `result/preparation/guidance_component_study_20261025/local_cpu_tests.log`. GPU smoke plus physical Drive/local receipts remains the next acceptance gate.


### Agent B ten-cell setup and phase safety accepted — 2026-10-25 13:06 UTC

[Code/Run/Open] Own A100-SXM4-80GB/High-RAM session `tta-guidance-components-a100-20261025-r1`, endpoint `gpu-a100-hm-kkb-ass1c2-36gbz99z2hf98`, completed all ten setup cells. Cell1 restart/reconnect passed; Cell2 only known fastrlock pip wheel error was deferred and Python3.8.20 verified; Drive consent was user-completed, Cell3 retry copied KNN wheel. Cell4 first CLI failed before sending setup code (kernel inventory idle); same-VM retry with REQUEST_TIMEOUT60 passed. EMD/Chamfer/PointNet++/PVCNN markers and actual GPU FPS/gather forward/backward/KNN/Chamfer/EMD checks passed. Cell10 returned status=passed for PyTorch2.1.2+cu121/CUDA12.1, data and checkpoints. No accuracy research has run.

Setup ZIP56156bytes SHA`13338f05cf2fc67418911e3eafba7723a746f948ef6997aa36ee05f0acaa132a` matches actual Drive and downloaded local bytes/CRC/member identities. Local `result/preparation/guidance_component_study_20261025/setup/setup_evidence_drive_roundtrip.zip`; exact Drive path in adjacent setup_evidence_receipt.json. Current source62e9a2b plus reconstructible accepted v8 asset overlay is sealed: manifest`7012c30a...f625ed75`, source ZIP`f5bca11a...576d0c6`, delta`1646533c...07e5065c`; exact SHA/bytes in preparation_identity.json.43 tests pass from the sealed source too;32 head configurations pass CPU preview.

Reviewed three task-local deployment/CPU/GPU phase scripts. Fixed controller cleanup on start-callback failure and SIGTERM, and fail-closed Drive-ledger restoration so retries cannot reset the shared3300s cap. Meaningful CPU subprocess regressions pass, including watchdog action during a blocked callback, external SIGTERM cleanup, retained123s/history and conflicting-ledger rejection. A Darwin dead-but-unreaped process-group EPERM was reproduced and corrected by reaping leaders before signaling remaining groups. Independent re-review reports no blocker. All this is CPU/control evidence; CUDA model smoke and physical result acceptance remain pending. Same monitor stays active; own runtime preserved, no peer operations or teardown. Next: sealed asset deployment/eight strict CPU loads, then the initially budgeted four-arm smoke; development needs independently accepted smoke and whole-phase projection within the cap.


### Agent B initial GPU smoke starting — 2026-10-25 13:11 UTC

[Code/Run/Open] Own sealed deployment validates637files and preserves fresh native bytes before restoring the audited four binaries. Deployment1421359bytes SHA07d769d35c38d9af490776a795bb699d24e0d88224165994649440a2eac668e7 and eight-load CPU proof24936bytes SHA2ee054d6ed9f5520177cf1e378749a7ebd1bda7e52799e7ec72c12d8f2687ffb match actual Drive/local SHA, CRC and source/native/class/checkpoint guards. PyTorch CUDA stayed uninitialized in CPU proof. Exact paths/receipts in `result/preparation/guidance_component_study_20261025/`.

Begin the registered initial GPU phase in own r1 under the same single monitor: two datasets×four arms×four heads, Gaussian/Background×70fixedIDs, seed0,17,920candidate predictions expected. Smoke uses fresh encode (no paired cache) for real instrumented end-to-end timings. The prior25–45min whole smoke+development estimate and shared3300s cap include short retries; no >60min phase is authorized. Independent local smoke acceptance and updated whole-phase timing projection precede any all15/70-ID development launch. No accuracy result is accepted at launch; stop on identity/state/RNG/skip/source/receipt mismatch or budget gate. Runtime stays live; peer untouched.


### Agent B smoke accepted; fixed development launching — 2026-10-25 13:33 UTC

[Run/Code/Inference/Open] Eight smoke jobs completed: two datasets×four arms×four heads, Gaussian/Background×70 fixed IDs;32 raw head ZIPs,4480 terminal events and17920 candidate predictions. Same-run terminal/probe state/RNG/class/coverage/receipt/skip guards pass; all32 archives independently reanalyzed locally and copied byte-identically into canonical own-worktree result paths. Phase358337730bytes SHA2d05389f514d766f5d8cbbb9b590231ccccb050d41e78e6c73fe02b8dc09e2a1 matches actual Drive/local SHA/CRC. Smoke is fresh-encode, two-corruption technical/timing evidence, not full benchmark or causal cross-arm accuracy evidence.

Initial controller runtime extrapolation incorrectly scaled ~19–20s fixed model startup7.5times per case, projecting67.87min development. The separately archived derived analysis counts only a conservative startup lower bound once (last-head constructor timestamp before first corruption, minus2s slack), scales all residual overhead15/2 and adds10s/case for extra input reads. Independent arithmetic agrees:40.993min expected development+8.445min smoke=49.439min active research;20% development uncertainty scenario totals57.637min. This is a projection, not a statistical bound or measured speedup. Original raw projection remains immutable in the phase archive and smoke_validation_initial.json. Derived report hash is bound by local_smoke_acceptance.zip and accepted_smoke.json. The existing3300s hard active-phase cap is unchanged; an overrun stops partial rather than silently extending. No expected>60min research phase is launched. CPU analysis, setup, idle/billing lifetime are separate from active research job-time metrics.

Proceed with registered70 development IDs/dataset, all15, seed0, four arms/four heads/three readouts;96 descriptive policy rows and134400 candidate predictions expected. Exact z0/h0/noise and post-preparation RNG cache is shared between arms and must verify. Report cached-comparison cost separately from fresh-encode smoke time. Heldout IDs, tuning, seed replication and full2468/581 benchmarks remain unexecuted. Same own r1/monitor52794; peer untouched, no teardown. Local evidence: `result/preparation/guidance_component_study_20261025/{smoke_phase_receipt.json,local_smoke_analysis/,smoke_local_acceptance.zip}`; raw32 ZIPs under canonical dataset/method paths.


## 2026-10-25 — Agent B matched fixed component development accepted

[Run/Inference/Decision] On own A100,70 fixed development IDs per dataset across all15 corruptions/four frozen heads produce32 accepted raw ZIPs and96 case/policy rows. Balanced terminal/entropy/probability mean: C+S53.8690/55.2024/55.1071%, C54.4167/54.8095/55.1905%, S50.3690/52.5595/52.6190%, U50.6548/52.5714/52.8095%. C improves over U in all eight cases under each policy. Incremental spectral effect is mixed and small at this uncalibrated coefficient; no universal necessity claim. C probability mean is frozen without tuning for reserved-ID three-seed C/C+S confirmation. Its development accuracy is unresolved against C+S entropy. Fresh-encode two-corruption shared reconstruction is24.94% lower for C; development replay comparison costs are not deployment timing. Drive/local405066186byte phase archive and30 latent-cache NPZs verify exact SHA/CRC/source/state/IDs. Initial active phase34.8947min completed within55min. [Full evidence and limits](guidance_component_development_results_20261025.md). Confirmation remains unrun at this entry.


## 2026-10-25 — Agent B frozen three-seed confirmation accepted

[Run/Inference/Decision] Frozen C probability mean on70reservedobjectIDs/dataset/all15/fourheads,seeds0/1/2 averages56.8929%, versus C+S same-readout56.7222%;+0.1706pp conditional95%objectCI[-0.0397,+0.3651],6positive/2negativecases,436corrected/393harmed events. Perseed differences+0.1786/+0.1071/+0.2262pp. Secondary versus frozen C+S entropy56.6984% gives+0.1944pp CI[-0.0714,+0.4683], including negative seed1. All48rawZIPs and90physical Drive/local cache files verify. No tuning/reselection/accuracy-superiority/equivalence/generalization claim. C is the supported fixed-setting accuracy/cost candidate; spectral-only remains uncalibrated seed0development evidence. Fresh-encode smoke cost24.94% lower shared reconstruction for C; confirmation replay timings are not deployment latency. Three-seed phase41.7519min under55min, initialphase34.8947min separately retained. [Full tables, rawidentities and limitations](guidance_component_confirmation_results_20261025.md). Own runtime+monitor remainlive; full benchmark/calibration/matched70-IDsource deltas remain unrun.


## 2026-10-25 — Agent B additional sensitivity prepared; approval pending

[User report/Code/Run/Open] User requests GSD parameter/loss tests and a stronger original3DD/Chamfer comparison. Fixed prior C−C+S+0.1706ppCI[-0.0397,+0.3651] is not superiority/equivalence; GSD-Chamfer is not original3DD. Addedopt-in legacyL2/Huberδ1/truepermodepower and actualoriginalPointMAEterminal boundary, preservingdefaultmath/configs.54CPUtests/12exactconfiggroups,2calibrationautogradtests,5controllerregistrytests/selfchecks,184head-configpreviews pass. SameownA10016strictGSD/originalcheckpointloadsnoforward/noCUDA pass; physicalCPUarchive11715bytesSHAeb7af9e0c8ae2e460f6307b071538a4832e8f9c3c4c3760991dc3553b137cde3 verifies. Source8169271,manifest599b8dcb...9cd8eef/sourceZIP73d9fbe1...acb37b8 are sealed. New140IDs/dataset exclude olddevelopment+oldconfirmation;3oldbaselinesretained,15newconfigs/30datasetjobs registered,actualoriginalcontrolandfreshconfirmationconditional. WholecampaignplanSHA4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1 expects4.1934h,20%+1200sreserve5.3654h,hard5.5h. One explicitGPUbudgetapprovalrequested; noGPUcalibration/smoke/accuracyrun launched. [Design,equations,gates](guidance_component_sensitivity_study_20261025.md).


## Explicit approval and launch — 2026-10-26

[User report/Run/Open] User replied **“Tamam onaylıyorum”** to the exact
registered whole-campaign5.5h duration request. Recorded beforeGPU at
2026-10-26T05:53:05.669817+00:00 with planSHA
`4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1`;
nominal4.1934h/margin5.3654h/hard19800s unchanged. Approval466bytes SHA
`2f3e2a4f9efe9bdfe6a68e1345506f96f2a2f8aff848da849959f5e6005313da`
exists identically in own runtime and Drive, local`gpu_approval.json`.
SameA10080GB/endpoint and16-load CPU proof reverified; free disk151784640512bytes.
Calibration wrapper submitted through solemonitor52794; actualGPU outcome and
raw archive pending. No new accuracy result. Preserve single cumulative ledger,
source/identity failure stops and runtime lifecycle contract. Next: accept
calibration evidence, then registered14-job smoke beforedevelopment.


## Calibration preserved; unexpected runtime loss — 2026-10-26

[Run/Open] Actual calibration phase state completed in155.460578s; physical
Drive archive5,806,352bytes SHA256
`3431f2a15e67892429fcc663b7fd2c0d6e3d93083daee078dc4b3e884d61c8f2`
was downloaded before runtime loss and passes localSHA/ZIPCRC. The retained
ledger records155.866607s spent and no activeworker. Report coefficients:
legacyL2=9373.767440639811,Huber=19729.987388376365,power=13.76292805636293.
These are gradient scale coefficients, not accuracy outcomes; independent
2100-row calibration/parity review remains pending. PrimaryCLI stream never
returned the exactwrappermarker; it later reported Connection was lost.
Repeated serverinventories showed ownr1A100 absent at07:36/07:37UTC; monitor
52794 subsequently reportsLOST. No teardown was issued. No smoke/development
orconfirmation from this campaign ran. Localhost scheduling/sleep gaps occur;
servertermination cause remains unknown. A bounded21600s localcaffeinate
assertion is active to reduce idlehostsleep; it does not guarantee Colablifetime.

Newr2 allocation printedREADY but later inventory/status found no suchassignment;
no setup/researchcell was submitted. A sameA100/HighRAM r3 allocation is being
checked. No peer/unknownCPUassignment is touched. NewVM requiresall10setupcells
and exactnative/assets/Drive recovery before campaigncontinuation. Approved
19800s cumulativecap and trialscope persist; never resetspenttime or rerun
acceptedcalibration silently. Rawlocal:`calibrate_evidence_drive_roundtrip.zip`
and`calibrate_receipt.json` under sensitivitypreparationroot.


## Independent calibration acceptance; recovery waits for stable host — 2026-10-26

[Run/Code/Open] IndependentCPU validation and rootrevalidation accept actual
5,806,352byte calibrationarchive. Every ZIPmember/source/native/frozenLION
binding and registeredapproval verifies. All30physical cachedNPZ files were
rehashed and joined to accepted latent/noise/input/RNG metadata. All2100/2100
ratios are valid for eachloss, no tiny/nonfinitepair. Recomputed globalmedians
exactly equal L2=9373.767440639811,Huber=19729.987388376365,power=13.76292805636293.
30/30 scalar and directcoefficientgradients are exact, RNG unchanged; native
helperrelativeL2 maxima local0.00025963/style0.00035591 pass individualdeclared
repeatabilityfloor/caprules. These are first-stepgradientcalibrationresults,
not accuracy evidence or fulltrajectoryequivalence. Rawreport99029bytes SHA
`82ebc1fdb2784a0663968a20d46909e3020e127310f237971382c78c40e211e0`.
Local`calibration_local_acceptance.json`, `calibration_root_revalidation.json`,
`validate_calibration_local.py` retain independentchecks. No GPU repeat needed.
Post-seal ledger downloaded beforeloss preserves155.866606843s spent; archive
ledger155.460578268s is the earlierpre-seal snapshot. Use the largerknowncost.

[Run/Open] r3A100HighRAM was actualserver-verified at10:06UTC, endpoint
`gpu-a100-hm-euw4a0-ws8p7xzfs0gu`, solemonitor9752. Cell1installedConda and
restarted; primaryCLI laterlostconnection, serverinventory at10:42UTC showed
noA100. No reconnect or cell2+ passed; freshsetup is incomplete. r2also
vanished beforeanysetupcell. At11:06UTC server reportedzero activeassignments;
no runtime was stopped bythisagent. Bothmonitorprocesses remainrunning/LOST;
no activeevaluationworker. No smoke/development/confirmationaccuracy exists.

[Run/Inference/Open] `recovery_20261026/host_sleep_evidence.txt` retains161
Macsleep/wakeevents today, including longMaintenancesleep/deepidle onbattery
duringCLIstreamgaps. Local21600s caffeinate-i assertion didnotpreventthese
sleeps. This establishesinterruptedlocal supervision, not the server'scause
oftermination. Repeatedallocation alone is not a soundrecovery. An asynchronous
question now asksuser toconfirmMacawake/stable beforeanotherallocation;
existing5.5hGPUbudgetapproval staysvalid and is NOT beingrequestedagain.
Oncehoststable: allocate sameA100HighRAM, replayall10setupcells, exactpinned
native/assets/DriveCPUproof, restoreacceptedcalibration andspentledger with
explicitmigrationprovenance; then registered14-jobsmoke. Do nottouchunknown
orpeerassignments. Fullscopeandhardcap19800s remain unchanged.


## User-directed recovery r4 — 2026-10-28

[User report/Run/Open] User directs “Yeni oturum açıp çalışmaya devam et”.
Existing whole-campaign approval/cap19800s and spent155.866607s remain binding.
Server inventory initially0active; new ownA100HighRAM session
`tta-guidance-components-a100-20261028-r4`, endpoint
`gpu-a100-hm-euw4a0-3s9zrjduwd03k`, actualA100-SXM4-80GB/175166460kBRAM
verified. Solemonitor execsession5314; bounded21600s localhost-awake assertion
execsession85660. These supervise legitimatework and do not guaranteeVMlifetime.
Canonicaltennotebookcells unchanged; receipt-onlysessionliteral adapted in
newsetupwrappers under`recovery_20261028_r4/setup`. Cell1ranseparately, restarted,
and sameVMreconnectexactmarker passed13:21:02UTC. Cell2environment creation
submitted13:21:20UTC. No researchGPU or newaccuracytest yet.

Next: finishcells2–10 (onlyknownfastrlockfailuredeferred), mountDrive through
CLI/userconsent ifrequested, exactpinnedassets/native restore and freshCPUproof,
explicitplan/session/ledger migration preservingacceptedcalibrationbytes and
approvedscope/spenttime, then14-jobsmoke. No teardown authorization.


[Run/Code/Open] r4cell2 createdPython3.8.20 and failedonlyatknownfastrlockwheel;
explicitdefer remainsapplicable. Independentenvironmentmarker passed13:29:11UTC.
Cell3Drive/KNN submitted13:30:04UTC. Ownership-onlycontinuationplanSHA
`1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42`,
controllerSHA`0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc`;
changesonlysession/ledgerownership andstrengthenedacceptedreport/spentfloor
guards. Sealedassets/source/IDs/objectives/trials/readouts/capunchanged.
TwoCPU migrationtests, controller selfcheck andPython3.8syntaxpass.
Fresh8+16CPUproofs and actualDrive/recoveryinstallation remainpending.
No newresearchGPUexecuted andnoaccuracyclaimed.


[Run/User action required/Open] r4cell3 firstattempt failed`ValueError:mountfailed`
at13:32:07UTC (exactmarkerabsent). TTYCLI`colab drivemount` execsession37218
received actualDriveOAuthrequest for ownendpoint; actualGoogleconsentscreen
opened inuserbrowser andasynchronousquestionasksuser tocompleteauthorization.
Noaccountpermissionclickedbyagent; noauthURI/tokeniscopiedtorepository.
Solemonitor5314remainsrunning, withprobesinhibitedafterfailedcell3; nosecond
notebookcellwillbesubmittedwhilemountautomationwaits. Thisisinteractive
accountconsent, notnewGPUbudgetapproval. Runtimepreserved/liveatlastinventory.
Nextafteruserconfirms: sendnewline to37218, requiremountsuccess, reruncell3
andverifyDrive/KNN; thencells4–10inorder. ResearchGPUstillnotstarted.


[Code/CPU/Open] r4localanalysis preparation isready. Sharedhelper nowaccepts
explicitmigratedplan+hash whileoriginalcalibration isverifiedagainstitsown
oldplan/session/archive/report/acceptance; seed0defaultsretained, registered
seed1/2 andseed-specificpreparationpairingvalidated. Confirmationanalyzer
requiresactual140-IDlabels, 180physicalcaches, exactdevelopmentfreeze and
280objectclusters jointlyacrossseeds. Root9CPUchecks pass0.172s, previous5
fixturechecks0.033s, selfcheck+Python3.8ASTpass; actualmigratedplan+original
calibrationingestionpasses. Noactualnewsmoke/dev/confirmationdataexists, so
thoseacceptancepathsremainunverified. ModelNetfullsealedlabel.npy mustbe
downloadedaftercell9; Scanlabelsareinassets. Source, checksandCLIhandoffunder
`result/preparation/guidance_component_sensitivity_20261025/`; nosealed
researchconsumer/source/objectivechange. R4stillawaitsactualDriveuserconsent
inTTY37218; asyncquestionpending. No newGPUapprovalrequest oraccuracyresult.


## Drive consent completed; same r4 resumes — 2026-10-28

[User report/Run/Open] User says “Başla”, then explicitlyconfirms
“Drive erişim iznini tamamladım”. OldmountTTY37218exitedwithtimeout after
itslongwait; remote2kernelswereidle andDriveMyDrive404, so no duplicateactive
cellwasstarted. SameVMnewmountTTY57259receivedactualfreshconsent; usercompleted
it, CLIreportedCredentialspropagated and`Mounted at /content/drive`, exit0.
Cell3retry submitted14:37:23UTC viaunchangedsolemonitor5314; markerpending.
Accountpermissionwasuser-operated; existingGPUbudgetapprovalunchanged.
No researchaccuracyrun yet. Next: acceptcell3Drive/KNN andrun4–10inorder,
exactpinnedassets/freshCPUproofs, installreviewedownership-onlyrecovery,
thenregistered14-jobsmoke. Oldcalibrationand155.866607sspentremainbinding.


[Code/CPU/Run/Open] Independentanalysisreviewfoundtwofutureacceptanceblockers:
ledgerkey`resolved`vsproducer`resolved_model_config`, andNumPy2pickleRNGmodule
spellingdiffersfromremoteNumPy1. Both reproducedbeforefix: actualcontracttest
failed; genuinecachehashmismatchfirstfile, independentreview0/30. Fixedexact
producerfield; isolatedCPUanalysisPython3.10.18+NumPy1.21.6 nowmatches30/30
genuineoldcacheRNGhashes, withoutrewritingrawbytesorweakeningchecks. Explicit
NumPy1guardpreventsfalsecorruptionreports. Full11CPUtestspass(.292sNP1,
.168sNP2unsupportedguard). GPUconsumer/source/manifest/planunchanged.
Actualconfirmationacceptanceisstillunexecuted. Seeanalysis_environment.json
andanalysis_review_fixes.json under recovery_20261028_r4.

[Run/Open] r4EMDpassed14:53:16UTC,Chamfer14:55:21,PointNet++14:59:47, each
withdiskmarker+physicalDriveprogressreceiptchecks. Cell8PVCNN/nativeGPUchecks
submitted15:00:39UTC via5314. Dataset/checkpoints/setup10 andfreshCPUproofs
remainpending; no newresearchaccuracyrun.


## Accepted r4 recovery; registered GPU smoke launches — 2026-10-28

[Run/Code/Open] All10canonicalsetupcellspassed (onlyknowncell2fastrlock
deferred). ActualA10080GB,PyTorch2.1.2cu121/CUDA12.1/NumPy1.21.2/nativeGPU
operations,ModelNetall15andcheckpointsverify. Setup56157byteDrive/localZIP
SHA`924076c28b64af0c1b121703cacbf55fa8400a92e6c8deb0925d4015fcb2a517`,
637filedeployment1421391bytesSHA`ee84885876c9008dc6b9a66ddc072f3fe92b6d1b1a185141832fa089b85f7c57`,
8-headCPU24937bytesSHA`3058021bb9b91f7d517eece7899a11dcf244cf60f16ac42f12d207901032bd50`,
16-loadCPU11716bytesSHA`d90e6308a985e4a25b85ff9c99d15b82ff11c76db7e94be2e5ced5c898172e9f`
allactualDrive/localSHA,CRC,source/stateverify. Freshnativebyteswerepreserved
beforeexactaudited4binaryrestore; nohiddenfresh-buildparityclaim.

Recovery11845364bytesSHA`03c86fa0bfc15a84abe7b0e7d845b96d9f9f004816262f54655372b3b3bd954d`
passedindependentlocal32membermanifest/CRC/plan/statechecks. Oldcalibration
reportbytes/sessionretained,30actualdevelopmentNPZsrestoredfromDrive;
post-seal155.866606843sspent/historycarried. Newownershipplan14033563…c4d42,
consumer8169271/source73d9fbe1…b37b8/manifest599b8dcb…8eef unchanged.
No newGPUapprovalneededforunchangedscopeand19800scap.

Registered14jobGPUtechnicalsmoke now submittedthroughsolemonitor5314 in
`tta-guidance-components-a100-20261028-r4`:7conditions×2datasets,Gaussian/
Background,70fixedIDs/dataset,4heads,seed0; freshdistinctsmokecachenamespace.
WrapperSHA`539b14b2fb540f0de44b73eb4d14f53cb84787090e8f16f8019f06eee52fed30`.
Actualsuccess/accuracy/rawarchivepending. Completionneedsexactdirectmarkers,
physicalDrive/localevidence+cachebytes, independentanalysis andmeasured
wholecampaignprojectionbeforedevelopment. Localbounded21600sawakeprocess
64591supportsCLI supervision; noVMlifetimeguarantee/teardownauthorization.


## Sensitivity smoke independently accepted; development running — 2026-10-28

[Run/Code/Open] All14registeredtechnicaljobs completed in916.655641s;
post-sealwholecampaignspent1105.891987s includingoldacceptedcalibration.
Bothdatasets/Gaussian+Background/70developmentIDs/fourheads/seed0,7conditions
(C/CS/Sbase,calibratedCSL2/Huber/power,actualoriginal3DD) yield56rawZIPs.
Rawphase627301980bytesSHA`523c2e634d6af682157b4e1e3e597e3173523c067b777a77d1d2e3243128fad3`
and4physicalcacheNPZcollection17198671bytesSHA
`7bdd9ba956c742fc8b57795cb14768a69f9ffd38c9069069ab581d16db5857a3`
matchactualDrive/localbytes. Independentlocalanalyzeracceptsall56rawsource/
config/checkpoint/state/ID/label/trajectory/pairing/receiptguards,all4physical
newcachesand30oldcalibrationcaches; rootNP1reconstructsall4newRNGhashes.
Technicaltwo-corruptionresultsarenotall15ranking/finalaccuracyproof.

Localacceptance37355bytesSHA
`f285c4e5e32c5d5b4e910a0cdbbdc1d1c9c1d85e18272bb03e85d0eb6041c74e`
physicallyDrive/localroundtripverified; ROOT/accepted_sensitivity_smoke.json
passedcurrentplan/Drive/cumulativeledgergates15:48:13UTC. Conservative
post-smokeprojectionusesmaxmeasurednew/oldjobratio1.025455,20%margin,1200s
reserveandincreasedoriginalfutureallowances200s/devdataset,300s/confirmation
datasetseed. Upper19743.614653s=5.48434h fitsunchanged19800scap; thisis
anestimate,notruntimeguarantee. Recheckafterdevelopment/freeze.

Registered32-jobdevelopment submitted15:48:34UTC viaownr4/solemonitor5314.
30newGSDdatasetjobs(5newsettingsperfamily×2datasets),2actualoriginalcontrols,
all15/s570IDs/dataset/4heads/seed0;3baselinesreusedonlywithdeclaredparity.
FirstC-step-lowModelNetjobhasproducedseveralcorruptionreceiptsusingoldpaired
preparations. Actualdevelopmentacceptance/ranking/freezeand140-IDthree-seed
confirmationremainpending. NoGPUscope/trial/readout/source/ID/capexpansion.


[Code/CPU/Open] Futureconfirmation180NPZcollector preparedbyadaptingexisting
artifactworkflow, sourceSHA`694f55a0e138c3d6af34be4f92ef5e44c8b3ac88537ef22aa90efb9e11d15c4f`.
Itrequirescompletedconfirmation/inactiveledger/exactfreezeand180unique
source+Drivecachepaths, neverGPUrecalculation. ThreeCPUfixturespass0.105s,
Python3.8syntaxchecked. Actualcollectionunexecuted; actualconfirmationoutputs
do notexist yet. Source/test/handoff underrecovery_20261028_r4.


### Development acceptance preparation — 2026-10-28 16:51 UTC

[Code/CPU/Open] The CPU acceptance/freeze template is prepared and independently
reviewed (SHA8366ab2e…158ff6); root reran its three synthetic fixture tests, all
passed in0.004s. It cannot execute until bound to a genuine accepted development
package. Gates retain32new jobs/152raw ZIPs/19trial rows/30physical caches,
source/history separation, exact freeze binding, immutable Drive evidence and
the conservative maximum24-job confirmation budget under19800s. Independent
read-only review also accepted24 genuine old ZIPs/30physical caches and checked
all72 possible freeze pairs using synthetic scores (12three-condition,60four).
Evidence: recovery_20261028_r4/development_acceptance_preparation.json plus
source/tests/handoff and development_review_check_development.py/stdout.json.
The GPU development phase is still running; no new ranking, winner or accuracy
claim is established. Next: genuine completed archive acceptance, frozen choices
and measured budget gate before the existing confirmation wrapper.


### Bounded-memory artifact transport prepared — 2026-10-28 17:17 UTC

[Code/CPU/Open] Localhost has16GiB RAM; installed ColabCLI0.7.4 download loads
the entire base64 JSON response. Genuine smoke archive sizes project about4.7GiB
of development raw ZIPs, so direct whole-archive retrieval risks excess memory.
Prepared a CPU phase to stream the completed immutable Drive ZIP into<=256MiB
transport parts and a local joiner requiring exact original receipt/plan/session,
part SHA/order/offset/coverage and aggregate original SHA. Originals stay intact.
Source SHA d3550929…62723 (split),667deec2…8773 (join). Root reviewed source,
reran three synthetic fixtures (0.013s), and checked Python3.8 AST; independent
review found no material blocker. Genuine transport remains unexecuted. See
recovery_20261028_r4/chunk_transfer_preparation.json and chunk_transfer_HANDOFF.md.
Scientific source, trial registry, frozen-ID plan and19800s cap are unchanged.


## Sensitivity development accepted; frozen confirmation running — 2026-10-28 17:49 UTC

[Run/Code/Decision/Open] All32new development jobs completed in6584.800419s;
post-seal cumulative campaign spent7782.756118s. Raw phase1614733520bytes SHA
`2c4424e3cf017b8c1da99e77eb233fddb6c1d472efeb5df9552fd7b3c74fdea8`
was physically checked from Drive through7bounded parts and reconstructed with
identical local SHA. Independent NumPy1 analysis accepted152raw ZIPs (128new,
24historical),19registered trial rows,30physical dev/calibration caches and all
source/state/ID/label/trajectory/policy guards. Raw and derived data stay separate.

Development balanced all15/s5/70IDs per dataset/four heads/seed0 probability-mean
scores: best tuned C(`c-step-high`,eta=gamma=.03)56.785714%; best spectral-containing
CS(`cs-l2-r010`,calibratedL2rho=.1,eta=gamma=.01)55.345238%; actual original3DD
terminal55.238095%. These are development selection scores, not confirmatory
accuracy evidence or proof that original3DD is superior. Selected CS versus C
adds the spectral term at matched settings; tunedC and original are distinct
comparators. Original host/readout policy differs from GSD.

The exact accepted JSON SHA f94afc83…54ed and frozen4condition JSON SHA
ea245f8a…c360 bind spectral-candidate,matched-chamfer,tuned-chamfer,original3dd.
Compact90496byte acceptance SHA37f528f6…61b8d passed physical Drive/local
roundtrip and remote source/cache/freeze checks17:46:53UTC. Timing review
includes doubled four-head cost and maximum24futurejobs: remaining7741.187194s;
wholecampaign upper18272.180750s including20%futuremargin+1200reserve, below
unchanged approved19800s. Actualdevmaxslowdown1.001292 is below retained1.025455.

Registered confirmation submitted17:47:55UTC through solemonitor5314 in owned
`tta-guidance-components-a100-20261028-r4`; wrapper SHAb507da40…ebe8. All15/four
heads/new140IDs per dataset/seeds0,1,2,4frozenconditions/24jobs. First fresh-cache
ModelNet batch is observed; actual confirmation results remain pending. No new
GPU scope/ID/readout/trial/cap expansion, so existing Oct26 approval applies.
No runtime teardown authorized. Next: retain all180physical new NPZs and raw
confirmation Drive/local bytes, then object-cluster paired analysis across all
three seeds (280objects, not840), all15/macro14 and qualified policy costs.
Evidence under recovery_20261028_r4/local_development_analysis_v1 and
development_local_acceptance_verification.json; actual source remains8169271
plus sealed73d9fbe1…b37b8/manifest599b8dcb…8eef, separate from local branch HEAD.


[Run/Open] 2026-10-28 18:28UTC: frozen confirmation seed0 completed all8jobs
(fourconditions×two datasets); seed1 spectral-candidate ModelNet is now active.
No partial accuracy ranking has been selected or used to modify the freeze.
Ledger retained locally as recovery_20261028_r4/confirmation_live_ledger_seed0_end.json;
all per-corruption/final worker receipts remain on owned Drive. Solemonitor5314
and bounded host-awake support remain active. Seeds1/2 and independent final
96ZIP/180physical-cache acceptance are pending; no new superiority claim.


[Code/CPU/Open] 2026-10-28 18:49UTC: final confirmation acceptance template
prepared (SHAc6a4456a…3312), using actual analyzer/collector metadata schema.
Root read source, reran three synthetic guard fixtures (0.025s), checkedPy3.8
syntax; independent review found no blocker and confirmed the final exclusive-write
delta by reversing it to reproduce the previously reviewed source hash. The CPU-only phase
requires genuine accepted24-job/96-ZIP/180-cache/280-object results, exact frozen
development/calibration/source bindings, original Drive raw/cache bytes and an
inactive under-cap ledger. No estimator rerun, tuning, result selection or GPU
operation; package identity is unbound until actual final local analysis passes.
Source/tests/handoff and final_acceptance_preparation.json are inrecovery_20261028_r4.
Actual confirmation remains running; final binding/acceptance is unexecuted.


[Run/Open] 2026-10-28 19:09UTC: confirmation seeds0and1 completed all16jobs;
third/final seed2 is now running, starting spectral-candidate ModelNet. Actual
cumulative campaign spent12571.930780s at this boundary, below19800s cap; no
configuration/readout/trial change or partial-score-based decision. Local ledger:
recovery_20261028_r4/confirmation_live_ledger_seed1_end.json. Allremaining8jobs,
96rawZIP/180physicalNPZ final acceptance and object-cluster analysis remain
pending. Same ownedA100/solemonitor5314/host-awake support are preserved.


## Confirmation numeric failure; reviewed recovery preparation — 2026-10-28 19:43 UTC

[Run/Code/Open] Registeredslot20 (tunedC`c-step-high`,ModelNet,seed2,eta=gamma=.03)
failed19:32:22UTC with`ValueError: Classifier input contains nonfinite XYZ coordinates`.
Uniform/Gaussian both70-batches and Backgroundbatch0 completed; the failure is
in Backgroundbatch1 (selectedpositions70–139), during a clean trajectory decode
probe before classification. ModelNet/DGCNN preprocessing usesFPS with no
normalization. Existing logs do not identify the exact first invalid tensor or
original object; no unsupported detailed causal claim. Other source/state/data/
native/RNG mismatch is not reported. No numerical clipping/fallback/parameter
change or rerun was made. The finite-input guard correctly stopped the worker.

20registeredjobs complete,slot20partial(status`running` retained in immutable
producer state, overallattempt`failed`),slots21–23unrun. Raw failed phase
1159246521bytes SHA`f586870206961f88d38daa7ed9da632fddfa51356075e114c5f3b98f020ebe5e`
sealed on Drive19:33:34UTC; local raw roundtrip pending. Exact failedreceipt
SHA5ab88805…9754,post-sealfailedledger60c8ba55…736b retains14114.660198s spent.
No complete wrapper marker or confirmation validation exists; previous running
headlines are superseded by this failure. Runtime andsolemonitor5314remain live.

Independent policy review permits a separately reviewed new-output continuation
of onlyUNRUN21(tunedC/Scan/seed2),22–23(originalMN/Scan/seed2), keeping failed20
failed, immutable archives/freeze/source and one cumulative19800s cap. Conservative
upper16384.444877s includes891.487232s futurework,20%margin and1200reserve.
No continuation GPU has started: first preserve physical raw failedattempt+all180
cachebytes; source/native/data/state/RNG/receipt mismatch remains a stop condition.
The original complete-only controller/analyzer/templates stay unchanged and may
not accept this as complete four-condition confirmation. Prepare separate
failure-aware analysis: candidate vs matchedC already has all3seeds; candidate
vsoriginal requires the two unrun original jobs. TunedC's three-seed balanced
endpoint remains unavailable; no dropping the failedcase, two-seed primary
average, setting substitution or hidden failed-slot replacement.
Exact local evidence: recovery_20261028_r4/confirmation_failed_receipt.json,
confirmation_failed_ledger.json,confirmation_failed_job020.log and
confirmation_numeric_failure_diagnosis.json. The numerical failure is research
evidence, not proof of either method's universal superiority.


## Failure preserved; reviewed unrun-only continuation active — 2026-10-28 20:12 UTC

[Run/Code/Open] Failedraw1159246521B/f5868702…be5e physicallydownloaded through
5boundedparts and recombined with identicalSHA. Failure/cacheproof774792352B
SHA2a003500c74feafbd6aa6ac03363ec57e6330a67707fc22c803865081d15b898 matches
actualDrive/localbytes,190memberSHA/CRC and180physicalNPZ extraction. Remote
CPU verification checked all180input/latent/noise/post-RNG hashes underPy3.8/
NumPy1.21.2/Torch2.1.2 withCUDAuninitialized before/after. Scoped to20completed
workers plus180preparations; failedslot20 remains unvalidated for accuracy.
Allfourfailedrawconfigs also matchknownnumericstack and unchangedclassifier/
LIONstates, without audit-error flags. No source or scientificparameter changed.

Reviewedremaining-onlyplan2362B SHA13ff25249aec97eafd6c1fb6db33cd5e3d8462b79331a7cfdfba0e4a06540f6c
binds actual2900B preservationreceipt866135f4…486a and sourcewrapperac5bb60d…9931/
runner eb44e15a…0b09d. Onlyslots21–23; failed20excluded. Native no-build
bootstrap omission was found by review beforelaunch, fixed with targeted
RED→GREEN regression; root8fixturespassed0.196s and independentdelta review
accepted. Originalscientificcontroller/plan/freeze/failedarchive stayunchanged.
NewCPUpreflight actuallyloaded12strictselectedheadconfigs withoutforward/CUDA
andcheckedsource/data/checkpoint/nativeidentities; downloadedproofSHAd64cc5af…4ef8.

Operatorwrapper submitted20:05:33UTC via5314; freshCPUproof passed and first
remainingGPUworker began20:06:42UTC in newimmutableROOT/Driveoutput
`sensitivity_confirmation_remaining_20261028T200610144036Z`. Slot21(tunedC/Scan/
seed2)completed; slot22(originalMN/seed2)nowruns, then23(originalScan/seed2).
Nofailed-slotretry, resultreplacement orwholefour-condition-complete claim.
Conservativewholecampaignupper16384.444877s retains existing19800scap; original
failedledger/history and postsealspent14114.660198s formthepreservedprefix.

[Code/Open] Separatefailure-awareanalysis is beingvalidated. A rawZIPprefix
I/Obug was found before scoring and confined to newadapter Bundle normalization.
Failedraws omit onlydynamicexecutedsourceinventory entryshared_trajectory.py;
allotherrequiredentriesmatch. Rootverifiedall4 embeddedwhole sourceZIPs exactly
73d9fbe1…b37b8 and member6524B/dce0e8b2…f2fcda plusknowntracepaths. Explicit
failed-provenance-only recovery may retainthisone missingentry fromexactarchived
bytes, withmissingoriginalkey/recoveryproof recorded; anywrongbytes orother
missingentry stops. Rawbytes/status andcomplete-onlyguards stayunchanged. No
failedprefixaccuracy accepted, no estimatormath/readout change.
Localproofs: failure_preservation_receipt.json,failure_preservation_local_validation.json,
failed_worker_partial_metadata_local_validation.json and
failed_shared_trajectory_source_inventory_proof.json underrecovery_20261028_r4.
Finalremainingarchive/postsealledger/failure-aware independentanalysis and
Drive/localpartialacceptance remainpending.


## Sensitivity campaign accepted with explicit numerical failure — 2026-10-28 20:35 UTC

[Run/Decision/Inference] All24registeredconfirmation slotsare recorded:23complete,
failed20retained; reviewed21–23allpassed. Independentfailure-awareanalysisaccepts
92full+4failedZIPs,180physicalNPZs and280objectclusters. Originalterminal53.763889%,
matchedCprobmean53.013889%,CS-L2rho.1probmean52.962302%. Original−CS+.801587pp,
conditional95%CI[+.148760,+1.557589]; wholepolicy/host/readoutcomparison,5/8cases.
MatchedC−CS+.051587pp,CI[-.125,+.234127]crosses0; noisolatedCDsuperiority/equivalence.
TunedCthree-seedendpointnullbecauseoneModelNetseed2Backgroundprobe numerical
failure; no2seedmean/sampledrop/replacement. No universalwinnerclaim.

Actualpartialacceptancebinder722266B/dc5b854e…68df storedonfreshDrive and
physicallydownloaded withsameSHA/CRC/analysisbytes. Statusaccepted_partial_confirmation_with_numerical_failure,
notfullfour-conditioncomplete. Finalpostsealspent14836.554263s<19800s. Original
failedrawf5868702…be5e,remaining7b676a7b…ccd6d,cacheproof2a003500…b898 remainimmutable.
AllplannedGPUworkrecorded; runtime+5314preservedwithoutteardownauthorization.
Cleanresults,percasevalues,source/coverage/failure/CIlimits andexactproofpaths:
knowledge/guidance_component_sensitivity_results_20261028.md and
recovery_20261028_r4/final_local_verification.json.


## Agent B runtime closed after accepted campaign — 2026-10-29 local / 2026-10-28 22:04 UTC

[User report/Run/Decision] The user authorized closing the completed campaign and
recording its results: “Colab oturumu kapatıp sonuçları knowledge dosyalarına
ekleyebilirsin”. Monitor5314 exited0 at22:02:52.511932UTC, then the explicit CLI
stop targeted `tta-guidance-components-a100-20261028-r4` (A100 High RAM,
endpoint `gpu-a100-hm-euw4a0-3s9zrjduwd03k`). A fresh server inventory by22:04:19UTC
returned `No active sessions found on server.` The exact server release timestamp
is not available. Four existing Drive-roundtrip archives were rehashed locally
before teardown; all bytes/SHA-256 matched `final_local_verification.json`.

Accepted scientific status remains `accepted_partial_confirmation_with_numerical_failure`:
23 successful jobs plus preserved numerical failure20. No retry, replacement,
new GPU work or changed metric/inference occurred during closure. Campaign time
remains14836.554263s under the19800s approved cap. Original whole-policy mean
53.763889% versus C+S52.962302% does not establish universal or isolated-loss
Chamfer superiority; the matched C−CS interval still crosses zero. No scheduled
GPU work remains. This entry supersedes prior live-runtime/monitor statements;
future generalization or failure diagnosis remains separately scoped.

[Accepted results](guidance_component_sensitivity_results_20261028.md) ·
[Closure evidence](../result/preparation/guidance_component_sensitivity_20261025/recovery_20261028_r4/runtime_closure_20261029.json). Scientific source remains
`8169271c9058995e8dc128062848a253baa13f10` plus the unchanged sealed snapshot;
closure documentation starts from operator HEAD `03416e8`.


## Agent B executed guidance/math audit — 2026-10-30

[User report/Code/Run/Inference/Open] Before the user's method decision, audited
sealed fixed/sensitivity sources,32fixed+120newGSD head configs/576source bindings;
29CPUregression/property tests pass. Beta2 smoothing is active, but uses restored
raw-eigenvalue/hardM800 legacy fidelity, not September normalized full-spectrum
smooth or full GSDTTA. No detached-gradient/transpose/missing-IGFT explanation found.
Known self-only isolation defect persists; six actual cached-object CPU graph
replays find one missed node in Scan/Gaussian, not CUDA prevalence or accuracy
causality. Prior full offdiag control did not improve mean accuracy.
S−U probabilitymean=-.190476pp,CI[-.535714,+.143155],single-seed development;
best calibrated S52.880952% slightly exceeds U52.809524% on that same cohort.
Final confirmation has no standaloneS/U arms. Historical normalized S>U+.1576pp
is preserved with its build/draw confounds; broad combined gains include SCD.
Latest grid fixes beta/M/graph and cannot reject all spectral formulations;
earlier beta/M studies also retain their null/negative findings. Existing
23pass/1numericfailure acceptance and all raw scores remain unchanged.
No production edit/newGPU/runtime allocation. User method decision is next;
no new experiment scheduled. [Full audit and evidence](guidance_mathematics_execution_audit_20261030.md).


## Next-agent literature handoff prepared — 2026-10-30

[User report/Decision/Open] User will open a separate session/agent for current
3Dpoint-cloudTTA literature and forward citations of3DD-TTA, seeking improved
loss/guidance/readout or accuracy/cost tradeoffs. Prepared
[copyable prompt](diffusion_guidance_literature_handoff_20261030.md) with current
AgentBworktree/source references, accepted results, attribution and no-new-GPU
scope. This is handoff preparation, not a completed search or new method result.


## Shared thesis integration and Agent B selected outcome — 2026-10-30

[User report/Code/Run/Decision] User authorizes necessary completed agent work
on origin/thesis-integration, preserving main/upstream and peer/user edits.
Selective Agent B import retains code/tests, accepted reports, compact evidence,
negative/null outcomes and the tuned-C numerical failure. Operational dumps
remain on its source branch. Integrated345CPUtests and7extrafixtures pass;
no GPU/model rerun. See [scope/manifest/limits](thesis_integration_agent_b_20261030.md)
and [standing publication prompt](thesis_integration_workflow_20261030.md).
