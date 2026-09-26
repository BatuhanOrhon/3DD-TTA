# GSD smooth-spectrum implementation review — 2026-09-26

**Latest status:** The historical findings below are repaired. A second
independent review and the expanded suite pass **111 CPU tests**. See the
second-review record at the end. Calibration/common-draw evidence and Colab
smoke remain required before interpreting the accuracy pilot.

**[Code/Run] Verdict:** The intended quadratic spectral metric is represented
correctly, but the working tree is not ready for a Colab pilot. The earlier
claim that v1 was preserved is superseded by this review. No implementation
source was changed during the review.

**Provenance:** `gsd-smooth-spectrum`, HEAD
`9650770f75cf0c37e1e16b873bd6b8ddd0a4276e`, uncommitted v2 changes.
Scope: compare the working tree with
`gsd_guidance_math_20260926.md` section 9 and the v2 test plan. Evidence is
local CPU unit/synthetic testing, not a model or accuracy experiment.

## Findings

1. **[Code/Run, blocking] Shared basis allocation breaks v1 and v2.**
   In `graph_spectral.py:144`, `basis` is allocated while `rank=0`. After rank
   selection it is never resized before `basis[active] = eigenvectors[:, :rank]`
   at line 183. Normal active graphs raise a shape mismatch; the rank-one
   corner can silently retain an empty basis and return zero v1 loss. The base
   implementation allocated the basis after rank selection. Both new profiles
   execute this code, even though smooth loss does not need the hard basis.
   Restore allocation after final rank selection, then verify both versions.

2. **[Code/Run, experimental validity] Launcher omits the agreed v1 arm.**
   `eval_gsd_smooth.py:46` maps `baseline` to `3dd_original`, while both other
   arms use `gsd_latent_spectral_smooth_v2`. The knowledge design requires
   A=v1 hard-M, B=v2 hard-M, C=v2 smooth. Nine commands are generated, but none
   runs v1. SCD-only is a useful additional control; it cannot substitute for
   the v1 comparator. Reconcile launcher, implementation plan and test-plan
   tables. Retaining SCD-only as a fourth arm means 12 bundles / 24
   corruption-seed cells for one beta, rather than 9 / 18.

3. **[Code/Run, numerical correctness] PSD tolerance mixes spectral units.**
   `graph_spectral.py:189-191` divides eigenvalues by mean active degree `mu`
   but compares them against a tolerance derived in raw-Laplacian units.
   Compare raw eigenvalues with the raw tolerance, or divide the tolerance by
   `mu` as well. After bypassing finding 1 in memory, a two-node fixture with
   distance 1, delta=.25, k=1, gamma=.6 has `mu=0.00033546262790251185`.
   Injecting a raw zero-mode roundoff of `-1e-8`, within raw atol `1e-7`,
   incorrectly raises `graph Laplacian has a materially negative eigenvalue`.
   This is a controlled mocked-eigensolver test, not evidence of occurrence
   in model data. For `mu>1`, the mismatch instead weakens rejection.

4. **[Code/Open, verification gap] V2 tests remain a plan.**
   Existing tests exercise v1; no dedicated smooth-profile, v2 protocol or
   new-launcher regression cases were added. The new plan lists these cases
   but does not implement them. This review's isolated operator check is
   narrower than the full planned coverage and does not validate CUDA.

5. **[Code/Open, protocol readiness] Calibration is still unspecified.**
   Requiring `--hard-weight`, `--smooth-weight` and `--beta` is not evidence
   that B/C have matched guidance scales. Declare calibration data, aggregate
   statistic, beta candidates and selection rule, then retain calibration
   evidence before pilot scoring. Automatic calibration was explicitly out
   of implementation scope; this is an unresolved launch prerequisite, not
   a claim that an automatic calibrator had been implemented incorrectly.
   Direct `run_baseline.py` v2 use also inherits weight=1 when omitted.
   Equal seeds alone do not establish common random draws: verify their
   pairing before interpreting paired effects.

## What is consistent with the mathematical decision

**[Code]** Reference XYZ and graph/filter are detached and static. The graph
rules are shared with v1. Smooth weighting covers all active modes, using
`F = U diag(exp(-beta * lambda / mu)) U.T`; excluded rows are zero padded.
The implementation computes `trace(E.T F E)/(3*N)`, summing samples. It does
not square `F E`, so it does not accidentally double beta. The hard v2 profile
uses the v1 subspace with the new point-count denominator. The TTA host
retains SCD, gradient propagation through the clean prediction to both local
state and conditioning, and the original decoder style. V2 is opt-in and
restricted to smoke/pilot. These structural properties do not override the
blocking shared-function regression.

**[Inference]** At fixed graph/rank `m`, v2 hard loss and gradients are `m/N`
times v1's, before the external coefficient. At nominal m=100,N=2048 this
is about 0.0488. Thus preserving alpha does not preserve guidance strength;
calibration matters, and a normalization change alone is not evidence for
better accuracy. Boundary expansion can change the actual m per sample.

## Verification evidence

**[Run]** Actual working tree:

```powershell
python -m unittest discover -s tests -v
```

92 tests: **69 passed, 1 failed, 22 errors**, 2.753 seconds. Errors arise at
the basis assignment; the failed two-vertex projector test detects the
rank-one empty-basis case. Direct four-vertex float64 probes independently
raise the same shape mismatch for v1 and smooth v2.

**[Run]** Diagnostic-only Python module loaded from the same source with one
in-memory insertion immediately before the assignment:

```python
basis = reference.new_zeros((count, rank))
```

With that module substituted via `sys.modules`, all **92 existing tests
passed**, 4.280 seconds. This establishes the allocation regression's role;
it is not a passing test result for the unchanged working tree.

**[Run]** With the same in-memory substitution, an independent four-node
square fixture (side .1, k=2, delta=.2, gamma=.6, float64, beta=1.3) compared
the produced operator against `torch.matrix_exp(-beta * L / mean(diag(L)))`.
Operator agreement passed at atol/rtol `1e-12`; quadratic loss and autograd
gradient agreed with `trace(E.T F E)/(3*N)` and `2 F E/(3*N)`. The raw-tolerance
roundoff probe described above still failed, independently of allocation.

**[Decision/Open]** Repair allocation and tolerance; add persistent v2 tests;
restore the agreed arm matrix; settle calibration and pairing. Then run the
CPU suite on the actual source and a Colab smoke before the accuracy pilot.
No model weights, GPU evaluation, archived results or accuracy claims were
changed by this review. A fully passing actual-source suite plus matched
smoke artifacts would supersede the blocked implementation status.

## Correction status, 2026-09-26

**[Code/Run]** The three code blockers above are repaired in the current
working tree. Basis allocation now follows final rank selection; PSD rejection
uses raw eigenvalues and raw-unit tolerance; `eval_gsd_smooth.py` launches v1,
v2 hard and v2 smooth. Smooth v2 now requires an explicit spectral weight.
Persistent tests cover these cases and independently check a literal two-node
heat kernel, loss and gradient. The focused 60-test suite passed. See the
dated correction entry in `findings_log.md` for implementation evidence.

The smooth-profile trajectory is also exercised through both local and
conditioning gradient routes. The full repository suite passes **101 tests**;
the launcher preview parses to 3 v1, 3 hard-v2 and 3 smooth-v2 commands.

**[Open]** The remaining gate is external guidance-scale calibration and the
predeclared beta-selection rule, followed by Colab smoke. No accuracy run was
made in this correction; the earlier failure counts describe the pre-fix
working tree only.

## Second review and push preparation, 2026-09-26

**[Code/Run]** Reviewed on `gsd-smooth-spectrum`, based on `9650770`, before
committing the complete v2 change. The independent reviewer confirmed the
allocation, raw-unit PSD check and v1/hard/smooth launcher fixes. Three
additional reproducible defects were addressed:

- Empty graphs retained dense zero filters but reported zero operator bytes.
  Storage now comes from the returned tensor, including empty graphs; the
  unnecessary eager dense allocation for active graphs was removed.
- Launcher `:g` formatting rounded user coefficients/beta to six significant
  digits. CLI values and the beta run-name suffix now preserve float precision.
- A finite beta beyond float32's range produced `inf * 0` at a zero mode.
  Exponential weighting now multiplies in float64 and returns weights in the
  reference dtype, preserving the exact-zero weight at beta=1e40.

**[Run]** New persistent checks compare hard-v2 loss/gradients with
`actual_rank/N * v1`, including boundary expansion and isolates; compare the
heat operator with an independent matrix exponential; exercise empty graphs,
finite differences, batch sum, detached references/operators, RNG preservation,
vertex permutation and repeated-eigenvalue basis rotation. Invalid beta and
materially negative eigenvalues are rejected. V2 zero-weight dispatch matches
the original output/RNG and bypasses graphs. Synthetic v2 complete/failed
bundles retain their profile, beta, reduction and seven-file schema.

The new defect tests failed before their fixes. Final actual-source command:
`python -m unittest discover -s tests -q`: **111 passed**, 3.489 seconds.
The independent follow-up review ran 21 relevant tests and found no new
actionable code issue. No model or GPU evaluation was performed.

**[Decision/Open]** Historical failed/pending statements above describe earlier
snapshots. CPU implementation review is complete. Pilot calibration, beta
selection, real-runtime common-draw confirmation and Colab smoke remain open.
The test plan now explicitly avoids attributing A-vs-B effects solely to the
denominator when the external alpha also changes.
