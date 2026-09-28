# Review of the rebuilt beta-rho interaction

[Code/Run/User report/Inference/Open] 2026-09-28. Runtime revision reviewed:
`gsd-smooth-spectrum@c1c7467`. This review changes documentation only.
See the [copyable next-agent handoff](gsd_interaction_handoff_20260928.md).

## Verdict and scope

[Code/Inference] The `--rebuild-prerequisites` experiment is consistent with
step 3 of the [calibration review](gsd_calibration_review_20260927.md) and the
[2026-09-28 audit](gsd_calibration_audit_20260928.md): complete beta {.5,2}
by rho {.001,.01}, retain both candidates, and use a matched SCD-only control.
It is a user-authorized repeat of the necessary development cells under a new
calibration reference, not an independent confirmation or a repeat of the
entire original seven-condition screen. It changes compute cost and provenance,
not the registered interaction question. Do not cancel or relaunch a running
Colab invocation merely because this documentation has been updated.

## Executed design in the launcher

| Order | Run | Count per corruption | Applied spectral guidance |
|---|---|---:|---|
| 1 | Fresh diagnostic reference | 64 | None; hard/.5/2/8 are common-state probes |
| 2 | SCD-only | 128 | alpha=0, SCD weight=1 |
| 3 | Smooth beta=.5, rho=.001 | 128 | Fixed alpha from this reference |
| 4 | Smooth beta=2, rho=.001 | 128 | Fixed alpha from this reference |
| 5 | Smooth beta=2, rho=.01 | 128 | Fixed alpha from this reference |
| 6 | Smooth beta=.5, rho=.01 | 128 | Fixed alpha from this reference |

[Code] All use Gaussian+Impulse, ModelNet40-C severity5, seed0, batch32,
raw/eval LION, EMA off, lambda=.95, gamma=eta=.01 and SCD weight1. Split seed
20260927 preserves the diagnostic prefix in the development pool. Coefficients
are derived as rho divided by the pooled median per-example/per-probe local
spectral/SCD ratio; they are fixed during guidance. Four prerequisite ZIPs are
generated and checked against that same reference before the final arm.
The analyzer then receives precisely the five development ZIPs and raw reference
config. Expect six seven-file run ZIPs plus a separate compact summary.

[Code] `c1c7467` changes the orchestration script, its tests and findings log;
it does not change spectral mathematics, host updates or the calibration
statistic. There are no extra beta8, hard-guidance or rho=.0001 accuracy arms.
Diagnostic probing of hard/beta8 is still part of the approved reference run.
The SCD-only run uses a hard profile label with zero spectral weight; it is not
a hard-guidance comparison. New run names and the summary include the reference
run ID. Old archives are not modified or selected in rebuild mode.

## Why the restart is justified, and what it does not establish

[User report] The old reference config was absent from Colab. Once prerequisites
were restored, the old launcher reported a coefficient mismatch for the stored
beta .5/rho .001 run and stopped before the target arm. The user then explicitly
requested rebuilding the ZIPs. The new mode supplies internally matched controls
without weakening coefficient/source checks.

[Open] The failing comparison used `rel_tol=1e-12, abs_tol=0`. The traceback
does not provide the fresh alpha, absolute/relative difference, or cause.
Do not infer a scientifically large drift, an eigensolver defect, environment
causation or invalid old accuracy results from this error alone. Compare the new
raw diagnostic report to the old report when it arrives. Rebuilding all cells
does not itself explain that discrepancy.

[Code/Open] Runtime source/data/checkpoint manifests are checked. They are not
a complete binary environment identity: installed package versions, CUDA and
compiled extension builds require inspection of environment/log evidence.
`eval_gsd_calibration.py` is in the recorded runtime manifest; the orchestrator
`scripts/run_gsd_interaction_from_scratch.py` is not in that list at c1c7467.
Do not edit runtime sources while the current reference is being used, or
alter stored hashes/configs to make checks pass.

[Code/Inference] The original staged protocol asks for diagnostic review before
screening. The rebuilt workflow proceeds automatically after structural/numeric
checks. Finite nonzero calibration medians are not a scientific stability gate.
Inspect new per-step SCD/state/DDIM ratios, tails, missing denominators and
local/style behavior before interpreting or expanding the resulting screen.
This operational compression is acceptable for the requested development rerun,
but does not replace the diagnostic assessment. If those metrics are abnormal,
retain the evidence and investigate rather than promoting a candidate.

## Current evidence and limits

[Run] Re-read seven original screen ZIPs locally: CRC checks and the current
strict analyzer reader pass, execution states are complete, and the stored
counts remain unchanged. SCD-only is
98/128 Gaussian and 92/128 Impulse; beta .5/rho .001 is 99/128 and 93/128;
beta2/rho .001 is 100/128 and 93/128. Beta2's +1.171875 pp over SCD-only is
three total correct predictions; its lead over beta .5 is one. Beta2/rho .01
is 100/128 and 92/128. These remain descriptive historical development results.
There are no new rebuild/interaction ZIPs locally at this review.

[Run/User report] Local failed ZIP `20260928-095332_gsd-cal-diagnose-reference-seed0-n64.zip`
records Diffusers .36.0, Hub .36.2, Torch 2.1.2+cu121 and an import-time
`torch.xpu` exception. It is not the earlier user-reported FPS failure at 092752.
The user subsequently restored Diffusers .11.1 and Hub .11.1 in the existing
Colab environment and reported `CUDA=True` and `DDPMScheduler import=OK`.
Those versions match the historical successful screen environment. The latest
coefficient-check traceback implies a diagnostic completed, but its raw bundle
has not yet been ingested. Neither the import check nor that traceback certifies
all CUDA behavior. Env files remain identical to `dev` as requested.

[Verification] `python -m unittest discover -s tests -q`: 133 tests pass in
this review. The two launcher tests cover selected prerequisite conditions and
reference-derived weights; they are not an end-to-end GPU orchestration test.
No new model/GPU evaluation was run locally.

## Required interpretation and next step

1. Obtain all six raw ZIPs from this exact invocation and its compact summary;
   preserve the raw diagnostic config. Do not rerun a completed expensive block
   just because a later summary/export step failed. There is no resume option
   in this launcher: another invocation starts fresh calibration and all cells.
2. Validate seven-file structure, completion, counts, reference run ID and SHA,
   deterministic indices, CLI/host/graph settings, source/assets and actual
   environment. For this rebuilt block require one exact common calibration
   identity, even though the analyzer can also handle equivalent generations.
3. Inspect fresh diagnostic scales and compare old/new alpha and median R
   values quantitatively. Do not manufacture the missing old raw reference
   from `report.json` or development-config snapshots.
4. Analyze only the five new development runs together. Report each corruption,
   total correct counts, macro accuracy, deltas from the new SCD-only control,
   and rho=.01 minus rho=.001 within each beta. The difference of those rho
   deltas is a descriptive interaction contrast, not a significance test.
5. Keep beta .5/2 as candidates until results support narrowing. Then declare
   larger nested development counts and seed0/1/2 with matched controls and
   a fixed accepted calibration. Freeze only after that development stage.

[Open] Native original/instrumented SCD prediction/RNG parity, per-example
classification pairing, cross-corruption object correspondence and a separate
object-disjoint confirmation stage remain open. Repeated seeds or larger
prefixes do not create held-out data. Style is monitored but not simultaneously
scale matched. Hard M100 boundary expansion remains unchanged and is not an
exact first-100-modes test. No new optimum or accuracy gain is established.
