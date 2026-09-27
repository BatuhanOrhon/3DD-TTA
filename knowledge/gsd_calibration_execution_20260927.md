# Fixed-state calibration and staged development screen

[User report] Approved 2026-09-27. Implements the revised design in
[the calibration review](gsd_calibration_review_20260927.md), based on
`gsd-smooth-spectrum@d75a32d`. Numerical/model experiments remain Colab-only.

## Bounded implementation and acceptance checks

1. Share one graph eigendecomposition per encoded example across hard-v2 and
   smooth beta .5/2/8. Check losses and gradients against the existing dense
   operators, including isolated graphs. Keep the fixed original `3*N` reduction.
2. Add an explicit calibration stage with spectral weight zero and SCD weight
   one. Probe the first, middle and last reverse steps without applying any
   candidate gradients. Check output and RNG parity against original SCD on a
   CPU fixture; real CUDA parity remains an empirical question.
3. Record individual sample/step local and style RMS, update/state ratios,
   local SCD/DDIM displacement ratio, spectral/SCD norm ratios and cosines.
   Keep raw denominators; ratios with denominator <= 1e-12 are null, not
   stabilized by adding epsilon. This cutoff is a numerical reporting rule,
   not a physical stability threshold.
4. Select a deterministic shuffled index pool without consuming model RNG.
   Diagnostic count 64 and screen count 128 per corruption share a prefix of
   that pool, with split seed 20260927. Store exact original indices. Index
   correspondence is NOT verified object identity; no held-out claim until
   cross-corruption object correspondence has been established separately.
5. Summarize only completed Gaussian+Impulse seed-0 calibration artifacts.
   For each candidate, R is the pooled median of per-example/per-probe local
   ratios; fixed alpha=rho/R at rho 1e-4/1e-3/1e-2. Keep zero spectral ratios
   in the median; never recommend alpha from a zero median or missing SCD
   denominator. Report style separately. No accuracy enters this calculation.
6. Prepare four actual screening conditions: SCD-only plus beta2 at the three
   fixed calibrated weights. Later beta .5/8 and matched hard need an explicit
   selected rho after reviewing this screen. No automatic winner/promotion.

## Interpretation gates

- A 64-example diagnostic run measures scales, not accuracy improvement.
- A 128-example development screen rejects gross failures; it cannot resolve
  a roughly one-percentage-point gain reliably.
- A very small/large update ratio alone does not prove useful/harmful SCD.
- Preserve SCD weight one. A .5 SCD ablation is deferred until evidence warrants it.
- Calibration and screening use target development inputs. Pooled calibration
  creates cross-example dependence, explicitly recorded as an exploratory variant.
- Freeze settings before confirmation; reserve all touched indices and verify
  their object identities before constructing a held-out confirmation set.
- Separate-run seeds do not prove common draws. Only candidates probed inside
  a single diagnostic forward share exactly the same state.

## Execution status

[Code] Implemented in `graph_spectral.py`, `tta_gsd.py`, `gsd_calibration.py`,
`gsd_protocol.py`, `run_baseline.py` and `eval_gsd_calibration.py`. The existing
Drive exporter now includes smooth-v2 archives with its `--stage all` filter.
The [Colab handoff](colab_gsd_calibration.md) gives exact diagnostic, report,
screen and Drive commands. Existing `eval_gsd_smooth.py` smoke/pilot usage remains
available; the new calibration experiment uses the new launcher.

[Code/Verification] `python -m unittest discover -s tests -q`: **125 tests
passed** on the local CPU environment on 2026-09-27. `git diff --check` passed.
New evidence covers shared-spectrum vs dense losses/gradients and one-eigh
count, empty spectra, per-sample ratios/cosines and null denominators, separate
SCD/spectral/total update ratios, median coefficient construction, nested
indices without RNG consumption, five-step probe/no-probe trajectory and RNG
parity, worker artifact integration, staged CLI parsing, failed/incomplete
calibration rejection, calibrated coefficient/source hash checks and export.

[Code/Review] Independent code review identified that a direct development
invocation could omit its calibration reference; the CLI now requires it.
SCD-only, spectral and total update/state metrics are explicitly separated.
No further concrete runtime or spectral-math bug was found in that review.

[Open] Native CUDA/Chamfer/DDIM/LION integration, actual GPU memory/time and
diagnostic scale values require the first Colab ZIP. CPU fixtures are not
model evidence. No new model run, optimum beta/alpha, SCD instability or
accuracy improvement is claimed.
