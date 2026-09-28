# GSD calibration implementation and experiment audit

[Code/Run/Inference] Review of `gsd-smooth-spectrum@6a0b4da` on 2026-09-28.
Scope: smooth spectral mathematics, `b7eb4d7` calibration changes, `1cb5c9b`
and `fd87777` analyzers, and the experiment decision in `6a0b4da`.
This review changes research notes only; runtime code and raw artifacts are
unchanged. It supersedes the immediate-freeze recommendation in that commit.

## Conclusion

The core implementation follows the approved mathematical/calibration design.
No new loss-sign, detached-prediction, double-beta or normalization defect was
found. The seven development results are internally consistent. The evidence
supports continuing development, not promoting beta2 as a stable optimum.
Two planned stages remain: a small interaction check and larger/repeated-seed
development comparisons before freezing settings for independent confirmation.

## Findings requiring follow-up

### 1. The new disjoint-validation recommendation has no execution path yet

[Code] `gsd_calibration.py:165` requires the calibration split seed to match
the development seed and requires the development count to contain the
diagnostic pool. `run_baseline.py:665` selects the same shuffled prefix, then
`:671` explicitly rejects a prefix that omits the calibration indices.
`eval_gsd_calibration.py` supports diagnose, report, screen-weight and
screen-beta only. Increasing count or changing model seed cannot create a
disjoint set: the former includes the original development prefix and the
latter does not change the split seed.

[Inference/Decision] This is correct for the implemented development protocol,
but contradicts the implied readiness of the latest next-test recommendation.
Independent confirmation needs a separately defined evaluation split/stage,
frozen coefficients, an overlap check and object correspondence evidence.
Do not remove the existing development guard to make a run pass. Further
development work can proceed while the confirmation split is prepared.

### 2. The immediate beta2 freeze skipped registered development stages

[Code] Steps 3 and 4 of `gsd_calibration_review_20260927.md` require an adjacent
rho check at a promising new beta, keeping multiple candidates when needed,
and expansion across development examples/seeds before freezing. The latest
README/backlog advanced directly to frozen beta2 confirmation.

[Run] Beta2 has 193/256 correct versus beta .5 at 192/256, and SCD-only at
190/256. This one-example beta difference cannot establish a reliable beta
ordering. The reported +1.171875 pp is a correct descriptive difference from
SCD-only, based on three additional correct predictions in aggregate.

[Decision] Restore the registered sequence. Keep beta .5 and 2 on the short
list. A bounded next interaction proposal is beta .5/rho .01 (alpha
81.39104941932808): it completes the 2x2 development comparison of beta .5/2
and rho .001/.01 using three existing corners. Declare this new condition
before execution; it is not a completed experiment or a mandatory new grid.
Then extend promising candidates and matched controls across seeds 0/1/2 and
a larger development pool. Reused/nested examples remain development data.
The staged launcher has no single-candidate interaction phase; the existing
direct runner can express this condition with the calibration reference.

### 3. The analyzer's protocol checks are incomplete

[Code] `scripts/analyze_gsd_calibration.py:152` and `:241` compare manifests,
seeds, splits and calibrated coefficients, but do not explicitly validate all
shared host/graph CLI settings. `config.get(manifest)` also accepts missing
manifests when all runs omit them. CSV counts are checked against stored
indices but not against declared `gsd_development_count`; summary CSV content
and several CSV metadata fields are not validated by the analyzer.

[Verification] Read the actual five beta-screen bundles into memory, made
independent deep copies and replaced `_read_screen` with those copies for
ranking. Each of these invalid variations was accepted:

- change one candidate's `gsd_scd_weight` to .5;
- change one candidate's `gsd_modes` to 400;
- change every declared count to 32 while retaining 128 indices and CSV rows
  (the output then reports 32 examples per corruption);
- remove all three manifests from every run.

No raw file was modified. Existing tests also accept synthetic bundles with
declared count 32 and 200 recorded samples. Passing 128 tests does not close
these uncovered checks. Harden shared protocol, required manifests, count
and CSV/config consistency checks before relying on the analyzer as a full
artifact validator. Keep stricter fixture rejection tests with that fix.

[Run] An independent audit of the real seven archives found no such mismatch:
all host CLI settings match apart from intended profile/beta/weight/rho/name;
all are seed0, count128. Scheduler, module inventories, checkpoint loads,
extensions, randomness policy, preprocessing and all three manifests agree.
CSV counts/metadata/summary, commands, logged results, CRC and common archive
root also agree. There are no logged tracebacks. Thus the current accuracy
ranking remains valid as a descriptive development result.

## What is implemented correctly

[Code] The graph/reference is detached and fixed, and both latent signals
are compared in its common basis. For each sample the smooth loss is
`sum_i exp(-beta*q_i) ||u_i^T(Y-R)||^2 / (3*N)`, summed over samples, where
`q_i=lambda_i/mean_active_degree`. The dense quadratic form and shared-spectrum
probe use this same weight once; neither accidentally squares the weight.
The signal is predicted-clean latent XYZ, N=2048, and differentiation passes
through the frozen prior to both local state and conditioning. Guidance stays
`SCD + alpha*spectral` with original host rates/decoder settings.

[Code] Diagnostic candidates share one eigensystem and each denoiser forward.
They are probed at steps 0/2/4 on an SCD-only trajectory, never applied as
candidate updates. Per-example local norm ratios are pooled over probe states;
alpha=rho/median(R). Missing denominators and near-zero medians are rejected.
Actual guidance uses a fixed alpha, not online norm balancing. Split selection
uses its own Python RNG and preserves model RNG consumption.

## Actual guided scale and graph evidence

[Run] Recomputed medians from all 5 steps x 128 samples per corruption in
the four rho=.001 guided ZIPs. These are weighted spectral/SCD gradient norm
ratios along the actual guided trajectories, not the calibration probes:

| Profile | Local Gaussian | Local Impulse | Style Gaussian | Style Impulse |
|---|---:|---:|---:|---:|
| beta .5 | .0009683 | .0010329 | .0009293 | .0006538 |
| beta 2 | .0009857 | .0010105 | .0011455 | .0007281 |
| beta 8 | .0010011 | .0009981 | .0014727 | .0009699 |
| hard-v2 | .0009743 | .0010167 | .0014786 | .0010848 |

Local matching works to within about 3.3% of the target median on this screen.
Style is not simultaneously scale matched, as the approved design explicitly
anticipated. Do not describe this as equal total two-route guidance strength.

[Run] Mean effective spectral mass (sum of weights) is about 1258.7/1260.8
for beta .5, 646.7/615.1 for beta2 and 329.3/286.2 for beta8
(Gaussian/Impulse). These are effective masses, not a number of nonzero
frequencies. The hard-v2 requested M=100 actually selects mean ranks
139.3/121.2, with maxima 291/240, because the existing numerical boundary
tolerance expands the selected subspace. This is a known implementation
policy, not an accidental new calibration change or an exact 100-mode test.
Actual-rank/eigensolver analysis remains open before attributing hard/smooth
differences solely to a clean mathematical cutoff.

[Run] Beta2/rho .001 corruption runtimes total 28.67 s versus SCD-only
11.76 s (about 2.44x), and peak recorded memory is 21009 versus 14603 MiB
(about 1.44x). These are short instrumented runs, excluding model loading;
they are not a full benchmark cost estimate.

## Verification and remaining evidence

[Verification] `python -m unittest discover -s tests -q`: 128 tests pass.
No local model/GPU experiment was run. The screen-weight archives record
`1cb5c9b`, beta-screen archives record `fd87777`; all recorded runtime source
manifests agree. Current working files also agree with recorded sources after
normalizing Windows CRLF to LF where needed; raw byte hashes differ only
for those line endings in five existing host files.

[Open] Instrumented spectral-off versus original native CUDA prediction/RNG
parity is still unmeasured. CPU fixture parity exists, and the current screen
uses a matched instrumented SCD-only host; that does not itself close the
original-3DD-TTA parity gate. No new v1 comparator was run in this screen.
Per-example predictions are not archived, so a paired classification test
cannot be computed from the aggregate correct counts. The original seven-file
diagnostic ZIP is still missing; `report.json` alone does not establish its
full raw provenance. Independent confirmation requires an untouched split
and a configuration frozen after the remaining development work.

Sources: all seven `*gsd-cal-screen-*.zip` files and `report.json` under
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/`; exact run IDs and
accuracy counts are preserved in the dated findings log entries.
