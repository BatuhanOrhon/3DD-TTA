# GSD Background reverse-step sensitivity

**Status: 30 valid runs ingested; one failed pre-inference attempt excluded.**
This note records the accepted 35-step reference results, the completed
5–25-step scan, and its limits. The ZIPs and run directories were validated;
derived results are linked below.

## Question

**[User report]** The Background scores around 23–24% seem unexpectedly low.
The user noted that the authors may use 30 or 35 reverse steps for Background;
the source supporting that paper-specific setting should be checked before
making a direct reproduction claim.

**[Code]** In the current `gsd-smooth-spectrum` checkout at
`aa8b725ba4daaf3448f0cda75bfc6aaec6c86d48` (2026-10-04; inference sources
unchanged by this knowledge update), `run_baseline.py` selects 35
reverse steps for Background and 5 for other corruptions. In `tta_gsd.py`,
`steps_back_local` is converted to a step count as
`(total * steps_back_local) // 100`; total is 100. Thus requested values
5/10/15/20/25 map to 5/10/15/20/25 actual reverse steps. The accepted prior
archives independently verify that the 35-step runs processed indices 0–34.

**Question being tested:** Does shortening the Background reverse trajectory
from 35 steps improve accuracy for the two no-SCD conditions? This is a
descriptive sensitivity test, not a claim that step count alone explains the
low score.

## Accepted 35-step references

All Background runs use 2,468 severity-5 examples. Percentages below were
recomputed from the six accepted all-15 ZIPs; the exact archives and full
provenance are linked in
[`gsd_guidance_ablation_results_20261004.md`](gsd_guidance_ablation_results_20261004.md)
and `result/modelnet40_c/gsd-guidance-ablations/analysis_20261004/analysis.json`.

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Unguided, SCD=0, spectral weight=0 | 23.703404% | 23.946515% | 23.217180% | 23.622366 ± 0.371359 pp |
| Smooth-only, SCD=0, beta=.5, alpha=8.140161356429882 | 24.149109% | 23.662885% | 23.824959% | 23.878984 ± 0.247573 pp |

**[Run]** The separate v1 spectral-only M100 seed-0 Background completion is
572/2468 = 23.176661%. It differs in method/version and is not another matched
seed for this sensitivity analysis. Its 14+1 combination is a cross-run
composite, not a homogeneous run.

## Declared run matrix

| Factor | Locked values |
|---|---|
| Corruption/scope | ModelNet40-C, severity 5, Background only, full file (2,468 examples) |
| Reverse steps | 5, 10, 15, 20, 25 |
| Arms | Unguided; smooth-only (beta=.5, alpha=8.140161356429882) |
| Replicates | Seeds 0, 1, 2 |
| Total new runs | 5 × 2 × 3 = 30 |
| Reused reference | Existing same-arm 35-step run for each seed; do not rerun |
| Shared controls | Batch 32; raw/eval LION; EMA off; frozen Point-MAE; original final decode style; lambda=.95; gamma=eta=.01; SCD weight=0; calibration reference `20260928-113047_gsd-cal-diagnose-reference-seed0-n64` |

The smooth-only arm retains beta and alpha while changing only the Background
reverse-step count. The unguided arm keeps both guidance weights at zero. All
other corruption schedules remain at the existing 5-step setting, but only
Background is evaluated in these runs.

## Launch and artifact handling

**[Code/Open]** A Colab cell was supplied in the conversation on 2026-10-04.
It temporarily adds the Background step argument and permits the existing
full-file ablation route to evaluate only Background, then restores
`run_baseline.py` and `gsd_protocol.py` on cell exit. It writes results under
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/`. Its first condition
failed before inference as described below; after correcting the calibration
input verification, all 30 planned matrix conditions completed. This records the launch plan at commit
`aa8b725ba4daaf3448f0cda75bfc6aaec6c86d48` on branch `gsd-smooth-spectrum`;
pre-existing workspace changes were left untouched.

### First launch failure (retained, excluded)

**[User report/Run failure]** The 5-step smooth-only, seed-0 attempt created
`20261004-121037_gsd-bg-step-5-smooth_only-seed0-20261004-121036.zip`, then
failed in `verify_calibration_inputs` with
`dataset_hash_manifest/gaussian` mismatch. `run_worker` hashed only the
selected Background data file, but the pinned calibration provenance check
also expects hashes for its Gaussian and Impulse calibration files. This is a
scope mismatch at preflight, not an inference result. The shell stopped after
the first failure; the remaining 29 runs did not start. Preserve this ZIP as a
failed artifact and exclude it from accuracy analysis.

**[Code/Decision] Correction for the Colab patch:** Before calling
`verify_calibration_inputs`, build a verification-only copy of the run config.
For every calibration `expected_manifests.dataset_hash_manifest` key absent
from the Background run's manifest, compute the current file identity with
`selected_data_path(args.dataset_root, name, config["severity"])`. Verify the
copy with the combined hashes. Store those identities separately in a config
field such as `calibration_input_hash_manifest`; leave
`dataset_hash_manifest` containing only the evaluated Background file. This
keeps the calibration inputs checked without claiming Gaussian/Impulse were
evaluated in this run. Keep the error ZIP and use a new run tag for retries.

### Accepted 5–25-step results

**[Run/Verification]** All 30 planned conditions are present and complete,
with one additional failed archive from the first pre-inference attempt.
Valid ZIPs pass CRC, safe seven-member contract, and exact ZIP-to-run-directory
content checks. Every run has a complete Background CSV row and summary for
2,468 examples; the 2-arm x 5-step x 3-seed matrix has no missing or duplicate
condition. Assets, Background data hash, calibration-input hashes, runtime
source hashes and native-extension inventory are identical across the 30
valid runs. LION eval and Dropout eval are recorded; EMA is off.

**[Code/Run]** `scheduler_timesteps.background` records all 100 points of the
DDIM grid, so its length is not the executed reverse-step count. Actual steps
are verified from aggregate per-batch diagnostics: 78 batches x requested
count; `step_index` min=0 and max=count-1 for each run. The algorithm uses the
last N scheduler points; changing N therefore changes both loop count and
initial noising timestep (N=25 starts at timestep 240; N=35 starts at 340).

| Arm | Steps | Seed 0 | Seed 1 | Seed 2 | Mean +/- sample SD (%) | Delta vs 35-step (pp) |
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

**[Run/Inference]** Scores remain low and the response is non-monotonic.
Step 25 gives the highest means, but improves only +1.0805 pp for smooth-only
and +1.3776 pp for unguided over same-arm, same-seed 35-step results. The scan
does not support reverse-schedule depth alone as an explanation of the
23–25% no-SCD Background scores.

**[Inference/Open]** Historical 35-step SCD-only and smooth+SCD Background
means are 60.6699% and 60.9130%. That large contrast is consistent with the
no-SCD condition being more salient than step count, but it is not a causal
estimate: prior controls differ from the new sweep in `run_baseline.py` and
`gsd_protocol.py` source hashes and native Chamfer/PointNet2 hashes. All new
step conditions are matched to one another; equal seeds still do not prove
common random draws. Do not select/promote step 25 from this same-test-set
scan or run another accuracy grid on these examples without a separate
mechanistic hypothesis.

Derived artifacts are separate from the raw ZIPs at
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/background_step_sensitivity_20261004/analysis_20261004_step_sensitivity/`:
`validation.md`, `analysis.json`, `per_step_summary.csv`, and
`run_validation.csv`.

Because each run captures source hashes while the temporary runtime patch is
active, retain the exact executed cell together with the downloaded ZIPs. The
background-only archives are not valid inputs to the all-15 ablation analyzer.
On ingestion, validate the standard run files and CRC, full completion and
2,468 examples, configuration/CLI agreement, calibration identity, data/model
hashes, requested and observed scheduler-step counts, and the temporary-source
fingerprints. Keep raw ZIPs unchanged and place derived tables beside them.

## Analysis and limits

For each arm and seed, calculate the percentage-point difference from that
same arm/seed's existing 35-step Background score. Then report each step's
three seed scores, mean and sample SD. Confirm the scheduler actually performs
the requested count; a CLI/config value alone is insufficient. Show all five
steps, including null or negative outcomes, without choosing only the best
step.

**[Inference/Open]** The repeated runs use the same benchmark examples, and
changing trajectory length can change random-number consumption and later
draws. Equal seed labels do not establish common-draw pairing. Since this scan
compares settings on ModelNet40-C test examples, it is development evidence;
the top observed step count cannot be presented as independent confirmation.

**Falsifier:** If scores remain near the 35-step reference or fail to improve
consistently across seeds, the hypothesis that a shorter Background schedule
recovers the low score is not supported. Any claimed step effect also fails if
the archived scheduler counts do not match the requested values or if the
other locked inputs differ.

## Next checklist

- [x] Record the 35-step reference scores by arm and seed.
- [x] Declare steps 5/10/15/20/25, both arms and seeds 0/1/2.
- [x] Diagnose and correct the first attempt's pre-inference calibration hash
  failure; retain its failed ZIP and exclude it from accuracy results.
- [x] Run all 30 conditions and validate ZIP integrity, complete counts,
  calibration/data/assets, source/runtime identity and actual step diagnostics.
- [x] Compare each condition to the existing same-arm/seed 35-step result;
  record all seeds, means, SDs, and the non-monotonic outcome.
- [x] Update `findings_log.md`, `open_questions.md`, and the knowledge README.
