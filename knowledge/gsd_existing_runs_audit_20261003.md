# Existing GSD/SCD runs and corrected reuse-first next step

> [Run] Later2026-10-04 update: unguided and fixed beta .5 smooth-only now
> both have complete all15/seeds0-2 evidence (six accepted ZIPs plus two
> summaries). These are no longer missing conditions. Reuse their artifacts
> from `result/modelnet40_c/gsd-guidance-ablations/`; see
> [accepted comparison and limits](gsd_guidance_ablation_results_20261004.md).

> [Run] 2026-10-04 coverage update: the full v1 spectral-only M100 Background
> seed0 archive has arrived and validates (23.1767%). Together with the old
> 14-row archive it yields a descriptive 61.1967% composite. Background is
> no longer missing; the remaining v1 spectral-only gap is 26 cells at
> seeds1/2. The inventory below is the historical 2026-10-03 snapshot.
> See [validation and provenance limits](../result/modelnet40_c/gsd_latent_spectral_v1/analysis_20261004_background/validation.md).

[Run/Code/User report/Inference] 2026-10-03, `gsd-smooth-spectrum@4be6afd`.
The user explicitly requests avoiding repeats of tests whose results already
exist. This supersedes the blanket 12-new-run recommendation in
`gsd_next_experiment_proposal_20261003.md`.

## Inventory actually checked

[Run/Verification] Read config.json and per_corruption.csv from every local
result ZIP: 141 ModelNet40-C and 3 ScanObjectNN-C archives, 144 total, with no
metadata/CSV read errors. Also checked standalone config.json run identities;
none added an unarchived run. This is a metadata/count inventory, not a new
full integrity/CRC or environment-equivalence audit. Cross-checked the
knowledge README, findings, prior branch comparison, all-15 matrix and plans.

The exact archive names, config hashes, methods, seeds, spectral contracts,
host fields and raw CSV rows are recorded in
`result/modelnet40_c/diagnostics/gsd_existing_run_inventory_20261003.json`.

| Condition | Completed coverage | Existing evidence | Default next action |
|---|---|---|---|
| Original SCD-only, eval/raw | all15, 2,468 each, seeds0/1/2 | 3dd_original all15-eval-raw ZIPs | Reuse |
| Full-screen SCD-only | all15, 2,468 each, seeds0/1/2 | 3 full-screen scd ZIPs | Reuse |
| v1 spectral-only, M100/weight1 | Gaussian+Impulse, 2,468 each, seeds0/1/2 | 3 spectral-only pilot ZIPs | Reuse |
| v1 spectral-only, M100/weight1 | all except Background, 2,468 each, seed0 | ablation14 spectral-only ZIP | Reuse; Background/other-seed coverage remains incomplete |
| SCD+v1 spectral, M100/M240/M400 | Gaussian+Impulse, 2,468 each, seeds0/1/2 per M | 9 on and 9 off pilot ZIPs | Reuse these comparisons; all15 extension is not yet present |
| SCD+smooth beta .5/rho .001 and beta2/rho .01 | all15, 2,468 each, seeds0/1/2 per candidate | 6 guided full-screen ZIPs | Reuse |
| SCD+v2 hard, calibrated rho .001 | Gaussian+Impulse, 128 each, seed0 | screen-beta-hard ZIP | Reuse at its subset scope |
| Diffusion with both guidance terms disabled | No corresponding archive found | No zero-guidance method/weight or zero-rate run found | Genuinely missing control |

[Code/Run] Legacy v1 configs may not have a separate scd_weight key. Their
serialized guidance is `SCD + weight * spectral`, and weight-zero behavior
explicitly delegates to the original baseline. These off runs are SCD-only;
they are not diffusion-only. Pure VAE and source-only are already completed
but also do not provide a diffusion-only control.

## Concrete archive identities

Under `result/modelnet40_c/gsd_latent_spectral_smooth_v2/`, the all15 SCD-only
ZIPs are:

- `20260928-133819_gsd-full-screen-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-scd-seed0.zip`
- `20260928-140814_gsd-full-screen-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-scd-seed1.zip`
- `20260928-143811_gsd-full-screen-20260928-113047_gsd-cal-diagnose-reference-seed0-n64-scd-seed2.zip`

Under `result/modelnet40_c/gsd_latent_spectral_v1/`, spectral-only ZIPs are:

- `20260923-183420_gsd-v1-pilot-on-seed0-spectral-only.zip`
- `20260923-184326_gsd-v1-pilot-on-seed1-spectral-only.zip`
- `20260923-185232_gsd-v1-pilot-on-seed2-spectral-only.zip`
- `20260926-141029_gsd-v1-ablation14-on-seed0-spectral-only.zip`

The pilot macros are 72.386548%, 72.629660%, 72.427066%; the 14-corruption
macro is 63.912364%. These scores retain their original evaluation scopes.

Original all15 eval/raw SCD reference ZIPs in `3dd_original/` are dated
20260913-163817 (seed0), 20260913-173819 (seed1), 20260913-183818 (seed2).
All use names `3dd-original-all15-eval-raw-seed<seed>.zip` after the timestamp.
Keep them distinct from legacy-mode, EMA, normalized-SCD and lambda=.96 runs.

## Corrected next step and limitations

[Decision/Proposal] Reuse completed SCD-only, spectral-only and combined
results. Withdraw the instruction to rerun all four arms at all three seeds.
The minimal new main experiment is unguided diffusion over all15 at seeds
0/1/2: three genuinely new full runs. Its host/assets/environment must be
compared with the archived controls; align settings wherever possible and
report residual differences rather than silently claiming exact pairing.
The existing code still needs explicit zero-guidance support.

[Decision] Missing scope or metrics are not the same as a never-run method.
At the corruption/seed coverage level, v1 spectral-only lacks Background at
seed0, plus 13 corruptions at each of seeds1/2 (27 missing cells total).
Gaussian/Impulse already exist at all seeds. V1 SCD+spectral at M100 lacks
13 corruptions per seed (39 cells). These are possible later extensions,
not automatic reruns or current requirements.

[Limit] Existing seed0 pilot/ablation14 rows may use different random-draw
assignments because scope/order changes. Do not splice their scores into a
claimed homogeneous new all15 run. Future missing-cell execution must record
its scope, manifest and randomization separately. Historical controls lack
paired per-example predictions/shared-noise identities, so reuse permits
descriptive comparisons but does not retroactively create a common-draw
factorial experiment. A dedicated causal replay would be a separately
justified question, not a default reason to repeat completed accuracy tests.

[Decision] Parallel diagnostics should first reuse the existing 113047
Gaussian/Impulse hard/smooth shared-state probes and full-screen aggregate
norm/graph statistics. Only missing observations justify new capture, such
as a v1-to-v2 direction cosine, actual graph connectivity, decoded displacement,
or diagnostics on previously unmeasured corruptions. Do not schedule the
entire earlier diagnostic proposal without identifying these gaps.

[User preference] Every future launch proposal must enumerate matching
existing artifacts and explicitly identify the new condition, missing coverage,
or missing measurement. No repeat full evaluation merely because a launcher,
logging format, or knowledge entry is new. No Colab run was launched here.
