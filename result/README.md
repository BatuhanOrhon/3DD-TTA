# Colab Result Archive

## GSD guidance composition v1 - implementation, no runs yet

[Code/Verification] The local implementation adds the separate
`gsd_guidance_composition_v1` result schema. Each arm bundle contains the
seven standard files plus `predictions.npz`,
`gradient_diagnostics.jsonl.gz`, and `experiment_manifest.json`; the runner
also writes phase/selection manifests and preserves partial bundles. Focused
CPU algebra, protocol, runner, analyzer, and trajectory tests pass. The code is
uncommitted and not yet available to Colab. No GPU run or accuracy result is
claimed. See the [Colab phase plan](../knowledge/colab_gsd_block_routing_20261004.md).

## Guidance ablations - ingested2026-10-04

[Run/Verification] Six new full15/seeds0-2 ZIPs and two supplied summaries
are preserved under the user's imported hierarchy:
`modelnet40_c/gsd-guidance-ablations/{unguided,smooth_only}/20260928-113047_gsd-cal-diagnose-reference-seed0-n64/`.
Treat these folders as the raw evidence locations for this batch; do not
duplicate them under the older method directory or count copies as new runs.
Derived JSON, corruption tables and prediction transitions are under
`modelnet40_c/gsd-guidance-ablations/analysis_20261004/`.
Both conditions are complete. See
[accepted results and provenance](../knowledge/gsd_guidance_ablation_results_20261004.md).
Native binary identities differ between arms; no common-draw pairing is claimed.

## GSD full-test-set candidate screen - completed

[Run/Verification] Nine full-file runs cover every example in all 15
ModelNet40-C severity-5 corruptions with seeds 0/1/2. They compare SCD-only,
beta .5/rho .001 and beta 2/rho .01 using calibration reference
`20260928-113047_gsd-cal-diagnose-reference-seed0-n64`. The archives and compact
summary are under `result/modelnet40_c/gsd_latent_spectral_smooth_v2/`.

[Run] Equal-weight macro mean +/- sample SD: SCD-only 63.8799 +/- 0.1326%;
beta .5/rho .001 63.8520 +/- 0.0353% (mean matched delta -0.0279 pp);
beta 2/rho .01 63.8214 +/- 0.1184% (delta -0.0585 pp). The first candidate
has mixed seed directions; the second is lower in all three seeds. This is
descriptive full-test-set development evidence, not independent confirmation.
See the [consolidated all-15 ModelNet40-C result matrix](../knowledge/modelnet40_c_all15_results_matrix_20261001.md)
for this screen and prior experiments.

## GSD calibration/development artifacts - 2026-09-27

[Code] Smooth-v2 `calibrate` and `development` stages keep the same seven-file
bundle. `config.json.development_split` contains original shuffled indices;
`gsd_sample_diagnostics` contains per-example state/probe rows; graph aggregates
remain in `gsd_diagnostics`. A successful subset run has execution_status
complete and coverage status partial. Calibration accuracy is SCD-only, since
candidate spectral gradients are measured but never applied. Development runs
include the calibration reference hash/run ID and declared target rho.
See [Colab commands](../knowledge/colab_gsd_calibration.md). Preserve the first
diagnostic ZIP for review before screening; no new Colab result is claimed yet.

## GSD v1 artifacts - 2026-09-23

[Code] `gsd_latent_spectral_v1` preserves the seven-file schema below and
stores graph/gradient aggregates in `config.json.gsd_diagnostics`. Each
method/seed/arm receives a fresh directory and ZIP; no resume or overwrite.
Source/asset hashes, resolved scheduler and before/after mode inventories are
recorded. See [GSD handoff](../knowledge/colab_gsd.md).
[Run] Four smoke ZIPs and six complete Gaussian/Impulse pilot ZIPs are present
locally. The pilot's paired three-seed GSD-on minus GSD-off mean is -0.0135
percentage points, so it does not pass the current promotion screen. Raw ZIPs
remain outside Git; no all-15 GSD result exists yet.

## GSD smooth-spectrum v2 smoke - 2026-09-27

[Run] The validated partial smooth-profile Gaussian smoke is archived at
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/20260927-192156_gsd-smooth-v2-smoke-smooth-seed0-beta2p0.zip`.
It contains 32 examples (26 correct) and is execution/diagnostic evidence,
not an accuracy evaluation. A separate
`20260927-192156_validation.md` records archive checks, gradient-scale
diagnostics and the calibration gate. No hard/smooth accuracy conclusion or
pilot promotion is supported by this smoke.

After a Colab all-15 run, `scripts/export_gsd_results.py` copies each local
ZIP into the matching Drive path
`thesis/result/modelnet40_c/<method>/<run_id>/<run_id>.zip` without modifying
the local artifact.

This directory stores immutable experiment evidence. See [`knowledge/experiment_protocol.md`](../knowledge/experiment_protocol.md) for the full protocol.

## Imported GSD improvements default CSV - incomplete metadata

[User report/Run] The supplied `modelnet40_c/gsd_tta_improvements/eval_results.csv`
contains 15 corruption accuracies at M=400/M_mid=600 and weights 16/2. Its
equal-corruption macro is 64.3868% (all 15) and 64.7459% (Background excluded).
Values are consistent with 2,468 examples per row, though the CSV omits explicit
counts. Preserve the raw file unchanged. It has no seed, command, config,
environment, checkpoint/data hashes or logs, so it is an incomplete result
artifact and must not be treated as a matched comparison. See the latest
entry in `knowledge/findings_log.md` for its comparison and interpretation.

## Required layout

```text
result/<dataset>/<method>/<YYYYMMDD-HHMMSS>_<short-run-name>/
  command.txt
  config.json
  environment.txt
  stdout.log
  summary.csv
  per_corruption.csv
  notes.md
```

Example:

```text
result/modelnet40_c/3dd_original/20260912-153000_paper-config_seed0/
```

Use one new directory per run. Do not overwrite, merge, or manually clean raw logs. Derived analyses should be added as new files while the supplied artifacts remain unchanged.

## What to send after a Colab run

Batch 1 smoke command and download instructions are in [the Colab handoff](../knowledge/colab_baseline_smoke.md). run_baseline.py produces all seven files and a sibling ZIP automatically; it currently supports original-TTA smoke prefixes only, with no resume or mode change.

Successful prefix runs have execution_status=complete in config.json but status=partial in CSVs because the dataset was not fully evaluated. Accuracy columns are fractions, not percentages. Failed runs retain their logs/counts and are not benchmark evidence.

Please download and provide the complete run directory as a ZIP, not only a screenshot or final percentage. Remove credentials, Drive tokens, and private paths first. The ZIP should contain the seven required files above and use the same run ID inside `config.json` and both CSV files.

If an older script cannot generate the full format, provide all available logs and the exact notebook/command. The run will be stored as incomplete rather than discarded, and missing metadata will be listed explicitly.

## Method names

Use the canonical IDs in [`knowledge/repository_map.md`](../knowledge/repository_map.md): `source_only`, `preprocessing_identity`, `preprocessing_identity_seed_stability`, `pure_vae_encode_decode`, `pure_vae_seed_stability`, `shared_trajectory_decoder_control`, `scd_normalization_control`, `scd_lambda96_control`, `lion_recon`, `3dd_original`, `gsd_static`, `gsd_dynamic`, `gsd_physical`, `dual_seq`, `dual_sync`, `pxp_priority`, or `pxp_symmetric`.

The `shared_trajectory_decoder_control` pilot/confirmation keeps the seven-file
archive contract and accepts either the locked Gaussian/Impulse pilot scope or
the complete canonical all-15 corruption scope. It adds decoder-control fields to `summary.csv` and
`per_corruption.csv`: original/updated-style correct counts and accuracies,
paired delta in percentage points, prediction disagreement, decoder-output
difference, and style displacement. `config.json` also records the control
contract, RNG snapshot/restore policy, commit, checkpoint hash and dataset
hash manifest.

The `scd_normalization_control` method keeps the original-style decoder and
accepts the same Gaussian/Impulse pilot or complete all-15 scope. Its
`config.json` records the Eq. 11 point-count normalization contract, fixed
2048 denominator, retained count, actual VAE/DDIM/decode pipeline, and the
Colab-only CUDA/Chamfer integration check; its standard CSVs record accuracy,
counts, runtime and memory.

The `scd_lambda96_control` method keeps the original-style decoder, legacy
unnormalized SCD reduction, raw LION eval mode and EMA disabled while changing
only the retained SCD fraction from `.95` to `.96`. It accepts the same locked
pilot/all-15 scopes and records the value, retained count and Colab integration
check in `config.json`.
