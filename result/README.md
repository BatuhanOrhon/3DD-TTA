# Colab Result Archive

This directory stores immutable experiment evidence. See [`knowledge/experiment_protocol.md`](../knowledge/experiment_protocol.md) for the full protocol.

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

Please download and provide the complete run directory as a ZIP, not only a screenshot or final percentage. Remove credentials, Drive tokens, and private paths first. The ZIP should contain the seven required files above and use the same run ID inside `config.json` and both CSV files.

If an older script cannot generate the full format, provide all available logs and the exact notebook/command. The run will be stored as incomplete rather than discarded, and missing metadata will be listed explicitly.

## Method names

Use the canonical IDs in [`knowledge/repository_map.md`](../knowledge/repository_map.md): `source_only`, `lion_recon`, `3dd_original`, `gsd_static`, `gsd_dynamic`, `gsd_physical`, `dual_seq`, `dual_sync`, `pxp_priority`, or `pxp_symmetric`.
