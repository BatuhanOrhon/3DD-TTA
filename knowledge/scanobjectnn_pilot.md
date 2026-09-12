# ScanObjectNN Pilot

Date: 2026-09-12

## Data confirmation

[User report] h5_files/main_split/test_objectdataset.h5 contains data (581, 2048, 3) float32, label (581,) int32, and mask (581, 2048) int32. The training file contains 2309 examples with the same field schema. The main split is retained as the initial benchmark input; only the test file is needed for TTA evaluation.

## Generator batch

[Code] datasets_mate/create_corrupted_dataset.py now accepts --severity, --corruptions, --seed, and --skip_ply. Existing defaults remain all corruption names, severity 8, and PLY export enabled. A supplied seed controls NumPy point subsampling and corruption randomness. Unknown corruption names fail before data generation.

[Open] The ScanObjectNN runner/checkpoint adapter is not implemented yet. The current ModelNet runner cannot be used for this dataset because it assumes 40 classes and severity-5 filenames.

## First experiment

Use one selected corruption (initially Gaussian), severity 8, fixed seed, and --skip_ply. Validate generated array shape/count and deterministic rerun before expanding the corruption set. Do not treat this pilot as a ModelNet40-C result.
## Validated input and runner adapter

[Run] The archived pilot input under result/scanobjectnn_c now has data shape (581, 2048, 3), float32, and label shape (581,), int32 with 15 classes and finite coordinates.

[Code] Commit ff6f6d7 adds dataset/severity-aware source-only artifact handling. The runner now supports dataset-name scanobjectnn-c, severity 8 filenames, 15-class Point-MAE configuration, and scanobjectnn_c result paths. A Point-MAE ScanObjectNN checkpoint is still required before evaluation.
