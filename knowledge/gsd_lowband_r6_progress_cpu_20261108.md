# R6 paired screen: local CPU progress, 2026-11-08

[Code/Run/Open] Frozen partial analyzer SHA256 `e431388507114ee888fd08b1446a635daeecdc6ebd3fc53a587b35f122527b39`; validator basis-fp32-v2 is an explicit post-hoc acceptance-rule change. Original v1 failure and all raw errors remain preserved. These are incomplete617-object seed0 screen prefixes, with all15 corruptions clustered by object in the intervals; no old B2 pooling, unseen-data claim, automated expansion or interim stopping. Native repeat failure29/30 remains unresolved; conditional intervals exclude native numerical variability. This note is owned only by the delegated local CPU supervisor. No CLI, remote, GPU, source, shared-note or Git operations. Final617 analysis belongs to the primary operator.

## 2026-11-08T14:12:59.292156+00:00 — existing128 baseline

[Run/CPU derived] `result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/derived/partial_128_v2.json`; SHA256 `0381b5da37e27d44b0091104d2af16d9d32d26a77f11953c33bc3f7dcfde8f2e`. All15 corruptions, seed0; only complete cache/control/half0/M100 ACK sets.

- control PM: 1250/1920 = 65.104167%.
- half0 PM: 1236/1920 = 64.375000%.
- M100 PM: 1250/1920 = 65.104167%.
- M100_vs_control: 0.000000pp; conditional object-cluster95%CI [-0.8854166666666666, 0.8854166666666666]pp.
- M100_vs_half0: 0.729167pp; conditional object-cluster95%CI [0.10416666666666673, 1.3541666666666667]pp.
- half0_vs_control: -0.729167pp; conditional object-cluster95%CI [-1.5104166666666667, 3.252606517456513e-17]pp.

Coverage: {"actual_prepared_caches": 60, "cache_jobs": 2, "inference_jobs": 6, "objects": 128, "predictions_per_arm": 1920, "shards": 2}. Legacy v1 projection failures retained: [{"label": "shard001_half0", "absolute_error": 0.0, "arithmetic": "fp64_per_object_energy_then_producer_dtype_left_fold", "case": "background/1", "legacy_absolute_error": 1.10527247443315e-06, "legacy_fp64_sum": 8.81187358757228, "legacy_gate_passed": false, "object_order": "archived_batch_order", "passed": true, "per_object_count": 32, "per_object_energy_dtype": "float64", "producer_dtype": "float32", "producer_replayed_sum": 8.811872482299805, "schema_version": 2, "tolerance": 1e-06, "validator_version": "basis-fp32-v2"}]. Partial screen decision remains null; final617 decision reserved for primary operator.

## 2026-11-08T14:32:00.753783+00:00 — new paired cohort

[Run/CPU derived] `result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/derived/partial_192_v2.json`; SHA256 `d379998b0d1e9892c068b3bc22cfc09ac342d922a39e15f3ac04963b3fe2caba`. All15 corruptions, seed0; only complete cache/control/half0/M100 ACK sets.

- control PM: 1861/2880 = 64.618056%.
- half0 PM: 1845/2880 = 64.062500%.
- M100 PM: 1860/2880 = 64.583333%.
- M100_vs_control: -0.034722pp; conditional object-cluster95%CI [-0.7291666666666666, 0.6597222222222222]pp.
- M100_vs_half0: 0.520833pp; conditional object-cluster95%CI [-2.891205793294678e-17, 1.0416666666666665]pp.
- half0_vs_control: -0.555556pp; conditional object-cluster95%CI [-1.2500000000000002, 0.10416666666666671]pp.

Coverage: {"actual_prepared_caches": 90, "cache_jobs": 3, "inference_jobs": 9, "objects": 192, "predictions_per_arm": 2880, "shards": 3}. Legacy v1 projection failures retained: [{"label": "shard001_half0", "absolute_error": 0.0, "arithmetic": "fp64_per_object_energy_then_producer_dtype_left_fold", "case": "background/1", "legacy_absolute_error": 1.10527247443315e-06, "legacy_fp64_sum": 8.81187358757228, "legacy_gate_passed": false, "object_order": "archived_batch_order", "passed": true, "per_object_count": 32, "per_object_energy_dtype": "float64", "producer_dtype": "float32", "producer_replayed_sum": 8.811872482299805, "schema_version": 2, "tolerance": 1e-06, "validator_version": "basis-fp32-v2"}]. Partial screen decision remains null; final617 decision reserved for primary operator.

## 2026-11-08T14:53:24.928803+00:00 — new paired cohort

[Run/CPU derived] `result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/derived/partial_256_v2.json`; SHA256 `484b0b8f58ca22cc44a4ce2f8e1a15b0148cb0a6dba2cda72a618d519aa9593e`. All15 corruptions, seed0; only complete cache/control/half0/M100 ACK sets.

- control PM: 2528/3840 = 65.833333%.
- half0 PM: 2513/3840 = 65.442708%.
- M100 PM: 2528/3840 = 65.833333%.
- M100_vs_control: 0.000000pp; conditional object-cluster95%CI [-0.5729166666666667, 0.5729166666666667]pp.
- M100_vs_half0: 0.390625pp; conditional object-cluster95%CI [-0.05273437499999946, 0.8333333333333335]pp.
- half0_vs_control: -0.390625pp; conditional object-cluster95%CI [-0.9635416666666667, 0.18229166666666666]pp.

Coverage: {"actual_prepared_caches": 120, "cache_jobs": 4, "inference_jobs": 12, "objects": 256, "predictions_per_arm": 3840, "shards": 4}. Legacy v1 projection failures retained: [{"label": "shard001_half0", "absolute_error": 0.0, "arithmetic": "fp64_per_object_energy_then_producer_dtype_left_fold", "case": "background/1", "legacy_absolute_error": 1.10527247443315e-06, "legacy_fp64_sum": 8.81187358757228, "legacy_gate_passed": false, "object_order": "archived_batch_order", "passed": true, "per_object_count": 32, "per_object_energy_dtype": "float64", "producer_dtype": "float32", "producer_replayed_sum": 8.811872482299805, "schema_version": 2, "tolerance": 1e-06, "validator_version": "basis-fp32-v2"}]. Partial screen decision remains null; final617 decision reserved for primary operator.

## 2026-11-08T15:11:53.409281+00:00 — new paired cohort

[Run/CPU derived] `result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/derived/partial_320_v2.json`; SHA256 `3588903f9739b7545422b47fc33042f50bdfd1bfe98a38f8fd143ac51711100a`. All15 corruptions, seed0; only complete cache/control/half0/M100 ACK sets.

- control PM: 3176/4800 = 66.166667%.
- half0 PM: 3158/4800 = 65.791667%.
- M100 PM: 3173/4800 = 66.104167%.
- M100_vs_control: -0.062500pp; conditional object-cluster95%CI [-0.5625000000000001, 0.4375000000000001]pp.
- M100_vs_half0: 0.312500pp; conditional object-cluster95%CI [-0.10416666666666671, 0.7291666666666666]pp.
- half0_vs_control: -0.375000pp; conditional object-cluster95%CI [-0.8958333333333334, 0.12500000000000006]pp.

Coverage: {"actual_prepared_caches": 150, "cache_jobs": 5, "inference_jobs": 15, "objects": 320, "predictions_per_arm": 4800, "shards": 5}. Legacy v1 projection failures retained: [{"label": "shard001_half0", "absolute_error": 0.0, "arithmetic": "fp64_per_object_energy_then_producer_dtype_left_fold", "case": "background/1", "legacy_absolute_error": 1.10527247443315e-06, "legacy_fp64_sum": 8.81187358757228, "legacy_gate_passed": false, "object_order": "archived_batch_order", "passed": true, "per_object_count": 32, "per_object_energy_dtype": "float64", "producer_dtype": "float32", "producer_replayed_sum": 8.811872482299805, "schema_version": 2, "tolerance": 1e-06, "validator_version": "basis-fp32-v2"}]. Partial screen decision remains null; final617 decision reserved for primary operator.

## 2026-11-08T15:34:17.742361+00:00 — new paired cohort

[Run/CPU derived] `result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/derived/partial_384_v2.json`; SHA256 `a28831c771ff1b7c98dc19b042511f4747c01420936b342a4009f8ea5dfa7c82`. All15 corruptions, seed0; only complete cache/control/half0/M100 ACK sets.

- control PM: 3745/5760 = 65.017361%.
- half0 PM: 3726/5760 = 64.687500%.
- M100 PM: 3740/5760 = 64.930556%.
- M100_vs_control: -0.086806pp; conditional object-cluster95%CI [-0.5381944444444445, 0.3645833333333334]pp.
- M100_vs_half0: 0.243056pp; conditional object-cluster95%CI [-0.15625000000000003, 0.6423611111111112]pp.
- half0_vs_control: -0.329861pp; conditional object-cluster95%CI [-0.7986111111111113, 0.1388888888888889]pp.

Coverage: {"actual_prepared_caches": 180, "cache_jobs": 6, "inference_jobs": 18, "objects": 384, "predictions_per_arm": 5760, "shards": 6}. Legacy v1 projection failures retained: [{"label": "shard001_half0", "absolute_error": 0.0, "arithmetic": "fp64_per_object_energy_then_producer_dtype_left_fold", "case": "background/1", "legacy_absolute_error": 1.10527247443315e-06, "legacy_fp64_sum": 8.81187358757228, "legacy_gate_passed": false, "object_order": "archived_batch_order", "passed": true, "per_object_count": 32, "per_object_energy_dtype": "float64", "producer_dtype": "float32", "producer_replayed_sum": 8.811872482299805, "schema_version": 2, "tolerance": 1e-06, "validator_version": "basis-fp32-v2"}]. Partial screen decision remains null; final617 decision reserved for primary operator.

## 2026-11-08T15:55:18.743483+00:00 — new paired cohort

[Run/CPU derived] `result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/derived/partial_448_v2.json`; SHA256 `2ef8a1e9b79298c080fcb80c6c528dce9b3d8b7e645c8bf1394c315ce662b5fb`. All15 corruptions, seed0; only complete cache/control/half0/M100 ACK sets.

- control PM: 4438/6720 = 66.041667%.
- half0 PM: 4410/6720 = 65.625000%.
- M100 PM: 4426/6720 = 65.863095%.
- M100_vs_control: -0.178571pp; conditional object-cluster95%CI [-0.6101190476190477, 0.23809523809523817]pp.
- M100_vs_half0: 0.238095pp; conditional object-cluster95%CI [-0.13392857142857148, 0.6101190476190477]pp.
- half0_vs_control: -0.416667pp; conditional object-cluster95%CI [-0.8630952380952381, 0.014880952380952377]pp.

Coverage: {"actual_prepared_caches": 210, "cache_jobs": 7, "inference_jobs": 21, "objects": 448, "predictions_per_arm": 6720, "shards": 7}. Legacy v1 projection failures retained: [{"label": "shard001_half0", "absolute_error": 0.0, "arithmetic": "fp64_per_object_energy_then_producer_dtype_left_fold", "case": "background/1", "legacy_absolute_error": 1.10527247443315e-06, "legacy_fp64_sum": 8.81187358757228, "legacy_gate_passed": false, "object_order": "archived_batch_order", "passed": true, "per_object_count": 32, "per_object_energy_dtype": "float64", "producer_dtype": "float32", "producer_replayed_sum": 8.811872482299805, "schema_version": 2, "tolerance": 1e-06, "validator_version": "basis-fp32-v2"}, {"label": "shard006_M100", "case": "background/1", "schema_version": 2, "validator_version": "basis-fp32-v2", "arithmetic": "fp64_per_object_energy_then_producer_dtype_left_fold", "producer_dtype": "float32", "per_object_energy_dtype": "float64", "object_order": "archived_batch_order", "per_object_count": 32, "producer_replayed_sum": 8.90091609954834, "absolute_error": 0.0, "passed": true, "legacy_fp64_sum": 8.900917487469078, "legacy_absolute_error": 1.3879207383382663e-06, "legacy_gate_passed": false, "tolerance": 1e-06}]. Partial screen decision remains null; final617 decision reserved for primary operator.

## 2026-11-08T16:12:54.955097+00:00 — local WAIT_USER stop

[Failed/Run/CPU] Local supervisor stopped on `result/modelnet40_c/lowband_quarter_fast/20261108-105629_a10080_screen617_r6/downloaded/failed_transfers_basis_fp32_v2/WAIT_USER_20261108T161146.json` (SHA256 `5566058ace74e821094cf1e48d07257fcef7248a8e2de461262be7910ae2f1cb`). Local error: `HTTP503 Service Unavailable while downloading acks_basis_fp32_v2/shard007_M100.json`. Exact job receipts=31; complete paired shards=[0, 1, 2, 3, 4, 5, 6]. Latest successful derived coverage448 objects; no512/576/final617 analysis attempted. No retries, remote/CLI/GPU actions or source changes were made. Pending artifacts and all errors remain preserved; primary operator owns the next decision.
