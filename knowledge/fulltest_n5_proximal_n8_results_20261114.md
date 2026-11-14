# Full-test uniform N5 versus proximal N8+PM — 2026-11-14

[Run/Code] Complete: all60 corruption/method/dataset groups,240raw head
archives,365880final predictions. Every ID/label/logit/PM prediction recounted
independently on the host; all required copies preserved and DriveFS final
cloud flush completed2026-11-14T18:09:00Z. The r9A100 runtime remains live;
Drive is unmounted after successful flush, not the runtime. No teardown
permission has been given.

## Locked protocol and scope

All15 corruptions at severity5, batch32 including the final4ModelNet/final5Scan
examples, fresh worker seed0 per corruption. ModelNet40-C all2468unique test
objects/corruption (37020predictions/head/method); accepted ScanObjectNN-C
main_split all581 (8715predictions/head/method). Classifier weights/state stay
frozen; all heads share each method's reconstruction.

Original3DD terminal uses N5 uniformly, gamma/eta.01/.01 and retained.95;
cheap proximalN8 uses tau.1/retained.95, no denoiser backward or direct style
update, equal probability mean candidates2/4/6/8 at T1. No special Background
horizon. Controlled backend, original encoded decoder style, DDIM100/final-alpha1,
raw/eval/EMAoff. Scientific source
`b651e259db9ec6341f39e606b01eef5254d90456`; scientific contract
`f2721cd29a1cfdc0937d552a63e4d6f34e0950c528625c3370e5d506d2210f43`.

## Accepted accuracy

| Dataset | Source model | Terminal N5 (%) | Proximal N8+PM (%) | Delta (pp) |
|---|---|---:|---:|---:|
| ModelNet40-C | Point-MAE | 61.5235 | 63.6764 | +2.1529 |
| ModelNet40-C | DGCNN | 63.2442 | 63.9411 | +0.6969 |
| ModelNet40-C | CurveNet | 64.5030 | 65.6780 | +1.1750 |
| ModelNet40-C | PointNeXt | 68.4684 | 70.6267 | +2.1583 |
| ScanObjectNN-C | Point-MAE | 35.7085 | 37.9231 | +2.2146 |
| ScanObjectNN-C | DGCNN | 36.0528 | 38.3477 | +2.2949 |
| ScanObjectNN-C | CurveNet | 54.2628 | 56.2937 | +2.0310 |
| ScanObjectNN-C | PointNeXt | 48.5714 | 50.0861 | +1.5146 |

All15 macro accuracy equals pooled accuracy because each corruption retains
all original IDs. N8+PM has higher observed accuracy in all8matched cells.
These are one-seed descriptive results comparing whole policies; horizon,
reconstruction and readout effects are not isolated. This is not a statistical
reliability/generalization or exact-paper-reproduction claim. Accepted published
alternative source checkpoints are used. Custom native determinism and Scan
corruption-generation seed parity to GSDTTA remain unproven. No rates, horizons,
readouts, checkpoints or IDs were tuned after observing these results. Previous
140-object/Background35 and paper65.7/66.1 rows are different protocols.

## Validation, persistence and reproducibility

[Code/Run] Actual fresh canonical ten-cell setup (only the explicitly deferred
Cell2fastrlock exception), native GPU checks, data/checkpoint checks and strict
8headCPU admission passed. All60acceptance guards checked ID/batch coverage,
actual scheduler horizons/final-alpha, source/native bindings, frozen classifier
and LION state, shared geometry and classifier RNG. Independent host recount
then verified all240raw ZIP hashes/CRC, admitted labels, endpoint logits and
PM from saved candidates, and all8overall/120paired corruption table rows.
Scientific CLI regression checks:23tests +21subtests pass; backup/controller
checks21pass. No local real model inference was run.

[Run] Two r7ScanUniform groups were salvaged on the host and reused only after
identical scientific-contract admission;58r9 groups are fresh. Earlier Drive
quota0 made cached readbacks falsely appear durable; three other old accepted
r7groups had no recoverable raw files and were recomputed, never inferred from
prose. Per-worker cache isolation fixes the earlier smoke/full RNG collision.
Mandatory host ACK precedes the next worker; all60groups are mirrored to the
requested shared thesis folder. Final official DriveFS push_changes_and_quit
returned success and unmounted Drive after host/table proof upload. The cloud
flush receipt is preserved on the host/in compact Git evidence. No VM stop.

[Checked counts, all240raw archive hashes, checkpoint/source identities,
final file hashes, timings and cloud proof](evidence/fulltest_n5_proximal_n8_20261112/complete_results_20261114.json).
Raw ZIPs remain under `result/<dataset>/<method>/<UTC-run-id>.zip`.
Derived CSV/Markdown and full independent recount are in
`result/preparation/fulltest_n5_proximal_n8_20261112/derived_r9/`;
Drive copies under `MyDrive/thesis/result/fulltest_n5_proximal_n8_20261112/`,
final reports `attempts/r9/derived/`. Actual source/native supplement is
`attempts/r9/deployment.zip`; executed host/mirror/finalizer sources are archived
separately there. Full current execution history is on source branch
`experiment/fulltest-n5-proximal-n8-20261112` and the preparation artifacts.

GPU campaign wall time was approximately4h04, within the6hwhole-phase cap.
The estimate at observed8.88CU/h is about36.05CU; global account billing includes
a protected peer and cannot establish exact task cost. Initial setup/recovery
costs are additional. No speed gain is inferred from these accuracy results.
