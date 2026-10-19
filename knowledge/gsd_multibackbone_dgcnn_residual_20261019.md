# DGCNN residual three-job phase — 2026-10-19

**Current status — 20:44UTC [Run]:** all three jobs and the18-row matrix
remain accepted, with the stated13+2 ModelNet GSD limitation. Runtime cleanup
completed18:40UTC; a fresh bounded server inventory again finds zero active
sessions and no task monitor, evaluation or host-awake process. Fresh local
verification passes23 ZIP SHA/CRC/path checks and all count-based matrix and
trajectory recomputations. No new inference, allocation or process shutdown
was necessary. Historical submitted/running text below is superseded by the
final acceptance/cleanup sections and
[detailed final synthesis](gsd_multibackbone_final_findings_20261019.md).

[User report/Code/Open] User: “Hala canlıysa kalan testleri de çalıştırabilirsin”.
Same live r9 is reused; no rebuild or duplicate completed ScanGSD/source/CNPN/
trajectory evaluation. The earlier four-job59386-event proposal is superseded
by this three-job50671-event scope. Prior smoke-only authorization stays intact;
its accepted exact-v7 tests are explicitly admitted without rerunning.

| Job | Coverage | Batch | Predictions |
|---|---|---:|---:|
| ModelNet40-C GSD-inspired + DGCNN | Occlusion, LiDAR;2468 each |70|4936|
| ModelNet40-C original3DD + DGCNN | all15;2468 each |32|37020|
| ScanObjectNN-C original3DD + DGCNN | all15;581 main_split each |32|8715|

Severity5, seed0, fixed published-alternative checkpoints, frozen classifiers.
GSD uses unchanged v7 M800/beta2/spectral1.17/chamfer1/horizons10–30/updated
style/finalalphaFalse. Original uses unchanged `tta.py`, p.95/gamma-eta.01,
SCDsum/horizons5–35/originalstyle/finalalphaTrue. LION raw/eval/EMAoff.
This is a whole-policy comparison, not full GSDTTA paper reproduction.

## Source, assets and budget

Executed immutable v7 manifest
`78cdb53891bc0714dacd100eb220a5f6deb26bfe36f4e10b0aaa16f9aced60f9`,
sourceZIP `f82cf5052b8100591df679b159acec9f172144c5f6a0a916c5a5255cadeed4cf`.
Local orchestration starts from `f5dee52`; global remote checkout stays distinct
`dev@a458cd46bba1288a3f41302548bc777700cbc7d4`. No model source changed.
Scope SHA `cc77f0d31634ef2cb5afa0c04bbccc8ca858bf74238400d30e81e6210ee2355b`;
controller SHA `707dc790fd4f73087b4a3c0640b040ab0a8202095e219dde380c0118a6b1759e`.
Local preparation: `result/preparation/dgcnn_residual_20261019/`.

ModelNet DGCNN checkpoint
`e9eb118cd5adb42b7a564e2cc396ce698dee6cad42342d5e6c518e47ea152494`;
Scan DGCNN `7816a3ccf915406c5cc613883d8deb187d78e8f505df925d83cdcb8ac1343bfe`;
LION `807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`.
Source references58.797947% ModelNet and32.782559% Scan remain fixed.

Expected whole GPU phase49.847min, upper58.824min. Estimate reuses actual full
original reconstruction1637.122s ModelNet/391.108s Scan once, adds measured
DGCNN40ms/head,150ms/batch persistence and45s/job; GSD72batches use measured
7.4s reconstruction+105ms head, then15percent compute margin and90s extra.
All three jobs share a new3600s job-wall cap. Expected duration stays under60min
within the latest requested task; no additional long-run approval is required.
Reassess before extending this scope or cap. Preparation/archival are CPU-only.

## Reuse and acceptance

Same r9 session `tta-multibackbone-dgcnn-a100-r9-20261019`, endpoint
`gpu-a100-hm-kkb-euw4a0-1h0qzqfkhb7tr`, A100 High-RAM. Sole monitor70119/
local session63241 owns execution. New bounded host awake PID83891/session33491
(10800seconds) replaces expired prior assertion; managed lifetime not guaranteed.

Reuse accepted four smoke tests scope4c2c4854…a4b139/evidenceeacc97ac…07f4aa,
with explicit new-scope admission proving unchanged v7 assets/config/runtime.
Reuse accepted Scan GSD8715-event terminal36.075731%, raw SHA
`c50efad82090e8c777e219b3647defffa294f30ff1b0dbcd6317ff343dc2be58`.
Do not relabel old scope authorization or alter immutable old ledgers/results.

Reuse13 complete recovered ModelNet GSD corruptions (32084events); exclude old
partial Occlusion910events. New complete Occlusion/LiDAR4936events combine
into separate derived37020-event all15 evidence. This is two-attempt coverage,
not exact RNG resume or a single uninterrupted seed0 run. Old13 lack a final
interrupted LION state hash; retain this limit. Source checkpoints/data and
per-corruption classifier hashes remain verified.

Runner persists indexed per-batch NPZ+diagnostics to actual Drive with hashes,
then each corruption ZIP and final model ZIP. Controller preserves its logs,
scope/authorization/admission and phase ledger on success or failure. No auto
retry/overwrite. All seven standard result files, raw logits/IDs/labels, frozen
states, actual original DDIM schedules, source/checkpoint/preprocessing binding,
Drive/local SHA and independent metric recomputation precede acceptance.

[Run] CPU/hash preflight passed17:44:14UTC:622 asset files, four pinned
native binaries, LION/config, prior raw smoke identities, no active worker,
reused ScanGSD raw and unchanged global checkout. Fresh14928byte immutable
preflight evidence SHA4214776d1bd8f04f56166fda81a16cced682ec85015837204d671967127f6158
passes actual Drive/local roundtrip and CRC/controller/scope/report binding.
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261019T174414Z_dgcnn_residual3_preflight/preflight_evidence.zip`;
local `preflight_evidence_drive_roundtrip.zip` in the residual preparation root.

Current status: three-job GPU phase submitted through the same sole monitor;
ModelNet GSD Occlusion/LiDAR first, then ModelNet and Scan original3DD all15.
No full new accuracy accepted until artifact validation.


## ModelNet GSD subset and composite accepted — 17:57UTC

[Run/Artifact] First job17:46:07–17:54:46UTC,519.250626s;72batches,
4936new predictions. Occlusion/LiDAR two-file accuracy34.724473% versus same
scope source22.042139% (+12.682334pp); do not compare this subset to source
full15 macro. New raw `result/modelnet40_c/gsd_inspired_latent_spectral_guidance/20261019-174608_dgcnn-seed0.zip`,
7036942bytes SHA3ebdb4e619960adacd3a8bfbca4c7274c50508cfac7a8cfb5f7d7565dfb1230d,
passes actual Drive/local SHA, sealed members, all counts/labels/logits,
strict checkpoint/source/preprocessing, frozen classifier/LION and RNG guards.

Independent composition revalidates all1472 recovery manifest members,
13nested corruption ZIP identities,468complete old batch receipts and
32084old predictions. All13 classifier states pass; their initial LION state
matches new before/after, but final interrupted old LION state remains missing.
Old partialOcclusion910 is excluded. All15 complete files now contain37020
predictions; composite accuracy63.833063%, +5.035116pp over matched source
58.797947%. This is two-attempt complete coverage, not an uninterrupted seed0
run. OriginalDGCNN comparison remains pending.

Immutable derived bundle5487419bytes SHA
`6298278eb5f0b0eab5c7adcbbacbf32a157b7d8ee03b94ce4049e0f9a47cd4ea`,
local `result/preparation/dgcnn_residual_20261019/modelnet_dgcnn_gsd_composite_20261019T175630Z.zip`;
actual Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/modelnet_dgcnn_gsd_composite_20261019T175630Z.zip`.
Contains all15 indexed composite prediction arrays, per-corruption counts,
explicit attempt provenance/limits, independent analyzers and first-job raw
identity/state binding. All member hashes/CRC and actual Drive/local roundtrip
pass. Uploaded/downloaded through file service while the sole research cell
continues; no concurrentexec or mutation of raw attempts.

Second job ModelNet original3DD all15 started17:54:46UTC, followed by Scan
original all15. Same source/config/budget; no new source or smoke runs.


## ModelNet original3DD accepted — 18:25UTC

[Run/Artifact] Second job17:54:46–18:24:12UTC,1766.376976s (29.439616min),
1170batches/37020predictions. Accuracy65.586170%; matchedsource58.797947%
(+6.788223pp). The qualified GSD composite63.833063% is1.753106pp below
original; both exceed own source in this seed, with whole-policy/two-attempt
limits unchanged. No isolated spectral attribution.

Raw `result/modelnet40_c/3dd_original/20261019-175448_dgcnn-seed0.zip`,
18629467bytes SHA
`f63399443e978a8ff673f8e6b0fffc840de963a457b0e28fae8495ff943e698e`.
Actual Drive same dataset/method/run ID `final.zip` matches local raw bytes.
Independent validation checks all15 original-ID/label/logit aggregates and
batch shards, strict same checkpoint/source/preprocessing, frozenclassifier/
LION, actual originalDDIM100/full5-or35/set_alpha_to_oneTrue, headRNG guards,
source-matched corrected/harmed counts and exactCSV/NPZ agreement.

Immutable partial-phase analysis11156bytes SHA
`6154f546ba85f2b3bcb3a89cf7b26418a8286542ee5ed0e6200b8e74d25c3738`;
local `modelnet_dgcnn_original_accepted_20261019T182520Z.zip` and
`modelnet_original_acceptance_drive_roundtrip.zip` in residual preparation;
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/modelnet_dgcnn_original_accepted_20261019T182520Z.zip`.
Actual Drive/local SHA/CRC and every member verify. This phase snapshot
contains both complete jobs (41956new events); final Scan original remains
running. Final physical all-receipt audit follows that last job.

Scan original3DD started18:24:12UTC on same r9; no additionalGPU phase,
checkpoint changes or repeats. New two-job elapsed2285.628s/3600s cap.


## Final residual acceptance — 18:33UTC

[Run/Artifact] All three requested jobs completed. Scan original3DD job
18:24:12–18:31:33UTC,440.879817s (7.347997min),285batches/8715predictions:
accuracy36.442915%, ownsource32.782559% (+3.660356pp). Its matched GSD terminal
36.075731% is0.367183pp lower; GSD corrects635original errors and harms667
correct events (net-32), improves5/worsens10corruptions. ModelNet GSD versus
original corrects2460/harms3109 (net-649), improves4/worsens11corruptions.
Whole-flow single-seed comparison only; neither case isolates spectral effects.

Final new Scan raw `result/scanobjectnn_c/3dd_original/20261019-182414_dgcnn-seed0.zip`,
7006542bytes SHA
`be908813ec640822f1a9b28b1bc26b153a18607c03bd3aab8cfc7338d7674139`;
actual Drive same dataset/method/run ID `final.zip`, identical SHA/size.
Prior two new raw paths/hashes above remain immutable.

All3jobs50671new events/1527batches pass strict checkpoint/source/preprocessing,
frozenclassifier/LION, IDs/labels/finite logits, originalDDIM5/35/alphaTrue,
RNG/head geometry guards and standard CSV/aggregate/batch NPZ consistency.
All six cases now join source/original/GSD by original IDs, labels and own
checkpoint/preprocessing:18accuracy rows,411615method-role prediction events.
Reused CN/PN12raw ZIPs independently reproduce all prior four-case values;
newDGCNN ScanGSD uses its accepted separate trajectory terminal, not a rerun.

| Dataset | Backbone | Source % | Original3DD % | GSD terminal % | GSD-original pp |
|---|---|---:|---:|---:|---:|
| modelnet40_c | CurveNet | 64.011345 | 67.071853 | 65.186386 | -1.885467 |
| modelnet40_c | DGCNN | 58.797947 | 65.586170 | 63.833063* | -1.753106 |
| modelnet40_c | PointNeXt | 65.831983 | 68.292815 | 66.472177 | -1.820637 |
| scanobjectnn_c | CurveNet | 52.071142 | 55.238095 | 55.008606 | -0.229489 |
| scanobjectnn_c | DGCNN | 32.782559 | 36.442915 | 36.075731 | -0.367183 |
| scanobjectnn_c | PointNeXt | 49.087780 | 49.684452 | 45.209409 | -4.475043 |

*ModelNet DGCNN GSD is13recovered+2fresh complete-corruption composition;
old13 final interrupted LION state missing, no exact RNG resume. Preserve all
raw attempts and this limitation; do not call it one uninterrupted run.

GSD terminal is below original in all6cases and exceeds own source in5/6.
Fixed Scan probability mean from the separately accepted same-run trace gives
CurveNet56.477338% (+1.239243pp vsoriginal), DGCNN37.326449% (+0.883534pp),
PointNeXt46.815835% (-2.868617pp). These are whole-policy separate-run
comparisons; within-trace readout deltas use their own terminal. Point-MAE's
trajectory study remains separate with no matched source/original main_split
row. No additional seed/control or paper reproduction claim.

Total new job wall2726.507419s=45.441790min, within expected49.8–58.8min and
shared3600s cap. Native/setup/accepted smoke/ScanGSD/source/CNPN/trajectory
runs were not repeated. Phase started17:46:07UTC, ended18:31:33UTC.

Phase evidence38057bytes SHA
`87201e964410b88afcdb4bef2b8544a278736325908480e2820864f308df05f3`;
local `phase_evidence_drive_roundtrip.zip` in residual preparation; Drive
`MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261019T174607Z_dgcnn_residual3/phase_evidence.zip`.
Final analysis package29705bytes SHA
`b87a4e392532836674aaf3e01998b48e2cfa9e18e3585050942f1bc0155ba0c4`.
Accepted analysis+actual physical receipts1105309bytes SHA
`227af6582cc8fa0371db71a261f3d6392e783a22e7e73b44804d7ec81ead693f`;
local `result/preparation/dgcnn_residual_20261019/final_acceptance_drive_roundtrip.zip`;
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/dgcnn_residual3_accepted_20261019T183309Z/dgcnn_residual3_accepted_20261019T183309Z.zip`.
Actual Drive/local roundtrip, every member and nested analysis manifest,1527
batch/32corruption/3final physical receipts and all new raw ZIPs verify.
Derived exact CSVs: `all3_analysis/` and `complete_matrix/` in residual root.

Requested remaining GPU tests are complete. Follow the user's earlier
conditional shutdown instruction after final report backup and verification;
archive the actual lifecycle result separately, with no new research inference.


## Authorized cleanup complete — 18:40UTC

[User report/Run] The user's earlier “Şimdi yapılacak başka bir testimiz
kalmadıysa colab oturumlarının tamamını ve background terminallerini
kapatabilirsin” condition is now satisfied by completion of all three named
remaining tests, independent validation and final report backup.

Final report238340bytes SHA
`22ca24428687516bc82ddf5d002b76480c22a83479bac32fcde223f02e27735f`
is backed up on actual Drive and locally verified, including every member and
results commit `bf2860341c5da4b909930f2e89ed2daa4f317fbe`.
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/completed_multibackbone_report_20261019T183511Z.zip`;
local `completed_multibackbone_report.zip` and `completed_report_drive_roundtrip.zip`
in the residual preparation directory. All raw/accepted evidence above was
safely archived before runtime termination.

Stopped all three local monitors gracefully (exit0 each): r7PID34633/session97631,
r8PID63286/session69569 and r9PID70119/session63241. r7/r8 server assignments
were already lost; stopping monitors did not terminate another active runtime.
`colab stop -s tta-multibackbone-dgcnn-a100-r9-20261019` returned exit0 and
“Session terminated.” Bounded `REQUEST_TIMEOUT=20 colab sessions` confirms
no active server sessions. One earlier unbounded inventory hung; only its
local clientPID85667 was stopped before the bounded verification succeeded.
Host awake assertionPID83891/session33491 stopped. All four known task PIDs
are absent. No model/transfer/inspection work remains active.

Post-shutdown closure receipt and immutable local evidence are preserved at
`result/preparation/dgcnn_residual_20261019/closure_receipt.json` and
`closure_evidence_local.zip`. This lifecycle-only artifact was produced after
teardown and is local; research results and final report were verified on
Drive beforehand. Current task is complete; no runtime is allocated.
Closure evidence3536bytes SHA`79ba5df38cf29c9c42dfb4ac9d8c75a2196944fc1fd872aafe6ea17dcca615ba`.
