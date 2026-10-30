# ScanObjectNN-C / DGCNN checkpoint investigation — 2026-10-30

[User report/Decision] Prioritize checkpoint/training/preprocessing provenance for
the unexplained DGCNN source gap. Keep the accepted581-object `main_split`,
severity5 and existing corruption arrays fixed. Other backbones' closer results
motivate this priority; they do not prove that every model sees equivalent
features or that author arrays are identical. This phase completes a CPU/static
audit and candidate search, not a corrected accuracy experiment.

## Accepted baseline and concrete checks

[Run/Paper] Our all15 source is **32.7825588%**; SVWA Table3 reports37.9%
(gap5.1174412pp), while GSDTTA Table2 reports55.72% (gap22.9374412pp).
Our clean control is399/581=68.6747%. Source run
`20261018-202249_dgcnn-seed0` uses1024 FPS points, raw XYZ, k20, embedding1024,
dropout0.5 and frozen eval/BN state. Earlier complete archives remain unchanged.
[Accepted source evidence](gsd_multibackbone_source_results_20261018.md),
[full cross-paper table](source_only_3dd_all_papers_20261025.md).

| Hypothesis/check | Evidence and outcome | Interpretation |
|---|---|---|
| Wrong download or local corruption | Fresh public-author download21694533bytes; SHA256 `7816a3ccf915406c5cc613883d8deb187d78e8f505df925d83cdcb8ac1343bfe`, identical to evaluated file | Rejected for this released file; exact paper checkpoint remains unverified |
| Wrong40-class checkpoint or unloaded head |15x256 final head; all70state entries strict-load, zero missing/unexpected, all finite | No support for class-count or partial-load failure |
| Reversed dataset selector calls the wrong classifier | `linear3_` versus `linear3` changes only attribute name; inverse key renaming gives identical states and strict compatibility | Naming quirk is not a class permutation; no fresh inference parity claim |
| Stored `acc=0` means a failed model | epoch100, `metrics.acc=best_metrics.acc=0`; published trainer fixes `a=0` and gates validation on `a==1` | Zero metadata is expected with disabled validation; it cannot select a best checkpoint |
| SVWA drops the last batch | Same archived predictions on first576objects give33.0092593%; any five omissions can change the581-object mean by at most0.8605852pp | Tail omission alone cannot close5.1174pp; FPS/RNG changes are outside this bound |
| Checkpoint training split differs from test split | Published Scan pretraining YAML selects objectonly; its ROOT is `main_split_nobg`; commands use `scanobject_nbg`; training requires an initializer | Strong training-family lead; no file-linked training log proves this specific weight's complete history |
|1024 versus512 points | Main paper says1024; supplementTable1 lists512/1024; author issue explicitly recommends512 for **ModelNet DGCNN adaptation** | Does not establish Scan source-only used512; do not silently change the accepted contract |

[Code/Run] Scan optimizer steps are14544, matching
`floor(2309/16)*(100+1)`. This is consistent with published small-split training,
not a unique split fingerprint. BN counters474130 exceed optimizer steps;
because published Scan training requires a pretrained initializer, counters can
be inherited. This is **not evidence of target leakage or an adapted release**.
The initializer and exact file-linked training log remain unknown.

## Primary sources and search outcome

[Code] SVWA upstream HEAD remains pinned
`59935d68d0ef12811619ce4776e1dc4a711997d1`. Relevant primary references:

- [Released checkpoint folder](https://drive.google.com/drive/folders/1wUH7gInq5OHwNo5c6bOVYMeguwZiEA0B): evaluated file is `objectnn_src_only_DGCNN.pth`, ID`1ZkXrXj_U5t1OGr-_J7sGFPcuNVrA5fh-`.
- [Scan training config](https://github.com/AliBahri94/SVWA_TTA/blob/59935d68d0ef12811619ce4776e1dc4a711997d1/cfgs/pre_train/pretrain_scanobject.yaml), [OBJ_ONLY root](https://github.com/AliBahri94/SVWA_TTA/blob/59935d68d0ef12811619ce4776e1dc4a711997d1/cfgs/dataset_configs/ScanObjectNN_objectonly.yaml), [training commands](https://github.com/AliBahri94/SVWA_TTA/blob/59935d68d0ef12811619ce4776e1dc4a711997d1/commands.sh).
- [Trainer, validation disabled at lines243–253](https://github.com/AliBahri94/SVWA_TTA/blob/59935d68d0ef12811619ce4776e1dc4a711997d1/tools/runner_pretrain.py#L243), [initializer required at lines100–101](https://github.com/AliBahri94/SVWA_TTA/blob/59935d68d0ef12811619ce4776e1dc4a711997d1/train.py#L100).
- [Author's512-point recommendation](https://github.com/AliBahri94/SVWA_TTA/issues/2#issuecomment-3242788004): ModelNet adaptation context; later user still reports DGCNN difficulty. It is not a Scan checkpoint correction or proof our source implementation is defective.
- [SVWA paper](https://openaccess.thecvf.com/content/WACV2025/papers/Bahri_Test-Time_Adaptation_in_Point_Clouds_Leveraging_Sampling_Variation_with_Weight_WACV_2025_paper.pdf): PDFp5 says1024, PDFp8Table3 reports Scan source37.9. [Supplement](https://openaccess.thecvf.com/content/WACV2025/supplemental/Bahri_Test-Time_Adaptation_in_WACV_2025_supplemental.pdf): PDFp1Table1 permits512/1024 without resolving this Scan row.

[Code/Open] Two15-class Scan checkpoints were retrieved from
[3DeformRS develop](https://github.com/gaperezsa/3DeformRS/tree/30a2385edbd6a0e7919caa01344cb47af81bb361/Pointnet2andDGCNN/trainedModels):
`dgcnnScanobjectBaseline` and `dgcnnScanobjectBaselineV2`. Safe CPU deserialization
succeeds. They use a **different PyG implementation** (two DynamicEdgeConv
stages, max pooling and log-softmax), OBJ_ONLY training, random1024sampling
and centroid/L-infinity normalization. They are not drop-in SVWA weights and
were not substituted or ranked by corrupted accuracy. Their byte hashes and
head shapes are retained in the compact evidence.

[Code/Open] The [official OpenPoints model zoo](https://guochengqian.github.io/PointNeXt/modelzoo/)
also links a Scan PB_T50_RS DGCNN release with improved training; its displayed
86.1% is **clean hardest-split OA**, not corruption accuracy. The public folder
contains run`scanobjectnn-train-dgcnn-ngpus1-seed4767-20220522-213201-6LdYSGosNJuBoBGh5f7b4P`
(folderID`1DR300jJidTadrrfLsXjJWT51CeqhuZ_9`). It is a separate candidate;
checkpoint/config bytes have not been inspected. No exact GSDTTA checkpoint
identity was established by this bounded search.

## Decision and next discriminating work

[Inference/Open] We have the actual current SVWA release, a compatible15-class
network and reproducible archived reductions. A malformed/wrong-class download
is therefore less plausible. **Training provenance and input contract remain
the leading investigation targets; the accuracy gap is not solved.** Correct
published-file identity is weaker than correct paper-experiment identity.

1. Trace the released weight to its initializer and training log. Establish
   OBJ_ONLY versus OBJ_BG/PB_T50_RS, source sampling, centering/scale, k and
   BN policy from evidence, rather than optimizing on corrupted test labels.
2. Under a separately scoped diagnostic phase, use this unchanged checkpoint
   on the same accepted data and compare the existing implementation against
   the full pinned author's source evaluator. Keep matched FPS inputs first;
   isolate512/1024 sampling only as an explicitly labelled sensitivity check.
3. Inspect OpenPoints' complete checkpoint/config and clean-validation contract
   before implementing another DGCNN variant. Any alternative remains a new
   source condition with its own matched source/original3DD/GSD results.
4. Do not replace historical32.7826% or combine old/new checkpoint rows. A
   confirmed checkpoint/config defect reopens affected matched comparisons.

## Evidence, ownership and publication scope

[Run/Code] CPU audit input base`f57b170460360b029a5b0f2a0ebc130a41083206`;
session-ownership skill source commit`817128a`. Compact
[asset audit](evidence/scan_dgcnn_checkpoint_audit_20261030/asset_audit.json),
[public download receipt](evidence/scan_dgcnn_checkpoint_audit_20261030/current_download_receipt.json),
[source snippets/identities](evidence/scan_dgcnn_checkpoint_audit_20261030/source_evidence.json)
and [verification/selection](evidence/scan_dgcnn_checkpoint_audit_20261030/verification.json)
preserve positive, negative and unresolved findings. Raw downloads, scripts and
snapshots stay under`result/preparation/scan_dgcnn_checkpoint_audit_20261030/`;
weights, full HTML/issues dumps and caches are excluded from integration.

[Open] No Colab session was allocated, queried, used or terminated, no GPU or
new model inference ran, and no author was contacted. Runtime/endpoint/monitor
ownership is **none**. No Drive artifact backup was performed; the public Drive
download above is source retrieval, not our result backup. A future allocated
session must identify this task, e.g.
`tta-scan-dgcnn-checkpoint-a100-<UTCdate>-r<N>`, record ownership and preserve
peer/unknown sessions. Integration authorization does not authorize new GPU work.
