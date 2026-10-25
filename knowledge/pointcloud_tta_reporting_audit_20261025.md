# Point-cloud TTA: source models and cross-paper result provenance — 2026-10-25

[Paper/Code/Run/Inference/Open] **The literature does not form one mutually reproduced numerical benchmark.** Several source-result families exist; some later baseline rows are explicitly inherited, some are author-reported reruns with changed training/protocols, and some differ without an identifiable execution history. Our ModelNet source results are very close to published Point-MAE, DGCNN and CurveNet references. Large gaps to particular other papers remain real but do not, by themselves, identify a broken local classifier.

This expands the [previous3DD citation audit](3dd_tta_citation_reproduction_audit_20261025.md). It contains **16 numerical reporting publications,823 macro-table observations,128 source observations and77 paired original→later cases**. There are15 TTA-method papers plus the original corruption benchmark. These counts include separately published tables, training/batch variants and repeated observations; they are **not counts of independent experiments or replications**. Six additional TTA papers and the original benchmark receive numerical treatment beyond the previous matrix. No model execution or Colab operation occurred.

- [Complete823-row readable table](evidence/pointcloud_tta_reporting_audit_20261025/all_reported_table.md) / [CSV with full protocol fields](evidence/pointcloud_tta_reporting_audit_20261025/reported_results.csv).
- [Complete128-source readable table](evidence/pointcloud_tta_reporting_audit_20261025/source_only_table.md) / [source CSV](evidence/pointcloud_tta_reporting_audit_20261025/source_only.csv).
- [77 paired citation cases](evidence/pointcloud_tta_reporting_audit_20261025/citation_comparisons_table.md) / [paired CSV](evidence/pointcloud_tta_reporting_audit_20261025/citation_comparisons.csv).
- [Publication catalogue](evidence/pointcloud_tta_reporting_audit_20261025/publication_catalogue.json), [artifact receipt](evidence/pointcloud_tta_reporting_audit_20261025/bundle_receipt.json), [validation](evidence/pointcloud_tta_reporting_audit_20261025/validation.json).

## Evidence and comparison rules

[Paper] means directly published text/table; [Code] inspected code or deterministic extraction; [Run] previously accepted local artifacts; [Inference] descriptive interpretation; [Open] not established. Reported reproduction is an author statement, not our verification of authors' raw logs, dataset or checkpoint bytes. Equal printed numbers cannot establish copying; differing printed numbers cannot establish rerunning.

Native metrics remain visible:744 accuracy rows,69 **ordinary classification-error** rows,10 macro-recall rows. The additional `derived_accuracy_percent` field equals100−error only for verified ordinary percentage classification error. It is not applied to normalized mCE or recall. Conversions and averages of displayed cells are derived descriptions, not fresh accuracy measurements. Batch size, standard/online/reset behavior, source-training variant, voting, backbone, corruption set, severity and averaging denominator stay separate.

The main15-corruption papers, original benchmark's five-severity average, CloudFixer's six-corruption ablation, PointDA-class protocols and seven-atomic-corruption VLFM studies are separate cohorts. Shared dataset names do not imply shared corruptions or files. None of the tables below is a pooled method ranking.

## Point-MAE source-only overview

[Paper/Run] Values are reported accuracy(%), except our explicitly labelled historical run. Main15-corruption observations are shown; paper severity is not always explicitly numbered. A missing cell means no corresponding observation here. All source training distinctions below matter.

| Reporting study / source training | ModelNet40-C | ScanObjectNN-C | ShapeNet-C | Main qualification |
|---|---:|---:|---:|---|
| Our frozen Point-MAE |53.6899|—|—|Accepted severity5; full Scan source macro not available in this comparison |
| MATE: classification-only source |53.9|45.7|57.6|Original source row, not joint-trained model |
| MATE: joint MAE+classification, no TTA |57.6|45.6|59.3|Different source training; both rows are no-adaptation controls |
| CloudFixer: full15 Point-MAE |53.70|—|—|SupplementI3; source/model protocol must not be replaced by main six-type table |
| SVWA: own clean pretrained source |53.9|33.1|61.3|Separate source training; different Point-MAE bytes from our released3DD asset |
| SMART-PC: Org-SO |54.0|37.0|61.3|Own reproduced classification-only source |
| SMART-PC: MATE GitHub source / detailed rows |53.7|34.5|56.5*|MN/Scan adaptation-only starred rows; Shape reproduction extent not resolved by its single star |
| SMART-PC: fully reproduced MATE source |54.4|29.8|—|Detailed double-star rows; not interchangeable with single-star source |
| SMART-PC: own skeleton-auxiliary source |61.7|38.7|64.5|New source training; not a generic frozen Point-MAE checkpoint |
| Original3DD-TTA source |57.6|61.5|59.3|MN/Shape numerically equal MATE joint-training; Scan does not |
| Purge-Gate |54.16|36.96|61.35|MATE weights stated; source-row independent execution identity unresolved |
| LN-TTA |54.16|36.96|61.35|Repeated source cells; baseline attribution/provenance caveats below |
| MAMVI |54.13|36.96|61.35|MN upsampling/lidar cells differ slightly from PG family |
| SATOR |54.16|36.96|61.35|Same source fingerprints; baseline execution history unavailable |
| LoTT-PC: own LoTT-PC-SO† |74.4|52.0|66.4|Different source-training procedure; caption explicitly averages severity levels too |

*Shape56.5 is a reproduced source value, not a verified adaptation-only checkpoint identity. †LoTT's all-severity caption makes that row a separate protocol cohort. Its cited MATE/SMART source rows are retained separately in the complete table; a caption is not evidence each baseline was independently rerun under that aggregation.

[Inference] Our ModelNet53.6899 differs from CloudFixer's53.70 by−0.0101pp, SVWA's53.9 by−0.2101pp and the PG-family54.16 by−0.4701pp. These are useful descriptive agreements, not proof of equal weights or arrays. Original3DD Scan61.5 is substantially above the33.1–36.96 source family and originalMATE45.6/45.7; it is not a universally repeated Point-MAE target. Our ScanGaussian31.4974 versus SVWA32.5 remains a **single-corruption**, different-checkpoint comparison, not our full source macro.

## DGCNN, CurveNet and PointNeXt source families

[Paper/Run] All entries below are accuracy(%). BFTT3D entries explicitly convert its ordinary classification errors; their source architectures/training are not our checkpoints. SVWA CurveNet is a newly reviewed **supplementary** reference, superseding the earlier table's missing SVWA CurveNet cell.

| Backbone | Reporting study | ModelNet40-C | ScanObjectNN-C | ShapeNet-C |
|---|---|---:|---:|---:|
| DGCNN | Our accepted source |58.7979|32.7826|—|
| DGCNN | SVWA |58.8|37.9|63.6|
| DGCNN | BFTT3D,100−error |65.35|53.91|—|
| DGCNN | GSDTTA |66.51|55.72|—|
| CurveNet | Our accepted source |64.0113|52.0711|—|
| CurveNet | SVWA supplementTable3 PDFp3 |64.0|—|—|
| CurveNet | BFTT3D,100−error |71.53|35.23|—|
| CurveNet | GSDTTA |71.38|53.06|—|
| PointNeXt | Our accepted source |65.8320|49.0878|—|
| PointNeXt | CloudFixer supplementI2 |67.03|—|—|
| PointNeXt | GSDTTA |66.99|50.66|—|

[Code/Run] Our DGCNN/CurveNet ModelNet checkpoints were selected from SVWA's released assets before evaluation. Their58.8/64.0 agreement therefore supports source evaluation consistency with that asset family. It does not supply an independent training replication. Our contracts, hashes and exact local values are in [accepted source results](gsd_multibackbone_source_results_20261018.md).

[Inference/Open] DGCNN has a58.8 versus65–66 ModelNet source family; CurveNet has64.0 versus71.4–71.5. On Scan, CurveNet itself ranges35.23 versus52–53 across reviewed source settings. These differences precede TTA. They suggest source-training/checkpoint/protocol must be matched, but do not quantify the contribution of any one factor. The remaining local ScanDGC gap to SVWA37.9 is5.1174pp; it is not resolved by the ModelNet match.

## How original method results change in citing publications

### Explicit reuse and author-reported reruns

[Paper] Examples below bind original and later row IDs in the77-case CSV. Values are accuracy(%), with BFT's error conversion explicitly noted. Changes are not controlled effects because conditions differ.

| Original X / method | Dataset | X's own value | Citing Y's value | What primary evidence actually establishes |
|---|---|---:|---:|---|
| MATE-Online → BFTT3D supplement |MN|71.3|71.3 (=100−28.7error)|BFT caption explicitly says MATE errors provided in MATE; numerical reuse with metric conversion |
| MATE TENT baseline → BFTT3D supplement |MN|26.5|26.5 (=100−73.5error)|Same explicit numerical reuse statement |
| MATE-Standard → SMART-PC adaptation-only |MN|64.3|58.9|Detailed single-star caption: MATE GitHub checkpoint, adaptation rerun |
| MATE-Standard → SMART-PC full reproduction |MN|64.3|63.0|Detailed double-star caption: pretraining+adaptation rerun; separate experiment |
| MATE-Standard → SMART-PC |Scan|47.0|36.9|Author-reported adaptation-only reproduction; exact causal gap unresolved |
| MATE-Online → SMART-PC |Scan|48.5|36.9|Author-reported adaptation-only reproduction |
| SMART-PC's MATE-Standard → Purge-Gate |MN|58.9|58.90|PG captions explicitly take TTT results from SMART-PC; not a second independent measurement |
| SVWA → SMART-PC |MN|75.0|57.1|SMART marks SVWA reproduced; changed protocol/source setting, no exact asset verification |
| SVWA → SMART-PC |Scan|46.1|37.4|Same author-reported reproduction qualification |
| SVWA → SMART-PC |Shape|67.0|50.5|Same qualification; the large difference is not itself a bug diagnosis |
| Original3DD → GSDTTA |MN /Scan|65.7 /64.3|71.69 /57.08|Author-reported published-code run, **Point-MAE→DGCNN** |
| Original3DD → Uni-Adapter |MN /Scan /Shape|65.7 /64.3 /69.8|56.06 /30.78 /55.30|Author-reported implementation, **Uni3D-Large/B1/s5/step search**, original-author overlap |

The MATE chain is particularly important: **original64.3 → SMART adaptation-only58.9 → PG58.90**. A reader may see “MATE” in both later tables and assume the original64.3 was reused. The actual PG donor is SMART-PC's different reproduction. SMART's compact table also compresses the detailed single-/double-star distinctions; its MN compact source54.4 and standard63.0 correspond to full retraining, while nearby prose and some downstream comparisons use adaptation-only53.7/58.9. Preserve both variants instead of choosing whichever improves agreement.

### Changed means with unresolved execution history

| Method X | Dataset | Original X | Later Y | Provenance limit |
|---|---|---:|---:|---|
| SVWA → MAMVI |MN|75.0|75.04|Compatible with original one-decimal rounding; no baseline-specific rerun/reuse statement located |
| SVWA → MAMVI |Scan|46.1|47.45|Changed reported value; execution/asset history unresolved |
| SVWA → MAMVI |Shape|67.0|65.74|Changed reported value; no independent exact reproduction established |
| PG-SF → SATOR |Scan|47.54|46.78|SATOR selects PG-SF; baseline supplementary execution history not obtained |
| LN-TTA → SATOR |Scan|49.31|48.50|Different mean, same backbone/source-row family; causal explanation unavailable |
| LN-TTA → SATOR |MN /Shape|69.74 /64.38|69.71 /64.35|Small differences preserved; do not infer seed variance or rerun from the difference |
| CloudFixer → SMART-PC |MN|67.97|68.0|Compatible rounding; diffusion marker is not a reproduction declaration |
| CloudFixer-O batch1 → SMART-PC |MN|77.20|77.2|Matching mean, execution/reuse unspecified |

LN-TTA§4.1.2 says other results come from MATE[17] unless otherwise indicated, while captions flag starred methods as reproduced. This generic statement does not uniquely resolve every listed baseline; for example SMART-PC postdates MATE. We retain the attribution ambiguity instead of asserting all rows were copied or rerun. Likewise SVWA says unstarred rows are reproduced; its12 starred comparison rows are excluded from that claim, but the immediate numerical donor is not explicitly identified. Equality with an older table is not sufficient to fill that gap.

## Corruption set, model-update variant and averaging denominator

### CloudFixer: same dataset/backbone name, different experiment

[Paper] CloudFixer mainTable7 PDFp12 uses **six** selected corruptions: Point-MAE source43.13, CloudFixer58.50, CloudFixer-O68.53(batch1)/72.68(batch64). SupplementTableI3 PDFp8 uses **all15**: source53.70, CloudFixer67.97, CloudFixer-O77.20(batch1)/80.36(batch64). This is not a discrepancy to explain through random regeneration; the evaluated corruption set differs explicitly.

The original CloudFixer numerical benchmarks cover ModelNet40-C and PointDA settings, not Scan-C or Shape-C. Later GSDTTA/Uni-Adapter Scan/Shape CloudFixer rows therefore cannot be copies of an original matching-dataset row. GSD's supplement describes model updates under its plain “CloudFixer” label, whereas original CloudFixer and CloudFixer-O distinguish input-only versus additional model adaptation. Retain variant ambiguity in GSD comparisons. Uni explicitly uses input-only transfer and new VLFM classifiers.

### Point-Cache → BayesMM: identical displayed accuracies, different mean definition

[Paper/Code] This is a **separate seven-atomic-corruption, usually severity2, VLFM cohort**, not the15-corruption severity5 benchmark. Models are ULIP,ULIP-2,OpenShape and **Uni3D-Giant**, not Uni-Adapter's Uni3D-Large. ScanOBJ_ONLY/OBJ_BG/PB_T50_RS are separate split-specific datasets.

Across **48 paired source/Point-CacheGlobal/Point-CacheHierarchical rows**, all clean+7 displayed accuracy cells are identical between Point-Cache and BayesMM. Point-Cache defines its Avg over seven corruptions; BayesMM's main prose says clean+corrupted, and its printed means predominantly use eight terms. Therefore changed Avg is not evidence of changed corruption performance or rerunning.

| ModelNet-C example | Point-Cache printed Avg | BayesMM printed Avg | Same clean+7 cells? | Derived common seven-corruption mean |
|---|---:|---:|---|---:|
| ULIP source |47.52|48.60|Yes|47.515714|
| ULIP Point-CacheHierarchical |53.70|55.02|Yes|Same corruption cells; see paired CSV |
| Uni3D-Giant source |67.95|69.69|Yes|Same corruption cells; see paired CSV |

For ULIP source, the seven corruption cells sum332.61:332.61/7=47.515714. Adding clean56.16 gives388.77/8=48.59625. Both printed means follow their respective denominators. **Copy versus rerun remains unspecified.** BayesMM uses additional ModelNet text descriptions for its own method; that is another protocol distinction, not a basis to assert the provenance of its Point-Cache baseline rows.

[Code/Open] Fourteen of112 seven-type macro rows differ by more than0.011pp from the corresponding seven- or eight-term mean of displayed cells. Some differences are small; the largest is1.12625pp. Example: Point-Cache Uni3D S-PB_T50_RS source prints37.38, although its displayed seven cells average37.781429. BayesMM repeats those cells and prints38.46, while clean+7 averages38.813750. Published values remain unchanged; derived means and flags are separate. Raw logs or author clarification are required to resolve these arithmetic/reporting differences. Do not silently repair them or turn them into a model-performance claim.

The separate Sim-to-Real appendix has28 rows and12 paired cases; nine pairs change displayed cells, but no explicit rerun/reuse statement was found. It is archived as context and excluded from the823 corruption-matrix count and rankings. PCoTTA was inspected but excluded because its metrics concern reconstruction/denoising/registration rather than the requested classification accuracy.

## Primary reading list and coverage

The catalogue lists every reporting publication and its main/supplement URLs. Important new original sources:

1. **MATE: Masked Autoencoders are Online 3D Test-Time Learners** — Mirza et al.,ICCV2023. [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Mirza_MATE_Masked_Autoencoders_are_Online_3D_Test-Time_Learners_ICCV_2023_paper.pdf). Tables1/3/4 distinguish source training and standard/online adaptation.
2. **Backpropagation-free Network for 3D Test-time Adaptation** — Wang et al.,CVPR2024. [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Backpropagation-free_Network_for_3D_Test-time_Adaptation_CVPR_2024_paper.pdf), [supplement](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wang_Backpropagation-free_Network_for_CVPR_2024_supplemental.pdf). MainTables1/2 ordinaryerrors; supplementTable2 explicit reusedMATE/TENT values.
3. **CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation** — Shim et al.,ECCV2024. [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf), [supplement](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603-supp.pdf). Full15 versus six-type and input/model-update variants.
4. **SMART-PC: Skeletal Model Adaptation for Robust Test-Time Training in Point Clouds** — Bahri et al.,ICML2025. [Paper](https://raw.githubusercontent.com/mlresearch/v267/main/assets/bahri25a/bahri25a.pdf). DetailedTables5/7 distinguish adaptation-only and full retraining.
5. **Point-Cache: Test-time Dynamic and Hierarchical Cache for Robust and Generalizable Point Cloud Analysis** — Sun et al.,CVPR2025. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Point-Cache_Test-time_Dynamic_and_Hierarchical_Cache_for_Robust_and_Generalizable_CVPR_2025_paper.pdf). Seven-corruption averages and source-model split controls.
6. **Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning (BayesMM)** — Zhu et al.,CVPR2026. [Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhu_Adapting_Point_Cloud_Analysis_via_Multimodal_Bayesian_Distribution_Learning_CVPR_2026_paper.pdf). Clean-inclusive averaging and repeated Point-Cache cells.
7. **Benchmarking Robustness of 3D Point Cloud Recognition Against Common Corruptions** — Sun et al.,ICML2022. [Original preprint](https://arxiv.org/pdf/2201.12296v1). Its source/BN/TENT errors average five severity levels; keep separate from severity5.

Existing originals3DD-TTA,GSDTTA,SVWA,Uni-Adapter,Purge-Gate,LN-TTA,MAMVI,SATOR andLoTT-PC were re-examined at table/provenance level. Primary data were read directly; discovery summaries and search-result parsing were not accepted as numerical evidence. Three non-classification citing publications inaccessible in the previous audit remain unassessed; this expansion does not pretend otherwise. Generic TTAC/TMLR candidates and unrelated segmentation/detection/registration works remain outside the current numerical extraction. Search is bounded, not a claim of complete world literature coverage.

Correction to previous audit: SATOR main numerical tables are PDFpp6/7 (Tables1/2 onp6;Table3 onp7), not the older matrix's5/6 offsets. The new records use corrected pages; the immutable prior artifacts are retained. There is no new accuracy result associated with this correction.

## Local research implications and falsifiers

[Inference/Decision] Report our fixed checkpoint/data protocol as a matched local comparison; identify which published source family it follows. Do not select another checkpoint, severity, split or seed merely to recover a paper mean. ModelNet byte parity already constrains a generic random-regeneration explanation; Scan/Shape author-array parity remains unknown. The literature establishes protocol/reporting variation, not a measured causal decomposition of our gap.

For a thesis table, distinguish: original reported number; explicit inherited number and immediate donor; author-reported rerun with known changes; provenance unspecified; independently artifact-verified exact reproduction. None of the new author-reported reruns was upgraded to the last category. Final method gains still require matched controls and repeated-seed confirmation; this audit launches neither peer GPU work nor new tuning.

Falsifiers: authors' exact arrays/checkpoint hashes/configurations and raw logs could establish asset parity or explain a particular gap; an explicit baseline reuse/rerun declaration could resolve unknown edges; corrected author tables or a disclosed alternative reduction could resolve arithmetic flags. Without these, preserve both published values and the uncertainty.

[Artifact] Research begins from study commit`51dd03b00163e22e4641658bbc0d532b6154e7b7`. Raw primary papers, extracted row coordinates, agent evidence, arithmetic checks and authoring scripts live under`result/preparation/tta_literature_expansion_20261025/`; its adjacent sealedZIP and committed receipt record exact SHA256/CRC/member identities. A local research archive is not a GPU result bundle. No Drive backup or Colab command was attempted; peer worktrees/runtimes and protected user files were not modified.
