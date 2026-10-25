# 3DD-TTA citation and reproduction audit — 2026-10-25

[Paper/Code/Run/Inference/Open] **No accessible later work in this bounded audit establishes an independently verified reproduction of the original 3DD-TTA Point-MAE results.** Two later works report executing 3DD-TTA on different classifiers: GSDTTA on DGCNN and Uni-Adapter on Uni3D-Large. Their reported results differ from the original numbers and cannot validate our Point-MAE target. Matching source-only numbers elsewhere, or a citation without a numerical comparison, does not establish a rerun.

The search verifies **15 distinct directly citing publications**, using primary bibliographies or publisher-deposited references. Twelve have accessible full primary texts; three have only preview/reference metadata available. The full-text numerical review finds **five later 3DD-TTA dataset/backbone rows**, all in the two works above. These counts describe this search, not an exhaustive global citation census. See the [search ledger](evidence/3dd_tta_citation_audit_20261025/search_ledger.md), [publication catalogue](evidence/3dd_tta_citation_audit_20261025/citing_publications.csv), [72-row evidence matrix](evidence/3dd_tta_citation_audit_20261025/evidence_matrix.csv) and [690 per-corruption facts](evidence/3dd_tta_citation_audit_20261025/per_corruption_fingerprints.csv).

## Question, scope and evidence standard

Descriptive question: which accessible papers actually cite Dastmalchi et al., report a 3DD-TTA comparison, and disclose enough provenance to distinguish reused numbers, author-reported reruns, and exact independent reproduction? Secondary question: does their data/protocol evidence explain our ModelNet40-C or ScanObjectNN-C gaps?

Audit branch `gsd-3dd-citation-audit` starts at committed study base `4c5be0eaf0ac933923129fe26738e0db5f96744a`. Only literature, public source snapshots and local derived documents were created. No model experiment, GPU work, Colab inventory query, allocation or runtime operation occurred. Peer worktrees and original-root user edits were read-only. Existing run evidence is historical, not fresh evaluation.

Labels: **[Paper]** primary paper or publisher reference; **[Code]** inspected pinned public source; **[Run]** existing archived local result/receipt; **[Inference]** interpretation; **[Open]** unresolved. An author statement that they ran code is [Paper], not our independent validation of their logs or assets.

| Provenance category | Requirement and use here |
|---|---|
| Original reference | Original authors' published row; retained without replacing it by README values |
| Explicitly copied | A direct statement that the later 3DD row was taken from the original; none located in the reviewed numerical 3DD comparisons |
| Author-reported rerun | Authors explicitly say they ran or implemented baselines; GSDTTA and Uni-Adapter qualify, under changed backbones/protocols |
| Independently verified exact reproduction | Matching classifier/checkpoint/data/preprocessing/severity/aggregation plus inspectable rerun evidence; none established |
| Provenance unspecified | A number exists but its rerun/reuse status is not explicitly established; equality is insufficient |
| Citation only | Actual primary reference verified, but no relevant numerical 3DD row in accessible full text |
| Contextual/non-citing | Related study without a verified direct 3DD citation; SVWA, Purge-Gate and LN-TTA belong here |
| Reference verified; full text unassessed | Publisher reference verifies citation, but numerical content cannot be audited from accessible material |

## Original reference and later numerical comparisons

All values in this section are **reported classification accuracy (%)**. No error-to-accuracy conversion was performed. `100−error` would be valid only for ordinary percentage classification error under the same denominator; it must not be applied to normalized corruption error/mCE. NA means absent in the accessible relevant table, or explicitly unassessed where full text was unavailable.

[Paper] Original: **Hamidreza Dastmalchi, Aijun An, Ali Cheraghian, Shafin Rahman and Sameera Ramasinghe**, *Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models*, arXiv:2411.14495 (2024), WACV 2025, pp.1566–1576. [Venue paper](https://openaccess.thecvf.com/content/WACV2025/papers/Dastmalchi_Test-Time_Adaptation_of_3D_Point_Clouds_via_Denoising_Diffusion_Models_WACV_2025_paper.pdf), [supplement](https://openaccess.thecvf.com/content/WACV2025/supplemental/Dastmalchi_Test-Time_Adaptation_of_WACV_2025_supplemental.pdf), [arXiv v1](https://arxiv.org/html/2411.14495v1). Venue PDF and arXiv-v1 target source/3DD rows agree; they are one publication, not two reproductions.

| Dataset | Original Point-MAE source | Original 3DD | Exact original location | Later GSDTTA DGCNN source /3DD | Later Uni-Adapter Uni3D-Large source /3DD |
|---|---:|---:|---|---:|---:|
| ModelNet40-C |57.6|65.7|Table2, PDFp7 /printed1572|66.51 /71.69|59.15 /56.06|
| ScanObjectNN-C |61.5|64.3|Table3, PDFp8 /printed1573|55.72 /57.08|38.07 /30.78|
| ShapeNet-C |59.3|69.8|Table1, PDFp6 /printed1571|NA /NA|57.92 /55.30|

### GSDTTA: external team, reported rerun, changed backbone

[Paper] **Xin Wei, Qin Yang, Yijie Fang, Mingrui Zhu and Nannan Wang**, *3D Test-time Adaptation via Graph Spectral Driven Point Shift*, ICCV 2025, pp.26762–26771, arXiv:2507.18225. [Primary paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf), [supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wei_3D_Test-time_Adaptation_ICCV_2025_supplemental.pdf).

Section4.2, PDFp5, explicitly states common pretrained backbone weights across comparable methods and results obtained by running published code. Tables1/2, PDFpp6/7, report 3DD only for **DGCNN**, with batch32 and fifteen corruptions. ModelNet has2468 test objects; Scan uses the581-object main split. CurveNet and PointNeXt have source and other-method rows, **no 3DD rows**. ShapeNet-C is not a numerical benchmark here. The supplement provides configurations for several baselines but no dedicated 3DD-TTA configuration. Exact 3DD input sampling, numerical severity, array hashes, backbone checkpoint hashes, seeds and raw rerun logs remain unavailable. This is an independent team's author-reported transfer/rerun, not exact Point-MAE reproduction.

[Code/Inference] Downloaded PDF cells were visually checked. Search-engine table extraction can corrupt cells: its Rotation79.53/Cutout77.70 variant is contradicted by the actual DGCNN3DD row, **59.93/72.77**; the PDF row averages71.6913 and supports printed71.69. Use archived PDF/fingerprints, not the search snippet.

### Uni-Adapter: reported implementation, new VLFM and timestep search

[Paper] **Mehran Tamjidi, Hamidreza Dastmalchi, Mohammadreza Alimoradijazi, Ali Cheraghian, Aijun An and Morteza Saberi**, *Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models*, arXiv:2511.15311 (2025), AAAI2026. [Venue PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37888/41850), [full arXiv PDF including supplement](https://arxiv.org/pdf/2511.15311).

Venue Tables1/2, PDFp5 /printed9300, and Table3, PDFp6 /printed9301, use Uni3D-Large. The baseline paragraph says the authors implemented twelve baselines including3DD. Implementation states1024 points, severity5; AppendixD in the full arXiv PDFp12 specifies batch1 generative adaptation and Table7's 3DD “Step Size” search `{1,5,10,25,35}`. Chosen steps per corruption and exact baseline preprocessing/assets are not supplied. The inspected public repository lacks a3DD baseline implementation. Three authors also coauthored original3DD; this is not wholly independent team confirmation. None of these rows verifies original Point-MAE accuracy.

## Other 3D-TTA source rows: useful cross-checks, different experiments

The table reports Point-MAE source rows; it does not present them as original3DD reruns. All three datasets are included. Full corruption fingerprints, PointNet/DGCNN SVWA source rows, supplementary GSD RSCNN source rows and PG Point-Mamba source rows are retained in CSV.

| Study and source row | ModelNet40-C | ScanObjectNN-C | ShapeNet-C | Location and provenance limit |
|---|---:|---:|---:|---|
| SVWA, own Point-MAE source |53.9|33.1|61.3|Tables1/2 PDFp6, Table3 PDFp7; own supervised clean pretraining described in supplement§1 |
| Purge-Gate, Point-MAE source |54.16|36.96|61.35|Tables2/1 PDFp7, Table3 PDFp8; MATE weights for reproduced results; source row rerun provenance unspecified |
| LN-TTA, Point-MAE source |54.16|36.96|61.35|Tables2/1 PDFp7, Table3 PDFp8; matches PG source cells; repetition does not prove independent measurement |
| MAMVI, Point-MAE source |54.13|36.96|61.35|Tables1/2 PDFp8, Table3 PDFp9; MATE weights stated; no3DD numerical row |
| SATOR, Point-MAE source |54.16|36.96|61.35|Tables1/2 PDFp5, Table3 PDFp6; no3DD numerical row; source provenance unspecified |
| LoTT-PC's cited MATE-SO row |57.6|45.6|59.3|Table1 PDFp7; caption/prose explicitly average corruption types **and severity levels**; no3DD numerical row |

[Paper] SVWA: **Ali Bahri, Moslem Yazdanpanah, Mehrdad Noori, Sahar Dastani, Milad Cheraghalikhani, David Osowiechi, Farzad Beizaee, Gustavo A. Vargas Hakim, Ismail Ben Ayed and Christian Desrosiers**, *Test-Time Adaptation in Point Clouds: Leveraging Sampling Variation with Weight Averaging*, WACV2025, pp.266–275, arXiv:2411.01116. [Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Bahri_Test-Time_Adaptation_in_Point_Clouds_Leveraging_Sampling_Variation_with_Weight_WACV_2025_paper.pdf), [supplement](https://openaccess.thecvf.com/content/WACV2025/supplemental/Bahri_Test-Time_Adaptation_in_WACV_2025_supplemental.pdf). It predates3DD's first preprint and the inspected venue paper does not cite3DD.

[Code/Run] SVWA commit`59935d68d0ef12811619ce4776e1dc4a711997d1` selects severity5, uses FPS1024 and a581-example main-split Scan generator. Its source loader uses `drop_last=True`; actual evaluated counts may differ. Its Gaussian source32.5% is close to our archived31.4974%, but full source33.1% is far below original3DD61.5. Existing read-only [checkpoint comparison receipt](/Users/batuhan/Desktop/projects/3DD-TTA/result/preparation/cli_scanobjectnn_s1_20261018/official_checkpoint_comparison_20261018.json) records 3DD SHA`53f6c2e4…535dac` versus SVWA`011a9efc…4a432`; all225 common named, same-shaped model tensors differ in stored bytes. This audit checked the receipt and fresh official code/PDF, not the large checkpoint tensor files again. Thus SVWA supports a severity5 **protocol-family cross-check with different weights**, not exact asset reproduction.

[Paper] Contextual non-citing studies:

- **Moslem Yazdanpanah, Ali Bahri, Mehrdad Noori, Sahar Dastani, Gustavo Adolfo Vargas Hakim, David Osowiechi, Ismail Ben Ayed and Christian Desrosiers**, *Purge-Gate: Backpropagation-Free Test-Time Adaptation for Point Clouds Classification via Token purging*, ICCV2025. [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Yazdanpanah_Purge-Gate_Backpropagation-Free_Test-Time_Adaptation_for_Point_Clouds_Classification_via_Token_ICCV_2025_paper.pdf).
- **Moslem Yazdanpanah, Ali Bahri, Mehrdad Noori, Sahar Dastani, Samuel Barbeau, David Osowiechi, Gustavo Adolfo Vargas Hakim, Ismail Ben Ayed and Christian Desrosiers**, *Revisiting Layer Normalization for Point Cloud Test Time Adaptation*, WACV2026, pp.537–546. [Paper](https://openaccess.thecvf.com/content/WACV2026/papers/Yazdanpanah_Revisiting_Layer_Normalization_for_Point_Cloud_Test_Time_Adaptation_WACV_2026_paper.pdf).

[Inference/Open] PG, LN and SATOR repeat all45 source cells across three datasets to printed precision; MAMVI's ModelNet row differs in upsampling/lidar and mean. These fingerprints help track reporting families but do not show whether each source row was independently rerun, inherited, or copied. No paper is accused of copying from numerical equality. LoTT's57.6/59.3 equality with original3DD source means is also insufficient to establish provenance, especially given its explicit all-severity description.

## Can corruption generation and randomness explain our gaps?

[Code] Fresh official3DD inspection pins `main` to`107305fd7baf40b359f31c07d235599198be7324`. The released generator has its NumPy seed call commented out and loops over severity8, whereas the released evaluator loads`data_<corruption>_5.npy`. Gaussian8 usesσ.045, Gaussian5σ.03. AppendixA, supplementPDFp1, statesσ.03, uniform±.05 and rotation±15; these support the severity5 interpretation, although the main tables do not explicitly name a numbered severity. The local generator was already repaired to explicit severity(default5) and optional seed. [Pinned official guide](https://github.com/hamidreza-dastmalchi/3DD-TTA/blob/107305fd7baf40b359f31c07d235599198be7324/data/readme.md) recommends the fixed Zenodo archive for ModelNet and generated corruptions for ShapeNet/Scan.

| Explanation | Evidence relevant to this repository | Supported conclusion |
|---|---|---|
| Wrong Scan severity8 instead of5 | Archived183/581=31.4974% Gaussian5 versus108/581=18.5886% Gaussian8 | Material known mistake, fixed; +12.9088pp does not close59.0 paper source target |
| Global paper severity1 instead of5 | Existing seed0 full13 source mean49.2784% versus corresponding paper69.0; only Gaussian is close | A simple global5→1 swap is rejected by the broader profile |
| Different Scan split | No-background Gaussian42.6850%, versus main31.4974%; PB_T50_RS is worse for released checkpoint | Split can matter materially; neither tested substitution reproduces paper profile |
| Corruption random draw/order | Unseeded released generators; stochastic point selection/noise and rendering; no author array hashes | Plausible for generated datasets, magnitude not established; cannot claim it explains27.5pp Gaussian gap |
| ModelNet regenerated noise | Canonical Zenodo archive/current arrays already have established byte parity in local evidence | Local random regeneration is not a sufficient explanation for current canonical-data gap |
| Classifier checkpoint/training | SVWA differs from3DD weights; GSD source checkpoints unresolved; later new backbones/VLFMs | A concrete confound; exact paper-asset attribution still open |
| Aggregation or reported values | LoTT all-severity statement; original table severity label not explicit; README internal inconsistency | Match denominator/severity and printed row before comparing; matching means are insufficient |
| LION sampling/dropout/environment | Existing original-path studies isolate some small changes; source gap occurs before LION | Relevant to TTA variance, cannot explain a source-only discrepancy by itself |

[Run] The historical original seed0 eval/raw ModelNet result is63.7061% with source53.6899%, versus paper65.7/source57.6. Recent six-backbone-case studies use published alternative weights, not exact GSDTTA assets. Their results cannot establish exact paper reproduction. This literature task adds no source-model experiment or new stochasticity estimate. See [reproduction diagnosis](reproduction_gap.md), [Scan severity evidence](colab_scanobjectnn_severity5_20261017.md) and [multibackbone final findings](gsd_multibackbone_final_findings_20261019.md).

[Code/Inference] Original README's ModelNet printed66.1 is separate from paper65.7. At the pinned revision its displayed rounded corruption cells average **65.44**, so the printed66.1 is not explained by rounding those cells. The paper's corresponding cells average65.6933. Preserve both records; this is a reporting inconsistency, not an inferred new benchmark result.

## Directly citing publication inventory

The catalogue below deduplicates venue/preprint identities. Full authors, primary URLs, exact reference locations and access status are in the committed [CSV](evidence/3dd_tta_citation_audit_20261025/citing_publications.csv). The three limited-access works are not counted as fully reviewed numerical negatives. Each has NA in all three dataset fields rather than invented numbers.

| Work | Full author list; year | Primary evidence and classification |
|---|---|---|
| [3D Test-time Adaptation via Graph Spectral Driven Point Shift](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf) | Xin Wei; Qin Yang; Yijie Fang; Mingrui Zhu; Nannan Wang; 2025 | Reference[33] PDFp10; related workp8; §4.2p5 author rerun statement. 3DD DGCNN ModelNet40-C and ScanObjectNN-C; no ShapeNet-C row |
| [Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](https://ojs.aaai.org/index.php/AAAI/article/download/37888/41850) | Mehran Tamjidi; Hamidreza Dastmalchi; Mohammadreza Alimoradijazi; Ali Cheraghian; Aijun An; Morteza Saberi; 2025 preprint /2026 AAAI | Author-year Dastmalchietal2025, bibliographyPDFp8; baseline/implementationp5. 3DD Uni3D-Large all3datasets; same authors overlap original |
| [MAMVI: 3D Test-Time Adaptation via Masked Multi-View Point Clouds](https://arxiv.org/abs/2606.12939) | Inseok Kong; Geunyoung Jung; Jiyoung Jung; 2026 | Reference[5] PDFp14; §1/2.1. Citation only; Tables1–3 no3DD row; source-only context extracted |
| [SATOR: Backpropagation-Free 3D Test-Time Adaptation via Sample-Adaptive Token Routing](https://arxiv.org/abs/2610.21367) | Inseok Kong; Geunyoung Jung; Jiyoung Jung; 2026 | Reference[5] PDFp9; §1/2.1. Citation only; Tables1–3 no3DD row; source-only context extracted |
| [Low-Rank Test-Time Training for Pre-Trained Point Cloud Models](https://openaccess.thecvf.com/content/CVPR2026/papers/Ye_Low-Rank_Test-Time_Training_for_Pre-Trained_Point_Cloud_Models_CVPR_2026_paper.pdf) | Ouyangzi Ye; Feifei Shao; Kexin Li; Yawei Luo; Zikai Song; Ping Liu; Fengda Zhang; Hongwei Wang; Jun Xiao; 2026 | Reference[5] PDFp9; related workp3. Citation only; Table1 no3DD row; allseverity MATE-SO source context |
| [ETTA: Efficient Test-Time Adaptation for Vision-Language Models through Dynamic Embedding Updates](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_1137/paper.pdf) | Hamidreza Dastmalchi; Aijun An; Ali Cheraghian; 2025 | Reference[9] PDFp11. Other task; no relevant3DD point-cloud benchmark row |
| [Hierarchical Adaptive networks with Task vectors for Test-Time Adaptation](https://arxiv.org/abs/2508.09223) | Sameer Ambekar; Marta Hasny; Laura Daza; Daniel M. Lang; Julia A. Schnabel; 2025 preprint /2026 WACV | Reference[15] PDFp18 of retrieved latest version. Other task; no relevant3DD point-cloud benchmark row |
| [Test-time Adaptation for Point Cloud Denoising under Unseen Noise](https://arxiv.org/abs/2610.27410) | Kun Jiang; Shida Fu; Fanhao Yi; Weiqing Li; Zhiyong Su; 2026 preprint /2027 journal | Reference[33] PDFp15; §2.2.1 (HTML numbering differs[7]). Geometric denoising; no3DD classification accuracy on3targetdatasets |
| [Test Time Adaptation Methods for Point Cloud Registration in Laparoscopic Surgery](https://arxiv.org/abs/2608.02883) | Nina Bodelot; Soufiane Belharbi; Eric Granger; 2026 | Reference[5] PDFp9. Registration; no3DD classification accuracy on3targetdatasets |
| [RGBD-to-3D Object Mesh Refinement via Depth Matching and Symmetry Propagation](https://arxiv.org/abs/2610.11187) | Ahyun Seo; Minsu Cho; 2026 | Reference[5] PDFp15. Mesh refinement; no3DD classification accuracy on3targetdatasets |
| [Beyond Global Editing: Per-Instance Disentangled Subspaces for Training-Free Hallucination Mitigation in LVLMs](https://arxiv.org/abs/2608.09344) | Ali Cheraghian; Hamidreza Dastmalchi; Hamed Barzamini; Morteza Saberi; Mojtaba Golzan; Shafin Rahman; Hossein Rahmani; 2026 | Reference[11] PDFp16. LVLM hallucination task; no relevant3DD point-cloud benchmark row |
| [Test-Time Adaptation with Steering Primitives for Realigning Representations](https://queensu.scholaris.ca/server/api/core/bitstreams/b7e60c54-6ed2-4202-a9bd-b3d3046c21a3/content) | Muhammad Sudipto Siam Dip; 2026 | Reference[9] PDFp82 (printedp72). Masters thesis other task. Companion arXiv2609.24111 searched but has no3DD reference: do not inherit thesis citation |
| [SA-P2SB: a structure-aware Schrödinger bridge for scene-level point cloud denoising](https://link.springer.com/article/10.1007/s00371-026-04670-0) | Yuxin Zhang; Tao Zuo; Zhaowei Wu; 2026 | Publisher accessible bibliography Reference43. Publisher preview only; fulltext unavailable; relevant numerical rows unassessed |
| [Geometry-Guided Diffusion SAR Point Cloud Denoising](https://doi.org/10.3390/rs18152458) | Chengwei Zhang; Tao Jiang; Xinhao Xu; Wenjie Li; Fubo Zhang; Longyong Chen; 2026 | Publisher-deposited Crossref reference ref_37 gives exact3DDDOI. Primary fulltext HTTP403; reference metadata only; numerical rows unassessed |
| [Diffusion-Driven Point Cloud Adaption for Depth-Only 6D Pose Estimation in Cluttered Robotics Bin-Picking](https://doi.org/10.1109/ICECET65726.2026.11632335) | Samed Doğan; Nico Leuze; Nicolas Rodriguez-Pena; Nikolas Ivakko; Alfred Schöttl; 2026 | Publisher-deposited Crossref reference ref46 gives exact3DDDOI. IEEEHTTP202 challenge page; fulltext unavailable; numerical rows unassessed |

## Search coverage, assets and limits

[Code/Paper/Open] On2026-10-25, Google Scholar's exact-title query and exact “Cited by” link were attempted first. The web reader failed; ordinary public HTTP returned200 without an access challenge. The title query exposed one citation-only record and “Cited by3,” cluster`11097964353367270493`. Its exact cited-by page yielded MAMVI, SATOR and the Queen's thesis. No CAPTCHA, authentication, proxy or access-control bypass was attempted. Both original HTML responses are archived.

Semantic Scholar's arXiv target returned12 citing entries; OpenAlex's merged WACV/arXiv target returned4. Their unions differ and omit accessible direct references found by topic search, including LoTT-PC. A second SemanticScholar venue-DOI query was rate-limited429; no repeated pressure or bypass followed. These counts cannot be added as if they were distinct or exhaustive. Primary bibliographies, not index edges alone, determined inclusion. [Search ledger](evidence/3dd_tta_citation_audit_20261025/search_ledger.md) lists the explicit search families, excluded candidates and access failures.

Raw literature and pinned code are stored under:

`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/3dd-citation-audit/result/preparation/3dd_citation_audit_20261025T123834Z/`

The adjacent sealed ZIP and SHA/CRC/member validation are identified by the committed [bundle receipt](evidence/3dd_tta_citation_audit_20261025/bundle_receipt.json). Raw PDFs are local ignored artifacts; the small catalogue/matrices/manifests are committed. Git integration alone does not transfer the raw bundle: preserve/copy the sealed ZIP and recheck its receipt before removing this worktree. No Drive backup was attempted in this literature-only task.

[Artifact] Coordinator review integrated audit commit `e661b33` into the study
branch as `659610f`. An identical sealed ZIP is also preserved at
`/Users/batuhan/Desktop/projects/3DD-TTA/result/preparation/3dd_citation_audit_20261025T123834Z.zip`.
Both copies match receipt SHA256 `04cc873fc3a696213a9657ae0551b31c1cef1eb1656260f286db86cd05c8217b`.
Independent coordinator checks passed ZIP size/CRC/safe paths, all107 manifested
member hashes, the extracted validator and tracked/raw CSV field equivalence
(committed CSVs use LF). This verifies archive consistency, not the authors'
experimental assets or results.

The nested`original_protocol_evidence.zip` was supplied by the coordinating root's independent read-only audit:66469bytes, SHA256`1c14607ed9d6ba4b8abcb616301161d2158261ae0767010f3c74097c7e7cd638`, CRC checked. Its arXiv-v1 extraction was cross-checked against venue PDF target rows here. Other public repository pins and URL/byte/SHA/CRC records are in the [retrieval manifest](evidence/3dd_tta_citation_audit_20261025/retrieval_manifest.json) and [GitHub inventory](evidence/3dd_tta_citation_audit_20261025/github_pin_inventory.json).

[Decision/Open] Continue using the documented severity5 released-asset protocol and qualified source controls. To close exact-paper reproduction, obtain original paper evaluation arrays/checkpoint hashes, preprocessing/point-sampling and aggregation details, seeds and raw logs, or an explicit author clarification. Do not tune seeds, severity or checkpoints to reproduce an isolated paper cell. The current conclusion would change if an accessible later study supplied matching Point-MAE assets and an auditable exact rerun, or if the authors' actual assets explained the source fingerprints. None was found in this bounded audit.
