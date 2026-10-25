# Search boundary and data schema — 2026-10-25

[Paper/Code/Open] This expansion starts at committed study51dd03b and reuses the previous sealed3DD audit as identified primary evidence, then retrieves original MATE/BFTT3D/SMART-PC/CloudFixer/Point-Cache/BayesMM and the original ModelNet40-C benchmark. Primary PDFs/supplements and exact table pages determine numerical inclusion. Sixteen numerical reporting publications are included; the older15direct-citation catalogue is a different inventory. No claim of exhaustive literature coverage or823independent reruns is made.

Three scoped workers reviewed early methods (~10 targeted queries), input methods (~8–12 planned queries plus primary followups; exact count not reconstructed), and VLFM/new classification studies (~13 targeted queries). Their findings, actual retrieval manifests, source URLs, hash checks, exclusion decisions and raw table extractions are preserved in the sealed archive. Full transient search responses were not retained. Counts are coverage descriptions, not a reproducible search-result census.

Coordinator queries, issued via the web tool:

1. `"point cloud" "test-time adaptation" "reproduced" "results"`
2. `"Purge-Gate" "results" "taken"`
3. `"Revisiting Layer Normalization" "reproduce"`
4. `"MAMVI" "baselines" "results"`
5. `"SATOR" "supplementary" "baseline configurations"`
6. `"MAMVI" "supplementary" "SVWA"`
7. `"Revisiting Layer Normalization" "results for other methods"`
8. `"SATOR: Backpropagation-Free" supplemental`
9. `"MAMVI: 3D" supplementary`
10. `"Revisiting Layer Normalization for Point Cloud Test Time Adaptation" supplemental`

Several searches returned unrelated normalization/LLM summaries; none supplied numerical evidence. Public primary downloads and previously hashed PDF snapshots were used for table review. No access-control bypass, author contact or model execution occurred. SATOR/MAMVI baseline supplements were not obtained; their changed baseline numbers remain execution-provenance-unspecified. PCoTTA was inspected but excluded from classification. Generic TTAC/TMLR candidates remain unassessed; segmentation/detection/registration results are not converted into classification entries. Prior limited-access publications remain limited.

## Records and metrics

`reported_results.csv`:823 macro-table observations. The tuple paper,document,page,table,backbone,method,protocol,batch identifies the reported setting; repeated compact/detailed tables are retained. `row_id` is a local pointer, not a run ID. `method_raw` preserves table labels even where a canonical method name is supplied. `is_source_only` includes no-adaptation source variants such as Joint-Training and skeleton-trained models; it never means all checkpoints were trained identically.

`reported_value` and `metric` preserve printed units. `derived_accuracy_percent` equals printed accuracy, or100−ordinary percentage classification error, and is blank for macro-recall. The69error rows are verified ordinary ER in BFTT3D/original corruption benchmark, not normalizedmCE. Empty values and unknown severity remain unknown. `severity_evidence` separates paper-level disclosure from code-level severity5 support; numeric mean agreement does not establish exact severity/asset parity. `batch_size` retains1/64CF-O variants. Protocol cohorts separate five-severity averaging, full15, six-type subsets, class imbalance/recall, and7atomic-corruption VLFM results.

`source_only.csv` is a filtered128-observation view, not128 independent models. Readable tables are generated from the same records. Native JSON records/extraction scripts are in the raw archive; the CSV is the committed complete matrix.

`citation_comparisons.csv` contains77 actual original/later row pairs:29 selected method-reporting cases and48Point-Cache/BayesMMcorruption pairs. Every case links to two row IDs and primary URLs/pages. `descriptive_delta_pp` is not a matched causal effect; `matched_protocol_claim` is false throughout. Explicit reuse, author-reported reruns, unknown provenance and changed aggregation are distinct categories. Equality alone does not assign reuse; difference alone does not assign rerun.

Seven-type VLFM rows retain clean+7cells and both derived denominators in raw records. Fourteen printed-mean flags exceed0.011pp, allowing normal two-decimal cell/mean rounding; flags indicate incompatibility with the tested reduction, not a diagnosis of fraud or a replacement benchmark measurement. Separate Sim-to-Real28rows/12cases are archived and excluded from the823 corruption-matrix count. Local historical results are linked in the report, not added to the16-paper counts.
