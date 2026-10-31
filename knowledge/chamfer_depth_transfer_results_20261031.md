# Frozen N8 transfer: accepted results — 2026-10-31

[Run/Code/Inference] The fixed candidate improves the all15 probability-mean accuracy over the historical C control on all eight dataset/head endpoints in the reused140-object cohorts. All eight conditional95% object-cluster intervals for those differences exclude zero. This is evidence for the complete selected policy, not a proof that8steps or its rates are globally optimal. No global default is changed.

All two technical smokes and six research jobs passed without numerical failures or retries. Each research job shares one reconstruction across four frozen heads; the heads are reported separately, not ensembled. Independent local acceptance validates32headZIPs (24research +8technical),454physical caches,7540snapshot files and all source/RNG/state/label/ID guards. A second reviewer recomputed all1800research prediction shards from24rawZIPs and reproduced the means/SD and joint object intervals to1e-10.

The candidate is normal8/Background35, eta=.003 (conditioning/style), gamma=.03 (local), p(lambda)=.95 retained fraction, batch32, original encoded decoder style, DDIM final-alpha-to-one=True, raw/eval LION with EMAoff. Equal temperature1 probability mean uses clean2/4/6 + terminal8; Background uses clean9/18/27 + terminal35. Spectral, semantic, sparse and weighted additions are off. The historical p=.95 setting is distinct from the paper's p=.96.

Both datasets use all15 corruptions/severity5, exactly140original IDs and seeds0/1/2. ModelNet arrays contain2468objects/40classes; Scan uses the accepted BFTT3D581-object main_split/clean1024/all15s5 asset package and15-class checkpoints. This does not reuse the invalid older Scan generation receipt. Each research head has6300prediction events. Cohorts were already exposed in C/H experiments, so this is exploratory transfer rather than externally unseen validation. The old256-object depth confirmation mean68.741319% belongs to a different cohort and must not be compared directly to the new Point-MAE66.000000%.

The following accuracies are three-seed all15 means; SD is sample SD in percentage points. Intervals resample140original objects jointly across corruptions and the three fixed seeds,10000replicates, bootstrap seed20261030. They are conditional and uncorrected for multiple comparisons.

| Dataset | Head | N8 PM % ± SD pp | Own terminal % | PM − own terminal pp [95% CI] |
|---|---|---:|---:|---|
|ModelNet40-C|Point-MAE|66.000000 ± 0.503953|64.968254|+1.032 [+0.333, +1.762]|
|ModelNet40-C|DGCNN|69.571429 ± 0.666667|68.476190|+1.095 [+0.397, +1.810]|
|ModelNet40-C|CurveNet|71.761905 ± 0.082479|70.111111|+1.651 [+0.937, +2.413]|
|ModelNet40-C|PointNeXt|70.920635 ± 0.262265|69.412698|+1.508 [+0.730, +2.317]|
|ScanObjectNN-C|Point-MAE|41.857143 ± 0.312259|40.809524|+1.048 [+0.190, +1.952]|
|ScanObjectNN-C|DGCNN|40.317460 ± 0.262265|38.936508|+1.381 [+0.524, +2.206]|
|ScanObjectNN-C|CurveNet|57.888889 ± 0.476983|56.650794|+1.238 [+0.540, +1.952]|
|ScanObjectNN-C|PointNeXt|50.698413 ± 0.262265|50.523810|+0.175 [-0.460, +0.825]|

PM exceeds its own terminal in all eight mean endpoints; seven intervals exclude zero. Scan PointNeXt is the exception: +.174603pp, CI[-.460317,+.825397]; its seed1 readout difference is negative (−.095238pp). Thus PM is not uniformly beneficial for every source/seed.

Historical C/H comparisons use the same140IDs, labels, checkpoints and fixed seeds, but C/H uses batch70, normal10/Background30, eta=gamma=.01, updated style and final-alpha-to-one=False. They are whole-policy comparisons across source/native processes, not common-trajectory or isolated depth/rate/style effects. The table keeps the main Point-MAE-teacher H phase separate from the later CurveNet-teacher followup.

| Dataset | Head | Historical C PM % | N8 PM % | N8 − C pp [95% CI] | N8 − main H pp [95% CI] |
|---|---|---:|---:|---|---|
|ModelNet40-C|Point-MAE|65.000000|66.000000|+1.000 [+0.127, +1.857]|+0.825 [-0.048, +1.698]|
|ModelNet40-C|DGCNN|66.777778|69.571429|+2.794 [+1.651, +3.968]|+2.381 [+1.190, +3.603]|
|ModelNet40-C|CurveNet|68.571429|71.761905|+3.190 [+2.143, +4.302]|+2.952 [+1.968, +3.984]|
|ModelNet40-C|PointNeXt|67.253968|70.920635|+3.667 [+2.413, +5.000]|+3.190 [+1.984, +4.492]|
|ScanObjectNN-C|Point-MAE|38.936508|41.857143|+2.921 [+1.793, +4.143]|+2.587 [+1.524, +3.698]|
|ScanObjectNN-C|DGCNN|38.492063|40.317460|+1.825 [+0.746, +2.921]|+1.460 [+0.286, +2.651]|
|ScanObjectNN-C|CurveNet|55.968254|57.888889|+1.921 [+0.841, +3.032]|+2.127 [+1.095, +3.190]|
|ScanObjectNN-C|PointNeXt|46.936508|50.698413|+3.762 [+2.524, +5.063]|+3.921 [+2.762, +5.111]|

All24per-seed contrasts against main C and all24against main H have positive point differences. Main H is exceeded in all eight mean point estimates, but ModelNet Point-MAE's +.825397pp CI[-.047619,+1.698413] includes zero. Do not report universal superiority over H.

| Scan head | Followup own C PM % | CurveNet-H PM % | N8 − followup C pp [95% CI] | N8 − CurveNet-H pp [95% CI] |
|---|---:|---:|---|---|
|Point-MAE|39.000000|38.936508|+2.857 [+1.746, +4.048]|+2.921 [+1.794, +4.095]|
|DGCNN|38.380952|38.571429|+1.937 [+0.841, +3.095]|+1.746 [+0.651, +2.889]|
|CurveNet|55.666667|55.904762|+2.222 [+1.206, +3.286]|+1.984 [+0.968, +3.048]|
|PointNeXt|46.698413|47.111111|+4.000 [+2.873, +5.159]|+3.587 [+2.429, +4.762]|

All four N8−CurveNet-H followup intervals exclude zero. These are separate historical endpoints with their own matched C; do not substitute the main C table into that followup.

Background materially contributes to the gains. N8−mainC Background changes range from+5.952381pp to+15.238095pp across the eight endpoints. Improvements are not uniform across individual corruptions: e.g. ModelNet Point-MAE Lidar−3.809524pp and Scan CurveNet Impulse−4.285714pp. Full per-corruption values are retained in corruption_comparison_main_c.csv.

| Dataset | Head | N8 − main C macro14 pp [95% CI] |
|---|---|---|
|ModelNet40-C|Point-MAE|+0.442 [-0.323, +1.224]|
|ModelNet40-C|DGCNN|+2.466 [+1.327, +3.639]|
|ModelNet40-C|CurveNet|+2.704 [+1.616, +3.844]|
|ModelNet40-C|PointNeXt|+3.027 [+1.820, +4.337]|
|ScanObjectNN-C|Point-MAE|+2.041 [+0.952, +3.248]|
|ScanObjectNN-C|DGCNN|+1.429 [+0.459, +2.415]|
|ScanObjectNN-C|CurveNet|+1.633 [+0.612, +2.704]|
|ScanObjectNN-C|PointNeXt|+3.129 [+1.905, +4.371]|

Seven macro14 intervals against main C exclude zero; ModelNet Point-MAE +.442177pp CI[−.323129,+1.224490] does not. This limits an across-corruption claim even when its all15 result is positive.

For original3DD, only six non-Point-MAE full seed0 archives have indexed predictions that can be subset to the same140IDs. Compare candidate seed0 with original seed0 once; no seed0 triplication and no three-seed original claim. ModelNet Point-MAE full originals lack indexed predictions; Scan Point-MAE full original archive is unavailable (see preserved errata).

| Dataset | Head | Original seed0 terminal % | N8 seed0 PM % | Difference pp [95% CI] |
|---|---|---:|---:|---|
|ModelNet40-C|DGCNN|67.238095|70.047619|+2.810 [+1.238, +4.381]|
|ModelNet40-C|CurveNet|69.428571|71.714286|+2.286 [+0.714, +3.810]|
|ModelNet40-C|PointNeXt|67.809524|70.904762|+3.095 [+1.381, +4.810]|
|ScanObjectNN-C|DGCNN|38.142857|40.142857|+2.000 [+0.381, +3.571]|
|ScanObjectNN-C|CurveNet|55.476190|57.714286|+2.238 [+0.905, +3.619]|
|ScanObjectNN-C|PointNeXt|51.000000|51.000000|+0.000 [-1.619, +1.571]|

Five original seed0 all15 contrasts have intervals above zero; Scan PointNeXt is a51.000000% tie with CI[−1.619048,+1.571429]. Its new three-seed mean50.698413% cannot be called a three-seed improvement over an original available only at seed0. ModelNet CurveNet's original comparison also loses a clear interval advantage under macro14 (+1.428571pp, CI[−.204082,+2.959184]).

Actual controller wall was2412.154642seconds (40min12s), below3300seconds. This includes workers, CPU acceptance and verified Drive copies. Research job accepted wall durations were approximately377–389seconds; two smokes57.915/55.393seconds. Setup, account-consent waiting and subsequent full artifact download are separate. No isolated speed improvement over the paper or historical B70 C/H protocol is established.

The full archive is1368701809bytes, SHA256 eb9386ee10694c56674bff745c26ba1bd6cc6887f6577146cc0cc6db705a6a8d. It contains32final headZIPs plus32embedded source_snapshot ZIPs, not64independent experiments. All six256MiB-or-smaller parts, full SHA, ZIP path/type/size/CRC and7540file identities passed the Drive/local roundtrip. Raw immutable archive and parts remain under result/preparation/chamfer_depth_transfer_20261031/retrieval_20261031T112631Z/;32byte-identical canonical rawZIP hardlinks are under result/{dataset}/3dd_chamfer_depth_study/.

Executed research: experiment/chamfer-depth-transfer-20261031 source commit4a5c137403fd747d465b2532c2f45419744dd0eb; sourceZIP b2293ca319b5dd1128b7a608b4e2085e28e07b75b53c11ba4fb07e845fae736d; manifest de27d8a56b3ef9d19c3c266a610111ef49d3b70150e7ee7528ca32d227be375e; actual plan9154038be293c5ec9482a7010c4c7e0f1777763ab9421e58ab40d1ee54793554. CPU strict8head/no-forward proof archive7117185bytes SHA91cba85025d54ea4881fa86ad217f5ec57efc4e9f828b8c99f5ef82888e49ea6; CPU receipt5ef05d72f6ab2f777aaa46abc291228bf0181f8bfd2a909aaa5b2632586cb972. Local analysis uses Python3.12/NumPy1, operator b80e362b2907623bbc940a5b25a2501e417da0a5daa5aa88ebcbd21ccb8804f9; it recomputes raw logits and physical RNG/cache proof rather than trusting controller scores.

The canonical ten-cell setup and GPU native/data/weight checks passed on A100-SXM4-80GB/Python3.8.20/Torch2.1.2+cu121/NumPy1.21.2. Only the known cell2 fastrlock failure was deferred; fastrlock0.8 later installed. Two pre-existing dependency-resolver warnings remain recorded. The setup checkout dev@a458cd46 differs from the archived research source. The required user-consented Drive mount and bounded execution adaptation are recorded in the study journal.

Decision: retain this fixed policy as a strong candidate for the next baseline decision. It transfers positively versus the recent C controls across all eight endpoints, with specific null/regression cases preserved. This finite reused-cohort study neither identifies global/per-model optimal N/eta/gamma/p nor replaces full-data or unseen-cohort confirmation. Background uses35steps, so the policy is not uniformly8step; no H/GSD combination was tested or made default. Native Background repeatability and Scan DGCNN source-provenance gaps remain separate open questions.

[Execution protocol and dated history](chamfer_depth_transfer_study_20261031.md) · [Historical depth confirmation](chamfer_depth_tuning_results_20261030.md) · [C/H context](chamfer_guidance_methods_results_20261031.md) · [Historical Point-MAE errata](../result/preparation/chamfer_depth_transfer_20261031/cohort140/errata.json).
