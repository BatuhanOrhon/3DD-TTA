# Agent B sensitivity results — 2026-10-28

[Run/Decision] The registered finite campaign is recorded with **23 successful confirmation jobs and one numerical failure**. Its accepted status is `accepted_partial_confirmation_with_numerical_failure`, not complete four-condition confirmation. The original 3DD-TTA policy has the highest balanced mean among the three fully evaluable frozen methods; the matched ablation does not establish Chamfer-only superiority.

The confirmation uses 140 newly reserved original objects per dataset, all 15 severity-5 corruptions, DGCNN/CurveNet/PointNeXt/Point-MAE, and fixed seeds 0, 1, 2. These 280 objects exclude both earlier 70-object sets. They are not an externally unseen benchmark; earlier benchmark exposure remains explicit. Each method has 50,400 prediction events, but bootstrap sampling has **280 object clusters**, jointly preserving all seeds, heads and corruptions.

| Frozen method / fixed readout | All15 accuracy (%) | Correct / events |
|---|---:|---:|
| Original 3DD-TTA / terminal | 53.763889 | 27097 / 50400 |
| Matched Chamfer / probability mean | 53.013889 | 26719 / 50400 |
| C+S calibrated L2 rho .1 / probability mean | 52.962302 | 26693 / 50400 |
| Tuned Chamfer eta=gamma=.03 / probability mean | Unavailable: numerical failure | No two-seed or successful-case substitution |

Original minus C+S is **+0.801587 percentage points**, conditional 95% object-bootstrap interval [0.148760, 1.557589] pp. The observed per-seed original advantages range from 0.535714 to 1.071429 pp; original wins 5/8 dataset–head means. This is a whole-policy comparison: batching, horizons, style update, final-alpha convention and readout differ. It does not isolate a Chamfer-loss cause.

Matched C minus C+S is **+0.051587 pp**, conditional interval [-0.125000, 0.234127] pp. It crosses zero. C+S corrects 1010 and harms 1036 matched prediction events. No accuracy superiority or equivalence is established for this isolated spectral-term ablation.

[Inference/Open] These are conditional, pointwise intervals across three fixed seeds, not a seed-population or universal-superiority proof. Macro14 excludes Background; other readouts and contrasts remain unadjusted diagnostics. The fork is GSD-inspired latent spectral guidance, not a full GSDTTA reproduction.

| Dataset | Head | C+S (%) | Matched C (%) | Original (%) | Original − C+S (pp) |
|---|---|---:|---:|---:|---:|
| modelnet40_c | dgcnn | 61.0159 | 60.8571 | 63.3651 | +2.3492 |
| modelnet40_c | curvenet | 63.0317 | 63.0000 | 64.3016 | +1.2698 |
| modelnet40_c | pointnext | 63.7143 | 63.8730 | 65.5397 | +1.8254 |
| modelnet40_c | pointmae | 64.3968 | 64.3492 | 63.2540 | -1.1429 |
| scanobjectnn_c | dgcnn | 36.3016 | 36.1587 | 35.3492 | -0.9524 |
| scanobjectnn_c | curvenet | 56.6984 | 56.5556 | 55.8730 | -0.8254 |
| scanobjectnn_c | pointnext | 44.6349 | 44.9365 | 47.6032 | +2.9683 |
| scanobjectnn_c | pointmae | 33.9048 | 34.3810 | 34.8254 | +0.9206 |

[Run] Development evaluated 19 registered configurations: six C, six C+S, six S including three explicitly reused accepted baselines, plus original3DD. The spectral-family choice had 12 candidates versus six C candidates. Calibrated L2, Huber and per-mode power losses were tested in development; only the frozen best spectral candidate was confirmed. Development selection scores were C-step-high 56.785714%, CS-L2-rho.1 55.345238%, original 55.238095%. They must not be compared with confirmation scores as a causal change: the object cohorts differ.

The frozen spectral candidate uses eta=gamma=.01, p=.95 and calibrated L2 rho=.1; matched C uses the same rates/p with the spectral term removed. Tuned C uses .03/.03/.95. Original keeps B32, 5/35 reverse updates, original style, finalalpha=True and terminal readout. GSD-host controls keep B70, 10/30, updated style, finalalpha=False and the registered probability-mean readout. C+S at .03/.03 was not in this finite grid.

[Run] Registered slot20 (tuned C / ModelNet / seed2) stopped at `Classifier input contains nonfinite XYZ coordinates` during a clean trajectory probe in Background batch1, selected positions70–139. Uniform/Gaussian both batches and Background batch0 completed. The exact first invalid tensor and individual original object are not identifiable from saved logs. Four head artifacts share this one worker failure; they are not four independent failed experiments. No clipping, fallback, parameter change, failed-slot retry or hidden replacement occurred. Slots21–23 were run once in a separately reviewed immutable output; they all passed. Tuned C has 5/6 successful dataset–seed jobs, so its full three-seed endpoint/CI stays null.

All 92 complete ZIPs and four failed ZIPs retain provenance. Complete-method statistics use the three full method grids (18 jobs, 72 ZIPs). Successful tuned-C rows are diagnostics only. The failed artifacts omit one late executed-source inventory entry; its absence is retained explicitly and only failed-provenance metadata recovers it from the exact embedded sealed source ZIP and member bytes. No failed-prefix prediction contributes to accuracy, and all other missing/mismatched source identities remain stops.

[Run/Code] Original raw archives remain immutable. Accepted analysis verifies source/native/checkpoint/frozen-state/IDs/labels/RNG/trajectory contracts, the two phase archives, one cumulative ledger and all180 physically downloaded NPZs. CPU cache verification used NumPy1 with CUDA uninitialized; local independent analysis used Python3.10.18/NumPy1.21.6. Twelve failure-adapter fixtures and eight remaining-runner fixtures pass; genuine full composite ingestion and a separate derived-only numerical review pass. Instrumented fresh/cache-replay costs in `costs.csv` are not deployment-latency comparisons.

| Evidence | Bytes | SHA-256 |
|---|---:|---|
| Original failed phase | 1159246521 | f586870206961f88d38daa7ed9da632fddfa51356075e114c5f3b98f020ebe5e |
| Reviewed remaining phase | 150730036 | 7b676a7b4d6fefc8f569e77cf962f599baa45c1f1bcb5cf0b490210e1cdccd6d |
| Failure +180-cache proof | 774792352 | 2a003500c74feafbd6aa6ac03363ec57e6330a67707fc22c803865081d15b898 |
| Partial acceptance binder | 722266 | dc5b854e27ecd96f1d04a5420332749e0b73a7574f011dadc7e4733fd46d68df |

The binder passed an actual Drive→local roundtrip and identical selected analysis bytes, CRC and SHA. Its Drive directory is `MyDrive/thesis/result/guidance_component_study_20261025/phase-evidence/sensitivity_partial_confirmation_acceptance_20261028T203251841461Z/`. Raw phase and cache Drive paths remain in their receipts.

Scientific source remains commit8169271c9058995e8dc128062848a253baa13f10 plus sealed source ZIP73d9fbe1…b37b8 and manifest599b8dcb…8eef; it is distinct from the local analysis/operator branch HEAD. Frozen selection SHAea245f8a…c360 and accepted-development SHAf94afc83…54ed remain unchanged. The reviewed continuation plan13ff2524…40f6c excludes failed20 and runs only21–23.

Measured cumulative campaign time is 14836.554263s (4.121265h), including failed work and stage sealing, below the approved19800s cap. CPU setup, external transfer/analysis and billing lifetime are separate as declared. No planned GPU research remains. Session `tta-guidance-components-a100-20261028-r4` and monitor5314 remain live; teardown was not authorized.

Derived tables and exact acceptance: [analysis folder](../result/preparation/guidance_component_sensitivity_20261025/recovery_20261028_r4/local_partial_confirmation_analysis_v1/), [final verification](../result/preparation/guidance_component_sensitivity_20261025/recovery_20261028_r4/final_local_verification.json), [current study history](guidance_component_sensitivity_study_20261025.md). Further full-dataset/severity/seed generalization, exact failure-tensor diagnosis and any new matched-rate grid require a separate scoped study; no post-hoc rerun is claimed here.
