# Common-horizon Background and all15 results — 2026-11-08

[Run/CPU/Inference] **With N5 terminal and N8 candidates applied to every corruption, cheap N8+PM saves 27.673026% pipeline time with -0.013506 percentage-point accuracy difference.** All three same-margin diagnostic checks pass; the paired original-object95% interval is[-0.180983,0.153971]pp, above the −.3pp lower margin. This is a separate user-directed protocol revision after seeing historicalN35 results, not the untouched original primary test or unseen confirmation.

[User decision/Run] The user rejects a special35-step Background horizon because corruption identity is unavailable at inference. This run changes only Background to the existing normal policy: N5terminal/gamma=eta=.01; tunedN8/gamma=.03,eta=.003; p=.95, PM[2,4,6,terminal]; cheapN8/tau=.1 unchanged. Original Background inputs/labels/name are retained; no corruption aliasing or label-dependent selection. Other14 results are reused byte-for-byte. Every corruption now has the same horizon within each method.

**Absolute Background accuracy drops sharply with the short fixed horizon.** HistoricalN35 tunedPM64.303079% becomes N8PM30.105348%; cheap60.696921% becomes30.591572%; the default-rate terminal control61.102107% atN35 becomes24.959481% atN5. Higher absolute N35 accuracy and its special routing are preserved in [the historical report](lion_guidance_all15_results_20261108.md). This new comparison does not establish that eight steps repair severe Background well, nor isolate reverse-update count from starting noise/readout positions/native variation.

## Background only: all2,468 objects, severity5, seed0, commonB128/final36

| Method | Correct/2,468 | Accuracy% | Pipeline seconds | B128-amortized ms/inference |
|---|---:|---:|---:|---:|
| N5 terminal, gamma/eta=.01/.01 | 616 | 24.959481 | 198.321 | 80.357 |
| Tuned N8+PM, gamma/eta=.03/.003 | 743 | 30.105348 | 365.276 | 148.005 |
| Cheap N8+PM, fixedtau=.1 | 755 | 30.591572 | 264.146 | 107.028 |
| Tuned N8 terminal from saved PM trace | 741 | 30.024311 | Not measured separately | Not measured separately |

[Run/Inference] Background cheap−PM is+0.486224pp (44corrected/32harmed/net+12), paired95%CI[-0.202593,1.175041]pp; 27.685902% less time. Its accuracy interval includes zero, so a positive point estimate alone is not established superiority. PM versus its own saved tuned N8 terminal adds2 correct predictions (+0.081037pp).

## New all15 common-horizon composite

| Method | Correct/37,020 | Accuracy% | Pipeline seconds | B128-amortized ms/inference |
|---|---:|---:|---:|---:|
| N5 terminal, gamma/eta=.01/.01 | 22,747 | 61.445165 | 2975.721 | 80.381 |
| Tuned N8+PM, gamma/eta=.03/.003 | 23,595 | 63.735818 | 5489.958 | 148.297 |
| Cheap N8+PM, fixedtau=.1 | 23,590 | 63.722312 | 3970.720 | 107.259 |
| Tuned N8 terminal from saved PM traces | 23,164 | 62.571583 | Not measured separately | Not measured separately |

[Run/Inference] Cheap versus N8PM: −0.013506pp,95%CI[-0.180983,0.153971], 27.673026% less time, 1.382610× throughput, 1519.237s saved across37,020 classifications. It corrects518 and harms523 predictions, net−5: macro14−17 plus newBackground+12. Time saving≥15%, lower95%accuracy>−.3pp, and nondegenerate differences all pass under this revised policy. The descriptive fixed-batch timing interval[27.655990,27.692698]% does not capture hardware/runtime uncertainty.

Cheap versus N5 terminal: +2.277147pp, but 33.437234% more time. N8PM versus N5 terminal:+2.290654pp and84.491660% more time. These whole-policy comparisons also change guidance rates/readouts, not just depth. Same-trace tuned N8 terminal is62.571583%; PM adds1.164236pp overall. On the preserved macro14 traces, tuned N8 terminal remains64.896388%, PMgain1.241607pp.

## Complete corruption-level record

[Run/CPU] Every row below uses all **2,468 original objects**, severity5, seed0, one measured pass, common B128/final36. Parentheses contain the exact number correct out of 2,468. N8 terminal comes from the last saved tuned N8+PM readout. All original/cheap/terminal labels, IDs and source identities are cross-checked against the sealed tables.

### Accuracy and exact counts

| Corruption | N5 terminal% (correct) | Tuned N8 terminal% (correct) | N8+PM% (correct) | Cheap N8+PM% (correct) |
|---|---:|---:|---:|---:|
| uniform | 77.3501 (1909) | 77.2285 (1906) | 80.2269 (1980) | 80.2674 (1981) |
| gaussian | 74.4327 (1837) | 76.8233 (1896) | 79.4165 (1960) | 79.4571 (1961) |
| background | 24.9595 (616) | 30.0243 (741) | 30.1053 (743) | 30.5916 (755) |
| impulse | 70.0162 (1728) | 77.3096 (1908) | 79.2545 (1956) | 79.4976 (1962) |
| upsampling | 82.2123 (2029) | 83.5089 (2061) | 84.7650 (2092) | 84.7245 (2091) |
| distortion_rbf | 63.1686 (1559) | 62.4797 (1542) | 64.3031 (1587) | 64.1410 (1583) |
| distortion_rbf_inv | 65.8833 (1626) | 65.3971 (1614) | 67.4230 (1664) | 66.8963 (1651) |
| density | 72.0421 (1778) | 71.7180 (1770) | 72.0827 (1779) | 72.7715 (1796) |
| density_inc | 85.5348 (2111) | 85.7780 (2117) | 85.8590 (2119) | 85.8590 (2119) |
| shear | 65.6402 (1620) | 64.6272 (1595) | 64.4246 (1590) | 65.0324 (1605) |
| rotation | 33.2658 (821) | 33.0227 (815) | 34.6029 (854) | 34.2788 (846) |
| cutout | 70.9076 (1750) | 72.3258 (1785) | 72.4473 (1788) | 72.6904 (1794) |
| distortion | 65.0324 (1605) | 64.3031 (1587) | 65.3160 (1612) | 65.5592 (1618) |
| occlusion | 40.1540 (991) | 40.3971 (997) | 40.8833 (1009) | 40.3971 (997) |
| lidar | 31.0778 (767) | 33.6305 (830) | 34.9271 (862) | 33.6710 (831) |

### Paired accuracy changes and PM contribution

All deltas below are **percentage points**. Corrected/harmed compares cheap with tuned N8+PM on the same object/corruption; it is not the difference in separately sampled cohorts. Arms use matched method-independent seeds with fresh native encodes; identical encoded states or bitwise trajectories are not established. Object bootstrap uses 10,000 draws and seed20261107. These per-corruption intervals are descriptive and unadjusted for 15 simultaneous comparisons.

| Corruption | PM−own N8 terminal pp | Cheap−PM pp | Cheap−PM95%CI pp | Corrected | Harmed | Net correct |
|---|---:|---:|---:|---:|---:|---:|
| uniform | +2.9984 | +0.0405 | [-0.6078, +0.6888] | 35 | 34 | +1 |
| gaussian | +2.5932 | +0.0405 | [-0.5673, +0.6483] | 30 | 29 | +1 |
| background | +0.0810 | +0.4862 | [-0.2026, +1.1750] | 44 | 32 | +12 |
| impulse | +1.9449 | +0.2431 | [-0.2836, +0.7699] | 27 | 21 | +6 |
| upsampling | +1.2561 | -0.0405 | [-0.5267, +0.4862] | 20 | 21 | -1 |
| distortion_rbf | +1.8233 | -0.1621 | [-0.8914, +0.5267] | 38 | 42 | -4 |
| distortion_rbf_inv | +2.0259 | -0.5267 | [-1.2561, +0.2026] | 36 | 49 | -13 |
| density | +0.3647 | +0.6888 | [-0.0405, +1.4587] | 54 | 37 | +17 |
| density_inc | +0.0810 | +0.0000 | [-0.5673, +0.6078] | 29 | 29 | +0 |
| shear | -0.2026 | +0.6078 | [-0.0810, +1.2966] | 44 | 29 | +15 |
| rotation | +1.5802 | -0.3241 | [-0.9319, +0.2836] | 26 | 34 | -8 |
| cutout | +0.1216 | +0.2431 | [-0.4467, +0.9724] | 44 | 38 | +6 |
| distortion | +1.0130 | +0.2431 | [-0.4457, +0.9319] | 42 | 36 | +6 |
| occlusion | +0.4862 | -0.4862 | [-1.1750, +0.1621] | 29 | 41 | -12 |
| lidar | +1.2966 | -1.2561 | [-1.9449, -0.6078] | 20 | 51 | -31 |

### Measured pipeline cost by corruption

Each time is summed once over 20 batches per method, including the final 36-object batch. Cells show **total seconds / amortized milliseconds per input**. They exclude setup/loading/archive work and are not batch1 latency. N8 terminal-only time remains **unmeasured for every corruption**; N8+PM time must not be assigned to that readout.

| Corruption | N5 terminal s / ms | N8+PM s / ms | Cheap N8+PM s / ms | Time saved s | Less time vs N8+PM% |
|---|---:|---:|---:|---:|---:|
| uniform | 198.872 / 80.580 | 366.683 / 148.575 | 265.393 / 107.534 | 101.290 | 27.6233 |
| gaussian | 198.933 / 80.605 | 366.802 / 148.623 | 265.467 / 107.564 | 101.335 | 27.6265 |
| background | 198.321 / 80.357 | 365.276 / 148.005 | 264.146 / 107.028 | 101.130 | 27.6859 |
| impulse | 199.024 / 80.642 | 366.959 / 148.687 | 265.578 / 107.609 | 101.381 | 27.6272 |
| upsampling | 194.478 / 78.800 | 362.495 / 146.878 | 261.029 / 105.765 | 101.467 | 27.9911 |
| distortion_rbf | 198.901 / 80.592 | 366.822 / 148.631 | 265.384 / 107.530 | 101.438 | 27.6531 |
| distortion_rbf_inv | 198.758 / 80.534 | 366.530 / 148.513 | 265.108 / 107.418 | 101.422 | 27.6709 |
| density | 199.200 / 80.713 | 366.676 / 148.572 | 265.391 / 107.533 | 101.285 | 27.6225 |
| density_inc | 198.520 / 80.437 | 366.180 / 148.371 | 264.889 / 107.330 | 101.291 | 27.6614 |
| shear | 198.599 / 80.470 | 366.342 / 148.437 | 264.971 / 107.363 | 101.371 | 27.6712 |
| rotation | 198.940 / 80.608 | 366.934 / 148.677 | 265.251 / 107.476 | 101.683 | 27.7115 |
| cutout | 199.259 / 80.737 | 366.831 / 148.635 | 265.526 / 107.588 | 101.305 | 27.6163 |
| distortion | 198.665 / 80.496 | 366.345 / 148.438 | 264.983 / 107.368 | 101.362 | 27.6684 |
| occlusion | 197.533 / 80.038 | 364.447 / 147.669 | 263.732 / 106.861 | 100.715 | 27.6350 |
| lidar | 197.720 / 80.113 | 364.635 / 147.745 | 263.871 / 106.917 | 100.764 | 27.6343 |

### Coverage and source group

| Corruption | Physical runtime group | Original measurement run |
|---|---|---|
| uniform | r3 | `20261107T215537Z_full14` |
| gaussian | r3 | `20261107T215537Z_full14` |
| background | r5 | `20261108T165415Z_background_uniform` |
| impulse | r4 | `20261108T082824Z_continue14` |
| upsampling | r4 | `20261108T082824Z_continue14` |
| distortion_rbf | r4 | `20261108T082824Z_continue14` |
| distortion_rbf_inv | r5 | `20261108T123641Z_continue14` |
| density | r5 | `20261108T123641Z_continue14` |
| density_inc | r5 | `20261108T123641Z_continue14` |
| shear | r5 | `20261108T123641Z_continue14` |
| rotation | r5 | `20261108T123641Z_continue14` |
| cutout | r5 | `20261108T123641Z_continue14` |
| distortion | r5 | `20261108T123641Z_continue14` |
| occlusion | r5 | `20261108T123641Z_continue14` |
| lidar | r5 | `20261108T123641Z_continue14` |

[Run/CPU] The detailed JSON stores all unrounded values, exact counts, source archive hashes, input hashes, timing intervals and observed peak allocated MiB per measured arm. Allocated peaks include the resident shared process; they do not establish standalone incremental memory use. [Machine-readable record](evidence/lion_guidance_speed_20261107/uniform_per_corruption_detail_20261108.json). Counts sum to 22,747 N5,23,164 tuned N8 terminal,23,595 N8+PM and23,590 cheap correct out of37,020; corrected518/harmed523/net−5. Timing sums reproduce2975.721249/5489.957533/3970.720134s. No experimental result or frozen input is overwritten.

## Provenance and validation

[Run] New `20261108T165415Z_background_uniform` ran16:54:15→17:10:26UTC (scientific elapsed969.276848s), same owned `tta-lion-guidance-speed-a100-20261108-r5` / `gpu-a100-hm-euw4a0-1svstdm1jnaaj`, A100-SXM4-80GB/High RAM.20closed cases/7404rows, exact BACKGROUND_UNIFORM_CELL_COMPLETE/exit0, fullB36/B128warmups, unchanged3-modelhashproof. No environment/native rebuild. Source345e85ef09eb5f3d8e5da10a5f152eddc353aabe, overlay5221f7effff99045cfbde8459f7797123dad00a707330e29dfca28bf4854ddde (777391bytes/261files); manifest53d2921caa1c52cc1f8afea191aa5ce690f00b2324c8dda2280de524c61e9b82. Only runner/shared explicit-horizon helper differ from historicalN35 source; ten nested mathematical paths retain exact AST except terminal horizon keyword. Historical defaults stay intact.

[Code/CPU] Initial9RED→43targetedGREEN; review caught stale checkpoint metadata and too-broad original override acceptance before GPU. Fixed with2RED→45targetedGREEN, real scheduler guard5/35, final848projecttests/2skip/500subtests48.49s. Integrated latest93cd210 source passes859tests/2skip/562subtests48.19s. Seven adversarial raw fixtures pass. A later host-only terminal derivation identity guard was added with RED→GREEN negative fixture; its Background acceptance must exactly equal the record embedded in the all15 summary. Initial synthetic completion-marker failure is retained and does not represent a GPU failure.

[Run] Pre-science archive5,736,171bytes,SHA219801c36b4e854799dcc3021b6b649070753d38a974559a3520053b340401b5,416members/native7/source/operators/CPUchecks independentlyaccepted before GPU. Actual model/data/backend/source and explicit5/8/8/checkpoints2/4/6 pass synchronously at first warmup. Raw `result/modelnet40_c/lion_guidance_speed/raw/20261108T165415Z_background_uniform.zip`,5,408,313bytes,SHA2e2ffbc391372b1646567cdc9636a0fa2be2a2760b86369df7e1ebd3cf70a25b, matches Drive `MyDrive/thesis/result/lion_guidance_background_uniform_20261108_r5/raw/20261108T165415Z_background_uniform.zip`. Rawacceptor verifies all member hashes/source/setup/bootstrap/native/model proofs, original labels,7404PMoutcomes/canonicalRNG/order/final36/60timings and resolvedhorizons/checkpoints.

[Run/CPU] Independent reviewer recounts exact111060composite rows/900deduplicated timings/37020terminalreadouts, threephysicalruntimes/fourmeasurementgroups, sourcecrossbindings, allbootstraps andcorrected518/harmed523. Macro14 summary/CSV hashes remain sealed; historicalN35 raw/report/summary remain unchanged. New common-policy analysis `result/modelnet40_c/lion_guidance_speed/analysis/20261108T1711Z_common_horizon_all15/`; terminal-only-readout accuracy `analysis/20261108T1711Z_common_N8_terminal_all15/`. [Review receipt](evidence/lion_guidance_speed_20261107/uniform_all15_independent_review_20261108.json).

[Open] One seed, development-exposed benchmark and user-directed revision after seeingN35; conditional uncertainty, native numerical variation and runtime/corruption confounding. B128 amortized timing is not B1 latency. Saved N8 terminal accuracy does not provide a fresh terminal-only run/time. No isolated cheap-versus-unguided attribution, algorithmic novelty or unseen generalization claim. This completes the requested fixed-policy evaluation, without a new GPU variant or automatic global production-default promotion. Runtime/solemonitor16794 retained; no teardown.


[Run/CPU] 2026-11-08T17:16:58.921552+00:00: Final common-horizon all15 derived bundle Drive-roundtrip verified:795,490bytes,SHA23623f382f8ec850085a6d6ab322a37127cd36b0fd03d47b7e391353624839c9,45members/allmemberidentities/CRC. Includes newall15/N8terminal summaries+CSVs, currentandhistoricalreports, source/raw/model/sealedmacro14receipts, exacthostacceptance/composition/derivation/checksources and independentreview. Drive `MyDrive/thesis/result/lion_guidance_background_uniform_20261108_r5/uniform_all15_derived_20261108T171636Z.zip`. Receipt/laterpublicationmetadata remain outside ownsealedbundle. Scientificraw and priorN35/macroseals unchanged.


[User decision/Run] 2026-11-08T17:40:29.127164+00:00: Following completed scientific runs and verified raw/source/derived/documentation backups, user authorized shutdown. Own speed r5 was stopped successfully after monitor16794 exited; server inventory independently confirms its absence. Earlier retained/live statements above describe their dated state, not the current runtime. Peer r6 archival/shutdown is owned by its original task; final combined closure receipt is in `evidence/lion_guidance_speed_20261107/colab_shutdown_20261108.json`.


[User decision/Run] 2026-11-08T17:42:59.179621+00:00: Both Colab runtimes and their owner monitors are now stopped, independently confirmed by emptyserverinventory after preservation of results. [Final closure receipt](evidence/lion_guidance_speed_20261107/colab_shutdown_20261108.json).
