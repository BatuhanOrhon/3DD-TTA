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

## Per-corruption accuracy

| Corruption | N5 terminal% | Tuned N8PM% | Cheap N8PM% | Cheap−PM pp |
|---|---:|---:|---:|---:|
| uniform | 77.3501 | 80.2269 | 80.2674 | +0.0405 |
| gaussian | 74.4327 | 79.4165 | 79.4571 | +0.0405 |
| background | 24.9595 | 30.1053 | 30.5916 | +0.4862 |
| impulse | 70.0162 | 79.2545 | 79.4976 | +0.2431 |
| upsampling | 82.2123 | 84.7650 | 84.7245 | -0.0405 |
| distortion_rbf | 63.1686 | 64.3031 | 64.1410 | -0.1621 |
| distortion_rbf_inv | 65.8833 | 67.4230 | 66.8963 | -0.5267 |
| density | 72.0421 | 72.0827 | 72.7715 | +0.6888 |
| density_inc | 85.5348 | 85.8590 | 85.8590 | +0.0000 |
| shear | 65.6402 | 64.4246 | 65.0324 | +0.6078 |
| rotation | 33.2658 | 34.6029 | 34.2788 | -0.3241 |
| cutout | 70.9076 | 72.4473 | 72.6904 | +0.2431 |
| distortion | 65.0324 | 65.3160 | 65.5592 | +0.2431 |
| occlusion | 40.1540 | 40.8833 | 40.3971 | -0.4862 |
| lidar | 31.0778 | 34.9271 | 33.6710 | -1.2561 |

## Provenance and validation

[Run] New `20261108T165415Z_background_uniform` ran16:54:15→17:10:26UTC (scientific elapsed969.276848s), same owned `tta-lion-guidance-speed-a100-20261108-r5` / `gpu-a100-hm-euw4a0-1svstdm1jnaaj`, A100-SXM4-80GB/High RAM.20closed cases/7404rows, exact BACKGROUND_UNIFORM_CELL_COMPLETE/exit0, fullB36/B128warmups, unchanged3-modelhashproof. No environment/native rebuild. Source345e85ef09eb5f3d8e5da10a5f152eddc353aabe, overlay5221f7effff99045cfbde8459f7797123dad00a707330e29dfca28bf4854ddde (777391bytes/261files); manifest53d2921caa1c52cc1f8afea191aa5ce690f00b2324c8dda2280de524c61e9b82. Only runner/shared explicit-horizon helper differ from historicalN35 source; ten nested mathematical paths retain exact AST except terminal horizon keyword. Historical defaults stay intact.

[Code/CPU] Initial9RED→43targetedGREEN; review caught stale checkpoint metadata and too-broad original override acceptance before GPU. Fixed with2RED→45targetedGREEN, real scheduler guard5/35, final848projecttests/2skip/500subtests48.49s. Integrated latest93cd210 source passes859tests/2skip/562subtests48.19s. Seven adversarial raw fixtures pass. A later host-only terminal derivation identity guard was added with RED→GREEN negative fixture; its Background acceptance must exactly equal the record embedded in the all15 summary. Initial synthetic completion-marker failure is retained and does not represent a GPU failure.

[Run] Pre-science archive5,736,171bytes,SHA219801c36b4e854799dcc3021b6b649070753d38a974559a3520053b340401b5,416members/native7/source/operators/CPUchecks independentlyaccepted before GPU. Actual model/data/backend/source and explicit5/8/8/checkpoints2/4/6 pass synchronously at first warmup. Raw `result/modelnet40_c/lion_guidance_speed/raw/20261108T165415Z_background_uniform.zip`,5,408,313bytes,SHA2e2ffbc391372b1646567cdc9636a0fa2be2a2760b86369df7e1ebd3cf70a25b, matches Drive `MyDrive/thesis/result/lion_guidance_background_uniform_20261108_r5/raw/20261108T165415Z_background_uniform.zip`. Rawacceptor verifies all member hashes/source/setup/bootstrap/native/model proofs, original labels,7404PMoutcomes/canonicalRNG/order/final36/60timings and resolvedhorizons/checkpoints.

[Run/CPU] Independent reviewer recounts exact111060composite rows/900deduplicated timings/37020terminalreadouts, threephysicalruntimes/fourmeasurementgroups, sourcecrossbindings, allbootstraps andcorrected518/harmed523. Macro14 summary/CSV hashes remain sealed; historicalN35 raw/report/summary remain unchanged. New common-policy analysis `result/modelnet40_c/lion_guidance_speed/analysis/20261108T1711Z_common_horizon_all15/`; terminal-only-readout accuracy `analysis/20261108T1711Z_common_N8_terminal_all15/`. [Review receipt](evidence/lion_guidance_speed_20261107/uniform_all15_independent_review_20261108.json).

[Open] One seed, development-exposed benchmark and user-directed revision after seeingN35; conditional uncertainty, native numerical variation and runtime/corruption confounding. B128 amortized timing is not B1 latency. Saved N8 terminal accuracy does not provide a fresh terminal-only run/time. No isolated cheap-versus-unguided attribution, algorithmic novelty or unseen generalization claim. This completes the requested fixed-policy evaluation, without a new GPU variant or automatic global production-default promotion. Runtime/solemonitor16794 retained; no teardown.


[Run/CPU] 2026-11-08T17:16:58.921552+00:00: Final common-horizon all15 derived bundle Drive-roundtrip verified:795,490bytes,SHA23623f382f8ec850085a6d6ab322a37127cd36b0fd03d47b7e391353624839c9,45members/allmemberidentities/CRC. Includes newall15/N8terminal summaries+CSVs, currentandhistoricalreports, source/raw/model/sealedmacro14receipts, exacthostacceptance/composition/derivation/checksources and independentreview. Drive `MyDrive/thesis/result/lion_guidance_background_uniform_20261108_r5/uniform_all15_derived_20261108T171636Z.zip`. Receipt/laterpublicationmetadata remain outside ownsealedbundle. Scientificraw and priorN35/macroseals unchanged.
