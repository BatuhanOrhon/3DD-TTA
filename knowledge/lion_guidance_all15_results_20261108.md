# Full ModelNet40-C all15 extension — 2026-11-08

[Run/CPU/Inference] **The all15 extension saves 29.562521% pipeline time, with -0.286332 percentage-point lower accuracy. The accuracy noninferiority check does not pass.** Its paired original-object 95% interval is [-0.472717,-0.097245]pp; its lower endpoint falls below the required −0.3pp margin. The interval straddles −0.3pp: this does not prove the loss exceeds0.3pp, but noninferiority at that margin is not established. This does not change the previously accepted [macro14 primary conclusion](lion_guidance_macro14_results_20261108.md), which passed. Background is an important failure case of the frozen cheap setting; no retuning or corruption-specific fallback was applied.

All 2,468 original IDs, severity5, all15 corruptions, seed0, one measured pass, common B128/final36. Each arm has37,020 classifications; the combined table has111,060 provenance-bound rows. Macro14 retains the accepted r3/r4/r5 disjoint groups; the separately requested Background phase used the same r5 after macro14 analysis. Thus this is three physical A10080 runtimes and four measurement groups, not four independent runtime replicates. Equal corruption sizes make macro and micro accuracy identical.

## Main results

| Method | Correct/37,020 | Accuracy% | Pipeline seconds | B128-amortized ms/inference |
|---|---:|---:|---:|---:|
| Original terminal N5 normal / N35 Background | 23,639 | 63.854673 | 3866.805 | 104.452 |
| Tuned N8 normal / N35 Background + four equal-PM readouts | 24,439 | 66.015667 | 6290.511 | 169.922 |
| Cheap correction N8 normal / N35 Background + four equal-PM readouts | 24,333 | 65.729335 | 4430.877 | 119.689 |
| Tuned terminal readout from saved N8/35+PM traces | 24,028 | 64.905457 | Not measured separately | Not measured separately |

[Run/Inference] Cheap versus tuned PM: −0.286332pp, 29.562521% less time, 1.419699× throughput, 1859.634s saved. It corrects581 and harms687 predictions, net−106. The descriptive fixed-suite batch timing interval [28.844121,30.270514]% does not measure hardware/runtime variability. Extension checks: time saving≥15% PASS; accuracy lower95%>−.3pp FAIL; nondegenerate differences PASS. Do not describe all15 as established accuracy-preserving acceleration.

Cheap versus original N5/35 terminal: +1.874662pp (object95%CI[1.564019,2.193409]), but14.587550% more time. Tuned PM versus N5/35: +2.160994pp,62.679800% more time. These comparisons combine depth, guidance rates and readouts; they do not isolate PM or a depth effect.

## Background N35, separately declared extension

| Method | Correct/2,468 | Accuracy% | Pipeline seconds | Amortized ms/inference |
|---|---:|---:|---:|---:|
| Original terminal N5 normal / N35 Background | 1,508 | 61.102107 | 1089.405 | 441.412 |
| Tuned N8 normal / N35 Background + four equal-PM readouts | 1,587 | 64.303079 | 1165.829 | 472.378 |
| Cheap correction N8 normal / N35 Background + four equal-PM readouts | 1,498 | 60.696921 | 724.303 | 293.478 |
| Tuned N35 terminal from saved PM trace | 1,605 | 65.032415 | Not measured separately | Not measured separately |

[Run/Inference] Cheap loses89 correct predictions versus tuned PM (107 corrected/196 harmed), −3.606159pp, object95%CI[−4.983793,−2.228525], while saving37.872281% time. It is0.405186pp below the original N35 terminal control while saving33.513866% time; that accuracy interval[−1.944895,1.135535]pp does not establish noninferiority. The −89 Background events combine with macro14's−17, producing−106 acrossall15. No root cause is isolated by this run.

[Run/CPU] Tuned N35's final readout alone is65.032415%, versus64.303079% for its PM; PM harms18 net Background predictions (−0.729335pp). On macro14, PM adds1.241607pp to the same tuned N8 traces. Acrossall15 the tuned N8/35 terminal readout is64.905457%; PM adds1.110211pp. These are exact saved-readout comparisons, not new terminal-only executions or latency measurements.

## Per-corruption accuracy

| Corruption | N5/35 terminal% | Tuned PM% | Cheap PM% | Cheap−tuned PM pp |
|---|---:|---:|---:|---:|
| uniform | 77.3501 | 80.2269 | 80.2674 | +0.0405 |
| gaussian | 74.4327 | 79.4165 | 79.4571 | +0.0405 |
| background | 61.1021 | 64.3031 | 60.6969 | -3.6062 |
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

## Protocol and provenance

[Code] The terminal control uses gamma=eta=.01, p=.95 and one terminal readout. Tuned PM uses gamma=.03, eta=.003, p=.95 and four equal softmax-probability readouts. Cheap retains the tuned horizon/readouts and fixedtau=.1, applies the frozen-correspondence bidirectional95%-trimmed Chamfer quadratic correction in predicted-clean latentXYZ, and transports the clean delta through DDIM. It avoids denoiser backward and direct style updates. Normal horizons are5 versus8; **Background is35 for all three** and uses clean[9,18,27]+terminal. Feature/style still participate in the model. This policy depends on corruption identity for the Background horizon. Unguided attribution and novelty are not established.

[Run] Background `20261108T152511Z_background` ran15:25:11→16:22:06UTC on `tta-lion-guidance-speed-a100-20261108-r5`, endpoint `gpu-a100-hm-euw4a0-1svstdm1jnaaj`, A100-SXM4-80GB/High RAM. Actual elapsed3413.114s; completeB36/B128 warmups,20 closed cases/7404rows, allthree frozen model hashes and exact final marker/exit0 passed. Scientific source5afdf0d/overlay17b49a8632f83b183dc654c334f6637c88e9fca88f0440ef4f59fedcd5301242; manifest01305b50e8e5de5321960772b0aaf49b5d142b726e80451df8a91c4dac91dcb6. Ten nested scientific functions remain AST-identical to macro14 source; only new phase/metadata differs. Macro14 scientific source remainsaf90b34/overlay72bc0a1, not the later Background commit.

Raw `result/modelnet40_c/lion_guidance_speed/raw/20261108T152511Z_background.zip`,5,425,487bytes,SHA256bf75c39c9477b3f5e292b33f45166acdbbbac2f77932dad59ea8f60faa5a7953, independently matches Drive `MyDrive/thesis/result/lion_guidance_background_20261108_r5/raw/20261108T152511Z_background.zip`. Actual pre-sciencev2 source/native/operator supplement SHA81c8aa41cced9f764c8bddb6cb00200a73d116c0c5b4cd6a55980ff47d26d54c is separately archived. Raw acceptor checks all member hashes, source/setup/bootstrap/model/gate/config, canonical RNG/order, original labels, every PM outcome, final36, complete timings and before/after model integrity.

Derived all15 summary/CSV: `result/modelnet40_c/lion_guidance_speed/analysis/20261108T1623Z_all15_extension/`. Accuracy-only terminal derivation: `analysis/20261108T1623Z_policy_terminal_all15/`. Compositor seals original macro14 CSV/summary before and after, binds raw environment/source/model/manifest to Background acceptance, verifies disjoint103656+7404 rows and exactly2468 IDs per arm/corruption. Host derivation scripts/receipts are archived with the derived backup; no GPU rerun.

[Open] One seed; development-exposed data, not a globally unseen holdout; native numerical variation; three-runtime/corruption confounding; B128 amortized timing is not B1 latency. Confidence intervals are conditional on the observed seed/runtime suite. No fresh N8/35 terminal-only timing, repeated-seed claim, isolated cheap-versus-unguided contribution or algorithmic novelty. The next research question is why the frozen cheap/PM settings lose accuracy on Background; this report authorizes no retuning or extra GPU phase.

[Run/Open] Independent actual raw audit has passed Background's7,404 predictions/labels/60timings. Final all15/terminal-derivation reviewer and derived Drive-backup receipt are pending at report creation; append their actual acceptance below.


## 2026-11-08T16:26:31.628374+00:00 — final independent audit accepted

[Run/CPU] Independent raw/composite audit verifies all111,060 rows,900 deduplicated timings, sealed macro14 inputs, three physical runtimes/four measurement groups, and all37,020 terminal readouts. Accuracy/timing intervals and corrected581/harmed687 reproduced exactly. All15 accuracy-margin criterion FAILS while macro14 primary remains PASS. Background losses and its negative PM effect remain explicit; no fallback or retuning. [Review receipt](evidence/lion_guidance_speed_20261107/all15_independent_review_20261108.json). Derived Drive-backup receipt follows separately.


[Run/CPU] 2026-11-08T16:27:46.152326+00:00: Final all15 derived bundle Drive-roundtrip verified:759,531bytes,SHAe6b5bd63c9947e310927a2f1e10d229abd4f043d9475398986a02b9e15ff3bed,34members/all member identities andCRC passed. Includes accepted all15/N8terminal CSV/summary, reports, exact host acceptance/composition/derivation sources, independent review and source/raw/sealed-macro14 receipts. Drive `MyDrive/thesis/result/lion_guidance_background_20261108_r5/all15_derived_results_20261108T162703Z.zip`. Receipt is outside its own sealed bundle; later publication metadata is separately recorded. Raw/source/native archives remain separate and unchanged.


[User decision/Run/CPU] 2026-11-08T17:15:39.099364+00:00: Userrequested a [separate common-horizon Background revision](lion_guidance_uniform_horizon_results_20261108.md) after this historicalN35 result. Newall15 appliesN5terminal/N8candidates uniformly: cheap−PM−.013506pp,27.673026%less time,95%CI[−.180983,.153971]; same-marginchecks pass but absoluteBackgroundaccuracy falls sharply. This report/raw/summary remains unchanged historical evidence; do not replace its values with the newpolicy totals.
