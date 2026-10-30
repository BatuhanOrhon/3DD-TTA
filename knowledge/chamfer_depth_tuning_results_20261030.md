# Chamfer depth tuning: accepted results — 2026-10-30

[Run/Code/Inference] All authorized experiments and independent local raw reanalysis are complete. Owned r2 was closed after the verified final derived Drive roundtrip at18:06:04UTC; a fresh inventory confirmed the protected peer remained present unchanged. This result concerns a bounded operating point in original Chamfer guidance with Probability Mean; no spectral guidance is used.

## Protocol and evidence

ModelNet40-C/severity5/all15, frozen Point-MAE, raw/eval LION EMAoff, batch32. Development128 reserved objects/seed0; confirmation256 disjoint reserved objects, fixed seeds0/1/2. Prior70+70+140 cohorts are excluded, but the benchmark has historical exposure. Background is35steps throughout. N changes initial noise depth and reverse-update count together. Probability Mean equally averages the three unique quarter/half/three-quarter predicted-clean checkpoints plus terminal at temperature1; probes use original encoded decoder style and preserve live input/RNG/model buffers. Gamma controls local latent; eta controls conditioning style. p(lambda) is retained Chamfer fraction.

Registered32-setting sequential search:10default depths1–10,16new eta/gamma settings at top2depths8/9, then6new fractions. All32 attempted:27finite and5numerically infeasible. User explicitly approved the postfailure infeasibility amendment; failures retain null accuracy and are never repeated or replaced. Frozen confirmation contains exactly the following three conditions, each at all three seeds;9/9passed. No default production setting was silently changed.

## Confirmation accuracy

Primary all15 Probability Mean, percent; sample SD over three fixed seeds is descriptive.

| Setting | Seed0 | Seed1 | Seed2 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| N5: eta=.01, gamma=.01, p=.95 | 66.718750 | 66.640625 | 66.171875 | 66.510417 ± 0.295776 |
| N8: eta=.01, gamma=.01, p=.95 | 67.265625 | 68.203125 | 67.630208 | 67.699653 ± 0.472592 |
| N8: eta=.003, gamma=.03, p=.95 | 68.385417 | 69.244792 | 68.593750 | 68.741319 ± 0.448290 |

Paired conditional95% intervals resample256 original objects jointly over15 corruptions and all three fixed seeds (10,000 bootstrap repetitions, seed20261030). There are11,520 prediction events, not11,520 independent observations. These are marginal conditional intervals, not a simultaneous/global superiority guarantee.

| Primary comparison | Difference (pp) | Conditional95% CI (pp) | Corrected / harmed events |
|---|---:|---:|---:|
| N8 default − N5 default | +1.189236 | [0.512153, 1.866319] | 424 / 287 |
| N8 tuned − N5 default | +2.230903 | [1.623047, 2.864583] | 496 / 239 |
| N8 tuned − N8 default | +1.041667 | [0.685764, 1.406250] | 276 / 156 |

Excluding Background (macro14), tunedN8−N5 is+1.832217pp [1.218378,2.474191]; tunedN8−defaultN8 is+.604539pp [.269717,.948661]. The improvements therefore persist outside Background. Background contributes60of 257 net corrected events versusN5, and55of 120 versusdefaultN8; its own mean gain is7.8125pp and7.161458pp respectively. These effects do not establish an isolated native-kernel cause.

N5 terminal is65.789931%; its own Probability Mean adds+.720486pp [.329861,1.111111]. TunedN8 PM versusN5 terminal is+2.951389pp [2.222005,3.697917], a combined depth/rate/readout comparison. It must not be called a pure depth gain. TunedN8 terminal/minimum-entropy means are67.647569/68.793403%; primary readout remains the frozen Probability Mean, not a posthoc entropy switch.

## Development, numerical failures and costs

Default development PM atN1–10:55.885417,58.645833,61.614583,63.020833,64.166667,64.375000,64.218750,65.000000,64.531250,64.322917%. N10−N5 is only+.15625pp under this common readout/seed; N10 was not included in heldout confirmation. The historical question of three-seed heldoutN5versusN10 remains unanswered.

Development best: N8/eta.003/gamma.03/p.95=65.468750%; runner-up N9/eta.003/gamma.03/p.96=65.416667%, a one-event margin out of1,920. This is not evidence that8 is a unique/global optimum. Retain N8/eta.003/gamma.03/p.95 as the supported candidate for a subsequent separately designed larger benchmark.

Five infeasible settings: N8 eta.03 with gamma.003/.01/.03; N9 eta.03 with gamma.003/.01. All fail at processed predicted-clean classifier XYZ rejection in Background35 batch0. Decoder/latent/normalization root cause and exact failing probe loop are not isolated; no failed tensor/purity snapshot was saved. N9 eta.03/gamma.03 passes, so no deterministic global eta threshold is proved. Source, model-state, finite paired cache and completed-prefix guards pass; prefixes do not receive an all15 score.

Background identical-default negative controls varied52.34375–57.8125% on128dev objects despite matched initial caches/RNG/schedule. Original Chamfer CUDA backward atomicAdds are a plausible contributor, not a demonstrated cause. Full-process histories differ; the cluster intervals do not capture all native/retraining/environment variability.

Matched development cache-replay costs (all15,128objects,seed0, probes included):

| Setting | Reconstruction seconds | Worker seconds incl. artifact IO |
|---|---:|---:|
| n5-e0.01-g0.01-p0.95 | 124.194418 | 166.316381 |
| n8-e0.01-g0.01-p0.95 | 152.260551 | 194.414387 |
| n8-e0.003-g0.03-p0.95 | 151.934632 | 194.114559 |
| n10-e0.01-g0.01-p0.95 | 170.656617 | 212.775409 |

TunedN8 reconstruction takes22.336%more time thanN5 in this matched cache-replay measurement, and is10.971%lower thanN10. Worker time is16.714%higher thanN5. These are observed replay costs, not fresh end-to-end speedups or an unmeasured terminal-only baseline. ConfirmationN5 creates caches whileN8 replays; its worker timing is confounded by preparation mode. Peak reported GPU memory is about25,218MB. Probe dispatch overlaps reconstruction and is not added twice.

Final hash-chained root ledger:10,073.248667657/10,800s (726.751332343sremaining). This charges worker/failure/analysis/copy wall, conservative final-copy reserves and420s recovery allowance; it is not a strict VM-lifetime or GPU-billing account. Controller finalanalysis embeds9,853.311075344s because generated before its own phase finish; use archivedroot ledger/localreanalysis for the final figure.

## Artifact validation and provenance

Full Drive/local archive:1,345,860,484bytes, SHA256`c2a4e46885f4b8ba8e4e5930418170c9126c2c1b576df4d3751b8593dae1a82c`. Six Drive parts and VM/Drive receipt bytes match; safe extraction, CRC and all 7,267 declared member hashes pass. Coverage44unique attempts=3smoke+32development+9confirmation;39 accepted/five infeasible,422 physical cache NPZs, no unresolved/unattempted slot.

Unchanged local NumPy1 analysis reproduces all 39 raw acceptances, all five infeasibility receipts, all 78 paired comparisons and the exact frozen stage plus its score-input hash. Physical latent/noise/RNG bytes, IDs/source/checkpoint/state/decoder/probe guards are checked. Independent reviewer separately reproduces the frozen stage, ledger coverage and tuned-versus-default object bootstrap. Local ad-hoc stage assertion initially omitted the registration wrapper score_input_sha256 field; adding that exact existing metadata digest resolved the assertion, without code/data/config changes or GPU rerun.

CPU code checks:363 tests plus274 subtests pass (six retained warnings) and69 campaign/continuation tests pass. PlanSHA`3d6e42dbfc49966fc8d9b5ed79a133355a06dac63bde8e061a6fe624dfb1e8a6`; executedscientificsourceZIP SHA`166df3799834bc72541f1189407bf56d992a23bfdce9827db1c422d9ba5b3636`; Point-MAEcheckpoint SHA`507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75`. Exact binding/approval/amendment/sourcebytes are preserved in raw archives and compact receipts.

Local:[compact summary](../result/preparation/chamfer_depth_tuning_20261030/recovery_r2/final/accepted_summary.json),[comparisons](../result/preparation/chamfer_depth_tuning_20261030/recovery_r2/final/paired_comparisons.json),[independent acceptance](../result/preparation/chamfer_depth_tuning_20261030/recovery_r2/final/independent_reanalysis_receipt.json),[archive acceptance](../result/preparation/chamfer_depth_tuning_20261030/recovery_r2/final/snapshot_local_acceptance.json). Fullrawtree stays at`result/preparation/chamfer_depth_tuning_20261030/recovery_r2/final/exported/campaign`; large raw ZIPs/caches are not for Git integration.

Drive:`MyDrive/thesis/result/chamfer_depth_tuning_20261030/phase-evidence/final_snapshot_20261030T174831414018Z/campaign_evidence.zip`. Progress record at`operator_progress_20261030T1654Z.json` was independently downloaded and hash-verified before user departure. Final derived local analysis/knowledge roundtrip and owned-only closure receipt are recorded after this report.

## Decision and remaining limits

[Inference/Decision] The frozen tunedN8 candidate improves paired reserved-ID accuracy versus historicalN5 and versus defaultN8, with positive directions across all three seeds and conditional intervals excludingzero. This supports this operating point on this checkpoint/protocol. It does not establish a global optimum, a full2,468-object benchmark gain, externally unseen generalization, superiority of Chamfer overGSD, or exact paper reproduction (paperp=.96 versus historicalcontrolp=.95). No settings above10 or directN5/N10heldout contrast were tested.

[Open/Falsifier] A separately designed larger/repeated evaluation that loses the paired gain would limit transfer of this selected point. A different backbone/dataset or unseen benchmark, seed/native variability, and the finite grid remain separate questions. Existing frozen payload identity must remain distinct from any later combined integration checkout.

## Final preservation and closure

[Run/Decision] Derived local-analysis/knowledge archive4,244,184bytes SHA256`da3653f7b0d386d1e8b84002013d6c14eb8ea87e4ef7cbb7d72c953f9bee7731` was copied to`MyDrive/thesis/result/chamfer_depth_tuning_20261030/phase-evidence/final_local_analysis_20261030T1806Z.zip` and downloaded back with matchingbytes/SHA. It contains36declared files including full independent analysis, compact evidence, research notes and reanalysis sources. It is the immutable preclosure copy; the later closure receipt is preserved locally and in selected Git integration.

Ownedmonitor31219 exited18:05:55UTC; officialCLI stop targetedonly`tta-chamfer-depth-tuning-a100-20261030-r2`. Fresh server inventory18:06:04UTC verifies ownedendpoint`gpu-a100-hm-ass1c0-79r779o7n5fk` absent and peer`tta-chamfer-guidance-variants-a100-20261030-r1` / `gpu-a100-hm-ass1c0-3nh8smksm50de` stillpresentunchanged. [Closure receipt](../result/preparation/chamfer_depth_tuning_20261030/recovery_r2/final/runtime_closure.json). No GPU work remains; user authorization was explicit and conditional on completed work/validation.
