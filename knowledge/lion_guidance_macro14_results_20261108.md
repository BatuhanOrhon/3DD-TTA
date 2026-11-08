# Full ModelNet40-C macro14 speed study — 2026-11-08

[Run/CPU/Inference] **The frozen cheap correction passes the predeclared macro14 speed/accuracy gate:** 27.6721% less measured pipeline time than tuned N8+PM with an observed accuracy difference of −0.0492 percentage points. This is one-seed evidence, not a repeated-seed or isolated proximal-mechanism claim.

All2,468 original IDs, severity5,14 corruptions excludingBackground, seed0, one measured pass, commonB128/final36. Each arm has34,552 classifications; the composite has103,656 rows. Equal corruption sizes make macro and micro accuracy identical here. Three A10080 runtimes supplied disjoint corruption groups; combined time is an operational composite, with per-runtime results retained.

| Method | Correct/34,552 | Accuracy% | Pipeline seconds | Batch-amortized ms/inference |
|---|---:|---:|---:|---:|
| Original terminal N5 SCD | 22,131 | 64.051285 | 2777.401 | 80.383 |
| Tuned N8 terminal readout, derived from saved N8+PM traces | 22,423 | 64.896388 | Not measured separately | Not measured separately |
| Tuned N8 + four equal-PM readouts | 22,852 | 66.137995 | 5124.682 | 148.318 |
| Cheap latent correction + four equal-PM readouts | 22,835 | 66.088794 | 3706.574 | 107.275 |

[Run/Inference] Cheap versus N8PM: **−0.049201pp**, paired original-object-cluster95%CI **[−0.219958,+0.124450]pp**; **27.672109% less time**,1.382592× throughput and1,418.108 seconds saved across34,552 classifications. It corrects474 and harms491 paired predictions, net−17. Frozen ≥15% time saving, accuracy CI lower>−.3pp and nondegenerate differences all pass. The descriptive fixed-batch timing CI[27.6542,27.6920]% does not capture runtime/hardware uncertainty.

N8PM versus N5 terminal: **+2.086710pp**, CI[1.771243,2.402176], **84.5136% more time**. Cheap versus N5 terminal: **+2.037509pp**, CI[1.722042,2.367446], **33.4548% more time**. Thus cheap improves the tuned N8PM accuracy–cost balance; it is not faster than the actual N5 terminal comparator.

[Run/CPU] The added N8 terminal row is an accuracy-only derivation from the final saved readout, independently checked against all 34,552 predictions. PM improves its own tuned N8 terminal accuracy by **1.241607pp**. The N8 terminal advantage over N5 is **0.845103pp**, combining depth and guidance-rate changes. The N8+PM acquisition time must not be assigned to a terminal-only N8 implementation.

## Per-corruption accuracy

| Corruption | Terminal% | N8PM% | Cheap% | Cheap−N8PM pp |
|---|---:|---:|---:|---:|
| uniform | 77.3501 | 80.2269 | 80.2674 | +0.0405 |
| gaussian | 74.4327 | 79.4165 | 79.4571 | +0.0405 |
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

[Inference] Gains are heterogeneous. Density+.6888pp andShear+.6078pp offset losses includingLiDAR−1.2561pp,RBFInverse−.5267pp andOcclusion−.4862pp. No parameter was selected from these final outcomes. Within-runtime time savings are27.6249%(r3),27.7562%(r4),27.6546%(r5).

## Method and attribution

[Code] Normal terminal usesN5/gamma.01/eta.01/p.95/one readout. Tuned original usesN8/gamma.03/eta.003/p.95/four readouts. Cheap retainsN8/four readouts, applies fixedtau.1 closed-form bidirectional95%-trimmed nearest-neighbor correction to predicted-clean latentXYZ, then transports it through DDIM. No denoiser backward or guidance style update occurs in cheap. Feature/style remain active in the model; only the direct guidance correction targetsXYZ. The N8-versus-N5 comparison changes horizon/start noise and rates as well as readout; the gain is not attributable to PM alone.

[Paper/Inference] Closest reviewed inspiration is [Guess & Guide,v1](https://arxiv.org/html/2603.07860v1), currently titled [Sparse Scheduled Diffusion Guidance for Inverse Problems,Spin,v3](https://arxiv.org/abs/2603.07860). [CloudFixer](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf) already avoids diffusion backward in3DTTA. The specific frozen-correspondence Chamfer quadratic/latent-DDIM implementation is a project adaptation; the generic gradient-free/proximal principle is established. Cheap-versus-unguided attribution remains unperformed, so these results do not isolate the benefit of the proximal correction or establish algorithmic novelty.

## Evidence and limits

[Run] Independently accepted r3Uniform/Gaussian14,808rows + r4Impulse/Upsampling/DistortionRBF22,212rows + newr5 remainingnine66,636rows. All raw SHAs, PM logits/probabilities/predictions, original label identities, canonical per-run RNG/order, exact grids/ragged36, source/native/setup/model proofs and deduplicated timings validated. Independent reviewer recounted all103,656 outcomes/labels/provenance,840 batch timings and42corruption/arm tables and reproduced the primary CI.

Actual scientific source is af90b34/frozenoverlay72bc0a1e4291480c2a1df4fd0d170a879f03f6fa5cadba264a9cdbadf593ed4d onr4/r5; r3scientificfunctions are AST-identical across the documented continuation bridge. CurrentnewBackgroundsource5afdf0d was not used for macro14. Currentr5manifest40b30562bb3bf823479bfbbcda51d33ced976c96478d09f522b542bcd40522ac; pinned native7 restored only after preserving freshbytes; allthree model tensorhashes match across runtimes. Miniconda bootstrap substitution is archived separately from scientific identity.

Newraw `result/modelnet40_c/lion_guidance_speed/raw/20261108T123641Z_continue14.zip`,64,514,252bytes,SHAe20117e4c48aafc47ad850b93fa74dc49782e038d655ba0e746e1a836c4c763b; Drive `MyDrive/thesis/result/lion_guidance_full14_20261108_r5/raw/20261108T123641Z_continue14.zip`. Pre-sciencesupplement6,589,156bytes,SHA8b30160c7e8fefeca0c8e9d4b288430092e2c151dd7766db2b31f0d683926bf4 is locally/Driveaccepted. Earlier r3/r4 fullrun directories remain partial; only explicitly closed accepted groups enter the composite.

Derived summary/CSV: `result/modelnet40_c/lion_guidance_speed/analysis/20261108T1501Z_full14_three_runtimes/`. Compact evidence: `knowledge/evidence/lion_guidance_speed_20261107/macro14_final_20261108.json` and `macro14_independent_review_20261108.json`. Execution/failures: [study record](lion_guidance_speed_20261107.md).

[Open] One seed; development-exposed benchmark, not an unseen holdout. Native numerical variation and runtime/corruption confounding remain. B128 amortized times are not B1 latency; prior smallB1calibration remains separately scoped. Comparator is repositoryp.95/controlledbackend, not exact publishedp.96/fullpaper reproduction. No all15 conclusion yet.

[User decision/Open] User requested a separate fullBackground extension before macro14 analysis was available. Preserve this frozen macro14 conclusion. Background uses35steps for allthree arms, one versusfour readouts, p.95/fixedtau.1, same2468IDs/seed0 and commonB128 ifN35warmupsmemorysafe. It is a separately declared experiment with its ownsource/receipt/RNGschedule and will yield a separately labelled all15 composite; no retuning or rerun of completed14.


## 2026-11-08T15:39:23.492320+00:00 — supplementary same-trace N8 terminal accuracy

[Run/CPU] Terminal-onlyreadoutfromacceptedN8PMtraces is22,423/34,552 = **64.896388%**, versusN5terminal64.051285%,N8PM66.137995%,cheapPM66.088794%. PM adds **1.241607pp** toitsownN8terminal; N8terminalexceedsN5terminalby.845103pp; cheapPMexceedsN8terminalby1.192406pp. N8ratesaregamma.03/eta.003, N5rates.01/.01, soN5→N8isnotdepth-only. Extractionverifiesall3rawSHAs/IDs/labelsandterminal-lastreadoutorder. No newGPU; no freshterminal-onlyN8latencymeasurement, andN8PMacquisitiontimemustnotberelabelled. [Derivedreadoutsummary](../result/modelnet40_c/lion_guidance_speed/analysis/20261108T153708Z_n8_terminal_macro14/summary.json). Primarymacro14frozenCSV/summaryunchanged.
