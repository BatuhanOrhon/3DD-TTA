[Run] **2026-11-20 update:** the locked policies were evaluated on full Point2Vec,
2,468 objects × all15/s5, seed2/B1. PlainCF77.244733%, latePM74.954079%,
N32 true terminal73.233387%, source61.985413%; both fixed proximal policies lose
overall to CF. [Full results and evidence](cloudfixer_point2vec_full_results_20261120.md).
The historical Point-MAE128 development results below remain unchanged and are
not directly comparable across classifier/cohort/seed. Independent readout
mechanism results are in the [held-out B confirmation](cloudfixer_pm_hypotheses_confirmation_20261118.md).

# CloudFixer-only coarse grid — accepted exploratory results, 2026-11-15

[Run/Validation]48settings,3120units,128fixed object-disjoint development IDs,
all15ModelNet40-C corruptions/severity5/seed0/B32. Each configuration has1920
object-corruption predictions; each corruption has128. Macro15 accuracy equals
micro accuracy here because counts match. Same copied author XYZ diffusion,
frozen Point-MAE head and input/RNG identity; no LION. Raw/profile and canonical
preparation source remain documented in cloudfixer_transfer_execution_20261114.md.

## Main outcomes

| Arm | N | K | tau | Correct/1920 | Macro15 accuracy |
|---|---:|---:|---:|---:|---:|
| Source | — | — | — |943|49.114583%|
| Original CF operational0 | — | — | — |1220|63.541667%|
| Original CF operational1 | — | — | — |1217|63.385417%|
| Original CF operational2 | — | — | — |1226|63.854167%|
| Best PM |32|80|.03|1266|65.937500%|
| Terminal at best PM setting |32|80|.03|1247|64.947917%|
| Best terminal |32|160|.03|1295|67.447917%|
| PM at best terminal setting |32|160|.03|1228|63.958333%|

Original CF mean63.593750%, sample SD0.238676percentage points across three
same-seed operational repeats (native CUDA variation; not three independent
seeds; no logit ensemble). Best PM is+2.343750pp and best terminal+3.854167pp
versus this mean on the tuned development subset; neither is confirmatory.
PM exceeds its own terminal in26/48settings, loses22/48, ties0. At best PM,
71terminal errors are corrected and52terminal successes become errors, net+19
(+0.989583pp). At best terminal, PM net−67/1920 (−3.489583pp).

[Code/Method] Original CF is the copied `adapt.cloudfixer(...)` algorithm,
optimizing per-point delta and6D rotation with Adamax for30updates, random-time
one-call clean estimates and Chamfer+delta regularization, authorPoint-MAE
configuration input_lr.2/rotation.02/lam_h10/lam_l1/vote1. It is not a diffusion-only
or our tau0 arm. Model/head weights remain frozen. Our tau0 control uses the
same added deterministic XYZ DDIM-type reverse chain with correction disabled;
nonzero tau applies gradient-free proximal SCD, and PM averages four saved
softmax vectors at quarter/half/three-quarter/terminal readouts.

[Run/Controls] At N32K80, tau0 gives PM65.520833%/terminal65.468750%; tau.03
changes these by+0.416667pp/−0.520833pp. At N32K160, tau0 gives
PM59.739583%/terminal63.437500%; tau.03 changes these by+4.218750pp/+4.010417pp.
Thus correction does not uniformly improve every readout. Negative high-strength
outcome preserved: N32K160tau.3 terminal43.333333%, below source49.114583%.
The transferred N8K80tau.1 gives PM65.104167%/terminal66.093750%.

[Run/Timing] Inference pipeline sums across1920predictions: originalCF
533.423844/533.016232/532.995325seconds across operational repeats; best PM
598.874860s, best terminal setting596.702133s. Our times include all four
classifier readouts and PM computation; terminal is read from the same trace,
not a terminal-only speed optimization. Best PM is~12.33%slower than meanCF;
no speed-superiority claim. Max allocated GPU memory9308.116699MiB for these
best sampler settings. Grid completed in~5h37m including archival versus the
8.125h conservative forecast; CPU retrieval did not rerun inference.

## Complete grid

N=reverse update count (N+1denoiser calls including t0 clean terminal),
K=initial integer level out of500, tau=per-step proximal strength; retained.95.

| N | K | τ | PM (%) | Terminal (%) |
|---:|---:|---:|---:|---:|
| 4 | 40 | 0 | 63.65 | 63.23 |
| 4 | 40 | 0.03 | 63.91 | 62.66 |
| 4 | 40 | 0.1 | 64.11 | 62.71 |
| 4 | 40 | 0.3 | 63.28 | 59.84 |
| 4 | 80 | 0 | 62.86 | 65.00 |
| 4 | 80 | 0.03 | 63.33 | 65.42 |
| 4 | 80 | 0.1 | 63.44 | 64.79 |
| 4 | 80 | 0.3 | 63.54 | 64.79 |
| 4 | 160 | 0 | 52.14 | 60.31 |
| 4 | 160 | 0.03 | 52.92 | 61.56 |
| 4 | 160 | 0.1 | 53.18 | 63.44 |
| 4 | 160 | 0.3 | 54.84 | 63.54 |
| 8 | 40 | 0 | 63.80 | 62.86 |
| 8 | 40 | 0.03 | 64.43 | 63.39 |
| 8 | 40 | 0.1 | 63.54 | 60.42 |
| 8 | 40 | 0.3 | 61.41 | 55.10 |
| 8 | 80 | 0 | 64.58 | 65.52 |
| 8 | 80 | 0.03 | 64.74 | 65.99 |
| 8 | 80 | 0.1 | 65.10 | 66.09 |
| 8 | 80 | 0.3 | 63.96 | 60.73 |
| 8 | 160 | 0 | 57.08 | 62.76 |
| 8 | 160 | 0.03 | 58.80 | 64.79 |
| 8 | 160 | 0.1 | 60.47 | 66.82 |
| 8 | 160 | 0.3 | 60.36 | 63.85 |
| 16 | 40 | 0 | 64.01 | 61.82 |
| 16 | 40 | 0.03 | 64.32 | 61.30 |
| 16 | 40 | 0.1 | 62.08 | 56.41 |
| 16 | 40 | 0.3 | 58.75 | 52.71 |
| 16 | 80 | 0 | 65.31 | 65.52 |
| 16 | 80 | 0.03 | 65.68 | 65.73 |
| 16 | 80 | 0.1 | 65.62 | 62.50 |
| 16 | 80 | 0.3 | 61.15 | 54.58 |
| 16 | 160 | 0 | 59.48 | 63.75 |
| 16 | 160 | 0.03 | 61.56 | 65.99 |
| 16 | 160 | 0.1 | 63.02 | 66.15 |
| 16 | 160 | 0.3 | 57.81 | 53.85 |
| 32 | 40 | 0 | 64.11 | 62.08 |
| 32 | 40 | 0.03 | 62.81 | 58.65 |
| 32 | 40 | 0.1 | 59.43 | 52.50 |
| 32 | 40 | 0.3 | 55.10 | 52.29 |
| 32 | 80 | 0 | 65.52 | 65.47 |
| 32 | 80 | 0.03 | **65.94** | 64.95 |
| 32 | 80 | 0.1 | 62.92 | 54.38 |
| 32 | 80 | 0.3 | 56.93 | 50.21 |
| 32 | 160 | 0 | 59.74 | 63.44 |
| 32 | 160 | 0.03 | 63.96 | **67.45** |
| 32 | 160 | 0.1 | 60.42 | 57.60 |
| 32 | 160 | 0.3 | 50.52 | 43.33 |

## Artifact and recovery acceptance

Raw: `result/modelnet40_c/cloudfixer_transfer/20261114-171811_development.zip`,
62985130bytes, SHA28baa27f2b80ae32c42fb5b56d569f8bb3dfac4e94fb6c055a2e89d1dabf50e7.
Independent second CPU VM mounted-Drive retrieval and Mac SHA/CRC/path checks
pass. All3120expected unique unit IDs, alloriginal label SHA/IDs, source/model/
manifest, frozen model hashes, common input/RNG hashes, masks and sampler time
indices pass; raw104summary rows/1560corruption rows and PMselection match
independent prediction recount. CSV `partial` coverage labels are retained raw;
complete3120registered-subset coverage is independently established, not full
benchmark coverage. Derived full CSV/JSON under the sibling `_derived` folder.

Full verification bundle327459bytes, SHA4c6c0f8caff68abf712bc2619c0e211367d2dda95a91f156c75a48b1364ca455,
Drive `thesis/result/cloudfixer_transfer_20261114_mount_b09f8b7730d1/verified_analysis_20261115.zip`;
fsync/SHA backup readback passed. Includes recount source, labels, payloadSHA,
raw report binders and nativeCF operational statistics. Compact result and grid
CSV retained in knowledge/evidence/cloudfixer_transfer_20261114/.

## Limits and next decision

[Open] This is label-tuned exploratory development on128unique objects and
one seed, not1920independent objects or a full benchmark. N32 is the tested
upper boundary for both winners; K160 is also an edge for the terminal winner.
No global optimum or general superiority established. No confirmation256IDs,
extra seeds, higherN, newPMweights/times, classifier tuning or refinement ran.
User defers PM redesign to joint planning after this grid. Runtime r4 was lost
after successful archival; user authorized a separate CPU recovery VM and close
after transfer. CPU tta-cloudfixer-retrieval-cpu-20261115-r1 (endpoint
m-s-usw4c1-32zk78x4avg9r) retrieved only, then was explicitly stopped successfully;
subsequent server inventory shows no active sessions. No peer runtime action.


## 2026-11-15 paper-protocol clarification and detailed corruption exports

[Paper/Code/Run] See [source/protocol audit](cloudfixer_paper_protocol_audit_20261115.md).
The reference is the unchangedauthorCFalgorithm evaluated in our128-object
Point-MAE pilot, not an exactpaper-table reproduction. RelevantpublishedplainCF
Point-MAE value67.97% (suppI3); mainTable1Point2Vec78.23% is anotherhead.
Oursource49.1146% versuspaperPoint-MAE53.70%; nativeCF63.59375% versus67.97%.
Corehyperparameters match the releasedcode; paper/oursubset,seed,RNG,B and
harness differ. Full2468-object nativePoint-MAE parity has not been established.
Detailed720setting×corruption rows plus60source/reference rows and15paper
comparison rows are versioned inknowledge/evidence/cloudfixer_transfer_20261114/.
All counts are traceable to the accepted3120payloads; no new inference.

## 2026-11-18 — subsequent held-out confirmation and transfer status

[Run/Code/Inference] New Point-MAE mechanism confirmation uses reserved256 objects and seeds0/1/2, with positive locked late-PM vs own-terminal gaps at three schedules. Those72%-range accuracies belong to a different cohort than the historical128 grid and cannot be treated as a grid improvement. The separate full2468 Point2Vec comparison is running with locked source/N8latePM/N32terminal/nativeCF policies; no full Point2Vec accuracy is accepted yet. [Full evidence and limitations](cloudfixer_pm_hypotheses_confirmation_20261118.md). [Current execution](cloudfixer_two_colab_execution_20261118.md).
