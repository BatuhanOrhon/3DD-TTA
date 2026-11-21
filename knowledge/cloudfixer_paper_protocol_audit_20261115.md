# CloudFixer paper/code/protocol audit — 2026-11-15

[User request] Check why paper ModelNet40-C numbers exceed our nativeCF pilot,
identify Point-MAE versus Point2Vec, verify official-source reuse, and preserve
all detailed accuracy records. This is a read-only paper/source/artifact audit;
no new inference, checkpoint download, runtime or implementation change.
Worktree research/cloudfixer-transfer-20261114, accepted-grid commit6933e72;
origin fetched for current research policy. Original user checkout untouched.

## Direct answer and the correct paper reference

[Code/Run] We used **Point-MAE**, not Point2Vec. Executed worker imports
`classifier.pointMAE.build_model_from_cfg` and loads
`cfgs/cfgs_mate/tta/tta_modelnet.yaml` (Point_MAE,32groups,32neighbors,40classes,
all classification tokens,rotateTrue). Source/head weights frozen.
Project `pointnet_ckpts/modelnet_jt.pth` is copied on the VM to
`/content/CloudFixer/checkpoints/pointMAE_modelnet40.pth` and loaded strictly
frombase_model after strippingmodule. ClassifierSHA
507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75.
Historical Nov10 peer asset audit records an independent author download as
byte-identical (351656113bytes); this audit rechecks executed manifest/model
identity, not a fresh remote checkpoint download. The diffusion checkpoint
is separate: authors' Transformer/Point-E-derived XYZ prior, SHA
b64841d022e08b1a89f58dd9b16d9023e00be52fbca3dca735d5798362bb5a5a.

[Paper] [Main paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf),
section5.1/Table1/page9 uses **Point2Vec**; the appropriate Point-MAE comparison
is [supplement TableI3/page8](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603-supp.pdf).

| Evaluation | Classifier | Objects/corruption | Source accuracy | Plain CloudFixer accuracy |
|---|---|---:|---:|---:|
| Paper main Table1 |Point2Vec|2468|62.09%|78.23%|
| Paper supplement TableI3 |Point-MAE|2468|53.70%|67.97%|
| Our completed pilot |Point-MAE|128|49.114583%|63.593750% ±0.238676pp|

PaperTable1 additionally reports independent-view votingK5=79.09% (not our
one-trajectory temporalPM). CloudFixer-O is another variant that updates the
classifier: Point2Vec81.49% in mainTable4; Point-MAE77.20%(B1) and80.36%(B64)
in supplementI3. Our reference is plain CloudFixer/vote1 with a frozen head,
so those O/voting rows are not directly matching reference values.

## Source reuse verified, not a rewritten CloudFixer

[Code/Artifact] Official clean clone is
`/Users/batuhan/Desktop/projects/CloudFixer`, pinned authorcommit
f043e24414fcf4165fdc892abff21215c9989141. The source pack usesgitarchive;
recomputed all513author files against the actual run manifest:512exact matches,
sole changedfileutils/misc.py uncomments the PointNet++ import. In particular
adapt.py, datasets/dataloader.py, classifier/pointMAE.py, diffusion/diffusion.py,
diffusion/transformer.py, parser and model config are unmodified. The worker
hash matches the executed manifest and its actualsys.argv literals match this
audit, not inferred defaults. Critical SHA witnesses retained below.

New code belongs to our overlay: diffusion sampler/proximal kernel, immutable
mount store, resumable worker/registration and packaging/reporting. It does not
rewrite author models or the native adaptation algorithm. Reference inference
calls the copied `adapt.cloudfixer(options, DataParallel(model), x, mask, ind)`
and then the copied Point-MAE forward. It does **not** execute the author's
completeadapt.main/forward_and_adapt evaluation program unchanged.

## Algorithm and hyperparameter parity

[Code] Reused nativeCF function initializes per-pointdelta and6Drotation,
optimizes both withAdamax30updates, obtains detached random-time clean estimates
from the author diffusion model, and minimizes bidirectional Chamfer plus
weighted displacement regularization. Matches released scripts/run_cloudfixer.sh:

| Item | Our nativeCF | Author recipe |
|---|---|---|
| delta learning rate |.2|.2|
| rotation learning rate |.02|.02|
| updates |30|30|
| t range |.02T through .12T|same|
| neighbor count / exponent |5 /1|same|
| weighted regularizer |True|True|
| lambda endpoints |10→1|same|
| warmup / final LR factor |.2 /.05|same|
| Adamax beta1 / beta2 / decay |.9 /.999 /0|same|
| diffusion T / schedule |500 /polynomial_2|same|
| subsample threshold |2048|same|
| vote |1|plainCF1|

[Code] AuthorPoint-MAE preprocessing is retained: authorModelNet40C duplicate
mask/index handling,unit_std andaxis conversion; classifier rotateTrue, unit-ball
normalization,32groups andalltokens. No extraexternalFPS1024 is inserted.
The config's npoints1024 is not an extra preprocessing operation in nativeadapt.py.
Frozen state verified for every batch. OneGPU classifier unwrapped versus author
DataParallel does not itself establish an accuracy discrepancy; no parity claim
beyond the source/config/state evidence has been dynamically tested here.

## Why this is not an exact paper-protocol reproduction

[Paper/Code/Run] Important differences:

1. Papermain usesPoint2Vec; our common head isPoint-MAE. Matching head reference
   is67.97%, not78.23%,79.09% or the CloudFixer-O rows.
2. Paper uses2468objects/corruption. Our grid used128fixed random development
   objects, representing34of40classes; sample difficulty/class weights differ.
3. Paperreproducibility appendixD fixesglobal seed2. We registerseed0 and hash
   corruption/seed/originalbatchIDs to independently reset per unit, with
   controlled warmups and resume; nativeCF3repeats measure operational variation.
   This does not recreate the paper's RNG stream.
4. PaperCF runsB1. Our actual batches are32; the published example script itself
   setsB128. The parser defaultsoptions.batch_size4 in our minimal argv, but
   this field is not used bynativeCF: actualxshapeB32 governs computation.
   Batch/RNG/numerical equivalence is unmeasured; do not claim it explains the gap.
5. The author's main evaluation shuffles normaltest data, evaluates source
   predictions before adaptation and computes its outputs viaforward_and_adapt.
   Our hash-seeded fixed-ID harness evaluates source as a separate control and
   scores the singleadapted output. Plainvote1 has the same output rule, but
   call ordering/backend/warmup/stream differ. We have not run a full entrypoint
   parity test on identical IDs/B1/seed2.
6. Environment/backend versions are our canonicalPython3.8/Torch2.1.2+cu121
   setup; repoREADME suggestsPython3.10.14. Native KNNbackward variation was
   measured and retained. This difference's effect is unisolated.

[Paper/Code discrepancy] SupplementG describes **cosine** lambda10→1; pinned
adapt.cloudfixer uses **linear** lambda10*(1-iter/30)+1*(iter/30), whose last
iteration is1.3. Our reference follows the unmodified releasedcode. This is an
upstream paper/code discrepancy, not a correction invented for this experiment.

## Interpreting the numeric gap conservatively

[Run] Our source49.114583% is4.585417pp below paperPoint-MAE53.70%. NativeCF
mean63.593750% is4.376250pp below paperPoint-MAE67.97%. Source→CF improvement
is14.479167pp locally versus14.27pp in the paper. Every per-corruption source
value in this subset is below the published full-data source value.

[Inference/Open] These observations are consistent with a harder subset and
protocol/RNG differences contributing to the absolute gap; they do not prove
its complete cause or validate full-data baseline parity. Our selectedPM/
terminal settings beat the nativeCF reference on this development subset only.
They are not a demonstrated improvement over the paper's complete benchmark.
The reference label means **author CF algorithm on our Point-MAE pilot**, not
**the published paper protocol reproduced end to end**.

## Per-corruption source/nativeCF comparison (percent)

| Corruption | Paper source | Our source | Paper CF | Our CF mean ± SD |
|---|---:|---:|---:|---:|
| occlusion | 34.72 | 28.12 | 34.24 | 27.08 ± 0.45 |
| lidar | 15.19 | 13.28 | 35.62 | 32.29 ± 1.19 |
| density_inc | 85.53 | 84.38 | 79.62 | 78.12 ± 0.00 |
| density | 78.65 | 77.34 | 71.15 | 65.89 ± 0.45 |
| cutout | 74.64 | 74.22 | 73.54 | 65.10 ± 1.19 |
| uniform | 61.35 | 60.94 | 86.71 | 81.77 ± 0.45 |
| gaussian | 51.30 | 42.19 | 87.16 | 85.68 ± 0.45 |
| impulse | 55.47 | 47.66 | 87.72 | 82.81 ± 0.00 |
| upsampling | 51.01 | 44.53 | 82.25 | 78.65 ± 1.63 |
| background | 16.87 | 6.25 | 50.36 | 48.18 ± 0.45 |
| rotation | 31.56 | 26.56 | 54.05 | 51.82 ± 0.45 |
| shear | 67.02 | 61.72 | 65.40 | 58.07 ± 0.90 |
| distortion | 61.87 | 57.03 | 68.60 | 63.02 ± 0.45 |
| distortion_rbf | 58.31 | 54.69 | 70.66 | 66.67 ± 0.45 |
| distortion_rbf_inv | 62.03 | 57.81 | 72.45 | 68.75 ± 0.00 |

## Detailed accuracy record

All files underknowledge/evidence/cloudfixer_transfer_20261114/:

- development_grid.csv:48rows, PM/terminal counts/accuracy and paired changes,
  inference runtime and peak memory.
- development_per_corruption.csv:720rows (48settings×15corruptions), both
  readouts' correct counts/128 and accuracy, delta and shared inference time.
- development_reference_per_corruption.csv:60rows (source+3nativeCFrepeats)×15,
  correct counts and accuracy/runtime, no ensemble.
- paper_pointmae_per_corruption_comparison.csv:15rows, paper/source/CF and
  local/source/CFmean/operationalSD and adaptation gains.
- development_independent_recount.json: accepted rawbinder, all48configs and
  selected source/nativeCF/best-readout/per-corruption outcomes.
- paper_results_reference.json andsource_reuse_protocol_audit.json: primary
  source URLs, table numbers, exact values, parsed options and code SHA witnesses.

Raw ZIP and completeNPZs remain undercanonicalresult path; independent full
3120payload recounted records were already accepted, not recomputed GPU outputs.
This audit checks720per-corruption rows sum back to each1920prediction config,
source/nativeCF rows and manifest/worker/source hashes. Priorraw Drive backup
is preserved; these newly derived audit CSVs are versioned locally and not
claimed to be uploaded after the CPU session was closed.

## Next falsifying checks, not executed

[Open] Before a paper-level claim, run the native author Point-MAE entrypoint
on all2468objects with seed2/paperB1 and verify source andplainCF, then isolate
harness/RNG/B1vsB32 differences underidentical IDs. Compare againstI3, keeping
CF-O andindependent-view voting separate. If targetingmain-tablePoint2Vec,
use that releasedhead/checkpoint for both our sampler andCF; do not change
only one comparator. No newGPU/runtime/weights, PM redesign, confirmatory test
or inference is launched by this audit.

## 2026-11-18 — subsequent held-out confirmation and transfer status

[Run/Code/Inference] Official Point2Vec support is implemented and actually GPU-validated: strict180-state/40-class head, pinned checkpoint, official class order/preprocessing, seed2/B1 author stream, native call/input/RNG/XYZ-head replay audits and scoped deterministic proximal branches. Full2468/all15/s5 A comparison is still running. Accepted B256 Point-MAE results concern gradient-free proximal readouts; they do not close the separate full-data native Point-MAE paper-baseline reproduction gate. [Full evidence and limitations](cloudfixer_pm_hypotheses_confirmation_20261118.md). [Current execution](cloudfixer_two_colab_execution_20261118.md).


## 2026-11-20 — Full Point2Vec comparison completed

[Code/Run] All three main arms and the preadaptation source control share frozen
Point2Vec/official XYZ diffusion, ModelNet40-C2,468/all15/s5, author seed2/B1,
loading/preprocessing/class order and checkpoint identities. Primary native CF
is first plain logits; diagnostic vote1 executes separately for author RNG.
Source61.985413%, PM74.954079%, N32 true terminal73.233387%, plainCF77.244733%;
diagnostic vote1 77.106969%. Whole-policy results do not establish isolated PM
causality or paper parity. Source is measured in the matched author stream,
not an independently reproduced standalone paper baseline. Native CF combines
four resumed A10040 epochs; saved bytes/RNG are preserved, but KNN backward
optimizer bitwise parity/seed variability were not established. No full-test
retuning occurred. [Full report and archive identities](cloudfixer_point2vec_full_results_20261120.md).
