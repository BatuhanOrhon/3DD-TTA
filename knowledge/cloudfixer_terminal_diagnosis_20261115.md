# Why canonical CloudFixer N8 PM trails its terminal — 2026-11-15

[User report/Question] Investigate the failure to exceed terminal, reconcile the
earlier LION multi-dataset/multi-head evidence, and propose falsifiable hypotheses.
Evidence below separates **[Run]**, **[Code]**, **[Inference]**, and **[Open]**.
Source at start: branch `research/cloudfixer-transfer-20261114`, commit `8ede8d6`.
No model inference, new GPU allocation, or production sampler change.

## What LION actually established

[Run] The [Oct31 frozen N8 transfer study](chamfer_depth_transfer_results_20261031.md)
used 140 original objects/dataset, all15 corruptions, severity5, seeds0/1/2,
four frozen heads, normal8/Background35, original encoded decoder style,
eta=.003/gamma=.03, p=.95, DDIM final-alpha1. These are *own same-run terminal*
comparisons; each entry is the three-seed all15 mean:

| Dataset | Head | Terminal % | Equal PM % | PM − terminal pp |
|---|---|---:|---:|---:|
| ModelNet40-C | Point-MAE | 64.968254 | 66.000000 | +1.031746 |
| ModelNet40-C | DGCNN | 68.476190 | 69.571429 | +1.095238 |
| ModelNet40-C | CurveNet | 70.111111 | 71.761905 | +1.650794 |
| ModelNet40-C | PointNeXt | 69.412698 | 70.920635 | +1.507937 |
| ScanObjectNN-C | Point-MAE | 40.809524 | 41.857143 | +1.047619 |
| ScanObjectNN-C | DGCNN | 38.936508 | 40.317460 | +1.380952 |
| ScanObjectNN-C | CurveNet | 56.650794 | 57.888889 | +1.238095 |
| ScanObjectNN-C | PointNeXt | 50.523810 | 50.698413 | +0.174603 |

[Run/Limits] All eight mean endpoints improve; seven conditional object-cluster
95% intervals exclude zero. Scan PointNeXt has CI[-.460317,+.825397]pp and
seed1 decreases by .095238pp. Its comparison with the *separately run original*
seed0 terminal is a 51% tie, a different comparator. Thus “all eight means” is
supported; “every corruption/seed/configuration” is contradicted by the records.

[Run] Larger single-seed GSD-inspired studies also show positive aggregate PM
differences: ModelNet2468 objects, DGCNN/CurveNet/PointNeXt +.734738/.913020/.886008pp;
Scan581 objects, CurveNet/PointNeXt/DGCNN/Point-MAE +1.342513/1.457258/1.250717/1.560528pp.
There are negative corruption rows in each study. See
[ModelNet handoff](modelnet_multibackbone_trajectory_handoff_20261025.md) and
[Scan findings](gsd_multibackbone_final_findings_20261019.md).
The raw ZIP paths documented there are absent from this worktree: the present
historical audit uses their accepted compact evidence, not a fresh raw-logit recount.

[Run/Scope] Later full ModelNet/Point-MAE common-N8 tuned-gradient study has
terminal62.571583% → PM63.735818% (+1.164236pp; 37,020 events/seed0).
Its separate cheap/proximal PM63.722312% is **not** a same-trace comparison with
that tuned-gradient terminal. The eight-head Oct31 study predates the current
gradient-free/proximal baseline; it is not eight-head validation of that baseline.
See [uniform-horizon results](lion_guidance_uniform_horizon_results_20261108.md).

## CloudFixer premise and new saved-logit diagnostics

[Run] This is our proximal/DDIM-type sampler around the author CloudFixer XYZ
denoiser with frozen author Point-MAE, not native CloudFixer R/Delta optimization.
Canonical PM already beats its own terminal in26/48 coarse-grid settings.
At K80/tau.1, N12 PM1257/1920 exceeds terminal1235/1920. The failure being
diagnosed is specifically canonical N8 four-candidate PM1250 versus terminal1269.
Do not compare a new N8 readout with the best terminal from another configuration
as if they shared a trajectory.

[Run/Exploratory] All rows below use the same128 original IDs/all15/s5/seed0,
K80/tau.1/retained.95/B32, unchanged input/RNG/model identities. New subset
means are posthoc explanatory diagnostics on previously exposed development
labels. All prefixes, suffixes, and leave-one-out means are retained in
[diagnostics.json](evidence/cloudfixer_terminal_diagnosis_20261115/diagnostics.json).
The three views have48 subset rows, plus3 entropy-ranking recounts. Snapshot
times are native indices, not loop counts or denoising-update counts.

| N / saved candidates averaged | Correct /1920 | Accuracy % | Repairs / harms vs own terminal |
|---|---:|---:|---:|
| 8 terminal | 1269 | 66.093750 | 0 /0 |
| 8 canonical [70,50,30,0] | 1250 | 65.104167 | 43 /62 |
| 8 canonical without70: [50,30,0] | 1264 | 65.833333 | 43 /48 |
| 8 canonical last2: [30,0] | 1264 | 65.833333 | 32 /37 |
| 8 dense all9: [80,70,60,50,40,30,20,10,0] | 1256 | 65.416667 | 49 /62 |
| 8 dense late3: [20,10,0] | 1283 | 66.822917 | 38 /24 |
| 8 single20 | 1284 | 66.875000 | 59 /44 |
| 12 terminal | 1235 | 64.322917 | 0 /0 |
| 12 canonical [67,47,27,0] | 1257 | 65.468750 | 72 /50 |
| 12 last2: [27,0] | 1272 | 66.250000 | 62 /25 |
| 12 single27 | 1286 | 66.979167 | 95 /44 |

[Run/Limits] N8 late3 adds14correct (+.729167pp), conditional object-cluster
95%CI[-.052083,+1.562500]pp, versus its own terminal. N12 last2 adds37correct
(+1.927083pp), CI[+.937500,+2.968750]. Bootstrap:128 original-object clusters,
joint across15 corruptions,10,000 draws/seed20261115. These are unadjusted
descriptive intervals after examining many readouts, conditional on one seed
and reused objects. They do not establish a selected policy's generalization.
The degenerate terminal-versus-itself [0,0] interval establishes no equivalence
outside this fixed trace. Dense observer runtime overhead remains as documented
in the [N-step study](cloudfixer_nstep_entropy_20261115.md); there is no new
terminal-only or late3-only latency measurement.

## Hypotheses, support and falsifiers

1. **[Run/Inference: strongest support] Early wrong opinions drown later correct
   opinions; a fractional-loop probe schedule did not transfer.** N8 individual
   probe correct counts at70/50/30/0 are1119/1224/1265/1269. On the62 examples
   that canonical PM breaks, those counts are1/5/15/62. Truth-minus-PM-winning-
   wrong-class probability margins average −.53654/−.44198/−.20193/+.32323;
   their equal sum explains these argmax reversals. Dropping70 reduces harms
   by14 without losing any of43 repairs, but alone still does not beat terminal.
   The missing late20/10 probes allow a mean that does beat it in this cohort.
   **Falsifier/control:** preregister one late schedule on held-out original IDs
   and new seeds; include fixed canonical PM and own terminal. A benefit that
   disappears there supports development-set overfitting instead.

2. **[Run/Inference: strong support] Low entropy identifies easy examples much
   better than it identifies the best time for one example.** Pooled correct/
   wrong entropy AUC is .861701/.862237/.863895 for N8four/N8nine/N12four.
   Restricting comparisons to a correct and wrong candidate of the *same*
   example gives .623256/.636615/.622677 (1075/6500/1076 pairs,325/474/330
   mixed-correctness events). Equal-weight-per-mixed-example alternatives are
   .623590/.647792/.632071. Thus pooled confidence evidence does not justify
   strong temporal weights. On canonical N8 PM's62 harms, early70 entropy
   averages1.0934 despite only1correct; terminal averages1.3849 and is62correct.
   Minimum-entropy candidate selection gives1251/1242/1256 correct: it does not
   solve N8 either. This explains why low-H weighting can reinforce confident
   early errors; it does not prove classifier miscalibration as the sole cause.
   **Falsifier/control:** held-out within-example rankings or calibration that
   demonstrably improve and translate into paired accuracy gains.

3. **[Code/Inference: plausible, not isolated] Prior/representation/noise and
   endpoint geometry differ.** LION probes decode clean latent estimates
   through a VAE with style conditioning (`shared_trajectory.py`); CloudFixer
   probes classify clean XYZ estimates directly (`cloudfixer_diffusion.py`).
   Native noise schedules and cumulative corrections differ even at equal N.
   Our XYZ candidates are recorded before the current proximal correction;
   terminal follows all corrections and an extra index0 denoiser call. LION's
   gradient observer also sees a clean estimate before the current guidance
   update (`tta_gsd_v2.py`), so “only CloudFixer has pre-update probes” would be
   false. No endpoint/correction-placement bug or causal decoder advantage is
   established. **Control:** log effective logSNR and cumulative corrections,
   align probe noise levels, then ablate observation placement without changing
   the chain. A new model run is required for unavailable post-correction probes.

4. **[Run/Inference: unresolved] Longer adaptation can make intermediate opinions
   useful by degrading the endpoint, rather than improving PM intrinsically.**
   N12 single27 has1286correct versus1235 atterminal; N8 single20 has1284 versus1269.
   This is consistent with some late overshoot/over-adaptation, not proof of it:
   N also changes cumulative tau and actual probe indices, and individual
   corruptions can move in opposite directions. Correlated temporal errors
   further limit the benefit of an ensemble from one chain. **Control:** align
   logSNR, hold cumulative correction strength fixed, and compare held-out
   intermediate/end trajectories and geometric drift; do not simply conclude
   larger N or larger entropy weight is better.

## Evidence and next decision

[Run/Validation] Pinned raw ZIPs are
`result/modelnet40_c/cloudfixer_exploration/20261115-110019_nstep_run.zip`
(SHA2a4c544f2f08de453205f556689a47a1aedabf37be4b87e07d8c9348e68d58f7) and
`20261115-114749_entropy_run.zip`
(SHA509ea9699cd2fcde8542fc52564c03146a1f2f30db0d4e20d176e1aa479506df).
The diagnostic script checks SHA/CRC, labels, IDs/coverage, entropy, normalization,
known counts and paired arithmetic. An alternate raw-archive recount validates
all48 subset counts/repairs/harms and all3 within-object rankings (51checks).
See [validation receipt](evidence/cloudfixer_terminal_diagnosis_20261115/validation.json)
and `scripts/diagnose_cloudfixer_terminal.py`. No historical LION raw archive,
new model inference, cross-prior paired geometry, or unseen-data confirmation
was performed. Preliminary local JSON is retained separately in `result/preparation/`.

[Decision/Open] PM has not failed universally on either model. Prioritize probe
timing and conditional confidence diagnostics over a larger entropy coefficient.
N8 late3 is a candidate for future held-out validation, not a promoted baseline;
its dense exploratory trace cannot establish sparse deployment cost/parity.

[Archive/Runtime] Scientific diagnosis/compact evidence/source/knowledge ZIP
`20261115-165611_diagnosis.zip`,427516B,
SHA696016fe13aa5a89c0720766222dc511900a3b41ccc1bcd31d25a57d7d956b18,
backed up16:56:38UTC to `MyDrive/thesis/result/cloudfixer_terminal_diagnosis_20261115/`.
DriveFS→CPU→Mac readback SHA/CRC pass; independent server sync is not proven.
[Receipt](evidence/cloudfixer_terminal_diagnosis_20261115/drive_backup_receipt.json).
Same owned transfer-only CPU `tta-cloudfixer-weighted-pm-drive-cpu-20261115-r1`
(endpoint m-s-usc1c9-1lut8c3l2e92z,monitor93130) remains live. New diagnosis
transfer assignment/600s cap recorded separately from completed weighted-PM
phase; no previous allocation/idle time reset or new allocation. Transfer took
~27s from phase registration, within cap; rate previously .08CU/hour, no new
CU usage measurement. No GPU setup/research cells or runtime teardown.

[User request/Planned followup — 2026-11-18] The four hypotheses above are the
basis of Colab B in the [two-Colab continuation handoff](cloudfixer_point2vec_hypothesis_handoff_20261118.md).
Colab A separately transfers the highest observed PM and true-terminal policy
candidates to Point2Vec alongside author plainCF. This is a future execution
request, not new validation of these hypotheses or a runtime-status update.

## 2026-11-18 — subsequent held-out confirmation and transfer status

[Run/Code/Inference] The reserved256/all15/s5/seeds0/1/2 Point-MAE study now independently confirms locked late PM over its own true terminal at allthree fixed schedules (+.903/+1.727/+.660pp; pooled nominal95% object-cluster CIs above0). Early wrong rival votes explain conditional canonical harms, but early correct votes also repair errors. Matching N*tau restores most of N12's terminal deficit. The extra t0 call improves against its noisy input state, so endpoint degradation cannot simply be assigned to that extra call. This is bounded evidence about this XYZ proximal host, not LION or Point2Vec causality. [Full evidence and limitations](cloudfixer_pm_hypotheses_confirmation_20261118.md). [Current execution](cloudfixer_two_colab_execution_20261118.md).
