# CloudFixer late probability averaging with Point-MAE on held-out ModelNet40-C objects

**2026-11-18 · Colab B · Independently recounted**

[Run] **Current-status reconciliation, 2026-11-21:** B and the separate full
Point2Vec A study are complete and preserved. A's owned r13 was officially
flushed/unmounted and stopped on2026-11-20; fresh server inventory is empty.
The dated statements below that A was still running describe the Nov18 B
acceptance checkpoint, not the present state. Raw/derived archive bytes and
their historical `a_status` fields remain unchanged.
[Detailed Turkish A/B test inventory](cloudfixer_ab_detailed_tests_results_20261121.md) ·
[Full A results](cloudfixer_point2vec_full_results_20261120.md).

**[Run]** On 256 reserved ModelNet40-C objects, the prespecified late-window probability mean (PM) beat its own native t0 terminal at each of three reverse-chain settings and in all nine schedule-by-seed comparisons. The pooled gains were **+0.903, +1.727 and +0.660 percentage points (pp)**. The largest within-chain gain occurred on the setting with the weakest terminal. N8/.1 had the highest absolute late-PM accuracy among the three locked settings.

### Protocol and scope

[Code/Run] Frozen official ModelNet40 XYZ diffusion and Point-MAE classifier. All 15 corruptions, severity 5, seeds 0/1/2, batches of 32, start index K80 on the 500-level schedule, retained fraction .95. Each schedule has 3,840 predictions per seed and 11,520 pooled predictions. The 1,080 units cover 34,560 schedule-specific trajectories. All readouts of a trajectory share the input, classifier and noise. Pre/post probes were observed with common classifier RNG after transport. N counts reverse updates and there is one additional denoiser call at t0.

The N8 window [20,10,0] and N12 window [27,0] were selected on historical development logits and locked before confirmation. An audit of six earlier raw archives found zero overlap between the 256 reserved IDs and 136 historical IDs. Independence applies to object selection for these hypotheses. The official benchmark and development results had already been examined. The confirmation set contains 256 of the benchmark’s 2,468 test objects. It estimates macro accuracy across all 15 corruptions, with identical denominators giving the same micro accuracy. [Handoff](cloudfixer_point2vec_hypothesis_handoff_20261118.md) · [Prior diagnosis](cloudfixer_terminal_diagnosis_20261115.md) · [ID audit](evidence/cloudfixer_two_colab_20261118/confirmation_id_audit.json).

Differences below are paired. A repair changes a wrong reference prediction to correct, and a harm changes a correct reference to wrong. CIs use 10,000 paired bootstrap draws over the 256 original IDs, keeping all corruptions and seeds for each sampled ID together. These are nominal 95% intervals conditional on the three seeds, with no multiplicity adjustment. Repeated seeds and corruptions do not create 11,520 independent objects.

### Primary result and timing controls

[Run] Equal PM averages float64 softmax vectors from pre-correction clean estimates and the true terminal. Each row uses the terminal of the same trajectory as reference. The canonical indices are [70,50,30,0] for N8 and [67,47,27,0] for N12. All-step PM uses all 9 or 13 candidates.

| Schedule N / τ | Window | Terminal correct / 11,520 (%) | PM correct / 11,520 (%) | Repairs | Harms | Δ pp | 95% CI (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N8 / .1 | Locked late | 8295 (72.0052%) | 8399 (72.9080%) | 213 | 109 | +0.903 | [+0.564, +1.241] |
| N8 / .1 | Canonical | 8295 (72.0052%) | 8233 (71.4670%) | 241 | 303 | -0.538 | [-1.094, +0.009] |
| N8 / .1 | All steps | 8295 (72.0052%) | 8229 (71.4323%) | 236 | 302 | -0.573 | [-1.137, -0.026] |
| N12 / .1 | Locked late | 8112 (70.4167%) | 8311 (72.1441%) | 315 | 116 | +1.727 | [+1.276, +2.170] |
| N12 / .1 | Canonical | 8112 (70.4167%) | 8265 (71.7448%) | 389 | 236 | +1.328 | [+0.729, +1.892] |
| N12 / .1 | All steps | 8112 (70.4167%) | 8242 (71.5451%) | 398 | 268 | +1.128 | [+0.503, +1.727] |
| N12 / .8÷12 | Locked late | 8286 (71.9271%) | 8362 (72.5868%) | 212 | 136 | +0.660 | [+0.295, +1.033] |
| N12 / .8÷12 | Canonical | 8286 (71.9271%) | 8226 (71.4062%) | 225 | 285 | -0.521 | [-1.033, -0.017] |
| N12 / .8÷12 | All steps | 8286 (71.9271%) | 8237 (71.5017%) | 251 | 300 | -0.425 | [-0.920, +0.069] |

Canonical and all-step PM changed direction across schedules. Late PM exceeded canonical PM by +1.441 pp at N8/.1, +0.399 pp at N12/.1 and +1.181 pp at N12/.8÷12. The N12/.1 interval for late versus canonical included zero [−0.026,+0.825]. At N8 the late policy also adds t20 and t10, so that comparison alone cannot attribute the entire benefit to removing early probes.

| Schedule | Seed | Terminal correct / 3,840 | Late PM correct / 3,840 | Repairs | Harms | Δ pp | 95% CI (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N8 / .1 | 0 | 2770 | 2790 | 65 | 45 | +0.521 | [-0.052, +1.068] |
| N8 / .1 | 1 | 2768 | 2816 | 81 | 33 | +1.250 | [+0.755, +1.771] |
| N8 / .1 | 2 | 2757 | 2793 | 67 | 31 | +0.938 | [+0.495, +1.406] |
| N12 / .1 | 0 | 2710 | 2772 | 95 | 33 | +1.615 | [+0.990, +2.292] |
| N12 / .1 | 1 | 2699 | 2769 | 114 | 44 | +1.823 | [+1.146, +2.500] |
| N12 / .1 | 2 | 2703 | 2770 | 106 | 39 | +1.745 | [+1.120, +2.370] |
| N12 / .8÷12 | 0 | 2773 | 2782 | 68 | 59 | +0.234 | [-0.391, +0.859] |
| N12 / .8÷12 | 1 | 2760 | 2791 | 73 | 42 | +0.807 | [+0.260, +1.354] |
| N12 / .8÷12 | 2 | 2753 | 2789 | 71 | 35 | +0.938 | [+0.391, +1.510] |

Seven of the nine per-seed intervals excluded zero. The two intervals containing zero were seed 0 at N8/.1 and seed 0 at matched-strength N12.

**Corruption directions.** Pooled over seeds, late minus terminal was positive on 15/15 corruptions for N8/.1. At N12/.1 it was positive on 13/15, with three fewer correct on each of cutout and density_inc. At N12/.8÷12 it was positive on 11/15, zero on uniform, and negative on cutout (−5), distortion (−1) and upsampling (−1). Distortion_rbf contributed +14, +31 and +20 correct respectively out of 768 observations per schedule. These counts are descriptive. No corruption received its own policy. The per-corruption file retains repairs, harms, correct counts, timing and peak memory per seed.

### What early votes contributed

[Run] The prospectively specified explanation uses the canonical four probes. On a harm, the fixed rival is the incorrect class selected by PM. On a repair it is the incorrect class selected at the terminal. Each entry below is the **mean of p(true) − p(fixed rival)** over that transition group at one probe. The four margins average to the PM margin by algebra. Their observed sizes, which probes favored the rival and how often each probe was correct are empirical results.

| Schedule | Transition | n | Mean margin, early → t0 | True top-1 counts | Rival top-1 counts | Confident wrong counts (≥.9) |
| --- | --- | --- | --- | --- | --- | --- |
| N8 / .1 | harm | 303 | -0.530 / -0.482 / -0.135 / +0.352 | 17 / 17 / 94 / 303 | 252 / 262 / 186 / 0 | 74 / 56 / 10 / 0 |
| N8 / .1 | repair | 241 | +0.295 / +0.460 / +0.369 / -0.225 | 147 / 202 / 188 / 0 | 36 / 12 / 37 / 241 | 2 / 0 / 0 / 3 |
| N12 / .1 | harm | 236 | -0.524 / -0.483 / -0.071 / +0.330 | 8 / 12 / 81 / 236 | 198 / 206 / 123 / 0 | 58 / 38 / 7 / 0 |
| N12 / .1 | repair | 389 | +0.390 / +0.534 / +0.465 / -0.290 | 276 / 335 / 318 / 0 | 51 / 25 / 39 / 389 | 7 / 0 / 0 / 15 |
| N12 / .8÷12 | harm | 285 | -0.538 / -0.484 / -0.056 / +0.352 | 12 / 20 / 112 / 285 | 238 / 244 / 149 / 0 | 71 / 50 / 8 / 0 |
| N12 / .8÷12 | repair | 225 | +0.310 / +0.479 / +0.374 / -0.236 | 143 / 201 / 184 / 0 | 30 / 15 / 33 / 225 | 0 / 0 / 0 / 2 |

All reported canonical reversals had strict truth-versus-rival preferences at PM and terminal. On the 303 N8 harms the first probe selected the eventual wrong rival 252 times and the truth 17 times. The first two mean margins were strongly negative, and a smaller negative third margin reinforced them. The repair rows show the opposite voting pattern. N12/.1 produced 389 repairs against 236 harms even though early wrong votes were pronounced within the harm group. This decomposition explains the classifier vote on an observed trajectory. It cannot establish the cause of the underlying geometry.

### How much does entropy identify the better probe?

[Run] The score is negative predictive entropy. The pooled AUC compares a randomly chosen correct candidate and wrong candidate across the full dense sample. The within-example AUC compares correct/wrong candidate pairs from the same original ID, corruption, seed and schedule, assigning half credit to ties. The example-weighted column gives each mixed-correctness trajectory equal weight.

| Schedule | Pooled AUC | Within, pair-weighted | Within, example-weighted | Mixed trajectories | Within-trajectory pairs | Wrong / all confident probes (max p ≥ .9) |
| --- | --- | --- | --- | --- | --- | --- |
| N8 / .1 | 0.867 | 0.647 | 0.658 | 2508 | 35468 | 5,825 / 58,903 |
| N12 / .1 | 0.868 | 0.640 | 0.658 | 2799 | 76090 | 8,196 / 84,677 |
| N12 / .8÷12 | 0.867 | 0.635 | 0.654 | 2689 | 73842 | 8,399 / 85,344 |

The pooled association was considerably stronger than the ordering within one trajectory. Remaining within-trajectory AUC above .5 indicates some ranking information. The weighting comparison tests whether that information improved a fixed readout. Each cell below is the pooled accuracy difference in **pp versus equal PM on the same window**. Linear weights are proportional to 1−H/log(40), with equal fallback. Exponential weights are proportional to exp(−βH). β=0 reproduces equal PM exactly.

| Schedule | Window | Linear | β0 | β.5 | β1 | β2 | β4 | β8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N8 / .1 | canonical | +0.009 | +0.000 | +0.000 | -0.052 | -0.113 | -0.139 | -0.182 |
| N8 / .1 | all | -0.017 | +0.000 | -0.043 | -0.061 | -0.234 | -0.295 | -0.486 |
| N8 / .1 | late | -0.052 | +0.000 | -0.043 | -0.069 | -0.087 | -0.052 | -0.095 |
| N12 / .1 | canonical | +0.026 | +0.000 | +0.009 | +0.043 | -0.165 | -0.200 | -0.339 |
| N12 / .1 | all | -0.078 | +0.000 | -0.095 | -0.174 | -0.365 | -0.538 | -0.712 |
| N12 / .1 | late | +0.035 | +0.000 | +0.052 | +0.017 | +0.017 | -0.052 | -0.069 |
| N12 / .8÷12 | canonical | +0.026 | +0.000 | +0.035 | +0.043 | -0.043 | -0.174 | -0.243 |
| N12 / .8÷12 | all | -0.122 | +0.000 | -0.139 | -0.148 | -0.260 | -0.417 | -0.608 |
| N12 / .8÷12 | late | -0.017 | +0.000 | -0.009 | -0.026 | -0.035 | -0.035 | -0.043 |

Neither linear nor β1 consistently improved a locked late window. Their nominal late-window intervals versus equal PM included zero in all three schedules. Strong weighting reduced dense PM in every setting. The registered rules are reported as a family, with no confirmation-based selection.

### Cumulative correction, observation position and the t0 readout

[Run] At fixed τ=.1, increasing N from 8 to 12 raises nominal Nτ from .8 to 1.2. The third schedule lowers τ at N12 so that Nτ returns to .8. Each cross-schedule comparison below uses matched original inputs and seeds. The correction and displacement columns give dataset means of Euclidean RMS point displacements in model-input coordinates. Summed corrections accumulate step magnitudes and differ from net motion.

| Schedule | Sum of correction RMS | Terminal displacement from input | First late probe logSNR | Mean true-minus-best-other logit margin, first late → terminal | Pre-extra state → terminal RMS |
| --- | --- | --- | --- | --- | --- |
| N8 / .1 | 0.09989 | 0.35187 | 5.7391 | 3.164 → 2.946 | 0.00199 |
| N12 / .1 | 0.13360 | 0.35244 | 5.1383 | 3.075 → 2.700 | 0.00197 |
| N12 / .8÷12 | 0.10360 | 0.35209 | 5.1383 | 3.131 → 2.920 | 0.00186 |

All schedules begin at native t80 with mean logSNR 2.9328 and finish at t0 with logSNR 11.5129. Fixed τ lowered N12 terminal accuracy by 1.589 pp versus N8 [−2.153,−1.033]. At N12, reducing τ recovered 174 correct predictions net (+1.510 pp [+.955,+2.075], 410 repairs and 236 harms). The matched N12 terminal differed from N8 by −0.078 pp [−.417,+.243]. Nτ matching narrowed both the difference in summed correction magnitudes and the terminal accuracy gap. Similar terminal accuracies coexist with 180 repairs and 189 harms across the two matched-strength schedules. Their discretizations and point trajectories differ. This bounded control supports sensitivity to cumulative correction strength and does not prove equal dynamics or a specific overshoot mechanism.

On the paired transition from the single first late probe to the terminal, 310/471/341 previously correct predictions became wrong, while 230/276/280 wrong predictions became correct, in schedule order. The first late probe (t20 or t27) alone gave 8,375/8,307/8,347 correct. These are single-probe diagnostics. The PM accuracy above also uses the other locked probes. On late-correct/terminal-wrong versus late-wrong/terminal-correct cases, mean remaining correction RMS was .02085/.02088, .03567/.03522 and .02903/.02896 respectively. Remaining displacement magnitude alone offers little separation between those groups.

The post-correction PM observes corrected clean XYZ at each nonzero selected index while retaining the same true terminal. The pre-extra-t0 state still contains noise. The last post-correction clean point set precedes transport to this state. The additional native t0 denoiser call produces the true terminal. Paired results below distinguish all three observations. Repairs and harms use the listed reference.

| Schedule | Observation contrast | First-named correct / 11,520 | Repairs | Harms | Δ pp | 95% CI (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| N8 / .1 | Post late PM vs pre late PM | 8361 | 93 | 131 | -0.330 | [-0.573, -0.078] |
| N8 / .1 | Last post clean vs terminal | 8321 | 139 | 113 | +0.226 | [-0.061, +0.512] |
| N8 / .1 | True terminal vs pre-extra state | 8295 | 90 | 57 | +0.286 | [+0.078, +0.495] |
| N12 / .1 | Post late PM vs pre late PM | 8285 | 98 | 124 | -0.226 | [-0.512, +0.061] |
| N12 / .1 | Last post clean vs terminal | 8128 | 127 | 111 | +0.139 | [-0.148, +0.417] |
| N12 / .1 | True terminal vs pre-extra state | 8112 | 105 | 72 | +0.286 | [+0.043, +0.530] |
| N12 / .8÷12 | Post late PM vs pre late PM | 8374 | 86 | 74 | +0.104 | [-0.104, +0.312] |
| N12 / .8÷12 | Last post clean vs terminal | 8290 | 110 | 106 | +0.035 | [-0.234, +0.304] |
| N12 / .8÷12 | True terminal vs pre-extra state | 8286 | 84 | 72 | +0.104 | [-0.122, +0.330] |

The extra t0 call gave a positive net accuracy change at all three settings. The interval included zero at matched-strength N12. Observing post-correction late clouds reduced PM at N8, gave an uncertain negative difference at fixed-strength N12, and an uncertain positive difference at matched-strength N12. There is no common post-correction advantage.

### Interpretation of the four hypotheses

| Prespecified question | Held-out conclusion |
| --- | --- |
| Does timing explain when PM beats its terminal? | Supported for the locked late windows. All three pooled intervals exclude zero. Canonical and all-step PM change sign with the schedule. Early rival votes explain a subset of reversals, and early correct votes also repair terminal errors. |
| Is low entropy a reliable way to select better steps on one trajectory? | Partially supported as a ranking signal. The large pooled association weakened within examples, and fixed entropy weighting showed no consistent gain over equal PM. |
| Does cumulative correction help explain the N8/N12 terminal gap? | Supported by the bounded strength control. Reducing N12 τ restored most of the fixed-τ accuracy gap. Three settings cannot identify a general optimum or establish overshoot on individual objects. |
| Does the observation position explain the terminal disagreement? | Position changes predictions and the extra t0 call can help. The evidence does not support a universal pre/post ranking. The study contains no LION decoder and cannot test a LION-specific decoder advantage. |

### Verification and evidence

[Run] Run `20261118-093757_confirmation` used session `tta-cloudfixer-pm-hypotheses-a100-20261118-r13`, endpoint `gpu-a100-hm-euw4a0-nap9xkiscfxe`, on an NVIDIA A100-SXM4-80GB. Its raw Drive destination is `MyDrive/thesis/result/cloudfixer_two_colab_20261118/b_412a15c74123/archives/20261118-093757_confirmation.zip`.

[Run] The technical pilot replayed the historical 128 development IDs on uniform/seed0 in four batches, with dense, sparse and terminal-only modes per batch. Every historical canonical-four and dense-nine logit comparison had maximum absolute difference zero. Input/model/RNG, transport and endpoint checks passed. The technical terminal count was 112/128. A fresh process reopened the 12 pilot payloads. Confirmation used the separate 256 IDs. The full phase took 8,414.28089 s (2 h 20 min 14 s). Measured diagnostic compute per schedule totaled 1,872.13, 2,619.04 and 2,618.91 s, with peak allocated GPU memory of 9,287.39, 9,293.40 and 9,293.40 MiB. Each trace includes pre/post classifier forwards, and that cost is shared by all its offline readouts. [Pilot validation](evidence/cloudfixer_two_colab_20261118/b_pilot_independent_validation.json) · [Recovery evidence](evidence/cloudfixer_two_colab_20261118/b_pilot_reopen_and_phase_estimate.json).

The accepted raw ZIP contains 2,199 members and 395,520,996 bytes, SHA256 `80ad0b9192690cc4959fafe31fcb53ba6a127a1831da4e44787cf9ed2a57ffb0`. It was downloaded through Colab A's separate DriveFS mount. Outer member/CRC/SHA, source/manifest and operator checks passed. The local analyzer validated registered unit coverage, receipts, label SHA, native environment/pilot evidence and saved probabilities against logits, then reproduced the archived CSV and JSON reports. Its completion marker was `CF_HYPOTHESES_RECOUNT_PASS c3c9dfbe949341bd5638f92ad626a3dfed0ce24b6a0c4ad12aac92f3e173a7f1 1080`. The canonical contribution report was calculated after acceptance and checked against the paired counts. [Raw archive](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation.zip) · [Recount log](../result/preparation/cloudfixer_two_colab_20261118/b/standard_recount.log).

| Identity | SHA256 or Git revision |
| --- | --- |
| B scientific source | `eec1e8b2fe272e9d1f42b6a628cacd5341f1c8c8` |
| Official CloudFixer source | `f043e24414fcf4165fdc892abff21215c9989141` |
| B task | `412a15c74123f6d3ee2243c1030d9bd4dfc1e030f3b18a8836a5772838310de6` |
| Loaded native environment | `f78e6d2040e50bd801217317d5a26af7dfdd329e02c5e3aeb36bbb73ae6487bd` |
| Point-MAE checkpoint | `507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75` |
| Diffusion checkpoint | `b64841d022e08b1a89f58dd9b16d9023e00be52fbca3dca735d5798362bb5a5a` |
| Canonical ordered confirmation IDs | `6793dfb437afdb6b90c491809d3f738df0713392eb1f239164b6b52cbaf622f9` |
| Original labels | `97b6e660820103074ab192f10b6d9c33aa0446f1547ddb6ad2ca35a029a63b31` |

[Open] The confirmed claim concerns this Point-MAE protocol. It supplies no full-benchmark Point-MAE estimate or evidence of a generally superior diffusion prior, decoder or classifier. Colab A's separate full Point2Vec run was ongoing when this B record was written; it subsequently completed on2026-11-20, as documented in the [full A report](cloudfixer_point2vec_full_results_20261120.md). [Execution record](cloudfixer_two_colab_execution_20261118.md).

Local derived records contain 279 schedule/seed/readout summaries, 4,185 corruption rows and 479 paired contrasts, including the cross-schedule and per-seed controls. The transition file identifies the original objects and corruption/seed combinations on which the first late probe and terminal disagree.

[Summary](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/summary.csv) · [Per corruption](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/per_corruption.csv) · [Paired differences and intervals](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/paired.csv) · [Entropy, geometry and margins](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/diagnostics.json) · [Object-level terminal transitions](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/terminal_transitions.csv) · [Canonical vote contributions](../result/modelnet40_c/cloudfixer_hypotheses/20261118-093757_confirmation_recount/canonical_contributions.json).

[Run/Lifecycle] At2026-11-18T13:55UTC, fresh server inventory no longer listed
B's assignment; usage reported only A at6.73CU/h. No agent stop/release/delete
command was issued, and the loss reason is unknown. B's complete raw and
derived archives had already passed independent A Drive downloads and local
acceptance, so no scientific output was lost. At that checkpoint this did not
block the then-running A experiment, which later completed. The earlier B
idle-close question is obsolete.
