# Diffusion trajectory selection, averaging and early stopping

**2026-10-25 attribution amendment [Paper/Code/Inference/Open]:** the dated
proposal below records the user's idea, not evidence of first invention.
Rosnati et al.(ML4H2023, section4/equation9), Wang et al.(2025,
section4.1/equation4), Li et al.(NeurIPS2025, appendixB.4) and Purge-Gate
(ICCV2025, section3.3.1/equations20–21) provide relevant prediction-ensemble
and entropy-selection precedents. These were audited retrospectively; do not
claim they were all the original inspiration or copied implementations.
[Exact primary citations, differences and thesis wording](gsd_trajectory_attribution_20261025.md).
Generic timestep ensembling/minimum-entropy selection is not a novelty claim.

[User report/Open] 2026-10-25 follow-up: the
[four-paper reading list](gsd_trajectory_attribution_20261025.md#reading-list-requested-by-the-user--2026-10-25)
retains full titles, primary links and section pointers. The user requests
considering logit mean as an alternative to the tested probability mean;
[this within-trajectory comparison](open_questions.md#2026-10-25--logit-mean-versus-probability-mean)
is recorded, not evaluated. Historical averaging over separate trajectories
is not evidence for this checkpoint policy.

Date: **2026-10-11**. Evidence: [User report], [Code], [Run], [Paper],
[Inference], [Open]. Inspected baseline: gsd-smooth-spectrum at
`f670c07440533f01e10fd149fad2480d36f9194c`.
Status: documented hypothesis; no implementation, new test or GPU execution.

## User proposal and priority

[User report] Evaluate clean point-cloud predictions from multiple diffusion
steps with frozen Point-MAE. Choose a low-entropy prediction or average
predictions across steps. If successful, profile GSD and SCD and consider
removing one component; alternatively stop diffusion when confidence is
sufficient to reduce total runtime. Accuracy remains the primary objective.

[Inference] Treat this as a concrete input-adaptation research candidate to
investigate before expanding classifier internals. The earlier research note's
PG-first ordering was an assistant recommendation, not an accepted experimental
decision. Purge-Gate remains an advisor discussion item and possible comparator;
no PG implementation prerequisite is imposed on this trajectory experiment.
See [advisor agenda G-001 through G-003](advisor_meeting_log.md).

## Claim to test

[Inference/Open] Later diffusion steps may improve some objects while erasing
class-relevant details in others. Clean intermediate predictions may therefore
complement the terminal prediction. Entropy or temporal consistency might
identify useful outputs without labels. None of these propositions has yet
been demonstrated for the restored legacy trajectory.

Keep Point-MAE weights, BatchNorm state, token flow, checkpoint, preprocessing,
and LION inference modes unchanged. Use the classifier only to score candidates;
do not optimize its weights or backpropagate classification entropy in this
proposal. No source dataset or target labels enter the inference rule.

[Run] The existing two-arm legacy/offdiag probability ensemble improved
macro15 by0.2782pp at seed0. This concerns two complete trajectories and does
not demonstrate intermediate-step complementarity, a stopping rule or a1-2pp
gain. Artifact: `result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/analysis_20261011_selection/`.
Raw SHA256: `534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968`.
Operational legacy accuracy is64.5705%, accepted as practical reproduction.

## Correct intermediate object and observation timing

[Code] In `tta_gsd_v2.py`, each prior forward and scheduler step already
produces `scheduler_output.pred_original_sample`, the clean **local latent**
estimate. This is not a physical point cloud. GSD and selective Chamfer in this
legacy path operate on latent coordinates; there is no full VAE decode at
every guidance step. The decoder is called at the end.

For a probe at reverse-loop index j and scheduler timestep t:

1. Capture the clean latent estimate and the same pre-update style condition
   used by the prior to produce it. Decode those together with VAE in eval/no_grad.
2. Apply the exact baseline inverse transform, normalization and FPS1024 path,
   then frozen Point-MAE; record40 logits/probabilities.
3. For full-trajectory observation, continue the original guidance/update
   untouched. Probes must not mutate style, scheduler, model buffers or main RNG.

The predicted clean latent is produced **before** the current guidance update.
Do not pair it silently with post-update style or call it the final result of
that guided step. The original terminal decode uses final updated latent/style;
it must remain a separate candidate and comparator, even at the final loop
iteration. A last-step clean prediction need not equal the legacy final output.
Decoder input dimensions and conditioning must match the existing contract.

Record loop index, scheduler t, actual style hash, pre/post-update designation,
and whether the candidate is predicted-clean or terminal. Use original object
indices as sample IDs. Point-MAE currently generates unused random masks even
on its all-token path; isolate/restore RNG state or consume cached prepared
inputs to prevent probes changing later baseline draws. Detach probe tensors;
do not retain all guidance autograd graphs or large GPU states across steps.

## Three separate inference policies

Let p_j=softmax(logits_j), with fixed temperature1 and40 classes.
Normalized entropy e_j=-sum_c p_jc log(p_jc)/log(40) lies in[0,1].
No temperature fitting or per-corruption threshold tuning is implied.

### A. Full-trajectory output selection

Choose j*=argmin_j e_j from a predeclared candidate set, including the baseline
terminal candidate. Use deterministic tie-breaking fixed before running.
All candidate steps have to be evaluated: selecting the best over the completed
trajectory does **not** save diffusion work.

### B. Temporal prediction averaging

Use p_bar=(1/K)sum_j p_j and argmax_c p_bar_c as the simplest fixed control.
Average probabilities, not label IDs or point coordinates. Logit averaging is
a distinct rule because logit scale changes its effective weighting. Begin
with equal probability weights; entropy-weighted versions are optional later
ablations and can amplify confidently wrong predictions. Highly correlated
steps may add little diversity; early inaccurate candidates can worsen a mean.

### C. Causal early stopping

At checkpoint j, a possible rule requires minimum elapsed steps, entropy below
tau, and unchanged top class across a predeclared number of recent checkpoints.
An optional probability-distance condition checks whether distributions, not
only argmax labels, have stabilized. Thresholds/checkpoint counts are **not
selected or registered yet**. No future predictions or ground-truth labels may
enter the stop decision. If no stop condition is met, use a predeclared terminal
fallback (or prefix aggregate if that is the separately named policy).

If probing before current guidance and stopping immediately, current-step
backward can also be skipped. State that convention explicitly; it differs
from completing the update before stopping. A rule returning the lowest-entropy
candidate among already observed steps is causal; returning the globally best
future step is not. A stopped probability ensemble can average only the prefix.

## Confidence, stability and correctness are different quantities

[Inference] Low entropy means a concentrated predictive distribution. It does
not establish either correctness or repeatability across steps/sampling views.
A sequence can remain confidently wrong. Searching more checkpoints can select
an unusually overconfident error. Temporal agreement helps diagnose instability
but also does not prove correctness. Measure these relationships rather than
assuming that an entropy minimum is an accuracy maximum.

[Paper] Modern neural networks can be miscalibrated: see
[Guo et al., ICML2017](https://proceedings.mlr.press/v70/guo17a.html).
This motivates caution; it is not empirical evidence that our particular
checkpoint is miscalibrated. Label-based ECE/Brier/NLL and entropy/correctness
diagnostics can be reported after inference, not used as test-label selectors.

## Compute model and component removal

[Inference] Approximate full observation cost:

`T_observe = T_legacy + sum_extra_probes(T_decode + T_preprocess + T_classifier) + overhead`.

For early stopping, saving requires omitted future prior/guidance work to exceed
extra probes plus active-sample management. The graph/eigendecomposition is
upfront in the static legacy path and already paid by an early exit. Do not
assume K probes multiply total latency by K, or that fewer steps guarantee
lower end-to-end time. Measure on the same hardware/batches.

[Code] Legacy constructs the graph even if spectral weight is zero. For fair
accuracy ablation first preserve the host; for deployment timing, separately
implement actual graph skipping and verify equivalent spectral-off outputs.
SCD and GSD share the local-prior backward path. Removing either loss may keep
most backward work; removing both can remove the guidance backward entirely.
Their time contributions therefore need not add linearly. The current code
also computes diagnostic losses separately; profile logging overhead separately
from the mathematical intervention.

If trajectory selection works, compare a matched2x2 guidance matrix:

| GSD | SCD | Interpretation |
|---|---|---|
| on | on | Legacy host + selected trajectory rule |
| off | on | SCD only + identical rule |
| on | off | GSD only + identical rule |
| off | off | Unguided diffusion + identical rule |

Maintain the same prepared inputs, noise, schedule, checkpoint and observation
rules. Profile preprocessing/encode, graph/eigh, prior forward, each loss,
combined backward, decoder, FPS, classifier, transfer/packaging separately.
Measure synchronized GPU timing and end-to-end wall time, warm-up, throughput,
latency distribution and allocated/reserved peak memory. Source computation
with a zero weight is not evidence for the speed of removing that component.
Choose deletion only from the measured accuracy/time tradeoff, including
corruption-specific damage and repeated seeds; no component is selected now.

At B70, marking some samples finished while still evaluating the entire batch
does not skip their GPU work. Real per-object exits need active-batch compaction
or grouping, with gather/scatter overhead and potential numerical/RNG changes.
Whole-batch stopping follows the hardest remaining object. A dense full run can
simulate stop decisions and saved step counts, but cannot establish actual
throughput gains. Verify live scheduling separately after the policy is fixed.

## Bounded experimental sequence (proposal, not launch authorization)

1. **Observe first, without early exit.** Keep all15 coverage, proposed280 fixed
   indices/corruption and seeds0/1/2 from the current development plan. Freeze
   IDs/checkpoint schedule before execution. Proposed sparse checkpoints at
   roughly25/50/75% of reverse iterations, plus the unchanged terminal decode:
   loop counts3/5/8 of10, and8/15/23 of30 (ceil convention). These are proposed
   observations, not new noise depths or registered optimum steps. A denser
   trace on the same bounded scope is conditional if sparse traces miss useful
   changes; do not begin with many complete reruns.
2. **Reuse one trace for simple policies.** Compare unchanged terminal, minimum
   entropy, equal probability mean, and a fixed earlier output as a control.
   Optional input/no-adaptation candidate is a separately named extension. Keep
   per-step logits/indices/timings and enough geometry to audit clean decoding.
   Compare same-cost repeated terminal FPS voting if any policy gain is claimed
   to arise specifically from time diversity. Extra sampling views are deferred
   until this simple comparison shows useful signal.
3. **Assess headroom and falsifiers.** Report actual selector corrected/broken
   counts vs terminal, entropy/calibration diagnostics, selected step histogram,
   temporal disagreement, and a label-using oracle selection ceiling. Oracle
   ceiling is solely a diagnostic; a large ceiling with a failing selector
   means selection is unsolved. No oracle gain means little selection headroom
   among those candidates, though probability averaging can still behave
   differently. A stable earlier fixed output matching adaptation would weaken
   the claim for a sample-specific selector.
4. **Only then optimize runtime.** Simulate causal prefix stopping to develop a
   bounded policy; lock thresholds on explicitly designated development data.
   Re-evaluate actual wall time with live stopping and active batching. Add
   the guidance2x2 matrix if the accuracy mechanism is useful, keeping old
   completed SCD-off/other-host evidence as context rather than substitutes.
5. **Freeze and evaluate.** Full2468x15x3 with matched controls after policy lock.
   Report macro15/14, Background, all corruption/seed differences, sample seed
   SD and within-corruption original-index cluster intervals retaining seeds.
   Timesteps are repeated measurements, never extra independent samples.
   This benchmark has already informed development; seed replication is not
   external validation. A separate evaluation is needed for generalization.

No numerical promotion, acceptable accuracy loss, entropy threshold or latency
budget is invented after observing results. Agree these before the relevant
experiment; the advisor agenda includes this decision. New method gain,
speedup and novelty remain unestablished. Literature novelty review is still
required specifically for trajectory selection/temporal ensembles/early exit.

## Realistic assessment

[Inference] Worth a bounded pilot: uses the existing clean latent estimates,
preserves the classifier, offers multiple predictions from one diffusion path,
and directly tests whether forcing every sample to the last step is harmful.
Main risks are confidently wrong selections, correlated trajectories, expensive
decoding and negligible actual batch speedup. A useful result could be improved
accuracy at tolerable cost or unchanged accuracy with measured faster inference.
Neither outcome, nor a1-2pp gain, is guaranteed. The strongest near-term action
is a sparse observation experiment; start with selection/averaging before
changing guidance or implementing complex early exits.

Existing raw archives, failed Background100/1000 and the separate M1800 branch
remain untouched. This record contains no new model results. No stage/commit/push.


## Later 2026-10-11 - User clarification: online latency is a separate target

[User report] The user emphasizes single-example/small-batch online TTA rather
than only processing a complete test archive in B70. [Inference] At B1 an early
exit directly omits subsequent diffusion calls; it does not wait for other
objects and requires no active-batch compaction. The B70 caveat above must not
be generalized to deny this saving mechanism. Net end-to-end speedup still
compares omitted future work against extra decode/FPS/classifier probes.
Measure onlineB1 mean/p50/p95 latency and accuracy against matched full-horizon
B1, separately from B70 benchmark throughput. Small batches above1 still need
handling of different finish times. Do not relabel B70 accuracy as B1 evidence.

No future samples, future trajectory steps or labels enter online stopping.
The legacy10/30 corruption-name horizon is a disclosed benchmark convention;
unknown-corruption online deployment needs a separately fixed or sample-derived
horizon without a corruption oracle. Do not change the baseline horizon while
implementing the initial observer. Next session starts from accepted legacy,
not offdiag or PG: [handoff](gsd_trajectory_session_handoff_20261011.md).


## Later 2026-10-11 — Observation infrastructure implemented, GPU evidence pending

[Code/CPU/Open] The initial observer/analyzer scope is now implemented in the
uncommitted working tree over f670c074. See [implementation/Colab handoff](gsd_trajectory_implementation_20261011.md).
Default off preserves Legacy; audit records state/terminal identity; observe
adds pre-update clean probes. Policies share one trace, use probability averaging
and terminal-first entropy ties. Fixed-probe controls, oracle diagnostics and
index-cluster intervals are explicit. This supersedes the initial no-implementation
status, without establishing selection benefit, stopping criteria or speedup.
GPU off/audit/observe smoke and subsequent development pilot remain unrun.


## Later 2026-10-11 — Existing5/10-step evidence and its limit

[Run/Code/Inference/Open] The [archived step-count audit](gsd_step_count_evidence_20261011.md)
finds consistent Background5-to10 gains for smooth-only(+1.080497pp) and
unguided(+0.540249pp), with non-monotonic behavior at larger horizons. Both
arms are SCD-off/B32. Large Impulse/LiDAR improvements and Shear losses between
Phase6 normal5 and restored Legacy normal10 are unmatched host contrasts.
The CLI step count changes initial noising timestep as well as loop length;
a5-step run is not a prefix exit of a10-step run. These observations motivate
but do not validate the same-trajectory selector/early-exit hypothesis. Preserve
the observer's10/30 horizon and measure its actual candidates before expansion.


## Recovered smoke results and next averaging candidate - 2026-10-11

[Run/Exploratory] The recovered six-arm smoke completed140 locked seed0 examples for each arm (ZIP SHA-256 `7cd75452ce7d64f3e3359ee66edadb5f08c9cf2d7a57c074b0c12bbf1bed488e`). Observe candidate accuracy: Gaussian terminal76.43%, minimum entropy79.29%, equal probability mean79.29%; Background terminal57.14%, minimum entropy54.29%, mean55.00%. Legacy-off terminal was75.00% Gaussian and57.86% Background. The smoke's small subset shows a mixed corruption response, not an all15 effect. Observe predictions differed from audit on10/140 Gaussian and22/140 Background. Exact RNG replay still produced different VAE latents in4/4 audit batches, so cross-process equality remains unresolved.

[Open/next] Evaluate entropy-weighted probability averaging only after the seed0 all15 pilot has been reviewed. A transparent candidate is a per-example softmax over negative normalized entropies, `weight_k = exp(-tau * H_k) / sum_j exp(-tau * H_j)`, then `mean_probability = sum_k weight_k * probability_k`. Compare with terminal, minimum entropy, equal probability mean and fixed-probe/equal-cost controls from identical saved logits. Freeze tau in advance or use separate calibration data; do not select ? on the same evaluation outcomes and describe that result as confirmatory. Low entropy can still assign more weight to a confidently wrong prediction; neither low entropy nor agreement guarantees correctness or temporal stability. No tau or benefit is claimed.


## 2026-10-11 - Seed0 all15 pilot completed and independently reanalyzed

[Run/CPU] User handoff `pilot_recovered_20261011T152044Z_handoff_20261011T153106Z.zip` is preserved at `result/modelnet40_c/legacy_trajectory/` (SHA-256 `da8092f933f3272416b166996b383f497d547e766b6a859bbd1528b37962cebb`; outer ZIP CRC passed, 392 members). Its nested `pilot_seed0.zip` SHA-256 is `145a0932e3924cee8c6f921673fbcf368fadef7cdc2f1d0e6acece52eb7d5a2e`. Local CPU analysis was rerun from that nested archive and validated full trace coverage. Recomputed `per_corruption_seed.csv` is byte-identical to the report in the handoff. The execution status is complete with 4,200/4,200 selected examples, all15 corruption types and 280 shared IDs per type. The run's `status: partial` denotes the deliberate 280-example development subset, not an interrupted run or full 2,468-example benchmark. The run resolves to current branch HEAD `a351e66ec58f08f3a2104e46cab2cb8b20cdf961`; model state/buffers were unchanged. Environment: NVIDIA A100-SXM4-80GB, PyTorch2.1.2+cu121. No baseline archive was changed.

[Run/Inference] On these same seed0 trajectories, macro accuracy over all15 sampled corruptions was:

| Policy | Accuracy | Difference vs terminal |
| --- | ---: | ---: |
| Original terminal | 63.2143% | reference |
| Minimum entropy | 63.7381% | +0.5238 percentage points |
| Equal class-probability mean | 63.9762% | +0.7619 percentage points |
| Fixed probe1 | 62.5000% | -0.7143 percentage points |
| Fixed probe2 | 62.3095% | -0.9048 percentage points |
| Fixed probe3 | 63.5238% | +0.3095 percentage points |

Against the terminal on the same 4,200 examples, minimum entropy corrected133 and broke111 predictions (net+22); probability mean corrected135 and broke103 (net+32). The 1,000-replicate stratified original-index bootstrap interval for the macro paired difference was[-0.215,+1.214]pp for minimum entropy and[+0.071,+1.500]pp for probability mean. These intervals are conditional on the selected indices and this one seed/trajectory; they do not show seed stability or independent generalization. Do not compare the 63.2143% subset terminal directly with the accepted full-data Legacy B70/M800 result64.570502%: the evaluation populations differ.

[Run/Inference] Corruption-specific paired deltas (percentage points vs terminal; minimum entropy / probability mean):

| Corruption | Min entropy | Probability mean |
| --- | ---: | ---: |
| uniform | +2.14 | +2.50 |
| gaussian | +2.50 | +3.93 |
| background | -3.57 | -3.57 |
| impulse | +1.79 | +0.71 |
| upsampling | +3.21 | +2.86 |
| distortion_rbf | +0.36 | +1.07 |
| distortion_rbf_inv | -0.71 | +0.36 |
| density | +0.71 | -0.71 |
| density_inc | +3.57 | +2.50 |
| shear | -1.43 | -0.71 |
| rotation | 0.00 | +0.36 |
| cutout | +1.79 | +1.43 |
| distortion | -2.50 | -1.79 |
| occlusion | -0.36 | +1.07 |
| lidar | +0.36 | +1.43 |

[Inference/Open] Excluding Background, terminal/minimum-entropy/probability-mean accuracy was63.5969/64.4133/64.6684%. Background fell from57.8571% terminal to54.2857% under both minimum entropy and probability mean. The macro gains therefore coexist with a clear Background regression and mixed per-corruption changes. Across examples, candidate argmax predictions disagreed at least once for25.81%;805 examples were stably wrong across every candidate. Minimum-entropy selection still made1,523 errors. Entropy is not a correctness or temporal-stability guarantee.

[Run/Cost] The Colab run completed in about10m17s by session/run timestamps. Recorded trajectory-runtime sum is544.4s over4,200 examples (~7.71 examples/s); this is observer-instrumented B70 throughput, not a matched baseline speedup. Peak allocated memory was about66.7GB (reserved about71.8GB) on the reported80GB A100. Four sparse probes were evaluated in each of60 batches. No early stop occurred and these full-trace policies do not save diffusion work.

[Open/next] The result supports further seed0 policy analysis, not a promotion claim: evaluate entropy-weighted probability averaging offline from the saved logits, document the weight formula and sensitivity to tau, and retain the Background/per-corruption loss review. Treat any tau sweep on these same280 IDs as exploratory; before claiming general benefit, freeze a policy and check seeds1/2 and an independent evaluation. Do not change the Legacy baseline or10/30 horizons based on this single pilot. Conditional B1 early-stop latency and GSD/SCD ablations remain later work.

## 2026-10-11 - Full-data seed0 result updates candidate assessment

[Run/CPU] The complete37,020-example all15 seed0 artifact and independent
CPU report are recorded in [the implementation handoff](gsd_trajectory_implementation_20261011.md).
Terminal64.592112%, minimum entropy65.051324% (+0.459211pp), equal class
probability mean65.156672% (+0.564560pp) on identical traces. The index-cluster
95% intervals are positive conditional on this single run; no seed variation
is estimated. Mean improves10/15 corruptions and loses5, including Background
-2.674230pp (minimum entropy-3.484603pp). Macro14 mean is65.7560%. The separate
accepted Legacy64.570502% comparison is descriptive, not paired. This expands
the pilot signal, but remains below+1pp and does not support promotion without
reviewing the Background tradeoff and seed repeatability.

Observer-instrumented runtime sums80.4min; total run timestamps span86.1min.
Prior Legacy records68.98min. This suggests around17% added measured compute
by per-corruption sums, but the executions are separate and observer timings
include probes/hash/I/O; not a matched latency benchmark. Peak is62.16GiB
allocated/69.05GiB reserved. Full-trace selection does not save diffusion steps.
Same-cost terminal voting, entropy-weighted offline analysis, locked policy,
seeds1/2 and online B1 stopping remain open. Do not select tau from this run.


## 2026-10-11 - Proposed next accuracy step after the full-data screen

[Run/Inference/Open] Recommend a bounded offline comparison on the saved full
trace before another GPU run. Background fixed clean probes at8/15/23 score
38.1686/49.7164/58.1442%, versus terminal59.4408%; early candidates are much
weaker in aggregate. Uniform/Gaussian early candidates outperform terminal,
so discarding early steps globally can also remove useful signal. These
aggregate results motivate analysis; they do not prove which candidate caused
each averaging error.

Proposed candidates (not implemented or evaluated): (1) a parameter-free,
mild entropy-weighted mean with normalized entropy H in[0,1], weights
w_k=(1-H_k)/sum_j(1-H_j), and equal weights if the denominator is zero;
(2) an equal mean of the latest clean probe and terminal, applied identically
to every corruption, as a late-candidate control. Retain existing terminal,
minimum entropy and equal four-candidate mean as references. No tau or entropy
threshold is selected. This linear weighting is a new explicit proposal,
distinct from the earlier exp(-tau*H) family; no superiority is assumed.

Evaluate paired corrected/broken counts against both terminal and equal mean,
all15/macro14 and each corruption, plus entropy/temporal-disagreement patterns
for corrected and newly broken predictions. Labels score policies only. No
corruption-specific output switch is proposed. A hypothetical repair that only
returns Background to terminal while leaving other policy results unchanged
would add0.1783pp to macro15, bringing the current+0.5646pp to about+0.7428pp;
this is diagnostic arithmetic, not a deployable policy or measured result.
Reaching1-2pp requires additional improvements beyond removing that loss.

Treat this offline comparison as development on already reviewed ModelNet40-C.
Freeze a supported policy before conditional seed1/2 checks; no GPU launch,
new measured accuracy, speedup, independent validation or final policy choice
is authorized or claimed by this recommendation. Causal stopping and GSD/SCD
removal remain later questions.


## 2026-10-11 - User redirects development to the common 10-step, 14-corruption scope

[User decision] The user rejects Background-specific accuracy recovery as the
next research emphasis: its30-step horizon requires known corruption identity,
which is unavailable in the intended inference setting. Prioritize accuracy
research on the other14 corruptions sharing the same10-step horizon. This
supersedes the preceding Background-repair emphasis and any suggestion that
Background recovery is required to select the next development candidate.

[Run/CPU] Re-read the independently analyzed full-data seed0 report, without
new inference. On these14 corruptions (34,552 records): terminal64.960060%,
minimum entropy65.700972% (+0.740912pp), equal probability mean65.755962%
(+0.795902pp). Fixed clean probes1/2/3 score64.470942/64.991896/65.200278%.
These remain one-seed development observations. The reference is the matching
14-corruption terminal; do not compare these percentages to all15 Legacy64.570502%.
Raw evidence remains at
`result/modelnet40_c/legacy_trajectory/full_all15_seed0_20261011T154536Z_handoff_20261011T171147Z.zip`;
report at its sibling `_ingested/independent_analysis/analysis.json`.
Local source revision is `a351e66ec58f08f3a2104e46cab2cb8b20cdf961`.

[Inference/next] Develop a single label-free prediction policy across all14,
using the same normal3/5/8 probes plus terminal. Per-corruption reporting is
diagnostic, not permission for inference to choose a rule from corruption names.
Assess the proposed parameter-free entropy-weighted mean against the existing
equal mean and terminal on this scope. The latest-clean-plus-terminal mean may
remain a generic control, but Background's weak early probes no longer justify
prioritizing it. Inspect confidence and temporal consistency across all14;
do not invent tau, thresholds or automatic entropy guarantees.

[Open/limits] Background's existing30-step outputs stay archived and reported
as a separate legacy-protocol result, along with the unchanged all15 result.
No rerun,10/30 horizon change, classifier/BN/token change or new policy
measurement occurred. Excluding Background from development does not establish
performance for unknown-corruption Background inputs at10 steps. A uniform
horizon for every incoming sample or a causal input-derived horizon remains a
separate deployment question. Any gain in this scope must be labelled macro14;
the original all15 research target is not silently relabelled. Policy freezing,
conditional seed1/2 and independent evaluation remain later stages.

## 2026-10-11 - Saved-logit entropy-weighted candidate evaluated

[Run/CPU] A strict analyzer evaluated the common10-step macro14 subset of the complete frozen-state seed0 full-data trace. Parameter-free weights are normalized `1-H`, with equal fallback when all entropies are maximal; no tau was fit. Entropy-weighted mean is65.7270% (+0.7670pp vs terminal64.9601%), while equal probability mean remains higher at65.7560% (+0.7959pp). Against the equal mean it fixes26 but breaks36 examples (net -10; -0.0289pp).

[Decision/Open] Do not advance entropy weighting as an accuracy winner. It is an interpretable control that is nearly tied with the equal mean on seed0. The equal mean remains the stronger development candidate, subject to conditional seed1/2 repeatability and an independent evaluation. Continue reporting per-corruption effects as diagnostics, with one shared rule; retain Background in the archived all15 legacy result but outside this macro14 focus. Full paired results: `..._ingested/macro14_policy_analysis_v2/`.

## 2026-10-11 - Next accuracy proposal: sampling consistency and input diversity

[Code/Run] Read-only diagnosis at `gsd-smooth-spectrum`, HEAD `a351e66ec58f08f3a2104e46cab2cb8b20cdf961`, from `result/modelnet40_c/legacy_trajectory/full_all15_seed0_20261011T154536Z_handoff_20261011T171147Z_ingested/macro14_policy_analysis_v2/paired_examples.csv`: group by (corruption,index), use fixed_probe_1/2/3 and terminal as four candidate predictions. Among34,552 examples,24,569 have at least one correct candidate (71.1073165% label-using selection oracle). The equal mean is wrong on1,850 examples with a correct candidate available. All four candidates unanimously predict the same wrong class on6,461 examples. The oracle is not a deployable policy or an averaging ceiling. These counts diagnose complementary and shared errors; they do not establish a reliable way to identify correct candidates. No new inference or policy measurement.

[Code] Nested `full_all15_seed0.zip` config has `trajectory.geometry_samples_per_batch=0`. The complete saved logits suffice for prediction aggregation, but these artifacts contain no saved decoded probe clouds for a full multi-view classifier reevaluation. New sampling views therefore require a new Colab probe run unless matching geometries are supplied from elsewhere.

[Inference/Proposal] First GPU candidate: evaluate two predeclared point-sampling views of each existing clean03/05/08 and terminal geometry, reusing each decode and the same10-step trajectory. Keep Point-MAE weights, BN, token algorithm and input point count fixed; change only external point sampling. Shared view seeds, RNG isolation and identical rules across macro14 are required. Compare existing four-prediction equal mean, eight-prediction equal mean, and a predeclared reliability selector over four view-averaged candidates using normalized H(mean probability) + normalized JS(two views), with equal coefficients and no tau/threshold fitting. This is a hypothesis, not an optimized or proven score. Agreement can be confidently wrong. Include terminal-only sampling votes with the same classifier-pass count and record total time including decode/FPS; equal pass counts alone do not make compute identical. Eight forwards versus the existing four add classification cost; diffusion steps need not increase. Avoid inferring a percentage runtime penalty before profiling.

[Inference/Alternative] Prepared-input prediction plus trajectory probability mean is another bounded candidate: one extra frozen-classifier prediction before diffusion may preserve evidence lost during reconstruction. Start with a predeclared equal combination as an exploratory control; do not fit a gate on evaluation labels or assume unprocessed input is reliable. Its premise is new error diversity, not another entropy temperature. Do not modify corruption-specific horizons.

[Paper/Scope] MEMO (NeurIPS2022, abstract and method overview, https://arxiv.org/abs/2110.09506) uses augmentation consistency through marginal-entropy optimization and adapts model parameters. Bahri et al. (WACV2025, abstract, https://arxiv.org/abs/2411.01116) use FPS/KNN sampling variation, TENT adaptation and weight averaging. They motivate testing sampling variation; neither validates our proposed frozen-classifier selector nor establishes its novelty or expected gain. Our proposal preserves weights and BN.

[Decision/Open] Recommend a bounded sampling-consistency pilot before larger entropy-weight searches. Primary comparison is versus macro14 equal mean65.755962%, with paired transitions, corruption rows and latency; also retain terminal64.960060%. Falsifier: consistency does not predict useful corrections, or gains vanish against equal view averaging / same-cost terminal votes. Seed0 is development; lock a candidate before conditional seeds1/2 and independent evaluation. A1-2pp total gain versus terminal is a target, not a forecast. No implementation or GPU launch is authorized by this recommendation alone.

## 2026-10-11 - Explicit attribution and two-view construction clarification

[User requirement/Paper] The user requires explicit attribution of the proposed sampling-consistency method. Primary inspiration is Ali Bahri et al., "Test-Time Adaptation in Point Clouds: Leveraging Sampling Variation with Weight Averaging", WACV2025, pp266-275; https://arxiv.org/html/2411.01116v2, sections3.1-3.2, equations1-3 and5. It varies FPS/KNN point-cloud representations, adapts normalization parameters with TENT and averages adapted weights. Our proposal borrows sampling diversity as a hypothesis while keeping classifier weights, BN and token algorithm frozen; it does not reproduce their adaptation/weight averaging.

[Paper] Second conceptual inspiration is Marvin Zhang, Sergey Levine and Chelsea Finn, "MEMO: Test Time Robustness via Adaptation and Augmentation", NeurIPS2022; https://arxiv.org/html/2110.09506v3, section3.1 equations1-3 and Algorithm1. MEMO minimizes entropy of the mean prediction across augmented views through parameter adaptation. Our proposal uses view predictions to score decoded trajectory candidates with a frozen classifier. Additional related precedent identified during this attribution review: Hendrycks et al., "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty", ICLR2020, https://arxiv.org/abs/1912.02781 and official implementation https://github.com/google-research/augmix. AugMix uses Jensen-Shannon consistency in training; it is background for the JS measurement, not evidence for our proposed inference selector.

[Inference/Open] The particular normalized H(mean)+normalized JS score and its use to select clean diffusion snapshots are proposed design choices, not equations copied from these papers. No claim of first use, proven novelty or demonstrated gain is supported by this review. Mean entropy already reflects disagreement; the extra JS term needs a separate ablation. Cite the two main inspirations explicitly, cite AugMix as related consistency work, and describe the combined frozen-classifier trajectory policy as an unvalidated synthesis.

[Code/Clarification] `eval_gsd_tta_v2.py` probe_preprocess rotates back, normalizes, then applies `misc.fps(points,1024)`; `utils_mate/misc.py` lines60-67 wraps the pointnet2 FPS operator. A/B are two potentially overlapping full-shape subsets from the same decoded cloud, each with1024 points, not disjoint halves or physical camera views. Proposal: A retains the original FPS sample; B uses a predeclared alternative FPS initialization, preserving coordinates, whole-cloud normalization and point count. Existing wrapper exposes no initial-point argument; merely changing a random seed is not established to change this kernel's output. A future implementation must explicitly construct/verify the alternative external sample and isolate its RNG effects, without altering the classifier's token algorithm. No implementation change or GPU run in this clarification.
