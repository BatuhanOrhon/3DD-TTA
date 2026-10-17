# Accuracy development after practical legacy reproduction - 2026-10-11

Evidence: **[User report]**, **[Run]**, **[Code]**, **[Paper]**,
**[Inference]**, **[Open]**. Repository: `gsd-smooth-spectrum`, inspected HEAD
`f670c07440533f01e10fd149fad2480d36f9194c`. This is a research proposal and
CPU artifact analysis, not an implemented adaptation method or GPU experiment.

## Decision and baseline

[User report/Run] The user accepts 64.570502% as practical reproduction of the
historical 64.5327% legacy result. Use the restored B70/M800 legacy method as
the operational baseline for accuracy development. Historical bitwise identity
remains undocumented, but recovering it is no longer the immediate prerequisite.
This supersedes earlier pending-reproduction priorities without deleting them.

[Run] Legacy: 23,904/37,020 correct; offdiag-only correction: 23,898/37,020.
The difference is -0.016207 pp at seed 0. Both arms and provenance were audited
in [the previous validation](../result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/analysis_ingested/validation.md).
The new target is approximately 65.57-66.57% macro15, requiring at least
371-741 additional correct classifications at this equal-size all15 coverage.
This objective is not a guarantee, a retrospective acceptance gate, or proof
that every corruption will improve by 1-2 pp.

[Inference] Prioritize a bounded classifier-side adaptation experiment on the
restored geometry, followed conditionally by sample-wise adaptation selection.
The strongest directly relevant literature candidate found is PG-SF token
purging. Keep the existing legacy configuration as the comparator. A numerical
bug correction and an accuracy-improving algorithm are separate questions.

## What has already been done, and what remains open

| Work | Existing evidence | Current decision |
|---|---|---|
| Legacy M800/B70/10-30 restoration | Full all15 seed0, 64.5705% | Practical reproduction accepted; no additional reproduction-only run |
| Offdiag isolation correction | Full all15 seed0, -0.0162 pp | Preserve corrected option; do not prioritize repeat seeds for an accuracy claim |
| Style routing/projection, P_PC | Phase6 all15/three seeds, +0.036917 pp vs its C_SCD | Completed; no new projection expansion justified by that result |
| Smooth beta/rho calibration | Completed screens and all15 comparisons in a different host | Do not repeat the old candidates as new work |
| Normal5 -> normal10 proposal of October7 | Superseded operationally by restored normal10/Background30 | Do not propose normal10 as a missing setting; its isolated causal effect was not established by restoration |
| Background step scan | Completed SCD-off unguided/spectral-only study | Does not answer strength adaptation in the restored SCD-on host |
| Spectral-off control in restored host | No validated matching B70/10-30 legacy control found | Add alongside a genuinely new candidate; older C_SCD is not interchangeable |
| Per-example strength/output selection | Earlier conceptual proposal, no implemented validated legacy experiment | Remains a plausible research question |
| Point-MAE token filtering / episodic BatchNorm | No experiment in supplied legacy results | New, testable candidate |

[Run] Failed Background spectral weights100/1000 remain failed, with nonfinite
gradients. Do not rerun them. M_max=1800 is a separate research branch and is
unchanged by this plan. No global-diffusion conclusion follows from style tests.

## New offline evidence: fixed rules on archived logits

[Run] Raw archive, unchanged:
`result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/20261010T114156946419Z_handoff_20261010T141536060023Z.zip`.
SHA-256 `534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968`;
size12,277,729 bytes. No new model outputs were generated.

The CPU analysis rechecked outer/full-arm CRCs, exact indices0..2467 in every
corruption, shared targets, finite logits, and prediction/argmax agreement.
It applied fixed temperature1 and equal weights, without fitting temperatures,
weights, corruption-specific rules, or thresholds. Labels enter scoring and the
explicit oracle calculation only. Selector ties fall back to legacy.

| Rule | Correct / 37,020 | Macro15 (%) | Delta vs legacy (pp) | Corrected / broken |
|---|---:|---:|---:|---:|
| Legacy | 23,904 | 64.570502 | 0 | 0 / 0 |
| Offdiag | 23,898 | 64.554295 | -0.016207 | 872 / 878 |
| Average probabilities, 50/50 | 24,007 | 64.848730 | +0.278228 | 496 / 393 |
| Average logits, 50/50 | 23,995 | 64.816316 | +0.245813 | 491 / 400 |
| Choose lower entropy | 23,989 | 64.800108 | +0.229606 | 477 / 392 |
| Choose larger maximum probability | 24,009 | 64.854133 | +0.283630 | 491 / 386 |
| Oracle: choose correct arm using labels | 24,776 | 66.925986 | +2.355484 | 872 / 0 |

[Inference] The oracle is an unattainable-by-this-rule diagnostic ceiling for
selecting between these two recorded predictions, not a deployable result or
promise of +2.36pp. Fixed label-free rules capture roughly0.23-0.28pp. Choosing
the largest observed number among them is posthoc development selection, not
independent evidence. No claim that offdiag itself causes useful diversity is
supported; a two-draw same-method ensemble would be needed to isolate that.

[Run] Probability averaging improves13/15 corruption rows. The exceptions are
Cutout -0.2431pp and Density_inc -0.0405pp; Background +0.6078pp. Its net103
extra correct is materially below the371 needed for a1pp macro15 gain.

[Run/Inference] Conditional95% percentile resampling interval for probability
averaging minus legacy is [+0.1162,+0.4349]pp. Method:20,000 paired index
bootstrap replicates, RNG20261011, resampling independently within each
corruption and averaging15 stratum means. One recorded prediction pair per
index; multinomial resampling of {-1,0,1} differences is equivalent for this
mean. Assumes exchangeability within corruption and treats corruption strata
independently. It neither pools cross-corruption object identities nor measures
seed variability or unknown cross-corruption dependence. Intervals are
conditional on these runs, exploratory, and not adjusted for multiple rules.
The previous note's lack of a seed interval does not prevent this narrower
conditional resampling calculation. Prepared-input/random-draw hashes are absent.

[Inference] Executing the ensemble on new data requires both reconstructions,
roughly two full inference costs (recorded68.98+69.21 minutes on A10080GB), so
it is a useful control and positive clue, not the preferred main method.
All details and every corruption are in
[selection_summary.json](../result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/analysis_20261011_selection/selection_summary.json)
and its neighboring CSV/script.

## Code diagnosis relevant to accuracy

[Code] `tta_gsd_v2.py` preserves the corrupted encoded local latent h0 using
both a selective Chamfer term and a spectral fidelity term. The latter uses a
fixed reference basis, first3 latent channels, raw eigenvalues, exp(-2*lambda),
rank cutoff800, and normalization by3*2048. It is a fidelity penalty, not a
learned semantic direction toward the correct class. A smaller loss need not
mean a more classifiable object. Both losses are summed across the batch;
raw loss magnitude alone cannot establish gradient dominance.

[Code] Noise depth is fixed by the normal10/Background30 rule; changing reverse
steps also changes starting noise level. Local and style updates both occur.
Actual update mapping is eta -> local latent, gamma -> style, despite reversed
descriptions in the docstring. Preserve both .01 in the first new comparison.
Classifier supervision, reliability selection, and token filtering are absent.

[Code] Our classifier is `Point_MAE`,64 groups of32 points, with CLS and max
token pooling and BatchNorm in both grouping encoder/head. The all-token
`classification_only(..., only_unmasked=False)` path is active. Its encoder
still creates an unused random mask: extra prediction passes must not change
later diffusion/interpolation random draws. Cache or isolate RNG state.

[Inference] Potentially useful new information comes from classifier features
and consistency, rather than another fixed penalty anchored solely to h0.
Low observed Rotation33.10%, LiDAR36.35%, and Occlusion41.45% make these
important diagnostic rows; they do not establish the cause or guarantee easy
recoverable information. Noise, deformation and missing data need not benefit
from the same adaptation strength.

## Literature and official implementation review

### Purge-Gate (ICCV2025): highest immediate compatibility

[Paper] [Paper, sections3.2-3.3 and Table2](https://arxiv.org/html/2509.09785v1):
PG-SF scores tokens using a CLS-derived reference and removes divergent tokens;
it selects purge length by prediction entropy. The reported ModelNet-C mean
is68.68%. PG-SP requires source statistics. Normalization adaptation is also
part of the method. The published number is not a prediction for our checkpoint,
batching or diffusion outputs. [Venue record](https://openaccess.thecvf.com/content/ICCV2025/html/Yazdanpanah_Purge-Gate_Backpropagation-Free_Test-Time_Adaptation_for_Point_Clouds_Classification_via_Token_ICCV_2025_paper.html).

[Code] Official GitHub inspected through the connector, pinned commit
`894d4b6372897986eb468038dc6ffbc93f89e38c` (2025-07-26):

- [models/Point_MAE.py, lines361-399](https://github.com/MosyMosy/Purge-Gate/blob/894d4b6372897986eb468038dc6ffbc93f89e38c/models/Point_MAE.py#L361):
  only the first transformer block performs PG-SF removal. It uses normalized
  token-plus-position Q/K projections, retains CLS, gathers matching token
  positions, and preserves order of retained indices. Later blocks receive
  the shortened sequence; output concatenates CLS and max-pooled patch features.
- [tools/tta_purge.py, lines239-295](https://github.com/MosyMosy/Purge-Gate/blob/894d4b6372897986eb468038dc6ffbc93f89e38c/tools/tta_purge.py#L239):
  random-start FPS, optional BatchNorm buffers set to None, one forward per
  purge size, per-example minimum-entropy output selection. Loader uses
  drop_last=True; a faithful benchmark comparison must account for counts.
- [commands.sh](https://github.com/MosyMosy/Purge-Gate/blob/894d4b6372897986eb468038dc6ffbc93f89e38c/commands.sh):
  published usage enables BN_reset. Parser defaults are batch32, one sampling
  view and purge sizes0,2,4,8,16,32.
- [ModelNet config](https://github.com/MosyMosy/Purge-Gate/blob/894d4b6372897986eb468038dc6ffbc93f89e38c/cfgs/tta_purge/tta_purge_modelnet.yaml):
  active model is PointTransformer, not our Point_MAE wrapper. Grouping uses
  PyTorch3D in this release, unlike our current operator. Do not transplant the
  runner, switch checkpoints or silently change FPS/normalization to obtain a
  nominally comparable score. Port the small inference operation to our encoder.

[Inference] This is a plausible adaptation, not a proven drop-in replacement.
Retain our checkpoint, native grouping, exact point count, FPS and all examples.
No parameter optimization is required for token-only inference. BatchNorm
adaptation changes classifier state/normalization semantics even when weights
remain frozen; it must be separately named and controlled. Test token-only
frozen-BN and the full BN-adapting variant. Occlusion/LiDAR may lose essential
evidence when tokens are removed; retain purge0 and inspect corrected/broken.

### Other recent directions and why they rank lower

| Primary source | Relevant idea | Transfer constraint / priority |
|---|---|---|
| [Point-Cache, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Point-Cache_Test-time_Dynamic_and_Hierarchical_Cache_for_Robust_and_Generalizable_CVPR_2025_paper.html) | Dynamic global/local test feature cache | Designed for multimodal open-vocabulary models; adapting it to Point-MAE adds pseudo-label, stream-order and cache-reset questions |
| [Uni-Adapter, AAAI2026](https://arxiv.org/abs/2511.15311) | Online class prototypes, graph smoothing and entropy-weighted prediction fusion | 3D vision-language model setting; a Point-MAE version would be an inspired extension |
| [BayesMM, CVPR2026](https://arxiv.org/abs/2603.22070) | Bayesian fusion of text priors and streaming visual class distributions | No text encoder in our classifier; not a direct module replacement |
| [Discriminator-guided adaptive diffusion, April2026 preprint](https://arxiv.org/html/2604.23636v1) | Sample-dependent forward noising depth with frozen base models | Image method; explicitly trains an additional discriminator. Motivation for adaptive strength, not an available training-free3D implementation |
| [FOCUS, August2025 preprint](https://arxiv.org/html/2508.14437v1) | Adaptive frequency conditioning for diffusion input adaptation | Image dense prediction, with an additionally trained27.5M frequency network; replacing fixed spectral fidelity is a research transfer |

[Inference] These papers motivate reliability and adaptation strength, but
their gains cannot be added to64.57%. The Point-Cache/Uni-Adapter/BayesMM review
here is a scope/abstract screen, not a replication or full algorithm audit.
The frozen Point-MAE constraint makes PG-SF the most bounded first experiment.

## Proposed next experiment: geometry x token gate x normalization

[Inference/Open] Primary hypothesis: filtering unreliable classifier tokens
after legacy reconstruction improves macro15 without rerunning diffusion for
every classifier option. This is the next experiment to implement, not yet run.

Use the same existing checkpoint and B70/severity5 protocol. First run a bounded
development pilot: **280 fixed indices per corruption, all15, seeds0/1/2**.
280 gives four full batches at70. Save the exact selected indices and run
configuration before execution; this report does not yet create a selection
manifest. Start with seed0 technical completion, then finish the predeclared
three-seed pilot without selecting by a favorable prefix.

| Input representation | Frozen BN, no gate | Frozen BN, PG-SF gate | Batch stats, no gate | Batch stats, PG-SF gate |
|---|---|---|---|---|
| Prepared input, no diffusion | Preprocessing/classifier control | Gate without diffusion | BN-only control | PG-style control without diffusion |
| Same legacy host, spectral weight0, SCD on | Missing matched SCD control | SCD + gate | SCD + BN | SCD + BN + gate |
| Legacy spectral1.17 + SCD1 | Baseline | Main token-only candidate | Legacy + BN | Full PG-style candidate |

This is twelve prediction outputs per sample, but only **two diffusion
trajectories**, shared across the four classifier policies. PG branches contain
six forward options0/2/4/8/16/32 and choose entropy minimum without labels.
Keep zero-purge output for a direct ablation. For batch-stat variants, use
current-batch statistics with dropout off, reset/restore buffers per variant,
and log batch composition. This is a batch-adaptive evaluation; later test
mixed corruptions/batch sizes if it is the selected method.

Primary comparisons: gate minus no-gate under identical BN; BN minus frozen
under no-gate; combined candidate minus legacy; combined candidate minus
prepared-input PG; and spectral-on minus spectral-off under the same classifier
policy. These answer whether gains arise from the new classifier operation,
normalization, diffusion, or spectral guidance. Existing Phase6 C_SCD cannot
replace this host-specific control.

Implementation requirements before any Colab execution:

1. Opt-in classifier path, with purge0/frozen-BN numerical parity against the
   existing classifier. Preserve same checkpoint loading and source identities.
   A future implementation needs explicit verification; no tests were added here.
2. Prepare interpolation/encoded h0/style/noise once per block and share them
   between the two diffusion arms. Hash actual components and replay RNG states.
   Preserve original batch order and all15 index coverage. Extra classifier
   probes must not advance the main RNG stream.
3. Cache final reconstructed2048-point geometry and final1024-point classifier
   input, plus logits for each policy, so classifier experiments do not repeat
   expensive diffusion. Never reuse features across frozen/batch-BN encoders
   when the features themselves depend on the normalization setting.
4. Existing ZIPs contain logits, not reconstructed point clouds or intermediate
   latents. They support the offline ensemble analysis, but cannot support PG
   evaluation. New pilot data capture is necessary for this new question; it is
   not another request to re-establish64.53%.
5. Record all corruption/seed accuracy, paired corrected/broken counts,
   macro15/macro14/Background, chosen purge size, confidence/calibration
   diagnostics, latency, allocated/reserved memory, sources/assets/native hashes.
   Do not infer wall time from attention FLOPs or literature speedup claims.

[Inference] Two pilot diffusion arms at280/2468 of full coverage correspond
roughly to16 minutes/seed by observed runtime scaling, or47 minutes for three
seeds, **before** extra classifier passes, preparation and packaging. This is
an extrapolation, not a measured run budget. Sequential inference should avoid
duplicating the60.8GiB live diffusion footprint, but peak memory must be measured.

Decision logic: if token-only helps at fixed normalization, pursue filtering;
if only BN helps, report BN adaptation as the source of improvement; if PG on
prepared input is as good as PG+legacy, do not attribute the gain to diffusion.
If spectral-off matches spectral-on, do not claim a GSD contribution. A null
pilot does not trigger an unbounded purge/BN/weight search. Lock a candidate
before full2468x15x3 evaluation. Freeze practical accuracy/latency decision
criteria before execution; no new numeric acceptance threshold is invented here.

## Conditional second method: sample-wise output reliability

[Inference/Open] The archived ensemble result supports a modest predictive
complementarity hypothesis. It does not establish the best way to choose
diffusion strength. After the first pilot, test a parameter-free selector over
prepared-input, intermediate predicted-clean and final outputs from one legacy
trajectory. Keep the terminal legacy output as an explicit candidate.

A concrete starting rule to predeclare is: use two fixed sampling views per
candidate, compute mean probability p_bar and
`score = H(p_bar) + JS(p_view1, p_view2)`, and choose minimum score. Equal
weights define one bounded proposal, not a fitted optimum. No target labels or
corruption names enter selection. Entropy-only selection and equal probability
averaging are required controls to establish value from consistency. Also
report the nondeployable oracle ceiling as a diagnostic, never as accuracy.

For an intermediate diffusion snapshot, decode the model's predicted clean
latent with its contemporaneous style. Decoding noisy h_t is not the same
operation. Capture at a predeclared halfway step and at the terminal output;
this is output selection on one trajectory, not an independent shorter-noise
run and not evidence that changing the starting time has been optimized.
Saved geometry permits classifier sampling probes without repeating diffusion.
Additional decode/classifier costs remain and must be compared with simple
voting at the same compute budget. This selector is a hypothesis, not a proven
contribution; confident errors and sparse-shape information loss may defeat it.

If unreliable anchoring remains the measured problem, a later spectral method
could weight reference fidelity by geometric reliability. That requires a
well-defined PSD loss and a validated reliability signal; changing bases must
respect sign/degenerate-subspace ambiguity. Dynamic graphs, global diffusion,
large coefficient scans and simultaneous step/style/graph changes rank lower
until a diagnostic identifies their benefit.

## Evidence limits and paper claim

[Inference] A1-2pp repeated improvement is a useful empirical target. A defensible
method contribution additionally needs a specific mechanism, equal-cost and
matched controls, ablations, and evaluation outside the already used development
set. Combining two published methods alone does not establish methodological
novelty. PG is also a relevant modern comparator even if its integration fails.

All ModelNet40-C results have already informed development. New seeds and
locked full coverage assess stability; they do not make it an unseen benchmark.
Reserve a separate dataset/protocol for generalization evidence and report its
own supported checkpoints. For repeated-seed within-corruption intervals,
resample original indices as clusters retaining all seeds; report seed SD
separately. Without proven cross-corruption correspondence, do not pool all
corruption records as independent objects or treat timestep rows as examples.

## Work performed and files

- Read knowledge/history and current inference/classifier source, and official
  primary papers. GitHub reads were pinned to the official PG commit above.
- CPU-only command:
  `python result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/analysis_20261011_selection/analyze_selection.py`.
  Generated `selection_summary.json` and `selection_per_corruption.csv` beside
  the script. SHA-256 was verified before and after; raw ZIP was untouched.
- Other read-only commands: `git status --short`, `git rev-parse HEAD`, focused
  `rg` searches and `Get-Content` source/knowledge reads. No model imports,
  local GPU inference, new model tests, stage, commit or push.
- Research record additions: this report, README headline, findings log,
  open questions and thesis-scope practical-reproduction decision. Existing
  user changes, historical failures and M1800 notes are preserved.


## Later 2026-10-11 user proposal - trajectory selection becomes the research focus

[User report/Inference/Open] The user specifically proposes evaluating clean
point-cloud predictions at multiple diffusion steps, minimum-entropy selection
or prediction averaging, and, if useful, early stopping and GSD/SCD removal
based on measured accuracy/runtime. The earlier PG-first ordering in this note
was an assistant recommendation and is superseded as the immediate research
emphasis; PG remains an advisor agenda/comparator option. No requirement to
implement PG before investigating trajectories applies.

See [the detailed proposal and realistic assessment](gsd_trajectory_selection_20261011.md)
and [advisor discussion log](advisor_meeting_log.md). Proposed first experiment
uses sparse clean-latent probes and the unchanged terminal output, with frozen
Point-MAE including BN. Confidence, temporal stability and correctness are
separate. Full-trajectory minimum entropy cannot itself save diffusion steps;
causal prefix stopping is a later policy. No new result or launch occurred.


## Session handoff and online clarification - later 2026-10-11

[User report/Inference/Open] User selects the last successful legacy64.5705%
B70/M800 run as the development base and requests a fresh-session handoff.
Implement opt-in clean trajectory observation/analyzer first; no PG prerequisite.
OnlineB1 stopping omits remaining diffusion work directly; the large-batch
waiting caveat must not be generalized. Evaluate net B1 latency including
probes against B1 full-horizon, separately from B70 throughput/accuracy.
[Code, knowledge, artifacts and implementation scope](gsd_trajectory_session_handoff_20261011.md).
No new model implementation, result, stage/commit/push in this handoff update.
