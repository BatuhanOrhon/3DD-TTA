## Further loss/parameter comparison requested — 2026-10-25

[User report/Decision/Open] User reopens Agent B after the bounded fixed study,
requesting different GSD parameters and several loss calculations to strengthen
the comparison with Chamfer/original3DD. The scientific question remains
falsifiable; no universal superiority conclusion is selected in advance.
[Registered campaign](guidance_component_sensitivity_study_20261025.md): equal6
configs perC/C+S/S family, label-free scale calibration, original3DD as a distinct
whole-policy comparator, then new object-disjoint140IDs/dataset/seeds0/1/2.
CPU preparation is accepted; the >60min whole GPU campaign awaits explicit
approval, not inherited from the prior short phases. Agent A remains untouched.

# Thesis Scope and Research Questions

## Parallel component-study authorization — 2026-10-25

[User report/Decision] A second future agent may investigate Chamfer-only,
spectral-only, combined and unguided controls and bounded method/parameter
development across ModelNet40-C/ScanObjectNN-C and four frozen heads. This
extends the prior locked-readout-only next scope without changing Agent A's
assignment. User requests a derived branch and separate worktree for each
agent, clearly named owned Colab sessions and no interference with peer work.
[Ownership contract](parallel_colab_agent_contract_20261025.md),
[Agent B scientific/compute gates](guidance_component_study_handoff_20261025.md).
This turn prepares prompts only. Results must support whichever arm performs
best under matched/development-confirmation controls; no result is preselected.

## Next-session scope amendment — 2026-10-25

[User report/Open] The next selected task is ModelNet40-C GSD-inspired
minimum-entropy/equal-probability-mean evaluation for DGCNN, CurveNet and
PointNeXt, with a same-run terminal control for each. Earlier requested
terminal/Scan scopes remain complete. This turn only audits memory and
prepares the [next-agent handoff](modelnet_multibackbone_trajectory_handoff_20261025.md);
the40-class observation extension and GPU work are pending. Preserve the
fixed policy, accepted alternative checkpoints and no-redundant-runs preference.
Logit averaging, tuning and extra seeds are not added to this next GPU scope.

## Current empirical amendment — 2026-10-19

[User report/Run] The requested multibackbone study and GSD-only four-head
Scan readout study are complete. All Colab runtimes and task background workers
are closed; fresh20:44UTC inventory confirms that state. The older empirical
snapshot and candidate plans below describe their original dates and are not
the current result inventory. See the
[final synthesis](gsd_multibackbone_final_findings_20261019.md).

[Inference/Open] There is no universal terminal GSD improvement over original:
GSD loses all six whole-policy comparisons. Scan probability mean exceeds
original for CurveNet and DGCNN but remains below it for PointNeXt. Both
readouts improve all four own terminals for seed0 only. Additional seeds,
same-flow spectral-off attribution and exact-paper assets remain open; the
new backbone settings are transferred legacy values, not proven optima.
This documentation/closure task does not launch those follow-up studies.

## Local/style composition planning request - 2026-10-04

[User report/Decision] The user requested assessment plus an implementation
plan and Colab scenarios for a different agent: local-SCD/style-spectral,
reverse routing, style-only PCGrad and conditional alternatives. This extends
the earlier research scope to planning composition; the September GSD-only
exclusion does not forbid this requested plan. No implementation or GPU run
was requested in the planning session. Global diffusion is a later gated
study with frozen pretrained weights, not model training. Read the
[design](gsd_block_routing_design_20261004.md) and
[handoff](gsd_block_routing_implementation_plan_20261004.md).

## Result reuse preference - 2026-10-03

[User report/Decision] Do not rerun tests whose results are already available.
Before proposing execution, inspect knowledge and result artifacts, identify
existing condition/corruption/seed coverage, and specify what is genuinely new.
Missing per-example diagnostics or a different launcher do not automatically
justify repeating a complete accuracy evaluation. Preserve historical pairing
limits rather than relabeling archived comparisons as common-draw experiments.
Current inventory: [GSD/SCD coverage audit](gsd_existing_runs_audit_20261003.md).

## GSD-only development authorization - 2026-09-23

[User report] The current task authorizes a new `gsd-development` branch
from the latest `baseline-repro-clean`, including GSD implementation,
unit/protocol tests, knowledge records and Colab scripts. It supersedes
the historical GSD deferral and default no-new-tests preference below.
[Code] The selected method is a static latent XYZ fidelity regularizer;
see [design](gsd_integration_20260922.md). PxP, gradient projection and
combined variants are outside this task. Local GPU experiments are prohibited.

## Working title

**Graph-Spectral and Conflict-Aware Test-Time Adaptation of Corrupted 3D Point Clouds with Denoising Diffusion Models**

This is a working title, not a finalized thesis claim.

## Problem statement

Point-cloud classifiers trained on clean source data degrade under unseen corruptions. Source data may be unavailable at deployment, and retraining or changing the classifier may be undesirable. The thesis investigates input-level, training-free adaptation that uses a pretrained 3D diffusion prior to transform each corrupted point cloud into a representation that the frozen source classifier can classify more reliably.

The base method, 3DD-TTA, guides LION's local latent denoising with a robust geometric reconstruction objective. The fork adds graph-spectral guidance intended to preserve global/mesoscopic structure and explores gradient-conflict handling when geometric and spectral objectives disagree.

## Primary research question

Can graph-spectral structural guidance, alone or with conflict-aware gradient composition, improve the corruption robustness of 3DD-TTA on ModelNet40-C while retaining its training-free, frozen-classifier setting?

## Secondary questions

1. Which part of the current accuracy gap is caused by environment, data/checkpoint identity, preprocessing, stochasticity, scheduler configuration, or metric aggregation?
2. Is a static spectral basis computed from the corrupted local latent useful, or must the basis be recomputed during denoising?
3. Do low-, mid-, or high-frequency constraints carry useful signal in LION's local latent space?
4. Does a graph built on physical coordinates define a meaningful basis for LION latent features when point correspondence is retained?
5. When SCD and spectral gradients conflict, does one-way or symmetric projection improve accuracy, stability, or both?
6. Does adapting/denoising the global latent help, and what is the correct LION decoder/style contract?

## Scope boundaries

### Implementation policy agreed on 2026-09-12

Start the new branch directly from main, not a GSD/PxP development branch. Future legacy branches are references for ideas and code inspection, not implementations to copy wholesale. Prefer simpler, smaller code; independently test inherited assumptions and repair confirmed omissions/errors before using a component. Introduce one reviewed increment at a time and preserve original baseline math as the comparator. GSD work is explicitly deferred while the baseline/artifact/dropout controls are established.

### In scope

Research workflow preference (2026-09-12): do not add unit-test files by default. Verify proportionately through syntax, baseline/protocol checks and controlled archived Colab experiments; local checks are not accuracy evidence.

- ModelNet40-C, severity 5, initially with the repository's Point-MAE classifier and LION checkpoints.
- Frozen classifier and training-free test-time input adaptation unless explicitly declared otherwise.
- Original 3DD-TTA reproduction and controlled ablations.
- Static/dynamic latent spectral guidance, physical-basis experiments, dual global/local denoising, and PxP-inspired gradient projection.
- Accuracy, per-corruption behavior, variance, runtime, memory, and mechanism diagnostics.

### Out of scope unless separately approved

- Claiming a reproduction of full GSDTTA while omitting its learned spectral shift and model-adaptation stage.
- Comparing numbers across different classifiers/backbones as if protocols were identical.
- Hyperparameter search on the final test set followed by an unbiased generalization claim.
- Treating a single stochastic run or a partial 25-sample grid search as a definitive improvement.
- Rewriting the classifier or diffusion model training pipeline before the baseline gap is characterized.

## Candidate contributions, not yet established

- A graph-spectral regularizer applied inside a hierarchical 3D latent diffusion TTA trajectory.
- An analysis of static versus dynamic graph bases in a changing latent space.
- Conflict-aware composition of SCD and spectral guidance gradients for 3D diffusion TTA.
- A reproducibility and diagnostic study of 3DD-TTA on ModelNet40-C.

Each item becomes a thesis contribution only after controlled evidence, ablations, statistical reporting, and comparison against a verified baseline.

## Success criteria

Minimum scientific success is a defensible explanation of the reproduction gap and a reproducible baseline. A method improvement should additionally show:

- repeated-seed improvement over the matched baseline;
- per-corruption results, not only a mean;
- uncertainty or variance reporting;
- a mechanistic diagnostic consistent with the proposed explanation;
- compute/memory costs;
- no hidden change in data, checkpoint, classifier, preprocessing, or evaluation set.

## Current empirical status

- **[User report]** Original code: approximately 63% mean accuracy.
- **[User report]** Best current GSD variant: approximately 63.5%.
- **[Open]** Exact commands, seeds, environment, per-corruption values, checkpoint hashes, and raw logs have not yet been archived under `result/`.

Therefore no gain is yet treated as confirmed.


## 2026-10-07 ? User priority: all15 accuracy improvement

[User decision] The immediate priority is a +1 to +2 absolute percentage-point
improvement in equal-weight all15 accuracy, as the intended empirical basis
for a paper contribution. Explanatory style ablations alone do not satisfy
this objective. This is a prospective research objective, not a retroactive
Phase6 acceptance threshold, guaranteed gain, or publication guarantee.

[Inference/Proposal] Deprioritize further P_PC/style-composition expansion
after the +0.036917pp all15 result. Use existing evidence to select the next
performance hypothesis; avoid repeating completed conditions merely to fill
mechanism tables. Keep shared inference protocol, seed variance, compute cost
and candidate selection on development data explicit. Local GPU inference
remains prohibited; no new Colab run is authorized by this priority note.

## 2026-10-11 - Practical legacy reproduction accepted

[User report/Run] The user accepts the restored legacy64.5705% result as
practical reproduction of historical64.5327%. Use it as the operational
accuracy-development baseline; exact historical bitwise identity remains
undocumented and is no longer the immediate prerequisite. This supersedes
earlier reproduction-first status statements without removing their evidence.

[Code/Paper/Inference/Open] The next proposed experiment is PG-SF-inspired
token filtering on the existing classifier/legacy outputs, with separate
BatchNorm and same-host spectral-off controls. Frozen weights with adapted
BatchNorm is a distinct protocol from fully frozen classifier state and must
be labelled. A later sample-wise output selector is conditional. See
[the dated research plan](gsd_accuracy_research_plan_20261011.md). These are
unimplemented hypotheses; the requested1-2pp gain and novelty remain open.


## Later 2026-10-11 - User-directed trajectory research and advisor record

[User report/Inference/Open] The user requests an advisor discussion log and
prioritizes examination of clean intermediate diffusion outputs, low-entropy
selection, temporal prediction averaging and conditional early stopping.
Potential GSD/SCD deletion is conditional on measured accuracy and runtime.
The prior PG-first proposal is not a mandatory implementation sequence.
This candidate preserves the classifier weights, BN and token computation;
classifier output only scores reconstructed inputs. It is distinct from PG's
classifier-internal adaptation. Neither novelty nor an accuracy/speed gain is
established. See [design](gsd_trajectory_selection_20261011.md) and
[advisor log](advisor_meeting_log.md); no advisor meeting/approval is invented.


## 2026-10-11 - Common-horizon development focus (user decision)

[User decision/Run/Open] Prioritize the14 non-Background corruptions sharing
10 reverse steps. The user points out that selecting30 steps specifically for
Background assumes corruption identity at inference; Background repair is no
longer the next development objective or candidate-selection requirement.
Preserve and report the existing Background/all15 legacy results separately.
On the existing full-data seed0 trace, macro14 terminal/minimum entropy/equal
mean are64.960060/65.700972/65.755962% (+0.740912/+0.795902pp vs terminal).
Use one shared label-free policy across14; per-corruption reporting must not
become corruption-specific inference routing. These are macro14 development
results, not all15 improvements or demonstrated unknown-corruption deployment.
No horizon/model/code/raw-artifact change or new inference. Entropy-weighted
averaging remains a proposed offline candidate, with no tau/threshold selected.
See [updated decision and limits](gsd_trajectory_selection_20261011.md).
