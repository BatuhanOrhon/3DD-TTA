# Diffusion trajectory readouts: provenance and prior-work attribution

Date: 2026-10-25. Evidence: [User report], [Code], [Paper], [Inference], [Open].
Review base: `gsd-multibackbone-study@ee8d218a923259ac8695e94df5aabd492fd242e2`.
Scope: literature/provenance review only; no new numerical experiment or Colab
allocation. Existing results and immutable raw artifacts are unchanged.

## Question and local recommendation history

[User report] The user asks whether probability averaging/minimum-entropy
selection was taken from an existing work and requires attribution wherever
the diffusion-step prediction/aggregation mechanism has prior precedent.

[Code/User report] The dated2026-10-11
[trajectory proposal](gsd_trajectory_selection_20261011.md#user-proposal-and-priority)
and [advisor agenda G-002](advisor_meeting_log.md) record evaluating intermediate
clean point clouds, selecting low-entropy predictions or averaging them as the
user's proposal. The assistant formalized the frozen-head candidate protocol.
These records do not identify a particular paper as the direct implementation
source. They do not establish first invention or absence of earlier literature.

[Code/Paper] Earlier notes explicitly discussed Purge-Gate for entropy-based
candidate selection and MEMO/SVWA/AugMix for the later sampling-consistency
extension. Those references are not an adequate review of diffusion-timestep
ensembling itself. This review identifies relevant prior work retrospectively;
do not rewrite the2026-10-11 history to claim these newly found papers were
the original inspiration or that their published implementations were copied.

## What the current implementation actually does

[Code] `scripts/analyze_legacy_trajectory.py:policies`, reused by
`scripts/analyze_scan_trajectory.py`, computes probabilities from each candidate's
logits, averages probabilities for `probability_mean`, and takes the candidate
with smallest predictive entropy for `minimum_entropy`. Deterministic ties
prefer terminal, then the earliest clean probe. Neither rule adapts classifier
weights, changes BN statistics or backpropagates the classification entropy.

For fixed candidate clouds X_j and the frozen classifier f:

`p_j = softmax(f(X_j)); p_mean = (1/K) sum_j p_j; prediction = argmax p_mean`.

For selection: `j* = argmin_j H(p_j); prediction = argmax p_j*`.

Normalizing H by log(class count) does not change the within-example argmin.
The present four candidates are three decoded predicted-clean latent probes
plus a distinct ordinary terminal output. These are sparse checkpoints of one
guided reverse trajectory, not all timesteps, independent full trajectories,
averaged coordinates, model-weight averaging or entropy minimization by updates.
Probe decoding, preprocessing and classifier passes have real additional cost.

## Verified primary references and precise differences

### 1. Rosnati, Roschewitz and Glocker — ML4H 2023

[Paper] **Robust semi-supervised segmentation with timestep ensembling diffusion
models**, PMLR225, pp512–527.
[Proceedings](https://proceedings.mlr.press/v225/rosnati23a.html);
[PDF](https://proceedings.mlr.press/v225/rosnati23a/rosnati23a.pdf), section4,
equations8–9, printed p516 (PDF page5).
Their shared pixel predictor consumes diffusion representations at several
timesteps; equation9 aggregates the resulting maps with an arithmetic mean.
Training also uses the selected representations. This directly precedes the
general timestep-prediction ensembling idea, but uses medical segmentation
features rather than decoded point clouds with existing frozen source heads.
Do not claim their equation is our exact probability implementation.

### 2. Wang and colleagues — 2025 preprint

[Paper] **Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language
Models**, arXiv2508.09138; reviewed v3,06Oct2025.
[Primary text](https://arxiv.org/html/2508.09138v3), section4.1, equation4.
They combine intermediate answers from one denoising trajectory through
semantic-answer voting with constant or timestep-dependent weights. This is
particularly close to reusing one trajectory, but voting over discrete answers
differs from averaging classifier probability vectors over decoded geometry.
Their temporal semantic entropy and post-training method are not our selector.

### 3. Li and colleagues — NeurIPS 2025

[Paper] **Understanding Representation Dynamics of Diffusion Models via
Low-Dimensional Modeling**.
[Conference PDF](https://papers.nips.cc/paper_files/paper/2025/file/9a3b1949b6da9b8facfdc28e1b4722ed-Paper-Conference.pdf),
appendixB.4, printed p28 (PDF page28).
They extract representations at five noise levels, train linear probing heads
and average their predicted logits for classification. This supplies a vision
classification precedent for combining timestep-specific predictions, not our
existing-head probability mean on a guided restoration trajectory. Logit mean
and mean of softmax probabilities are different inference rules.

### 4. Yazdanpanah and colleagues — ICCV 2025

[Paper] **Purge-Gate: Backpropagation-Free Test-Time Adaptation for Point Clouds
Classification via Token Purging**.
[Primary text](https://arxiv.org/html/2509.09785v1), section3.3.1,
equations20–21. It evaluates several token-purging lengths and selects the
candidate prediction with minimum entropy. This is a particularly relevant
point-cloud precedent for our selection criterion; its candidates are purging
settings, not diffusion checkpoints. Its normalization handling is also distinct
from our fully frozen-head protocol. Cite it as related entropy-based selection,
without claiming implementation or evaluation of Purge-Gate in this study.

### Additional context, not exact sources for the tested readout

- [Paper] [DiffusionDet](https://arxiv.org/html/2211.09788), Chen et al.,
  ICCV2023: Figure3 and appendix discussion around Table8 combine detections
  from sampling steps using NMS; not class-probability averaging.
- [Paper] [DDA](https://arxiv.org/html/2207.03442), Gao et al., CVPR2023,
  section3.3: self-ensemble before/after adaptation; not sparse intermediate
  checkpoints along one trajectory.
- [Paper] [MEMO](https://arxiv.org/html/2110.09506v3), Zhang, Levine and Finn,
  NeurIPS2022, section3.1: augmentation-marginal entropy with parameter
  adaptation; not frozen-head timestep output selection.

[Open] A search result for a2026 TopoDiff submission describes averaging DDIM
logits, but the full primary PDF could not be fetched through the web tool.
It is not used as a verified citation or an accepted-publication claim here.
Other search hits are not evidence that the identical four-head point-cloud
pipeline has or has not been published. The review is bounded, not exhaustive.

## Citation decision and permissible thesis wording

[Inference/Decision] Cite Rosnati et al. for timestep-prediction aggregation,
Wang et al. for a single trajectory's intermediate-output voting, Li et al. for
diffusion-timestep prediction ensembles in vision classification, and
Yazdanpanah et al. for entropy-based candidate selection in point-cloud TTA.
These are related precedents identified by this review, not retroactive claims
about where the original local proposal came from.

Suggested manuscript wording (adapt citation keys to the thesis bibliography):

> Prediction aggregation across diffusion timesteps has precedents in
> segmentation, visual representation learning and diffusion language models
> (Rosnati et al.,2023; Li et al.,2025; Wang et al.,2025). We evaluate equal
> probability averaging over sparse decoded clean estimates and the terminal
> output of a GSD-inspired latent spectral guidance trajectory, keeping the
> pretrained point-cloud classifier fixed. We also evaluate an entropy-based
> candidate selector, related to test-time selection in Purge-Gate
> (Yazdanpanah et al.,2025), with diffusion checkpoints as the candidates.

This paragraph is our synthesis, not a quotation from any paper. It describes
an evaluated adaptation of established ideas. Do not call temporal averaging,
minimum-entropy selection, or the generic strategy of using intermediate
diffusion predictions a new invention. Whether the particular point-cloud
combination supports a distinct contribution remains open and needs broader
prior-work comparison, appropriate controls and replication.

[Inference/Open] Citation is appropriate even if the local idea was arrived at
independently. Conversely, similar literature does not show code copying.
Higher accuracy does not establish originality or remove the single-seed and
whole-policy confounds. The existing missing same-flow spectral-off control
and seed replication remain open; no new experiment is authorized by this review.

## Retrieval and audit evidence

[Code/Paper] Seven primary documents (two PDFs, five arXiv HTML pages) were
downloaded successfully at2026-10-25 10:53UTC. Exact URLs, resolved URLs,
byte sizes and SHA-256 identities are in
`result/preparation/trajectory_attribution_review_20261025T105306Z/sources.json`.
Local HTML text extracts support section inspection. No dependency was
installed when a local PDF text extractor was unavailable; PDF equations and
section details were checked through the web tool's primary PDF text instead.
This is local literature evidence, not an inference result bundle or a new
Drive copy. No runtime was allocated or queried during this review.

Sealed local evidence ZIP:
`result/preparation/trajectory_attribution_review_20261025T105306Z.zip`,
8897535bytes,14 members, SHA-256
`4c5b8d2bd0b79e0253a1c3adb3057519d7bde0cf91bdb55eea67cc48c743fec3`.
All seven downloaded-source identities, ZIP CRC and member hashes pass.
The bundled review-note snapshot precedes this seal-identity paragraph;
subsequent knowledge changes are preserved separately in Git.

Falsifiers/limits: an earlier project source linking a specific paper would
revise recommendation provenance; a method-level match in further literature
would narrow contribution claims. Neither uncertainty changes accepted raw
accuracy values. This note supplements the
[final findings](gsd_multibackbone_final_findings_20261019.md), without changing
their results or the recorded runtime closure.
