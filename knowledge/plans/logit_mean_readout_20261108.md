**2026-11-10 status:** CPU implementation/tests complete, owned High-RAM A10080 setup awaiting Drive consent. Latest [execution record](../terminal_logit_pilot_study_20261108.md) supersedes historical unsupported-code statements; GPU smoke/pilot and results remain pending. Full-data expansion is outside this task.

# Logit Mean versus Probability Mean — registered 2026-11-08

[User report/Open] **Status: PLANNED / NOT_RUN.** The user asks whether
averaging pre-softmax class logits across our fixed diffusion readouts gives
higher accuracy than averaging probabilities. This refines the existing
[25 October question](../open_questions.md#2026-10-25--logit-mean-versus-probability-mean),
now anchored to the promoted [proximal N8+PM baseline](../current_baseline.md).
No readout implementation, offline result, baseline replacement or new Colab
allocation is claimed by this registration.

## Motivation and attribution

[Paper] Li et al., *Understanding Representation Dynamics of Diffusion Models
via Low-Dimensional Modeling*, NeurIPS2025. The explicitly checked
[arXiv v3, appendix B.4 and appendix D](https://arxiv.org/html/2502.05743v3#A2.SS4)
ensembles output logits from five independently trained timestep-specific
linear probes. [v1 section4.1 and appendixA.3](https://arxiv.org/html/2502.05743v1#S4.SS1)
also describes this procedure. Version identifiers matter: the v4 HTML does
not expose that same ensemble subsection. Cite the checked version and
section rather than assuming every revision has identical content.

[Inference] Our proposed adaptation uses one existing frozen 3D classifier
on four decoded reconstructions of one guided trajectory. It does not train
timestep probes or average diffusion hidden features. Generic logit ensembling
has prior art; a positive outcome would be evidence for this readout choice
in our setting, not first invention of logit averaging.

## Frozen hypothesis and comparison

Let `l_j` be the **C-dimensional class-score vector immediately before
softmax**, after the classifier's final class projection, at each candidate
`j in {clean_02, clean_04, clean_06, terminal}`. This is not an arbitrary
penultimate hidden embedding. Use equal weights and temperature1:

```text
PM: p_PM = (1/4) sum_j softmax(l_j)
LM: p_LM = softmax((1/4) sum_j l_j)
prediction = argmax_c p[c]
```

[Inference/Open] Hypothesis: LM yields a positive paired all15 accuracy
difference versus PM on identical candidate logits. Softmax and averaging
do not commute. LM is also the normalized geometric mean of the per-step
softmax probabilities; it can reward agreement but can overemphasize a
confident wrong timestep. Thus an accuracy gain is not guaranteed. Softmax
does not change the argmax of the averaged logits; retain probabilities for
calibration diagnostics.

## Ordered future steps

1. **Audit/reuse archives first, CPU only.** Primary scope is cheap/proximal
   N8, ModelNet40-C, Point-MAE, all2468 original IDs × all15 corruptions,
   severity5/seed0/B128. Start from the immutable raw inputs referenced by
   [the common-horizon report](../lion_guidance_uniform_horizon_results_20261108.md)
   and `evidence/lion_guidance_speed_20261107/uniform_per_corruption_detail_20261108.json`.
   `scripts/run_lion_guidance_speed.py` stores full `logits` shaped B×4×40
   for PM/cheap arms, plus IDs/labels and saved PM probabilities/predictions.
   The shared host stores `trajectory_logits` with candidate names. Verify
   actual coverage/order/source/asset identity, finite values, labels and no
   duplicate IDs before analysis. Keep special-N35 Background archives out of
   the common-N8 primary comparison; retain their history separately.
2. **Compute two reductions from exactly the same tensors.** Recount saved PM
   and terminal predictions first. Match the archived GPU float32 PM policy;
   if stable CPU float64 recomputation changes near-tie predictions, quantify
   and disclose those cases rather than attributing them to LM. Compare PM
   and LM at the same declared precision and deterministic tie rule. No new
   denoising, decoder/classifier forward, resampling, guidance or checkpoint
   selection. Missing logits are missing coverage, not permission to infer
   them from predictions or silently rerun a full benchmark.
3. **Report matched outcomes.** Record macro-all15 accuracy, each corruption's
   accuracy/counts/delta, macro14 separately, corrected/harmed/net counts,
   candidate disagreement and terminal reference. Cluster paired bootstrap
   by original object across corruptions (10000 draws, seed20261108), report
   the95%CI in percentage points and disclose unadjusted per-corruption CIs.
   Save per-object predictions/probabilities, raw input hashes, source commit,
   full config, command/log and derived tables in a new `result/.../analysis/`
   directory; never rewrite the accepted source CSV/ZIPs.
4. **Inspect scale and confidence without tuning.** Report per-step centered
   logit norms/ranges, entropy, margins and where LM/PM disagree; optionally
   NLL/Brier using labels only for evaluation. Centering is diagnostic only;
   no per-step scaling, learned weights, temperature search or corruption
   routing enters the registered equal-weight LM policy. Each such change
   would require a separately labelled development experiment.
5. **Interpret before expansion.** Positive mean with a95%CI including zero is
   inconclusive. A lower CI above zero supports a conditional advantage on
   the exposed cohort, not unseen generalization. A tie/loss is retained and
   does not replace PM. If promising, later test the frozen policy on fresh
   seed1/2 traces and matched additional heads/dataset; reuse complete old
   logits where possible and label their historical hosts separately. No
   new GPU phase is started by this plan.

[Open] Diffusion/decode/classifier counts are unchanged, so major pipeline
speedup is not predicted. Measure reduction overhead separately if needed;
CPU archive processing time is not GPU inference latency. This readout
ablation is independent of the
[future PM-off terminal acceleration experiment](proximal_terminal_without_pm_20261108.md).

## Execution checklist

- [ ] Audit complete common-N8 cheap candidate logits and archived PM parity.
- [ ] Implement/review a minimal offline paired reducer and verify it on
  synthetic cases where PM=LM and where their predictions differ.
- [ ] Produce all15/corruption-level accuracy, uncertainty and scale diagnostics.
- [ ] Record positive/null/negative result and decide on separate frozen confirmation.

All boxes remain open; no LM accuracy has been computed in this update.
