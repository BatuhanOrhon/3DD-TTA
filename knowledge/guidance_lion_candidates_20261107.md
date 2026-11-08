# LION guidance candidates after the spectral studies — 2026-11-07

[Decision/Inference/Open] Prioritize a bounded test of **cheap clean-latent Chamfer correction with occasional full backward**, and a separate **fourth-channel correspondence diagnostic**. These are research hypotheses, not implemented methods, established novelty, GPU results or a selected implementation design. Keep `chamfer_pm_n8_v1` as the baseline. A third, higher-risk feature-only semantic correction is described for comparison, not immediate expansion.

## Evidence and scope

[Code] Reviewed shared `ecba1e44036236dfa52a60d8ef17b1e4c251d504`, not the dirty original `gsd-multibackbone-study@5d0b89b` checkout. Work branch: `research/guidance-ideas-20261107`. No production source/default changes, Colab calls, model inference, new training or GPU execution. Existing peer lowband-halfmix hold is unaffected. Before publication, reconciled incoming documentation-only `96e51ad` and its [complete GSD inventory](gsd_session_test_inventory_20261107.md); its completed/unrun distinctions and all peer records are preserved. The publication commit containing this report identifies its integration revision.

[Run] Accepted [depth results](chamfer_depth_tuning_results_20261030.md): 256 object-disjoint confirmation IDs, all15/s5/Point-MAE/seeds0,1,2; N5 terminal65.789931%, N5 PM66.510417%, tuned N8 PM68.741319%. Thus +2.951388pp versus N5 terminal and +2.230902pp versus N5 PM (rounding at the last digit). These are confirmation-subset results, not new full-benchmark measurements. N8 changes starting noise as well as reverse-step count; Background remains35. Current gains are a useful baseline; a further accuracy gain is not the only possible scientific contribution. Refer to [readiness review](publication_readiness_20261107.md) for exposure, transfer and attribution limits.

## What the old fourth-channel experiment actually did

[Code] Recovered `53ba252519c7cf65f836a9c1c564027142ab1573` (`pxp-gradient-projection` history):

- `graph_spectral.py:17,55–60,108–111`: `use_4d_gft` changes the GFT **signal** from XYZ to XYZ+feature; the graph is still built from XYZ only. This is not a four-dimensional neighborhood graph or a four-dimensional physical reconstruction.
- `tta_gsd.py:173–194`: live predicted-clean XYZ+feature is transformed in the fixed basis and compared to original coefficients with low/mid/high MSE. The later live-gradient computation matters; an earlier no-grad diagnostic does not establish disconnected guidance.
- `eval_gsd_tta.py:48` and `main_gsd_tta.py:50` expose `--use_4d_gft`. Changing a mean loss from three to four channels also changes the XYZ contribution's normalization, besides adding feature fidelity.

[User report/Open] The user remembers this test as unsuccessful. Source support is recovered; a verified numerical run receipt uniquely tying that flag, checkpoint, protocol and result together was not found in the reviewed records. Do not manufacture an accuracy number or diagnose the cause of failure.

[Code] At the reviewed shared revision, `models/latent_points_ada.py` encodes B×2048×4 and the local prior consumes all four channels. Decoder `LatentPointDecPVC` consumes all4, returns XYZ plus a learned residual. In `tta.py:176–237`, SCD measures predicted-clean XYZ but backpropagates through the denoiser to **all four noisy channels and conditioning style**. Decoder/PM use original encoded style in the baseline. Thus the feature is already active; the question is how to use it explicitly and usefully in guidance. The decoder's .01 residual coefficient is not a derivative bound. Native coordinate-gradient/repeat limitations remain.

## Closest literature and novelty boundaries

Primary sources checked on 2026-11-07; versions below pin preprints. This is a focused search around diffusion guidance/backward cost, point correspondences and LION features, not proof that no equivalent method exists.

| Source | Relevant mechanism | Consequence here |
|---|---|---|
| [LION, NeurIPS2022](https://research.nvidia.com/labs/toronto-ai/LION/), §3 | Hierarchical z/local point latent h and conditional local prior | Respect actual four-channel coupling; a scalar feature is not automatically a semantic class/confidence. |
| [CloudFixer, ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf), §3 | Optimize geometric input transformations using detached diffusion targets, avoiding denoiser backward | “Remove diffusion backward for point-cloud TTA” already has a direct precedent. |
| [Guess & Guide, v1 2026-03-09](https://arxiv.org/html/2603.07860v1), §3/Algorithm1/App.B | Pixel-space fidelity/proximal updates, latent decode/encode and noise-preserving transitions without denoiser/decoder backward | Generic proximal or gradient-free guidance is not new; known measurement operator differs from unknown corruption with a contaminated anchor. |
| [Piecewise Guidance, v1 2025-07-22](https://arxiv.org/html/2507.18654v1), §III | Different conditional-score approximations at different noise levels | Generic mixed expensive/cheap guidance is not new either. Its inverse-problem assumptions/theory do not transfer automatically. |
| [RPM-Net, CVPR2020, v1](https://arxiv.org/html/2003.13479v1), §5.1–5.3 | Learned geometry/spatial features, soft correspondences and Sinkhorn | Feature-assisted matching itself is established. Our candidate must exploit frozen LION's feature without registration training. |
| [SeaLion, CVPR2025, v1](https://arxiv.org/html/2505.17721v1), §3.1–3.2 | Joint noise/part segmentation prediction and segmentation-conditioned decoding | Semantic part awareness is trained in; it is not evidence that our original LION scalar already has those semantics. |
| [Readout Guidance, CVPR2024, v1](https://arxiv.org/html/2312.02150v1), §3–4 | Trained readout heads extract spatial/correspondence control signals | Feature guidance is established; these heads are not a free plug-in for our frozen checkpoint. |
| [DragonDiffusion, ICLR2024](https://arxiv.org/abs/2307.02421) | Feature-correspondence guidance for image editing | Do not claim the general feature-guidance idea. |
| [DiffStateGrad, ICLR2025](https://arxiv.org/abs/2410.03463) | Intermediate-state subspace projection of guidance | Generic projected guidance is not a new contribution; N×4 rank/units make naive transplantation questionable. |
| [RACER, v1 2026-08-03](https://arxiv.org/html/2608.01740v1), §3 | Disagreement-controlled diffusion feature forecasting/cache refresh | Gradient forecasting is an adaptation hypothesis already proposed in the Oct31 project audit, not newly discovered here. |

## A — cheap clean-latent correction between full backward steps

[Inference] The existing [sparse result](chamfer_guidance_methods_results_20261031.md) saved about21% reconstruction time on fresh Gaussian/Background cases but lost1.3571pp on ModelNet and .9683pp on Scan in its older10/30-step protocol. It zeros guidance between retained backward steps. Proposed difference: still correct geometry on those steps, using a cheap analytic update instead of the denoiser Jacobian. No speed percentage transfers from that experiment.

Let predicted-clean latent be `(q_hat,f_hat)`, corrupted encoded reference XYZ be `r`, and E be the multiset of both-direction nearest-neighbor edges retained by the existing SCD trimming **at q_hat**. Count an edge twice if both directions select it. Define:

`Q(q) = (1/2) Σ_(i,j in E) ||q_i − r_j||²`.

The minimizer of `(1/2)||q−q_hat||² + τ Q(q)` is pointwise:

`q_i_plus = (q_hat_i + τ Σ_j m_ij r_j) / (1 + τ Σ_j m_ij)`.

[Inference/math] `m_ij` is edge multiplicity, including all incoming retained edges and the outgoing retained edge. No incident edge means no movement. For fixed trimming counts and exact NN distances, Q upper-bounds half the trimmed symmetric Chamfer F and touches it at q_hat. Therefore `F(q_plus) <= Q(q_plus) <= Q(q_hat) = F(q_hat)`. This is a limited clean-coordinate surrogate property, **not** descent of clean-reference error, decoder output error, classification loss, or the complete diffusion iteration. The code's unnormalized SCD has no1/2, so τ must absorb that scale; copying gamma numerically is not justified.

[Run/CPU] [Synthetic algebra witness](evidence/guidance_ideas_20261107/proximal_algebra.json): 200 random cases check surrogate bound, proximal objective descent and trimmed objective descent. This exercises neither LION nor CUDA nor actual native tie/reduction behavior. The analytic expression and its inequalities are the evidence; no accuracy claim follows.

[Inference] On cheap steps, run the local prior without backward, correct q while leaving f's predicted-clean value unchanged, and transport the correction through the deterministic DDIM equation. With the same epsilon estimate, the candidate prev-state increment is `sqrt(alpha_bar_prev) * (q_plus−q_hat, 0)` relative to the ordinary transition. This convention must be validated against the installed scheduler; it is not the baseline's exact `−gamma * Jᵀ grad SCD` update or an exact posterior sampler. No extra decode/reencode is required for this latent-space candidate. Preserve PM observation timing (currently before current-step guidance) and its readout budget.

[Inference/Open] Cheap steps omit the baseline style update and XYZ-to-feature Jacobian coupling. Occasional full-backward refreshes are intended to retain some of these effects, not guaranteed to recover them. First compare a simple fixed refresh schedule; only if useful, test whether NN/trim-set turnover predicts exact-update approximation error and can schedule refreshes. A turnover statistic is a hypothesis, not an error certificate. Normalize/calibrate update scale on label-free development states and account for its cost.

**Minimum discriminating controls:** full N8 C+PM; sparse-zero with the same refresh schedule; cheap-only with matched style-frozen C control; hybrid exact/cheap. Keep starting noise, draws, eta/gamma semantics, p, decoder style and PM candidates fixed. Compare geometry/update norms, feature/style drift, end-to-end time including decoder/classifier/PM, peak memory, per-corruption accuracy and corrected/harmed objects. If the hybrid cannot beat sparse-zero's accuracy–cost tradeoff or cannot save total time at a predeclared accuracy margin, stop. Do not tune an adaptive refresh controller to rescue an ineffective cheap update.

**Potential contribution:** a tested symmetric-trimmed, latent-structure-aware guidance approximation with controlled exact refresh under unknown corruptions. The formula alone is a simple quadratic solution; its derivation or renaming is not strong algorithmic novelty. Closest-method baselines and a demonstrated advantage are necessary.

## B — use the fourth channel to choose matches, not to preserve its value

[Inference] Chamfer can pull nearby but structurally different surface points together. The frozen scalar feature might distinguish some of those ambiguous matches even if it is unsuitable as a spectral fidelity target. This is a different mechanism from `use_4d_gft`.

For each predicted-clean point take a small XYZ neighbor shortlist in the encoded reference, then rank or softly weight it using `||q_i−r_j||²/s_q² + β (f_i−f_ref_j)²/s_f²`. Use a shared/reference-derived robust scale with a defined nonzero floor, not independently standardized clouds that erase comparability. Detach the assignments/weights; the actual guidance loss still penalizes **XYZ displacement**, with ordinary denoiser gradients. Feature affects which geometric target is selected, rather than adding a direct penalty to preserve the corrupted feature. Account for both directions and trimming consistently. Cached shortlists require explicit validity checks; do not silently reuse stale neighbors.

**First falsifier before implementation:** on paired diagnostic states, compare XYZ-only matching with feature-assisted choices, feature shuffled across points and constant-feature controls. If a trusted clean/original correspondence exists, use it only for posthoc mechanism diagnosis, never inference or test-set tuning. Otherwise restrict claims to measurable perturbation/trajectory stability; that alone does not prove correct correspondence. Record whether the exact required clean/paired tensors exist rather than assuming archived logits contain them. A scalar with no useful signal, unstable scale or no advantage over shuffled features is grounds to stop.

**Applied controls if the diagnostic survives:** C+PM; geometry-only soft matching (isolates smoothing); feature-assisted geometric matching; direct normalized4D distance (isolates the proposed routing); shuffled feature. Match update norm/calibration and PM budget. No eigendecomposition, extra decoder or trained head is inherently needed, but kNN and ranking costs count. Feature stability may later gate β, only after it predicts usefulness on development data; the negative H5 study prevents treating arbitrary confidence proxies as reliable.

**Potential contribution:** training-free, reliability-tested use of LION's existing scalar to disambiguate correspondence under corruption. “Feature matching” is not new; neither SeaLion nor RPM-Net proves this scalar will work. Accuracy gain must transfer beyond the classifier used to select it.

## C — higher-risk feature-only semantic correction from past PM predictions

[Inference/Open] At an already decoded clean checkpoint, freeze q and z and make one bounded update only to f using a frozen classifier loss toward a detached consensus of **earlier** PM predictions. The first usable checkpoint has no such teacher and is left uncorrected. Future PM predictions cannot supply an online teacher without a separately costed second pass. Bound decoded displacement and retain a no-correction control. Use corrected candidates consistently in the final readout; this changes readout as well as latent optimization and must be isolated.

This differs from the old fixed corrupted-input semantic teacher and from spectral4D fidelity. It is also close to standard self-distillation/classifier guidance and is the weakest immediate recommendation. The scalar controls a nonlinear decoder residual; freezing latent XYZ does not freeze decoded geometry. One-quarter as many variables does not imply one-quarter compute. Additional decoder/classifier backward can be expensive and can optimize confidence without repairing shape. Compare fixed-input teacher versus causal PM teacher, all-channel versus f-only updates, matched geometry/norm constraints, and cross-head accuracy. Stop if confidence grows while independent geometry or classifier accuracy worsens. Do not claim a new semantic representation without evidence.

## Proposed order and decision criteria

[Inference/Proposal] These are suggested next experiments, not authorized/launched runs or a user-selected design:

1. Audit saved tensor coverage; reuse valid states for B's correspondence diagnostic and A's update-error checks. Require fresh matched controls where native nondeterminism exceeds the signal. Missing arrays require a separately scoped capture, not invented offline results.
2. A small mechanism pilot can use16 development objects across Gaussian, Background, density change and occlusion, seed0. This is for rejecting mechanisms, not claiming accuracy. Include same-input native repeats and fix its source/configuration. Native failures remain failures; do not relax tolerance to make an idea pass.
3. Only the survivor gets an all15 pilot, a locked schedule/scale and then a fresh or clearly exposure-labelled256-object, three-seed paired confirmation. Aggregate uncertainty by original object, not by treating corruptions/seeds as independent objects. Then validate transfer and severity coverage.
4. For speed, a proposed engineering target is >=15% **end-to-end** time saving with a predeclared noninferiority margin (e.g.0.3pp): a paired CI lower bound above−0.3pp supports that margin; an ordinary nonsignificant accuracy difference does not. These thresholds are proposals, not venue rules. For accuracy, seek a positive paired gain with cost and per-corruption harm disclosed; neither +0.5pp nor +1pp is a universal acceptance threshold.
5. Ultimately benchmark the new component **on top of current N8+PM**, plus PM-off and old-terminal references. Re-demonstrating the existing≈3pp against the old baseline would not identify the new component's contribution. Compare both fixed inference budget and fixed accuracy, and include relevant published methods under matched conditions before a submission claim.

[Decision] Do not restart broad GSD coefficient/feature-dimension/global-z scans. The [Nov6 campaign](gsd_comprehensive_results_20261106.md) and [spectral-z confirmation](scdh_spectralz_results_20261101.md) found no reliable net benefit in their scopes. Per-object semantic caps and gradient hold/forecast remain separate **already proposed** options in the [Oct31 audit](chamfer_guidance_implementation_literature_audit_20261031.md), not new findings. The held peer halfmix experiment remains independent.

[Open] A is the first recommendation when preserving the existing accuracy gain at lower cost is the goal. B is the most direct low-cost test of the user's LION-feature idea. C should wait for evidence that the feature is controllable without semantic or geometric damage. None currently establishes publication novelty, accuracy gain or GPU speedup.


## 2026-11-08T13:46:26.763617+00:00 — user clarification: currentcheap mechanism and versioned paper attribution

[Code/Paper/Inference] User requests brief method/source summary while GPUruncontinues. Actual `lion_guidance_pilot.py` currentcheap setsfull=False onallsteps: denoiserforward only, bidirectional95%trimmedChamfer matches on predictedcleanlatentXYZ, frozen-correspondence closedformproximalcorrection tau.1, DDIMclean-delta bridge, styleunchanged; eightsteps/fourPMreadouts retained. Hybrid occasionalbackward is a distincttestedarm, not currentcheap. Our exactquadratic Chamferformula is projectderivation/adaptation; genericbackward-free/proximalguidance haspriorart. Primaryinspiration reviewedversion `Guess & Guide: Gradient-Free Zero-Shot Diffusion Guidance`, arXiv2603.07860v1(9Mar2026): https://arxiv.org/html/2603.07860v1 . Currentprimaryabstract now identifies v3(28May2026) as `Sparse Scheduled Diffusion Guidance for Inverse Problems`(Spin): https://arxiv.org/abs/2603.07860 . Preserveversionedv1title, discloseupdatedname. CloudFixerECCV2024 also explicitlyavoidsdiffusionbackprop for3DinputTTA: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf . These do not establish novelty of our adaptation; unguidedattribution remainsunperformed. No experiment/default/scope change.
