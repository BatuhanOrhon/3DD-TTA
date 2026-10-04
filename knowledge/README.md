# 3DD-TTA Thesis Knowledge Base

**[User report/Code/Planning] 2026-10-04 block-routing handoff:**
The user requests an honest assessment and implementation/Colab plan for
local-SCD/style-spectral routing, its reverse, and style-only projection.
Read the [design](gsd_block_routing_design_20261004.md),
[implementation plan](gsd_block_routing_implementation_plan_20261004.md) and
[Colab scenarios](colab_gsd_block_routing_20261004.md).
Style conditioning already receives SCD gradients; global diffusion is a
separate, currently inactive mechanism. Existing updated-final-style decoding
was null on average (-.0315 pp), which is not a style-off/global-prior test.
The plan preserves hard v1, original decode, matched inputs and frozen models;
global diffusion is conditional. The runner and analyzers are now implemented
locally and independently reviewed; GPU phases have not yet been run.
The user also registered the hypothesis that the small spectral gradient
limits accuracy gains. A v1 coefficient pilot at 0/1/100/1000 is planned on
common four-corruption draws, separately from routing. See `open_questions.md`
and the linked Colab scenario.

**[Paper/Run/Inference] 2026-10-04 SCD/GSD composition research:**
The [composition review](gsd_scd_gradient_composition_review_20261004.md)
separates magnitude dominance, conflict and prediction complementarity.
New offline arithmetic on existing smooth probes gives median local sum
rotation only .054-.568 degrees from SCD and no local negative cosines;
style conflicts are present. These are seed-0 Gaussian/Impulse reference
states, not hard-v1 or new guided-run results. The proposed next step is
missing v1 diagnostics, followed conditionally by per-example style projection
with a magnitude control. No adaptation algorithm or GPU run was added.

**[Run/Inference] 2026-10-04 Background step sensitivity ingested:**
All 30 Background-only runs (2 arms x 5/10/15/20/25 steps x seeds 0/1/2)
validate; one separate pre-inference hash failure is preserved and excluded.
Measured counts are 78 batches x requested steps. Scores remain about
23.8–25.0%; the best observed mean is step 25 (smooth-only 24.9595%,
unguided 25.0000%), only +1.08/+1.38 pp against same-arm/seed 35-step
references. Results are non-monotonic and do not resolve the low accuracy.
See the [validated analysis](../result/modelnet40_c/gsd_latent_spectral_smooth_v2/background_step_sensitivity_20261004/analysis_20261004_step_sensitivity/validation.md)
and [research record](gsd_background_step_sensitivity_20261004.md).

**[Run/Inference] 2026-10-04 unguided and smooth-only ablations accepted:**
All six full15/seeds0-2 ZIPs and both supplied summaries validate.
Unguided is **61.2021 +/- 0.1274%**; smooth-only beta .5/alpha8.140161 is
**61.3596 +/- 0.0721%**. The observed smooth-only increment is +0.1576 pp,
positive in all three seeds and 12/15 corruption means. Background stays
near24% in both, with actual35-step execution verified. Archived SCD-only
is63.8799%; same-smooth+SCD is63.8520%. Native binary identities differ
between the two new arms, and the historical controls also differ in source
and package inventory; no common-draw causal claim is established.
Read the [accepted results, provenance and next decision](gsd_guidance_ablation_results_20261004.md).
Reuse these completed conditions. The user-directed Background step scan is a
separate development sensitivity test; inspect its [predeclared plan](gsd_background_step_sensitivity_20261004.md).

**[Run/Code] 2026-10-04 v1 Background completion ingested:** Full seed0
spectral-only M100 Background is **572/2468 = 23.1767%**. Reusing the archived
14 rows yields a **61.1967%** descriptive 14+1 composite (22655/37020), not one
homogeneous all15 run. Source/binary differences and the launcher's graph-file
compatibility guard are documented in the
[validation report](../result/modelnet40_c/gsd_latent_spectral_v1/analysis_20261004_background/validation.md).
Do not rerun Background. No Colab composite JSON was supplied; the local
derived summary is explicitly separate. The subsequent unguided/smooth-only
ingestion above supersedes the earlier pending-output status.

**[User decision/Code] 2026-10-03 missing ablations prepared:**
[Colab handoff](gsd_guidance_ablation_handoff_20261003.md) provides unguided
diffusion and fixed beta=.5 smooth spectral-only (SCD=0), all15/fullfiles,
seeds0/1/2: six new runs total. Existing controls are reused. Scripts and
upload notebook are locally ready; no GPU results or push yet.

**[Run/User preference] 2026-10-03 reuse completed results:** The
[GSD/SCD coverage audit](gsd_existing_runs_audit_20261003.md) checks all 144
local result ZIP configs/CSVs. SCD-only/all15 and spectral-only/combined
results already exist at their recorded scopes. The 12-new-run proposal is
withdrawn; only unguided diffusion is a wholly missing main control. Inspect
existing coverage before every launch and honor the user's no-repeat preference.

**[Run/Code/Inference] 2026-10-01 smoothing diagnosis:** Read the
[historical versus smooth-gradient analysis](gsd_smoothing_diagnosis_20261001.md)
for full Gaussian/Impulse recomputation, weighted gradient scales, graph/profile
scope, conditioning conflicts and limits of attributing historical gains.

**[Run/Verification] 2026-10-01 all-15 test matrix:** See the consolidated
[ModelNet40-C 15-corruption results matrix](modelnet40_c_all15_results_matrix_20261001.md),
covering 40 complete local archives across source-only severity, reconstruction,
3DD-TTA controls and the full GSD screen. Incomplete/pilot runs and external
paper/reference scores are labeled separately.

**[Run/Inference] 2026-10-01 full-test screen ingested:** All nine ZIPs and the
derived summary for reference `20260928-113047_gsd-cal-diagnose-reference-seed0-n64`
pass strict archive and protocol checks. Across seeds 0/1/2, SCD-only is
63.8799% +/- 0.1326 pp; beta .5/rho .001 is 63.8520% +/- 0.0353 pp
(-0.0279 pp against matched SCD-only, mixed seed directions); beta2/rho .01
is 63.8214% +/- 0.1184 pp (-0.0585 pp, all three seeds lower). Neither spectral
candidate improves the full-set mean. This is a full-test-set development
screen with descriptive candidate selection, not independent confirmation.
Do not repeat candidate tuning on the same examples or promote a spectral
candidate from this result. See the latest entry in [`findings_log.md`](findings_log.md).

**[Run/Inference] 2026-09-28 rebuilt interaction ingested:** All six ZIPs under
the same calibration reference pass the recorded artifact/protocol checks; the
supplied summary matches raw recomputation. Read [the accepted results and
historical development proposal](gsd_interaction_results_20260928.md). New
SCD-only macro is 74.609375%; beta .5/rho .001 is 75.390625%, beta2/rho .01 is
75.781250%, while beta2/rho .001 is 73.828125%. This supersedes the earlier
screen and prevents treating the old beta2/rho .001 lead as stable. The new
block is internally matched; old/new package and extension identities differ.
Calibration alpha drift is small (.01298%/.00513% for beta .5/2), with cause
unisolated. The later full-test screen above supersedes its proposed 512-example
follow-up. No independent confirmation is available.

**[Code/Run/Inference] 2026-09-28 review: finish development before freezing.**
The fixed-state diagnostic and five-arm seed-0 development screen have now
run. Beta2/rho .001 ranks first (+1.1719 pp over SCD-only) on 128 Gaussian and
128 Impulse examples, but this is exploratory and not held-out confirmation.
The [implementation/plan audit](gsd_calibration_audit_20260928.md) corrects
the preceding immediate-freeze recommendation: beta .5 is only one correct
prediction behind beta2. Restore the planned small interaction check and
larger/repeated-seed development comparison, keeping both candidates.
Prepare object-disjoint confirmation separately; the current launcher requires
overlap with the calibration pool and cannot execute that confirmation yet.
Analyzer validation gaps are recorded in the audit; independent checks found
no corresponding mismatch in the current seven archives. The stricter analyzer
and the one-arm beta .5/rho .01 handoff are prepared; the latter has not run.
The original calibration config is missing from Colab. A regenerated reference
failed the coefficient-equivalence check against an old cell (user report);
the mismatch magnitude/cause remain unmeasured. At the user's request,
`c1c7467` adds `--rebuild-prerequisites`: one new reference, four regenerated
comparison cells, then beta .5/rho .01 and the five-cell summary. New results
must form their own matched block. No rebuilt result ZIP has been ingested yet.
Read the [current restart review](gsd_interaction_review_20260928.md) and
[copyable next-agent handoff](gsd_interaction_handoff_20260928.md) before
continuing. Do not restart an already running Colab invocation.
See
[`findings_log.md`](findings_log.md), [`open_questions.md`](open_questions.md),
and the exact run summary at
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/beta_screen_summary.json`.
The complete raw diagnostic ZIP is still needed for full provenance.
For compact analysis of the large report, use
[`scripts/analyze_gsd_calibration.py`](../scripts/analyze_gsd_calibration.py);
it ranks beta by observed development accuracy only after all five screen
bundles at the same rho are supplied.

**[Code/Inference] 2026-09-27 calibration review:** The
[revised proposal](gsd_calibration_review_20260927.md) uses mean-normalized
eigenvalue scale for a small fixed beta grid, measures SCD update sizes rather
than judging raw gradient norms, and calibrates fixed relative spectral
contributions on common reference states. It corrects the earlier blanket
restriction on unlabeled target-input statistics. The prior q95/v1-target
proposal is superseded; the diagnostic runner is now implemented, but no
new model experiment has been run.

**[Run] 2026-09-27 smooth-v2 smoke:** The fixed Colab launcher now completes
the Gaussian smooth-profile one-batch smoke at commit `d75a32d`; its validated
32-example ZIP and diagnostics are recorded in
[`findings_log.md`](findings_log.md) and beside the raw artifact. This is
execution/gradient evidence only, not an accuracy comparison. Beta/weight
calibration and the hard/smooth pilot remain open.

**[Code/Run] 2026-09-26 implementation correction:** The three blockers in the
[implementation review](gsd_smooth_review_20260926.md) have been repaired:
rank-sized basis allocation, raw-unit PSD tolerance, and the v1/hard-v2/smooth-v2
launcher matrix. Persistent algebra, protocol and launcher tests were added;
all 112 CPU tests pass, including a regression guard for the Colab Python 3.8
annotation failure. Launcher floats now
round-trip exactly, empty-filter storage is correct, and large finite beta
avoids float32 overflow at zero modes. This is CPU evidence only. Beta/alpha calibration and
Colab smoke/pilot evidence remain open; no accuracy claim is supported.

**[Paper/Inference] 2026-09-26 mathematical design:** Read the
[GSDTTA reread and guidance decision](gsd_guidance_math_20260926.md).
The paper learns low-frequency shifts, not a two-spectrum matching loss.
The next scalar guidance uses a fixed latent reference graph and common-basis
smooth spectral fidelity. The opt-in `gsd_latent_spectral_smooth_v2` code and
pilot runner are now on branch `gsd-smooth-spectrum`, based on
`gsd-development@9650770`; no model/Colab accuracy evidence is available.
The existing v1 code/results remain the comparator.
The next GSD test is registered as
[`gsd_smooth_spectrum_profile_pilot`](gsd_guidance_math_20260926.md#9-difference-from-the-current-gsd-code-and-next-test-case):
preserve v1, then compare hard and smooth profiles in the same new host with
the guidance scale calibrated. Beta and calibration values remain explicit
inputs; they must be fixed before evaluation.

**[Code/Run] 2026-09-23 branch-comparison update:** Read the
[current versus legacy GSD diagnosis](gsd_branch_comparison_20260923.md).
The M100/240/400 Gaussian/Impulse pilots are archived and have null/negative
mean increments; weight1 local spectral gradients are approximately
.04--.10% of SCD norms. Legacy eval defaults differ in trajectory, graph,
multiband objective and batch scaling, so reported old gains are not yet
isolated spectral effects. Current GSD lambda remains **.95** after `509b901`;
historical .96-lock and GPU-pending statements below are superseded for this
method. No all-15 GSD result or confirmed causal explanation is available.

**[Code] 2026-09-23 GSD development update:** The user authorized GSD-only
development on `gsd-development`, derived from current `baseline-repro-clean`
at `79cc027`. Read [the short GSD design](gsd_design.md),
[audit and plan](gsd_integration_20260922.md),
[verification record](gsd_verification_20260923.md) and
[Colab handoff](colab_gsd.md). The opt-in method is
`gsd_latent_spectral_v1`, named GSD-inspired latent spectral guidance.
Raw/eval LION, EMA off, frozen Point-MAE and original-style decoding remain
the operational controls. Earlier statements parking GSD are superseded for
this scope; PxP and combined-method work remain parked. **[Open]** GPU
validation and accuracy evidence remain pending; local tests are CPU-only.

This directory is the canonical, repository-local memory for the thesis. A new research session must start here instead of reconstructing the project from chat history.

## Current thesis in one paragraph

The project studies training-free test-time input adaptation for corrupted 3D point clouds. The base system is 3DD-TTA, which uses a pretrained LION latent diffusion model and Selective Chamfer Distance (SCD) guidance to reconstruct a classifier-friendly point cloud while keeping the source classifier frozen. This fork investigates whether graph-spectral structure constraints inspired by GSDTTA, dynamic spectral bases, global/local diffusion variants, and PixelAsParam-inspired conflicting-gradient projection can improve ModelNet40-C robustness. The immediate scientific priority is not another variant: it is first closing or explaining the gap between the published/repository baseline and locally observed Colab results under a controlled, reproducible protocol.

## Required reading order

1. [Thesis scope](thesis_scope.md)
2. [Reproduction gap](reproduction_gap.md)
3. [Experiment protocol](experiment_protocol.md)
4. [Repository map](repository_map.md)
5. [Method synthesis](method_synthesis.md)
6. Paper notes: [3DD-TTA](papers/3dd_tta.md), [GSDTTA](papers/gsdtta.md), [PixelAsParam](papers/pixelasparam.md)
7. [Development history](development_history.md)
8. [Findings log](findings_log.md) and [open questions](open_questions.md)
9. [Source index](source_index.md)

Before clean-restart implementation also read the [follow-up code audit](code_audit_20260912.md) and [small implementation batches](clean_restart_batches.md). These refine earlier informal next-test ordering.

**[Run/Inference] Reproduction context:** The source-only severity 1--5 probe is complete and shows a 22.1907 pp severity-1-to-5 drop, but no tested severity matches the paper's 57.6% source row. Internal artifact manifests and the canonical Zenodo archive now agree: the downloaded archive identity matches Zenodo, all 15 archive members match the audited manifest, and all 15 current Colab files match byte size and SHA-256. The remaining source-only provenance caveat is the author/canonical Point-MAE checkpoint identity. The preprocessing identity control is stable across seeds 0/1/2 at 55.0135% +/- 0.0602 pp sample SD (+1.3236 pp versus deterministic source-only). Pure VAE encode/decode is also stable across seeds 0/1/2 at 54.8469% +/- 0.0790 pp sample SD (+1.1570 pp versus source-only), but remains 0.1297--0.2296 pp below the matched preprocessing identity result at every seed. The SCD normalization all-15 control is complete at 61.2678% +/- 0.0979 pp and is 2.4806 pp below the matched original-style unnormalized control; preserve the original baseline and keep the separate lambda=.96 check open. The common-draw control is deferred because it is not expected to change accuracy and would only strengthen a causal diffusion/guidance claim; that claim remains explicitly open. EMA, GSD/PxP and other datasets remain parked.

**[Inference/Open] Current GSD next action:** The requested step scan is
complete. Do not promote 25 steps from this same-test-set sensitivity result;
the remaining Background gap is much larger than the observed step effect.
See the validated analysis and its source/native-build comparison limits above.

**[Run/Inference] 2026-09-22 update:** The three-seed Gaussian/Impulse
lambda=.96 pilot is complete and null: +0.0068 pp mean versus the matched
lambda=.95 original-style pilot, with mixed seed directions. Do not promote it
to all-15; preserve the original baseline and keep any scale-matched Eq. 11
test separate.

**[Run/Inference] 2026-09-23 update:** The user-requested all-15 lambda=.96
confirmation is complete and validated at
`result/modelnet40_c/scd_lambda96_control/`. Original-style macro mean is
63.9339% +/- 0.1878 pp versus 63.7484% +/- 0.1399 pp for the matched
lambda=.95 original-style rows; all three paired seed deltas are positive and
the mean delta is +0.1855 pp. This supersedes the pilot-only stopping note as
a paper-setting check. Treat `.96` as the paper-conformant reference for new
baseline/GSD runs; retain `.95` as the historical comparator. The existing
GSD `.95` pilot is not relabeled and a `.96` GSD pilot would require a fresh
matched weight-zero control.

After the accepted Batch 1 smoke, follow [Batch 2 source-only instructions](colab_source_only.md).

After the source-only protocol audit, use the [clean Point-MAE control](colab_clean_control.md) before interpreting the remaining corrupted-source gap.

Historical/parked task: [ShapeNet Audit Instructions](colab_shapenet_audit.md).

## Evidence labels

Use these labels in all future updates:

- **[Paper]** Directly reported in a cited paper.
- **[Code]** Verified in the named repository revision/file.
- **[Run]** Supported by an archived artifact under `result/`.
- **[User report]** Reported conversationally but not yet backed by an archived artifact.
- **[Inference]** A reasoned interpretation that still needs a controlled test.
- **[Open]** Unknown, ambiguous, or awaiting evidence.

Never silently promote an inference or user report into a run-backed finding. If sources disagree, retain both values and describe the disagreement.

## Headline state as of 2026-09-12

- **[Paper]** The WACV 2025 3DD-TTA paper reports **65.7%** ModelNet40-C mean accuracy for its Point-MAE setting.
- **[Code]** The repository README reports **66.1%**; several per-corruption cells differ from the published table. Both references must remain visible.
- **[User report]** Local original-code runs are approximately **63%**, while current GSD variants reach approximately **63.5%**. No complete run bundle is currently archived, so these numbers are provisional.
- **[Run]** Batch 1 Gaussian smoke is archived at `result/modelnet40_c/3dd_original/20260912-110541_baseline-smoke_seed0/`: 47/64 on a two-batch prefix, with validated files/config/counts. It is not a full baseline or accuracy comparison.
- **[Run]** Full source-only identity evaluation is archived at `result/modelnet40_c/source_only/20260912-111822_source-only_seed0/`: 53.69% macro over all 15 corruptions, 3.91 points below README's 57.6%. The reproduction gap therefore precedes LION/TTA under this recorded protocol.
- **[Run]** The locked pure VAE control is archived at `result/modelnet40_c/pure_vae_encode_decode/20260919-144513_pure-vae-s5-all15-seed0.zip`: 54.7947% (20,285/37,020), +1.1048 points over source-only but 0.2296 points below preprocessing identity. This places the positive one-seed delta primarily in preprocessing rather than VAE reconstruction; see the findings log.
- **[Run]** Pure VAE seed-stability is archived at `result/modelnet40_c/pure_vae_seed_stability/`: seed 0/1/2 are 54.7947%, 54.9379%, and 54.8082%, respectively; mean 54.8469% with 0.0790 pp sample SD and +1.1570 pp mean over source-only. The early/late timestamp archives are the seed-1/seed-2 runs despite both filenames saying seed1; configs and commands record the true seeds.
- **[Run]** Preprocessing identity seed-stability is archived at `result/modelnet40_c/preprocessing_identity_seed_stability/`: seeds 0/1/2 are 55.0243%, 55.0675%, and 54.9487%, respectively; mean 55.0135% with 0.0602 pp sample SD and +1.3236 pp mean over source-only. The early/late timestamp archives are the seed-1/seed-2 runs despite both filenames saying seed1; configs and commands record the true seeds.
- **[Run]** The paired clean input control at `result/modelnet40_c/source_only/20260912-115021_clean-control_seed0/` obtains **90.64%** (2237/2468) using the same frozen Point-MAE checkpoint, labels and FPS(1024) path. This rules down an obvious clean classifier/data failure, but author-published clean checkpoint parity remains unavailable.
- **[Code]** Active branch is `baseline-repro-clean`, created from main `107305f` in the same repository folder. Batch 0 docs commit is `b31fd23`. Batch 1 adds a smoke runner/artifact module and optional observers; observer-stripped ASTs of the three changed baseline modules match main. LION/dependency files remain unchanged. Legacy GSD/PxP code remains on `pxp-gradient-projection` at `53ba252`; historical method notes refer to that audited revision.
- **[Inference]** Reproducibility risks include unseeded stochastic interpolation/noise, environment drift, scheduler details, checkpoint/data identity, batch-size sensitivity, and incomplete result logging.
- **[Code]** Original LION trainer inference disables dropout; the inherited demo wrapper leaves LION in training mode. **[Run]** A same-commit all-15 seed-1/2 screen favors eval mode by **+.7577 +/- .1203 pp** and 13/15 corruption means. It is seed-controlled rather than common-draw paired, so eval is a provisional operational baseline rather than an isolated causal dropout claim; see [the full screen](dropout_eval_mode_20260913.md).
- **[Code]** Mean spectral loss plus summed SCD changes relative guidance with batch size. **[User report]** Mean settings previously outperformed sum; sum/smaller-step tuning is deferred until baseline and spectral-off parity.
- The user selected same-folder development and requested skill/knowledge preservation. No new stash was needed; existing dev stash is untouched. Research memory/skill/result protocol were committed in `b31fd23`; source PDFs remain local/untracked. Batch 1 local checks passed but Colab smoke acceptance is pending. No accuracy improvement or dropout effect is established.
- **[Code]** The fork's spectral method is not a reproduction of full GSDTTA: it regularizes a LION local latent and keeps the classifier frozen instead of learning a physical-coordinate spectral shift and alternating input/model adaptation.
- **[Code]** The fork's PxP variants are PixelAsParam-inspired gradient-conflict methods between spectral and Chamfer guidance; they do not implement PixelAsParam's denoising/diversity/classification decomposition.

## Update rule

After any material experiment, diagnosis, code change, or paper review:

1. archive the immutable output under `result/` using [the experiment protocol](experiment_protocol.md);
2. append the outcome—positive, negative, or inconclusive—to [findings_log.md](findings_log.md);
3. update affected method/reproduction notes and [open_questions.md](open_questions.md);
4. include date, Git commit, run path, evidence label, and what would falsify the conclusion;
5. do not erase superseded findings—mark them superseded and link the newer evidence.

## Terminology

- `z` / global latent: LION global shape/style representation.
- `h` / local latent: per-point LION latent, represented in this repo as `B × 2048 × 4` for spectral work.
- `U`: graph Laplacian eigenvector matrix (a basis), not a single vector. `U_current` is recomputed in dynamic variants.
- SCD: Selective Chamfer Distance.
- TTA: test-time adaptation.
- PxP-inspired projection: this fork's use of projection for conflicting loss gradients; not an exact implementation of PixelAsParam.

## First action for the next session

Read the [current session handoff](session_handoff_20260915.md). It supersedes historical smoke/ShapeNet-next-action statements above and records the selected eval/raw baseline, EMA decision, current blockers, and immediate Colab experiment. Numerical experiments remain Colab-only.
