# CloudFixer denoiser transfer: checkout and comparison preparation

Date: 2026-11-14. Branch: `research/cloudfixer-transfer-20261114`.
Base: `origin/thesis-integration@64b7345138777a7cc6833c6130be28bccd166c01`.
Scope: repository/literature review, a fresh worktree and official-source clone.
No new adaptation implementation, checkpoint download, training or GPU run.

## Follow-up: explicit CloudFixer-only scope and recovery/search request

**[User report/Decision]** The user clarifies that LION must be absent from the
new inference pipeline: use CloudFixer's models/preprocessing and replace its
adaptation rule with gradient-free SCD during diffusion and temporal PM.
The user also requests incremental Drive persistence, resumable evaluation and
grid search; eight reverse steps are not assumed optimal on the new prior.

**[Code/Inference/Open]** A [concrete proposed design](../docs/superpowers/specs/2026-11-14-cloudfixer-only-proximal-pm-design.md)
specifies native polynomial XYZ sampling, separate initial-noise/update-count
search, mask-aware reference matching and a clean terminal readout. Native t=0
still has noise, so terminal cleanup has a real denoiser-call cost. Per-batch
committed artifacts and method-independent batch RNG keys support recovery;
changing source/assets/policy starts a distinct campaign. Prior peer FUSE
write/read success did not establish new-VM availability, motivating Drive
API upload and independent cloud-download SHA validation. No credentials are
copied. Interactive Google consent remains user-operated if required.

The proposed48-cell grid and128-object development cohort are implementer
choices, not measured optima or user-approved numerical settings. The design
passed an independent builder/source review and awaits its design decision before implementation;
no production code, checkpoint download or GPU/runtime action has occurred.
Local static checks validate48 cells, strict integer schedules, distinct PM
locations and source links. The global polynomial round/truncation difference
at float32 indices251/253 is outside the initial proposedK<=160 range; actual
model index consistency remains a required test, not a claimed repaired bug.

## Research objective and current thesis state

**[User report/Decision]** Test the methods developed on LION using the diffusion
model employed by CloudFixer, then compare with CloudFixer under shared conditions.
The immediate request prepares that work; it supplies no new experimental result.

**[Code]** The main Desktop checkout is an older `gsd-multibackbone-study`
revision with existing user edits. The fresh worktree uses the latest fetched
shared integration instead, at
`/Users/batuhan/.codex/worktrees/0e7f/3DD-TTA`. Existing edits and peer worktrees
are preserved. Official source was cloned with `git clone` into the sibling
`/Users/batuhan/Desktop/projects/CloudFixer`, at author commit
`f043e24414fcf4165fdc892abff21215c9989141`, with a clean working tree.

**[Code/Run, historical]** The thesis investigates training-free input adaptation
with frozen classifiers. Earlier GSD-inspired spectral and PxP-inspired
composition results are mixed/null and are not the current operational method.
The [active baseline](current_baseline.md) is `lion_proximal_pm_n8_v1`:
eight updates for every corruption, frozen bidirectional 95%-trimmed latent-XYZ
Chamfer correction, tau=.1, original encoded decoder style and probability mean
at clean2/4/6 plus terminal8. It avoids denoiser backward. The accepted
[common-horizon study](lion_guidance_uniform_horizon_results_20261108.md)
reports 63.722312% versus full-backward N8+PM 63.735818%, using 27.673026%
less measured pipeline time on Point-MAE/ModelNet40-C/seed0/B128. These are
historical exposed-data measurements, not fresh measurements of this checkout.
The [later terminal/logit pilot](terminal_logit_pilot_results_20261110.md) did
not establish a replacement; PM remains active. Native repeatability limits persist.

## Paper and source findings

**[Paper]** [CloudFixer, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf),
sections4.1–4.3, guides rotation/displacement optimization with detached diffusion
predictions. Voting uses independent adapted views. CloudFixer-O additionally
adapts the classifier and belongs in a separate protocol. The
[supplement](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603-supp.pdf),
appendicesD/E/G, specifies seed2, severity5, normalization and adaptation details.
Its training-dataset naming and regularizer schedule must be reconciled with code.

**[Code]** The ModelNet40 training script selects clean
`modelnet40_ply_hdf5_2048` train H5s, not corrupted ModelNet40-C test arrays.
`train_dm.py` uses clean test examples for validation/checkpoint selection;
there is no demonstrated independent validation split. The ModelNet40 recipe is
5000 epochs, AdamW2e-4, batch32/accumulation2, polynomial_2/500 diffusion steps.
This is a published recipe, not reconstructed released-weight training history.
The [Nov12 audit](cloudfixer_denoiser_transfer_audit_20261112.md) remains the
detailed architecture/training and CloudFixer-O paper/code reference.

| Boundary | Current LION implementation | Official CloudFixer source |
|---|---|---|
| Denoiser state | B×2048×4 local latent, conditioned on z128 | Physical B×N×3 XYZ; Point-E-derived XYZ-only Transformer |
| Training domain | ShapeNet pretrained prior | ModelNet40 source-training recipe |
| Noise contract | Linear DDIM grid; LION t+1 convention | polynomial_2/500; normalized t mapped to integer t×500 |
| Input/output | LION scale/axis conversion, encoder and decoder | unit_std/axis preparation; classifier unit-ball/axis conversion |
| Update | Reverse trajectory with frozen-neighbor proximal correction | 30 AdaMax rotation/displacement updates; random noise levels |
| Readout | Four candidates along one trajectory, frozen classifier | Terminal view or independent-view voting; O optionally updates classifier |

**[Code/Inference]** `diffusion/diffusion.py` already exposes epsilon/clean
prediction and deterministic DDIM transitions. This is a useful integration
surface, not an implemented transfer. With its alpha convention,
`x0_hat=(xt-sigma_t*epsilon_hat)/alpha_t` and
`xs=alpha_s*x0_hat+sigma_s*epsilon_hat`; a clean correction transports as
`alpha_s*(x0_corrected-x0_hat)` at fixed epsilon. Our LION helper instead expects
cumulative alpha squared and embeds XYZ into four channels. Reuse the
`proximal_chamfer` quadratic where appropriate; do not call the four-channel
`bridge_clean_delta` unchanged or substitute `lion.priors[1]` weights.
No transfer setting for tau, trimming, timestep range or readout schedule has
been validated. Direct XYZ spectral guidance is a separate possible ablation.

**[Code]** Important reviewed files: `scripts/eval_chamfer_baseline.py`,
`lion_guidance_pilot.py`, `tta.py`, `scripts/eval_chamfer_depth.py` in this repo;
`adapt.py`, `diffusion/{build_model,diffusion,transformer}.py`,
`scripts/{train_dm,run_cloudfixer,set_env}.sh`, `train_dm.py`,
`datasets/dataloader.py`, `classifier/pointMAE.py` and its config in CloudFixer.
The official example defaults to Point2Vec/seed2/B128 and runs both CF and CF-O.
It is not a ready-made matched Point-MAE command. The known commented
`pointnet2_utils` import in `utils/misc.py` is still present in the untouched clone;
prior Point-MAE execution required restoring it. No environment was installed.

## Existing work found outside shared integration

**[Code/Run, historical]** Peer branch
`experiment/cloudfixer-pointmae-pilot-20261110@d91a64d` contains prior pilot,
latency, full-run/recovery scripts and notes. It is preserved and not merged or
resumed here. Its local worktree is
`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/cloudfixer-pointmae-pilot-20261110`.
The archived three-corruption pilot compared native CF against native LION
proximal+PM: 71/96 versus55/96, seed0/B32/IDs0–31. It did **not** run our
method with the CloudFixer denoiser. Classifier checkpoint bytes match but
CF uses32groups/no external FPS1024 versus LION64groups/FPS1024. Neither the
accuracy difference nor measured time isolates the adaptation rule or prior.

**[Code]** This review freshly checks both pilot ZIP SHA-256/CRC and CSV totals,
not a new NPZ prediction recount. Full15 completion is not established: the
peer note records an interrupted evaluation, an accepted Background case and
Drive-consent-dependent recovery. Those are dated records, not a current
runtime inventory. No peer runtime was queried or operated. A previously
recorded diffusion weight hash is `b64841d022e08b1a89f58dd9b16d9023e00be52fbca3dca735d5798362bb5a5a`;
it is an asset-reuse lead, not a newly verified local checkpoint or EMA identity.

## Controls needed before interpreting a transfer

**[Inference/Proposal]** The discriminating comparison holds the author denoiser,
classifier state/config, coordinate preprocessing, data hashes/IDs and evaluation
code fixed. It includes source-only, ordinary denoising, transferred proximal
terminal, transferred proximal+PM and frozen-classifier CloudFixer. Keep native
pipeline comparisons with LION separate. Add GSD/PxP only as explicitly identified
secondary methods; their historical latent-space gains cannot be assumed in XYZ.

**[Open]** Establish classifier readout parity, author-weight provenance/load,
duplicate masks/cardinality, polynomial scheduler endpoints and same-state
numerical controls before accuracy. Match seeds and stochastic draws where the
algorithms permit; equal seed alone does not imply identical draws. Fix a common
corruption-independent policy, separate development from confirmation, and report
all15/severity5 counts, paired object uncertainty, memory, denoiser/classifier
calls, B1 latency and matched-batch pipeline time. CF independent votes and our
trajectory votes have different compute. No architecture-only causal claim follows
from changing prior domain and representation together.

**[Code]** Verification: clean clone/origin/commit, new branch/base ancestry,
fresh native-free baseline CLI preview, historical pilot SHA/CRC/CSV checks,
source hashes, local links and whitespace. Model/GPU tests and full CPU suite
were not run; local Python lacks numpy/torch/pytest. No dependencies or temporary
wrappers were added. The [compact receipt](evidence/cloudfixer_transfer_20261114/sources.json)
pins sources and these limited checks. This preparation stays on the requested
separate research branch; no shared integration publication is made in this task.
