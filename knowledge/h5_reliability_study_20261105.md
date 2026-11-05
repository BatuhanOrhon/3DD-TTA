# H5 — first diagnostic phase, 2026-11-05

[User report/Proposal/Open] User selected H5 (bandwise reference reliability) for
this agent's own Colab study. The hard-M/half-mixture idea is separately handed
off in `gsd_lowband_halfmix_agent_handoff_20261105.md`, local branch
`research/gsd-lowband-halfmix-handoff-20261105`@563ad7b. Root now works in
`research/gsd-h5-reliability-20261105` in the same own worktree. Other agents
must create their own worktrees; no branch switching in an active shared tree.

## Question and bounded scope

Can clean-free features measured at existing PM checkpoints predict whether
pulling a decoded geometry toward corrupted X in a spectral band improves or
harms clean-set geometry? This tests the premise of H5 before applying q as
guidance. It is not a selected accuracy method or a Wiener optimality claim.

First GPU science phase only:8 previously used mechanism IDs
459,545,1229,1435,1441,1557,1881,2095; Gaussian/Background/Shear/LiDAR, s5,
seed0/B2/N8-B35, PointMAE/LION eval/raw/EMAoff, original SCD .03/.003/.95,
DDIM100/final-alpha1. Three fresh-process zero-physical-weight captures share
exact prepared-input/encoded-state/noise caches. Existing accepted accuracy
runs are not replaced; these new runs capture previously missing intermediate
geometry/eigensystems and numerical repeat-floor evidence. No applied H5 loss,
coefficient tuning or new accuracy selection in this phase. Estimate15–25min,
whole GPU hard45min including repeats/technical checks. No >60min phase planned.

## Measurements and leakage boundary

Opt-in `--capture-h5` uses the operational control arm. Save all4 existing
PM decoder outputs in the exact prepared-X physical frame, loop counts,
reference X, corrupted normalization center/scale and original IDs. No extra
decoder forward or classifier backward. Capture overhead is timed explicitly.
After each trajectory and baseline timing/peak have finished, build the existing
GPU probe eigensystem once on X; archive actual active indices,U,normalized
eigenvalues, effective rank, graph diagnostics and tensor identities. This avoids
CPU reconstruction of a different metric, which failed earlier fixed gates.

All bands use one archived first-repeat per-object reference eigensystem
across time AND all3fresh repeats, after prepared-X tensor hash equality.
Do not compare unmatched per-repeat eigenvector coefficients. Archive other
eigensystems as reproducibility evidence but use one common metric for features. Initial
candidate band edges in normalized lambda:0,.25,.5,1,2,4,infinity. Preserve
whole numerically repeated eigenspace blocks; archive actual band membership.
Isolated vertices are excluded as in the existing operator. Empty bands and
zero-residual directions are unavailable, not repaired by arbitrary scaling.

For current clean estimate Y_t and its preceding *already observed* PM estimate:
`r_b=||P_b(Y_t-X)||²/(||P_bY_t||²+||P_bX||²+eps)`;
`v_b=||P_b(Y_t-Y_prev)||²/(||P_bY_t||²+||P_bY_prev||²+eps)`.
First checkpoint lacks v and must be flagged unavailable for two-time features.
Use no future checkpoint to construct an earlier feature. These are heuristic
disagreement/drift features; v is not an unbiased noise variance. Repeated
processes estimate native numerical variation separately from temporal drift.

Predeclared score candidates: frequency-only reliability, negative r alone,
and `s=r/(1+v)` as a proposed reference-unreliability ranking (large persistent
departure from X lowers reference trust). An optional illustrative bounded q
is exp(-s), but no q is fed into guidance in this phase and no scale is fitted.
Disagreement can reflect a bad prior; stability can reflect stable bias.
The purpose of the oracle below is to falsify these candidate interpretations.

Only after GPU trajectories/features are fixed, use the verified clean asset
as a CPU diagnostic oracle. Map clean points through the recorded corrupted
normalization/scale/rotation, then score a normalized negative band-fidelity
gradient perturbation of magnitude .001*||Y||. Evaluate symmetric mean-NN
squared clean Chamfer before/after. This avoids inventing matched clean latent
slots or assuming raw clean/corrupted arrays share vertex indices. Float64
quadratic/symmetry and permutation checks remain mandatory. Numerical ties
within1e-12*(1+baselineCD) are unresolved; no label/clean use during generation,
feature construction, band/score selection or future applied coefficient fit.

Report per-band/corruption distributions, Spearman association and AUROC when
both oracle classes exist; otherwise unavailable. Compare feature rankings to
frequency-only and residual-only controls. Cluster uncertainty by8 original
objects, keeping all corruptions/checkpoints/bands together; no population or
heldout accuracy claim. All selected IDs have historical benchmark exposure.
No new score is chosen from a larger posthoc search. A later disjoint feature
validation and an applied q accuracy study require a separately concrete plan.

## Known confound and gates

Background same-observer repeats already failed; do not assume controlled
cuDNN makes native operations deterministic. Use the same prior numerical
gates (logitmax.05, per-object/checkpointRMS.01, firstdecodedRMS1e-4) without
relaxation, and compare repeat floor to temporal feature variation. If native
variation dominates a feature or it fails to predict oracle benefit, H5 is not
ready for an accuracy run. A failure is retained rather than patched by retuning.

## Runtime/setup and batches

New runtime must use all10 canonical setup cells, cell1 separately then reconnect
sameVM and2–10 in order. Only known cell2 fastrlock failure may be deferred;
native/CUDA/data/checkpoint failures stop setup. CLI only; no account changes,
no full research notebook. Assigned peer
`tta-gsd-anchor-correspondence-a100-20261105-r3` is protected.

Allocation r1 requested A100/HighRAM and failed ServiceUnavailable before any
runtime assignment. Same-shape r2 also failed ServiceUnavailable. Standard-RAM A100 r3 allocated
endpoint gpu-a100-s-usw4b0-31aq3a2nhfbij, actualA100-SXM4-40GB. No peer runtime
was reused or closed. Monitor51470/PID91277 owns cells. Cell1 completed17:47:22UTC,
sameVM Conda/hardware proof17:47:48UTC; cell2 started17:48:39UTC and is
running. No H5 research GPU run has started.
Prepared canonical wrapper hashes match all10 previously verified cell sources.
Fresh paths: local `result/preparation/gsd_h5_reliability_20261105/`, remote
`/content/studies/gsd-h5-reliability-20261105`, Drive
`thesis/result/gsd_h5_reliability_20261105`. Actual endpoint/owner/monitor and
source/asset hashes will be recorded after assignment/sealing.

Batch1: handoff branch/prompt complete before allocation (563ad7b; whitespace
check). Batch2: development-agent implements opt-in capture + meaningful CPU
tests, independent read-only review, no defaults changed. Batch3: canonical
setup/Drive/native/data/checkpoint proof. Batch4: seal protocol/IDs/source,
bounded captures, perbatch Drive persistence, archive source/native/cache/NPZ
bytes and success/failure receipts. Batch5: independent CPU oracle/features,
raw SHA/count/RNG/model checks, knowledge findings/open questions and curated
integration. User already selected this diagnostic direction; no extra run
approval for the bounded phase. Runtime closure requires a later user request.

[Run/Code] Current17:52UTC: handoff tipd5869c2, integratedc41cbcd after preserving peer reencode notesb5c76cb. Owned H5r3Standard A10040GB endpointgpu-a100-s-usw4b0-31aq3a2nhfbij; monitor51470/PID91277; cell1/reconnect passed, cell2 running. Two HighRAM allocation failures were503ServiceUnavailable, not GPU research attempts. CPU capture tests12pass; native H5 capture remains unchecked.
