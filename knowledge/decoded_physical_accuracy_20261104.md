# Decoded physical spectral guidance: applied accuracy pilot, 2026-11-04

**[Code/Run/Open] Status: requested pilot complete; raw evidence accepted; main numerical-attribution gate failed; no promotion.** User asked to continue the next tests. This stage follows the accepted [P0/P1/P2 follow-up](decoded_physical_followup_20261104.md). Prior derivative agreement does not establish accuracy improvement.

## Accepted raw results and failed numerical gate

**[Run/Inference] No established probability-mean accuracy gain; keep the baseline.** All20 raw runs,964B2 cases,241 actual paired caches,248 source files and6 native ELF payloads passed independent local integrity/protocol/count checks. The main compatibility gate failed45/240B2 cases; this is retained as a failed scientific control, not repaired by relaxing thresholds. An independent second reviewer reproduced all counts, paired CIs, timings and physical-filter equality.

| Arm | PM correct/480 | PM accuracy | Terminal accuracy | Reconstruction s | Graph s | Worker wall s | Measured peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Operational SCD+PM | 313 | 65.2083% | 64.5833% | 334.593 | 0.000 | 439.117 | 3752.104 |
| Live observer, weight0 | 315 | 65.6250% | 64.5833% | 339.572 | 25.369 | 526.247 | 4450.551 |
| SCD + full physical | 313 | 65.2083% | 65.4167% | 385.266 | 25.375 | 572.049 | 4565.966 |
| SCD + H8 physical | 312 | 65.0000% | 65.4167% | 340.354 | 25.369 | 527.389 | 3783.808 |

All rows use the same32 objects/all15 corruptions/seed0. PM is the frozen primary readout; the terminal result does not replace it after observing outcomes. Full/H8 both score314/480 terminal, versus310/480 controls, but this is a secondary readout under the failed numerical control. [All per-corruption counts](evidence/decoded_physical_accuracy_20261104/per_corruption.csv).

| Paired PM contrast | Delta pp | Object-cluster95% CI pp | Corrected / harmed |
|---|---:|---|---:|
| full_vs_live0 | -0.4167 | [-1.6667, +0.8333] | 5 / 7 |
| h8_vs_live0 | -0.6250 | [-1.6667, +0.4167] | 4 / 7 |
| full_vs_control | +0.0000 | [-1.2500, +1.2500] | 6 / 6 |
| h8_vs_control | -0.2083 | [-1.2500, +0.8333] | 5 / 6 |
| h8_vs_full | -0.2083 | [-0.8333, +0.4167] | 1 / 2 |
| live0_vs_control | +0.4167 | [+0.0000, +1.0417] | 2 / 0 |

Intervals resample32 original objects, retaining all15 corruptions;10,000 draws, seed20261101. They are pointwise, conditional development-cohort intervals at one inference seed. They do not cover unquantified native numerical instability or historical test exposure. Negative point estimates do not establish harm; none of the physical-versus-control intervals establishes a gain.

### Efficiency and mathematical interpretation

- H8 records11.6575% less reconstruction time than full; graph+reconstruction10.9386% less, worker wall7.8071% less, and the measured post-graph peak17.1302% lower. These descriptive totals are from one prepared-cache B2 pilot, not deployment throughput or confidence intervals on timing.
- Against operational SCD+PM, H8 records1.7217% more reconstruction time, **9.3036% more graph+reconstruction time**, and20.1022% more worker wall. Thus the present additive SCD+physical objective does not achieve simultaneous accuracy and speed improvement over baseline.
- Main encoder/noise preparation was common and outside arm timers:16.3016s summed preparation boundaries,21.9125s loop including copies,42.1015s outer process including loading. Graph peak memory was not measured because the existing counter resets after graph construction. Worker wall includes setup/IO through the loop but excludes final sealing/publication.
- The physical reference/filter hashes are exactly equal across live0/full/H8 for all240B2 cases. This supports a consistent frozen quadratic metric, not a causal claim that the corrupted anchor or vertex correspondence is correct. See the [GSD math audit](gsd_guidance_math_efficiency_20261103.md) and [actual-GPU-filter anchor counterexample](decoded_physical_followup_20261104.md).
- H8 and full are not prediction-equivalent:468/480 PM predictions agree,448/480 terminal predictions agree. Of12 PM differences, H8 corrects1 and harms2;9 differ while both are wrong. Similar gradient directions/accuracy do not prove equivalent trajectories.

### Numerical failure that changes the next gate

The old two-ID Gaussian technical gate passes again (worstlogit1.4305e-5), but it does not generalize. Main control/live0 fails45/240B2 cases, including **16/16Background** cases. Worstlogit difference3.3738575, worst per-object/checkpoint RMS1.1059471, first decoded per-object RMS up to.07278594. Control/live0 differ on2/480 PM class predictions and3/480 terminal predictions. The two PM changes are Background ID905 (26→24,label24) andID1305 (25→28,label28), yielding a zero-weight+0.4167pp shift.

Source/native/checkpoint/model states, encoded states/noise, input and post-reconstruction RNG identities match. First clean-latent hashes already differ before the first decoder probe in240/240 cases; original decoder-style hashes match240/240. Hash mismatch measures bit differences, not their magnitude. This localizes some variation before the first decoded-loss evaluation and prevents blaming only the later decoder gradient branch. Custom native/discrete-neighborhood/roundoff amplification remains a hypothesis; no kernel is causally identified and no speculative numerical patch was applied. Controlled cuDNN flags are not a general reproducibility guarantee.

The main numerical-attribution gate remains **failed**, even though raw integrity and all physical-metric equality checks pass. Small observed accuracy differences are descriptive and cannot establish an isolated guidance effect. Preserve the narrower previous P0 result as valid for its tested cases, superseded as a general stability assumption.

### Decision and testable next steps

1. **First, numerical diagnosis:** freeze selected failing Background/LiDAR cases plus a passing case; repeat the same live0 observer in fresh processes, then isolate the earliest prior/SCD boundary with exact cached inputs, outputs and discrete indices. Falsifier for a decoder-only explanation: divergence before any decoder probe, already suggested by the hashes. This needs new bounded native runs; none was launched after this phase.
2. **Then, anchor bias:** compare the corrupted prepared-X anchor with its frozen LION encode/decode reconstruction under the same physical graph, holding every other component fixed and calibrating update scale without labels. This tests whether raw corruption is being reintroduced; the present null pilot alone does not prove that cause.
3. **Check correspondence:** on frozen states, compare slot residuals with a declared geometric correspondence and permutation witnesses before another spectral coefficient search. An identical F does not establish that encoder/decoder vertex indices are semantically aligned.
4. **Conditional efficiency work:** if a valid objective improves accuracy, compare the exact heat quadratic with its sparse square-root polynomial approximation and measure fresh preparation plus all end-to-end stages. H8 input-detachment remains unpromoted; this phase did not test it.

No coefficient search, default promotion, full-benchmark claim, seed replication, or further GPU phase. Keep the prepared runtime and monitor open.

### Evidence and publication

Raw package `result/preparation/gsd_physical_accuracy_20261104/accuracy_raw_package.zip`,598124006bytes, SHA`25848ee32f6fab40dd2b58e5b428ed61f78da3a0899f92c3038e291bb81b3111`; Drive `/content/drive/MyDrive/thesis/result/gsd_physical_accuracy_20261104/accuracy_raw_package.zip`.407payloads+package manifest verified; canonical20 raw ZIPs retained under `result/modelnet40_c/decoded_physical_spectral_diagnostic/`.

[Main recount and gates](evidence/decoded_physical_accuracy_20261104/main_results.json), [compact costs/boundaries](evidence/decoded_physical_accuracy_20261104/compact_summary.json), [raw acceptance](evidence/decoded_physical_accuracy_20261104/local_acceptance.json), [technical recount](evidence/decoded_physical_accuracy_20261104/technical_results.json). Analysis scripts retain exact executed bytes; `accept_download.py` and `summarize_accuracy.py` expect their original `result/preparation/gsd_physical_accuracy_20261104/` layout and raw inputs.

Source GPU commit `cdbdfb1995296b08bcc4345161af6c01213f4b2f`; analyzer `bd335d8`, SHA`33b5fcab3ca9dd0c0090dd09c2b91ed0175a008c633524d64f37d0c53da6da37`. CPU507tests/1skip passed; final16 affected tests pass in the clean integration checkout. Whole GPU phase2497.5627s (41min37.6s), under55min cap. Derived Drive backup and final integration identity are recorded in the publication receipt below.

## Frozen protocol before execution

Question: does the frozen physical decoded-space loss improve probability-mean accuracy, and does H8 retain that behavior while reducing decoder-backward cost? This is **SCD plus physical guidance**, not replacement of SCD or full GSDTTA.

- ModelNet40-C, all15 canonical corruptions, severity5, PointMAE/LION raw eval, EMA off; seed0, B2;32 development IDs in four immutable8-ID shards. Selection uses SHA256 rank with salt `decoded-physical-accuracy-20261104-v1`, excluding the prior five stage cohorts, old pilot IDs, and follow-up8. [Selection and exclusions](evidence/decoded_physical_accuracy_20261104/development_ids.json). No label/clean-data selection. Historical full-test exposure remains; this is not unseen confirmation.
- Original SCD sum, gamma.03/eta.003/p.95; DDIM100/final-alpha1; normal8/background35 reverse steps; clean PM checkpoints2/4/6 or9/18/27 plus terminal. Controlled cuDNN backend as in the accepted follow-up; custom CUDA atomics remain uncontrolled.
- Four arms: `control` (no-grad operational SCD+PM), `live0` (grad-enabled physical observer, zero applied loss), `full` (full decoded-space derivative), `h8` (same decoder values, frozen residual surrogate). Full/H8 use the previously frozen label-free coefficient **802.5000841164167**. No accuracy-based retuning, decoder input-detachment optimization, default promotion, or additional coefficient search.
- Same frozen prepared-input graph: k10/delta.1/graph-gamma.6, smooth heat profile beta2 with all active modes; exact prepared encoder input versus decoded PM geometry. Fixed reference and vertex-correspondence assumption retained.
- Prebuild every main paired encoded state/noise cache using the existing sampler's preparation callback, stopping before reverse diffusion. All measured arms replay exact same caches. Preparation cost is separate and common. First technical GaussianB2 uses prior IDs44/91 and their pinned cache.
- Technical four-arm gate: unchanged source/native/model/prepared/state/RNG identities; finite complete outputs. Control/live0 per-object/checkpoint logit max≤.05, RMS≤.01, first decoded RMS≤1e-4. Same gates checked on main all15; failed equivalence limits isolated attribution rather than silently changing thresholds. Positive full/H8 need not have identical outputs.
- Main orders rotate across shards: control/live0/full/h8; live0/full/h8/control; full/h8/control/live0; h8/control/live0/full. Compare full and H8 primarily to live0; separately report operational comparisons against control and H8 versus full.
- Outcomes: independent PM/terminal recount; per-corruption/all15 accuracy and paired differences;10,000 bootstrap samples (seed20261101) of32 original objects, retaining all15 outcomes per object. One seed, development-only pointwise intervals, no simultaneous/multiple-comparison guarantee. No treating480 observations as independent objects.
- Timing: report graph-build, reconstruction/PM, per-case elapsed, worker wall, common cache preparation and outer process wall separately. Graph construction is outside existing reconstruction timers; peaks reset after graph construction, so graph-build peak is unmeasured. Warm prepared-cache timing is not deployment throughput. No end-to-end speed claim from reconstruction alone.
- Whole GPU phase, including preparation and technical runs: estimate30–45minutes, hard55minutes from the first GPU invocation. No new run if remaining budget is insufficient. Accept only complete common four-arm/all15 shards; preserve all partial/failed artifacts and never choose inclusion by scores. No >60minute phase authorized or planned.

## Execution and persistence

Owned worktree `3DD-TTA-worktrees/gsd-guidance-math-audit-20261103`, branch `research/gsd-physical-accuracy-20261104`, based on `origin/thesis-integration`73e117a. Minimal driver change permits canonical all15 inputs while preserving existing defaults. Source is sealed before deployment.

Reuse owned session `tta-gsd-physical-spectral-a100-20261103-r1`, endpoint `gpu-a100-hm-ass1c0-i933qghuewn8`, A10080GB; monitor97932/PID34871 owns cells. Canonical environment was already verified. New remote root `/content/studies/gsd-physical-accuracy-20261104`; Drive `/content/drive/MyDrive/thesis/result/gsd_physical_accuracy_20261104`; local `result/preparation/gsd_physical_accuracy_20261104/`. Fresh immutable deployment and output directories; per-batch Drive copies and sealed raw bundles. Source, native binaries, caches, commands, success/failure receipts retained. Runtime remains protected; no teardown permission.

Next gate: CPU deployment/source/native/data verification, technical comparison, then bounded main phase. A positive pilot requires stage-disjoint multi-seed confirmation and fresh deployment timing before a final claim; a null or harmful result rejects promotion at this frozen coefficient without proving every physical spectral objective impossible.

CPU deployment accepted07:45:20UTC: sourcecdbdfb1, sourceoverlay682718bytes/SHA`a85a5c794ddf8003ba7302c153f98c9381deb18188bc75f6f255e65209f0ebdb`; assetmanifest181793bytes/SHA`05b6fcde7c6982954594bec34fd6bd7d62b8b020f86129d2d8d16fafd8c0363c`. Technical operatorpackage20883bytes/SHA`c6e097f2a7f95c93d374f1f0df079a127a6c9347fa03f8a05e567776a96f1b41`. Immutable phase deadline08:43:34UTC. Preparation caches are copied/verified to Drive per B2, with fresh per-corruption/shard prefix receipts.

[Run] Technical four-arm phase complete07:50:56UTC. Control/live0 worstlogit1.4305115e-5, per-object/checkpointRMS4.6221261e-6, firstdecodedRMS≤9.212753e-8; all fixed gates pass, both PM predictions agree. All four arms2/2 PM/terminal on oldGaussian2, not accuracy evidence. Reconstruction control/live0/full/H8=1.617304/1.624602/1.821383/1.633384s, respective measuredpeaks3081.547/3779.524/3894.674/3113.594MiB; graph excluded and oneB2 insufficientforperformanceclaims. Strict local reanalysis of raw archives pending. Main preparation and16 paired runs launched07:52:26UTC under the unchanged clock.

[Code] Final analyzer commitbd335d8, SHA`33b5fcab3ca9dd0c0090dd09c2b91ed0175a008c633524d64f37d0c53da6da37`; independent technical/main review passed; full CPU507tests/1skip in39.286s and final6focused tests pass. Main source and coefficients unchanged. First main control/full raw bundles downloaded early and accepted independently (60B2cases each, completeall15/8IDs, matching source/native/model/prepared/state/RNG); no outcome-based run decisions.

[Run] All4technical+16main runs completed, no GPU failure/retry or scope change. Immutable scienceclock elapsed2497.562670469284s (41.626min), below3300s cap. Common cache preparation240B2states and main960B2cases complete; archival CPU collector started08:30:28UTC. No accuracy claim before independent local recount.
