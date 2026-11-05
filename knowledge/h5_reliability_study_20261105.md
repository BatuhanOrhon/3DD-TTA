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

[Run/Open] 17:58UTC: cell2 ended with the specifically deferred fastrlock pip wheel failure; no other failure accepted. Separate proof verified Python3.8.20 at17:56:39UTC. Monitor51470/PID91277 was cleanly stopped17:57:27UTC (runtime retained) to hand exclusive cell/TTY ownership to Drive CLI exec79873. That CLI now waits for actual Google Drive user consent; browser consent page opened. Do not launch any other cell until consent/mount completes. Then restart the sole monitor on the same endpoint, run canonicalcell3 followed4–10. Research GPU capture has not started; no accuracy result exists.

[Run] 18:08UTC: initial Drive mount failed after delayed consent; its kernel call completed before retry. Fresh request exec63312 received user consent, credentials propagated and mount confirmed `/content/drive`; no account/CLI change. Resume canonicalcell3 then4–10. No research model inference yet.

[Run/Open] 18:44–18:45UTC: r3 disappeared from two fresh server inventories; console/status returned session-not-found. Cell4 CLI remained stale after last dependency output18:18:33UTC and beyond configured1800s. No research capture had started; no runtime stop/unassign command was issued. Cause unknown, not certified OOM/idle loss. Preservedlocal wrapper/stream/source and runtime_r3_loss.json; stopped only own stale monitor/CLI. A100HighRAM r4 recovery requested; user authorizes all GPU phases in this task without further approval and closure only once comprehensive tests/archives finish. Peer runtimes remain protected.

[Run/User decision] 19:04UTC: recovery r4 HighRAM allocated endpointgpu-a100-hm-euw4a0-2rsp767om7kzr; actualA100-SXM4-80GB, cell1 and sameVM reconnect proof passed. Sole monitor exec40692 replaces idle PID588 (no remote interrupt); fresh canonicalcell2 running. New ownership_r4.json records identity. User explicitly preauthorizes whole comprehensive GPU phase and conditional own-runtime closure after completed tests/archives; this supersedes earlier future-approval/closure wording, not human Google account consent. H5 full CPU suite517tests/1skip, no actual captures yet. Peer halfmix CPU publication origin1ccda41/7fc231d remains separately user-held pending session availability; no duplicate GPU work.

[Run/Open] 19:15UTC: r4 canonicalcell2 failed solely at fastrlock0.8.4 wheel build (raw CLI execution history retained), independently verified Python3.8.20. No other setup failure deferred. Plain exec mount adaptation had no CLI Drive-auth hook and waited for an unanswered frontend request; corrected using own restart-kernel, which interrupted that known setup call (KeyboardInterrupt), preserving VM/files/environment. Both corrective calls ended, then proper `colab drivemount` exec86410 became sole cell owner. It now requests actual human Google Drive consent for this new endpoint; user notified asynchronously and browser auth page opened. Monitor40692 cleanly stopped for exclusive TTY ownership, runtime retained. Canonicalcell3 and4–10 remain unrun; no H5 research capture/new accuracy. Do not send Enter before user confirms account consent, bypass Drive receipt or claim GPU-run approval is missing.

[Code/CPU] Frozen firstH5overlay atb766e61 contains249files,693404bytes/SHA99cd1bcf4e6a1101120c903468a03bcc73951287db99d80361aa965dc063ee57. Independent review verified exact16oldcachepaths and frozen CLI scope/deadline. Local operator wrappers are prepared, not executed research. Separate native-trace opt-in CPU suite521tests/1skip; native execution remains unchecked. Comprehensive remaining H1–H8 matrix links `gsd_comprehensive_test_matrix_20261105.md`; completed spectral-z grid and peer held-halfmix scope are preserved.

[Code/Publication/Open] CPU capture/analyzer/trace and campaign records were curated into latest7fc231d and normally pushed as39a8d0928700f386fbe0354262b8fadefdaa1092; actualremote ref verified. Peer lowband code/knowledge preserved; merged trace additionally rejects nonunit decoded_scd_weight. Merged fullsuite528tests/1skip in51.971s,31focused plus independent40trace checks; originalguard rejection witness red/green verified. Compactpublication.json records selection/exclusion and logSHA. This publishes completed CPU preparation, not H5/GPU results. Ownr4 remains assigned and awaiting new-VM human Drive account consent; proper soleCLIexec86410 owns the pending mount, localmonitor40692 stopped for TTY exclusivity. After user confirms, inspect/resume this mount (retry fresh consent if expired), restart solemonitor, canonicalcell3 then4–10, setupverification/strictCPUdeployment and only then frozenH5captures. Do not close own VM until actual comprehensive tests/archives are complete.

[Run/Open] 19:51UTC: user requests fresh Drive link if runtime live. Fresh server inventory confirms same r4/endpoint assigned. Old36min mount request cancelled using own-kernel restart; stale localauthCLI3448 stopped only after restart, no VM/peer teardown. New properCLI drivemount exec95845 waits fresh human consent; URL supplied/opened, no account credentials stored in knowledge. ScienceGPU remains unrun; monitor stays stopped for authTTY exclusivity. Resume this exact mount after actual consent, verify canonicalcell3, restart solemonitor and continue4–10.

[Run/User decision] 19:56UTC: user confirms fresh Drive consent and again authorizes continuation through comprehensive tests/analysis then ownGPUclosure. SoleCLI95845 propagated credentials and mountedDrive. Restarted solemonitor26139 with full localJSONLtee, canonicalcell3 exactmarker passed19:55:29UTC and KNNwheelcopySHAequal. Canonicalcell4 now running; execution-only60sUTC heartbeat outerwrapperSHA47e3e06d521edb8e9a265e06f410dde3159a225d6d3a16f3329d1645aaa95b78 preserves every saved canonical source/command and checked-shell failure behavior. Timeout2700s, nativeprepare marker still required;5–10 pending. No scienceGPU capture yet.

[Run/Recovery] 20:18UTC: canonicalcell4/prepare passed20:08:18UTC andcell5/EMD20:11:14UTC. Setup-only backupCLI lostWebSocket20:12:08 and hung locally. Read-onlyrawconsole confirmed sameVM, prepare/emdmarkers, mountedDrive, correctuploadedZIPSHAs and allkernelstatesidle; no researchworker. The backupdestination didnotexist. Stopped onlyownstaleCLI10978/monitor26139 after this proof; new solemonitor66602 reconnected, read-onlyinspection passed, then exactfresh-outputbackupretry completedmarker20:17:58UTC and verified actualDrivecopies. No kernel/VM teardown, no researchduplication; roottransportcauseunresolved. Canonicalcell6 follows;5–8 are not relabelled from markersalone.

[Run] 20:46UTC: r4 freshcanonicalten-cell replay nowverifiedthroughcell10 (cell2knownfastrlockdeferredonly; step4pinneddependenciesrepaired). cell7PointNet++20:24:38, cell8PVCNN+nativeGPUFPS/gatherbackward20:33:30, cell9datasetcopy/CRC/extract20:36:06, cell10fullverifier20:44:21 statuspassed/all15arrays. Firstcell10attemptlostconnectioninCLIprelude execution.py217 beforebody; rawconsole provedallkernelsidle/noactiveinstaller/checkpointfilesabsent, ownstaleCLI14570/monitor66602stoppedonlylocally. Newsolemonitor77067/read-onlyreconnect thenexactcell10retrycompleted; sameVM/Drivemount preserved. Actualcanonicalreceipt nowbeingDrivepersisted inreviewedCPUdeployment; frozeninitialH5overlayb766 remainsfirstsciencepayload.

[Run] 20:48UTC: reviewedinitialH5CPUdeployment/strictno-forward/no-CUDA proof passed20:46:54. Assetmanifest182211bytes/SHA18968fce29b4318233dc2bbdc2349d1b90081a9fddd8a3614b68e0f312c02813; allfreshnativebytes backedDrivebeforepinnedrestore,707580bytes/SHA31f9a43a2ca20f890d0bf1f60aa327fb50477319aa511dc484d6b459ecdfa3a7. Newsolemonitor53240 inheritslocal REQUEST_TIMEOUT=60 (installedclientdocumentedstartupknob; notremoteenv), unchangedaccount/CLIinstall/runtime/kernel. This testslongerstartupwindow, notclaimedroottransportfix. InitialH5sourceb766/overlay99cd…ee57/operator482c…f88 nowlaunched3freshcontrolcaptures withfixed2700scap; nohypothesisscoretuning/oracle. Laterprobeoverlay254files/731868bytes/SHA809573e8351877b8de4408090cf64300de5696dcf3b12a18911c14eba594b713 remainsseparate, notyetdeployed. FullnewCPU suite554tests/1skip39.573s, logSHA7e6aa74c785ffa5a7019adb92357ca0ab8a4b3d1813b261ae4d1592ff00a496e.

## 2026-11-05 21:32UTC — H5 accepted and supplemental mechanism running

[Run/Decision] Canonicalsetup all15 finalchecks passed on ownr4 A10080GB; Drive authorization verified. H5 three fresh zero-weight control captures completed20:55UTC. ActualDrive/local raw package1,391,073,853bytes SHA256`ee58f23c92546858b7c82220e5c353f4aae68a6997df8e69289e3541dadbe105` accepted. [Detailed numerical validation and results](h5_reliability_results_20261105.md): all three native repeat comparisons fail. Frozen combined reliability AUROC.271341, whole8object cluster95%CI[.137627,.386096]; paired superiority gates fail. H5 applied guidance is NOT_ADVANCED under the previously frozen rule; no score inversion, tuning or accuracy gain claimed.

[Run/Open] Supplemental source72e0eb4/overlay809573e8351877b8de4408090cf64300de5696dcf3b12a18911c14eba594b713 deployed without native rebuild. Six H0 baseline traces and one four-ID shared first/last PM capture complete21:15:28UTC; seven actualDrive ZIPs downloaded and byte/SHA verified under`result/preparation/gsd_comprehensive_20261105/received_captures_v2/`. H1/H2/H3/H6/H7 common-state probes began21:30:47UTC using separately archived validation-v2 operator; strict original numeric gates and raw q/U order unchanged. Metadata-only first wrapper failed with undefined`__file__` before subprocess/model/GPU work; retainedv1, correctedrunpy wrapper and freshmetadata_v2. H4 technical equivalence/componentcost and frozen posthoc PointMAE readouts remain planned. Sole monitor53240; ownruntime endpointgpu-a100-hm-euw4a0-2rsp767om7kzr remains live. User authorizes campaign and ownclosure onlyafter completed/classifiedtests and verifiedarchives.

[Run/CPU/Decision] 2026-11-06: H5 and all mandatory supplemental diagnostics completed, including exact-input prior witness; conditional phases NOT_ADVANCED. [Final campaign result](gsd_comprehensive_results_20261106.md).18raw/derived archives verified actualDrive/local; final derived backup/publication and ownr4 closure are next. No hypothesis/score/tolerance retuning or new benchmark claim.

[Run/Publication] 2026-11-05 22:04UTC: all required analyses and conditional decisions completed, sharedscience published b8783f3, finalderivedDrive/local195file roundtrip verified; ownr4 explicitlyclosed and serverabsence verified, monitor53240exit0. [Final result and closure evidence](gsd_comprehensive_results_20261106.md).
