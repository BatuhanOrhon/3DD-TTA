# Agent B: matched guidance components and accuracy/cost development

Date:2026-10-25. Evidence: [User report], [Code], [Run], [Inference], [Open].
Planning base:`gsd-multibackbone-study@47e69be`; inspect/pin actual committed
base after this documentation batch. Status: planned only; no experiment launched.

## Objective, evidence and attribution

[User report] Determine whether selective Chamfer, spectral guidance, their
combination or neither gives the most useful accuracy/cost tradeoff when using
terminal, minimum entropy and equal probability mean. Explore both datasets
and frozen source heads; small method improvements and bounded eta/gamma/
lambda experiments are permitted. Spectral improvement is desired, but a
supported Chamfer-only or null result is also a valid research outcome.

[Run/Inference] Current GSD terminal is below original3DD in all six matched
new-backbone cases. Scan probability mean exceeds original for CurveNet/DGCNN,
not PointNeXt; full matched Point-MAE Scan source/original rows are unavailable.
Thus a universal superiority claim is unsupported. Original3DD also changes
batch/horizon/style/scheduler, so it does not isolate spectral contribution.

[Run] Earlier Point-MAE original-style ablations already cover three seeds:
unguided61.2021%, smooth-only61.3596%, historical SCD63.8799%, combined63.8520%
on ModelNet. Build/source and host differences limit causal reuse; these are
not missing tests and not the current B70/M800 trajectory-policy ablation.
Read [existing-run audit](gsd_existing_runs_audit_20261003.md) and
[accepted ablations](gsd_guidance_ablation_results_20261004.md) before launches.

[Decision] Follow the
[parallel ownership contract](parallel_colab_agent_contract_20261025.md):
branch`gsd-guidance-components`, separate worktree and sessions prefixed
`tta-guidance-components-a100-<UTCdate>-r<N>`. Agent A's ModelNet readout task
must remain unchanged and untouched. Preparation here does not activate Agent B.

## Matched component matrix

Use the current GSD reconstruction host as the controlled reference, not the
original3DD host. Begin with the locked B70/M800/beta2, eta=gamma=.01,
p=.95, raw/eval LION/EMAoff, updated style, DDIM finalalphaFalse,
normal10/Background30 and existing sparse probes. Invariant weight remains0.

| Arm | Selective Chamfer weight | Spectral weight | Scientific role |
|---|---:|---:|---|
| C+S |1|1.17|current combined reference|
| C |1|0|incremental spectral contribution conditional on Chamfer|
| S |0|1.17|spectral alone; effect of removing Chamfer|
| U |0|0|unguided DDIM control; not source-only or direct VAE reconstruction|

For every arm record terminal, minimum entropy and equal probability mean from
the same three clean probes plus ordinary terminal. Use each arm's own terminal
for readout effects and identical readout policies for arm comparisons. Saved
candidate logits allow this without three independent diffusion jobs per arm.
Guidance trajectories differ across arms, so reconstructed inputs need not match.

Scope: ModelNet40-C2468/40classes and ScanObjectNN-C581/15classes `main_split`,
all15 severity5 for final evaluation; DGCNN, CurveNet, PointNeXt and Point-MAE.
Screen on small, label-free selected fixed development IDs across both datasets
before expanding; seed0 first is exploratory. Record exact subset/seed manifests
and whole phase budget before GPU execution. Full four-arm × four-head × two-
dataset evaluation would yield96 policy rows per seed and is not an automatic
first launch. Stage it according to evidence, runtime and user approval.

[Code] Current selective Chamfer and spectral fidelity operate on local latent
coordinates, not decoded physical clouds. Preserve that objective for the fixed
component matrix; moving a loss into physical space is a separately named change.

Keep the same input IDs, checkpoints, preprocessing, batching, horizons, native
build and environment across matched arms. Share prepared inputs, encoded states
and perturbation noise where technically correct, with hashes and isolation;
equal seeds alone do not establish common draws with native nondeterminism.
Head-independent guidance permits shared reconstruction across frozen heads.
Reuse peer C+S results only if their full protocol/source/assets/IDs are compatible;
otherwise label them historical and capture the small missing matched control.
Do not repeat accepted full accuracy merely for a new launcher or log format.

## Parameters and runtime: inspect actual semantics

[Code] In `tta_gsd_v2.py` actual updates use `eta` for the local latent and
`gamma` for style. The function docstring's descriptions are reversed; use the
executed equations and `legacy_gsd_artifacts.py:update_rates` contract.
CLI `lambdaa` is passed as `p`, the retained Chamfer-distance fraction, not a
generic loss multiplier. Spectral/chamfer loss weights are separate; graph
`gamma=.6` is also separate from style-update `gamma=.01`.
The retained-fraction parameter p has no active Chamfer role in a spectral-only
arm; apparent sensitivity there would require checking the implementation.

Log component losses, weighted gradient/update norms and reductions together;
equal numeric loss weights do not imply comparable guidance magnitude. A weak
spectral-only arm at the transferred coefficient does not establish that a
properly calibrated spectral-only method cannot work.

[Code] The graph/eigendecomposition is currently computed even with spectral
weight0. Chamfer is instantiated only when its weight is positive. Both active
losses feed one combined backward; loss removal does not imply proportional
wall-time savings. Distinguish numerical ablation from measured deployment cost.

First verify component switches against zero-weight/reference behavior on the
same prepared states. For real timing, skip inactive graph/eigh or Chamfer
kernels and diagnostics, and skip guidance backward when both weights are0.
Verify numerical/state/RNG contracts after these changes. Preserve default
combined behavior; do not silently change the peer's or legacy implementation.
Do not add loss wall times as though the shared backward were independent.

Measure synchronized GPU and end-to-end time, warmup, throughput, peak memory,
graph/encode/prior/backward/decode/classifier/IO costs and observer overhead.
Report matched hardware/probe counts/batch settings. No early-stop speed claim;
removing one component may improve, worsen or leave cost nearly unchanged.

## Selection, tuning and claims

[User report/Decision] After the fixed matrix, allow a small predeclared search
over eta/gamma/p or guidance weights and a justified small spectral modification.
Change one factor at a time first; name each modified method and compare it with
the unmodified arm under the same search budget. Predeclare candidate ranges,
max trials, development IDs, seeds, stopping/failure rules and compute cap.
Full Cartesian sweeps or unlimited optimization are not implied authorization.

Use balanced mean over dataset/backbone cases, each case's equal-weight all15
macro; do not let ModelNet's larger example count dominate a pooled micro mean.
Report accuracy and deltas versus own source where available, C+S and C; retain
per-case/per-corruption signs and Pareto accuracy/cost tradeoffs. Missing matched
Point-MAE Scan source/original values stay missing unless separately measured
and justified; they do not prevent component-to-component comparisons.

Keep tuning/development evidence separate from held-out evaluation. Freeze a
single candidate/policy before confirmation on held-out IDs and additional seeds
when feasible. Existing benchmarks already informed development; disclose this
and do not call a test-selected maximum unbiased generalization. No per-corruption
or per-test-label routing. Record unsuccessful spectral variants and support
the arm actually favored by evidence. A null increment does not universally
prove a guidance component unnecessary.

Split by original object IDs consistently across all corruptions so the same
object does not appear in both development and prospective held-out subsets.
Previously inspected full-benchmark results mean this is not an entirely
unseen external test set; record that limitation explicitly.

## Ordered handoff batches

### Batch 1 - Isolation, coverage and concrete phase proposal

- Goal: own workspace/runtime scope and identify genuinely missing evidence.
- Scope: read prior artifacts; register branch/worktree/ownership; no research GPU.
- Touched areas: this note, task-local preparation/ownership records.
- Stack context: local CPU inspection and Colab CLI inventory/setup if requested.
- Dependencies: shared ownership contract and original workspace's current skill.
- Implementation notes for $development-agent: preserve peer runtime/source,
  user edits and accepted archives; pin actual source/assets and study IDs.
- Verification: existing-row/ID/hash audit; cost and full proposed phase scope.
- Knowledge artifact to update: this note and branch-local findings/open questions.
- User review gate: >60min whole GPU phase needs one concrete approval before launch.
- Out of scope: peer execution, full sweep, automatic teardown.

### Batch 2 - Strict component/readout switches and smoke

- Goal: implement smallest C+S/C/S/U path plus dataset-correct observation.
- Scope: isolated opt-in controls; ModelNet40/Scan15 support; no default-policy drift.
- Touched areas: runner, observer/analyzer and minimal guidance skip branches.
- Stack context: existing Python/PyTorch/CUDA stack, own prepared Colab VM.
- Dependencies: Batch1 inventory and phase approval where required.
- Implementation notes for $development-agent: preserve actual parameter semantics,
  preprocessing, frozen states, shared draws and independently verified skip paths.
- Verification: focused CPU contracts, old-mode regressions and genuine GPU smoke;
  exact shapes, candidate purity, component gradient/update presence/absence, receipts.
- Knowledge artifact to update: this note, findings/open questions and sealed smoke.
- User review gate: smoke acceptance before broader execution; no approval repeats
  within unchanged authorized scope.
- Out of scope: tuning before mathematical controls pass.

### Batch 3 - Matched development matrix and measured cost

- Goal: identify which component/readout combinations merit confirmation.
- Scope: fixed shared development IDs,4 arms,3 readouts, both datasets/frozen heads.
- Touched areas: task-local run configs, immutable outputs and paired analysis.
- Stack context: exclusively owned GPU; one reconstruction per arm across heads.
- Dependencies: Batch2 smoke and whole-phase duration/approval gate.
- Implementation notes for $development-agent: reuse compatible evidence; distinguish
  paired control from historical reference; profile actual skipped computation.
- Verification: ID/count/state/source/native/receipt checks, per-case deltas and time.
- Knowledge artifact to update: this note and branch-local accepted result tables.
- User review gate: review evidence/cost before expanding or materially changing phase.
- Out of scope: universal necessity/speed claims from a small seed0 screen.

### Batch 4 - Bounded development and frozen confirmation

- Goal: strengthen a justified candidate and assess reproducibility/accuracy-cost tradeoff.
- Scope: registered small parameter/method trials, then frozen held-out/full evaluation.
- Touched areas: opt-in candidate changes, trial ledger, validation manifests and analysis.
- Stack context: same owned runtime if valid; fresh phase budget if scope changes.
- Dependencies: Batch3 findings, selected search limits and explicit >60min approval.
- Implementation notes for $development-agent: equal tuning opportunity for comparators;
  report unsuccessful trials, all15/macro14 and source/normalization differences.
- Verification: paired per-case/per-corruption effects, repeat seeds where feasible,
  exact Drive/local SHA and actual performance measurements.
- Knowledge artifact to update: this note, findings/open questions/README and final matrix.
- User review gate: candidate freeze before confirmatory scoring; user-authorized closure only.
- Out of scope: claiming an optimal/general solution from a test-set maximum.

## Required references and acceptance

Read `AGENTS.md`, `/Users/batuhan/.codex/RTK.md`, current original-workspace
research/Colab skills, `knowledge/README.md`, `thesis_scope.md`,
`experiment_protocol.md`, `result/README.md`, then:

- [Final accepted results](gsd_multibackbone_final_findings_20261019.md),
  [Scan readouts](gsd_scan_trajectory_selection_20261019.md),
  [ModelNet A scope](modelnet_multibackbone_trajectory_handoff_20261025.md).
- [Existing-run audit](gsd_existing_runs_audit_20261003.md),
  [historical component results](gsd_guidance_ablation_results_20261004.md),
  [trajectory selection/cost discussion](gsd_trajectory_selection_20261011.md),
  [paper reading list](gsd_trajectory_attribution_20261025.md).
- Code: `tta_gsd_v2.py`, `graph_spectral_v2.py`, `eval_gsd_tta_v2.py`,
  `legacy_gsd_artifacts.py`, `scripts/eval_multibackbone_guidance.py`,
  `shared_trajectory.py`, `scripts/analyze_legacy_trajectory.py`,
  `scripts/analyze_scan_trajectory.py`, `scripts/eval_multibackbone_source.py`,
  `classifier_backbones.py`, `scripts/run_gsd_guidance_ablation.py`.

Use a new task-local source snapshot and run IDs, with batch/corruption/model
Drive persistence and verified local SHA. Accepted control/tuning batches are
separate commits on Agent B's branch. Final deliverables distinguish fixed
ablations, development winners and held-out/seed confirmation, plus actual
runtime/memory and remaining limitations. No new result exists at planning time.


## Agent B activated — 2026-10-25

[User report/Code/Open] User assigned this session to Agent B. Own branch/worktree was created from committed base `4c5be0eaf0ac933923129fe26738e0db5f96744a`; original user edits and Agent A worktree remain untouched. Existing Point-MAE original-host ablations are reused as historical context, not relabeled matched B70/M800 controls. Initial fixed screen:70 label-free development IDs/dataset across15 corruptions, four arms/four heads, three readouts;70 object-disjoint IDs reserved per dataset. Exact manifests and [execution plan](../result/preparation/guidance_component_study_20261025/execution_plan.md) are prepared. Whole initial research GPU phase estimate25–45min, cap55min including smoke/screen/retries; real smoke projection gates screen. No research GPU has run; no accuracy or speed result yet. Tuning and full benchmark remain future scoped decisions. Evidence artifacts: `result/preparation/guidance_component_study_20261025/`; separate Drive root per contract.


### Agent B CPU controls accepted — 2026-10-25

[Code/Open] On own branch from `54ae61f`, opt-in C+S/C/S/U controls, strict ModelNet40/Scan15 observation/analyzer, nonprefix original-ID selections and immutable z0/h0/noise plus post-preparation RNG replay are implemented. Default combined/terminal equations and dataset coordinate contracts are preserved. Inactive graph/Chamfer/backward call counters verify genuine skipped computation in CPU fake-native tests. Component run IDs include task+arm. Final local seven-module suite passes43 tests; independent review found no blocking issue before the final run-ID test. Source checks include Python3.8 syntax and `git diff --check`. This is CPU contract evidence, not native CUDA/model accuracy. Smoke will measure encode-inclusive normal execution; development cache replay timing is separately labeled and cannot establish deployment speedup. No research GPU run yet; own runtime remains in setup. Raw CPU log: `result/preparation/guidance_component_study_20261025/local_cpu_tests.log`. GPU smoke plus physical Drive/local receipts remains the next acceptance gate.


### Agent B ten-cell setup and phase safety accepted — 2026-10-25 13:06 UTC

[Code/Run/Open] Own A100-SXM4-80GB/High-RAM session `tta-guidance-components-a100-20261025-r1`, endpoint `gpu-a100-hm-kkb-ass1c2-36gbz99z2hf98`, completed all ten setup cells. Cell1 restart/reconnect passed; Cell2 only known fastrlock pip wheel error was deferred and Python3.8.20 verified; Drive consent was user-completed, Cell3 retry copied KNN wheel. Cell4 first CLI failed before sending setup code (kernel inventory idle); same-VM retry with REQUEST_TIMEOUT60 passed. EMD/Chamfer/PointNet++/PVCNN markers and actual GPU FPS/gather forward/backward/KNN/Chamfer/EMD checks passed. Cell10 returned status=passed for PyTorch2.1.2+cu121/CUDA12.1, data and checkpoints. No accuracy research has run.

Setup ZIP56156bytes SHA`13338f05cf2fc67418911e3eafba7723a746f948ef6997aa36ee05f0acaa132a` matches actual Drive and downloaded local bytes/CRC/member identities. Local `result/preparation/guidance_component_study_20261025/setup/setup_evidence_drive_roundtrip.zip`; exact Drive path in adjacent setup_evidence_receipt.json. Current source62e9a2b plus reconstructible accepted v8 asset overlay is sealed: manifest`7012c30a...f625ed75`, source ZIP`f5bca11a...576d0c6`, delta`1646533c...07e5065c`; exact SHA/bytes in preparation_identity.json.43 tests pass from the sealed source too;32 head configurations pass CPU preview.

Reviewed three task-local deployment/CPU/GPU phase scripts. Fixed controller cleanup on start-callback failure and SIGTERM, and fail-closed Drive-ledger restoration so retries cannot reset the shared3300s cap. Meaningful CPU subprocess regressions pass, including watchdog action during a blocked callback, external SIGTERM cleanup, retained123s/history and conflicting-ledger rejection. A Darwin dead-but-unreaped process-group EPERM was reproduced and corrected by reaping leaders before signaling remaining groups. Independent re-review reports no blocker. All this is CPU/control evidence; CUDA model smoke and physical result acceptance remain pending. Same monitor stays active; own runtime preserved, no peer operations or teardown. Next: sealed asset deployment/eight strict CPU loads, then the initially budgeted four-arm smoke; development needs independently accepted smoke and whole-phase projection within the cap.


### Agent B initial GPU smoke starting — 2026-10-25 13:11 UTC

[Code/Run/Open] Own sealed deployment validates637files and preserves fresh native bytes before restoring the audited four binaries. Deployment1421359bytes SHA07d769d35c38d9af490776a795bb699d24e0d88224165994649440a2eac668e7 and eight-load CPU proof24936bytes SHA2ee054d6ed9f5520177cf1e378749a7ebd1bda7e52799e7ec72c12d8f2687ffb match actual Drive/local SHA, CRC and source/native/class/checkpoint guards. PyTorch CUDA stayed uninitialized in CPU proof. Exact paths/receipts in `result/preparation/guidance_component_study_20261025/`.

Begin the registered initial GPU phase in own r1 under the same single monitor: two datasets×four arms×four heads, Gaussian/Background×70fixedIDs, seed0,17,920candidate predictions expected. Smoke uses fresh encode (no paired cache) for real instrumented end-to-end timings. The prior25–45min whole smoke+development estimate and shared3300s cap include short retries; no >60min phase is authorized. Independent local smoke acceptance and updated whole-phase timing projection precede any all15/70-ID development launch. No accuracy result is accepted at launch; stop on identity/state/RNG/skip/source/receipt mismatch or budget gate. Runtime stays live; peer untouched.


### Agent B smoke accepted; fixed development launching — 2026-10-25 13:33 UTC

[Run/Code/Inference/Open] Eight smoke jobs completed: two datasets×four arms×four heads, Gaussian/Background×70 fixed IDs;32 raw head ZIPs,4480 terminal events and17920 candidate predictions. Same-run terminal/probe state/RNG/class/coverage/receipt/skip guards pass; all32 archives independently reanalyzed locally and copied byte-identically into canonical own-worktree result paths. Phase358337730bytes SHA2d05389f514d766f5d8cbbb9b590231ccccb050d41e78e6c73fe02b8dc09e2a1 matches actual Drive/local SHA/CRC. Smoke is fresh-encode, two-corruption technical/timing evidence, not full benchmark or causal cross-arm accuracy evidence.

Initial controller runtime extrapolation incorrectly scaled ~19–20s fixed model startup7.5times per case, projecting67.87min development. The separately archived derived analysis counts only a conservative startup lower bound once (last-head constructor timestamp before first corruption, minus2s slack), scales all residual overhead15/2 and adds10s/case for extra input reads. Independent arithmetic agrees:40.993min expected development+8.445min smoke=49.439min active research;20% development uncertainty scenario totals57.637min. This is a projection, not a statistical bound or measured speedup. Original raw projection remains immutable in the phase archive and smoke_validation_initial.json. Derived report hash is bound by local_smoke_acceptance.zip and accepted_smoke.json. The existing3300s hard active-phase cap is unchanged; an overrun stops partial rather than silently extending. No expected>60min research phase is launched. CPU analysis, setup, idle/billing lifetime are separate from active research job-time metrics.

Proceed with registered70 development IDs/dataset, all15, seed0, four arms/four heads/three readouts;96 descriptive policy rows and134400 candidate predictions expected. Exact z0/h0/noise and post-preparation RNG cache is shared between arms and must verify. Report cached-comparison cost separately from fresh-encode smoke time. Heldout IDs, tuning, seed replication and full2468/581 benchmarks remain unexecuted. Same own r1/monitor52794; peer untouched, no teardown. Local evidence: `result/preparation/guidance_component_study_20261025/{smoke_phase_receipt.json,local_smoke_analysis/,smoke_local_acceptance.zip}`; raw32 ZIPs under canonical dataset/method paths.


### Agent B fixed development accepted and confirmation frozen — 2026-10-25 14:05 UTC

[Run/Decision/Open] All32 head ZIPs,96 case/policy rows,33600 terminal inputs and134400 candidate predictions pass independent local readout/state/source/ID validation. Exact Drive/local development archive SHA8abd0ec0...15901d and all30 cache NPZ bytes verify. Balanced C probability mean55.1905% nearly matches C+S minimum entropy55.2024%; no accuracy separation is established. At transferred weights, Chamfer improves over unguided in every case, whereas spectral increments vary by readout; standalone spectral update norms are much smaller and its coefficient was not calibrated. Fresh-encode instrumented smoke shared reconstruction is24.94% lower for C; cached development cost must not be called deployment latency. Initial active phase2093.6846s/34.8947min is complete under3300s.

[Results, paired uncertainty, cost and immutable evidence](guidance_component_development_results_20261025.md). Frozen plan `result/preparation/guidance_component_study_20261025/frozen_confirmation_plan.json` SHA256`c6ea3e01a03061afd227c82eb30eb128c2a4f1f342dbf937732dd5e61b122d45`: no tuning, select C probability mean, compare same-policy C+S primarily and frozen C+S entropy secondarily; both datasets/all15/four heads/reserved70 original IDs/seeds0,1,2. One new whole confirmation phase expects41.10min;20% margin+300s reserve54.32min, hard55min cap across all seeds together. No confirmation score has been read. New phase gets its own ledger while completed initial evidence remains unchanged. Full benchmark and calibrated spectral search stay unrun. Own runtime/monitor remain live; peer/original files untouched.
