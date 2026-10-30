# Chamfer-only guidance variants — 2026-10-30

[User report/Decision] User selects three independent studies on the existing
Chamfer-only, probability-mean host: sparse real guidance backward; semantic
guidance with KL and marginal entropy as separate arms (explicit follow-up);
reference-density-weighted selective Chamfer. No spectral loss, graph or
eigendecomposition may execute. Do not combine interventions until isolated
accuracy/time benefit is established. Causal early exit is a future-only test.
The user's decision to exclude spectral guidance applies to this campaign;
it does not retroactively turn earlier inconclusive comparisons into a proof
of universal absence of spectral benefit.

[Code] Worktree `chamfer-guidance-variants-20261030`, branch
`research/chamfer-guidance-variants-20261030`, starts from integration4da29b3.
Preserve root user edits, peer guidance-components worktree and peer runtime
`tta-chamfer-depth-tuning-a100-20261030-r1`.

[Run/Open] Owned runtime `tta-chamfer-guidance-variants-a100-20261030-r1`,
endpoint `gpu-a100-hm-ass1c0-3nh8smksm50de`, A10080GB High-RAM. Ten-cell
fresh setup is verified with only the authorized cell2fastrlock deferral.
Cells3–10 passed their exact markers; all five native completion files exist.
656full task-asset identities,8strict frozen CPUhead loads and restored pinned
native GPU forward/backward checks passed. Sole local monitor36160/PID59645
remains active. No research GPUworker/approvalfile/ledger/results exists.
User approved the exact whole5h research phase at13:23:44UTC; initial technical launch is next.

Local evidence: `result/preparation/chamfer_guidance_variants_20261030/`.
Drive: `MyDrive/thesis/result/chamfer_guidance_variants_20261030/`.
[Accepted launch readiness and archive identities](evidence/chamfer_author_code_audit_20261030/launch_readiness_20261030.json).
The source commit is a1d298a; later knowledge-only commits do not alter its
sealed source bytes or approved-scope plan e4bf0104…dcff4.

## Agreed design and execution batches

The detailed prior design was presented in conversation; user selected
implementation and Colab work. Preserve legacy C B70, rates.01/.01, p.95,
10/30 schedule and probability mean over3probes+terminal. Original3DD5/35 is
context only, never an isolated method control. Freeze all classifier weights,
BN/buffers/token algorithms. No R/Delta physical transform is introduced.

### Batch 1 - Sampler and mathematical controls

- Goal: opt-in variants with unchanged default C path and prohibited spectral work.
- Scope: tta_gsd_v2.py, new chamfer_guidance_variants.py and focused tests.
- Stack context: Py3.8 remote/Torch2.1.2+cu121; CPU synthetic local verification.
- Dependencies: current sampler/component fixtures and this design.
- Implementation notes for $development-agent: one optional experiment mapping;
  require Chamfer-only+skip_inactive; sparse interval3 including final step;
  static nonself density weights, squared-distance convention declared;
  accept optional differentiable scalar returned by clean observer, scaled by
  explicit semantic weight. No code duplication/new general framework.
- Verification: test-first, default parity, actual zero graph/spectral calls,
  backward counts/timesteps, nearest-index weight routing and finite gradients.
- Knowledge artifact to update: this note, own batch receipt section.
- User review gate: already authorized; researchGPU >60min gate stays separate.
- Out of scope: semantic observer/runner wiring, global diffusion, combinations.

### Batch 2 - Reused decoded probes and independent experiment entrypoints

- Goal: preserve baseline four-candidate probability mean while adding semantic
  KL or marginal-H only at two existing checkpoints.
- Scope: dedicated semantic observer, runner opt-in hooks, three experiment
  scripts, config/provenance/call-count tests.
- Stack context: frozen differentiable Point-MAE; external deterministic FPS
  views with selected-coordinate gradient and untouched main RNG stream.
- Dependencies: Batch1 API; native gradient viability requires later GPU smoke.
- Implementation notes for $development-agent: reuse one gradient decode and
  baseline-view logits for readout; one cached, detached deterministic-FPS
  corrupted-X Point-MAE forward per batch supplies the semantic reference. Existing probability mean uses baseline view only. No duplicate
  observer decode or extra vote silently added. Classifier weights/BN frozen.
  Semantic trajectory is head-dependent; label chosen guidance head explicitly.
- Verification: baseline/sham parity; same logits reused; correct local/style
  gradients; no classifier parameter gradients; separate per-op and total cost.
- Knowledge artifact to update: this note.
- User review gate: KL versus H choice resolved by user; no new permission needed.
- Out of scope: classifier training, SATOR tokens, fitted temperatures, early exit.

### Batch 3 - CPU proof, ten-cell setup and prepared GPU campaign

- Goal: concrete auditable phase with source/assets/IDs/seed/time freeze.
- Scope: task-owned setup/deploy scripts, experiment manifests and acceptance.
- Stack context: own A100; exact ten-cell setup, no research notebook Run-all.
- Dependencies: batches1–2 tests, actual native/data/checkpoint readiness.
- Implementation notes: development/confirmation object split, seeds0/1/2;
  no unbounded grid, no whole-benchmark repeats to replace existing results.
  Common prepared states/noise for accuracy; fresh encode timing separately.
  GPU whole phase estimated as one campaign incl smoke/calibration/timing.
- Verification: full local suite; actual frozen/native/zero-spectral smoke;
  Drive/local SHA/CRC; all15/macro14, paired corrected/harmed, object-cluster CI.
- Knowledge artifact to update: this note and final findings/open questions.
- User review gate: explicit duration approval before >60min research phase;
  no split into short CLI calls to bypass it. Runtime remains until explicit close.
- Out of scope: joint variants until independent benefit; no forecast accuracy claim.

## Future-only

[User report/Decision] Causal early exit remains deferred. Later evaluate
prefix-only confidence+stability, real B1 loop break/B70 compaction, full probe
cost and confident-wrong errors. No early-exit implementation in this campaign.

## Batch 2 receipt - 2026-10-30

[Code] Added strict opt-in `--chamfer-experiment sparse|semantic|weighted`
and three thin `scripts/eval_chamfer_{sparse,semantic,weighted}.py` entrypoints
sharing the existing runner. Existing default method/protocol is unchanged.
Experiments select the Chamfer component, lock C1/S0/invariant0 and graph=None,
and retain baseline 3 clean probes plus terminal probability mean. Variant and
semantic-kind fields distinguish independent studies while METHOD retains the
existing host ID for bundle/analyzer compatibility. Mixing variants, nonzero
spectral options or unrelated variant options fails before runtime setup.

[Code] `SemanticTrajectoryObserver` inherits shared trajectory recording. Both
semantic arms use the same pair: cached detached deterministic-FPS Point-MAE
logits from the original corrupted raw X, and the existing decoded baseline-view
Point-MAE logits. X uses unchanged source preprocessing with no alternate
normalization. A single frozen X forward per batch is cached for both semantic
checkpoints, at loop5/8 or15/23. KL is half the sum of KL(p_decoded||p_X) and
KL(p_X||p_decoded); marginal entropy is per-object H((p_X+p_decoded)/2). Both
losses sum over objects to match Chamfer's sum reduction and preserve per-object
gradient scale across batch sizes. This is not entropy of a minibatch class
histogram. X is detached in both.
The semantic scalar joins the sampler's existing backward and affects local
latent/style through the frozen decoder and frozen classifier, without updating
classifier weights, normalization buffers or token algorithms.

[Code] Each semantic checkpoint reuses one differentiable decode and each
head's baseline logits for readout; no reference vote is added. Probe operation
counts remain3 decodes and3 forwards per selected head, plus one cached X source
forward per batch. Terminal evaluation is unchanged. Other selected heads
evaluate the same geometry, which is explicitly labelled Point-MAE-dependent.
Weight must be explicitly supplied for actual semantic execution; weight0 runs
the same semantic work as a sham. Reference geometry/logit hashes, model state,
probe raw losses, reuse indicators, RNG guards and reference/probe dispatch
times are archived. Total reconstruction timing includes the X forward and is
synchronized by the existing runner; dispatch components are not standalone
GPU speed claims. Shared initial diagnostics now record actual sampler-boundary
`initial_rng_sha256`, including after a paired-state replay restores RNG.

[Test] Test-first red/green CPU verification covers strict/default configuration,
15/40-class semantics, finite nonzero local/style gradients, no model weight
gradients, baseline-view/logit reuse, baseline/sham candidate parity, cached X
forward count/detachment, RNG restoration even on failure, frozen-weight checks,
background checkpoints and initial RNG provenance. Seven focused suites pass
59 tests. All three standalone scripts also preview successfully under Python
`-S` with no native/CUDA dependency imports. `git diff --check` passes.

[Open] No GPU research or commit in Batch2. Native CUDA FPS/gather and real
Point-MAE/decoder backward remain subject to the later owned-runtime smoke;
CPU synthetic success is not evidence of accuracy improvement or GPU speed.

## Batch 1 receipt - 2026-10-30

[Code] Added the trailing opt-in `chamfer_experiment` sampler argument and
`chamfer_guidance_variants.py`. Experimental configuration requires inactive
component skipping, positive Chamfer, zero spectral/invariant weights,
`graph=None` and no dynamic graph. Unknown/mixed options fail before model or
graph work. The legacy default path retains its original sum and update order.

[Code] Sparse guidance retains all forward timesteps/noise and guides at
one-based counts `{1,4,7,...}` plus the final count: 4 backwards for 10 forward
steps and 11 for 30. Off steps skip loss calculation and updates and disable
the prior/scheduler autograd graph. Semantic mode accepts a finite scalar from
the existing `clean` callback, preserving its local/style graph, with an explicit
finite nonnegative weight. Existing observers returning `None` are unchanged.

[Code] Weighted SCD computes one detached reference-latent-XYZ density vector:
inverse mean nonself-kNN squared Euclidean distance with epsilon `1e-6`,
normalized to per-object mean1. kNN uses 128-query chunks. Trimming still sorts
the native Chamfer distances before weighting; predicted matches gather the
nearest reference weight through the native returned indices, while the reverse
direction uses each reference point's own weight. The default k is10.

[Test] New CPU fake-native tests cover exact interval1 C points/style/RNG
parity, complete sparse schedules and actual backward counts, semantic zero
weight parity and active local/style changes, validation before graph calls,
weighted nearest-index routing, distance trimming, equal-weight parity,
permutation invariance and a full weighted sampler path. All experimental test
paths replace spectral loss with a raising sentinel. Numeric counters expose
prior forwards, guidance backwards, semantic callbacks/scalars, density
precomputation and zero spectral calls.

[Test] Test-first failures were observed before sampler API implementation and
before the final callback-counter/invalid-variant validation. Focused suites:
28 tests passed. Full CPU discovery with Torch/NumPy/SciPy/h5py/PyYAML:
364 tests passed in20.632s; `git diff --check` passed. No Colab access, native
CUDA execution, GPU accuracy/timing evidence or commit in this batch. Native
gradient viability and realistic accuracy/runtime remain Batch3 obligations.

[Paper/Decision] CloudFixer Sec4.2 applies inverse average neighbor-distance
weights to a displacement penalty, allowing isolated points more freedom. This
experiment transfers that confidence principle to Chamfer fidelity using the
explicitly proposed inverse **squared** distance formula. Call it
**density-confidence weighted SCD**, not inverse-density balancing or an exact
CloudFixer implementation. Sparse reference regions receive less fidelity
weight. The paper's written denominator uses distance, so the squared-distance
choice is an experimental inference. [Primary paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf).

[Decision/Open] Per-object mean1 weights preserve the uniform-weight loss and
gradient exactly, but distance trimming and nearest-match multiplicity can change
the total retained weight, and therefore update magnitude. Retain the proposed
formula; measure a label-free actual local/style update-norm ratio against C for
scale calibration before interpreting accuracy gains as better geometry. Do not
silently add per-direction normalization. Added an unconditional actual spectral
loss-call counter so baseline C proves zero while active legacy spectral modes
report real calls (two per guided step); 29 focused CPU tests passed.

## Batch 3 asset preparation receipt - 2026-10-30

[Code/Test] `scripts/prepare_chamfer_variants_assets.py` and five small fixture
tests prepare the task's future `assets_v1` source/audit/study overlay. Default
invocation only previews; `--seal` is reserved for the coordinated source freeze.
Existing destination/delta/receipt files cause refusal rather than overwrite.
The builder pins sensitivity manifest
`599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef`,
refreshes its runtime-source inventory from the current checkout, includes new
code, and excludes newly discovered build/results/tests outputs. It retains the
inherited dataset/checkpoint/native identities without opening bulk files. Local
`assets_v1` therefore intentionally requires reconstruction over the full pinned
base before complete-manifest verification. ZIP CRC, copied-source identities,
six ID hashes and unchanged source inputs are checked before publication.

[Code] Added task-owned `study_scope.json` and
`asset_deployment_requirements.json` under
`result/preparation/chamfer_guidance_variants_20261030/`. Scope declares all four
frozen scoring heads, shared geometry with Point-MAE semantic guidance, separate
C/sparse/KL/marginal-H/weighted arms, B70 all15 severity5, development70 seed0
and heldout140 seeds0/1/2. The deployment record pins exact full-v7, components
delta and sensitivity delta identities plus the two additional Point-MAE
checkpoint identities. A private complete sensitivity tree is an equivalent
base; peer assets/runtimes must remain untouched.

[Test/Open] Five asset-builder tests passed, including intentionally absent huge
weight/data fixtures, preserved native/checkpoint metadata, refused overwrites,
tampered metadata, unsafe paths and generated-file filtering. A combined
legacy-restore/sampler/asset run passed26 tests. Legacy numerical regression now
compares every historical metric key, exact output/context/style and RNG,
separately checking the additive actual spectral-call metric. No final task
assets, deployment or GPU execution has been produced by this preparation batch;
source freeze and final broad CPU verification remain pending coordination.

[Code/Test] Prepared ignored task-owned `remote/deploy_chamfer_assets.py` and
`remote/preflight_chamfer_cpu.py` wrappers. They reuse an exact copy of the house
deployment helper and a hash-bound CPU proof copy with only `combined` changed
to `chamfer`. Helper globals bind to `/content/studies/chamfer-guidance-variants`
and the task Drive/session paths. Deployment verifies each complete reconstruction
boundary, including missing Point-MAE weights, and uses the existing native
preservation/restore, safe ZIP and evidence-sealing functions. Pinned native
restore is gated by the actual ten-cell acceptance, exact wrapper hashes/markers,
cell1 reconnection, the sole explicitly recorded cell2 fastrlock deferral,
cell10 report identity/native checks and all five fresh native phase markers.

[Test/Open] Both wrapper dry self-check entrypoints passed the inherited safe
ZIP/seal and eight-head row validation checks. Three offline gate tests passed
for exact acceptance, the known report suffix, forbidden extra deferrals,
missing native completion and peer-session/echoed-marker rejection; all phase
sources parse for Python3.8. These checks did not import native backends, load
weights, access a runtime or initialize CUDA. No phase source upload, deployment,
final seal or GPU experiment was executed. Actual cell10 acceptance and final
source identities must be supplied by the owning coordinator before use.

[Code/Test] Final seal provenance now computes `source_dirty` from actual
`git status --porcelain --untracked-files=all` restricted to the executed-source
inventory, and records that exact status plus every copied source identity.
It no longer claims a dirty source after a clean committed freeze merely because
the builder supports overlays. Six builder fixture tests passed, covering both
clean and modified source states; final seal remains unexecuted.

### Batch 3 CPU analysis receipt — 2026-10-30

[Code/Verified] `scripts/analyze_chamfer_variants.py` reuses the existing strict
sealed `Bundle`, trajectory validator and probability policies. It accepts an
exhaustively completed declared subset as `complete_subset`; a full benchmark
still requires all original objects and all15 corruptions. Running/failed,
truncated, undeclared or unsealed results fail acceptance. C, sparse, KL,
marginal-H and weighted arms stay independent. Every batch must record zero
actual spectral/graph calls, zero spectral/invariant weights and inactive
component skipping, including C. Classifier/LION guards, preserved executed
source bytes, asset/native hashes, host protocol, labels and actual
input/h0/noise/scheduler plus sampler-initial RNG hashes must match paired C.
Final trajectory hashes are allowed to differ by method.

[Code] The report includes per-dataset/head/seed/corruption terminal and
probability-mean accuracy; same-set all15 primary and macro14 secondary where
available; paired corrected/harmed original IDs; and object-cluster percentile
CIs within each dataset retaining every head/seed/corruption observation.
These CIs condition on the included heads and seeds; they do not estimate an
independent seed population. Partial corruption scope is labelled only
`declared_subset`. Shared reconstruction wall time counts once per actual
group; terminal head costs are added. `cache_replay` and `fresh_encode` are
separate cost modes, with same-mode/same-condition cost deltas only. Cached
execution costs do not establish deployment speed or an equal-time voting
comparison.

[Verified] Test-first collection failed on the absent module before
implementation. `/tmp/3dd-depth-cpu312-20261030/bin/python -m pytest
 tests/test_analyze_legacy_trajectory_macro14.py
 tests/test_chamfer_variants_analysis.py -q`: 26 passed (23 new analyzer tests).
CLI `--help` also passed. No GPU, network, runtime or peer-worktree action was
performed by this CPU analysis batch. Synthetic sealed-bundle coverage is not
native classifier-gradient or real-run evidence; real sealed artifacts remain
required before accuracy/speed conclusions.

## Setup and integration status — 2026-10-30 12:03 UTC

[User report/Run] User authorized temporarily stopping the local monitor for
interactive Drive authentication, completed actual Google consent, and explicitly
requested restarting the monitor. Monitor80411 stopped without runtime teardown;
TTY83781 mounted Drive and exited0. Sole monitor36160 restarted12:02:03UTC on
the same endpoint and submitted exact setup cell3. Peer runtime remains untouched.

[Code/Decision] Semantic KL and marginal entropy are separate arms using the
same pair: frozen cached corrupted-X FPS logits and existing decoded baseline
FPS logits. X is detached; gradients flow only through decoded input/local/style.
No alternate vote enters probability mean. Fixed-X source-forward cost is inside
total reconstruction timing. This supersedes the initial alternate-decoded-view
implementation discussed during preparation; no GPU result used that version.

[Code/Open] Density-confidence SCD retains the proposed inverse mean squared
nonself-kNN-distance weights. This is an inferred fidelity transfer of the
CloudFixer displacement regularizer, not inverse-density balancing or a
paper-exact CloudFixer implementation. Squared distance differs from the paper's
written mean distance. Mean1 weights alone do not control retained gradient
magnitude; label-free update-norm calibration is being prepared before execution.

[Code] Task-owned manifests exclude280previously used IDs per dataset; select
new development70 and disjoint confirmation140 with Python Random20261030,
without labels. Smoke2 is inside development. Both are benchmark-internal
partitions, not an externally unseen test. Selection source SHA/ID receipts live
in result/preparation/chamfer_guidance_variants_20261030/selection_receipt.json.

[Test/Open] Root broad check first selected a CPU interpreter missing Torch;
no project code inference follows from its collection failure. Proper dependency
check found an added metrics-key mismatch in four legacy parity assertions
(points/style/RNG still matched), plus missing pytest in the selected test
environment. The metrics assertion is being updated to preserve legacy numerical
checks while accepting the new actual-call counter; final suite remains pending.

### Batch 3 campaign preparation and calibrated analysis receipt — 2026-10-30

[Code/Verified] `scripts/run_chamfer_variants_campaign.py` prepares a fresh,
SHA-bound command ledger and validates retained explicit approval-file hashes;
it deliberately has no GPU worker-launch mode. The exact plan schema/sample
is exercised by `tests/test_chamfer_variants_campaign.py:plan`. Inputs cover
absolute source/assets/research/Drive/Python/data/label/Point-MAE paths, pinned
asset-manifest and source-receipt identities, and sealed smoke2/development70/
heldout140 selection identities. Optional CPU preflight verifies references,
coverage and split disjointness. Before a real GPU launcher runs, source/native
readiness, whole-campaign approval, current budget projection and fresh local/
Drive destinations still require verification.

[Code] Prepared scope is 86 jobs: two all15 development70 Point-MAE-only
label-free calibration jobs; fourteen Gaussian/background technical smoke jobs
(five independent arms plus zero-weight KL and H shams per dataset); ten
all15 development70 seed0 jobs; thirty all15 heldout140 seeds0/1/2 jobs; thirty
separate fresh-encode Gaussian/background70 seed0 timing jobs (three repeats,
rotating arm order). Every evaluation job uses all four frozen heads, B70,
Chamfer-only and the existing probability mean. Accuracy stages share immutable
initial-state caches within each stage/seed; timing jobs omit replay. No arm
combination or accuracy-driven coefficient selection is included. Semantic
native gradients must be finite/nonzero for both local/style latents and the
two weight-zero shams must match C geometry/all-head logits before accuracy.

[Inference/Planned] Nominal 13,780s (3.83h): calibration900, smoke840,
development1,780, confirmation9,000, timing1,260. A10% planning margin and1,800s
packing/retry reserve yield16,958s (4.71h), under the tentative18,000s (5h)
whole-GPU cap. Basis: prior C70 about178s/all15, C140 about300s/all15;
Gaussian/background timing about42s. Calibration/smoke overhead remains
uncertain: re-estimate the whole remaining phase after smoke and stop if the
cap cannot hold. This is a preparation estimate, not run evidence or user
approval. GPU approval remains pending. Calibration-dependent argv tokens are
explicitly non-executable until the fixed formula report and its SHA exist.

[Code/Verified] The analyzer now permits weighted eta/gamma scaling only with
an actual sealed fixed-calibration report: exact bytes/SHA, frozen median rule,
summed per-object losses, both dataset proofs, matching source/native/assets/
LION and Point-MAE hashes, and both rates equal to.01 times the proven scalar.
All other host protocol fields must match C. Unproven rate changes, synthetic
reports, coefficients inconsistent with recorded medians and changed gamma
fail acceptance. Semantic calibrated weights also match the exact report.

[Verified] Existing local Python environment only, no install/network/GPU:
`/tmp/3dd-depth-cpu312-20261030/bin/python -m pytest
 tests/test_chamfer_variants_campaign.py tests/test_chamfer_variants_analysis.py
 tests/test_analyze_legacy_trajectory_macro14.py -q`:39 passed (12 planner,
24 analyzer,3 legacy). Both preparation/analyzer modules compile; preparation
CLI help passes. Root owns real path/hash population, immutable source freeze,
reviewable preparation outputs and any later approved launcher.

## Label-free calibration receipt - 2026-10-30

[Code/Decision] Added research-only `scripts/calibrate_chamfer_variants.py`.
`run` reuses the shared runner with an explicit worker-script/observer injection;
there is no separate copied GPU evaluator lifecycle. A tiny optional sampler
`guidance_components` callback sees the actual Chamfer loss, live noisy local
latent/style, predicted clean latent and original h0 before the existing backward.
Existing callers have no callback and preserve their path. Calibration follows
the C trajectory with semantic weight0, one cached detached corrupted-X anchor
and the same reused baseline decoded logits for both semantic scalars.

[Code/Decision] Collection is fixed to seed0, B70, declared development70 IDs,
all15 corruptions and Point-MAE for each dataset. At normal5/8 or background15/23,
diagnostics retain C, symmetric KL, marginal-H and static-density weighted-C
graphs. `autograd.grad(retain_graph=True)` measures per-object local/style norms
and cosines without assigning `.grad`. Per batch this adds8 diagnostic gradient
calls,2 native weighted-counterpart Chamfer calls and one static density
precomputation, alongside one cached X source forward. Semantic and both Chamfer
losses sum over objects; duplicating a batch doubles the scalar and preserves
each object's gradient, so B2 smoke/B70 calibration/partial batches share scale.

[Code/Decision] Update norm uses the actual sampler mapping
sqrt((eta*local_gradient_norm)^2+(gamma*style_gradient_norm)^2). Both baseline
rates are.01. `summarize` verifies sealed bundles, exact per-object/probe/candidate
coverage, frozen models, source/native/assets/CUDA evidence and zero actual
spectral/graph calls. It reads gradient records only. Evaluator labels,
predictions and scores remain archived for audit and never select coefficients.
Zero/near-zero or nonfinite combined norms/ratios, duplicate/missing evidence or
incomplete datasets reject calibration; there is no fallback coefficient.

[Code/Decision] Pool the two equal70 development subsets with equal object,
probe and corruption weights. Freeze semantic_kl and semantic_h separately as
0.1/median(candidate/C update-norm ratio); freeze weighted_scale as
1/median(weighted/C). No accuracy fitting or grid is introduced. The report has
schema1, purpose `label_free_chamfer_calibration`, full two-dataset proof,
coefficient/ratio summaries and an adjacent SHA/size receipt. Final GPU status
is produced only by validated bundle summarization, not the CPU math helper.

[Code] Optional runner `--calibration-report` plus
`--calibration-report-sha256` binds the report before execution and copies exact
bytes into the sealed bundle as `calibration_report.json`. Semantic weight must
match its reported arm. Weighted Chamfer keeps C1 and uses eta=gamma=.01 times
the reported scale; this scales the update without changing the selective loss.
Runtime verifies corresponding assets, LION/frozen semantic head, native bytes
and common source bytes. Direct explicit experiments without a report remain
available and are marked uncalibrated. Freeze all coefficients before heldout.

[Test] Calibration tests were first red, then green. Covered rate mapping with
unequal rates, gradient-only fixed coefficients, missing/duplicate coverage,
near-zero rejection, both fixed-X scalar graphs, `.grad`/RNG purity, diagnostic
hook zero-weight C points/style/RNG parity, and sealed summaries with no label
or prediction files at all. Semantic duplicate-batch scaling and report SHA/
weighted-rate validation have dedicated tests. Eight focused unittest suites
pass66 tests; both dataset calibration previews pass; scoped `git diff --check`
passes.

[Open] GPU calibration has not run. Real Point-MAE/decoder/native backward must
pass the technical smoke before collection; failure stops the phase. The root
campaign controller owns whole-phase duration approval and concrete asset/ID
paths. Reviewable commands use the existing runner arguments:

```sh
python scripts/calibrate_chamfer_variants.py run --dataset DATASET --dataset-root DATA --label-path LABELS --asset-root ASSETS --asset-manifest-sha256 SHA --result-root RESULTS --drive-root DRIVE --pointmae-config CONFIG --pointmae-checkpoint CHECKPOINT --index-manifest DEVELOPMENT70 --seed 0 --corruptions all
python scripts/calibrate_chamfer_variants.py summarize MODELNET_BUNDLE SCAN_BUNDLE --output calibration.json
```

## Calibration acceptance review fix - 2026-10-30

[Code/Test] Review found that runtime accepted coefficient reports whose formula
or other candidate statistics the analyzer would reject. A test first exposed
six invalid reports accepted despite updated valid file hashes. Added the
dependency-free shared `chamfer_calibration_report.validate_calibration_report`
for runtime, planner and collector; root wires the analyzer to the same rule.
It requires the exact frozen formula, summed reduction, target.1, all three
positive finite coefficients and complete ordered finite ratio summaries:
4200 rows per candidate, missing0 (2 datasets x70 objects x15 corruptions x2
probes). Every coefficient must equal its frozen target divided by the recorded
median within relative tolerance1e-12 and zero absolute tolerance. Existing
source/native/assets/state provenance checks remain. Realistic valid fixtures
now replace the previously under-specified runtime report fixture. Nine scoped
unittest suites pass70 tests, the shared validator passes3 tests under Python
`-S`, and scoped diff whitespace checks pass. No GPU or commit.


## Author-code gate and engineering review — 2026-10-30

[User report/Code/Inference] User requested author-repository inspection after
implementation and before research tests. CPU unit verification predated this
request; no research GPU run occurred. The
[completed consistency audit](chamfer_author_code_consistency_audit_20261030.md)
compares CloudFixer/MAMVI/GDA pinned source and records unavailable official
Adaptive Guidance sampler code. Differences are explicit adaptations, not
claims of faithful reimplementation. No static semantic/weighted/sparse
algorithm defect was identified; native gradients remain open.

[Code/Test] Independent review fixes: shared strict calibration-formula,
reduction,4200-ratio-coverage and coefficient validation before GPU setup;
separate3repeat fresh-timing analysis with strict paired provenance, no reused
groups and unchanged accuracy duplicate rejection.54targeted tests plus46
subtests passed. Root added a red/green source-receipt test: identity-only
preflight accepts sealed_CPU_only and explicitly reports bulk_assets_verified
false; remote fullasset/native/checkpoint verification remains mandatory.

[Run/Open] Monitor36160 remains active. Cell4 first failed solely on a
files.pythonhosted.org TensorFlow wheel read timeout, with raw stream retained.
A subsequent transport-setting call lost its CLI connection in the prelude
before user source; the stuck local child was terminated, without runtime
teardown. Same-VM read-only audit confirmed Drive/project survived, no native
completion or active pip/conda installer. PIP_DEFAULT_TIMEOUT=120 was then
confirmed by exact marker. Exact cell4wrapperSHA2ce4fed2…ab8e is replaying
from12:20:11UTC with only that transport-environment change; no package/version
pin or original notebook cell was edited.


## Final local source verification — 2026-10-30

[Test] Root integrated pytest:436passed,331subtests,6warnings,23.21s.
Python3.13.7/Torch2.15.0/NumPy2.5.4/pytest9.2.0 local CPU environment;
all13changed runtime modules also parse with Python3.8 grammar. This does not
prove runtime equivalence to remote Torch2.1.2/nativeCUDA.
[Exact commands/source/log identities](evidence/chamfer_author_code_audit_20261030/cpu_validation.json).
Default point/style/RNG parity remains asserted; calibrated report and repeated
timing acceptance regressions pass. `git diff --check` passes.
No research GPU result exists; source sealing/remote fullasset and native
preflight plus explicit whole5h-phase approval remain launch gates.


## Source seal — 2026-10-30

[Code] Experimental source commit33ff8457b2269227ca0007c8e18e09648e59dbcf
is clean at seal. TaskmanifestSHA9fbdddc3692a0ad191f8a24e183da3bd320f80d0b6fcca132c2ac2938e13eaba;
sourcearchiveSHA920f0fa665848dfe198f4d072ebd7eaf40748249f4bb1ebeedb505dfc21d3f20
(5,567,321bytes,296runtime source paths); deltaSHA
9dd062c24f30ee212048fa0af49fc2e26115b3a676f412dd17c500a1a27e3111
(5,518,571bytes). Source/ID overlays are sealed; bulk checkpoint/data bytes
are inherited exact identities, requiring full remote verification before GPU.
[Compact seal receipt](evidence/chamfer_author_code_audit_20261030/source_seal.json).
NoGPUlaunch or duration approval is implied by sealing.


## Prelaunch integration revision2 — 2026-10-30

[Code/Test] Actual RunBundle boundary review found semantic weight strings
like0.0/0.1 contain dots and positive scientific exponents contain+, violating
the existing safe run-name contract. Names now encode those characters safely;
coefficients and math are unchanged. A red test exercised real RunBundle.create
for KL/H weights0,.1,1e-9,1e20 before any model/CLI work; all8cases now pass.
Final integrated437tests/339subtests pass in22.52s,6preexisting warnings.
[Updated source/log receipt](evidence/chamfer_author_code_audit_20261030/cpu_validation_revision2.json).
Revision1source/delta remain preserved and were never GPU-executed; revision2
source/delta will replace them in the eventual approved plan.

[Code/Test] Reviewed ignored GPUstage executor is prepared, not executed:
remote/gpu_chamfer_phase.py SHAe03582aff64156efd120d12a93af33c6a7a3e478aae865cfff832b3d39349770
(19,671bytes);12offline wrapper/fake-child tests pass. Exact stage order is
initial8technical jobs,2calibration,6active smokes,10development,30confirmation,
30fresh timing. Whole5h continuous research clock includes between-stage wait;
active/job/packing times also persist. Full source/setup/CPU/assets/approval
gates, actualnative gradient/parity proof, immutable physical cache Drive
copy and failure preservation are required. A ZIP-depth regression uses actual
RunBundle layout and excludes nested source_snapshot ZIPs. No GPUlaunch yet.


## Frozen scope awaiting duration approval — 2026-10-30 12:54 UTC

[Code/Test/Run/Open] Current executed sourcea1d298a4d64bd9b580253dfa53224aa69b94df33
is clean at revision2seal; sourceZIPd638c2e455a225d6597ff10eeda63778e3963711e05e542450ae5e47726e0551.
ManifestSHAa9f017789fda66de310931f63cde5e540985b96ac5097ae8fe729358e9d06da0;
deltaSHA398891e17854e4025379bc484ea3e261e4185824801a2fdf9e6ccdf75d36fb95.
Revision1raw files and source pins remain historical/unexecuted.

[Code] Concrete86jobplanSHAe4bf0104fe997157810820aad1b6f12e9f0f7e6b9915334329c16f4a754dcff4;
commandsSHA9825e3d51b73b85f25fdf11876c885c86b43d2a3a5d063afcc2cda3985bb78ef.
CPU parser preflight accepted86workers/338head configurations without bulkdata,
weights or GPU. Nonexecuting parser dummy coefficients are not calibrated
research coefficients.12phase/fakechild gates and437CPUtests/339subtests pass.
[Scope identities](evidence/chamfer_author_code_audit_20261030/prepared_scope_revision2.json).

[User action required/Open] Explicit whole-phase5happroval question was sent;
no answer/approval has been received. Estimate3.83h nominal/4.71h withmargin;
continuous5h cap starts only at first research GPUworker, setup separately.
The reviewable request is
result/preparation/chamfer_guidance_variants_20261030/gpu_approval_request.md;
plan+allargv+gatedkernel wrappers live under remote/. No GPU approval file
exists, and no research worker has launched. Only independently useful arms
may later combine; such combinations are outside this frozen phase.

[Run/Open] Controlbundle3d07ca6a…16f1f62 and its reviewed installer were uploaded
via CLI filesystem API while the sole monitor owns native compilation. They
have not yet been extracted/Driveverified. PVCNN cell8 is active; EMD, Chamfer
andPointNet++5/6/7 completed. User explicitly required monitor restart after
Driveconsent; monitor36160 remains active on the same owned endpoint.


## Preparation accepted; research launch pending — 2026-10-30 13:21 UTC

[Run/Test/Open] Canonical ten-cell split replay passed on the same fresh A100
with only known cell2fastrlock deferred. Setupreport95098b5e…6c1c2e (9005bytes),
acceptancec64209b9…d9f637 (3510bytes),15ModelNetseverity5 arrays verified.
Private deployment verified656files and matched LION/config/checkpoint/data
identities. Fresh native binaries were preserved before restoring the four
pinned historical binaries; no peer runtime or user/root edits changed.

[Run/Test] Eight strict frozen CPUloads passed: both datasets×DGCNN/CurveNet/
PointNeXt/Point-MAE, no missing/unexpected checkpoint keys and CUDA remains
uninitialized in that proof. After native restore, a separate environment-only
synthetic32point smoke passed KNNforward and Chamfer/EMD/PVCNNgrouping backward
in5.75s on actualTorch2.1.2+cu121/A10080GB. No classifier/LION inference,
guidance trajectory or accuracy evaluation executed. Native import of the
graph module is not a graph/spectral calculation; no such function was called.

[Run] DeploymentZIP71bfd21c…, CPUproofZIP794d10d1…, pinned-nativeZIP518c9aaf…
and readinessZIPa8e53ce3… match Drive/local size/SHA/CRC and member digests.
Raw ten-cell sources/stdout/report acceptance ZIPbc0afb9f…076c85,273259bytes,
36members completed actual Drive→local roundtrip. Literature44member ZIP
f1a1e6cd… and57member author-source ZIP5efec812… were Drivehash/CRCverified.
Compact accepted identities are in launch_readiness_20261030.json.

[Open/User action required] The concrete86job source/plan/commands are ready;
explicit5h whole-research approval has not arrived. No research GPUworker,
calibration report, paired cache or accuracy/runtime result exists. The
5hclock has not started; completed setup is separate. Actual semantic native
decoder→classifier→local/style gradients, shamparity and calibratedcoefficients
remain the first research gates after approval. Keep source/plan/cap fixed;
retain failed/negative outcomes. Runtime and monitor stay live; no teardown
authorization was given.


## Whole5h research phase approved — 2026-10-30 13:23:44 UTC

[User report/Decision] Explicit reply: “5 saat üst sınırla onaylıyorum”,
questioncall_5f07a438c7ed48b7bfa95b075ce5c171. ApprovalSHA
a041f13d1eea8a3025d55af220ac8c4621c8efbdbb33cfb0d497c7672ea0207f
binds plan e4bf0104…dcff4, sourcea1d298a, exact86jobs and18000scontinuous
research cap. Setup is separate. Approvalfile uploaded to ownedVM andDrive;
no further permission is needed for unchanged execution/monitoring/download.
[Retained approval](evidence/chamfer_author_code_audit_20261030/gpu_approval_20261030.json).
Initial8technical jobs launch next via monitor36160; no empirical outcome yet.
All earlier pending-approval entries above are dated history superseded here.


## Initial native smoke stopped at exact sham parity — 2026-10-30 13:39 UTC

[Run/Open] All8 initial technical workers completed their declared2IDs×Gaussian/
Background×4heads;32raw head archives are independently readable and match
actual Drive SHA. The stage failed its final semantic-weight0/C exact geometry/
logit comparison at13:31UTC. Calibration, positive-weight semantic/weighted,
development, confirmation and timing stages have not started. No accuracy or
speed benefit is established by these technical outputs.

[Run] Failed phase archive97160bytes SHA
2354d2cda1113a086bc2216da9d7e69ee5d85d80429a3130a5819f89002c4e68;
local `result/preparation/chamfer_guidance_variants_20261030/gpu_smoke_initial_failed_evidence.zip`,
Drive `thesis/result/chamfer_guidance_variants_20261030/phase-evidence/gpu_smoke_initial/evidence.zip`.
Read-only CPUfield audit archive6426bytes SHA
e6b845bb89b3e6ff77123cf352626e8229443c90e2796ef555a4667e90b344ea,
local `initial_failure_audit.zip` in the same preparation directory; actual
Drive roundtrip size/SHA/CRC passed. Original failed files are preserved.

[Run/Code/Inference] Initial input/encoded states/noise/style/initialRNG and final
RNG match; frozen weights/native/config identities match. The first clean
probe already differs before semantic activation; all terminal local/style
hashes differ. Logit maximum absolute differences reach11.57 and some candidate
argmaxes change. Native voxelization/devoxelization uses floating atomicAdd,
but this is only a candidate cause, not yet demonstrated. A baseline-vs-baseline
native replay with numeric intermediate arrays is needed before any acceptance
rule or implementation change. Do not weaken exact parity speculatively.

[Decision/Open] Original research clock started13:25:16.894181UTC; unchanged
18000s continuous cap ends18:25:16.894181UTC, including debugging/waits. Monitor
36160 and owned A100 remain live; no teardown authorization. Short bounded
recovery diagnostics are within the authorized work; no failed directory may
be overwritten or silently resumed. Reassess remaining whole-phase projection
before launch. Source remains a1d298a; no algorithm fix has been made.


## Native self-repeat and metadata repair — 2026-10-30 13:55 UTC

[Run] Frozena1 source C1/C2/KL0 diagnostic completed140.5913s, all cached
replays on identical2ModelNet IDs/Gaussian+Background. Raw47memberZIP
47507841bytes SHA17314883a99194c5fb6ce17be9893e58e3323bf70852c89144a1d3927b25473d
matches actual Drive/local size/SHA/CRC. Numeric audit retains actual perstep
prior inputs/outputs, scheduler outputs, leaf gradients and decoded points;
[compact evidence](evidence/chamfer_author_code_audit_20261030/native_repeat_diagnostic_20261030.json).

[Run/Inference] C1/C2 initial tensors and first prior input/style are byte-exact,
but first prior output maxabs differences are7.7614e-4 Gaussian and5.2428e-4
Background. Differences precede every semantic callback and amplify through
C-only gradients; terminal decoded relativeL2 differs.004808/.040121 and some
predictions change. Thus exact whole-trajectory parity fails even for C/C in
the observed execution policy. This does not establish zero semantic effect:
no arbitrary epsilon or trajectory-equivalence acceptance is inferred. Next
is exact native autograd(0*S) local/style-zero verification on live probes.

[Code/Test/Run] Found independent artifact-schema bug: when REPO lies within
assets/source_snapshot, an elif prevented canonical repo/* entries and perfile
source copies. Commit0fd258110abf48dc92d739b288010ebfbee475b2 changes that elif
to independent if after retained asset declaration/hash verification.438CPUtests/
339subtests pass24.86s,6pre-existingwarnings. New regression rejects undeclared
and tampered nested sources. No sampler/semantic/math code changed.

[Run] New immutable prepared_v2 deployment verifies656files; exactly the helper
and source ZIP asset hashes differ. All296runtime sources match except the
one-word helper patch. Newmanifestef0412d3…19b777, sourceZIPfd8c163f…b1a69;
CPUcanonical-source probe passes, CUDA stays uninitialized. Existing verified
8strict head loads are explicitly reused, not claimed as rerun. DeploymentZIP
5500079bytes SHA1dda3c561376238cbcec4d39a9c0bef32eac848e1cd3ef50409f0e87c131b17e
matches Drive/local size/SHA/CRC. Original deployment/failedstage/approval remain
immutable. Same originalcontinuous5h clock/deadline; recovery may consume the
registered1800s packing/retry reserve, never extend18000s. Accuracy stages remain
gated on pointwise zero-effect proof and reviewed metadata-only recovery.


## Exact zero effect accepted; calibration running — 2026-10-30 14:04 UTC

[Run] Native KL0/H0×Gaussian/Background live autograd(0*S) checks complete:
8probe rows (normal5/8,Background15/23), B2; every local/style gradient is
finite and has exactly zero count/max/norm. RNG and existing leaf .grad fields
are unchanged. Default runner frozen-model and declared-ID checks pass.
Raw91382byteZIP SHA0bb7b512421b9a20e123491cd614ea860df010050ee2c2535756ccafdfa07be1
matches actual Drive/local SHA/CRC; receipt1db3130a…388bf8 retained. A CLI
prelude Connection-was-lost attempt never submitted user code (history0matches,
same owned endpoint verified); only its local CLI child was stopped and same
reviewed phase retried. The persistent monitor and VM stayed live.

[Decision/Inference] Initial exact whole-trajectory equality remains a failed
historical outcome. C/C native drift establishes why that gate cannot pass
under the observed policy; exact-zero contribution proves the semantic
zero-weight scalar contributes no local/style update at the tested probes.
This is not a trajectory-equivalence claim or arbitrary tolerance. Exact
initial tensor/RNG/source/asset/frozen checks remain required for comparisons;
three-seed uncertainty is conditional on native nondeterminism.

[Run/Decision] Metadata-only recovery binding46318318…0b466 and controller
9cfd4769…b00772 verify all296actual source files, the exact elif→if patch,
unchanged bulk/checkpoints/IDs, original approval/plan/commands,32initialrawZIPs,
CPUrepair and both native diagnostic receipts. Original failed stage/approval
are immutable. Six CPU recovery tests pass including9bad-zero subcases.
The original1800s packing/retry reserve is now consumed by actual recovery;
pinned accounting fd8c08c7…aefa0e9 records0remaining without changing nominal
compute estimates,10%margin or absolute18:25:16.894181UTC deadline.

[Run/Open] Both label-free calibration jobs launched14:04:11UTC through sole
monitor36160 on the same owned A100. Exactdev70/B70/seed0/all15,Point-MAE-only;
coefficients use the predeclared gradient-ratio rule, not accuracy. ModelNet
completed its15declared corruptions; Scan follows. No coefficient/report or
new accuracy/speed benefit is accepted yet. Next: independently validate both
raw calibration bundles and frozen report, then positive-weight technical
smokes before independent development/confirmation/timing. Spectral/graph
calls remain forbidden; no combination or early exit is enabled.


## Calibration independently accepted; active smoke complete — 2026-10-30 14:18 UTC

[Run/Test] Both70-object/all15/B70/seed0 calibration raw ZIPs match actualDrive/
local SHA/CRC and strict selection/frozen/source checks. Independent CPU
summarize_bundles recomputes the exact report after normalizing only local
versus remote provenance paths. All12candidate×dataset×local/style groups
contain2100finite strictly positive norms; graph/spectral calls0 in30batches.
All4physically downloaded smoke cache NPZs pass actual tensor/RNG/byte checks.
Report72065bytes SHA6c03ad4990197cc8f8a17d6ad4f8c5eaad88d359800b557dd55d1597fb8e5a25.

[Run/Decision] Frozen coefficients: KL0.6285361973358488,
H1.4715424107401223, weightedscale0.9171509895707142. Per-candidate4200ratios
have medians .15909982658734056/.06795590753630018/1.0903330109997096;
predeclared .1/median,.1/median,1/median reproduced. No accuracy-based choice.
Calibration controls the pooled median, not every object's update: scaled
maximum ratios can reach approximately7.01/3.72/2.24 respectively. Do not
reinterpret .1 as a per-object cap or retune from future accuracy.

[Run] Calibration took360.9799s (estimate900s); phaseZIP144129bytes SHA
dceec26aa53ca5bda6e725c33fb244f999a738d0fd6fa9488dc3ee983f6b0b10
matchesDrive/localSHA/CRC. Positive-weight KL/H/weighted6technical jobs
completed281.0998s at14:17:41UTC; phaseZIP144117bytes SHA
dc46ce76c330390ebfb169b99ed51447696f7c8ac77e1ed76b832090187cab5c
also matchesDrive/localSHA/CRC. Its24rawhead ZIPs are downloading/awaiting
independent acceptance before development. No small-smoke accuracy claim.
SameA100/monitor36160, original18:25:16.894181UTC deadline, no teardown.


## Technical acceptance passed; development running — 2026-10-30 14:23 UTC

[Run/Test] All24positive-weight rawheads/48batches independently accepted with
actualDrive/localSHA,size,CRC,2IDs×Gaussian+Background×4heads×bothdatasets.
Frozen KL/H weights and W eta/gamma match the actual frozen calibration report;
all graph/spectral calls0. Four-candidate probability mean/source preprocessing,
semantic10/30callbacks+2losses+1Xforward and W1density+10/30backwards pass.
Actualinitial tensors/RNG/IDs/labels exactly match oldC in every batch; only
asset-source ZIP/helper identities differ by the exact reviewed metadata
repair. Initial8sparse heads also verify4/11backwards and10/30prior forwards.
Geometry finiteness was checked by the producing runtime; raw evidence retains
shape/hash and finite logits, not geometry coordinates. No tiny-smoke accuracy
conclusion. See active_smoke_acceptance_20261030.json.

[Code/Open] Independent raw loading exposed an offline analyzer integration
bug: Point-MAE-only calibration reports4native modules, while all4head runs
add PointNeXt pointnet2_batch_cuda. Shared native bytes and LION/source match;
the extra module exactly matches oldC. Runtime already validates the complete
calibration-native subset correctly. Offline analyzer whole-map equality must
be repaired/tested to allow additional run-only modules while still rejecting
missing/tampered calibration modules. Full C/treatment native-map equality
for paired accuracy remains strict. No GPU source/coefficient rerun is needed.

[Run/Open] Development launched14:22:52UTC through36160: exact10jobs,
5independent arms×2datasets×4heads,dev70/B70/seed0/all15. Actual same-source
C caches initialize each stage; no coefficient or hyperparameter selection
from scores. Then frozen heldout140×3seeds and3fresh timing repeats, unchanged
original18:25:16.894181UTC deadline. Runtime/monitor remain live; no teardown.


## Offline analysis repairs and first partial development result — 2026-10-30 14:36 UTC

[Code/Test/Run] Offline analyzer commit83dbe21 fixes the observed4-native
Point-MAE calibration→5-native four-head inventory mismatch: every calibration
module must match run bytes/SHA, while extra run-only modules are allowed.
Full baseline/treatment native equality remains strict.55targeted tests pass;
all24actual positive smoke ZIPs now load. No GPU source/report/coefficient
changed; the executing source remains sealed0fd2581.

[Run/Code] Existing sameModelNetdev70/B70/seed0/all15 calibration and baseline
fresh encodes show15/15exact input/noise/timesteps/alpha/RNG but15/15different
z0/h0/style/noisy-local hashes. All4baseline heads agree internally. This is
observed encode drift in existing runs with different calibration/readout
workloads, not a claim about future timing data or performance. Configured
encoders use floating CUDA atomic reductions; no bitwise guarantee is valid.

[Code/Test] Offline commit75cec0a gives only fresh timing an explicit pairing
mode: exact raw input/noise/scheduler/RNG plus unchanged provenance/IDs/config;
encoded differences are reported with actual reference/compared hashes.
Cached-accuracy and shared-geometry head checks retain full initial equality.
69targeted tests pass including control-tamper rejection. All30fresh timing
jobs and GPUsource0fd2581 remain unchanged. Timing supplies costs only, never
accuracy inference. Actual timing artifacts remain pending.

[Run/Inference/Open] First partial ModelNet C/sparse development comparison:
70objects/seed0/all15/fourheads probability mean62.738095% C versus61.238095%
sparse, delta−1.500000pp, conditional object-bootstrap95%interval
[−2.261905,−.761310]pp (2000replicates,seed0). Strict complete initial-state/RNG
pairing passes on all8raw heads. This is one dataset/one seed development
only, not a full-study/heldout conclusion. Do not change interval/coefficient
from this score. Other arms, Scan and140-object×3seed confirmation are pending.
Derived local `result/preparation/chamfer_guidance_variants_20261030/development_modelnet_C_sparse_partial_analysis.json`;
analysis83dbe21, GPUsource0fd2581. No speed percentage from fresh-C/cache-arm
development timings; the3fresh repeats remain the speed endpoint.


## Development complete; heldout confirmation executing — 2026-10-30 15:03 UTC

[Run/Test] Exact10developmentjobs/40rawhead ZIPs/600head-batches and34physical
caches are locally size/SHA/CRCverified. Root fullanalyze_campaign validates
all40paired initial states/RNG/IDs/labels/source/native/frozen conditions;
independent32-head/480-batch+all34cache proof plus root last8/120-batch guards
cover the complete stage. Independent final numeric recheck remains pending.
Phase1849.8476s,160829byteZIP SHA
cca33a5a08c9866b3e12d95e4c8bff6aad7cffc04eec66244038ce1f7aa7b4c8
matchesDrive/local. Full local CPU suite455tests/339subtests passes23.48s,
6pre-existingwarnings after offline fixes; GPUsource remains0fd2581.

[Run/Inference] Development probability mean (fourheadall15means,seed0,70IDs):
ModelNetC62.738095%, ScanC40.595238%. Deltas pp (MN/Scan): sparse−1.500000/
−1.119048; KL+.261905/−.452381; H+.047619/−.309524; weighted−.309524/−.928571.
CIs and exactfractions are retained in development_results_20261030.json.
No robust accuracy benefit is established. This is fixed development evidence,
not heldout confirmation; no interval/weight/arm is retuned or dropped.
Cached treatment timings cannot supply deployment speed against fresh C.

[Run/Open] Submitted confirmation15:00:36UTC encountered only CLI prelude
Connection-was-lost; history0matching user-code executions and sameownedVM
proved no GPUworker launched. Only ownedlocalCLIchild79187 was terminated;
monitor36160/VM stayedlive. Same reviewedwrapper retried15:02:00UTC; actual
fourhead ModelNetC/seed0 runbundles began15:02:04UTC. Exact30jobs: bothdatasets,
all5arms, reserved140IDs/B70/all15/seeds0,1,2. No score-based choices, spectral,
combination or earlyexit. User says“Tamamdır devam et” after status; this keeps
the original approval/cap, not a new five-hour clock. Absolute deadline remains
18:25:16.894181UTC. Reassess measured140-object timings and remaining fullphase
cost before further stages. SameA100 and solemonitor36160 remain live.


## Final development recheck and first heldout structure accepted — 2026-10-30 15:24 UTC

[Run/Test] Independent final40-head/600-batch development report exactly
matches Root's2000bootstrap/seed0 report, including corrected/harmedIDs and
CIs, excluding only Root-added analysis_provenance. All34cache proofs remain
accepted. Earlier“independent final recheck pending” is superseded here.
See development_acceptance_20261030.json.

[Run/Test] First3heldout workers (ModelNetC/sparse/KL,seed0),12rawheads and
360batches independently pass structural acceptance with no accuracy/argmax
calculation. Actual140IDs match the frozen heldout manifest and are disjoint
from dev70. Both70-object batches/all15 match30physical confirmation caches:
input/z0/h0/noise and producer postprepare→samplerinitial→postreconstruction
RNG are exact. Native/source/frozen/labels, sparse4/11backwards10/30forwards,
KL2loss/1Xforward/3decodes/12probeheadforwards and graph/spectral0 pass.
Raw audit c81ca85d…cc6228,22375bytes; producing NumPy1 protocol4 hash spelling
conversion is explicit. First C/sparse/KL worker times305.5412/259.1948/318.3081s
are whole-worker durations, not deployment inference speed measurements.

[Run/Open] At15:23UTC4/30confirmationjobs complete; ModelNetseed0weighted
runs. Job order is all5arms×bothdatasets for seed0, then seed1, then seed2;
a conversational“seed1”progress label was corrected against the frozen argv.
The source/coefficients/IDs/order remain unchanged; no clock reset or scope
extension. SameVM/36160 remains live. Full3seed accuracy and30fresh timing jobs
are pending; no gain/adoption conclusion yet.


## Additional source-model control authorized after the current campaign — 2026-10-30

[User report/Decision] The user requests a follow-up because Point-MAE source-only
accuracy on ScanObjectNN-C is low: test only the accuracy-positive method
(current candidate entropy guidance) on Scan with a different source-model and
record results. Explicit instruction: no additional GPU approval question is
needed; perform this after the existing phases finish and their data are
collected. Verbatim authorization is retained in
post_campaign_source_model_authorization_20261030.json.

[Decision/Open] The added experiment changes the frozen classifier producing
semantic guidance, not merely the classifier reading the existing Point-MAE-
guided points. Select an alternate model using validated source-only evidence;
retain matched C versus C+semantic accuracy and runtime controls, frozen model
parameters, existing probability mean, no spectral/combined methods. Prepare
and verify any required gradient-head/calibration/source changes before
launch. Do not infer a new five-hour clock for the existing campaign: its
original18:25:16.894181UTC deadline remains. The explicitly authorized follow-up
is a separate bounded phase after current data collection, with its actual
source/scope/duration recorded. It has not been implemented or launched yet.

## 2026-10-30 — completion-conditioned owned runtime closure authorized

[User report/Decision] The user leaves and directs autonomous continuation,
a progress backup to Drive, and closing our own Colab runtime without another
question ONLY AFTER all required original and alternate-model studies and
verified evidence collection finish. This supersedes the previous keep-live
closure state; no early teardown is authorized. Exact owned session/endpoint,
protected peer and verbatim authorization are in
`knowledge/evidence/chamfer_author_code_audit_20261030/completion_teardown_authorization_20261030.json`.
No peer or unknown runtime/monitor may be stopped.

## 2026-10-30 17:38UTC — heldout confirmation accepted; fresh timing active

[Run/Code/Inference]30/30workers complete in9086.4653s. Phase archive232946bytes,
SHA28e1e16a05b0b0fc9babc26b284010b80b790c59bc58edbdbdb0aa877a0ec9f5;
Drive/local120raw and214physical cache bytes/CRC verified. Independent proof
SHA9484d1f1011af898889bb2ae2f142677a00a83449d6c30087e95e3643743c15b
accepts3600batches and exactly reproduces Root analysis. Four-head all15 H
MN+.3254pp CI[+.0794,+.5714], Scan+.0833pp CI[-.1071,+.2778]; S/W negative
on both/allfourheads. Per-head and limits are in the consolidated results note.
No HPO or method combinations. Three fresh timing repeats launched17:35:16UTC
through solemonitor36160, source0fd unchanged; no remote transfers during
measurement. Later CurveNet/H remains pending until original data collection.

## 2026-10-30 18:16UTC — original timing and complete collection accepted

[Run/Code]30timingworkers/120heads/240batches complete in1918.9431s; phase
ZIP207451bytes SHA81302e31025b7b639f02876863d8c6e7f467e65fd06827aced632a5aafbf4838
passes Drive/local checks.240confirmation+timing raw ZIPs and214physical
caches rehashed into454-record original_combined_physical_manifest. Independent
timing proof finalSHAd5a623ba9bc176309a7378ba0228f14935b25b100d5f9e83172077269432c303
matches Root costs/drift modulo record order. Median paired pipelinecostchange:
S−21.153/−21.069%, H+7.426/+8.155%, KL+7.491/+8.165%, W+.670/+.447% (MN/Scan).
Startup/data loading/archival excluded; G+B B70/all4heads, no full15 or
singlehead deployment claim. Originalcontinuouswall4h42m under5h; ledger
active14093.35s differs. New CurveNet/H-only deployment prepared, no new
research GPU yet. Details in consolidated results note.

## 2026-10-30 18:33UTC — alternate teacher native gate passed; C/H running

[Run/Code] New95a4997/CurveNetH-only164s calibration,2100rows/30native
zero/nonzero XYZ/local/style proofs independently accepted. Fixedlambda
3.153885581354952; frozenreportSHA8e8dd33e15edc17487bbcc1636064bf3de1b4a1fdad5714d6dd777f62e2e5d07.
CPU deploy archivefdd668b3...c789 verified; actual newplan297badb3...f9d46
after original454physicaldata collection. Six new matched C/H workers now
run on reused140Scan IDs/seed0,1,2, fourreadouts; no newaccuracyclaim yet.
One0-usercode CLIprelude retry preserved sameVM/source/cap. Newdeadline
19:33:23.664750UTC; old5hclock not reset. Details/evidence in CurveNet note.

## 2026-10-30 19:08UTC — alternate CurveNet heldout accepted

[Run/Inference]6newC/Hworkers complete in1897.8977s; phase188788bytes
SHAb7268477715a285237812358097df3d56002657eb9ea26822b9762cd5c89074f
passes Drive/local.24raw/720batches/90physicalcaches independently accepted,
proofSHA2a31bba3fbce637ba2b98ca91c6b3799b6a967522248da8de696d3e23b0e2ed7.
PrimaryCurveNet55.6667→55.9048%,+.2381pp CI[−.1587,+.6349]; all3seeds
positive butCIcontains0. Secondaryfour-head+.1944pp; PointNext+.4127pp
doesnotreplaceprimary. Sameheld140posthoc, noHPO/noGSL/noBNupdates.
NewthreefreshC/Htimingrepeats run19:07:55UTC; no remote transferduring
measurement. Newdeadline19:33:23.664750UTCunchanged. Finalcost and
Drive/localsummary thenonlyownedruntimeclosurearepending.

## 2026-10-30 — tüm deneyler ve fresh süre ölçümleri tamamlandı

[Run] CurveNet takip aşamasının 13 worker/49 head ZIP ve90 fiziksel cache'i
kabul edildi. Üç döndürülmüş fresh tekrar: dev70/B70/seed0,
Gaussian+Background, ortak rekonstrüksiyon bir kez ve dört terminal classifier.
C medyanı20.955446s, H medyanı22.759866s; birincil eşlenmiş H/C oranları
1.08557173,1.08981485,1.08468403; medyan ek maliyet **%8.5572**.
Gaussian ek maliyet medyanı%13.0753, Background%6.0390.
Model/native kurulum, toplu veri yükleme, paketleme ve Drive aktarımı bu
batch ölçümüne dahil değildir; tek classifier veya all15 deploy süresi değildir.

[Run/Code]24 timing head/48 batch bağımsız kabul edildi. Input/noise/scheduler/
RNG, kaynak95a4997,658asset, donmuş checkpoint/state ve katsayı tam eşleşir.
40 head-batch encoded-state farkı açıkça kayıtlıdır; fresh encode bitwise
paritesi iddia edilmez ve bu koşulardan accuracy çıkarılmaz. Root ve bağımsız
tam JSON, yalnız encoded-difference satırlarının sırası normalize edilince
birebir aynıdır. Guidance spectral/graph hesap sayıları sıfırdır; classifier
mimarisinin kendi KNN/Curve işlemleri aynen korunur.
[Süre kabulü](evidence/chamfer_author_code_audit_20261030/scan_curvenet_timing_acceptance_20261030.json).
Timing phase ZIP183436bytes SHA1e106a454c0df905ba444355f431ecd8ac9b0164bb2b14f9c6a368e0fb2e2cd9.

[Run] Yeni sürekli GPU saati18:23:23.664750→19:14:23.569UTC,
yaklaşık3059.90s/51dakika;4200s sınırının altında. Ledger active_seconds
2448.191s bekleme/aktarım aralarını içermez; bütçede sürekli süre kullanılır.
Orijinal4saat42dakika/5saat bütçesi değiştirilmedi.

[Inference/Decision] CurveNet kendi accuracy'sindeki +.2381pp farkın95%CI'si
[−.1587,+.6349] olduğundan, %8.56 batch maliyetine karşı güçlü bir fayda
kanıtlanmadı. PointNeXt/ortalama ikincil olumlu işaretleri birincil belirsizliği
kaldırmaz. Şimdilik hiçbir varyant varsayılan yapılmadı veya birleştirilmedi.
Tüm sonuçlar Chamfer-only ve mevcut dört-zamanlı probability mean üzerindedir.
Kullanıcının talimatıyla kalan iş yalnız nihai Drive özeti, dosya doğrulaması
ve sadece sahip olduğumuz runtime'ın kapatılmasıdır; yeni GPU işi yoktur.

## 2026-10-30 — nihai Drive kabulü ve yalnız sahipli runtime kapanışı

[Run/Artifact]407ham head ZIP+304NPZ'nin711yerel dosyası SHA/CRC/güvenli yol
kontrolünden geçti; eksik/çakışma yok.304physicalcache ayrıca Drive'da
`scan_curvenet_h_postcampaign_20261030/phase-evidence/paired_physical_caches_final_remote_20261030T1926Z.zip`:
1290931819bytes SHAea98e68d50330bd4b988594089564c35010a4e86e674c4920c65e510e85bbebd.
Geri indirilen ZIP'nin tam SHA/CRC'si ve304member SHA'sı ayrı doğrulandı.
Nihai260member bilimsel özet aynı Drive klasöründe
`final_research_summary_20261030T1930Z_verified.zip`:78981869bytes,
SHA19e7c6735f0b60438c3347d68d87183b1ee6d886aaca084cb1605746742077f5.
Geri indirme ve259member SHA/güvenli yol/CRC kabul edildi. Dört kurulum
history üyesi paylaşılan özet kopyasından çıkarıldı; orijinal kaynak ZIP'leri
ve ham araştırma kanıtları değiştirilmedi. Özet, kapanış öncesi bilimsel
snapshot'tır; gerçek kapanış makbuzu aşağıdaki yerel/Git kaydıdır.
Son roundtrip kabul makbuzları da Drive'a ayrıca başarıyla yüklendi.

[Run] CLI'nin büyük doğrudan upload'ları400/brokenpipe verdi; başarı olarak
raporlanmadı.304cache VM üzerinde CPU ile paketlenip SHA/CRC doğrulandı;
özet5×16MiBparça ile yüklenip aynı byte'lara birleştirildi. Bir CPU archive
CLI prelude bağlantı kaybı kullanıcı kodu çalışmadan önce kanıtlandı;
yalnız sahipli lokal CLI5577 sonlandırılıp aynı script yeniden çalıştı.
Bu arşiv işleri yeni GPU deneyi veya araştırma bütçesi uzatımı değildir.
Private URL/token içeren ham CLI history paylaşılmadı.

[User report/Run/Decision] Kullanıcının tamamlanınca yalnız kendi session'ımızı
onaysız kapatma talimatı uygulandı. `tta-chamfer-guidance-variants-a100-20261030-r1`,
endpoint`gpu-a100-hm-ass1c0-3nh8smksm50de`; izleyici36160/PID59645
19:34:38.530UTC'de stop protokolüyle exit0 kapandı. Ardından yalnız tam bu
adla `colab stop -s tta-chamfer-guidance-variants-a100-20261030-r1` exit0;
19:34:46UTC itibarıyla sunucu envanteri aktif assignment göstermiyor.
Peer zaten önceki envanterde yoktu; diğer agent'ın runtime/izleyicisine veya
bilinmeyen süreçlere stop/exec/reconnect işlemi yapılmadı. Yeni GPU işi yoktur.
[Cache yedeği](evidence/chamfer_author_code_audit_20261030/final_cache_backup_acceptance_20261030.json) ·
[Özet yedeği](evidence/chamfer_author_code_audit_20261030/final_summary_backup_acceptance_20261030.json) ·
[Kapanış kanıtı](evidence/chamfer_author_code_audit_20261030/owned_runtime_closure_20261030.json).
