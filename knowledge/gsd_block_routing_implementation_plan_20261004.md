# Local/style guidance routing implementation plan

> Implementation is underway in the shared workspace. Batches 1 and 2 are
> complete; Batch 3 is in final review. Checkboxes below track the requested
> implementation and Colab-only experiments remain unlaunched.

**Goal:** isolate which loss should update local latent and style conditioning,
and whether style conflict handling adds value beyond routing or step scaling.

**Architecture:** pure per-example composition; an opt-in extension to the
current sampler; a paired experiment runner that shares prepared inputs and
random draws; versioned result analysis. Keep legacy CLI contracts intact.

**Stack:** Python compatible with the pinned Colab environment (3.8 syntax),
PyTorch, existing DDIM/Chamfer/PointNet2/LION, standard-library CLI/JSON/CSV.
No new dependency installation or neural-network training.

**Spec:** [design and scientific rationale](gsd_block_routing_design_20261004.md).
**Colab:** [scenarios and command contract](colab_gsd_block_routing_20261004.md).
**Base inspected:** `gsd-smooth-spectrum@aa8b725ba4daaf3448f0cda75bfc6aaec6c86d48`.
The workspace is dirty with unrelated work and research artifacts; preserve
all changes. At execution use a separate branch/worktree if needed; explicitly
carry this uncommitted handoff and dependencies into it. Do not reset/stash
other work, switch the shared branch, or stage the entire result tree.

## Global constraints

- Initial method/profile/settings are exactly those locked in the design.
- No simultaneous mean-normalization, smoothing, decoder or global-prior change.
- No raw archive modification and no repeated completed full benchmark for
  missing diagnostics. New small matched controls have a distinct purpose.
- Defaults on old methods must preserve arithmetic, draws and artifacts.
- Labels never enter routing, norm calibration, projections or candidate gates
  during inference; offline accuracy selects only development candidates.
- Pending/running user Colab jobs are not interrupted or reconfigured.
- Planning does not authorize running GPU experiments locally or publishing
  a branch. Document concrete commit/branch before producing executable cells.

## Review focus

1. A new route must not accidentally take the old spectral-weight-zero bypass.
2. Loss needed only for probes must have a live graph even when its applied
   coefficient is zero; unused versus connected-zero gradients are distinct.
3. The final style update is unused with original-style decode; don't claim
   a last-step style change altered the reconstruction.
4. Paired preparation and classifier RNG must not depend on arm execution order.
5. Partial/foreign artifacts must not be silently resumed as a matched block.

## Batch 1 - Pure routing and projection algebra

**Status: complete; spec and quality review approved.**

- Goal: one tested, sample-independent composition function.
- Scope: `off/scd/spectral/sum/pcgrad/scd_priority/sum_norm_pcgrad`; local
  routing initially only off/scd/spectral/sum. Style permits every mode.
- Touched areas: create `gsd_composition.py`, `tests/test_gsd_composition.py`.
- Stack context: torch tensors, no CUDA/native imports needed for CPU checks.
- Dependencies: read design, especially norm cap and near-zero semantics.
- Implementation notes for $development-agent:
  - Interface `compose_block(scd_grad, spectral_grad, *, scd_weight,
    spectral_weight, mode, norm_floor=1e-12) -> (direction, diagnostics)`.
    Inputs have identical shape `[B,...]`, floating dtype/device; weights
    finite and nonnegative. Apply each weight once; no in-place input mutation.
  - Diagnostics return detached per-example records, including raw/capped
    projected norm and cap factor. Undefined cosine/angle serializes as null,
    not NaN. Zero-weight omission is not a disconnected-gradient flag.
  - `CompositionConfig` stores `local_mode`, `style_mode`, four explicit
    weights (local/style x SCD/spectral), `norm_floor`, schema version 1.
    Initial coefficients all 1; masked gradients still available for probes.
  - Validate mode/block combination and dimensions before reduction.
- Verification:
  - [ ] Write CPU tests before implementation: agreeing vectors unchanged;
    `c=(1,0), q=(-1,1)` gives raw symmetric sum `(.5,1.5)` then norm cap 1;
    applied direction is `(.5,1.5)/sqrt(2.5)`, matched sum is `(0,1)`.
  - [ ] Test opposite vectors produce zero, zero/tiny/absent/nonfinite cases,
    dtype preservation, weight application exactly once, no in-place mutation.
  - [ ] Opposite conflict signs in a two-example batch must be treated
    separately; batch permutation must permute outputs only.
  - [ ] Test first-order dot signs for symmetric direction and that
    SCD-priority can leave spectral dot negative.
  - [ ] Run `python -m unittest discover -s tests -p "test_gsd_composition.py"`.
- Knowledge artifact to update: append implementation evidence to design
  and `findings_log.md`, without accuracy claims.
- User review gate: inspect algebra/API diff and passing CPU evidence; this
  is a reviewable checkpoint, not a request for an additional experiment.
- Out of scope: sampler, CAGrad, norm calibration and new GPU execution.

## Batch 2 - Paired sampler inputs and hard-v1 diagnostics

**Status: implemented and reviewed; focused trajectory tests pass.**

- Goal: opt-in routing on the unchanged host trajectory with controlled draws.
- Scope: prepare once, clone state per arm, route both block gradients and
  archive exact v1 probes. Default path remains backward compatible.
- Touched areas: `tta_gsd.py`; create `gsd_paired_inputs.py`,
  `tests/test_gsd_composition_trajectory.py`; read `gsd_calibration.py`,
  `graph_spectral.py`, `tta.py`, `tests/test_gsd_trajectory.py`.
- Stack context: model parameters frozen, state gradients enabled, eval mode.
- Dependencies: Batch 1 accepted.
- Implementation notes for $development-agent:
  - Define `PreparedGuidanceInputs` with detached original shape latent,
    original local latent, original style conditioning, local Gaussian noise,
    timesteps/scheduler contract and input/config hashes. Do not store labels.
  - `prepare_guidance_inputs(x, lion, *, total, steps_back_local)` returns
    this bundle, using exactly the existing encode/noising procedure.
    `reference_xyz` is derived from original local latent; preserve ordering.
  - Add optional keyword arguments `composition_config=None`,
    `prepared_inputs=None`, `composition_observer=None` to
    `tta_gsd_reconstruct`. A supplied composition config explicitly enters
    the new path; legacy zero-weight delegation remains for config=None.
    Reject ambiguous combinations with old calibration probes.
  - New composition weights are authoritative; reject nondefault legacy
    scalar-weight arguments rather than silently combining two weight systems.
  - Clone working states per arm. Resolve frozen graph from the same encoded
    reference/config. Gradient probes must call actual v1 `SpectralTarget.loss`
    even for an SCD/off applied route; do not substitute smooth hard-v2 probes.
  - Original shape latent remains final decode input. Observer records step,
    timestep, both gradients' block statistics, actual routed displacements,
    style drift and signed local guidance/DDIM geometry. No gradient vectors
    or computation graphs retained across steps; retain only detached scalars.
  - Keep graph lifetime sufficient for both losses, then release it. Mark
    unused gradients before replacing them with shape-correct zeros.
  - The pure `off/off` route can share this instrumented host; verify numerical
    parity with unguided sampling. Diagnostic losses must not change outputs.
- Verification:
  - [ ] With mocked frozen models/scheduler, `scd/scd`, `spectral/spectral`,
    `sum/sum` match existing corresponding sampler outputs given same states.
  - [ ] Style off is unchanged at every step; local off still follows DDIM.
    Routing changes only intended gradient contribution at a shared state.
  - [ ] Same prepared bundle replay and reversed arm order give identical
    outputs in the deterministic CPU fixture; assert original bundle unchanged.
  - [ ] Test nonidentity `global2style`; final decoder still receives original
    raw shape latent. Last-step-only style perturbation changes logged style
    but not final reconstructed points under this decode contract.
  - [ ] Observer on/off parity; parameter gradients absent; state gradients
    connected; actual 5/35 step counts and scheduler defaults preserved.
  - [ ] Run new trajectory and existing `test_gsd_trajectory.py` tests via
    `python -m unittest discover -s tests -p "test_gsd*trajectory.py"`;
    run `test_gsd_composition_trajectory.py` explicitly if glob differs.
- Knowledge artifact to update: design/code map plus findings log; record
  any actual scope/config mismatch rather than weakening old tests.
- User review gate: sampler parity and finite-gradient evidence before Colab.
- Out of scope: global-prior activation, updated final decode, new loss.

## Batch 3 - Colab runner, manifests and analysis

**Status: implemented; focused CPU tests pass; final independent review pending.**

- Goal: one reproducible suite with dry-run planning, partial-run export and
  diagnostics that distinguish routing, conflict repair and smaller steps.
- Scope: independent new-method runner; do not relax old method stage guards.
- Touched areas: create `gsd_composition_protocol.py`,
  `scripts/run_gsd_composition.py`, `scripts/analyze_gsd_composition.py`,
  `tests/test_gsd_composition_protocol.py`, `tests/test_gsd_composition_runner.py`,
  `tests/test_analyze_gsd_composition.py`; reuse `research_artifacts.py` and
  read/import suitable preprocessing/classification helpers in `run_baseline.py`.
- Stack context: Python CLI; existing conda environment `3dd_tta_env`.
- Dependencies: Batch 2 accepted, scenario table below treated as normative.
- Implementation notes for $development-agent:
  - Protocol owns `ArmSpec`, named phase arm lists, split manifests, locked
    configuration and stable experiment fingerprint. No new options on old
    runners just to bypass their validation.
  - Launcher contract: `--phase {smoke,diagnose,scale,routing,projection,replicate,all15}`,
    `--result-root PATH`, optional `--reference-manifest PATH`,
    `--selection-manifest PATH`, `--execute`; default prints commands/scopes
    and performs no model inference. Candidate choices come from manifests,
    not ad hoc command strings. JSON output records resolved commit and args.
  - One worker per seed/phase loads models once and runs arms on each shared
    prepared batch. Preprocess once, encode once, draw local noise once;
    reset classification/FPS RNG to a shared snapshot for each arm. Different
    reconstructed geometries need not yield identical FPS indices.
  - Preparation draw keys must be phase/arm independent: SHA256 of a canonical
    serialization of dataset, corruption, model seed and ordered batch indices;
    derive RNG seeds from this key and save/restore Python/NumPy/torch CPU/CUDA
    RNG states around preparation. Use a separate classification key. Record
    the derivation version; never use Python's randomized `hash()`.
  - Save tensor/input/noise/config hashes proving the shared preparation.
    If cross-sample independence is not established, label batch-paired scope;
    do not claim batching-invariant inference from shared seeds.
  - Resume by a complete corruption/seed paired block. A partial block is
    preserved and recomputed into a new attempt directory; never merge arms
    from different preparation/runtime identities. A later phase may reuse
    archived controls only after exact fingerprint/input/hash agreement.
  - Use new method `gsd_guidance_composition_v1`; per-arm run directory contains
    the seven standard files plus `predictions.npz`,
    `gradient_diagnostics.jsonl.gz`, `experiment_manifest.json`. Its new
    schema/export validator accepts this exact inventory; old seven-file
    validators stay unchanged. Store labels, original indices, predicted
    classes and 40 logits in predictions. Persist failure/partial status.
  - Hash source/native libraries/checkpoints/data and record actual timesteps,
    reduction/rank, all route weights, eval/frozen status and shared-draw keys.
    Export ZIPs automatically after successful or interrupted phases when safe.
  - Analyzer: `--result-root PATH --output PATH`; write validation, arm/seed/
    corruption accuracy, paired transitions, gradient geometry summaries,
    complementarity/oracle, runtime/memory and uncertainty. Reject duplicate
    indices, wrong counts, nonfinite logits, config conflicts or mixed inputs.
    Keep macro15, macro14-excluding-background and Background separate.
    A pooled fraction must not silently replace macro accuracy.
  - Report preparation/graph cost, per-arm sampling and classification time
    separately: shared-work suite timings are not standalone deployment timings.
- Verification:
  - [ ] CPU protocol tests pin phase counts/indices/mode names, immutable legacy
    defaults, reject impossible routes and absent selection manifests.
  - [ ] Runner dry-run makes no CUDA/model calls; resume rejects wrong binary,
    seed, loss reduction, weights, indices and incomplete arm inventory.
  - [ ] Fixture archives validate predictions versus correct counts; invalid
    paths/duplicates/mixed manifests fail; export preserves raw data.
  - [ ] Run `python -m unittest discover -s tests -p "test_gsd_composition*.py"`
    and `python -m unittest discover -s tests -p "test_analyze_gsd_composition.py"`.
  - [ ] Run every phase dry-run, including selected manifest fixtures; no real
    accuracy or model-runtime claim from these checks.
- Knowledge artifact to update: Colab scenario file with actual CLI and
  tested branch/commit, `result/README.md` with method-scoped schema, findings.
- User review gate: deliver concrete two-cell Colab instructions after code
  exists and smoke is ready; user launches GPU work.
- Out of scope: launching all phases automatically or auto-promoting winners.

## Batch 4 - Colab mechanism pilot and evidence review

- Goal: complete the phase gates in the scenario document and select at most
  one mechanism candidate for replication.
- Scope: smoke -> diagnose -> routing; projection only if v1 style conflicts
  are observed. Preserve null results; no automatic all15 launch.
- Touched areas: Colab artifacts under new method path; analysis outputs and
  the three knowledge documents, findings/open questions.
- Stack context: user-run pinned Colab GPU runtime; no local GPU commands.
- Dependencies: CPU parity and complete launch manifest from Batch 3.
- Implementation notes for $development-agent: ingest complete ZIPs, validate
  identity before comparison; distinguish technical completion from evidence
  of an accuracy effect; report actual measured cost, not estimates as facts.
- Verification: scenario acceptance/counts/provenance; paired uncertainty
  clusters by underlying object where correspondence is verified. Seed repeats
  are not independent new objects. Development selection is not confirmation.
- Knowledge artifact to update: dated result note linked from README plus
  scenario gates checked with source artifacts.
- User review gate: review diagnostic/pilot findings before replication;
  review replicated effect/cost before one optional all15 run block.
- Out of scope: full grid, new datasets, conclusion from one lucky seed.

## Conditional follow-ups: record them, do not implement all now

Each follow-up is a separate batch after the relevant evidence gate. The next
agent may finish Batches 1-3 without implementing these features.

### Registered next test: is v1 guidance too weak to affect accuracy?

[User decision/Open] The user explicitly requests this question in
`open_questions.md` and an isolated subsequent study. This experiment changes
only the scalar v1 spectral coefficient in the existing SCD+GSD sum. It does
not test routing or projection and must not be combined with the mean-reduction
or smooth-profile work. It is planned after the runner and v1 diagnostics pass
technical smoke; implementation may include the `scale` phase at that point.

Use common prepared data/noise and exactly matched SCD weight, loss reduction,
graph/rank, DDIM trajectory, update rates and decoder. Pilot coefficients are
spectral `gsd_weight ∈ {0, 1, 100, 1000}`; all other settings remain the locked
v1 defaults. The 0 arm is the matched SCD-only control. With historical local
ratio around .09%, 100 approximates a 9% local spectral/SCD norm contribution;
1000 approaches parity. This is a rough scale interpretation, not a predicted
performance optimum; recompute ratios on the shared current states. Keep one
scalar coefficient for both blocks, and report local/style separately.

- Pilot: 64 examples each on Gaussian, Impulse, Background and Shear, seed0;
  four arms x four corruptions x64 =1,024 predictions.
- Diagnostics: weighted blockwise spectral/SCD norm ratios, local/style
  guidance displacement versus DDIM displacement, per-step update size and
  finite-state checks. Pair predictions and retain correctness transitions.
- Gate: if 1000 is nonfinite or repeatedly gives guidance displacement over
  2x DDIM displacement on more than10% of nonterminal example-steps, flag it
  as an unstable exploratory scale and do not replicate it. This is a safety
  diagnostic, not evidence that smaller is accurate. Rank non-unstable scales
  against the 0 arm; positive pilot evidence only nominates one scale.
- Replicate one nominated scale and the 0/1 controls on the next256 manifest
  examples, seeds0/1/2, four corruptions. Preserve per-corruption outcomes;
  require positive mean paired accuracy change across the three seeds to keep
  the hypothesis alive. This is still development data, not independent
  confirmation. Do not launch all15 automatically.
- Null/negative results are useful: if materially larger applied gradients
  change predictions but not accuracy, small gradient magnitude alone is
  insufficient. If it barely changes state/predictions until the unstable
  region, attribution remains unresolved and the hypothesis is weakened.

[Plan] Phase CLI contract adds `scale`; the launcher must enforce exactly the
four pilot coefficients and locked config above. Replication reads a frozen
scale selection manifest. Report both raw GSD weight and observed ratios;
never label coefficient 1000 a calibrated optimum.

### Registered next test: is v1 guidance too weak to affect accuracy?

[User decision/Open] The user explicitly requests this question in
`open_questions.md` and an isolated subsequent study. This experiment changes
only the scalar v1 spectral coefficient in the existing SCD+GSD sum. It does
not test routing or projection and must not be combined with the mean-reduction
or smooth-profile work. It is planned after the runner and v1 diagnostics pass
technical smoke; implementation may include the `scale` phase at that point.

Use common prepared data/noise and exactly matched SCD weight, loss reduction,
graph/rank, DDIM trajectory, update rates and decoder. Pilot coefficients are
spectral `gsd_weight ∈ {0, 1, 100, 1000}`; all other settings remain the locked
v1 defaults. The 0 arm is the matched SCD-only control. With historical local
ratio around .09%, 100 approximates a 9% local spectral/SCD norm contribution;
1000 approaches parity. This is a rough scale interpretation, not a predicted
performance optimum; recompute ratios on the shared current states. Keep one
scalar coefficient for both blocks, and report local/style separately.

- Pilot: 64 examples each on Gaussian, Impulse, Background and Shear, seed0;
  four arms x four corruptions x64 =1,024 predictions.
- Diagnostics: weighted blockwise spectral/SCD norm ratios, local/style
  guidance displacement versus DDIM displacement, per-step update size and
  finite-state checks. Pair predictions and retain correctness transitions.
- Gate: if 1000 is nonfinite or repeatedly gives guidance displacement over
  2x DDIM displacement on more than10% of nonterminal example-steps, flag it
  as an unstable exploratory scale and do not replicate it. This is a safety
  diagnostic, not evidence that smaller is accurate. Rank non-unstable scales
  against the 0 arm; positive pilot evidence only nominates one scale.
- Replicate one nominated scale and the 0/1 controls on the next256 manifest
  examples, seeds0/1/2, four corruptions. Preserve per-corruption outcomes;
  require positive mean paired accuracy change across the three seeds to keep
  the hypothesis alive. This is still development data, not independent
  confirmation. Do not launch all15 automatically.
- Null/negative results are useful: if materially larger applied guidance
  changes predictions but not accuracy, small gradient magnitude alone is
  insufficient. If it barely changes state/predictions until the unstable
  region, attribution remains unresolved and the hypothesis is weakened.

[Plan] Phase CLI contract adds `scale`; the launcher must enforce exactly the
four pilot coefficients and locked config above. Replication reads a frozen
scale selection manifest. Report both raw GSD weight and observed ratios;
never label coefficient 1000 a calibrated optimum.

| Follow-up | Gate / smallest implementation | Required comparison and stopping rule |
|---|---|---|
| Fixed style-scale calibration | Native spectral style route acts like off; add label-free median nonzero `norm(scd)/norm(spec)` on diagnose states, capped multiplier at 100, record capped fraction. Initial alternative target ratio .1, one frozen coefficient; no per-example equal-norm forcing. | Native route, calibrated route and style-off with same local SCD; stop if effect is negligible or unstable. This is a new scale study, not a rerun of archived smooth grids. |
| SCD-priority style | Symmetric style projection actually activates and changes result | Add `scd/scd_priority`; check how much spectral progress is sacrificed, not just SCD loss. |
| Local projection | V1 local negative cosines appear | Enable per-example local PCGrad using same helper; same local-norm control, fixed style route. |
| CAGrad style | Widespread conflicts and symmetric projection yields problematic stagnation/trade-offs | Separate helper with weighted gradients; mean `g0=(c+q)/2`, trust parameter .5. Solve published two-task dual, verify against a dense scalar-grid CPU reference, cap its final direction at ordinary-sum norm. Record solver tolerance and compare with norm-matched sum; no diffusion convergence claim. |
| Probability ensemble | Paired standalone predictions show complementarity | Use `.5*softmax(logits_scd)+.5*softmax(logits_spec)` without labels or fitted weights. Compare two SCD trajectories and two spectral trajectories using same two draw IDs and equal adaptation compute. |
| Timestep schedule | Per-step geometry suggests roles differ with time | One early-spectral/late-SCD style schedule and reversed order, local SCD fixed; early means first ceil(K/2) reverse steps for K=5/35. Same weights and explicit per-step applied norms; no extra substeps. Recomputed sequential substeps are a separate future design. |
| Frequency subspace routing | A chain-rule-correct state-space formulation exists | No generic 'apply U to style' implementation. Requires a new mathematical spec and correspondence checks before code. |
| Global diffusion | Replicated style contribution beyond style-off OR a separately accepted global-prior mechanism hypothesis | Perform the representation audit and small four-arm sequential experiment in the Colab document; no unfreezing or direct legacy dual migration. |

For norm calibration, omit near-zero denominators using the same 1e-12 norm
threshold; if no valid samples, fail calibration rather than invent a scale.
The style spectral multiplier is exactly
`min(100, .1 * median(norm(g_scd_style)/norm(g_spec_style)))` over valid
nonterminal C_SCD probe observations in the declared diagnostic manifest;
keep the local spectral coefficient unchanged. Save the uncapped coefficient
and whether the cap activated; do not call this per-example adaptive scaling.
The .1 target and 100 multiplier cap are **predeclared exploratory choices**,
not measured optima. All follow-ups retain development/confirmation labels.

## Copyable next-agent handoff

```text
Use $development-agent. Implement the core of the 2026-10-04 local/style
guidance plan, Batches 1-3, and stop at the ready-for-Colab handoff.
Read knowledge/README.md, thesis_scope.md, experiment_protocol.md,
gsd_block_routing_design_20261004.md,
gsd_block_routing_implementation_plan_20261004.md and
colab_gsd_block_routing_20261004.md first.
Inspect current branch/status; preserve dirty user files and raw result ZIPs.
Implement pure per-example block composition, opt-in hard-v1 sampler routing,
shared prepared-input/random-draw experiment execution, and strict artifacts.
Keep old methods numerically unchanged. Test CPU algebra/trajectory/protocol
as specified. No global diffusion, decoder change, smoothing/mean experiment,
automatic full benchmark, or local GPU run. Update knowledge after each batch.
Deliver exact branch/commit plus two Colab cells for the smoke phase and
later phase commands. Report unsupported environment checks explicitly.
```

## Plan self-review

[Planning] Design coverage: routing, style-only norm-capped symmetric PCGrad,
near-zero/batch semantics, frozen/original decode, hard-v1 probes, paired RNG,
strict resume, missing predictions and Colab gates each have an owning batch.
Broader methods and global diffusion are deliberately conditional. No proposed
CLI is represented as already implemented. The user's request is for this
plan and handoff; implementation remains with their chosen separate agent.
