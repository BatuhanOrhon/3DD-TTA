# Legacy trajectory observer — implementation and Colab handoff

Date: 2026-10-11. Evidence: [Code], [CPU test], [Run], [Inference], [Open].
Base HEAD: `f670c07440533f01e10fd149fad2480d36f9194c`, branch
`gsd-smooth-spectrum`. The user subsequently authorized GitHub publication;
source distribution now uses this branch. No reset/clean, local trained-model
inference or GPU run. Unrelated working-tree changes remain separate.

## Preserved baseline and bounded question

[Run/User report] Operational reference remains **Legacy B70/M800 seed0,
64.570502%**, accepted as practical reproduction of historical64.5327%.
Offdiag and Phase6 C_SCD are not substituted. Static Legacy graph, spectral1.17,
Chamfer1, beta2, gamma/eta.01, lambda.95, normal10/Background30 and original
updated-style terminal decode are unchanged. `M_max` CLI default remains1200
for backward compatibility; supplied Colab commands explicitly use800.

Raw archive remains at
`result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/20261010T114156946419Z_handoff_20261010T141536060023Z.zip`.
Its SHA-256 was rechecked:
`534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968`.
Existing validation, M1800 and failed Background100/1000 evidence are preserved.

[Inference/Open] Test whether clean intermediate outputs complement terminal
errors and whether label-free minimum entropy or equal probability averaging
can exploit them. No accuracy increase, entropy threshold, promotion threshold,
latency gain or novelty claim is established by this implementation.

## Implementation contract

- `tta_gsd_v2.py`: optional `trajectory_observer=None`; no observer imports,
  probes or timing work in the default loop. Clean observation occurs immediately
  after `scheduler.step`, before current guidance, with detached/cloned
  `pred_original_sample` and the exact pre-update `style_cond`. Normal3/5/8 and
  Background8/15/23 are one-based reverse-loop observation counts, not noise
  depth changes. Terminal remains the existing final noisy-local/style decode.
- `eval_gsd_tta_v2.py`: `--trajectory_mode off|audit|observe`, default off.
  Audit records state/terminal identity without extra model forwards; observe
  adds three clean probes. Probe preprocessing repeats the unchanged ModelNet
  rotate-back/normalize/FPS1024 path; fixed `classification_only(...,
  only_unmasked=False)`. No classifier weight, BN, token or masking changes.
- `legacy_trajectory.py`: Python/NumPy/Torch CPU and initialized CUDA RNG
  restore in `finally`; eval-only, no-grad probe. Restore changed buffers only,
  avoiding version increments on unchanged buffers used by the live autograd
  graph. Clean inputs are clones; no classifier entropy backward. Record
  prepared input, encoded latent, style, noise, initial state, schedule and
  terminal latent/style hashes; main RNG hashes at each batch boundary.
- `legacy_gsd_artifacts.py`: explicit original-index manifests, complete/failure
  metadata, source/assets/native identity, model state hashes before/after,
  trajectory artifact SHA/size manifest and sealed ZIPs. Existing predictions
  retain unchanged original terminal logits. Execution can be complete while
  subset coverage remains partial; no subset is labelled a full benchmark.
- `scripts/analyze_legacy_trajectory.py`: NumPy/standard-library CPU analysis,
  no torch/model/native imports. Validate finite40 logits, candidates/timing,
  coverage, original IDs/labels, terminal identity, file integrity, frozen
  model state and matched source/assets/runtime across pooled seeds.

Per-batch `trajectory/<corruption>/NNNN.json` contains seed, IDs, analysis-only
targets, candidate names/kinds/style timing, zero-based loop index, one-based
loop count, timestep, hashes and component times. Candidate NPZs contain logits,
argmax and normalized entropy. Geometry is optional, bounded to0–2 examples
per batch/clean candidate, in raw LION decoder coordinates; default0, smoke1,
pilot0. Large latent states or guidance graphs are not retained.

Timing uses synchronized component wall seconds: input preprocessing, encoding,
graph/eigh, prior/scheduler, guidance losses/diagnostics, backward/update,
probe decode/preprocess/FPS/classifier, terminal decode/preprocess/classifier,
hashing/artifact work. Overall time and peak allocated/reserved memory are
reported. These instrumented timings include overhead; they are not a warmed,
pure-inference B1 latency benchmark or a separable GSD/SCD cost ablation.

## Analysis policies and interpretation

1. Original terminal logits, no additional terminal forward.
2. Minimum normalized entropy at temperature1 across the three clean candidates
   and terminal. Exact ties prefer terminal, then the earliest clean checkpoint.
3. Equal arithmetic mean of **class probabilities**, not logits, IDs or geometry.
4. Fixed probe1/2/3 controls; probe positions map to3/5/8 or8/15/23 by horizon.
5. Label-using any-candidate-correct oracle, explicitly diagnostic only. It is
   a ceiling for choosing an existing candidate, not a probability-mean ceiling.

Outputs: `analysis.json`, `per_corruption_seed.csv`, `paired_examples.csv`.
Report per-corruption/seed accuracy, corrected/broken, selected histogram,
temporal disagreement, stable wrong predictions, entropy of correct/wrong
candidates, macro over available corruption coverage, without-Background and
Background separately, sample seed SD only when multiple seeds exist (null
for this seed0-only pilot), and1000 fixed-seed stratified bootstrap
replicates of original indices retaining their seed observations. Available
coverage is explicit; timesteps are not extra independent examples.

[Inference] Low entropy does not guarantee correctness or temporal stability.
Stable predictions may be confidently wrong. Full-trace selection uses all
future candidates and does not constitute an online stop rule or save diffusion
steps. Same-cost repeated terminal voting remains a control if gains are later
attributed specifically to time diversity.

## Colab delivery and stops

Use [the separate continuation notebook](../scripts/gsd_legacy_trajectory_colab_20261011.ipynb)
after the first6 **code** preparation cells in the user's original notebook.
The original notebook and its user changes remain intact. Do not run its later
historical experiments or use Run all on the continuation notebook.

Source delivery is GitHub `origin/gsd-smooth-spectrum`: fetch, switch, then
`git pull --ff-only origin gsd-smooth-spectrum`. No source ZIP upload/extraction
is required. Existing preparation ZIPs remain historical local artifacts and
are not experiment results. The continuation notebook verifies sources with:

```text
conda run --no-capture-output -n 3dd_tta_env python scripts/legacy_trajectory_colab.py verify
```

`cfgs/legacy_trajectory_provenance.json` carries the accepted baseline's asset
hashes and explicit inference-source inventory (77 baseline sources plus the
observer, analyzer, launcher and classifier config). Runtime verification
checks those81 files and the provenance file against one resolved Git commit,
normalizing CRLF only. Source mismatches stop before inference. Unrelated
build/lib changes are ignored. The session records commit, branch, per-file
SHA/size and baseline identities; changing that identity requires a new session.
No local raw baseline ZIP is needed on Colab. Data/checkpoints are still supplied
by the first6 preparation cells and verified against accepted baseline hashes.

Sequence:

1. Git branch update and conda CPU initialization/preview. A label-free
   shuffle with local seed20261011 chooses280 original indices, sorted and
   shared across all corruptions/seeds. The smoke uses the first140 of that
   list, solely for integration. Checkpoint/index manifests precede inference.
2. **Smoke:** Gaussian + Background, seed0,140 examples each, modes off/audit/
   observe:6 arms,840 example processings. Each arm validates complete locked
   IDs, finite logits and (when enabled) unchanged model state. Audit replays
   VAE encode with the exact pre-encode RNG state and records latent/buffer
   equality. Observe records RNG/buffer hashes around every probe. Cross-process
   latent/logit equality is descriptive only because a same-mode GPU repeat
   already varied. **Stop, download and review the full smoke ZIP before pilot.**
3. **Pilot:** all15 ×280 ×seed0 =4,200 terminal classifications plus
   three clean probes each. Four B70 batches/corruption. Same10/30 horizons,
   no early exit and no full2468 rerun. CPU analyzer uses these same traces.
   **Stop and review; no automatic benchmark promotion.**
4. Every sealed arm ZIP, including failed ones, is copied to mounted Drive
   with SHA verification. The packaging/download cells can run after smoke,
   pilot or failure and include partial directories/logs/status too. A new
   timestamped handoff ZIP goes beside the session directory, plus
   `MyDrive/thesis/results/legacy_trajectory/`. Download and provide the full
   ZIP, not screenshots/percentages; check for credentials before sharing.

Session path:
`result/modelnet40_c/legacy_trajectory/<UTC>_trajectory/`.
No overwrite/resume or automatic rerun into existing output paths. On OOM or
partial pilot, preserve the attempt; review missing coverage before scheduling
anything again. Completed historical conditions are not rerun by this plan.

All computational Python jobs in Colab use:

```text
conda run --no-capture-output -n 3dd_tta_env python ...
```

Only Colab's drive mount/download UI helpers run in the notebook kernel.
Local conda was not installed/discoverable; the user explicitly approved the
existing local Python for synthetic CPU verification only.

## Verification and remaining questions

[CPU test] Tests cover actual Legacy loop execution with synthetic CPU
scheduler/prior/decoder/Chamfer substitutes, pre-update style timing, complete
guidance/terminal equality across off/audit/observe, Python/NumPy/Torch RNG,
buffer restoration on failure and unchanged buffer version counters, finite
policy arithmetic/ties, artifact corruption rejection, ID/manifest guards,
and CPU-simulated bundle lifecycle. Observer-stripped AST still matches pinned
Legacy mathematics; the graph file remains byte-identical to the restoration.
The final `python -m unittest discover -s tests -q` passed **234 tests**
in28.480s (exit0). Receipt/log:
`result/modelnet40_c/diagnostics/legacy_trajectory_20261011/verification.json`
and `cpu_tests.log`.
Python3.8 syntax is checked for the new Colab modules; this is not proof of the
installed Colab dependency/native-kernel behavior.

[Open] Real CUDA smoke/peak memory and pilot accuracy/cost await the user's
Colab execution. A review agent completed a partial review and found a pooling
provenance omission, now fixed (native/external sources/environment checked);
the reviewer stopped at its usage limit, so do not claim a completed independent
review of the final patch.

[Open] Conditional later work: useful selection/averaging signal; equal-cost
terminal voting; prefix-only locked stopping criteria/fallback; actual onlineB1
exit and same-B1 full-horizon net latency including probes; separate B70
throughput; horizon when corruption identity is unknown; matched compute-skipping
GSD/SCD ablations; independent evaluation/novelty. Current10/30 corruption-name
horizon is disclosed and unchanged. Purge-Gate remains advisor agenda/comparator,
not implemented and not advisor-approved.


## Later 2026-10-11 — User-directed seed0 screening stage

[User decision/Code/Open] The initial three-seed launch plan is superseded:
all currently prepared smoke/pilot experiments use inference seed0 only.
All15 coverage,280 IDs per corruption, B70, M800,10/30 horizon and probes stay
fixed. Pilot work falls from12,600 to4,200 examples. Index-selection seed20261011
is intentionally unchanged to preserve the subset. The seed0 source ZIP was the intermediate delivery mechanism, now superseded
by Git delivery. Both preparation ZIPs and previous verification are retained
locally for history.

Review seed0 policy gains/losses, corruption heterogeneity and added cost first.
If useful, freeze the selected policy before separately evaluating seeds1/2.
No empirical effect size, significance cutoff or gain is invented. A single
seed's index-bootstrap intervals do not establish cross-seed stability; seed SD
is null. Analyzer still supports later multi-seed input, but the current runner
and notebook request only `pilot_seed0.zip`. No full2468 run is added.

Historical source-overlay installation is superseded by Git delivery.
Git conflicts or source mismatches stop; no automatic overwrite is performed. Existing run directories are
never rewritten. If an earlier smoke has already completed, inspect it before
scheduling redundant execution under a new session.


## Smoke handoff recovery — 2026-10-11

[Artifact/recovery] User-provided
`20261011T145239Z_trajectory_handoff_20261011T145805Z.zip` was verified and
preserved at `result/modelnet40_c/legacy_trajectory/` (SHA-256
`7cd75452ce7d64f3e3359ee66edadb5f08c9cf2d7a57c074b0c12bbf1bed488e`). All six
smoke arms completed their locked 140 examples. The runner failed after those
arms when it tried to serialize tuple-keyed `(corruption, mode)` validation
results as JSON. This is a reporting failure; no inference arm needs repeating.
The launcher now uses JSON-safe string keys, with a CPU regression test.

[CPU reconstruction] All six nested archives pass coverage, locked-ID, finite
logit, and artifact checks. Observe arms record six clean probes each and
preserve their RNG and model buffers; audit records preserve the replay RNG
boundary and buffers. However, exact same-RNG VAE latent replay failed in all
four audit batches, even though RNG was replayed/restored and buffers stayed
unchanged. The precise nondeterministic operation remains unknown. This finding
keeps cross-process parity descriptive and does not invalidate the completed
within-arm checks. Off arms do not report a model-state audit (`null`).

[Exploratory only] On the partial 140-example observe subsets, Gaussian
terminal/minimum-entropy/probability-mean accuracy was 76.43/79.29/79.29%;
Background was 57.14/54.29/55.00%. The direction differs by corruption and
the smoke subset is not an all-15 evaluation or evidence of a general gain.
Gaussian and Background observe predictions differed from their audit controls
on 10/140 and 22/140 examples respectively. No accuracy threshold or early
stop rule is inferred. Full recovered CPU reports are under
`result/modelnet40_c/diagnostics/legacy_trajectory_smoke_20261011_145239Z/`.

[Open] Before relying on exact replay parity, isolate the VAE encode/operator
that changes latents under identical recorded RNG and unchanged buffers. Pilot
execution should use a valid gate produced by the fixed runner; do not rerun
these already completed smoke inference arms solely to repair JSON reporting.

[Recovery workflow] `legacy_trajectory_colab.py recover-smoke` imports the six
completed Drive ZIPs into a fresh session only after validating current source
identity, smoke coverage/IDs, seed0, archive CRCs, and that intervening Git
changes are limited to reporting/orchestration files. It refuses collisions,
analyzes the recovered candidates, and writes the gate last. The notebook has
separate recovery/review, pilot, and package/download cells; the pilot remains
an explicit user-run Colab step.


[CPU verification, seed0 revision]234 tests passed in29.834s. The launcher/session
checks assert only seed0 and4,200 pilot examples; the one-seed analyzer reports
null seed SD. Notebook syntax, all81 package source hashes and safe prior-overlay
upgrade were verified. Receipt:
`result/modelnet40_c/diagnostics/legacy_trajectory_seed0_20261011/verification.json`.
No GPU inference or Git publication occurred.


## Later 2026-10-11 — Git delivery at user request

[User decision/Code] Publish the observer, analyzer, CPU tests, provenance spec
and separate continuation notebook on `gsd-smooth-spectrum`. Remove the
`package-source`/upload/overlay workflow; preserve results ZIP packaging,
per-arm Drive backups and smoke/pilot review stops. Only this development's
files are selected for publication; existing user notebook edits and unrelated
research records/artifacts remain untouched. No inference or experimental
conclusion changes. Seed0-only screening and10/30 horizons remain fixed.
