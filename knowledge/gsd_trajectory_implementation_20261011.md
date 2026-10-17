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


## 2026-10-11 - Recovered smoke evidence and pilot failure handling

[Run/CPU] The user-provided `20261011T145239Z_trajectory_handoff_20261011T145805Z.zip` is preserved locally with SHA-256 `7cd75452ce7d64f3e3359ee66edadb5f08c9cf2d7a57c074b0c12bbf1bed488e`. CPU recovery validated all six completed seed0 arms and all 140 locked examples per arm; no inference was repeated. The earlier smoke crash occurred during post-run JSON reporting, after inference had completed.

[Exploratory/Inference] In the 140-example observe subsets, Gaussian terminal/minimum-entropy/equal-probability-mean accuracy was76.43/79.29/79.29%; Background was57.14/54.29/55.00%. Corresponding Legacy-off terminal controls were75.00% Gaussian and57.86% Background. Directions differ by corruption; this small smoke does not establish all15 benefit, threshold, or speedup. Observe differed from audit predictions on10/140 Gaussian and22/140 Background examples.

[Run/Inference/Open] Exact same-RNG VAE latent replay differed in all four audit batches despite RNG restoration and unchanged buffers. This replay variability is unresolved and does not establish that probes caused cross-process differences. Keep it as a limitation when interpreting the pilot.

[Code/control] Pilot persistence was inspected: each completed batch writes prediction NPZs and updates result rows; a caught exception records failed status and the runner's `finally` block seals the available run directory as a ZIP. The launcher copies that ZIP to Drive in its own `finally`, including when the inference subprocess exits unsuccessfully. CPU analysis starts only after the inference ZIP has been backed up, so a late analysis/report error does not require rerunning inference. The prepared packaging cell includes partial run directories, logs and status.

[Open/risk] A forced Colab runtime termination or kernel reset can bypass Python `finally`; the pilot is one all15 subprocess and does not continuously mirror partial batches to Drive or automatically resume. In that exceptional case, local partial files may be lost with the runtime. Before launch, confirm Drive is mounted and enough time/runtime is available; after any ordinary error, preserve the session and run the packaging/download cell before retrying. Do not rerun an existing output path.

[Open/next] After first reviewing terminal, minimum-entropy and equal probability mean on the same pilot traces, consider an entropy-weighted probability mean as a separate label-free policy. A candidate family is per-example `weight_k = exp(-tau * H_k) / sum_j exp(-tau * H_j)` using normalized entropy, followed by `sum_k weight_k * probability_k`; specify/freeze tau before confirmatory use or calibrate on separate data. Compare corruption-wise accuracy, temporal disagreements, confidence/correctness calibration and compute cost. This is a hypothesis only; no weight formula, tau, gain or safety guarantee is selected here.


## 2026-10-11 - Seed0 all15 pilot completed and independently reanalyzed

[Run/CPU] User handoff `pilot_recovered_20261011T152044Z_handoff_20261011T153106Z.zip` is preserved at `result/modelnet40_c/legacy_trajectory/` (SHA-256 `da8092f933f3272416b166996b383f497d547e766b6a859bbd1528b37962cebb`; outer ZIP CRC passed, 392 members). Its nested `pilot_seed0.zip` SHA-256 is `145a0932e3924cee8c6f921673fbcf368fadef7cdc2f1d0e6acece52eb7d5a2e`. Local CPU analysis was rerun from that nested archive and validated full trace coverage. Recomputed `per_corruption_seed.csv` is byte-identical to the report in the handoff. The execution status is complete with 4,200/4,200 selected examples, all15 corruption types and 280 shared IDs per type. The run's `status: partial` denotes the deliberate 280-example development subset, not an interrupted run or full 2,468-example benchmark. The run resolves to current branch HEAD `a351e66ec58f08f3a2104e46cab2cb8b20cdf961`; model state/buffers were unchanged. Environment: NVIDIA A100-SXM4-80GB, PyTorch2.1.2+cu121. No baseline archive was changed.

[Run/Inference] On these same seed0 trajectories, macro accuracy over all15 sampled corruptions was:

| Policy | Accuracy | Difference vs terminal |
| --- | ---: | ---: |
| Original terminal | 63.2143% | reference |
| Minimum entropy | 63.7381% | +0.5238 percentage points |
| Equal class-probability mean | 63.9762% | +0.7619 percentage points |
| Fixed probe1 | 62.5000% | -0.7143 percentage points |
| Fixed probe2 | 62.3095% | -0.9048 percentage points |
| Fixed probe3 | 63.5238% | +0.3095 percentage points |

Against the terminal on the same 4,200 examples, minimum entropy corrected133 and broke111 predictions (net+22); probability mean corrected135 and broke103 (net+32). The 1,000-replicate stratified original-index bootstrap interval for the macro paired difference was[-0.215,+1.214]pp for minimum entropy and[+0.071,+1.500]pp for probability mean. These intervals are conditional on the selected indices and this one seed/trajectory; they do not show seed stability or independent generalization. Do not compare the 63.2143% subset terminal directly with the accepted full-data Legacy B70/M800 result64.570502%: the evaluation populations differ.

[Run/Inference] Corruption-specific paired deltas (percentage points vs terminal; minimum entropy / probability mean):

| Corruption | Min entropy | Probability mean |
| --- | ---: | ---: |
| uniform | +2.14 | +2.50 |
| gaussian | +2.50 | +3.93 |
| background | -3.57 | -3.57 |
| impulse | +1.79 | +0.71 |
| upsampling | +3.21 | +2.86 |
| distortion_rbf | +0.36 | +1.07 |
| distortion_rbf_inv | -0.71 | +0.36 |
| density | +0.71 | -0.71 |
| density_inc | +3.57 | +2.50 |
| shear | -1.43 | -0.71 |
| rotation | 0.00 | +0.36 |
| cutout | +1.79 | +1.43 |
| distortion | -2.50 | -1.79 |
| occlusion | -0.36 | +1.07 |
| lidar | +0.36 | +1.43 |

[Inference/Open] Excluding Background, terminal/minimum-entropy/probability-mean accuracy was63.5969/64.4133/64.6684%. Background fell from57.8571% terminal to54.2857% under both minimum entropy and probability mean. The macro gains therefore coexist with a clear Background regression and mixed per-corruption changes. Across examples, candidate argmax predictions disagreed at least once for25.81%;805 examples were stably wrong across every candidate. Minimum-entropy selection still made1,523 errors. Entropy is not a correctness or temporal-stability guarantee.

[Run/Cost] The Colab run completed in about10m17s by session/run timestamps. Recorded trajectory-runtime sum is544.4s over4,200 examples (~7.71 examples/s); this is observer-instrumented B70 throughput, not a matched baseline speedup. Peak allocated memory was about66.7GB (reserved about71.8GB) on the reported80GB A100. Four sparse probes were evaluated in each of60 batches. No early stop occurred and these full-trace policies do not save diffusion work.

[Open/next] The result supports further seed0 policy analysis, not a promotion claim: evaluate entropy-weighted probability averaging offline from the saved logits, document the weight formula and sensitivity to tau, and retain the Background/per-corruption loss review. Treat any tau sweep on these same280 IDs as exploratory; before claiming general benefit, freeze a policy and check seeds1/2 and an independent evaluation. Do not change the Legacy baseline or10/30 horizons based on this single pilot. Conditional B1 early-stop latency and GSD/SCD ablations remain later work.


## 2026-10-11 - Full-data seed0 all15 launch prepared

[Code/Run/Open] After reviewing the completed 4,200-example seed0 pilot, the user approved a 37,020-example all15 screen on the same source commit `a351e66ec58f08f3a2104e46cab2cb8b20cdf961` and unchanged Legacy B70/M800 settings. The Colab notebook now appends a preflight, inference, packaging/analysis/Drive-backup, and download workflow; the prior14 cells are preserved. Preflight verifies branch/commit, baseline assets, all15 data files and labels, mounted Drive, and a GPU with at least75GiB total/70GiB free. The run uses all original IDs0..2467, seed0, severity5, Legacy isolate policy, normal/background horizons10/30, checkpoints3/5/8 and8/15/23, and observe mode. It does not early-stop or alter the accepted baseline.

The pilot took about10m17 for4,200 examples; the full run is expected to take about80-100 minutes, with at least2 hours reserved. The observed pilot peak was about66.7GB allocated/71.8GB reserved on A100 80GB. There is no periodic Drive checkpoint: after an ordinary error while the runtime remains alive, run the packaging cell to save available partial outputs. A hard runtime reset can lose local partial files. Packaging analyzes only a complete 37,020-example immutable-state trace, then creates a handoff ZIP, records its path/SHA, verifies the Drive copy, and prepares browser download; incomplete runs are still packaged and labeled, not analyzed as full results. No inference was launched locally. This is a one-seed development screen; seed replication and any policy promotion remain conditional on review of paired/per-corruption results, especially Background.


## 2026-10-11 - Colab preflight rerun behavior and current handoff

[User report/Open] In the full-data all15 section of `scripts/gsd_guidance_ablations_colab.ipynb`, the first execution of its preflight code cell reportedly completed without error; rerunning that same cell produced a notebook `CalledProcessError`. The shell deliberately refuses a second initialization when `/content/legacy_trajectory_full_session.txt` already exists, to protect the session and avoid overwriting outputs. This is the expected behavior if the initial pass created the session pointer. The reported traceback showed only the IPython wrapper, not the underlying Bash stderr, so the specific guard is a strong inference rather than directly confirmed from the printed error line. If the first pass printed `Preflight passed`, keep the pointer/session and proceed to the next cell, ?Start the full all-15 run?; do not rerun preflight or delete its pointer. The inference cell separately refuses any existing run output path.

[User report/Run/Open] As of this handoff, the user has not reported starting the inference cell or receiving inference output. Treat the full 37,020-example seed0 run as pending. Current inspected local branch is `gsd-smooth-spectrum`, HEAD `a351e66ec58f08f3a2104e46cab2cb8b20cdf961`. The workspace has many pre-existing tracked and untracked user/research changes, including the Colab notebooks, PDFs, results and knowledge records. Preserve them; no reset, clean, deletion, stage, commit or push.

## 2026-10-11 - Full-data seed0 all15 screen ingested and independently analyzed

[Run/CPU] User handoff `full_all15_seed0_20261011T154536Z_handoff_20261011T171147Z.zip`
was received from Downloads, safely inspected, and preserved unchanged at
`result/modelnet40_c/legacy_trajectory/full_all15_seed0_20261011T154536Z_handoff_20261011T171147Z.zip`.
Outer SHA-256 is `44f5366448263dc02d7932111685948ed0a5ead39d4a8f53c3bda4ec2969d07a`
(size33,579,609 bytes). Outer ZIP has7 safe members and passes CRC; nested
`full_all15_seed0.zip` has3,249 members and passes CRC. A separate extraction
is under the sibling `_ingested/` directory. Current CPU analyzer wrote an
independent report to `_ingested/independent_analysis/`; raw ZIP bytes were
not rewritten.

[Run/Code] Run plan and config agree: branch `gsd-smooth-spectrum`, commit
`a351e66ec58f08f3a2104e46cab2cb8b20cdf961`, seed0, all15 severity5,
2,468 original IDs per corruption,37,020/37,020 classifications, B70/M800,
beta2, spectral1.17, Chamfer1, gamma/eta.01, lambda.95, Legacy isolate,
normal/Background horizons10/30, clean checkpoints3/5/8 and8/15/23. Execution
and coverage statuses are complete; model_state_unchanged is true and the
before/after state hashes match. All79 source manifest entries match the
recorded Git tree (and local HEAD). The archived 226 native inventory, model
modes, scheduler, assets and full environment are retained. Raw baseline ZIP
SHA remains `534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968`.

[Run/Inference] On the same full-data traces, macro15 results are terminal
64.592112% (23,912/37,020), minimum entropy65.051324% (24,082), and equal
class-probability mean65.156672% (24,121). Relative to that run's terminal,
minimum entropy is+0.459211pp (1,073 corrected,903 broken); probability mean
is+0.564560pp (953 corrected,744 broken). A 1,000-replicate original-index
cluster bootstrap conditional on this one seed/run gives95% intervals
[+0.2269,+0.6942]pp and[+0.3431,+0.7915]pp, respectively. These do not
include seed or trajectory variability. Relative to the accepted separate
Legacy64.570502% run, probability mean is descriptively+0.586170pp; this is
not a paired comparison and does not establish a1pp gain. Macro14 excluding
Background is terminal64.9601%, minimum entropy65.7010%, probability mean
65.7560%.

Background is the key damage: terminal59.4408%, minimum entropy55.9562%
(-3.4846pp;74 corrected/160 broken), equal mean56.7666% (-2.6742pp;77/143).
Probability mean improves10/15 corruption rows and regresses5/15. The full
per-corruption transitions are in `independent_analysis/per_corruption_seed.csv`;
summary, diagnostics and intervals are in `independent_analysis/analysis.json`.
Fixed probe3 alone is64.7299%; its paired bootstrap interval versus terminal
includes zero. Same-cost repeated-terminal-vote control has not been run.

[Run/Cost] Colab run timestamps span about86.1 minutes; summed per-corruption
instrumented trajectory runtime is4,825.6s (80.4min). The prior accepted
Legacy run records68.98min, so this suggests a moderate observer cost (about
17% by instrumented sums), with a non-identical, non-concurrent run comparison;
observer timing includes hashing/artifact I/O and extra decode/FPS/classifier
probes. Do not call this a synchronized matched speed benchmark. Peak memory
was62.16GiB allocated and69.05GiB reserved. No early stop occurred and these
policies save no diffusion steps.

[Inference/Open] Full-data evidence strengthens the trajectory-policy signal
beyond the280-example pilot, but this remains one seed on ModelNet40-C,
which has informed development. It reaches about+0.56pp over the same-trace
terminal, not the1–2pp research target; the Background decline prevents an
unqualified promotion. Do not tune tau on this evaluation. Next: inspect the
per-corruption failure pattern, then (only as explicitly exploratory offline
analysis) assess the predeclared entropy-weighted probability candidate on
saved logits without selecting a confirmatory tau. If a policy is frozen,
seeds1/2 assess repeatability but are not independent-dataset validation.
Causal stopping, online B1 latency, speedup and guidance ablations remain
untested.

## 2026-10-11 - Macro14 entropy-weighted offline analysis prepared and run

[Code/CPU] Added `scripts/analyze_legacy_trajectory_macro14.py` and focused tests in `tests/test_analyze_legacy_trajectory_macro14.py`. The analyzer accepts only a complete, frozen-state seed0 all15 bundle with the approved Legacy B70/M800, 10/30 horizons and 3/5/8 normal checkpoints; it filters Background for the primary macro14 summary and applies shared policies without labels or corruption-name routing. It computes parameter-free normalized-entropy weights `w ∝ 1-H`, with equal weights only if total weight is zero; no tau is fit. It retains terminal, minimum-entropy, equal-mean and fixed-probe references; latest-clean plus terminal is only an extra generic control. CSV outputs include paired per-example correctness and transitions versus both terminal and equal probability mean. Output directories are exclusive-create to protect prior data.

[Run/Inference] CPU analysis of the immutable nested archive `full_all15_seed0.zip` (SHA-256 `1711d875d190e79e207f13960e97c131d12410541a1b6e8bbb1bbe176cf31ec7`) wrote `..._ingested/macro14_policy_analysis_v2/`. Provenance gates verified the full 37,020-example all15 trace and frozen model state before selecting the 34,552 non-Background examples. Macro14 terminal is64.9601%; equal probability mean65.7560% (+0.7959pp,876 corrected/601 broken); entropy-weighted mean65.7270% (+0.7670pp,875/610 vs terminal). Against the equal mean, entropy weighting corrects26 examples and breaks36, a net10-example decrease (-0.0289pp). It is near the equal mean and does not improve it on this seed. Per-corruption output shows mixed effects: the weighted mean is positive versus terminal in10/14 rows and negative in4/14, including Occlusion (-0.365pp), Lidar (-0.324pp), Cutout (-0.284pp) and Density (-0.203pp). The latest-clean/terminal control scores65.4752% (+0.5152pp).

[Limitations] One seed and previously reviewed ModelNet40-C only; bootstrap intervals resample original IDs, not seeds or trajectory variability. These full-trace policies do not save denoising steps or measure runtime. Background remains preserved in the original all15 report but is excluded from this development macro. Keep Legacy as reference; do not promote entropy weighting over the equal mean from these data. Conditional seed1/2 repeatability and an independent evaluation remain later decisions. Validation: focused3 tests and45 existing GSD/trajectory regression tests passed; raw ZIP unchanged.

## 2026-10-11 - Sampling-consistency development implemented; GPU pending

[User decision/Code/CPU] User authorized implementation after recording the plan/test and reference paper names. Canonical plan: [sampling consistency](gsd_sampling_consistency_20261011.md). It explicitly cites Bahri et al., **Test-Time Adaptation in Point Clouds: Leveraging Sampling Variation with Weight Averaging** (WACV2025), Zhang/Levine/Finn, **MEMO: Test Time Robustness via Adaptation and Augmentation** (NeurIPS2022), and Hendrycks et al., **AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty** (ICLR2020). Sampling diversity, confidence/consistency and JS precedent are attributed separately. The frozen-classifier H+JS trajectory selector remains an unvalidated synthesis with no novelty claim.

[Code/CPU] New opt-in external FPS views, six offline policies, strict artifact/source/index/model-state checks, paired transitions versus terminal A and four-A mean, source snapshot and Colab launch/recovery are ready. The approved host is Legacy B70/M800 seed0 common10-step macro14, probes3/5/8; original A logits and main RNG are preserved in CPU tests. The acquisition includes terminal8 voting and therefore14 forwards/example, while the proposed method itself uses8. Defaults and original guidance/model/BN/token algorithm are preserved. 66 targeted tests passed; source snapshot CRC/318-file SHA manifest and extracted CPU preview passed; changed runtime modules/new cells parse as Python3.8.

[Artifact] New source ZIP: `result/preparation/trajectory_sampling_source_20261011_v1.zip`,640,698 bytes, SHA-256 `6e42659c5cb3684a03bd4b741e098385877dadaaa1c45d971b8a44a10b005686`. Base is `a351e66ec58f08f3a2104e46cab2cb8b20cdf961` plus explicitly hashed working-tree overrides. This is not a clean-HEAD-only delivery. No stage/commit/push. Notebook section **Sampling consistency: macro14 seed-0 smoke and pilot** appends9 cells after the existing21; all original notebook bytes/cells were retained. Preservation receipt: `result/preparation/trajectory_sampling_notebook_receipt_20261011.json`.

[Open/Next] User-operated Colab preflight -> Gaussian140 smoke -> download full handoff ZIP and review -> separate macro14x280 (3920-example) pilot. Do not Run all, rerun prior experiments, or rerun preflight on an initialized session. A normal failed run is packaged/backed up; recovery is available while runtime remains alive. CUDA kernels, accuracy, memory and actual runtime remain untested. No new inference, accuracy gain, seed stability or speedup claim.


## 2026-10-12 - Sampling source delivery v2 (current)

[Code/CPU/Open] Base remains `gsd-smooth-spectrum@a351e66ec58f08f3a2104e46cab2cb8b20cdf961` plus the declared hashed working-tree overrides. Use `result/preparation/trajectory_sampling_source_20261011_v2.zip` (642,170 bytes; 318 source files), SHA-256 `d0c48e7f5a90f8fe19bfc615961e0ed1db3451b19cf5c98494cb613181b57295`. This supersedes v1 for new launches; v1 is preserved.

The standalone CPU analyzer now independently compares recorded checkpoint/config/label/selected-data SHA-256 and sizes with the accepted baseline specification and verifies that specification against source delivery. Changed or absent identities are rejected before scoring. The launcher still hashes actual assets before inference.67 targeted CPU tests pass; extracted v2 CRC, all source SHA checks, CPU macro14 preview (34,552 full-scope examples; the staged runs remain140/3920), and Python3.8 syntax checks pass. Tests use synthetic CPU fixtures, not native CUDA.

The notebook now requests v2 and its exact SHA. Only the filename/hash in the newly added section were revised; original21 cells are unchanged. Amendment receipt: `result/preparation/trajectory_sampling_notebook_receipt_20261012_v2.json`. The plan and its explicit paper attribution remain in `knowledge/gsd_sampling_consistency_20261011.md` (sampling variation/weight averaging paper, MEMO, AugMix; our H+JS selector is an unvalidated synthesis).

Next: user-operated Colab preparation, new sampling preflight once, Gaussian140 seed0 smoke, then download/provide its complete handoff ZIP for review. Pilot macro14x280 remains separate and conditional on smoke review. No Run all, old-run repetition, local GPU, stage/commit/push, or raw baseline changes. Native CUDA behavior, accuracy, cost and repeatability remain open; failed invariants or no paired gain over the controls would reject promotion. Acquisition14 forwards is not deployment latency for the8-forward method.
