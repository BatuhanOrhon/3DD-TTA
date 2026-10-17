# New agent handoff — thesis objective and current status

**Prepared:** 2026-10-17 (Europe/Istanbul)

**Branch:** `gsd-smooth-spectrum`

**HEAD at handoff:** `a351e66ec58f08f3a2104e46cab2cb8b20cdf961`

**Evidence labels:** Run, Code, Inference, User report/decision, Open.

## Why this handoff exists

Use this file to orient a fresh agent without treating old handoffs as current.
The repository contains dated records written before later runs. Preserve that
history; use the newest evidence and explicitly identify when it supersedes an
older statement. The worktree is intentionally dirty and contains user and
research changes. Do not reset, clean, delete, stage, commit, or push any of it.

## Project objective

The thesis studies training-free, input-level test-time adaptation for 3D point
cloud classifiers under corruption. 3DD-TTA uses a pretrained LION diffusion
prior to repair corrupted point clouds before a frozen Point-MAE classifier.
The `gsd-smooth-spectrum` branch investigates Graph-Spectral Driven Point
Shift (GSD) guidance and its interaction with the existing self-consistency
denoising (SCD) guidance. The current practical research priority is to find a
reproducible all15 accuracy improvement of roughly 1–2 percentage points while
keeping the source classifier frozen. Treat that as a target, not a result or
promotion threshold. Do not conflate accuracy development with causal proof
about GSD components or generalization to an independent dataset.

## Current evidence and its limits

- **[Run] Operational Legacy reference:** all15, severity5, 2,468 examples per
  corruption, seed0, 37,020 classifications: **64.570502%** (23,904 correct),
  B70/M800, spectral weight1.17, Chamfer1.0, beta2, gamma/eta0.01,
  lambda0.95, Legacy isolate policy, Normal/Background denoising horizons
  10/30. The raw handoff is
  `result/modelnet40_c/legacy_isolate_control/attempt-0002_all15_seed0/20261010T114156946419Z_handoff_20261010T141536060023Z.zip`, SHA-256
  `534c8634ed16aa0a0787e3be2f92294268453029a0d0a175ebcd7b6618726968`.
  This is accepted as a practical reproduction of the historical
  user-reported64.5327%, not a bit-identical or provenance-complete rerun of
  that old experiment.
- **[Run/Inference] Phase6 all15 composition comparison:** P_PC versus its
  paired C_SCD comparison produced only **+0.036917pp** mean difference. It
  does not support a large all15 gain. Phase5 P_PC65.9505% was over four
  development corruptions; never label it an all15 score.
- **[Run/Inference] Highest full all15 trajectory-policy score recorded so
  far:** on one seed0 full Legacy trace, the original terminal is64.592112%,
  minimum-entropy selection65.051324%, and equal class-probability averaging
  across three clean probes plus terminal is **65.156672%** (+0.564560pp paired
  with that trace's terminal). A 1,000-replicate original-index cluster
  bootstrap conditional on this one seed/run gave95% CI[+0.3431,+0.7915]pp.
  This is a saved-trace policy result, not a separate full rerun or a matched
  paired comparison with the64.570502% Legacy archive. It is not seed-stable,
  independent-dataset, or causal evidence. Background decreases2.6742pp and
  five of15 corruption rows regress.
- **[Run/Inference] Macro14 follow-up:** excluding Background, the same trace
  scores64.9601% terminal and65.7560% equal probability mean (+0.7959pp).
  The fixed entropy-weighted mean reaches65.7270%, slightly below equal
  averaging; it should not replace the equal mean on current evidence.
- **[Code/CPU] Sampling consistency:** a new frozen-classifier, paired-FPS-view
  experiment is implemented and has a pinned source delivery, but its native
  CUDA smoke has not been reviewed and it has no accuracy result. The next
  gate is a user-run Gaussian140 seed0 smoke, followed by ZIP review; only if
  that passes should the separate macro14×280 pilot be considered.
- **[Open]** No causal online early-stop, B1 latency/speedup, same-cost
  terminal-vote result, seed1/2 trajectory replication, or independent-dataset
  validation is established. Do not promise +1–2pp or a speedup.

The terminal/selection values above come from one run with correlated
trajectory candidates. Its bootstrap resamples original example IDs within
corruption and keeps seed/trajectory fixed; it does not estimate seed or
trajectory variability. ModelNet40-C has informed method development, so more
seeds there provide repeatability, not independent external validation.

## Source and worktree provenance

The current Git HEAD is the base of the trajectory/sampling development, but
it does **not** by itself represent every delivered source file. The working
tree has user-owned modifications and untracked implementation files. Check
`git status --short` before inspecting and leave it intact. The sampling
Colab delivery is a pinned source snapshot plus declared working-tree
overrides:

- `result/preparation/trajectory_sampling_source_20261011_v2.zip`
- Size642,170 bytes; SHA-256
  `d0c48e7f5a90f8fe19bfc615961e0ed1db3451b19cf5c98494cb613181b57295`
- Specification and staged gates: `knowledge/gsd_sampling_consistency_20261011.md`
- Colab cells appended to `scripts/gsd_guidance_ablations_colab.ipynb` under
  **Sampling consistency: macro14 seed-0 smoke and pilot**.

Do not launch GPU inference from the local workstation. Do not use `Run all`,
repeat the earlier trajectory run, or advance to the macro14×280 pilot before
the new Gaussian140 smoke handoff is reviewed. Do not change or overwrite raw
result ZIPs. Keep the separate M_max=1800 line and recorded Background
non-finite failures out of this experiment.

## Reading order for the fresh agent

1. `knowledge/README.md` — read the 2026-10-17 status block first, then use
   dated entries as history.
2. `knowledge/thesis_scope.md` and `knowledge/experiment_protocol.md` — thesis
   purpose, comparison limits and evidence rules.
3. `knowledge/findings_log.md` and `knowledge/open_questions.md` — latest
   dated evidence and outstanding gates; do not infer that a checkbox or old
   note is current without checking its date.
4. `knowledge/gsd_trajectory_implementation_20261011.md` — validated full
   trajectory artifact, independent analysis, later macro14 offline analysis,
   and the sampling source v2 amendment.
5. `knowledge/gsd_trajectory_selection_20261011.md`,
   `knowledge/gsd_sampling_consistency_20261011.md`, and
   `knowledge/gsd_step_count_evidence_20261011.md` — hypothesis, locked
   experiment, literature attribution, controls and compute limits.
6. `knowledge/gsd_accuracy_research_plan_20261011.md`,
   `knowledge/gsd_spectral_math_audit_20261008.md`,
   `knowledge/method_synthesis.md`, and `knowledge/advisor_meeting_log.md` —
   previous accuracy direction, mathematical review, candidate synthesis and
   advisor discussion agenda (agenda items are not advisor decisions).
7. Relevant code: `eval_gsd_tta_v2.py`, `legacy_trajectory.py`,
   `legacy_gsd_artifacts.py`, `trajectory_sampling.py`,
   `scripts/analyze_legacy_trajectory_macro14.py`,
   `scripts/analyze_trajectory_sampling.py`, and
   `scripts/trajectory_sampling_colab.py`. Compare these with the pinned ZIP
   and recorded source manifests; do not silently attribute working-tree code
   to HEAD.
8. Inspect, without modifying, the accepted baseline ZIP and full trajectory
   handoff under `result/modelnet40_c/legacy_isolate_control/` and
   `result/modelnet40_c/legacy_trajectory/`, plus the source snapshot above.
   Check whether a later sampling smoke handoff has since arrived before
   recommending any launch.

## Copyable prompt for the next agent

```text
You are joining the 3DD-TTA thesis project. Begin by reading
knowledge/new_agent_handoff_20261017.md and follow its reading order. Then
inspect the current branch, HEAD, git status, latest dated knowledge entries,
and available result handoffs. Preserve every tracked/untracked user change;
do not reset, clean, delete, stage, commit, or push.

Explain the thesis objective and distinguish (a) the accepted Legacy all15
reference 64.570502%, (b) the Phase6 P_PC-vs-C_SCD all15 difference
+0.036917pp, (c) the Phase5 four-corruption P_PC result65.9505%, and (d) the
one-seed full all15 trajectory-policy mean65.156672% versus its own
same-trace terminal64.592112%. State the evidence limits, including the
Background loss, one-seed/development-data status, and the lack of matched
speed or causal early-stop evidence.

Determine whether a Gaussian140 seed0 sampling-consistency v2 smoke ZIP has
arrived since this handoff. If it has, review it against
knowledge/gsd_sampling_consistency_20261011.md, verify all recorded source,
assets, coverage, FPS-view, state/RNG and archive integrity requirements, and
report concrete discrepancies before recommending the conditional macro14x280
pilot. If it has not, report that the next gate is still the user-operated
Colab smoke and provide the correct notebook section/cell sequence. Do not
launch GPU work yourself. Do not claim accuracy or latency gains for sampling
consistency. Keep any proposed work aligned with the locked protocol and
explain what evidence would falsify the candidate.

Use the current worktree and source ZIP as distinct provenance identities;
never assume dirty source is present in HEAD. Treat historical knowledge notes
as dated snapshots, not current truth. Do not edit source or run a full test
suite unless asked; answer with linked evidence, current status, and the
smallest justified next action.
```
