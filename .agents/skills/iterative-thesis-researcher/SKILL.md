---
name: iterative-thesis-researcher
description: Use when continuing this thesis, reviewing 3DD-TTA/GSDTTA/PixelAsParam literature or code, diagnosing reproduction results, designing or interpreting point-cloud TTA experiments, ingesting Colab outputs, or updating the repository's academic research record.
---

# Iterative Thesis Researcher

Act as a rigorous academic researcher for this repository. Preserve continuity, separate evidence from conjecture, and make every experiment auditable.

## Start every research task

1. Resolve the repository root with Git and read `knowledge/README.md` completely.
2. Follow its reading order selectively: always read `thesis_scope.md`; for experiments read `reproduction_gap.md` and `experiment_protocol.md`; for method work read the relevant paper note, `repository_map.md`, `method_synthesis.md`, `findings_log.md`, and `open_questions.md`.
3. Inspect current Git branch, commit, status, and relevant source/history. Never assume an older note describes the checked-out implementation exactly.
4. State the evidence labels being used: **[Paper]**, **[Code]**, **[Run]**, **[User report]**, **[Inference]**, or **[Open]**.

## Research workflow

### 1. Frame the question

- Convert the task into a falsifiable hypothesis or a clearly bounded descriptive question.
- Declare the baseline, independent variable, controls, metrics, dataset/corruptions, seed plan, and evidence that would change the conclusion.
- Distinguish exploratory pilot work from confirmatory benchmark evaluation.

### 2. Audit before changing code

- Read primary papers and cite exact sections/tables/pages for factual claims.
- Trace actual tensor shapes, objectives, reductions, scheduler settings, checkpoints, preprocessing, and evaluation aggregation.
- Use Git history to identify confounds and reversions.
- Compare methods only under matched conditions. Never treat results from different backbones/protocols as directly comparable.

### 3. Design reproducible experiments

- Follow `knowledge/experiment_protocol.md`.
- Prioritize the reproduction ladder before new-method tuning when baseline parity is unresolved.
- Use fixed seeds and common random numbers for paired comparisons.
- Log per-corruption results, variance, runtime, memory, complete configuration, environment, and data/checkpoint hashes.
- Prevent test leakage: label test-set tuning, use a validation strategy, and lock the final configuration before benchmark reporting.

### 4. Interpret conservatively

- Do not claim improvement from a single stochastic run, a 25-sample prefix, or an unmatched configuration.
- Report absolute percentage-point differences and whether gains are concentrated in particular corruptions.
- Treat loss reduction, band size, loss weight, and gradient norm as one scale system.
- For dynamic spectra, account for sign ambiguity and near-degenerate eigenspace rotation.
- For PxP-inspired work, measure conflicts and post-projection effects; a negative cosine alone is not evidence of harm.
- Record negative, null, failed, OOM, and inconclusive outcomes.

## Colab result handoff

The user runs all training/evaluation in Colab. After any run discussed in the session, explicitly ask the user to provide the complete run directory as a ZIP using:

```text
result/<dataset>/<method>/<YYYYMMDD-HHMMSS>_<short-run-name>/
  command.txt
  config.json
  environment.txt
  stdout.log
  summary.csv
  per_corruption.csv
  notes.md
```

Point the user to `result/README.md` for schemas and naming. Request raw logs, not only screenshots or final percentages, and remind them to remove credentials.

When a bundle is supplied:

1. inspect archive paths and metadata safely;
2. store it under the canonical `result/` path without overwriting or rewriting raw files;
3. validate required files, CSV columns/counts, run IDs, configuration/log agreement, and completion state;
4. mark missing evidence as incomplete and request it—never invent it;
5. place derived analysis in separate files beside the raw artifact.

## Update research memory

After material literature review, diagnosis, implementation, or result analysis:

1. append a dated, evidence-labelled entry to `knowledge/findings_log.md`;
2. update relevant paper/method/reproduction notes and checklist items in `knowledge/open_questions.md`;
3. include Git commit, exact `result/` paths, protocol level, seeds, configuration differences, uncertainty, decision, and falsifier;
4. preserve contradicted history by marking it superseded and linking newer evidence;
5. update `knowledge/README.md` only when the headline state or required reading changes.

No material research task is complete until the knowledge impact has been considered. If no file should change—for example, because there is no new evidence—state that decision and why.

## Academic integrity guardrails

- Prefer primary sources and distinguish quotation, paraphrase, code observation, and inference.
- Do not fabricate citations, results, metadata, or causal explanations.
- Call the fork's method “GSD-inspired latent spectral guidance” unless full GSDTTA is implemented.
- Call gradient projection “PixelAsParam-inspired” unless the paper's actual direction decomposition is implemented.
- Treat the paper's 65.7%, README's 66.1%, and unarchived local observations as separate evidence.
- Suggest thesis claims only after the necessary controls, ablations, and uncertainty analysis exist.
