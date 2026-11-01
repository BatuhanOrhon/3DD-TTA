# LION Reencoding Implementation Plan

> **For agentic workers:** use superpowers:executing-plans inline for this authorized task.

**Goal:** Evaluate one N8/B35 → full z+h reencode → N3 local pass with paired evidence.
**Architecture:** Reuse original tta, depth preprocessing, SharedTrajectoryObserver and shared artifact worker; add an opt-in runner/observer and posterior bridge, leaving baseline defaults intact.
**Tech Stack:** Existing Python3.8/PyTorch/NumPy/Colab CLI.
**Spec:** knowledge/lion_reencode_pilot_20261101.md

## Global Constraints

No local model inference; frozen weights; no new dependency; no full benchmark/tuning; preserve canonical first pass and new output paths; no runtime teardown.

## Review Focus

Incorrect source/cache binding; accidental second-frame normalization; wrong noise depth; stale decoder style/anchor; misreported first/second/pooled readouts; hidden encoder RNG/model-buffer changes.

### Task1 — opt-in paired pilot

Files: create lion_reencode.py, scripts/eval_lion_reencode.py, tests/test_lion_reencode.py; reuse shape_latent_refresh.py persistence and original runner.
- [ ] Write failing tests for posterior coupling/conditional h, RNG/buffers, second own-style/anchor/depth, readout phase weights, and invalid input/config.
- [ ] Run targeted tests RED, implement minimum bridge/observer/runner, run GREEN and existing impacted tests.
- [ ] Seal source/assets/operator; strict CPU load proof; complete canonical setup and small CUDA smoke.
- [ ] Independently accept smoke, run64/all15 pilot under shared55min scientific cap, retrieve raw/source/proof archives and independently recount outputs.
- [ ] Update knowledge with negative/null/positive results and limits; review, verify and selectively integrate completed necessary changes.
