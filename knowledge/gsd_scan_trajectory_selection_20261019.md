# ScanObjectNN-C GSD-inspired trajectory selection — 2026-10-19

Status: implementation/preparation; no new trajectory GPU result accepted yet.

The user selected **only the GSD-inspired guidance pipeline** for this study.
Evaluate CurveNet, PointNeXt, DGCNN and Point-MAE on severity5, seed0,
all15 corruptions of the 581-example ScanObjectNN-C `main_split`.
The primary comparison is each head's same-run terminal prediction.
Published alternative backbone checkpoints remain distinct from exact GSDTTA
paper reproduction. DGCNN's weak source checkpoint remains exploratory.

Fixed candidates: predicted-clean outputs at reverse loop counts3/5/8 of10,
or8/15/23 of30 for Background, plus the ordinary terminal output.
Decode each selected latent once, then apply each frozen head's established
preprocessing. Scan outputs must use the terminal coordinate transform:
rotate back, divide3.3885, inverse the input bounding box; no output normalization.
Policies: terminal, minimum normalized entropy (ties favor terminal then the
earliest checkpoint), equal class-probability mean at temperature1.
No labels, threshold fitting, early stopping, or extra diffusion per policy.
Existing Point-MAE ModelNet trajectory results do not establish Scan performance.

## Batch 1 — observer and isolated runner integration

- Goal: collect four-head, four-candidate logits without changing GSD updates.
- Scope: opt-in Scan trajectory mode, shared decoder observation, trusted exact
  Point-MAE checkpoint loading, indexed batch trace persistence, pure analysis.
- Touched areas: `scripts/eval_multibackbone_guidance.py`, a small shared observer,
  focused tests and a Scan trajectory analyzer. Default old modes stay unchanged.
- Stack context: local CPU checks; actual runtime Python3.8/PyTorch2.1.2/cu121.
- Dependencies: existing `tta_gsd_v2` hooks, `isolated_probe`, source preprocessing,
  immutable bundle and Drive shard helpers, class-count-independent `policies`.
- Implementation notes for development-agent: preserve classifier/LION state and
  RNG; preserve the live autograd graph; decode once per checkpoint; require all
  finite B×4×15 logits; preserve original IDs including last21 examples. Do not
  invent a full Point-MAE matched source row; its primary comparator is terminal.
- Verification: targeted CPU tests for purity, geometry transforms, default-off
  behavior, policy ties and complete indexed coverage; then genuine GPU smoke.
- Knowledge artifact to update: this note (agent implementation section).
- User review gate: selected scope is already authorized; automatically verify
  acceptance before the GPU phase, with explicit approval only if its complete
  expected duration exceeds60minutes.
- Out of scope: original3DD trajectory selection, hyperparameter tuning, completing
  the separate pending full DGCNN ModelNet/original3DD phase.

## Batch 2 — sealed source and GPU pilot

- Goal: validate actual four-head observations and estimate the whole GPU phase.
- Scope: fresh immutable source/asset manifest; existing r9 environment, no build;
  Gaussian and Background one-batch pilot, per-batch Drive receipts and SHA checks.
- Touched areas: preparation artifacts and this note; never mutate prepared_v7.
- Stack context: existing r9 A10080GB, same four pinned native binaries and LION.
- Dependencies: accepted Batch1 CPU checks and source provenance.
- Implementation notes: record the executed snapshot/hash; validate states/RNG,
  candidate counts and same terminal transforms; account for pilot plus fullrun.
- Verification: actual GPU shapes, purity, peak memory, finite losses/logits,
  independent Drive/local hash round trip and policy analyzer.
- Knowledge artifact to update: this note, findings_log, open_questions.
- User review gate: no extra approval under60minutes; if estimate exceeds60,
  request one comprehensive approval before the dependent full GPU phase.
- Out of scope: runtime replacement/reinstallation or other Colab sessions.

## Batch 3 — all15 evaluation and acceptance

- Goal: obtain12 accuracy rows (four heads × three selection policies).
- Scope: one shared full Scan diffusion trajectory, immutable candidate shards
  copied to Drive after every batch, per-head/corruption and final bundles.
- Touched areas: results, independent analysis, research records and commits.
- Stack context: fixed B70, normal10/Background30 reverse updates, seed0.
- Dependencies: accepted pilot and whole-phase duration/approval gate.
- Implementation notes: analyze all8715 original-ID examples per head, report
  all15 macro and paired terminal changes; save prefix on interruption.
- Verification: independent indexed predictions and SHA-256 checks, all135
  shared batches, four classifier and LION state guards, policy recomputation.
- Knowledge artifact to update: this note, findings_log, open_questions, README.
- User review gate: report concrete evidence and limitations; push accepted batches.
- Out of scope: single-seed generalization or a claim of full GSDTTA reproduction.

## Inventory and runtime baseline

CurveNet and PointNeXt source/original3DD/GSD comparisons are complete on both
datasets. DGCNN source-only is complete on both; its full guidance/original3DD
comparison remains incomplete. The four newly accepted DGCNN GPU smoke tests
are408 predictions and are not full accuracy rows. No validated four-head Scan
trajectory trace exists in the prior terminal-only archives.

The prior shared Scan CurveNet+PointNeXt GSD job took1004.737seconds (16.746min),
including924.644seconds shared reconstruction for135 batches. This is an
empirical baseline, not yet the new four-head observer duration estimate.
The live r9 session is reused. Remote checkout and sealed executed research
source have distinct provenance; see the DGCNN smoke note and runtime handoff.

## Shared observer and independent analyzer implementation — 2026-10-19

`shared_trajectory.SharedTrajectoryObserver` implements the existing GSD hooks
without editing GSD updates. Each selected pre-update clean latent is decoded
once under `isolated_probe`, then a callback performs the Scan terminal coordinate
transform and classifies that shared geometry with all selected frozen heads.
The callback does not receive labels. Three clean candidates and the already
computed terminal logits form finite B×4×15 arrays, with explicit candidate names
and loop counts. Missing/reordered checkpoints, mismatched heads, changed RNG,
buffers or live latent/style are rejected. Diagnostics preserve initial/final
hashes and each probe's RNG, buffer and decoded-geometry identities. Component
times are unsynchronized dispatch wall times; the runner owns GPU wall timing.

`scripts/analyze_scan_trajectory.py` independently verifies every sealed bundle
member's SHA-256/size and accepts the existing `run_id/` ZIP prefix. It requires
the declared GSD-inspired, severity5/seed0, 581-example `main_split`, 15-class,
B70 protocol and unchanged classifier/LION state guards. Full acceptance requires
all15 corruptions, all original IDs0–580 exactly once in order, and nine batches
per corruption including the final21 examples. Candidate logits must contain
the exact stored terminal logits. Policies reuse the existing class-agnostic
temperature1 implementation and report per-corruption/macro accuracy, changes
against same-run terminal, corrected and harmed predictions. `--allow-partial`
permits a clearly labelled complete-batch prefix for the pilot; it cannot create
a full accuracy row from an incomplete trace. No Point-MAE source-only result is
invented by this analysis.

Local verification: `uv run --offline --no-project --with torch --with numpy
python -m unittest tests.test_shared_trajectory
tests.test_legacy_trajectory_observer -v` passed all15 tests. The new tests cover
stochastic single-decoder observation, RNG/buffer restoration with a live BN
autograd graph, missing checkpoints, four-head/15-class policy ties, all135
synthetic batch records including remainder21, duplicate IDs, partial scope,
prefixed sealed ZIPs and member tampering. These are synthetic CPU checks;
actual r9 CUDA memory, model/checkpoint compatibility and full trajectory
results remain pending the real GPU pilot and evaluation.


## Sealed source and actual r9 CPU proof — 15:37UTC

[Code/Run] Implementation commit `746bc3024f26de67a3cab410c1e15cfbdfa2eb2a`
is pushed. A fresh `prepared_scan_trajectory_v8` preserves v7 and declares
manifest `7c3b55fe7cf179a63ee47c4c62a1a02cacd68372a645b534dd31f4d8e616806c`,
sourceZIP `f30c224d99e8d62263b55c47b0e6bfb6ac8e5575efcdae875a51aedce058948c`.
Existing diffusion/guidance/native consumer source bytes are unchanged.
Scan Point-MAE checkpoint351525407bytes SHA53f6c2e…b9535dac is now declared
with the three existing frozen Scan checkpoints. The actual r9 strict CPU
load passes all4 models, with no forward or CUDA initialization; the pinned
four-backend no-build bootstrap also passes. CPUproof wall14.703seconds.

Immutable CPUevidence51514bytes SHA
`c15bbe0ccd25663d760ae314b646b8efcfa15be29fffbae68262ece075506c9a`
and delta5487366bytes SHA
`a050f0dcf0cc360354d6f0d1b2504abde7499d6d0efecfdc138a21e16005e008`
pass actual Drive/local roundtrip. Local artifacts:
`result/preparation/scan_trajectory_20261019/`; actual Drive prefix:
`MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261019T153709Z_scan_trajectory_v8/`.

[Run/Open] Pilot started15:37:49UTC in r9, shared Gaussian+Background first70
IDs, four heads and four candidates. Phase
`/content/gsd_multibackbone_study_20261018/gpu_phases/20261019T153749Z_scan_trajectory_pilot`.
Controller SHA27b7e74a…2d699e7; actual GPU result acceptance remains pending.
Planned whole new GPU phase25–45min based on the16.75min prior two-head job;
pilot recalibrates before fullrun. One shared3600s job-wall cap includes pilot
and fullrun, never a per-call loophole. No long DGCNN completion launch.


## Actual four-head GPU pilot accepted — 15:41UTC

[Run] Pilot completed15:38:54UTC:65.652680296seconds, two shared70-example
batches,560 terminal predictions and2240 four-candidate predictions. Gaussian
shared reconstruction9.959955540s; Background16.086486018s. Peakallocated
36335.369MiB (35.484GiB) on A10080GB. Allfour strict classifier states and
LION states remain equal; each probe preserves RNG, modelbuffers and live
latent/style bytes. The four heads share identical indexed trajectory hashes.
Original main_split labels match the declared local dataset; all15-class
candidate shapes, terminal equality, policy recomputation and sealed manifests
pass. Prefix140examples/head is smoke evidence, not benchmark accuracy.

All4rawZIPs are under `result/scanobjectnn_c/gsd_inspired_latent_spectral_guidance/`:
`20261019-153750_curvenet-seed0.zip` SHA153e1b9b…9906150;
`20261019-153751_pointnext-seed0.zip` SHA10d4b0a0…c1e0c91;
`20261019-153752_dgcnn-seed0.zip` SHA967db8cb…730c1b9;
`20261019-153753_pointmae-seed0.zip` SHAbc268fa2…ed94f2.
Exact full identities are in `result/preparation/scan_trajectory_20261019/gpu_pilot_receipt.json`.
Allfour are downloaded from their actual Drive finalZIP and hash-verified.

Pilot phase evidence48595bytes SHA380a8046…1fcb036b verifies Drive/locally.
Accepted analysis/physical-receipt binder37464bytes SHA
`182a8f2415a3b4bbd3b5b04d07c70733c7dc37b2401d5ef2dceac47e02e9cdd9`
verifies actual Drive/local roundtrip and every sealed member, including
8physical batch receipts,8corruption receipts and4final receipts.
Local `pilot_accepted_evidence_drive_roundtrip.zip`; Drive
`MyDrive/thesis/result/preparation/gsd_multibackbone_study/scan_trajectory_pilot_accepted_20261019T154120Z/`.

[Inference/Run] Whole-phase estimate includes pilot,126normal+9Background
batches: central1795.385s (29.923min), conservative2550.279s (42.505min).
Upper adds40percent compute margin,3s/batch persistence and120s fixed overhead.
Thus expected duration stays under60min; no new long-run approval required.
The shared3600s job-wall cap still includes pilot and fullrun.

Full all15 started15:41:44UTC in the same r9 session, source/manifest unchanged.
Phase `/content/gsd_multibackbone_study_20261018/gpu_phases/20261019T154144Z_scan_trajectory_all15`;
controllerSHA94c512da…a80dba. Expected135shared batches,8715examples/head,
139440candidate predictions. Batch receipts copy allcandidates to Drive,
followed by per-corruption and per-head final immutable ZIPs.
No full accuracy claim until independent local acceptance.
