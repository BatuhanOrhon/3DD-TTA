# GSD-inspired latent spectral guidance: multibackbone study

## Scope and gates

[User decision] 2026-10-18: create `gsd-multibackbone-study` from
`2684f11922d77696a26c2b954dcd0c43de5f5a6c`, preserve user edits, and study
DGCNN, CurveNet and PointNeXt on ModelNet40-C and ScanObjectNN-C. The latter
must use the 581-example `main_split`, never `main_split_nobg`. Later user
clarification authorizes searching for published dataset-trained alternatives
when exact GSDTTA checkpoint links cannot be found. Alternatives must remain
explicitly distinct from paper reproduction.

Scientific question: for each fixed published checkpoint, does the existing
GSD-inspired latent spectral guidance improve all15 macro accuracy against
its own matched source-only evaluation? Backbone, weights, classifier-facing
preprocessing, seed and input IDs are controls. No target-label tuning or new
GSDTTA model adaptation is included. Seed0 is the initial descriptive screen;
replication follows only after cost and source protocol review.

Gates: provenance -> Point-MAE parity -> per-backbone source-only smoke ->
full source-only all15 -> explain differences -> frozen-classifier GSD smoke
and matched all15. No GSD integration before the source-only gate. No numerical
paper-parity threshold is invented. A checkpoint substitution explains lack of
identity, but does not by itself establish the cause of an accuracy difference.

GPU approval is not yet requested or granted: total conditional GPU work is
expected to exceed60min once full GSD evaluation is included. Prepare concrete
commands, snapshots, dataset/checkpoint identities, destinations and cost
estimate before the single comprehensive approval. CPU inspection, downloads
and preparation are already authorized. Do not close r7 without the user's
explicit task-complete plus close authorization; leave other assignments alone.

## First audit

[Code] Local branch was created at the requested commit from a clean worktree.
During preparation an external edit appeared in
`.codex/skills/colab-research-operator/SKILL.md`; preserve it and exclude it from
this study's commits. `.codex/knowledge/INDEX.md` is absent; canonical research
memory is `knowledge/README.md`. The requested undated all15 matrix path is
absent; its actual file is `modelnet40_c_all15_results_matrix_20261001.md`.

[Run] Initial server inventory confirms r7 at
`gpu-a100-hm-kkb-euw4a2-1d8qxj97k4u94`, NVIDIA A100-SXM4-80GB/High-RAM,
6MiB GPU allocation. Inspection completed with exact marker
`MULTIBACKBONE_R7_INSPECTION_COMPLETE`. Actual remote source is
`dev@a458cd46bba1288a3f41302548bc777700cbc7d4`, with native build repairs,
modified `utilities_3dd_tta.py`, and untracked `run_baseline.py`,
`research_artifacts.py`, `gsd_protocol.py`. Do not equate it to the local branch.
Conda environment is `/usr/local/envs/3dd_tta_env`. Existing weights on Drive
and in the project are Point-MAE/LION; no requested backbone weights were found
under the inspected thesis weights directory. No GPU inference was launched.

[Paper] Official ICCV PDF, p5/Sec4.1–4.2, specifies 2,468 ModelNet40 test
objects and 581 ScanObjectNN test objects, three backbones and common pretrained
weights for compared methods. Tables1–2 (PDF pp6–7) report source-only targets:

| Dataset | DGCNN | CurveNet | PointNeXt |
|---|---:|---:|---:|
| ModelNet40-C |66.51|71.38|66.99|
| ScanObjectNN-C |55.72|53.06|50.66|

The main PDF and complete three-page supplement do not identify checkpoint
bytes, a GSDTTA code release, exact backbone configurations, classifier input
point counts, normalization or FPS policy. Supplement Sec2 links baseline
implementations, including CloudFixer, ModelNet40-C and BFTT3D; those links do
not prove checkpoint identity. Inspect the primary sources rather than treating
the arXiv v1 or a third-party model zoo as the final ICCV protocol.

Primary sources:
- [Main paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.pdf)
- [Supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wei_3D_Test-time_Adaptation_ICCV_2025_supplemental.pdf)

## Published candidates (not paper checkpoint identity)

[Code/Run] Download and strict-load checks completed below. Predeclare candidates
by published provenance; do not choose among weights by final corrupted accuracy.

| Pair | Published source | Identity / provenance caveat |
|---|---|---|
| DGCNN / ModelNet40 | SVWA | Drive `1RmuvcOR3nFE9TQIiS2DPwEk1PfNNCguA`; exact SVWA DGCNN_cls |
| DGCNN / ScanObjectNN | SVWA | Drive `1ZkXrXj_U5t1OGr-_J7sGFPcuNVrA5fh-`; source training variant must be audited |
| CurveNet / ModelNet40 | SVWA | Drive `1jlDHnLwQfDRZwGl1DdpuAcoUqcO-OSBm`; exact SVWA CurveNet |
| CurveNet / ScanObjectNN | 3DeformRS | `CurveNet/checkpoints/BetterCurveNetOnScanObjectnn/models/model.t7`; 15-class OBJ_ONLY alternative; earlier `curvenetScanobjectBaseline` candidate rejected (40-class head) |
| PointNeXt / ModelNet40 | official model zoo | Drive `1TUOR3j3xqOoLNzEneiXccR5eRmeZxKq9`; PointNeXt-S width64 |
| PointNeXt / ScanObjectNN | official model zoo | Drive `1tFmyb3rmKMVzjv1qzlPXPplFRQnDiulr`; PointNeXt-S width32, hardest PB_T50_RS training |

Pinned inspected source revisions: SVWA
`59935d68d0ef12811619ce4776e1dc4a711997d1`; 3DeformRS develop
`30a2385edbd6a0e7919caa01344cb47af81bb361`; CloudFixer
`f043e24414fcf4165fdc892abff21215c9989141`; ModelNet40-C
`fad786c430c007cfbb7b4fe90ed1c9873d960921`; BFTT3D
`cacd91b7b4afe6907e2534a7673a24b539f3ee9e`.

CloudFixer's public weights list has DGCNN ModelNet40/PointDA weights but no
CurveNet, PointNeXt or ScanObjectNN checkpoint in the inspected folder.
BFTT3D's currently linked checkpoint folder exposes only `pointnet.pth`.
SVWA's linked folder has DGCNN and CurveNet for ModelNet40, DGCNN for
ScanObjectNN, but no ScanObjectNN CurveNet file. 3DeformRS supplies the missing
CurveNet alternative in its repository; evaluate only on the requested test
split and explicitly report the training-split mismatch.

## Ordered implementation / research batches

### Batch1 — provenance and reproducible assets
- Goal: establish six candidate identities and valid evaluation arrays.
- Scope/touched areas: preparation artifacts, this note, findings and questions.
- Stack context: CPU/network and existing r7 environment; no new setup.
- Dependencies: complete primary-source audit and checkpoint/data integrity.
- Implementation notes for $development-agent: pin source bytes and hashes,
  reject partial classifier heads, preserve raw receipts and failed attempts.
- Verification: SHA256, safe archive checks, tensor names/shapes, class count,
  preprocessing trace and581-example main_split provenance.
- Knowledge artifact to update: this note.
- User review gate: existing task authorization; GPU>60min approval outstanding.
- Out of scope: adaptation or any paper-reproduction claim for alternatives.

### Batch2 — smallest classifier boundary and Point-MAE parity
- Goal: a frozen logits interface without altering established Point-MAE paths.
- Scope/touched areas: `classifier_backbones.py`, focused CPU tests; subsequent
  opt-in source-only runner wiring only after concrete assets are validated.
- Stack context: PyTorch, existing native FPS, no added runtime dependency.
- Dependencies: inspect actual model call contracts; primary audit above.
- Implementation notes for $development-agent: accept BxNx3 XYZ; Point-MAE
  calls `classification_only(..., only_unmasked=False)` unchanged. DGCNN and
  CurveNet accept Bx3xN; PointNeXt uses its audited pos/features contract.
  Keep sampling/normalization outside logits dispatch; preserve input gradients
  while freezing parameter and BN state. No broad registration/plugin system.
- Verification: exact Point-MAE tensor/logit/RNG/buffer equivalence, batch1,
  wrong shape/point-feature input rejection, frozen parameters and finite logits.
  Real CUDA parity belongs to the approved remote phase, not CPU evidence.
- Knowledge artifact to update: this note; root agent handles findings/questions.
- User review gate: existing task authorization, targeted tests before commit.
- Out of scope: source loader guesses, GSD wiring, changes to trajectory math.

### Batch3 — source-only evaluation and discrepancy diagnosis
- Goal: six matched full15 source rows with per-corruption paper deltas.
- Scope/touched areas: opt-in source-only runner, checkpoint preprocessing
  contracts, immutable seven-file bundles plus indexed logits/provenance.
- Dependencies: Batch1/2 and GPU approval; smoke before full coverage.
- Verification: strict checkpoint load;15 rows ×2468 or581 original IDs,
  fraction/percent units, independent macro recomputation, frozen-state hashes.
- Knowledge artifact to update: this note, findings and questions.
- User review gate: explain all source differences before next batch.
- Out of scope: selecting preprocessing or weights to maximize test accuracy.

### Batch4 — conditional frozen-classifier GSD comparison
- Goal: GSD-inspired latent spectral guidance + each validated backbone.
- Scope/touched areas: opt-in existing runner/legacy artifact wiring, no
  diffusion or spectral equation changes; matched source-only per checkpoint.
- Dependencies: accepted source diagnosis, parity and complete GPU approval.
- Verification: reconstruction/frozen-state smoke, full15 coverage and matched
  checkpoint/preprocessing/IDs, per-corruption deltas and resource use.
- Knowledge artifact to update: this note, findings and questions.
- User review gate: inspect each material immutable bundle before promotion.
- Out of scope: full GSDTTA, classifier adaptation, parameter sweeps or training.

Each material batch receives a fresh immutable bundle, source bytes and
manifest, verified Drive/local SHA256, and a separate selected-file commit/push.

## Batch2 CPU boundary evidence — 2026-10-18

[Code] Added `classifier_backbones.FrozenClassifier` and seven focused CPU
tests without changing existing callers. Layout is a required explicit
`input_layout="bnc"|"bcn"` argument: SVWA models already transpose internally,
whereas the alternative CurveNet implementation expects BCN. Point-MAE keeps
the input tensor and `classification_only(..., only_unmasked=False)` call.
Parameters and eval/BatchNorm state stay frozen while XYZ gradients remain
available. Sampling, normalization and checkpoint loading remain external.
PointNeXt raises `NotImplementedError` until its checkpoint-specific
pos/features contract is audited; no height feature was invented.

[Run] Observed seven RED failures before implementation, then seven passing
tests with cached CPU PyTorch2.14.1. Final command:
`uv run --offline --no-project --with torch --with numpy python -m unittest discover -s tests -p test_classifier_backbones.py -v`.
Tests cover exact stub Point-MAE logits/input/RNG/buffer parity, both explicit
layouts and batch1, input gradients, parameter/BN freezing, malformed XYZ and
logit rejection. This is boundary evidence, not real CUDA/checkpoint parity.
Full CPU unittest discovery additionally reported one unrelated failure in
`test_colab_research_operator_skill.test_connection_resilience_preserves_real_work_without_fake_keepalive`:
the concurrently edited skill no longer contains its expected anti-idle
sentence. That external edit and its test were preserved. No GPU, runner
wiring, commit, push or immutable remote result bundle occurred in this batch.

### PointNeXt boundary extension — 2026-10-18

[Code] The earlier PointNeXt block is superseded after inspecting official
OpenPoints `db31e0d94ede35bc672ef06d1b6e4073a9c269d5` source in
`result/preparation/multibackbone_20261018/openpoints_source.zip` and the saved
`official_openpoints_cls_base.py`. `BaseCls.forward` passes the data dictionary
to `PointNextEncoder.forward_cls_feat`, which reads BNC `pos` and BCN `x`.
The boundary now supports `forward(points, *, features=None)`: default `x`
is transposed contiguous XYZ; explicit BCN features pass through unchanged.
PointNeXt requires declared BNC positions. Explicit features must have matching
batch/point dimensions, nonempty channels, XYZ device/dtype and finite floating
values; other backbones reject them. ScanObjectNN's externally prepared fourth
height channel is never derived or normalized by this boundary.

[Run] Observed the missing-contract RED errors before implementation, then all
10 focused CPU tests passed with the same command recorded above. New checks
cover dictionary position identity, exact default feature layout, explicit
four-channel feature identity, gradients through both XYZ and features, and
invalid feature rejection before model execution. Existing Point-MAE parity
checks still pass. No full-suite rerun, source loader, GPU execution, checkpoint
inference or caller integration was included in this extension.
Annotations use `typing.Optional` for the remote Python3.8 environment; both
module and tests passed `ast.parse(..., feature_version=(3, 8))`. The 10 tests
were rerun successfully after this compatibility fix. Native Python3.8 and
PyTorch2.1.2 runtime execution remains a separate remote verification step.


## Accepted asset audit and native preparation — 2026-10-18

[Run/Code] All six selected alternatives load with `strict=True` and
`weights_only=True` on CPU. Four DGCNN/CurveNet checks are recorded in
`result/preparation/multibackbone_20261018/strict_load_probe.json`; the two
PointNeXt checks are in `pointnext_strict_cpu_attempt3.log`. This is model-state
compatibility evidence, not inference or paper-checkpoint identity. The two
3DeformRS `curvenetScanobjectBaseline` / `curvenetScanobjectBaseline_v2` candidates
have 40-output heads despite their directory names and were rejected. The
accepted `BetterCurveNetOnScanObjectnn` file has 15 outputs. No checkpoint was
chosen by corrupted-test accuracy.

| Accepted file | SHA-256 |
|---|---|
| svwa_dgcnn_modelnet40.pth | e9eb118cd5adb42b7a564e2cc396ce698dee6cad42342d5e6c518e47ea152494 |
| svwa_dgcnn_scanobjectnn.pth | 7816a3ccf915406c5cc613883d8deb187d78e8f505df925d83cdcb8ac1343bfe |
| svwa_curvenet_modelnet40.pth | 1288beba37b65d063f87d45b25c24895f739188daf7d4001ed6ae2f44beabe20 |
| 3deformrs_curvenet_scanobjectnn_better.t7 | 470725f49015908c9ec8c4ab6cf27d827ce021ab7d91b638bfe416c0d9649085 |
| pointnext_modelnet40.pth | 4c9c6deacc82235b864c6adfe3251c29bc4848038f03fc32402ba302f0b3ff97 |
| pointnext_scanobjectnn.pth | 5b51fe01381c65889f268c85a1a52fa37f00941c874cf228e39dcc881a85e52a |

[Code] SVWA DGCNN's `transformer_config.dataset_name` selects only its head
parameter name: the ModelNet checkpoint requires the string `scanobject`, and
the Scan checkpoint requires `modelnet`. The runner records this quirk rather
than silently renaming heads or interpreting the string as split provenance.
SVWA DGCNN and CurveNet accept BNC (they transpose internally); the accepted
3DeformRS CurveNet expects BCN. Official PointNeXt-S uses width64/in3 for the
selected ModelNet checkpoint and width32/in4 for Scan. Its Scan training is
PB_T50_RS; the CurveNet training backup is OBJ_ONLY. Exact SVWA Scan training
log/checkpoint linkage remains uncertain.

[Code] The source protocol is predeclared in
[specific runner note](gsd_multibackbone_source_runner_20261018.md): raw XYZ and
native PointNet2 FPS1024 for SVWA; mean-center/L-infinity then FPS for the
alternative Scan CurveNet (transfer from its random-sampling trainer); raw FPS
XYZ for ModelNet PointNeXt; FPS, unscaled y-height, mean-center and unit-sphere
XYZ for Scan PointNeXt. These benchmark-transfer choices are explicit and
must remain identical in matched guidance. No accuracy-based preprocessing
selection is allowed.

[Run] BFTT3D's published ScanObjectNN-C archive (Drive ID
`1PpS4oPoPA03-RWTiVp-huvhqa7jh3PbW`, 488341020 bytes, SHA-256
`e236151c9f0ad805f59d3571a956b75ada2e1208b030f56f84546cc993b38eb2`)
passes CRC and safe-path checks. All 581 released clean clouds contain 1024
points that are exact coordinate members of the corresponding official
`main_split` 2048-point H5 object: maximum nearest distance is zero for all
581 objects, and labels match in original order. This proves split/order
membership, not GSDTTA corruption seed or byte identity. Use its canonical15
severity5 arrays; exclude extra `scale`. The existing Drive `scanobjectnn_c`
arrays were severity8 and the other s5 receipt covers Gaussian only; neither
is silently reused as all15. The new bundle has an independent manifest.

[Run] PointNeXt's official PointNet2 batch native extension built on r7 with
CUDA12.1, GCC11.4, torch2.1.2 and arch8.0; import and expected symbols pass.
No GPU tensor/forward was run in this phase. Binary is 12449056 bytes,
SHA-256 `a2db745611c01ea03fab6e5d5e1d9e5f08837e847780dd27555eec561bc58aab`.
The original CLI stream stalled after its initial messages despite the on-disk
build completion at19:23:30 UTC. After independently confirming state and
`.complete`, only the local waiting CLI client was interrupted. A bounded
recovery inspection confirmed no active build and emitted exact recovery and
archive markers; the runtime was not stopped and the completed build was not
replayed.

[Artifact] Recovery ZIP preserves 254 actual remote source/config/native-source
files, Git status, environment and build log/state. Local
`result/preparation/multibackbone_20261018/r7_source_environment.zip` and Drive
`MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261018T193557Z/audit_20261018T193557Z.zip`
both have 478900 bytes and SHA-256
`1043480475cfc0f4262ad175ce22a092921a90ffb5ea507ce1bf43af3c089bfd`.
Local safe paths and CRC pass. This archive describes the existing dirty
`dev@a458cd4` runtime; planned inference will use a separately pinned local
base2684 snapshot with only this study's code overlay.

[Open] Real CUDA Point-MAE parity, six source smokes and six complete source
rows remain pending. Source discrepancies must be documented before any GSD
integration. No GPU accuracy claim, long-phase approval, or GSD result exists.


## Sealed preparation and actual r7 CPU preflight — 2026-10-18 19:51 UTC

[Run/Code] Final focused command ran26 tests successfully in1.860s with
ResourceWarning promoted to error: boundary10, source13, parity3. Log:
`result/preparation/multibackbone_20261018/cpu_tests_final.log`. The combined
run exposed PyTorch's synthetic relative `_classes.py` module filename in the
source inventory. Nonexistent synthetic relative paths are skipped; missing or
undeclared absolute asset modules remain strict errors, covered by regression.
Both runners and the boundary pass Python3.8 AST syntax checks and scoped diff
checks. Boundary batch is committed as `e91cca51b3867f3abed306603911012a1c9b72a9`.

[Artifact] Source ZIP (589 files; repository base2684 plus only the six declared
study code/test overlay files, pinned backbone sources/configs and native binary)
SHA-256 is `683924c4e8caad6459dfdf48caccfd545185501d3ea04257ae7f114028e09dd5`.
Preparation base ZIP249654807 bytes has SHA-256
`7cb6db90868a4151a9096324ebdd95ae4d3816433281a58f778403ed1c55df2b`;
remote and Drive copies match local. A single large CLI upload returned500;
eight individually hashed chunks transferred and reassembled to the exact
original ZIP. No partial upload was treated as success.

[Run/Diagnosis] First CPU preflight correctly rejected a wrong manifest field:
local preparation inferred ModelNet labels as `[2468]`, while actual NPY headers
are `[2468,1]`/uint8. Every dataset byte/hash already matched; the metadata
inference was the root cause. A bounded read of all17 actual NPY headers
confirmed the remaining shapes. Do not weaken the runner's exact-shape check.
The failed v1 bundle is preserved and verified locally/Drive:2752 bytes,
SHA-256 `ddfa7c9d44b19d1ccbf477b37f32167e274379b994dac205265639db97ff4597`.

[Artifact/Run] Revision2 creates a fresh isolated directory, preserves v1, and
reuses identical data/weights/source ZIP with corrected observed metadata.
Manifest SHA-256 is
`e6421333fdaab6fdd30167ea905f3d829d40de77f78fe91b696c035caa02a07a`.
Its39060-byte delta ZIP SHA-256 is
`032b10d264a2ebbe8bef1a4e715802d39f9f70fd3ab2ec3038620e9998eb8f82`.
Source/code hashes are unchanged. Execute from
`/content/gsd_multibackbone_study_20261018/prepared_v2/assets/source_snapshot`.
Restore base payload plus this delta for recovery; old manifest is superseded.

[Run] Actual r7 Python3.8.20/torch2.1.2+cu121 loads all six classifiers strictly
with no missing/unexpected keys and validates both full datasets against the
manifest. The preflight temporarily substitutes only Module.cuda's final device
transfer with an identity and forbids Tensor.cuda; no forward is executed.
`torch.cuda.is_initialized()` staysFalse. This establishes exact runtime CPU
loader/import/state compatibility, not a CUDA-forward result. Wrapper and
six-load completion markers both appear directly in the monitored primary
stream. CPU phase took12.719s.

[Artifact] Passed CPU bundle4198 bytes/SHA-256
`782225ef24f418fe4cdf41c9752a378684f02f0eed66921a21391075f8941d7e`,
Drive directory
`MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261018T195047Z/`.
Base payload Drive directory is `20261018T194915Z/` under the same parent.
Local receipts, safe paths, CRC, report/receipt agreement, per-model strict-load
checks and Drive/local ZIP SHA checks pass. Full paths and six state hashes are
in [the provenance summary](gsd_multibackbone_provenance_20261018.json).

[Open/User gate] Prepared whole GPU campaign, estimates, exact initial commands,
asset IDs and conditional guidance settings are in
[the GPU phase note](gsd_multibackbone_gpu_phase_20261018.md).
Historical full15 legacy runtime4138.711s gives an uncertain4–8h campaign
estimate; the proposed cap is8 GPU-work hours, seed0 only. Approval is still
pending. No accuracy, real CUDA parity or GSD result exists. r7 responded to
the completed CPU phase and remains live; the local monitor owns CLI execution.


## GPU parity accepted; persistence and legacy payload fix — 2026-10-18

[User decision] Whole conditional8-hour GPU campaign approved; save every
completed model to Drive and checkpoint after each corruption. Verified optional
Drive snapshots and final receipts are implemented without numerical changes.
All29 focused CPU tests pass (boundary10/source15/parity4). Revision4 uses
manifest `bf964525b6d3e1d747f7d6ba959c02bd7f7153b0b9d79725ba08007b049cb7f8`
and sourceZIP `ec136350e58506e700fb30e35c3e33a39fcdc3cc2cabfc542d54f9f28e039183`.
Preparation deltaSHA
`cdb4d6599038a0f5bba1bf41a86abf28822312315467374c0e4dbe08c9f23284`
and CPU ZIP `d6588b0e84d5ab63460768e6d20f9bb5def174d419c0592f930ca6d0262aed2b`
match Drive/local. Actual CPU preflight strict-loads all six alternatives and
both original Point-MAE models; CUDA stays uninitialized.

[Run/Diagnosis] Initial parity stopped before forward because torch2.1.2's
weights-only unpickler rejects legacy NumPy scalar training metadata. Only the
two exact-SHA-pinned Point-MAE files now use the original repository's legacy
payload loading semantics (`weights_only=False`). The helper hashes the exact
open handle immediately before loading it; CPU regression proves a wrong SHA
is rejected before torch.load. All new-backbone loads remain weights-only.
Original checkpoint bytes, strict model loading and classifier math are
unchanged. Failed attempt ZIP/traceback is retained with Drive/local SHA
`84b108a8ca186d3b0f171feb14acc832010f8f5ae5046a0279fcc7a208357599`.

[Run] Actual CUDA parity passed on both datasets, Gaussian/Density/Background,
32 original IDs each, seed0, batch32. Direct Point-MAE and FrozenClassifier have
bitwise equal logits/predictions, equal post-call Python/NumPy/Torch/CUDA RNG,
unchanged inputs and unchanged full model-state hashes. Maximum logits
absolute difference0.0 for all six conditions. This validates the classifier
call boundary, not end-to-end GSD preprocessing or benchmark accuracy.
The accepted phase took30.967s; the failed loading attempt consumed11.328s.
No source-only alternative model forward has run at this gate.

[Artifact] Accepted raw ZIPs:
- `result/modelnet40_c/pointmae_boundary_parity/20261018-201336_pointmae-seed0.zip`,
  SHA `61cef430e37e75cbff274c5be66a8e2f6bd4fb32d11a03c2833b1b51f7b7a456`.
- `result/scanobjectnn_c/pointmae_boundary_parity/20261018-201351_pointmae-seed0.zip`,
  SHA `ab42b3b38a229dedebbe82a9a58d9a5e849f791c43ca1602cdf0f36b5f87a19d`.
Each has canonical seven files, exact sourceZIP, indexed paired logits and
per-file manifest. CRC, safe paths, every member hash, exact NPZ equality and
Drive/local hashes independently pass. Drive directory is
`MyDrive/thesis/result/gsd_multibackbone_study/<dataset>/pointmae_boundary_parity/<run_id>/`;
three intermediate parity receipts plus final verified ZIP are present per run.

[User decision/Run] The later instruction to close assignments unused by this
task explicitly supersedes the earlier no-touch rule for the two observed CPU
assignments. Both were released; only r7 remains, at6.77CU/hour in the latest
inventory. A second inactive local monitor was discovered and stopped locally;
PID34633/session97631 is now sole CLI owner. No runtime teardown occurred for
r7. Monitoring cannot guarantee Colab lifetime/quota retention. The next gate is
six source smokes, then complete all15 evaluation and discrepancy diagnosis.
