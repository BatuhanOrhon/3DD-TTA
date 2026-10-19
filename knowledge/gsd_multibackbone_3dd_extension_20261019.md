# Original 3DD-TTA and explicit DGCNN extension — 2026-10-19

[User decision] Prepare original 3DD-TTA for CurveNet+PointNeXt on both
datasets, followed by DGCNN with both the GSD-inspired and original policies.
This is a code preparation batch; GPU launch and the proposed campaign budget
remain the main workflow's separate approval/execution gate.

[Code] Extended only `scripts/eval_multibackbone_guidance.py` and added
`tests/test_multibackbone_3dd_extension.py`. Existing GSD default selection is
still CurveNet+PointNeXt, method `gsd_inspired_latent_spectral_guidance`, B70
and every established reconstruction equation/setting unchanged.
`--method original3dd` selects the canonical `3dd_original` result family;
`--backbones dgcnn` explicitly selects the previously audited DGCNN alternative.
Unique arbitrary selections of the three already supported heads are accepted;
default selection and existing source-only behavior are preserved.

The original branch calls unchanged `tta.tta_reconstruct` directly, without
spectral/graph arguments or a graph constructor. It locks the validated
repository operational policy: B32, normal5/Background35 of100 DDIM steps,
gamma local0.01, eta style0.01, p0.95, unnormalized SCD sums,
original `shape_latent` final decoder, raw LION eval/EMA off, final DDIM
alpha-one=True and cuDNN benchmark=True. The command-line B40 default in
`main_3dd_tta.py` is a distinct historical default, not the B32 all15 reference.
Cross-method batch values are rejected rather than silently overridden.

The existing scheduler observer asserts the actual runtime scheduler's linear
beta0.0001–0.02,1000 training steps, epsilon prediction, no clipping,
alpha-one=True, offset0, exact100 inference timesteps990,980,...,0 and final
alpha1. It retains the full/active timestep lists plus complete scheduler config
in every original per-batch diagnostic under `scheduler` and in
`config.scheduler_records[corruption]`. A scheduler change within a corruption
fails closed. Original reconstruction has no fabricated loss metrics; its
`guidance_metrics` dictionary is empty.

Both policies retain the same existing dataset normalization/upsampling,
scale/rotation and dataset-specific inverse/output normalization. The full
2048-point decoded geometry is shared before each frozen head's own audited
preprocessing/FPS. Manifest/checkpoint/source-baseline/data identity checks,
head initialization and inference RNG isolation, explicit post-initialization
seed boundary, frozen-state audits and immutable Drive batch/corruption/final
artifacts remain in use. Native bootstrap is unchanged; importing `tta` after
the audited bootstrap reuses already imported LION/Chamfer dependencies.

Batch iteration, schema method fields and reconstruction-time allocation now
follow the selected method/head count. A one-head DGCNN invocation receives
all shared reconstruction time; a two-head invocation receives half per head.
Default two-head GSD timing/memory descriptions remain unchanged. DGCNN config
records the weaker ScanObjectNN clean/source performance and limits inference
to its own fixed alternative baseline. No checkpoint, class mapping, dataset
split, classifier preprocessing, original `tta.py` or spectral math changed.

[Interpretation] Original B32/5–35/original-style/alpha-True/benchmark-True
versus GSD B70/10–30/updated-style/alpha-False/benchmark-False is a whole-policy
comparison. It does not isolate spectral causality. A future same-flow
spectral0/SCD1 control would require the existing GSD path and all its other
settings, and is not implemented by this branch. Neither method with these
published-alternative classifier weights establishes paper reproduction.

[Run/CPU] RED was observed for unrecognized method/head arguments and missing
original scheduler/dispatch support. Root then ran37 multibackbone tests
successfully in3.022s. Four new tests exercise all four requested mode/head
combinations on both datasets using native-free previews; reject invalid batch
and duplicate-head selections; validate full/active actual scheduler fields
and reject mismatches; compare dispatch bitwise against the actual unchanged
`tta_reconstruct` under synthetic CPU dependencies for both datasets, checking
RNG and original decoder style; and verify canonical method counts plus a
failed one-head DGCNN invocation's sealed/verified Drive final artifact.
The synthetic partial batch18 proves generic partial-batch handling; actual
B32 all15 tails are ModelNet4 and ScanObjectNN5. CPU tests do not establish
trained LION/CUDA parity or the installed remote scheduler/native ABI.

[Open] Main workflow still needs fresh immutable source sealing, installed
runtime CPU/no-build/scheduler proof, selected CUDA smokes, a concrete approved
GPU campaign and independently verified full15 comparisons. No Colab/GPU
operation, commit or push was performed by this implementation batch.

## Prepared GPU phase and accepted CPU gate — 2026-10-19 06:02 UTC

[User decision/Code/CPU] This supersedes the initial implementation-only open
gate above. DGCNN is explicitly back in scope by the user's new request.
Implementation and CPU preparation are authorized; a distinct GPU phase
expected to exceed60 minutes still awaits the single informed launch approval.
No GPU smoke/full job from this extension has started.

### Batch1 — implementation and preparation (accepted CPU gate)

- Goal/scope: extend the opt-in runner to select unchanged repository-original
  3DD reconstruction and explicit DGCNN, retaining default two-head GSD.
- Touched areas: `scripts/eval_multibackbone_guidance.py`, new
  `tests/test_multibackbone_3dd_extension.py`, this note and current research
  records. Existing `tta.py`, spectral math, source contracts and results stay
  unchanged. User's concurrent skill/monitor/setup edits are excluded.
- Context/dependencies: existing r7 A10080GB High-RAM, Python3.8.20,
  torch2.1.2+cu121/CUDA12.1/GCC11.4, pinned six published alternatives and data.
- Verification: root37 multibackbone +10 classifier CPU tests pass; Py3.8
  grammar/diff checks pass. Actual r7 CPU preflight completes in21.329s:8
  strict models (6 alternatives+2 Point-MAE),6 matched-source rows,4 audited
  native binaries without rebuild, unchanged original `tta.py` and actual
  installed DDIM config/full100/active5-or35 schedules. CUDA remains
  uninitialized; no model forward/GPU accuracy is inferred from this gate.
- Review gate: source frozen; GPU duration approval is the next user gate.
- Out of scope: tuning, new checkpoint selection, spectral0, seed replication.

### Batch2 — conditional CUDA smoke then fixed full15 (pending launch approval)

- Goal: complete the requested source/original3DD/GSD comparison for all3 heads.
- Exact full-job order: original3DD CurveNet+PointNeXt jointly on ModelNet,
  then Scan; DGCNN GSD on ModelNet then Scan; DGCNN original3DD on ModelNet
  then Scan. Eight new model/dataset results,182940 indexed predictions.
- Scope: seed0; all15 severity5 corruptions;2468 ModelNet objects and581 Scan
  `main_split` objects, all original IDs. Original B32/5–35 policy above; GSD
  B70/10–30/M800/beta2/spectral1.17/SCD1 prior selected policy. Same pinned
  checkpoint/preprocessing/data for each own source row. Do not use nobg.
- Dependencies: accepted Batch1 and explicit GPU approval bound to scope SHA
  below. Three-corruption Gaussian/Background/Density one-batch smokes on
  each of6 joint jobs produce996 predictions in8 model bundles. Full phase
  requires an independently verified smoke-admission record matching scope
  and manifest SHA; numeric gains are not a smoke acceptance threshold.
- Verification/stopping: strict loads, finite geometry/logits, no state/RNG
  mutation, actual original scheduler, exact IDs/counts and physical Drive
  identities. Stop on mismatch, failure/OOM, missing Drive receipt or4h
  accumulated job wall limit; only child jobs stop, never the runtime.
- Estimate:2.5–3.5h, including all selected smokes/full jobs; cap14400s=4h.
  Based on prior same-r7 GSD full5169.979s and historical original3DD
  ModelNet1773.985s (historical hardware not reverified; new-head times remain
  estimates). Current observed rate6.77CU/h implies roughly17–24CU for the
  estimate, at most~27.08CU for4h of job wall time; idle/preparation/runtime
  lifetime and actual billing are separate. Latest CLI balance495.86CU.
- User review gate: one comprehensive launch approval; no research run yet.
- Out of scope: repeating the4 accepted CurveNet/PointNeXt GSD all15 rows,
  training/fine-tuning, same-flow spectral0 or broader search.

### Batch3 — archive, independently analyze and publish accepted records

- Scope: retrieve all8 raw final bundles; verify182940 predictions,3585
  shared geometry batches,5040 batch receipts and120 corruption receipts.
  Reuse the4 previously accepted GSD results to form6 matched per-backbone
  source/original/GSD comparisons, with per-corruption corrected/harmed counts.
- Verification: safe paths/CRC/member SHA, pinned labels/IDs, finite logits/
  argmax/counts, strict/frozen state and RNG guards, exact actual original
  scheduler; Drive/local byte roundtrips. No isolated spectral or seed-stability
  claim from whole-policy single-seed results. Negative DGCNN results retained.
- Knowledge targets: this note, findings/open questions, README, study and
  provenance; accepted batches committed/pushed separately on current branch.
- Review gate: artifact acceptance, no new GPU authorization for CPU analysis.

## Immutable preparation identities

Parent checkout `acc27d3e16666cd4638744988a1cfd249fa45164`; executed assets are
the separate source snapshot, not assumed identical to remote dirty dev.

- Revision7 asset manifest SHA:
  `78cdb53891bc0714dacd100eb220a5f6deb26bfe36f4e10b0aaa16f9aced60f9`.
- Source ZIP5498028 bytes, SHA:
  `f82cf5052b8100591df679b159acec9f172144c5f6a0a916c5a5255cadeed4cf`.
- Runner SHA `d06a19306aa64de812d3589f81cdcc7364c11c034f3de9c22bec9ae1967092e1`.
- Unchanged `tta.py` SHA
  `c9a901ddeef755359325ad177478f21156fbe1792b2cf4e9797ecfb9de65d4ec`.
- Approval scope SHA:
  `e0eafe5938907d6de45803daa550f486567e86e532cd1f3cbb81362541b74f41`.
- Preparation delta5464063 bytes, SHA
  `d97fe9e5b169db79b1dfe534c8db7150309f6693e521d1358174ba3ce1c27e3e`.
- Actual CPU proof33902 bytes, SHA
  `c725f6cfe02de9a845562e7bb938f979e9c1a8709fe084d452ad0ebf04f5c769`.
  Local `result/preparation/multibackbone_20261018/cpu_preflight_v7.zip`.
  Actual Drive download, CRC and all nested phase-source manifest entries
  match. Exact executed CPU deployer is preserved in nested phase ZIP26751
  bytes SHA `bb9f44dd4987ffad2b0046917bf03bda621ed502469e757dc59676898ec9e98f`.
- Drive preparation root:
  `MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261019T060200Z/`.
  Contains the exact preparation delta, CPU proof and `receipt.json`; delta
  also passes actual Drive upload/download byte equality and CRC.

Exact reviewed phase files under local preparation root:
`gpu_phase_extension_smoke_v7_r7.py` SHA
`6fb832b14bc27d3355cf836bb0b2119ab7d623b0ce0ac5c9faf585ed702591c5`,
`gpu_phase_extension_all15_v7_r7.py` SHA
`ccd395b0fbda7662da17daa13542151fcbc410e9368d10f98775645238cdb041`.
Remote assets `/content/gsd_multibackbone_study_20261018/prepared_v7/assets`;
run through the existing foreground monitor only. Both controllers require
`extension_approval_20261019.json` with approved status, exact scope SHA and
14400s cap. That activation receipt has not been created while approval is pending.
Original campaign ledger5815.594s/28800s is preserved; extension gets its own
4h ledger, not a relabelled expansion of the old approval.

## Persistence and runtime contract

New results: `MyDrive/thesis/result/gsd_multibackbone_study/<dataset>/<method>/<run_id>/`,
method `3dd_original` or `gsd_inspired_latent_spectral_guidance`. Each completed
batch saves immutable indexed NPZ logits/predictions/labels plus diagnostics
and a verified receipt; each corruption saves cumulative ZIP and each model
saves final ZIP. Frozen checkpoint hashes identify weights without duplicating
them per result. Raw decoded geometry/latents/RNG state are not saved and
automatic resume is not implemented. Completed outputs survive interruption;
exact random-stream continuation is not guaranteed.

Protected runtime `tta-scanobjectnn-nobg-a100-r7-20261018`, endpoint
`gpu-a100-hm-kkb-euw4a2-1d8qxj97k4u94`, remains live under sole monitor97631.
At05:47 UTC, a5h-stalled *local liveness CLI client* was terminated; runtime
and monitor were preserved. Fresh bounded audit and subsequent probes/CPU
deployment respond successfully, A10080GB present, no active experiment or
campaign lock. No environment reinstallation or runtime shutdown occurred.
Colab-managed lifetime/quota cannot be guaranteed.

[Review] Focused independent read-only review found no critical/important
blocker in runner, tests, v7 preparation or GPU controller. Actual CUDA
correctness/memory remain the smoke gate after launch approval.


[User decision/Run/running] 2026-10-19 06:16 UTC — user “Devam et”
approves the prepared2.5–3.5h extension with14400s cap, selected smokes and
conditional full15. Exact scopeSHA e0eafe5938907d6de45803daa550f486567e86e532cd1f3cbb81362541b74f41
unchanged; accepted preparationcommit fd7a82d. Approvalactivation is persisted
to Drive and locally,3756-byte immutable ZIP SHA
`a85a588bfaf49e6c389da4525864aed6581dd13c934167f2544013783a8e0bee`,
Drive preparation root20261019T061550Z; exact scope/approval/executed source
bytes, CRC and Drive/local SHA pass. No additional launch approval is needed
for this same phase within its cap.

GPU smoke starts on protectedr7 via solemonitor97631: phase
`/content/gsd_multibackbone_study_20261018/gpu_phases/20261019T061610Z_extension_smoke/`.
Firstjob original3DD CN+PN ModelNet/B32; remainingpredeclared6joborder and
all8outputmodels unchanged. Full15 still requires independent smoke acceptance.
Each batch/corruption/model writes verified Drive evidence. Runtime preserved.


[Run/Artifact] 2026-10-19 06:21 UTC — all6selected extension CUDA smoke
jobs/8model bundles pass independent996-prediction/18shared-geometry validation.
Native4no-build, exact checkpoint/source preprocessing, finite geometry/logits,
strict/frozen classifier+LION hashes and RNG isolation pass. Original method
actual full/5-or35 DDIM schedules match the locked config. All24batch receipts,
24corruption receipts and8final ZIPs physically match Drive/local SHA.
Smokes remain partial prefixes, not full15 accuracies or gain evidence.
Peak whole-process allocation: originalCNPN ModelNet35447.222MiB,
Scan17931.125MiB; DGCNN GSD32366.378MiB, original14494.007MiB (maxdatasets).
Cumulative smoke jobs250.287s of14400s new cap; fixed source/settings unchanged.

Immutable smoke evidence21819bytes SHA
`0b1a91476eb42a649225f8bf2c91ac004ccdab9eb7adf64afddcb4598de0808b`;
local `result/preparation/multibackbone_20261018/extension_smoke_evidence_v1.zip`,
Drive preparation root20261019T062114Z. Actual download, CRC, all nested
manifest members, exact executed finalizer and analysis source verify.
Nested analysis ZIP10896bytes SHA
`a863fea56c71b469ff08c3794440be1f7ec0a39a0df1d490d212dfb83be62404`.
Standalone analyzer revision3 fixes a not-yet-executed revision1 local variable
shadow and adds declared partial/full job coverage checks; actual raw inference
source is unchanged. No GPU rerun was needed for that CPU-only analyzer fix.

[Run/running] Fixed full15 begins06:21:48UTC on protectedr7/monitor97631:
`/content/gsd_multibackbone_study_20261018/gpu_phases/20261019T062148Z_extension_all15/`.
Exact controllerSHA ccd395b0fbda7662da17daa13542151fcbc410e9368d10f98775645238cdb041.
OriginalCNPN ModelNet first, then Scan, then DGCNN GSD/original pairs.
Batch/corruption/model Drive persistence and whole14400s job cap remain active.
No further approval required for this approved scope; no teardown authorized.


[CPU/Artifact] 2026-10-19 — verification hardening during unchanged full GPU
phase: finalizer review required exact NPZ+diagnostic receipt membership and
comparison to phase-pinned sealed ZIP member identities, rather than mutable
bundle-directory files alone. New `finalize_extension_all15_template_v2.py`
implements both. Rechecked all24 actual smoke receipts against pinned sealed
ZIPs: all contain exactly2 expected files and every byte identity matches.
No missing/mismatched actual artifact or inference/source change occurred.
Derived binding verifier+validation ZIP1399bytes SHA
`693b2bc329802b53845d160424de4676ca8f513aa97c1e049b44ba8b3453657a`,
local `result/preparation/multibackbone_20261018/smoke_sealed_member_binding_v1.zip`;
Drive062114Z preparation folder has exact uploaded/downloaded bytes and CRC.
This closes the generic receipt-verifier gap without repeating any GPU work.


[Run/Artifact/Inference] 2026-10-19 06:56 UTC — ModelNet original3DD full15
accepted: CurveNet67.0719% (source64.0113,+3.0605pp), PointNeXt68.2928%
(source65.8320,+2.4608pp). Previously accepted GSD65.1864/66.4722% is lower
than original by1.8855/1.8206pp. Exact paired IDs show GSD corrects2250/2273
but harms2948/2947 events relative to original. Single-seed whole-policy
comparison, not isolated spectral causality; no setting changed.

74040predictions/1170shared batches, frozen states/RNG, actual scheduler, all
member hashes and Drive/local raw ZIP SHA pass. Joint original job2023.697s.
Derived16321-byte matched analysis ZIP SHA
`85ac78483c9ad5eb32e41aa7a40d6b5c9d82e4676ae76b6d957681d26418c5a8`
passes actual Drive roundtrip and every manifest member. Full raw identities,
source/oldGSD links and per-corruption results in
[matched comparison](gsd_multibackbone_method_comparison_20261019.md).
Scan original CN/PN now running; DGCNN jobs follow unchanged within4h cap.
Replicated seeds or matched same-flow controls could falsify/attribute the
one-seed ordering. Preserve r7/monitor; no additional GPU policy/search.


[Run/Artifact/Inference] 2026-10-19 07:06 UTC — Scan main_split581
original3DD accepted: CurveNet55.2381% (source52.0711,+3.1670pp),
PointNeXt49.6845% (source49.0878,+0.5967pp). PriorGSD55.0086/45.2094%
is0.2295/4.4750pp lower than original. Whole-policy seed0 comparison only;
small CurveNet difference is not stable superiority evidence. No settings
changed after observations. Both original datasets now91470 predictions/1455
shared batches independently pass frozen/RNG/scheduler/labels/logits/ZIP
checks. Joint Scan job508.593s; ledger2782.578s before active DGCNN job.
Immutable22758-byte combined analysis ZIP SHA
`8f3be7464f78c12dcf4eb588019fa131ba41f9241980eda71a21d2b7ccd35d64`
passes actualDrive upload/download, CRC/all members. Exact paths/raw SHAs and
paired corruption analysis are in the current method-comparison note.

Primary CLI stream became silent after07:01, but read-only file-service
audits confirmed Scan had completed normally and DGCNN GSD ModelNet was
active/advancing (run20261019-070402_dgcnn-seed0). Server still lists the
exact r7/A100 endpoint. Do not rerun/duplicate or kill/restart the runtime;
continue solemonitor plus immutable file-progress snapshots. At final phase
completion, independently check .complete/state/ledger before repairing any
stale local client and submitting the CPU finalizer.

## 08:05 UTC — Runtime loss; original4 rows retained, DGCNN partial

[Run/Open] r7 disappeared while DGCNN ModelNet GSD was active. Two original
CN/PN joint jobs (4model rows) are accepted on bothdatasets. Last file audit
08:03:20UTC shows13 complete corruptions plus Occlusion490/2468, no traceback.
Repeated server inventories around08:05UTC show no assignments; cause unknown,
no runtime stop/release sent. Preserve exact execution snapshot and raw evidence;
no runtime `.complete` or full-DGCNN acceptance exists. DGCNN ScanGSD and both
originalDGCNN jobs were never launched. Sole local monitor34633/session97631
remains alive/LOST; only orphaned localCLI58698 terminated after confirmedloss.

Loss archive62808bytes SHA256
`68986b855cfc1e1d30633aa5ace7d84a81ab0e302ed02fd8a140d12aa6853b3c`, local
`result/preparation/multibackbone_20261018/r7_runtime_loss_20261019T080916Z.zip`.
CRC and16member/manifests verify; Drive copy pending sitepermission. Original
accepted4-row comparison and Drive/local evidence remain valid. Completed
ledger2782.578s excludes interrupted approximately61min; include that work when
reassessing approved4h budget. A fresh full attempt is distinct from exact
mid-batch resume because only RNG hashes, not actual state, were persisted.
Recover Drive coverage before a new plan; new runtime must replay ten setup
cells/strict asset and native checks before GPU inference. No new run launched.
[Loss evidence and outstanding gates](gsd_multibackbone_method_comparison_20261019.md#unexpected-r7-loss--0805-utc).
