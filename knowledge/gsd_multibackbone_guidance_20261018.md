# Matched multibackbone latent spectral guidance — 2026-10-18

[Decision] The independently verified six all15 and six clean source controls
are eligible for a published-alternative matched experiment, as recorded in
`gsd_multibackbone_source_results_20261018.md`. Exact paper checkpoint identity
and a unique causal explanation of its accuracy gaps remain unresolved. This
study is **GSD-inspired latent spectral guidance + <backbone>**, not full
GSDTTA or paper reproduction. Following the latest scope steering, guidance
currently selects CurveNet and PointNeXt on both datasets; DGCNN is parked.
The six existing source rows remain unchanged evidence.

[Code] Added the opt-in `scripts/eval_multibackbone_guidance.py` without
changing existing source defaults, Point-MAE runners, spectral equations or
checkpoints. Each dataset invocation creates two separate standard RunBundles
under `gsd_inspired_latent_spectral_guidance`, preserving indexed labels,
predictions/logits and fraction-based count columns. Actual execution requires
an immutable manifest and the 581-example `main_split` ScanObjectNN contract.

The pinned source validation and its original all15 asset manifest are required
as `manifest.guidance.matched_source_results` and `matched_source_manifest`.
The former must link the latter by SHA256. Each selected row must contain all15
original-ID counts and the identical classifier checkpoint/preprocessing;
labels and selected corrupted file identities/shapes must also match. Guidance
does not substitute a numerically closer checkpoint or tune preprocessing.

One classifier-independent reconstruction is shared per dataset/batch. Legacy
math is bbox normalization -> unchanged upsample2048 -> scale3.3885/rotation ->
`tta_gsd_v2.tta_gsd_reconstruct` with observer off -> rotation back. ScanObjectNN
then divides3.3885 and reverses the original bbox normalization; ModelNet40
applies the existing output bbox normalization. The complete decoded2048 cloud
reaches each head's separately audited source preprocessing. No terminal
FPS1024 output is shared across heads.

Locked protocol: B70, Mmax800, beta2, spectral1.17, invariant0, SCD1,
local/style rates0.01, p0.95, normal10/background30 of100 DDIM steps, static XYZ
graph, legacy isolation, updated decoder style and final DDIM alpha-one=False.
VAE/priors stay eval/frozen, raw LION weights, EMA off. LION checkpoint SHA is
`807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`;
the config SHA is
`9d4a826f35fe9b3053210105eb4857811c8bfd53cd8c6a998f8ea7953e975816`.
External absolute path identities are verified before the legacy load; the
resolved LION config and strict classifier load reports are preserved.

`manifest.guidance.native_bootstrap` names a pinned source helper. Its
`bootstrap(asset_root)` must load independently audited prebuilt native modules
and return `no_build=True` before any LION/GSD import. The runner invents no
binary locations or JIT build fallback. Native readiness and actual CUDA
execution remain separate verification gates owned by the main workflow.

Head initialization is enclosed in the existing `isolated_probe` RNG guard.
Python/NumPy/CPU/CUDA RNGs are explicitly seeded after all initialization.
Each head's entire preprocessing plus forward call is independently isolated;
RNG hashes and the full geometry hash must remain unchanged after it. Labels
enter only prediction artifacts/counts after inference. LION/classifier state
hashes cover before/after execution, including best-effort failure audits that
preserve the original error. Fresh legacy cuDNN defaults are explicitly
benchmark=False, deterministic=False, deterministic algorithms off; this is
recorded alongside the source-only benchmark=True policy and does not assert
kernel identity or common-draw pairing with the original source stream.

CSV runtime allocates shared reconstruction time equally to the two heads,
then adds each head's separately synchronized preprocessing/forward time.
Summing both heads therefore counts reconstruction once. Configs preserve the
unallocated shared seconds and common reconstruction group ID. Peak GPU memory
is explicitly the entire joint process, repeated per head and never summed.
Head timing includes the isolation guard and its RNG/geometry hashing.
Artifact/Drive I/O and model initialization are outside these component timers.
No pure-inference speed or speedup claim follows from this bookkeeping.

Every batch writes a fresh compact NPZ plus geometry/RNG/loss diagnostics. The
same exact bytes are copied to Drive with SHA/byte verification before an
atomic completion receipt. Paths are
`<drive>/<dataset>/<method>/<run_id>/batches/<corruption>/batchNNNN/`.
An interrupted copy has no complete receipt; collisions fail rather than
resume or overwrite. Completed corruptions reuse the source runner's immutable
ZIP checkpoints; final and failed bundles are separately sealed and copied for
each head. Intermediate ZIP staging stays outside bundles, and partial
corruption ZIP logs are explicitly prefixes. No automatic resume is provided.

[Run] Eight focused CPU tests passed after initial missing-module RED evidence.
They exercise the actual existing normalization/rotation/inverse functions,
2048-point geometry and locked reconstruction kwargs, separate FPS calls,
real RNG/buffer isolation even on classifier failure, initialization/seed
boundary, strict matched-source linkage, compact Drive shard receipts and
hash failures, native-free preview, and sealing both failed bundles plus their
Drive final archives. These are synthetic CPU contracts, not LION/CUDA parity
or measured adaptation results. The combined boundary/source/Point-MAE/guidance
suite passed all38 tests with resource warnings treated as errors. The main
workflow pins final source hashes and performs deployment/remote verification.

[Open] Real no-build bootstrap, CUDA smokes, actual B70 memory/time, all15
guidance metrics and matched source deltas are not yet established by this
implementation batch. No Colab/GPU execution or commit occurred here.


## 2026-10-18 20:47 UTC — two-head guidance integration CPU gate accepted

[User decision] PointNeXt+CurveNet explicitly confirmed on both datasets;
DGCNN parked, its six source/clean results preserved. Original3DD-TTA and a
same-flow spectral0 comparator remain proposed next stages, not launched.

[Code/Run] New opt-in shared reconstruction runner and scoped native binding
pass33 focused CPU tests (plus10 unchanged FrozenClassifier checks earlier).
Actual r7 Python3.8/torch2.1.2 CPU preflight loads all4 audited native binaries
without JIT build, restores the original loader, leaves CUDA uninitialized,
strict-loads six alternative and two original Point-MAE models, and verifies
four selected matched-source rows against original all15 manifest/data bytes.
The original Point-MAE/GSD entrypoints and equations remain unchanged. Real
LION loading, B70 GPU memory/time and adaptation metrics remain the next smoke
gate; CPU checks do not establish those.

[Artifact] Revision6 manifest SHA
`c88e96de38aa3498f3e3989b286d52e1970a4f1a62d33f86d8e71bd8c9329e25`;
sourceZIP SHA `a43b60d5e62c3a25c6a302cb7509b0042b04846bfe40918570dbfe6bc3e01c5e`.
Preparation delta5523990 bytes SHA
`9dfca4fe1fe6595326a77f95646273826612f6553ce818366b4477551c207706`.
Actual CPU proof ZIP28303 bytes SHA
`e7a69695ff9cc09650040fb2c607b67f598ef62b230521454f2c00a2e37167ee`,
local `result/preparation/multibackbone_20261018/cpu_preflight_v6.zip`;
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261018T204639Z/`.
Source/preparation and CPU proof copies match Drive/local SHA; CRC, all nested
phase source members and exact executed deployer bytes pass. Audit v2 native
ZIP SHA `c77d71e934e7b52f65493e49957f2ff3f45986babf4cd4b6f3d862b10932fc2e`
also passes all99 member hashes and Driveroundtrip. CPU phase elapsed21.095s.

[Decision/Next] Retain fixed legacy800/beta2/spectral1.17/SCD1, normal10/BG30,
seed0/B70, all other predeclared raw/eval/updated-decoder policies. Each head's
full source preprocessing is separately RNG-isolated on the same decoded2048
geometry. Batch shards and corruption/final ZIPs have immutable SHA-verified
Drive receipts. Proceed to3-condition one-batch GPU smoke per dataset/head;
only after artifacts/finite geometry/logits/losses/frozen states/RNG guards pass, run full15.
Continue within the user's unchanged28800s GPU-work cap and preserve r7.


## 2026-10-18 20:53 UTC — two-backbone CUDA smokes accepted

[Run/Artifact] Existing r7 completed both selected heads on Gaussian,
Background and Density, first70 original IDs each, seed0/B70. All840 indexed
predictions independently validate finite logits/argmax/pinned labels/counts;
shards exactly concatenate to corruption NPZs. Six shared geometry batches
match across the two heads, all RNG guards and frozen classifier/LION state
hashes pass. Native bootstrap reports all4 pinned binaries/no-build. Raw/eval
LION loads successfully; peak whole-process allocation32381.92MiB (~31.62GiB).
Reconstruction takes~7s/normal batch and~13.8s/Background; full two-dataset phase
is estimated1.5–2h, within the existing8h approval. Smokes are partial subsets,
not all15 accuracies and not evidence of improvement over full source macros.

All4 final ZIPs and12 batch NPZ/diagnostic receipts were independently checked
against exact local/Drive bytes, in addition to per-corruption checkpoints.
Raw ZIPs live under canonical `result/<dataset>/gsd_inspired_latent_spectral_guidance/`;
run IDs and full hashes are in the provenance JSON and guidance smoke validation.
Smoke jobs consumed57.106+56.524s; no model/parameter setting changed afterward.

Immutable evidence ZIP local
`result/preparation/multibackbone_20261018/guidance_smoke_evidence_v1.zip`,15264bytes,
SHA `4760b814c94b3bab81f26b9954e872a85663dd106407f72184af2922df44de75`;
Drive `MyDrive/thesis/result/preparation/gsd_multibackbone_study/20261018T205237Z/`.
It contains the exact executed CPU verification phase, ledger, final phase
state,12 Drive receipt audit and independent derived analysis/source bytes.
Local/Drive SHA, CRC and all member identities pass. Current code batch is
pushed as fbb096b; executed snapshot remains separately identified by revision6.

[Decision/Next] Run fixed full15 seed0 for CurveNet+PointNeXt, both datasets,
with batch-level immutable receipts and per-corruption/final model ZIPs. Compare
only to each own fixed full source baseline. Original3DD-TTA and spectral0
controls stay future proposals. Preserve r7 and the sole local monitor.


[Run/running] 2026-10-18 20:53:36 UTC — fixed full15 two-head phase launched
on protected r7 through monitor97631. Exact controller SHA
`c609a312d72068154c244ddeaccb3c9b3f8620c9431c17fc4b3a76f84f6981d7`,
revision6 manifest/source unchanged. Phase path
`/content/gsd_multibackbone_study_20261018/gpu_phases/20261018T205336Z_guidance_all15/`.
ModelNet run IDs `20261018-205338_curvenet-seed0` and
`20261018-205339_pointnext-seed0`; Scan follows automatically only after both
ModelNet final ZIPs verify on Drive. First Uniform batches are producing
verified immutable Drive shards. Prior GPU-job consumption645.614s of28800s.
Estimate1.5–2h for full phase, existing approval unchanged; no new3DD-TTA run.
Do not start another cell or duplicate job while this phase is active.


## Proposed subsequent comparison (read-only audit, 2026-10-18)

[Code/Artifact/Plan] `tta.tta_reconstruct` already returns classifier-independent
2048 decoded points before the terminal FPS. A future opt-in repository-original
3DD-TTA eval/raw runner can reuse the strict heads, separately audited source
preprocessing, RNG isolation, native bindings and artifact helpers. Existing
`main_3dd_tta.process_batches` owns the common geometry transforms and final
Point-MAE/FPS boundary; preserve the full decoded cloud for the new heads.

The archived operational 3DD eval/raw reference is B32, normal5/Background35,
100 scheduled steps, p.95, unnormalized SCD sum, gamma local=.01, eta style=.01,
original shape-latent decoder style, actual DDIM final-alpha-one=True,
cuDNN benchmark=True, eval/raw LION and EMA off. The main CLI's batch40 default
is a separate setting. Existing `scheduler_observer` can verify actual scheduler
config/timesteps; this remains source/protocol planning, not new execution.

Current guidance instead uses B70,10/30,updated style,alphaFalse,benchmarkFalse.
Therefore the later reference comparison measures the entire method policy.
To isolate added spectral guidance, a distinct same-flow `tta_gsd_v2` spectral0/
SCD1 arm must retain every other current setting; original `tta.py` cannot stand
in for this ablation. Existing Point-MAE scores do not transfer to new heads.
No new original3DD-TTA or spectral0 GPU work was launched or claimed complete.


[Run/Partial campaign] 2026-10-18 21:00 UTC (local date2026-10-19) — first
full corruption accepted while the unchanged full15 phase continues. ModelNet
Uniform has2468 indexed predictions per head. CurveNet source1661/2468=67.3015%
versus guidance1774/2468=71.8801%, +4.5786pp (271 corrected/158 harmed objects).
PointNeXt source1117/2468=45.2593% versus guidance1880/2468=76.1750%, +30.9157pp
(844 corrected/81 harmed). Both immutable corruption ZIPs were downloaded from
Drive; receipt SHA/bytes, safe paths, CRC, every checkpoint member hash, labels,
IDs, finite logits/argmax/counts and original-source predictions independently
pass. Local paths are
`result/preparation/multibackbone_20261018/full_modelnet_<backbone>_uniform_checkpoint.zip`;
ZIP SHA values are `d2ed9e9edb465f28d85b41652fba4c56ece8976986fc011ec80ef66a1dd6cf32`
(CurveNet) and `579b5a7b74ecb9076e8515a152fc7c627e342453d218b0e5a97ef4367b80526c`
(PointNeXt). Drive roots are the current205338/205339 run directories,
`corruption-uniform.zip` plus receipt. Derived counts are in
`uniform_full_checkpoint_validation.json` under the same local preparation root.
These are single-corruption descriptive gains, not all15 means or isolated
spectral contributions. No setting changed after observing the labels.


[Run/Partial campaign] 2026-10-18 21:07 UTC — full ModelNet Gaussian
checkpoint downloaded from Drive and independently verified for both heads:
CurveNet source1902/2468=77.0665% ->1770/2468=71.7180%, -5.3485pp
(175 corrected/307 harmed); PointNeXt1356/2468=54.9433% ->1871/2468=75.8104%,
+20.8671pp (641 corrected/126 harmed). Receipt size/SHA, CRC/safe paths,
checkpoint member hashes, pinned IDs/labels, finite logits/argmax/counts and
frozen classifier hashes pass. ZIP SHA values:
CurveNet `b1251937f74ee9ecb1a9457c9db405826bfd80c461f30a1e8e8e0d44f5ad0379`;
PointNeXt `ef5424b8f100ba9e8bdccfd808c23a6e379e2c567e2e19cf0a77e1e92d5a48a5`.
Local files are `full_modelnet_<backbone>_gaussian_checkpoint.zip` and
`gaussian_full_checkpoint_validation.json` in the preparation root. Drive
roots remain205338/205339, `corruption-gaussian.zip` plus receipt. Mixed
positive/negative corruption effects preclude an all15 benefit claim now.
Full15 continues unchanged; no target-based configuration selection occurred.


[Run/Artifact] 2026-10-18 22:05 UTC — ModelNet full15 accepted: CurveNet
64.0113->65.1864%, +1.1750pp; PointNeXt65.8320->66.4722%, +0.6402pp. Independent
74040-prediction/540-shared-batch checks pass, full classifier/LION states and
RNG guards unchanged. Each head improves6/worsens9 corruptions; no universal
benefit or isolated spectral claim. Joint duration4165.243s; raw final ZIPs
are verified on Drive and locally. Derived8837-byte analysis ZIP SHA
77ff619a008b881702358b83e4fbe54e6518bf3904cbdb17c13a6500389f1eff passes actual
Drive upload/download roundtrip and all member identities. Full identities,
per-corruption table and interpretation are in the new guidance-results note.
Scan full15 now runs; no extra GPU phase or config change.
