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
