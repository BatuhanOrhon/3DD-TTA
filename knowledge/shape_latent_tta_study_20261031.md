# Bounded shape latent: initial test plan — 2026-10-31

[User report/Decision] User approves the reported proposal and asks to start
tests. This supersedes the earlier report-only/no-GPU restriction for this
scoped implementation and initial exploratory experiment. Worktree7bf0 and
branch integration/chamfer-guidance-variants-20261030 are preserved, including
all previous uncommitted research records. Evidence labels remain Paper,
Code, Run, User report, Inference, Open.

Spec: [research report](shape_latent_tta_research_20261031.md), followed by
the user's clarification: one clean-global query at t20, then a3-call guided
DDIM20/10/0 candidate; local N8/Background35. On fetch, origin advanced to
d96617f with accepted fixed N8 transfer. Its normal8/B35, original-style,
raw/eval/EMAoff policy and whole-policy comparison limits are retained.
No branch switch or remote whole-file overwrite is needed.

## Fixed initial scope

- ModelNet40-C severity5, frozen Point-MAE, B32, local8/Background35,
  gamma(local)=.03, eta(style)=.003, p=.95, original DDIM finalalphaTrue.
- Probability mean of clean2/4/6 + terminal8; B35 clean9/18/27 + terminal35.
  No spectral, semantic, sparse, weighted, entropy selection or classifier update.
- Five paired arms: `baseline` (unchanged N8 original decoder style),
  `updated` (no global prior/no bound, current style in clean/final decode),
  `anchor` (updated + every style update within .05||initial z||),
  `proposal` (anchor + one global clean prediction at t20),
  `guided3` (anchor + global DDIM20/10/0 and clean-estimate anchor gradient).
- Guided3 anchor loss is per-object mean squared displacement (128dims),
  summed over objects before differentiation; coefficient1, correction capped
  at .01||initial z|| per object per global step. No coefficient search.
  Final clean global estimate is projected to the same .05 radius.
- Common original z/h/local epsilon and post-prepare RNG; global Gaussian
  drawn with an independent generator keyed by seed+cache batch filename.
  Proposal/guided3 share that draw. Original h remains fixed; this tests the
  downstream global intervention, not conditional-h reencoding.
- Native technical smoke:32 explicitly registered IDs, Gaussian/Background,
  seed0; five arms. Then fixed all15 seed0 exploratory pilot on64 registered
  development IDs. No automatic parameter selection, multi-seed/full benchmark
  or Scan phase is included. Reused/exposed IDs are explicitly exploratory.
- Record indexed predictions, candidate logits, all paired cache identities,
  scheduler, global/local calls, anchor/displacement/correction diagnostics,
  frozen states, runtime and memory. Primary all15 PM; macro14/B separate.

Ruling: add the `updated` control — the accepted N8 host decodes original z,
whereas the proposal decodes updated style. Without this control a decoder
change could be misattributed to a bound/global prior. Cost: one extra arm.
Ruling: keep the first experiment bounded to a fixed pilot, not a tuning grid
or full confirmation — new-method accuracy/runtime is unknown. No unseen-data
claim follows; a later confirmation is a separate scientific decision.

## Ordered implementation and verification

1. Add `shape_latent_guidance.py`: per-object projection, one clean proposal,
   three-step differentiable clean-estimate guidance and compact diagnostics.
   Add an opt-in adapter to `tta.tta_reconstruct`; None preserves default math,
   decoding, probes and RNG. Test missing feature first, then CPU RED→GREEN:
   radius bound/zero-radius, no RNG advance, true global Jacobian, exact1/3
   calls, clean rather than noisy anchor, and unchanged default host parity.
2. Add `scripts/eval_shape_latent.py` using existing depth/shared artifact and
   classifier infrastructure. Fixed parameters/selection/arm guards; preserve
   every existing CLI. Test fixed config, technical/research split, malformed
   settings and real sampler/probe integration with synthetic CPU networks.
3. Seal current source+IDs over existing verified assets; allocate a uniquely
   owned A100 session only when ready. Use canonical ten-cell setup, explicit
   Drive consent if needed, native/checkpoint/source proof and five-arm smoke.
   Estimate the whole planned initial GPU phase from historical costs and smoke;
   obey the >60min approval threshold. A changed estimate must gate continuation.
4. Execute the registered pilot only after technical gates. Preserve failed
   attempts; no silent retry or replacement. Retrieve raw/cache/source bytes,
   verify Drive/local SHA, independently recount paired metrics and publish the
   necessary completed code/knowledge under standing integration authorization.

Knowledge target: this note, findings_log, open_questions and README.
Acceptance: finite bounded states, input/state/RNG/source controls, recovered
source bytes, exact coverage and validated complete/partial artifacts. Any
nonfinite state/gradient, source/asset mismatch or missing completion marker
stops the phase. Numerical benefit is not an acceptance prerequisite.

## Current execution state

[Code/Open] The initial planning-only state is superseded by the dated setup
records below. No teardown authorization; any allocated runtime must be
preserved until the user declares completion and permits closure.

### Implementation / setup started — 12:05 UTC

[Code/Run] Test-first implementation is in shape_latent_guidance.py, opt-in
tta.py adapter and scripts/eval_shape_latent.py. New21 contract tests pass;
full repository CPU suite500passed/1historical-fixture skip/373subtests,
six pre-existing warnings. Default sampler math/RNG parity, bounded updates,
clean-global Jacobian, time/call counts and worker guards pass. Independent
fresh-context review is in progress before GPU research. No accuracy yet.

[Run/Open] New owned session `tta-shape-latent-a100-20261031-r1` allocated;
canonical Cell1 is submitted. Owner=root/shape-latent, current worktree/branch;
sole primary Cell1 exec (session recorded in local command output), host-awake
exec38840 has7200s bound. Existing tta-depth-transfer-a100-20261031-r1 is peer
state: no execution/reuse/teardown. Endpoint pending server inventory.

[Decision] Entire initial research phase: five GB32 technical arms then five
all15/dev64 arms, seed0. Historical tunedN8 all15/dev128 worker194.115s;
scaled baseline five64 arms about485s before fresh encoding/global overhead.
Conservative research expectation15–30min with55min hard cap including
technical research, worker/source/artifact overhead. Environment setup is
tracked separately. No planned tuning/full benchmark/confirmation phase is
being divided to evade approval. If updated projection exceeds60min, stop
before continuation and seek explicit duration approval under the Colab skill.

[Code] Sealed334current source files over accepted native compatibility base
599b8dcb…8eef. Payload shape_assets_v1.zip2248317bytes SHA
55e2c1c14026d4da0ffef552eeaad3241d6321f009722a17bfbbd42fb2124e5b;
manifest1bd8d9066f089a996640cebe3743bc7715d9e06616230c8e7b3221bc4fc275b7.
Exact source bytes/identity and10canonical setup cell hashes are in
`result/preparation/shape_latent_tta_study_20261031/`. Development64 is the
fixed first64 original IDs of the exposed128 depth-development cohort;
technical32 is its first32, explicitly overlapping. No unseen-object claim.
Local outputs will use `result/modelnet40_c/3dd_bounded_shape_pilot/`; Drive
`MyDrive/thesis/result/shape_latent_tta_study_20261031/`. No teardown authorized.

### Review acceptance / live setup — 12:18 UTC

[Code/Run] Independent read-only reviewer found no demonstrated sampler bug.
Important validation gap fixed: two complete independent guided-DDIM oracles
cover signed updates and active/inactive per-object correction caps. Removing
the actual correction makes both tests fail; restored implementation passes.
Final CPU suite502passed/1historical fixture skip/373subtests, six pre-existing
warnings. Bare root collection also sweeps ignored snapshots; excluding result
still fails four vendor collections (hydra, emd_cuda, removed importlib API),
so only the repository tests/ suite is claimed passing. Exact logs are retained.
Separate operator stdout/nonzero/partial-line deadline selfchecks3/3 pass.
Final: fixed the guided3 validation gap with omission mutation RED→GREEN.
Final: minor (deferred): default-only original-style probe comment in tta.py;
the explicit adapter expression beneath it implements current-style probes.
Ruling: guided3 versus proposal is a combined chain-length/guidance contrast,
not an isolated guidance ablation; no sixth arm is added to this fixed pilot.

[Run/Open] Owned endpoint `gpu-a100-hm-ass1c0-1ulu15s7e5w42`, actual NVIDIA
A100-SXM4-80GB, same-VM Conda Cell1/reconnect pass. Two CLI connection losses
occurred in the prelude before Cell2/read-only inspection was sent. Exact
disconnected local CLI processes were interrupted; own setup kernel restarted
without stopping/reallocating VM. Read-only proof then showed no project/env
and no conda/research child. Cell2 submitted once after this proof, now creates
Python3.8 environment and installs dependencies. Only known fastrlock failure
may be deferred; no numerical experiment has begun. Drive consent and Cells3–10
remain pending. No peer operation.

[Code] Superseding source seal v2 includes the stronger tests, same runtime
code: payload2249362bytes SHA50dbf140f09267b416c30f072468392ab5f7ad00f70bf4ffae534288d75d77f4;
manifest e9a75f8b90374b96b6fd49bb59a0ba646747e7e98a01d01d4604ddf373fc5786;
source snapshot d7c360debc73597c7a8449f456e8aaad5580ae5694538a080bbf620e7282de3d.
Operator ZIP16449bytes SHAd8e25cc2960c601eca7d9611e24ab4200d7a40bfda3ad96dd11d679b2a3b6d60
registers10 files, exact commands/IDs, shared3300s research ledger and immutable
success/failure archives. Upload is in progress; execution/remote hashes pending.

### Drive authorization / setup recovery — 12:33 UTC

[User report/Run] User requested the actual Drive URL and then confirmed
authorization. The first interactive mount had already timed out; grant was
propagated but physical mount was not accepted from that exit0/error output.
Regular exec cannot answer dfs_ephemeral requests (the dedicated CLI hook is
only in automation.py); two redundant canonical mount calls were cancelled
with only own setup-kernel restarts, no research state or VM teardown.
Installed SDK uses REQUEST_TIMEOUT env default10s. Same CLI with
REQUEST_TIMEOUT=60 then authorized/mounted Drive successfully without new
user consent. All later calls use that process-local setting; account/CLI
installation and credential/session-state files are untouched.

[Code/Run/Open] Reused the accepted transfer study's verified-existing-mount
adapter for canonical Cell3: first assert real FUSE/MyDrive/assets, intercept
only its redundant drive.mount call, preserve/hash the complete original
cell and verify copied KNN wheel bytes. No fake/invalid receipt is accepted.
Sequential exact setup wrappers4–10 now run in one primary CLI connection;
each shell nonzero raises, native phase completion files are asserted and
per-cell progress markers persist. Only setup cells, no research notebook.
Actual Cell3/4–10 completion remains pending streamed markers.

Ruling: replace sorted-prefix draft64 selection before any research launch
with label-free SHA256 ranking within the exposed128 depth-development IDs,
then sort selected IDs. Smoke32 is separately hash-ranked within pilot64.
This avoids class/order prefix bias without using labels or changing count.
New operator v2 is16761bytes SHA6a02aabbb3085c777641c229016b5e702fce52d4f55141ca56e998a614807df9;
v1 is preserved and never executed. Asset source v2 remains unchanged.

### Setup accepted / model technical phase submitted — 13:02 UTC

[Run] Exact canonical setup markers and physical native phase assertions pass
through Cell10. Downloaded actual_setup_verification.json parses with status
passed: Python3.8.20, Torch2.1.2+cu121, NumPy1.21.2, A100-SXM4-80GB,
15severity5 arrays plus clean/labels and checkpoint identities. GPU FPS/gather
forward/backward, KNN, Chamfer and EMD setup checks pass. PointNet2 here is
the sampling CUDA library used by the Point-MAE evaluation FPS/gather path,
not an additional classifier. PVCNN is LION infrastructure.

[Run] Uploaded operator v2/source v2 verified physically. Fresh native bytes
were archived to Drive before restoring the accepted pinned four-backend
bindings, so no fresh-build evidence was overwritten. Strict CPU checkpoint
loading and no-build bootstrap pass with CUDA uninitialized and forward
prohibited. Actual receipt downloaded and accepted against the two sealed
SHA identities. Deployment archive1453627bytes SHA
fd1342a474238633cae781eb352d47db2f2701ea0206310556a1077d850187ca
at Drive phase-evidence/deployment_20261031T130046370029Z/deployment_cpu_evidence.zip.

[Run/Open] Model technical phase submitted via own primary exec to five fixed
Gaussian/Background32 arms, seed0, same cached initial states. No accepted
model result yet. Shared initial research cap3300s now begins; setup is separate.
Full64/all15 pilot remains behind technical/raw/cache/source acceptance.
All code/settings remain frozen; no accuracy-driven selection. Baseline makes
the initial cache whereas later arms replay, so primary timing is instrumented
fresh-versus-replay and cannot establish a matched end-to-end speed ratio.

### Administrative smoke stop / validated baseline reuse — 13:10 UTC

[Run/Code] Baseline completed both GB32 corruptions, frozen-state/RNG/paired
cache guards and all source/Drive checkpoints. Controller then incorrectly
required config.status=complete; the shared runner intentionally labels any
less-than15-corruption request partial even when every requested object is
complete. No numerical/model failure occurred. Original aggregate phase stays
failed, archive7007462bytes SHAe25e903e0d055367c2fa615865a558f2412f7c5e730df706c7dcdebf8fb9c061
physically verified Drive/local, raw config/CRC/source and two actual NumPy1
cache/RNG proofs independently accepted. Baseline24/32 Gaussian and21/32
Background correct under PM are technical counts only, not model selection.

[Code/Run] Versioned controller/analyzer guard now requires exact requested
scope: complete batches, zero max-batch truncation, valid status and full raw
shard coverage; partial is accepted only for the registered GB technical
subset. Regression uses the actual downloaded config: RED→GREEN; malformed
failed/truncated/missing-batch/full-pilot variants reject. Four operator tests
pass. Scientific worker/source snapshot e9a75f8b…5786 and all settings unchanged.

[Decision/Run/Open] Explicit fresh continuation reuses the accepted baseline
bytes and physical caches, executes only updated/anchor/proposal/guided3,
and records the original failed phase/reuse paths. Old directory/ledger/source
remain immutable. New ledger links the old identity and charges total elapsed
research time from13:02:35.879279UTC, including diagnosis/retrieval gaps;
55min cap is not reset. New8320byte continuation payload SHA
3879a2c388412bae70ce1081b37803dc3f153d41a9f89e2f6a8ef0189eb8dcf0.
Continuation submitted; no complete five-arm result or pilot yet.

### Five-arm technical acceptance / pilot launch — 13:15 UTC

[Run] All four remaining GB32 arms complete. The successful baseline was
reused byte-for-byte, not rerun. Full five-arm archive19218443bytes SHA
4e96bdd77293832645a9f9647ccf0b76a4bcf9742975ca69d4d68263b2507b3a
matches actual Drive/local bytes, CRC and safe extraction. Independent local
NumPy1 raw/cache/source/state/RNG and PM recount exactly equals remote
acceptance. Two physical cache files shared across all five arms verify.

| Technical arm | Gaussian PM% | Background PM% | Two-case macro% |
|---|---:|---:|---:|
| baseline |75.000|65.625|70.3125|
| updated |75.000|62.500|68.7500|
| anchor |75.000|62.500|68.7500|
| proposal |71.875|62.500|67.1875|
| guided3 |71.875|65.625|68.7500|

Only32original objects, each appearing in two corruption conditions; not64
independent objects. One event changes this macro by1.5625pp. No gain or
method selection follows. Guided3 exceeds proposal by one event but changes
both chain length and guidance. Original failed phase remains preserved.

[Run/Open] Same frozen64/all15/seed0/five-arm pilot submitted, no parameter
change after these scores. Technical continuation spent572.989s total elapsed
since original13:02:35.879 launch, including diagnosis. Pilot start charges
the intervening retrieval/review too; whole3300s cap persists. Historical128-ID
cost and measured35s/smoke-arm project pilot about10–20min, within remaining
budget. No multi-seed/full benchmark, Scan or extra timing phase is added.

### Pilot complete / independently accepted — 13:27 UTC

[Run] All five registered64/all15/seed0 jobs complete, no parameter change.
150batch shards,30physical caches and30proposal/guided3 global-noise SHA pairs
pass independent local NumPy1 source/state/RNG/PM recount. Local acceptance
exactly equals remote JSON. Pilot archive95791160bytes SHA
c2a58d37641e549cf3fc5403143c913483f92dd6bba912efe8d715f23d688ec3
passes CRC/safe extraction/local SHA and controller Drive verification.
Whole research ledger1337.696126s from original launch, including failure/gaps;
55min cap not reset/exceeded. [Results and limits](shape_latent_tta_results_20261031.md).

[Run/Inference] All15 PM baseline60.9375%, updated60.729167%, anchor60.3125%,
proposal60.3125%, guided3 60.416667%. No general gain. Proposal−anchor0;
guided3−anchor+.104167pp (one event), conditional paired-object95% interval
[-.729167,+.937500]. All comparison intervals include0. Not confirmation,
not960independent objects, not guidance-only ablation. Background unbounded
style drift median6.7474%, bounded about5%; bounding does not improve accuracy.
Original failed technical aggregate remains intact. No extraGPU phase.
Own runtime remains live; no teardown authorization.

### Derived archive / integration CPU verification — 13:36 UTC

[Run/Code] Final-derived ZIP158311bytes SHA
7b2310c398fc32636e02aafe19c317f43c0063e9f9411738bf14677ae47b8ad7
verified on real Drive and downloaded byte-identically. Derived archive
contains completed research records/evidence and the exact five source files;
later archive/integration receipts are separate, no self-containing hash claim.
Latest14d4644 integration base preserves promoted N8 reference/accepted transfer.
518CPUtest/412subtest pass with60NumPy1 tests passing separately; one historical
sealed-source fixture stays skipped. Temporary checkout's absolute tmp path
triggered existing native-fixture filtering; relocating that clean checkout
resolved both failures without code/test edits. Exact failed/successful logs
retained with compact hashes. Source worktree/branch and raw bytes preserved.
No new GPU job or runtime closure.
