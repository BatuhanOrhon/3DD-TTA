# CloudFixer-only diffusion: gradient-free SCD, PM and resumable search

2026-11-14 — approved design; implementation authorized by the user.
Worktree: `/Users/batuhan/.codex/worktrees/0e7f/3DD-TTA`.
Branch: `research/cloudfixer-transfer-20261114`; current base `d6b9469`.

## User requirements

Use only the models and preprocessing in the CloudFixer TTA pipeline. Do not
load LION, its VAE/encoder/decoder, either prior or latent/style state. Replace
CloudFixer's geometric optimization with our gradient-free SCD correction during
diffusion and Probability Mean. Identify transfer difficulties before execution.
Search suitable parameters rather than assume eight steps. Save incremental
results to Drive and resume after interruption without repeating saved work.

## Approach and scope

The user has selected the physical-space approach: the author XYZ denoiser,
with an implementer-proposed Point-MAE comparison using the author implementation/
configuration and the same source weights for our sampler and the original
CloudFixer reference. Point-MAE is already the project's primary comparator;
the user has not selected a new head explicitly. Author commit:
`f043e24414fcf4165fdc892abff21215c9989141`.

Putting author weights into LION violates the request and its tensor contract.
Applying PM to repeated original CloudFixer optimizations tests a different
method; independent-view voting remains a separate reference. Implement one
XYZ reverse-diffusion trajectory with our correction and temporal PM.
CloudFixer-O, new training and spectral/PxP variants are outside this batch.
The existing LION baseline and peer CloudFixer jobs remain unchanged.

Only shared geometric algebra may be reused. Move `proximal_chamfer` unchanged
to a small model-independent module, retaining its old import/re-export for
existing callers. The new path must have no LION imports or model assets.

## Transfer assessment and sampler contract

[Code/Inference] There is no fundamental obstacle to epsilon prediction,
gradient-free coordinate correction or averaging frozen classifier probabilities.
Their accuracy and cost on this prior are unmeasured. CloudFixer's original
optimizer is not a reverse trajectory, so this is a new sampler, not a drop-in
replacement of a function's hyperparameters.

Use the author's polynomial_2 schedule, T=500 and precision1e-5 unchanged.
Choose an integer initial noise index K and a strictly decreasing integer grid
of N reverse updates ending at0, requiring4<=N<=K. Declare every actual index.
Initialize `xK=alphaK*x+sigmaK*epsilon` with one recorded standard Gaussian draw.
Construct grid index i as `round(K*(N-i)/N)`, i=0..N, using Python's declared
round-to-even rule; validate strict decrease and endpoints. Compare the loaded
`gamma.gamma` table with the intended polynomial500 table: strict checkpoint
loading can overwrite this schedule parameter, so constructor arguments alone
do not establish the executed schedule.
Use the same index for noise coefficients and Transformer conditioning;
author gamma rounds normalized t while phi truncates, so test consistency.

For a frozen epsilon estimate, compute `q=(xt-sigma_t*epsilon)/alpha_t`.
Apply the frozen bidirectional trimmed Chamfer quadratic solve to q against
the observed input in the author's unit_std/axis coordinate frame. Transport
the correction using `xs=alpha_s*q_corrected+sigma_s*epsilon`.
Here alpha is a signal amplitude; the LION helper's sqrt(cumulative-alpha)
and four-channel layout must not be copied. All model execution is no_grad;
no denoiser backward, rotation optimizer, displacement optimizer or model update.

Keep the author's complete point array/cardinality as denoiser input. Use its
mask/indices to identify valid observed anchors for SCD; masked duplicate or
unselected anchors must not multiply reverse-match weight. Predicted points
remain eligible for forward matching. Declare separate retained counts for
the predicted and valid-reference sets; reject empty/nonfinite inputs.
This mask-aware multiset treatment is an explicit transfer choice, not proven
equivalent to the historical LION objective. Retain mask/cardinality diagnostics.
Removing CloudFixer's explicit rotation optimizer also removes its dedicated
alignment mechanism. SCD anchoring to rotated observed XYZ may resist useful
rotation; tuning N/K/tau cannot be assumed to recover that mechanism. This is
an untested risk to disclose, not a reason to add a rotation optimizer silently.

At quarter/half/three-quarter reverse-update counts, collect predicted-clean q
**before that update's proximal correction**, matching the existing PM timing.
After the last update, use the author's deterministic clean estimate at t=0
as the terminal candidate. Its alpha0 is not1: raw z0 is not the clean output.
Thus N reverse updates require **N+1 denoiser calls**, including terminal cleanup.
Count this additional call in every comparison. Isolate classifier RNG from
sampling; classifier weights, buffers, grouping, axis/unit-ball convention and
all-token inference remain frozen and identical across methods.

Save four class-logit vectors, timestep/loop identities and terminal logits.
PM is the equal mean of their temperature1 softmax probabilities; no LM,
entropy selection, class-label feedback or corruption-specific routing.
Terminal and PM accuracy can be recovered from the same trace, but that trace's
timing is not standalone terminal-only latency. Measure terminal-only separately
when reporting a terminal speed claim. Tau0 bypasses the Chamfer computation.

## Proposed bounded grid

These numerical choices are proposals by the implementer, not user-selected optima.

| Parameter | First grid |
|---|---|
| Reverse updates N |4,8,16,32 |
| Initial index K on500-step grid |40,80,160 (normalized .08,.16,.32) |
| Proximal tau |0,.03,.1,.3 |
| Retained fraction |.95 initially |
| PM locations |quarter/half/three-quarter updates plus terminal |

This is48 trajectories per batch, not96: terminal and PM share reconstruction.
Tau0 supplies12 ordinary-denoising controls. Source-only and original CloudFixer
vote1/30-update are fixed reference conditions on the same classifier/data.
Ordinary denoising with PM also separates readout benefit from SCD benefit.

First freeze disjoint8-smoke/128-development/256-confirmation original-object IDs
using a recorded label-free seed20261114; use each cohort across all15/s5
corruptions. Batch32, inference seed0 for development, seeds0/1/2 for confirmation.
These are prospective object-disjoint cohorts from an already exposed benchmark,
not an untouched official validation set. Do not select configurations per
corruption or present a small development subset as full-benchmark accuracy.

Rank by development macro15 accuracy; break exact ties by measured pipeline
time, then a stable config ID. Preserve the complete accuracy/time frontier.
Tau0 is an eligible winner: if it wins, report no demonstrated SCD benefit
instead of selecting a worse positive-tau configuration to favor the method.
Freeze the selected configuration before confirmation; do not retune on it.
An edge winner flags that the grid has not bracketed an optimum. Wider/local
refinement must be explicitly registered before its results are examined.
A finite grid establishes its best tested configuration, not global optimality.

Prepare both the coarse screen and confirmation commands. Initially execute
technical validation and the coarse screen; report its complete results and
remaining search boundary before launching refinement or confirmation. No
automatic full2468-object benchmark is part of this first implementation.

## Drive persistence and recovery

The recoverable unit is a **completed inference batch**, not a diffusion step.
Mid-batch loss reruns only that uncommitted batch. Store each successful unit
as an immutable artifact with trial/config/corruption/seed/original IDs, logits,
candidate/terminal predictions, exact counts, synchronized time/memory,
source/model/data/config hashes, actual grid and method-independent RNG key.
Archive setup/source/environment and each failed attempt separately.

Bind campaign identity to the source snapshot, author source/patch, checkpoints,
all data/label hashes, cohort, batch size and numerical policy. A semantic
change starts a new campaign; a different local restoration path alone does not.
Compute deterministic per-batch seeds from inference seed/corruption/original
IDs, independent of trial order, process-global RNG and resume position.
Common-noise comparisons use the same initial Gaussian draws where applicable;
equal seeds do not make original CF's random-noise optimizer the same algorithm.
Do not promise bitwise equality across hardware/native libraries.

The old peer campaign observed missing files after a new-VM Drive read despite
same-VM mounted-path checks. Therefore use **Drive API-backed binary upload and
download verification** for completed units, not just FUSE write/read as proof.
Use the user's Colab-authenticated account, never copied tokens or embedded
credentials; actual Google consent remains user-operated when required.
Each upload yields a Drive file ID and server size/checksum; download through
the API and verify local SHA256 before publishing its immutable commit receipt.
Keep a local VM copy and retrieve closed evidence to the Mac through Colab CLI.

On resume, list paginated commit receipts from the campaign's Drive folder,
verify identity/content/coverage, reconstruct the completed set and execute only
missing batches. Partial uploads and files without a commit receipt remain
uncommitted; preserve them as attempts. Handle an interrupted acknowledgement
idempotently: matching repeated records count once, conflicting content stops
the campaign. Never trust a mutable last-index file or double-count duplicates.
Use one owned worker; never start another while its liveness is unknown.

Refuse resume after corrupt/missing committed content or identity mismatch.
Drive/auth/quota failures stop new scientific computation and preserve the
pending local result for retry; they do not silently advance progress.
Derive progress CSVs and final standard seven-file reports from verified units.
An interrupted grid remains partial, not a completed selection.

## Runtime and verification

No GPU has been allocated for this design. After design/implementation checks,
use a distinct owned Colab CLI runtime and the canonical ten setup cells.
Canonical setup may build LION-related dependencies because repository policy
requires it; the research worker itself must never load or invoke LION.
Protect peer runtimes and keep separate teardown authorization.

Initial rough scientific estimate is3–8hours for the screen; point count,
Transformer attention and per-batch cloud uploads make this uncertain. Measure
actual B32 timings by point count in the smoke and publish the whole-phase
projection/cost estimate before launching the grid. Proposed first-attempt
whole-phase cap10hours includes setup, validation, search and archival. Persist
spent-time accounting across interruptions/recovery VMs; a new process or VM
does not reset the campaign budget. Preserve committed progress and stop at
the cap. No duration permission gate is added.
Quota/auth/native/data/model/nonfinite/OOM/identity failures stop their phase.

Required checks: polynomial/epsilon/transport/endpoints with synthetic witnesses;
zero-tau no-Chamfer and frozen-model invariants; duplicate-mask/variable-point
cases; PM versus saved logits; interruption before/during/after upload/receipt;
same-batch restart, changed config rejection, corrupted records, duplicate
receipts, partial grids and tail batches. Test with CPU fixtures first, then
real author checkpoint/native smoke; fixture success is not model evidence.
On GPU, run fixed-input repeat checks and report any native numerical floor.

Implementation boundaries: model-independent quadratic; XYZ sampler; one
campaign CLI with preview/register/run/resume/report; Drive unit store; focused
tests and a Colab phase/handoff using existing setup. Reuse existing result
schemas rather than create another reporting format. No general job scheduler,
new classifier training, automatic runtime teardown or unrelated refactor.

## Sources

- [Prior preparation and exact source receipt](../../../knowledge/cloudfixer_transfer_preparation_20261114.md).
- [CloudFixer model/training audit](../../../knowledge/cloudfixer_denoiser_transfer_audit_20261112.md).
- [Drive uploads](https://developers.google.com/workspace/drive/api/guides/manage-uploads).
- [Drive binary-file metadata](https://developers.google.com/workspace/drive/api/reference/rest/v3/files).
- [Official Colab external-data notebook](https://colab.research.google.com/notebooks/io.ipynb).

## Builder review

Independent source review found no conceptual blocker. Incorporated clarifications:
explicit initial noising/integer grid, checkpoint gamma verification, eligible
tau0 winner, persistent recovery budget and the loss of explicit rotation repair.
N+1 terminal accounting and reference-only mask use were checked against the
official source. Point-MAE remains an implementer proposal. No real model/Drive
test or new scientific outcome follows from this document review.

## 2026-11-14 — user-directed storage revision

The user explicitly abandons API authorization and selects the existing Colab
Drive mount. This supersedes API-only requirements above. Keep the registered
science, author source/models, cohorts, grid and whole-campaign budget unchanged.
Use a single-writer mounted folder with temporary write, fsync, atomic rename,
byte/SHA readback and payload-before-receipt commits; ignore orphan temporary
files and validate every completed payload on recovery. DriveFS readback may
be cached and does not establish independent server persistence. Preserve local
run artifacts and download completed phase archives to Mac as a second copy.
No further account-consent step or model rewrite is required for this revision.

### Released gamma table compatibility

Actual author checkpoint gamma differs from freshly constructed polynomial500
by up to0.0007076263 log-SNR. All501 values exactly match the fixed point of the
author's EMA(beta=.9999), including its averaging of frozen gamma. Preserve the
released table; accept native or reproduced EMA values at the original strict
tolerance, record the mode, and reject other schedules. This supersedes an
assumption of byte-identical fresh polynomial construction, not the native
coefficient/sampler contract. See the dated execution evidence for full hashes.

### Evidence-driven repeat policy after actual GPU calibration

Actual author CF has matching input/CPU+CUDA RNG hashes but different outputs;
strict torch names KNearestNeighborBackwardCuda as lacking deterministic support.
Strict torch makes source/proximal repeated logits identical on all8smoke objects
and all15corruptions. Preserve author CF algorithm/native kernels. For continued
comparison, require strict source/proximal repeats and matching input/RNG for
all arms, while record CF repeat deltas and run3operational CF repetitions with
the same seed. Report individual accuracies and mean/sampleSD; do not ensemble
CF logits. This explicitly supersedes the all-arms bitwise-repeat prerequisite.
The author native variance is a reported limitation, not a new gradient rule.

The48proximal grid cells and fixed cohorts/seeds remain unchanged. Development
now has source+3CF controls+48grid cells =52configurations/3120batch units.
Profile keeps one CF timing representative (60units total); smoke includes all
three native controls (360units). Torch deterministic algorithms apply to
source/proximal, while original CF uses its unmodified native backward; set
CUBLAS_WORKSPACE_CONFIG=:4096:8 before model initialization. Record the policy
in source/campaign/gate metadata. Whole-phase budget, hash recovery and measured
B32 forecast remain mandatory; no confirmation/full benchmark is added.


## 2026-11-14 duration-only revision after measured B32 profile

User explicitly requests continuing GPU testing without a new approval even
when the pilot is long; incremental mounted Drive commits and recovery remain
required. Measured48-setting grid forecast is8.125h including25% margin.
The internal total bound is24h instead of10h, preserving all elapsed/setup/
retry receipts and per-operation deadlines. Grid/cohorts/seeds/models/readouts
are unchanged. Budget-only source migration validates exact scientific
identity and adopts already passed smoke/profile gates with explicit parent
task and gate-SHA lineage; it does not claim a new gate run. No automatic
confirmation/full benchmark or teardown. See current execution note.
