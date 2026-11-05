# Shared baseline states — implementation, 2026-11-05

[Code/CPU/Open] Default-off CPU snapshot instrumentation for later frozen-state
H3/H1/H6/H7 diagnostics. This batch runs no GPU experiment, model replay,
new guidance, classifier-gradient computation or clean-oracle selection.
The initial H5 overlay at b766e61 was sealed separately and is unchanged.

## Opt-in contract

`scripts/diagnose_decoded_spectral.py --arm control --weight 0
--capture-shared-states` selects a thin `SharedTrajectoryObserver` subclass.
The flag is default-off, worker-revalidated and mutually exclusive with H5
and native-loop-trace captures. The default observer/hook path stays intact.
The explicitly selected mechanism supplement additionally permits count4 only
with this flag, guarded again in the worker. Legacy/H5/native-trace runs retain
count2/8; the four sorted mechanism IDs are supplied via the pinned selection
manifest. No count4 accuracy or H5 expansion is authorized by this CLI change.

Only the first and last existing clean PM checkpoints are captured:
normal N8 loops2/6, Background B35 loops9/27. The middle existing PM checkpoint
still runs and votes normally. All original SCD forwards/backward, prior/decoder/
classifier forwards, local/style updates and terminal handling remain unchanged.
There is no new production `tta.py` change in this batch.

## Exact stages and arrays

Each batch retains seven NPZ files plus incremental `shared_states.json`:

- `shared_state_loop00_frame.npz`: original `index`, corrupted normalization
  center and scale. Physical mapping is `((raw-center)/scale)*3.3885` then
  rotate `(-z,y,x)`; no clean registration or classifier normalization applies.
- `shared_state_loop02_pre_prior.npz` (also loop06; Background09/27):
  actual current `noisy_local`, optimized `conditioning_style`, and actual prior
  `prior_t` input from an opt-in forward pre-hook. Full RNG is sampled before
  copying these inputs and before the existing prior forward.
- `shared_state_loop02_pre_decoder.npz`: actual `predicted_clean_latent`, all
  8192 dimensions including the fourth per-point feature, and original encoded
  `original_decoder_style` z0. Full RNG is sampled before the existing PM decode.
- `shared_state_loop02_decoded.npz`: already computed physical `decoded_xyz`
  intercepted at the existing classifier callback, before its preprocessing.
  No decoder is rerun; corresponding files exist at the last selected PM point.

The original decoder-z and optimized prior condition are deliberately separate.
The pre-decoder RNG is archived even when equal to pre-prior RNG. Existing
SharedTrajectoryObserver decode uses detached-clone inputs under `no_grad`;
that branch is labeled in metadata and matters for later native replay.

Existing `reference.npz` provides exact prepared X and h0 XYZ anchor; its bytes/
SHA are bound in `shared_states.json`. The production paired preparation NPZ
is also bound by path/bytes/SHA and retains complete original h0/z0/epsilon.
The production recorder creates these references before observer.initial runs.

## RNG codec, metadata and failures

Full Python/NumPy/Torch CPU/all initialized CUDA-device RNG values are stored
without object arrays or pickled payloads. `rng_metadata` is a Unicode JSON
scalar for Python state, NumPy scalar fields, CUDA-device count and legacy hash;
`numpy_rng_keys`, `torch_rng_cpu`, `torch_rng_cuda_0` etc. are numeric arrays.
`load_shared_state_rng(archive)` reconstructs the state with `allow_pickle=False`
and verifies the existing `_rng_state_hash` before returning it. The legacy
hash implementation internally serializes its in-memory state for hashing;
there is no pickle load or pickled state in the artifact.

Each stage records original IDs, loop/timestep, source tta/driver file identities,
tensor hashes/shapes/dtypes/finite flags and original requires-grad flags, Unix
capture time, copy/serialization duration and RNG before/after writer guards.
The selected decoded/clean-latent identities are paired with existing observer
probe hashes. Completion requires both selected clean callbacks to succeed;
missing selected prior pre-hook fails rather than claiming complete coverage.

Each NPZ and incremental manifest is written before proceeding. A decoder failure
therefore retains pre-prior and pre-decoder evidence. Existing batch Drive shard
copy includes all successful batch files; a later failed batch retains partial
files in the existing final sealed/Drive bundle. Existing diagnostics links
`shared_states.json` by bytes/SHA. No separate GPU exporter or inference path.

## Evidence and limits

[CPU] Focused toy tests compare captured/plain runs: exact final local and style
states, PM logits, RNG and prior/decoder call counts. Saved tensors match the
ordinary probe hashes after later updates, demonstrating snapshot immutability.
Full CPU RNG restoration/hash is checked; synthetic cached-Gaussian and CUDA
byte-array encoding round-trip, and tampered CUDA bytes are rejected. Synthetic
CUDA bytes are not a claim of CUDA execution. Failed decoder evidence and
control/exclusivity/default CLI guards are covered. Python3.8 syntax is required.
An independently recorded stochastic prior/decoder entry also matches each
archived live RNG boundary; prior RNG use makes the two archived hashes differ.

[Limitation/Open] Detached CPU copies and serialization add synchronization and
can change native scheduling/allocator history in this instrumented run. These
captures are reusable states, not whole-process CUDA noninterference evidence
or timing benchmarks. Snapshot costs inside sampling are included in existing
reconstruction timing; initial CPU frame/source serialization occurs before the
batch timer. No actual native state archive or downstream H3/H1/H6/H7 replay
has been validated in this implementation batch.
