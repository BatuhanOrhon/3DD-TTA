# Baseline native variation trace — 2026-11-05

[Code/CPU/Open] Opt-in instrumentation only. No native GPU trace has run in
this implementation batch; no nondeterministic kernel has been identified.
The separately sealed initial H5 overlay remains unchanged by these later edits.

## Motivation and boundary

[Run] Accepted old zero-repeat archives show equal prepared/initial state and
initial RNG, but unequal clean-latent hashes before the first decoder probe:
Gaussian loop2 and Background loop9. Background control/control repeat gates
failed3/3 (worst logit2.01245165; first decoded RMS0.03937847). Gaussian
passed3/3 with first decoded RMS below9.72e-8. Original-z decoder style matched.
See `evidence/gsd_zero_repeat_20261104/summary.json` and the sealed12rawZIPs
under `result/preparation/gsd_zero_repeat_20261104/downloaded/raw_runs/`.

[Inference/Open] Prior forward, scheduler arithmetic, SCD and its backward/
update remain candidate boundaries. Existing captures do not distinguish them.
Original decoder-z equality does not imply optimized prior-conditioning equality.
Chamfer backward's CUDA atomicAdd is a possible contributor, not a causal finding.

## Interface and exact captures

[Code] `tta_reconstruct(..., native_trace=None)` preserves the default path.
A callback receives `(loop_count, timestep, boundary, tensors)` for loops1..9,
or all loops if the trajectory has fewer than9. Only the original zero-weight
joint local/style SCD host accepts this callback: shape/style spectral guidance,
alternative Chamfer anchors and SCD normalization are rejected with tracing.

`scripts/diagnose_decoded_spectral.py --arm control --weight 0
--capture-native-trace` installs the CPU artifact writer. The flag is default-off,
is worker-revalidated, and cannot accompany `--capture-h5`.

Each completed loop has five snapshots, in execution order:

1. `prior_input`: current noisy local h, optimized `conditioning_style`, fixed
   encoded `original_decoder_style` z0, original h0 XYZ SCD anchor, actual prior t.
2. `scheduler`: existing prior `noise_pred`, scheduler `pred_original_sample`
   and `prev_sample`, captured before any PM decoder/classifier callback.
3. `scd`: existing unsorted native distances and NN indices in both directions;
   existing full sorted distances, retained sort indices, last retained cutoff,
   first excluded distance where available, and original scalar SCD sum.
   Cutoff gap can be reconstructed by subtracting the two archived distances.
4. `gradients`: actual noisy-local and conditioning-style `.grad` after the
   existing single backward, before the existing update.
5. `post_update`: resulting noisy local h and conditioning style, with the
   original gamma/eta arithmetic unchanged.

The trace makes no additional model, decoder, Chamfer forward or backward calls.
Tensor dtypes are preserved in raw arrays. No gradients are requested by the
writer; values are detached and copied to CPU. Missing tensors are explicitly
unavailable. Nonfinite raw evidence is retained, with finite=false and null norm.

## Artifacts, pairing and interpretation

[Code] Under each existing `batches/<corruption>/batchNNNN/` directory:

- `native_trace_loop01_prior_input.npz` through matching loop08 or09 boundary
  files: original `index` and named CPU arrays; no object pickle.
- `native_trace.json`: schema1, original IDs, loop limit, complete flag, ordered
  snapshot names/timesteps, raw file SHA/bytes, each tensor's legacy hash,
  dtype/shape, finite flag, float64 CPU norm/per-object norms, RNG before/after,
  and copy/serialization wall seconds.
- Existing `diagnostics.json.native_trace` identifies the JSON by SHA/bytes.

NPZ files are written at each boundary, followed by incremental JSON, so a later
native error retains prior evidence in the failed sealed bundle. All files enter
the existing per-batch Drive shard and final archive path; no separate exporter.
Normal N8 has40snapshots, Background B35 has45. The trace stops at loop9 and
does not bracket later-only divergence. A callback error can leave an unindexed
partial NPZ if serialization itself fails; bundle sealing still preserves files.

[Open] Compare equal cached initial states/source/native/backend/RNG across
fresh instrumented repeats. The earliest numeric divergence, including magnitude
and not hash alone, brackets the next investigation. Matching prior inputs with
different noise_pred points to forward computation; equal scheduler/SCD values
but different actual gradients points to backward; changed retained membership
or cutoff ties may explain sensitivity. This is localization, not proof of a
particular native kernel. No extra replay/FD/backward experiment is implemented.

[Limitation] CPU copies, hashing and disk serialization synchronize and slow
the instrumented trajectory. Their timing is included in reconstruction timing;
these runs are not speed measurements. GPU allocator/scheduling/history can
change, so CPU parity is not whole-process CUDA noninterference evidence.
Do not compare a traced trajectory to an untraced one as a causal method arm.
The original H5 phase uses its previously sealed source without this trace.

## CPU validation

[CPU] The synthetic native-free sampler checks exact final local/style equality,
RNG and model-buffer equality,11prior/11Chamfer calls in both modes, loop1..9 cap,
actual update equations, distinct original-z/optimized-condition semantics,
retained/cutoff membership, tensor hashes, default/invalid CLI guards, and failed
numeric/partial artifact retention. Existing frozen-parent sampler parity tests
also cover the default path. Python3.8 syntax is required. No CUDA, model-weight
or cross-process repeat validation is claimed by these tests.

[Code/CPU] Curated integration with peer lowband code additionally rejects `decoded_scd_weight!=1` when native_trace is enabled and retains all operational-control lowband/probe guards. The merged API rejection witness failed before this guard; final merged checks are recorded in the publication receipt. No native trace or new accuracy claim follows from CPU parity.
