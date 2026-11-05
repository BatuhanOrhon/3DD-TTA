# Exact-input local-prior repeat — CPU preparation, 2026-11-05/06

[Code/CPU/Open] Selected after actual H0 traced first noise_pred drift in all
six repeat pairs, with exact loop1 h/optimized condition/t/RNG. This new bounded
native operator has not run on GPU in this implementation batch. No kernel or
production repair is claimed. [H0 findings](h0_native_trace_results_20261105.md).

Canonical implementation: `scripts/probe_local_prior_repeat.py` and three focused
CPU tests. Parent wrapper is separately archived under comprehensive operator_v1;
the immutable8095/source72e0eb4 deployment is not modified. Tests import the
canonical script, not an ignored external operator copy.

## Frozen actual operation

Use the first Gaussian44/91 and Background495/512 loop1 prior snapshots. Three
actual `lion.priors[1](x=h,t=prior_t,condition_input=style,clip_feat=None)` forwards
per case; exactly six network calls total, no backward, decoder, classifier,
Chamfer forward, new trajectory or guidance. Actual Gaussian t input71 and
Background341 come from the producer's stored float32 prior_t arrays.

Model loading reuses original raw/EMA-off no-build runtime, exact model-state,
native ELF/bootstrap/resolved-config/version/hardware/backend identities. All
modules must be eval, parameters frozen. h/style are grad-enabled leaf clones
inside gradient_probe; global grad remains enabled, as in the actual sampler.
No no_grad/toy/seed-only production fallback exists.

The first live paired cache's bytes/SHA binds numeric Python/NumPy/TorchCPU/
CUDA RNG. Computed state hash must equal cache rng_after, captured trajectory
initial RNG and first pre-prior RNG. Restore that full state immediately before
each actual forward and verify again. Both actual cache chains were already
proved on CPU using old accepted zero-repeat byte-identical caches. Missing or
unmatched state fails; it is never promoted as an exact-RNG witness.

## Outputs, optional localization and limits

Every actual network output is stored as numeric `repeat{0,1,2}_output.npz`,
finite checked, with tensor hashes, input hashes/leaf+grad flags, full-RNG hash
boundary and pairwise maxabs/per-object RMS/relativeL2. The first new result is
also compared descriptively with the archived producer noise_pred. No new
numeric materiality or equivalence threshold is introduced.

Leaf pre/post hooks run INSIDE those six forwards, without an extra model call.
Pre-hooks hash actual tensor/scalar inputs BEFORE possible in-place mutation;
post-hooks hash outputs. Up to64MiB of CPU first-repeat outputs are retained
for full numeric differences on subsequent repeats. No huge activation dump.
Unknown input objects, missing reference coverage, unequal input/RNG hashes or
hash-only changes do not identify an exact-input numeric module candidate.
These are module/network brackets, not proof about a particular CUDA kernel.

Hooks copy/hash and synchronize native work; allocator/scheduling history can
change. There is no whole-process CUDA noninterference or speed claim. A
within-process repeat can differ from cross-process behavior even at matched
observable states; absence of reproduced drift does not repair old H0 failures.

RNG/buffer isolation preserves caller state, original inputs are rehashed, and
model state is checked after completion. Top-level report explicitly records
entry RNG before/after/restored, model unchanged, and source/native guards.
Per-repeat output/leaf trace/partial summaries persist before proceeding.

## Parent execution contract

Upload the canonical script, `run_prior_repeat.py` and
`prior_repeat_input_selection.json` together to a separate operator directory.
Parent invokes wrapper through runpy; child uses pinned Python3.8 environment,
820s timeout plus20s kill fallback, within the selected whole-phase900s cap.
The wrapper saves original operator sources, logs, outputs and every partial
failure to a ZIP with manifest/CRC, copies it to canonical Drive and verifies
byte/SHA equality. Parent requires status=complete, model_unchanged=true and
entry_rng_restored=true; mutation flags cannot pass on a zero child exit code.

[CPU] Three tests cover exact numeric cache/RNG chain and tamper rejection,
three-only forwards with actual leaf/global-grad contracts, immutable input/RNG/
buffer state, and pre-hook handling of in-place modules plus exact-input toy
module drift detection. Combined local-prior/H0 tests pass9tests; Python3.8
grammar and whitespace checks pass. These CPU toys test instrumentation only;
actual LION/native GPU compatibility remains the parent's next bounded phase.
