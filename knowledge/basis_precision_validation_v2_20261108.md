# Physical-basis precision validation v2 — 2026-11-08

[Run/Failed] R6 shard001 half0 completed its worker, but validator v1 rejected
`background/batch0001`. The immutable 957625938-byte ZIP has SHA-256
`a51c70bdfd743efce1a2b8b164b3c2ad456b6e7bc56abbbf5dc4aedd43fb5ea7`.
Its recorded FP32 batch loss is 8.811872482299805; the independent per-object
FP64 basis energies sum to 8.81187358757228. The absolute difference,
1.10527247443315e-6, fails v1's 1e-6 rule. All 30 first-decoded hashes match
the first probes; prepared-reference and basis hashes match. The failed raw
archive, v1 source, rejection and existing receipts remain unchanged.

[Code/Inference] The producer constructs FP32 dense filters and computes an
ordered FP32 scalar sum over objects. V1 instead sums independent basis energies
in FP64. Casting each independent FP64 object energy to FP32 and replaying the
producer's scalar addition order reproduces this failing recorded scalar exactly.
This establishes an aggregate arithmetic mismatch; actual native per-object
loss terms were not logged, so it does not prove each native term or gradient
is bit-identical to the independent calculation.

[Decision/Code] A separate, explicitly post-hoc validator `basis-fp32-v2` now
uses `fp64_per_object_energy_then_producer_dtype_left_fold`. The per-object
`||U.T E||²/(3*2048)` calculation remains FP64. Only the reference scalar
reduction is replayed in the actual producer dtype and archived object order;
FP64 producers retain FP64 reduction. The numeric tolerance remains 1e-6.
**The ACK acceptance rule and accepted set change post-hoc; v1 did not pass.**
There is no old/new minimum, tolerance increase, batch mean, clamp, rank crop
or special case for Background or any object.

[Code] Each v2 projection records schema2, exact arithmetic/version, producer
dtype, object order/count, replay sum/error and tolerance, plus the explicit
v1 FP64 left-fold sum, error and `legacy_gate_passed`. The legacy left fold is
explicit rather than Python's version-dependent built-in `sum`. Receipt details
retain every projection and the validator source identity. V2 also checks the
first decoded NPZ hash against the first probe and requires first/reference/basis
dtypes to agree. Sampling, gradients, source918780b, frozen alpha, checkpoints,
native binaries, cohort and raw archives are unchanged.

[Test] TDD preserved the original failing RED run (14 failures/4 inherited passes)
and a second RED run for the Python3.12 FP64-sum and dtype guards. Final focused
verification passed21 tests, including the actual32-term fixture, high dynamic
range/order alternatives, FP64 contracts, B1/9/32/41, altered/nonfinite losses,
wrong first witnesses outside the basis support, corrupted/resealed basis
energies, frames, hashes and dtypes. Existing r6 consumer33 tests plus project
`tests/` passed859 tests, skipped2, with562 subtests and7 existing warnings.
Independent review found no blocker and independently passed21 tests and
Python3.8 syntax checks.

[Run/CPU] The complete v2 `load_run` qualified all90 actual spectral cases
(shard000 half0/M100 and shard001 half0), plus30 control cases. Two existing
cache archives/60 prepared states passed the unchanged cache checks after
preflighting every local member as already byte-identical. No old raw file or
receipt was installed or changed. Maximum v2 error is 9.5367431640625e-7.
Only shard001 half0 Background1 retains `legacy_gate_passed=false`. Running
v1 again on that archive reproduced the original rejection.

[Evidence] New code, tests, RED/GREEN logs and qualification-only sidecars are in
`result/preparation/gsd_lowband_quarter_fast_20261108_r6_basis_fp32_v2/`.
The v1 acceptor SHA is
`c27c8eae4213695d2bd20d17417eee2048547106e5da4e2a54ef887391d9ec58`;
the 27478-byte v2 acceptor SHA is
`7f082803c10ff55300ed993bb1b32606e574ffcffc032fb0369609719b9e50bc`.
`actual_qualification_summary.json` binds the six archive identities and keeps
the v1 failure; each inference sidecar retains complete v1/v2 projection data.

[Open] This is CPU qualification and an independently reviewed administrative
validator change, not deployment, an ACK, or a completed617 result. Future
cases can still fail the strict v2 comparator and must remain preserved.
The unresolved native-repeat caveat remains. No GPU, CLI, Git, scientific retry,
source mutation or acceptance change was performed in this implementation batch.
