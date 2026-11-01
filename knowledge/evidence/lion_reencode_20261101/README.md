# LION two-pass reencoding pilot evidence

Accepted ModelNet40-C severity5 pilot:64 previously exposed objects/all15/seed0,
preceded by32-object Gaussian/Background technical gate. Second N3 PM does not
improve accuracy over its same-run first PM. See
[full results](../../lion_reencode_results_20261101.md).

The acceptance files preserve independent local raw/cache/posterior/RNG/source
and prediction recounts. Per-corruption and bootstrap inputs are compact derived
evidence. Clean-z bridge diagnostics use archived clean targets posthoc only.
`analyze_phase.py` and `diagnose_bridge.py` retain their assert-based self-checks.
`archive_roundtrip.json` verifies the Drive final capsule and all75 manifested
members; its snapshot precedes final publication metadata. Full immutable
raw/source/posterior archives remain under result/preparation/lion_reencode_20261101
and Drive thesis/result/lion_reencode_20261101, outside Git.
