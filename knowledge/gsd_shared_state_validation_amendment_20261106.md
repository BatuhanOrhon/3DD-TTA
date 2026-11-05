# Shared-state probe: anchored-spectrum validation amendment — 2026-11-06

[Run/CPU/Code/Open] Narrow pre-GPU validation amendment authorized by the parent
after actual H5 archive inspection. The scientific source overlay
`809573…b713`, active GPU assets, capture producer and original
`scripts/probe_gsd_shared_states.py` remain untouched. The amendment is a
separately archived operator:

`result/preparation/gsd_comprehensive_20261105/operator_v1/probe_shared_states_validation_v2.py`.

## Evidence and reason

[Run/CPU] The accepted H5 raw package is 1,391,073,853 bytes, SHA256
`ee58f23c92546858b7c82220e5c353f4aae68a6997df8e69289e3541dadbe105`.
Its initial CPU analysis encountered24 LiDAR spectra (eight original objects ×
three fresh-process repeats) with174 tiny adjacent normalized-eigenvalue
reversals; maximum backward step **3.5762786865234375e-7**. All these archived
spectra are finite/nonnegative. The already-frozen normalized anchored-block
tolerances are approximately.0025–.0028, and the read-only audit found **no
reversal crossing an existing anchored block**. No raw q/U/order/hash was
changed. Evidence: parent
`result/preparation/gsd_h5_reliability_20261105/downloaded/acceptance_v1/q_order_anchored_block_audit.json`
and `knowledge/h5_reliability_results_20261105.md`.

[Inference] Strict adjacent monotonicity is stronger than the existing
unresolved-eigenspace rule and can incorrectly block the planned shared-state
probe on this native output. This is a validator mismatch, not evidence for
sorting the eigenpairs, changing the graph/filter/bands, expanding a tolerance
or relaxing any forward/finite-difference gate. Actual H3/mechanism results
remain unrun by this implementation task.

## Exact amendment

[Code] Only `band_bounds` and loader/report wiring in `main` differ from the
frozen script's function ASTs. `band_bounds` retains the exact original
raw-L-normalized tolerance:

`atol=max(config.eigenspace_atol/mean_degree,
          2*eps*n_active*||L_normalized||_inf)`.

It scans blocks in the existing raw q order, anchoring each block once at its
first q and using `atol+rtol*abs(anchor)`. The original one-sided forward block
rule is followed by the full check
`abs(q[block]-anchor)<=block_tolerance` for every member. Tiny reversals inside
that same unresolved block are valid; material/backward tails outside it fail.
It neither sorts q/U nor chains successively close values into a wider block.
The actual production low rank and the original requested-high-boundary
selection/formula remain unchanged. No gradient, objective, scheduler, update,
RNG, band perturbation, branch or numeric gate changes.

[Code] The standalone operator resolves model `REPO` from
`--asset-root/source_snapshot`, and requires its source helper to have imported
from that immutable directory. Parent launches a fresh process with
`PYTHONPATH=ASSETS/source_snapshot`. All existing capture executed-source,
native, model-state, dataset/input/frame/state/RNG and manifest checks continue
to validate against those original pinned files. Runtime metadata records the
actual standalone `__file__` bytes/SHA as `executed_operator`, separately from
`captured_probe_source` and imported model-helper identities. The standalone
operator is not silently substituted into the manifest's original source entry.
There is no monkeypatch, active-worker change or asset rewrite.

Invocation has the same CLI as the original probe:

```sh
PYTHONPATH=/absolute/ASSETS/source_snapshot \
  /pinned/gpu/python /absolute/operator_v1/probe_shared_states_validation_v2.py \
  --capture-root /absolute/frozen_shared_capture \
  --capture-config-sha256 CAPTURE_CONFIG_SHA256 \
  --asset-root /absolute/ASSETS \
  --asset-manifest-sha256 ASSET_MANIFEST_SHA256 \
  --output /absolute/fresh_mechanism_probe_directory
```

[Code] Parent archives the amended operator and its unified diff against the
original source as separate operator evidence. The existing immutable overlay
and any active H0 workers are preserved. Any actual GPU execution follows
independent review and the parent's phase/source receipts.

## Verification and limits

[CPU] Four new focused tests in `tests/test_shared_probe_validation_v2.py` pass.
They reproduce an exact float32 lambda≈1 reversal of3eps (3.5762786865234375e-7):
the old strict validator rejects it, the amended validator accepts it without
changing tensor bytes or high-band selection. Material tails and cross-block
backward movement still fail. Multiple already-sorted float32/float64 and
zero-block spectra return exactly the original output dictionaries. AST checks
prove all other function bodies and all frozen numeric constants are unchanged;
a loader test rejects the wrong model-source root before loading any model.
Both amended files parse with Python3.8 grammar and compile locally.

[Code/Review] Independent H4/math review accepted the anchored-block validator
and explicit loader/provenance separation, with no blocker. The reviewer
independently reran all four amendment tests successfully (0.031s) and confirmed
that low/high selection, q/U ordering and FD/forward constants are unchanged.

[CPU] All nine original shared-state focused tests also pass in the persistent
parent CPU environment, including the repaired actual next-prior/PM-decoder
flags, common RNG ordering, H6 norm/feature/style preservation and sampler-step
arithmetic. Whitespace checks pass. Command:

```sh
result/preparation/gsd_comprehensive_20261105/cpu_venv/bin/python -m unittest discover -s tests -p test_shared_probe_validation_v2.py -v
result/preparation/gsd_comprehensive_20261105/cpu_venv/bin/python -m unittest discover -s tests -p test_gsd_shared_state_probes.py -v
```

[Open] No local model-weight forward, GPU/Colab call, full suite, source-overlay
reseal or commit was performed. This amendment does not establish a valid
native derivative, decoder band transfer, applied accuracy gain or deterministic
sampler. Existing max5e-4/RMS1e-5 forward gates, joint directional FD scope and
all unavailable/failed evidence rules remain binding. The actual common-state
GPU diagnostic may still reject its native forward/gradient gates.

## Canonical publication repair after runtime-source freeze

[Code/CPU] Parent reviewed and copied only the executed v2 `band_bounds` function into canonical `scripts/probe_gsd_shared_states.py`; no loader/operator duplication is needed in production source. The immutable8095 runtime overlay remains byte-identical. A regression in existing `tests/test_gsd_shared_state_probes.py` verifies native3eps reversal acceptance without q reordering and rejection of material/backward tails. Ten focused probe checks pass;29 focused H5/bootstrap/native/probe checks pass. The earlier four standalone-operator comparison tests remain in the ignored evidence archive `publication_validation_archive_v1/`, rather than publishing tests that require an ignored temporary operator. This canonical repair changes no graph, bandselection, scientificconstant or numericalgate.
