# Chamfer + Probability Mean baseline promotion

**Goal / user specification:** Promote the accepted N8/.003/.03/.95 policy
to the thesis operational baseline and publish to `thesis-integration`.
The user supplied the exact policy and requested implementation; no design
choice or further GPU experiment needs approval.

**Architecture:** A named baseline entrypoint reuses the accepted frozen
transfer configuration and reconstruction worker. Original reproduction,
depth-search and calibrated variant entrypoints retain their defaults.
Changing their historical defaults would invalidate existing comparators;
a documentation-only promotion would leave terminal predictions in the main
CSV. The new entrypoint therefore reports probability mean in its standard
CSV/prediction field and retains terminal outputs as diagnostics.

**Constraints:** normal8/Background35; eta=.003 conditioning style,
gamma=.03 local latent, p=.95 retained fraction; B32; original decode style;
DDIM100/final-alpha1; raw/eval LION/EMA off; no spectral/semantic losses.
Equal T=1 probability mean: clean2/4/6+terminal8, clean9/18/27+terminal35.
Explicit selected original IDs, asset SHA and existing failure/archive
guards remain required. No new GPU run or runtime teardown.

**Files / acceptance:**

- [x] Extend `tests/test_chamfer_depth.py` with a native-free baseline CLI
  preview, rejected overrides/tampered worker payloads, and real worker
  CSV/NPZ composition where terminal is wrong and probability mean is right.
- [x] Observe failures before implementation.
- [x] Add `scripts/eval_chamfer_baseline.py`; make narrowly scoped changes to
  `scripts/eval_chamfer_depth.py` validation and the shared worker's selected
  prediction reporting in `scripts/eval_multibackbone_guidance.py`.
- [x] Run targeted checks, then the CPU suite; verify old-study reporting,
  corruption horizon, selected-head independence and source recording.
- [x] Update root README and knowledge baseline/README/findings/open questions
  with promotion, exact command, evidence, limitations and superseded status.
- [ ] Review the diff; fetch latest target; normal fast-forward publication.

**Review focus:** reject changed rates/losses, mixed old/new heads, unlabelled
cohort scope, accidental terminal CSV, and changes to original trajectories.
Historical raw archives and frozen source evidence remain immutable.

**Review outcome:** Independent review found a worker replay guard gap and
two incorrect example asset paths. Both replay routes were reproduced with
failing tests before fixing. The baseline worker now requires its profile;
legacy depth rejects PM/baseline metadata. Example asset paths match the
asset manifest. No sampler/model/preprocessing changes or deferred findings.

**Final checks:** 495passed/61skipped/412subtests in the general CPU suite;
78passed in pinned NumPy1, covering60conditional skips; one optional historical
fixture unavailable. No new GPU inference; accepted GPU source remains4a5c137.
