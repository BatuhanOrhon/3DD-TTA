# Background with the common horizon implementation plan

> Execute inline with superpowers:executing-plans. User supplied the fixed policy and standing GPU authorization; no additional design/permission round.

**Goal:** Evaluate Background under the same N5 terminal / tuned N8+PM / cheap N8+PM policies as the accepted macro14, and derive tuned N8 terminal accuracy.

**Architecture:** Add explicit `background_uniform` phase to the existing speed runner. Pass an explicit original-reconstruction horizon through the existing geometry helper, retaining all previous defaults. Use a separate immutable asset/output/Drive root on the existing owned r5; preserve accepted macro14 and historical N35 artifacts.

**Tech stack:** Existing Python/PyTorch/Colab CLI, no dependency or native changes.

**Specification:** User request dated 2026-11-08: corruption identity cannot select a special Background horizon; use the normal policies for Background and recompute totals.

## Constraints and review focus

- Background severity5, all2468 IDs, seed0, one pass, B128/final36, 20 cases/7404 rows. Only this corruption reruns.
- Terminal N5 gamma/eta=.01/.01, one readout; tuned N8 gamma/eta=.03/.003, PM[2,4,6,terminal]; cheap N8/tau=.1 and same PM.
- Tuned terminal accuracy comes from the same final N8 readout; no invented terminal-only latency.
- No corruption aliasing: retain the true Background data/name; the phase explicitly fixes horizons.
- Unknown/invalid horizon overrides must fail before preprocessing; defaults retain N5/Background35 behavior.
- Preserve source/data/model/native identities, synchronous model gate, warmups, per-case Drive checkpoints and before/after model integrity.
- Estimated whole GPU phase17–22min. If actual VM is lost, preserve and await user; no replacement.
- This is a user-requested post-N35-result protocol revision, not an untouched confirmatory test or permission to erase earlier findings.

## Tasks

- [x] Write failing phase/parser/terminal-forwarding tests and explicit helper horizon/default/invalid tests; observe RED.
- [x] Implement the optional original horizon and `background_uniform` phase; run targeted then project tests; commit only scoped code/tests/plan.
- [x] Seal exact source overlay; reuse existing reviewed deployment/capture/launcher/runtime gate in a new root, validate CPU fixtures and pre-science archive before GPU.
- [x] Run the three actual GPU arms, derive tuned terminal accuracy, independently validate raw predictions/timings/provenance and sealed macro14 composition.
- [x] Record separate uniform-policy all15 results including N35 history, uncertainty/limitations; Drive roundtrip, curate shared integration, retain runtime.
