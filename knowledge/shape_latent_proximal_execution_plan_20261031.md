# Global proximal shape pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.

**Goal:** Test the user-authorized global shape diffusion guidance hypothesis on the currently assigned replacement VM.
**Architecture:** Five opt-in arms use immutable imported prior-study state/proxy;
three global DDIM steps optionally apply a fixed quadratic proximal clean correction.
**Tech Stack:** existing Torch/NumPy/diffusers, Point-MAE original N8/B35 host, Colab CLI.
**Spec:** [design and authority](shape_latent_proximal_pilot_20261031.md).

## Global Constraints

Prior study already published1bacabf. Own replacement runtime
tta-shape-proximal-a100-20261031-r1; old refresh VM disappeared without agent teardown.
See [current handoff and exact setup state](shape_latent_proximal_handoff_20261031.md).
32exposed IDs,
seed0,B32,G/B/Rotation/LiDAR; technicalGB only. Whole new GPU cap2400s,
expected15–25min. Original/mean/prior3/proxy3/oracle3, explicit oracle isolation.
Global20/10/0, half-target correction, no radius/grid. h0/noise/RNG/anchor fixed.

## Review Focus

DDIM alpha_prev/finalalpha, actual target isolation, imported file/tensor/RNG
and old-source lineage, current-source proposal binding, own-z decoder/readout,
scope/partial status, model/source guards and immutable failure/archive behavior.

### Task1: Proposal and fixed runner

Create shape_latent_proximal.py, scripts/eval_shape_proximal.py and corresponding tests.
- [x] RED→GREEN analytic zero-prior3half-step solution, private-noise/RNG/control
  arms and oracle argument rejection; immutable reference load checks.
- [x] Fixed scope/clean path and five-arm h/noise/anchor routing checks;
  mispaired valid-cache versus proof rejection RED→GREEN.19CPUtests pass.

### Task2: Seal inputs/source, review and execute

Reuse existing accepted asset/CPU/phase operator patterns in the new ignored root.
- [x] Prepare first32IDs/four-corruption imported state/proof identities and
  separate clean oracle target files; seal newsource/data manifest.
- [x] Fresh whole-change reviewer, substantive fixes with regressions;4operator checks.
- [x] Allocate replacement A100; canonicalCell1/reconnect and Cell2 complete,
  only user-deferred fastrlock wheel failure. Uploadassetsv1/operatorv2 complete.
- [x] VerifyCell2 Python; realDrive mount; canonicalCell3–10/native/data/checkpoint gates.
- [x] Fix external install wrapper's stale session assertion; check against v2ZIPregistration.
- [x] StrictCPU deployment and realdiffusers CPUalgebra gate on replacement VM.
- [x] Current runtime GB32 technical gate, then four-corruption5arm pilot under2400s.
- [x] Prepare remaining four-corruption32-object five-arm distance/cluster-bootstrap summary.
- [x] Independent raw+latent-distance+PM/object-bootstrap recount, Drive/local/source
  archival, knowledge update and selective normal integration publication.

Ruling: user delegated this hypothesis test; no new design approval loop.
Existing user branch/dirty state stays. No runtime teardown authorized.

[Run] Execution complete; see shape_latent_proximal_results_20261031.md.
Independent acceptance/distance/cluster analysis passed. Last archival and
selective publication receipt remain the final documentation gates; no GPU work.

[Run] Final source/operator/derived Drive roundtrip accepted; selective publication is the final gate.
