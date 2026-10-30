# Author-code consistency audit before GPU tests — 2026-10-30

[User report/Decision] After implementation, inspect relevant author GitHub
repositories for inconsistent transfers before research tests. Existing CPU
unit tests already ran; no research GPU test has started. This gate is separate
from environment CUDA checks and the >60min whole-phase approval.

## Outcome

[Code/Inference] No unintended loss sign, reference detach, cross-object
entropy reduction, classifier update or nearest-reference-weight routing defect
was found in the implemented three interventions by static tracing. They are
explicit adaptations under frozen-head/input-guidance constraints, not faithful
reimplementations of the named papers. Real CUDA gradients and empirical
accuracy/runtime remain open.

| Predecessor | Author-code evidence | Intentional transfer and boundary |
|---|---|---|
| Adaptive Guidance | Official paper/project/author links inspected. Pinned website2fe5f7f is website source; no official sampler implementation found. | Original cosine-triggered CFG branch removal differs from our fixed every-third real SCD backward. All prior forwards stay; exact author caching/autograd implementation remains unverified. |
| CloudFixer | Author commitf043e244; adapt.py, pc_utils.py, parser blob bytes verified. | Released default inversely weights mean **squared** neighbor distances. Our direction agrees; latent reference, nonselfk10, epsilon, mean1 and weighted trimmed Chamfer replace physical displacement regularization/R/Delta optimization. Prior gradients are intentionally retained. |
| MAMVI | Author commit0b553973; adaptation/normalization helper blobs verified. | Author uses directed view-to-detached-consensus KL plus per-view entropy and BN/LN adaptation. Ours uses fixed-X symmetric KL or arithmetic marginal H as separate input-guidance arms; no parameter/buffer adaptation,12mask views, reliability threshold or adaptive LR. |
| GDA | Official author repo yunyuntsai/gda_robust_tta commit4197ff4;10files blob-verified. | Plain code marginalizes16AugMix predictions; recommended weighted branch fuses original/generated log probabilities without renormalizing. Our H((p_X+p_decoded)/2) is a different normalized arithmetic objective, fixed reference and one reused decoded view. |

The earlier CloudFixer paper-distance caveat needs this code nuance: the
released default already uses squared distances. Neither predecessor supplies
evidence that our density estimate denotes clean geometry or that fixed-X
confidence is correct. Keep isolated accuracy/harmed and measured cost controls.

## Preserved audit evidence

- [CloudFixer/MAMVI details and local source mapping](evidence/chamfer_author_code_audit_20261030/cloudfixer_mamvi_audit.md).
- [GDA author repository, loss plumbing and gradient caveats](evidence/chamfer_author_code_audit_20261030/gda_audit.md).
- [Adaptive Guidance official-link search and implementation limit](evidence/chamfer_author_code_audit_20261030/adaptive_guidance_audit.md).
- Raw sources/PDF/site metadata and pinned local snapshots:
  `result/preparation/chamfer_guidance_variants_20261030/author_code/`.

## Engineering review and next gate

[Code/Test] Independent review found two acceptance/integration defects, not
predecessor-loss reproduction defects: runtime calibration proof accepted
formulas later rejected by analysis; three timing repeats collided in the
accuracy analyzer's unique keys. The first now uses one dependency-free strict
validator in runtime/planner/collector/analyzer. The second has a separate
three-repeat fresh-cost path while preserving duplicate rejection for accuracy.
Regression tests cover malformed formula/reduction/4200-row coverage and
reused/unmatched timing groups. Root's final integrated suite passes436tests and331subtests after
these fixes and the CPU-only-source receipt integration fix.

[Decision/Open] Author-code audit gate is complete with AG sampler availability
explicitly limited. Source freeze must follow final CPU verification. Actual
GPU calibration must establish finite/nonzero component gradients; zero-weight
shams must preserve C geometry/logits/frozen states and RNG. Any native
failure, spectral/graph call, pairing mismatch or source/asset discrepancy
stops collection and retains failed evidence. No speed or accuracy benefit
is established by this audit.
