# Selective integration: clean-z condition-gradient research — 2026-10-31

[Code/Decision] Source worktree7bf0 stays on
`integration/chamfer-guidance-variants-20261030`, HEAD
`a82200adbe117ce56a134dc3c22601cb73dc2654`, existing dirty state preserved.
Latest fetched shared target before preparation:
`e8523e023f53650a057ddb1b2fa097bdf17078dd`.

Selected scope: [research report](shape_latent_gradient_research_20261031.md),
compact URL/SHA/pin/code/runtime/validation evidence and new semantic entries
in README/findings/method/open-questions/LION notes. Prior accepted results
and dated history remain. Shared files are updated by adding the new entry,
not replaced with an older whole source-worktree copy.

Excluded: all pre-existing code/test/config work (already selectively
integrated or outside this new literature scope), third-party full texts,
raw data/weights/archives, temporary PDF extraction environment and wrappers.
There is no algorithm/config/default change and no GPU or research rerun.

[Run] Checks: physical primary receipt hashes, official Git blob identity,
local audited code unchanged from accepted e8523e0, new local links/JSON,
source branch/HEAD preservation, diff whitespace and staged scope.
[Evidence](evidence/shape_latent_gradient_research_20261031/validation.json).
No production test suite rerun for a documentation-only change; no native
gradient, convergence, clean-z recovery or accuracy claim.

[Decision] Use the existing clean publication checkout based on the latest
shared target; normal fast-forward push only to origin/thesis-integration.
Final published commit/remote verification is recorded outside Git at
`result/preparation/shape_latent_gradient_research_20261031/publication_receipt.json`
and reported to the user. The published commit is also discoverable with
`git log origin/thesis-integration -- knowledge/shape_latent_gradient_research_20261031.md`.
No original worktree reset/stash/clean/branch switch or runtime operation is
part of publication.
