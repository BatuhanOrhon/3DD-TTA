# Chamfer depth campaign preparation — 2026-10-30

The registered study has executed in owned runtime
`tta-chamfer-depth-tuning-a100-20261030-r2`; remote study root is
`/content/studies/chamfer-depth-tuning`. Initial r1 was unexpectedly lost before
research. See `knowledge/chamfer_depth_tuning_study_20261030.md` for dated
execution, ownership, exact source/checkpoint identities and approval history.

The study is accepted:32development settings attempted (27finite/5numeric
infeasible),9/9frozenconfirmation jobs passed. N5/default,N8/default and
N8/eta=.003/gamma=.03/p=.95 yield all15 Probability Mean66.510417%,
67.699653%,68.741319% on256reservedobjects/seeds0/1/2. TunedN8−N5 is
+2.230903pp conditional95%CI[1.623047,2.864583]. See
`knowledge/chamfer_depth_tuning_results_20261030.md` for all costs/limits.
FullDrive/localraw archive and7267members/422caches verify; independentlocal
analysis reproduces39acceptances/5infeasibilityreceipts/78comparisons andthe
frozenstage/scorehash. Compactevidence is in`recovery_r2/final/`. Large raw
ZIPs/caches stay local/Drive. Derivedanalysis/knowledgeDriveroundtrip also passes; ownedr2closed18:06UTC
under explicituserauthorization. Protectedpeerremainedpresentunchanged;
no furtherGPUjobpending.

The historical control is the original-code Chamfer host with p=.95. This is
not an exact paper replication (paper p=.96 is represented in the tuned grid).
Primary selection is all15 probability mean; terminal, minimum entropy and
macro14 excluding Background are diagnostics. N1 eta is unidentifiable for
normal14 because original-style final decoding ignores its final conditioning
style update; eta effects can come only from Background35 in that case.

## Files and dependencies

- `campaign.py`: deterministic IDs, immutable registration/stages, exact finite
  selection, explicit whole-phase approval gate, fresh slots, cumulative ledger.
- `operations.py`: bounded subprocess supervision and stdout streaming, real
  CPU/Drive preflight check, immutable SHA-verified Drive preservation.
- `analyze.py`: independently read sealed raw bundles and physical cache bytes,
  derive policies, report cases/corruptions/seeds/cost, paired object bootstrap.
- `make_binding.py`: verify the actual sealed operator payload and nine critical
  current source files; generate the binding without fabricating identities.
- `test_campaign.py`: synthetic CPU fixtures; these are not research evidence.
- `ids/`: exact sorted 128 development +256 confirmation IDs. SHA-ranked with
  seed20261030, disjoint from the prior70+70+140 tracked cohorts. No labels read.
- `binding_v1.json`: actual asset/source/data/checkpoint identity declarations
  from the parent's sealed payload; runtime CPU/Drive verification remains due.

Existing dependencies are `research_artifacts.py` (corruption order),
`scripts/analyze_scan_trajectory.py` (`Bundle` safe ZIP/manifest reader), and
`scripts/analyze_legacy_trajectory.py` (temperature-one policies). No old large
controller is copied. Physical cache/RNG verification requires **NumPy1**;
the temporary local interpreter is
`/private/tmp/3dd-depth-cpu312-20261030/bin/python` (Python3.12, NumPy1.26.4).
The runtime remains the canonical project's Python3.8 environment. Production
worker entrypoint is `scripts/eval_chamfer_depth.py`.

## Exact bounded search

1. Technical smoke: N1/N5/N10 defaults, Gaussian+Background, seed0, first32 of
   the sorted development IDs; separate cache namespace, no selection scores.
2. Depth: N1..10 with p=.95, gamma(local)=eta(conditioning style)=.01.
3. Rates: StageA best2 depths, full independent3×3 gamma/eta {.003,.01,.03};
   reuse the two defaults,16 new jobs.
4. Fraction: best rate pair at each chosen depth, p {.90,.96,.99}; reuse .95,
   six new jobs. Maximum32 distinct development settings total. Exact score
   scope is enforced; no extra configurations or failed-slot replacement.
5. Freeze at most3 unique conditions: depth5/default, StageA best/default,
   best tuned settings. Preserve role aliases when conditions coincide. Fixed
   seeds0/1/2 on the same256 reserved objects, at most9 confirmation jobs.

Ties prefer lower depth, then eta/gamma/p ascending. Original registration
stopped on unavailable required endpoints. After the first numerical failure,
the user explicitly approved the immutable `continue_campaign.py` amendment:
attempt every registered point once; retain only narrowly verified numerical
failures as infeasible/null; select among complete finite endpoints. This is a
post-failure exploratory amendment. Other failures still stop for review.
This finite sequential search cannot establish a global optimum. Previously
exposed benchmark objects are excluded from these stage reservations, but the
benchmark itself has historical exposure; no externally unseen claim is made.

## Registration, continuation and independent reanalysis

`plan_v1.json` is the actual immutable registered plan, SHA256
`3d6e42dbfc49966fc8d9b5ed79a133355a06dac63bde8e061a6fe624dfb1e8a6`.
`approval_v1.json` records the user's whole-phase approval, estimated9360s with
10800s charged budget. Do not regenerate or overwrite these receipts.

`continue_campaign.py` preserves the original plan/math/IDs/grid and consumes
actual proposal, user acknowledgment, amendment and original approval source
receipts under `recovery_r2/rates_failure/`. Its 69 combined CPU tests with
`test_campaign.py` and same-VM CPU proof passed before continuation. Failed
slots are never retried or replaced; incomplete confirmation seeds cannot be
converted to completed endpoints.

To independently analyze a retrieved immutable campaign tree in a fresh output:

```sh
PREP=result/preparation/chamfer_depth_tuning_20261030
python "$PREP/analyze.py" --plan "$PREP/plan_v1.json" \
  --root /path/to/extracted/campaign --output /path/to/fresh-analysis
```

Use NumPy1 for physical latent/RNG verification. Analysis verifies sealed raw
bundles, sources/models/IDs/cache bytes and reproduces controller acceptance;
then derives per-case/corruption/seed policies and paired cluster intervals.
Failed partial prefixes remain unavailable. Final archived root ledger includes
phase-finish charges that the controller's earlier embedded analysis may omit.

## Budget, failures and archive semantics

The legacy field `cumulative_wall_cap_seconds` is a **measured stop-before-next-
worker budget**, not a strict clock guarantee. Worker allowance reserves180s
for acceptance/copy and60s for final ledger persistence. Every phase charges
actual worker/probe/failed/analysis/Drive-copy wall plus a conservative60s final
ledger-copy reserve. Acceptance/Drive operations can overrun their allowance;
retain the actual overrun and stop before more GPU work. This includes failure
preservation, not GPU billing or VM idle/setup time. The final ledger copy is
checked against its60s reserve; an excess is explicitly recorded and requires
review. No new GPU runs are authorized by an elapsed budget.

Each job's sealed raw ZIP uses the existing worker Drive protocol. Controller
snapshots are under Drive `.../campaign/<planSHA>/snapshots/<sequence>/`, with
physical cache bytes under the same plan root's `paired/` or `paired-smoke/`.
Cache bytes are copied once and readback verified; reuse checks recorded raw/
published identities. Snapshots include exact plan/approval/stages/scores,
job/selection/command/log/acceptance/failure files and the latest ledger.
Failures, timeouts and interrupts also recursively copy every existing
`slot/raw` file/ZIP with hashes; no final ZIP/manifest is invented, and those
partial prefixes remain unavailable endpoints. Nothing overwrites old raw
files or retries a slot. SIGKILL/VM loss can leave an unresolved lock/phase;
reconcile and preserve it explicitly before any continuation.

Independent analysis reports terminal/entropy/PM per case/corruption/seed,
all15/macro14, reconstruction/probe/terminal/IO/worker costs and the campaign
ledger. It compares matching policies and named PM-versus-N5-terminal controls.
Object-cluster bootstrap resamples256 original objects jointly over all15
corruptions and three fixed seeds, never768 independent objects. Intervals are
conditional on the selected settings/model/corruptions/seeds. Missing failed
seeds remain unavailable; no successful-prefix replacement is allowed.

Synthetic tests are distinct from real GPU evidence. R2 technical smoke and
all development stages have actual sealed raw/Drive validation; final local
confirmation retrieval and independent reanalysis have passed. Finalderived
Drivecopy andexplicitlyauthorizedowned-onlyclosure have also passed. Bulk ZIPs/caches stay local and on Drive; integrate
necessary code, knowledge and compact scientific evidence only.
