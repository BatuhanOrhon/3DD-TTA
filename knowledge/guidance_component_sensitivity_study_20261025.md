# Agent B parameter and loss sensitivity — 2026-10-25

Status: CPU preparation; no new GPU research authorized or launched.
Required evidence labels: [User report], [Code], [Run], [Inference], [Open].

## Question and evidence boundary

[User report] Extend the fixed component study with different GSD parameters
and several loss calculations, and assess whether Chamfer/original3DD is more
reliably supported. This reopens research; preserve own live A100 and monitor.
The earlier optional close question is not teardown authorization.

[Run/Inference] Existing C probability mean56.8929% versus C+S56.7222% has
conditional interval[-0.0397,+0.3651]pp. That is not proof of superiority or
equivalence. At transferred coefficient1.17, standalone spectral local updates
are several orders smaller than Chamfer's. Chamfer-only in the GSD host also
differs from original3DD in batch, horizon, final style and DDIM endpoint.
Historical Scan readouts exceed original3DD in some cases. No finite menu of
experiments can establish universal superiority over every GSD parameter/loss.

Falsifiable question: after a declared finite search and label-free scale
calibration, does a frozen spectral-containing candidate outperform a matched
GSD-host Chamfer comparator and the distinct original3DD whole-policy baseline
on newly reserved objects across three seeds? Retain all negative/null/failure
results; the conclusion must be allowed to favor either family.

## Selected design and alternatives

The existing four-arm matrix is accepted evidence and will not be rerun merely
for a new launcher. Exhaustive tuning or trying variants until Chamfer wins
would not answer the question. The selected extension is a bounded exploratory
screen followed by a newly frozen confirmation. This is the smallest useful
way to test the weak-scale explanation, a few distinct losses, parameter
sensitivity and the actual original host together.

Use the same two datasets, four heads, all15 severity5 corruptions, native
binaries/checkpoints and GSD B70/M800/beta2/10-or30-step/update-style policy.
Graph construction, isolate policy, eigensolver, frequency cutoff and scheduler
remain fixed. New objectives are named experimental variants, not repairs or
full GSDTTA reproduction. No model training or gradient projection is added.

## Loss definitions and calibration

For predicted/reference graph coefficients H and Q, N latent points, and the
unchanged weights w_i=exp(-2 lambda_i) below800 (zero above), use:

1. `legacy_l2`: sum(w*(H-Q)^2)/(3N), preserving the exact default arithmetic.
2. `huber`: sum(w*smooth_l1(H,Q,beta=1))/(3N). Delta1 is fixed before labels.
3. `power`: sum_i w_i*(sum_c H_ic^2 - sum_c Q_ic^2)^2/(3N).
   This is per-mode squared energy, not amplitude or a claim of invariance
   under arbitrary mixing of degenerate eigenvectors.

For each loss, compute actual first-step gradients through the frozen local
prior and DDIM predicted-clean path on all30 existing development caches,
seed0. Use per-object RMS local updates at eta=.01; compare spectral coefficient1
with Chamfer1,p=.95. Freeze one scalar per loss from the median of per-object
Chamfer/spectral update ratios across2100 input events, equally represented
across both datasets and15 corruptions. No labels/classifier scores enter this
calibration. Style updates are recorded separately; one scalar is not asserted
to equalize both blocks. Ratios rho=.1 and1 multiply that frozen scalar.

Reject nonfinite gradients; record zero/tiny norms. A loss needs at least90%
valid local-gradient pairs; no silent coefficient clipping. Candidate scalar
must be finite and in[1e-12,1e12], otherwise stop for diagnosis. No calibration
per corruption/backbone/target label, or recalibration for a trial's rates.

## Fixed development trial budget

Development remains the original70 objects/dataset, all15, seed0. Each family
has six configurations INCLUDING its already accepted baseline. Each new
configuration sees both datasets/four heads and the same prepared states/noise.
The accepted baseline may be reused only after source/numerical/state parity;
its original source identity remains explicit rather than being relabeled v2.

| Family | Configurations, in registered order |
|---|---|
|C|accepted base; eta=gamma=.003; eta=gamma=.03; eta=.003/gamma=.01; p=.80; p=.99|
|C+S|accepted base1.17; calibrated L2 rho.1; calibrated L2 rho1; Huber rho.1; power rho.1; calibrated L2 rho.1 with eta=gamma=.003|
|S|accepted base1.17; calibrated L2 rho.1; calibrated L2 rho1; Huber rho.1; power rho.1; calibrated L2 rho.1 with eta=gamma=.003|

Other rates remain.01,p=.95. A coupled eta/gamma scale is one declared factor.
The trial count is equal, not equal wall time or exhaustive search-space
coverage. There are15 genuinely new configurations/30 dataset jobs; failures
consume their trial slot and are archived. No adaptive replacement or broad
Cartesian grid. Terminal/entropy/probability readouts are retained; the fixed
selection score is probability-mean balanced all15 across eight cases.
Ties favor earlier registered configurations; no per-case/corruption routing.

## Original baseline and prospective confirmation

Enable terminal-only original3DD for all four existing frozen heads, including
the currently blocked Point-MAE path, with its explicit checkpoint/config.
Preserve batch32,5/35 reverse updates, original final style, DDIM finalalphaTrue
and original tta.py. A missing matched-source row stays explicitly unavailable.
This is a new matched-subset/native study control, not a full accuracy rerun.
GPU smoke must verify the newly enabled boundary before broad execution.

Reserve140 NEW original IDs/dataset now, excluding BOTH previously used70-ID
sets; use the same new IDs across all corruptions/seeds. They are reserved from
this development stage, not an externally unseen benchmark. Do not inspect
their predictions until configuration freeze.

After screen acceptance, freeze the best spectral-containing configuration
across the six C+S and six S entries; this larger family-level choice opportunity
must be disclosed. Also freeze the best C configuration. Confirm:

- the chosen spectral candidate;
- a GSD-host Chamfer-only comparator with the candidate's eta/gamma/p;
- the best tuned C configuration if it differs from that comparator;
- locked original3DD as a separate whole-policy comparator.

Maximum four conditions, both datasets/all15/four heads/seeds0/1/2 on140 IDs.
If the spectral candidate is C+S, its matched C contrast isolates removing
spectral guidance at the chosen setting. If it is S-only, the contrast replaces
the objective; it is not a conditional spectral increment. Confirm equal
probability readouts for GSD-host conditions; original has ordinary terminal.
Report same-policy terminal/entropy diagnostics, all15/macro14, every case,
per-seed signs/ranges, corrected/harmed events and measured cost. Original's
different host permits whole-policy outcome comparison, not shared-latent or
isolated-loss causal claims.

Resample original object IDs jointly across corruptions, methods, heads and
all three seeds; independent draws between datasets. There are280 reserved
object clusters, not840 independent objects. Failed/null cases and unequal
family selection opportunities remain visible. No optimum/equivalence claim.

## Batches, compute and gates

### Batch 1 — explicit losses and original Point-MAE boundary

- Goal/scope: minimal opt-in objectives and missing terminal original head.
- Touched areas: tta_gsd_v2.py, runner, shared spectral-loss function and tests.
- Stack: existing CPU PyTorch contracts; no GPU or dependency installation.
- Dependencies: accepted fixed-source artifacts and ownership contract.
- Development notes: preserve default math/RNG/configs; pin new dependency.
- Verification: old-mode regressions, finite-difference/robust/power checks,
  strict CLI and original-policy tests; Python3.8 syntax and diff checks.
- Knowledge target: this note; append implementation and actual test evidence.
- User gate: authorized CPU preparation; no new GPU before duration approval.
- Outside scope: graph changes, original trajectory readouts, paper reproduction.

### Batch 2 — concrete sealed campaign and calibration

- Goal/scope: label-free gradient calibration, immutable trial/ID registry,
  complete commands, source/assets hashes and single whole-campaign ledger.
- Touched areas: own preparation root and isolated new remote source directory.
- Stack: existing Colab CLI/monitor, pinned environment/native assets.
- Dependencies: Batch1 tests; safe reuse/parity controls and reviewed preflight.
- Verification: CPU dry commands, trial counts/IDs/hash guards, calibration
  formula checks, strict expected trial parameters and timeout/recovery tests.
- Knowledge target: this note, findings/open questions/README and phase plan.
- User gate: one comprehensive >60min research-run approval BEFORE any GPU.
- Outside scope: experiments selected to force a desired conclusion.

### Batch 3 — bounded search, new freeze and confirmation

- Goal/scope: approved calibration/smokes/30 development jobs/original control,
  then evidence-gated frozen confirmation (maximum24 condition/dataset/seed jobs).
- Dependencies: exact sealed proposal approval, CPU/source/Drive acceptance.
- Verification: same-state/noise within GSD, source/native/checkpoint/ID/state/
  readout checks, physical Drive/local SHA, paired object-cluster analysis.
- Knowledge target: separate exploratory and confirmation result notes.
- User gate: keep the entire approved campaign budget, not per-CLI subdivision;
  any material scope/duration increase needs reassessment.
- Outside scope: full2468/581 benchmark, unlimited tuning, automatic teardown.

[Open] Preliminary whole campaign estimate3.5–5 GPU hours, provisional hard cap
5.5h, includes calibration, all smokes, bounded retries, development, original
baseline, confirmation and evidence persistence. Exact measured-cost proposal
will replace this estimate before approval; no GPU launch is authorized by
this document. Setup/transfer/billing lifetime is distinct. Existing own session
`tta-guidance-components-a100-20261025-r1` and monitor52794 are preserved.

## Batch 1 implementation and CPU verification — 2026-10-25

[Code/CPU/Open] Working base `3bc8ebb05911a76af1b831d5a295d1d59be51eed`;
this batch is an uncommitted source change, not an executed Colab revision.
`spectral_guidance_losses.py` now owns `legacy_l2`, fixed-delta-1 `huber`,
and per-mode squared-energy `power`, all with the declared `3N` reduction and
unchanged exponential weights/cutoff. Logging and applied spectral guidance
use the same helper. Invariant loss, graph/eigh/isolate behavior and original
`tta.py` are unchanged. Exact old-inline legacy values, gradients, reconstructed
outputs/style, metrics and RNG are checked. Power tests cover XYZ orthogonal
rotation and per-mode sign changes and explicitly demonstrate that arbitrary
mode mixing is not an invariance.

The opt-in `--spectral-loss-kind` requires an active-spectral component arm;
explicit selections record the kind and modified flag. Omitting the option
preserves existing configuration dictionaries. Terminal-only original3DD now
accepts Point-MAE with both explicit checkpoint/config paths, retaining B32,
5/35 updates, original style and finalalphaTrue. Its unbound source row stays
unavailable; no original trajectory observation is enabled. The prior observed
Point-MAE unavailable-source reason string is preserved.

[Code/CPU] The root's unrestricted focused suite passes **54 tests in 3.672s**;
the new 11-test module separately passes in 0.527s. All 12 pre/post full default
configuration groups compare exactly equal; Python 3.8 AST checks and
`git diff --check` pass. Tests include finite differences for all three losses,
Huber outlier gradients, exact legacy parity, CLI/original policy guards and
an actual `snapshot_sources` copy/hash check for the new helper. The first child
suite's five legacy temporary-directory errors were sandbox write denials;
the unrestricted rerun resolves that verification limitation.

[Open] No GPU/Colab execution, calibration result or accuracy/speed improvement
is established by this batch. The next sealed source ZIP must include the new
helper and its declared identity; the generic multibackbone source inventory
already captures it. Historical legacy entrypoints/inventories are outside
this campaign and remain unchanged. A new GPU smoke must verify the original
Point-MAE boundary and selected losses after the whole-campaign approval gate.
Source/helper hash, gradient/RNG or locked-original-policy mismatches invalidate
the affected future comparison.
