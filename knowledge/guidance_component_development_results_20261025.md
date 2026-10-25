# Agent B fixed component development — 2026-10-25

[Run] Accepted local analysis of32 immutable head ZIPs: two datasets, four
arms, four frozen heads, all15 severity5 corruptions,70 registered development
object IDs/dataset, seed0. The96 case/policy rows contain33600 terminal head
inputs and134400 candidate predictions. This is a development subset, not a
full benchmark. The actual GSD host, native binaries, checkpoints, preprocessing,
prepared cloud/z0/h0/noise and post-preparation RNG are matched across arms.

Balanced accuracy gives each of the eight dataset/backbone cases equal weight;
each case averages all15 corruptions. Policies use three probes plus terminal.

| Arm | Terminal % | Minimum entropy % | Probability mean % |
|---|---:|---:|---:|
| C+S |53.8690|55.2024|55.1071|
| C |54.4167|54.8095|55.1905|
| S |50.3690|52.5595|52.6190|
| U |50.6548|52.5714|52.8095|

[Run/Inference] Chamfer improves over unguided in all eight cases under each
readout: +3.7619pp terminal, +2.2381pp entropy, +2.3810pp probability mean.
Adding the transferred spectral coefficient to Chamfer gives -0.5476pp terminal
(2 positive/6 negative cases), +0.3929pp entropy (5/2/1 positive/negative/tied),
and -0.0833pp probability mean (4/2/2). Spectral-only versus unguided is
-0.2857/-0.0119/-0.1905pp respectively. This supports Chamfer at this fixed
coefficient; it does not establish that every calibrated spectral method fails.

[Run] Conditional paired object bootstrap,2000 replicates, resamples the same
70 original IDs jointly across all corruptions/heads/arms within each dataset,
independently between datasets. C+S minus C probability mean has95% interval
[-0.4643,+0.2976]pp; entropy [-0.0238,+0.8098]pp. These are seed0 development
intervals, without selection/multiplicity adjustment, seed uncertainty or
generalization guarantees. Full paired corrected/harmed and case/corruption
tables are in the accepted analysis directory below.

[Run] Spectral-only mean local update norms are0.0003453 ModelNet and0.0003127
Scan; Chamfer-only values are2.0249511 and2.0985394. These are unweighted means
over15 batch summaries, not isolated combined-loss gradients. Raw loss units
and numeric weights are not comparable. Spectral coefficient1.17 was transferred,
not calibrated here; the very small standalone updates limit a necessity claim.

## Measured cost and limitations

Shared reconstruction is counted once across four heads. Fresh-encode smoke
uses both Gaussian/Background, B70, identical probes and all four heads on the
same A10080GB. Averaging the two dataset totals: C+S26.1688s, C19.6421s,
S25.8991s, U12.1786s. C is24.94% below C+S for this instrumented reconstruction
scope; complete job+validation time is9.86% lower. These are two-corruption
measurements with observer/IO overhead, not B1 or standalone-head deployment
latency and not an early-stop speed claim.

Development shared times summed over both datasets are C+S302.3303s,
C196.1765s, S288.0571s and U132.9136s. C+S creates15 fresh caches/dataset;
the remaining arms replay those exact states, with cache IO included. Therefore
the35.11% C reduction here is comparison cost and must not be presented as a
measured deployment speedup. Whole-process peak allocations are about35.5GiB
C+S/S,34.4GiB C and5.5GiB U, with four heads loaded. Shared backward cost is
not independently additive by loss.

The completed initial smoke+development active research phase consumed
2093.6846s (34.8947min), below3300s. Setup, CPU analysis, transfer and idle VM
billing are separate. Development elapsed before final packing was1564.0199s.

## Frozen next stage

[Decision/Open] Freeze C + probability mean without parameter tuning. Its
55.1905% is0.0119pp below development-best C+S + minimum entropy55.2024%, an
unresolved difference, and its measured cost is lower. Both arms receive zero
tuning trials. No spectral modification is selected or claimed optimal.

The [frozen plan](../result/preparation/guidance_component_study_20261025/frozen_confirmation_plan.json)
registers C versus C+S on the reserved70 IDs/dataset, all15/four heads,
seeds0/1/2, before scoring any confirmation output. Primary comparison uses
probability mean for both; secondary compares frozen C probability mean with
frozen C+S minimum entropy. Terminal and own-terminal readout effects remain
diagnostics, without per-case policy reselection. The same held-out IDs are used
for every seed; original objects are disjoint from development across corruptions.
Previous benchmark exposure prevents an external unseen-test claim.

This is one new evidence-gated confirmation phase, expected41.10min from four
actual selected jobs/seed;20% margin plus300s preparation/receipt/retry reserve
gives54.32min. Its own hard3300s cap covers all three seeds together; the
completed initial ledger is preserved. Revised expectation above60min requires
explicit approval. Confirmation remains unrun at this record's creation.

## Immutable evidence

Preparation root: `result/preparation/guidance_component_study_20261025/`.

- `local_development_analysis/validation.json`, six CSV tables and
  `cache_roundtrip_validation.json`: independent CPU readout/count/state/source
  checks,32 canonical raw archive copies,30 physically downloaded cache NPZs.
- Development Drive/local archive405066186bytes SHA256
  `8abd0ec00c71996407d0fabc3933a7984c2e9860c8844e2d2940bff1db15901d`.
- Exact30-cache Drive/local archive128993199bytes SHA256
  `2eee44c204d556e028ed3e9be15580b38e5ea688806bd2915280b0068d7bf3e8`.
- All archives pass exact bytes/SHA, safe unique member paths/types and CRC;
  extracted evidence manifests and raw-cache records are retained. Raw archives
  remain local+Drive; compact metadata and analysis are versioned separately.
- Sealed manifest `7012c30a...f625ed75`, source ZIP `f5bca11a...576d0c6`,
  project source commit `62e9a2b`; full identities are in the acceptance records.
- Own live session `tta-guidance-components-a100-20261025-r1`; its foreground
  monitor remains active. Agent A resources and original user edits are untouched.

[Open] Confirmation, broad seed stability, full2468/581-object evaluation,
matched missing Point-MAE Scan source/original rows and calibrated spectral
search remain separate evidence gaps. No thesis contribution, equivalence or
universal component-necessity claim follows from this screen.


[Run/Open] Confirmation update14:36UTC: the plan-bound controller passed local
watchdog/process-group/ledger self-checks, independent review and remote Python3.8
self-check. Physical accepteddevelopment/source ZIPs and original plan hashes
verify. The whole three-seed phase began14:35UTC through the existing sole
monitor; first ModelNet C+S heldout batch and four Drive receipts are observed.
Source manifest SHA`b95170f32c73967dd97b55057e57e9107d037b0ca8b8bd7a2fdcd39ed2337229`.
This is execution progress, not an accepted confirmation result.

[Run] Supplemental macro14 excludes Background only, retaining all other14
corruptions and equal case weights: C+S terminal/entropy/probability mean
54.1964/55.8036/55.5102%; C54.7832/55.5102/55.7526%;
S52.3724/54.3495/54.5153%; U52.5765/54.4133/54.6556%.
The primary metric and already frozen candidate remain all15; this diagnostic
does not trigger reselection. Table:`local_development_analysis/macro14_without_background.csv`.


[Run/Code] Observer dispatch spans independently re-read from64 accepted smoke
+development raw ZIPs agree across four heads and are counted once. Supplemental
`component_dispatch_costs.csv` contains all16 dataset/arm/stage rows and update
norms. For development, mean per-dataset summed shared backward+update spans are
C+S31.5695s, C30.9416s, S31.6770s, U0.1256s; mean graph/eigh spans46.6838s
C+S and45.3979s S, absent in C/U. Probe spans are about35–36s per15 batches.
These are unsynchronized host dispatch wall spans, not isolated CUDA event
timings or an additive GPU decomposition. Terminal-decode spans also contain
terminal classifier work; the synchronized runner reconstruction and separate
terminal-head counters remain the cost totals. The shared backward executes
once/step, not separately per active loss. Fine-grained CUDA profiling and
standalone-head/B1 deployment latency are unmeasured.
