# Full ModelNet40-C confirmation of frozen M100 halfmix — 2026-11-07

[User report/Decision] User requests full-dataset testing of the observed M100
mixture improvement, followed by another dataset and separately trained source
model if a visible gain remains. Prior explicit instruction authorizes all needed
GPU tests without further duration approval; preserve staged knowledge/results,
negative outcomes, peer isolation and runtime until explicit teardown approval.
This extends, rather than rewrites, the accepted [16-ID pilot](gsd_lowband_halfmix_results_20261107.md).

[Code/Run/Open] Own existing isolated worktree
`/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/gsd-lowband-halfmix-20261105`,
new branch `research/gsd-lowband-full-20261107`, base0b6faebe916f3b1a797d65c6e5120931e2103bf3.
Own runtime `tta-gsd-lowband-halfmix-a100-20261107-r1`, endpoint
`gpu-a100-s-usc1c0-gz8ghzlq66bg`, standardA10040GB; sole foreground monitor91825.
Server assignment exists15:19UTC; last kernel liveness15:13:45UTC. No peer touched.
New immutable task root/Drive suffix `gsd_lowband_full_20261107`; previous pilot
source, artifacts, calibration and output directories remain unchanged.

## Frozen design and resource expectation

- Full official2468IDs,15corruptions,severity5,seed0,B2:37020object-corruption
  pairs/arm. Reuse exact pilot SCD/PM, DDIM100, N8/B35, gamma.03/eta.003,p.95,
  raw/eval/EMAoff/frozenPointMAE, hardM100 boundary-eigenspace rules.
- alpha100=4433.612924010353 stays frozen; weight=2216.8064620051765.
  No label-driven alpha/rate/rank/schedule selection or default promotion.
- Three full arms: operational SCD, live half-SCD/GSD0, M100 halfmix.
  This measures practical mixture gain and separates simple SCD attenuation.
  Existing full/live0 technical repeat failures remain; bounded matched technical
  controls verify the new runner rather than pretending native determinism fixed.
- Keep B2; changing batch size would require a separate protocol/equivalence
  study and can alter native paths/RNG. Prior measured costs project24–30GPUhours
  for three full arms before retries; approximately30–40hours end-to-end once
  initialization/archive/transfer IO is included. Actual throughput will be measured. The previous
  pilot55minute clock is not reused/reset to disguise this new longer phase.
- Fixed sorted64-IDshards, final36IDs; each shard all15corruptions, arm order
  rotates without looking at scores. Per-B2 original preparation/cache/noise/RNG
  paired across arms, retained graph bases and all standard inference artifacts.
  Journal completed whole shard/arm jobs and recover solely from verified artifacts;
  failed/partial attempts remain immutable, no silent in-place restart.
- Full coverage is primary. Historical full-benchmark evaluations already exposed
  all2468IDs; there is no globally unseen subset. Separately report the20prior
  halfmix calibration/pilot IDs and other2448 only as a prior-M100 subset split,
  not an exhaustive exposure classification. Broader targeted-manifest audit
  identifies at least1067registered IDs; historical full-data exposure dominates.
- PM primary, terminal secondary; paired object-cluster confidence intervals,
  per-corruption counts, correction/harm, component/graph/reconstruction/IO costs.
- Conditional external confirmation, selected before outcomes: advance if full
  M100 PM gain vs operational is >=1.0percentage point and paired95%object-cluster
  CI lower bound>0. This is an operational definition of the user's visible gain,
  not a claim of native causal attribution. Preserve half-control comparison and
  native variability caveat. Otherwise preserve the full negative/null result.
  Prefer accepted ScanObjectNN-C s5 with an independently trained, compatible
  source checkpoint; exact data/receipt/checkpoint contract must be audited before
  that stage. Do not bypass an invalid Drive receipt. No external GPU stage yet.

## Ordered batches / checks

1. Runner boundary: explicit benchmark-shard flag, positive even count<=2468,
   mandatory hashed ID manifest, previous diagnostic defaults/capture guards intact.
   No sampler/loss/graph/head math change. CPU RED/GREEN and impacted tests.
2. Durable operator: fresh source/native/data/calibration binding, bounded technical
   checks, full coverage manifest, per-shard immutable job journaling and verified
   Drive/local archives. Estimate actual runtime/storage before launching full.
3. Full three-arm ModelNet confirmation: no outcome-dependent stopping/subsets;
   update knowledge after completed chunks, retain runtime-loss/retry failures.
4. Independent raw/count/ID/CSV/PM/terminal/source/cache/native acceptance. Freeze
   and apply the external-dataset gate, publish necessary completed code/evidence
   via clean latest thesis-integration. If gate passes, execute audited separate
   source-model/dataset confirmation, with label-free calibration if required.

[Open] Batch1 implementation/CPU checks, remote binding, technical/storage checks,
full run and conditional external confirmation are pending. No new accuracy result.


[Code/Decision] Builder review distinguishes24–30hours device/graph work from
30–40hours end-to-end, based on prior measured external wall. Recorded targeted
subset union582+20halfmix+14newLION=616; historical full-data exposure already
includes all2468IDs. New full comparison is confirmation on existing benchmark,
not unseen-data evidence.

[Run/Code] Existing ownVM preflight15:30:43UTC: A10040GB/0MiB, no research
workers; Drive mounted, pinned native/checkpoints and original frozen calibration
hashes pass. Local filesystem186GiBfree, mounted Drive reports177GiBfree
(filesystem report, not a verified cloud-account quota). Full raw volume pilot
projection about99GB compressed/102GB unpacked before larger-shard amortization;
keep acceptedZIPs on Drive+Mac and remove only successful owned duplicate runtime
payloads after both copies are hash-verified. Failed/partial directories retained.

[Code/Test] Batch1 commit1cbd135: explicit hashed benchmark shards, legacy guards
preserved, no math changes;58affected tests/78subtests pass. Bare broad collection
encountered6known local native/vendor/demo errors; intended tests/ suite remains
the validation boundary and complete logs are retained. No full-data GPU yet.

## 16:00UTC — new-runner technical acceptance before full launch

[Code/CPU/Run] Source boundary commit1cbd135; driver patch98b63f46...; new asset
manifest e2cd2942fc9893c863e50018ab5d2373bac00ba5af30a5dbcc75edf99f263f47,
full actualsource snapshot43601f4554882ec1365a9ea52d1641c224df2f4f55ce3ea7e012a7eb1ec38e69.
817tests/2skip/533subtests pass51.55s. Whole scientific clock began15:49:23UTC,
newtask48hbudget; previous pilotclock unchanged. Fifteen actual technical caches
and4technical inference jobs completed, all7files/IDs/source/native/model/RNG/
predictions/basis checks passed; cache-worker native/model equality and exact
half0/M100samebasis+filter hashes pass. MaxCPU basisloss error2.16094e-8.
Native repeated-control gate stillfails; this is retained, not an advancement
waiver or fix. Full paired benchmark proceeds by user request; causal attribution
remains limited. Technicalraw79,360,548bytes/SHAed4fda35c95e88b7bea845efce30c64106654041e7f6618d2c71ac53547089c0,
Drive and localCLIparts/wholeSHA/CRC validated; localactualcaches accepted.

[Code/CPU] Longrun operator tests all156synthetic steps, lock contention,
truncated ACK, exactDrive+localack pruning, process-group failure/timeout/orphan,
failed-attempt refusal and verified resume. Local transfer transaction selfcheck
passes injected ACK-network failure, durable receipt-before-ACK replay and
unchanged cache-recovery hashes. Static independent review found no remaining
launch blocker. Source and plan are frozen before full outcomes.

[Decision] The earlier616subset union was not exhaustive and is superseded:
all2468already had historical benchmark exposure. Report only20prior-M100IDs
versus2448others as a narrowly named stratum, not globally unseen data.
External dataset/source is accepted BFTT3D581main_split/all15/s5 plus separate
Scan-trainedPointMAE SHA53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac.
Its separate oddtail/dataset boundary will be prepared without GPU execution
until the predeclared fullModelNet gain gate passes. No full-data score exists yet.

## 16:20:46UTC — full three-arm campaign RUNNING

[Run/User report] Launched on the same owned StandardA10040GB session
`tta-gsd-lowband-halfmix-a100-20261107-r1`, endpoint
`gpu-a100-s-usc1c0-gz8ghzlq66bg`; sole monitor PTY91825; bounded local sleep
inhibitor PTY71002. User explicitly authorizes all necessary GPU tests without
additional permission; whole new phase estimate30–40h remains recorded.
First actual unit `shard000_cache` started, no full accuracy yet. Immutable
operator_v1.zip36,034bytes/SHA344cfb5169a00e4adce58e0dfd49b155e3ea221bc84e17da5ca45bb9e4d74042
verified on Drive before computation. Executed full_main_phase.py
SHA99448730b4c63e5de6d13b7115fb21a1066c77a52b704b3cda55d215ddde4fff;
controller58c466f291e20b02c16b23f8691a1dee46c57616b16902743913a288f07b40b2.
39deterministic shards,117inference+39cache jobs, frozen source/alpha/plan;48h
wholephase cap measured from15:49:23technical start. Timeout49h is transport
allowance, not a reset of that scientific cap. Old pilot/peer paths untouched.

[Open] Finish all156accepted/locally+Driveverified transfers, reconstruct37,020
object-corruption predictions per arm, and only then apply the preregistered
external gate. Runtime lifetime/network loss remains possible; never infer
process death from a lost stream or launch a duplicate while state is unknown.

[Run/Code] First full cache archive accepted on Mac and ACKuploaded: shard000,
480actual B2caches/64IDs/all15;64,948,028bytes/SHA4fe6d2dc0f42d0edd3c2e5b2e535c35aaa2fff253770c6f77b5d80208632267f.
Local strict source/cache tensor/RNG checks pass; transferconsumer PTY12727.
It polls the actual remote journal through Contents API every60s to recover
missed transfer events without another kernel cell. Downloaded actualcache
metadata are retained; only redundant splitparts removed after wholeZIP+ACK.
Consumer crash/replay+missedevent/unownedpath checks pass;
SHA9a70d4ef4d309295a8f6086e05d42055981825550f9e600b2e05a37d85359b5b.

### Completed preparation publication

[Code/Run] Source branch `research/gsd-lowband-full-20261107`, runner1cbd135
and protocol/acceptedtechnical484d335 selected into clean integration checkout
from origin0b6faeb; integration e40e39e/214fe6b pushed normally to
`origin/thesis-integration`. Integrated runner tests14passed/55subtests;
original full suite817passed/2skip. Source snapshots retain executed controller,
cache preparation, fixed plan and local acceptor/selfchecks. RawZIPs/native
binaries/weights, temporary deployment wrappers and incomplete external changes
excluded. This completes the preparation batch, not the full benchmark.

### Recovery boundary (before any runtime loss)

[Decision/Open] Runtime may expire before30–40h. On stream loss, audit server
assignment and actual journal/artifacts before another cell. New ownVM requires
canonical10cell setup followed by exact validated source/native capsule/data/
weights/calibration restoration; same runtime/native identity checks remain.
Restore accepted receipts, exact selection/plan/operator bytes, actual paired
caches, ACKs, transfer manifests and cleanup receipts from verified Drive/local
archives. Do not redownload or rerun accepted inference. Preserve the last
journal as a new immutable recovery evidence file before any derived continuation
journal is written. A genuinely lost unaccepted job requires an explicit dated
recovery record, preserving every available partial Drive batch/log/receipt and
recording unrecoverable ephemeral files; no silent deletion or same-directory
overwrite. New inference uses fresh UTC bundle names, retaining earlier partial
Drive bundles. Do not infer that WebSocket loss means worker death.

## 16:35UTC — first full-shard operational arm accepted

[Run] shard000_control64IDs/all15,480actualB2cases completed764.2278s
(process scope). Drive+Mac raw125,858,027bytes/
SHA5f9977c1c803147145efee8fa473a06294e7b4ce404f77cd672d4be6d8106433;
all strict source/native/model/actualcache/RNG/PM/terminal/count/CSV checks pass.
Half0 running; M100 pending. No complete three-arm shard yet, no contrast/gate.

[Code/CPU] Added full aggregation with exact39shard/2468object/37020pair
coverage, final156rawZIPhashes, independent15corruption-objectclusterCI,
20priorM100vs2448 narrowstrata, final componenttiming. Partial output leaves
external gate null. Synthetic positive/negative/count/gate/hash tests pass;
independent reviewer found no concrete blocker; actualpartial consumes current
cache+control receipts and reports0complete pairedshards/gatenull. Archives
remain at their absolute localreceipt paths. Local archivedtools moved to the
evidence root to preserve repository-relative imports; sourcebytes unchanged,
archivedselfchecks pass. Conditional externalCPU boundary published76d9eb3;
actualScanGPU/calibration notstarted.

## 16:53:08UTC — first half0 accepted; M100 running

[Run] First64/all15 half0 accepted480B2cases/480basischecks, requestedhard100
actualrank100–256 (complete boundaryeigenspaces). Source/native/model/cache/RNG/
PM/terminal/count checks pass; Drive+MacZIP1,004,528,916bytes/
SHAad7e155bcc6901924568089ac316986bc0ae713e89314ffc818d2ae0c94d576d.
992.5963s outer process scope versus control764.2278s; these include startup/
sealing/Drive and are not isolated graph/recon costs. ActualM100 started
16:51:47UTC; no complete3armshard orfullaccuracy yet.

[Code/CPU] Final timingreader nowretains measured preprocessing, terminal
geometry/head/IO(combined), and fullcasewall(beforeDrive), preserving overlap
scopes. Baseline decoder/hash is separatelyUNMEASURED(null), not zero; baseline
spectral0is actual absentoperation. Decoder/PM remainwithinrecon total.
Realacceptedcontrol480JSONtimers parsed:preproc2.75415s,terminalcombined13.03175s,
casewall698.06322s. Full/Scan timing selfchecks pass; fullanalyzer
SHAa5db46a075a759705dc6ee0f66760c83a96306f62b9262135cc696494ab78373.
Native repeat gate remains failed; no new sampler or GPU math changes.

## 17:13UTC — first complete paired shard: NEGATIVE versus operational

[Run/Inference] All3first64-ID/all15/s5/B2 bundles accepted on Drive+Mac,
480pairedcases/960predictions perarm. PMoperational629/960(65.52083%),
half0619/960(64.47917%),M100621/960(64.68750%). M100-operational-0.83333pp,
pairedobjectCI95[-1.97917,+.31250]pp; M100-half0+.20833pp,
CI[-.93750,+1.35417]pp. This retains the unfavorable observed firstshard;
64/2468objects(2.6%) is not the full result and is not an early stopping rule.
Full39shardfixedcoverage continues; secondshardrunning. Nativevariability
excludedfromthese conditional intervals; no causal guidanceclaim.

M100raw1,004,501,023bytes/SHA
b641e5039ffef29d0ffbc137acd261e2fc8173dbb56cbbf298bcce03cfdc229e,
outerworker1077.6603s. Half0/M100 all480exactgraphbasis/filter/reference/support
identitiesmatch,960perobjectrankrecords perarm100–256;480basislosschecks perarm
pass. Togetherwithfirstcontrol allsource/native/model/actualcache/RNG/PM/
terminal/count/CSV guards pass. Derivedpartial JSON31,018bytes/SHA
79e7d07541733be574b04d89085e1edf25d2190b5a3361b994fbb584e58f7a4b;
CLIupload/download roundtriptoownDrive SHA verified.

[Code/CPU] Localanalyzer ranklowerbound corrected1→0 to honor existing
zero-support graph semantics (`test_empty_underflow_graph_has_connected_zero_loss`);
this does not change observed positive-rank results or GPU/math/numeric gates.
Pairedrank0 retained inhist/counts; -1/2049rejected; alpha calibration still
requiresactive finite norms. Full/Scanreader selfchecks and sourcezeroGraph test
pass. CurrentfullanalyzerSHA
e13ee21e8a48295e510e657ea5fae17a3a482679afbbc79a1e8adc20b94a7f44,
archivedentrypoint executedforcanonicalpartial.
