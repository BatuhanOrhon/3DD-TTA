[User report/Run/Open] 2026-10-30: APPROVED by “Tamam devam et”.
Originalr1 unexpectedly vanished13:04UTC before this reply or any research
GPU work. Same scientific plan/source/IDs/10800sbudget now recovers inr2;
fresh ten-cell setup and actual CPU/Drive proof must precede GPU launch.
The original preparation/approval package below is retained as dated history.

# Planned Chamfer depth study — approval package

Status: READY FOR DURATION APPROVAL. Canonical setup and actual CPU/native/Drive preflight passed; local/Drive archive SHA+CRC and every member identity agree.
No research GPU launch or duration approval exists yet.

Use frozen Point-MAE and original raw/eval LION on ModelNet40-C severity5, all15
corruptions, batch32. Probability mean is the primary readout. Background35
stays fixed. Gamma updates the local latent; eta updates conditioning style;
lambda/p is the retained Chamfer fraction. Preserve original decoder style,
DDIMfinalalphaTrue, unnormalized original Chamfer and EMAoff.

| Stage | Fixed scope |
|---|---|
| Technical smoke | N1/5/10, Gaussian+Background, same32 dev IDs, seed0; no selection |
| Development A | N1..10, p=.95, eta=gamma=.01, same128 IDs, seed0 |
| Development B | Best2 depths, eta×gamma={.003,.01,.03}²;16 new settings |
| Development C | Each selected depth's best rates, p=.90/.96/.99;6 new settings |
| Confirmation | Depth5/default, best-depth/default, best tuned; deduplicate; same256 reserved IDs, seeds0/1/2; up to9 jobs |

Estimate9360s (~2.6A100hours), cumulative measured stop-before-next-worker
budget10800s (3h), including technical smoke, failed work, per-job validation,
probes and Drive copies. Setup and idle VM lifetime are separate; this is not a
billing estimate. Post-worker CPU/Drive copying can overrun; record it and do
not start more GPU work. Numerical failure, timeout, budget exhaustion or
source/native/data/checkpoint mismatch stops for review; no failed replacement.

Local implementation checkpoint: ca87a6a on experiment/chamfer-depth-tuning-20261030.
Executed source is a recoverable sealed snapshot, not a remote Git checkout.
Manifest records4da29b3 plus exact dirty overlays at sealing.
Source SHA256:166df3799834bc72541f1189407bf56d992a23bfdce9827db1c422d9ba5b3636.
Manifest SHA256:30552944975518231718ceec7dca183202f9b368d43892342ce9490fb01705da.
Plan SHA256:3d6e42dbfc49966fc8d9b5ed79a133355a06dac63bde8e061a6fe624dfb1e8a6.
Point-MAE SHA256:507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75.
LION SHA256:807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2.
Exact dataset/file and selected-ID hashes are in the registered plan.

Session: tta-chamfer-depth-tuning-a100-20261030-r1, A100HighRAM,
endpoint gpu-a100-hm-euw4a0-22m9t1i56rxn3, monitor59557.
Actual no-forward CPU checkpoint/native/data proof and a real Drive roundtrip
are mandatory before any research subprocess.

Executed repository root:
/content/studies/chamfer-depth-tuning/prepared/assets/source_snapshot

The exact controller is result/preparation/chamfer_depth_tuning_20261030/campaign.py.
For each registered phase use the environment's Python3.8:

```sh
/usr/local/envs/3dd_tta_env/bin/python -u \
  result/preparation/chamfer_depth_tuning_20261030/campaign.py run \
  --plan result/preparation/chamfer_depth_tuning_20261030/plan_v1.json \
  --stage <registered-phase.json> --scores <accepted-development-scores-if-needed> \
  --approval <actual-user-approval.json> \
  --root /content/studies/chamfer-depth-tuning
```

Fixed initial phases are stage_smoke_v1.json and stage_depth_v1.json. Rates,
fraction and confirmation manifests are derived only by the registered stage
selector from accepted development results; freeze before confirmation.
Worker is scripts/eval_chamfer_depth.py; the plan binds all phase sources.

Raw worker bundles go to the immutable result/modelnet40_c/3dd_chamfer_depth_study
hierarchy under each fresh slot. Drive destination is
MyDrive/thesis/result/chamfer_depth_tuning_20261030. Preserve successful/failed
raw bundles, logs/config/commands/environment/predictions/shards, actual sources,
physical paired latent/noise/RNG cache, selections, freeze, ledger and verified
Drive receipts. Retrieve through CLI to a fresh local result path and verify
SHA/CRC before interpreting; never overwrite a failed attempt.

This identifies a finite, model/dataset-specific operating point, not a global
optimum or externally unseen benchmark improvement. Varying N changes initial
noise, update count and probe positions; N1/2 have fewer distinct votes.
Repeated seeds retain256 object clusters, not768 independent samples. Cache
replay timing omits encoding; it cannot alone prove fresh end-to-end speedup.

Actual CPU preflight archive:1450667bytes, SHA256
5422f43ff4fb31bacda65ed2318ae1863f301f6147a98a91034b8c30c44df44e.
StrictPoint-MAE load has zero missing/unexpected keys and frozen CPU state
7b212d65e82d3b20dc2476bdb99df30f4ab5d25d69b8cac24d7c2cf1f80b875e;
forward_executed=False and cuda_initialized=False. Actual A100SXM4-80GB,
Python3.8.20/Torch2.1.2+cu121/CUDA12.1/NumPy1.21.2.
First exact notebook-phase launcher: operator/run_smoke_phase.py
(relative to the study preparation directory), SHA256 12e92e942bc03f0aab641d553eb0ad926055b081a4d4dee80e407b7dbfbc57a8.
It invokes only the registered smoke through campaign.py; absent/mismatched
user approval, source/preflight failure or reused slots refuse GPU launch.

[Run/Decision] 2026-10-30T18:04:40.252268+00:00: approvedcampaigncomplete, finalchargedrootledger10073.248667657s<10800s. All32development/9confirmation/3smoke attemptedonce;5numericfailurespreservedunderexplicitamendment. No furtherGPUrunrequired. Userexplicitlyauthorizesownedr2closureAFTERvalidation/archive,withoutnewpermission; peerprotected. Seeacceptedresultsnote.

## Owned runtime closed after verified completion — 2026-10-30T18:06:04.631230+00:00

[Run/User report/Decision] Allnecessaryresearchandindependentvalidationcomplete. Finalderivedanalysis/knowledge ZIP4244184bytes SHA da3653f7b0d386d1e8b84002013d6c14eb8ea87e4ef7cbb7d72c953f9bee7731 wasuploadedtoDrive`phase-evidence/final_local_analysis_20261030T1806Z.zip`,downloadedbackandbyte/SHAverified18:05:23UTC. FullrawDrive/localZIPandall7267members/422cachesalreadyverify. User'sexplicitconditionalcompletion-and-closeinstructionwasappliedwithoutanotherpermissionprompt.

[Run] Solemonitor31219 stopped18:05:55UTCwithoutteardown; `colab stop -s tta-chamfer-depth-tuning-a100-20261030-r2` returned0andSessionterminated. Freshserverinventory18:06:04UTCshowsowned79r779o7n5fkabsent,protectedguidance-variantspeer3nh8smksm50destillpresentunchanged. Nopeer runtime/process/monitoroperationwasperformed. Closure/precondition/CLIstdout/before-afterinventoryreceiptsareinresult/preparation/chamfer_depth_tuning_20261030/recovery_r2/final/. NoGPUtaskremains. Earlierlive/pendingentriesarepreservedhistory,supersededbythisclosure. ScientificresultstaysselectedtunedN8PM68.741319%vsN5PM66.510417%on256reservedobjects/threefixedseeds,notfullbenchmark/globaloptimality.
