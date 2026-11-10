# Terminal acceleration and same-logit LM–PM pilot

## 2026-11-10 — active preparation

[User decision/Code/Open] Execute the [Nov8 handoff](terminal_logit_pilot_agent_handoff_20261108.md)
without another GPU approval question. No full-data expansion. On runtime or
connection loss, preserve evidence and WAIT_USER; no replacement, retry or new
remote cell while state is unknown. Runtime teardown is not authorized.

Owner: terminal/logit pilot agent. Worktree
`/Users/batuhan/.codex/worktrees/c117/3DD-TTA`; branch
`experiment/terminal-logit-pilot-20261110`; fetched base
`bb1526d461081a0cae5401dbfee3c5745624cb1a`. Other checkouts and runtimes are protected.
Evidence labels: [Paper], [Code], [Run], [User report], [Inference], [Open].

### Frozen design and implementation sequence

The user's execution request adopts the five-arm design in the handoff; no
additional design approval is pending. All15 ModelNet40-C corruptions, s5,
frozen40-class Point-MAE, seed0,32 common IDs/B32, three fresh operational
repeats. Repeat0 primary, repeats1/2 reported separately. Two fixed warmups
per arm/corruption. Smoke uses four disjoint IDs/B2 on Gaussian, Background,
Density and LiDAR. IDs come from one default_rng(20261108) permutation before
accuracy, stored in [registration](evidence/terminal_logit_pilot_20261110/registration.json).

T5/P5 and T8/P8 are matched terminal comparisons. Classic uses real `tta.py`
with rates(.01,.01)/(.03,.003); cheap uses tau.1/retained.95, no backward or
style update. P8-PM alone reads clean2/4/6+terminal8; offline LM and PM share
its B×4×40 logits and float64 arithmetic. Saved GPUfloat32 PM differences are
counted separately. No baseline defaults change.

- [x] Extend `lion_guidance_pilot.sample(..., terminal_only=True)` to N5 and
  omit intermediate readouts/snapshots; retain old N8/35 defaults.
- [x] Add bounded runner and CPU reducer; test count/grad/style/policy/finite/ID
  guards and same-logit reductions; run impacted suite and independent review.
- [ ] Seal source/registration/actual weights/data/native receipts before GPU.
- [ ] Allocate a new uniquely named High-RAM A100; canonical cells1–10 with
  exact markers; only known cell2 fastrlock exception may be deferred.
- [ ] Technical smoke: actual scheduler/update and1/4readout counts, frozen
  model hashes, finite outputs, three same-state/RNG repeats; disclose failures.
- [ ] Pilot with 45minute soft science budget; initial estimate15–45minutes,
  revise after smoke. Fresh preprocessing/encode per timed arm, balanced order,
  synchronized runtime and allocated/reserved peaks; exclude loading/warmup/
  RNG seeding/hashes/D2H/artifact I/O equally. B32ms/input is not B1 latency.
- [ ] Checkpoint each corruption, verify immutable rawZIPs Drive/Mac SHA/CRC,
  then report counts/all15/macro14/per-corruption, paired corrected/harmed/net,
  object-cluster10000draw/seed20261108 CIs per repeat and logit diagnostics.
- [ ] Reconcile latest integration and normal fast-forward push only necessary
  code/tests/knowledge/compact evidence; exclude rawZIPs, setup wrappers/weights.

[Open] Source commit/hash, Colab session/endpoint, Drive/local result roots,
actual setup/native/asset identities and measurements are pending. No new
accuracy, latency, repeatability or causal improvement is claimed. Small exposed
cohort/one inference seed and native variation limit conclusions. No −.3pp
noninferiority proof is expected from32objects.

[Code/CPU] Preparation checks:13 new pilot tests pass, including a complete225-case synthetic grid with known same-logit LM/PM decisions and changed-ID rejection. Latest broad project run:887 passed/2skipped/564subtests (the subsequent full-grid fixture separately passes). Python3.8 syntax and diff checks pass. Independent review identified source-seal binding and missing host-contract tests; both corrected. Duplicate cheap-only discarded style projection removed; eval global2style is pure and called once by each sampler.

Registration identity: `{"bytes": 1801, "sha256": "ac18925b95bf88d5ef4e95c7b16ea4b9fd780a579a058fadfed4d3c2a995e3b7"}`. Accuracy not inspected when IDs selected.

[CPU/Open] Unrestricted root `pytest -q` additionally collects unsupported vendor tests and fails collection in Pointnet2_PyTorch/tests (hydra absent), extensions/emd/test_emd_loss.py and third_party/PyTorchEMD/test_emd_loss.py (emd_cuda absent), and third_party/ChamferDistancePytorch/unit_test.py (Python3.12 importlib.find_loader removed). Raw log: `result/preparation/terminal_logit_pilot_20261110/root_pytest.log`. Project `pytest tests -q` passes; no Linux/CUDA model environment was installed on Mac.
