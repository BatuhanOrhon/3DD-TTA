# Two-pass LION reencoding pilot — 2026-11-01

[User report/Decision] User authorizes tests after the method discussion, retains the full baseline first horizon, and delegates the second horizon (3 or5). Choose3 for a bounded first test; no search or entropy tuning. First pass is the operational N8/Background35 baseline, not the first5 iterations or a fresh N5 schedule.

## Approaches and selected scope

Earlier z-only refresh retains the original h and does not test this hypothesis. Selected: decode the complete first pass in the LION2048 frame, reencode both global z and local h conditional on new z, apply a fresh3-step local DDIM with matching initial noise level. Keep global diffusion off. No multiple cycles or new training.

ModelNet40-C/all15/severity5/64 previously exposed IDs/seed0/Point-MAE/B32. Technical gate first: same32 IDs Gaussian+Background; pilot runs only after physical/source/state/RNG/PM acceptance. First gamma=.03 eta=.003 p=.95 total100 finalalpha1 raw/eval/EMAoff; preserve all rates for second3. Fresh h1 XYZ is the second SCD reference (intentional whole-policy change, not isolated z gain). Use common explicit eps_z/eps_h and local diffusion Gaussian between passes; posterior means/sigmas may change. No classifier-frame FPS/normalization before reencoding.

Readouts: first-pass terminal and PM; bridge decode-reencode-decode terminal as cycle-loss diagnostic; second terminal and equal PM on clean1/2/3+terminal; pooled equal probability mean over four candidates from each pass (eight total, equal phase weights). No entropy weighting/selection or label-dependent policy. Clean3 and terminal3 are distinct pre-guidance and post-guidance candidates. Primary comparison second PM versus first PM; combined PM secondary; terminal/cycle diagnostics descriptive. Report latency of the full two-pass pipeline, including probes and extra encoder/decoder; do not compare replay costs as deployment gains.

Persist original prepared states, explicit posterior Gaussians, bridge XYZ, new posterior parameters/selected z+h/noise, cycle logits, both four-candidate logits, scheduler/source/frozen-state/RNG guards. Recount probabilities and paired object-cluster intervals independently;64 objects, not960 independent events. No full benchmark/seed stability claim.

Own runtime tta-lion-reencode-a100-20261101-r1; endpoint gpu-a100-hm-ass1c0-1d6llui8hqweg; branch research/lion-reencode-20261101, base dbbdbee. Scientific phase estimate10–20min, conservative55min cap for smoke+pilot+failure handling/remote evidence. Setup is separately canonical10 cells; only known cell2 fastrlock can be deferred. Preserve peer/unknown runtimes. No teardown without explicit completion+close instruction.

Local result root result/preparation/lion_reencode_20261101; Drive MyDrive/thesis/result/lion_reencode_20261101. Stop on nonfinite tensors, source/state/cache mismatch, failed setup/technical gate, cap or unknown remote execution. Preserve failure bytes and avoid duplicate jobs.

## 2026-11-01 setup/CPU checkpoint

[Code/Run/Open] Owned A100 High-RAM allocated; Cell1 separately restarted and same endpoint reconnected. Initial Cell2 CLI failed in prelude; read-only audit proved no repo/environment/installer, so same-cell retry was safe. Cell2 completed Conda environment but its only pip wheel failure is the already-deferred fastrlock; Python3.8.20 verified. Drive consent/setup3–10 still pending; no scientific GPU inference.

CPU tests:12new contracts pass; complete project tests/ 640pass,1historical skip,412subtests. Bare pytest also collects unsupported vendored hydra/CUDA and Python3.13 importlib.find_loader paths (four collection errors); no local model inference attempted. Independent review found late policy-module import would change final source seal, and unchecked cycle logits could admit nonfinite diagnostics. Both have RED→GREEN regressions and were fixed before any GPU inference. Existing production baseline files unchanged.
