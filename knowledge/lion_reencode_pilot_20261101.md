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

## 2026-11-01 09:21 UTC — prepared and technical CUDA phase started

[Run/Code/Open] Canonical setup1/reconnect/2–10 complete on same A10080GB VM; only known Cell2fastrlock deferred. CUDA12.1/GCC11.4/Torch2.1.2+cu121 native GPU/data/weight verifier passed. Downloaded strict Point-MAE/no-forward/no-CUDA CPU deployment receipt validates all4 pinned native bindings, source manifest e202f5985c5ff9963f3ecabeb94ad0f8a8b220a40c39c9541ce0abeabc6f91dd, source05e599d. Scientific sourceZIP593d953c…3d1af3, payload3aa76cdb…611c84. Original source/native bytes preserved before pinned restore. Operator v1 had an obsolete session assertion caught in local audit before execution; v2 fixes it without scientific-source changes. Owned monitor28423 submits technical32-ID Gaussian/Background phase at09:21:43UTC. No accuracy accepted yet; shared scientific55min cap starts with this phase.

## 2026-11-01 09:26 UTC — technical smoke independently accepted

[Run/Code/Open] Initial smoke command failed only in CLI prelude; read-only audit proved no ledger/phase/process. Local monitor was restarted with documented REQUEST_TIMEOUT60, preserving the same VM; no runtime teardown. First-attempt wall09:21:43UTC is conservatively charged in the research ledger. Actual technical phase09:24:20UTC completes two32-ID Gaussian/Background batches; all first/second/pooled PM, frozen states, source, physical posteriors/Gaussians/RNG/frame/own-z/new-h-anchor and actual N3 scheduler checks pass in remote and independent local NumPy1 recount.25,178,429-byte raw ZIP SHA78c6748a1318a7edd673c151f959ad26e418ba0d2b22593e9fccdd0126d30223 validates Drive/local SHA/CRC/paths. Local acceptance is exactly equal to remote JSON. No tuning based on smoke; full registered64/all15 pilot follows. Mainsource identityc19849148f8f5ea12d65e999daaf814181ddb4d8337816be024e162c70d9abf2.

## Completed pilot — supersedes preceding pending/running states

[Run/Inference/Decision] [Final results](lion_reencode_results_20261101.md). Both technical and64/all15/seed0 phases independently accepted; second3 and pooled PM do not beat first baseline PM. Full raw335MB locally retrieved and verified. Runtime preserved; no additional GPU phase or teardown.
