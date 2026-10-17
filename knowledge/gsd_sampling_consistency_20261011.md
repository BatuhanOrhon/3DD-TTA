# Frozen-classifier sampling consistency on Legacy trajectories

## Authorization and source baseline

[User decision/Code] 2026-10-11: user approved implementation after explicit
literature attribution and recording the proposed test. Base is
`gsd-smooth-spectrum@a351e66ec58f08f3a2104e46cab2cb8b20cdf961` plus an explicitly
hashed working-tree source delivery. Preserve all unrelated edits, notebooks,
raw ZIPs, failures and M1800 evidence; no Git writes or local GPU launch.

## References and attribution

- [Paper] Ali Bahri et al., **Test-Time Adaptation in Point Clouds: Leveraging
  Sampling Variation with Weight Averaging**, WACV 2025, pp. 266-275.
  https://arxiv.org/html/2411.01116v2, sections 3.1-3.2, equations 1-3 and 5:
  FPS/KNN sampling variation, TENT normalization adaptation, weight averaging.
  Inspiration: sampling diversity. We preserve classifier weights, BN and
  token algorithm and do not implement their adaptation or weight averaging.
- [Paper] Marvin Zhang, Sergey Levine, Chelsea Finn, **MEMO: Test Time
  Robustness via Adaptation and Augmentation**, NeurIPS 2022.
  https://arxiv.org/html/2110.09506v3, section 3.1, equations 1-3, Algorithm 1:
  entropy of mean predictions across augmented views and parameter adaptation.
  Inspiration: confidence and view consistency; our selector does no adaptation.
- [Paper] Dan Hendrycks et al., **AugMix: A Simple Data Processing Method to
  Improve Robustness and Uncertainty**, ICLR 2020.
  https://arxiv.org/abs/1912.02781 and https://github.com/google-research/augmix:
  related precedent for Jensen-Shannon consistency during training. We do not
  implement AugMix transformations/training or attribute our selector to it.

[Inference/Open] The particular normalized H(mean)+JS selector over decoded
diffusion snapshots is our proposed synthesis, not a reproduced paper formula,
established novel contribution or proven improvement. Mean entropy already
contains disagreement information; explicitly ablate the additional JS term.

## Locked experiment

[Proposal] Legacy B70/M800, seed0, severity5, common10-step macro14 only;
Background excluded from new runs and preserved in old reports. Keep beta2,
spectral1.17, Chamfer1, gamma/eta .01, lambda .95, updated style, checkpoints
3/5/8 and original terminal. No threshold, tau, model/BN/token adaptation,
early stop, horizon search, PG or guidance removal.

For each of four decoded candidates, normalize the whole cloud once and keep
the original 1024-point FPS sample as A. For B, choose a deterministic nonzero
start index from (seed, original sample ID, view number), swap it to index0,
run the unchanged native FPS/gather, then map selected indices back to original
cloud indices. Assert native FPS starts at index0. Sampling changes coordinates
neither by jitter nor by normalization per subset. Views may overlap; record
indices and subset overlap, do not resample until a desired disagreement appears.
RNG/buffer isolation protects subsequent diffusion and original A logits.

Terminal has A plus seven predetermined alternate views. Its first alternate
is the same B used in the two-view method. Acquisition therefore uses14
classifier forwards (three clean candidates x2 + terminal x8), with no extra
diffusion or decode for the additional views. The two-view method uses8
forwards; terminal voting uses8. Acquisition runtime is not method latency.
Record synchronized component times, actual forwards and peak memory; a later
matched B1 benchmark is required for deployment latency/speed claims.

Policies from the same trace: terminal A; four-A equal mean; eight-view equal
mean; minimum normalized H of each two-view mean; minimum H+JS; terminal
eight-view equal mean. H is divided by log40, two-view JS by log2; weights1
are a declared design choice, not an optimized setting. Ties prefer terminal,
then earlier clean candidate. Labels score only; corruption names never route
sampling or prediction decisions.

## Test and launch gates

1. CPU: deterministic view indices independent of batching/global RNG;
   exact identity for A/native sampling; bounds/start mapping; overlap cases;
   finite normalized entropy/JS, agreement and confident-disagreement cases;
   tie rules; frozen buffers/RNG including exceptions; saved-logit analysis,
   hash/coverage rejection; default-off regressions; CLI preview and source
   delivery checks. No local GPU inference.
2. Colab smoke: Gaussian, seed0,140 predeclared original IDs, B70, same10-step
   host. Verify native FPS start, finite logits, source/assets, complete views,
   point indices, RNG/buffers and post-model-state identity. Cross-process
   CUDA logits equality is not a gate (previous variability is documented).
3. Separate user-run pilot after smoke gate: all14 x280 predeclared IDs, seed0.
   ID selection uses the existing label-free shuffle seed20261011; no new
   selection based on labels or correctness. Do not repeat the old all15 run.
4. CPU report: paired corrected/broken counts versus same-trace terminal A
   and four-A mean, per-corruption and macro14, sampling overlap and selected
   candidate histograms, actual measured acquisition cost. Incomplete runs
   are packaged for recovery but do not pass analysis. Fresh output paths,
   archive SHA and verified Drive backups; user executes via
   `conda run --no-capture-output -n 3dd_tta_env python ...`.

[Falsifier/Open] Reject the extra reliability mechanism if H+JS does not add
benefit over equal eight-view averaging or H alone. Terminal votes test whether
extra classification alone explains gains. Accuracy remains exploratory on
previously reviewed ModelNet40-C; freeze a supported candidate before seed1/2
repeatability and independent evaluation. No new accuracy/runtime claim yet.

## Implementation and CPU verification

[Code/CPU] Added `trajectory_sampling.py` (explicit external FPS starts,
isolated view collection, six label-free policies),
`scripts/analyze_trajectory_sampling.py` (strict source/artifact/coverage/model
state validation and paired reports), and `scripts/trajectory_sampling_colab.py`
(source snapshot, fresh session, preview, separate smoke/pilot, partial recovery,
verified Drive copy). Opt-in flags are `--trajectory_sampling paired` and
`--corruption_scope macro14`; default execution remains unchanged. Runtime
source is explicitly identified as pinned base plus hashed working-tree
overrides, not misreported as a clean Git revision. Source delivery requires
no commit/push. Sample-ID manifests are hashed at initialization and locked.

The local pointnet2 CUDA source starts FPS at index0
(`Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu`,
line85). Each actual native call must also satisfy that runtime assertion.
Starts are generated using SHA-256(seed:original_id), a local deterministic
shuffle of nonzero indices, and its fixed prefix. The B start therefore agrees
between the two-view method and the eight-terminal-vote control regardless of
batch boundaries. Points are swapped only to specify native FPS initialization.

[CPU] 66 targeted tests pass: new view determinism/mapping, H versus H+JS,
RNG/buffer restoration including failure, end-to-end synthetic artifact analysis,
original A logit equality, source/indices tamper rejection, partial packaging
and backup, plus Legacy numerics/observer/decoder/trajectory regression. The
real Legacy CPU loop includes a paired-view arm and preserves original guidance,
decoder timing/style, output and RNG. The existing AST equality guard now
removes only the explicit optional sampling branch and its context variable
before comparing the default body with the historical source.

Colab cells are appended under **Sampling consistency: macro14 seed-0 smoke
and pilot** in `scripts/gsd_guidance_ablations_colab.ipynb`. Existing preparation
must already provide `/content/3DD-TTA`, its data/checkpoints and `3dd_tta_env`.
The new preflight uploads a hashed source ZIP to a fresh directory, links
read-only-used assets, checks A10080GB-class capacity, initializes a new pointer,
and previews140 Gaussian smoke examples. Do not Run all or rerun old pilot/full
cells. After smoke, download/provide the complete handoff ZIP for review before
the separate3920-example macro14 pilot cell. Inference is user-operated in
Colab. A normal failure is packaged in `finally`; use recovery if runtime is
still alive. A hard runtime reset may lose local partial files.

[Open] Native CUDA execution, true per-example coordinate diversity, accuracy,
acquisition time and peak memory are not yet measured. Index-set overlap is
reported; different indices may still have duplicate coordinates. First-run
native extensions may compile in the fresh source directory. Full-data runs,
seeds1/2 and matched B1 deployment timing remain conditional later work.


## 2026-10-12 - Final asset-provenance gate

[Code/CPU] The standalone CPU analyzer now checks recorded model config,
Point-MAE/LION checkpoint, label and selected-corruption data SHA-256 and byte
counts against the accepted Legacy baseline specification. The specification
itself must match the delivered source manifest. Relocated paths are accepted
only with identical content identities. Missing or changed identities fail
before accuracy is scored; raw assets are verified by the Colab launcher.

67 targeted CPU tests pass, including negative model/data/label/specification
checks and the saved-trace end-to-end analysis. This supersedes the earlier
66-test count. Source delivery v2 supersedes v1; retain v1 for provenance.
No GPU experiment or accuracy/latency improvement is claimed.
