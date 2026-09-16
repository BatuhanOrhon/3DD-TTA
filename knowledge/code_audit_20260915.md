# Main / LION accuracy-path audit - 2026-09-15

## Scope and provenance

Static diagnosis and reanalysis of existing artifacts; no new GPU evaluation or
accuracy-improving implementation. The user runs the approved source-only
severity 1--5 probe in Colab while this audit proceeds.

- **[Code]** Active branch `baseline-repro-clean`, HEAD
  `262f3a668b3f5a7bc44c6282c4a8a2723ac6f00a`.
- **[Code]** Local and remote upstream 3DD-TTA `main` both resolve to
  `107305fd7baf40b359f31c07d235599198be7324`; original LION local and remote
  `main` both resolve to `7711b3d185752eeb632d095494876e4de15f3195`.
  Remote refs were checked with read-only `git ls-remote` on this date.
- **[Code]** The pre-existing working change to `run_baseline.py` describes
  source-only runs in `notes.md`; it remains outside this documentation commit.
  All PDFs, ZIPs, caches and historical files are preserved.
- **[Paper]** Re-read 3DD-TTA Sections 3--4 / Algorithm 1 / Eq. 11--12 /
  Table 2 in the local 11-page WACV PDF; visually checked PDF pages 5 and 7
  (printed 1570 and 1572). Re-read LION Section 3 and normalization discussion
  in Section 5.1; visually checked local PDF pages 4 and 7. The local LION PDF
  has 19 pages; the separate NeurIPS supplement could not be retrieved by the
  web reader, so this is not an exhaustive supplement audit.
- **[Code]** Compared the 100 overlapping `.py/.cpp/.cu/.h` files under
  `models/`, `utils/`, `third_party/` in 3DD-TTA main with local LION, ignoring
  CRLF differences: 99 identical. The sole difference is
  `third_party/PyTorchEMD/cuda/emd_kernel.cu`, not called by the SCD TTA path.
  This comparison does not cover every dependency or checkpoint tensor.

The selected baseline remains raw LION + `--lion-eval-mode`, severity 5,
batch 32, gamma=eta=.01, lambda=.95, 100 DDIM steps and 5/35 reverse steps.
EMA remains an ablation. Source-data reproduction remains the first gate.

## 1. New priority: source-only FPS requests more points than exist

**[Run]** The accepted source bundle
`result/modelnet40_c/source_only/20260912-111822_source-only_seed0/`
records Density `[2468,649,3]`, Cutout `[2468,724,3]`, LiDAR `[2468,768,3]`.
Other files mostly have 1,024 points; Background has 1,075 and Upsampling 2,048.

**[Code]** `run_baseline.py` always calls `misc.fps(...,1024)`;
`utils_mate/misc.py:60-67` uses Pointnet2 indices and gather. The binding
`Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling.cpp`
allocates the requested count without checking `nsamples <= input count`.
Its CUDA loop runs to the requested count. It selects existing indices and
cannot synthesize new points. Thus successful 1,024-slot outputs for these
files contain at least 375, 300 and 256 repeated-index slots, respectively.
These are mathematical lower bounds, not measured Colab unique-index counts.

**[Code]** A second feature of the same FPS kernel is more subtle:
`.../_ext-src/src/sampling_gpu.cu:85-101` starts from index 0 and excludes
later candidates with `x*x+y*y+z*z <= 1e-3` (radius about .03162).
Therefore even requesting N points from an N-point input need not preserve
every input point. The first seed is an exception to that filter.
This absolute-coordinate filter also breaks translation/scale invariance of
candidate eligibility. Point-MAE uses the same FPS helper again to choose
64 group centers (`models_mate/Point_MAE.py:77`). LION's separate PVCNN FPS
kernel has no analogous magnitude filter in its candidate loop.

**[Inference]** Repeated points can change nearest-neighbor groups and classifier
features. This is a concrete source-path sensitivity, especially because the
largest source deficits include Density, Cutout and LiDAR. It is inherited
upstream behavior, so it does not itself explain why authors obtained different
results. The actual Colab binary must be checked: the archived extension hash
identifies a binary, not its exact build-source semantics.

**Next diagnostic / falsifier:** after severity results, inspect installed FPS
indices on Density, Cutout, LiDAR and a 1,024-point Gaussian control. Record
input N, unique input/output indices, repeat multiplicities, near-origin counts,
and the same statistics for 64 centers. Include small synthetic clouds in Colab
to distinguish origin filtering from index repetition. Preserve direct loading
and batch 32. If the loaded binary does not exhibit the source behavior, revise
the claim about that runtime. Any alternate FPS/padding implementation is a
separate ablation requiring prediction comparison; do not silently replace the
source-only comparator or its severity sweep.

Primary external implementation corroboration:
[Pointnet2 FPS source](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu).

## 2. Reconfirmed: final decoder consumes the old global latent

**[Code]** `tta.py:53,64-86` updates `style_cond`, but `:89-92` decodes with
the original `shape_latent`. Updated style affects later local-prior calls;
only the last style update has no downstream consumer. `global2style` is an
identity for the supplied empty `style_mlp` configuration
(`models/vae_adain.py:120-127`), so the proposed replacement is well-defined
for this checkpoint. Other configurations need their own style contract.

**[Paper]** Algorithm 1 lines 9--11 and Eq. 12 use the updated z0 when decoding
after the loop. LION's VAE `recont` also passes the encoding's conditioning
style to its decoder. This is a verified paper/code inconsistency, already
identified in `code_audit_20260913.md`, not a newly proven accuracy gain.

**[Code]** `models/latent_points_ada.py:253-271` reveals a useful limit on the
hypothesis: decoder output is latent XYZ plus `.01 * network_output` for this
config. Holding final local latent fixed, changing style affects that residual
branch. This factor does not bound classifier changes or network output size,
but cautions against predicting a large gain just from the missing consumer.

**History:** `158eae8` tried updated-style decoding; `72335a3` restored author
behavior. No archived matched experiment establishes the isolated effect.

**Test:** one shared eval/raw denoising trajectory, decode its final local latent
twice with original versus updated style; classify both. This permits a stronger
paired comparison than two independently sampled runs. Record both accuracies,
disagreements, decoder-output difference and style displacement. Snapshot/restore
NumPy RNG around the extra classifier call, because of Section 5 below. Full
Gaussian/Impulse files, seeds 0/1/2, followed by all-15 confirmation only if
justified. A null/negative paired result rules down accuracy benefit.

## 3. Reconfirmed: SCD scale and lambda differ from the paper

**[Paper]** Eq. 11 divides each directed retained-distance sum by the original
point-set cardinality; Section 4 setup reports lambda=.96 and both rates=.01.
**[Code]** `tta.py:75-77` sums retained squared distances without these
denominators; CLI lambda defaults to .95. With 2,048 points per latent cloud,
the sum loss and its gradients are 2,048 times the point-normalized formula
at the same state. A batch-wide mean would introduce an additional B and
retained-count factor and would not implement Eq. 11.

**[Inference]** Strong guidance could preserve corruption; very weak guidance
could lose identity. Neither direction is an established improvement.

**Separate tests:** (a) current loss with gamma=eta=.01/2048
(`0.0000048828125`), which matches the point-normalized update scale at fixed
point count; (b) lambda .95 versus .96 at the current rates; (c) zero guidance
as a mechanism control if the scale probe is informative. Do not combine these
factors. The zero-guidance diffusion path still perturbs/denoises and is distinct
from direct VAE reconstruction. CLI gamma updates local state, eta style;
the paper names them oppositely, which matters only when unequal.

## 4. Missing attribution control: preprocessing without LION

**[Code]** Source-only is direct load -> FPS(1024) -> Point-MAE. TTA adds
bbox normalization, interpolation to 2,048, scaling/rotation, LION, inverse
rotation, output bbox normalization and FPS(1024)
(`main_3dd_tta.py:89-114`). Therefore TTA-minus-source combines preprocessing
and generative adaptation effects.

**[Inference]** Before assigning a gain/loss to LION, measure a preprocessing
identity control that follows that same preprocessing chain and bypasses the
LION call, followed separately by pure VAE encode/decode. This can reveal
whether normalization/resampling explains part of the classifier change.
If required, split bbox normalization from interpolation in subsequent controls.
This does not modify the canonical direct-loading source comparator.

**[Open]** The constant 3.3885 has no provenance in the examined wrapper.
The all55 YAML contains both `normalize_global: true` and the historical
`datasets.neuralspline_datasets` type; the latter module is absent from the
public local LION checkout. LION Section 5.1 distinguishes globally normalized
PointFlow from per-shape-normalized ShapeNet-vol data. Consequently, the current
PointFlow loader alone cannot prove the exact all55 training normalization.
Obtain checkpoint-specific statistics before calling the fixed scale wrong.
Changing scale also changes network geometry and guidance magnitude.

## 5. Newly traced reproducibility issue: unused classifier masks consume RNG

**[Code]** `classification_only(...,only_unmasked=False)` calls the MAE encoder
with `noaug=False` by default (`models_mate/Point_MAE.py:484-486`). Its forward
creates a random mask before choosing the all-token branch (`:304-355`). Active
config has mask_ratio=.9 and 64 groups; `np.random.shuffle` consumes RNG per
example even though that mask does not select the classification tokens.

**[Inference]** This explains how classifier calls can alter later NumPy-driven
interpolation in the TTA loop. It does not make the current source-only logits
depend on the unused mask. Do not present removing it as an accuracy repair:
the first direct predictions should match, but later TTA batches can receive
different interpolated inputs if the RNG stream changes. For paired diagnostics,
cache prepared inputs or isolate/restore RNG state, including extra diagnostic
classifier calls. Merely matching the initial seed is insufficient.

## 6. Lower-priority limitation: partial coordinate gradients in PVCNN

**[Code]** `models/pvcnn2_ada.py:176` detaches voxelization coordinates;
`third_party/pvcnn/functional/interpolatation.py:38` returns no gradients for
point/center coordinates, and `functional/devoxelization.py` similarly returns
only feature gradients. These are inherited LION operators. The guidance still
has direct latent and feature-gradient paths; it is not gradient-free.

**[Inference]** Training these layers' weights does not require all derivatives
needed when optimizing their input coordinates. Thus the computed guidance is
not necessarily the full derivative of the numerical geometry-dependent forward
map. A directional finite-difference diagnostic over several small step sizes,
with eval/raw and fixed draws, could quantify this. Neighbor/grid changes are
non-smooth and must be monitored. This is an engineering-heavy hypothesis,
not an approved kernel rewrite or the next accuracy test.

## 7. Explanations ruled down by this audit

- Model time t+1 agrees with LION's 1..1000 indexing; no demonstrated off-by-one.
- `set_alpha_to_one=True` agrees with native LION's final alpha=1 convention.
  Original LION DDIM uses endpoint-inclusive spacing and default kappa=1,
  whereas 3DD-TTA uses deterministic DDIM. Copying that sampler wholesale would
  change both schedule and stochasticity; it is not a drop-in correction.
- In diffusers 0.11.1, `scale_model_input` is identity; no omitted scaling fix.
  `scheduler.step` defaults to eta=0. The TTA CLI eta is a guidance rate.
  See [versioned DDIM implementation](https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/schedulers/scheduling_ddim.py).
- LION's main paper describes mixed-score prediction, but this checkpoint config
  disables it. Enabling mixed prediction on weights trained without it is not
  justified. Posterior sampling is explicitly in both papers; posterior means
  would be an ablation.
- Latent layout is (B,2048,4); the XYZ slice is consistent with encoder/prior
  layouts and the Chamfer wrapper makes it contiguous.
- Classifier inference is eval + no_grad with no optimizer. The source artifact's
  nonzero `trainable_parameters` inventory counts requires_grad flags; it does
  not show that classifier weights were updated.
- Confirmed eval/raw improvement and inconclusive EMA findings remain as recorded
  in the inference-mode matrix; this audit adds no new measured accuracy gain.

## 8. Existing results help select diagnostics

**[Run]** Recomputed from the accepted source seed-0 directory above and the
three complete eval/raw archives under `result/modelnet40_c/3dd_original/`:

- `20260913-163817_3dd-original-all15-eval-raw-seed0.zip`
- `20260913-173819_3dd-original-all15-eval-raw-seed1.zip`
- `20260913-183818_3dd-original-all15-eval-raw-seed2.zip`

All three eval bundles use `b999a1eb809690a625c1075b205cb042b384443f`;
source uses `ef87692042e337cf628b9438bfc06a3e362b2e0b`. This is historical
cross-commit context, not a new matched intervention. Counts (15 complete rows,
2,468 examples each) were rechecked; raw files were read without extraction/edit.

| Corruption | Local source | Paper source | Source gap (pp) | Local eval/raw mean | Local TTA gain | Paper TTA gain |
|---|---:|---:|---:|---:|---:|---:|
| Density | 65.2350% | 75.1% | -9.8650 | 72.4878% | +7.2528 pp | +4.2 pp |
| Cutout | 62.2771% | 70.4% | -8.1229 | 70.5700% | +8.2928 pp | +4.3 pp |
| LiDAR | 19.9352% | 29.1% | -9.1648 | 30.7942% | +10.8590 pp | +13.1 pp |
| Gaussian | 51.2966% | 57.0% | -5.7034 | 74.6623% | +23.3657 pp | +22.1 pp |
| Impulse | 55.3485% | 58.8% | -3.4515 | 70.4079% | +15.0594 pp | +21.5 pp |

Paper values are rounded Table-2 cells. Subtracting source accuracies is a
descriptive aid, not causal adjustment for different protocols. Density/Cutout
support prioritizing the source path; Impulse remains informative for an
adaptation-specific probe (local gain is about 6.44 pp smaller than the paper's).

## Decision and Colab handoff

1. The approved source-only severity 1--5 ZIPs are complete and validated:
   75.8806%, 73.2739%, 68.5062%, 62.0205%, and 53.6899% for severities 1--5.
   The -22.1907 pp curve is descriptive evidence, but no tested level matches
   the paper's 57.6% source row; severity 5 remains the benchmark condition.
2. Because the source gap remains, prioritize installed FPS/index diagnostics and
   preprocessing identity, alongside checkpoint/data provenance. These controls
   require a separately scoped runner extension; current CLI does not implement
   them. Keep seed 0 initially for index inspection; use 0/1/2 for stochastic
   preprocessing/accuracy comparisons. Keep severity 5 and batch 32.
3. Queue updated-style paired decode as the first small TTA implementation
   candidate after the source-data gate is reviewed. Then test SCD scale and
   lambda separately. Start complete diagnostic corruption files at seeds 0/1/2;
   promote a supported candidate to all-15 before selecting any new baseline.
4. All are exploratory tests on the corruption test set. Predeclare the contrast,
   report negative/null outcomes, and lock any later benchmark configuration.
   GSD/PxP, other datasets, and EMA baseline promotion stay parked.

Every Colab evaluation must use `!conda run --no-capture-output -n 3dd_tta_env
python ...` and produce a fresh immutable seven-file run directory/ZIP per
condition/seed, following `result/README.md`. Request the full ZIP, including
command, config, environment, stdout, summary, per-corruption counts and notes;
omit credentials. The severity probe is recorded as `[Run]` evidence; no new
TTA accuracy gain is claimed here.
