# Findings Log

## 2026-09-13 - LION prior EMA inventory and opt-in loader

**Evidence:** [Run] `result/modelnet40_c/diagnostics/checkpoint_ema_inventory.json` and `checkpoint_sha256.txt`; [Code] `models/lion.py`, `main_3dd_tta.py`, `run_baseline.py`.

The checkpoint contains 462/462 prior EMA tensors matching the 462 prior model entries by count and shape. The VAE has no EMA entries. Config metadata reports EMA enabled with decay `.9999`. An opt-in, shape-validated `--lion-ema-mode` loader was added; default raw loading remains unchanged. VAE weights are never replaced. `py_compile` and `git diff --check` passed; no GPU inference was run.

**Decision:** Run eval+raw versus eval+EMA on complete Gaussian and Impulse severity 5 with identical seed, batch, scheduler, gamma, eta and lambda. Do not combine with legacy dropout mode. Repeat any positive effect at seeds 1/2 before changing the selected baseline.

## 2026-09-13 - First prior-EMA pilot: Gaussian and Impulse

**Evidence:** [Run] `result/modelnet40_c/3dd_original/20260913-154218_3dd-original-gaussian-impulse-eval-raw-seed0.zip` and `20260913-154546_3dd-original-gaussian-impulse-eval-ema-seed0.zip`.

**Protocol:** Commit `c91c1c3`; ModelNet40-C severity 5; Gaussian and Impulse; 2,468 examples/corruption; seed 0; batch 32; 100 DDIM / 5 reverse steps; gamma=eta=.01; lambda=.95; LION eval mode. Artifact manifests confirm identical Point-MAE and LION checkpoint hashes. The only intended CLI difference is `lion_ema_mode` false/true. EMA stdout confirms 462 prior EMA parameters loaded; both bundles are complete and traceback-free.

**Result:** Raw/EMA Gaussian: 74.3112/74.7974% (+.4862 pp, +12 correct). Raw/EMA Impulse: 70.2188/70.6240% (+.4052 pp, +10 correct). Two-corruption macro: 72.2650/72.7107% (+.4457 pp, +22/4936 correct).

**Interpretation/decision:** A modest, same-direction seed-0 pilot supports testing EMA further, but does not establish a baseline change: runs are seed-controlled, not common-draw paired, and cuDNN benchmark remains enabled. Repeat the exact pair at seeds 1 and 2. Do not merge EMA with legacy/train-mode LION in this confirmation.

## 2026-09-13 - Prior-EMA three-seed confirmation: inconclusive

**Evidence:** [Run] six complete raw/EMA Gaussian+Impulse bundles at seeds 0/1/2, listed in `ema_inventory_20260913.md`; all use commit `c91c1c3`, eval mode and identical model/data hashes.

**Result:** Macro raw/EMA by seed: 72.2650/72.7107 (+.4457), 72.3663/72.8525 (+.4862), and 72.6904/72.4878 (-.2026) percent. Three-seed means are 72.4406% raw and 72.6837% EMA: +.2431 pp with .3865 pp sample SD. Gaussian mean is -.0540 pp; Impulse mean is +.5402 pp.

**Decision (superseded):** The two-corruption result is small, seed/corruption dependent, and not common-draw paired, but is insufficient to rule out an effect across the remaining corruptions.

**Active decision:** Run all 15 corruptions for seeds 0 and 1 under matched `eval+raw` and `eval+EMA` conditions before accepting or rejecting EMA as a candidate. This is an exploratory two-seed screen, not a final benchmark claim.

## 2026-09-13 - All-15 three-seed EMA screen: small, non-robust aggregate effect

**Evidence:** [Run] six complete all-15 raw/EMA ZIPs listed in `ema_inventory_20260913.md`, commit `b999a1e`, seeds 0/1/2, severity 5, batch 32, eval mode, matching assets and settings except EMA flag.

**Result:** Raw/EMA macro by seed: 63.7061/63.8466 (+.1405), 63.8088/64.0546 (+.2458), and 63.9006/63.8250 (-.0756) percent. Means are 63.8052/63.9087: **+.1035 ± .1639 pp** sample SD; 70,862/70,977 correct over 111,060 examples. Largest mean gains: Background +.5808 and Distortion +.3647 pp. Largest mean declines: Impulse -.3241 and LiDAR -.2431 pp.

**Decision:** Keep EMA as a documented ablation, not the selected baseline. The small aggregate gain is seed/corruption dependent and non-common-draw paired. Next run legacy+raw for all 15 corruptions at seeds 1/2 to isolate the dropout/eval effect against current eval+raw results.

## 2026-09-15 - EMA provenance audit: code-backed, not paper-reported

**Evidence:** [Paper] local `lion.pdf` exact-term scan; [Code] original local LION repository `../LION/trainers/common_fun_prior_train.py`, `trainers/train_prior.py`, `utils/ema.py`, plus this fork's supplied `lion_ckpts/unconditional_all55_cfg.yml`; [Run] checkpoint inventory under `result/modelnet40_c/diagnostics/`.

**Result:** The LION paper text contains no exact `EMA`, `exponential moving average`, or `moving average` mention. The original training code does wrap the prior/DAE optimizer in an EMA optimizer, serializes its state as `dae_optimizer`, and swaps those weights into the prior around sampling when `cfg.ddpm.ema` is enabled. The supplied all55 configuration enables it with decay .9999. The checkpoint has 462 matching prior EMA tensors and no VAE EMA tensors.

**Decision:** Retain the EMA experiment as a code-derived fork ablation, not a paper reproduction claim. Its existing three-seed all-15 result remains +.1035 +/- .1639 pp and is not selected. Full audit: `ema_inventory_20260913.md`.

Append entries chronologically. Never delete negative or superseded results. Use exact run paths for **[Run]** claims.

## 2026-09-19 - Preprocessing identity control implemented

**Evidence:** [Code] run_baseline.py, tests/test_preprocessing_identity.py,
and result/README.md; no [Run] artifact yet. The implementation was committed
on top of `4096d4ec1426a8b2682f6fd2d45c9d094ad4c199` after local checks.

**Question:** How much of the source-only to TTA accuracy difference is
attributable to the TTA preprocessing/output chain rather than LION?

**Implementation:** Added the opt-in --method preprocessing_identity path.
It is locked to ModelNet40-C severity 5, all 15 canonical corruptions, batch
32, seed 0, complete evaluation, direct corruption-file loading and the
existing gamma/eta/lambda values. The path performs per-shape normalize,
the existing interpolation/upsampling to 2048, scale 3.3885,
rotate_pointcloud, rotateback_pointcloud, existing ModelNet output
normalization, FPS(1024), and frozen Point-MAE classification_only under
torch.no_grad(). It does not load or call LION, mutate data, alter FPS,
use EMA, scheduler, guidance, GSD, or PxP.

The runner reuses the existing immutable seven-file directory/ZIP contract.
Identity config records method, stage, preprocessing contract,
lion_loaded=false, lion_mode_policy=bypassed, final_decode_style,
point counts, scale, scheduler/spectral/projection fields, asset manifests and
the exact CLI. The canonical source_only branch AST is unchanged relative
to HEAD.

**Local verification:** [Code] py_compile passed for changed/affected
modules; the four-test unittest control suite passed; CLI scope guards,
identity transform order, and identity config serialization are covered.
No CUDA/model evaluation was run locally.

**Decision/Open:** The code is ready for the predeclared Colab run, but no
accuracy or causal conclusion is made until the complete ZIP is supplied and
validated. The result must be stored under
result/modelnet40_c/preprocessing_identity/ and contain exactly the seven
required files. A positive identity delta permits planning pure VAE
encode/decode next; a null/negative result sends the next one-factor checks to
updated final-style decoder, Eq. 11 SCD normalization, lambda .95/.96, and
RNG controls in that order.

## 2026-09-19 - Pure VAE encode/decode control implemented

**Evidence:** [Code] `run_baseline.py`, `tests/test_preprocessing_identity.py`,
and `result/README.md`; no [Run] artifact yet.

**Question:** After the positive preprocessing identity delta, how much of the
remaining source-only to 3DD-TTA difference is explained by VAE reconstruction
itself, without LION priors, diffusion scheduling, or guidance?

**Implementation:** Added the opt-in `--method pure_vae_encode_decode` path.
It is locked to ModelNet40-C severity 5, all 15 canonical corruptions, batch 32,
seed 0, complete evaluation, direct corruption-file loading and the existing
gamma/eta/lambda values. The path uses the same preprocessing/output contract as
the identity control, then calls LION VAE `encode` -> `decompose_eps` -> `sample`
with the encoded latents. It loads raw LION weights, forces VAE/prior modules to
eval mode for provenance, but never calls the priors; no EMA, scheduler,
guidance, GSD, PxP, alternate FPS policy, or dataset mutation is introduced.

The seven-file immutable artifact contract is reused. Pure VAE config records
`lion_loaded=true`, `lion_mode_policy="raw VAE eval; priors bypassed"`,
`vae_contract="encode -> decompose_eps -> sample"`, and `prior_used=false`.
The canonical source-only branch remains structurally unchanged relative to the
pre-implementation HEAD.

**Local verification:** [Code] the focused unittest suite passes 7/7, including
locked CLI scope, config serialization, exact preprocessing order, and the
absence of prior/guidance calls in the fake-VAE control. No CUDA/model
evaluation was run locally.

**Decision/Open:** The implementation is ready for the predeclared Colab run.
Do not infer an accuracy result until the complete ZIP is supplied and checked
for the seven required files, status/traceback, commit, checkpoint/data hashes,
total accuracy, and all 15 per-corruption rows. If pure VAE is near source-only,
the positive identity delta is primarily preprocessing-local; if it is much
higher, VAE reconstruction explains a larger share of the TTA delta. These are
[Inference] decision rules, not [Run] conclusions.

## 2026-09-19 - Preprocessing identity result: modest positive delta

Evidence: [Run] result/modelnet40_c/preprocessing_identity/20260919-140904_identity-s5-all15-seed0.zip; archive SHA-256 9348bcbf613b8e77ee1c758c7542b1e90cde74e59d1c8896a188669a3e969629.
Git commit: 4096d4ec1426a8b2682f6fd2d45c9d094ad4c199; branch baseline-repro-clean.

Protocol: Complete ModelNet40-C severity 5, all 15 corruptions, 2,468 examples/corruption, batch 32, seed 0, direct corruption-file loading, identity preprocessing chain, LION bypassed, frozen Point-MAE, FPS(1024), gamma=.01, eta=.01, lambda=.95. The ZIP passes testzip(), contains exactly the seven required files, has 15 complete rows, and stdout has no traceback/error signature. Classifier, label and all 15 data hashes match the archived source-only comparator; Colab records torch 2.1.2+cu121, CUDA 12.1, Diffusers 0.11.1, PointNet2 3.0.0 and an A100-SXM4-80GB.

### Result

Identity macro/micro accuracy is 55.0243% (20,370/37,020). The source-only severity-5 comparator is 53.6899% (19,876/37,020), so identity improves by +1.3344 pp (+494 correct). The seed-0 eval/raw 3DD-TTA context run is 63.7061%; identity remains 8.6818 pp below it and closes 13.32% of the 10.0162 pp source-to-TTA gap.

Identity is higher than source-only on 9/15 corruptions, equal on Upsampling, and lower on 5/15. Largest gains are Density Increase +8.1848 pp, Cutout +6.0373 pp, Density +3.6467 pp and Occlusion +2.8363 pp. Largest declines are LiDAR -4.6596 pp and Background -4.2950 pp.

## 2026-09-19 - Pure VAE encode/decode result: preprocessing-localized positive delta

**Evidence:** [Run] `result/modelnet40_c/pure_vae_encode_decode/20260919-144513_pure-vae-s5-all15-seed0.zip`; archive SHA-256 `ade0a5c0199be5cbbf6cd95b3deb5324a0572b16dd57dc9fc64d3c753dc0eb0d`.

**Provenance/validation:** The archive passes `testzip()` and contains exactly
the seven required files under one run directory. Config and CSV status are
`complete`; all 15 canonical corruption rows contain 2,468 examples; stdout
contains 15 corruption results and no traceback, error, exception, or OOM
marker. The run is branch `baseline-repro-clean`, commit
`3630e7903f1a888e0f015ef5bb4346bcdc99221e`, ModelNet40-C severity 5, batch 32,
seed 0, direct file loading, frozen Point-MAE, and raw VAE eval with
`encode -> decompose_eps -> sample`. `scheduler_config` is `{}` and no
`scheduler_class` or timestep list is recorded. Classifier, label, and all 15
data hashes match the source-only and preprocessing-identity artifacts; the
LION checkpoint hash is `807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2`.

### Result

Pure VAE macro/micro accuracy is **54.7947%** (20,285/37,020). Relative to the
source-only severity-5 comparator at 53.6899% (19,876/37,020), this is
**+1.1048 pp**. Relative to preprocessing identity at 55.0243%, it is
**-0.2296 pp**. Relative to the seed-0 eval/raw 3DD-TTA context at 63.7061%,
it is **-8.9114 pp** and closes 11.03% of the source-to-TTA gap. This last
comparison is contextual, not a causal estimate, because the TTA artifact is
from a different commit and random draw.

Pure VAE is higher than source-only on 12/15 corruptions and lower on
Background (-5.1864 pp), Shear (-0.9319 pp), and LiDAR (-3.2010 pp). Largest
gains are Density Increase (+6.6451 pp), Cutout (+4.7407 pp), Gaussian
(+2.6742 pp), and Uniform (+2.2285 pp). It is equal to identity on Upsampling
and below identity by 0.2296 pp overall.

**Decision:** [Inference] The positive but identity-near result localizes most
of the modest source-only delta to the preprocessing/interpolation/rotation/
output-normalization chain. Pure VAE reconstruction does not explain the
large remaining TTA gap and slightly reduces the identity result in aggregate.
This does not by itself prove that diffusion guidance is ineffective: pure VAE
and full TTA differ in more than one operation, and the available TTA context
run is not a common-random-number, same-commit comparison.

**Open/falsifier:** A matched common-draw comparison or a separately isolated
decoder/protocol control could revise the localization. Do not add GSD/PxP,
EMA, or another dataset on the strength of this single seed; preserve the raw
ZIP unchanged.

## 2026-09-19 - Pure VAE seed-stability diagnostic implemented

**Evidence:** [Code] `run_baseline.py`, `tests/test_preprocessing_identity.py`,
`result/README.md`, and `knowledge/repository_map.md`; no [Run] artifact yet.

**Implementation:** Added the opt-in `--method pure_vae_seed_stability` path.
It reuses the already validated pure VAE encode/decode implementation and is
locked to ModelNet40-C severity 5, all 15 canonical corruptions, batch 32,
complete evaluation, direct corruption-file loading, raw VAE eval, and the
same gamma/eta/lambda, checkpoint, data, preprocessing, and FPS contracts.
Only seed 1 or seed 2 is accepted; seed 0 remains represented by the archived
`pure_vae_encode_decode` run. The new path is a stochastic stability diagnostic,
not a new TTA method, and records
`seed_stability_reference="pure_vae_encode_decode seed0 archive"`.

No priors, EMA, diffusion scheduler, guidance, GSD, PxP, alternate FPS policy,
or dataset mutation is added. The existing seed-0 pure VAE and source-only
paths retain their locked validation rules. Local tests cover seed-1 acceptance,
seed-2 config serialization, and seed-0 rejection.

**Decision/Open:** The implementation is ready for two separate Colab runs,
one at seed 1 and one at seed 2. Combine their complete seven-file ZIPs with
the archived seed-0 pure VAE ZIP to calculate mean and standard deviation. Do
not interpret the one-seed `+1.1048 pp` pure-VAE delta as stable until this
screen is complete.

## 2026-09-19 - Pure VAE seed-stability result

**Evidence:** [Run] the complete archives
`result/modelnet40_c/pure_vae_seed_stability/20260919-162638_pure-vae-s5-all15-seed1.zip`
and
`result/modelnet40_c/pure_vae_seed_stability/20260919-163345_pure-vae-s5-all15-seed1.zip`.
Their SHA-256 values are `5593e519efa0a4d7a87fae8e39bb551a78cbdc260d00172f96cbd2bf41101490`
and `a5fe7d7b1b21736ce6403f0371f2cb022e4ab68a1003c0351fd8a1153210db67`.

**Validation/provenance:** Both archives pass `testzip()`, contain exactly the
seven required files, have complete status, 15/15 rows, 2,468 examples per
corruption, and no traceback/error/exception/OOM marker. Both record method
`pure_vae_seed_stability`, raw VAE `encode -> decompose_eps -> sample`, no
priors, empty scheduler config, and the same batch-32 ModelNet40-C severity-5
all-15 contract. Both record commit `e8be9a4e0ef6a0289fb0a749dd0ac72d1600d48c`;
the archives also record the Colab working tree as dirty, so the commit and
runtime source hashes are the provenance anchors. The filenames both end in
`seed1`, but the earlier timestamp `162638` records seed 1 in config and
command, while the later timestamp `163345` records seed 2; this discrepancy
is preserved and the raw files are unchanged.

### Result

Seed 0 (archived pure VAE control) is **54.7947%** (20,285/37,020); seed 1 is
**54.9379%** (20,338/37,020); seed 2 is **54.8082%** (20,290/37,020). Across
seeds 0/1/2, mean accuracy is **54.8469%**, sample standard deviation
**0.0790 pp**, population standard deviation **0.0645 pp**, and range
**0.1432 pp**. The mean is **+1.1570 pp** over deterministic source-only at
53.6899% (19,876/37,020).

The paired pure-VAE minus preprocessing-identity differences are **-0.2296 pp**
(seed 0), **-0.1297 pp** (seed 1), and **-0.1405 pp** (seed 2). Thus VAE
reconstruction is consistently below the matched preprocessing-only control;
the one-seed localization is supported, although the absolute VAE delta is
still modest and stochastic.

**Decision:** [Run]/[Inference] The positive source-model effect persists in
the pure-VAE control, but the paired comparison assigns the aggregate gain
primarily to preprocessing rather than VAE reconstruction. This is not yet a
causal estimate of diffusion/guidance: the pure-VAE and available 3DD-TTA
context runs are not same-commit common-draw pairs. The next test should be a
same-commit, common-draw comparison between pure VAE encode/decode and the
operational eval/raw 3DD-TTA path, with preprocessing held fixed. Keep EMA,
GSD/PxP, alternate FPS policies, and other datasets parked. This was the
initial candidate; the later decision below defers it unless a stronger causal
thesis claim is required.

## 2026-09-19 - Common-draw control deferred

**Evidence:** [Run] the validated pure-VAE seed screen above, the validated
preprocessing identity seed screen above, and the operational eval/raw LION
screen documented in `knowledge/dropout_eval_mode_20260913.md` under
`result/modelnet40_c/3dd_original/`.

**Assessment:** The existing independent screens already establish the
operational ordering: source-only is 53.6899%, preprocessing identity averages
55.0135%, pure VAE averages 54.8469%, and the available eval/raw 3DD-TTA screen
averages 63.8547% over its two repeated seeds (with the historical seed-0
screen also archived). A common-draw run would not be expected to change these
accuracy values; its purpose would be to tighten the causal attribution of the
remaining 3DD-TTA gap by reusing one preprocessing realization in both branches.

**Decision:** [User report]/[Inference] Do not spend the next Colab run on this
paired control. Treat the result as an optional confirmatory experiment only if
the thesis requires the stronger claim that diffusion/guidance, rather than
unmatched stochastic preprocessing or VAE reconstruction, is causally
responsible for the remaining gap. Until then, report the preprocessing
positive effect and the consistently non-positive pure-VAE contribution, while
leaving diffusion/guidance causality [Open]. No common-draw code was retained,
committed, or pushed.

## 2026-09-19 - Preprocessing identity seed-stability diagnostic implemented

**Evidence:** [Code] `run_baseline.py`, `tests/test_preprocessing_identity.py`,
`result/README.md`, and `knowledge/repository_map.md`; no [Run] artifact yet.

**Implementation:** Added the opt-in
`--method preprocessing_identity_seed_stability` path. It reuses the validated
preprocessing identity chain with LION fully bypassed and is locked to
ModelNet40-C severity 5, all 15 corruptions, batch 32, complete evaluation,
direct files, seed 1 or 2, the same gamma/eta/lambda, data/checkpoint assets,
and FPS(1024) contract as the seed-0 identity archive. Seed 0 remains the
archived `preprocessing_identity` reference. The config records
`seed_stability_reference="preprocessing_identity seed0 archive"`.

This control is necessary because `upsample_all` uses NumPy random interpolation
and downsampling. It isolates preprocessing stochasticity separately from the
pure VAE seed screen, whose randomness includes both this preprocessing and VAE
latent sampling. No LION, VAE, EMA, scheduler, guidance, alternate FPS policy,
or dataset mutation is introduced.

**Decision/Open:** Run seeds 1 and 2 as separate seven-file ZIPs and compare
them with the archived seed-0 identity ZIP before interpreting the pure VAE
seed screen. The useful paired quantity is pure-VAE accuracy minus
preprocessing-identity accuracy at the same seed.

## 2026-09-19 - Preprocessing seed-stability routing failure and fix

**Evidence:** [Run] failed archive
`result/modelnet40_c/preprocessing_identity_seed_stability/20260919-164103_preprocessing-identity-s5-all15-seed1.zip`;
[Code] `run_baseline.py`, `tests/test_preprocessing_identity.py`.

The first Colab attempt reached the evaluation loop but failed before the first
corruption result with `AttributeError: 'NoneType' object has no attribute
'vae'`. Root cause: model setup recognized the new method as a LION-free
preprocessing route, but the batch-evaluation branch still matched only the
exact `preprocessing_identity` string and fell through to
`baseline.process_batches(..., lion=None, ...)`. The ZIP is preserved as a
failed/incomplete run and provides no accuracy evidence.

The fix centralizes the route predicate in
`is_preprocessing_identity_method()` and uses it for both model setup and batch
evaluation. The focused suite now covers both method identifiers; the failed
archive must not be overwritten or interpreted as a benchmark result.

## 2026-09-19 - Preprocessing identity seed-stability result

**Evidence:** [Run] the complete archives
`result/modelnet40_c/preprocessing_identity_seed_stability/20260919-165516_preprocessing-identity-s5-all15-seed1.zip`
and
`result/modelnet40_c/preprocessing_identity_seed_stability/20260919-165857_preprocessing-identity-s5-all15-seed1.zip`.
Their SHA-256 values are `a9afc629d81a8c1c08c2b7ab4b1b331d492f53d45ad82d60bbe2923599e7a358`
and `82720bc9378e4b235d31286007430d68c46af676f0ed08f95e572b3c34b56ae9`.

**Validation/provenance:** Both archives pass `testzip()`, contain exactly the
seven required files, have complete status, 15/15 rows, 2,468 examples per
corruption, and no traceback/error/exception/OOM marker. Both record commit
`9ad172848844467307427d7b7e1cbed72b3387d7`, raw preprocessing identity with
LION bypassed, empty scheduler config, matching classifier/data/label hashes,
and the same batch-32 ModelNet40-C severity-5 all-15 contract. The filenames
both end in `seed1`, but the earlier timestamp `165516` records seed 1 in both
config and command, while the later timestamp `165857` records seed 2; this
labeling discrepancy is preserved and does not alter the raw files.

### Result

Seed 0 (archived identity control, commit `4096d4e`) is **55.0243%**
(20,370/37,020); seed 1 is **55.0675%** (20,386/37,020); seed 2 is
**54.9487%** (20,342/37,020). Across seeds 0/1/2, mean accuracy is
**55.0135%**, sample standard deviation **0.0602 pp**, population standard
deviation **0.0491 pp**, and range **0.1189 pp**. The mean is **+1.3236 pp**
over deterministic source-only at 53.6899% (19,876/37,020); per-seed deltas
are +1.3344, +1.3776, and +1.2588 pp.

**Decision:** [Run]/[Inference] The preprocessing-only positive source-model
delta is stable across these three seeds at the aggregate level and is not
explained by a large interpolation-randomness swing. This strengthens the
finding that preprocessing itself raises source-only accuracy modestly. The
pure VAE seed screen remains necessary because it adds a second stochastic
source, VAE latent sampling; compare pure VAE and preprocessing identity at
the same seed before attributing any VAE contribution.

## 2026-09-12 — Initial repository and literature audit

**Evidence:** paper PDFs, current branch source, full Git history; no archived Colab artifacts.  
**Git branch/commit:** `pxp-gradient-projection` / `53ba252` at audit start.  
**Status:** mixed **[Paper]**, **[Code]**, **[User report]**, and **[Inference]** evidence.

### Confirmed from sources

- **[Paper]** Published 3DD-TTA ModelNet40-C mean is 65.7%; repository README says 66.1%.
- **[Code]** The original baseline path remains separated from GSD/PxP scripts.
- **[Code]** The main GSD path is a latent spectral regularizer, not full GSDTTA.
- **[Code]** Dynamic mode recomputes an eigenvector matrix/basis, not a single `U_0` vector.
- **[Code]** PxP variants project spectral and SCD gradients and therefore are PixelAsParam-inspired rather than direct implementations.
- **[Code]** Runs are stochastic and current output schemas do not capture enough provenance for definitive comparison.
- **[Code]** Loss reductions and scheduler/style choices have changed across commits, creating confounds.

### Provisional observations

- **[User report]** Original baseline is approximately 63% locally.
- **[User report]** A GSD variant reaches approximately 63.5%.
- **[Inference]** The first investigation should prioritize environment/checkpoint/data identity, seed variance, scheduler serialization, and metric aggregation before further method search.

### What would update this conclusion

A complete archived baseline run with per-corruption results, exact command/config, package/GPU environment, data/checkpoint hashes, and repeated seeds.

## 2026-09-12 — Follow-up code audit and clean restart decision

**Evidence:** [Code], [Paper], [User report], [Inference]; no new [Run].  
**Fork commit:** `53ba252519c7cf65f836a9c1c564027142ab1573`  
**Original LION commit:** `7711b3d185752eeb632d095494876e4de15f3195`  
**Run paths:** none; static audit only, not an accuracy experiment.

### Findings

- [Code] Original LION trainer inference disables VAE/prior dropout. Its demo wrapper does not; the fork wrapper is byte-identical. Current baseline/GSD setup sets only Point-MAE to eval. This verified discrepancy is inherited, not caused by GSD additions; accuracy impact remains [Inference].
- [Code] Scheduler, reverse steps, final style, batch and unequal gamma/eta semantics confound current baseline/GSD comparisons.
- [Code] Mean spectral MSE plus summed SCD changes relative guidance with actual batch/band/channel counts.
- [User report] Mean settings previously gave smaller reported spectral losses and higher accuracy than sum. Exact runs/configurations are not archived; a smaller mean number alone is not spectral-fidelity evidence.
- [Paper/Code/Open] Threshold normalization differs from GSDTTA Eq. (10). Keep visual/symbol/adjacency verification before correction; self-neighbors, distance units and isolated-node handling need small tests.
- [Code] Conditional missing-key failures, unsafe resume aggregation, symmetric CSV header/row mismatch, stale analyzer shapes and batch-flattened projection require targeted tests rather than wholesale legacy migration.
- [Code] SCD uses the first three local-latent channels; it is not computed after decoding. Earlier method/map wording is superseded and corrected.
- [Inference] Shared eigenvector sign flips/within-band rotations do not change complete-band same-basis MSE; changing projectors and band boundaries are the dynamic diagnostics.

Full source paths, wrapper hash, caveats and falsifiers: [code audit](code_audit_20260912.md).

### Decision and verification gates

The user endorsed an incremental clean restart from main. Local main/origin/main/upstream/main equal `107305fd7baf40b359f31c07d235599198be7324`. Proposed `baseline-repro-clean` is not created yet. Preserve legacy branches/user work; add artifact and source-only controls, then isolated dropout A/B, then lock the baseline ladder and require spectral-off parity. Sum plus smaller eta/gamma and dynamic/PxP tuning are deferred. [Small batches](clean_restart_batches.md) define scopes, tests, knowledge updates and Colab review gates.

This update changes documentation only. No branch switch or Python modification; no gain claimed. Gaussian/background seeds 0,1,2 under fixed configuration/common draws are next accuracy evidence. Repeated null/negative dropout effects weaken the gap hypothesis. A source-only mismatch redirects diagnosis to data/classifier; a failed spectral-off parity check exposes accidental method differences.

## 2026-09-12 — Batch 0 branch reference prepared, placement pending

**Evidence:** [Code] Git/ref/status checks; no numerical run.  
**Legacy checkout:** `pxp-gradient-projection` / `53ba252519c7cf65f836a9c1c564027142ab1573`  
**New branch/main:** `baseline-repro-clean` / `107305fd7baf40b359f31c07d235599198be7324`  
**Run paths:** none.

The user approved proceeding. Created the branch reference with `git branch baseline-repro-clean main`, without switching checkout, staging, commit or push. Ref equality and `git diff --exit-code` verified; pre-existing untracked knowledge/skill/protocol/PDF/notes remain. Worktree inspection found only the normal root checkout. Legacy tracked scripts and source PDFs would disappear from the folder on an in-place switch, though still retained on the old branch; asked the user for placement preference before this transition.

Batch 0 remains partial until safe placement, curated memory preservation and review. No Python changes, dependency install or local accuracy test. Keep Batch 1 on hold until the clean branch is actually checked out in its declared workspace. See `clean_restart_batches.md` for the next handoff. This supersedes the earlier status that the new branch did not exist; it does not supersede the audit's inference findings.

## 2026-09-12 — Batch 0 same-folder checkout and documentation preservation

**Evidence:** [Code] Git state, SHA-256 and parity checks; no [Run].  
**Active baseline source:** `baseline-repro-clean`, main `107305fd7baf40b359f31c07d235599198be7324`  
**Legacy experiments:** `pxp-gradient-projection` / `53ba252519c7cf65f836a9c1c564027142ab1573`  
**Run paths:** none.

User selected same-folder development and approved carrying skill/knowledge. Switched to the prepared clean branch. No tracked or staged code edits existed, so no new stash; existing dev stash `02ddaf533c93164d69643e43c55ad37df6fa0343` remains. Restored four absent scholarly PDFs from legacy Git history to local untracked files; did not overwrite the user's PixelAsParam PDF. All 23 selected memory/skill/protocol/PDF hashes matched across checkout before documentation updates. Tracked Python/requirements/environment match main; old variants are not imported.

Curated memory, researcher skill and result protocol are selected for a local documentation-only commit, with all PDFs and unrelated docs/tmp excluded. The containing documentation commit is recorded in the handoff rather than embedded as a self-referential hash. No push, dependency installation or numerical test. The prior placement-pending status is superseded. Next is Batch 1 after handoff review, not dropout modification yet.

## 2026-09-12 - Batch 1 baseline smoke artifact implementation

**Evidence:** [Code], not [Run].

**Base Git commit:** b31fd23193bbcb9a5c189cfb4118be41506f9333 on baseline-repro-clean; implementation commit reported in the handoff.

**Run paths:** none supplied yet. Expected result/modelnet40_c/3dd_original/<UTC-timestamp>_baseline-smoke_seed0/.

**Question:** can we produce an internally consistent, diagnosable original-TTA smoke bundle without changing adaptation math/modes?

Added run_baseline.py and research_artifacts.py, with optional read-only checkpoint/scheduler/batch observers in three baseline modules. The runner reuses the original preprocessing/TTA/classification path. Seeds and actual runtime flags/configs, checkpoint/data/source hashes, load incompatibilities, dropout/module modes, installed extension identities and count-based fraction CSVs are logged; subprocess capture retains Python/native stderr. Each invocation creates a fresh seven-file directory and a sibling ZIP, with partial/failed status rather than fabricated full accuracy. No resume, GSD, LION eval, scheduler/rate/style/reduction or dependency changes.

Local syntax/CLI and temporary artifact/count/macro-micro/duplicate/collision/failure-state checks passed. Observer-stripped ASTs of all three modified baseline files match main. The first AST comparison failed because its checker omitted a nested batch-observer branch; the corrected recursive checker passed. This was a verification-script limitation, not an adaptation-code change. No unit-test files were created per user request.

**Protocol:** pending Level 0 Colab smoke, Gaussian severity5, first two batches, seed0, explicit batch32 and repository lambda0.95; gamma/eta0.01, normal/background5/35, unchanged legacy modes. Runner default batch remains40. Same seed is not a guarantee of common-draw pairing.

**Decision:** request the complete smoke ZIP using colab_baseline_smoke.md; validate before Batch 2. No numerical result, dropout effect or accuracy-gap cause is inferred. A Colab runtime/import/schema/count failure or unexplained GPU behavior rejects the end-to-end handoff until resolved.

## 2026-09-12 - Chamfer import restoration for Batch 1 smoke

**Evidence:** [Code] plus [User report] Colab traceback; no successful run yet.

The Batch 1 smoke failed while constructing Point-MAE, before LION/TTA or classification. `models_mate/Point_MAE.py` referenced `ChamferDistanceL2` for `cdl2` but its import was commented out. The same import is active on the previously working `pxp-gradient-projection` branch. Git history identifies commit `8183863` as restoring it for Colab.

Decision: restore only the missing import. This is a baseline construction repair, not a TTA, dropout, scheduler or GSD change. Re-run the same Level 0 Gaussian two-batch command with seed 0. The new ZIP falsifies this diagnosis if it still reaches the same undefined-name error.

## 2026-09-12 - Batch 1 Gaussian smoke accepted

**Evidence:** [Run] `result/modelnet40_c/3dd_original/20260912-110541_baseline-smoke_seed0/` and its unchanged ZIP, SHA-256 `1f7db8dee1fe80bc2cb3a7b4d9d36b5bd19ce355b98e4a4cb17de36bfd8d1611`.

**Git commit:** `6a60b611b8ce03c236d541474fd4fd151c0da112` on `baseline-repro-clean`.

**Protocol:** Level 0 smoke; ModelNet40-C Gaussian severity 5; first two file-order batches; 64 of 2468 examples; batch 32; seed 0; gamma=eta=0.01; lambda=0.95; normal reverse steps=5. NVIDIA A100-SXM4-80GB, driver 580.82.07, CUDA 13.0; Python 3.8.20.

The seven required files have one safe archive root and no unexpected members. `execution_status=complete` and CSV `status=partial` correctly distinguish a finished prefix from full dataset coverage. Counts are internally consistent: 47/64 = 0.734375 for both recorded macro and micro values. Runtime is 4.3506 seconds and peak allocated GPU memory is 25748.2 MiB. `stdout.log` has no traceback or error line.

Point-MAE loaded with strict=False but zero missing/unexpected keys; both LION modules loaded strictly with zero missing/unexpected keys. The actual scheduler is DDIMScheduler with set_alpha_to_one=True, epsilon prediction and 100 timesteps from 990 to 0. Classifier is eval; LION VAE/priors are train mode, retaining the intended legacy-mode control. Source hashes match the recorded Git commit blobs; apparent local Windows hash differences were CRLF line endings. Colab git dirty state contains compiled extension/cache artifacts, not the runner source files.

**Decision:** accept Batch 1 artifact/output contract. This result is not an accuracy benchmark or reproduction claim. Next implementation gate is Batch 2 source-only full-corruption identity evaluation; do not yet change LION mode or add GSD.

## Entry template

## 2026-09-12 - Source-only all-corruption identity result

**Evidence:** [Run] `result/modelnet40_c/source_only/20260912-111822_source-only_seed0/`; ZIP SHA-256 `bf1ff8884401b4617213299ff7e1f7a90476345e25a6ea79d8d514cfff09eabe`.

**Git commit:** `ef87692042e337cf628b9438bfc06a3e362b2e0b`.

**Protocol:** source-only identity evaluation, ModelNet40-C severity 5, all 15 corruptions, 2468 examples each, 37020 total; FPS(1024), frozen Point-MAE, batch32, seed0. No LION loaded, no normalize/rotation/diffusion/guidance. Execution complete with no traceback.

Macro and micro accuracy are both `0.536899` (19876/37020), or 53.69%. This is 3.91 percentage points below the repository README source-only reference 57.6%. Per-corruption lows are lidar 19.94%, background 28.16%, rotation 30.19% and occlusion 37.24%; highest is density_inc 77.35%.

**Follow-up code/data audit (2026-09-12):** [Code] the repository's data guide directs ModelNet40-C users to download the already-corrupted Zenodo package; it only directs users to generate corruptions for ShapeNetCore and ScanObjectNN. [Run] the archived package has the expected 15 severity-5 arrays, 2,468 examples and labels in `[0,39]` for every corruption. [Code] the source-only runner is behaviorally identical to the upstream `tools/runner_finetune.py` source evaluation for this path: direct `np.load(data_<corruption>_5.npy)` / `label.npy`, then `misc.fps(points, 1024)`, then frozen `classification_only(..., only_unmasked=False)`. It adds logging only; it does not normalize, rotate, denoise, or augment inputs.

**Interpretation:** data/checkpoint/FPS/classifier protocol is operational and the locally used source-only path is not a preprocessing divergence from the upstream source-evaluation path. The actual downloaded files and Point-MAE checkpoint still have no author-published checksums, so structural validity does not prove byte identity. Because only one seed and one Colab environment are archived, this is a diagnosed reproduction gap, not a causal conclusion. Do not interpret LION/TTA accuracy before this gap is investigated.

**Decision:** preserve this as the source-only comparator. Next, compare data/checkpoint hashes and Point-MAE preprocessing/evaluation details against the reference protocol before dropout A/B. No GSD work.

## 2026-09-12 - Source-only seed variance result

**Evidence:** [Run] `result/modelnet40_c/source_only/20260912-120122_source-only_seed1/` and `result/modelnet40_c/source_only/20260912-120349_source-only_seed2/`.
**Git commit:** `ef87692042e337cf628b9438bfc06a3e362b2e0b`.
**Protocol:** source-only identity evaluation, ModelNet40-C severity 5, all 15 corruptions, 2468 examples each; FPS(1024), frozen Point-MAE, batch32. Seeds 1 and 2.

### Result
Both Seed 1 and Seed 2 have identically matching configuration, dataset hashes, and per-corruption metrics compared to Seed 0. 
Zero variance across all 3 seeds: macro and micro accuracy are perfectly equal at 53.69% (19876/37020). 

### Interpretation
The source-only evaluation pipeline is deterministic across these three seeds. It confirms that the 53.69% accuracy gap observed for Point-MAE source-only is stable and not an artifact of random sampling in the FPS step (which was potentially stochastic if unseeded, but it appears to yield identical outcomes here, likely because numpy/torch seeds were fixed).

### Decision
Source-only baseline is fully verified and stable at 53.69%. Proceed with data/checkpoint provenance investigation and ShapeNet preparations.

### Falsifier / next evidence
A different checkpoint or data source showing the 57.6% source-only accuracy.

```markdown
## YYYY-MM-DD — Short finding title

**Evidence:** [Run]/[Paper]/[Code]/[User report]/[Inference]  
**Git commit:** full hash  
**Run paths:** `result/...`  
**Hypothesis:** falsifiable statement  
**Protocol:** level, dataset, corruptions, seeds, important config  

### Result

Exact values, uncertainty, failures, runtime/memory, and per-corruption pattern.

### Interpretation

What the evidence supports and what it does not support.

### Decision

Continue, modify, reject, reproduce, or escalate to full evaluation.

### Falsifier / next evidence

What result would overturn or materially revise the interpretation.
```

## 2026-09-12 � ScanObjectNN Gaussian source-only pilot

- **[Run]** Artifact: result/scanobjectnn_c/20260912-135346_source-only-gaussian-seed0-label-fix.zip (user-supplied; complete after validation).
- **[Run]** Dataset: main_split-derived ScanObjectNN, Gaussian severity 8, 581 examples, seed 0, batch size 32, frozen 15-class Point-MAE checkpoint scanobject_jt.pth.
- **[Run]** Result: 108/581 correct, accuracy 0.1858864028 (18.59%), runtime 2.884 s, no traceback, checkpoint missing/unexpected keys empty.
- **[Inference]** This is a valid corruption result but not yet interpretable as adaptation evidence. A clean-input control with the same checkpoint and preprocessing is required first; the checkpoint is user-supplied and its clean OBJ-BG parity is not established.
- **[Open]** Create data_original.npy from the official main_split test H5, run the complete clean source-only control, then compare Gaussian degradation.

## 2026-09-12 � ScanObjectNN clean control

- **[Run]** Artifact: result/scanobjectnn_c/source_only/20260912-135748_clean-control-seed0.zip (user-supplied; complete after validation).
- **[Run]** Same seed/checkpoint/protocol as Gaussian pilot; 581 clean main_split examples, source-only, severity 0.
- **[Run]** Result: 415/581, accuracy 0.7142857143 (71.43%), runtime 2.875 s, no traceback.
- **[Run]** Clean and Gaussian artifacts use the identical label SHA-256 bb145670...eeb7a68; both inventory shapes are 581 examples and both checkpoint loads have empty missing/unexpected keys. This rules out a run-to-run label-file mismatch.
- **[Inference]** The 52.84-point clean-to-Gaussian drop is not explained by an observed label shift. Remaining candidates are the severity-8 corruption strength, checkpoint/preprocessing mismatch, or a generator/data-content issue; external H5-label equality and a severity-1 control remain open.

## 2026-09-12 � ScanObjectNN 3DD-TTA smoke

- **[Run]** Artifact: result/scanobjectnn_c/3dd_original/20260912-154840_3dd-original-gaussian-smoke-seed0-device-fix.zip.
- **[Run]** LION priors/VAE and Point-MAE loaded with empty missing/unexpected keys; the CPU/GPU unnormalization error did not recur.
- **[Run]** First two batches, 64 examples, Gaussian severity 8: 11/64 correct (17.1875%), runtime 5.167 s; status partial by design.
- **[Open]** Full 581-example 3DD-TTA run remains necessary; smoke accuracy is not benchmark evidence.

## 2026-09-12 - LION eval-mode TTA autograd compatibility repair

- **[User report]** The ModelNet40-C 3dd_original Gaussian eval-mode pilot failed at ch_loss.backward() because PVCNN devoxelization ran with is_training=False, so it did not retain the interpolation indices/weights required by its custom backward function.
- **[Code]** models/pvcnn2.py and models/pvcnn2_ada.py now pass self.training or torch.is_grad_enabled() to trilinear_devoxelize. Thus lion.eval() continues to disable dropout, while gradient-enabled TTA retains the custom CUDA operator's backward state. Normal no-grad eval remains unchanged.
- **[Code]** Local structural verification: python -m py_compile models/pvcnn2.py models/pvcnn2_ada.py third_party/pvcnn/functional/devoxelization.py completed successfully. No local CUDA execution was attempted.
- **[Open]** The prior artifact 20260912-162954_3dd-original-gaussian-eval-seed0.zip is a failed run, not an accuracy result. Rerun the same smoke command after pulling this repair, using a fresh run name.

## 2026-09-12 - LION dropout mode A/B: full Gaussian pilot

**Evidence:** [Run] result/modelnet40_c/3dd_original/20260912-162758_3dd-original-gaussian-legacy-seed0/ and result/modelnet40_c/3dd_original/20260912-163734_3dd-original-gaussian-eval-fix_seed0/.

**Protocol:** Complete ModelNet40-C Gaussian severity 5, 2,468 examples, batch 32, seed 0, gamma=eta=0.01, lambda=0.95, 100 DDIM steps and 5 reverse steps. The classifier checkpoint, LION checkpoint, Gaussian data file, configuration assets, batch size, and all listed TTA settings have identical hashes/values. The independent variable is LION mode: legacy train versus --lion-eval-mode.

**Result:** Legacy LION mode: 1,826/2,468 = 0.739870 (73.99%), runtime 96.60 s, peak memory 25,750 MiB. Eval mode after commit 9ce5553: 1,843/2,468 = 0.746759 (74.68%), runtime 95.55 s, peak memory 25,249 MiB. The observed change is +17 correct examples, or +0.6888 percentage points. Both archives contain all seven required files, have complete status, and their stdout logs contain no Traceback, RuntimeError, or ValueError.

**Code verification:** Legacy config records priors/VAE training=True. Eval config records CLI lion_eval_mode=True and priors/VAE training=False; this confirms dropout is disabled. The successful eval run also accepts the PVCNN autograd repair. The config field lion_mode_policy remains the stale string 'legacy; unchanged' in the eval artifact, so module inventory and CLI are the authoritative mode evidence for this pair.

**Interpretation:** This is encouraging but inconclusive. The runner records seed-controlled, not common-random-number-paired sampling; stochastic interpolation/noise remains a confound. The two runs are also on adjacent commits, although 9ce5553 only repairs eval-mode autograd. One corruption and one seed cannot establish a reliable accuracy effect or a benchmark gain.

**Decision:** Keep eval mode as a viable candidate. Repeat the exact legacy/eval pair for seeds 1 and 2 before selecting a mode; then test background under the same paired design. Do not yet use this +0.69 pp pilot to change the full benchmark protocol or make a thesis claim.

## 2026-09-12 - LION dropout A/B: Gaussian three-seed result

**Evidence:** [Run] seed-0 archives in the preceding entry plus 20260912-165102 legacy seed1, 20260912-165259 eval seed1, 20260912-165454 legacy seed2, and 20260912-165650 eval seed2 under result/modelnet40_c/3dd_original/.

**Protocol:** Full Gaussian severity 5; 2,468 examples; seeds 0,1,2; batch32; gamma=eta=0.01; lambda=.95; 100 DDIM / 5 reverse steps. All six archives have seven required files, complete CSV status, matching within-seed assets, and no traceback/runtime error. Seed1/2 pairs both use 9ce5553; seed0 legacy is 2611da8 and eval is 9ce5553.

**Result:** legacy/eval: seed0 73.9870/74.6759 (+0.6888 pp); seed1 73.3387/75.2431 (+1.9044 pp); seed2 73.9870/75.2836 (+1.2966 pp). Means: legacy 73.7709% (SD .3743 pp), eval 75.0675% (SD .3398 pp); paired mean delta +1.2966 pp (SD .6078 pp). All three deltas favor eval.

**Interpretation and decision:** Gaussian replicates a favorable eval-mode effect, but this is not a significance or benchmark claim: n=3, seed-controlled rather than fully common-draw-paired sampling, and seed0 spans the repair commit. The matched-commit seed1/2 deltas are still positive. Run background at seeds 0,1,2 (35 reverse steps) before a full 15-corruption comparison.


## 2026-09-12 - LION dropout A/B: full Background three-seed result

**Evidence:** [Run] `result/modelnet40_c/3dd_original/20260912-171041_3dd-original-background-legacy-seed0.zip`, `20260912-171913_3dd-original-background-eval-seed0.zip`, `20260912-172738_3dd-original-background-legacy-seed1.zip`, `20260912-173612_3dd-original-background-eval-seed1.zip`, `20260912-174437_3dd-original-background-legacy-seed2.zip`, and `20260912-175310_3dd-original-background-eval-seed2.zip`.

**Git commit:** `9ce5553de9277f9b9d7e26fc729e70b912a7c2d7`.

**Protocol:** Complete ModelNet40-C Background severity 5; 2,468 examples per run; seeds 0, 1, 2; batch 32; gamma=eta=.01; lambda=.95; 100 DDIM steps and 35 background reverse steps. The six archives each contain the required seven files, one archive root, one complete summary/per-corruption row, and no Traceback, RuntimeError, or ValueError. Every artifact records the same Point-MAE, LION, config, label, and Background-file identities. Legacy CLI has lion_eval_mode=false; eval CLI has lion_eval_mode=true. Eval module inventory confirms LION dropout entries have training=false; legacy retains training mode.

**Result:** Legacy accuracies for seeds 0/1/2 are 60.6564%, 61.5073%, and 60.0486%; eval accuracies are 60.8185%, 60.3323%, and 60.7374%. Paired eval-minus-legacy deltas are +0.1621, -1.1750, and +0.6888 percentage points. Means are 60.7374% legacy (SD 0.7327 pp) and 60.6294% eval (SD 0.2605 pp), for a paired mean delta of -0.1080 pp (SD about 0.961 pp). Eval is faster by about 8.21 seconds per run on average (485.69 versus 493.89 seconds) and uses about 500 MiB less peak allocated GPU memory (about 25,250 versus 25,750 MiB).

**Interpretation:** Background does not replicate Gaussian's consistent eval advantage (+1.2966 pp mean across its three seed-controlled pairs). The current evidence falsifies a claim that disabling LION dropout uniformly improves 3DD-TTA over corruptions. It does not isolate a causal dropout effect because the runs are seed-controlled rather than common-random-number paired; changing the mode changes stochastic-draw consumption. The observed mode interaction is nevertheless large enough that a Gaussian-only mode selection would be unjustified.

**Decision:** Do not lock either LION mode as the global baseline yet. Keep batch 32 unchanged. The next protocol decision must evaluate mode behavior across more corruptions before any all-corruption baseline is declared; do not interpret the Background mean as a paper-level reproduction metric.

**Falsifier / next evidence:** A matched common-draw A/B or a broader multi-corruption paired evaluation that shows a stable same-direction difference would revise this conclusion.


## 2026-09-13 - 15-corruption LION-mode screen at seed 0

**Evidence:** [Run] Complete seed-0 legacy/eval ZIP pairs under `result/modelnet40_c/3dd_original/`, consisting of the existing Gaussian/Background pairs and 26 newly supplied `*-screen-seed0.zip` artifacts for Cutout, Density, Density Increase, Distortion, RBF Distortion, Inverse-RBF Distortion, Impulse, LiDAR, Occlusion, Rotation, Shear, Uniform, and Upsampling.

**Git revisions:** `9ce5553de9277f9b9d7e26fc729e70b912a7c2d7` for every newly added screen artifact, all Background artifacts, and Gaussian eval. Gaussian legacy seed 0 is `2611da8a7507a449f5c34c5788f92e4506a63ad3`; therefore that one pair spans the PVCNN eval-autograd repair commit.

**Protocol:** ModelNet40-C severity 5, all 15 corruptions, 2,468 examples per corruption, seed 0, batch 32, gamma=eta=.01, lambda=.95, 100 DDIM steps; Background uses 35 reverse steps and other corruptions use 5. Each included ZIP has exactly seven expected files under one root; each selected CSV row is complete, has 2,468 examples, and its stdout has no Traceback, RuntimeError, or ValueError. This is a seed-0 exploratory mode screen, not a locked Level-3 benchmark and not a common-random-number pair.

**Result:** Macro accuracy is 63.0578% legacy (23,344/37,020 correct) and 63.8817% eval (23,649/37,020), a +0.8239 percentage-point eval-minus-legacy difference (+305 correct). Eval is higher in 13/15 corruptions; it is lower only on Rotation (-.5267 pp) and Shear (-.3241 pp). Largest gains are RBF Distortion (+1.9854 pp), Cutout (+1.6613), Density Increase (+1.5802), LiDAR (+1.5802), and Upsampling (+1.2561). Total recorded adaptation runtime is 1,818.34 s eval versus 1,846.22 s legacy; mean peak allocated memory is 25,249.40 MiB eval versus 25,749.44 MiB legacy.

**Interpretation:** The full seed-0 screen strengthens the evidence that LION eval mode can improve the aggregate result under this implementation, but it does not establish a reproducible global advantage. Background seed variation already changes the comparison direction, the new 13 corruption pairs have only seed 0, draws are not common-random-number paired, and the Gaussian seed-0 legacy/eval pair crosses a code revision. The 63.8817% eval screen remains 1.8183 pp below the paper's 65.7% and 2.2183 pp below the README's 66.1%; it must not be used as a claim of paper-level parity.

**Decision:** Keep batch 32. Treat eval mode as the leading candidate for the next confirmed baseline, but do not lock it until matched-commit repeated-seed evidence is obtained. Do not resume GSD/PxP tuning from this screen alone.

**Falsifier / next evidence:** Re-run a predeclared full all-15-corruption evaluation for seeds 1 and 2 under a clean matched commit (or introduce common-draw pairing) and report the three-seed macro mean and variance.

## 2026-09-13 - Matched all-15 LION eval-mode screen, seeds 1 and 2

**Evidence:** [Run] `result/modelnet40_c/3dd_original/20260913-173819_3dd-original-all15-eval-raw-seed1.zip`, `20260913-183818_3dd-original-all15-eval-raw-seed2.zip`, `20260913-200206_3dd-original-all15-legacy-raw-seed1.zip`, and `20260913-203231_3dd-original-all15-legacy-raw-seed2.zip`. All are safe seven-file ZIPs with 15 complete 2,468-example rows and no error signature in `stdout.log`.

**Git commit:** `b999a1eb809690a625c1075b205cb042b384443f` for all four bundles.

**Protocol:** ModelNet40-C severity 5, all 15 corruptions, 37,020 examples/run, batch 32, gamma=eta=.01, lambda=.95, raw LION prior (`lion_ema_mode=false`) and seeds 1/2. The sole planned condition difference is `lion_eval_mode`: legacy has false and records LION prior/VAE dropout `training=true`; eval has true and records `training=false`. Recorded LION and Point-MAE checkpoint identities agree across all four bundles.

**Result:** Legacy macro accuracies are 63.1361% (seed 1) and 63.0578% (seed 2), mean 63.0970%. Eval/raw is 63.8088% and 63.9006%, mean 63.8547%. Eval minus legacy is +.6726 and +.8428 pp, respectively: **+.7577 +/- .1203 pp** sample SD, or +561 correct predictions across the two complete runs. Eval is higher in 13/15 two-seed per-corruption means; the largest mean gains are Density (+1.742 pp), Shear (+1.682), Impulse (+1.479), Cutout (+1.074), and LiDAR (+1.033). Uniform (-.122) and Background (-.101) are lower.

**Interpretation:** This is the first full, same-commit repeated-seed evidence that the LION eval-mode path is operationally preferable in this fork. It remains a seed-controlled, non-common-draw comparison, so it cannot isolate the causal effect of dropout alone; it also does not resolve the source-only gap or establish parity with the 65.7% paper mean.

**Decision:** Select `--lion-eval-mode` with raw weights as the provisional baseline for subsequent reproduction diagnostics. Keep legacy/raw as the comparator, retain EMA as an ablation, and keep GSD/PxP parked. Full table and protocol caveats: `knowledge/dropout_eval_mode_20260913.md`.

**Falsifier / next evidence:** A common-random-number legacy/eval control that reverses the result would overturn the operational choice. Independently, a labelled source-only severity 1--5 probe is the next P0 test for the reproduction gap.

## 2026-09-15 - Main/LION re-audit: FPS and integration candidates

**Evidence:** [Code], [Paper], [Inference]; [Run] reanalysis only of existing
source/eval bundles. Active HEAD `262f3a668b3f5a7bc44c6282c4a8a2723ac6f00a`;
3DD-TTA main `107305fd7baf40b359f31c07d235599198be7324`; LION
`7711b3d185752eeb632d095494876e4de15f3195`. Remote main refs agree with local.
Full source locations, exact archive paths, protocol and falsifiers are in
`knowledge/code_audit_20260915.md`. No new GPU evaluation or algorithm change.

**New observations:** [Code/Run] source-only always requests FPS(1024), although
Density/Cutout/LiDAR inputs contain only 649/724/768 points in
`result/modelnet40_c/source_only/20260912-111822_source-only_seed0/`. Gather-based
sampling necessarily repeats indices. The bundled FPS kernel also excludes
later candidates of squared radius <=.001, even when input/output counts match.
The binary's actual behavior requires Colab inspection. [Code] Point-MAE creates
unused NumPy random masks in all-token classification, which can alter subsequent
TTA interpolation; added classifier diagnostics need RNG isolation. Inherited
PVCNN interpolation/voxelization supplies only partial coordinate gradients.

**Reconfirmed:** [Paper/Code] final decoding uses old global style despite its
updates; Eq. 11 point-count denominators are absent from the SCD sum; paper
lambda=.96 differs from code .95. Decoder residual weight .01 makes the style
effect a measured question, not a promised gain. A shared-trajectory paired
decode is the smallest targeted TTA experiment. A preprocessing-only identity
control is missing from attribution of TTA-minus-source gains.

**Reference correction:** visually checked Table 2 confirms paper source=57.6%;
corrected the stale corruption headings in `papers/3dd_tta.md`. The earlier
2026-09-13 audit claimed that correction but the old headings had remained.

**Decision:** finish severity 1--5 first; if unresolved, prioritize installed
FPS/index and preprocessing controls before source-data gate closure. Queue
updated-style, SCD-scale and lambda as separate exploratory TTA tests after
reviewing that gate. Keep eval/raw, batch32, severity5 and parked-method decisions.
No candidate has a newly established accuracy benefit. Preserve all raw evidence;
commit only the relevant knowledge changes. The pre-existing runner note-only
working change remains uncommitted and was not modified by this audit.

**Falsifiers:** an installed FPS binary without the suspected behavior weakens
that runtime explanation; matched source controls with unchanged predictions
rule down sampling/preprocessing effects. Null/negative shared-trajectory style
results rule down the decoder candidate. Record full seven-file Colab ZIPs for
each condition/seed; do not infer causality from cross-paper gain subtraction.

## 2026-09-16 - Source-only severity 1--5 probe: severity explains a large descriptive component

**Evidence:** [Run] Five complete seven-file ZIPs under
`result/modelnet40_c/source_only/`:

- `20260915-184540_source-only-sev1-all15_seed0.zip` (SHA-256 `a99c8998212d006a94418e8168ab21d299989f022a5fc87ccf6f9d70b76e8c7b`)
- `20260915-184848_source-only-sev2-all15_seed0.zip` (SHA-256 `a5272e15c439772357ec1ef9d51b9dd67635158f08d725d6fa4091db400c23f6`)
- `20260915-185118_source-only-sev3-all15_seed0.zip` (SHA-256 `a716844b0974efeee723ae36d447cc5dcaea198cbec8b86dfe735b2d2f5b60f1`)
- `20260915-185346_source-only-sev4-all15_seed0.zip` (SHA-256 `49cef14e82ae709e9ed67c96ef4d6104d263dce7599fda71661c434b4fc6c458`)
- `20260915-185615_source-only-sev5-all15_seed0.zip` (SHA-256 `01fdcfdab370f06a4d9b950603174fb6b978cd0c3acc1b2fd2a812eb7a0ceea6`)

All five use commit `262f3a668b3f5a7bc44c6282c4a8a2723ac6f00a`, seed 0,
batch 32, frozen Point-MAE, direct corruption-file loading, FPS(1024), all
15 corruptions and 2,468 examples per corruption. Classifier checkpoint hash
is `507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75`; the
label hash is identical across levels. Each archive has a safe single root,
all required files, 15 complete rows, 37,020 total examples, matching
config/CSV severity and no traceback/error signature. The generated `notes.md`
files retain the pre-patch generic smoke wording; this is documented metadata
debt and raw artifacts are not rewritten.

**Hypothesis:** released corruption severity could explain part of the source
Point-MAE discrepancy before LION/TTA.

### Result

| Severity | Macro = micro accuracy | Correct / 37,020 | Difference from paper source 57.6% |
|---:|---:|---:|---:|
| 1 | 75.8806% | 28,091 | +18.2806 pp |
| 2 | 73.2739% | 27,126 | +15.6739 pp |
| 3 | 68.5062% | 25,361 | +10.9062 pp |
| 4 | 62.0205% | 22,960 | +4.4205 pp |
| 5 | 53.6899% | 19,876 | -3.9101 pp |

Severity 1 to 5 changes the macro by **-22.1907 pp**. Thirteen of 15
corruption curves are non-increasing. Occlusion rises from 41.7747% (s1) to
43.7196% (s3); LiDAR rises from 20.2188% to 23.7439% before declining. The
severity-5 row exactly reproduces the earlier source-only 53.6899% result.
The largest s5 deficits against rounded paper source cells are Density
(65.2350% vs 75.1%), LiDAR (19.9352% vs 29.1%), Cutout (62.2771% vs 70.4%)
and Gaussian (51.2966% vs 57.0%).

### Interpretation and decision

**[Inference]** Severity is a strong descriptive determinant of source-only
accuracy and can account for more than the observed s5 gap if a different
severity were used. No tested severity is a provenance match for the paper:
s4 is 4.4205 points above 57.6%, while s5 is 3.9101 points below it. The paper
reports 57.6% in Table 2 but does not state severity; released 3DD-TTA code
uses `_5` files. Checkpoint/data identity and corruption-generation version
remain open. This is not TTA evidence and does not redefine the severity-5
benchmark.

Accept the probe as complete descriptive evidence. Keep severity 5 as the
operational protocol and retain all five levels as a diagnostic curve. Next P0
is provenance plus installed FPS/index inspection, followed by a preprocessing
identity control. Do not start GSD/PxP or select a lower severity post hoc.

**Falsifier / next evidence:** a verified paper-specific severity declaration
or byte-identical data/checkpoint bundle would revise provenance. A different
validated asset set would test asset-specificity. FPS diagnostics showing no
suspected repeats/origin filter in the actual Colab binary would weaken that
candidate explanation.

## 2026-09-19 - FPS diagnostic blocked by dependency drift

**Evidence:** `[Run]` The first Colab diagnostic artifact
`result/modelnet40_c/source_only/20260919-104821_source-only-fpsdiag-s5-seed0.zip`
failed before evaluation. Its traceback reaches `main_3dd_tta.py` ->
`models/lion.py` -> `diffusers.utils.peft_utils` ->
`diffusers.utils.torch_utils`, where the installed Diffusers code evaluates
`torch.xpu.empty_cache` and the installed PyTorch has no `torch.xpu` attribute.
The parent runner still sealed a failed ZIP; it is incomplete evidence, not an
accuracy result.

**[Code]** The source-only runner imports `main_3dd_tta` at worker startup,
which imports LION/Diffusers even though source-only does not use LION. The
repository's `env.yaml` pins `torch==2.0.1+cu121` but leaves `diffusers`
unpinned; `requirements.txt` contains the historical compatible pins
`diffusers==0.11.1` and `huggingface-hub==0.11.1`. This explains why older
environments could run while a newly resolved environment fails at import.
`ninja: no work to do` and `_pvcnn_backend` loading are preceding normal output,
not the failure source.

**Decision:** Do not add a fake `torch.xpu` attribute. First verify the Colab
package versions, then restore the historical Diffusers/Hub pins without
changing the dataset or FPS code. Separately consider lazy LION/Diffusers
imports so source-only diagnostics do not require unused TTA dependencies.

**[Code] Superseding update, 2026-09-19:** The diagnostic runner commit
`47d3e3b` was reverted by `cc52437` at the user's request before any valid FPS
diagnostic result. `run_baseline.py` is back to its pre-diagnostic behavior;
the failed import remains an environment/provenance finding, not evidence
against FPS and not an accuracy result.

**[Code/User report] Update, 2026-09-19:** After the user reported that the
environment-rebuilt source-only smoke completed successfully, the same
read-only `--fps-diagnostics` implementation was re-enabled. It remains an
opt-in source-only path; no alternate resampling policy or dataset mutation is
included. The complete smoke ZIP is still required before treating the
environment gate as `[Run]` evidence.

## 2026-09-19 - Legacy FPS diagnostic completed on four severity-5 corruptions

**[Run]** The complete archive
`result/modelnet40_c/source_only/20260919-122509_source-only-fpsdiag-s5-seed0.zip`
was validated without modifying it. SHA-256 is
`1397e8640b2b0e7688d5a55caaad1872a5a8f4c4d459515b5185d03c505f5539`.
It contains exactly the seven required files, four complete corruption rows
(2,468 examples each; 9,872 total), and no traceback/error signature. The run
records commit `36a2d602a121acf46a8462a58992ab648d446bb9`, Diffusers `0.11.1`,
PointNet2 extension `3.0.0`, and classifier hash
`507e0bbfc91b9293ef021b9078e86c0f333c04f408fa21e9c3320a83f53aec75`.

| Corruption | input N | mean unique indices | mean duplicate slots | input-origin total | selected-origin total | source accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Density | 649 | 648.7338 | 375.2662 | 657 | 0 | 65.2350% |
| Cutout | 724 | 723.7034 | 300.2966 | 733 | 1 | 62.2771% |
| LiDAR | 768 | 396.1528 | 627.8472 | 405 | 0 | 19.9352% |
| Gaussian | 1024 | 1023.6297 | 0.3703 | 914 | 0 | 51.2966% |

**[Code/Run]** The four accuracies exactly match the corresponding prior
source-only values, so the diagnostics did not change the classifier input or
predictions. Density, Cutout, and Gaussian duplicate totals are approximately
the unavoidable `1024-N` padding plus skipped near-origin points. LiDAR is
qualitatively different: it averages only about 396 unique indices and about
628 repeats per example; the 405 input-origin points cannot explain that scale.

**[Inference/Open]** LiDAR may contain repeated/degenerate coordinates,
non-finite values, or extension tie behavior, but these aggregate counters do
not distinguish them. The next diagnostic should record finite/NaN/Inf counts
and coordinate-unique counts, prioritizing LiDAR. No alternate resampling
policy has been implemented or benchmarked.

## 2026-09-19 - FPS diagnostic v2 prepared for LiDAR localization

**[Code]** `run_baseline.py` now records additional read-only counters behind
the existing opt-in `--fps-diagnostics` flag. For each selected corruption it
aggregates finite and non-finite input-point counts, scalar NaN and Inf counts,
finite-point coordinate-unique counts, and finite/non-finite selected-point
counts. The schema is explicitly marked
`legacy_fps_v2_finite_coordinate_unique` in `config.json`.

**[Code/Inference]** The legacy FPS indices, gather operation, classifier input,
and predictions are unchanged. Coordinate uniqueness is computed over exact
finite `[x,y,z]` rows only; non-finite rows are counted separately rather than
silently included in the uniqueness statistic. This is a localization
diagnostic, not a resampling-policy change.

**[Open]** No Colab v2 archive exists yet. The next run must use the same
severity-5, seed-0, batch-32, four-corruption scope so its counters remain
directly comparable to the validated v1 archive.

## 2026-09-19 - Uploaded v2-named ZIP used stale diagnostic code

**[Run]** Archive
`result/modelnet40_c/source_only/20260919-133003_source-only-fpsdiag-v2-s5-seed0.zip`
is structurally complete and error-free: seven required files, four complete
corruptions, 9,872 examples, and no traceback. SHA-256 is
`dfa4efd18637850b4cb0b5d62463fa797e3a7cd671cd799b28c524ea574d26a6`.
Its source-only accuracies reproduce the validated v1 values exactly: Density
65.2350%, Cutout 62.2771%, LiDAR 19.9352%, Gaussian 51.2966%.

**[Run/Provenance]** This is not v2 diagnostic evidence. `config.json` records
`git_commit=36a2d602a121acf46a8462a58992ab648d446bb9`, while v2 was added in
later commit `ceb9576`; `fps_diagnostics_schema` is absent and the new finite,
NaN/Inf, and finite-coordinate-unique fields are absent. `notes.md` also has
the v1 diagnostic wording. The run therefore confirms the old classifier path
only, not the LiDAR localization hypothesis.

**[Open]** Re-run after fetching `origin/baseline-repro-clean` at or beyond
`ceb9576`; verify `git rev-parse HEAD` and the schema field before accepting
the archive as v2 evidence. Raw ZIP remains unchanged.

## 2026-09-19 - FPS diagnostic v2 localizes LiDAR repetition

**[Run]** Archive
`result/modelnet40_c/source_only/20260919-133400_source-only-fpsdiag-v2-s5-seed0.zip`
is complete and valid (seven files, four complete rows, 9,872 examples, no
traceback). SHA-256 is
`9a9cc4cd8dcdd432344d19e43c70394bd8c05be6efae748b9a39e37d56802330`.
The run records commit `0003743b6362be322352ba42a70cbd8260c98f6d` and schema
`legacy_fps_v2_finite_coordinate_unique`. Accuracy is unchanged from v1:
Density 65.2350%, Cutout 62.2771%, LiDAR 19.9352%, Gaussian 51.2966%.

| Corruption | input N | mean finite points | mean finite-coordinate unique | mean FPS-index unique | mean input coordinate duplicates | mean FPS duplicate slots | input NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---:|
| Density | 649 | 649 | 649.0000 | 648.7338 | 0.0000 | 375.2662 | 0 / 0 |
| Cutout | 724 | 724 | 724.0000 | 723.7034 | 0.0000 | 300.2966 | 0 / 0 |
| LiDAR | 768 | 768 | 396.2273 | 396.1528 | 371.7727 | 627.8472 | 0 / 0 |
| Gaussian | 1024 | 1024 | 1024.0000 | 1023.6297 | 0.0000 | 0.3703 | 0 / 0 |

**[Run/Inference]** LiDAR has no non-finite input points, but about 372 exact
coordinate duplicates per example. FPS-index uniqueness nearly equals
finite-coordinate uniqueness, so the large duplicate-slot count is explained
by the input itself plus padding to 1024, not by NaN/Inf handling. The small
184-index aggregate difference is consistent with FPS selecting among repeated
coordinate rows. Density, Cutout, and Gaussian have no exact coordinate
duplicates; their earlier padding/origin-filter interpretation remains intact.

**[Code]** The repository generator's `simulate_lidar` uses
`np.random.choice(new_pc.shape[0], 768)` at
`datasets_mate/create_corrupted_dataset.py:655`; without an explicit
`replace=False`, NumPy samples with replacement. This is a direct code-level
mechanism consistent with the observed LiDAR duplicates, although the current
artifact does not by itself prove which historical generator invocation
created the archived `.npy` files.

**[Decision/Open]** Do not change inference resampling or claim an FPS bug yet.
First reconcile the corruption-file provenance/generator version. The v2 gate
is now complete; alternate policies remain parked.

## 2026-09-19 - Upstream generator confirms LiDAR replacement sampling

**[Code]** The canonical ModelNet40-C repository's `data/generate_c.py` uses
the same LiDAR construction at lines 240--242:
`index = np.random.choice(new_pc.shape[0], 768)` followed by `new_pc[index]`.
No `replace=False` is supplied. The fork's
`datasets_mate/create_corrupted_dataset.py:655-657` mirrors this behavior.
The upstream README documents both direct pre-corrupted download and generation
with `python data/process.py` / `python data/generate_c.py`.

**[Paper/Code/Run]** Together with the v2 counters, this strongly supports
that repeated LiDAR coordinates are an intended property of the ModelNet40-C
generator, not evidence that our inference FPS implementation is wrong. The
current archive still does not prove byte identity with the Zenodo package, so
asset identity remains an open provenance question, but `replace=False` is not
a justified benchmark correction.

**[Decision]** Do not regenerate `data_lidar_5.npy`, alter inference resampling,
or claim a LiDAR preprocessing contribution. Keep the validated archive as the
benchmark input and return to the broader source-only gap / asset-checkpoint
identity investigation.
this uncommitted batch is db1482ae3becb2e5f9f44a6811c775c5570b0501.
**Implementation commit:** `1e66374b7d9fbe3f193c51cdeb9895d1c7ecd5de`,
pushed to `origin/baseline-repro-clean`; no raw artifacts were included.
Identity is higher than source-only on 9/15 corruptions, equal on Upsampling, and lower on 5/15. Largest gains are Density Increase +8.1848 pp, Cutout +6.0373 pp, Density +3.6467 pp and Occlusion +2.8363 pp. Largest declines are LiDAR -4.6596 pp and Background -4.2950 pp.

Interpretation: [Run/Inference] The preprocessing chain has a real but modest positive aggregate effect under this seed and explains only a minority of the observed source-to-TTA difference. It is not sufficient to account for the TTA result, and the corruption-dependent signs prevent a claim of uniform preprocessing improvement. The run config has git_dirty=true because Colab generated or modified compiled extensions, caches and local data artifacts; the recorded source commit and runtime source manifest identify the intended code, and no method-source modification is visible in the recorded status.

Decision: This is a positive identity control under the predeclared decision rule, so plan the next pure VAE encode/decode control. Do not add a new TTA method or tune GSD/PxP. Keep the result exploratory because it is one stochastic seed and the source comparator is archived at an older commit, even though source-only behavior was structurally preserved.

Falsifier / next evidence: A pure VAE encode/decode result near source-only would localize the modest gain to preprocessing/interpolation/rotation/output normalization; a large pure-VAE gain would show that generative reconstruction, not guidance, explains more of the TTA difference. A repeat identity seed or common-draw control would test the stability of the +1.3344 pp estimate.
