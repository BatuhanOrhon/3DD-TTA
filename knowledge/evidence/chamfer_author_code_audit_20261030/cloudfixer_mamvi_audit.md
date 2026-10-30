# CloudFixer and MAMVI author-code consistency audit — 2026-10-30

[User report/Decision] Before any research GPU tests, compare the implemented
transfers against author repositories. This audit distinguishes intentional
changes from reproduction defects; author implementations were not executed.

## Pinned sources and preserved bytes

[Code] GitHub connector fetched full files at the previously reviewed commits.
All five downloaded source files reproduce the GitHub Git-blob SHA exactly;
[retrieval manifest](cloudfixer_mamvi_retrieval.json) records SHA256/size/URLs.
Raw files remain under task preparation `author_code/`, outside Git.

- CloudFixer commit `f043e24414fcf4165fdc892abff21215c9989141`:
  [adapt.py](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/adapt.py#L72),
  [utils/pc_utils.py](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/utils/pc_utils.py#L21),
  [cfgs/cfgs_adapt.py](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/cfgs/cfgs_adapt.py#L133).
- MAMVI commit `0b553973b7123da6198b79bba4ff704b8eec914c`:
  [tools/tta.py](https://github.com/iskong17/MAMVI/blob/0b553973b7123da6198b79bba4ff704b8eec914c/tools/tta.py#L368),
  [utils/tent_shot.py](https://github.com/iskong17/MAMVI/blob/0b553973b7123da6198b79bba4ff704b8eec914c/utils/tent_shot.py#L186).

## CloudFixer: what the released implementation actually does

[Code] `adapt.py:97–110` computes static weights from detached local-neighbor
**mean squared distances**, raises the result to `args.pow`, takes its reciprocal,
then normalizes weights to sum1 per object and applies the validity mask.
The parser default is `knn=5`, `pow=1`. `utils/pc_utils.py:21–53` computes
squared Euclidean distances, includes self among k selected entries, adds1e-8
inside neighbor distances, excludes masked/duplicated entries and detaches the
returned statistic. It does not balance density by upweighting isolated points.

[Code] `adapt.py:112–178` optimizes physical `delta` and 6D rotation with
Adamax, sampling noise/timestep each update. The denoiser target is inside
`torch.no_grad()`. Chamfer matching gradients act on transformed physical X;
weighted displacement regularization makes moving dense regions more costly.
The denoiser Jacobian is absent because target and optimization variable are
different branches. Detaching our complete predicted latent instead would
remove the intended local-prior guidance route.

[Code/Inference] Our density-confidence SCD deliberately uses encoded reference
XYZ, nonself k10, epsilon1e-6, weights of mean1 and distance-trimmed bidirectional
Chamfer residuals. Reference-to-predicted uses own reference weight; the reverse
uses the native nearest-reference index. We preserve the original trim ordering.
We optimize existing local/style states with the existing rates, retain prior
gradients and use neither physical rotation nor delta. The inverse-squared
distance direction agrees with the released default statistic; our earlier
paper-distance caveat remains a paper-versus-code distinction, not evidence
that the author code used ordinary Euclidean distance. The residual location,
k/self/duplicate policy, epsilon and normalization are explicit adaptations.

[Decision] No author-code inconsistency requiring a sampler change was found.
Do not call this CloudFixer reproduction or inverse-density balancing. Keep
fixed label-free update-norm calibration because weighted trim changes gradient
magnitude even with mean1 weights. Sparse backward is a separate compute
policy; it does not implement CloudFixer's detached-target optimization.

## MAMVI: loss target, detach and optimization boundary

[Code] `tools/tta.py:368–443` makes masked-view classifier predictions, retains
their graphs, and detaches copies used to derive view weights and consensus.
The consensus is a weighted mean of detached view probabilities; an optional
entropy reliability mask can remove uncertain objects. `:481–533` uses
**one-direction KL(view || detached consensus)** plus each view's own entropy,
averages over reliable samples and weights across views. `:535–637` performs
a single backward and optimizer step, with optional confidence/consistency/LR
logic. `tta_cd_mmvec` calls `tent_shot_utils.setup_tent_shot`; its configure/
collect helpers enable BN/LN affine adaptation and change BN running-stat use.

[Code/Inference] Our semantic branch intentionally keeps all classifier and
normalization parameters/buffers fixed. One deterministic FPS source prediction
from corrupted X is cached and detached per batch. At the last two existing
probe checkpoints, the existing decoded baseline prediction is reused for
readout and guidance. KL is symmetric between decoded and fixed X distributions;
marginal entropy is per-object H((p_decoded+p_X)/2). These are separate arms,
with summed per-object losses to match the Chamfer sum. Only the decoded branch
retains gradients, through frozen decoder/classifier/preprocessing into the
local/style variables. The reference never becomes an extra probability-mean
vote. No masked12-view consensus, reliability threshold or adaptive LR is copied.

[Decision] These are declared input-guidance hypotheses, not a faithful MAMVI
loss/optimizer implementation. The differences implement the user's frozen-head
and decode-reuse constraints. CPU tests cover fixed-reference detach, decoded
input gradients, frozen model state/RNG, unchanged sham readout, reduction and
call reuse. No mathematical author-code mismatch demands replacing these with
normalization-parameter updates. Native CUDA backward remains unverified.

## GPU gate and limits

[Open] Author-code reading does not prove accuracy, speed or hardware viability.
Keep GPU experiments unstarted until the full author audit is reconciled.
Then require actual finite/nonzero semantic gradient calibration, sham parity,
zero spectral/graph counters and paired initial-state proof before accuracy
collection. Independent engineering review found two separate acceptance issues
(repeated timing groups and runtime calibration-formula validation); fixes and
regressions are being completed before source freeze.
