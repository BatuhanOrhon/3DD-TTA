# LION eval-mode (dropout) screen - 2026-09-13

## Question

Does setting LION prior and VAE to evaluation mode, which disables their
dropout layers, improve the released 3DD-TTA baseline relative to the inherited
legacy train-mode wrapper?

## Evidence and protocol

**[Run]** Four immutable, seven-file Colab bundles were validated under
`result/modelnet40_c/3dd_original/`:

- `20260913-173819_3dd-original-all15-eval-raw-seed1.zip`
- `20260913-183818_3dd-original-all15-eval-raw-seed2.zip`
- `20260913-200206_3dd-original-all15-legacy-raw-seed1.zip`
- `20260913-203231_3dd-original-all15-legacy-raw-seed2.zip`

All four use Git commit `b999a1eb809690a625c1075b205cb042b384443f`, raw
LION weights (`lion_ema_mode=false`), the same recorded LION and Point-MAE
checkpoint SHA-256 values, ModelNet40-C severity 5, all 15 corruptions,
2,468 examples/corruption, batch 32, gamma=eta=.01, lambda=.95, and seeds 1
or 2. Every `per_corruption.csv` has 15 complete rows and each `stdout.log`
has no Python error signature.

`lion_eval_mode=false` in the legacy bundles and `true` in the eval bundles.
Their module inventories independently confirm that the LION prior and VAE
dropout modules are `training=true` for legacy and `training=false` for eval.
This is therefore a matched-code inference-mode comparison, not an EMA
comparison.

The runs are seed-controlled but not common-random-number paired: changing the
mode can change the stochastic-draw sequence. The effect should be described as
the **LION eval-mode effect**, not as an isolated causal estimate of only one
dropout layer.

## Result

| Corruption | Legacy mean | Eval mean | Eval - legacy |
|---|---:|---:|---:|
| uniform | 76.763% | 76.641% | -0.122 pp |
| gaussian | 73.906% | 74.635% | +0.729 pp |
| background | 60.413% | 60.312% | -0.101 pp |
| impulse | 69.125% | 70.604% | +1.479 pp |
| upsampling | 81.686% | 82.314% | +0.628 pp |
| distortion_rbf | 62.703% | 62.865% | +0.162 pp |
| distortion_rbf_inv | 64.364% | 65.215% | +0.851 pp |
| density | 70.766% | 72.508% | +1.742 pp |
| density_inc | 85.312% | 86.062% | +0.750 pp |
| shear | 64.749% | 66.430% | +1.682 pp |
| rotation | 32.820% | 33.124% | +0.304 pp |
| cutout | 69.672% | 70.746% | +1.074 pp |
| distortion | 64.810% | 65.215% | +0.405 pp |
| occlusion | 39.587% | 40.336% | +0.750 pp |
| lidar | 29.781% | 30.814% | +1.033 pp |
| **Macro** | **63.0970%** | **63.8547%** | **+0.7577 pp** |

Seed-level macro values are 63.1361% legacy versus 63.8088% eval at seed 1
(+.6726 pp), and 63.0578% legacy versus 63.9006% eval at seed 2 (+.8428 pp).
The two-seed eval-minus-legacy mean is **+.7577 +/- .1203 pp** sample SD,
equivalent to 561 more correct predictions over 74,040 evaluated examples.
Eval is higher for 13 of 15 two-seed corruption means; only Uniform and
Background are slightly lower.

## Interpretation and decision

**[Inference]** The effect is substantially more consistent than the earlier
Gaussian-only or Background-only pilots. Together with the historical seed-0
screen, it supports `--lion-eval-mode` as the **provisional selected
reproduction baseline** for subsequent diagnostic work. It does not eliminate
the upstream reproduction gap: its 63.8547% two-seed mean is still below the
paper Table-2 mean (65.7%), while source-only parity remains unresolved.

Keep legacy/raw as the explicit comparator and retain `--lion-ema-mode` only as
an ablation. Do not characterize the result as a paper-level improvement or as
a fully isolated dropout causal effect until common-draw control is available.

## Next evidence

The next P0 investigation should address the source-only/released-data gap,
including the already planned labelled severity 1--5 source-only probe, rather
than tuning GSD. A common-random-number legacy/eval pair would be the direct
falsifier if this mode decision must be made causal rather than operational.
