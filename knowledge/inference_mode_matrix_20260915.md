# LION inference-mode matrix - 2026-09-15

This note is the compact decision table for the legacy, raw, eval, and EMA
terminology used in ModelNet40-C reproduction runs. Values are all-15
corruption severity-5 macro accuracies in percent, with batch 32,
gamma=eta=.01 and lambda=.95.

## Terms

| Term | Exact meaning in this fork |
|---|---|
| **Legacy** | Inherited wrapper behavior: no `lion.vae.eval()` / `lion.priors.eval()` call. LION VAE and prior dropout modules record `training=true`. Point-MAE remains frozen and eval in every condition. |
| **Eval** | Pass `--lion-eval-mode`: LION VAE and prior enter evaluation mode, so their dropout modules record `training=false`. TTA input-gradient computation remains enabled; this is not a `torch.no_grad()` inference. |
| **Raw** | Direct checkpoint network weights (`dae_state_dict`) after normal loading. “Raw” does not mean uncorrupted input and does not remove diffusion noise. |
| **EMA** | With `--lion-ema-mode`, replace only the LION prior’s raw weights with the 462 validated EMA tensors from `dae_optimizer`. The checkpoint has no VAE EMA, so the VAE remains raw. All EMA trials also use eval mode. |

No legacy+EMA run was made: EMA was tested only against eval+raw, which is the
appropriate trained-sampling comparison. The raw/eval condition is the current
provisional baseline.

## Complete-run matrix

| LION condition | Seed 0 | Seed 1 | Seed 2 | Mean +/- sample SD | Evidence status |
|---|---:|---:|---:|---:|---|
| Legacy + raw | 63.0578%* | 63.1361% | 63.0578% | 63.0839 +/- 0.0452%* | Seed 1/2 are matched `b999a1e` bundles; seed 0 is a historical composite screen, not a strict same-commit pair. |
| Eval + raw | 63.7061% | 63.8088% | 63.9006% | 63.8052 +/- 0.0973% | Three complete matched `b999a1e` bundles. |
| Eval + EMA | 63.8466% | 64.0546% | 63.8250% | 63.9087 +/- 0.1268% | Three complete matched `b999a1e` bundles; 462 prior EMA tensors confirmed. |

`Seed 0`, `Seed 1`, and `Seed 2` are the three experiment seeds; they are not
dataset severity labels. Raw artifacts are under
`result/modelnet40_c/3dd_original/`; exact archive names are recorded in
`dropout_eval_mode_20260913.md` and `ema_inventory_20260913.md`.

## Valid contrasts

| Contrast | Seed 0 | Seed 1 | Seed 2 | Summary and decision |
|---|---:|---:|---:|---|
| Eval + raw minus Legacy + raw | +0.6483 pp* | +0.6726 pp | +0.8428 pp | **Matched seed-1/2 mean: +0.7577 +/- 0.1203 pp.** Eval is higher for 13/15 two-seed corruption means; select eval+raw provisionally. Seed 0 is supporting only. |
| Eval + EMA minus Eval + raw | +0.1405 pp | +0.2458 pp | -0.0756 pp | **Three-seed mean: +0.1035 +/- 0.1639 pp.** Sign is not stable; retain EMA as an ablation, not the baseline. |

## Interpretation boundary

All comparisons use fixed seeds but do not preserve common random draws. LION
TTA adds initial latent Gaussian noise; point interpolation can use NumPy random
choices; and legacy dropout consumes extra PyTorch random draws. Thus a matching
seed makes the protocol controlled, but it does not make eval/legacy trajectories
identical apart from one dropout mask. Report the first contrast as the
**eval-mode effect**, not a fully isolated causal dropout effect.
