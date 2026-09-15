# LION EMA inventory — 2026-09-13

Evidence: [Run] `result/modelnet40_c/diagnostics/checkpoint_ema_inventory.json` and `checkpoint_sha256.txt`.

The Colab checkpoint contains `dae_state_dict`, `dae_optimizer`, `vae_state_dict`, and `vae_optimizer`. The prior has 462 model entries and 462/462 optimizer EMA entries, all float32 with matching shapes. The VAE has 867 model entries and no EMA entries. Config metadata enables EMA with decay 0.9999.

Implementation: `models/lion.py` now has opt-in `use_ema`, exposed by `--lion-ema-mode` in `main_3dd_tta.py` and `run_baseline.py`. Default raw loading is unchanged. The loader validates optimizer parameter count and every EMA tensor shape before copying parameters.

## Provenance clarification - 2026-09-15

**[Paper]** Text extraction from the local LION paper (`lion.pdf`, 19 pages)
finds zero occurrences of the exact terms `EMA`, `exponential moving average`,
and `moving average`. EMA is therefore **not** a paper-reported LION method
claim or a stated explanation for any paper result.

**[Code]** It is nevertheless a real implementation detail in the original
LION training repository at `../LION/`: `trainers/common_fun_prior_train.py`
wraps the prior/DAE optimizer in `EMA(..., ema_decay=args.ema_decay)`;
`utils/ema.py` updates each optimizer-state `ema` tensor after the base
optimizer step; and `trainers/train_prior.py` saves that optimizer state in
`dae_optimizer`. At sampling, `trainers/train_prior.py` conditionally calls
`dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)` before and
after sampling. The supplied matching configuration in this fork,
`lion_ckpts/unconditional_all55_cfg.yml`, sets `ddpm.ema: 1` and
`sde.ema_decay: .9999`.

**[Run]** The supplied LION checkpoint inventory independently records 462/462
shape-matching prior EMA tensors in `dae_optimizer`, and no VAE EMA tensors.
Our opt-in loader reconstructs this **prior-only sampling swap** because
3DD-TTA loads modules directly instead of instantiating the original trainer.
The reconstruction is shape/count validated, but it remains a fork ablation;
it must not be presented as a paper-specified 3DD-TTA or LION setting.

Next: compare eval+raw versus eval+EMA on complete Gaussian and Impulse severity-5 files with identical seed, scheduler, rates, lambda and batch. Do not replace VAE weights.

## First pilot result

**[Run]** `result/modelnet40_c/3dd_original/20260913-154218_3dd-original-gaussian-impulse-eval-raw-seed0.zip` and `20260913-154546_3dd-original-gaussian-impulse-eval-ema-seed0.zip` are complete seven-file bundles. Both use commit `c91c1c3`, seed 0, severity 5, batch 32, eval mode, gamma=eta=.01, lambda=.95, and identical Point-MAE/LION checkpoint hashes. The EMA bundle has `lion_ema_mode=true` and stdout confirms `INFO loaded prior EMA parameters: 462`; raw has `lion_ema_mode=false`.

| Corruption | Eval + raw | Eval + EMA | EMA minus raw |
|---|---:|---:|---:|
| Gaussian | 74.3112% (1834/2468) | 74.7974% (1846/2468) | +0.4862 pp |
| Impulse | 70.2188% (1733/2468) | 70.6240% (1743/2468) | +0.4052 pp |
| Two-corruption macro | 72.2650% (3567/4936) | 72.7107% (3589/4936) | +0.4457 pp |

**Interpretation:** Both corruptions favor EMA by 22 correct predictions total. This is an exploratory, seed-controlled but not common-random-number-paired result; runtime configuration explicitly records cuDNN benchmark/non-deterministic behavior. It does not establish a final baseline improvement. Repeat the exact raw/EMA pair at seeds 1 and 2 before deciding whether EMA belongs in the selected baseline.

## Three-seed confirmation

**[Run]** The seed-1/2 pairs are `20260913-155608_3dd-original-gaussian-impulse-eval-raw-seed1.zip`, `20260913-155936_3dd-original-gaussian-impulse-eval-ema-seed1.zip`, `20260913-160304_3dd-original-gaussian-impulse-eval-raw-seed2.zip`, and `20260913-160632_3dd-original-gaussian-impulse-eval-ema-seed2.zip`, under `result/modelnet40_c/3dd_original/`. All six seed-0/1/2 bundles are complete, use commit `c91c1c3`, identical classifier/LION checkpoint hashes and eval mode; EMA bundles confirm 462 loaded prior EMA parameters.

| Seed | Raw macro | EMA macro | EMA minus raw |
|---:|---:|---:|---:|
| 0 | 72.2650% | 72.7107% | +0.4457 pp |
| 1 | 72.3663% | 72.8525% | +0.4862 pp |
| 2 | 72.6904% | 72.4878% | -0.2026 pp |
| Mean ± sample SD | 72.4406% | 72.6837% | +0.2431 ± 0.3865 pp |

Across the three seeds, Gaussian changes by +.4862, -.2431, and -.4052 pp (mean **-.0540 pp**); Impulse changes by +.4052, +1.2156, and 0.0000 pp (mean **+.5402 pp**). Aggregate correct predictions are 10,727 raw and 10,763 EMA out of 14,808 paired evaluated examples.

**Decision (superseded):** The two-corruption result alone is inconclusive. The earlier decision not to run all 15 corruptions was premature because EMA may be corruption-dependent.

**Active next test:** Run a matched screening benchmark over all 15 ModelNet40-C corruptions at seeds 0 and 1: `eval+raw` versus `eval+EMA`, with commit `c91c1c3`, severity 5, batch 32, gamma=eta=.01 and lambda=.95. Interpret the four runs as an exploratory two-seed screen, then decide whether a seed-2 confirmation or source-only severity-4 control is next.

## All-15, three-seed screen

**[Run]** Six complete ZIPs at commit `b999a1e` are archived under `result/modelnet40_c/3dd_original/`: raw/EMA pairs for seeds 0 (`20260913-163817`, `20260913-170822`), 1 (`20260913-173819`, `20260913-180820`), and 2 (`20260913-183818`, `20260913-190813`). Each has 15 complete 2,468-example rows; identical LION/Point-MAE checkpoint hashes; LION eval mode; seed/batch/severity/guidance settings matched; and EMA logs confirm 462 prior tensors.

| Corruption | Raw mean | EMA mean | EMA minus raw |
|---|---:|---:|---:|
| uniform | 76.9584% | 77.1880% | +0.2296 pp |
| gaussian | 74.6623% | 74.9325% | +0.2701 pp |
| background | 60.4538% | 61.0346% | +0.5808 pp |
| impulse | 70.4079% | 70.0837% | -0.3241 pp |
| upsampling | 82.0232% | 82.3204% | +0.2971 pp |
| distortion_rbf | 62.7634% | 62.8309% | +0.0675 pp |
| distortion_rbf_inv | 65.1540% | 65.2215% | +0.0675 pp |
| density | 72.4878% | 72.4743% | -0.0135 pp |
| density_inc | 86.0886% | 86.3182% | +0.2296 pp |
| shear | 66.1669% | 66.2075% | +0.0405 pp |
| rotation | 33.1983% | 33.3739% | +0.1756 pp |
| cutout | 70.5700% | 70.5700% | +0.0000 pp |
| distortion | 65.0459% | 65.4106% | +0.3647 pp |
| occlusion | 40.3025% | 40.1135% | -0.1891 pp |
| lidar | 30.7942% | 30.5511% | -0.2431 pp |
| Macro | 63.8052% | 63.9087% | +0.1035 pp |

Seed macro deltas are +.1405, +.2458, and -.0756 pp: **+.1035 ± .1639 pp** (sample SD), equivalent to +115 correct predictions over 111,060 evaluated examples. Ten corruption means are positive, one is zero, and four are negative; neither direction is stable across every seed.

**Decision:** The all-15 screen does not support selecting EMA as a robust baseline improvement. Preserve `--lion-ema-mode` as an ablation and report the small, corruption-dependent result. The next active baseline test is dropout: legacy+raw at seeds 1/2, compared with the now-available eval+raw seeds 1/2.
