# LION EMA inventory — 2026-09-13

Evidence: [Run] `result/modelnet40_c/diagnostics/checkpoint_ema_inventory.json` and `checkpoint_sha256.txt`.

The Colab checkpoint contains `dae_state_dict`, `dae_optimizer`, `vae_state_dict`, and `vae_optimizer`. The prior has 462 model entries and 462/462 optimizer EMA entries, all float32 with matching shapes. The VAE has 867 model entries and no EMA entries. Config metadata enables EMA with decay 0.9999.

Implementation: `models/lion.py` now has opt-in `use_ema`, exposed by `--lion-ema-mode` in `main_3dd_tta.py` and `run_baseline.py`. Default raw loading is unchanged. The loader validates optimizer parameter count and every EMA tensor shape before copying parameters.

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
