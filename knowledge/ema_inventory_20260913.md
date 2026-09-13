# LION EMA inventory — 2026-09-13

Evidence: [Run] `result/modelnet40_c/diagnostics/checkpoint_ema_inventory.json` and `checkpoint_sha256.txt`.

The Colab checkpoint contains `dae_state_dict`, `dae_optimizer`, `vae_state_dict`, and `vae_optimizer`. The prior has 462 model entries and 462/462 optimizer EMA entries, all float32 with matching shapes. The VAE has 867 model entries and no EMA entries. Config metadata enables EMA with decay 0.9999.

Implementation: `models/lion.py` now has opt-in `use_ema`, exposed by `--lion-ema-mode` in `main_3dd_tta.py` and `run_baseline.py`. Default raw loading is unchanged. The loader validates optimizer parameter count and every EMA tensor shape before copying parameters.

Next: compare eval+raw versus eval+EMA on complete Gaussian and Impulse severity-5 files with identical seed, scheduler, rates, lambda and batch. Do not replace VAE weights.
