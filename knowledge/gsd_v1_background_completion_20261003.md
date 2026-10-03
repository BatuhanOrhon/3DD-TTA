# GSD v1 spectral-only M100 Background completion

[Code/User report] The archived seed-0 `ablation_no_background` run evaluates
14 ModelNet40-C corruption files at severity5 with weight1, SCD weight0 and
M100. Background is the only missing corruption. A dedicated `background_completion`
stage locks the same settings and evaluates the full 2,468-example Background
file; it cannot run a prefix or a different corruption.

The Colab driver validates the immutable 14-corruption ZIP
`result/modelnet40_c/gsd_latent_spectral_v1/20260926-141029_gsd-v1-ablation14-on-seed0-spectral-only.zip`
(SHA256 `a083cf795dc7e2b951fa80f47ea32e939cbeb8f9e41c027449b02431ee6b804a`),
checks its per-corruption counts and current dataset-file hashes, and runs only
Background. It derives the equal-corruption macro over the 14 archived rows and
the new Background row. The JSON and findings-log entry explicitly label this
as a cross-run 14+1 composite, not a single contemporaneous all-15 run.
Environment and extension inventory differences are reported alongside the
score. The knowledge record is appended only after both ZIPs pass validation.

## Colab dependency repair

[User report/Code] Both new guidance launchers stopped before inference because
the `3dd_tta_env` Python could not import `pointnet2_ops`. The subsequent setup
trace first identified system nvcc13.0 versus the installed PyTorch cu121. After
pinning nvcc to CUDA12.1, the compiler output showed the remaining failure:
CUDA12.1 rejects the host GCC because it is newer than GCC12. The notebook now
installs CUDA12.1 plus GCC/G++12 in the existing conda environment, selects the
active GPU architecture, rebuilds the extension, and calls the FPS CUDA kernel
before evaluation. `TORCH_CUDA_ARCH_LIST` selects GPU architecture, not toolkit
version. The notebook does not reinstall `requirements.txt` or replace PyTorch.

[Open] The Colab build and FPS check have not yet completed successfully. Once
the user runs the repair cell, retry the two failed guidance launchers; their
incomplete attempts are retained and new attempts use distinct run names.
Then run `scripts/run_gsd_v1_background_completion.py`; it creates
`result/modelnet40_c/gsd_latent_spectral_v1/spectral_only_m100_all15_summary_seed0.json`
and appends a labeled result section to `knowledge/findings_log.md`.

The combined script additionally requires the archived 14 corruption files to
match their recorded hashes against the current Colab dataset before accepting
the new Background result. It records runtime, source and extension identities
in the derived JSON so environment changes remain visible.
