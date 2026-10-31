# Shape-latent research search ledger — 2026-10-31

Scope: research only in worktree7bf0, integration/chamfer-guidance-variants-20261030,
HEAD a82200adbe117ce56a134dc3c22601cb73dc2654. No branch switch or runtime launch.

## Older experiment search

- Current knowledge Markdown searched case-insensitively for dual sync/seq,
  global diffusion, shape latent, global latent and latent diffusion.
- All reachable refs:249 commits touching knowledge. Earliest knowledge
  b31fd23 (2026-09-12). Historical names inspected; no deleted Markdown
  paths from `git log --all --diff-filter=D --name-only --format= -- '*.md'`.
- Seven commits touch the three legacy dual scripts, all2026-08-09;
  initial0124b98, latest a458cd46bba1288a3f41302548bc777700cbc7d4.
- `3dd_tta.ipynb` at a458cd4 and53ba252 parsed as JSON: sources and saved text
  outputs contain no dual/global_steps/shape-diffusion experiment match.
- Filename search over result directories in the requested worktree, original
  project and preserved chamfer-guidance worktree for dual/global/latent
  names found source snapshots but no matching historical dual result archive.
  The original project's knowledge filenames were also included.
- These searches cover reachable local history and the named roots. They do
  not inspect inaccessible historical machines, dangling Git objects, remote
  Drive archives or an unknown external result directory. No exact old trial
  command, source receipt, accuracy or runtime was recovered.
- Decoder-control all15 raw ZIPs are a different study, physically found and
  rehashed/recounted by recount_decoder_control.py. Gaussian/Impulse pilot
  ZIPs were inspected separately and not pooled with the all15 control.

## Primary literature and code

- LION project, NeurIPS2022/full arXiv paper; official train_2prior,
  train_prior and wrapper at7711b3d. Architecture checked against current
  vae_adain, local encoder/decoder/prior, AdaGN, global prior and all55 config.
- Original3DD WACV2025 paper and official tta.py at107305fd. Updated
  conditioning and initial-z final decode confirmed; no inspected explanation
  establishing a deliberate negative-global-update ablation.
- SDEdit2108.01073v2; DPS2209.14687v4 and official condition_methods.py;
  PSLD NeurIPS2023 venue PDF, Sec.2.1 Eq.6–7; DAPS2407.01521v3 and official
  project/repo. DAPS implementation was not executed or fully audited.
- Recent primary full texts: L-DPS2606.26592v1, Sec.4.1/4.2 Eq.27/30/32;
  CLAMP2605.27990v1 and official project/repo. L-DPS official code was not
  verified. CLAMP matrix-free solver was not implemented or benchmarked.
- Supplemental broad search queries:
  `LION shape latent test time adaptation diffusion 2026 global latent guidance corruption`;
  `"latent diffusion" "test-time adaptation" 3D 2025 2026`;
  `CLAMP diffusion inverse problems 2605.27990 manifold 2026`.
- Additional surfaced works were not promoted from search abstracts to
  validated implementation candidates. Dual Ascent Diffusion/CVPR2026,
  discriminator-guided image adaptation and corruption-editor methods would
  need separate protocol/training/full-text review. No exhaustive absence or
  novelty claim follows from this search.
- Six primary PDFs downloaded directly and text extracted; URLs, timestamps,
  HTTP status, byte count and SHA in primary_retrieval.json. LION extraction
  warned of form-XObject limits, so no quantitative figure-only claim used.
  Repo heads/dates in author_pins.json;19 inspected file hashes in
  source_manifest.json. Ignored local source/PDF copies remain under
  result/preparation/shape_latent_research_20261031/.

## Interpretation constraints

The180-cache analysis measures between-corruption/sample variation, not clean
latent error. Eight CPU checks verify formulas/RNG/static legacy behavior,
not model accuracy or native gradients. No pretrained inference, Colab/GPU,
production edits, commit/push or remote archival operation occurred.
