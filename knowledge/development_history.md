# Development History

This is a research-oriented history, not a full changelog. Dates and hashes come from the local Git graph. Inspect the commit itself before relying on an implementation detail because later commits may revert it.

## Upstream and baseline

- `a1d9c306` (2024-11-20): repository README result table. It predates the final WACV publication and differs from the published ModelNet40-C table.
- `107305f` (`main`): upstream/fork baseline ancestry.
- The current branch keeps original TTA logic in `tta.py`; GSD and PxP are separate scripts.

## Initial GSD synthesis — July 2026

- `4cf0670` (2026-07-01): initial GSD integration.
- `71de186` (2026-07-02): restored the original baseline and isolated GSD logic. This separation is methodologically valuable and must be preserved.
- `09580b2` (2026-07-02): changed the graph threshold formula described as a tau fix. The equation still needs visual confirmation against the ICCV paper.
- `58247f7` (2026-07-15): changed gamma/eta mapping in variants/baseline. Later branch state means unequal-rate experiments must inspect current code, not rely on this message.
- `158eae8` (2026-07-15): changed final decode toward updated style conditioning; later baseline history reverted/isolated this behavior.
- `23b1a44` (2026-07-15): made the LION scale factor `3.3885` unconditional.
- `59e41b8` (2026-07-16): integrated scale/step behavior.
- `819c9e5` (2026-07-23): introduced multiband spectral guidance.

## Global/local and dynamic variants — August 2026

- `0124b98` (2026-08-09): added dual-diffusion work.
- `a458cd4` (2026-08-09): made global-shape gradient updates continuous in synchronized mode.
- `c364b0a` (2026-08-25): added dynamic graph work and paper artifacts.
- `4079f49` (2026-08-25): reverted transpose logic, returning to the `B × 2048 × 4` interpretation of flat local latents.
- `8450a83` (2026-08-25): added physical-to-latent spectral architecture.
- `6b34a30`, `25d8bf2`, `38935f1`: exposed/fixed static-style and dynamic-evaluation controls.

## Loss-scale instability signal

Several commits alternate reductions:

- `8a48017`: spectral loss from mean to sum.
- `6712c11`: reverted reduction to mean.
- `04af1af`: changed Chamfer to mean.
- `06561f3`: restored Chamfer sum for baseline comparison.

This sequence is evidence that optimization scale has not yet been normalized. A reported weight is meaningless without its loss reduction, point count, band size, batch size, and observed gradient norm. Future experiments must log all five.

## Scheduler and performance changes

- `9997de8` (2026-08-27): disabled unnecessary graphs in pure-LION tests to address memory use.
- `f63587b` (2026-08-27): changed GSD DDIM scheduler initialization to align with native LION, including `set_alpha_to_one=False` in current GSD code. The original `tta.py` scheduler is not configured identically, so this must be an explicit ablation rather than an unnoticed method difference.
- `6e8b3a3` (2026-09-06): grid search and spectral computation optimizations.

## PixelAsParam-inspired work — September 2026

- `33c4868` (2026-09-06): introduced one-way PCGrad-style projection.
- `4676ddd`: fixed retained computation graphs that caused CUDA out-of-memory behavior.
- `b1da2e9`: added symmetric projection with adjustable deltas.
- `8c68daa`: added file logging to symmetric search/evaluation.
- `654ac93` and `53ba252`: exposed spectral reduction selection.

## Branch interpretation

- `main`: baseline ancestry.
- `dev`: GSD and dual-diffusion development.
- `origin/gsd-tta-improvements`: later dynamic/physical experiments.
- `pxp-gradient-projection`: current line with PxP-inspired variants.

Do not compare branch-level accuracy without recording the exact commit and config. The branches differ in scheduler, reduction, style decode, spectral bands, and logging—not just the headline method.

## Clean restart — 2026-09-12

- Created branch reference `baseline-repro-clean` from main `107305fd7baf40b359f31c07d235599198be7324` with no checkout/commit/push. Current legacy checkout remains `53ba252`; tracked working files unchanged.
- Worktree versus in-place placement and curated untracked knowledge/skill/result-protocol preservation are pending user preference/review. Preserve legacy source PDFs; do not confuse a prepared ref with a clean active code workspace.
- Batch 0 is partial; no algorithm changes and no numerical tests. See `clean_restart_batches.md` and the dated findings entry before Batch 1.

### Same-folder placement follow-up

- User chose the existing repository folder. Clean branch is checked out; original main Python/config/dependency files retained, legacy experiments remain on `pxp-gradient-projection`.
- No new stash: tracked/index state was clean. Existing dev stash is unchanged. Skill/knowledge/result protocol and all selected local PDF sources were preserved, with 23 SHA-256 matches before status-note edits.
- Only curated memory/skill/protocol enter the documentation commit; PDFs and unrelated docs/tmp stay untracked. No push or algorithm changes. The earlier partial placement status is superseded; Batch 1 is next after the handoff review.

## Historical lessons

1. Preserve `tta.py` as an untouched reference path.
2. Name algorithm changes separately from engineering fixes.
3. Pair every loss-reduction change with gradient-norm measurements.
4. Never infer current behavior solely from commit messages; inspect the checked-out file.
5. Archive run outputs before changing branches or resuming an evaluation.
