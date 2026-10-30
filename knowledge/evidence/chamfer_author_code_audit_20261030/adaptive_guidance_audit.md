# Adaptive Guidance: author-code availability and sparse-SCD audit

2026-10-30. Evidence labels: **[Paper]**, **[Code]**, **[Inference]**, **[Open]**.
Read-only audit; no GPU/Colab access, algorithm execution or core-code edits.

## Finding and source identity

**[Code/Open] No official sampling implementation was located.** The official
paper links the [author project](https://bcv-uniandes.github.io/adaptiveguidance-wp/).
Its [GitHub repository](https://github.com/BCV-Uniandes/adaptiveguidance-wp)
describes itself as website source. The complete, untruncated recursive tree
contains HTML/CSS/JavaScript and media, with no Python sampler or notebook.
The HTML's Code button is commented out and points to the unrelated Nerfies
template; the footer's source-code link also credits that website template.
Those links are not an Adaptive Guidance implementation.

The website repository is pinned at
[`2fe5f7f03063476e5873eb5e32be8e6038147e3b`](https://github.com/BCV-Uniandes/adaptiveguidance-wp/commit/2fe5f7f03063476e5873eb5e32be8e6038147e3b),
2023-12-21. This is a **website commit**, not an algorithm commit.
Pinned [README](https://github.com/BCV-Uniandes/adaptiveguidance-wp/blob/2fe5f7f03063476e5873eb5e32be8e6038147e3b/README.md)
and [HTML](https://github.com/BCV-Uniandes/adaptiveguidance-wp/blob/2fe5f7f03063476e5873eb5e32be8e6038147e3b/index.html)
were retrieved through the GitHub connector; saved bytes match their Git blob
SHAs `bb1a2a480b3f0c4432625da120a25dee77e672d7` and
`cb502ea670a0b79991b88779d28bdea299579e7c` exactly.

**[Code/Open] Search boundary:** official arXiv/AAAI paper, linked project and
complete website tree, current [Castillo](https://angelacast135.github.io/)
and [Pérez](https://www.juancprzs.com/) homepages, and bounded title/ID/author
GitHub searches. Both author homepages return to the same project without a
sampler link. This establishes an availability limit, not proof that private or
unindexed code cannot exist. Unaffiliated ComfyUI/T-GATE/FastVideo repositories
were not substituted for author code.

The GitHub connector's repository query `adaptiveguidance` returned only the
website repository; `"Adaptive Guidance" Castillo` returned no repositories.
Raw structured responses are saved separately as
`github_repository_search_results.json`; search indexing is a further bound on
this negative result.

## What the primary paper specifies

**[Paper]** CFG combines conditional and unconditional noise predictions.
AG preserves denoising iterations, computes their cosine similarity and switches
once from CFG to conditional-only predictions after a threshold crossing.
It does not prescribe every-third-step guidance. Under the standard CFG formula,
conditional-only corresponds to guidance scale1, not scale0. The published
AAAI policy and cosine equations are Eq.7 and Eq.8 on PDFp5/printed1966;
arXivv1 places the cosine in Eq.7 and the method in §5.
[AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/download/32192/34347),
[arXiv §5](https://arxiv.org/html/2312.12487v1#S5).

**[Paper]** LinearAG is separate: it estimates unconditional predictions with
OLS coefficients fitted to200 saved trajectories, combining conditional/current
and past unconditional scores and interleaving exact/estimated CFG. It is not
simple reuse of the last gradient. See AAAI Eq.10, PDFp7/printed1968 and
[arXiv §5.1](https://arxiv.org/html/2312.12487v1#S5.SS1).

**[Open]** Without author implementation, the exact tensor axes, batch gating,
numerical epsilon, threshold inequality, negative-prompt branches, cache
detach/lifetime and PyTorch `no_grad` placement cannot be audited. The stated
inference equations need noise predictions rather than a differentiable
external-loss backward; the paper's separate NAS search uses backpropagation.
No source-code-level verification of either path is claimed.

## Comparison with our implemented sparse arm

**[Code]** Audit base is branch
`research/chamfer-guidance-variants-20261030`, HEAD
`4da29b32f1027806ace77a6bbcf7e6468431883a`, with uncommitted experiment files.
Exact audited copies and SHA256 identities are in the raw audit manifest below;
HEAD alone does not identify those working-tree changes.

| Aspect | AG paper | Our sparse-SCD code | Interpretation |
|---|---|---|---|
| Removed computation | Unconditional forward | Selected loss/backward/update work | Intentional transfer to a different cost |
| Gate | Prediction cosine threshold | Fixed one-based loop membership | Not adaptive AG |
| Later behavior | One-way conditional-only suffix | Guidance resumes at the next selected count | Intentional departure |
| Final step | Governed by the switch | Always guided | Own design choice |
| History approximation | LinearAG OLS score history | None | No LinearAG implementation |

**[Code]** [`guidance_loop_counts`](../../../chamfer_guidance_variants.py)
returns `{1,4,7,...} ∪ {horizon}` for interval3. The
[`sampler`](../../../tta_gsd_v2.py), lines141–186, keeps all active scheduler
timesteps. `step_requires_guidance` controls latent/style `requires_grad` and
`torch.set_grad_enabled` around the local prior and DDIM scheduler. Every loop
still evaluates the local prior. Off steps take `prev_sample` directly and
continue before Chamfer diagnostics, loss and backward. Style carries its last
update without an off-step guidance update.

**[Code]** Guided steps retain the actual prior Jacobian and update both local
latent and style. Freezing model parameters does not detach these input
derivatives. There is no cached prior-output surrogate, no cosine calculation,
no unconditional LION call and no physical CloudFixer R/Δ optimization.

**[Code]** Consequences: 10 local-prior forwards with guidance at1/4/7/10
(4 backwards); 30 forwards with guidance at1/4/7/10/13/16/19/22/25/28/30
(11 backwards). The existing test source asserts those full forward schedules,
grad-enabled counts and `chamfer_calls=2×backward_calls`:
[`test_sparse_uses_same_timesteps_with_four_or_eleven_backwards`](../../../tests/test_chamfer_guidance_variants.py).
Those tests were inspected rather than rerun in this audit; they are fake-native
CPU contracts and do not establish CUDA speed or accuracy.

**[Code]** The ordinary
[`SharedTrajectoryObserver`](../../../shared_trajectory.py), lines20/60–85,
still probes3/5/8 for horizon10 or8/15/23 for horizon30, then the terminal.
Its isolated probe detaches live inputs and uses `no_grad`; sparse backward
does not remove these decoder/classifier readout costs. Prepared encoder/noise
replay is a paired-experiment cache, not AG/LinearAG score caching.

## Bugs, departures and remaining risks

**[Inference]** No unintended sparse-loop implementation error was identified
by this static trace relative to our registered fixed schedule. It is correct
to label this arm **fixed-interval sparse SCD backward, inspired by guidance
redundancy**. Calling it an official AG reproduction or an author-code-verified
port would be incorrect. Existing literature review already distinguishes the
transfer; the present record adds the code-availability limit.

**[Code/Inference]** A diagnostic caveat remains: off steps insert zeros in loss
history, so `mean_raw_chamfer` averages unevaluated entries as zero. Its reduction
must not be presented as improved geometric fidelity. Update-norm means cover
guided updates rather than the whole trajectory; sparse also reduces total
guidance exposure at unchanged eta/gamma. Neither is an author-algorithm bug.
The diagnostic meaning should be explicit in analysis.

**[Inference/Open]** Fewer backwards are not fewer local-prior forwards, and
published AG NFE savings do not forecast our wall time. Keep actual forward,
backward, Chamfer, decode/head and elapsed-time counters. Accuracy preservation,
native autograd behavior and total speed remain unmeasured. This audit neither
launches a test nor grants GPU-run approval.

## Raw evidence

Ignored task-owned directory:
`result/preparation/chamfer_guidance_variants_20261030/author_code/adaptive_guidance/`.
It contains connector-derived website/tree/commit evidence, exact local source
copies, `audit_manifest.json`, and the official nine-page AAAI PDF
(19,469,470bytes, SHA256
`35d6700d770fad39ec7c45eb8e58dbd32403f9f83b3537fc70376ab2fc368a0b`).
Text was extracted with transient `pypdf`; font-encoding warnings occurred,
so equation interpretation was cross-checked against the arXiv HTML. No project
dependency file or environment was modified.
