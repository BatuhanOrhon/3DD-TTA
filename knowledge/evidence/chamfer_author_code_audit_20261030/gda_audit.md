# GDA author-code audit — 2026-10-30

[Code/Paper/Inference/Open] Read-only audit before any GPU test. No author
program, checkpoint, classifier, native extension, Colab cell or GPU experiment
was executed. Only primary-source retrieval, source inspection, local copies
and file/blob hashing were performed. No implementation file was changed.

## Source identity and author association

[Code] Author repository: [yunyuntsai/gda_robust_tta](https://github.com/yunyuntsai/gda_robust_tta),
pinned commit **4197ff4e43b03b896830f5e70980c7e613be5126**, dated
2025-10-06T21:38:44Z; [pinned commit](https://github.com/yunyuntsai/gda_robust_tta/commit/4197ff4e43b03b896830f5e70980c7e613be5126).
The first author's [publication page](https://yunyuntsai.github.io/) links
the matching paper; the repository belongs to the matching GitHub Pages owner,
and its [README](https://github.com/yunyuntsai/gda_robust_tta/blob/4197ff4e43b03b896830f5e70980c7e613be5126/README.md#L1)
explicitly identifies itself as this CVPR2024 paper's source code and links
arXiv2404.00095. The inspected publication entry and
[coauthor project page](https://www.cs.columbia.edu/~junfeng/papers/gda/)
did not directly link the repository. A GitHub connector search restricted to
the author account located it; this is not a guessed repository name.

[Open] This pins the author's2025 public checkout, not the exact historical
execution environment for the2024 paper. The README clones external packages
without immutable dependency commits; runtime equivalence is not established.

[Code] Inspected the README-recommended `main.py`, alternative `main2.py` and
`main_imgnetr.py`, `third_party.py`, `learning/resnet.py`,
`guided_diffusion/gaussian_diffusion.py`, `guided_diffusion/respace.py`,
`scripts/script.sh` and `utils.py`. All ten downloaded files were checked
against the pinned tree's Git blob SHA1, then SHA256-inventoried. No install or
model download was required. Raw connector responses, source bytes and primary
HTML are retained only in ignored
`result/preparation/chamfer_guidance_variants_20261030/author_code/gda/`;
`retrieval_manifest.json` records every source identity and
`primary_pages_manifest.json` records the primary HTML identities.

## Paper versus the released loss branches

[Paper] [GDA, arXiv2404.00095v2, §3 Eqs.7–8](https://arxiv.org/html/2404.00095v2#S3)
defines entropy after averaging predictions over augmentations of a generated
sample. Algorithm1 evaluates guidance through the denoised estimate; §4.2 uses
16 augmentations, and the supplement describes AugMix. These are views of the
generated sample, not a fixed corrupted-input/decoded-output pair. The paper
also includes style/content terms and confidence filtering; those components
are not part of our proposed H arm. The main text's DDIM discussion and
50-step description do not establish the README command's actual sampler or
executed step count; those must be traced in code.

[Code] `main.py:153–160`
([permalink](https://github.com/yunyuntsai/gda_robust_tta/blob/4197ff4e43b03b896830f5e70980c7e613be5126/main.py#L153))
implements the plain `marginal` branch as

\[
p_j=\operatorname{softmax}(z_j),\qquad
\bar p=\frac1K\sum_{j=1}^{K}p_j,\qquad
L_{\mathrm{plain}}=-\sum_c\bar p_c\log\bar p_c.
\]

The log-space `logsumexp` across dimension0 averages **augmentations of one
image**. `main.py:256–261` constructs16 AugMix samples. With the default
adaptation batch size1 (`main.py:311`), dimension0 is the view dimension, not a
batch of unrelated test objects. There is no additional mean across test
objects in this scalar loss. `main_imgnetr.py:253/260` applies the same losses
after its ImageNet-R class mask.

[Code] The README and shell script select **`weighted-marginal`**, not the
plain branch. `main.py:164–199`
([permalink](https://github.com/yunyuntsai/gda_robust_tta/blob/4197ff4e43b03b896830f5e70980c7e613be5126/main.py#L164))
additionally computes the classifier prediction on the original corrupted
input. Let `q` be that prediction, `a=log(bar p)`, `b=log(q)`,
`h_a=H(bar p)`, and `h_b=H(q)`. Its actual computation is

\[
f=\frac{h_ba+h_ab}{h_a+h_b},\qquad
r_c=\exp(f_c)=\bar p_c^{\,h_b/(h_a+h_b)}q_c^{\,h_a/(h_a+h_b)},
\qquad L_{\mathrm{weighted}}=-\sum_c r_cf_c.
\]

There is **no normalization of `r` before the returned loss**. Thus this is
not generally `H(softmax(f))`, and it is not
`H((bar p+q)/2)`. The arithmetic operations are directly visible in the
released code. Whether the unnormalized objective reflects an intended
variation or an author defect is undocumented; this audit does not infer
author intent. We should not copy it while describing it as the paper's
plain marginal entropy. The entropy-derived mixing weights also vary with
the generated prediction, unlike our fixed half weights. `main.py:248–254`
recomputes the original-input logits during each guided step.

## Gradients, frozen states, augmentation and time

[Code] `main.py:226–279`
([permalink](https://github.com/yunyuntsai/gda_robust_tta/blob/4197ff4e43b03b896830f5e70980c7e613be5126/main.py#L226))
detaches the current noisy image, enables its gradient, predicts the clean
image, and blends clean/noisy images with a noise-level-dependent factor
before applying losses. It returns the **negative input gradient** of the
combined loss. `gaussian_diffusion.py:357–370`
([permalink](https://github.com/yunyuntsai/gda_robust_tta/blob/4197ff4e43b03b896830f5e70980c7e613be5126/guided_diffusion/gaussian_diffusion.py#L357))
adds variance times that gradient to the previous-step mean. This has the
loss-minimization sign. It is a local per-step graph, not differentiation
through the complete previous sampling trajectory.

[Code] The classifier is put in evaluation mode at `main.py:348–357`.
Its parameter `requires_grad` flags are not explicitly disabled there, but
the inspected adaptation path has no classifier optimizer/parameter update:
`autograd.grad` requests the image tensor only. Therefore the released code
keeps classifier **values** fixed without our stricter parameter-flag and
buffer/RNG guards. CLIP is explicitly frozen; some diffusion parameters are
re-enabled at `main.py:45–48` for input-gradient computation, again without a
parameter optimizer in this path. These details do not justify unfreezing
our source classifier.

[Code/Inference] `third_party.py:28–46`
([permalink](https://github.com/yunyuntsai/gda_robust_tta/blob/4197ff4e43b03b896830f5e70980c7e613be5126/third_party.py#L28))
mixes a tensor preprocessing path with PIL augmentation branches. PIL
conversion/operations are not a general differentiable augmentation path;
the direct tensor mixture component retains an input-gradient route. We
have not executed the author's TorchVision/version combination and do not
claim every AugMix branch supplies native input gradients. This is a code
caveat, not a tested runtime failure.

[Code] In the README command, `main.py:28` respaces the diffusion to50 states,
`main.py:398–401` selects the ancestral sampler because the string does not
start with `ddim`, and `--skip_timesteps45` leaves **five** reverse iterations.
`gaussian_diffusion.py:619/631` executes those indices. Conditioning runs at
each of these steps. `respace.py:123–128` maps to the original noise indices;
five executed iterations must not be described as the full original
1000-step process or as50 executed guidance steps. Other CLI settings may
select different behavior. With `--ensemble`, `main.py:451–458` chooses the
lower-entropy original/adapted prediction; it does not perform our
four-candidate probability mean.

[Inference] These GDA five-step settings are not the original3DD-TTA
five-step control: models, image/point-cloud domains and losses differ. They
provide no matched accuracy or speed comparison to our10/30-step protocol.

## Comparison with this worktree

[Code] Audited our dirty source against checkout base
`4da29b32f1027806ace77a6bbcf7e6468431883a`, preserving the actual compared
bytes in the ignored `ours/` evidence directory. A base commit alone does
not identify the new implementation. SHA256:

| File | SHA256 of inspected bytes |
| --- | --- |
| `semantic_trajectory.py` | `11ef7b8273941bdf17926243cb94e3db8dab9d29f1b3f642400682f62ca322be` |
| `tta_gsd_v2.py` | `c4c883d08ac3a152de8642ced7ef59ba8bee45d6c9ed61cf749cd2914faa7d5c` |
| `scripts/eval_multibackbone_guidance.py` | `b433be53606b4323b8733abed68a3b3e4698c6e23fc4f2415f73ba9c95cbbfcc` |

[Code] Our `semantic_trajectory.py:33–48` defines

\[
p_i=\operatorname{softmax}F(D(h_{0,t,i},s_{t,i})),\qquad
q_i=\operatorname{stopgrad}\operatorname{softmax}F(\operatorname{FPS}(X_i)),
\]
\[
L_H=\sum_i H\!\left(\frac{p_i+q_i}{2}\right),\qquad
L_{KL}=\frac12\sum_i\left[KL(p_i\Vert q_i)+KL(q_i\Vert p_i)\right].
\]

The two views are paired **within each object**; objects/classes are summed
after constructing that object's marginal. There is no averaging of
probabilities across unrelated objects. The sum matches our summed
Chamfer objective's batch scaling. The symmetric KL is an independent
consistency arm; the author's unused `kl_entropy` helper at `main.py:201–205`
does not make our branch a reproduction of GDA's selected entropy branch.

[Code] `semantic_trajectory.py:13–30` checks frozen/eval models and restores
RNG/buffers. `:86–95` clones the live local/style inputs **without detaching**,
decodes once, retains Point-MAE logits for the loss, and copies detached
logits only for readout. `scripts/eval_multibackbone_guidance.py:432–460`
keeps the guidance-head forward differentiable, but caches detached
corrupted-X logits once per batch. `classifier_backbones.py:33–40` explicitly
freezes all classifier weights and modes. `tta_gsd_v2.py:150–180/282–291`
builds the per-step local/style graph, adds the scaled semantic loss to C,
and subtracts the resulting `eta`/`gamma` gradients. The semantic checkpoints
are5/8 of10 or15/23 of30; the first readout probe and terminal still contribute
to the existing probability mean, and the fixed-X reference supplies no vote.

[Inference] These are intentional adaptations: fixed corrupted-X anchor,
two equally weighted views, per-object sum, frozen3D classifier/LION,
local-and-style gradients, sparse semantic checkpoints, Chamfer fidelity and
unchanged probability-mean readout. They are not author-code parity or the
complete GDA method. A truthful label is **fixed-X semantic input guidance
inspired by marginal-entropy diffusion guidance**, with KL and H isolated.

[Inference] For our H objective,
`H((p+q)/2)=(H(p)+H(q))/2+JS(p,q)`. Since q is fixed, its own entropy has no
gradient, but it biases agreement and decoded confidence. A confident wrong
corrupted-X anchor can therefore pull adaptation toward an incorrect class.
That is a limitation of the chosen objective, not evidence of a coding bug or
an established accuracy improvement.

## Decision and remaining gate

[Code/Inference] Static inspection found no unintended detach, reversed
minimization sign, cross-object marginalization or classifier parameter
update in our semantic path. The non-equivalence to GDA is substantive and
must remain explicit. The author's weighted-log fusion and PIL branches are
not implementation requirements to copy into this intentionally modified3D
experiment.

[Open] Native FPS/grouping/decoder gradients to both local and style latents,
finite/nonzero update magnitudes, weight-zero sham parity, classifier/buffer
and RNG invariance, actual cost and accuracy are **not established by this
read-only audit**. No GPU launch is authorized by this note. The user's
author-repository audit gate and later whole-phase approval remain separate;
the root agent must reconcile all requested method audits before any GPU test.
