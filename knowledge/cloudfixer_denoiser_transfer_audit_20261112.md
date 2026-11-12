# CloudFixer denoiser: training provenance and LION replacement boundary

Date: 2026-11-12. Research branch `research/cloudfixer-denoiser-audit-20261112`,
based on `origin/thesis-integration@3789bcce8f0f9d4e42e50f48cc2ff14def16750b`.
Scope: primary-paper/author-code inspection only. No training, model inference,
GPU allocation, checkpoint substitution or peer-runtime action.

## Answer and evidence labels

**[Paper/Code]** CloudFixer uses the Point-E-derived `base40M-uncond`
`PointDiffusionTransformer`, with an XYZ-only input/output modification. Its
authors explicitly report training the diffusion model on each source dataset.
It is not evidence that an unchanged public OpenAI Point-E checkpoint was used.

**[Inference/Open]** Replacing the LION prior might improve our results, but
neither that improvement nor its magnitude has been measured. The training
domain, latent/physical representation, scheduler, normalization, adaptation
rule and readout budget must be separated from architecture in attribution.
`[Paper]` denotes published statements; `[Code]` inspected sources;
`[Inference]` deductions; `[Open]` missing evidence. No new `[Run]` result exists.

## 1. What model actually runs?

**[Paper]** [CloudFixer ECCV2024, section5.1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04603.pdf),
printed page8, identifies Point-E `base40M-uncond` and source-specific training.

**[Code]** Author repository pinned to
`f043e24414fcf4165fdc892abff21215c9989141`:

- [build_model.py](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/diffusion/build_model.py#L7)
  instantiates this configuration and wraps it in `DiffusionModel(n_dims=3)`.
- [transformer.py](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/diffusion/transformer.py#L41)
  specifies12 layers, width512,8 attention heads,3 input channels,3 output
  channels and a timestep token. It has no image/text/class conditioning input.
  The `40M` label identifies the model family, not a newly counted exact
  checkpoint parameter total.
- [Original Point-E configuration](https://github.com/openai/point-e/blob/fc8a607c08a3ea804cc82bf1ef8628f88a3a5d2f/point_e/models/configs.py)
  has6 input and12 output channels for the same family name. The CloudFixer
  source explicitly edits those dimensions to3/3. Do not equate the weights.
- [diffusion.py](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/diffusion/diffusion.py#L300)
  passes physical coordinates `B x N x3` to the Transformer and predicts noise;
  its `return_x0_est` branch reconstructs the clean-coordinate estimate.
  There is no LION-style VAE/global-style contract in this path.

**[Code]** `n_ctx=1024` is a configuration value, not a demonstrated fixed
input-point limit: the matching forward assertion is commented out and the
released run script selects `subsample=2048`. No capacity benchmark was run.

## 2. Did the authors train it themselves?

**[Paper/Code]** Yes. The paper's statement is corroborated by
[train_dm.sh](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/scripts/train_dm.sh)
and [train_dm.py](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/train_dm.py#L184).
The normal non-resume path constructs a model without loading OpenAI weights;
checkpoint loading in the training entrypoint is for an explicit resume.
Thus the published training path supports training from initialized weights,
rather than merely using an off-the-shelf OpenAI checkpoint. Individual released
weight provenance is not independently reconstructed without training logs.

| Released training recipe | Setting |
|---|---|
| ModelNet40 data | clean `modelnet40_ply_hdf5_2048`, train partition |
| ModelNet40 epoch budget |5000 |
| PointDA ModelNet / ShapeNet / ScanNet budgets |20000 /5000 /20000 |
| Diffusion grid |500, `polynomial_2` |
| Loss setting |`l2`, epsilon prediction |
| Optimizer |AdamW, learning rate2e-4 |
| ModelNet40 batch / gradient accumulation |32 /2 |
| Coordinate convention |`unit_std` preparation |

These are code settings, not verified elapsed training times or proof that
every released checkpoint completed the entire budget. Only the ModelNet40
function is called by default; the other three functions are provided but
commented out at the script's end. The experiment name contains
`random_scale`, but the command does not enable `--random_scale`; an experiment
name is not evidence of an active augmentation.

**[Code/Open]** Training uses clean train examples; its ModelNet validation
loader uses the clean test partition and selects checkpoints by evaluation NLL.
This is not gradient training on corrupted test inputs, but do not describe the
released recipe as having an independently held-out validation split.
The script does not establish a dedicated ScanObjectNN-trained denoiser;
PointDA's ScanNet is a different dataset.

## 3. Where are the trained weights?

**[Code]** The [README](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/README.md#pre-trained-models)
announces classifier and diffusion checkpoints in an
[author Google Drive folder](https://drive.google.com/drive/folders/1nkoyupwKIxUOkoI8qmrqFb03EpjPmB6e).
[run_cloudfixer.sh](https://github.com/shimazing/CloudFixer/blob/f043e24414fcf4165fdc892abff21215c9989141/scripts/run_cloudfixer.sh)
expects `checkpoints/diffusion_model_transformer_modelnet40.npy` and500-step
polynomial diffusion. The `.npy` suffix alone does not define the serialization;
the adaptation entrypoint uses PyTorch checkpoint loading.

**[Open]** The Drive folder could not be inspected by the web reader this turn.
No weight was downloaded, deserialized or hashed. README publication and the
expected filename are established; present folder contents, exact checkpoint
bytes, selection epoch and raw-versus-EMA identity remain unverified here.
Reusing verified author weights could avoid new training; availability must
be checked before claiming a ready-to-run asset.

## 4. Could this improve our method?

**[Paper]** [3DD-TTA section4](https://arxiv.org/html/2411.14495v1) uses a
ShapeNet-pretrained LION and treats ModelNet40/ScanObjectNN as transfer datasets.
CloudFixer's ModelNet40 recipe instead learns that source's geometry directly.
**[Inference]** Better source-domain alignment is a plausible contributor to
the accuracy difference, not a measured explanation or proof of a superior
Transformer. Online CloudFixer-O updates, classifier checkpoints, geometric
transforms and voting are additional confounds.

**[Code/Inference]** An unchanged replacement of `lion.priors[1]` is invalid:
our prior consumes a four-channel local latent conditioned on global style;
the CloudFixer model consumes physical XYZ under its own training distribution.
Matching three coordinates numerically does not make the spaces interchangeable.

A physical-space adaptation could reuse the **principles** of our method:
point-cloud noising/reverse sampling, frozen trimmed-neighbor proximal
correction and trajectory probability averaging. It would operate on XYZ,
remove the LION encoder/decoder/global-style path, and require the correct
polynomial noise schedule, timestep convention, normalization and clean-state
transport. `N8`, `.1`, `.95` and readouts2/4/6 are not prevalidated settings for
the new denoiser. Removing VAE passes may reduce overhead, but attention and
sampling costs mean no speed or memory advantage is established.

**[Open]** Discriminating evidence would compare verified source-matched
weights and frozen classifier/preprocessing under fixed compute and include
ordinary denoising, proximal correction and proximal+PM separately. A prior
swap plus a latent-to-physical rewrite cannot isolate architecture alone.
This review registers no implementation/GPU phase and makes no baseline change.

## Verification and publication scope

Selected scope: this note, compact primary-source hashes, and navigation,
findings/method/open-question links. Production code, checkpoints, raw source
dumps, external training and all peer work are excluded. Source texts were
retrieved through the GitHub connector; local raw copies are research temp
files. [Compact source receipt](evidence/cloudfixer_denoiser_20261112/sources.json)
records exact inspected file bytes. Documentation checks only; no model tests
or scientific results. The commit containing this note is the publication
identity; the source branch and base are recorded above.
