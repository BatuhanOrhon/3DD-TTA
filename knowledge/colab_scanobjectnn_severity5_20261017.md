# ScanObjectNN severity-5 audit and Colab plan

Date: 2026-10-17. Evidence: [Code], [Run], [Inference], [User decision], [Open].

## Implemented Colab continuation (current; supersedes draft commands below)

[User decision] Gaussian + clean is the approved first scope. Priority is
the authors' published corruption procedure. The numerical choices below
are fixed for provenance, not tuned to obtain an accuracy target.

[Code/CPU/Open] The notebook now has a **ScanObjectNN: author-code Gaussian
severity 5 + clean -> Drive** section immediately after its first6 preparation
code cells. Run its generation/backup cell and then its download cell in the
already prepared Colab session. It does CPU data preparation only. The cell
contains its own hashed source bundle, so it also works in an existing `dev`
session without a Git write, new commit, source upload or environment rebuild.
Leave `ORIGINAL_SOURCE=''` for unique main_split discovery under
`/content/drive/MyDrive/thesis/dataset`, or set the exact original H5/ZIP path.
Ambiguity, different splits and unexpected shapes fail before generation.
The cell reuses an already complete attempt rather than regenerating on rerun;
an incomplete attempt must be downloaded/reviewed before an explicit new run.

The fifth preparation code cell now restores a complete, verified package if
present; on the first run without one it reports that generation is needed.
No need to rerun preparation in an already initialized session: the new
generation cell also bootstraps its embedded tools. All other original30
cell objects (except that intentional cell5 addition) are preserved, including
prior Legacy/sampling cells; three new cells were inserted, for33 in total.
Preservation/bundle receipt:
`result/preparation/scanobjectnn_s5_notebook_receipt_20261017.json`.

Generation executes unchanged `normalize`, `uniform_noise` and `gaussian_noise`
function ASTs from pinned author source at upstream `107305f`, not our own
replacement Gaussian algorithm. Full source is preserved in
`result/preparation/scanobjectnn_author_generator_107305f.py`, SHA-256
`3c18c0b3a7018d84f4517a5b431eefab8a52427dba49385d294b69fa645cd8bb`.
The adapter normalizes no raw/clean data and does not reorder examples.
It only flattens the H5 label vector, checks the main_split test schema,
applies explicit severity5 and seed0, and skips unrelated exports/imports.
The original script's list-shaped loader/label serialization issue is not
copied into the output layout.

To match the published default MAP order through Gaussian, the adapter first
executes Uniform with its point permutation and random draws, discards that
output, then executes Gaussian with its own point permutation. Thus the RNG
sequence differs intentionally from the earlier draft's Gaussian-only CLI
command below, which is now historical design text. A separate CPU replay
checks the same point selection, sigma0.03, float32 and clipping exactly.
Direct raw H5 values/order are saved as `data_original.npy` for the control.
Seed0 is our reproducibility setting: author seed and original experimental
file identities remain unpublished, so byte-identical paper data is not claimed.

Verified archives go to:

```text
/content/drive/MyDrive/thesis/dataset/scanobjectnn_c_s5_seed0/
  runs/<fresh-UTC-ID>/dataset.zip
  runs/<fresh-UTC-ID>/receipt.json
  latest_gaussian.json
```

The ZIP includes `dataset/{data_gaussian_5.npy,data_original.npy,label.npy,
manifest.json}` plus source, command/environment, status and generation log
under `audit/`. Original H5 bytes are identified by hash rather than copied
into the handoff. A failed attempt is backed up with complete=false and cannot
replace `latest_gaussian.json`. Successful publication requires array/coverage
validation, archive CRC/path validation and exact Drive-copy SHA/size equality.
Future restore verifies those identities and the pinned source again, extracts
to a fresh content-addressed local directory and sets `SCANOBJECTNN_S5_ROOT`.
Use that path for the later source evaluation; never infer the data root from
the old severity8 `scanobjectnn_c` directory.

Six focused synthetic CPU checks pass: author Gaussian/sigma/order replay and
clean identity; Drive round trip/reload; severity/label tamper rejection; failed
attempt non-promotion; unsafe ZIP rejection; changed archive rejection; input
split/ambiguity checks (some are combined in one test). Python3.8 syntax,
embedded bundle identity and original-cell preservation also pass. Actual H5
reading and Drive/Colab behavior await the user-run handoff; local h5py is not
installed and no real-data generation, inference or accuracy run was performed.

After running, provide the full downloaded ZIP for review before accuracy
evaluation or expansion to other corruptions. Do not Run all. A hard runtime
reset before the cell reaches Drive backup can lose local partial output;
the download/recovery cell can recover an ordinary error in a live session.

## Finding and scope

[User decision] Temporarily switch from sampling-consistency development to
checking ScanObjectNN source accuracy and preparing severity-5 data from the
original dataset already on Drive. GPU evaluation remains user-operated.

[Code] Inspected `main:data/readme.md` at
`107305fd7baf40b359f31c07d235599198be7324`. Its layout names
`scanobjectnn_c/data_*_5.npy`. Its command points to
`datasets/create_corrupted_dataset.py`, which is absent from that tree;
the actual generator is `datasets_mate/create_corrupted_dataset.py`.
The main generator hardcodes `for sev in [8]`. Thus the README's example
command alone does not ensure severity 5. The current generator accepts
`--severity`, but still defaults to 8: explicitly pass `--severity 5`.

[Run] Reopened all three ScanObjectNN result ZIPs; their CRC checks pass.
The Gaussian source-only archive's command, config and CSV all specify
severity8, seed0, batch32, 581 examples and 108 correct (18.588640%).
Its selected data path is `data_gaussian_8.npy`, not a severity-5 file.
The local array SHA-256 matches the recorded runtime identity:
`586b8cae83b7eda99b29da4b8c9ff9d6d0a48590b123d8bdba4d50a9f25492ed`.
Label SHA-256 also matches:
`bb1456708f5724e43abf01e14315c0b3059af68e6fe8531eb97aa4c38eeb7a68`.
Clean source accuracy was415/581 (71.428571%); 3DD-TTA was only a
64-example severity8 smoke. No severity5 ScanObjectNN result was found.

[Code/Inference] Gaussian sigma is0.03 at severity5 and0.045 at severity8,
followed by clipping to[-1,1]. This is a concrete protocol mismatch;
it does not prove that severity alone explains the low accuracy. Raw
coordinate scale/clipping, checkpoint/split compatibility and labels still
need inspection. No extra normalization is introduced in this experiment.

## Source and environment preflight (Colab cell 1)

The user confirms that the first SIX CODE cells (not counting the opening
markdown cell) of `scripts/gsd_guidance_ablations_colab.ipynb` prepare their
working GPU session. These are notebook indices1..6. Reuse that session and
run the ScanObjectNN work after preparation. No repeat environment setup is
required by this plan. Inspection confirms those cells clone the fork and
checkout `dev`, create `3dd_tta_env`, mount Drive, install/compile dependencies,
prepare ModelNet data and copy ModelNet/LION weights. Record the actual `dev`
source at runtime; do not assume it is the audited `gsd-smooth-spectrum`
revision. The sixth cell explicitly copies `modelnet_jt.pth`, so separately
verify/copy `scanobject_jt.pth` before source evaluation. Data generation
itself needs no classifier weights.

Use the already prepared `/content/3DD-TTA` and `3dd_tta_env` environment.
Do not run old sampling/trajectory inference cells. Record Colab commit and
dirty source hashes rather than assuming they equal this local worktree.
The audited source base here is `gsd-smooth-spectrum@a351e66ec58f08f3a2104e46cab2cb8b20cdf961`.
The generator, runner and ScanObjectNN YAML match HEAD after CRLF/LF
normalization; unrelated dirty files are not required for this plan.

Check the generator exists and supports `--severity`, `--corruptions`, `--seed`.
The old main generator is unsuitable as-is. The current `--skip_ply` option
is parsed but not used by its export loop, so do not claim it suppresses PLY
output. Accept the extra PLY files in the isolated staging directory.

Run dependency and asset checks in `3dd_tta_env`: numpy, h5py and open3d for
Gaussian generation, plus the existing Point-MAE/native CUDA environment
for source evaluation. Gaussian needs neither PyGeM deformation nor meshes.
Check `cfgs/tta_scanobj.yaml`, `pointnet_ckpts/scanobject_jt.pth`, GPU access
and the runner's `--dataset-name scanobjectnn-c` support.

For comparison with the old pilot, the classifier checkpoint must have
SHA-256 `53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac`.
The archived YAML SHA is
`bb0f6adecbf60b7ae5e81efe4303c2344bec2518d832580a1d82d9489cac92a4`;
this equals current HEAD bytes and the local YAML after newline normalization.
Record actual bytes and distinguish newline-only changes from config changes.

## Original-data audit and isolated paths (Colab cell 2)

Mount Drive and locate `h5_files/main_split/test_objectdataset.h5` beneath
the user's dataset directory. Confirm the exact file before generation;
do not substitute a perturbed variant or training split. Original source
is expected to contain581 examples,2048 xyz points and labels0..14.
Read only this test H5, not the2309-example training H5.

Record source H5 SHA/size, field shapes/dtypes, label counts, finite-coordinate
checks, coordinate min/max and the fraction outside[-1,1]. Check the raw H5
labels equal the prior label array when available; save that comparison.
Measure clipping fractions after generation, because the Gaussian function
clips coordinates. Out-of-range raw geometry can make clipping consequential.

Use a unique UTC run root, for example:

```text
/content/scanobjectnn_s5_<UTC>/
  scanobjectnn/h5_files/main_split/test_objectdataset.h5  # symlink to original H5
  scanobjectnn_c/                                    # NEW generated output
  audit/                                            # commands, hashes, inventories
  results/                                          # source-only run bundles
```

The generator concatenates `--main_path` strings and overrides both
`--dataset_path` and `--corrupted_dataset_path`. Supply a trailing slash
on `--main_path`; the staging layout above is required. Require the new
root not to exist before initializing it. Retain the resulting path for
later cells. Do not reuse the old Drive `scanobjectnn_c` output directory.

## Gaussian generation and identity validation (Colab cell 3)

From `/content/3DD-TTA`, replace `<NEW_ROOT>` with the initialized run root:

```bash
conda run --no-capture-output -n 3dd_tta_env python -u datasets_mate/create_corrupted_dataset.py --dataset scanobjectnn --main_path <NEW_ROOT>/ --severity 5 --corruptions gaussian --seed 0
```

Capture the exact command, stdout/stderr, environment and generator SHA.
Require successful completion with `Done with Corruption: gaussian with Severity: 5`.
Do not rename an `_8.npy` file to `_5.npy`.

Before evaluation, use numpy/h5py in the same environment to verify:

- `data_gaussian_5.npy` is float32, shape(581,2048,3), finite and within[-1,1].
- Generated `label.npy` is one-dimensional and exactly equals H5 labels
  in original example order; require all15 classes and matching counts.
- Save H5 `data` directly as `data_original.npy`, preserving values and
  original point/example order. Verify exact equality on reload. This
  reproduces the direct clean control rather than the generator's shuffled
  `clean.npy` convention.
- Record SHA/size/shape/dtype for every array and source file. The generator
  permutes the point dimension before Gaussian noise; example order is retained.
- Repeat Gaussian generation once in a separate fresh CPU staging root
  with the same H5, seed, source and environment; require identical array
  hashes. This checks generation only, without extra GPU evaluation.

For additional severity assurance, replay the generator's NumPy sequence
in CPU audit: seed0 -> `choice(2048,2048,replace=False)` -> per-example
`normal(size=(2048,3))*0.03` -> float32 -> clip[-1,1]. Compare the saved
Gaussian array exactly with that reconstruction and record the outcome.
Keep source/data untouched if validation fails; preserve logs for review.

## Full clean and Gaussian source-only controls (Colab cell 4)

Use the same checkpoint/YAML, FPS1024, batch32 and seed0 for both. Run clean
and Gaussian as separate invocations, evaluating all581 examples each.
The runner uses15 output classes for `scanobjectnn-c`. LION is not loaded
by source-only evaluation. No adaptation is needed at this stage.

```bash
conda run --no-capture-output -n 3dd_tta_env python -u run_baseline.py --method source_only --dataset-name scanobjectnn-c --pointmae_config ./cfgs/tta_scanobj.yaml --pointmae_ckpt ./pointnet_ckpts/scanobject_jt.pth --dataset_root <NEW_ROOT>/scanobjectnn_c --label_path <NEW_ROOT>/scanobjectnn_c/label.npy --corruptions original --severity 0 --max-batches 0 --batch_size 32 --seed 0 --run-name scanobjectnn-clean-s5-audit-seed0 --result-root <NEW_ROOT>/results

conda run --no-capture-output -n 3dd_tta_env python -u run_baseline.py --method source_only --dataset-name scanobjectnn-c --pointmae_config ./cfgs/tta_scanobj.yaml --pointmae_ckpt ./pointnet_ckpts/scanobject_jt.pth --dataset_root <NEW_ROOT>/scanobjectnn_c --label_path <NEW_ROOT>/scanobjectnn_c/label.npy --corruptions gaussian --severity 5 --max-batches 0 --batch_size 32 --seed 0 --run-name scanobjectnn-gaussian-s5-seed0 --result-root <NEW_ROOT>/results
```

Keep the old clean71.43% and Gaussian-s8 18.59% as dated references. A new
clean mismatch requires inspection of H5/checkpoint/YAML/preprocessing/native
identities. The new versus old Gaussian difference is descriptive unless
generation randomness/source/assets are also matched. Do not promise an
accuracy target or attribute all change to severity from those two runs.

## Packaging and conditional expansion (Colab cell 5)

Package the complete new run directory with generated arrays, source-H5
identity (not necessarily a duplicate H5), generation/replay logs, source
identity, environment, audit JSON and both seven-file result bundles.
Preserve partial/failed outputs too. Create a fresh ZIP, verify CRC and
safe member paths, compute SHA-256 and copy to a new Drive destination;
verify copied SHA before browser download. Provide the entire handoff ZIP
for review, following `result/README.md`; omit credentials.

After Gaussian/input review, expand to the13 array-based benchmark corruption
names at severity5: uniform, gaussian, background, impulse, upsampling,
distortion_rbf, distortion_rbf_inv, density, density_inc, shear, rotation,
cutout, distortion. Use one declared list/order and seed per fresh production
dataset; random draws depend on generation order. If regenerating Gaussian
as part of that production set, label it as a new generated identity.
Record resulting point counts, which vary by corruption. RBF/FFD need the
correct PyGeM deformation dependency, not an unrelated package with that name.
Do not silently include the generator's extra `scale` corruption in macro15.

Occlusion and lidar need a separately checked mesh/rendering pipeline; the
original test H5 alone does not establish that pipeline. H5-only generation
must not be reported as a complete15-corruption benchmark. Any synthesized
meshes require explicit provenance and would define an additional protocol.

[Falsifier/Open] If verified severity5 Gaussian remains poor with a matching
clean control, severity8 is insufficient as the explanation. Inspect
coordinate/clipping behavior, checkpoint/split compatibility and preprocessing
before TTA or new guidance. No new generation, GPU inference, accuracy or
speed result has been obtained during this local audit.

## Primary-source follow-up: why8, and evidence for5

[Code] Local `upstream/main` is `107305fd7baf40b359f31c07d235599198be7324`.
The author's initial public commit
`e5b7278b35dda713e96df55be55e93fb8d10266e` (2024-12-05) already has
`for sev in [8]` in the generator. Fork commit
`1f28b29146ee4f2dd4469873e7b09834fa02dc04` (2026-09-12) replaced that
constant with `--severity`, retaining default8; its pilot note explicitly
selected8. Thus8 was inherited from generation code, not derived from the
paper's ScanObjectNN evaluation protocol. No documented experimental reason
for preferring8 in that pilot was found. The author's motive for leaving8
in the released generator remains unknown.

[Code] The upstream evaluation chain is different: `main_3dd_tta.py` accepts
`scanobjectnn-c`, then calls `PointDataset`; upstream
`utilities_3dd_tta.py:22` loads `data_<corruption>_5.npy` unconditionally,
including ScanObjectNN. This was verified in local upstream history and
against the live official raw files on2026-10-17:
https://raw.githubusercontent.com/hamidreza-dastmalchi/3DD-TTA/main/main_3dd_tta.py
https://raw.githubusercontent.com/hamidreza-dastmalchi/3DD-TTA/main/utilities_3dd_tta.py
The live generator endpoint could not be fetched; its8 claim above refers
to the pinned local upstream tree, not a fresh remote-generator audit.

[Paper] Main paper Section4.1 says ScanObjectNN-c uses the ModelNet40-c
corruption code; it does not explicitly name a severity number. Official
supplement Appendix A specifies Gaussian sigma0.03, uniform range+/-0.05,
background5%, impulse10%, shear+/-0.25 and rotation+/-15degrees. Comparing
these numeric parameters with the generator tables maps each to severity5
(background uses integer floor N//20). For Gaussian, this gives direct
parameter evidence independently of the `_5.npy` filenames. Main Table3
reports ScanObjectNN Gaussian source59.0% and 3DD-TTA68.0%, with means61.5%
and64.3%; these are published references, not predictions for our assets.

Primary sources:
https://arxiv.org/html/2411.14495v1#S4.SS1
https://arxiv.org/html/2411.14495v1#S4.SS3
https://openaccess.thecvf.com/content/WACV2025/supplemental/Dastmalchi_Test-Time_Adaptation_of_WACV_2025_supplemental.pdf

[Inference/Limit] Severity5 is the supported released evaluation setting and
the Appendix-A Gaussian setting. It is not a proof of byte-identical authors'
experimental data. Appendix prose is not fully identical to every generator
function: for example cut-out describes five centers/100 neighbors while
the inherited severity5 table uses10 iterations/30 neighbors. Preserve the
released-code definition for our planned dataset and disclose this discrepancy;
do not silently rewrite corruption functions to claim exact paper reproduction.

[Decision] Proceed with explicit severity5 in the already prepared Colab
session. Generating8 only to preserve the old default is not supported for
the intended evaluation. The first task remains H5/source preflight followed
by fresh Gaussian5 generation and integrity review.
