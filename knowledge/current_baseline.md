# Current thesis operational baseline — 2026-10-31

[User report/Decision] After accepting the depth transfer results, the user
explicitly promotes this complete policy to the baseline for future thesis
work. It supersedes the earlier decision to leave the operational default
unchanged. Original 3DD-TTA remains the historical reproduction comparator.

| Setting | Locked value |
|---|---|
| Profile / artifact method | `chamfer_pm_n8_v1` / `3dd_chamfer_pm_baseline` |
| Normal / Background reverse steps | 8 / 35 |
| eta / gamma | .003 conditioning style / .03 local latent |
| lambda (`p`) | .95 retained Chamfer fraction; not a spectral coefficient |
| Guidance | Original summed robust Chamfer, no normalization, spectral or semantic term |
| Batch / DDIM total steps | 32 / 100 |
| Decoder / final alpha | Original encoded shape latent / set-alpha-to-one=True |
| LION / classifiers | Raw eval, EMA off / separately frozen, no ensemble |
| Probability Mean | Equal softmax probabilities, temperature1; no logit averaging |
| Normal candidates | `clean_02`, `clean_04`, `clean_06`, `terminal` at step8 |
| Background candidates | `clean_09`, `clean_18`, `clean_27`, `terminal` at step35 |

## Running the baseline

[Code] Use `scripts/eval_chamfer_baseline.py`. It reuses the accepted depth
worker, rejects changed policy parameters and records the baseline profile.
The same entrypoint supports `modelnet40_c` and `scanobjectnn_c`, with
`pointmae`, `dgcnn`, `curvenet`, `pointnext` independently (default: all four).
Point-MAE requires its matching config and checkpoint; omit those flags when
selecting other heads only. There is no default hidden object subset: an
explicit original-ID manifest is required, as in the accepted study.

Example configuration preview from the repository root, after preparing a
source/asset receipt for the **current commit** and matching dataset/head:

```bash
python scripts/eval_chamfer_baseline.py \
  --dataset modelnet40_c \
  --dataset-root /content/data/modelnet40_c \
  --label-path /content/data/modelnet40_c/label.npy \
  --asset-root /content/baseline_assets \
  --asset-manifest-sha256 'VERIFIED_ASSET_MANIFEST_SHA256' \
  --pointmae-config source_snapshot/cfgs/tta_modelnet.yaml \
  --pointmae-checkpoint checkpoints/modelnet_jt.pth \
  --index-manifest result/preparation/chamfer_depth_transfer_20261031/cohort140/modelnet40_c_heldout_ids.json \
  --paired-state-root /content/baseline_states_seed0 \
  --result-root result --seed 0 --preview
```

Replace the SHA placeholder with the verified manifest hash.
Remove `--preview` only for an authorized, prepared Colab execution; this
promotion itself launches no new inference. Scan uses its accepted581-object
asset package and matching15-class heads, label path and
`cohort140/scanobjectnn_c_heldout_ids.json`. Use distinct cache roots per seed
and asset/source identity. All15/severity5 remains the research scope.
The `ids/*_full_ids.json` manifests describe a possible full-data scope; their
existence is not evidence that the full benchmark was run or approved.

[Code] Standard `summary.csv`, `per_corruption.csv` and NPZ `prediction`
now report **probability mean** for this entrypoint. Batch shards retain
all trajectory logits, candidate probabilities and terminal/entropy readouts.
Both batch and corruption prediction NPZs retain `terminal_prediction`;
their `logits` field is terminal logits, not logits of the probability mean.
Do not infer the primary prediction with `logits.argmax`.
Old depth and original/variant entrypoints keep their existing terminal CSV
semantics; archived results and pinned calibration/source records are intact.

## Evidence and boundaries

[Run/Inference] [Accepted transfer results](chamfer_depth_transfer_results_20261031.md)
at integration commit `d96617f0ef1619fc90da9b9c5ad6a527bf90ee7d` cover
140 previously exposed objects per dataset, all15/severity5, seeds0/1/2,
four fixed heads. All eight all15 differences versus historical main C are
positive (+1.000–3.762pp), with conditional95% intervals above zero.
The earlier [256-object Point-MAE confirmation](chamfer_depth_tuning_results_20261030.md)
uses a different cohort; its68.741319% is not the new140-cohort66.000000%.

[Open] Operational promotion is a user decision based on this evidence,
not a new accuracy experiment, global optimum, external replication or full
benchmark result. The whole policy changes multiple factors versus old C/H;
N8's isolated contribution is not identified by that transfer comparison.
Scan PointNeXt PM−terminal interval includes zero and originalseed0 ties51%.
Background35 still assumes corruption identity, so this is not a uniform
unknown-corruption deployment protocol. GSD remains an experimental arm,
not part of the promoted baseline; previous positive/null findings remain.

[Code] Promotion verification: native-free CLI preview and policy guards;
CPU worker fixtures exercise both datasets/four heads, with deliberately
wrong terminal and correct PM predictions; archive fields, CSV aggregation,
source/state guards and unchanged old reporting are checked. These checks
are software evidence, not new GPU accuracy evidence.

[Run/Decision] Published normally to `origin/thesis-integration` at
`f89944285cf4a18878185689538fc51822c32252`, with495CPU tests/412subtests
and78separate pinned NumPy1 checks passing. After the user's explicit
follow-up instruction, owned `tta-depth-transfer-a100-20261031-r1` was
closed and the server inventory verified empty at12:04UTC. Both full archive
SHA identities were rechecked locally and on Drive before closure.
[Closure receipt](../result/preparation/chamfer_depth_transfer_20261031/runtime_closure_20261031/runtime_closure.json).


## 2026-11-08 — matched High-RAM speed-study batch override

[Run/User decision] The [full speed study](lion_guidance_macro14_results_20261108.md)
uses common B128/final36 after a measured B32/B64/B128 throughput-and-memory
selection on A10080 and verified transfer to the continuation runtimes. This
is the comparison baseline for this Point-MAE/ModelNet40-C study. The separate
multi-head/multi-dataset entrypoint above retains its B32 locked profile;
B128 capacity and numerical behavior have not been established for every
other head or dataset. No global policy/default or existing artifact is rewritten.


## 2026-11-08 — current comparison record with corruption-independent horizons

[User decision/Run] For the completed Point-MAE/ModelNet40-C speed comparison,
use the [common-horizon all15 record](lion_guidance_uniform_horizon_results_20261108.md):
N5 terminal at gamma/eta=.01/.01; tuned N8 terminal and N8+PM at .03/.003;
p=.95; cheap N8+PM with tau=.1; all15 use the same horizon within a method,
PM[2,4,6,terminal], B128/final36, seed0, all2468objects/corruption. TerminalN8
accuracy is the last saved N8+PM readout; its separate latency is unmeasured.

[Run] N5terminal61.445165%, tunedN8terminal62.571583%, N8PM63.735818%,
cheap63.722312%; cheap saves27.673026% time with−.013506pp accuracy difference,
conditional95%CI[−.180983,.153971]. The report contains all15 individual
accuracies/counts, paired diagnostics and timings. Original8/35 and B32
multi-head entrypoint settings above are historical executable profiles,
not the new corruption-independent comparison. Their code defaults are not
silently changed by this documentation update. Background short-horizon
accuracy is substantially belowN35; retain both policies/results.

[Open] One seed/development-exposed policy revision, native/runtime effects,
no fresh terminal-onlyN8 latency, repeated-seed/unseen-generalization or
isolated unguided-attribution claim. No automatic cheap/default promotion.
