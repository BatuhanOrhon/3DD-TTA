# Missing guidance ablations: Colab handoff, 2026-10-03

[User decision/Code] Implement only the two requested missing conditions.
Reuse completed SCD-only and SCD+smooth controls; do not repeat their runs.
This supersedes the implementation-readiness paragraph and 12-run proposal
in `gsd_next_experiment_proposal_20261003.md`; see the coverage audit first.

Base Git: `gsd-smooth-spectrum`, `4be6afd86629f3d52648fb543fc2eb350a5f9f1d`.
Changes are local, not committed or pushed. No Colab/GPU execution was started.

| Launcher | Guidance | New full runs |
|---|---|---:|
| `scripts/run_gsd_unguided.py` | SCD=0, spectral=0; encode, noise, DDIM, decode retained | 3 |
| `scripts/run_gsd_smooth_only.py` | SCD=0, smooth beta=.5, alpha=8.140161356429882 | 3 |

Both use all 15 ModelNet40-C test corruptions, severity5, all2468 examples per
corruption, seeds0/1/2, batch32, raw/eval LION, noEMA, gamma=eta=.01,
lambda=.95, original final decoder style and the existing 100-step DDIM
schedule with5 reverse steps (35 Background). No subset or new coefficient
search. Reference `20260928-113047_gsd-cal-diagnose-reference-seed0-n64`, raw
config SHA256 `550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6`.
Rho=.001 describes the archived SCD reference; with SCD disabled it is not an
active gradient ratio. Fixed alpha may be weak for smooth-only; the experiment
tests this existing setting, not all possible smoothing strengths.

## Colab use

Open `scripts/gsd_guidance_ablations_colab.ipynb` in Colab with the existing
environment/data/checkpoints. Its first cell uploads
`result/modelnet40_c/diagnostics/gsd_guidance_ablation_colab_20261003.zip`.
It checks base commit, existing file hashes, package hashes and safe paths
before writing the changed code and byte-identical calibration config.
It is repeatable and refuses to overwrite divergent user changes. No git
pull, package reinstall or runtime restart is required. If an experiment
is currently running, wait until it finishes before installing code.

Execute the next two cells sequentially in the same runtime, or run one
experiment per separately prepared Colab runtime. Each launcher always skips
validated complete seed archives (including successful attempt suffixes).
A failed/incomplete seed restarts under a new attempt name; partial batches
are not resumed. Source-mismatched completions stop rather than trigger a
silent duplicate. Do not execute the same launcher concurrently in one root.

## Outputs and interpretation

Method directory: `result/modelnet40_c/gsd_latent_spectral_smooth_v2/`.
Run tags: `gsd-ablation-unguided-<reference>-seedN` and
`gsd-ablation-smooth_only-<reference>-seedN`.
Collect 6 completed seven-file ZIPs and both
`guidance_ablation_summary_<arm>_<reference>.json` summaries. Failed attempts
remain separate evidence. `config.json` now includes ordered sample indices,
labels and predictions for every corruption; accuracy is checked against CSV.
Labels are recorded only after predictions and do not enter adaptation.

[Inference/Open] New scores can be compared descriptively with archived
controls. Shared seed numbers do not establish common-draw pairing across
historical runs. V1 versus smooth-only differs in operator normalization and
coefficient as well as smoothing, so it cannot isolate a pure smoothing effect.
No accuracy outcome is claimed before the Colab artifacts return.

## Validation

[Code/Verification] 158 CPU unit tests passed; both launcher dry-runs generated
exactly three expected commands using the real pinned calibration. All six
commands passed CLI/config construction. All nine historical full-screen ZIPs
still pass the strict reader. The unguided trajectory CPU test checks the
reverse equations and RNG consumption independently and fails if Chamfer,
graph construction or guidance gradients are invoked. No CUDA/LION accuracy
run occurred locally. Independent review identified and resolved an existing
calibrated-stage source-compatibility regression: new configs explicitly
declare the `tta_gsd.py` branch extension, old archived declarations retain
their stricter checks. Exact current source hashes remain logged.

[Verification] The notebook installer was exercised against a temporary
base-file copy: fresh installation, repeated installation and refusal to
overwrite a divergent file passed. Every shipped Python file matches the
current source after LF normalization; ZIP member hashes and CRC passed.

Falsifiers: nonzero unguided guidance, changed reference coefficient, missing
samples/corruptions or manifest/CSV/prediction disagreement invalidate a run.
Once raw results arrive, validate before assessing how much denoising alone
and smooth-only account for the archived SCD/combined scores.
