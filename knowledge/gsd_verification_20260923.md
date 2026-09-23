# GSD v1 implementation verification

[Code] Date: 2026-09-23. Branch: `gsd-development`. Base:
`79cc02774e5fa85a7c2f84a08506617670416642`. Implementation revision is the
commit containing this record; runtime bundles record its full hash and
individual source SHA-256 values. No GPU experiment or new raw result ZIP.

## Checks performed

[Code] `python -B -m unittest discover -s tests` passed **84 tests** in
2.649 seconds (exit 0). The suite includes existing baseline controls and
new graph, trajectory, launcher, protocol and worker tests. Graph tests use
small real CPU tensors; trajectory/worker tests replace DDIM, native CUDA
operators and pretrained models with test doubles.

[Code] Coverage includes analytic/finite-difference spectral gradients,
batch scaling, sign and within-band rotation invariance, nonself/duplicate
neighbors, directed thresholds, union symmetry, isolated/empty graphs,
malformed/nonfinite inputs, dtype-aware repeated-eigenvalue bands, RNG
preservation, zero-weight original tensor equality, unequal gamma/eta mapping,
conditioning gradients, original-style decode, frozen weights, no_grad callers,
protocol rejection, old-method configuration parity, actual worker dispatch
for 5/35 steps, seven-file artifacts and preserved counters on failure.

[Code] Python AST parsing passed for the five runtime modules. Both
`scripts/colab_gsd_pilot.sh` and `scripts/colab_gsd_all15.sh` passed `bash -n`
using Git Bash. Initial sandbox execution of Bash hit Windows signal-pipe
permission errors; the same syntax-only checks passed with authorized
escalation. No shell script was executed as an experiment.

[Code] `python -B eval_gsd_tta.py --stage pilot` prints nine commands;
all-15 dry-run command generation also passed. Launcher tests validate every
generated smoke/pilot/all-15 command through the real CLI parser. The
`--execute` GPU path remains a Colab check.

[Code] `git diff --exit-code 79cc027 -- tta.py main_3dd_tta.py models
utilities_3dd_tta.py datasets_mate research_artifacts.py
tests/test_preprocessing_identity.py tests/test_shared_trajectory_decoder_control.py`
passed. Existing baseline method arguments/configuration/notes are also
compared with the pinned starting runner in a CPU test. `git diff --check`
passed; Git emitted only its existing LF-to-CRLF normalization notices.

[Code] Independent review found and reproduced one float32 eigenspace
truncation defect. Four regression tests and a dtype/size/Laplacian-scale
roundoff floor resolved it; follow-up review reported no new findings.
Zero-mode counts are numerical diagnostics, not exact connected-component
counts; nearby unresolved eigenvalues may be retained.

[Code] Protected refs remained `baseline-repro-clean=79cc027` and
`main=107305f`. Only named GSD code/tests/docs are selected for the local
commit. Existing untracked source PDFs, raw results/ZIPs, caches and
historical notes are retained. The branch has not been pushed.

## Files in this change

[Code] Runtime and launchers:

- `graph_spectral.py`
- `tta_gsd.py`
- `gsd_protocol.py`
- `eval_gsd_tta.py`
- `run_baseline.py` (additive opt-in dispatch)
- `scripts/colab_gsd_pilot.sh`
- `scripts/colab_gsd_all15.sh`

[Code] Tests:

- `tests/test_graph_spectral.py`
- `tests/test_gsd_trajectory.py`
- `tests/test_gsd_protocol.py`
- `tests/test_gsd_worker.py`
- `tests/test_gsd_launcher.py`

[Code] Research/artifact records:

- `knowledge/README.md`
- `knowledge/thesis_scope.md`
- `knowledge/reproduction_gap.md`
- `knowledge/experiment_protocol.md`
- `knowledge/repository_map.md`
- `knowledge/method_synthesis.md`
- `knowledge/findings_log.md`
- `knowledge/open_questions.md`
- `knowledge/papers/3dd_tta.md`
- `knowledge/papers/gsdtta.md`
- `knowledge/papers/lion.md`
- `knowledge/gsd_design.md`
- `knowledge/gsd_integration_20260922.md`
- `knowledge/gsd_verification_20260923.md`
- `knowledge/colab_gsd.md`
- `result/README.md`

## Remaining evidence

[Open] Run the [Colab smoke/pilot](colab_gsd.md), inspect original/off/on ZIPs,
then apply the declared all-15 promotion rule. Full CUDA correctness,
N=2048 time/memory, bandwidth/weight suitability, per-corruption effects and
three-seed accuracy benefit are unverified. Baseline reproduction/checkpoint
provenance questions remain separate. No GSDTTA reproduction claim is made.
