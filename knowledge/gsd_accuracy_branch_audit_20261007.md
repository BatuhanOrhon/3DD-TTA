# Accuracy priority: existing branch evidence and next decision - 2026-10-07

## Objective and decision

[User decision] Seek +1 to +2 absolute percentage points in equal-weight
all15 accuracy. This prospective objective does not revise Phase6 gates.

[Inference] Park further style/projection variants. The next bounded candidate
is stronger SCD-guided adaptation in the current controlled pipeline:
normal reverse steps 5 -> 10, retaining Background35 and all other settings.
This is a proposed performance experiment, not an implemented change or a
prediction of a +1pp gain. No GPU run was launched.

## Existing numerical evidence

[Run] Phase6 paired C_SCD mean is63.974428%, P_PC64.011345%:
delta+0.036917pp. The complete audit remains authoritative:
[Phase6 report](../result/modelnet40_c/gsd_guidance_composition_v1/all15/analysis_20261006_independent/validation_20261007.md).

[Run: CSV-only] Recomputed all15 means from the three standalone CSVs:

| Existing result directory | Macro15 accuracy |
|---|---:|
| gsd_tta_improvements | 64.386817% |
| gsd_smooth_optimal (M800) | 64.532684% |
| gsd_smooth_optimal_1800 | 64.505672% |

Each has exactly one row for each of the15 expected corruptions. CSVs lack
seed, prediction records, run-source manifest, asset/build identities and
matched SCD controls. Their numerical order does not establish an optimum.
Historical M_max1800 notes and raw files are preserved.

[Run/Inference] The following M800-versus-Phase6 differences are **unmatched
descriptive contrasts**, comparing a CSV record to a three-seed mean:

| Corruption | Phase6 C_SCD | Legacy M800 | Difference, pp | Contribution to macro15, pp |
|---|---:|---:|---:|---:|
| Impulse | 69.6786 | 79.0113 | +9.3328 | +0.6222 |
| LiDAR | 31.0508 | 36.3047 | +5.2539 | +0.3503 |
| Gaussian | 74.8649 | 76.5802 | +1.7153 | +0.1144 |
| Shear | 66.3155 | 62.3582 | -3.9573 | -0.2638 |
| Distortion | 65.5997 | 63.3306 | -2.2690 | -0.1513 |
| Background | 60.9941 | 60.0891 | -0.9049 | -0.0603 |
| All15 | 63.9744 | 64.5327 | +0.5583 | +0.5583 |

Impulse/LiDAR account for+0.9724pp of the descriptive macro difference;
losses elsewhere offset much of that. This motivates investigating the
adaptation trade-off, not selecting a winning method by corruption label.
It neither proves those gains are reproducible nor attributes them to steps.

## Source and missing provenance

[Code] Main HEAD:c18926ff7751ed9023313869229f5d4dffe218be.
Read-only comparison worktree: `D:/Akademik/Okul/Thesis/code/3DD-TTA-gsd-improvements`,
branch gsd-tta-smooth-integration, HEAD:5c3ed88314f1c3d50ab77d17f7c6ba22da59c441.
Seven inspected tracked files match that HEAD after CRLF normalization.
This establishes the inspected source identity, **not the CSV execution identity**.

| Setting | Phase6/current control | Inspected legacy smooth launcher/source |
|---|---|---|
| Batch | 32 | 70 |
| Normal / Background steps | 5 / 35 | 10 / 30 |
| DDIM final-alpha setting | recorded True | explicit False |
| Final decode style | original | updated by default |
| Spectral composition | C_SCD none; P_PC local SCD/style projection | summed SCD+spectral on both latent blocks |
| Spectral operator | Phase6 hard M100 with boundary expansion | exp(-2*raw eigenvalue), truncate M800 |
| Graph filtering | directed degree cutoff .06*mean at defaults | symmetric degree cutoff .6*mean |
| Isolates/boundary | active eigensystem, boundary expansion | diagonal1000, jitter1e-5, fixed slice |
| Model mode | raw/eval | inspected LION wrapper sets VAE/prior eval |
| Randomness/provenance | seeded paired artifacts | eval script has no seed argument/seeding |

[Code] Legacy smooth loss sums weighted squared spectral residuals divided
by3*2048; SCD is summed. This differs from the older multiband mean reduction
and from current rank-normalized hard guidance. Equal numerical weights or
beta values across these operators do not imply equal guidance.
Legacy PointDataset explicitly reads severity5 files. Data/checkpoint equality
with Phase6 cannot be established from the CSV.

[Open] The launcher writes `gsd_smooth_optimal_800`, whereas the supplied CSV
is in `gsd_smooth_optimal`. There is no executed command/source manifest to
resolve this naming difference. Do not silently assign HEAD5c3ed88 to the run.

[Code/Open] Untracked `run_scd_only_improvements.sh` exists in that worktree;
it calls **eval_gsd_tta.py**, not eval_gsd_tta_v2.py. Its existence is not run
evidence, and its host cannot automatically control smooth-v2. No corresponding
scd_only_improvements/gsd_only_improvements result directories were found in
the main result root; the other worktree has no result directory. The October3
metadata inventory contains144 archives and zero normal10 records. This is a
bounded search, not proof that no unprovided run exists. A broad rg traversal
also encountered an inaccessible old temporary workcopy; no access changes
or deletion were attempted.

## Next action: one candidate, all15 coverage

[Inference/Proposal] Hypothesis: current normal5 adaptation may leave useful
SCD-guided correction unrealized; normal10 may improve macro15, with possible
losses on deformation corruptions. Evidence for this mechanism is currently
absent; the legacy pattern only makes it worth a bounded comparison.

1. Freeze candidate **SCD, normal10/Background35**, against current C_SCD
   normal5/Background35. Keep lambda.95, gamma/eta.01, original final style,
   DDIM final-alpha, batch32, assets, preprocessing and all other settings fixed.
   No spectral/projection change or M search accompanies this candidate.
2. First use an all15 development subset, proposed256 indices per corruption
   and seeds0/1/2. Freeze indices before execution; do not screen only the four
   previously favorable corruptions. This benchmark is already development
   data; an unused subset label must not be represented as independent evidence.
3. Reuse existing baseline counts only if exact indices, prepared state,
   underlying random draws, runtime and sources can be shown equivalent.
   Otherwise a fresh paired baseline is necessary to estimate this **new**
   intervention; do not describe it as a repeat of the style experiment.
   Initial encoded latent and epsilon draws can be shared; the noised initial
   state intentionally differs because the start time changes.
4. Report macro15/14, Background, all corruption/seed deltas, corrected/broken
   counts and runtime. Decide promotion criteria before execution. No numerical
   pilot gate or implementation was registered in this read-only audit.
   A null/negative candidate should not trigger an automatic large step sweep.
5. If supported on development data, lock the one candidate and compare full
   2468 indices x15 x3 with matched controls. This remains full-test development
   evaluation. An independent evaluation claim needs separately reserved data.

[Code/Inference] Steps5 ->10 also changes the starting diffusion time from
t40 to t90 under the current100-step schedule; it changes noise strength and
the number of guided updates together. It is not a pure runtime-only test.
With Background35 held, total per-corruption reverse-step budget becomes
175 versus105 (+66.7%); wall time must be measured, not assumed proportional.

[Run] The completed Background5/10/15/20/25 sensitivity scan was for unguided
and smooth-only arms (SCD off). It does not answer this normal-corruption,
SCD-on question. Completed decoder/style/projection and smooth beta/rho
comparisons are not scheduled again. Failed Background100/1000 remain failed.

[Inference] Even a successful fixed10 setting alone is an operating-point
improvement, not sufficient evidence of a new mechanism. If correction versus
damage changes by sample, a later label-free adaptation-strength rule could
become a method question. Such a rule is not justified, designed or selected
by these CSVs. The +1-2pp goal is not guaranteed by this plan.

## Reproducible local work

CPU/standard-library only:
`python result/modelnet40_c/diagnostics/accuracy_priority_20261007/analyze_existing.py`.
The script recomputes the15-row means, descriptive contrasts, source hashes
against git blobs, and the older inventory's normal10 records. Outputs:
[JSON](../result/modelnet40_c/diagnostics/accuracy_priority_20261007/existing_evidence.json),
[all15 CSV](../result/modelnet40_c/diagnostics/accuracy_priority_20261007/descriptive_comparison.csv).
Other commands: `git status --short`, `git -C <legacy> status --short`,
`git show HEAD:<source>`, `rg --files`, `rg -n`, `Get-Content`, `Get-ChildItem`.
No model imports, GPU execution, environment installs, tests, raw-archive
mutation or stage/commit/push. Only this report, derived diagnostics and dated
knowledge additions were written.
