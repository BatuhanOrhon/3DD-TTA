# Full ModelNet40-C Test-Set GSD Candidate Screen

**Status:** proposed design for user review
**Date:** 2026-09-28
**Branch:** `gsd-smooth-spectrum`

## Purpose

Run the already selected GSD-inspired smooth guidance candidates over every
example in all 15 ModelNet40-C severity-5 corruptions. This replaces the
proposed 100-example-per-corruption extension. The run is a full-test-set
development scan to compare the candidates and assess their behavior across
the corruption suite.

This test set has already supplied labels for the 128-example Gaussian/Impulse
development screens. Selecting a candidate from the full-set scores therefore
does not create an untouched benchmark confirmation. Preserve this distinction
in the run notes, summary, and thesis-facing knowledge record. A later
independent confirmation requires a separately defined untouched evaluation
protocol.

## Experiment matrix

Evaluate three arms at seeds 0, 1, and 2, for nine full-suite run bundles:

| Arm | Profile and target contribution | Fixed spectral coefficient |
|---|---|---:|
| Matched SCD-only | hard placeholder, weight 0, no target rho | 0 |
| Candidate A | smooth beta .5, rho .001 | 8.140161356429882 |
| Candidate B | smooth beta 2, rho .01 | 113.10823980075075 |

The coefficients come from calibration reference
`20260928-113047_gsd-cal-diagnose-reference-seed0-n64`, whose config SHA-256
is `550d73dc83375395c905db2e6cda3845dc3362632117d3371f8d4ad4e787e2d6`.
Every seed and arm reuses these exact coefficients. Do not recalibrate by seed
or fit values using the full-test accuracy.

Each run evaluates the complete input file for every entry in
`research_artifacts.CORRUPTIONS`, in canonical order. Expected coverage is all
15 corruption names and each corruption's full recorded dataset length. The
runner must fail on empty, truncated, or data/label-mismatched inputs. Preserve
the existing protocol: ModelNet40-C severity 5, batch32, frozen Point-MAE,
frozen raw/eval LION, EMA disabled, lambda .95, gamma=eta=.01, current graph,
preprocessing, DDIM scheduler and decoder. Background retains its existing
35-step reverse schedule; other corruptions retain the existing five-step
schedule.

This is 15 x 2,468 = 37,020 example-corruption evaluations per arm/seed when
the local archive has 2,468 examples in every corruption, or 333,180 total
across all nine runs. Record actual counts from each input rather than assuming
this expected count without checking the loaded files.

The runs remain separately seed-controlled. The existing implementation does
not guarantee common diffusion/interpolation draws across arms, so report
same-seed accuracy deltas as seed-matched comparisons and do not claim
per-example common-random-number pairing.

## Protocol and implementation shape

Add a distinct GSD smooth-v2 stage named `full_dataset_development`. Keep the
existing `calibrate`, Gaussian/Impulse `development`, `pilot`, and benchmark
contracts intact. The new stage:

- requires the smooth-v2 method, all 15 canonical corruption names, severity5,
  max-batches0, batch32, and seed0/1/2;
- requires a calibration reference, verifies its run ID, file hash, source,
  checkpoint, data and graph/host settings, and checks each guided arm's
  declared rho and alpha against that reference;
- processes each complete corruption file without selecting/shuffling a
  prefix; does not require a subset count or calibration-index overlap;
- records `stage=full_dataset_development`, full-file counts, canonical order,
  calibration provenance, complete dataset scope, and the explicit
  development/test-set selection caveat in every run config and notes;
- reports `execution_status=complete` and CSV coverage `complete` only after
  all 15 corruption files have been fully processed;
- accumulates the existing bounded per-corruption scalar diagnostics, without
  storing sample-by-step diagnostic rows for all 37,020 examples;
- leaves graph construction, spectral loss, coefficient math, SCD behavior,
  random-number use, and environment dependency pins unchanged.

Add `scripts/run_gsd_full_dataset_screen.py` as the Colab entry point. It
preflights the one raw calibration config and derives/verifies the fixed arm
coefficients before any model work. It launches the nine arm/seed commands
sequentially in the existing `3dd_tta_env`, prints each exact command and
archive path, stops at the first failed arm, and never installs packages or
performs a Git pull. Give every logical arm/seed a unique immutable archive
identity. A `--resume` option may skip only complete seven-file archives that
pass protocol and provenance checks; it must never overwrite an existing raw
archive. If a matching incomplete/failed attempt exists, retain it and create
a distinct attempt for the rerun.

Add `scripts/analyze_gsd_full_dataset_screen.py`. It accepts an explicit list
of exactly nine run ZIPs, validates rather than selecting archives by glob,
and writes a compact derived report beside (not inside) the raw runs. It must
check safe archive members, CRC, seven-file schema, run/CSV/log agreement,
complete 15-corruption coverage, each recorded full data count, all arm/seed
combinations exactly once, the pinned calibration ID/hash/alpha, common
source/data/checkpoint manifests, and matching recorded package/runtime and
compiled-extension identities. It must reject missing, duplicate, mixed or
incomplete conditions.

## Analysis output and interpretation

Report per arm and seed:

- correct/total and accuracy for every corruption;
- unweighted 15-corruption macro accuracy (with the aggregation formula);
- total examples/correct, total runtime and peak allocated GPU memory;
- each guided arm's absolute percentage-point delta from the SCD-only arm at
  the same seed, overall and per corruption;
- mean and sample standard deviation of the three seed-level macro values and
  seed-matched macro deltas, plus the sign of every seed delta.

Do not pool all examples across corruptions into the macro score. Do not
present the best arm's full-test score as an unbiased final performance claim
after inspecting it for selection. Do not call the three seeds independent
objects or common-draw paired. The purpose is to select or reject candidates
for further work; an independent evaluation remains outstanding.

## Validation and acceptance

Before implementation is considered complete, CPU-only tests must cover:

1. stage acceptance of all 15 corruptions and rejection of partial/incorrect
   lists, prefix sampling, unsupported seeds, and incompatible host settings;
2. full-file iteration, recorded input counts, complete/failed status, and
   absence of per-sample trajectory dumps in the run artifact;
3. calibration ID/hash/manifest and fixed-alpha checks for all three arms;
4. launcher command matrix of nine runs, stop-on-failure behavior, and resume
   skipping only valid complete matching archives;
5. analyzer acceptance of a valid synthetic 15-corruption x three-seed
   fixture and rejection of wrong counts, duplicate/missing arms, seed/config
   drift, manifest or environment mismatch, and malformed archives;
6. exact macro, per-corruption and seed-delta calculations.

Colab is the only place to execute LION/GPU runs. Local verification is limited
to syntax/static checks and CPU fixtures. Before a result is interpreted, all
nine seven-file bundles and the compact summary must be ingested and checked.
Interrupted or failed runs remain archived; resume must preserve and reuse
valid completed arms rather than relaunching the full matrix.

## Scope exclusions

Do not change the spectral equation, choose new beta/rho values, tune on a new
grid, regenerate calibration, modify the classifier/LION/checkpoints/dataset,
change requirements or installed packages, implement dynamic graphs or PxP,
claim a full GSDTTA reproduction, or call this full test-set scan independent
confirmation. Do not modify or combine the immutable 113047 interaction ZIPs.

## User decisions captured

- “15 corruption'ın tamamı” means all 15 corruption files in the ModelNet40-C
  **test set**, not an inferred noise-only subset.
- The user accepts evaluating the entire dataset for this important candidate
  comparison and is willing to spend the additional runtime.
- The candidate arms are the currently retained beta .5/rho .001 and beta
  2/rho .01 conditions with a matched SCD-only control, across seeds0/1/2.
