# Existing 5-versus-10 step evidence and trajectory-selection interpretation

Date: 2026-10-11. Evidence: [Run], [Code], [Inference], [Open]. Inspected HEAD
`f670c07440533f01e10fd149fad2480d36f9194c`, with uncommitted observer changes.
Read-only source/archive inspection and CPU arithmetic; no new model run.

## Internally matched evidence: Background, SCD disabled

[Run] The accepted October4 sensitivity sweep covers full2468 Background
examples, B32, seeds0/1/2, unguided and smooth-only beta.5/alpha8.140161356429882.
Assets/source/native identities match within this sweep. These are not current
Legacy B70/M800/SCD-on results. Same seeds do not prove common prepared states
or shared random draws.

Recomputed from raw CSV correct counts after verifying24 archived ZIP hashes
and CRCs for steps5/10/15/20. New derived files:
`result/modelnet40_c/diagnostics/trajectory_step_evidence_20261011/analysis.json`,
`background_step_effects.csv`, `unmatched_host_contrasts.csv`.
The JSON lists each immutable input ZIP and SHA. The older30-run validation
remains at
`result/modelnet40_c/gsd_latent_spectral_smooth_v2/background_step_sensitivity_20261004/analysis_20261004_step_sensitivity/validation.md`.

| Arm | Steps5 mean % | Steps10 mean % | 10-minus-5 pp | Seed0/1/2 differences, pp |
|---|---:|---:|---:|---|
| Smooth-only, SCD off |23.851972|24.932469|+1.080497|+1.458671 / +0.607780 / +1.175041|
| Unguided |24.135602|24.675851|+0.540249|+0.607780 / +0.688817 / +0.324149|

Both5-to10 differences are positive in all three seeds. Response is
non-monotonic: smooth-only10-to15 is-0.877904pp and10-to20 is-1.148028pp,
negative in each seed. Unguided10-to15 averages-0.675311pp (mixed seed signs),
and10-to20 is-0.891410pp (all three negative). The already accepted25-step
means rebound to24.959481% smooth-only and25% unguided; do not select10 as an
optimum by omitting the rest of the sweep. No significance or independent
generalization claim is made from these seed patterns.

## Larger, unmatched differences across hosts

[Run/Inference] Compare Phase6 C_SCD normal5 (three-seed mean) to the accepted
restored Legacy normal10 (seed0). This is a descriptive host contrast, not an
isolated step intervention and not a replacement baseline:

| Corruption | Phase6 C_SCD mean % | Restored Legacy seed0 % | Unmatched delta pp |
|---|---:|---:|---:|
| Impulse |69.678552|78.363047|+8.684495|
| LiDAR |31.050783|36.345219|+5.294435|
| Gaussian |74.864938|76.620746|+1.755808|
| Shear |66.315505|62.763371|-3.552134|
| Distortion |65.599676|64.019449|-1.580227|

These use the recent64.570502% Legacy result, not the older CSV-only64.532684%.
The derived table retains all15 rows, not only these illustrative differences.
Inputs are the accepted Phase6 `per_corruption_seed.csv` and accepted Legacy
`analysis_ingested/per_corruption_seed0.csv`. Batch32/70, spectral guidance,
graph, scheduler terminal-alpha, final decode style and random protocol also
differ. Background uses35/30, so its contrast is not5-versus10 at all.

[Open] The searched knowledge records do not supply a completed current-Legacy
all15 control differing only in5 versus10. The October7 proposed SCD
normal10/Background35 experiment was superseded by Legacy restoration; a
proposal is not execution evidence. Do not schedule a new sweep by default.

## What this supports, and what the observer still has to establish

[Code] Both samplers choose the last N entries of a100-step DDIM grid and
initialize noise at its first selected timestep. Steps5 starts at t40;
steps10 starts at t90. Thus changing `denoising_step` changes initial noise
strength and trajectory length together. A5-step run's terminal result is
not the fifth checkpoint of a10-step run.

[Inference] The controlled Background evidence supports sensitivity to the
denoising/noise-depth setting and rejects a simple monotonic-more-is-better
story for those two SCD-off arms. It motivates examining adaptation extent.
The unmatched corruption trade-offs motivate inspecting heterogeneity but
cannot establish that extra steps caused the large Impulse/LiDAR gains or
Shear losses. Neither result proves same-trajectory intermediate superiority,
entropy's ability to identify a good exit, or a speedup.

[Open] The new observer directly addresses the missing distinction: keep the
same initial state/noise, Legacy guidance and10/30 horizon, record intermediate
clean outputs with matching pre-update style plus original terminal, then
measure paired corrected/broken predictions and actual entropy/averaging
policies. A useful oracle with a failing selector means selection remains
unsolved. No useful candidate/policy advantage weakens this trajectory-selection
hypothesis. Only later evaluate a locked prefix-only exit and measured onlineB1
latency including probes. Existing scans and failures must not be repeated.
