# Terminal acceleration and same-logit LM–PM: bounded pilot results

[Run/CPU/Inference] Completed on 2026-11-10. Fixed 32 original objects × all15 ModelNet40-C corruptions, severity5, Point-MAE, seed0, B32. Three fresh operational repeats, repeat0 primary; repeats1/2 remain separate. This is exposed development data, one inference seed and one VM. No full-data expansion or default change.

Pilot runner elapsed 1185.561 seconds (19.759 minutes), including its loading/warmup/checkpoint work; registered softcap2700s. This is distinct from the measured arm inference times below.

## Interpretation within this pilot

[Inference] Proximal terminal N5 and N8 reduce measured total inference time
by32.1167% and35.3848%, respectively, but primary point accuracies decrease
by1.0417pp and1.8750pp. Paired confidence intervals include zero and extend
well below−.3pp: neither terminal candidate satisfies the planned accuracy
retention criterion. This is descriptive speed evidence with an uncertain
accuracy tradeoff, not a validated baseline replacement.

LM has345/480 versus PM344/480 (+.2083pp,95%CI[0,.6250]pp), only one corrected
pair and no harmed pair in repeat0. Repeat1 also gains one; repeat2 gains two.
Every interval touches zero. A reliable accuracy benefit is unestablished;
keep PM active. Separate P8-PM beats P8 by19 pairs (+3.9583pp) while taking
42.0284% more time; failed native repeats prevent attributing every difference
solely to the extra readouts. No full dataset or additional seed is scheduled.

## Accuracy, operational time and memory

Each accuracy denominator is480 object-corruption pairs, clustered into32 original objects. Macro14 excludes Background (448pairs). Timings include fresh preprocessing/encoding/noise, guidance and terminal or four-candidate readout; CUDA synchronized. Loading, two warmups per arm/corruption, RNG seeding, audit hashes, D2H and artifact I/O are excluded equally. B32 amortized ms/input is not B1 latency. Memory is CUDA peak allocated/reserved MiB, max over15 corruptions.

| Repeat | Arm | Correct/480 | All15 % | Macro14 % | Time s | ms/input | Allocated MiB | Reserved MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | T5 | 329/480 | 68.5417 | 70.9821 | 41.9593 | 87.4152 | 14599.22 | 15068.00 |
| 0 | P5 | 324/480 | 67.5000 | 69.4196 | 28.4833 | 59.3403 | 11601.46 | 15068.00 |
| 0 | T8 | 334/480 | 69.5833 | 70.7589 | 60.8550 | 126.7812 | 14599.22 | 15068.00 |
| 0 | P8 | 325/480 | 67.7083 | 68.9732 | 39.3216 | 81.9200 | 11601.46 | 15068.00 |
| 0 | P8-PM | 344/480 | 71.6667 | 72.9911 | 55.8478 | 116.3496 | 11605.47 | 15068.00 |
| 0 | P8-LM | 345/480 | 71.8750 | 73.2143 | same trace, offline | — | — | — |
| 1 | T5 | 330/480 | 68.7500 | 70.9821 | 41.9537 | 87.4035 | 14599.22 | 15068.00 |
| 1 | P5 | 324/480 | 67.5000 | 69.4196 | 28.4787 | 59.3306 | 11601.46 | 15068.00 |
| 1 | T8 | 335/480 | 69.7917 | 70.9821 | 60.8669 | 126.8061 | 14599.22 | 15068.00 |
| 1 | P8 | 325/480 | 67.7083 | 68.9732 | 39.4496 | 82.1867 | 11601.46 | 15068.00 |
| 1 | P8-PM | 344/480 | 71.6667 | 72.9911 | 55.8373 | 116.3278 | 11605.47 | 15068.00 |
| 1 | P8-LM | 345/480 | 71.8750 | 73.2143 | same trace, offline | — | — | — |
| 2 | T5 | 330/480 | 68.7500 | 70.9821 | 41.9379 | 87.3706 | 14599.22 | 15068.00 |
| 2 | P5 | 324/480 | 67.5000 | 69.4196 | 28.4696 | 59.3116 | 11601.46 | 15068.00 |
| 2 | T8 | 335/480 | 69.7917 | 70.9821 | 60.8567 | 126.7848 | 14599.22 | 15068.00 |
| 2 | P8 | 325/480 | 67.7083 | 68.9732 | 39.3213 | 81.9194 | 11601.46 | 15068.00 |
| 2 | P8-PM | 344/480 | 71.6667 | 72.9911 | 55.8387 | 116.3306 | 11605.47 | 15068.00 |
| 2 | P8-LM | 346/480 | 72.0833 | 73.4375 | same trace, offline | — | — | — |

T5/P5: matched classic/proximal terminal N5. T8/P8: matched terminal N8. T5 gamma=eta=.01; T8 gamma=.03/eta=.003. P5/P8 use tau=.1 and retained=.95, no backward/direct style update or intermediate PM probes. P8-PM reads clean2/4/6+terminal8; P8-LM reduces those exact same four saved logits, equal weight, temperature1, CPU float64. Its shared acquisition time is P8-PM time; no standalone GPU LM latency was measured.

## Paired differences and speed

Candidate minus reference, percentage points. Conditional paired object-cluster bootstrap,10000 draws/seed20261108,32 objects; no pooling operational repeats. All15 and macro14 below. Corrected/harmed/net count object-corruption outcomes. Unadjusted intervals are descriptive; a degenerate interval is not equivalence.

| Repeat | Reference → candidate | Group | Δ pp | 95% CI pp | Corrected/harmed/net | Speedup | Time saved % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | T5 → P5 | all15 | -1.0417 | [-3.5417, +1.2500] | 10/15/-5 | 1.4731x | 32.1167 |
| 0 | T5 → P5 | macro14 | -1.5625 | [-4.2411, +0.6696] | 8/15/-7 | 1.4732x | 32.1204 |
| 0 | T8 → P8 | all15 | -1.8750 | [-4.3750, +0.6250] | 11/20/-9 | 1.5476x | 35.3848 |
| 0 | T8 → P8 | macro14 | -1.7857 | [-4.4643, +0.8929] | 11/19/-8 | 1.5476x | 35.3856 |
| 0 | P8-PM → P8-LM | all15 | +0.2083 | [+0.0000, +0.6250] | 1/0/1 | same logits | — |
| 0 | P8-PM → P8-LM | macro14 | +0.2232 | [+0.0000, +0.6696] | 1/0/1 | same logits | — |
| 0 | P8 → P8-PM | all15 | +3.9583 | [+2.0833, +6.0417] | 23/4/19 | 0.7041x | -42.0284 |
| 0 | P8 → P8-PM | macro14 | +4.0179 | [+2.0089, +6.0268] | 22/4/18 | 0.7039x | -42.0636 |
| 1 | T5 → P5 | all15 | -1.2500 | [-3.7500, +0.8333] | 9/15/-6 | 1.4732x | 32.1187 |
| 1 | T5 → P5 | macro14 | -1.5625 | [-4.2411, +0.6696] | 8/15/-7 | 1.4735x | 32.1324 |
| 1 | T8 → P8 | all15 | -2.0833 | [-4.3750, +0.4167] | 10/20/-10 | 1.5429x | 35.1871 |
| 1 | T8 → P8 | macro14 | -2.0089 | [-4.4643, +0.6696] | 10/19/-9 | 1.5426x | 35.1761 |
| 1 | P8-PM → P8-LM | all15 | +0.2083 | [+0.0000, +0.6250] | 1/0/1 | same logits | — |
| 1 | P8-PM → P8-LM | macro14 | +0.2232 | [+0.0000, +0.6696] | 1/0/1 | same logits | — |
| 1 | P8 → P8-PM | all15 | +3.9583 | [+2.0833, +6.0417] | 23/4/19 | 0.7065x | -41.5409 |
| 1 | P8 → P8-PM | macro14 | +4.0179 | [+2.0089, +6.0268] | 22/4/18 | 0.7065x | -41.5471 |
| 2 | T5 → P5 | all15 | -1.2500 | [-3.7500, +0.8333] | 9/15/-6 | 1.4731x | 32.1150 |
| 2 | T5 → P5 | macro14 | -1.5625 | [-4.2411, +0.6696] | 8/15/-7 | 1.4732x | 32.1190 |
| 2 | T8 → P8 | all15 | -2.0833 | [-4.5833, +0.4167] | 10/20/-10 | 1.5477x | 35.3870 |
| 2 | T8 → P8 | macro14 | -2.0089 | [-4.6875, +0.6696] | 10/19/-9 | 1.5475x | 35.3809 |
| 2 | P8-PM → P8-LM | all15 | +0.4167 | [+0.0000, +1.0417] | 2/0/2 | same logits | — |
| 2 | P8-PM → P8-LM | macro14 | +0.4464 | [+0.0000, +1.1161] | 2/0/2 | same logits | — |
| 2 | P8 → P8-PM | all15 | +3.9583 | [+2.0833, +6.0417] | 23/4/19 | 0.7042x | -42.0061 |
| 2 | P8 → P8-PM | macro14 | +4.0179 | [+2.0089, +6.0268] | 22/4/18 | 0.7040x | -42.0385 |

## Every corruption: primary repeat0

| Corruption | T5 | P5 | T8 | P8 | P8-PM | P8-LM |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 26/32 (81.250%) | 26/32 (81.250%) | 28/32 (87.500%) | 27/32 (84.375%) | 28/32 (87.500%) | 28/32 (87.500%) |
| gaussian | 24/32 (75.000%) | 24/32 (75.000%) | 25/32 (78.125%) | 27/32 (84.375%) | 28/32 (87.500%) | 28/32 (87.500%) |
| background | 11/32 (34.375%) | 13/32 (40.625%) | 17/32 (53.125%) | 16/32 (50.000%) | 17/32 (53.125%) | 17/32 (53.125%) |
| impulse | 30/32 (93.750%) | 29/32 (90.625%) | 31/32 (96.875%) | 30/32 (93.750%) | 31/32 (96.875%) | 31/32 (96.875%) |
| upsampling | 27/32 (84.375%) | 26/32 (81.250%) | 26/32 (81.250%) | 28/32 (87.500%) | 29/32 (90.625%) | 29/32 (90.625%) |
| distortion_rbf | 23/32 (71.875%) | 22/32 (68.750%) | 23/32 (71.875%) | 23/32 (71.875%) | 24/32 (75.000%) | 24/32 (75.000%) |
| distortion_rbf_inv | 22/32 (68.750%) | 22/32 (68.750%) | 21/32 (65.625%) | 20/32 (62.500%) | 22/32 (68.750%) | 22/32 (68.750%) |
| density | 28/32 (87.500%) | 27/32 (84.375%) | 28/32 (87.500%) | 25/32 (78.125%) | 27/32 (84.375%) | 27/32 (84.375%) |
| density_inc | 30/32 (93.750%) | 31/32 (96.875%) | 31/32 (96.875%) | 30/32 (93.750%) | 31/32 (96.875%) | 31/32 (96.875%) |
| shear | 25/32 (78.125%) | 22/32 (68.750%) | 24/32 (75.000%) | 24/32 (75.000%) | 24/32 (75.000%) | 25/32 (78.125%) |
| rotation | 12/32 (37.500%) | 12/32 (37.500%) | 12/32 (37.500%) | 12/32 (37.500%) | 13/32 (40.625%) | 13/32 (40.625%) |
| cutout | 27/32 (84.375%) | 26/32 (81.250%) | 27/32 (84.375%) | 26/32 (81.250%) | 27/32 (84.375%) | 27/32 (84.375%) |
| distortion | 20/32 (62.500%) | 21/32 (65.625%) | 19/32 (59.375%) | 18/32 (56.250%) | 20/32 (62.500%) | 20/32 (62.500%) |
| occlusion | 14/32 (43.750%) | 13/32 (40.625%) | 11/32 (34.375%) | 10/32 (31.250%) | 12/32 (37.500%) | 12/32 (37.500%) |
| lidar | 10/32 (31.250%) | 10/32 (31.250%) | 11/32 (34.375%) | 9/32 (28.125%) | 11/32 (34.375%) | 11/32 (34.375%) |

All three repeats and per-corruption time/memory are in [per_corruption.csv](evidence/terminal_logit_pilot_20261110/per_corruption.csv); [timing_memory.csv](evidence/terminal_logit_pilot_20261110/timing_memory.csv) contains225 per-arm/corruption/repeat time and allocated/reserved peak records. Raw JSONs retain hook status (smoke-only instrumentation). [Full paired differences](evidence/terminal_logit_pilot_20261110/comparisons.csv) include all15 corruptions, all three repeats and corrected/harmed/net/95%CIs.

## Every corruption: primary paired effects

| Corruption | Comparison | Δ pp | 95% CI pp | Corrected/harmed/net | Time saved % |
| --- | --- | --- | --- | --- | --- |
| uniform | T5 → P5 | +0.0000 | [-9.3750, +9.3750] | 1/1/0 | 32.020 |
| gaussian | T5 → P5 | +0.0000 | [-9.3750, +9.3750] | 1/1/0 | 32.062 |
| background | T5 → P5 | +6.2500 | [+0.0000, +15.6250] | 2/0/2 | 32.066 |
| impulse | T5 → P5 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 32.068 |
| upsampling | T5 → P5 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 32.762 |
| distortion_rbf | T5 → P5 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 32.198 |
| distortion_rbf_inv | T5 → P5 | +0.0000 | [-9.3750, +9.3750] | 1/1/0 | 31.978 |
| density | T5 → P5 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 32.173 |
| density_inc | T5 → P5 | +3.1250 | [+0.0000, +9.3750] | 1/0/1 | 31.993 |
| shear | T5 → P5 | -9.3750 | [-21.8750, +0.0000] | 0/3/-3 | 32.054 |
| rotation | T5 → P5 | +0.0000 | [-9.3750, +9.3750] | 1/1/0 | 32.061 |
| cutout | T5 → P5 | -3.1250 | [-12.5000, +6.2500] | 1/2/-1 | 31.894 |
| distortion | T5 → P5 | +3.1250 | [-6.2500, +15.6250] | 2/1/1 | 32.101 |
| occlusion | T5 → P5 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 32.166 |
| lidar | T5 → P5 | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | 32.170 |
| uniform | T8 → P8 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 35.286 |
| gaussian | T8 → P8 | +6.2500 | [-6.2500, +18.7500] | 3/1/2 | 35.295 |
| background | T8 → P8 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 35.374 |
| impulse | T8 → P8 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 35.341 |
| upsampling | T8 → P8 | +6.2500 | [-6.2500, +18.7500] | 3/1/2 | 35.941 |
| distortion_rbf | T8 → P8 | +0.0000 | [-9.3750, +9.3750] | 1/1/0 | 35.357 |
| distortion_rbf_inv | T8 → P8 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 35.339 |
| density | T8 → P8 | -9.3750 | [-21.8750, +3.1250] | 1/4/-3 | 35.171 |
| density_inc | T8 → P8 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 35.285 |
| shear | T8 → P8 | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | 35.457 |
| rotation | T8 → P8 | +0.0000 | [-9.3750, +9.3750] | 1/1/0 | 35.342 |
| cutout | T8 → P8 | -3.1250 | [-9.3750, +0.0000] | 0/1/-1 | 35.279 |
| distortion | T8 → P8 | -3.1250 | [-12.5000, +6.2500] | 1/2/-1 | 35.430 |
| occlusion | T8 → P8 | -3.1250 | [-15.6250, +6.2500] | 1/2/-1 | 35.499 |
| lidar | T8 → P8 | -6.2500 | [-15.6250, +0.0000] | 0/2/-2 | 35.384 |
| uniform | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| gaussian | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| background | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| impulse | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| upsampling | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| distortion_rbf | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| distortion_rbf_inv | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| density | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| density_inc | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| shear | P8-PM → P8-LM | +3.1250 | [+0.0000, +9.3750] | 1/0/1 | — |
| rotation | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| cutout | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| distortion | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| occlusion | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| lidar | P8-PM → P8-LM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | — |
| uniform | P8 → P8-PM | +3.1250 | [-6.2500, +12.5000] | 2/1/1 | -41.892 |
| gaussian | P8 → P8-PM | +3.1250 | [+0.0000, +9.3750] | 1/0/1 | -41.926 |
| background | P8 → P8-PM | +3.1250 | [+0.0000, +9.3750] | 1/0/1 | -41.538 |
| impulse | P8 → P8-PM | +3.1250 | [+0.0000, +9.3750] | 1/0/1 | -42.076 |
| upsampling | P8 → P8-PM | +3.1250 | [-6.2500, +12.5781] | 2/1/1 | -43.055 |
| distortion_rbf | P8 → P8-PM | +3.1250 | [+0.0000, +9.3750] | 1/0/1 | -42.173 |
| distortion_rbf_inv | P8 → P8-PM | +6.2500 | [+0.0000, +15.6250] | 2/0/2 | -42.001 |
| density | P8 → P8-PM | +6.2500 | [+0.0000, +15.6250] | 2/0/2 | -41.656 |
| density_inc | P8 → P8-PM | +3.1250 | [-6.2500, +15.6250] | 2/1/1 | -41.856 |
| shear | P8 → P8-PM | +0.0000 | [+0.0000, +0.0000] | 0/0/0 | -42.143 |
| rotation | P8 → P8-PM | +3.1250 | [+0.0000, +9.3750] | 1/0/1 | -42.095 |
| cutout | P8 → P8-PM | +3.1250 | [+0.0000, +9.3750] | 1/0/1 | -41.842 |
| distortion | P8 → P8-PM | +6.2500 | [-6.2500, +18.7500] | 3/1/2 | -42.245 |
| occlusion | P8 → P8-PM | +6.2500 | [+0.0000, +15.6250] | 2/0/2 | -42.161 |
| lidar | P8 → P8-PM | +6.2500 | [+0.0000, +15.6250] | 2/0/2 | -41.794 |

## Repeat stability and same-logit diagnostics

| Arm | Repeat | Prediction changes vs repeat0 /480 |
| --- | --- | --- |
| T5 | 1 | 2 |
| T5 | 2 | 2 |
| P5 | 1 | 0 |
| P5 | 2 | 1 |
| T8 | 1 | 2 |
| T8 | 2 | 3 |
| P8 | 1 | 0 |
| P8 | 2 | 2 |
| P8-PM | 1 | 1 |
| P8-PM | 2 | 1 |
| P8-LM | 1 | 1 |
| P8-LM | 2 | 1 |

Saved GPUfloat32 PM versus CPUfloat64 reduction prediction changes, by arm across three repeats: `{"T5": 0, "P5": 0, "T8": 0, "P8": 0, "P8-PM": 0}`. These changes are precision diagnostics and must not be credited to LM.

[Candidate diagnostics](evidence/terminal_logit_pilot_20261110/diagnostics.csv) contain centered logit norm/range, entropy, top-two probability margin, candidate/LM–PM disagreements and separate P8-PM-terminal versus operational P8 predictions. No temperature, scale, weight or parameter tuning was performed.

## Primary same-logit scale/confidence diagnostics

| Candidate | Centered norm | Logit range | Entropy nats | Probability margin |
| --- | --- | --- | --- | --- |
| clean_02 | 24.971407 | 18.783504 | 0.647416 | 0.678764 |
| clean_04 | 25.022925 | 18.814928 | 0.644530 | 0.679306 |
| clean_06 | 25.023501 | 18.828016 | 0.629020 | 0.685213 |
| terminal | 24.710239 | 18.575914 | 0.638885 | 0.683162 |

Repeat0, equal average across15 corruptions. LM–PM prediction disagreements: 3/480; cases with candidate prediction disagreement: 115/480. Separate operational P8-PM-terminal versus P8 prediction changes: 0/480 (native-repeat confounding; no acquisition parity claim).

## Admission, provenance and limits

Canonical ten-cell fresh-VM setup completed in order, except the user-deferred known cell2 fastrlock install failure; Python3.8 independently verified, later requirements installed fastrlock0.8. Setup dependencies emitted recorded nonfatal compatibility warnings. Native CUDA, dataset and checkpoints passed; [actual setup report](evidence/terminal_logit_pilot_20261110/setup_verification.json).

Actual A100-SXM4-80GB/81920MiB, High-RAM175166460kB, Python3.8.20, Torch2.1.2+cu121, NumPy1.21.2. Session `tta-terminal-logit-pilot-a100-20261110-r1`, endpoint `gpu-a100-hm-ass1c0-vcmn4mvmkhfd`, kernel896. Runtime closed under explicit user authorization after artifact verification; [closure receipt](evidence/terminal_logit_pilot_20261110/runtime_closure_20261110.json). Initial connection loss occurred before Drive verification; only explicit user reconnection command reopened work on the same kernel. No automatic replacement/retry or peer-runtime intervention.

Core scientific source `c76d6b967ea63a6efd2988ec674fbb196756cfad` plus actual fresh canonical native-source overrides; setup checkout dev@a458cd46bba1288a3f41302548bc777700cbc7d4 is distinct. Assembled source snapshot SHA256 `b5ff4e7284274aab24be890a932a0df7261b7d04f50ec126986f5c78f32d7d43`; manifest SHA256 `8bc589db3dd6eb5cc83522cd716d7b59ff3ef5101d567d8708edb280821a09d5`. Fresh seven native libraries and exact sources, strict CPU Point-MAE load and unchanged model hashes are bound in [deployment receipt](evidence/terminal_logit_pilot_20261110/deployment_receipt.json). No historical binaries restored.

GPU smoke:120 cases; actual5/8 prior/scheduler and1/4 decoder/head counts, finite outputs, zero cheap backward steps, frozen models, IDs and labels passed. **Native same-state/RNG repeat floor failed55/80** at unchanged1e-5 maxabs-logit threshold (max1.1585958003997803); zero prediction changes on four smoke IDs. Handoff permits descriptive fresh-repeat pilot despite this failure. No strict repeat parity, isolated guidance causal attribution or noninferiority claim. LM–PM reductions still use the exact same stored tensors.

Pilot225 raw cases/7200 GPU predictions/8640 derived policy predictions; each raw label independently matched SHA-pinned label.npy. Exact ID/order/source/native/model proofs, archive SHA/CRC/all member hashes/standard seven files, full local independent recount and independent bootstrap pass. Actual call/update counts were instrumented in smoke only; timed pilot deliberately has audit=null and zero placeholder counters, independently checked against sealed source. No pilot call-count instrumentation is claimed. Remote and local summary/comparison/per-object output agree. [Acceptance](evidence/terminal_logit_pilot_20261110/pilot_acceptance.json).

Raw run `20261110-191704_pilot_seed0`, archive 4777541 bytes, SHA256 `8570672072e7bbcb9b62e4c06246f33dbfb14863b20f7f9e7612f5b8d0fc8111`. Local immutable ZIP `result/modelnet40_c/terminal_logit_pilot/raw/20261110-191704_pilot_seed0.zip`, extracted sibling. Drive `/content/drive/MyDrive/thesis/result/terminal_logit_pilot_20261110_r1/raw/20261110-191704_pilot_seed0.zip`. Complete setup archive SHA b7622715ff93bc6cb2acd83bcd184ffafaad3e621425aebdd7a022c39df837de; deployment archive c94a11bea7931ee660d8fb5b945273485aaa23dbbe4602d9745e5e9993ad2a36; smoke archive ad6b46a65237a0ba9faee5fd486a84fd57679566735f761ab7362d0f4b0c1b6e.

CPU project verification:888passed/2skipped/564subtests,13new pilot tests; unrestricted root pytest vendor collection failures recorded in [execution history](terminal_logit_pilot_study_20261108.md). Full benchmark, additional inference seeds, held-out data, other classifiers/datasets, B1 latency, component profiler and standalone LM reduction microbenchmark were not run. No automatic expansion or baseline promotion.


Derived analysis/documentation backup:43members,639522B/SHA256
`6496ec9d85bfe76fa2b886cf90f184e189d144674b7359d1667431f91833e09e`,
Drive `/content/drive/MyDrive/thesis/result/terminal_logit_pilot_20261110_r1/documentation_20261110T193956Z.zip`;
whole SHA/CRC/member hashes verified after Drive→Mac roundtrip.
[Receipt and precise prepublication archive scope](evidence/terminal_logit_pilot_20261110/documentation_archive_receipt.json).
