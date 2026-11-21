# Point2Vec full ModelNet40-C comparison — 2026-11-20

[Code/Run] 2026-11-21: a [detailed Turkish A/B test inventory and interpretation](cloudfixer_ab_detailed_tests_results_20261121.md)
now connects this full benchmark with the separate Point-MAE mechanism study.
It reuses the accepted artifacts; no new inference or test-set tuning was performed.

**[Run] The two locked proximal policies did not beat original plain CloudFixer in this full-test Point2Vec condition.** Accuracy was **77.244733%** for plain CF, **74.954079%** for N8 late PM, **73.233387%** for N32 true terminal and **61.985413%** for the source control. This rejects the tested transfer benefit in this condition; it does not establish global method inferiority or isolate a backbone effect.

## Locked protocol

All **2,468 original test objects × 15 corruptions × severity 5 = 37,020 observations per arm**. Author seed **2**, batch **1**, frozen Point2Vec and official CloudFixer XYZ diffusion; no LION in these scientific runs. Official loading, setup, preprocessing, class order and checkpoint verification were used.

- **N8/K80/tau .1/retained .95:** equal probability mean at native indices **[20,10,0]**.
- **N32/K160/tau .03/retained .95:** only the actual extra-t0 terminal; PM disabled.
- **Original CloudFixer:** official rotation/delta optimization, first plain logits as the primary readout.
- **Source control:** the frozen classifier's preadaptation forward in the matched author stream.

[Code] **True terminal definition:** N counts reverse updates. N32/K160 queries
native indices160,155,…,5 during32 updates, then makes one additional native
t0 denoiser call:33 denoiser calls total. Only the final clean XYZ from that
extra call is classified for this arm; no PM and no additional proximal update
at t0. The last post-correction clean cloud, the transported pre-extra-t0 state
and the true terminal are distinct. N8/K80 late PM uses9 denoiser calls and
three classifier observations at[20,10,0]. Comparing those two A arms changes
N/K/tau/readout together; B supplies the same-trajectory PM/terminal comparison.

The native second vote1 forward still executes to preserve the author's next-object RNG stream and is diagnostic. Source control is not an independently reproduced standalone paper baseline. No full-test tuning, window reselection or corruption exclusion was performed. An intermediate probe accuracy is not relabeled PM or true terminal.

## Primary results

| Policy | Correct / 37,020 | Accuracy (%) | Recorded seconds | Peak allocated (MiB) |
|---|---:|---:|---:|---:|
| Source control | 22947 | 61.985413 | 713.338 | 436.935 |
| N8/K80/tau.1 late PM | 27748 | 74.954079 | 9071.108 | 586.330 |
| N32/K160/tau.03 true t0 terminal | 27111 | 73.233387 | 25822.797 | 586.283 |
| Original CloudFixer, plain first logits | 28596 | 77.244733 | 29013.512 | 586.407 |



The legacy column name `peak_gpu_memory_mb` contains **MiB**. Times sum synchronized per-object measurements across the resumed A10040 VMs. They exclude comparison gates, diagnostic vote forward, I/O and setup. N8 PM used about **1/3.20** of plain CF's recorded method time, with lower accuracy. These are not end-to-end deployment or billing measurements.

Diagnostic vote1: **28,545/37,020 = 77.106969%**, distinct from the primary plain-CF count of **28,596**.

## Paired policy comparisons

| Policy minus plain CF | Difference (pp) | Paired nominal 95% CI (pp) | Repairs / harms |
|---|---:|---:|---:|
| N8/K80/tau.1 late PM | -2.290654 | [-2.663425, -1.925918] | 1240 / 2088 |
| N32/K160/tau.03 true t0 terminal | -4.011345 | [-4.513844, -3.506213] | 1507 / 2992 |



Nominal 95% percentile intervals use **10,000 PCG64 draws**, seed **20261118**, resampling the **2,468 original objects jointly across all 15 corruptions**. They condition on this evaluation seed and native realization; they do not measure seed or optimizer variability.

N8 PM exceeds N32 terminal by **637 cases / +1.7206915 pp**. N, K, tau and readout all differ, so this is a whole-policy comparison, not an isolated PM effect. The [held-out Point-MAE study](cloudfixer_pm_hypotheses_confirmation_20261118.md) separately tests readouts on shared trajectories.

## Per-corruption accuracy

| Corruption | Source (%) | N8 PM (%) | N32 terminal (%) | Plain CF (%) | PM minus CF (pp) |
|---|---:|---:|---:|---:|---:|
| background | 36.831442 | 65.316045 | 76.458671 | 76.094003 | -10.777958 |
| cutout | 83.995138 | 85.251216 | 82.333874 | 81.442464 | +3.808752 |
| density | 86.547812 | 79.659643 | 75.000000 | 79.335494 | +0.324149 |
| density_inc | 90.113452 | 86.912480 | 83.184765 | 87.763371 | -0.850891 |
| distortion | 75.810373 | 76.337115 | 72.325770 | 78.687196 | -2.350081 |
| distortion_rbf | 74.432739 | 79.376013 | 77.876823 | 80.753647 | -1.377634 |
| distortion_rbf_inv | 77.025932 | 81.969206 | 80.064830 | 83.427877 | -1.458671 |
| gaussian | 49.554295 | 90.842788 | 87.641815 | 91.572123 | -0.729335 |
| impulse | 65.599676 | 91.936791 | 89.262561 | 91.572123 | +0.364668 |
| lidar | 20.299838 | 40.842788 | 39.870340 | 38.371151 | +2.471637 |
| occlusion | 41.531605 | 41.369530 | 41.004862 | 41.166937 | +0.202593 |
| rotation | 46.920583 | 48.055105 | 44.935170 | 69.489465 | -21.434360 |
| shear | 79.416532 | 75.891410 | 72.730956 | 78.970827 | -3.079417 |
| uniform | 63.614263 | 90.437601 | 87.803890 | 90.964344 | -0.526742 |
| upsampling | 38.087520 | 90.113452 | 88.006483 | 89.059968 | +1.053485 |



PM wins **6/15** corruptions and N32 terminal **3/15** against plain CF. Rotation accounts for **529** and background for **266** of CF's **848-case** net advantage over PM: **795/848 = 93.75%**. This is descriptive error analysis; it does not justify excluding those corruptions or retuning on their labels, and it does not prove that rotation optimization caused the gap.

## Validation and recovery

The run is `20261120-184955_full`; campaign `4dc0b06afa30c61cc8e68e761ae5c34e6564f4b108627a883eb62bac55646e95`. Scientific source is `9afa8436b1770e38e148f9093d3edd85a4d9770a`; official commit `f043e24414fcf4165fdc892abff21215c9989141`. Frozen source, checkpoint, dataset, environment and execution identities are archived in the raw and derived evidence.

The raw ZIP is **136,139,832 bytes / 2,388 members**, SHA256:
`b2f0685cb1f0c3e0a053dff6e19f71503e9b793ba2ed340ff1eb0c6f6081e42d`.
Drive: `a_8c70334b1806/archives/20261120-184955_full.zip` under the recorded study root. Drive atomic write/readback and installed CLI API download passed SHA, CRC, all member hashes and required-file checks. All **2,212 historical raw files / 35,000 observations** remain byte-for-byte identical to the pre-recovery snapshot.

| Runtime stage | New blocks | New observations |
|---|---:|---:|
| r3 | 847 | 26,824 |
| r9 | 182 | 5,740 |
| r12 | 77 | 2,436 |
| r13 | 64 | 2,020 |
| Total | 1,170 | 37,020 |

Old complete blocks were replayed from immutable receipts and per-block RNG, without rescoring or selection. Native CF is one operational realization stitched across four A10040 VMs. Fresh deterministic source/proximal goldens and historical adapted-XYZ classifier replay were exact; repeated native KNN backward can change floating-point optimization outputs. Native bitwise reproducibility was not established.

The independent CPU analyzer validated all labels, raw logits, probabilities, entropy, indices, logSNR, timings, memory and RNG chains; its 4 summary and 60 per-corruption rows match the archived CSVs. A separate reviewer recounted every raw block through a stdlib NPY parser and float64 softmax and confirmed all counts and repair/harm pairs. Bootstrap recomputation and GPU replay were outside that second review.

## Interpretation and limits

A uses one fixed seed and includes the earlier development objects. The PM window was selected in an earlier Point-MAE development exploration, then locked before A. Backbone, cohort and seed also differ from that exploration. This is not an independently established optimum or a pure backbone-effect test; it does not establish paper parity.

The accepted [Point-MAE confirmation](cloudfixer_pm_hypotheses_confirmation_20261118.md) used 256 held-out objects, three seeds and all 15 corruptions at severity 5. Late PM beat its own terminal by pooled **+0.903 / +1.727 / +0.660 pp** in the three locked settings. Entropy ranked candidates much less strongly within trajectories than in pooled data; entropy weighting gave no consistent extra gain. Cumulative-correction, observation-position and extra-t0 effects remained conditional. Its absolute accuracies are not directly comparable with this full Point2Vec cohort.

All prior setup, consent-wait and budget charges remain. R13 setup was **8,562.107 s**; combined A+B setup **108,028.702 s = 30.008 h**, under the explicit amended **32 h** bound. The scientific manifest retains its registered 30 h estimate; the separate operational A cap is 54 h, with task cap 400 CU and reserve 54.33 CU. R13 full-wrapper wall time was **5,555.988 s**. These differ from per-method recorded time and total billing.

## Artifacts and operational completion

- [Raw ZIP](../result/modelnet40_c/cloudfixer_point2vec/20261120-184955_full.zip)
- [Summary CSV](../result/modelnet40_c/cloudfixer_point2vec/20261120-184955_full_recount/summary.csv), [per-corruption CSV](../result/modelnet40_c/cloudfixer_point2vec/20261120-184955_full_recount/per_corruption.csv), [paired comparisons](../result/modelnet40_c/cloudfixer_point2vec/20261120-184955_full_recount/paired_policy_differences.csv)
- [Validation](../result/modelnet40_c/cloudfixer_point2vec/20261120-184955_full_recount/validation.json), [derived source/evidence ZIP](../result/modelnet40_c/cloudfixer_point2vec/20261120-184955_full_recount.zip)
- [Full execution and evidence timeline](cloudfixer_two_colab_execution_20261118.md)

[Run] Derived ZIP: **192,235 bytes / 23 members**, SHA256
`8c6f9da76d509213d31de0da88f555ff28acdf98c95588350e123750e80960aa`.
It is stored under task `a_8c70334b1806/analysis/20261120-184955_full_recount/20261120-184955_full_recount.zip`.
Atomic Drive write/readback, installed CLI API download, byte equality, CRC and
all archived member hashes passed. Source, environment, command, stdout,
validation and tables are included. Raw bytes were not modified.

[Run] **Operational completion verified 2026-11-20T20:35:58.158429+00:00:** official Drive flush/unmount
finished at 2026-11-20T20:33:43.145207+00:00. Owned
`tta-cloudfixer-point2vec-recovery-a100-20261120-r13` was then stopped using the
user's prior completion-and-close instruction. CLI stop returned0; independent
server inventory confirms its assignment absent, with **0 active assignments,
0 CU/hour** and **90.10 CU** balance. No peer runtime was stopped.
Scientific/preservation acceptance was committed at
`973262d564da7817c8c6748f5e7d50f593d35318` before flush/teardown. Raw and derived
archives remain unchanged. B scientific results were already accepted; its
runtime had disappeared externally earlier, not through this A teardown.

The r13 conservative allocation-to-stop interval was approximately4.176h,
including all consent/setup wait. Observed account balance changed from117.81CU
before r13 allocation to90.10CU after closure (27.71CU); this is an account
observation, distinct from logged method time and an exact billing interval.
The final operator-source/receipt archive is preserved under knowledge evidence;
its receipt records all source/member hashes.
