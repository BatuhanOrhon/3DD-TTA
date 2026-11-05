# Shared-state mechanism results — 2026-11-06

[Run/CPU/Inference] H1/H2/H3/H6/H7 diagnostics and their separate posthoc
Point-MAE readouts are complete, including failed/unavailable evidence. H3
shows decoder band mixing on these states; H2 shows substantial graph/operator
drift. The numerical and clean-geometry conditions do **not** support advancing
a new applied accuracy experiment. No threshold, epsilon, profile, branch,
fraction or coefficient was selected or changed after outcome lookup.

## Protocol, sources and raw acceptance

[Code/Run] Four previously exposed ModelNet40-C objects459/545/1229/1435,
Gaussian/Background/Shear/LiDAR, s5/seed0/B2. First/last existing clean-PM
loops2/6 (Background9/27), yielding16 B2 states. Capture is the separate original
SCD control with weight0, N8/BG35, DDIM100/finalalpha1 and raw/eval/EMAoff LION.
Original decoder z0 is fixed; prior conditioning is the actual optimized style.
The frozen [execution protocol](gsd_comprehensive_execution_20261105.md) and
[validation amendment](gsd_shared_state_validation_amendment_20261106.md) apply.

[Run/CPU] Parent verified actual Drive/local archives before this CPU analysis:

| Raw archive | Bytes | SHA256 |
|---|---:|---|
|`received_captures_v2/shared-states.zip`|6,237,002|`279a30f0c7266d963f70790c35c0daa01fe3a234be5e3945ff87a7b3d4024777`|
|`received_mechanism_v1.zip`|1,000,097,904|`6613da2eb5d2dd0ebeaf6aefc775c95ecf3b28e6665b754d3423639ed528a2ee`|
|`received_readouts_v1.zip`|5,058,388|`d563015eb124a1f58d299e7b19b46d1696b1c59d96ac0cc2fbc207821cc1f585`|

Paths are under `result/preparation/gsd_comprehensive_20261105/`.
The immutable model/source overlay is731,868 bytes,
SHA`809573e8351877b8de4408090cf64300de5696dcf3b12a18911c14eba594b713`.
The executed external validation-v2 probe is54,660 bytes,
SHA`c6dc6308c328a6ecfa4c6b512887d40a6f36b2213de3f131663c82d91a9bd535`.
It preserves raw q/U order and the original anchored tolerance; active GPU
ASSETS and the source overlay were not overwritten. Actual operator provenance
is separate from the captured model-helper source identities. Parent later
promoted the validation-only canonical helper change in9a4cbce; that does not
change which external bytes ran on the GPU.

[Run/CPU] Independent CPU acceptance checked SHA, ZIP path/type/size/CRC,
outer/inner sealed inventories, all declared mechanism payloads, embedded
capture source identities, original snapshot/frame/input/ID links, final model/
RNG boundaries and actual operator identity. All16 state receipts and224
predeclared B2 branch keys are accounted for. Final status is
`complete_with_unavailable`, not success of every derivative. Native binary
identities remain linked to the already accepted pinned setup/capture receipts;
this compact analysis is not a second independent ELF extraction.

[CPU] New `scripts/analyze_gsd_shared_probes.py` writes raw gates, all states,
coverage and raw gradient comparisons before opening the accepted H5 clean
oracle. It reuses the accepted `load_oracle`, `physical_clean` and symmetric
mean-nearest squared Chamfer implementation. Clean data are mapped with the
exact corrupted center/scale, factor3.3885 and rotation(-z,y,x). Set-distance
permutation checks pass. Full output is in
`result/preparation/gsd_comprehensive_20261105/mechanism_analysis_v2/`;
V1 is preserved separately. V2 adds exact logit payload/key lineage and paired
margin summaries; it does not change any generated cloud or raw outcome.

## Technical eligibility and frozen FD failures

[Run/CPU] H2 has16 completed graph comparisons. H3 has16 completed decoder
checks and128 B2 perturbations. Direction diagnostics have3 fully eligible
states,12 complete-with-unavailable states and1 failed live-decoder forward gate.
That failure is LiDAR IDs1229/1435, loop2: prior gate passes, but object1435 live
XYZ RMS2.1756511e-5 exceeds the unchanged1e-5 limit (maxabs3.3926964e-4 remains
below5e-4). No forward tolerance was relaxed.

| Frozen joint-B2 directional FD witness | Passed | Attempted |
|---|---:|---:|
|C selective Chamfer|3|15|
|Latent beta2 heat|7|15|
|Physical beta2 heat|8|15|

[Run/Inference] These are the predeclared ±.001 joint-state-norm central
differences, absolute1e-5 + relative.1 tolerance, including repeated-forward
uncertainty. A failure can reflect finite-step nonlinearity, nearest-neighbor/
retained-set changes or native numerical variation; the experiment does not
separate these explanations or establish an incorrect native backward kernel.
There was no epsilon sweep or surrogate replacement. The witness is one joint
B2 direction, not a certificate of every object's complete derivative.

[Run/CPU] C's technical eligibility controls the normmatched branches. Only:

- Gaussian, IDs1229/1435, loop6;
- Shear, IDs1229/1435, loop6;
- LiDAR, IDs459/545, loop6.

support all seven arms. These are **three B2 states containing four original
objects**, with six dependent object-condition rows per arm/fraction. They are
not six independent subjects. There are66/224 available B2 clouds:30 zero,
and6 each for C/S_lat/S_phys/P100_C/CS_lat/CS_phys. The other158 keys retain their
failure/unavailable reasons; no missing cloud is filled with zeros. Zero's
broader30-row-per-fraction readout cannot be compared with a nonzero arm's
six-row denominator as an accuracy effect.

## H2 graph drift and H3 decoder transfer

[Run/CPU] Medians over eight dependent object/checkpoint observations per
corruption, using actual GPU U/q heat operators, common vertex domain and the
same residual Y-X with the original prepared-X anchor:

|Corruption|Edge Jaccard|Heat-F relative difference|Heat-action relative difference|Heat-action cosine|
|---|---:|---:|---:|---:|
|Gaussian|.1219|.9070|.9791|.5027|
|Background|.00621|1.0503|1.0773|.3354|
|Shear|.1273|.9016|.9593|.5233|
|LiDAR|.1234|.9019|.9456|.5218|

[Inference] Fixed-reference operators materially differ from current-Y
operators on these states, especially Background. This establishes staleness
as a measured property, not a benefit from refreshing the graph. No refresh
sampler arm or applied accuracy run was executed or promoted. CPU-reconstructed
filters never replace the actual GPU operators in this comparison.

[Run/CPU] Three same-state detached/no-grad decoder baselines match the archived
PM branch at the original max5e-4/RMS1e-5 gates. Maximum sampled fresh/fresh
output-delta norm is2.85863e-6. All256 dependent object perturbations (128 low,
128 high; both signs/two fixed magnitudes) exceed that norm floor; their
other-active-mode energy also exceeds the sampled modal repeat floor.

|Latent perturbation band|Median physical other-active energy fraction, identity-XYZ control|Median after actual decoder|
|---|---:|---:|
|Low expanded100|.163799|.505981|
|High last100 expanded block|.999984|.853276|

[Inference] The identity control already exposes latent-versus-physical graph
domain mismatch. The decoded differences additionally redistribute spectral
energy: median paired other-active-fraction change is+.34660 for low inputs
and−.14662 for high inputs. This supports decoder band mixing on the sampled
states above the sampled floor. It is neither an isometry theorem nor proof
that spectral guidance causes classification harm. Physical isolates are
reported separately in the raw energy records; “other-active” excludes them.

## Clean-set and frozen classifier readouts

[Run/CPU] The posthoc classifier ran only after geometry choices froze. It uses
the exact existing ModelNet rotate-back/per-cloud normalization, native FPS1024
and frozen Point-MAE, with no classifier gradient or extra PM vote. All present
model/source/native/RNG guards pass. Raw B2×40 logits, keyed predictions,
classifier geometry and actual FPS indices/points are archived without labels.
The external readout operator is24,768 bytes,
SHA`2a16c543ec4bc259060276d766e921c68008813ad01f67388c838c8d55009b91`.
It was promoted byte-for-byte to `scripts/classify_shared_probe_readouts.py`
for reproducible source/tests; this does not change the external executed bytes.

[CPU] Classification analysis opens the official label field from the already
sealed capture only after raw geometry and readout acceptance. It checks every
readout payload's keys, cloud identity, original IDs and independent argmax.
True-class margin is true-class logit minus the largest other-class logit.
Every comparison below uses the same corruption/checkpoint/fraction/object;
original objects receive equal weight, retaining all repeated condition rows.

[Run/CPU] **Every nonzero arm is4/6 correct at both fractions on the same four
original objects and exactly the same matched rows.** Directly checking all60
candidate-versus-C correct indicators confirms they are identical, rather than
inferring equality from totals. Margins and geometry still change. Table entries
are equal-object means of candidate minus C; negative clean-CD difference is
better, positive margin difference is better:

The four correct rows are Gaussian1229/1435 and Shear1229/1435; LiDAR459/545
remain incorrect in every nonzero arm at both fractions.

|Arm|Clean ΔCD, fraction.5|Clean ΔCD, fraction1|Margin Δ, fraction.5|Margin Δ, fraction1|
|---|---:|---:|---:|---:|
|S_lat|−.000185714|−.001541526|+.007369|−.095016|
|S_phys|+.000242540|+.000210493|+.060525|−.008376|
|P100_C|+.000014713|−.000893111|−.017584|+.163289|
|CS_lat|−.001068320|−.000259992|−.379814|−.220561|
|CS_phys|+.000241040|−.000113158|−.464499|−.253585|

[Run/CPU] C itself increases clean CD versus the matched zero step on equal-
object means by.000721798/.001016241 at fractions.5/1: three object clusters
worsen and one improves. There is no observed case among these eligible states
where a helpful C step becomes nonhelpful after adding CS_lat/CS_phys. The
small support therefore does not establish the proposed H1 suppression example.
The existing accepted output-space counterexample remains separate evidence.

[Run/CPU] Independent H6 saved-gradient witness: feature and style updates are
byte-exactly unchanged in all six eligible object-condition rows. Maximum XYZ
norm relative error is1.03841e-7; projection residual against the actual saved
latent GPU low basis is1.63256e-6. This confirms the intended noisy-state
preconditioning/norm behavior, without identifying it with clean-latent or
physical projection. Its clean/margin benefit is not consistent at both fixed
fractions, so an applied H6 expansion is not advanced.

[Decision/Inference] Physical heat fails to beat C clean geometry at either
fraction on equal-object means and has mixed object effects; H7's prerequisite
fails. H2 refresh usefulness was not established, H6 lacks both-fraction
benefit, and most direction states fail the frozen technical witness. Together
with the separate [H5 AUROC.271341/native-repeat rejection](h5_reliability_results_20261105.md),
no new applied H5/H2/H6/H7 accuracy campaign is supported. This is a bounded
not-advanced decision, not a universal impossibility claim or an unrun accuracy
experiment labelled failed.

## Evidence, review and limits

[CPU] [Compact identities, coverage, H2/H3 summaries, H6 saved-gradient witness
and paired four-object records](evidence/gsd_shared_probe_20261106/summary.json).
Raw V2 files retain all forward/FD errors, native floors,128 perturbations,
224 coverage keys,132 individual classifier rows and all clean/margin readouts.
There is no rowwise significance or confidence interval treating these rows as
independent samples, no unseen-cohort confirmation, no trajectory/PM benchmark
and no end-to-end speed claim.

[Code/CPU] Four focused CPU analyzer tests and three frozen readout tests pass;
Python3.8 AST, compile and whitespace checks pass. Tests cover cluster weighting,
accepted physical frame/set invariance, SHA/path/CRC/manifest/exclusive output,
classifier key/label/margin joins and tamper rejection. The independent H4/math
reviewer found no blocker in actual lineage or metric arithmetic. That independent raw-data review also
checked45 FD formulas and66/224 readout keys. Runtime launcher
runpy/900-second cap/partial archive handling was separately reviewed before
parent execution. CPU environment is the persistent root-owned Python3.12.10/
NumPy1.26.4/Torch2.5.1/SciPy1.14.1 environment; no local weights/GPU/Colab run or
commit was performed in this analysis task. Parent owns final Drive backup,
integration and authorized runtime closure after all phases are accounted for.
