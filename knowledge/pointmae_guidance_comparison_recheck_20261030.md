# Point-MAE / ModelNet40-C comparison recheck — 2026-10-30

[User report] The user recalls a C+S accuracy advantage and asks whether the
latest code or reporting may have been wrong. Recheck the historical phases,
not only the latest aggregate or a different backbone/dataset.

[Run/Code] CPU-only independent recount of17canonical raw head ZIPs:2fixed
development,6fixed confirmation and9sensitivity confirmation (including3original
controls). Outer SHA/bytes, CRC and every sealed member hash match the accepted
archive inventories. Direct float64 softmax/probability means and terminal
argmax/counts reproduce the accepted CSVs. Labels/indices, frozen model-state
guards, pinned TTA source, exact loss weights/rates, and C/CS initial input/
latent/noise hashes match. Batch labels/terminal logits equal their global
corruption arrays. Three cohorts are disjoint. No GPU/model inference occurred.

## Matched C versus C+S

All values are accuracy percentages. Differences are C+S minus C, in percentage
points. Each phase uses the same objects/readout/host within its C/CS contrast.
Do not compare absolute scores between phases as a treatment effect.

| Phase | Objects/dataset; seeds | Readout | C | C+S | C+S−C pp |
|---|---|---|---:|---:|---:|
| Oct25 development |70;0|terminal|66.952381|67.809524|+.857143|
| Oct25 development |70;0|probability mean|67.809524|67.809524|0|
| Oct25 frozen confirmation |different70;0/1/2|terminal|62.825397|62.761905|−.063492|
| Oct25 frozen confirmation |different70;0/1/2|probability mean|64.222222|63.809524|−.412698|
| Oct28 calibrated confirmation |new140;0/1/2|terminal|63.333333|63.111111|−.222222|
| Oct28 calibrated confirmation |new140;0/1/2|probability mean|64.349206|64.396825|+.047619|

[Run/Inference] The development phase **does** exhibit the user's proposed
pattern: positive terminal spectral increment and equal probability-mean
accuracy. The preceding conversational answer should have named this stage;
its latest-confirmation counterexample cannot erase that earlier observation.
This is not evidence that probability mean generally replaces spectral loss:
the other phases do not reproduce the same pattern.

The first confirmation also has an actual >1pp matched terminal increment in
seed1: C61.714286%,CS62.952381%,delta+1.238095pp. But seed0 is−.095238pp and
seed2−1.333333pp; the three-seed mean is−.063492pp. Do not select that positive
seed alone as proof of a stable benefit. Probability mean in seed1 is tied.

## Latest confirmation and the original comparator

[Run] Latest three-seed Point-MAE means: original3DD terminal63.253968%,
matched C terminal63.333333%,CS terminal63.111111%,matched C probability mean
64.349206%,CS probability mean64.396825%. Therefore:

- CS probability mean beats matched C probability mean by+.047619pp
  (4057versus4054correct/6300prediction events).
- CS probability mean beats original terminal by+1.142857pp.
- CS terminal is−.142857pp versus original terminal.

These are separate comparisons. The +1.142857pp whole-policy/readout advantage
is not an isolated spectral effect. Repeated prediction events are not6300
independent original objects. No new per-cell confidence interval/significance
claim is made by this recount.

## Protocol differences and code conclusion

[Code/Run] Fixed phases use spectral coefficient1.17; latest calibrated CS uses
937.3767440639812 (label-free L2 scalar timesrho.1). Matched C has spectral0;
both have Chamfer1,eta/gamma.01,p.95,beta2,M800,static graph/updated style.
Latest140IDs exclude both earlier70-IDsets. Fixed phases pin TTAa3b8452f…4650e;
latest pins71f8f75b…284a3, with explicit loss-helper/default-parity history.
The graphs remain legacy. These are registered changes, not an unexpected
wrong-file/parameter substitution detected by this check.

[Decision/Inference/Open] No archive/count/arm-label/protocol mismatch was found
that reverses the published latest table. A remembered earlier CS advantage
really exists, but does not require the later different-cohort result to have
the same ordering. This recount does not prove absence of every algorithm or
native-CUDA defect. The existing isolate-policy defect and its unproven accuracy
impact remain as documented in the mathematical audit. Do not change or rerun
the scientific consumer merely to force the expected ordering.

[Recount script](evidence/pointmae_guidance_recheck_20261030/recheck.py) ·
[Exact archives/protocols/counts](evidence/pointmae_guidance_recheck_20261030/verification.json) ·
[Mathematics audit](guidance_mathematics_execution_audit_20261030.md).
