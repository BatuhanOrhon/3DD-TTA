# Agent B guidance mathematics and executed-source audit — 2026-10-30

## Question, scope and decision boundary

[User report] Before deciding whether to remove GSD from the thesis, inspect
whether smoothing, guidance mathematics, historical progression and actual
execution match. Do not invent errors to rescue a desired result.

[Code/Run] Read-only scientific/CPU audit starting from operator branch
`gsd-guidance-components@59ed079`. No production algorithm was edited, no GPU
inference was run and no Colab runtime was allocated. The owned r4 runtime was
previously closed with user authorization; no new live-runtime claim is made.

[Inference/Decision] The campaign ran its declared **legacy smooth-M800 latent
fidelity** protocol. It did not run full GSDTTA or exhaust the normalized,
full-spectrum smooth formulation developed in September. Its negative/null
results remain valid at their registered scope. They neither prove an execution
failure nor justify discarding every spectral-guidance formulation. Conversely,
the evidence does not support promoting spectral guidance as a demonstrated
accuracy contribution. The user retains the method decision; no new campaign
is scheduled by this audit.

## Actual sources and execution

[Run] Fresh SHA-256 checks match the retained source ZIPs:

- Fixed component source: `f5bca11a543fd5463ef5a09b26cf216b2bd049523d61a97da24e184e5576d0c6`.
- Sensitivity source: `73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8`.
- Both execute `graph_spectral_v2.py`, SHA
  `1cadd63642dc58bb0aa22969c08e00e4c9dd597696cf53107dbe081d3c68807e`;
  this also equals the graph at historical restoration commit `12525b8`.
- Sensitivity `tta_gsd_v2.py`: `71f8f75b07e3317cf55b301801b093b90dd926d077d2db99f0f3cbe291a284a3`.
- Sensitivity loss helper: `937c73a00165b0cff0792776dba4c6ba25a66b08e82bf28c399657a3c02381b4`.

[Run] Checked 32 fixed-development head configurations and 120 newly executed
sensitivity GSD head configurations (30 dataset jobs). All use beta2, M800,
static XYZ graph, legacy isolate policy, updated terminal style and DDIM
finalalphaFalse. In total 576 relevant executed-source bindings match their
sealed sources. Eight new original3DD head records and the separately reused
24 baseline head records are outside this 120-new-GSD count; they are not
silently represented as new experiments.

The graph defaults k10/delta.1/gamma.6 are instantiated explicitly. Runner
overrides M_max1200 with800; graph object's default M100 is not the active
spectral-loss cutoff. Its separate invariant cutoff inconsistency is inactive
because invariant weight is zero. The label/ID/state/native/cache checks from
accepted analyses remain the basis for accuracy acceptance.

## Which smoothing was applied?

Let Q be corrupted encoded latent XYZ, Y the predicted-clean latent XYZ,
and U the fixed reference eigenbasis. The executed legacy L2 objective is:

`L = sum_b sum_(i<800) exp(-2*lambda_bi) * ||[U_b^T(Y_b-Q_b)]_i||^2 / (3*2048)`.

[Code] Smoothing is present: high graph frequencies get exponentially smaller
weights. It is smoothing of the **fidelity residual metric**, not an explicit
low-pass replacement of Y or a guarantee that the decoded cloud is cleaner.
The raw graph eigenvalues are not normalized; indices800 onward receive zero
weight. An outer spectral coefficient scales the loss. C/S/U flags control the
actual applied objective; the diagnostic no_grad block is followed by a separate
gradient-enabled spectral calculation.

| Formulation | Graph/filter and optimization |
|---|---|
| September normalized smooth | Nonself graph; directed gamma/(Nk) threshold; active induced graph; lambda/mean_active_degree; all active modes; original terminal style/5–35 host |
| October restored legacy / Agent B | CUDA self-query graph; gamma*mean symmetric degree; legacy1000 isolation penalty; raw lambda; exp(-beta*lambda) plus hard M800 cutoff; updated style/10–30 host |
| GSDTTA paper | Physical-coordinate graph; learned low-frequency displacement; eigenmap-guided objectives; alternating classifier adaptation |

[Code/History] The October8 restoration explicitly preserved the successful
legacy command structure and these graph/host differences. This was not an
accidental omission of the September code. See
[restoration record](gsd_legacy_smooth_restore_20261008.md) and
[September mathematical design](gsd_guidance_math_20260926.md).

[Paper] GSDTTA Eqs.11–13 use `X_s = X + U_M DeltaC`; Eq.17 optimizes the
shift with pseudo-label/entropy/diversity/Chamfer objectives, and Eq.18 adapts
the classifier. Our latent fidelity penalizes low-band displacement from Q;
it does not implement those learned shifts or their driving objective.
[Primary paper, sections3.1–3.2](https://arxiv.org/html/2507.18225v1#S3).

## Verified correct routes; no invented execution defect

[Code/Run] Shared U on both signals makes the GFT comparison consistent;
there is no missing inverse GFT required for this scalar coefficient loss.
The fourth LION channel is a feature, and Bx2048x4 then XYZ slicing matches
the encoder/decoder layout. Model freezing preserves gradients through the
predicted-clean denoiser to local latent and style. Updates subtract those
gradients. Eta is local and gamma is style in this legacy path; equal defaults
remove any old name ambiguity, and unequal-rate tests exercise the actual map.

[Run] Re-ran 29 focused CPU tests: components4, explicit losses11, legacy
numerics5, normalized smooth9; all pass in0.703s with local torch2.14.1/
NumPy2.5.4. Tests cover old-inline/default parity, active/inactive calls, cached
state/RNG, finite differences, one-step guidance to both variables, heat operator
equivalence and the known isolate witness. This is synthetic CPU verification,
not fresh CUDA/model accuracy or native cross-version parity.

Existing actual-GPU calibration additionally has2100/2100 valid local-gradient
pairs for every loss, preserved native parity checks and nonzero applied
updates. Thus an accidentally detached/disconnected spectral route is not
supported. No evidence establishes a fourth-power KNN-distance error; the new
CPU adapter assumes Euclidean distances and does not verify the old CUDA wheel's
distance units or tie order. No such error is diagnosed here.

## Concrete defect, with bounded evidence of relevance

[Code/Run] The October8 self-only-isolate defect remains in the executed graph:
retained A_ii can be positive while all off-diagonal edges vanish. Then full
row degree is positive, but D-A has a zero row; `degree == 0` fails to apply
the intended1000 penalty. This is a real graph-policy defect, not a new
explanation inferred from low accuracy. The existing offdiag variant was not
selected in Agent B. The hard index cutoff also need not preserve a repeated
eigenspace crossing index800; no such exact boundary degeneracy is established
for the sampled real caches below.

[Run] Fresh CPU replay used the **actual archived graph forward**, substituting
explicit Euclidean CPU kNN, on the first cached development object in six
predeclared cells. No model forward or accuracy was computed:

| Dataset / corruption | Retained self-only nodes missed by legacy penalty | Zero-row nodes penalized | Sum of applied spectral weights |
|---|---:|---:|---:|
| ModelNet / Gaussian |0|384|482.0802|
| ModelNet / Background |0|406|469.7426|
| ModelNet / DensityIncrease |0|475|434.8913|
| Scan / Gaussian |1|478|223.2057|
| Scan / Background |0|154|44.8211|
| Scan / DensityIncrease |0|417|163.3319|

The Scan/Gaussian object's latent vertex281 has filter diagonal0.9993922,
despite being isolated from other vertices in this CPU replay. This establishes
a real-cache witness under the stated CPU adapter, not installed-CUDA prevalence
or the cause/size/sign of accuracy harm. The other five sampled objects have
no missed self-only nodes. These six objects cannot characterize2100 inputs.
Raw-lambda beta2 also yields substantially different effective weight mass
across the sampled objects; a single coefficient calibrates strength, not
filter shape or graph quality.

[Run/History] This defect already had an all15 Point-MAE seed0 legacy/offdiag
control on October10:64.5705% versus64.5543% (-.0162pp), with unmatched prepared
draws. Therefore there is no positive existing accuracy evidence that fixing
it rescues GSD. Preserve the known defect and this negative control together.

## What the U/S evidence actually establishes

[Run] Fixed development probability mean is S52.6190% versus U52.8095%:
S−U=-.190476pp, conditional95% object-bootstrap interval[-.535714,+.143155].
There are102 corrected and118 harmed events among8400, four positive/four
negative dataset–head cells, one seed. Terminal S−U=-.285714pp with interval
[-.786012,+.178571]; entropy difference=-.011905pp with interval
[-.416667,+.392857]. None proves systematic accuracy harm.

Sensitivity S-L2/rho1 reaches52.880952% on the same development cohort versus
the accepted U52.809524%, only+.071429pp. This is exploratory selection evidence,
not an independent U/S confirmation. Other S variants remain near U. The final
new140-ID/three-seed confirmation contains neither standalone S nor U, so no
three-seed U/S conclusion follows from it. C+S versus U is positive in the fixed
screen, but includes Chamfer; it cannot attribute that benefit to spectral loss.

## Historical positive results and search coverage

[Run/History] The October4 accepted full ModelNet/Point-MAE normalized-smooth
comparison records U61.2021% and S61.3596% (+.1576pp), positive in3/3 seeds.
Different native build hashes and no shared-state proof prevent a clean causal
interpretation. It is a genuine small positive observation, not a large
generalized spectral advantage. Historical combined63.8520% versus U61.2021%
also contains SCD; SCD-only63.8799% already accounts for that large descriptive
gain. See [accepted historical ablations](gsd_guidance_ablation_results_20261004.md).

The user's legacy64.5327% report was closely approached by restored64.5705%,
but lacks its exact historical seed/build and an equally matched U/S/C control.
The current two-dataset/eight-head-case subset macro must not be compared with
a full ModelNet/Point-MAE-only macro as an implementation regression.

[Code/Run] The latest registered search was valid for its declared finite
question: six C, six CS, six S settings including reused baselines; label-free
first-step calibration; new disjoint confirmation objects and three seeds.
It fixed beta2/M800/k10/delta.1/gamma.6, static graph, local XYZ and both gradient
routes. It did not search beta, M, graph quality, full-spectrum normalized
smoothing, conditioning-route composition or explicit learned spectral shifts.
CS.03/.03 was absent. Earlier studies did examine normalized beta.5/2 and hard
M100/240/400, with recorded null/negative increments; those are existing evidence
and must not be presented as never tested or automatically repeated.

[Inference] The loss is mathematically a fidelity regularizer anchored to
corrupted Q: Y=Q minimizes it. It can oppose useful low-frequency denoising;
this explains why correct mathematics alone does not imply classification
improvement. Actual per-object clean-reference/margin causality remains unmeasured.
Coefficient calibration does not fix a poor target or establish the direction
helps classification. This is an objective limitation, not proof of a bug.

## Decision gate and reproducibility

[Decision/Open] Preserve all existing scores and the failed tuned-C slot. Do not
declare the whole campaign invalid, advertise spectral accuracy benefit, or
remove every GSD-inspired idea solely from this U/S mean. For the tested legacy
variant, incremental accuracy benefit is unestablished despite bounded tuning
and nonzero guidance. Any continuation must name a specific unresolved mechanism
and reuse existing controls; failure to show a useful matched increment would
support excluding that variant from the main contribution. No new run or fix
is authorized/started by this report; the user's decision is next.

Derived evidence: [audit source](../result/preparation/guidance_component_audit_20261030/audit.py),
[identities/configs/CPU cache probes](../result/preparation/guidance_component_audit_20261030/audit.json),
[CPU verification receipt](../result/preparation/guidance_component_audit_20261030/cpu_verification.json).
The source script contains assertions and reads immutable inputs. Reproduce:
`uv run --offline --no-project --with torch --with numpy python result/preparation/guidance_component_audit_20261030/audit.py`.
