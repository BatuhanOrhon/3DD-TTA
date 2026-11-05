# Shared-state H1/H2/H3/H6/H7 probe implementation — 2026-11-05

[Code/CPU/Open] Adds only `scripts/probe_gsd_shared_states.py` and its focused
native-free tests. Parent owns the capture driver, H4 helper, H5 study, all
GPU/Colab work and shared knowledge integration. Checkout during implementation:
`research/gsd-h5-reliability-20261105`, HEAD
`78658d07e168f337ab3472556f5e0ca849748ebe` plus the parent/sibling working edits.
This note records implementation and toy checks, not actual LION findings.
No branch switch, commit, model-weight inference, GPU or Colab call was made.

## Frozen consumer and identity boundary

[Code] CLI, executed later from the verified asset-root `source_snapshot`:

```sh
python scripts/probe_gsd_shared_states.py \
  --capture-root /absolute/accepted/shared_capture_bundle \
  --capture-config-sha256 PINNED_CONFIG_SHA256 \
  --asset-root /absolute/verified_assets \
  --asset-manifest-sha256 PINNED_ASSET_MANIFEST_SHA256 \
  --output /absolute/fresh_probe_directory
```

Requires the separate zero-weight/control `--capture-shared-states` capture,
original IDs **459/545/1229/1435**, ordered Gaussian/Background/Shear/LiDAR,
B2, seed0, N8/BG35, actual first/last clean PM loops2/6 or9/27. It validates
shared schema1 completeness, seven snapshot identities, selected IDs/timesteps,
tensor hashes/shapes/dtypes, full numeric/JSON RNG hashes, paired original
h0/z0/noise, exact prepared-X and normalization-frame identities, original PM
probe hashes and the current pinned corrupted input rows. No label or clean
array is loaded by the consumer. Artifact SHA verification may read manifest
files as bytes; it does not interpret clean/label arrays or select a direction.

[Code] Uses existing source helpers and `load_guidance_runtime` with original
method, raw LION weights, EMAoff, all-eval modules, frozen parameters, audited
no-build bootstrap. Requires exact Torch2.1.2+cu121/CUDA12.1. Validates source
files loaded by the capture, native binary paths/hashes/bootstrap, resolved
LION config, full model hashes and the complete existing DDIM scheduler record,
including100 timesteps and finalalpha1. It imports PointNet++ solely to bind
and verify the captured PM extension; no classifier/head forward is run.
The CLI rejects ordinary local execution before model loading. Local tests
use tiny toy tensors/modules, never LION weights or native geometry kernels.

[Code] Each output directory/file is fresh (`mkdir exist_ok=False`, exclusive
JSON/NPZ writes). Per-stage arrays precede their immutable receipts; Python
exceptions retain partial/failed evidence. `complete.json` inventories every
written payload by bytes/SHA and distinguishes complete, unavailable and
technical failures. Intermediate failures do not erase earlier states.
Existing capture config/shared manifests remain external immutable references.

## H2: graph staleness, without applying a refresh

[Code] Builds actual current-GPU `build_probe_spectral_target` eigensystems on
prepared X and each captured physical Y, plus sparse normalized Laplacians
from the matching `sparse_heat.build_sparse_heat_target` recipe. Stores actual
active indices, full U, normalized q, sparse L values/indices, rank/mean degree
and diagnostics. Every U/sparse support must agree. Full-vertex zero extension
allows support changes to be compared in one vertex domain.

[Code] Reports symmetric directed edge overlap, active support overlap,
normalized-L and beta2 all-active-heat F Frobenius/action differences/cosines,
expanded low ranks and boundary gaps, and nonself kth/k+1 squared-distance gaps.
Both operators act on the **same residual Y-X with original X anchor**. It
never compares unmatched individual eigenvector coefficients. Sparse L versus
U/q reconstructed L discrepancy is reported separately; matching support does
not establish operator equality. Heat F always comes from actual GPU U/q,
never a CPU-reconstructed graph or basis. Dense graph/eigh, separate sparse
rebuild, boundary gaps and operator comparison costs are recorded separately.
A refreshed full trajectory/accuracy experiment is not implemented here.

## H3: actual decoder band transfer

[Code] Original h0XYZ defines the latent reference graph; exact prepared X
independently defines physical readout bands. Low100 uses the actual production
expanded rank. High last100 expands downward through the whole unresolved
boundary eigenspace. For normalized q the anchored tolerance is
`max(atol/mean_degree, 2*eps*n_active*||L_normalized||_inf)
+ rtol*abs(q_boundary)`, equivalent to the production raw-L boundary allowance.
No chaining of adjacent tolerances; empty support and band overlaps are explicit.
Both use existing k10/delta.1/gamma.6/modes100, atol1e-7/rtol1e-5.

[Code] A fixed `sin((vertex+1)*(coordinate+1))` witness is projected into each
latent low/high subspace and normalized. Tiny projected witnesses are unavailable.
Perturb captured predicted-clean XYZ by ±.001 and ±.0005 times its XYZ norm;
features stay fixed, decoder always uses original z0 and restores exact captured
predecoder RNG for every independent forward. Three ordinary detached/no-grad
baseline decodes match the archived PM branch; archive/fresh and fresh/fresh
max5e-4/RMS1e-5 gates run before perturbations. Failures remain unavailable.

[Code] Saves all raw output deltas, total/low/other-active/isolate energies,
Parseval residual and the identity-XYZ control (input perturbation itself read
in physical bands). Native repeated-forward deltas and their modal energies
are retained. Per-mode energy/norm resolution is reported against that sampled
floor; a below-floor variation is unresolved. This can distinguish mismatch
between latent/physical graph domains from additional decoder mixing in these
states. It proves neither a decoder isometry theorem nor classification harm.

## H1/H6/H7: prior gradients and one real sampler-step branches

[Code] Replays the exact local prior at archived noisy h, optimized conditioning
and preprior RNG; then exact scheduler. Before any native gradient it checks
predicted clean against the captured tensor and live decoded XYZ against the
original capture at fixed max5e-4/RMS1e-5. Decoder uses original z0 and exact
predecoder RNG. Each objective's forward is separately saved and gated before
backward, so unsupported native gradients preserve their forward evidence.

[Code] Objectives are original selective Chamfer C (sum of both directions,
retained floor(.95N), no1/N), latent all-active normalized beta2 heat residual
against original h0XYZ, and physical all-active normalized beta2 heat residual
against prepared X. Heat reduction is fixed3N with batch sum. Autograd traverses
the actual prior/scheduler and physical decoder to both noisy h and optimized
conditioning. Raw local/style gradients, losses and norms are saved. Each gets
one fixed joint-gradient directional central-FD witness with ±.001 times joint
state norm, absolute1e-5 + relative.1 derivative tolerance and repeat-floor
check. It is a local directional witness, not a global gradient certificate.
Failed/unresolved FD or native backward makes that branch unavailable; there
is no surrogate replacement or epsilon search.

[Code] H6 projects **only the already-computed C noisy-state XYZ gradient**
through fixed latent low100, normalizes each projected XYZ gradient back to its
original XYZ norm, and leaves feature/style updates unchanged. Projected norm
<=1e-6*(1+original XYZ norm) is unavailable without amplification. This is
noisy-state preconditioning: P100 and the denoiser Jacobian do not commute.
It is distinct from the peer hard100/200 halfmix objective experiment. Projection
computation occurs before all branch forwards, so the plain-C branch shares
that diagnostic compute history; no separate sham full trajectory is claimed.

[Code] Apply actual .03/.003 scales first. Spectral directions are label-free
per-object normmatched to joint `[.03*g_h, .003*g_style]` C. Parent additionally
froze only CS_lat and CS_phys: sum the matched C/S updates then normalize the
sum back to the C joint norm; a tiny sum is unavailable. Seven branches are
zero/C/S_lat/S_phys/P100_C/CS_lat/CS_phys, each at fractions.5/1. No coefficient
or direction is chosen using a clean geometry, label or classifier margin.

[Code] Actual update is `h_next=scheduler.prev_sample-fraction*update_h` and
`style_next=current_optimized_style-fraction*update_style`. It then executes
**one next active scheduler timestep**, with its correct prior t+1, and decodes
the next predicted-clean h with original z0. Every branch restores a declared
fresh common full RNG from seed1729; its numeric/JSON archive/hash is saved.
This next prior/decoder pair follows normal RNG order without an intervening
reset. The next prior/scheduler runs with global grad enabled and fresh local/
conditioning leaf inputs requiring gradients, matching the actual sampler;
only the following existing PM decoder runs no-grad with detached-clone inputs.
States/updates/clouds are saved before posthoc CPU clean-set or margin
analysis; labels never select branches. The two zero fractions also repeat the
same state and permit an additional native-forward floor readout.

[Inference/Open] C-versus-CS may demonstrate a bounded state where adding
spectral fidelity removes a helpful C effect, if that difference clears the
native floor. Negative-step controls are omitted by the parent-frozen scope.
There is no full trajectory, probability-mean accuracy, unseen confirmation,
universal suppression, refresh benefit or full-pipeline speed claim. Extra
forwards/VJPs and full eigensystems are diagnostic costs, not deployment cost.
Guarded forward timings include buffer/RNG preservation and input cloning;
they are not isolated native-kernel or uninstrumented decoder timings.

## Verification and handoff

[CPU] Nine focused native-free tests pass with
`/tmp/tta-lion-reencode-venv/bin/python -m unittest discover -s tests -p test_gsd_shared_state_probes.py -v`.
Checks include anchored repeated blocks/empty bands, rotation/sign-invariant
energies and projected directions, exact H6 feature/style preservation, XYZ and
joint norm preservation, tiny rejection, operator/action identity, combined
normmatching and sampler.prev_sample arithmetic/next-timestep validation,
immutable numeric outputs/invalid-input rejection, all H3 ±/two-scale probes,
failed FD evidence, and a complete toy prior/scheduler/decoder gradient/FD/
14-branch replay with RNG/input preservation. Python3.8 AST, current-interpreter
compile, CLI help and whitespace checks pass. No full suite was run because
only a new isolated script and its tests were added.

[Code/Review] The H4/math sibling independently inspected boundary tolerance,
operator embedding/comparison and target storage; no numerical blocker found.
It emphasized symmetric-directed edge counts and that matching support alone
does not certify U/q/sparse-L equality. The capture sibling independently
confirmed consumer keys/shapes/RNG/paired/initial-state and scheduler alignment.
Its review found the producer CLI lacked count4; parent authorized the sibling
to add count4 only for shared-state capture, leaving H5/native counts unchanged.
Backend flags now precede CUDA RNG initialization. Actual GPU capture/consumer
compatibility remains a deployment gate.

[Code/CPU] Review P1 found that the original next-step helper put both prior
and decoder under global no-grad, unlike the actual sampler's next prior.
The strengthened toy regression failed that old helper at the expected prior
flag tuple, then passed after the scoped inner `torch.enable_grad` repair.
It explicitly checks all14 next-prior flags (global-grad enabled, local/style
requires-grad and leaf flags true), next-PM decoder global/input grad flags
false, and the exact first/second draws from common seed1729 before prior/
decoder without a reset between them. Existing sampler.prev_sample arithmetic,
outer RNG/input preservation and all nine focused tests still pass. No new
branch, FD scope or actual model inference was added. The FD remains one joint
B2 direction and cannot certify each object's individual derivative.
The independent H4/math reviewer accepted the repaired scope and independently
reran all nine focused tests successfully; no remaining blocker was reported.

[Open] Parent must seal this supplemental source with the capture and sparse
helper, run accepted canonical setup and capture first, then execute the CLI
under its whole-phase clock. Preserve failed/native-dominated branches and
all raw arrays before oracle/margin readout. Native kernels may fail gradients,
finite differences or forward gates even when the toy interface passes. No
GPU result, scalar gate relaxation, applied candidate promotion or research
knowledge claim follows from this CPU implementation.
