# Clean-z oracle pilot — 2026-10-31

[User report/Decision] User approves a small pilot of the four-arm diagnostic
discussed in chat and explicitly requests saving current knowledge before
starting. No full dataset sweep; inspect results before deciding continuation.
Worktree7bf0/branch integration/chamfer-guidance-variants-20261030 preserved.

## State before this pilot

[Run/Inference] Previous fixed global-prior pilot is complete and published at
be3bdcaa5e336a8ebee06db9369fbc5918219add. ModelNet40-C/Point-MAE/64exposed
objects/all15/seed0, localN8/Background35 PM: baseline60.9375%, updated60.729167%,
anchor60.3125%, proposal60.3125%, guided3 60.416667%. No general gain; all paired
object-bootstrap comparison intervals include0. Background drift capping
does not improve accuracy. Gaussian global proposals move z much more than
native style updates. Global diffusion stays off in the operational policy;
this does not disable existing gradient-based conditioning-style updates.
[Full result and provenance](shape_latent_tta_results_20261031.md).

[Code/Inference/Open] LION factorizes q(z|X)q(h|X,z) and p(z)p(h|z)p(X|h,z).
Previous interventions retained h sampled under the original z. Harm from
changing this pair is a hypothesis, not a proven explanation of null results.
Clean-z anchoring alone may also preserve corruption. A proposed second
compatibility guidance is deferred until clean global information shows useful
headroom. Candidate metrics discussed: conditional local-denoising error as
a proxy for p(h|z) compatibility; encoder-posterior consistency as a regularizer.
Neither is implemented, approved as an extra GPU arm, or a calibrated exact
joint-likelihood/guarantee. No new guidance is included in this oracle pilot.

## Approved small design and interpretation

| Arm | Global z source | Initial h source / condition |
|---|---|---|
| baseline | corrupted X | corrupted X, corrupted z |
| clean_z_fixed_h | clean X | exact baseline h retained |
| clean_z_reencode_h | clean X | corrupted X, clean z |
| clean_pair | clean X | clean X, clean z |

[Decision] Clean z means an encoder sample from the same original object's
clean data_original.npy, not a global-denoiser prediction. All four arms keep
global diffusion off. C is conditional reencoding, not a guarantee of clean h.
D is a full clean-latent diagnostic witness, not a formal accuracy upper bound.
Chamfer target is fixed to baseline corrupted-input h XYZ in every arm, includingD.
The raw-X wording in the initial chat was imprecise; actual host anchors
initial h XYZ, so replacing h must not silently replace that anchor.
Decoder/probes use the supplied initial z, matching the reference host policy;
existing local conditioning-style gradient updates remain unchanged.

[Decision] Same exposed64 original IDs and all15/severity5/seed0/Point-MAE as
the previous pilot; technical32 Gaussian/Background gate, no tuning. Common
global posterior standard-normal draw for clean/corrupt z, common local
posterior draw for all h branches, common local diffusion epsilon and common
post-preparation RNG. Recompute reference under this matched preparation;
do not transplant old60.9375% as the paired oracle reference. Clean and corrupt
preprocessing use the same declared normalization/rotation convention; record
their transforms, point subsets and object correspondence explicitly.

[Decision] Main contrast clean_z_reencode_h−baseline; B−A includes mismatch,
C−B tests conditional reencoding, D−C replaces the whole canonical local
input/preparation, including normalization and interpolation. It is not a
pure local-denoising or compatibility contrast. PositiveC with
negativeB supports the compatibility explanation; positiveD without positiveC
suggests clean global information alone is insufficient under this protocol.
Null results weaken further global-prior investment in this host but do not
prove a universal no-headroom claim. +1pp is a proposed practical signal to
examine, not a significance threshold or an accuracy-driven parameter search.
One seed/exposed objects/clean oracle: exploratory diagnosis, not deployable
TTA, confirmation or full benchmark. No additional phase launches automatically.

## Execution plan / acceptance

1. Inspect exact clean/corrupt asset identities and ID/shape correspondence;
   preserve existing source, outputs and own runtime. Prepare separate immutable
   oracle source/asset/state/output roots. No old archive/cache overwrite.
2. Test paired conditional preparation and fixed-arm wrapper first on CPU:
   exact common posterior noise, B retains h_A, C uses corrupt input, D uses
   clean input, correct log-sigma offsets, no global-prior calls, source/RNG
   guards and unmodified reference sampler/probe behavior.
3. Seal recoverable code and fixed commands/IDs; verify current own A100,
   existing setup/Drive/native/checkpoints/source. Run native gate then fixed
   four-arm64/all15 pilot. Expected entire research phase15–25min; separate
   shared3300s/55min hard cap covers gate, jobs, retries and artifact overhead.
   This is a new user-requested question after the completed prior phase, not
   a reset/retry of that phase. Reassess if estimate would exceed60min.
4. Preserve success/partial/failure, verify Drive/local ZIP SHA+CRC/source and
   physical posterior/cache/RNG proofs, independently recount PM outcomes,
   report paired all15/macro14/Background and conditional object intervals.
   Stop at results; no new solver, grid, seed, full data or dataset extension.

[Open] Status: documented before implementation/GPU launch. Source HEADa82200ad
+preserved dirty research; final new snapshot/commands/SHAs pending preparation.
Assigned existing own session tta-shape-latent-a100-20261031-r1, endpoint
gpu-a100-hm-ass1c0-1ulu15s7e5w42; owner=root/shape-latent, current task oracle
follow-up. This assignment does not authorize VM closure. Verify before reuse.
Local preparation result/preparation/shape_latent_oracle_pilot_20261031/;
Drive MyDrive/thesis/result/shape_latent_oracle_pilot_20261031/ (fresh outputs).
No numerical result from this new pilot exists yet.

### Unexpected runtime loss before follow-up — 13:57 UTC

[Run/Failure/Open] Read-only server inventory reports no active sessions and
prunes one stale local record. Previously owned r1 was not stopped by us;
this is unexpected runtime loss, not authorized teardown or successful cleanup.
Prior raw/derived archives are already verified on Drive and locally. No new
oracle inference existed to lose. New uniquely owned A100/setup is required;
canonical ten-cell preparation must run for the new VM before research.

### CPU implementation / posterior-noise trap found before inference

[Code/Run] New shape_latent_oracle.py and scripts/eval_shape_oracle.py are
opt-in; existing sampler/production paths are unchanged. Tests first fail on
missing feature, then pass. An additional independent standard-normal-draw
check found that models/distributions.py sample_normal_jit mutates rho in-place
and returns rho aliased to the final sample. It is not the original Gaussian
noise. The oracle now draws/stores eps_z/eps_h explicitly and uses
sample_given_rho. RED→GREEN proof retained. Existing encode() ignores that
second return; no prior accuracy result is invalidated by this observation.
CPU test confirms baseline samples/noise/RNG match the original encode path.

[Code/Decision] Fresh review confirms ModelNet decoded_to_geometry renormalizes
outputs and ignores passed corruption center/scale (Scan differs). Keep clean
and corrupt normalization independent, preserving genuinely clean global z.
Putting clean z in the corrupted bbox frame would contaminate the main oracle.
D−C includes clean normalization/interpolation; D has no strict upper-bound
interpretation. Explicit common Chamfer reference is new opt-in sampler input;
default math remains unchanged, targeted old-host parity checks pass.

[User report/Run] User granted new VM Drive access. First request had timed
out, then proper CLI retry propagated the granted credentials and physically
mounted /content/drive; no second user consent was required. User explicitly
accepts continuing in the new session after the old server assignment remained
absent. Canonical Cell3 verified real mount/wheel; Cells4–10 now run sequentially.
Final CPU suite519passed/1historical fixture skip/373subtests, seven warnings
(including existing TorchScript deprecation). No oracle GPU inference yet.

### Source and CPU acceptance — 14:20 UTC

[Code/Run] Scientific snapshot338files sealed before research. Asset manifest
85057eeba260c7f2608f8ee38b3f9df7948af0754c444ba211654cf851a5e3bb;
source ZIP42d7f973b5897e63c813fb36684bfcbe83eddb15b45819694910a1a341263b55;
payload2266031bytes SHA1e6d86afef987fa66fd2b4e04a2c231a1ca75b79882dd4921ded5e4336329505.
Operator ZIP17728bytes SHA603204bacd3afc887484dac60a31b66a8ab47fbac41b6685e8db778d26707294.
Final519CPUtests/373subtests and4operator checks pass; historical fixture skip
retained. Fresh review found no blocking math/cache/RNG/ID defect. Explicit
scientific metadata and physical prepared-input audit gaps are fixed together
with RED→GREEN tests before sealing. Native readiness/data lineage still pending.

[User report/Decision] User accepts new oracle r1 after confirming the old
prepared runtime is absent from current server inventory; no old-runtime
closure was performed. New Drive consent has propagated successfully.

### Setup/deployment accepted; technical gate launching — 14:42 UTC

[Run] New A100-SXM4-80GB passes exact canonical10-cell markers and physical
phase assertions. Actual downloaded receipt passes Python3.8.20/Torch2.1.2+cu121/
CUDA12.1/NumPy1.21.2;15corrupted arrays, clean/labels and checkpoint identities.
Real CUDA FPS/KNN/Chamfer/EMD checks passed. Fresh native bytes preserved to
Drive before restoring four accepted pinned bindings; no-build CPU load proof
passes strict checkpoint/native/source/data validation with CUDA uninitialized
and forward prohibited. Actual deployment receipt accepted against sealed
manifest/source hashes. No scientific source changed after sealing.

[Run/Open] Launching four fixed GB32 technical arms under a fresh shared3300s
research cap, with64/all15 pilot behind raw/physical posterior/cache acceptance.
This follows user-approved new diagnosis; no reset of an incomplete prior run.

### Technical gate accepted / fixed pilot launch — 14:47 UTC

[Run] Four GB32 arms complete; full smokeZIP52073193bytes SHA
d89420d72f2e53ccd260195a97969ef4c482a430763490a6ea31928b8c9d60f9
verified Drive/local/CRC/safe extraction. Independent local NumPy1 posterior
reparameterization, physical cache/RNG, branch/anchor/source/frozen-state/PM
recount exactly equals remote acceptance. Two common states, two posterior
proofs and eight selected states are complete. Shared research start
14:41:56.363042UTC; technical controller elapsed148.914365s, with subsequent
retrieval/analysis also charged by the continuing ledger, not reset.

[Run/Inference] Technical contrasts only: B−A+3.125pp, C−A0, D−A+6.25pp,
32original objects ×2conditions, not confirmation or parameter selection.
Proceed with the already fixed64/all15/seed0 four-arm pilot, no settings
change or scope expansion. No extra seed/full-data phase will auto-launch.

### Pilot complete / independently accepted — 15:06 UTC

[Run] All four64/all15 jobs complete. Pilot rawZIP621835687bytes SHA
52bffe3f220bc76c3e9cdd486f033f954148b240515d3f33ad659795aa9ce033
passes Drive/local SHA,CRC,safe extraction and local NumPy1 full physical
posterior/cache/RNG/source/state/PM recount. Outcome matrices equal remote too.
30common states/30posterior proofs/120selected caches/120batch shards. Research
ledger907.6917s from14:41:56.363042UTC, no cap reset or failure retry.

[Run/Inference] A60.833333%, B64.895833%, C63.020833%, D77.708333%. Main C−A
+2.1875pp CI[.625000,3.854167]; B−A+4.0625pp[1.979167,6.354167]. C−B−1.875pp
[-3.958333,0], no reencoding gain. [Report/limits](shape_latent_oracle_results_20261031.md).
No full-data/extra-seed/new-guidance run. Clean global headroom exists in this
host; global prior stays operationally off. User decides subsequent research.

### Final archival/runtime observation — 15:20 UTC

[Run/Failure/Open] After all scientific phases and raw retrieval/acceptance,
final derived upload returned Session not found. Read-only server inventory
then reported no active runtime. No stop/release/teardown command was executed;
loss time/cause is unknown. User's teardown authorization was never inferred.
All raw phase archives remain SHA-verified on Drive and locally; results and
independent acceptance are complete. This is not loss of the scientific data.

[Run/Open] Extra derived/source/operator capsule is locally CRC-validated,
2311574bytes SHA70eb6a04c9a18011129cf993ffcc68dd8fa399083b130a44c1fb8ebd2f5dc926.
Its additional Drive copy/roundtrip is pending, not claimed successful. Browser
Drive fallback was denied by browser security policy (user declined); no retry
or workaround. Existing authorized Git publication preserves compact analysis/
code/knowledge; raw ZIPs and local capsule are retained. No new GPU allocated
solely for this extra copy. [Archival status](evidence/shape_latent_oracle_pilot_20261031/derived_archive_status.json).

[User report/Decision] User explicitly accepts local retrieval as sufficient
and says there is no need to reconnect Drive. Extra derived Drive copy is
deferred by user decision; all Drive/browser recovery stops. Full local raw
ZIPs, independent analyses and source/operator capsule are retained. No
scientific result or verification remains pending for this pilot.
