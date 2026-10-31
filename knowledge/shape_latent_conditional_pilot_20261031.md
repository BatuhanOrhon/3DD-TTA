[Current Run/Decision] 2026-10-31 21:34 UTC: completed study closed on explicit
user request. CLI stop/server absence and owned monitor shutdown verified.
[Final observations, scope and closure](shape_latent_conditional_results_20261031.md).
Earlier running/keep-live statements below are historical, superseded operationally.

# Conditional z-gradient: encoder versus final SCD start — 2026-10-31

[Latest Run/Decision] Scientific phase complete; both starts and all seven
arms accepted Drive+local,874-file pilot manifest and independent exact
recount. [Final results and limits](shape_latent_conditional_results_20261031.md):
no reliable accuracy gain, guided final clean-z MSE farther than either
start. Runtime/monitor preserved; no further GPU phase. Entries below are
dated design/setup/failure/recovery history, superseded by the final result.

[User report/Code/Decision] User requests both initializations and a new Colab
runtime with the operator skill's monitoring. `tta.py` returns original_style
(encoder z) and updated_style (last SCD conditioning update); these differ.
This is an authorized bounded exploratory prototype, not a production/default
change. Existing code/dirty work and all completed studies remain preserved.

[Proposal] Same32 exposed original IDs and gaussian/background/rotation/lidar,
seed0/Point-MAE, original h0/local noise/anchor and own-z PM decoder. A fresh
shared baseline pass on imported, physically bound preparation states supplies
fixed detached terminal h* and final updated_style. Compare no global prior,
prior3 and conditional-energy-guided3 for each of encoder/SCD starts. The
encoder/no-prior replay also observes native repeat variation; firstpass is
not silently substituted for that control. No clean z enters the objective.

[Proposal] Hold h* fixed across starts. Global DDIM20/10/0, epsilon,1000train/
100infer, clipoff/finalalpha1. Local energy uses all8192channels, unit weight,
four fixed training schedule indices99/299/599/899 and independent private
Gaussian draws, common across arms. Per-object gradient uses feature mean
and batch sum to avoid batch-size scaling. Sequential local condition-VJPs
then global clean-estimate VJP implement the first-order chain rule without
retaining four PVCNN graphs. No detach on the live differentiation path.

[Proposal] Single fixed guidance step norm .02*||encoder z|| per global step,
shared object-specific scale across starts; no oracle/accuracy step tuning,
no grid and no inherited5%ball. Actual vectors/gradients/norms/energies are
saved. Independent held-out noise energy, posthoc clean-z MSE/cosines and
matched accuracy are distinct outcomes. Schedule/weight choice is exploratory,
not a calibrated likelihood or convergence guarantee. Failure does not license
an automatic hyperparameter sweep.

[Open] First technical gate: real frozen CUDA condition/global gradients,
directional finite differences with fixed noise, finite state/RNG/source/
decoder invariants and small timing/memory test. Abort on nonfinite/OOM/
identity/gradient failure; preserve failed output and review before retry.
If bounded pilot cannot fit the budget, retain partial evidence instead of
expanding scope. No unseen/seed-stable benchmark claim.

[Run/Open] Planned session `tta-shape-conditional-a100-20261031-r1`, owner/root,
worktree7bf0/source a82200adbe117ce56a134dc3c22601cb73dc2654 dirty.
Estimate whole phase45min: canonical setup15, diagnostic/pilot20, archive10;
single3300s cap for active setup/research/transfer work, excluding waiting for
user-operated Drive consent. More than60min expected work requires the skill's
separate explicit approval. No GPU work has run at this note's creation.
Session endpoint, primary CLI/monitor identities and source seals pending.

Local root `result/preparation/shape_latent_conditional_pilot_20261031/`;
new Drive root `MyDrive/thesis/result/shape_latent_conditional_pilot_20261031/`.
Fresh output paths only. Ten canonical setup cells, cell1 separate/reconnect,
then2–10; only known cell2 fastrlock error deferred. Stream real progress and
read-only server inventory; no fake anti-idle cells. Monitor cannot prevent
Colab lifetime/quota termination. Do not stop runtime without explicit user
task-complete and close authorization.

## Setup progress — 2026-10-31 20:23 UTC

[Run/Code] Owned endpoint gpu-a100-hm-euw4a0-2jl6gm4ux3xzj, A100/High-RAM.
Read-only local monitor PID85488; no fake kernel cells. Cell1 completed and
reconnected sameVM; Cell2 only known fastrlock failed, Python3.8.20 verified.
User authorized this VM's Drive request and actual mount/Cell3 wheel copy
passed. Canonical4–10 driver active via primary CLI62845; all seven run_cell
sources verified byte-contained from current saved notebook. Native/data/
weights strict gates still pending.

[Code] Prototype files stay under local result/preparation, packaged as
explicit additions to accepted source snapshot; production files unchanged.
CPU condition gradient/dense-reference/finite-difference/global-VJP/batch-scale/
private-RNG/seven-arm scope checks pass. Assets29981388bytes SHA
4cab03682e0d56d956c967da3ae31518f385335d6c82f51476fc82e733c5f09b;
manifest307e157437f623b2052500efb7a0bae64b1c54bf8b228e9e3d34c0ea7ed68061.
Operator v2 adds absolute55min deadline; prepared v1 never deployed.
Technical GB32 firstpass then real native condition/global-chain finite-
difference gate precedes remaining arms. Numerical derivative fractions
.001/.01 of point norm, pass relative error<=.1 or absolute<=1e-5; fixed
noise and frozen-state/RNG proofs saved. Failure stops and archives. No new
GPU accuracy, no runtime teardown, no method promotion.

[Code] Predeployment archive audit found references/manifest.json absent from
v1 declared file map (basename exclusion was too broad). Fixed before any
deployment/research; all v2 declared members now physically verify. V1 ZIP
and operator v1/v2 remain preserved and were never executed. Active assets
v2 SHA8afc9e8e795660606a77a360e29d49f9b6d22b34ec0584a914f8e41b346a095b,
manifestd6231c4aa807ddd72af17c5c9482ef26ead1f9a179124ae66fe9d26655d70054,
operatorv3. Prior pending asset identities above are superseded.

[Run] All canonical cells4–10 completed with exact final marker. A10080GB
CUDA12.1/GCC11.4, EMD/Chamfer/PointNet++/PVCNN physical markers and actual
native GPU checks passed; ModelNet arrays/checkpoints verified. Primary
setup CLI62845 exited0 after SETUP_ENVIRONMENT_VERIFIED and exact wrapper
markers. Sealed source/assetsv2/operatorv3 strict CPU/native deployment now
running; scientific GPU stage has not started.

[Run/Code] Initial CPU deployment stopped before any GPU research: copied
operator called worker.make_scheduler, while the new runner exposes that
helper through worker.parent.make_scheduler. Root cause is operator API
adaptation, not the energy algorithm. Failed log/output and immutable v3
operator retained. Native restoration/extraction already completed. CPU-only
resume_v2 verifies existing assets/native bytes, uses the correct helper and
creates a fresh CPU output; no reinstall/rebuild or failed-directory overwrite.
Scientific sealed source/manifest unchanged. First GPU phase still pending.

[Run] CPU-only resume passed exact SHAPE_CPU_PREFLIGHT_COMPLETE and
CONDITIONAL_CPU_RESUME_COMPLETE at20:49UTC. No CUDA initialization or
forward during CPU proof; strict frozen Point-MAE state/source/native/data
verified. GB32 technical stage now starts, firstpass then native derivative
gate before proposal arms; no claim of gradient/accuracy success yet.

[Run/Code/Open] First GB32 firstpass completed and archived, but native-check
serialization called file_identity(str) instead of Path, before saving its
computed derivative report. No native-pass claim accepted. Failed stage
archive36783790bytes SHA9ecdeec539eff2d7923ba30c9d639e9b72d68c10ef16db48b1fd0d5d4cd074b4
verified on Drive; original failed receipt/ledger/output immutable. Native
check_v2 fixes only Path conversion. Explicit reviewed recovery_v2 preserves
the original research start/deadline and reuses completed firstpass raw/cache
byte-for-byte in a fresh output. It reruns native proof before six proposal
arms, does not rerun the completed baseline. Scientific sealed source unchanged.

[Run] Native gradient recovery passed on one Gaussian object for both encoder
and final-SCD starts: local condition and global-chain derivatives, two fixed
finite-difference scales each; relative errors3.16–6.49% within preregistered
10% tolerance, all eight trials pass. Models/RNG unchanged; peak1,953,943,552
bytes for this small gate. Not full-batch derivative certification or clean-z
recovery/accuracy evidence. Failed raw archive downloaded and exact SHA/size
verified locally; no repeated firstpass. Six smoke proposal arms now running.

[Decision/Code] Prospective timing-only amendment before full pilot: whole
cap55→59min (3540s), after slow fresh setup and187MB technical transfer.
Scientific source, targets, coefficients, seeds and coverage unchanged.
Original55min history retained; no claim it remains the final cap. Research
subcap1165s stays unchanged, including original scientific start/recovery
time. v3 pilot controller changes only absolute whole deadline; stage-v2
ledger and accepted technical evidence reused. Expected remaining pilot
plus archive420s; no >60min phase authorized or launched.

[Run] Recovered GB32 technical stage complete. Raw186869678bytes SHA
262e39804c3947d2a1021a6d1ea9884a7dbf6eb1f94223f32686d2033242747f
Drive+local verified;807-file physical manifest/CRC and independent NumPy1
cache/source/RNG/own-z-decoder/PM acceptance exactly matches remote outcomes.
Whole download completed before an attempted client search found no process;
no client was interrupted. Extra split parts are derived copies, not used.
Technical values are exposed development and do not select a coefficient.
Four-corruption pilot launched21:00:04UTC under timing-only v3 controller;
scientific source/parameters unchanged. Original failed stage/firstpass and
independent acceptance retained. Turkey local date is now2026-11-01; run IDs
and study paths retain UTC2026-10-31.

[Run] Four-corruption pilot remote complete; all seven arms/28shards pass
remote physical acceptance, raw325899956bytes SHA
13ab784c339e5514b2a897cc9e45a5ed4754cd8b280fd04c84d29a394669b609
verified on Drive. Shared research ledger970.088193s=16.168min includes
failed technical attempt, recovery, technical, transition and pilot/raw
archival; under1165s research cap and prospective59min whole deadline.
GPU research ended; no further GPU phase. Large raw is being retrieved in
64MiB derived parts before local independent interpretation. Own runtime
and monitor remain live, no teardown.

[Run/Open] Post-GPU CPU extra-backup CLI lost its connection in execution.py
line227 prelude, before sending the backup script. Server inventory/monitor
remain live; read-only console confirms no extra receipt/local ZIP/final
Drive directory exists, so no duplicate backup job. Same-VM retry sets the
installed jupyter_kernel_client REQUEST_TIMEOUT environment to60; scientific
GPU work is complete and unchanged.

[Run/Code] Extra-backup retry remote completed21:18:16UTC despite local
client interruption. During local failed-client cleanup two matching CLI
clients, including retry, were interrupted; no kernel/runtime was stopped.
Read-only console independently found verified receipt and completed final
Drive directory, so no third backup execution was launched. Source payload
and operator/CPU/setup57-member ZIP verified remotely; actual Drive download
roundtrip now pending. Scientific raw already Drive+local independently valid.
