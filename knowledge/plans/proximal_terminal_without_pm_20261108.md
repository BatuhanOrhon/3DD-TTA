**2026-11-10 status:** CPU implementation/testing in progress under the user-requested small pilot. Latest [execution record](../terminal_logit_pilot_study_20261108.md) supersedes the historical NOT_RUN/unsupported-code status below; no new GPU result yet. Full-data expansion is outside this task.

# Terminal-only acceleration without PM — registered 2026-11-08

[User report/Open] **PLANNED / NOT_RUN; terminal-only acceleration arms not
implemented.** The user asks to integrate the speed method into terminal
3DD-TTA without PM. Both runtimes remain closed; this plan allocates none.

## Question and controls

Does the existing frozen-correspondence latent Chamfer solve/DDIM bridge
improve terminal-only speed while retaining accuracy, independently of PM?
Use a matched 2×2 comparison to distinguish the original N5 policy from the
tuned N8 policy. Every arm has just one final decode/classification, no
intermediate2/4/6 readouts and no PM. Within a horizon, change only guidance:

| Horizon / all15 | Classic terminal control | Proximal terminal candidate |
|---|---|---|
| N5 | Original rates gamma=eta=.01, p=.95 | Same preparation/horizon, cheap tau=.1, retained .95 |
| N8 | Tuned gamma=.03/eta=.003, p=.95 | Same preparation/horizon, cheap tau=.1, retained .95 |

[Code/Open] The current cheap kernel/CLI baseline is N8+PM; N5 cheap and
readout-free terminal execution require future implementation and a technical
smoke before scientific launch. Gamma/eta are reference controls; cheap uses
tau and has no denoiser backward/direct style step. Transferring tau=.1 to N5
is a fixed exploratory transfer, not a known optimum. No new parameter search
or corruption-specific Background35 fallback is part of this plan.

## Future execution order and record

1. Review sampler/observer changes; verify actual5/8 scheduler steps, exactly
   one final readout, no hidden PM probes and original encoded decoder style.
   Check source/native/weights/data identities, frozen model state, finite
   outputs and native repeats before accuracy/timing. Failure stops advancement.
2. Pilot on a fixed explicit ID manifest across all15 before estimating the
   full GPU phase. Then evaluate all2468 ModelNet40-C IDs/s5/seed0, Point-MAE,
   common B128/final36 if actual hardware validation still passes. Preserve
   seed1/2 confirmation as a separately estimated phase. Record real controls,
   not an assumed saved-trace equivalence after removing observers.
3. Use fresh native encode/noise preparation for **every timed arm** with the
   same method-independent draw seed and balanced arm order. Do not compare a
   cache replay against a cache miss. Synchronize CUDA and measure preprocessing,
   encode, guidance, terminal decode/classify and total time/allocated memory;
   archive preparation/loading/I/O separately. Repeated timed passes are needed
   to qualify throughput beyond a single pass. B128 amortized time is not B1 latency.
4. Report per-corruption exact counts/accuracy, all15/macro14, paired
   corrected/harmed/net, object-cluster95%CIs, runtime ratio/time saved and
   memory. Publish negative/null/numerical failures and protocol identities.
   A prospective useful operating point has at least15% less total time and
   paired95% accuracy lower bound above−.3pp versus its matched classic horizon;
   this is a planning criterion, not a passed test. Compare N5 and N8 whole
   policies separately, without attributing their difference solely to guidance.

[Run/Open] Existing N8 terminal accuracy from PM acquisition is useful context
only. **No standalone N8 terminal latency exists**; PM acquisition times cannot
be assigned to it. Original N5 timing/accuracy and current cheap N8+PM results
remain separate archived controls. New software/source receipts and matched
hardware timing are required before calling this terminal variant validated.
