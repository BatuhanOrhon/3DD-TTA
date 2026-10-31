# Conditional z guidance: accepted exploratory evidence

See [results and limits](../../shape_latent_conditional_results_20261031.md)
and [prospective design/history](../../shape_latent_conditional_pilot_20261031.md).
UTC2026-10-31; local completion crosses2026-11-01.

- `acceptance.json`, `outcomes.npz`, `per_corruption.csv`: independent raw PM
  recount for seven arms × four corruptions ×32exposed original IDs.
- `paired_latents.npz`, `direction.json`, `analyze_direction.py`: physical
  selected global vectors and posthoc paired-clean-z distance/alignment.
  Clean z never enters the inference objective or per-example step selection.
- `native_gradient_report.json`, `native_gradient_vectors.npz`,
  `native_check_v2.py`: actual small CUDA finite-difference gate for both starts,
  after fixing only proof serialization. Not full-Jacobian certification.
- `pilot_receipt.json`, `smoke_receipt.json`, `failed_receipt.json` and
  independent acceptance receipts: all raw attempts preserved, including the
  failed native report and byte-identically reused completed firstpass.
- `technical_pilot_repeatability.json`, `compare_repeats.py`: existing GB
  source/state/RNG-matched repeat variation; no additional inference run.
- `shape_latent_conditional.py`, `eval_shape_conditional.py`, `selfcheck.py`:
  executed exploratory source and host synthetic contracts; these are audit
  copies, not installed production/default modules. Host selfcheck records its
  original local repository/test paths.
- `analyze_conditional.py`, `legacy_acceptance.py`, `extract_verified.py`:
  actual independent NumPy1 acceptance and archive validation sources.
- Setup/asset/timing/verification receipts identify checkpoint/data/native/
  source bytes and distinguish original budget from prospective amendment.

Full immutable raw ZIPs, operator versions, used source payload and CPU
attempts are archived under
`MyDrive/thesis/result/shape_latent_conditional_pilot_20261031/`.
Local artifacts live under
`result/preparation/shape_latent_conditional_pilot_20261031/`.
The source payload restores `prepared/assets/source_snapshot`; put that path
and the audit-script directory on PYTHONPATH for CPU reanalysis. Use NumPy1
for the physical pickle/RNG schema. `analyze_conditional.py` takes raw-phase,
paired-cache and **separate derived-output** directories; never analyze in
place over immutable raw files. `analyze_direction.py` takes that derived
directory, raw phase and a fresh JSON output path.

All seven accuracy contrasts include0 in their conditional object-cluster
95% interval. New global guidance moves final z farther from paired clean z
than each initial condition, despite slightly reducing held-out energy and
the drift versus prior3. No benchmark/seed-stability/speed/default claim.
