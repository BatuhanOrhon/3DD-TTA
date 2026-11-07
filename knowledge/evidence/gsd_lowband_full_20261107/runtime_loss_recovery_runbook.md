# Runtime-loss recovery — 2026-11-07 18:39 UTC

Prepared and locally verified only. No runtime access, allocation, GPU work, or
actual restore was performed by these scripts. The parent owns the replacement
Standard A100 session. The LION HighRAM runtime is protected.

All eleven local accepted ZIPs pass their recorded whole SHA/size: three cache
packages and eight inference packages. These eight inference jobs must **never
run again**. The only lost unaccepted job is `shard002_half0`, last reported at
93/480 batches. That progress is not a completed arm or an accuracy result.

## Immutable scientific bindings

- Manifest: `e2cd2942fc9893c863e50018ab5d2373bac00ba5af30a5dbcc75edf99f263f47`.
- `full_source_snapshot.zip`: 1,219,439 bytes,
  `43601f4554882ec1365a9ea52d1641c224df2f4f55ce3ea7e012a7eb1ec38e69`.
- Executed driver: 59,722 bytes,
  `767020693aae803364789a9e345f4ab4fdbad89d5f50c638a1b5b8cf2f7c6b73`;
  commit `1cbd135004e1a298aad650fe9f57bf741ee4982b` over the pinned pilot source.
- Native capsule: 6,451,408 bytes,
  `cc5eacf18ab9e8fcff449645855876f46c5ba3198583ac44d97207d208f76944`.
- Calibration: 52,405 bytes,
  `fb121b127d0ac78ed8ac1ee8dc500d52c5f2a631c4dabaac2a932d6595eca9a1`.
- Whole-phase clock remains `1794066564.0503306` (15:49:24 UTC), 172,800 seconds.
  Setup/recovery time is not subtracted and the clock is not restarted.

`verified_context/inventory.json` contains each accepted label/archive identity,
Drive location, exact code binding and consumed absolute file identities. It also
distinguishes VM-only lost metadata from preserved original evidence.

## Ordered restore, parent review required before execution

1. Finish and verify the ten canonical setup cells on the newly assigned own VM.
   Keep its fresh `/content/cli_setup_verification.json`. Never replace that fresh
   proof with the historical setup report recovered below.
2. Recreate `/content/studies/gsd-lowband-full-20261107/prepared/assets` from the
   original pinned base archive listed in inventory, then overlay **only the
   archived 436 source bytes** and recovered top-level metadata. Do not use the
   current development checkout's driver. Install the original e2 manifest byte
   for byte; do not rewrite roots or any fields. Restore the released ModelNet
   PointMAE checkpoint at `ASSETS/checkpoints/modelnet_jt.pth` from the verified
   canonical checkpoint if absent. It is not necessarily in the base alternatives
   archive.
3. The real data root is `/content/3DD-TTA/data/modelnet40_c` (all17 arrays,
   including original and labels). LION checkpoint/config remain in
   `/content/3DD-TTA/lion_ckpts`. Recreate these actual directories without
   symlinks. Inventory also includes three KNN Python/CUDA source files under the
   exact Conda `site-packages/knn_cuda` path; canonical installation must match
   their old hashes or restore those exact audited source bytes. Project native
   sources are read from ASSETS/source_snapshot by the frozen bootstrap.
4. Restore `full_source_snapshot.zip`, the five top-level metadata JSON files,
   and `ROOT/calibration_frozen.json` from `verified_context`. The historical
   `physical_setup_verification.json` was reconstructed from archived config and
   original serialization order; its exact9003-byte SHA matches e2. It is old
   evidence, never proof that r2 passed setup. The other four metadata files were
   read from historical local assets and matched e2 byte for byte.
5. Upload/extract the recovery context into an owned staging folder. After
   reviewing the new VM's canonical proof, file-only native restoration is:
   `PYTHON restore_context.py STAGING --restore-native`. It preserves the fresh
   binaries/proof in an owned dated recovery folder and restores all10 original
   ELF paths. It imports no native module and launches no GPU work.
6. With Drive mounted, run default `PYTHON restore_context.py STAGING` for the
   audit. It checks all11 Drive job receipts and full ZIP SHA/size, original
   clock/binding, exact consumed paths without symlinks and original native
   bytes. It refuses any lost-job accepted receipt or possible final ZIP: audit
   such evidence before deciding a retry. The actual latest Drive journal and
   lost partial Drive artifacts were **not available locally at preparation**;
   this audit remains necessary after mounting.
7. After review, `PYTHON restore_context.py STAGING --apply` first preserves the
   original Drive journal as dated immutable bytes, writes an explicit loss/new
   attempt record, and ZIPs/verifies all surviving Drive files of the lost job.
   It restores exact accepted job receipts from Drive, transfer manifest objects
   and ACKs from durable local acceptance, original operator files, and actual
   three cache packages/states. It deletes **only** the lost running entry from a
   derived continuation journal. Original journal/clock/accepted entries remain
   separately preserved. Fetch and hash the dated recovery ZIP/records to Mac
   before starting new inference.
8. Run a fresh-process environment/source/native/model/runtime preflight before
   continuation. Match the seven-key **operator** common identity and all model
   state hashes exactly. The local consumer common identity has ten keys and
   must not be copied unchanged into VM ROOT. Check PointMAE raw/eval/frozen,
   LION EMAoff, old backend flags and actual Standard A100/Python/Torch/CUDA names.
   Verify old prepared/noise/cache/RNG hashes for the third shard. Check the
   third-shard M100 graph bases/filter/reference/support against a bounded new-VM
   matched graph preflight before paying for the missing half0. Any mismatch
   stops continuation; do not relax native or graph equality gates.
9. Only after preflight and parent review, the original frozen controller can
   run `PYTHON -u ROOT/operator/campaign.py --stage full --budget-hours 48
   --ack-timeout 3600 --resume`. This file contains no automatic launch. It will
   replay eleven accepted receipts, rehash permanent Drive archives and rebuild
   missing VM cleanup receipts; it will skip all eight accepted inference jobs.
   The first new inference is `shard002_half0` in a fresh UTC bundle. Reconnect
   the existing local consumer/watcher; never start competing consumers.

## Important preservation boundaries

- Old partial Drive runs stay under the same job-label prefix. A new UTC run_id
  creates a distinct directory. Do not restore an old partial bundle under the
  active `ROOT/raw_runs/shard002_half0` search path or its log under active
  `phase_logs`: the frozen controller rejects duplicate configs/log collisions.
  The apply script archives any such VM remnants under ROOT/recovery first.
- Accepted VM bundles/ZIPs/parts need not be restored. Exact ACKs plus verified
  permanent Drive ZIPs allow the original reaper to confirm their absence.
- Transfer manifests and ACKs were VM-only, but their complete objects are in
  local durable acceptance receipts. No manifest identity was bound elsewhere;
  preserve the same data and paths. Original VM cleanup receipt bytes are lost;
  the existing reaper creates new truthful verification receipts, not invented
  copies. The mutable Drive journal is snapshotted before any continuation write.
- Native, model, source, paired RNG and graph comparisons remain exact. A new VM
  can fail eigenbasis equality even with identical code and caches. That is a
  recorded recovery failure, not permission to weaken the experiment.
