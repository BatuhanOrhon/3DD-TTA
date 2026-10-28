# Failed confirmation evidence preservation — CPU only

This task preserves the failed attempt. It does not complete or accept confirmation, rerun job20, change objectives, or weaken any complete-only helper.

Pinned actual anchors:

- Failed receipt3,676,445bytes SHA`5ab88805d14edbc9713275816aa11bef1d5e80784ea385b477199f7d6b779754`
- Post-seal inactive failed ledger12,181bytes SHA`60c8ba55d787c8e1e9ea4fa34f27eeb3e9aa8522300252c6c74863e1cad2736b`
- Failed raw ZIP1,159,246,521bytes SHA`f586870206961f88d38daa7ed9da632fddfa51356075e114c5f3b98f020ebe5e`
-20 completed jobs; job20 remains recorded `running` in the failed archived state, which is stale/incomplete metadata rather than a live worker; jobs21–23 were unrun. Ledger is inactive. The union of completed producer metadata contains all180 preparation cache identities.

Upload these unchanged files to `/content/studies/guidance-components/transport_helpers/`:

- `split_stage_archive.py` SHA`d355092956650f71fd130d51bd2b3d6720f5b6c8200361f240f2939eb9362723` (the original complete-only helper copied byte-identically; import pure streaming primitives only)
- `failed_preservation_phase.py` SHA`a4e712716ffb260c64d8bfdf41b5092445176ae15e21fb4b972d684477d602ff`
- `preserve_failed_cache_phase.py` SHA`77d073fad144faaf953ade9938ad07892c600fa5eef4e1c1ef2849cb1ba80f33`

## Failed raw archive transport

Submit `failed_preservation_phase.py` through the sole monitor. It pins the actual failed receipt, post-seal ledger, raw archive, current r4 controller/plan and existing freeze. It streams/verifies the original local ZIP and original Drive ZIP into fresh at-most256MiB transport parts with the unchanged bounded-memory primitives. All status labels remain `failed_attempt`, never accepted/validated/complete experiment results.

Exact marker: `AGENT_B_CONFIRMATION_FAILED_ARTIFACT_TRANSPORT_COMPLETE`.

Small pointer: `ROOT/sensitivity_confirmation_failed_transport_receipt.json`. It identifies the manifest and exact preserved failed receipt/ledger copies. Download those small files, then each part via ordinary `colab download` strictly sequentially. The original Drive ZIP/proof remains unchanged; no Drive part copy is needed.

Keep `join_failed_archive.py`, `failed_preservation_phase.py`, and the unchanged `split_stage_archive.py` adjacent locally. Reconstruct into a new output:

```sh
python join_failed_archive.py \
 --manifest ACTUAL_FAILED_TRANSPORT_MANIFEST.json \
 --failed-receipt ACTUAL_ORIGINAL_FAILED_RECEIPT.json \
 --failed-ledger ACTUAL_POSTSEAL_FAILED_LEDGER.json \
 --plan ACTUAL_CURRENT_R4_PLAN.json \
 --parts-root FRESH_SEQUENTIAL_DOWNLOAD_DIRECTORY \
 --output FRESH_FAILED_ATTEMPT_RAW.zip
```

It verifies failed-only metadata/part namespace, order/offset/size/SHA, aggregate SHA and persisted output SHA with1MiB buffers. Success marker is `AGENT_B_FAILED_CONFIRMATION_ARCHIVE_BYTES_PRESERVED`; the returned status is `failed_attempt_archive_bytes_preserved`. This proves exact failure archive bytes, not successful confirmation. Existing outputs are never overwritten; failed partial join output is retained for inspection.

## All180 cache files and failure proof

Submit `run_failed_cache_preservation_cpu_phase.py` (SHA`cd24aa76cff126bdd6c8f60219603b7093c7979799edff67a84859b40938d8c4`) through the same monitor. It invokes the uploaded collector under `conda run --no-capture-output -n3dd_tta_env python`, not the incidental notebook Python.

Wrapper marker: `AGENT_B_CONFIRMATION_FAILURE_PRESERVATION_WRAPPER_COMPLETE`; inner marker: `AGENT_B_CONFIRMATION_FAILURE_PRESERVATION_COMPLETE`.

The collector requires the pinned actual failed receipt/post-seal ledger and original local+Drive raw archive SHA, frozen plan/development bytes, exact archived job020 log and signature `ValueError: Classifier input contains nonfinite XYZ coordinates`. It only uses consistent records from20 completed producers. It verifies all180 local/Drive NPZ SHA identities and registered140-ID/two-B70 paths, classifier/LION contracts against the original ledger, common paired traces, fresh CPU proof and pinned source files. Under Python3.8/PyTorch2.1.2+cu121/NumPy1, it imports the sealed `legacy_trajectory` source and reuses `tensor_hash`, `_rng_state_hash`, `rng_hash` to check every cache's input/tensor/post-preparation RNG metadata exactly as the producer. CUDA must remain uninitialized; no LION instance or model/GPU forward is created.

Canonical small receipt: `ROOT/sensitivity_confirmation_failure_preservation_receipt.json`.

Its status `verified_failure_preserved` means failure evidence is preserved. `experimental_result_status` remains `failed`. It includes exact parent-agreed fields: session/plan/source/manifest, original_confirmation_archive4fields, original_receipt_identity, failed_ledger_identity, failed_job_index20, numeric_failure_signature, physical_cache_count180, local_and_drive_original_archive_verifiedTrue, physical_cache_completeTrue, mismatch_detectedFalse, scoped integrity_verification and archive/source_log identities. Scope is20 completed jobs+180 preparation caches; job20 final state is explicitly unvalidated. Remaining-slot execution still needs its separate read-only source/native/data preflight.

Fresh sealed proof ZIP layout:

- `cache_records.json` —180 original local_path/drive_path/bytes/SHA records
- `cache_metadata_validation.json` —180 CPU input/tensor/post-RNG verification records
- Original NPZ paths `paired_sensitivity_confirmation_v2_seed{0,1,2}/dataset/corruption/hash.npz`
- `failed_phase_receipt.json`, `failed_campaign_ledger.json`, `job_020.log`
- `sensitivity_confirmation_freeze.json`, `accepted_sensitivity_development.json`, `campaign_plan.json`
- `fresh_cpu_preflight_receipt.json`, `state.json`, `evidence_manifest.json`

Download the actual receipt and physical Drive proof ZIP, verify their actual local SHA/CRC before reporting preservation or using them as continuation gates. Neither raw/classifier partial outputs nor missing jobs become accepted by preserving all180 preparation files. The complete-only collector and final acceptance template remain unchanged and inapplicable to this failed attempt.

Validation: three CPU fixtures pass0.123s: toy failed-archive roundtrip, explicit rejection of complete/accepted labels/foreign path, actual failed metadata's180-record consistency plus tampering rejection. Actual20 completed classifier/LION metadata contracts also match the preserved ledger. Python3.8 AST passes. Local NumPy2/PyTorch negative check rejects the incompatible environment before cache access. Compatible local NumPy1 environment lacks torch, so actual180 tensor/RNG/byte verification is pending the reviewed remote CPU execution; no genuine bytes are claimed preserved by these preparation tests alone.
