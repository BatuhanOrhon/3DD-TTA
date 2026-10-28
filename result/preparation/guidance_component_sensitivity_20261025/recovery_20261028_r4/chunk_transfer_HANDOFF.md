# Bounded-memory r4 raw-archive transport — preparation only

Files:

- `split_stage_archive.py` —10,455bytes SHA256 `d355092956650f71fd130d51bd2b3d6720f5b6c8200361f240f2939eb9362723`
- `join_stage_archive.py` —2,819bytes SHA256 `667deec2f69d54bac50ff354347b209bc6cf1ca4c3780bc507217f7f00698773`

Keep both files adjacent on the local machine: the join helper imports the split helper's pure manifest validators. Importing it performs no runtime action.

1. After actual development completion, submit `split_stage_archive.py` through the sole r4 monitor. It pins the current controller/plan and requires completed stage receipt/validation plus an inactive completed ledger. The exact marker is `AGENT_B_DEVELOPMENT_CHUNK_TRANSPORT_COMPLETE`.
2. Download the small `/content/studies/guidance-components/sensitivity_development_chunk_transport_receipt.json`. It points to a small manifest with byte/SHA identity. Download and verify that manifest before trusting its part list.
3. Download each manifest part using the ordinary configured `colab download -s tta-guidance-components-a100-20261028-r4 REMOTE_PATH FRESH_LOCAL_PART_PATH`, strictly one at a time. Preserve the exact `part00000.bin`, `part00001.bin`, etc. basenames in one fresh local parts directory. Do not download the original multi-GiB ZIP as a single CLI call or parallelize part downloads.
4. Join locally using the genuine downloaded phase receipt and current plan:

```sh
python join_stage_archive.py \
  --stage development \
  --manifest ACTUAL_TRANSPORT_MANIFEST.json \
  --phase-receipt ACTUAL_DEVELOPMENT_RECEIPT.json \
  --plan ACTUAL_CURRENT_R4_PLAN.json \
  --parts-root FRESH_DOWNLOADED_PARTS_DIRECTORY \
  --output FRESH_ORIGINAL_ARCHIVE.zip
```

The output must not exist. On success, the helper emits `AGENT_B_SENSITIVITY_ARCHIVE_JOIN_COMPLETE` and the exact original archive SHA. On failure, the partial output is deliberately preserved and must not be used as accepted evidence; select a new output path after diagnosing the bad/missing part. No original archive bytes are rewritten.

For confirmation, make a reviewed copy of the remote source changing only `STAGE = 'development'` to `STAGE = 'confirmation'`. Its marker becomes `AGENT_B_CONFIRMATION_CHUNK_TRANSPORT_COMPLETE`, and the small pointer becomes `sensitivity_confirmation_chunk_transport_receipt.json`. Use `--stage confirmation` locally; the local join code requires no change.

The remote helper streams the original Drive ZIP into at most256MiB parts with1MiB buffers. It first hashes the existing original local ZIP, verifies each persisted part's size/SHA, and checks aggregate bytes/SHA against the untouched original phase receipt. Stage paths, part names, offsets/counts and source identities are checked. A fresh unique ROOT directory holds parts; no Drive part copies are made because the original sealed Drive ZIP remains the recovery source. Ready manifest/receipt are written only after the complete match and unchanged inactive ledger. Failed parts remain with failure metadata for inspection.

The local join helper validates the manifest against the genuine phase receipt and pinned current plan/session/stage, enforces canonical owned remote paths and sequential names/offsets, hashes every downloaded part with1MiB buffers, joins exclusively to a fresh local file, and rehashes the persisted result against the original full SHA. The installed CLI still uses base64 JSON internally per download;256MiB bounds each such allocation, so sequential download is required on the16GiB host.

Verification: three synthetic fixtures with257-byte test parts pass in0.013s, covering exact roundtrip/exclusive output, corrupt/missing parts with preserved failure output, out-of-order records and foreign/traversing paths. Python3.8 AST and whitespace checks pass. No CLI, GPU, remote execution or genuine phase transfer was performed by this preparation.
